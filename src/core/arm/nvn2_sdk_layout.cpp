// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>

#include "common/alignment.h"
#include "core/arm/nvn2_sdk_layout.h"

namespace Core::Nvn2 {
namespace {

// The procedure the bootstrap hands out, so every table has it.
constexpr std::string_view AnchorName{"DeviceGetProcAddress"};
constexpr std::string_view AnchorNeedle{"\0DeviceGetProcAddress\0", 22};

constexpr u64 DT_HASH = 4;
constexpr u64 DT_STRTAB = 5;
constexpr u64 DT_SYMTAB = 6;
constexpr u64 DT_STRSZ = 10;
constexpr u64 DT_GNU_HASH = 0x6FFFFEF5;
constexpr size_t MaxDynamicEntries = 256;
constexpr size_t SymbolSize = 24;
constexpr size_t MaxBootstrapSize = 64;

} // Anonymous namespace

std::optional<u32> SdkLayoutFinder::U32(size_t offset) const {
    if (offset > image.size() || image.size() - offset < sizeof(u32)) {
        return std::nullopt;
    }
    u32 value;
    std::memcpy(&value, image.data() + offset, sizeof(value));
    return value;
}

std::optional<u64> SdkLayoutFinder::U64(size_t offset) const {
    if (offset > image.size() || image.size() - offset < sizeof(u64)) {
        return std::nullopt;
    }
    u64 value;
    std::memcpy(&value, image.data() + offset, sizeof(value));
    return value;
}

bool SdkLayoutFinder::IsCode(u64 offset) const {
    return offset != 0 && offset < text_end && offset % sizeof(u32) == 0;
}

std::optional<SdkLayoutFinder::SymbolTable> SdkLayoutFinder::DynamicSymbols() const {
    const auto mod = U32(4);
    if (!mod || U32(*mod) != u32{0x30444F4D}) { // "MOD0"
        return std::nullopt;
    }
    const auto dynamic_offset = U32(size_t{*mod} + 4);
    if (!dynamic_offset) {
        return std::nullopt;
    }
    // Wraps for a table before the header and for garbage alike; the reads below check.
    size_t entry = size_t{*mod} + static_cast<size_t>(static_cast<s32>(*dynamic_offset));
    std::optional<u64> symbols, strings, strings_size, hash, gnu_hash;
    for (size_t i = 0; i < MaxDynamicEntries; ++i, entry += 16) {
        const auto tag = U64(entry);
        const auto value = U64(entry + 8);
        if (!tag || !value) {
            return std::nullopt;
        }
        if (*tag == 0) {
            break;
        }
        if (*tag == DT_SYMTAB) {
            symbols = value;
        } else if (*tag == DT_STRTAB) {
            strings = value;
        } else if (*tag == DT_STRSZ) {
            strings_size = value;
        } else if (*tag == DT_HASH) {
            hash = value;
        } else if (*tag == DT_GNU_HASH) {
            gnu_hash = value;
        }
    }
    if (!symbols || !strings || !strings_size || *symbols >= image.size() ||
        *strings >= image.size() || *strings_size > image.size() - *strings) {
        return std::nullopt;
    }
    // Neither hash table says how many symbols there are in so many words.
    size_t count{};
    if (hash && *hash < image.size()) {
        count = U32(static_cast<size_t>(*hash) + 4).value_or(0);
    } else if (gnu_hash && *gnu_hash < image.size()) {
        count = GnuHashSymbolCount(static_cast<size_t>(*gnu_hash));
    }
    count = std::min(count, (image.size() - static_cast<size_t>(*symbols)) / SymbolSize);
    return SymbolTable{static_cast<size_t>(*symbols), count, static_cast<size_t>(*strings),
                       static_cast<size_t>(*strings_size)};
}

size_t SdkLayoutFinder::GnuHashSymbolCount(size_t table) const {
    const auto bucket_count = U32(table);
    const auto first_hashed = U32(table + 4);
    const auto bloom_size = U32(table + 8);
    if (!bucket_count || !first_hashed || !bloom_size) {
        return 0;
    }
    const size_t buckets = table + 16 + size_t{*bloom_size} * sizeof(u64);
    const size_t chains = buckets + size_t{*bucket_count} * sizeof(u32);
    if (chains < table || chains > image.size()) {
        return 0;
    }
    // The last symbol is the end of the chain that starts highest.
    size_t last{};
    for (size_t i = 0; i < *bucket_count; ++i) {
        last = std::max<size_t>(last, U32(buckets + i * sizeof(u32)).value_or(0));
    }
    if (last < *first_hashed) {
        return *first_hashed;
    }
    while (true) {
        const auto link = U32(chains + (last - *first_hashed) * sizeof(u32));
        if (!link) {
            return 0;
        }
        if ((*link & 1) != 0) {
            return last + 1;
        }
        ++last;
    }
}

std::optional<SdkLayoutFinder::Symbol> SdkLayoutFinder::FindSymbol(const SymbolTable& table,
                                                                   std::string_view name) const {
    for (size_t i = 0; i < table.count; ++i) {
        const size_t symbol = table.symbols + i * SymbolSize;
        const auto name_offset = U32(symbol);
        const auto info = U32(symbol + 4); // info, other, section index
        const auto value = U64(symbol + 8);
        const auto size = U64(symbol + 16);
        if (!name_offset || !info || !value || !size) {
            return std::nullopt;
        }
        const bool defined = (*info >> 16) != 0;
        if (!defined || *name_offset >= table.strings_size ||
            table.strings_size - *name_offset <= name.size()) {
            continue;
        }
        const u8* const text = image.data() + table.strings + *name_offset;
        if (std::memcmp(text, name.data(), name.size()) == 0 && text[name.size()] == 0) {
            return Symbol{*value, *size};
        }
    }
    return std::nullopt;
}

std::optional<size_t> SdkLayoutFinder::BranchTarget(const Symbol& function) const {
    if (function.size == 0 || function.size > MaxBootstrapSize) {
        return std::nullopt;
    }
    for (u64 at = function.value; at < function.value + function.size; at += sizeof(u32)) {
        const auto word = U32(static_cast<size_t>(at));
        if (!word) {
            return std::nullopt;
        }
        if (*word >> 26 == 0b000101) { // b
            const s64 words = static_cast<s64>(static_cast<u64>(*word) << 38) >> 38;
            const u64 target = at + static_cast<u64>(words * 4);
            return IsCode(target) ? std::optional<size_t>{static_cast<size_t>(target)}
                                  : std::nullopt;
        }
    }
    return std::nullopt;
}

// The length of the name at offset, 0 when there is none that ends before limit.
size_t SdkLayoutFinder::NameLength(size_t offset, size_t limit) const {
    if (offset >= limit) {
        return 0;
    }
    const size_t room = std::min(limit - offset, SdkLayout::MaxNameLength + 1);
    const void* const end = std::memchr(image.data() + offset, 0, room);
    return end ? static_cast<size_t>(static_cast<const u8*>(end) - (image.data() + offset)) : 0;
}

std::optional<SdkLayoutFinder::Names> SdkLayoutFinder::FindNames() const {
    // The name alone, as another string of the image, also reads as a table of one entry:
    // the table is the largest.
    const std::string_view data{reinterpret_cast<const char*>(image.data()), image.size()};
    std::optional<Names> best;
    for (size_t at = data.find(AnchorNeedle, text_end); at != std::string_view::npos;
         at = data.find(AnchorNeedle, at + 1)) {
        const auto names = NamesAround(at + 1);
        if (names && (!best || names->count > best->count)) {
            best = names;
        }
    }
    return best;
}

std::optional<SdkLayoutFinder::Names> SdkLayoutFinder::NamesAround(size_t anchor) const {
    // The names lie back to back, so the block ends at the first empty one, and the offsets
    // follow it.
    size_t end = anchor;
    while (const size_t length = NameLength(end, image.size())) {
        end += length + 1;
    }
    const size_t offsets = Common::AlignUp(end, sizeof(u32));
    std::array<u32, SdkLayout::MaxEntries> run;
    size_t run_size{};
    for (; run_size < run.size(); ++run_size) {
        const auto value = U32(offsets + run_size * sizeof(u32));
        if (!value || *value >= end - text_end ||
            (run_size == 0 ? *value != 0 : *value <= run[run_size - 1])) {
            break;
        }
        run[run_size] = *value;
    }
    // The block starts where one of the offsets, taken back from the anchor, makes every
    // offset the start of its name.
    for (size_t index = 0; index < run_size; ++index) {
        if (run[index] > anchor - text_end ||
            (index + 1 < run_size && run[index + 1] - run[index] != AnchorName.size() + 1)) {
            continue;
        }
        const size_t names = anchor - run[index];
        const size_t count = CountNames(names, end, std::span{run}.first(run_size));
        if (count > index) {
            return Names{names, offsets, count, index};
        }
    }
    return std::nullopt;
}

// How many names the block [names, end) holds, 0 unless they are packed, sorted and at
// their offsets.
size_t SdkLayoutFinder::CountNames(size_t names, size_t end, std::span<const u32> offsets) const {
    const auto name_at = [&](size_t offset, size_t length) {
        return std::string_view{reinterpret_cast<const char*>(image.data() + offset), length};
    };
    size_t at = names;
    size_t count{};
    std::string_view previous;
    while (at < end) {
        const size_t length = NameLength(at, end);
        if (length == 0 || count == offsets.size() || offsets[count] != at - names ||
            name_at(at, length) <= previous) {
            return 0;
        }
        previous = name_at(at, length);
        at += length + 1;
        ++count;
    }
    return count;
}

bool SdkLayoutFinder::ValidTargets(size_t offset, size_t count) const {
    std::array<u64, SdkLayout::MaxEntries> targets;
    if (offset > image.size() || (image.size() - offset) / sizeof(u64) < count) {
        return false;
    }
    std::memcpy(targets.data(), image.data() + offset, count * sizeof(u64));
    const auto used = std::span{targets}.first(count);
    if (!std::ranges::all_of(used, [this](u64 target) { return IsCode(target); })) {
        return false;
    }
    std::ranges::sort(used);
    return std::ranges::adjacent_find(used) == used.end();
}

// The table is the run of procedures that has the anchor's code at the anchor's index.
std::optional<size_t> SdkLayoutFinder::FindTargets(const Names& names,
                                                   size_t anchor_target) const {
    const size_t before = names.anchor_index * sizeof(u64);
    const size_t bytes = names.count * sizeof(u64);
    std::optional<size_t> found;
    for (size_t at = Common::AlignUp(text_end, sizeof(u64)); at + sizeof(u64) <= image.size();
         at += sizeof(u64)) {
        if (U64(at) != u64{anchor_target} || at - text_end < before ||
            !ValidTargets(at - before, names.count)) {
            continue;
        }
        if (found && std::memcmp(image.data() + *found, image.data() + at - before, bytes) != 0) {
            return std::nullopt; // two tables that disagree
        }
        if (!found) {
            found = at - before;
        }
    }
    return found;
}

std::optional<SdkLayout> SdkLayoutFinder::Find() const {
    if (text_end > image.size()) {
        return std::nullopt;
    }
    const auto symbols = DynamicSymbols();
    if (!symbols) {
        return std::nullopt;
    }
    const auto bootstrap = FindSymbol(*symbols, "nvnBootstrapLoader");
    const auto internal = FindSymbol(*symbols, "nvnBootstrapLoaderInternal");
    if (!bootstrap || !internal || !IsCode(bootstrap->value) || !IsCode(internal->value)) {
        return std::nullopt;
    }
    // The internal bootstrap is a jump to the anchor procedure.
    const auto anchor_target = BranchTarget(*internal);
    if (!anchor_target) {
        return std::nullopt;
    }
    const auto names = FindNames();
    if (!names) {
        return std::nullopt;
    }
    const auto targets = FindTargets(*names, *anchor_target);
    if (!targets) {
        return std::nullopt;
    }
    // The padding after the internal bootstrap: a word of the text that nothing executes.
    const u64 padding = internal->value + internal->size;
    const bool has_padding = IsCode(padding) && U32(static_cast<size_t>(padding)) == u32{0};
    return SdkLayout{
        .names = names->names,
        .name_offsets = names->offsets,
        .targets = *targets,
        .text_end = text_end,
        .entry_count = names->count,
        .bootstrap = static_cast<size_t>(bootstrap->value),
        .internal_bootstrap = static_cast<size_t>(internal->value),
        .callback_return = has_padding ? static_cast<size_t>(padding) : 0,
    };
}

} // namespace Core::Nvn2
