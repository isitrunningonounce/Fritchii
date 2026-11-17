// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include <span>
#include <string_view>

#include "common/common_types.h"

namespace Core::Nvn2 {

// Where an SDK image keeps its NVN procedure table. Every value is an offset into the image.
struct SdkLayout {
    static constexpr size_t MaxEntries = 1024;
    static constexpr size_t MaxNameLength = 96;

    size_t names;        // the procedure names without "nvn": sorted, each ended by a NUL
    size_t name_offsets; // one u32 an entry, the name's offset into names
    size_t targets;      // one u64 an entry, the procedure's code
    size_t text_end;
    size_t entry_count;
    size_t bootstrap;
    size_t internal_bootstrap;
    size_t callback_return{}; // a zero word of the text, 0 when the image has none

    bool operator==(const SdkLayout&) const = default;
};

// Finds the procedure table of an SDK image from the image itself: the bootstrap functions
// from the dynamic symbols, the names by the procedure every table has, the targets by the
// one the internal bootstrap jumps to. The image is game data: nothing in it is trusted.
class SdkLayoutFinder {
public:
    SdkLayoutFinder(std::span<const u8> image_, size_t text_end_)
        : image{image_}, text_end{text_end_} {}

    std::optional<SdkLayout> Find() const;

private:
    struct Symbol {
        u64 value;
        u64 size;
    };
    struct SymbolTable {
        size_t symbols;
        size_t count;
        size_t strings;
        size_t strings_size;
    };
    struct Names {
        size_t names;
        size_t offsets;
        size_t count;
        size_t anchor_index;
    };

    std::optional<u32> U32(size_t offset) const;
    std::optional<u64> U64(size_t offset) const;
    bool IsCode(u64 offset) const;

    std::optional<SymbolTable> DynamicSymbols() const;
    size_t GnuHashSymbolCount(size_t table) const;
    std::optional<Symbol> FindSymbol(const SymbolTable& table, std::string_view name) const;
    std::optional<size_t> BranchTarget(const Symbol& function) const;

    size_t NameLength(size_t offset, size_t limit) const;
    std::optional<Names> FindNames() const;
    std::optional<Names> NamesAround(size_t anchor) const;
    size_t CountNames(size_t names, size_t end, std::span<const u32> offsets) const;

    bool ValidTargets(size_t offset, size_t count) const;
    std::optional<size_t> FindTargets(const Names& names, size_t anchor_target) const;

    std::span<const u8> image;
    size_t text_end;
};

} // namespace Core::Nvn2
