// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <system_error>
#include <tuple>

#include "common/logging.h"
#include "video_core/renderer_vulkan/lichii_precompile_io.h"
#include "video_core/renderer_vulkan/vk_bananza_nvn2_compute_list.h"

namespace Vulkan {
namespace {

constexpr std::array<char, 8> Magic{'N', 'V', 'N', '2', 'C', 'O', 'M', 'P'};
constexpr u32 Version = 1;
constexpr size_t HeaderSize = Magic.size() + sizeof(u32);

// The lists come out of hash tables, in no order: compared sorted, field by field (the pairs
// have padding, so not as bytes).
template <typename Pairs, typename Key>
bool SameList(Pairs a, Pairs b, Key&& key) {
    if (a.size() != b.size()) {
        return false;
    }
    const auto less = [&key](const auto& x, const auto& y) { return key(x) < key(y); };
    std::ranges::sort(a, less);
    std::ranges::sort(b, less);
    return std::ranges::equal(a, b, [&key](const auto& x, const auto& y) {
        return key(x) == key(y);
    });
}

auto Location(const Shader::HandleKey& at) {
    return std::tuple{at.index, at.offset, at.shift_left, at.sec_index,
                      at.sec_offset, at.sec_shift_left, at.count, at.has_secondary};
}

} // Anonymous namespace

bool BananzaNvn2ComputeList::Same(const VideoCommon::RecordedEnvQueries& a,
                                  const VideoCommon::RecordedEnvQueries& b) {
    const auto plain = [](const auto& entry) {
        return std::pair{static_cast<u64>(entry.first), static_cast<u64>(entry.second)};
    };
    const auto located = [](const auto& entry) {
        return std::pair{Location(entry.first), static_cast<u64>(entry.second)};
    };
    return a.texture_bound == b.texture_bound &&
           a.viewport_transform_state == b.viewport_transform_state &&
           a.has_hle_engine_state == b.has_hle_engine_state &&
           a.has_unbound_instructions == b.has_unbound_instructions &&
           a.cbuf_replacements.empty() && b.cbuf_replacements.empty() &&
           SameList(a.texture_types, b.texture_types, plain) &&
           SameList(a.texture_pixel_formats, b.texture_pixel_formats, plain) &&
           SameList(a.cbuf_values, b.cbuf_values, plain) &&
           SameList(a.storage_pointer_types, b.storage_pointer_types, plain) &&
           SameList(a.location_texture_types, b.location_texture_types, located) &&
           SameList(a.location_texture_formats, b.location_texture_formats, located) &&
           SameList(a.location_integer_formats, b.location_integer_formats, located);
}

std::vector<u8> BananzaNvn2ComputeList::Pack(const Variant& variant) {
    LichiiIO::Writer body;
    body.Put(variant.program);
    body.Put(variant.texture_bank);
    LichiiIO::PutQueries(body, variant.queries);
    LichiiIO::Writer record;
    record.Put(static_cast<u32>(body.buffer.size()));
    record.PutBytes(body.buffer.data(), body.buffer.size());
    return std::move(record.buffer);
}

BananzaNvn2ComputeList::BananzaNvn2ComputeList(std::filesystem::path file_)
    : file{std::move(file_)} {
    std::vector<u8> bytes;
    if (std::ifstream in{file, std::ios::binary}) {
        bytes.assign(std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{});
    }
    bool whole = bytes.size() >= HeaderSize &&
                 std::equal(Magic.begin(), Magic.end(), bytes.begin());
    if (whole) {
        u32 version{};
        std::memcpy(&version, bytes.data() + Magic.size(), sizeof(version));
        whole = version == Version;
    }
    if (whole) {
        size_t offset = HeaderSize;
        try {
            while (offset < bytes.size()) {
                LichiiIO::Reader size_reader{std::span<const u8>{bytes}.subspan(offset)};
                const u32 size = size_reader.Get<u32>();
                if (size > bytes.size() - offset - sizeof(u32)) {
                    throw std::runtime_error{"cut record"};
                }
                LichiiIO::Reader reader{
                    std::span<const u8>{bytes}.subspan(offset + sizeof(u32), size)};
                Variant variant;
                variant.program = reader.Get<u64>();
                variant.texture_bank = reader.Get<u32>();
                variant.queries = LichiiIO::GetQueries(reader);
                by_program[variant.program].push_back(variants.size());
                variants.push_back(std::move(variant));
                offset += sizeof(u32) + size;
            }
        } catch (const std::exception&) {
            // A record cut short by a crash: what was read before it is kept.
            whole = false;
        }
    }
    if (whole) {
        LOG_INFO(Render_Vulkan, "NVN 2 compute list: {} variants in {}", variants.size(),
                 file.string());
        return;
    }
    std::error_code error;
    std::filesystem::create_directories(file.parent_path(), error);
    std::ofstream out{file, std::ios::binary | std::ios::trunc};
    out.write(Magic.data(), static_cast<std::streamsize>(Magic.size()));
    out.write(reinterpret_cast<const char*>(&Version), sizeof(Version));
    for (const Variant& variant : variants) {
        const std::vector<u8> packed = Pack(variant);
        out.write(reinterpret_cast<const char*>(packed.data()),
                  static_cast<std::streamsize>(packed.size()));
    }
    LOG_INFO(Render_Vulkan, "NVN 2 compute list: started {} with {} variants", file.string(),
             variants.size());
}

size_t BananzaNvn2ComputeList::Add(u64 program, u32 texture_bank,
                                   const VideoCommon::RecordedEnvQueries& queries) {
    auto& indices = by_program[program];
    for (const size_t index : indices) {
        if (variants[index].texture_bank == texture_bank && Same(variants[index].queries, queries)) {
            return index;
        }
    }
    const size_t index = variants.size();
    indices.push_back(index);
    variants.push_back({program, texture_bank, queries});
    const std::vector<u8> packed = Pack(variants.back());
    std::ofstream out{file, std::ios::binary | std::ios::app};
    out.write(reinterpret_cast<const char*>(packed.data()),
              static_cast<std::streamsize>(packed.size()));
    return index;
}

const std::vector<size_t>& BananzaNvn2ComputeList::Of(u64 program) const {
    static const std::vector<size_t> none;
    const auto found = by_program.find(program);
    return found != by_program.end() ? found->second : none;
}

} // namespace Vulkan
