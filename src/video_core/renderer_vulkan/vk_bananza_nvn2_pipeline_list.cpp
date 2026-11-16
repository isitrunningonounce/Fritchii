// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <system_error>

#include "common/cityhash.h"
#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_bananza_nvn2_pipeline_list.h"

namespace Vulkan {
namespace {

constexpr std::array<char, 8> Magic{'N', 'V', 'N', '2', 'P', 'I', 'P', 'E'};
constexpr u32 Version = 1;

template <typename T>
void Put(std::vector<u8>& out, const T& value) {
    const auto* const bytes = reinterpret_cast<const u8*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

template <typename T>
void Get(std::span<const u8>& in, T& value) {
    std::memcpy(&value, in.data(), sizeof(T));
    in = in.subspan(sizeof(T));
}

// Each field by itself: the padding of the structures is not written.
template <class Field>
void Fields(Field&& field, auto& entry) {
    field(entry.programs);
    field(entry.formats);
    field(entry.color_formats);
    field(entry.color_count);
    field(entry.topology);
    field(entry.vertex_attributes);
    field(entry.vertex_strides);
    field(entry.vertex_divisors);
    field(entry.color_state.blend);
    field(entry.color_state.control);
    field(entry.color_state.channel_mask);
    field(entry.color_state.alpha_ref);
    field(entry.raster.depth_stencil);
    field(entry.raster.polygon);
    field(entry.raster.depth_format);
    field(entry.raster.depth_aspects);
    field(entry.raster.negative_one_to_one);
    field(entry.raster.origin_lower_left);
    field(entry.raster.depth_clamp);
    field(entry.raster.discard);
    field(entry.raster.depth_bias);
}

u64 HashOf(std::span<const u8> bytes) {
    return Common::CityHash64(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

} // Anonymous namespace

std::vector<u8> BananzaNvn2PipelineList::Pack(const Entry& entry) {
    std::vector<u8> bytes;
    Fields([&bytes](const auto& value) { Put(bytes, value); }, entry);
    return bytes;
}

BananzaNvn2PipelineList::Entry BananzaNvn2PipelineList::Unpack(std::span<const u8> bytes) {
    Entry entry;
    Fields([&bytes](auto& value) { Get(bytes, value); }, entry);
    return entry;
}

std::vector<u8> BananzaNvn2PipelineList::Header() const {
    std::vector<u8> header;
    Put(header, Magic);
    Put(header, Version);
    Put(header, vendor);
    Put(header, device);
    Put(header, static_cast<u32>(Pack(Entry{}).size()));
    return header;
}

void BananzaNvn2PipelineList::Hold(const Entry& entry, u64 hash) {
    held.insert(hash);
    for (size_t i = 0; i < entry.programs.size(); ++i) {
        // Once for a program the entry names twice.
        const u64 program = entry.programs[i];
        if (program != 0 && std::find(entry.programs.begin(), entry.programs.begin() + i,
                                      program) == entry.programs.begin() + i) {
            by_program.emplace(program, entries.size());
        }
    }
    entries.push_back(entry);
}

BananzaNvn2PipelineList::BananzaNvn2PipelineList(std::filesystem::path file_, u32 vendor_,
                                                 u32 device_)
    : file{std::move(file_)}, vendor{vendor_}, device{device_} {
    const std::vector<u8> header = Header();
    const size_t record = Pack(Entry{}).size();
    std::vector<u8> bytes;
    if (std::ifstream in{file, std::ios::binary}) {
        bytes.assign(std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{});
    }
    if (bytes.size() >= header.size() && std::equal(header.begin(), header.end(), bytes.begin())) {
        // A record cut short by a crash is left out.
        for (size_t offset = header.size(); offset + record <= bytes.size(); offset += record) {
            const std::span<const u8> packed{bytes.data() + offset, record};
            if (const u64 hash = HashOf(packed); !held.contains(hash)) {
                Hold(Unpack(packed), hash);
            }
        }
        if ((bytes.size() - header.size()) % record == 0) {
            LOG_INFO(Render_Vulkan, "Bananza NVN 2 pipeline list: {} pipelines in {}", entries.size(),
                     file.string());
            return;
        }
    }
    // No file, another GPU or version, or a cut record: written again from what was read.
    std::error_code error;
    std::filesystem::create_directories(file.parent_path(), error);
    std::ofstream out{file, std::ios::binary | std::ios::trunc};
    out.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));
    for (const Entry& entry : entries) {
        const std::vector<u8> packed = Pack(entry);
        out.write(reinterpret_cast<const char*>(packed.data()),
                  static_cast<std::streamsize>(packed.size()));
    }
    LOG_INFO(Render_Vulkan, "Bananza NVN 2 pipeline list: started {} with {} pipelines",
             file.string(), entries.size());
}

void BananzaNvn2PipelineList::Add(const Entry& entry) {
    const std::vector<u8> packed = Pack(entry);
    const u64 hash = HashOf(packed);
    if (held.contains(hash)) {
        return;
    }
    Hold(entry, hash);
    std::ofstream out{file, std::ios::binary | std::ios::app};
    out.write(reinterpret_cast<const char*>(packed.data()),
              static_cast<std::streamsize>(packed.size()));
}

} // namespace Vulkan
