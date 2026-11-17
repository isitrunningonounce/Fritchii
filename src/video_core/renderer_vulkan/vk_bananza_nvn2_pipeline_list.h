// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_bananza_nvn2_pipeline_state.h"

namespace Vulkan {

/// The graphics pipelines the game drew with, kept in a file across runs: the next run builds
/// them on worker threads as soon as the game has made their shader programs, so a draw that
/// needs one for the first time does not have to wait for it. An entry names its programs by
/// the hash of their contents. One thread uses the list.
class BananzaNvn2PipelineList {
public:
    static constexpr u32 NoColor = ~0U;

    struct Entry {
        // Graphics, fragment and geometry program; zero where there is none.
        std::array<u64, 3> programs{};
        BananzaNvn2ColorFormats formats{};
        // The guest format of each bound colour texture, NoColor where none is bound.
        std::array<u32, 8> color_formats{};
        u32 color_count{};
        VkPrimitiveTopology topology{};
        std::array<u32, 32> vertex_attributes{};
        std::array<u32, 32> vertex_strides{};
        std::array<u32, 32> vertex_divisors{};
        BananzaNvn2ColorState color_state;
        BananzaNvn2RasterState raster;
    };

    /// Reads the file. A file written for another GPU or by another version of the list is
    /// started again.
    BananzaNvn2PipelineList(std::filesystem::path file, u32 vendor, u32 device);

    /// Adds an entry the list does not hold yet and appends it to the file.
    void Add(const Entry& entry);

    /// Whether an entry names the program.
    [[nodiscard]] bool Uses(u64 program) const {
        return by_program.contains(program);
    }

    /// Calls ready(entry) for every entry that names the program and whose programs are all
    /// known(hash).
    template <class Known, class Ready>
    void ForProgram(u64 program, Known&& known, Ready&& ready) const {
        const auto [first, last] = by_program.equal_range(program);
        for (auto it = first; it != last; ++it) {
            const Entry& entry = entries[it->second];
            bool complete = true;
            for (const u64 hash : entry.programs) {
                complete = complete && (hash == 0 || known(hash));
            }
            if (complete) {
                ready(entry);
            }
        }
    }

    [[nodiscard]] size_t Size() const {
        return entries.size();
    }

private:
    static std::vector<u8> Pack(const Entry& entry);
    static Entry Unpack(std::span<const u8> bytes);
    std::vector<u8> Header() const;
    void Hold(const Entry& entry, u64 hash);

    std::filesystem::path file;
    u32 vendor;
    u32 device;
    std::vector<Entry> entries;
    // Hashes of the packed entries.
    std::unordered_set<u64> held;
    // For each program, the entries that name it.
    std::unordered_multimap<u64, size_t> by_program;
};

} // namespace Vulkan
