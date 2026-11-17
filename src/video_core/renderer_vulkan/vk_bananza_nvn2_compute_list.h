// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "video_core/shader_environment.h"

namespace Vulkan {

/// The compute pipeline variants the game dispatched, kept in a file across runs: a variant is
/// a compute program, named by the hash of its contents, with what its translation asked of
/// the bindings and was answered. The next run builds them on worker threads when the game
/// makes the program. One thread uses the list.
class BananzaNvn2ComputeList {
public:
    struct Variant {
        u64 program{};
        u32 texture_bank{};
        VideoCommon::RecordedEnvQueries queries;
    };

    explicit BananzaNvn2ComputeList(std::filesystem::path file);

    /// The index of the variant, added and appended to the file when the list did not hold it.
    size_t Add(u64 program, u32 texture_bank, const VideoCommon::RecordedEnvQueries& queries);

    /// The indices of a program's variants.
    [[nodiscard]] const std::vector<size_t>& Of(u64 program) const;

    [[nodiscard]] const Variant& At(size_t index) const {
        return variants[index];
    }

    [[nodiscard]] size_t Size() const {
        return variants.size();
    }

    /// Whether two records hold the same reads and answers, in any order.
    [[nodiscard]] static bool Same(const VideoCommon::RecordedEnvQueries& a,
                                   const VideoCommon::RecordedEnvQueries& b);

private:
    static std::vector<u8> Pack(const Variant& variant);

    std::filesystem::path file;
    std::vector<Variant> variants;
    std::unordered_map<u64, std::vector<size_t>> by_program;
};

} // namespace Vulkan
