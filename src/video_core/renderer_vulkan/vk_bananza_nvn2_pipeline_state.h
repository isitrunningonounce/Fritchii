// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <compare>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

/// The state a Bananza NVN 2 graphics pipeline is made for, besides its programs and vertex
/// layout: shared by the replay and the list of pipelines kept across runs.
using BananzaNvn2ColorFormats = std::array<VkFormat, 8>;

struct BananzaNvn2ColorState {
    std::array<u64, 8> blend{};
    u32 control{0x08030000};
    u32 channel_mask{0xffffffff};
    u32 alpha_ref{};
    auto operator<=>(const BananzaNvn2ColorState&) const = default;
};

struct BananzaNvn2RasterState {
    u64 depth_stencil{0x1118111800000022ULL};
    u32 polygon{0x14};
    VkFormat depth_format{};
    VkImageAspectFlags depth_aspects{};
    bool negative_one_to_one{};
    bool origin_lower_left{};
    bool depth_clamp{};
    bool discard{};
    bool depth_bias{};
    auto operator<=>(const BananzaNvn2RasterState&) const = default;
};

} // namespace Vulkan
