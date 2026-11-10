// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstdlib>
#include <span>
#include "common/common_types.h"
#include "video_core/mkw_upscale.h"
#include "video_core/framebuffer_config.h"

namespace VideoCommon {

inline constexpr u64 MarioKartPresentVertex = 0xa11ab97193a0ba8aULL;
inline constexpr u64 MarioKartPresentFragment = 0x52402d8e8ab55263ULL;

struct MarioKartHudlessScene {
    u64 frame{};
    MarioKartBlendImage scene{};
    u64 present{};

    bool MatchCopy(u64 current_frame, u64 vertex, u64 fragment, u32 elements, u32 instances,
                   const MarioKartBlendImage& source, const MarioKartBlendImage& target) const {
        using VideoCore::Surface::PixelFormat;
        return current_frame == frame && scene.identity != 0 && present == 0 &&
               vertex == MarioKartPresentVertex && fragment == MarioKartPresentFragment &&
               elements == 6 && instances == 1 && source.identity == scene.identity &&
               source.format == PixelFormat::B10G11R11_FLOAT && source.width == scene.width &&
               source.height == scene.height && target.identity != 0 &&
               target.identity != source.identity && target.format == PixelFormat::A8B8G8R8_SRGB &&
               target.width >= scene.width && target.height >= scene.height &&
               target.width <= 8192 && target.height <= 8192;
    }

    bool MatchesPresent(u64 current_frame, u64 identity) const {
        return current_frame == frame && present != 0 && identity == present;
    }
};

inline bool CanComposeMarioKartHudless(const Tegra::FramebufferConfig& config,
                                      u32 scene_width, u32 scene_height, size_t layers,
                                      bool effects, bool temporal_filter) {
    if (layers != 1 || effects || temporal_filter || config.blending != Tegra::BlendMode::Opaque ||
        (static_cast<u32>(config.transform_flags) & ~3u) != 0 ||
        scene_width == 0 || scene_height == 0) {
        return false;
    }
    const auto crop = Tegra::NormalizeCrop(config, scene_width, scene_height);
    return crop.left != crop.right && crop.top != crop.bottom &&
           std::min(crop.left, crop.right) >= 0 && std::max(crop.left, crop.right) <= 1 &&
           std::min(crop.top, crop.bottom) >= 0 && std::max(crop.top, crop.bottom) <= 1;
}

struct MarioKartHudlessDifference {
    size_t pixels{};
    size_t changed{};
    u32 maximum{};
};

inline MarioKartHudlessDifference CompareMarioKartHudless(std::span<const u8> final_color,
                                                          std::span<const u8> hudless) {
    MarioKartHudlessDifference result{std::min(final_color.size(), hudless.size()) / 4};
    for (size_t pixel = 0; pixel < result.pixels; ++pixel) {
        u32 difference{};
        for (size_t channel = 0; channel < 3; ++channel) {
            const size_t offset = pixel * 4 + channel;
            difference = std::max(difference,
                static_cast<u32>(std::abs(int(final_color[offset]) - int(hudless[offset]))));
        }
        result.changed += difference > 2;
        result.maximum = std::max(result.maximum, difference);
    }
    return result;
}

} // namespace VideoCommon
