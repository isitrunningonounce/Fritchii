// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <span>
#include <string_view>

#include "video_core/mkw_upscale.h"
#include "video_core/texture_cache/types.h"

namespace VideoCommon {

inline constexpr u64 BananzaUpscaleShader = 0xcd25b1bef2c4558aULL;
inline constexpr Extent2D BananzaOutputExtent{2880, 1620};
inline constexpr u32 BananzaSceneWidthPadding = 63;

constexpr OcarinaUpscaleBackend SelectBananzaUpscaler(bool dlss_enabled, bool dlss_available,
                                                      std::string_view override) {
    return SelectOcarinaUpscaler(false, dlss_enabled, dlss_available, override);
}

struct BananzaCamera {
    MarioKartCamera matrices;
    Extent2D extent;
    f32 near_plane;
    f32 far_plane;
};

constexpr bool IsBananzaSceneTarget(Extent2D scene, Extent2D target) {
    return target.height == scene.height && target.width >= scene.width &&
           target.width - scene.width <= BananzaSceneWidthPadding;
}

inline std::optional<BananzaCamera> ReadBananzaCamera(std::span<const f32> block, Extent2D target,
                                                      bool gpu_modified) {
    if (gpu_modified || block.size_bytes() < 0x1c0) {
        return std::nullopt;
    }
    const f32 width = block[0x120 / 4];
    const f32 near_plane = block[0xe0 / 4];
    const f32 far_plane = block[0xe4 / 4];
    if (!(width >= 640.0f && width <= static_cast<f32>(BananzaOutputExtent.width)) ||
        std::trunc(width) != width || !std::isfinite(near_plane) || !std::isfinite(far_plane) ||
        near_plane <= 0.0f || far_plane <= near_plane ||
        target.height > BananzaOutputExtent.height) {
        return std::nullopt;
    }
    const Extent2D scene{static_cast<u32>(width), target.height};
    if (!IsBananzaSceneTarget(scene, target)) {
        return std::nullopt;
    }
    const auto matrices = ReadMarioKartCamera(block, scene.width, scene.height, false);
    if (!matrices) {
        return std::nullopt;
    }
    return BananzaCamera{*matrices, scene, near_plane, far_plane};
}

inline std::optional<TemporalCameraConstants> MakeBananzaTemporalCamera(const BananzaCamera& camera) {
    auto constants = MakeMarioKartTemporalCamera(camera.matrices, camera.extent.width,
                                                 camera.extent.height);
    if (constants) {
        constants->near_plane = camera.near_plane;
        constants->far_plane = camera.far_plane;
    }
    return constants;
}

enum class BananzaFrameShape { Upscale, NativeSize };

constexpr std::optional<BananzaFrameShape> ClassifyBananzaBlend(Extent2D scene, Extent2D target) {
    if (scene == BananzaOutputExtent && target == BananzaOutputExtent) {
        return BananzaFrameShape::NativeSize;
    }
    if (scene.width >= 640 && scene.height >= 360 && scene.width < BananzaOutputExtent.width &&
        scene.height < BananzaOutputExtent.height && IsBananzaSceneTarget(scene, target)) {
        return BananzaFrameShape::Upscale;
    }
    return std::nullopt;
}

inline std::optional<BananzaFrameShape> RecognizeBananzaBlend(
    u64 fragment, bool indexed, bool indirect, u32 vertices, u32 instances,
    std::span<const MarioKartBlendImage> sampled, const MarioKartBlendImage& target, Extent2D scene) {
    const auto shape = ClassifyBananzaBlend(scene, {target.width, target.height});
    const bool in_place = shape == BananzaFrameShape::NativeSize;
    if (shape && IsMarioKartBlend(fragment, indexed, indirect, vertices, instances, sampled, target,
                                 scene.width, scene.height, in_place,
                                 in_place ? 0U : BananzaSceneWidthPadding)) {
        return shape;
    }
    return std::nullopt;
}

inline bool IsBananzaUpscaleDispatch(u64 hash, bool indirect, Extent2D scene, u64 blend_target,
                                     u64 tonemapped_scene, u64 scene_depth,
                                     std::span<const MarioKartBlendImage> sampled,
                                     std::span<const MarioKartBlendImage> written,
                                     std::span<const u32> constants) {
    using VideoCore::Surface::PixelFormat;
    if (hash != BananzaUpscaleShader || indirect || sampled.size() != 1 || written.size() != 1 ||
        constants.size() < 12 || !blend_target || !tonemapped_scene || !scene_depth ||
        scene.width < 640 || scene.height < 360 || scene.width >= BananzaOutputExtent.width ||
        scene.height >= BananzaOutputExtent.height) {
        return false;
    }
    const auto& input = sampled.front();
    const auto& output = written.front();
    return input.identity == blend_target && input.format == PixelFormat::B10G11R11_FLOAT &&
           Extent2D{input.width, input.height} == scene && output.identity != 0 &&
           output.identity != blend_target && output.identity != tonemapped_scene &&
           output.identity != scene_depth && output.format == PixelFormat::B10G11R11_FLOAT &&
           Extent2D{output.width, output.height} == BananzaOutputExtent &&
           constants[6] == scene.width - 1 && constants[7] == scene.height - 1 &&
           constants[10] == output.width - 1 && constants[11] == output.height - 1;
}

constexpr u32 BananzaJitterPhaseCount(u32 width, bool dlss) {
    const u32 minimum = dlss ? 16U : 8U;
    if (width == 0) {
        return minimum;
    }
    const f32 ratio = static_cast<f32>(BananzaOutputExtent.width) / static_cast<f32>(width);
    return std::max(minimum, static_cast<u32>(8.0f * ratio * ratio));
}

} // namespace VideoCommon
