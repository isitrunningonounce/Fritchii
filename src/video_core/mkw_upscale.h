// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <span>

#include "common/common_types.h"
#include "video_core/ocarina_upscale.h"
#include "video_core/surface.h"

namespace VideoCommon {

inline constexpr u64 MarioKartBlendFragment = 0x40c24603fc78bacaULL;

constexpr OcarinaUpscaleBackend SelectMarioKartUpscaler(bool dlss_enabled, bool dlss_available,
                                                       std::string_view override) {
    return SelectOcarinaUpscaler(false, dlss_enabled, dlss_available, override);
}

struct MarioKartCamera {
    std::array<f32, 16> current;
    std::array<f32, 16> previous;
};

inline std::optional<MarioKartCamera> ReadMarioKartCamera(std::span<const f32> block,
                                                         u32 width, u32 height,
                                                         bool gpu_modified) {
    if (gpu_modified || block.size_bytes() < 0x1c0 || width < 640 || height < 360 ||
        block[0x120 / 4] != static_cast<f32>(width) ||
        block[0x124 / 4] != static_cast<f32>(height) ||
        block[0xa8 / 4] != -1.0f || block[0xac / 4] != 0.0f) {
        return std::nullopt;
    }
    MarioKartCamera camera;
    std::copy_n(block.begin() + 0x30 / 4, 16, camera.current.begin());
    std::copy_n(block.begin() + 0x180 / 4, 16, camera.previous.begin());
    const auto finite = [](f32 value) { return std::isfinite(value); };
    if (!std::ranges::all_of(camera.current, finite) ||
        !std::ranges::all_of(camera.previous, finite)) {
        return std::nullopt;
    }
    return camera;
}

struct MarioKartBlendImage {
    u64 identity;
    VideoCore::Surface::PixelFormat format;
    u32 width;
    u32 height;
};

inline bool IsMarioKartBlend(u64 fragment, bool indexed, bool indirect, u32 vertices,
                            u32 instances, std::span<const MarioKartBlendImage> sampled,
                            const MarioKartBlendImage& target, u32 width, u32 height) {
    using VideoCore::Surface::PixelFormat;
    if (fragment != MarioKartBlendFragment || indexed || indirect || vertices != 3 ||
        instances != 1 || sampled.size() != 2 || width < 640 || height < 360 ||
        target.format != PixelFormat::B10G11R11_FLOAT || target.width != width ||
        target.height != height || target.identity == 0) {
        return false;
    }
    u32 colors{}, weights{};
    for (const auto& image : sampled) {
        if (!image.identity || image.identity == target.identity || image.width != width ||
            image.height != height) {
            return false;
        }
        colors += image.format == PixelFormat::B10G11R11_FLOAT;
        weights += image.format == PixelFormat::A8B8G8R8_UNORM;
    }
    return colors == 1 && weights == 1 && sampled[0].identity != sampled[1].identity;
}

} // namespace VideoCommon
