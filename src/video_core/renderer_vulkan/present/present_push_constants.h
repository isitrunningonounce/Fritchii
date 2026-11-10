// SPDX-FileCopyrightText: Copyright 2024 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/common_types.h"
#include "common/math_util.h"

namespace Vulkan {

struct ScreenRectVertex {
    ScreenRectVertex() = default;
    explicit ScreenRectVertex(f32 x, f32 y, f32 u, f32 v) : position{{x, y}}, tex_coord{{u, v}} {}

    std::array<f32, 2> position;
    std::array<f32, 2> tex_coord;
};

inline std::array<ScreenRectVertex, 4> MakeScreenVertices(const Common::Rectangle<u32>& screen,
                                                        const Common::Rectangle<f32>& crop) {
    const auto x = f32(screen.left);
    const auto y = f32(screen.top);
    const auto w = f32(screen.GetWidth());
    const auto h = f32(screen.GetHeight());
    return {ScreenRectVertex(x, y, crop.left, crop.top),
            ScreenRectVertex(x + w, y, crop.right, crop.top),
            ScreenRectVertex(x, y + h, crop.left, crop.bottom),
            ScreenRectVertex(x + w, y + h, crop.right, crop.bottom)};
}

static inline std::array<f32, 4 * 4> MakeOrthographicMatrix(f32 width, f32 height) {
    // clang-format off
    return { 2.f / width, 0.f,          0.f, 0.f,
             0.f,         2.f / height, 0.f, 0.f,
             0.f,         0.f,          1.f, 0.f,
            -1.f,        -1.f,          0.f, 1.f};
    // clang-format on
}

struct PresentPushConstants {
    std::array<f32, 4 * 4> modelview_matrix;
    std::array<ScreenRectVertex, 4> vertices;
};

static_assert(sizeof(PresentPushConstants) <= 128, "Push constants are too large");

} // namespace Vulkan
