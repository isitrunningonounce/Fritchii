// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include "common/common_types.h"

namespace VideoCommon {

// Row-major matrices for row-vector multiplication, with device depth in [0, 1].
struct TemporalCameraConstants {
    std::array<f32, 16> view_to_clip;
    std::array<f32, 16> clip_to_view;
    std::array<f32, 16> clip_to_previous_clip;
    std::array<f32, 16> previous_clip_to_clip;
    std::array<f32, 3> position;
    std::array<f32, 3> right;
    std::array<f32, 3> up;
    std::array<f32, 3> forward;
    f32 near_plane;
    f32 far_plane;
    f32 vertical_fov;
    f32 aspect_ratio;
};

} // namespace VideoCommon
