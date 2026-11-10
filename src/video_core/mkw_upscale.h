// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <span>

#include "common/common_types.h"
#include "video_core/ocarina_upscale.h"
#include "video_core/surface.h"
#include "video_core/temporal_camera.h"

namespace VideoCommon {

inline constexpr u64 MarioKartBlendFragment = 0x40c24603fc78bacaULL;

constexpr OcarinaUpscaleBackend SelectMarioKartUpscaler(bool dlss_enabled, bool dlss_available,
                                                       std::string_view override) {
    return SelectOcarinaUpscaler(false, dlss_enabled, dlss_available, override);
}

struct MarioKartCamera {
    std::array<f32, 16> current;
    std::array<f32, 16> previous;
    std::array<f32, 16> projection;
    std::array<f32, 16> view;
    std::array<f32, 16> camera_to_world;
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
    MarioKartCamera camera{};
    std::copy_n(block.begin() + 0x70 / 4, 16, camera.projection.begin());
    std::copy_n(block.begin(), 12, camera.view.begin());
    std::copy_n(block.begin() + 0xb0 / 4, 12, camera.camera_to_world.begin());
    camera.view[15] = camera.camera_to_world[15] = 1.0f;
    std::copy_n(block.begin() + 0x30 / 4, 16, camera.current.begin());
    std::copy_n(block.begin() + 0x180 / 4, 16, camera.previous.begin());
    const auto finite = [](f32 value) { return std::isfinite(value); };
    if (!std::ranges::all_of(camera.current, finite) ||
        !std::ranges::all_of(camera.previous, finite)) {
        return std::nullopt;
    }
    return camera;
}

namespace MarioKartCameraDetail {

using Matrix = std::array<f64, 16>;
inline constexpr Matrix Identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

inline Matrix Widen(const std::array<f32, 16>& source) {
    Matrix result;
    std::copy(source.begin(), source.end(), result.begin());
    return result;
}

inline Matrix Multiply(const Matrix& a, const Matrix& b) {
    Matrix result{};
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 4; ++col) {
            for (size_t k = 0; k < 4; ++k) {
                result[row * 4 + col] += a[row * 4 + k] * b[k * 4 + col];
            }
        }
    }
    return result;
}

inline std::optional<Matrix> Inverse(const Matrix& matrix) {
    std::array<std::array<f64, 8>, 4> rows{};
    for (size_t i = 0; i < 4; ++i) {
        std::copy_n(matrix.begin() + i * 4, 4, rows[i].begin());
        rows[i][i + 4] = 1.0;
    }
    for (size_t col = 0; col < 4; ++col) {
        const auto pivot = std::max_element(rows.begin() + col, rows.end(),
            [col](const auto& a, const auto& b) { return std::abs(a[col]) < std::abs(b[col]); });
        std::swap(rows[col], *pivot);
        const f64 divisor = rows[col][col];
        if (!std::isfinite(divisor) || std::abs(divisor) < 1e-10) {
            return std::nullopt;
        }
        for (auto& value : rows[col]) {
            value /= divisor;
        }
        for (size_t row = 0; row < 4; ++row) {
            if (row != col) {
                const f64 scale = rows[row][col];
                for (size_t i = 0; i < 8; ++i) {
                    rows[row][i] -= scale * rows[col][i];
                }
            }
        }
    }
    Matrix result;
    for (size_t i = 0; i < 4; ++i) {
        std::copy_n(rows[i].begin() + 4, 4, result.begin() + i * 4);
    }
    return result;
}

inline bool Near(const Matrix& a, const Matrix& b, f64 tolerance) {
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || std::abs(a[i] - b[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

inline bool NearProduct(const Matrix& a, const Matrix& b, const Matrix& expected) {
    const auto actual = Multiply(a, b);
    for (size_t i = 0; i < 16; ++i) {
        f64 tolerance = 1e-5;
        if (i % 4 == 3) {
            // Translation cancellation grows with distance from the world origin.
            f64 magnitude = std::abs(expected[i]);
            for (size_t k = 0; k < 4; ++k) {
                magnitude += std::abs(a[(i / 4) * 4 + k] * b[k * 4 + 3]);
            }
            tolerance = std::max(0.002, 4.0 * std::numeric_limits<f32>::epsilon() * magnitude);
        }
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
            !std::isfinite(tolerance) || std::abs(actual[i] - expected[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

inline std::array<f32, 16> PackRowVectorMatrix(const Matrix& matrix) {
    std::array<f32, 16> result;
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 4; ++col) {
            result[row * 4 + col] = static_cast<f32>(matrix[col * 4 + row]);
        }
    }
    return result;
}

} // namespace MarioKartCameraDetail

inline std::optional<TemporalCameraConstants> MakeMarioKartTemporalCamera(
    const MarioKartCamera& camera, u32 width, u32 height) {
    using namespace MarioKartCameraDetail;
    if (width == 0 || height == 0) {
        return std::nullopt;
    }
    const auto projection = Widen(camera.projection);
    const auto view = Widen(camera.view);
    const auto pose = Widen(camera.camera_to_world);
    const auto current = Widen(camera.current);
    const auto previous = Widen(camera.previous);
    if (!NearProduct(projection, view, current) ||
        !NearProduct(view, pose, Identity) || !NearProduct(pose, view, Identity)) {
        return std::nullopt;
    }
    for (size_t row = 0; row < 3; ++row) {
        for (size_t col = 0; col < 3; ++col) {
            f64 dot{};
            for (size_t k = 0; k < 3; ++k) {
                dot += pose[k * 4 + row] * pose[k * 4 + col];
            }
            if (!std::isfinite(dot) || std::abs(dot - (row == col ? 1.0 : 0.0)) > 0.001) {
                return std::nullopt;
            }
        }
    }
    // Captured RH perspective, allowing its constant off-axis x/y shift.
    constexpr std::array zero_indices{1, 3, 4, 7, 8, 9, 12, 13, 15};
    if (projection[14] != -1.0 || projection[0] <= 0 || projection[5] <= 0 ||
        std::ranges::any_of(zero_indices, [&](size_t i) { return projection[i] != 0.0; })) {
        return std::nullopt;
    }
    const f64 near_plane = projection[11] / (projection[10] - 1.0);
    const f64 far_plane = projection[11] / (projection[10] + 1.0);
    const f64 aspect = projection[5] / projection[0];
    const f64 fov = 2.0 * std::atan(1.0 / projection[5]);
    if (!std::isfinite(near_plane) || !std::isfinite(far_plane) || !std::isfinite(fov) ||
        near_plane <= 0 || far_plane <= near_plane || far_plane > 1e8 ||
        std::abs(aspect - static_cast<f64>(width) / height) > 0.001) {
        return std::nullopt;
    }
    // The motion shader consumes GL clip z = 2*depth-1; SL consumes the tagged device depth.
    constexpr Matrix gl_to_device{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 1};
    constexpr Matrix device_to_gl{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, -1, 0, 0, 0, 1};
    const auto inverse_current = Inverse(current);
    const auto device_projection = Multiply(gl_to_device, projection);
    const auto inverse_projection = Inverse(device_projection);
    if (!inverse_current || !inverse_projection) {
        return std::nullopt;
    }
    const auto reprojection = Multiply(Multiply(gl_to_device, Multiply(previous, *inverse_current)),
                                       device_to_gl);
    const auto inverse_reprojection = Inverse(reprojection);
    if (!inverse_reprojection || !Near(Multiply(reprojection, *inverse_reprojection), Identity, 1e-6)) {
        return std::nullopt;
    }
    const auto axis = [&](size_t column, f64 sign) {
        const f64 length = std::hypot(pose[column], pose[4 + column], pose[8 + column]);
        return std::array<f32, 3>{static_cast<f32>(sign * pose[column] / length),
                                  static_cast<f32>(sign * pose[4 + column] / length),
                                  static_cast<f32>(sign * pose[8 + column] / length)};
    };
    return TemporalCameraConstants{
        PackRowVectorMatrix(device_projection), PackRowVectorMatrix(*inverse_projection),
        PackRowVectorMatrix(reprojection), PackRowVectorMatrix(*inverse_reprojection),
        {camera.camera_to_world[3], camera.camera_to_world[7], camera.camera_to_world[11]},
        axis(0, 1.0), axis(1, 1.0), axis(2, -1.0),
        static_cast<f32>(near_plane), static_cast<f32>(far_plane),
        static_cast<f32>(fov), static_cast<f32>(aspect),
    };
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
