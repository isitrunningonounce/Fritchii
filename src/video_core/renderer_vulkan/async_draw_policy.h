// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>

#include "common/common_types.h"

namespace Vulkan {

/// Whether a draw whose graphics pipeline is still building may be skipped (asynchronous
/// shaders) instead of waiting for the build.
/// @param native         The draw comes from a Switch 2 (native Turing macro) program.
/// @param depth_enabled  The depth test is on for the draw.
/// @param index_count    Index count of the draw.
/// @param vertex_count   Vertex count of the draw.
/// @param discard_native Let native draws follow the heuristic too (on unless EDEN_NX2_ASYNC_DISCARD=0).
/// @param recurring_target Every render target of the draw is cleared and redrawn every frame
///                         (EDEN_NX2_ASYNC_DISCARD=2 asks; the other modes pass true). Native
///                         draws into any other target wait.
/// @return True when the draw may be skipped.
constexpr bool CanDiscardPendingDraw(bool native, bool depth_enabled, u32 index_count,
                                    u32 vertex_count, bool discard_native = false,
                                    bool recurring_target = true) {
    // Native draws can produce persistent resources, even with depth enabled. They are skipped
    // unless EDEN_NX2_ASYNC_DISCARD=0, trading a possible missing effect for no stall while a
    // new pipeline builds. With EDEN_NX2_ASYNC_DISCARD=2 only those into targets the frame
    // clears and redraws are: a skipped draw into a layer rendered once is missing for good
    // (title screen text, chat bubbles), one into a per-frame target for a frame.
    if (native && (!discard_native || !recurring_target)) {
        return false;
    }
    // Preserve the legacy async-shader heuristic.
    return depth_enabled || (index_count > 6 && vertex_count > 6);
}

// Which native (Switch 2) draws may be skipped while their pipeline builds, picked with
// LICHII_NATIVE_DISCARD. Skipping can lose persistent resources, so the default keeps them all.
enum class NativeDiscardMode : u8 {
    Never,  // hold every native draw until its pipeline is built (LICHII_NATIVE_DISCARD=never)
    Depth,  // skip native draws with depth enabled (scene geometry)
    Legacy, // default, the legacy heuristic: depth enabled, or more than 6 indices and vertices
    Always, // skip every native draw whose pipeline is not built
};

constexpr bool CanDiscardPendingDraw(bool native, bool depth_enabled, u32 index_count,
                                    u32 vertex_count, NativeDiscardMode native_mode) {
    if (native) {
        switch (native_mode) {
        case NativeDiscardMode::Never:
            return false;
        case NativeDiscardMode::Depth:
            return depth_enabled;
        case NativeDiscardMode::Always:
            return true;
        case NativeDiscardMode::Legacy:
            break;
        }
    }
    return depth_enabled || (index_count > 6 && vertex_count > 6);
}

/// Draws skipped or deferred for unready pipelines since the last call; the presenter reads it
/// to keep frames with missing draws off the screen (LICHII_HIDE_GLITCH_FRAMES).
u32 TakeSkippedDrawsSinceLastFrame() noexcept;

constexpr bool DefersShaderEmission(bool enabled, bool native, size_t pending, size_t capacity) {
    return enabled && native && pending < capacity;
}

} // namespace Vulkan
