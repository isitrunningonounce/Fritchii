// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstdlib>

#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_render_pass_cache.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

/// Lichii: the game's pipelines become shader objects (VK_EXT_shader_object): each stage compiled
/// on its own, every draw state set at draw time, so a new combination of shaders and state costs
/// no driver compile. LICHII_SHADER_OBJECTS=1 turns it on (and dynamic rendering with it).
inline bool UseShaderObjects(const Device& device) {
    static const bool requested = [] {
        const char* value = std::getenv("LICHII_SHADER_OBJECTS");
        return value != nullptr && value[0] == '1';
    }();
    return requested && device.IsExtShaderObjectSupported() && !device.IsTiler();
}

/// Lichii: the game's draws run in dynamic rendering instances (VK_KHR_dynamic_rendering) instead
/// of render passes, which shader objects need. LICHII_DYNAMIC_RENDERING=1 turns it on. Tilers
/// keep render passes (their MSAA resolve attachments live there).
inline bool UseDynamicRendering(const Device& device) {
    static const bool requested = [] {
        const char* value = std::getenv("LICHII_DYNAMIC_RENDERING");
        return value != nullptr && value[0] == '1';
    }();
    return (requested && device.IsDynamicRenderingSupported() && !device.IsTiler()) ||
           UseShaderObjects(device);
}

/// The attachment formats of a render pass key as dynamic rendering takes them: colour slots up to
/// the last one used (unused ones VK_FORMAT_UNDEFINED), then depth and stencil by aspect. The
/// pipeline and the rendering instance both derive theirs from this, so they always match.
struct DynamicRenderingFormats {
    std::array<VkFormat, 8> colors{};
    u32 color_count{};
    VkFormat depth{VK_FORMAT_UNDEFINED};
    VkFormat stencil{VK_FORMAT_UNDEFINED};
};

inline DynamicRenderingFormats MakeDynamicRenderingFormats(const Device& device,
                                                           const RenderPassKey& key) {
    using VideoCore::Surface::PixelFormat;
    using VideoCore::Surface::SurfaceType;
    DynamicRenderingFormats result;
    for (size_t index = 0; index < key.color_formats.size(); ++index) {
        const PixelFormat format = key.color_formats[index];
        if (format == PixelFormat::Invalid) {
            result.colors[index] = VK_FORMAT_UNDEFINED;
            continue;
        }
        result.colors[index] =
            MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, true, format).format;
        result.color_count = static_cast<u32>(index + 1);
    }
    if (key.depth_format != PixelFormat::Invalid) {
        const VkFormat format =
            MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, true, key.depth_format).format;
        const SurfaceType type = VideoCore::Surface::GetFormatType(key.depth_format);
        if (type == SurfaceType::Depth || type == SurfaceType::DepthStencil) {
            result.depth = format;
        }
        if (type == SurfaceType::Stencil || type == SurfaceType::DepthStencil) {
            result.stencil = format;
        }
    }
    return result;
}

} // namespace Vulkan
