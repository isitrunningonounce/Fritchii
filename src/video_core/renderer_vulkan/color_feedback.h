// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>

#include "shader_recompiler/shader_info.h"
#include "video_core/texture_cache/texture_cache_base.h"

#include "video_core/renderer_vulkan/vk_update_descriptor.h"

namespace Vulkan {

// Run on the draw's payload after memo replay/capture. The memo keeps only guest image views.
template <typename Copy>
void ApplyColorFeedbackCopies(std::span<const Shader::TextureDescriptor> textures,
                              std::span<const VideoCommon::ImageViewInOut> views,
                              std::span<DescriptorUpdateEntry> descriptors, Copy&& copy) {
    size_t index{};
    for (const auto& texture : textures) {
        for (u32 element = 0; element < texture.count; ++element, ++index) {
            if (texture.type != Shader::TextureType::Color2D || texture.is_depth ||
                texture.is_multisample) {
                continue;
            }
            if (const VkImageView replacement = copy(views[index].id)) {
                descriptors[index].image.imageView = replacement;
            }
        }
    }
}

} // namespace Vulkan
