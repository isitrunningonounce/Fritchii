// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/math_util.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;

struct MarioKartUiImages {
    vk::Image color;
    vk::ImageView color_view;
    vk::Image alpha;
    vk::ImageView alpha_view;
    VkExtent2D extent{};
    VkFormat format{};
};

// Owned and recorded only by the presenter. Each slot is retired with its SL inputs fence.
class MarioKartUiAlpha {
public:
    MarioKartUiAlpha(const Device& device, u32 slots);
    static bool Supported(const Device& device, VkFormat format);
    void Prepare(const Device& device, MemoryAllocator& allocator, MarioKartUiImages& images,
                 VkExtent2D extent, VkFormat format, VkImageView hudless, u32 slot);
    void Record(vk::CommandBuffer cmdbuf, const MarioKartUiImages& images, VkImage source,
                VkImage hudless, const VkImageBlit& blit, const VkImageCopy& copy, bool use_blit,
                Common::Rectangle<u32> rect, u32 slot);

private:
    vk::Sampler sampler;
    vk::DescriptorPool pool;
    vk::DescriptorSetLayout set_layout;
    vk::DescriptorSets sets;
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
};

} // namespace Vulkan
