// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstdint>
#include <vector>

#include "video_core/host_shaders/mkw_ui_alpha_comp_spv.h"
#include "video_core/renderer_vulkan/present/mkw_ui_alpha.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {
namespace {
VkImageMemoryBarrier Barrier(VkImage image, VkAccessFlags src, VkAccessFlags dst,
                             VkImageLayout before, VkImageLayout after) {
    return {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = src,
        .dstAccessMask = dst,
        .oldLayout = before,
        .newLayout = after,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
}
} // namespace

bool MarioKartUiAlpha::Supported(const Device& device, VkFormat format) {
    // Sampling an sRGB view would compare linear values instead of the encoded bytes.
    if (!device.IsStorageImageExtendedFormatsSupported() ||
        (format != VK_FORMAT_B8G8R8A8_UNORM && format != VK_FORMAT_R8G8B8A8_UNORM)) {
        return false;
    }
    const auto& physical = device.GetPhysical();
    const auto queues = physical.GetQueueFamilyProperties();
    constexpr auto mask_features =
        VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    return (queues.at(device.GetGraphicsFamily()).queueFlags & VK_QUEUE_COMPUTE_BIT) &&
           (physical.GetFormatProperties(VK_FORMAT_R8_UNORM).optimalTilingFeatures &
            mask_features) == mask_features &&
           (physical.GetFormatProperties(format).optimalTilingFeatures &
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);
}

MarioKartUiAlpha::MarioKartUiAlpha(const Device& device, u32 slots)
    : sampler{CreateWrappedSampler(device, VK_FILTER_NEAREST)},
      pool{CreateWrappedDescriptorPool(
          device, slots * 2, slots,
          {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE})},
      set_layout{CreateWrappedDescriptorSetLayout(device,
                                                  {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                   VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                   VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
                                                  VK_SHADER_STAGE_COMPUTE_BIT)} {
    const std::vector<VkDescriptorSetLayout> layouts(slots, *set_layout);
    sets = CreateWrappedDescriptorSets(pool, layouts);
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(u32) * 4};
    layout = device.GetLogical().CreatePipelineLayout({
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = set_layout.address(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push,
    });
    const auto shader = CreateWrappedShaderModule(device, MKW_UI_ALPHA_COMP_SPV);
    pipeline = CreateWrappedComputePipeline(device, layout, *shader);
}

void MarioKartUiAlpha::Prepare(const Device& device, MemoryAllocator& allocator,
                               MarioKartUiImages& images, VkExtent2D extent, VkFormat format,
                               VkImageView hudless, u32 slot) {
    if (!images.color || images.extent.width != extent.width ||
        images.extent.height != extent.height || images.format != format) {
        const auto create = [&](VkFormat image_format, VkImageUsageFlags usage) {
            return allocator.CreateImage({
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = image_format,
                .extent = {extent.width, extent.height, 1},
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = usage,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            });
        };
        MarioKartUiImages next;
        next.color =
            create(format, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                               VK_IMAGE_USAGE_SAMPLED_BIT);
        next.color_view = CreateWrappedImageView(device, next.color, format);
        next.alpha =
            create(VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        next.alpha_view = CreateWrappedImageView(device, next.alpha, VK_FORMAT_R8_UNORM);
        next.extent = extent;
        next.format = format;
        images.color_view.reset();
        images.alpha_view.reset();
        images = std::move(next);
    }
    const std::array infos{
        VkDescriptorImageInfo{*sampler, *images.color_view, VK_IMAGE_LAYOUT_GENERAL},
        VkDescriptorImageInfo{*sampler, hudless, VK_IMAGE_LAYOUT_GENERAL},
        VkDescriptorImageInfo{VK_NULL_HANDLE, *images.alpha_view, VK_IMAGE_LAYOUT_GENERAL},
    };
    std::array<VkWriteDescriptorSet, 3> writes{};
    for (u32 i = 0; i < writes.size(); ++i) {
        writes[i] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sets[slot],
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = i == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                     : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &infos[i],
        };
    }
    device.GetLogical().UpdateDescriptorSets(writes, {});
}

void MarioKartUiAlpha::Record(vk::CommandBuffer cmdbuf, const MarioKartUiImages& images,
                              VkImage source, VkImage hudless, const VkImageBlit& blit,
                              const VkImageCopy& copy, bool use_blit, Common::Rectangle<u32> rect,
                              u32 slot) {
    constexpr auto GENERAL = VK_IMAGE_LAYOUT_GENERAL;
    constexpr auto SRC = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    constexpr auto DST = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    const std::array before{
        Barrier(source, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT, GENERAL, SRC),
        Barrier(*images.color, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, DST),
        Barrier(*images.alpha, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, GENERAL),
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                           {}, {}, before);
    if (use_blit) {
        cmdbuf.BlitImage(source, SRC, *images.color, DST, blit, VK_FILTER_LINEAR);
    } else {
        cmdbuf.CopyImage(source, SRC, *images.color, DST, copy);
    }
    const std::array ready{
        Barrier(source, VK_ACCESS_TRANSFER_READ_BIT,
                VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, SRC, GENERAL),
        Barrier(*images.color, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, DST,
                GENERAL),
        Barrier(hudless, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, GENERAL, GENERAL),
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                           {}, {}, ready);
    cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
    cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, sets[slot], {});
    const std::array push{rect.left, rect.top, rect.right, rect.bottom};
    cmdbuf.PushConstants(*layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push.data());
    cmdbuf.Dispatch((images.extent.width + 15) / 16, (images.extent.height + 15) / 16, 1);
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           0,
                           Barrier(*images.alpha, VK_ACCESS_SHADER_WRITE_BIT,
                                   VK_ACCESS_SHADER_READ_BIT, GENERAL, GENERAL));
}

} // namespace Vulkan
