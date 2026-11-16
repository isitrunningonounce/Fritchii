// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "core/arm/bananza_nvn2_resources.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

/// How a texture's guest bytes become the bytes its image is filled from: unswizzled level by
/// level, ASTC decoded. Planned by the thread that owns the image; Run touches only its
/// arguments, so it may run on any thread.
class BananzaNvn2TextureFill {
public:
    /// Nothing when the format or the layout is not one the replay fills, or the bytes are
    /// too few.
    static std::optional<BananzaNvn2TextureFill> Plan(
        const Core::BananzaNvn2Hle::ResourceTextureInfo& texture, u32 layers, u32 depth,
        VkImageAspectFlags aspects, size_t data_size);

    /// Fills target, of Size() bytes, from the texture's guest bytes.
    void Run(std::span<const u8> data, std::span<u8> target) const;

    [[nodiscard]] size_t Size() const {
        return upload_size;
    }
    /// The regions of the filled buffer, one per level.
    [[nodiscard]] const std::vector<VkBufferImageCopy>& Copies() const {
        return copies;
    }

private:
    struct Level {
        size_t guest_offset;
        size_t bytes_per_layer;
        size_t upload_stride;
        u32 width;
        u32 height;
        u32 tile_height;
        u32 tile_depth;
    };

    std::vector<Level> levels;
    std::vector<VkBufferImageCopy> copies;
    size_t upload_size{};
    size_t stride{};
    u32 layers{};
    u32 block_bytes{};
    u32 block_width{};
    u32 block_height{};
    bool decode_astc{};
};

} // namespace Vulkan
