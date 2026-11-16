// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <numeric>

#include "core/arm/bananza_nvn2_texture.h"
#include "video_core/renderer_vulkan/vk_bananza_nvn2_texture_fill.h"
#include "video_core/textures/astc.h"
#include "video_core/textures/decoders.h"

namespace Vulkan {

std::optional<BananzaNvn2TextureFill> BananzaNvn2TextureFill::Plan(
    const Core::BananzaNvn2Hle::ResourceTextureInfo& texture, u32 layers, u32 depth,
    VkImageAspectFlags aspects, size_t data_size) {
    const auto& d = texture.description;
    const auto block = Core::BananzaNvn2Hle::GetTextureFormatBlock(d.format);
    if (!block || (texture.layout.stride && d.levels != 1)) {
        return std::nullopt;
    }
    BananzaNvn2TextureFill fill;
    fill.decode_astc = d.format >= 0x79 && d.format <= 0x94;
    fill.stride = texture.layout.stride;
    fill.layers = layers;
    fill.block_bytes = block->bytes;
    fill.block_width = block->width;
    fill.block_height = block->height;
    for (u32 level = 0; level < d.levels; ++level) {
        const auto offset = Core::BananzaNvn2Hle::TextureLevelOffset(d, level);
        const u32 width = std::max(d.width >> level, 1U);
        const u32 height = std::max(d.height >> level, 1U);
        const u32 level_depth = std::max(depth >> level, 1U);
        const u32 columns = (width + block->width - 1) / block->width;
        const u32 rows = (height + block->height - 1) / block->height;
        u32 tile = texture.layout.tile_height;
        while (tile && (8U << (tile - 1)) >= rows) {
            --tile;
        }
        u32 tile_depth = texture.layout.tile_depth;
        while (tile_depth && (1U << (tile_depth - 1)) >= level_depth) {
            --tile_depth;
        }
        const size_t bytes = texture.layout.stride
                                 ? texture.layout.stride * (rows - 1) + columns * block->bytes
                                 : Tegra::Texture::CalculateSize(true, block->bytes, columns, rows,
                                                                  level_depth, tile, tile_depth);
        const auto last_layer = (layers - 1) * texture.layout.layer_size;
        if (!offset || last_layer > data_size || *offset > data_size - last_layer ||
            bytes > data_size - last_layer - *offset) {
            return std::nullopt;
        }
        const size_t alignment =
            fill.decode_astc ? 4 : std::lcm(size_t{4}, size_t{block->bytes});
        fill.upload_size = (fill.upload_size + alignment - 1) / alignment * alignment;
        fill.copies.push_back({
            .bufferOffset = fill.upload_size,
            .imageSubresource = {aspects, level, 0, layers},
            .imageExtent = {width, height, level_depth},
        });
        const auto upload_stride = fill.decode_astc
                                       ? size_t{width} * height * level_depth * 4
                                       : size_t{columns} * rows * level_depth * block->bytes;
        fill.levels.push_back({*offset, texture.layout.layer_size, upload_stride, columns, rows,
                               tile, tile_depth});
        fill.upload_size += upload_stride * layers;
    }
    return fill;
}

void BananzaNvn2TextureFill::Run(std::span<const u8> data, std::span<u8> target) const {
    for (u32 level = 0; level < levels.size(); ++level) {
        const auto& src = levels[level];
        const size_t row = size_t{src.width} * block_bytes;
        const auto extent = copies[level].imageExtent;
        std::vector<u8> astc(decode_astc ? row * src.height * extent.depth : 0);
        for (u32 layer = 0; layer < layers; ++layer) {
            const size_t offset = src.guest_offset + layer * src.bytes_per_layer;
            auto linear = target.subspan(copies[level].bufferOffset + layer * src.upload_stride,
                                         src.upload_stride);
            auto dest = decode_astc ? std::span{astc} : linear;
            if (stride) {
                for (u32 y = 0; y < src.height; ++y) {
                    std::memcpy(dest.data() + row * y, data.data() + offset + stride * y, row);
                }
            } else {
                Tegra::Texture::UnswizzleTexture(dest, data.subspan(offset), block_bytes,
                                                src.width, src.height, extent.depth,
                                                src.tile_height, src.tile_depth, 0, true);
            }
            if (decode_astc) {
                Tegra::Texture::ASTC::Decompress(astc, extent.width, extent.height, extent.depth,
                                                block_width, block_height, linear);
            }
        }
    }
}

} // namespace Vulkan
