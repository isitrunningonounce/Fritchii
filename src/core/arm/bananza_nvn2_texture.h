// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>

#include "common/common_types.h"

namespace Core::BananzaNvn2Hle {

struct TextureDescription {
    u32 flags{};
    u32 target{1};
    u32 width{1};
    u32 height{1};
    u32 depth{1};
    u32 levels{1};
    u32 format{};
    u32 samples{};
    u32 stride{};
    u32 depth_mode{};
    std::array<u32, 4> swizzle{2, 3, 4, 5};
    std::optional<std::array<u8, 8>> packaged_layout;
};

struct TextureFormatBlock {
    u32 bytes{};
    u32 width{1};
    u32 height{1};
};

struct TextureLayout {
    u64 size{};
    u64 layer_size{};
    u64 stride{};
    u32 alignment{};
    u32 layers{1};
    u32 dimensions{};
    u32 tile_height{};
    u32 tile_depth{};
};

std::optional<TextureFormatBlock> GetTextureFormatBlock(u32 format);
std::optional<TextureLayout> ComputeTextureLayout(const TextureDescription& description);
std::optional<u64> TextureLevelOffset(const TextureDescription& description, u32 level);

} // namespace Core::BananzaNvn2Hle
