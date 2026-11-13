// SPDX-License-Identifier: Zlib
//
// Altered C++ port of Recomp runtime/nvn_tex.c at
// fac48bcdc5ec0edd6296bc3672bdebb540d5958d, via MacNativeGames b75be8cf.
// Based on deko3d source/dk_image.cpp. This version adds checked input limits and
// optional results; it does not implement sparse or video-decoder texture layouts.
//
// Copyright (C) 2018-2020 fincs
//
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any
// damages arising from the use of this software.
//
// Permission is granted to anyone to use this software for any
// purpose, including commercial applications, and to alter it and
// redistribute it freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you
//    must not claim that you wrote the original software. If you use
//    this software in a product, an acknowledgment in the product
//    documentation would be appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and
//    must not be misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source
//    distribution.

#include <algorithm>

#include "core/arm/bananza_nvn2_texture.h"

namespace Core::BananzaNvn2Hle {
namespace {

u64 DivideUp(u64 value, u64 divisor) {
    return (value + divisor - 1) / divisor;
}

u64 AlignUp(u64 value, u64 alignment) {
    return DivideUp(value, alignment) * alignment;
}

u64 Mip(u64 value, u32 level) {
    return std::max<u64>(value >> level, 1);
}

u32 PickTile(u64 gobs) {
    return gobs >= 16 ? 4 : gobs >= 8 ? 3 : gobs >= 4 ? 2 : gobs >= 2 ? 1 : 0;
}

u32 AdjustTile(u32 shift, u64 unit, u64 dimension) {
    while (shift && (unit << (shift - 1)) >= dimension) {
        --shift;
    }
    return shift;
}

struct Shape {
    u64 width;
    u64 height;
    u64 depth;
    u32 dimensions{2};
    bool layered{};
};

Shape GetShape(const TextureDescription& d) {
    Shape shape{d.width, d.height, d.depth};
    switch (d.target) {
    case 0: shape.dimensions = 1; shape.height = shape.depth = 1; break;
    case 3:
        shape.dimensions = 1;
        shape.depth = d.height;
        shape.height = 1;
        shape.layered = true;
        break;
    case 2: shape.dimensions = 3; break;
    case 4:
    case 6:
    case 9: shape.layered = true; break;
    case 8: shape.depth = 6; shape.layered = true; break;
    default: shape.depth = 1; break;
    }
    return shape;
}

std::array<u32, 2> SampleShape(u32 samples) {
    switch (samples) {
    case 2: return {2, 1};
    case 4: return {2, 2};
    case 8: return {4, 2};
    default: return {1, 1};
    }
}

u64 LevelBytes(const TextureDescription& d, const Shape& shape, const TextureFormatBlock& format,
               u32 levels, u32 tile_height, u32 tile_depth) {
    const auto samples = SampleShape(d.samples);
    u64 size{};
    for (u32 level = 0; level < levels; ++level) {
        const u64 width = DivideUp(Mip(shape.width * samples[0], level), format.width);
        const u64 height = shape.dimensions >= 2
                               ? DivideUp(Mip(shape.height * samples[1], level), format.height)
                               : 1;
        const u64 depth = shape.dimensions == 3 ? Mip(shape.depth, level) : 1;
        const u32 th = AdjustTile(tile_height, 8, height);
        const u32 td = AdjustTile(tile_depth, 1, depth);
        size += DivideUp(width * format.bytes, 64) *
                DivideUp(DivideUp(height, 8), 1ULL << th) * DivideUp(depth, 1ULL << td) *
                (1ULL << (9 + th + td));
    }
    return size;
}

u64 LayerBytes(const TextureDescription& d, const Shape& shape, const TextureFormatBlock& format,
               u32 tile_height, u32 tile_depth) {
    const u64 size = LevelBytes(d, shape, format, d.levels, tile_height, tile_depth);
    if (!shape.layered) {
        return size;
    }
    const u32 th = AdjustTile(tile_height, 8, DivideUp(shape.height, format.height));
    const u32 td = AdjustTile(tile_depth, 1, shape.depth);
    return AlignUp(size, 1ULL << (9 + th + td));
}

} // Anonymous namespace

std::optional<TextureFormatBlock> GetTextureFormatBlock(u32 format) {
    constexpr std::array<std::array<u32, 2>, 14> astc{{
        {4, 4}, {5, 4}, {5, 5}, {6, 5}, {6, 6}, {8, 5}, {8, 6},
        {8, 8}, {10, 5}, {10, 6}, {10, 8}, {10, 10}, {12, 10}, {12, 12},
    }};
    if (format >= 0x79 && format <= 0x94) {
        const auto block = astc[(format - 0x79) % astc.size()];
        return TextureFormatBlock{16, block[0], block[1]};
    }
    if (format >= 0x41 && format <= 0x50) {
        const bool small = format <= 0x42 || format == 0x45 || format == 0x46 ||
                           format == 0x49 || format == 0x4A;
        return TextureFormatBlock{small ? 8U : 16U, 4, 4};
    }
    struct Range {
        u32 first;
        u32 last;
        u32 bytes;
    };
    constexpr Range ranges[]{
        {0x01,0x04,1}, {0x05,0x09,2}, {0x0A,0x0C,4}, {0x0D,0x10,2},
        {0x11,0x15,4}, {0x16,0x18,8}, {0x19,0x1C,3}, {0x1D,0x21,6},
        {0x22,0x24,12}, {0x25,0x28,4}, {0x29,0x2D,8}, {0x2E,0x30,16},
        {0x31,0x31,1}, {0x32,0x32,2}, {0x33,0x35,4}, {0x36,0x36,8},
        {0x37,0x38,4}, {0x39,0x3C,2}, {0x3D,0x40,4},
        {0x51,0x52,1}, {0x53,0x54,2}, {0x55,0x56,4}, {0x57,0x58,2},
        {0x59,0x5A,4}, {0x5B,0x5C,8}, {0x5D,0x5E,3}, {0x5F,0x60,6},
        {0x61,0x62,12}, {0x63,0x64,4}, {0x65,0x66,8}, {0x67,0x68,16},
        {0x69,0x6C,4}, {0x6D,0x70,4}, {0x71,0x75,8}, {0x76,0x78,16},
        {0x95,0x98,2}, {0x99,0x9C,4},
    };
    for (const auto& range : ranges) {
        if (format >= range.first && format <= range.last) {
            return TextureFormatBlock{range.bytes, 1, 1};
        }
    }
    return std::nullopt;
}

std::optional<TextureLayout> ComputeTextureLayout(const TextureDescription& d) {
    const auto format = GetTextureFormatBlock(d.format);
    // These bounds also keep every dimension product and mip accumulation within u64.
    if (!format || d.target > 10 || !d.width || !d.height || !d.depth || !d.levels ||
        d.levels > 32 || (d.flags & ~0x7DDU) ||
        (d.samples != 0 && d.samples != 1 && d.samples != 2 && d.samples != 4 && d.samples != 8)) {
        return std::nullopt;
    }
    TextureLayout result;
    if (d.target == 10) {
        if (d.width > (1U << 27) || d.levels != 1 || format->width != 1 || format->height != 1) {
            return std::nullopt;
        }
        result.size = result.layer_size = u64{d.width} * format->bytes;
        result.alignment = format->bytes == 12 ? 4 : format->bytes;
        result.dimensions = 1;
        return result;
    }
    if (d.width > 16384 || d.height > 65536 || d.depth > 65536 ||
        ((d.target == 8 || d.target == 9) && d.width != d.height) ||
        (d.target == 9 && d.depth % 6 != 0)) {
        return std::nullopt;
    }
    const auto shape = GetShape(d);
    result.layers = shape.layered ? static_cast<u32>(shape.depth) : 1;
    result.dimensions = shape.dimensions;
    if (d.flags & 0x50) {
        if (shape.layered || shape.dimensions != 2 || d.levels != 1 || d.samples > 1) {
            return std::nullopt;
        }
        result.alignment = d.flags & 0x40 ? 128 : 32;
        const u64 row_bytes = DivideUp(shape.width, format->width) * format->bytes;
        result.stride = d.stride ? d.stride : AlignUp(row_bytes, result.alignment);
        if (result.stride < row_bytes || result.stride % result.alignment) {
            return std::nullopt;
        }
        result.size = result.layer_size = result.stride * shape.height;
        return result;
    }
    if (d.packaged_layout) {
        const u32 packed = (*d.packaged_layout)[0];
        if ((packed & 15) > 5 || (packed >> 4) > 5) {
            return std::nullopt;
        }
        result.tile_height = shape.dimensions >= 2 ? packed & 15 : 0;
        result.tile_depth = shape.dimensions == 3 ? packed >> 4 : 0;
    } else if (shape.dimensions == 2) {
        const auto samples = SampleShape(d.samples);
        const u64 height = DivideUp(shape.height * samples[1], format->height);
        result.tile_height = PickTile((height + height / 2) / 8);
    } else if (shape.dimensions == 3) {
        result.tile_depth = PickTile(shape.depth + shape.depth / 2);
    }
    result.layer_size = LayerBytes(d, shape, *format, result.tile_height, result.tile_depth);
    if ((d.flags & 0x200) && !d.packaged_layout) {
        const u64 minimal = LayerBytes(d, shape, *format, 0, 0);
        if (minimal != result.layer_size) {
            result.tile_height = result.tile_depth = 0;
            result.layer_size = minimal;
        }
    }
    result.size = result.layer_size * result.layers;
    result.alignment = 512;
    return result;
}

std::optional<u64> TextureLevelOffset(const TextureDescription& d, u32 level) {
    const auto layout = ComputeTextureLayout(d);
    if (!layout || level >= d.levels) {
        return std::nullopt;
    }
    if (d.target == 10 || (d.flags & 0x50)) {
        return 0;
    }
    return LevelBytes(d, GetShape(d), *GetTextureFormatBlock(d.format), level,
                      layout->tile_height, layout->tile_depth);
}

} // namespace Core::BananzaNvn2Hle
