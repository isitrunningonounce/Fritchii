// SPDX-License-Identifier: GPL-3.0-or-later
//
// ABI and object behavior reference: Recomp runtime/nvn_api_obj.c and nvn_obj.h
// at fac48bcdc5ec0edd6296bc3672bdebb540d5958d (MacNativeGames b75be8cf,
// MIT, Copyright (c) 2026 jhaizhou). Host state and guest-memory access rewritten
// for Lichii. Texture views use the verified Bananza 3.2.0 SDK layout.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <utility>
#include <fmt/format.h>

#include "common/cityhash.h"
#include "common/logging.h"
#include "core/arm/bananza_nvn2_hle.h"
#include "core/arm/bananza_nvn2_resources.h"
#include "core/memory.h"
#include "video_core/native_indirect_reset.h"

namespace Core::BananzaNvn2Hle {
namespace {

bool Fits(u64 offset, u64 size, u64 capacity) {
    return offset <= capacity && size <= capacity - offset;
}

bool ValidRange(Memory::Memory& memory, u64 address, u64 size) {
    return address && Fits(address, size, std::numeric_limits<u64>::max()) &&
           memory.IsValidVirtualAddressRange(address, size);
}

bool WriteWords(Memory::Memory& memory, std::span<const u64> addresses,
                std::span<const u32> values) {
    if (addresses.size() != values.size() ||
        !std::ranges::all_of(addresses, [&](u64 address) { return ValidRange(memory, address, 4); })) {
        return false;
    }
    for (size_t i = 0; i < values.size(); ++i) {
        memory.Write32(addresses[i], values[i]);
    }
    return true;
}

float FloatArg(const GuestCall& call, size_t index) {
    return std::bit_cast<float>(static_cast<u32>(call.v[index]));
}

std::optional<u64> RawStorageClass(const TextureDescription& d) {
    if (!GetTextureFormatBlock(d.format)) return std::nullopt;
    // SDK 18265D, +0x5ED3A0 and its jump table at +0xBA094F.
    if (d.flags & 0x50) return 1;
    switch (d.format) {
    case 0x31:
    case 0x32: return 3;
    case 0x33:
    case 0x35: return 4;
    case 0x36: return 5;
    default: return 2;
    }
}

std::optional<u64> StorageClass(const TextureDescription& d) {
    const auto block = GetTextureFormatBlock(d.format);
    if (!block || d.target > 10) return std::nullopt;
    // SDK +0x5ED260: linear textures and buffers use the device's pitch kind.
    // Native Bananza probes verify the resulting class is 0x100.
    if ((d.flags & 0x50) || d.target == 10) return 0x100;
    if ((d.samples != 0 && d.samples != 2 && d.samples != 4 && d.samples != 8) ||
        (d.flags & 0x1000)) return std::nullopt;

    // SDK format table +0xE03968, stride 0x2C. Formats without bit 0 in
    // byte 0 return kind zero before calling the driver (+0x5E2F90).
    if (block->width > 1 || (d.format >= 0x19 && d.format <= 0x21) ||
        (d.format >= 0x51 && d.format <= 0x6C)) return 0x100;

    // Bananza's boot probe returns the same kind for 0, 2, 4 and 8 samples.
    if (d.format >= 0x32 && d.format <= 0x36) {
        constexpr std::array<u64, 5> depth_classes{0x10B, 0x10C, 0x109, 0x10C, 0x10D};
        return d.flags & 8 ? depth_classes[d.format - 0x32] : 0x100;
    }
    const bool known_color = d.format <= 0x18 ||
        (d.format >= 0x22 && d.format <= 0x30) ||
        (d.format >= 0x38 && d.format <= 0x40) ||
        (d.format >= 0x6D && d.format <= 0x75) ||
        (d.format >= 0x95 && d.format <= 0x98) ||
        d.format == 0x9A || d.format == 0x9C;
    if (!known_color) return std::nullopt;
    return d.flags & 8 ? 0x108 : 0x100;
}

std::optional<u64> ZCullStorageSize(const TextureDescription& d) {
    if (!GetTextureFormatBlock(d.format)) return std::nullopt;
    // SDK +0x5EDAF0 only allocates for depth, or stencil with flag 0x400.
    if (d.format < 0x31 || d.format > 0x36 ||
        (d.format == 0x31 && !(d.flags & 0x400))) return 0;
    if (d.flags & 0x80) return std::nullopt;

    // Native device probe: alignment 32x32, tile area 1024, bytes per tile
    // 32, fixed storage 32+192. Legacy helper +0x632160 uses multiplier one
    // for both depth (mode 2) and stencil (mode 1), independent of samples.
    constexpr u64 MaxNative = std::numeric_limits<u32>::max();
    if (d.width > MaxNative - 31 || d.height > MaxNative - 31) return std::nullopt;
    const u64 tiles = ((u64{d.width} + 31) / 32) * ((u64{d.height} + 31) / 32);
    if (tiles > (MaxNative - 1023) / 1024) return std::nullopt;
    const u64 layers = std::max<u64>(d.depth, 1);
    const u64 size = tiles * layers * 32 + 224;
    return size <= MaxNative ? std::optional<u64>{size} : std::nullopt;
}

std::optional<std::array<u8, 16>> ReadView(Memory::Memory& memory, u64 address) {
    std::array<u8, 16> result{};
    if (!ValidRange(memory, address, result.size()) ||
        !memory.ReadBlock(address, result.data(), result.size())) {
        return std::nullopt;
    }
    return result;
}


u64 DivideUp(u64 value, u64 divisor) {
    return (value + divisor - 1) / divisor;
}

u32 ShrinkTile(u32 shift, u64 unit, u64 extent) {
    while (shift && (unit << (shift - 1)) >= extent) --shift;
    return shift;
}

// Same native GOB bit permutation as video_core/textures/decoders.cpp.
u64 NativeTexelOffset(u64 x, u64 y, u64 z, u64 row_bytes, u64 rows, u32 th, u32 td) {
    const u64 block_size = DivideUp(row_bytes, 64) << (9 + th + td);
    const u64 slice_size = DivideUp(rows, 8ULL << th) * block_size;
    u64 offset = (z >> td) * slice_size + ((z & ((1ULL << td) - 1)) << (9 + th)) +
                 (y >> (3 + th)) * block_size + (((y >> 3) & ((1ULL << th) - 1)) << 9) +
                 ((x >> 6) << (9 + th + td)) + ((x & 32) << 3) + ((y & 6) << 5) +
                 ((x & 16) << 1) + ((y & 1) << 4) + (x & 15);
    const u64 difference = ((offset >> 5) ^ (offset >> 6)) & 1;
    return offset ^ (difference << 5) ^ (difference << 6);
}

bool TransferTexels(Memory::Memory& memory, const ResourceTextureInfo& texture,
                    const GuestCall& call, bool write, bool strided,
                    const Resources::GpuAddressLookup& resolve_gpu) {
    constexpr u64 MaxTransfer = 64ULL << 20;
    const auto& x = call.x;
    const auto& d = texture.description;
    const auto block = GetTextureFormatBlock(d.format);
    if (!block || d.samples > 1 || !ValidRange(memory, x[2], 24) ||
        texture.layout.size > (256ULL << 20)) return false;
    u32 level{};
    u32 first_layer{};
    u32 view_layers = texture.layout.layers;
    if (x[1]) {
        const auto view = ReadView(memory, x[1]);
        if (!view) return false;
        const u8 flags = (*view)[7];
        if (flags & 1) {
            level = (*view)[4];
            if (level >= d.levels || ((*view)[5] && !Fits(level, (*view)[5], d.levels))) {
                return false;
            }
        }
        if (flags & 2) {
            first_layer = u32{(*view)[0]} | (u32{(*view)[1]} << 8);
            view_layers = u32{(*view)[2]} | (u32{(*view)[3]} << 8);
            if (first_layer >= texture.layout.layers) return false;
            if (!view_layers) {
                view_layers = texture.layout.layers - first_layer;
            }
            if (!Fits(first_layer, view_layers, texture.layout.layers)) return false;
        }
        if ((flags & 4) && memory.Read32(x[1] + 8) != d.format) return false;
        if ((flags & 0x20) && (*view)[6] != d.target) return false;
        if ((flags & 0x10) && (flags >> 6) != d.depth_mode) return false;
    }
    const auto level_offset = TextureLevelOffset(d, level);
    if (!level_offset) return false;
    std::array<u64, 6> region{};
    for (size_t i = 0; i < region.size(); ++i) region[i] = memory.Read32(x[2] + i * 4);
    const bool three_d = d.target == 2;
    const bool one_d = d.target == 0 || d.target == 3 || d.target == 10;
    const u64 width = std::max<u64>(d.width >> level, 1);
    const u64 height = one_d ? 1 : std::max<u64>(d.height >> level, 1);
    const u64 depth = three_d ? std::max<u64>(d.depth >> level, 1) : view_layers;
    if ((three_d && first_layer) || !Fits(region[0], region[3], width) ||
        !Fits(region[1], region[4], height) || !Fits(region[2], region[5], depth) ||
        region[0] % block->width || region[1] % block->height ||
        (region[3] % block->width && region[0] + region[3] != width) ||
        (region[4] % block->height && region[1] + region[4] != height)) return false;
    if (!region[3] || !region[4] || !region[5]) return true;
    const u64 rows = DivideUp(region[4], block->height);
    const u64 row_bytes = DivideUp(region[3], block->width) * block->bytes;
    const u64 row_stride = strided && x[4] ? x[4] : row_bytes;
    if (row_stride < row_bytes || row_stride > MaxTransfer || rows > MaxTransfer / row_stride) {
        return false;
    }
    const u64 image_stride = strided && x[5] ? x[5] : row_stride * rows;
    if (image_stride < row_stride * rows || image_stride > MaxTransfer ||
        region[5] > MaxTransfer / image_stride) return false;
    const u64 footprint = (region[5] - 1) * image_stride + (rows - 1) * row_stride + row_bytes;
    if (!ValidRange(memory, x[3], footprint) ||
        !Fits(texture.storage.gpu_address, texture.offset, std::numeric_limits<u64>::max())) {
        return false;
    }
    const u64 gpu = texture.storage.gpu_address + texture.offset;
    auto backing = resolve_gpu ? resolve_gpu(gpu, texture.layout.size) : std::nullopt;
    if (!backing && !(texture.storage.flags & 1) &&
        Fits(texture.storage.storage, texture.offset, std::numeric_limits<u64>::max())) {
        backing = texture.storage.storage + texture.offset;
    }
    if (!backing || !ValidRange(memory, *backing, texture.layout.size)) return false;
    const bool linear = (d.flags & 0x50) || d.target == 10;
    const u64 full_rows = DivideUp(height, block->height);
    const u64 full_row_bytes = DivideUp(width, block->width) * block->bytes;
    const u32 th = ShrinkTile(texture.layout.tile_height, 8, full_rows);
    const u32 td = ShrinkTile(texture.layout.tile_depth, 1, three_d ? depth : 1);
    const u64 x_byte = region[0] / block->width * block->bytes;
    const u64 y_block = region[1] / block->height;
    const auto offset_at = [&](u64 z, u64 row, u64 column) {
        const u64 layer = three_d ? 0 : first_layer + region[2] + z;
        const u64 base = *level_offset + layer * texture.layout.layer_size;
        if (linear) return base + (y_block + row) * texture.layout.stride + x_byte + column;
        return base + NativeTexelOffset(x_byte + column, y_block + row,
                                       three_d ? region[2] + z : 0, full_row_bytes, full_rows, th, td);
    };
    for (u64 z = 0; z < region[5]; ++z) {
        for (u64 row = 0; row < rows; ++row) {
            for (u64 column = 0; column < row_bytes;) {
                const u64 count = linear ? row_bytes : std::min(row_bytes - column, 16 - ((x_byte + column) & 15));
                if (!Fits(offset_at(z, row, column), count, texture.layout.size)) return false;
                column += count;
            }
        }
    }
    std::vector<u8> bytes(static_cast<size_t>(row_bytes * rows * region[5]));
    if (write) {
        for (u64 z = 0; z < region[5]; ++z) {
            for (u64 row = 0; row < rows; ++row) {
                if (!memory.ReadBlock(x[3] + z * image_stride + row * row_stride,
                                      bytes.data() + (z * rows + row) * row_bytes, row_bytes)) return false;
            }
        }
    }
    for (u64 z = 0; z < region[5]; ++z) {
        for (u64 row = 0; row < rows; ++row) {
            for (u64 column = 0; column < row_bytes;) {
                const u64 count = linear ? row_bytes : std::min(row_bytes - column, 16 - ((x_byte + column) & 15));
                const u64 address = *backing + offset_at(z, row, column);
                u8* data = bytes.data() + (z * rows + row) * row_bytes + column;
                if (write ? !memory.WriteBlock(address, data, count) : !memory.ReadBlock(address, data, count)) {
                    return false;
                }
                column += count;
            }
        }
    }
    if (!write) {
        for (u64 z = 0; z < region[5]; ++z) {
            for (u64 row = 0; row < rows; ++row) {
                if (!memory.WriteBlock(x[3] + z * image_stride + row * row_stride,
                                       bytes.data() + (z * rows + row) * row_bytes, row_bytes)) return false;
            }
        }
    }
    return true;
}

} // Anonymous namespace

std::optional<u64> Resources::Dispatch(Memory::Memory& memory, std::string_view name,
                                       const GuestCall& call, const DeviceLookup& has_device,
                                       const PoolLookup& find_pool,
                                       const GpuAddressLookup& resolve_gpu) {
    if (!name.starts_with("nvnTexture") && !name.starts_with("nvnSampler") &&
        !name.starts_with("nvnProgram")) return std::nullopt;
    if (!ValidRange(memory, call.x[0], 1)) {
        DiagnoseTexture(name, call, "unmapped object");
        return std::nullopt;
    }
    std::optional<u64> result;
    if (name.starts_with("nvnTextureView")) {
        result = HandleView(memory, name, call);
    } else if (name.starts_with("nvnTexturePool") || name.starts_with("nvnSamplerPool")) {
        result = HandleDescriptorPool(memory, name, call, find_pool);
    } else if (name.starts_with("nvnTexture")) {
        result = HandleTexture(memory, name, call, has_device, find_pool, resolve_gpu);
    } else if (name.starts_with("nvnSampler")) {
        result = HandleSampler(memory, name, call, has_device);
    } else if (name.starts_with("nvnProgram")) {
        result = HandleProgram(memory, name, call, has_device, resolve_gpu);
    }
    if (!result) DiagnoseTexture(name, call, "unhandled or invalid arguments");
    return result;
}

bool Resources::HasTexture(u64 device, u64 texture) const {
    const auto found = textures.find(texture);
    return found != textures.end() && found->second.device == device;
}

std::optional<ResourceTextureInfo> Resources::GetTexture(u64 texture) const {
    const auto found = textures.find(texture);
    return found == textures.end() ? std::nullopt : std::optional{found->second};
}

std::shared_ptr<const ResourceProgramInfo> Resources::GetProgram(u64 program) const {
    const auto found = programs.find(program);
    return found == programs.end() ? nullptr : found->second.snapshot;
}

const std::unordered_map<u32, ResourceTextureDescriptor>* Resources::GetTexturePool(u64 pool) const {
    const auto found = texture_pools.find(pool);
    return found != texture_pools.end() ? &found->second.textures : nullptr;
}

const std::unordered_map<u32, ResourceSamplerInfo>* Resources::GetSamplerPool(u64 pool) const {
    const auto found = sampler_pools.find(pool);
    return found != sampler_pools.end() ? &found->second.samplers : nullptr;
}

bool Resources::IsFloatReturn(std::string_view name) {
    return (name.starts_with("nvnSamplerBuilder") || name.starts_with("nvnSampler")) &&
           (name.ends_with("GetLodBias") || name.ends_with("GetMaxAnisotropy") ||
            name.ends_with("GetLodSnap"));
}


void Resources::NoteDiagnostic(std::string key, std::string message) {
    if (diagnostics.size() >= 64 || !diagnostics.insert(std::move(key)).second) return;
    LOG_WARNING(Core_ARM, "Bananza NVN 2 resource: {}", message);
}

void Resources::DiagnoseTexture(std::string_view name, const GuestCall& call,
                                std::string_view reason, const TextureBuilder* builder) {
    const auto& x = call.x;
    if (!builder) {
        NoteDiagnostic(fmt::format("{}:{}", name, reason),
                       fmt::format("{} {} x0={:#x} x1={:#x} x2={:#x} x3={:#x}",
                                   name, reason, x[0], x[1], x[2], x[3]));
        return;
    }
    const auto& d = builder->description;
    NoteDiagnostic(fmt::format("{}:{}:{:x}:{:x}:{}:{}:{}:{}:{}:{}", name, reason, d.format,
                               d.flags, d.target, d.width, d.height, d.depth, d.levels, d.samples),
                   fmt::format("{} {} object={:#x} device={:#x} pool={:#x} offset={:#x} "
                               "format={:#x} flags={:#x} target={} dimensions={}x{}x{} levels={} "
                               "samples={} stride={} packaged={:#x}",
                               name, reason, x[0], builder->device, builder->pool, builder->offset,
                               d.format, d.flags, d.target, d.width, d.height, d.depth, d.levels,
                               d.samples, d.stride, builder->layout_word));
}

std::optional<Resources::TextureBuilder> Resources::ReadTextureBuilder(Memory::Memory& memory,
                                                                       u64 address) {
    // SDK 18265D... setters at +0x5ECB50..+0x5ECC90 and +0x5ED390..+0x5ED450.
    if (!ValidRange(memory, address, 0x68)) return std::nullopt;
    TextureBuilder result;
    result.device = memory.Read64(address + 8);
    result.pool = memory.Read64(address + 0x10);
    result.offset = memory.Read64(address + 0x18);
    auto& d = result.description;
    d.flags = memory.Read32(address + 0x20);
    d.target = memory.Read32(address + 0x24);
    d.width = memory.Read32(address + 0x28);
    d.height = memory.Read32(address + 0x2C);
    d.depth = memory.Read32(address + 0x30);
    d.levels = memory.Read32(address + 0x34);
    d.format = memory.Read32(address + 0x38);
    d.samples = memory.Read32(address + 0x3C);
    for (size_t i = 0; i < 4; ++i) d.swizzle[i] = memory.Read32(address + 0x40 + i * 4);
    d.depth_mode = memory.Read32(address + 0x50);
    result.layout_word = memory.Read16(address + 0x54);
    if (result.layout_word & 1) {
        std::array<u8, 8> layout{};
        layout[0] = static_cast<u8>(result.layout_word >> 4);
        layout[1] = static_cast<u8>(result.layout_word >> 12);
        d.packaged_layout = layout;
    }
    d.stride = memory.Read32(address + 0x58);
    result.packaged_data = memory.Read64(address + 0x60);
    return result;
}

void Resources::WriteTextureBuilder(Memory::Memory& memory, u64 address,
                                     const TextureBuilder& builder) {
    memory.Write64(address + 8, builder.device);
    memory.Write64(address + 0x10, builder.pool);
    memory.Write64(address + 0x18, builder.offset);
    const auto& d = builder.description;
    memory.Write32(address + 0x20, d.flags);
    memory.Write32(address + 0x24, d.target);
    memory.Write32(address + 0x28, d.width);
    memory.Write32(address + 0x2C, d.height);
    memory.Write32(address + 0x30, d.depth);
    memory.Write32(address + 0x34, d.levels);
    memory.Write32(address + 0x38, d.format);
    memory.Write32(address + 0x3C, d.samples);
    for (size_t i = 0; i < 4; ++i) memory.Write32(address + 0x40 + i * 4, d.swizzle[i]);
    memory.Write32(address + 0x50, d.depth_mode);
    memory.Write16(address + 0x54, builder.layout_word);
    memory.Write32(address + 0x58, d.stride);
    memory.Write64(address + 0x60, builder.packaged_data);
}

std::optional<u64> Resources::HandleTexture(Memory::Memory& memory, std::string_view name,
                                            const GuestCall& call, const DeviceLookup& has_device,
                                            const PoolLookup& find_pool,
                                            const GpuAddressLookup& resolve_gpu) {
    const auto& x = call.x;
    if (name == "nvnTextureBuilderSetDefaults") {
        if (!ValidRange(memory, x[0], 0x68)) {
            DiagnoseTexture(name, call, "unmapped native builder");
            return std::nullopt;
        }
        TextureBuilder builder{.device = memory.Read64(x[0] + 8)};
        // Native defaults +0x5ECB60: target=2D, dimensions=0, levels=1, swizzle=RGBA.
        builder.description.width = builder.description.height = builder.description.depth = 0;
        WriteTextureBuilder(memory, x[0], builder);
        memory.Write32(x[0] + 0x5C, 0);
        return 0;
    }
    if (name == "nvnTextureBuilderSetDevice") {
        if (!ValidRange(memory, x[0], 0x10)) return std::nullopt;
        memory.Write64(x[0] + 8, x[1]);
        return 0;
    }
    if (name.starts_with("nvnTextureBuilder")) {
        auto decoded = ReadTextureBuilder(memory, x[0]);
        if (!decoded) {
            DiagnoseTexture(name, call, "unmapped native builder");
            return std::nullopt;
        }
        auto& builder = *decoded;
        auto& d = builder.description;
        const auto action = name.substr(std::string_view{"nvnTextureBuilder"}.size());
        if (action == "SetFlags") d.flags = static_cast<u32>(x[1]);
        else if (action == "SetTarget") d.target = static_cast<u32>(x[1]);
        else if (action == "SetWidth") d.width = static_cast<u32>(x[1]);
        else if (action == "SetHeight") d.height = static_cast<u32>(x[1]);
        else if (action == "SetDepth") d.depth = static_cast<u32>(x[1]);
        else if (action == "SetLevels") d.levels = static_cast<u32>(x[1]);
        else if (action == "SetFormat") d.format = static_cast<u32>(x[1]);
        else if (action == "SetSamples") d.samples = static_cast<u32>(x[1]);
        else if (action == "SetStride") d.stride = static_cast<u32>(x[1]);
        else if (action == "SetDepthStencilMode") {
            if (x[1] > 1) return std::nullopt;
            d.depth_mode = static_cast<u32>(x[1]);
        } else if (action == "SetSize1D" || action == "SetSize2D" || action == "SetSize3D") {
            d.width = static_cast<u32>(x[1]);
            d.height = action == "SetSize1D" ? 1 : static_cast<u32>(x[2]);
            d.depth = action == "SetSize3D" ? static_cast<u32>(x[3]) : 1;
        } else if (action == "SetStorage") {
            builder.pool = x[1];
            builder.offset = x[2];
        } else if (action == "SetSwizzle") {
            if (std::ranges::any_of(std::span{x}.subspan(1, 4), [](u64 value) { return value > 5; })) {
                return std::nullopt;
            }
            for (size_t i = 0; i < 4; ++i) d.swizzle[i] = static_cast<u32>(x[i + 1]);
        } else if (action == "SetPackagedTextureLayout") {
            if (!x[1]) {
                d.packaged_layout.reset();
                builder.layout_word = 0;
            } else {
                std::array<u8, 8> layout;
                if (!ValidRange(memory, x[1], layout.size()) ||
                    !memory.ReadBlock(x[1], layout.data(), layout.size())) return std::nullopt;
                d.packaged_layout = layout;
                builder.layout_word = static_cast<u16>(1 | (u16{layout[0]} << 4) |
                                                       ((u16{layout[1]} & 15) << 12));
            }
        } else if (action == "SetPackagedTextureData") builder.packaged_data = x[1];
        else if (action == "GetDevice") return builder.device;
        else if (action == "GetMemoryPool") return builder.pool;
        else if (action == "GetMemoryOffset") return builder.offset;
        else if (action == "GetPackagedTextureData") return builder.packaged_data;
        else if (action == "GetFlags") return d.flags;
        else if (action == "GetTarget") return d.target;
        else if (action == "GetWidth") return d.width;
        else if (action == "GetHeight") return d.height;
        else if (action == "GetDepth") return d.depth;
        else if (action == "GetLevels") return d.levels;
        else if (action == "GetFormat") return d.format;
        else if (action == "GetSamples") return d.samples;
        else if (action == "GetStride") return d.stride;
        else if (action == "GetRawStorageClass") return RawStorageClass(d);
        else if (action == "GetStorageClass") return StorageClass(d);
        else if (action == "GetZCullStorageSize") return ZCullStorageSize(d);
        else if (action == "GetDepthStencilMode") return d.depth_mode;
        else if (action == "GetSwizzle") {
            if (!WriteWords(memory, std::span{x}.subspan(1, 4), d.swizzle)) return std::nullopt;
        } else if (action == "GetStorageSize" || action == "GetStorageAlignment") {
            const auto layout = ComputeTextureLayout(d);
            if (!layout) {
                DiagnoseTexture(name, call, "unsupported or invalid layout", &builder);
                return std::nullopt;
            }
            return action == "GetStorageSize" ? layout->size : layout->alignment;
        } else {
            DiagnoseTexture(name, call, "unsupported builder operation");
            return std::nullopt;
        }
        if (action.starts_with("Set")) WriteTextureBuilder(memory, x[0], builder);
        return 0;
    }
    if (name == "nvnTextureInitialize") {
        const auto decoded = ReadTextureBuilder(memory, x[1]);
        if (!decoded) {
            DiagnoseTexture(name, call, "unmapped native builder");
            return 0;
        }
        const auto& builder = *decoded;
        if (!has_device || !has_device(builder.device) || !find_pool || next_serial == 0) {
            DiagnoseTexture(name, call, "missing device or pool lookup", &builder);
            return 0;
        }
        const auto pool = find_pool(builder.pool);
        const auto layout = ComputeTextureLayout(builder.description);
        if (!pool || !layout || pool->device != builder.device || (pool->flags & 8) ||
            !Fits(builder.offset, layout->size, pool->size) ||
            builder.offset % layout->alignment ||
            !Fits(pool->gpu_address, builder.offset + layout->size, std::numeric_limits<u64>::max())) {
            DiagnoseTexture(name, call, !pool ? "missing pool" : !layout ? "unsupported layout" :
                            "pool bounds, device, access or alignment mismatch", &builder);
            if (pool && layout) {
                NoteDiagnostic(fmt::format("pool-bounds:{:x}:{:x}:{}", pool->flags, builder.description.format,
                                           layout->alignment),
                               fmt::format("texture pool={:#x} flags={:#x} size={:#x} GPU={:#x} "
                                           "required={:#x} offset={:#x} alignment={:#x}",
                                           builder.pool, pool->flags, pool->size, pool->gpu_address,
                                           layout->size, builder.offset, layout->alignment));
            }
            return 0;
        }
        if (builder.packaged_data) {
            if ((pool->flags & 1) || layout->size > (64ULL << 20) ||
                !ValidRange(memory, builder.packaged_data, layout->size) ||
                !Fits(pool->storage, builder.offset, std::numeric_limits<u64>::max()) ||
                !ValidRange(memory, pool->storage + builder.offset, layout->size)) return 0;
            std::vector<u8> bytes(static_cast<size_t>(layout->size));
            if (!memory.ReadBlock(builder.packaged_data, bytes.data(), bytes.size()) ||
                !memory.WriteBlock(pool->storage + builder.offset, bytes.data(), bytes.size())) return 0;
        }
        textures.insert_or_assign(x[0], ResourceTextureInfo{
            .device = builder.device, .pool = builder.pool, .offset = builder.offset,
            .serial = next_serial++, .storage = *pool, .description = builder.description,
            .layout = *layout,
        });
        return 1;
    }
    if (name == "nvnTextureFinalize") {
        textures.erase(x[0]);
        return 0;
    }
    const auto found = textures.find(x[0]);
    if (found == textures.end()) return std::nullopt;
    const auto& texture = found->second;
    const auto& d = texture.description;
    if (name == "nvnTextureWriteTexels" || name == "nvnTextureWriteTexelsStrided" ||
        name == "nvnTextureReadTexels" || name == "nvnTextureReadTexelsStrided") {
        const auto pool = find_pool ? find_pool(texture.pool) : std::nullopt;
        const bool write = name.starts_with("nvnTextureWrite");
        const bool transferred = pool && pool->gpu_address == texture.storage.gpu_address &&
                                 TransferTexels(memory, texture, call, write,
                                                name.ends_with("Strided"), resolve_gpu);
        static const bool trace = [] {
            const char* flag = std::getenv("EDEN_NX2_NVN_TRACE_IMAGES");
            return flag && *flag == '1';
        }();
        if (trace && write && d.width >= 900 && ValidRange(memory, x[2], 24)) {
            LOG_INFO(Core_ARM, "NVN 2 texture write {:#x} {}x{} region {} {} {} {} {} {}, success {}",
                     texture.storage.gpu_address + texture.offset, d.width, d.height,
                     memory.Read32(x[2]), memory.Read32(x[2] + 4), memory.Read32(x[2] + 8),
                     memory.Read32(x[2] + 12), memory.Read32(x[2] + 16),
                     memory.Read32(x[2] + 20), transferred);
        }
        if (!transferred) {
            return std::nullopt;
        }
        return 0;
    }
    // CPU transfers use Memory's cache-aware accessors; no GPU execution is enabled here.
    if (name == "nvnTextureFlushTexels" || name == "nvnTextureInvalidateTexels") return 0;
    if (name == "nvnTextureGetFlags") return d.flags;
    if (name == "nvnTextureGetTarget") return d.target;
    if (name == "nvnTextureGetWidth") return d.width;
    if (name == "nvnTextureGetHeight") return d.height;
    if (name == "nvnTextureGetDepth") return d.depth;
    if (name == "nvnTextureGetLevels") return d.levels;
    if (name == "nvnTextureGetFormat") return d.format;
    if (name == "nvnTextureGetSamples") return d.samples;
    if (name == "nvnTextureGetStride") return texture.layout.stride;
    if (name == "nvnTextureGetDepthStencilMode") return d.depth_mode;
    if (name == "nvnTextureGetStorageSize") return texture.layout.size;
    if (name == "nvnTextureGetRawStorageClass") return RawStorageClass(d);
    if (name == "nvnTextureGetStorageClass") return StorageClass(d);
    if (name == "nvnTextureGetZCullStorageSize") return ZCullStorageSize(d);
    if (name == "nvnTextureGetMemoryPool") return texture.pool;
    if (name == "nvnTextureGetMemoryOffset") return texture.offset;
    if (name == "nvnTextureGetDebugID") return texture.serial;
    if (name == "nvnTextureGetSwizzle") {
        if (!WriteWords(memory, std::span{x}.subspan(1, 4), d.swizzle)) return std::nullopt;
        return 0;
    }
    if (name == "nvnTextureGetViewOffset") {
        if (!x[1]) return 0;
        const auto view = ReadView(memory, x[1]);
        if (!view) return std::nullopt;
        const u32 level = ((*view)[7] & 1) ? (*view)[4] : 0;
        const u32 layer = ((*view)[7] & 2) ? u32{(*view)[0]} | (u32{(*view)[1]} << 8) : 0;
        const auto offset = TextureLevelOffset(d, level);
        if (!offset || layer >= texture.layout.layers) return std::nullopt;
        return *offset + u64{layer} * texture.layout.layer_size;
    }
    if (name == "nvnTextureSetDebugLabel") return 0;
    return std::nullopt;
}

std::optional<u64> Resources::HandleView(Memory::Memory& memory, std::string_view name,
                                         const GuestCall& call) {
    const auto& x = call.x;
    if (!ValidRange(memory, x[0], 16)) return std::nullopt;
    const auto action = name.substr(std::string_view{"nvnTextureView"}.size());
    if (action == "SetDefaults") {
        memory.Write64(x[0], 0);
        memory.Write64(x[0] + 8, 0);
        return 0;
    }
    u64 packed = memory.Read64(x[0]);
    if (action == "SetLevels") {
        if (x[1] > 255 || x[2] > 255) return std::nullopt;
        packed = (packed & 0xFFFF0000FFFFFFFFULL) | (x[1] << 32) | (x[2] << 40) | (1ULL << 56);
    } else if (action == "SetLayers") {
        if (x[1] > 65535 || x[2] > 65535) return std::nullopt;
        packed = (packed & 0xFFFFFFFF00000000ULL) | x[1] | (x[2] << 16) | (1ULL << 57);
    } else if (action == "SetFormat") {
        memory.Write32(x[0] + 8, static_cast<u32>(x[1]));
        packed |= 1ULL << 58;
    } else if (action == "SetSwizzle") {
        if (std::ranges::any_of(std::span{x}.subspan(1, 4), [](u64 value) { return value > 5; })) {
            return std::nullopt;
        }
        for (size_t i = 0; i < 4; ++i) memory.Write8(x[0] + 12 + i, static_cast<u8>(x[i + 1]));
        packed |= 1ULL << 59;
    } else if (action == "SetDepthStencilMode") {
        if (x[1] > 1) return std::nullopt;
        packed = (packed & 0x3FFFFFFFFFFFFFFFULL) | (x[1] << 62) | (1ULL << 60);
    } else if (action == "SetTarget") {
        if (x[1] > 10) return std::nullopt;
        packed = (packed & 0xFF00FFFFFFFFFFFFULL) | (x[1] << 48) | (1ULL << 61);
    } else if (action == "Compare") {
        const auto other = ReadView(memory, x[1]);
        const auto view = ReadView(memory, x[0]);
        if (!view || !other) return std::nullopt;
        return *view == *other;
    } else {
        std::array<u32, 4> values{};
        size_t count{};
        u32 flag{};
        if (action == "GetLevels") {
            values[0] = (packed >> 32) & 255; values[1] = (packed >> 40) & 255;
            count = 2; flag = 56;
        } else if (action == "GetLayers") {
            values[0] = packed & 65535; values[1] = (packed >> 16) & 65535;
            count = 2; flag = 57;
        } else if (action == "GetFormat") {
            values[0] = memory.Read32(x[0] + 8); count = 1; flag = 58;
        } else if (action == "GetSwizzle") {
            for (size_t i = 0; i < 4; ++i) values[i] = memory.Read8(x[0] + 12 + i);
            count = 4; flag = 59;
        } else if (action == "GetDepthStencilMode") {
            values[0] = static_cast<u32>(packed >> 62); count = 1; flag = 60;
        } else if (action == "GetTarget") {
            values[0] = (packed >> 48) & 255; count = 1; flag = 61;
        } else return std::nullopt;
        if (!WriteWords(memory, std::span{x}.subspan(1, count), std::span{values}.first(count))) {
            return std::nullopt;
        }
        return (packed >> flag) & 1;
    }
    memory.Write64(x[0], packed);
    return 0;
}


std::optional<u64> Resources::HandleSampler(Memory::Memory& memory, std::string_view name,
                                            const GuestCall& call, const DeviceLookup& has_device) {
    const auto& x = call.x;
    if (name == "nvnSamplerBuilderSetDefaults") {
        const auto found = sampler_builders.find(x[0]);
        const u64 device = found == sampler_builders.end() ? 0 : found->second.device;
        sampler_builders.insert_or_assign(x[0], Sampler{.device = device});
        return 0;
    }
    if (name == "nvnSamplerBuilderSetDevice") {
        sampler_builders[x[0]].device = x[1];
        return 0;
    }
    if (name == "nvnSamplerInitialize") {
        const auto found = sampler_builders.find(x[1]);
        if (found == sampler_builders.end() || !has_device || !has_device(found->second.device)) {
            return 0;
        }
        samplers.insert_or_assign(x[0], found->second);
        return 1;
    }
    if (name == "nvnSamplerFinalize") {
        samplers.erase(x[0]);
        return 0;
    }
    const bool builder = name.starts_with("nvnSamplerBuilder");
    auto& objects = builder ? sampler_builders : samplers;
    const auto found = objects.find(x[0]);
    if (found == objects.end()) return std::nullopt;
    auto& sampler = found->second;
    const auto action = name.substr(builder ? std::string_view{"nvnSamplerBuilder"}.size()
                                           : std::string_view{"nvnSampler"}.size());
    if (action == "SetDebugLabel") return 0;
    if (builder && action.starts_with("Set")) {
        if (action == "SetMinMagFilter") {
            if (x[1] > 5 || x[2] > 1) return std::nullopt;
            sampler.min_filter = static_cast<u32>(x[1]);
            sampler.mag_filter = static_cast<u32>(x[2]);
        } else if (action == "SetWrapMode") {
            if (x[1] > 7 || x[2] > 7 || x[3] > 7) return std::nullopt;
            for (size_t i = 0; i < 3; ++i) sampler.wrap[i] = static_cast<u32>(x[i + 1]);
        } else if (action == "SetCompare") {
            if (x[1] > 1 || x[2] < 1 || x[2] > 8) return std::nullopt;
            sampler.compare_mode = static_cast<u32>(x[1]);
            sampler.compare_func = static_cast<u32>(x[2]);
        } else if (action == "SetReductionFilter") {
            if (x[1] > 2) return std::nullopt;
            sampler.reduction = static_cast<u32>(x[1]);
        } else if (action == "SetBorderColor" || action == "SetBorderColori" ||
                   action == "SetBorderColorui") {
            if (!ValidRange(memory, x[1], 16)) return std::nullopt;
            for (size_t i = 0; i < 4; ++i) sampler.border[i] = memory.Read32(x[1] + i * 4);
            sampler.border_kind = action == "SetBorderColori" ? 1 : action == "SetBorderColorui" ? 2 : 0;
        } else if (action == "SetLodClamp") {
            const float minimum = FloatArg(call, 0), maximum = FloatArg(call, 1);
            if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum) {
                return std::nullopt;
            }
            sampler.min_lod = minimum;
            sampler.max_lod = maximum;
        } else if (action == "SetLodBias" || action == "SetLodSnap" ||
                   action == "SetMaxAnisotropy") {
            const float value = FloatArg(call, 0);
            if (!std::isfinite(value) ||
                (action == "SetMaxAnisotropy" && (value < 1.0f || value > 16.0f))) {
                return std::nullopt;
            }
            if (action == "SetLodBias") sampler.lod_bias = value;
            else if (action == "SetLodSnap") sampler.lod_snap = value;
            else sampler.anisotropy = value;
        } else return std::nullopt;
        return 0;
    }
    if (action == "GetDevice" && builder) return sampler.device;
    if (action == "GetReductionFilter") return sampler.reduction;
    if (action == "GetLodBias") return std::bit_cast<u32>(sampler.lod_bias);
    if (action == "GetLodSnap") return std::bit_cast<u32>(sampler.lod_snap);
    if (action == "GetMaxAnisotropy") return std::bit_cast<u32>(sampler.anisotropy);
    if (action == "GetBorderColor" || action == "GetBorderColori" || action == "GetBorderColorui") {
        const u32 kind = action == "GetBorderColori" ? 1 : action == "GetBorderColorui" ? 2 : 0;
        if (kind != sampler.border_kind || !ValidRange(memory, x[1], 16)) return std::nullopt;
        for (size_t i = 0; i < 4; ++i) memory.Write32(x[1] + i * 4, sampler.border[i]);
        return 0;
    }
    std::array<u32, 3> values{};
    size_t count{};
    if (action == "GetMinMagFilter") {
        values[0] = sampler.min_filter; values[1] = sampler.mag_filter; count = 2;
    } else if (action == "GetWrapMode") {
        values = sampler.wrap; count = 3;
    } else if (action == "GetLodClamp") {
        values[0] = std::bit_cast<u32>(sampler.min_lod);
        values[1] = std::bit_cast<u32>(sampler.max_lod);
        count = 2;
    } else if (action == "GetCompare") {
        values[0] = sampler.compare_mode; values[1] = sampler.compare_func; count = 2;
    } else return std::nullopt;
    if (!WriteWords(memory, std::span{x}.subspan(1, count), std::span{values}.first(count))) {
        return std::nullopt;
    }
    return 0;
}

std::optional<u64> Resources::HandleDescriptorPool(Memory::Memory& memory, std::string_view name,
                                                   const GuestCall& call,
                                                   const PoolLookup& find_pool) {
    const auto& x = call.x;
    const bool texture = name.starts_with("nvnTexturePool");
    auto& objects = texture ? texture_pools : sampler_pools;
    const auto action = name.substr(texture ? std::string_view{"nvnTexturePool"}.size()
                                           : std::string_view{"nvnSamplerPool"}.size());
    if (action == "Initialize") {
        const auto storage = find_pool ? find_pool(x[1]) : std::nullopt;
        const u64 count = static_cast<u32>(x[3]);
        if (!storage || (storage->flags & 8) || count == 0 || count > (1U << 20) ||
            x[2] % 32 || !Fits(x[2], count * 32, storage->size)) {
            NoteDiagnostic(fmt::format("{}:initialize", name),
                           fmt::format("{} rejected object={:#x} pool={:#x} offset={:#x} count={} "
                                       "pool_found={} pool_size={:#x} pool_flags={:#x}", name,
                                       x[0], x[1], x[2], count, storage.has_value(),
                                       storage ? storage->size : 0, storage ? storage->flags : 0));
            return 0;
        }
        DescriptorPool pool;
        pool.pool = x[1];
        pool.offset = x[2];
        pool.count = static_cast<u32>(count);
        pool.storage = *storage;
        objects.insert_or_assign(x[0], std::move(pool));
        return 1;
    }
    if (action == "Finalize") {
        objects.erase(x[0]);
        return 0;
    }
    const auto found = objects.find(x[0]);
    if (found == objects.end()) return std::nullopt;
    auto& pool = found->second;
    const auto storage = find_pool ? find_pool(pool.pool) : std::nullopt;
    if (!storage || storage->gpu_address != pool.storage.gpu_address ||
        storage->size != pool.storage.size || storage->device != pool.storage.device) {
        return std::nullopt;
    }
    if (action == "GetMemoryPool") return pool.pool;
    if (action == "GetMemoryOffset") return pool.offset;
    if (action == "GetSize") return pool.count;
    if (action == "SetDebugLabel") return 0;
    const u32 id = static_cast<u32>(x[1]);
    if (id >= pool.count) return std::nullopt;
    if (texture && action == "RegisterTexture") {
        const auto resource = textures.find(x[2]);
        if (resource == textures.end() || resource->second.device != storage->device) {
            NoteDiagnostic(fmt::format("{}:texture-missing", name),
                           fmt::format("{} rejected descriptor_pool={:#x} id={} texture={:#x} "
                                       "view={:#x} texture_found={} device={:#x}", name, x[0], id,
                                       x[2], x[3], resource != textures.end(), storage->device));
            return std::nullopt;
        }
        TextureDescriptor descriptor{.texture = resource->second};
        if (x[3]) {
            descriptor.view = ReadView(memory, x[3]);
            if (!descriptor.view) return std::nullopt;
        }
        pool.textures.insert_or_assign(id, std::move(descriptor));
        return 0;
    }
    if (!texture && (action == "RegisterSampler" || action == "RegisterSamplerBuilder")) {
        const auto& source = action == "RegisterSampler" ? samplers : sampler_builders;
        const auto resource = source.find(x[2]);
        if (resource == source.end() || resource->second.device != storage->device) {
            return std::nullopt;
        }
        pool.samplers.insert_or_assign(id, resource->second);
        return 0;
    }
    return std::nullopt;
}

std::optional<std::vector<Resources::Shader>> Resources::ReadShaders(
    Memory::Memory& memory, u64 address, u64 count, bool extended) {
    const u64 stride = extended ? 24 : 16;
    if (count > 6 || (count && !ValidRange(memory, address, count * stride))) return std::nullopt;
    std::vector<Shader> result;
    result.reserve(static_cast<size_t>(count));
    for (u64 i = 0; i < count; ++i) {
        Shader shader{
            .data = memory.Read64(address + i * stride),
            .control = memory.Read64(address + i * stride + 8),
            .reserved = extended ? memory.Read64(address + i * stride + 16) : 0,
        };
        if (!shader.data || shader.data % 4 || shader.reserved ||
            !ValidRange(memory, shader.control, 0x48) ||
            memory.Read32(shader.control) != 0x98761234 ||
            memory.Read32(shader.control + 4) != 201 ||
            memory.Read32(shader.control + 0x38) == 0) {
            return std::nullopt;
        }
        result.push_back(shader);
    }
    return result;
}

std::vector<std::shared_ptr<const ResourceProgramInfo>> Resources::TakeNewPrograms() {
    std::scoped_lock lock{new_programs_mutex};
    return std::exchange(new_programs, {});
}

std::shared_ptr<const ResourceProgramInfo> Resources::StageProgram(
    Memory::Memory& memory, const Program& program, const GpuAddressLookup& resolve_gpu) {
    const auto reject = [&](std::string_view reason) -> std::shared_ptr<const ResourceProgramInfo> {
        NoteDiagnostic(fmt::format("shader:{}", reason),
                       fmt::format("shader staging rejected: {}", reason));
        return {};
    };
    if (next_serial == 0) return reject("resource serial exhausted");
    auto snapshot = std::make_shared<ResourceProgramInfo>();
    snapshot->device = program.device;
    u64 total_bytes{};
    u32 stages{};
    for (const auto& shader : program.shaders) {
        if (!ValidRange(memory, shader.control, 0x48)) return reject("unmapped control header");
        const u32 reported_control_size = memory.Read32(shader.control + 0x30);
        // Pinned Recomp gfx_shader.c nvn2_sizes: Bananza's zero-size controls occupy 0xD8.
        const u32 control_size = reported_control_size ? reported_control_size : 0xD8;
        const u32 code_size = memory.Read32(shader.control + 0x38);
        const u32 constant_begin = memory.Read32(shader.control + 0x40);
        const u32 constant_end = memory.Read32(shader.control + 0x44);
        const u32 storage_size = std::max(code_size, constant_end);
        if (control_size < 0x48 || control_size > 0x10000 || code_size < 16 ||
            storage_size > (16U << 20) || constant_end < constant_begin ||
            (constant_end != constant_begin && constant_begin < code_size) ||
            storage_size - code_size > 0x10000 || code_size % 16 ||
            !Fits(total_bytes, u64{control_size} + storage_size, 64ULL << 20) ||
            !ValidRange(memory, shader.control, control_size)) {
            NoteDiagnostic(fmt::format("shader-extents:{:x}:{:x}:{:x}:{:x}",
                                       reported_control_size, code_size, constant_begin, constant_end),
                           fmt::format("shader extents control={:#x} data={:#x} control_size={:#x} "
                                       "code_size={:#x} constants={:#x}..{:#x}", shader.control,
                                       shader.data, reported_control_size, code_size, constant_begin,
                                       constant_end));
            return reject("invalid native control or code extent");
        }
        const auto address = resolve_gpu ? resolve_gpu(shader.data, storage_size) : std::nullopt;
        if (!address || !ValidRange(memory, *address, storage_size)) {
            return reject("shader GPU address has no complete CPU backing");
        }
        ResourceShaderInfo copy;
        copy.data_address = shader.data;
        copy.control_address = shader.control;
        copy.code_size = code_size;
        copy.constant_offset = constant_begin;
        copy.constant_size = constant_end - constant_begin;
        copy.control.resize(control_size);
        copy.code.resize(storage_size);
        if (!memory.ReadBlock(shader.control, copy.control.data(), copy.control.size()) ||
            !memory.ReadBlock(*address, copy.code.data(), copy.code.size())) {
            return reject("failed to copy shader inputs");
        }
        const auto word = [](const std::vector<u8>& bytes, size_t offset) {
            return u32{bytes[offset]} | (u32{bytes[offset + 1]} << 8) |
                   (u32{bytes[offset + 2]} << 16) | (u32{bytes[offset + 3]} << 24);
        };
        if (word(copy.control, 0) != 0x98761234 || word(copy.control, 4) != 201 ||
            word(copy.control, 0x30) != reported_control_size ||
            word(copy.control, 0x38) != code_size || word(copy.control, 0x40) != constant_begin ||
            word(copy.control, 0x44) != constant_end) {
            return reject("unrecognized or changing NVN 2 control");
        }
        const u32 control_stage = control_size > 0x56 ? copy.control[0x56] : 0;
        if (control_stage >= 5 && control_stage <= 7) {
            copy.stage = control_stage;
        } else {
            const u32 first_word = word(copy.code, 0);
            const u32 header_stage = (first_word >> 10) & 15;
            const u32 header_type = first_word & 31;
            const u32 header_version = (first_word >> 5) & 31;
            if (code_size <= 128 || header_stage < 1 || header_stage > 5 || header_version != 4 ||
                (header_type != 1 && header_type != 2) ||
                (header_type == 2) != (header_stage == 5)) {
                NoteDiagnostic(fmt::format("shader-header:{:x}:{}:{}", first_word, control_stage,
                                           reported_control_size),
                               fmt::format("shader header control={:#x} data={:#x} CPU={:#x} "
                                           "control_size={:#x} effective_control_size={:#x} stage_byte={} "
                                           "code_size={:#x} storage_size={:#x} words={:08x},{:08x},"
                                           "{:08x},{:08x} word_at_80={:08x}", shader.control, shader.data,
                                           *address, reported_control_size, control_size, control_stage,
                                           code_size, storage_size, first_word, word(copy.code, 4),
                                           word(copy.code, 8), word(copy.code, 12),
                                           storage_size >= 0x84 ? word(copy.code, 0x80) : 0));
                return reject("unrecognized graphics stage header");
            }
            copy.stage = header_stage - 1;
            copy.header_size = 0x80;
        }
        if (stages & (1U << copy.stage)) return reject("duplicate stage in shader variant");
        stages |= 1U << copy.stage;
        if (copy.stage == 0 &&
            copy.code_size >= copy.header_size + sizeof(VideoCommon::NativeIndirectReset::Program)) {
            std::array<u32, VideoCommon::NativeIndirectReset::Program.size()> instructions;
            std::memcpy(instructions.data(), copy.code.data() + copy.header_size, sizeof(instructions));
            snapshot->resets_indirect_draws = VideoCommon::NativeIndirectReset::Matches(instructions);
        }
        total_bytes += u64{control_size} + storage_size;
        snapshot->shaders.push_back(std::move(copy));
    }
    if (((stages & (1U << 5)) && program.shaders.size() != 1) ||
        ((stages & 0xC0) && (stages & 0x0F))) {
        return reject("incompatible compute, mesh or conventional graphics stages");
    }
    snapshot->serial = next_serial++;
    u64 hash{};
    for (const auto& shader : snapshot->shaders) {
        const std::array<u32, 5> shape{shader.stage, shader.code_size, shader.header_size,
                                       shader.constant_offset, shader.constant_size};
        const auto mix = [&hash](const void* data, size_t size) {
            hash = hash * 0x9E3779B97F4A7C15ULL +
                   Common::CityHash64(static_cast<const char*>(data), size);
        };
        mix(shape.data(), sizeof(shape));
        mix(shader.control.data(), shader.control.size());
        mix(shader.code.data(), shader.code.size());
    }
    snapshot->content_hash = hash ? hash : 1;
    {
        std::scoped_lock lock{new_programs_mutex};
        new_programs.push_back(snapshot);
    }
    if (!reported_shader_staging) {
        LOG_INFO(Core_ARM, "Bananza NVN 2 HLE: storing immutable native shader inputs; "
                           "programs remain non-executable until a replay backend is connected");
        reported_shader_staging = true;
    }
    return snapshot;
}

std::optional<u64> Resources::HandleProgram(Memory::Memory& memory, std::string_view name,
                                            const GuestCall& call, const DeviceLookup& has_device,
                                            const GpuAddressLookup& resolve_gpu) {
    const auto& x = call.x;
    if (name == "nvnProgramBuilderSetDefaults") {
        const auto found = program_builders.find(x[0]);
        const u64 device = found == program_builders.end() ? 0 : found->second.device;
        program_builders.insert_or_assign(x[0], Program{.device = device});
        return 0;
    }
    if (name == "nvnProgramBuilderSetDevice") {
        program_builders[x[0]].device = x[1];
        return 0;
    }
    if (name == "nvnProgramBuilderSetShaders") {
        const auto found = program_builders.find(x[0]);
        const auto shaders = ReadShaders(memory, x[2], x[1], false);
        if (found == program_builders.end() || !shaders) {
            LOG_WARNING(Core_ARM, "Bananza NVN 2 HLE: invalid ProgramBuilderSetShaders arguments");
            return std::nullopt;
        }
        found->second.shaders = *shaders;
        return 0;
    }
    if (name == "nvnProgramBuilderGetDevice" || name == "nvnProgramBuilderGetShaderCount") {
        const auto found = program_builders.find(x[0]);
        if (found == program_builders.end()) return std::nullopt;
        return name.ends_with("GetDevice") ? found->second.device : found->second.shaders.size();
    }
    if (name == "nvnProgramInitialize") {
        if (!has_device || !has_device(x[1])) return 0;
        programs.insert_or_assign(x[0], Program{.device = x[1]});
        return 1;
    }
    if (name == "nvnProgramFinalize") {
        programs.erase(x[0]);
        return 0;
    }
    if (name == "nvnProgramSetShaders" || name == "nvnProgramSetShadersExt") {
        const auto found = programs.find(x[0]);
        const auto shaders = ReadShaders(memory, x[2], x[1], name.ends_with("Ext"));
        if (found == programs.end() || !shaders) {
            LOG_WARNING(Core_ARM, "Bananza NVN 2 HLE: invalid {} arguments", name);
            return 0;
        }
        Program next{.device = found->second.device, .shaders = *shaders};
        next.snapshot = StageProgram(memory, next, resolve_gpu);
        if (!next.snapshot) return 0;
        found->second = std::move(next);
        return 1;
    }
    if (name == "nvnProgramInitializeFromBuilder") {
        const auto found = program_builders.find(x[1]);
        if (found == program_builders.end() || !has_device || !has_device(found->second.device)) return 0;
        Program next = found->second;
        next.snapshot = StageProgram(memory, next, resolve_gpu);
        if (!next.snapshot) return 0;
        programs.insert_or_assign(x[0], std::move(next));
        return 1;
    }
    if (name == "nvnProgramSetDebugLabel" && programs.contains(x[0])) return 0;
    return std::nullopt;
}

} // namespace Core::BananzaNvn2Hle
