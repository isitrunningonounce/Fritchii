// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "common/logging.h"
#include "common/cityhash.h"
#include "common/swap.h"
#include "core/arm/bananza_nvn2_commands.h"
#include "core/memory.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/profile.h"
#include "video_core/native_indirect_reset.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/vk_bananza_nvn2_replay.h"
#include "video_core/renderer_vulkan/vk_bananza_nvn2_shaders.h"
#include "video_core/renderer_vulkan/vk_frame_dump.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/textures/astc.h"
#include "video_core/textures/decoders.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan {
namespace {

using Core::BananzaNvn2Commands::ReplayOp;
using VideoCore::Surface::PixelFormat;
constexpr std::array<u32, 3> GraphicsStorageBase{0x110, 0x510, 0x410};

struct ColorFormat {
    VkFormat vk;
    PixelFormat surface;
    u32 bytes;
    VkImageAspectFlags aspects{VK_IMAGE_ASPECT_COLOR_BIT};
};

std::optional<ColorFormat> FindColorFormat(u32 format) {
    if (format >= 0x79 && format <= 0x94) {
        // Decode ASTC into RGBA8 for hosts without native ASTC sampling.
        return format < 0x87
                   ? ColorFormat{VK_FORMAT_R8G8B8A8_UNORM, PixelFormat::A8B8G8R8_UNORM, 4}
                   : ColorFormat{VK_FORMAT_R8G8B8A8_SRGB, PixelFormat::A8B8G8R8_SRGB, 4};
    }
    switch (format) {
    case 0x01:
        return ColorFormat{VK_FORMAT_R8_UNORM, PixelFormat::R8_UNORM, 1};
    case 0x02:
        return ColorFormat{VK_FORMAT_R8_SNORM, PixelFormat::R8_SNORM, 1};
    case 0x03:
        return ColorFormat{VK_FORMAT_R8_UINT, PixelFormat::R8_UINT, 1};
    case 0x04:
        return ColorFormat{VK_FORMAT_R8_SINT, PixelFormat::R8_SINT, 1};
    case 0x05:
        return ColorFormat{VK_FORMAT_R16_SFLOAT, PixelFormat::R16_FLOAT, 2};
    case 0x06:
        return ColorFormat{VK_FORMAT_R16_UNORM, PixelFormat::R16_UNORM, 2};
    case 0x07:
        return ColorFormat{VK_FORMAT_R16_SNORM, PixelFormat::R16_SNORM, 2};
    case 0x08:
        return ColorFormat{VK_FORMAT_R16_UINT, PixelFormat::R16_UINT, 2};
    case 0x09:
        return ColorFormat{VK_FORMAT_R16_SINT, PixelFormat::R16_SINT, 2};
    case 0x0a:
        return ColorFormat{VK_FORMAT_R32_SFLOAT, PixelFormat::R32_FLOAT, 4};
    case 0x0b:
        return ColorFormat{VK_FORMAT_R32_UINT, PixelFormat::R32_UINT, 4};
    case 0x0c:
        return ColorFormat{VK_FORMAT_R32_SINT, PixelFormat::R32_SINT, 4};
    case 0x0d:
        return ColorFormat{VK_FORMAT_R8G8_UNORM, PixelFormat::R8G8_UNORM, 2};
    case 0x0e:
        return ColorFormat{VK_FORMAT_R8G8_SNORM, PixelFormat::R8G8_SNORM, 2};
    case 0x0f:
        return ColorFormat{VK_FORMAT_R8G8_UINT, PixelFormat::R8G8_UINT, 2};
    case 0x10:
        return ColorFormat{VK_FORMAT_R8G8_SINT, PixelFormat::R8G8_SINT, 2};
    case 0x11:
        return ColorFormat{VK_FORMAT_R16G16_SFLOAT, PixelFormat::R16G16_FLOAT, 4};
    case 0x12:
        return ColorFormat{VK_FORMAT_R16G16_UNORM, PixelFormat::R16G16_UNORM, 4};
    case 0x13:
        return ColorFormat{VK_FORMAT_R16G16_SNORM, PixelFormat::R16G16_SNORM, 4};
    case 0x14:
        return ColorFormat{VK_FORMAT_R16G16_UINT, PixelFormat::R16G16_UINT, 4};
    case 0x15:
        return ColorFormat{VK_FORMAT_R16G16_SINT, PixelFormat::R16G16_SINT, 4};
    case 0x16:
        return ColorFormat{VK_FORMAT_R32G32_SFLOAT, PixelFormat::R32G32_FLOAT, 8};
    case 0x17:
        return ColorFormat{VK_FORMAT_R32G32_UINT, PixelFormat::R32G32_UINT, 8};
    case 0x18:
        return ColorFormat{VK_FORMAT_R32G32_SINT, PixelFormat::R32G32_SINT, 8};
    case 0x25:
        return ColorFormat{VK_FORMAT_R8G8B8A8_UNORM, PixelFormat::A8B8G8R8_UNORM, 4};
    case 0x26:
        return ColorFormat{VK_FORMAT_R8G8B8A8_SNORM, PixelFormat::A8B8G8R8_SNORM, 4};
    case 0x27:
        return ColorFormat{VK_FORMAT_R8G8B8A8_UINT, PixelFormat::A8B8G8R8_UINT, 4};
    case 0x28:
        return ColorFormat{VK_FORMAT_R8G8B8A8_SINT, PixelFormat::A8B8G8R8_SINT, 4};
    case 0x38:
        return ColorFormat{VK_FORMAT_R8G8B8A8_SRGB, PixelFormat::A8B8G8R8_SRGB, 4};
    case 0x9a:
        return ColorFormat{VK_FORMAT_B8G8R8A8_UNORM, PixelFormat::B8G8R8A8_UNORM, 4};
    case 0x9c:
        return ColorFormat{VK_FORMAT_B8G8R8A8_SRGB, PixelFormat::B8G8R8A8_SRGB, 4};
    case 0x29:
        return ColorFormat{VK_FORMAT_R16G16B16A16_SFLOAT, PixelFormat::R16G16B16A16_FLOAT, 8};
    case 0x2a:
        return ColorFormat{VK_FORMAT_R16G16B16A16_UNORM, PixelFormat::R16G16B16A16_UNORM, 8};
    case 0x2b:
        return ColorFormat{VK_FORMAT_R16G16B16A16_SNORM, PixelFormat::R16G16B16A16_SNORM, 8};
    case 0x2c:
        return ColorFormat{VK_FORMAT_R16G16B16A16_UINT, PixelFormat::R16G16B16A16_UINT, 8};
    case 0x2d:
        return ColorFormat{VK_FORMAT_R16G16B16A16_SINT, PixelFormat::R16G16B16A16_SINT, 8};
    case 0x2e:
        return ColorFormat{VK_FORMAT_R32G32B32A32_SFLOAT, PixelFormat::R32G32B32A32_FLOAT, 16};
    case 0x2f:
        return ColorFormat{VK_FORMAT_R32G32B32A32_UINT, PixelFormat::R32G32B32A32_UINT, 16};
    case 0x30:
        return ColorFormat{VK_FORMAT_R32G32B32A32_SINT, PixelFormat::R32G32B32A32_SINT, 16};
    case 0x31:
        return ColorFormat{VK_FORMAT_S8_UINT, PixelFormat::S8_UINT, 1, VK_IMAGE_ASPECT_STENCIL_BIT};
    case 0x32:
        return ColorFormat{VK_FORMAT_D16_UNORM, PixelFormat::D16_UNORM, 2, VK_IMAGE_ASPECT_DEPTH_BIT};
    case 0x33:
        return ColorFormat{VK_FORMAT_X8_D24_UNORM_PACK32, PixelFormat::X8_D24_UNORM, 4,
                           VK_IMAGE_ASPECT_DEPTH_BIT};
    case 0x34:
        return ColorFormat{VK_FORMAT_D32_SFLOAT, PixelFormat::D32_FLOAT, 4, VK_IMAGE_ASPECT_DEPTH_BIT};
    case 0x35:
        return ColorFormat{VK_FORMAT_D24_UNORM_S8_UINT, PixelFormat::D24_UNORM_S8_UINT, 4,
                           VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT};
    case 0x36:
        return ColorFormat{VK_FORMAT_D32_SFLOAT_S8_UINT, PixelFormat::D32_FLOAT_S8_UINT, 8,
                           VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT};
    case 0x3d:
        return ColorFormat{VK_FORMAT_A2B10G10R10_UNORM_PACK32, PixelFormat::A2B10G10R10_UNORM, 4};
    case 0x3e:
        return ColorFormat{VK_FORMAT_A2B10G10R10_UINT_PACK32, PixelFormat::A2B10G10R10_UINT, 4};
    case 0x3f:
        return ColorFormat{VK_FORMAT_B10G11R11_UFLOAT_PACK32, PixelFormat::B10G11R11_FLOAT, 4};
    case 0x41:
        return ColorFormat{VK_FORMAT_BC1_RGB_UNORM_BLOCK, PixelFormat::BC1_RGBA_UNORM, 8};
    case 0x42:
        return ColorFormat{VK_FORMAT_BC1_RGBA_UNORM_BLOCK, PixelFormat::BC1_RGBA_UNORM, 8};
    case 0x43:
        return ColorFormat{VK_FORMAT_BC2_UNORM_BLOCK, PixelFormat::BC2_UNORM, 16};
    case 0x44:
        return ColorFormat{VK_FORMAT_BC3_UNORM_BLOCK, PixelFormat::BC3_UNORM, 16};
    case 0x45:
        return ColorFormat{VK_FORMAT_BC1_RGB_SRGB_BLOCK, PixelFormat::BC1_RGBA_SRGB, 8};
    case 0x46:
        return ColorFormat{VK_FORMAT_BC1_RGBA_SRGB_BLOCK, PixelFormat::BC1_RGBA_SRGB, 8};
    case 0x47:
        return ColorFormat{VK_FORMAT_BC2_SRGB_BLOCK, PixelFormat::BC2_SRGB, 16};
    case 0x48:
        return ColorFormat{VK_FORMAT_BC3_SRGB_BLOCK, PixelFormat::BC3_SRGB, 16};
    case 0x49:
        return ColorFormat{VK_FORMAT_BC4_UNORM_BLOCK, PixelFormat::BC4_UNORM, 8};
    case 0x4a:
        return ColorFormat{VK_FORMAT_BC4_SNORM_BLOCK, PixelFormat::BC4_SNORM, 8};
    case 0x4b:
        return ColorFormat{VK_FORMAT_BC5_UNORM_BLOCK, PixelFormat::BC5_UNORM, 16};
    case 0x4c:
        return ColorFormat{VK_FORMAT_BC5_SNORM_BLOCK, PixelFormat::BC5_SNORM, 16};
    case 0x4d:
        return ColorFormat{VK_FORMAT_BC7_UNORM_BLOCK, PixelFormat::BC7_UNORM, 16};
    case 0x4e:
        return ColorFormat{VK_FORMAT_BC7_SRGB_BLOCK, PixelFormat::BC7_SRGB, 16};
    case 0x4f:
        return ColorFormat{VK_FORMAT_BC6H_SFLOAT_BLOCK, PixelFormat::BC6H_SFLOAT, 16};
    case 0x50:
        return ColorFormat{VK_FORMAT_BC6H_UFLOAT_BLOCK, PixelFormat::BC6H_UFLOAT, 16};
    default:
        return std::nullopt;
    }
}

constexpr VkImageSubresourceRange BaseColorRange{
    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel = 0,
    .levelCount = 1,
    .baseArrayLayer = 0,
    .layerCount = 1,
};

constexpr VkImageSubresourceLayers BaseColorLayer{
    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .mipLevel = 0,
    .baseArrayLayer = 0,
    .layerCount = 1,
};

u32 ViewFormat(std::span<const u8> data, u32 base_format) {
    u32_le format;
    std::memcpy(&format, data.data() + 8, sizeof(format));
    return data[7] & 4 ? static_cast<u32>(format) : base_format;
}

bool HasCompatibleFormat(u32 format, u32 base) {
    return format == base || ((format == 0x25 || format == 0x38) &&
                              (base == 0x25 || base == 0x38)) ||
           ((format == 0x9a || format == 0x9c) && (base == 0x9a || base == 0x9c));
}

struct CopyRegion {
    u32 x;
    u32 y;
    u32 z;
    u32 width;
    u32 height;
    u32 depth;
};

VkFormat VertexFormat(u32 format) {
    switch (format) {
    case 0x02: return VK_FORMAT_R8_SNORM;
    case 0x03: return VK_FORMAT_R8_UINT;
    case 0x04: return VK_FORMAT_R8_SINT;
    case 0x07: return VK_FORMAT_R16_SNORM;
    case 0x08: return VK_FORMAT_R16_UINT;
    case 0x09: return VK_FORMAT_R16_SINT;
    case 0x0b: return VK_FORMAT_R32_UINT;
    case 0x0c: return VK_FORMAT_R32_SINT;
    case 0x0e: return VK_FORMAT_R8G8_SNORM;
    case 0x0f: return VK_FORMAT_R8G8_UINT;
    case 0x10: return VK_FORMAT_R8G8_SINT;
    case 0x13: return VK_FORMAT_R16G16_SNORM;
    case 0x14: return VK_FORMAT_R16G16_UINT;
    case 0x15: return VK_FORMAT_R16G16_SINT;
    case 0x17: return VK_FORMAT_R32G32_UINT;
    case 0x18: return VK_FORMAT_R32G32_SINT;
    case 0x19: return VK_FORMAT_R8G8B8_UNORM;
    case 0x1a: return VK_FORMAT_R8G8B8_SNORM;
    case 0x1b: return VK_FORMAT_R8G8B8_UINT;
    case 0x1c: return VK_FORMAT_R8G8B8_SINT;
    case 0x1d: return VK_FORMAT_R16G16B16_SFLOAT;
    case 0x1e: return VK_FORMAT_R16G16B16_UNORM;
    case 0x1f: return VK_FORMAT_R16G16B16_SNORM;
    case 0x20: return VK_FORMAT_R16G16B16_UINT;
    case 0x21: return VK_FORMAT_R16G16B16_SINT;
    case 0x22: return VK_FORMAT_R32G32B32_SFLOAT;
    case 0x23: return VK_FORMAT_R32G32B32_UINT;
    case 0x24: return VK_FORMAT_R32G32B32_SINT;
    case 0x26: return VK_FORMAT_R8G8B8A8_SNORM;
    case 0x27: return VK_FORMAT_R8G8B8A8_UINT;
    case 0x28: return VK_FORMAT_R8G8B8A8_SINT;
    case 0x2b: return VK_FORMAT_R16G16B16A16_SNORM;
    case 0x2c: return VK_FORMAT_R16G16B16A16_UINT;
    case 0x2d: return VK_FORMAT_R16G16B16A16_SINT;
    case 0x2f: return VK_FORMAT_R32G32B32A32_UINT;
    case 0x30: return VK_FORMAT_R32G32B32A32_SINT;
    case 0x3e: return VK_FORMAT_A2B10G10R10_UINT_PACK32;
    case 0x69: return VK_FORMAT_A2B10G10R10_SNORM_PACK32;
    case 0x6a: return VK_FORMAT_A2B10G10R10_SINT_PACK32;
    case 0x6b: return VK_FORMAT_A2B10G10R10_USCALED_PACK32;
    case 0x6c: return VK_FORMAT_A2B10G10R10_SSCALED_PACK32;
    default: {
        const auto color = FindColorFormat(format);
        return color && color->aspects == VK_IMAGE_ASPECT_COLOR_BIT ? color->vk
                                                                   : VK_FORMAT_UNDEFINED;
    }
    }
}

VkPrimitiveTopology Topology(u64 value) {
    switch (value) {
    case 0: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case 1: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case 3: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case 4: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case 5: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    case 6: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    case 7:
    case 8: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case 9: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    default: return VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
    }
}

VkBlendFactor BlendFactor(u32 value) {
    switch (value) {
    case 0x01: return VK_BLEND_FACTOR_ZERO;
    case 0x02: return VK_BLEND_FACTOR_ONE;
    case 0x03: return VK_BLEND_FACTOR_SRC_COLOR;
    case 0x04: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 0x05: return VK_BLEND_FACTOR_SRC_ALPHA;
    case 0x06: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 0x07: return VK_BLEND_FACTOR_DST_ALPHA;
    case 0x08: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 0x09: return VK_BLEND_FACTOR_DST_COLOR;
    case 0x0a: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 0x0b: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case 0x10: return VK_BLEND_FACTOR_SRC1_COLOR;
    case 0x11: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    case 0x12: return VK_BLEND_FACTOR_SRC1_ALPHA;
    case 0x13: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    case 0x61: return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case 0x62: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case 0x63: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case 0x64: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    default: return VK_BLEND_FACTOR_MAX_ENUM;
    }
}

} // Anonymous namespace

BananzaNvn2Replay::BananzaNvn2Replay(const Device& device_, MemoryAllocator& allocator_,
                                     Scheduler& scheduler_, const PipelineCache& pipeline_cache_,
                                     FrameDump& frame_dump_)
    : device{device_}, allocator{allocator_}, scheduler{scheduler_},
      pipeline_cache{pipeline_cache_}, frame_dump{frame_dump_},
      trace_images{std::getenv("EDEN_NX2_NVN_TRACE_IMAGES") != nullptr} {}

BananzaNvn2Replay::Image* BananzaNvn2Replay::FindImage(const Texture& texture, bool create) {
    const auto& d = texture.description;
    const u32 layers = texture.layout.layers;
    const bool volume = d.target == 2;
    const u32 depth = volume ? d.depth : 1;
    const auto& limits = device.GetPhysical().GetProperties().limits;
    if (!texture.serial || !texture.storage.gpu_address ||
        (d.target != 1 && !volume && d.target != 4 && d.target != 8 && d.target != 9) ||
        !layers || layers > device.GetPhysical().GetProperties().limits.maxImageArrayLayers ||
        d.samples > 1 || !depth || !d.width || !d.height ||
        d.width > (volume ? limits.maxImageDimension3D : limits.maxImageDimension2D) ||
        d.height > (volume ? limits.maxImageDimension3D : limits.maxImageDimension2D) ||
        (volume && depth > limits.maxImageDimension3D) || !d.levels ||
        d.levels > static_cast<u32>(std::bit_width(std::max({d.width, d.height, depth})))) {
        return nullptr;
    }
    const ImageKey key{texture.storage.gpu_address + texture.offset, d.width, d.height, d.format,
                       d.levels, d.target, layers, depth};
    if (const auto it = images.find(key); it != images.end()) {
        return &it->second;
    }
    if (!create) {
        return nullptr;
    }
    const auto format = FindColorFormat(d.format);
    if (!format) {
        return nullptr;
    }
    const bool depth_stencil = format->aspects != VK_IMAGE_ASPECT_COLOR_BIT;
    const auto depth_feature = depth_stencil ? VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT : 0;
    if (!device.IsFormatSupported(format->vk, VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | depth_feature |
                                                         VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                                         VK_FORMAT_FEATURE_TRANSFER_DST_BIT,
                                             FormatType::Optimal)) {
        return nullptr;
    }
    const bool attachment = !depth_stencil && device.IsFormatSupported(format->vk,
                                                      VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT,
                                                      FormatType::Optimal) &&
                            !(d.format >= 0x79 && d.format <= 0x94);
    const auto block = Core::BananzaNvn2Hle::GetTextureFormatBlock(d.format);
    const bool storage = !depth_stencil && block && block->width == 1 &&
                         device.IsFormatSupported(format->vk, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT,
                                                  FormatType::Optimal);
    const u64 bytes = std::max<u64>(texture.layout.size,
                                    u64{d.width} * d.height * format->bytes * layers * depth);
    const u64 max_image_bytes = std::clamp<u64>(device.GetDeviceLocalMemory() / 4,
                                                384ULL << 20, 4ULL << 30);
    if (bytes > max_image_bytes - image_bytes) {
        if (!image_budget_logged) {
            image_budget_logged = true;
            LOG_WARNING(Render_Vulkan, "Bananza NVN 2 image limit reached at {} MiB for format {:#x}, {}x{}",
                        image_bytes >> 20, d.format, d.width, d.height);
        }
        return nullptr;
    }
    const VkExtent2D extent{d.width, d.height};
    const bool mutable_format = d.format == 0x25 || d.format == 0x38 ||
                                d.format == 0x9a || d.format == 0x9c;
    auto image = allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .flags = (mutable_format ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                                       VK_IMAGE_CREATE_EXTENDED_USAGE_BIT : 0U) |
                 (d.target == 8 || d.target == 9 ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0U) |
                 (volume && attachment ? VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT : 0U),
        .imageType = volume ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
        .format = format->vk,
        .extent = {extent.width, extent.height, depth},
        .mipLevels = d.levels,
        .arrayLayers = layers,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                 VK_IMAGE_USAGE_SAMPLED_BIT | (attachment ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0U) |
                 (depth_stencil ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : 0U) |
                 (storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0U),
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    if (trace_images) {
        image.SetObjectNameEXT(fmt::format("NVN 2 {:#x} format {:#x} {}x{}",
                                           std::get<0>(key), d.format, d.width, d.height).c_str());
    }
    auto view = device.GetLogical().CreateImageView({
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = *image,
        .viewType = volume ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D,
        .format = format->vk,
        .subresourceRange = {format->aspects, 0, 1, 0, 1},
    });
    image_bytes += bytes;
    return &images.emplace(key, Image{std::move(image), std::move(view), extent, format->vk,
                                      format->surface, d.levels, layers, depth, volume, attachment, storage,
                                      format->aspects})
                .first->second;
}

VkImageView BananzaNvn2Replay::GetAttachmentView(Image& image, const AttachmentView& view) {
    if (!view.format || view.level >= image.levels) {
        return VK_NULL_HANDLE;
    }
    const u32 layer_count = image.volume ? std::max(1U, image.depth >> view.level)
                                          : image.layers;
    if (view.layer >= layer_count || !view.layers || view.layers > layer_count - view.layer) {
        return VK_NULL_HANDLE;
    }
    if (!image.volume && view.format == image.format && !view.level && !view.layer &&
        view.layers == 1) {
        return *image.view;
    }
    const auto key = std::tuple{view.format, view.level, view.layer, view.layers};
    auto found = image.attachment_views.find(key);
    if (found != image.attachment_views.end()) {
        return *found->second;
    }
    const bool color = image.aspects == VK_IMAGE_ASPECT_COLOR_BIT;
    const auto feature = color ? VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT
                               : VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (!device.IsFormatSupported(view.format, feature, FormatType::Optimal)) {
        return VK_NULL_HANDLE;
    }
    const VkImageViewUsageCreateInfo usage{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
        .usage = color ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                       : VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
    };
    auto host_view = device.GetLogical().CreateImageView({
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = &usage,
        .image = *image.image,
        .viewType = view.layers == 1 ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        .format = view.format,
        .subresourceRange = {image.aspects, view.level, 1, view.layer, view.layers},
    });
    return *image.attachment_views.emplace(key, std::move(host_view)).first->second;
}

void BananzaNvn2Replay::Initialize(Image& image) {
    if (image.initialized) {
        return;
    }
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([handle = *image.image, levels = image.levels, layers = image.layers,
                      aspects = image.aspects](vk::CommandBuffer cmdbuf) {
        const VkImageMemoryBarrier transition{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = handle,
            .subresourceRange = {aspects, 0, levels, 0, layers},
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, transition);
    });
    image.initialized = true;
}

void BananzaNvn2Replay::Barrier() {
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([](vk::CommandBuffer cmdbuf) {
        const VkMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, barrier);
    });
}

VkRect2D BananzaNvn2Replay::ScissorFor(VkExtent2D extent) const {
    if (!has_scissor) {
        return {{}, extent};
    }
    const s64 left = std::clamp<s64>(scissor.offset.x, 0, extent.width);
    const s64 top = std::clamp<s64>(scissor.offset.y, 0, extent.height);
    const s64 right = std::clamp<s64>(s64{scissor.offset.x} + scissor.extent.width, left,
                                      extent.width);
    const s64 bottom = std::clamp<s64>(s64{scissor.offset.y} + scissor.extent.height, top,
                                       extent.height);
    const s64 y = origin_lower_left ? extent.height - bottom : top;
    return {{static_cast<s32>(left), static_cast<s32>(y)},
            {static_cast<u32>(right - left), static_cast<u32>(bottom - top)}};
}

void BananzaNvn2Replay::ClearColor(const Operation& op) {
    if (!render_enabled || op.args.size() != 2 || op.payload.size() != 16 || op.args[0] >= colors.size() ||
        op.args[1] != 0xf || !colors[op.args[0]] || !color_views[op.args[0]].format) {
        return;
    }
    auto* image = FindImage(*colors[op.args[0]], true);
    if (!image || !image->color_attachment) {
        return;
    }
    const auto& view = color_views[op.args[0]];
    const auto attachment = GetAttachmentView(*image, view);
    if (!attachment) {
        return;
    }
    const VkExtent2D extent{std::max(1U, image->extent.width >> view.level),
                            std::max(1U, image->extent.height >> view.level)};
    const VkRect2D rect = ScissorFor(extent);
    if (!rect.extent.width || !rect.extent.height) {
        return;
    }
    const bool whole = rect.extent.width == extent.width && rect.extent.height == extent.height;
    const bool initialized = image->initialized;
    Initialize(*image);
    Barrier();
    VkClearColorValue value{};
    std::memcpy(&value, op.payload.data(), op.payload.size());
    TraceImage("clear", *colors[op.args[0]]);
    image->gpu_written = true;
    ++image->write_version;
    scheduler.Record([attachment, value, extent, rect, whole, initialized,
                      layers = view.layers](vk::CommandBuffer cmdbuf) {
        const VkRenderingAttachmentInfo color{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = attachment,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .loadOp = whole ? VK_ATTACHMENT_LOAD_OP_CLEAR
                            : initialized ? VK_ATTACHMENT_LOAD_OP_LOAD
                                          : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = {.color = value},
        };
        cmdbuf.BeginRendering({
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = {{}, extent},
            .layerCount = layers,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color,
        });
        if (!whole) {
            const VkClearAttachment clear{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .clearValue = {.color = value},
            };
            cmdbuf.ClearAttachments({clear}, {VkClearRect{rect, 0, layers}});
        }
        cmdbuf.EndRendering();
    });
    ++clears;
}

void BananzaNvn2Replay::ClearDepthStencil(const Operation& op) {
    if (!render_enabled || !depth_texture || op.args.size() != 3 || op.payload.size() != sizeof(float)) {
        return;
    }
    auto* image = FindImage(*depth_texture, true);
    if (!image || image->aspects == VK_IMAGE_ASPECT_COLOR_BIT) {
        return;
    }
    const auto attachment = GetAttachmentView(*image, depth_view);
    if (!attachment) {
        return;
    }
    const VkImageAspectFlags mask = ((op.args[0] & 255) ? VK_IMAGE_ASPECT_DEPTH_BIT : 0U) |
                                    ((op.args[2] & 255) == 255 ? VK_IMAGE_ASPECT_STENCIL_BIT : 0U);
    const VkImageAspectFlags aspects = image->aspects & mask;
    if (!aspects) {
        return;
    }
    VkClearDepthStencilValue value{.stencil = static_cast<u32>(op.args[1]) & 255};
    std::memcpy(&value.depth, op.payload.data(), sizeof(value.depth));
    const VkExtent2D extent{std::max(1U, image->extent.width >> depth_view.level),
                            std::max(1U, image->extent.height >> depth_view.level)};
    const VkRect2D rect = ScissorFor(extent);
    if (!rect.extent.width || !rect.extent.height) {
        return;
    }
    const bool initialized = image->initialized;
    Initialize(*image);
    image->gpu_written = true;
    ++image->write_version;
    Barrier();
    scheduler.Record([view = attachment, rect, aspects, image_aspects = image->aspects, value,
                      initialized, extent, layers = depth_view.layers](vk::CommandBuffer cmdbuf) {
        const VkRenderingAttachmentInfo target{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .loadOp = initialized ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        };
        cmdbuf.BeginRendering({
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = {{}, extent},
            .layerCount = layers,
            .pDepthAttachment = image_aspects & VK_IMAGE_ASPECT_DEPTH_BIT ? &target : nullptr,
            .pStencilAttachment = image_aspects & VK_IMAGE_ASPECT_STENCIL_BIT ? &target : nullptr,
        });
        const VkClearAttachment clear{.aspectMask = aspects, .clearValue = {.depthStencil = value}};
        const VkClearRect clear_rect{.rect = rect, .layerCount = layers};
        cmdbuf.ClearAttachments({clear}, {clear_rect});
        cmdbuf.EndRendering();
    });
    ++clears;
}

void BananzaNvn2Replay::CopyTexture(const Operation& op, const Submission& submission) {
    if (op.textures.size() != 2 || op.args.size() != 5 ||
        op.payload.size() != 32 + 2 * sizeof(CopyRegion)) {
        return;
    }
    const auto source_address = op.textures[0].storage.gpu_address + op.textures[0].offset;
    const auto reject = [&](const char* reason) {
        TraceImage(reason, op.textures[1], source_address);
    };
    TraceImage("copy request", op.textures[1], source_address);
    if (op.args[2] & ~0x1fULL) {
        reject("copy flags unsupported");
        return;
    }
    struct CopyView {
        u32 level;
        u32 layer;
        bool volume;
        CopyRegion region;
    };
    const auto decode = [&](size_t index) -> std::optional<CopyView> {
        const auto& texture = op.textures[index];
        const auto& d = texture.description;
        const auto view = std::span{op.payload}.subspan(16 * index, 16);
        const u8 flags = op.args[3 + index] ? view[7] : 0;
        const u32 level = flags & 1 ? view[4] : 0;
        const u32 layer = flags & 2 ? u32{view[0]} | (u32{view[1]} << 8) : 0;
        if (level >= d.levels || ((flags & 4) && ViewFormat(view, d.format) != d.format) ||
            ((flags & 0x20) && view[6] != d.target)) {
            return std::nullopt;
        }
        CopyRegion region{};
        std::memcpy(&region, op.payload.data() + 32 + index * sizeof(region), sizeof(region));
        const u32 width = std::max(1U, d.width >> level);
        const u32 height = std::max(1U, d.height >> level);
        const bool volume = d.target == 2;
        const u32 slices = volume ? std::max(1U, d.depth >> level) : texture.layout.layers;
        if (!region.width || !region.height || !region.depth || region.x > width ||
            region.width > width - region.x || region.y > height ||
            region.height > height - region.y || layer >= slices ||
            region.z >= slices - layer || region.depth > slices - layer - region.z) {
            return std::nullopt;
        }
        return CopyView{level, layer + region.z, volume, region};
    };
    const auto source_view = decode(0);
    const auto dest_view = decode(1);
    if (!source_view || !dest_view || source_view->volume != dest_view->volume ||
        source_view->region.depth != dest_view->region.depth) {
        reject("copy subresource unsupported");
        return;
    }
    const auto& from = source_view->region;
    const auto& to = dest_view->region;
    auto* source = FindImage(op.textures[0], true);
    if (!source || (source->aspects != VK_IMAGE_ASPECT_COLOR_BIT &&
                    source->aspects != VK_IMAGE_ASPECT_DEPTH_BIT) ||
        !(source->aspects == VK_IMAGE_ASPECT_COLOR_BIT
              ? PrepareSampledImage(*source, op.textures[0], submission)
              : source->initialized)) {
        reject("copy source unavailable");
        return;
    }
    auto* dest = FindImage(op.textures[1], true);
    if (!dest || dest == source ||
        (dest->aspects != VK_IMAGE_ASPECT_COLOR_BIT && dest->aspects != VK_IMAGE_ASPECT_DEPTH_BIT)) {
        reject("copy destination unsupported");
        return;
    }
    const bool flip_x = op.args[2] & 4;
    const bool flip_y = op.args[2] & 8;
    const bool flip_z = op.args[2] & 16;
    const auto subresource = [](const CopyView& view, VkImageAspectFlags aspect, u32 slice) {
        return VkImageSubresourceLayers{aspect, view.level,
                                         view.volume ? 0U : view.layer + slice, 1};
    };
    const auto offset = [](const CopyView& view, u32 slice) {
        return VkOffset3D{static_cast<s32>(view.region.x), static_cast<s32>(view.region.y),
                          view.volume ? static_cast<s32>(view.layer + slice) : 0};
    };
    const auto prepare_dest = [&] {
        if (dest->aspects == VK_IMAGE_ASPECT_COLOR_BIT && !dest->gpu_written &&
            submission.texture_snapshots.contains(op.textures[1].serial)) {
            if (!PrepareSampledImage(*dest, op.textures[1], submission)) {
                return false;
            }
        }
        Initialize(*dest);
        Barrier();
        return true;
    };
    if (source->aspects != dest->aspects) {
        const bool depth_to_color =
            (source->format == VK_FORMAT_D32_SFLOAT && dest->format == VK_FORMAT_R32_SFLOAT) ||
            (source->format == VK_FORMAT_D16_UNORM && dest->format == VK_FORMAT_R16_UNORM);
        if (!depth_to_color || from.width != to.width || from.height != to.height ||
            flip_x || flip_y) {
            reject("copy aspect conversion unsupported");
            return;
        }
        const u64 bytes = u64{from.width} * from.height *
                          (source->format == VK_FORMAT_D32_SFLOAT ? 4 : 2);
        if (bytes > (64ULL << 20) || !prepare_dest()) {
            return;
        }
        auto scratch = CreateWrappedBuffer(allocator, bytes, MemoryUsage::DeviceLocal);
        for (u32 slice = 0; slice < to.depth; ++slice) {
            const u32 source_slice = flip_z ? from.depth - 1 - slice : slice;
            const VkBufferImageCopy read{
                .imageSubresource = subresource(*source_view, VK_IMAGE_ASPECT_DEPTH_BIT, source_slice),
                .imageOffset = offset(*source_view, source_slice),
                .imageExtent = {from.width, from.height, 1},
            };
            const VkBufferImageCopy write{
                .imageSubresource = subresource(*dest_view, VK_IMAGE_ASPECT_COLOR_BIT, slice),
                .imageOffset = offset(*dest_view, slice),
                .imageExtent = {to.width, to.height, 1},
            };
            scheduler.Record([src = *source->image, dst = *dest->image, buffer = *scratch,
                              read, write](vk::CommandBuffer cmdbuf) {
                const VkMemoryBarrier transfer{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                };
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       0, transfer);
                cmdbuf.CopyImageToBuffer(src, VK_IMAGE_LAYOUT_GENERAL, buffer, {read});
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       0, transfer);
                cmdbuf.CopyBufferToImage(buffer, dst, VK_IMAGE_LAYOUT_GENERAL, {write});
            });
        }
        uploads.push_back(std::move(scratch));
        dest->gpu_written = true;
        ++dest->write_version;
        TraceImage("depth copy", op.textures[1], source_address);
        ++copies;
        return;
    }
    const bool depth = source->aspects == VK_IMAGE_ASPECT_DEPTH_BIT;
    const bool linear = (op.args[2] & 1) != 0;
    const bool blit = source->format != dest->format || from.width != to.width ||
                      from.height != to.height || flip_x || flip_y;
    if (depth && (source->format != dest->format || (blit && linear))) {
        if (trace_images && traced_images.emplace("depth blit filter", source_address, op.args[2]).second) {
            LOG_INFO(Render_Vulkan,
                     "NVN 2 depth blit flags {:#x}, source {}x{}, destination {}x{}",
                     op.args[2], from.width, from.height, to.width, to.height);
        }
        reject("depth blit unsupported");
        return;
    }
    const auto filter = linear && !depth ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    if (blit) {
        const VkFormatFeatureFlags source_features =
            VK_FORMAT_FEATURE_BLIT_SRC_BIT |
            (filter == VK_FILTER_LINEAR ? VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT : 0);
        if (!device.IsFormatSupported(source->format, source_features, FormatType::Optimal) ||
            !device.IsFormatSupported(dest->format, VK_FORMAT_FEATURE_BLIT_DST_BIT,
                                      FormatType::Optimal)) {
            reject("copy blit format unsupported");
            return;
        }
    } else {
        for (size_t index = 0; index < op.textures.size(); ++index) {
            const auto& texture = op.textures[index];
            const auto& view = index == 0 ? *source_view : *dest_view;
            const auto& region = view.region;
            const u32 width = std::max(1U, texture.description.width >> view.level);
            const u32 height = std::max(1U, texture.description.height >> view.level);
            const auto block = Core::BananzaNvn2Hle::GetTextureFormatBlock(texture.description.format);
            if (block && (region.x % block->width || region.y % block->height ||
                          (region.width % block->width && region.x + region.width != width) ||
                          (region.height % block->height && region.y + region.height != height))) {
                reject("copy block alignment unsupported");
                return;
            }
        }
    }
    if (!prepare_dest()) {
        return;
    }
    dest->gpu_written = true;
    ++dest->write_version;
    TraceImage("copy", op.textures[1], source_address);
    for (u32 slice = 0; slice < to.depth; ++slice) {
        const u32 source_slice = flip_z ? from.depth - 1 - slice : slice;
        const auto source_offset = offset(*source_view, source_slice);
        const auto dest_offset = offset(*dest_view, slice);
        if (blit) {
            VkImageBlit region{
                .srcSubresource = subresource(*source_view, source->aspects, source_slice),
                .srcOffsets = {source_offset,
                               {source_offset.x + static_cast<s32>(from.width),
                                source_offset.y + static_cast<s32>(from.height), source_offset.z + 1}},
                .dstSubresource = subresource(*dest_view, dest->aspects, slice),
                .dstOffsets = {dest_offset,
                               {dest_offset.x + static_cast<s32>(to.width),
                                dest_offset.y + static_cast<s32>(to.height), dest_offset.z + 1}},
            };
            if (flip_x) std::swap(region.srcOffsets[0].x, region.srcOffsets[1].x);
            if (flip_y) std::swap(region.srcOffsets[0].y, region.srcOffsets[1].y);
            scheduler.Record([src = *source->image, dst = *dest->image, region, filter](
                                 vk::CommandBuffer cmdbuf) {
                cmdbuf.BlitImage(src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL,
                                 {region}, filter);
            });
        } else {
            const VkImageCopy copy{
                .srcSubresource = subresource(*source_view, source->aspects, source_slice),
                .srcOffset = source_offset,
                .dstSubresource = subresource(*dest_view, dest->aspects, slice),
                .dstOffset = dest_offset,
                .extent = {from.width, from.height, 1},
            };
            scheduler.Record([src = *source->image, dst = *dest->image, copy](vk::CommandBuffer cmdbuf) {
                cmdbuf.CopyImage(src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, {copy});
            });
        }
    }
    ++copies;
}

void BananzaNvn2Replay::CopyBufferToTexture(const Operation& op, const Submission& submission) {
    if (op.args.size() != 3 || op.args[1] & ~2ULL || op.textures.size() != 1 ||
        op.payload.size() != 16 + sizeof(CopyRegion)) {
        return;
    }
    const auto& texture = op.textures[0];
    const auto& desc = texture.description;
    const auto block = Core::BananzaNvn2Hle::GetTextureFormatBlock(desc.format);
    const auto view = std::span{op.payload}.first(16);
    const u8 flags = op.args[2] ? view[7] : 0;
    const u32 level = flags & 1 ? view[4] : 0;
    const u32 base_layer = flags & 2 ? u32{view[0]} | (u32{view[1]} << 8) : 0;
    if (!block || level >= desc.levels || base_layer >= texture.layout.layers ||
        ((flags & 4) && ViewFormat(view, desc.format) != desc.format)) {
        return;
    }
    CopyRegion region{};
    std::memcpy(&region, op.payload.data() + 16, sizeof(region));
    const VkExtent2D extent{std::max(1U, desc.width >> level), std::max(1U, desc.height >> level)};
    const bool volume = desc.target == 2;
    const u32 available = volume ? std::max(1U, desc.depth >> level)
                                 : texture.layout.layers - base_layer;
    if (!region.width || !region.height || !region.depth || region.x > extent.width ||
        region.y > extent.height || region.width > extent.width - region.x ||
        region.height > extent.height - region.y || region.z >= available ||
        region.depth > available - region.z || region.x % block->width ||
        region.y % block->height ||
        (region.width % block->width && region.x + region.width != extent.width) ||
        (region.height % block->height && region.y + region.height != extent.height)) {
        return;
    }
    const u64 columns = (u64{region.width} + block->width - 1) / block->width;
    const u64 rows = (u64{region.height} + block->height - 1) / block->height;
    const u64 row_bytes = columns * block->bytes;
    const u64 stride = copy_row_stride ? copy_row_stride : row_bytes;
    if (stride < row_bytes || stride > (8ULL << 20) / (rows * region.depth)) {
        return;
    }
    const auto data = submission.buffer_snapshots.find(
        {op.args[0], stride * (rows * region.depth - 1) + row_bytes});
    if (data == submission.buffer_snapshots.end()) {
        TraceImage("buffer copy source unavailable", texture, op.args[0]);
        return;
    }
    auto* image = FindImage(texture, true);
    if (!image || image->aspects != VK_IMAGE_ASPECT_COLOR_BIT) {
        return;
    }
    const bool astc = desc.format >= 0x79 && desc.format <= 0x94;
    const u64 layer_bytes = astc ? u64{region.width} * region.height * 4 : row_bytes * rows;
    auto upload = CreateWrappedBuffer(allocator, layer_bytes * region.depth, MemoryUsage::Upload);
    std::vector<u8> compressed(astc ? row_bytes * rows : 0);
    for (u32 z = 0; z < region.depth; ++z) {
        auto layer = upload.Mapped().subspan(z * layer_bytes, layer_bytes);
        auto output = astc ? std::span{compressed} : layer;
        for (u64 row = 0; row < rows; ++row) {
            std::memcpy(output.data() + row * row_bytes,
                        data->second.data() + (z * rows + row) * stride, row_bytes);
        }
        if (astc) {
            Tegra::Texture::ASTC::Decompress(compressed, region.width, region.height, 1,
                                            block->width, block->height, layer);
        }
    }
    upload.Flush();
    Initialize(*image);
    Barrier();
    const VkBufferImageCopy copy{
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level,
                             volume ? 0U : base_layer + region.z, volume ? 1U : region.depth},
        .imageOffset = {static_cast<s32>(region.x), static_cast<s32>(region.y),
                        volume ? static_cast<s32>(region.z) : 0},
        .imageExtent = {region.width, region.height, volume ? region.depth : 1U},
    };
    scheduler.Record([buffer = *upload, target = *image->image, copy](vk::CommandBuffer cmdbuf) {
        cmdbuf.CopyBufferToImage(buffer, target, VK_IMAGE_LAYOUT_GENERAL, {copy});
    });
    uploads.push_back(std::move(upload));
    image->gpu_written = true;
    ++image->write_version;
    TraceImage("buffer copy", texture, op.args[0]);
    ++copies;
}

VkBuffer BananzaNvn2Replay::Upload(std::span<const u8> bytes, VkBufferUsageFlags extra_usage) {
    if (bytes.empty()) {
        return VK_NULL_HANDLE;
    }
    auto buffer = allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = bytes.size(),
            .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | extra_usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        },
        MemoryUsage::Upload);
    std::memcpy(buffer.Mapped().data(), bytes.data(), bytes.size());
    buffer.Flush();
    const VkBuffer handle = *buffer;
    uploads.push_back(std::move(buffer));
    return handle;
}

VkBuffer BananzaNvn2Replay::UploadRange(const Submission& submission, Range range) {
    if (const auto storage = storage_uploads.find(range); storage != storage_uploads.end()) {
        return *storage->second.buffer;
    }
    if (const auto cached = uploaded_ranges.find(range); cached != uploaded_ranges.end()) {
        return cached->second;
    }
    const auto bytes = submission.buffer_snapshots.find(range);
    if (bytes == submission.buffer_snapshots.end()) {
        return VK_NULL_HANDLE;
    }
    std::vector<u8> updated;
    for (const auto& write : buffer_updates) {
        const auto begin = std::max(range.first, write.address);
        const auto end = std::min(range.first + range.second, write.address + write.bytes.size());
        if (begin >= end) {
            continue;
        }
        if (updated.empty()) {
            updated = bytes->second;
        }
        std::memcpy(updated.data() + (begin - range.first),
                    write.bytes.data() + (begin - write.address), end - begin);
    }
    const auto buffer = Upload(updated.empty() ? bytes->second : updated);
    uploaded_ranges.emplace(range, buffer);
    return buffer;
}

std::map<BananzaNvn2Replay::Range, BananzaNvn2Replay::StorageUpload>::iterator
BananzaNvn2Replay::FindStorageRange(Range range) {
    auto upper = storage_uploads.upper_bound({range.first, std::numeric_limits<u64>::max()});
    if (upper != storage_uploads.begin()) {
        const auto entry = std::prev(upper);
        const auto& [start, size] = entry->first;
        const u64 offset = range.first - start;
        if (offset < size && range.second <= size - offset) {
            return entry;
        }
    }
    return storage_uploads.end();
}

std::pair<VkBuffer, u64> BananzaNvn2Replay::UploadBufferRange(const Submission& submission,
                                                            Range range) {
    if (const auto storage = FindStorageRange(range); storage != storage_uploads.end()) {
        return {*storage->second.buffer, range.first - storage->first.first};
    }
    if (const auto [upload, offset] = WritableStorage(submission, range); upload) {
        return {*upload->buffer, offset};
    }
    return {UploadRange(submission, range), 0};
}

void BananzaNvn2Replay::CollectStorageSpans(const Submission& submission) {
    storage_spans.clear();
    const u64 alignment =
        device.GetPhysical().GetProperties().limits.minStorageBufferOffsetAlignment;
    StorageSpan* span{};
    for (const auto& [range, guest] : submission.storage_addresses) {
        const auto [address, size] = range;
        // Only share Vulkan storage when both addresses describe the same aligned bytes.
        if (!span || address >= span->range.first + span->range.second ||
            address - span->range.first != guest - span->guest_address ||
            (address - span->range.first) % alignment) {
            span = &storage_spans.emplace(range, StorageSpan{range, guest, {range}}).first->second;
            continue;
        }
        span->range.second = std::max(span->range.second, address - span->range.first + size);
        span->snapshots.push_back(range);
    }
}

std::pair<BananzaNvn2Replay::StorageUpload*, u64> BananzaNvn2Replay::WritableStorage(
    const Submission& submission, Range range) {
    if (const auto found = FindStorageRange(range); found != storage_uploads.end()) {
        const u64 offset = range.first - found->first.first;
        const auto align = device.GetPhysical().GetProperties().limits.minStorageBufferOffsetAlignment;
        return offset % align == 0 ? std::pair{&found->second, offset}
                                   : std::pair<StorageUpload*, u64>{};
    }
    if (!submission.storage_addresses.contains(range)) {
        return {};
    }
    auto entry = storage_spans.upper_bound({range.first, std::numeric_limits<u64>::max()});
    if (entry == storage_spans.begin()) {
        return {};
    }
    const auto& span = std::prev(entry)->second;
    const u64 offset = range.first - span.range.first;
    if (offset >= span.range.second || range.second > span.range.second - offset ||
        !std::ranges::binary_search(span.snapshots, range)) {
        return {};
    }
    const u64 end = span.guest_address + span.range.second;
    for (const auto& [other_range, upload] : storage_uploads) {
        if (upload.guest_address < end && span.guest_address < upload.guest_address + other_range.second) {
            return {};
        }
    }
    auto buffer = allocator.CreateBuffer({
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = span.range.second,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    }, MemoryUsage::Download);
    auto mapped = buffer.Mapped();
    for (const auto& snapshot : span.snapshots) {
        const auto& bytes = submission.buffer_snapshots.at(snapshot);
        std::ranges::copy(bytes, mapped.begin() + (snapshot.first - span.range.first));
    }
    for (const auto& write : buffer_updates) {
        const auto begin = std::max(span.range.first, write.address);
        const auto limit = std::min(span.range.first + span.range.second,
                                    write.address + write.bytes.size());
        if (begin < limit) {
            std::memcpy(mapped.data() + begin - span.range.first,
                        write.bytes.data() + begin - write.address,
                        limit - begin);
        }
    }
    buffer.Flush();
    const auto upload = storage_uploads.emplace(
        span.range, StorageUpload{std::move(buffer), span.guest_address});
    return {&upload.first->second, offset};
}

void BananzaNvn2Replay::WriteBackStorage(const Submission& submission) {
    if (!submission.guest_memory) {
        return;
    }
    for (auto& [range, upload] : storage_uploads) {
        if (upload.written_bytes) {
            upload.buffer.Invalidate();
            submission.guest_memory->WriteBlockUnsafe(upload.guest_address,
                                                       upload.buffer.Mapped().data(),
                                                       upload.written_bytes);
        }
    }
}

bool BananzaNvn2Replay::UploadTexture(Image& image, const Texture& texture,
                                     std::span<const u8> data) {
    const auto& d = texture.description;
    const auto block = Core::BananzaNvn2Hle::GetTextureFormatBlock(d.format);
    if (!block || (texture.layout.stride && d.levels != 1)) {
        return false;
    }
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
    const bool decode_astc = d.format >= 0x79 && d.format <= 0x94;
    size_t upload_size{};
    for (u32 level = 0; level < d.levels; ++level) {
        const auto offset = Core::BananzaNvn2Hle::TextureLevelOffset(d, level);
        const u32 width = std::max(d.width >> level, 1U);
        const u32 height = std::max(d.height >> level, 1U);
        const u32 depth = std::max(image.depth >> level, 1U);
        const u32 columns = (width + block->width - 1) / block->width;
        const u32 rows = (height + block->height - 1) / block->height;
        u32 tile = texture.layout.tile_height;
        while (tile && (8U << (tile - 1)) >= rows) {
            --tile;
        }
        u32 tile_depth = texture.layout.tile_depth;
        while (tile_depth && (1U << (tile_depth - 1)) >= depth) {
            --tile_depth;
        }
        const size_t bytes = texture.layout.stride
                                 ? texture.layout.stride * (rows - 1) + columns * block->bytes
                                 : Tegra::Texture::CalculateSize(true, block->bytes, columns,
                                                                  rows, depth, tile, tile_depth);
        const auto last_layer = (image.layers - 1) * texture.layout.layer_size;
        if (!offset || last_layer > data.size() || *offset > data.size() - last_layer ||
            bytes > data.size() - last_layer - *offset) {
            return false;
        }
        const size_t alignment = decode_astc ? 4 : std::lcm(size_t{4}, size_t{block->bytes});
        upload_size = (upload_size + alignment - 1) / alignment * alignment;
        copies.push_back({
            .bufferOffset = upload_size,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, image.layers},
            .imageExtent = {width, height, depth},
        });
        const auto upload_stride = decode_astc ? size_t{width} * height * depth * 4
                                               : size_t{columns} * rows * depth * block->bytes;
        levels.push_back({*offset, texture.layout.layer_size, upload_stride, columns, rows, tile,
                          tile_depth});
        upload_size += upload_stride * image.layers;
    }
    auto upload = CreateWrappedBuffer(allocator, upload_size, MemoryUsage::Upload);
    for (u32 level = 0; level < levels.size(); ++level) {
        const auto& src = levels[level];
        const size_t row = size_t{src.width} * block->bytes;
        const auto extent = copies[level].imageExtent;
        std::vector<u8> astc(decode_astc ? row * src.height * extent.depth : 0);
        for (u32 layer = 0; layer < image.layers; ++layer) {
            const size_t offset = src.guest_offset + layer * src.bytes_per_layer;
            auto linear = upload.Mapped().subspan(copies[level].bufferOffset +
                                                      layer * src.upload_stride,
                                                  src.upload_stride);
            auto dest = decode_astc ? std::span{astc} : linear;
            if (texture.layout.stride) {
                for (u32 y = 0; y < src.height; ++y) {
                    std::memcpy(dest.data() + row * y,
                                data.data() + offset + texture.layout.stride * y, row);
                }
            } else {
                Tegra::Texture::UnswizzleTexture(dest, data.subspan(offset), block->bytes,
                                                src.width, src.height, extent.depth,
                                                src.tile_height, src.tile_depth, 0, true);
            }
            if (decode_astc) {
                Tegra::Texture::ASTC::Decompress(astc, extent.width, extent.height, extent.depth, block->width,
                                                block->height, linear);
            }
        }
    }
    upload.Flush();
    Initialize(image);
    Barrier();
    scheduler.Record([buffer = *upload, target = *image.image, copies = std::move(copies)](
                         vk::CommandBuffer cmdbuf) {
        cmdbuf.CopyBufferToImage(buffer, target, VK_IMAGE_LAYOUT_GENERAL, copies);
    });
    uploads.push_back(std::move(upload));
    ++image.write_version;
    return true;
}

VkImage BananzaNvn2Replay::PrepareDepthComparisonImage(const Image& source, VkFormat depth_format) {
    const auto found = depth_comparison_images.find(&source);
    if (found != depth_comparison_images.end() && found->second.write_version == source.write_version) {
        return *found->second.image;
    }
    const auto features = device.GetPhysical().GetFormatProperties3(depth_format).optimalTilingFeatures;
    constexpr auto required = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT |
                              VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT;
    if ((features & required) != required ||
        (depth_format == VK_FORMAT_D32_SFLOAT && !device.IsExtDepthRangeUnrestrictedSupported())) {
        return VK_NULL_HANDLE;
    }

    std::vector<VkBufferImageCopy> color_regions;
    std::vector<VkBufferImageCopy> depth_regions;
    color_regions.reserve(source.levels);
    depth_regions.reserve(source.levels);
    u64 bytes{};
    const u32 texel_bytes = depth_format == VK_FORMAT_D32_SFLOAT ? 4 : 2;
    for (u32 level = 0; level < source.levels; ++level) {
        const u32 width = std::max(1U, source.extent.width >> level);
        const u32 height = std::max(1U, source.extent.height >> level);
        color_regions.push_back({
            .bufferOffset = bytes,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, source.layers},
            .imageExtent = {width, height, 1},
        });
        auto& depth_region = depth_regions.emplace_back(color_regions.back());
        depth_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        bytes = Common::AlignUp(bytes + u64{width} * height * source.layers * texel_bytes, u64{4});
    }
    const u64 max_image_bytes = std::clamp<u64>(device.GetDeviceLocalMemory() / 4,
                                                384ULL << 20, 4ULL << 30);
    if (bytes > (64ULL << 20) ||
        (found == depth_comparison_images.end() && bytes > max_image_bytes - image_bytes)) {
        return VK_NULL_HANDLE;
    }

    auto entry = found;
    const bool created = entry == depth_comparison_images.end();
    if (created) {
        auto image = allocator.CreateImage(VkImageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .flags = source.layers >= 6 && source.extent.width == source.extent.height
                         ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0U,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = depth_format,
            .extent = {source.extent.width, source.extent.height, 1},
            .mipLevels = source.levels,
            .arrayLayers = source.layers,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        });
        entry = depth_comparison_images.emplace(&source, DepthComparisonImage{std::move(image)}).first;
        image_bytes += bytes;
        if (trace_images) {
            LOG_INFO(Render_Vulkan, "NVN 2 depth comparison copy: {}x{}, {} layers, {} levels, format {}",
                     source.extent.width, source.extent.height, source.layers, source.levels,
                     static_cast<u32>(depth_format));
        }
    }
    auto& depth = entry->second;
    auto scratch = CreateWrappedBuffer(allocator, bytes, MemoryUsage::DeviceLocal);
    const auto target = *depth.image;
    Barrier();
    // Preserve the texel bits across the color and depth formats, as in DKB Recomp.
    scheduler.Record([src = *source.image, target, buffer = *scratch,
                      color_regions = std::move(color_regions),
                      depth_regions = std::move(depth_regions),
                      created, levels = source.levels, layers = source.layers](
                         vk::CommandBuffer cmdbuf) {
        if (created) {
            const VkImageMemoryBarrier transition{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = target,
                .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, levels, 0, layers},
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, transition);
        }
        cmdbuf.CopyImageToBuffer(src, VK_IMAGE_LAYOUT_GENERAL, buffer, color_regions);
        const VkMemoryBarrier transfer{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, transfer);
        cmdbuf.CopyBufferToImage(buffer, target, VK_IMAGE_LAYOUT_GENERAL, depth_regions);
    });
    uploads.push_back(std::move(scratch));
    depth.write_version = source.write_version;
    ++copies;
    return target;
}

VkDescriptorImageInfo BananzaNvn2Replay::BindUnboundTexture(VkImageViewType type) {
    const auto& logical = device.GetLogical();
    auto found = unbound_textures.find(type);
    if (found == unbound_textures.end()) {
        const bool cube = type == VK_IMAGE_VIEW_TYPE_CUBE || type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
        const u32 layers = cube ? 6 : 1;
        constexpr VkFormat Format = VK_FORMAT_R8G8B8A8_UNORM;
        auto image = allocator.CreateImage(VkImageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0U,
            .imageType = type == VK_IMAGE_VIEW_TYPE_3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
            .format = Format,
            .extent = {1, 1, 1},
            .mipLevels = 1,
            .arrayLayers = layers,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        });
        const VkImageSubresourceRange subresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
        auto view = logical.CreateImageView({
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = *image,
            .viewType = type,
            .format = Format,
            .subresourceRange = subresource,
        });
        if (!unbound_texture_sampler) {
            unbound_texture_sampler = logical.CreateSampler({
                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            });
        }
        scheduler.RequestOutsideRenderPassOperationContext();
        scheduler.Record([handle = *image, subresource](vk::CommandBuffer cmdbuf) {
            const VkImageMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = handle,
                .subresourceRange = subresource,
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, barrier);
            constexpr VkClearColorValue zero{};
            cmdbuf.ClearColorImage(handle, VK_IMAGE_LAYOUT_GENERAL, zero, {subresource});
        });
        found = unbound_textures.emplace(type, UnboundTexture{std::move(image), std::move(view)}).first;
        ++clears;
    }
    return {*unbound_texture_sampler, *found->second.view, VK_IMAGE_LAYOUT_GENERAL};
}

std::optional<VkDescriptorImageInfo> BananzaNvn2Replay::BindTexture(
    const Submission& submission, u32 stage, const Shader::TextureDescriptor& binding,
    std::span<Image* const> destinations, std::optional<u64> compute_handle) {
    const auto& bound_program = stage == 5 ? compute_program
                              : stage == 1 ? fragment_program
                              : stage == 2 ? geometry_program
                                           : graphics_program;
    const auto& program = bound_program->shaders;
    const auto staged = std::ranges::find(program, stage == 1 ? 4U : stage == 2 ? 3U : stage,
                                           &Core::BananzaNvn2Hle::ResourceShaderInfo::stage);
    if (staged == program.end()) {
        NoteSkip("missing texture shader stage");
        return std::nullopt;
    }
    const u32 offset = binding.cbuf_offset;
    u64 handle{};
    // The SDK's direct handle table begins at 0x20 and uses eight bytes per slot.
    constexpr u32 HandleBase = 0x20;
    constexpr u32 HandleStride = 8;
    if (compute_handle) {
        handle = *compute_handle;
    } else if ((binding.cbuf_index == 2 || (stage == 0 && binding.cbuf_index == 0)) &&
        offset >= HandleBase &&
        (offset - HandleBase) % HandleStride == 0 &&
        (offset - HandleBase) / HandleStride < texture_handles[stage].size()) {
        handle = texture_handles[stage][(offset - HandleBase) / HandleStride];
    } else if (binding.cbuf_index == 1 && offset <= staged->constant_size &&
               staged->constant_size - offset >= 4) {
        u32_le word;
        std::memcpy(&word, staged->code.data() + staged->constant_offset + offset, 4);
        handle = word;
    } else if (binding.cbuf_index >= 3 && binding.cbuf_index - 3 < uniform_buffers[stage].size()) {
        const auto found = submission.buffer_snapshots.find(uniform_buffers[stage][binding.cbuf_index - 3]);
        if (found == submission.buffer_snapshots.end() || offset > found->second.size() ||
            found->second.size() - offset < 4) {
            NoteSkip("texture handle buffer range missing at submit");
            return std::nullopt;
        }
        u32_le word;
        std::memcpy(&word, found->second.data() + offset, 4);
        handle = word;
    } else {
        if (trace_images && draw_diagnostics.emplace(bound_program->serial,
                                                      "texture handle location").second) {
            LOG_INFO(Render_Vulkan, "NVN 2 program {} stage {} texture at bank {} offset {:#x}",
                     bound_program->serial, stage, binding.cbuf_index, offset);
        }
        NoteSkip("unsupported texture handle bank or offset");
        return std::nullopt;
    }
    if (binding.shift_left >= 32) {
        return std::nullopt;
    }
    handle <<= binding.shift_left;
    const u32 texture_id = static_cast<u32>(handle) & 0xfffff;
    const u32 sampler_id = static_cast<u32>(handle >> 20) & 0xfff;
    const auto textures = submission.texture_descriptors.find(texture_pool);
    if (textures == submission.texture_descriptors.end()) {
        NoteSkip("missing texture pool snapshot");
        return std::nullopt;
    }
    const auto texture = textures->second.find(texture_id);
    Core::BananzaNvn2Hle::ResourceSamplerInfo fetch_sampler{.min_filter = 0, .mag_filter = 0,
                                                          .wrap = {0, 0, 0}};
    const auto sampler_table = submission.sampler_descriptors.find(sampler_pool);
    const auto* desc_ptr = sampler_id == 0 ? &fetch_sampler : nullptr;
    if (sampler_table != submission.sampler_descriptors.end()) {
        const auto found = sampler_table->second.find(sampler_id);
        if (found != sampler_table->second.end()) {
            desc_ptr = &found->second;
        }
    }
    if (texture == textures->second.end() || !desc_ptr) {
        NoteSkip("missing texture or sampler descriptor");
        return std::nullopt;
    }
    const auto& tex = texture->second.texture;
    const auto& desc = *desc_ptr;
    const auto& d = tex.description;
    u32 format = d.format;
    u32 first_level{};
    u32 levels = d.levels;
    u32 first_layer{};
    u32 layers = tex.layout.layers;
    u32 target = d.target;
    u32 depth_mode = d.depth_mode;
    auto swizzle = d.swizzle;
    const auto invalid_view = [&] {
        if (trace_images && draw_diagnostics.emplace(bound_program->serial,
                                                      "sampled texture view detail").second) {
            LOG_WARNING(Render_Vulkan,
                        "NVN 2 program {} stage {} sampled bank {} offset {:#x} handle {:#x}, type {}, target {} from {}, format {:#x} from {:#x}, mip {}+{}/{}, layer {}+{}/{}, view {:02x}",
                        bound_program->serial, stage, binding.cbuf_index, offset, handle,
                        static_cast<u32>(binding.type), target,
                        d.target, format, d.format, first_level, levels, d.levels,
                        first_layer, layers, tex.layout.layers,
                        texture->second.view ? (*texture->second.view)[7] : 0);
        }
        NoteSkip("unsupported sampled texture view");
    };
    if (const auto& v = texture->second.view) {
        const u8 flags = (*v)[7];
        format = ViewFormat(*v, d.format);
        if (!HasCompatibleFormat(format, d.format)) {
            invalid_view();
            return std::nullopt;
        }
        if (flags & 1) {
            first_level = (*v)[4];
            levels = (*v)[5];
            if (first_level < d.levels) {
                const u32 remaining = d.levels - first_level;
                levels = levels ? std::min(levels, remaining) : remaining;
            }
        }
        if (flags & 2) {
            first_layer = u32{(*v)[0]} | (u32{(*v)[1]} << 8);
            layers = u32{(*v)[2]} | (u32{(*v)[3]} << 8);
            if (!layers && first_layer < tex.layout.layers) {
                layers = tex.layout.layers - first_layer;
            }
        }
        if (flags & 0x20) {
            target = (*v)[6];
        }
        if (flags & 8) {
            std::ranges::copy(std::span{*v}.subspan(12, 4), swizzle.begin());
        }
        if (flags & 0x10) {
            depth_mode = flags >> 6;
        }
    }
    constexpr std::array components{VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ONE,
                                    VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
                                    VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
    VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_MAX_ENUM;
    if (binding.type == Shader::TextureType::Color2D &&
        (target == 1 || target == 4 || target == 8 || target == 9)) {
        // A non-array 2D instruction addresses the first layer of an array or cube view.
        layers = 1;
        view_type = VK_IMAGE_VIEW_TYPE_2D;
    } else if (binding.type == Shader::TextureType::ColorArray2D &&
               (target == 1 || target == 4 || target == 8 || target == 9)) {
        view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    } else if (binding.type == Shader::TextureType::Color3D && target == 2) {
        view_type = VK_IMAGE_VIEW_TYPE_3D;
    } else if (binding.type == Shader::TextureType::ColorCube && (target == 8 || target == 9) &&
               layers == 6) {
        view_type = VK_IMAGE_VIEW_TYPE_CUBE;
    } else if (binding.type == Shader::TextureType::ColorArrayCube && target == 9 && layers % 6 == 0) {
        view_type = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    }
    if (!binding.is_depth && !binding.is_integer && view_type == VK_IMAGE_VIEW_TYPE_MAX_ENUM) {
        // Incompatible targets have no texels addressable by the shader's view type.
        if (binding.type == Shader::TextureType::Color3D && target != 2) {
            return BindUnboundTexture(VK_IMAGE_VIEW_TYPE_3D);
        }
        if (target == 1 && binding.type == Shader::TextureType::ColorCube) {
            return BindUnboundTexture(VK_IMAGE_VIEW_TYPE_CUBE);
        }
        if (target == 1 && binding.type == Shader::TextureType::ColorArrayCube &&
            device.GetPhysical().GetFeatures().imageCubeArray) {
            return BindUnboundTexture(VK_IMAGE_VIEW_TYPE_CUBE_ARRAY);
        }
    }
    if (view_type == VK_IMAGE_VIEW_TYPE_MAX_ENUM ||
        first_layer >= tex.layout.layers || !layers || layers > tex.layout.layers - first_layer ||
        first_level >= d.levels || !levels || levels > d.levels - first_level ||
        std::ranges::any_of(swizzle, [&](u32 s) { return s >= components.size(); })) {
        invalid_view();
        return std::nullopt;
    }
    if ((binding.is_depth && (desc.compare_mode != 1 || desc.compare_func < 1 ||
                              desc.compare_func > 8)) ||
        desc.reduction > VK_SAMPLER_REDUCTION_MODE_MAX) {
        if (trace_images && draw_diagnostics.emplace(bound_program->serial, "sampler state").second) {
            LOG_WARNING(Render_Vulkan,
                        "NVN 2 program {} sampler {} compare instruction={}, mode={}, func={}, reduction={}",
                        bound_program->serial, sampler_id, binding.is_depth, desc.compare_mode,
                        desc.compare_func, desc.reduction);
        }
        NoteSkip("unsupported sampler comparison or reduction");
        return std::nullopt;
    }
    auto* image = FindImage(tex, true);
    const VkImageAspectFlags aspect = !image ? 0 : image->aspects == VK_IMAGE_ASPECT_COLOR_BIT
        ? VK_IMAGE_ASPECT_COLOR_BIT
        : depth_mode ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    if (!image || !(image->aspects & aspect) || (binding.is_depth && aspect == VK_IMAGE_ASPECT_STENCIL_BIT)) {
        if (trace_images && draw_diagnostics.emplace(bound_program->serial,
                                                      "sampled texture format detail").second) {
            LOG_WARNING(Render_Vulkan, "NVN 2 program {} texture format {:#x}, target {}, depth mode {}, compare {}, image {}",
                        bound_program->serial, format, target, depth_mode, binding.is_depth,
                        image != nullptr);
        }
        NoteSkip("unsupported sampled texture format");
        return std::nullopt;
    }
    const bool color = aspect == VK_IMAGE_ASPECT_COLOR_BIT;
    if (!(color ? PrepareSampledImage(*image, tex, submission) : image->initialized)) {
        TraceImage("missing sampled snapshot", tex, bound_program->serial);
        if (trace_images && draw_diagnostics.emplace(bound_program->serial,
                                                      "sample snapshot detail").second) {
            LOG_INFO(Render_Vulkan,
                     "NVN 2 sample program {}, bank {}, offset {:#x}, handle {:#x}, format {:#x}, flags {:#x}, address {:#x}, {}x{}, {} bytes, {} levels",
                     bound_program->serial, binding.cbuf_index, offset, handle, d.format,
                     d.flags, tex.storage.gpu_address + tex.offset, d.width, d.height,
                     tex.layout.size, d.levels);
        }
        NoteSkip("sampled texture has no replayed write or CPU snapshot");
        return std::nullopt;
    }
    TraceImage("sampled", tex, bound_program->serial);
    const auto address = tex.storage.gpu_address + tex.offset;
    if (frame_dump.CapturingSampled() && (d.width >= 900 || aspect == VK_IMAGE_ASPECT_DEPTH_BIT) &&
        !image->volume &&
        (color || aspect == VK_IMAGE_ASPECT_DEPTH_BIT) &&
        captured_inputs.emplace(address, first_level, first_layer).second) {
        frame_dump.CaptureNvn2Input(*image->image, image->surface_format,
                                    std::max(1U, d.width >> first_level),
                                    std::max(1U, d.height >> first_level), address,
                                    color ? VK_FORMAT_UNDEFINED : image->format, first_level,
                                    first_layer, bound_program->serial);
    }
    auto vk_format = FindColorFormat(format)->vk;
    auto sampled_aspect = aspect;
    VkImage sampled_image = *image->image;
    bool comparison_copy{};
    if (binding.is_depth && color &&
        !(device.GetPhysical().GetFormatProperties3(vk_format).optimalTilingFeatures &
          VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT)) {
        const VkFormat depth_format = vk_format == VK_FORMAT_R32_SFLOAT ? VK_FORMAT_D32_SFLOAT
                                      : vk_format == VK_FORMAT_R16_UNORM ? VK_FORMAT_D16_UNORM
                                                                        : VK_FORMAT_UNDEFINED;
        if (depth_format == VK_FORMAT_UNDEFINED || image->volume || vk_format != image->format ||
            !(sampled_image = PrepareDepthComparisonImage(*image, depth_format))) {
            NoteSkip("sampled color texture has no depth comparison view");
            return std::nullopt;
        }
        comparison_copy = true;
        vk_format = depth_format;
        sampled_aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    const auto key = std::tuple{vk_format, first_level, levels, first_layer, layers, view_type,
                                sampled_aspect, swizzle};
    const auto overlaps = [&](size_t slot) {
        const auto& attachment = color_views[slot];
        return destinations[slot] == image && attachment.level >= first_level &&
               attachment.level < first_level + levels &&
               (image->volume || (attachment.layer < first_layer + layers &&
                                  first_layer < attachment.layer + attachment.layers));
    };
    bool feedback{};
    for (size_t i = 0; i < destinations.size(); ++i) {
        feedback |= overlaps(i);
    }
    auto* views = &image->sampled_views;
    if (feedback && !comparison_copy) {
        const auto [found, inserted] = feedback_images.try_emplace(image);
        auto& copy = found->second;
        if (inserted) {
            copy.image = allocator.CreateImage(VkImageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                         (d.target == 8 || d.target == 9 ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0U),
                .imageType = image->volume ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
                .format = image->format,
                .extent = {image->extent.width, image->extent.height, image->depth},
                .mipLevels = image->levels,
                .arrayLayers = image->layers,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            });
        }
        sampled_image = *copy.image;
        views = &copy.views;
        Barrier();
        scheduler.Record([source = *image->image, sampled_image, inserted, first_level,
                          levels, first_layer, layers, image_levels = image->levels,
                          image_layers = image->layers, extent = image->extent,
                          depth = image->depth](vk::CommandBuffer cmdbuf) {
            if (inserted) {
                const VkImageMemoryBarrier transition{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = sampled_image,
                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, image_levels, 0, image_layers},
                };
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, transition);
            }
            for (u32 level = first_level; level < first_level + levels; ++level) {
                const VkImageSubresourceLayers subresource{VK_IMAGE_ASPECT_COLOR_BIT, level,
                                                           first_layer, layers};
                const VkImageCopy region{
                    .srcSubresource = subresource,
                    .dstSubresource = subresource,
                    .extent = {std::max(1U, extent.width >> level),
                               std::max(1U, extent.height >> level), std::max(1U, depth >> level)},
                };
                cmdbuf.CopyImage(source, VK_IMAGE_LAYOUT_GENERAL, sampled_image,
                                 VK_IMAGE_LAYOUT_GENERAL, {region});
            }
        });
        ++copies;
    }
    auto sampled_view = views->find(key);
    if (sampled_view == views->end()) {
        const VkImageViewUsageCreateInfo usage{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
        };
        auto view = device.GetLogical().CreateImageView({
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = &usage,
            .image = sampled_image,
            .viewType = view_type,
            .format = vk_format,
            .components = {components[swizzle[0]], components[swizzle[1]],
                           components[swizzle[2]], components[swizzle[3]]},
            .subresourceRange = {sampled_aspect, first_level, levels, first_layer, layers},
        });
        sampled_view = views->emplace(key, std::move(view)).first;
    }
    constexpr std::array modes{
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_REPEAT,
        VK_SAMPLER_ADDRESS_MODE_MAX_ENUM, VK_SAMPLER_ADDRESS_MODE_MAX_ENUM,
        VK_SAMPLER_ADDRESS_MODE_MAX_ENUM, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    };
    if (!std::ranges::all_of(desc.wrap, [&](u32 wrap) {
            return wrap < modes.size() && modes[wrap] != VK_SAMPLER_ADDRESS_MODE_MAX_ENUM;
        })) {
        NoteSkip("unsupported sampler wrap mode");
        return std::nullopt;
    }
    const VkSamplerReductionModeCreateInfo reduction{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO,
        .reductionMode = static_cast<VkSamplerReductionMode>(desc.reduction),
    };
    const auto physical = device.GetPhysical();
    const bool point_depth = aspect == VK_IMAGE_ASPECT_DEPTH_BIT && !binding.is_depth;
    const bool use_reduction = desc.reduction && !binding.is_depth &&
        device.IsExtSamplerFilterMinmaxSupported() &&
        (physical.GetFormatProperties(vk_format).optimalTilingFeatures &
         VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT);
    const bool anisotropic = desc.anisotropy > 1.0f && desc.mag_filter == 1 &&
                             (desc.min_filter & 1) && !point_depth && !binding.is_depth &&
                             !use_reduction && physical.GetFeatures().samplerAnisotropy;
    const auto& limits = physical.GetProperties().limits;
    VkBorderColor border = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    if (std::ranges::find(desc.wrap, 5U) != desc.wrap.end() && desc.border_kind == 0) {
        const auto rgba = std::bit_cast<std::array<float, 4>>(desc.border);
        if (rgba == std::array{1.0f, 1.0f, 1.0f, 1.0f}) {
            border = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        } else if (rgba == std::array{0.0f, 0.0f, 0.0f, 1.0f}) {
            border = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        }
    }
    auto host_sampler = device.GetLogical().CreateSampler({
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = use_reduction ? &reduction : nullptr,
        .magFilter = !point_depth && desc.mag_filter == 1 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .minFilter = !point_depth && (desc.min_filter & 1) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .mipmapMode = !point_depth && desc.min_filter >= 4 ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                                          : VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = modes[desc.wrap[0]],
        .addressModeV = modes[desc.wrap[1]],
        .addressModeW = modes[desc.wrap[2]],
        .mipLodBias = std::clamp(desc.lod_bias, -limits.maxSamplerLodBias, limits.maxSamplerLodBias),
        .anisotropyEnable = anisotropic,
        .maxAnisotropy = anisotropic ? std::min(desc.anisotropy, limits.maxSamplerAnisotropy) : 1.0f,
        .compareEnable = binding.is_depth,
        .compareOp = binding.is_depth ? static_cast<VkCompareOp>(desc.compare_func - 1)
                                      : VK_COMPARE_OP_ALWAYS,
        .minLod = desc.min_filter < 2 ? 0.0f : std::clamp(desc.min_lod, 0.0f, float(levels - 1)),
        .maxLod = desc.min_filter < 2 ? 0.25f : std::clamp(desc.max_lod, 0.0f, float(levels - 1)),
        .borderColor = border,
    });
    const VkSampler handle_sampler = *host_sampler;
    samplers.push_back(std::move(host_sampler));
    return VkDescriptorImageInfo{handle_sampler, *sampled_view->second, VK_IMAGE_LAYOUT_GENERAL};
}

bool BananzaNvn2Replay::PrepareSampledImage(Image& image, const Texture& texture,
                                           const Submission& submission) {
    if (!image.gpu_written && image.checked_submission != submission.sequence) {
        const auto data = submission.texture_snapshots.find(texture.serial);
        if (data != submission.texture_snapshots.end()) {
            const auto& bytes = data->second;
            const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(bytes.data()),
                                                bytes.size());
            if (!image.initialized || image.cpu_serial != texture.serial || image.cpu_hash != hash) {
                if (!UploadTexture(image, texture, bytes)) {
                    return false;
                }
                image.cpu_hash = hash;
                image.cpu_serial = texture.serial;
                ++texture_uploads;
            }
        }
        image.checked_submission = submission.sequence;
    }
    return image.initialized && (image.gpu_written || image.cpu_serial == texture.serial);
}

std::optional<BananzaNvn2Replay::DrawState> BananzaNvn2Replay::DecodeDraw(
    const Operation& op) const {
    const auto& x = op.args;
    DrawState state;
    if (op.replay == ReplayOp::Draw || op.replay == ReplayOp::DrawInstanced) {
        const bool instanced = op.replay == ReplayOp::DrawInstanced;
        if (x.size() != (instanced ? 5 : 3)) {
            return std::nullopt;
        }
        state.first = static_cast<u32>(x[1]);
        state.count = static_cast<u32>(x[2]);
        if (instanced) {
            state.base_instance = static_cast<u32>(x[3]);
            state.instances = static_cast<u32>(x[4]);
        }
    } else {
        const bool instanced = op.replay == ReplayOp::DrawIndexedInstanced;
        const bool base = op.replay == ReplayOp::DrawIndexedBaseVertex;
        if (x.size() != (instanced ? 7 : base ? 5 : 4) || x[1] > 2) {
            return std::nullopt;
        }
        constexpr std::array types{VK_INDEX_TYPE_UINT8_EXT, VK_INDEX_TYPE_UINT16,
                                    VK_INDEX_TYPE_UINT32};
        if (x[1] == 0 && !device.IsExtIndexTypeUint8Supported() && x[0] != 7 && x[0] != 8) {
            return std::nullopt;
        }
        state.index_type = types[x[1]];
        state.count = static_cast<u32>(x[2]);
        state.index_range = {x[3], u64{state.count} << x[1]};
        if (base || instanced) {
            state.base_vertex = static_cast<s32>(x[4]);
        }
        if (instanced) {
            state.base_instance = static_cast<u32>(x[5]);
            state.instances = static_cast<u32>(x[6]);
        }
    }
    if (state.count > (1U << 20) || state.instances > (1U << 20)) {
        return std::nullopt;
    }
    return state;
}

std::optional<BananzaNvn2Replay::Pipeline> BananzaNvn2Replay::CreatePipeline(
    const ColorFormats& formats, u32 color_count, VkPrimitiveTopology topology,
    const RasterState& raster) {
    const auto& shaders = graphics_program->shaders;
    const auto vertex = std::ranges::find(shaders, 0U,
                                          &Core::BananzaNvn2Hle::ResourceShaderInfo::stage);
    const auto find_shader = [](const auto& program, u32 stage)
        -> const Core::BananzaNvn2Hle::ResourceShaderInfo* {
        if (!program) {
            return nullptr;
        }
        const auto& shaders = program->shaders;
        const auto shader = std::ranges::find(shaders, stage,
                                              &Core::BananzaNvn2Hle::ResourceShaderInfo::stage);
        return shader == shaders.end() ? nullptr : &*shader;
    };
    const auto* fragment = find_shader(fragment_program, 4);
    const auto* geometry = find_shader(geometry_program, 3);
    if (vertex == shaders.end() || (fragment_program && !fragment) ||
        (geometry_program && (!geometry || !device.GetPhysical().GetFeatures().geometryShader)) ||
        !device.IsDynamicRenderingSupported()) {
        return std::nullopt;
    }
    Shader::RuntimeInfo runtime;
    runtime.generic_input_types.fill(Shader::AttributeType::Float);
    const auto attribute_type = [](u32 format) {
        const auto color = FindColorFormat(format);
        if (color && VideoCore::Surface::IsPixelFormatInteger(color->surface)) {
            return VideoCore::Surface::IsPixelFormatSignedInteger(color->surface)
                       ? Shader::AttributeType::SignedInt : Shader::AttributeType::UnsignedInt;
        }
        return Shader::AttributeType::Float;
    };
    for (u32 i = 0; i < vertex_attributes.size(); ++i) {
        if (!(vertex_attributes[i] & 0x40)) {
            runtime.generic_input_types[i] = Shader::AttributeType::Disabled;
            continue;
        }
        const u32 format = (vertex_attributes[i] >> 21) & 0x3ff;
        switch (format) {
        case 0x1b:
        case 0x20:
        case 0x23:
            runtime.generic_input_types[i] = Shader::AttributeType::UnsignedInt;
            break;
        case 0x1c:
        case 0x21:
        case 0x24:
        case 0x6a:
            runtime.generic_input_types[i] = Shader::AttributeType::SignedInt;
            break;
        default:
            runtime.generic_input_types[i] = attribute_type(format);
            break;
        }
    }
    for (u32 i = 0; i < formats.size(); ++i) {
        if (colors[i]) {
            const auto format = FindColorFormat(colors[i]->description.format);
            if (format && VideoCore::Surface::IsPixelFormatInteger(format->surface)) {
                runtime.color_output_types[i] = attribute_type(colors[i]->description.format);
            }
        }
    }
    runtime.previous_stage_stores.mask.set();
    runtime.convert_depth_mode = raster.negative_one_to_one;
    runtime.y_negate = raster.origin_lower_left;
    const auto alpha_func = (color_state.control >> 24) & 15;
    if (alpha_func >= 1 && alpha_func < 8) {
        runtime.alpha_test_func = static_cast<Shader::CompareFunction>(alpha_func - 1);
        runtime.alpha_test_reference = std::bit_cast<float>(color_state.alpha_ref);
    }
    Shader::Backend::Bindings bindings;
    const auto& profile = pipeline_cache.ShaderProfile();
    const auto& host_info = pipeline_cache.ShaderHostInfo();
    const auto vs = TranslateBananzaNvn2Shader(*vertex, profile, host_info, runtime, bindings);
    if (vs.status != BananzaNvn2ShaderStatus::Success) {
        LOG_WARNING(Render_Vulkan, "Bananza NVN 2 vertex shader at {:#x}: {}",
                    vertex->data_address, vs.error);
        return std::nullopt;
    }
    const u32 geometry_binding = bindings.unified;
    runtime.previous_stage_stores = vs.info->stores;
    runtime.generic_input_types.fill(Shader::AttributeType::Float);
    BananzaNvn2ShaderResult gs;
    if (geometry) {
        if (topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST) {
            runtime.input_topology = Shader::InputTopology::Points;
        } else if (topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ||
                   topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP) {
            runtime.input_topology = Shader::InputTopology::Lines;
        } else {
            runtime.input_topology = Shader::InputTopology::Triangles;
        }
        std::array<u32, 8> mask;
        mask.fill(~0U);
        const auto enable = [&](Shader::IR::Attribute attribute) {
            const u32 index = static_cast<u32>(attribute);
            mask[index / 32] &= ~(1U << (index % 32));
        };
        for (u32 i = 0; i < 4; ++i) {
            enable(static_cast<Shader::IR::Attribute>(static_cast<u32>(Shader::IR::Attribute::PositionX) + i));
        }
        for (u32 i = 0; i < 32; ++i) {
            if (vs.info->stores.Generic(i)) {
                for (u32 component = 0; component < 4; ++component) {
                    enable(static_cast<Shader::IR::Attribute>(
                        static_cast<u32>(Shader::IR::Attribute::Generic0X) + 4 * i + component));
                }
            }
        }
        gs = TranslateBananzaNvn2Shader(*geometry, profile, host_info, runtime, bindings, nullptr, &mask);
        if (gs.status != BananzaNvn2ShaderStatus::Success) {
            LOG_WARNING(Render_Vulkan, "Bananza NVN 2 geometry shader at {:#x}: {}",
                        geometry->data_address, gs.error);
            return std::nullopt;
        }
        runtime.previous_stage_stores = gs.info->stores;
        runtime.previous_stage_stores.mask |= gs.info->passthrough.mask;
    }
    const u32 fragment_binding = bindings.unified;
    BananzaNvn2ShaderResult fs;
    if (fragment) {
        fs = TranslateBananzaNvn2Shader(*fragment, profile, host_info, runtime, bindings);
        if (fs.status != BananzaNvn2ShaderStatus::Success) {
            LOG_WARNING(Render_Vulkan, "Bananza NVN 2 fragment shader at {:#x}: {}",
                        fragment->data_address, fs.error);
            return std::nullopt;
        }
    }
    if (const char* dir = std::getenv("EDEN_NX2_NVN_SPIRV_DUMP")) {
        const auto serial = graphics_program->serial;
        const auto save = [&](std::string_view stage, const BananzaNvn2ShaderResult& shader,
                              const Core::BananzaNvn2Hle::ResourceShaderInfo& native) {
            const auto path = std::filesystem::path{dir} /
                              fmt::format("{}-{}-{:016x}.spv", serial, stage, shader.native_hash);
            if (!std::filesystem::exists(path)) {
                std::ofstream file{path, std::ios::binary};
                file.write(reinterpret_cast<const char*>(shader.spirv.data()),
                           shader.spirv.size() * sizeof(u32));
                std::ofstream code{path.string() + ".native", std::ios::binary};
                code.write(reinterpret_cast<const char*>(native.code.data()), native.code_size);
            }
        };
        save("vs", vs, *vertex);
        if (geometry) {
            save("gs", gs, *geometry);
        }
        if (fragment) {
            save("fs", fs, *fragment);
        }
    }
    const auto can_bind = [&](const BananzaNvn2ShaderResult& shader,
                             const Core::BananzaNvn2Hle::ResourceShaderInfo& staged, u32 stage) {
        const auto& info = *shader.info;
        const auto& queries = shader.queries;
        const u32 storage_base = GraphicsStorageBase[stage];
        const auto direct_storage = [&](u32 bank, u32 offset) {
            return bank == 0 && offset >= storage_base && offset < storage_base + 0x100 &&
                   (offset - storage_base) % 16 == 0;
        };
        for (const auto& storage : info.storage_buffers_descriptors) {
            const auto features = device.GetPhysical().GetFeatures();
            const bool supports_write = stage == 1 ? features.fragmentStoresAndAtomics
                                                    : features.vertexPipelineStoresAndAtomics;
            if ((storage.is_written && !supports_write) ||
                storage.count != 1 || storage.indirect_offset != ~0U ||
                !direct_storage(storage.cbuf_index, storage.cbuf_offset)) {
                if (draw_diagnostics.emplace(graphics_program->serial, "graphics storage detail").second) {
                    LOG_WARNING(Render_Vulkan,
                                "NVN 2 graphics storage: stage {}, bank {}, offset {:#x}, count {}, write {}, indirect {:#x}",
                                staged.stage, storage.cbuf_index, storage.cbuf_offset, storage.count,
                                storage.is_written, storage.indirect_offset);
                }
                NoteSkip("unsupported graphics storage write or pointer");
                return false;
            }
        }
        for (const auto& [key, mapped] : queries.storage_pointer_types) {
            if (!mapped || !direct_storage(static_cast<u32>(key >> 32), static_cast<u32>(key))) {
                NoteSkip("unsupported graphics storage pointer classification");
                return false;
            }
        }
        return info.image_descriptors.empty() && info.texture_buffer_descriptors.empty() &&
               info.image_buffer_descriptors.empty() && info.handle_programs.empty() &&
               !info.uses_global_memory && !info.stores_global_memory &&
               queries.cbuf_replacements.empty() &&
               std::ranges::all_of(info.texture_descriptors, [](const auto& descriptor) {
                   return (descriptor.type == Shader::TextureType::Color2D ||
                           descriptor.type == Shader::TextureType::ColorArray2D ||
                           descriptor.type == Shader::TextureType::Color3D ||
                           descriptor.type == Shader::TextureType::ColorCube ||
                           descriptor.type == Shader::TextureType::ColorArrayCube) &&
                          !descriptor.is_integer && !descriptor.is_multisample &&
                          !descriptor.has_secondary && descriptor.count == 1 &&
                          descriptor.handle_program == ~0U;
               }) &&
               !queries.has_hle_engine_state && !queries.has_unbound_instructions &&
               std::ranges::all_of(queries.cbuf_values, [&](const auto& entry) {
                   const u32 offset = static_cast<u32>(entry.first);
                   if ((entry.first >> 32) != 1 || offset > staged.constant_size ||
                       staged.constant_size - offset < sizeof(u32)) {
                       return false;
                   }
                   u32_le word;
                   std::memcpy(&word, staged.code.data() + staged.constant_offset + offset,
                               sizeof(word));
                   return word == entry.second;
               });
    };
    if (!can_bind(vs, *vertex, 0) || (fragment && !can_bind(fs, *fragment, 1)) ||
        (geometry && !can_bind(gs, *geometry, 2))) {
        NoteSkip("translation needs unsupported descriptors or constant values");
        return std::nullopt;
    }
    std::vector<VkVertexInputAttributeDescription> attributes;
    std::vector<VkVertexInputBindingDescription> streams;
    Pipeline result;
    result.uses_render_area = vs.info->uses_render_area || (fragment && fs.info->uses_render_area) ||
                              (geometry && gs.info->uses_render_area);
    result.stages = VK_SHADER_STAGE_VERTEX_BIT |
                    (geometry ? VK_SHADER_STAGE_GEOMETRY_BIT : 0U) |
                    (fragment ? VK_SHADER_STAGE_FRAGMENT_BIT : 0U);
    for (u32 i = 0; i < vertex_attributes.size(); ++i) {
        if (!vs.info->loads.Generic(i) || !(vertex_attributes[i] & 0x40)) {
            continue;
        }
        const u32 attribute = vertex_attributes[i];
        const u32 stream = attribute & 31;
        const auto vf = VertexFormat((attribute >> 21) & 0x3ff);
        if (!vertex_strides[stream] || vertex_divisors[stream] > 1 ||
            vf == VK_FORMAT_UNDEFINED ||
            !device.IsFormatSupported(vf, VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT, FormatType::Buffer)) {
            if (trace_images && draw_diagnostics.emplace(graphics_program->serial,
                                                          "vertex attribute detail").second) {
                LOG_WARNING(Render_Vulkan, "NVN 2 program {} attribute {}: word {:#x}, format {:#x}, stream {}, stride {}, divisor {}",
                            graphics_program->serial, i, attribute, (attribute >> 21) & 0x3ff,
                            stream, vertex_strides[stream], vertex_divisors[stream]);
            }
            NoteSkip("unsupported vertex attribute format or stream");
            return std::nullopt;
        }
        attributes.push_back({i, stream, vf, (attribute >> 7) & 0x3fff});
        if (std::ranges::find(result.streams, stream) == result.streams.end()) {
            result.streams.push_back(stream);
            streams.push_back({stream, vertex_strides[stream],
                               vertex_divisors[stream] ? VK_VERTEX_INPUT_RATE_INSTANCE
                                                       : VK_VERTEX_INPUT_RATE_VERTEX});
        }
    }
    std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
    const auto describe = [&](const Shader::Info& info,
                               const Core::BananzaNvn2Hle::ResourceShaderInfo& staged,
                               u32 stage, u32 binding, VkShaderStageFlags flags) {
        for (const auto& uniform : info.constant_buffer_descriptors) {
            if (uniform.count != 1 || uniform.index >= 32) {
                return false;
            }
            UniformBinding item{stage, uniform.index, binding, info.constant_buffer_used_sizes[uniform.index]};
            if (uniform.index == 1) {
                if (!staged.constant_size) {
                    return false;
                }
                const auto literal =
                    std::span{staged.code}.subspan(staged.constant_offset, staged.constant_size);
                item.literal.assign(literal.begin(), literal.end());
            }
            layout_bindings.push_back({binding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, flags});
            result.uniforms.push_back(std::move(item));
            ++binding;
        }
        for (const auto& storage : info.storage_buffers_descriptors) {
            const u32 base = GraphicsStorageBase[stage];
            layout_bindings.push_back({binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flags});
            result.storage.push_back({stage, binding++, (storage.cbuf_offset - base) / 16,
                                      storage.is_written});
        }
        for (const auto& descriptor : info.texture_descriptors) {
            layout_bindings.push_back({binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, flags});
            result.textures.emplace_back(stage, binding++, descriptor);
        }
        return true;
    };
    if (!describe(*vs.info, *vertex, 0, 0, VK_SHADER_STAGE_VERTEX_BIT) ||
        (geometry && !describe(*gs.info, *geometry, 2, geometry_binding, VK_SHADER_STAGE_GEOMETRY_BIT)) ||
        (fragment && !describe(*fs.info, *fragment, 1, fragment_binding, VK_SHADER_STAGE_FRAGMENT_BIT)) ||
        layout_bindings.size() != bindings.unified) {
        NoteSkip("shader descriptor layout could not be matched");
        return std::nullopt;
    }
    const auto& logical = device.GetLogical();
    if (!layout_bindings.empty()) {
        result.set_layout = logical.CreateDescriptorSetLayout({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = static_cast<u32>(layout_bindings.size()),
            .pBindings = layout_bindings.data(),
        });
    }
    const VkPushConstantRange push{
        .stageFlags = result.stages,
        .size = sizeof(Shader::Backend::SPIRV::RescalingLayout) +
                sizeof(Shader::Backend::SPIRV::RenderAreaLayout),
    };
    const VkDescriptorSetLayout set_layout = result.set_layout ? *result.set_layout : VK_NULL_HANDLE;
    result.layout = logical.CreatePipelineLayout({
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = set_layout ? 1U : 0U,
        .pSetLayouts = set_layout ? &set_layout : nullptr,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push,
    });
    const auto make_shader = [&](std::span<const u32> spirv) {
        return logical.CreateShaderModule({
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv.size_bytes(),
            .pCode = spirv.data(),
        });
    };
    const auto vertex_module = make_shader(vs.spirv);
    const auto fragment_module = fragment ? make_shader(fs.spirv) : vk::ShaderModule{};
    const auto geometry_module = geometry ? make_shader(gs.spirv) : vk::ShaderModule{};
    if (trace_images) {
        vertex_module.SetObjectNameEXT(fmt::format("NVN 2 {} VS {:016x}",
                                                  graphics_program->serial, vs.native_hash).c_str());
        if (fragment) {
            fragment_module.SetObjectNameEXT(fmt::format("NVN 2 {} FS {:016x}",
                                                        fragment_program->serial, fs.native_hash).c_str());
        }
        if (geometry) {
            geometry_module.SetObjectNameEXT(fmt::format("NVN 2 {} GS {:016x}",
                                                        geometry_program->serial, gs.native_hash).c_str());
        }
    }
    std::array<VkPipelineShaderStageCreateInfo, 3> stages{};
    u32 stage_count{};
    const auto append_stage = [&](VkShaderStageFlagBits stage, VkShaderModule module) {
        stages[stage_count++] = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                .stage = stage, .module = module, .pName = "main"};
    };
    append_stage(VK_SHADER_STAGE_VERTEX_BIT, *vertex_module);
    if (geometry) {
        append_stage(VK_SHADER_STAGE_GEOMETRY_BIT, *geometry_module);
    }
    if (fragment) {
        append_stage(VK_SHADER_STAGE_FRAGMENT_BIT, *fragment_module);
    }
    const VkPipelineVertexInputStateCreateInfo vertex_input{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = static_cast<u32>(streams.size()),
        .pVertexBindingDescriptions = streams.data(),
        .vertexAttributeDescriptionCount = static_cast<u32>(attributes.size()),
        .pVertexAttributeDescriptions = attributes.data(),
    };
    const VkPipelineInputAssemblyStateCreateInfo input{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = topology,
    };
    const VkPipelineViewportDepthClipControlCreateInfoEXT depth_clip{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
        .negativeOneToOne = VK_TRUE,
    };
    const VkPipelineViewportStateCreateInfo view{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = raster.negative_one_to_one && profile.support_native_ndc ? &depth_clip : nullptr,
        .viewportCount = 1,
        .scissorCount = 1,
    };
    const u32 polygon = (raster.polygon >> 3) & 3;
    if (polygon > 2 || (polygon < 2 && !device.GetPhysical().GetFeatures().fillModeNonSolid)) {
        NoteSkip("unsupported polygon mode");
        return std::nullopt;
    }
    const bool clockwise = ((raster.polygon & 4) == 0) != raster.origin_lower_left;
    const auto features = device.GetPhysical().GetFeatures();
    const VkPipelineRasterizationStateCreateInfo rasterization{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = raster.depth_clamp && features.depthClamp,
        .rasterizerDiscardEnable = raster.discard,
        .polygonMode = static_cast<VkPolygonMode>(2 - polygon),
        .cullMode = raster.polygon & 3,
        .frontFace = clockwise ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = raster.depth_bias,
        .lineWidth = 1.0f,
    };
    const VkPipelineMultisampleStateCreateInfo samples{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    const u64 ds = raster.depth_stencil;
    const auto compare = [](u32 nvn) {
        return nvn >= 1 && nvn <= 8 ? static_cast<VkCompareOp>(nvn - 1) : VK_COMPARE_OP_ALWAYS;
    };
    const auto stencil_op = [](u32 nvn) {
        return nvn >= 1 && nvn <= 8 ? static_cast<VkStencilOp>(nvn - 1) : VK_STENCIL_OP_KEEP;
    };
    const auto stencil = [&](u32 word) {
        return VkStencilOpState{.failOp = stencil_op((word >> 4) & 15),
                                .passOp = stencil_op((word >> 8) & 15),
                                .depthFailOp = stencil_op((word >> 12) & 15),
                                .compareOp = compare(word & 15)};
    };
    const bool has_depth = (raster.depth_aspects & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
    const bool has_stencil = (raster.depth_aspects & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
    const VkPipelineDepthStencilStateCreateInfo depth_stencil{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = has_depth && (ds & 1),
        .depthWriteEnable = has_depth && (ds & 2),
        .depthCompareOp = compare((ds >> 4) & 15),
        .stencilTestEnable = has_stencil && (ds & 4),
        .front = stencil(static_cast<u32>(ds >> 32)),
        .back = stencil(static_cast<u32>(ds >> 48)),
        .maxDepthBounds = 1.0f,
    };
    std::array<VkPipelineColorBlendAttachmentState, 8> attachments{};
    constexpr std::array blend_ops{VK_BLEND_OP_MAX_ENUM, VK_BLEND_OP_ADD, VK_BLEND_OP_SUBTRACT,
                                   VK_BLEND_OP_REVERSE_SUBTRACT, VK_BLEND_OP_MIN, VK_BLEND_OP_MAX};
    for (u32 slot = 0; slot < color_count; ++slot) {
        auto& attachment = attachments[slot];
        const u32 shader_mask = fs.info ? fs.info->frag_color_components[slot] : 0;
        attachment.colorWriteMask = ((color_state.channel_mask >> (slot * 4)) & 15) & shader_mask;
        attachment.blendEnable = (color_state.control >> (slot + 8)) & 1;
        if (!attachment.blendEnable) {
            continue;
        }
        constexpr u64 DefaultBlend = 0x01020102110c0000ULL;
        const u64 packed = color_state.blend[slot] ? color_state.blend[slot] : DefaultBlend;
        const u32 rgb_op = (packed >> 24) & 7;
        const u32 alpha_op = (packed >> 28) & 7;
        attachment.srcColorBlendFactor = BlendFactor((packed >> 32) & 127);
        attachment.dstColorBlendFactor = BlendFactor((packed >> 40) & 127);
        attachment.srcAlphaBlendFactor = BlendFactor((packed >> 48) & 127);
        attachment.dstAlphaBlendFactor = BlendFactor((packed >> 56) & 127);
        if (((packed >> 8) & 63) || !rgb_op || rgb_op >= blend_ops.size() || !alpha_op ||
            alpha_op >= blend_ops.size() || attachment.srcColorBlendFactor == VK_BLEND_FACTOR_MAX_ENUM ||
            attachment.dstColorBlendFactor == VK_BLEND_FACTOR_MAX_ENUM ||
            attachment.srcAlphaBlendFactor == VK_BLEND_FACTOR_MAX_ENUM ||
            attachment.dstAlphaBlendFactor == VK_BLEND_FACTOR_MAX_ENUM ||
            (formats[slot] != VK_FORMAT_UNDEFINED &&
             !device.IsFormatSupported(formats[slot], VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT,
                                       FormatType::Optimal))) {
            NoteSkip("unsupported blend state");
            return std::nullopt;
        }
        attachment.colorBlendOp = blend_ops[rgb_op];
        attachment.alphaBlendOp = blend_ops[alpha_op];
    }
    const u32 logic_op = (color_state.control >> 16) & 255;
    if (logic_op > VK_LOGIC_OP_SET) {
        NoteSkip("unsupported color logic operation");
        return std::nullopt;
    }
    const VkPipelineColorBlendStateCreateInfo blend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = logic_op != VK_LOGIC_OP_COPY,
        .logicOp = static_cast<VkLogicOp>(logic_op),
        .attachmentCount = color_count,
        .pAttachments = attachments.data(),
    };
    constexpr std::array dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                        VK_DYNAMIC_STATE_BLEND_CONSTANTS,
                                        VK_DYNAMIC_STATE_DEPTH_BIAS,
                                        VK_DYNAMIC_STATE_STENCIL_REFERENCE,
                                        VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
                                        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK};
    const VkPipelineDynamicStateCreateInfo dynamic{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };
    const VkPipelineRenderingCreateInfo rendering{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = color_count,
        .pColorAttachmentFormats = formats.data(),
        .depthAttachmentFormat = has_depth ? raster.depth_format : VK_FORMAT_UNDEFINED,
        .stencilAttachmentFormat = has_stencil ? raster.depth_format : VK_FORMAT_UNDEFINED,
    };
    result.pipeline = logical.CreateGraphicsPipeline({
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &rendering,
        .stageCount = stage_count,
        .pStages = stages.data(),
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input,
        .pViewportState = &view,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &samples,
        .pDepthStencilState = &depth_stencil,
        .pColorBlendState = &blend,
        .pDynamicState = &dynamic,
        .layout = *result.layout,
        .basePipelineIndex = -1,
    });
    return result;
}

void BananzaNvn2Replay::NoteSkip(std::string_view reason) {
    ++draw_skips[reason];
    const u64 serial = graphics_program ? graphics_program->serial : 0;
    const bool full_size = trace_images && colors[0] && colors[0]->description.width >= 1900;
    if ((draw_diagnostics.size() < 40 || (full_size && draw_diagnostics.size() < 150)) &&
        draw_diagnostics.emplace(serial, reason).second) {
        LOG_WARNING(Render_Vulkan, "Bananza NVN 2 draw skipped (program {}): {}", serial, reason);
    }
}

void BananzaNvn2Replay::TraceImage(std::string_view operation, const Texture& texture, u64 detail) {
    const auto& d = texture.description;
    const auto address = texture.storage.gpu_address + texture.offset;
    const bool small_depth = d.format == 0x32 || d.format == 0x34;
    if (!trace_images || (!small_depth && d.width < 900) || traced_images.size() >= 600 ||
        !traced_images.emplace(operation, address, detail).second) {
        return;
    }
    LOG_INFO(Render_Vulkan, "NVN 2 image {} {:#x} {}x{} format {:#x}, detail {:#x}",
             operation, address, d.width, d.height, d.format, detail);
}

bool BananzaNvn2Replay::ResetIndirectDraws(const DrawState& draw, const Submission& submission) {
    namespace Reset = VideoCommon::NativeIndirectReset;
    const auto& [address, size] = storage_buffers[0][0];
    const auto list = Reset::GetListRange(address, static_cast<u32>(size), draw.first, draw.count);
    if (!list || !submission.guest_memory) {
        return false;
    }
    const auto snapshot = submission.buffer_snapshots.find({list->address, list->bytes});
    if (snapshot == submission.buffer_snapshots.end()) {
        return false;
    }
    std::vector<u64> records;
    records.reserve(draw.count);
    for (size_t offset = 0; offset < list->bytes; offset += sizeof(u64)) {
        u64_le record;
        std::memcpy(&record, snapshot->second.data() + offset, sizeof(record));
        if (!record) {
            continue;
        }
        if (record % alignof(u32) || !submission.indirect_reset_addresses.contains(record) ||
            (record < list->address + list->bytes && list->address < record + sizeof(Reset::Record))) {
            return false;
        }
        records.push_back(record);
    }
    static constexpr std::array<u32_le, 5> reset_words{0, 1, 0, 0, 0};
    const std::span<const u8> reset_bytes{reinterpret_cast<const u8*>(reset_words.data()),
                                         sizeof(reset_words)};
    const std::set<u64> changed(records.begin(), records.end());
    std::vector<std::pair<VkBuffer, VkBufferCopy>> copies_to_storage;
    for (const u64 record : records) {
        submission.guest_memory->WriteBlockUnsafe(submission.indirect_reset_addresses.at(record),
                                                   reset_bytes.data(), reset_bytes.size());
        buffer_updates.push_back({record, reset_bytes});
        if (const auto storage = FindStorageRange({record, reset_bytes.size()});
            storage != storage_uploads.end()) {
            copies_to_storage.emplace_back(*storage->second.buffer,
                                           VkBufferCopy{0, record - storage->first.first,
                                                        reset_bytes.size()});
        }
        reset_indirect_draws.insert(record);
    }
    std::erase_if(uploaded_ranges, [&](const auto& entry) {
        const auto [start, size] = entry.first;
        const auto next = changed.lower_bound(start);
        return (next != changed.end() && *next < start + size) ||
               (next != changed.begin() && *std::prev(next) + reset_bytes.size() > start);
    });
    if (!copies_to_storage.empty()) {
        const auto source = Upload(reset_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        Barrier();
        bool overlaps = changed.size() != records.size();
        if (!changed.empty()) {
            auto last = changed.begin();
            for (auto next = std::next(last); next != changed.end(); last = next++) {
                overlaps |= *next - *last < reset_bytes.size();
            }
        }
        scheduler.Record([source, destinations = std::move(copies_to_storage), overlaps](vk::CommandBuffer cmdbuf) {
            const VkMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            };
            for (const auto& [buffer, region] : destinations) {
                if (overlaps) {
                    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, barrier);
                }
                cmdbuf.CopyBuffer(source, buffer, region);
            }
        });
    }
    if (trace_images && draw_diagnostics.emplace(graphics_program->serial, "indirect draw reset").second) {
        LOG_INFO(Render_Vulkan, "NVN 2 program {} reset {} indirect draw records", graphics_program->serial,
                 draw.count);
    }
    return true;
}

void BananzaNvn2Replay::Draw(const Operation& op, const Submission& submission) {
    if (!render_enabled) {
        return;
    }
    if (graphics_program && colors[0]) {
        TraceImage("draw request", *colors[0], graphics_program->serial);
    }
    if (op.args.empty()) {
        return;
    }
    if (!graphics_program) {
        NoteSkip("no vertex program bound");
        return;
    }
    const auto topology = Topology(op.args[0]);
    std::optional<DrawState> draw;
    if (op.replay == ReplayOp::DrawIndirect && op.args.size() == 2) {
        draw = DrawState{.indirect_range = {op.args[1], sizeof(VkDrawIndirectCommand)}};
    } else if (op.replay == ReplayOp::DrawIndexedIndirect && op.args.size() == 4 &&
               op.args[1] <= 2) {
        if (op.args[1] == 0 && !device.IsExtIndexTypeUint8Supported()) {
            NoteSkip("indirect uint8 indices unsupported by host");
            return;
        }
        constexpr std::array types{VK_INDEX_TYPE_UINT8_EXT, VK_INDEX_TYPE_UINT16,
                                   VK_INDEX_TYPE_UINT32};
        const Range arguments{op.args[3], sizeof(VkDrawIndexedIndirectCommand)};
        const auto argument_storage = FindStorageRange(arguments);
        const bool gpu_written = argument_storage != storage_uploads.end() &&
                                 arguments.first - argument_storage->first.first <
                                     argument_storage->second.written_bytes;
        const auto slice = submission.indirect_index_slices.find({op.args[3], op.args[2], op.args[1]});
        const auto snapshot = submission.buffer_snapshots.find(arguments);
        if (!gpu_written && reset_indirect_draws.contains(arguments.first)) {
            return;
        }
        if (!gpu_written && snapshot != submission.buffer_snapshots.end()) {
            VkDrawIndexedIndirectCommand command{};
            std::memcpy(&command, snapshot->second.data(), sizeof(command));
            if (!command.indexCount || !command.instanceCount) {
                return;
            }
            if (slice != submission.indirect_index_slices.end()) {
                draw = DrawState{.count = command.indexCount, .instances = command.instanceCount,
                                 .base_instance = command.firstInstance,
                                 .base_vertex = command.vertexOffset,
                                 .index_type = types[op.args[1]], .index_range = slice->second};
            }
        }
        if (!draw) {
            const auto index_storage = FindStorageRange({op.args[2], u64{1} << op.args[1]});
            const auto saved_size = submission.indirect_index_sizes.find(op.args[2]);
            const u64 size = index_storage != storage_uploads.end()
                                 ? index_storage->first.second -
                                       (op.args[2] - index_storage->first.first)
                             : saved_size != submission.indirect_index_sizes.end() ? saved_size->second
                                                                                  : 0;
            if (!size) {
                if (trace_images && draw_diagnostics.emplace(graphics_program->serial,
                                                              "indirect index detail").second) {
                    LOG_INFO(Render_Vulkan,
                             "NVN 2 program {} indirect indices {:#x} type {}, args {:#x}, GPU written {}, args snapshot {}, index slice {}",
                             graphics_program->serial, op.args[2], op.args[1], arguments.first,
                             gpu_written, snapshot != submission.buffer_snapshots.end(),
                             slice != submission.indirect_index_slices.end());
                }
                NoteSkip("indirect index buffer range missing at submit");
                return;
            }
            draw = DrawState{.index_type = types[op.args[1]], .index_range = {op.args[2], size},
                             .indirect_range = arguments, .indexed_indirect = true};
        }
    } else {
        draw = DecodeDraw(op);
    }
    if (!draw || topology == VK_PRIMITIVE_TOPOLOGY_MAX_ENUM) {
        NoteSkip("unsupported draw arguments or topology");
        if (trace_images && draw_diagnostics.emplace(graphics_program->serial,
                                                      "draw topology detail").second) {
            LOG_INFO(Render_Vulkan,
                     "NVN 2 program {} draw op {}, topology {}, arguments {} [{}]",
                     graphics_program->serial, static_cast<u32>(op.replay), op.args[0],
                     op.args.size(), fmt::join(op.args, ", "));
        }
        return;
    }
    if (!draw->indirect_range.second && (!draw->count || !draw->instances)) {
        return;
    }
    if (graphics_program->resets_indirect_draws && op.replay == ReplayOp::Draw && op.args[0] == 0 &&
        ResetIndirectDraws(*draw, submission)) {
        return;
    }
    VkBuffer indirect_buffer{};
    u64 indirect_offset{};
    if (draw->indirect_range.second) {
        if (op.args[0] == 7 || op.args[0] == 8) {
            NoteSkip("indirect quad topology");
            return;
        }
        std::tie(indirect_buffer, indirect_offset) = UploadBufferRange(submission, draw->indirect_range);
        if (!indirect_buffer || indirect_offset % 4) {
            NoteSkip("draw indirect argument buffer unavailable");
            return;
        }
        if (draw->indexed_indirect &&
            FindStorageRange(draw->indirect_range) == storage_uploads.end()) {
            const auto args = submission.buffer_snapshots.find(draw->indirect_range);
            if (args == submission.buffer_snapshots.end()) {
                return;
            }
            VkDrawIndexedIndirectCommand command{};
            std::memcpy(&command, args->second.data(), sizeof(command));
            const u64 element_size = u64{1} << op.args[1];
            const u64 end = (u64{command.firstIndex} + command.indexCount) * element_size;
            if (end > draw->index_range.second) {
                NoteSkip("indirect index range exceeds snapshot");
                return;
            }
        }
    }
    ColorFormats formats{};
    std::array<Image*, 8> targets{};
    std::array<VkImageView, 8> views{};
    VkExtent2D extent{std::numeric_limits<u32>::max(), std::numeric_limits<u32>::max()};
    u32 layers = std::numeric_limits<u32>::max();
    const auto include = [&](const Image& target, const AttachmentView& view) {
        extent.width = std::min(extent.width, std::max(1U, target.extent.width >> view.level));
        extent.height = std::min(extent.height, std::max(1U, target.extent.height >> view.level));
        layers = std::min(layers, view.layers);
    };
    u32 color_count{};
    for (u32 slot = 0; slot < colors.size(); ++slot) {
        if (!colors[slot] || !colors[slot]->serial) {
            continue;
        }
        const auto& view = color_views[slot];
        if (!view.format) {
            NoteSkip("unsupported color attachment view");
            return;
        }
        auto* target = FindImage(*colors[slot], true);
        if (!target || !target->color_attachment) {
            const auto& d = colors[slot]->description;
            if (unsupported_color_formats.size() < 12 &&
                unsupported_color_formats.emplace(d.format, d.target, d.samples).second) {
                LOG_WARNING(Render_Vulkan, "Bananza NVN 2 attachment format {:#x}, target {}, samples {}, {}x{}",
                            d.format, d.target, d.samples, d.width, d.height);
            }
            NoteSkip("unsupported color attachment");
            return;
        }
        views[slot] = GetAttachmentView(*target, view);
        if (!views[slot]) {
            NoteSkip("unsupported color attachment subresource");
            return;
        }
        include(*target, view);
        formats[slot] = view.format;
        targets[slot] = target;
        color_count = slot + 1;
    }
    auto* depth = depth_texture ? FindImage(*depth_texture, true) : nullptr;
    const auto depth_attachment_view =
        depth && depth->aspects != VK_IMAGE_ASPECT_COLOR_BIT
            ? GetAttachmentView(*depth, depth_view)
            : VK_NULL_HANDLE;
    if (depth_texture && !depth_attachment_view) {
        NoteSkip("unsupported depth attachment");
        return;
    }
    if (!color_count && !depth) {
        const auto limits = device.GetPhysical().GetProperties().limits;
        const auto dimension = [](float viewport_end, s64 scissor_end, u32 limit) {
            const double end = std::isfinite(viewport_end) && viewport_end >= 1.0f
                                   ? std::ceil(viewport_end)
                                   : static_cast<double>(scissor_end);
            return static_cast<u32>(std::clamp(end, 1.0, static_cast<double>(limit)));
        };
        extent = {dimension(viewport.x + viewport.width,
                            s64{scissor.offset.x} + scissor.extent.width, limits.maxFramebufferWidth),
                  dimension(viewport.y + viewport.height,
                            s64{scissor.offset.y} + scissor.extent.height, limits.maxFramebufferHeight)};
        layers = 1;
    }
    if (depth) {
        include(*depth, depth_view);
    }
    auto raster = raster_state;
    raster.depth_format = depth ? depth->format : VK_FORMAT_UNDEFINED;
    raster.depth_aspects = depth ? depth->aspects : 0;
    raster.negative_one_to_one = submission.depth_minus_one_to_one;
    raster.origin_lower_left = submission.origin_lower_left;
    const bool biased = raster.depth_bias;
    const auto bias = depth_bias;
    const float bias_clamp = device.GetPhysical().GetFeatures().depthBiasClamp ? bias[2] : 0.0f;
    const PipelineKey key{graphics_program->serial, fragment_program ? fragment_program->serial : 0,
                          geometry_program ? geometry_program->serial : 0,
                          formats, topology, vertex_attributes,
                          vertex_strides, vertex_divisors, color_state, raster};
    const auto [entry, inserted] = pipelines.try_emplace(key);
    if (inserted) {
        try {
            entry->second = CreatePipeline(formats, color_count, topology, raster);
        } catch (const vk::Exception& error) {
            LOG_WARNING(Render_Vulkan, "Bananza NVN 2 graphics pipeline {}: {}",
                        graphics_program->serial, error.what());
        }
    }
    if (!entry->second) {
        NoteSkip("graphics pipeline unavailable");
        if (!missing_draw_logged) {
            missing_draw_logged = true;
            LOG_WARNING(Render_Vulkan, "Bananza NVN 2 Vulkan replay skipped a draw with unsupported pipeline state");
        }
        return;
    }
    const auto& pipeline = *entry->second;
    std::vector<std::pair<u32, VkBuffer>> vertices;
    for (const u32 stream : pipeline.streams) {
        const auto buffer = UploadRange(submission, vertex_buffers[stream]);
        if (!buffer) {
            NoteSkip("vertex buffer range missing at submit");
            return;
        }
        vertices.emplace_back(stream, buffer);
    }
    VkBuffer index_buffer{};
    u64 index_offset{};
    if (op.args[0] == 7 || op.args[0] == 8) {
        const bool strip = op.args[0] == 8;
        const u32 quads = draw->count < 4 ? 0 : strip ? (draw->count - 2) / 2 : draw->count / 4;
        if (!quads) {
            return;
        }
        std::span<const u8> source;
        if (draw->index_type != VK_INDEX_TYPE_MAX_ENUM) {
            const auto found = submission.buffer_snapshots.find(draw->index_range);
            if (found == submission.buffer_snapshots.end()) {
                NoteSkip("quad index buffer range missing at submit");
                return;
            }
            source = found->second;
        }
        const auto index = [&](u32 offset) -> u32 {
            if (source.empty()) {
                return draw->first + offset;
            }
            if (draw->index_type == VK_INDEX_TYPE_UINT8_EXT) {
                return source[offset];
            }
            if (draw->index_type == VK_INDEX_TYPE_UINT16) {
                u16_le value;
                std::memcpy(&value, source.data() + offset * sizeof(value), sizeof(value));
                return value;
            }
            u32_le value;
            std::memcpy(&value, source.data() + offset * sizeof(value), sizeof(value));
            return value;
        };
        std::vector<u32> triangles;
        triangles.reserve(quads * 6);
        const auto corners = strip ? std::array{0U, 3U, 1U, 0U, 2U, 3U}
                                   : std::array{0U, 1U, 2U, 0U, 2U, 3U};
        for (u32 quad = 0; quad < quads; ++quad) {
            for (const auto corner : corners) {
                triangles.push_back(index(quad * (strip ? 2 : 4) + corner));
            }
        }
        index_buffer = Upload({reinterpret_cast<const u8*>(triangles.data()),
                               triangles.size() * sizeof(u32)});
        draw->index_type = VK_INDEX_TYPE_UINT32;
        draw->count = static_cast<u32>(triangles.size());
    } else if (draw->index_type != VK_INDEX_TYPE_MAX_ENUM) {
        std::tie(index_buffer, index_offset) = UploadBufferRange(submission, draw->index_range);
        if (!index_buffer) {
            NoteSkip("index buffer range missing at submit");
            return;
        }
    }
    VkDescriptorSet set{};
    std::vector<std::pair<StorageUpload*, u64>> written_storage;
    if (!pipeline.uniforms.empty() || !pipeline.storage.empty() || !pipeline.textures.empty()) {
        std::array<u8, 0x1000> driver{};
        constexpr float max_point_size = 2047.0f;
        const u32_le base_instance = draw->base_instance;
        std::memcpy(driver.data() + 0x610, &max_point_size, sizeof(max_point_size));
        std::memcpy(driver.data() + 0x644, &base_instance, sizeof(base_instance));
        for (u32 stage = 0; stage < GraphicsStorageBase.size(); ++stage) {
            const u32 base = GraphicsStorageBase[stage];
            for (u32 slot = 0; slot < 16; ++slot) {
                const auto& range = storage_buffers[stage][slot];
                const u64_le address = range.first;
                const u32_le size = static_cast<u32>(range.second);
                std::memcpy(driver.data() + base + slot * 16, &address, sizeof(address));
                std::memcpy(driver.data() + base + slot * 16 + 8, &size, sizeof(size));
            }
        }
        std::vector<VkDescriptorBufferInfo> infos;
        infos.reserve(pipeline.uniforms.size());
        for (const auto& binding : pipeline.uniforms) {
            const auto& bound = uniform_buffers[binding.stage];
            // NVN user uniform slots follow the driver's three internal banks.
            const auto range = binding.bank >= 3 ? bound[binding.bank - 3] : Range{};
            std::span<const u8> literal{binding.literal};
            if (binding.bank == 0) {
                literal = driver;
            } else if (binding.bank == 17) {
                literal = graphics_constants[0];
            } else if (binding.bank >= 24) {
                literal = graphics_constants[binding.bank - 24];
            }
            const u64 size = literal.empty() ? range.second : literal.size();
            if (!size || size > 65536) {
                if (draw_diagnostics.size() < 40 &&
                    draw_diagnostics.emplace(graphics_program->serial, "uniform detail").second) {
                    LOG_WARNING(Render_Vulkan,
                                "Bananza NVN 2 uniform: stage {}, bank {}, expected {}, available {}",
                                binding.stage, binding.bank, binding.size, size);
                }
                NoteSkip("uniform binding has insufficient data");
                return;
            }
            const auto buffer = literal.empty() ? UploadRange(submission, range) : Upload(literal);
            if (!buffer) {
                NoteSkip("uniform buffer range missing at submit");
                return;
            }
            infos.push_back({buffer, 0, size});
        }
        std::vector<VkDescriptorBufferInfo> storage_infos;
        storage_infos.reserve(pipeline.storage.size());
        const auto storage_limits = device.GetPhysical().GetProperties().limits;
        for (const auto& [stage, binding, slot, writable] : pipeline.storage) {
            const auto range = storage_buffers[stage][slot];
            if (trace_images && stage == 2 &&
                draw_diagnostics.emplace(geometry_program->serial,
                                         slot == 0 ? "geometry storage 0" : "geometry storage 1").second) {
                LOG_INFO(Render_Vulkan,
                         "NVN 2 geometry program {} storage slot {} at {:#x}, {} bytes, write {}, snapshot {}, mapping {}",
                         geometry_program->serial, slot, range.first, range.second, writable,
                         submission.buffer_snapshots.contains(range),
                         submission.storage_addresses.contains(range));
            }
            if (!range.first || !range.second || range.second > storage_limits.maxStorageBufferRange) {
                NoteSkip("graphics storage range missing or exceeds host limit");
                return;
            }
            VkBuffer buffer{};
            u64 offset{};
            if (writable) {
                if (const auto [upload, start] = WritableStorage(submission, range); upload) {
                    buffer = *upload->buffer;
                    offset = start;
                    written_storage.emplace_back(upload, offset + range.second);
                }
            } else {
                std::tie(buffer, offset) = UploadBufferRange(submission, range);
            }
            if (!buffer || offset % storage_limits.minStorageBufferOffsetAlignment) {
                NoteSkip("graphics storage range missing at submit");
                return;
            }
            storage_infos.push_back({buffer, offset, range.second});
        }
        std::vector<VkDescriptorImageInfo> image_infos;
        image_infos.reserve(pipeline.textures.size());
        for (const auto& [stage, binding, descriptor] : pipeline.textures) {
            const auto info = BindTexture(submission, stage, descriptor, targets);
            if (!info) {
                return;
            }
            image_infos.push_back(*info);
        }
        const auto& logical = device.GetLogical();
        std::vector<VkDescriptorPoolSize> pool_sizes;
        if (!infos.empty()) {
            pool_sizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<u32>(infos.size())});
        }
        if (!storage_infos.empty()) {
            pool_sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                  static_cast<u32>(storage_infos.size())});
        }
        if (!image_infos.empty()) {
            pool_sizes.push_back({VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                  static_cast<u32>(image_infos.size())});
        }
        auto pool = logical.CreateDescriptorPool({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = static_cast<u32>(pool_sizes.size()),
            .pPoolSizes = pool_sizes.data(),
        });
        const VkDescriptorSetLayout set_layout = *pipeline.set_layout;
        const auto sets = pool.Allocate({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = *pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &set_layout,
        });
        set = sets[0];
        std::vector<VkWriteDescriptorSet> writes;
        for (size_t i = 0; i < infos.size(); ++i) {
            writes.push_back({
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = set,
                .dstBinding = pipeline.uniforms[i].binding,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &infos[i],
            });
        }
        for (size_t i = 0; i < storage_infos.size(); ++i) {
            writes.push_back({
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = set,
                .dstBinding = pipeline.storage[i].binding,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &storage_infos[i],
            });
        }
        for (size_t i = 0; i < image_infos.size(); ++i) {
            writes.push_back({
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = set,
                .dstBinding = std::get<1>(pipeline.textures[i]),
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &image_infos[i],
            });
        }
        logical.UpdateDescriptorSets(writes, {});
        descriptor_pools.push_back(std::move(pool));
    }

    VkViewport draw_viewport = viewport;
    if (!(draw_viewport.width > 0.0f && draw_viewport.height > 0.0f)) {
        draw_viewport = {0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height),
                         0.0f, 1.0f};
    }
    draw_viewport.minDepth = depth_range[0];
    draw_viewport.maxDepth = depth_range[1];
    const auto limits = device.GetPhysical().GetProperties().limits;
    const auto& bounds = limits.viewportBoundsRange;
    if (draw_viewport.x < bounds[0] || draw_viewport.y < bounds[0] ||
        draw_viewport.x + draw_viewport.width > bounds[1] ||
        draw_viewport.y + draw_viewport.height > bounds[1] ||
        draw_viewport.width > limits.maxViewportDimensions[0] ||
        draw_viewport.height > limits.maxViewportDimensions[1]) {
        return;
    }
    const VkRect2D draw_scissor = ScissorFor(extent);
    draw_viewport.y = submission.origin_lower_left
                          ? static_cast<float>(extent.height) - draw_viewport.y
                          : draw_viewport.y + draw_viewport.height;
    draw_viewport.height = -draw_viewport.height;
    std::array<VkRenderingAttachmentInfo, 8> attachments{};
    for (u32 i = 0; i < color_count; ++i) {
        auto* target = targets[i];
        attachments[i] = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = views[i],
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .loadOp = target && target->initialized ? VK_ATTACHMENT_LOAD_OP_LOAD
                                                    : VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        };
        if (target) {
            Initialize(*target);
            target->gpu_written = true;
            ++target->write_version;
        }
    }
    VkRenderingAttachmentInfo depth_attachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = depth_attachment_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        .loadOp = depth && depth->initialized ? VK_ATTACHMENT_LOAD_OP_LOAD
                                              : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };
    if (depth) {
        Initialize(*depth);
        depth->gpu_written = true;
        ++depth->write_version;
    }
    for (const auto& [upload, size] : written_storage) {
        upload->written_bytes = std::max(upload->written_bytes, size);
    }
    Barrier();
    std::optional<u32> probe;
    if (frame_dump.CapturingSampled() && colors[0] && extent.width >= 1900) {
        const std::array shaders{graphics_program->serial};
        frame_dump.NoteDraw(shaders);
        if (!count_samples) {
            probe = frame_dump.BeginProbe();
        }
        if (probe) {
            LOG_INFO(Render_Vulkan,
                     "NVN 2 probe {} program {} target {:#x}, {} elements, viewport {} {} {} {}, scissor {} {} {} {}, polygon {:#x}, depth {:#x}/{}, range {} {}",
                     *probe, graphics_program->serial,
                     colors[0]->storage.gpu_address + colors[0]->offset, draw->count,
                     draw_viewport.x, draw_viewport.y, draw_viewport.width, draw_viewport.height,
                     draw_scissor.offset.x, draw_scissor.offset.y,
                     draw_scissor.extent.width, draw_scissor.extent.height, raster.polygon,
                     raster.depth_stencil, raster.depth_aspects, draw_viewport.minDepth,
                     draw_viewport.maxDepth);
        }
    }
    if (occlusion_queries == OcclusionQueryCount) {
        CollectOcclusion();
    }
    const VkQueryPool query_pool = count_samples ? *occlusion_pool : VK_NULL_HANDLE;
    const u32 query = count_samples ? occlusion_queries++ : 0;
    const VkQueryControlFlags query_flags = precise_queries ? VK_QUERY_CONTROL_PRECISE_BIT : 0;
    scheduler.Record([attachments, color_count, depth_attachment, depth_aspects = raster.depth_aspects,
                      graphics = *pipeline.pipeline, layout = *pipeline.layout,
                      extent, layers, draw_viewport, draw_scissor, command = *draw, set, index_buffer,
                      index_offset,
                      indirect_buffer, indirect_offset, biased, bias, bias_clamp,
                      query_pool, query, query_flags,
                      blend_color = blend_constants, uses_render_area = pipeline.uses_render_area,
                      stages = pipeline.stages,
                      refs = stencil_ref, compare_masks = stencil_compare_mask,
                      write_masks = stencil_write_mask,
                      vertices = std::move(vertices)](
                         vk::CommandBuffer cmdbuf) {
        if (query_pool) {
            cmdbuf.ResetQueryPool(query_pool, query, 1);
        }
        cmdbuf.BeginRendering(VkRenderingInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = {{}, extent},
            .layerCount = layers,
            .colorAttachmentCount = color_count,
            .pColorAttachments = attachments.data(),
            .pDepthAttachment = depth_aspects & VK_IMAGE_ASPECT_DEPTH_BIT ? &depth_attachment : nullptr,
            .pStencilAttachment = depth_aspects & VK_IMAGE_ASPECT_STENCIL_BIT ? &depth_attachment : nullptr,
        });
        if (query_pool) {
            cmdbuf.BeginQuery(query_pool, query, query_flags);
        }
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, graphics);
        cmdbuf.SetViewport(0, {draw_viewport});
        cmdbuf.SetScissor(0, {draw_scissor});
        cmdbuf.SetBlendConstants(blend_color.data());
        if (biased) {
            cmdbuf.SetDepthBias(bias[1], bias_clamp, bias[0]);
        }
        for (u32 i = 0; i < 2; ++i) {
            const VkStencilFaceFlags face = i == 0 ? VK_STENCIL_FACE_FRONT_BIT : VK_STENCIL_FACE_BACK_BIT;
            cmdbuf.SetStencilReference(face, refs[i]);
            cmdbuf.SetStencilCompareMask(face, compare_masks[i]);
            cmdbuf.SetStencilWriteMask(face, write_masks[i]);
        }
        using namespace Shader::Backend::SPIRV;
        constexpr RescalingLayout rescale{.down_factor = std::bit_cast<u32>(1.0f)};
        cmdbuf.PushConstants(layout, stages, rescale);
        if (uses_render_area) {
            const RenderAreaLayout area{{static_cast<float>(extent.width),
                                         static_cast<float>(extent.height), 0, 0}};
            cmdbuf.PushConstants(layout, stages, RENDERAREA_LAYOUT_OFFSET, sizeof(area), &area);
        }
        if (set) {
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, {set}, {});
        }
        for (const auto& [stream, buffer] : vertices) {
            cmdbuf.BindVertexBuffer(stream, buffer, 0);
        }
        if (command.indexed_indirect) {
            cmdbuf.BindIndexBuffer(index_buffer, index_offset, command.index_type);
            cmdbuf.DrawIndexedIndirect(indirect_buffer, indirect_offset, 1, sizeof(VkDrawIndexedIndirectCommand));
        } else if (indirect_buffer) {
            cmdbuf.DrawIndirect(indirect_buffer, indirect_offset, 1, sizeof(VkDrawIndirectCommand));
        } else if (index_buffer) {
            cmdbuf.BindIndexBuffer(index_buffer, index_offset, command.index_type);
            cmdbuf.DrawIndexed(command.count, command.instances, 0, command.base_vertex,
                               command.base_instance);
        } else {
            cmdbuf.Draw(command.count, command.instances, command.first, command.base_instance);
        }
        if (query_pool) {
            cmdbuf.EndQuery(query_pool, query);
        }
        cmdbuf.EndRendering();
    });
    if (probe) {
        frame_dump.EndProbe(*probe, {.first = draw->first, .elements = draw->count,
                                     .instances = draw->instances,
                                     .color0 = (color_state.channel_mask & 15) != 0});
    }
    if (colors[0]) {
        TraceImage("draw", *colors[0], graphics_program->serial);
    }
    if (++draws == 1) {
        LOG_INFO(Render_Vulkan, "Bananza NVN 2 Vulkan replay recorded its first guest draw");
    }
}

void BananzaNvn2Replay::ReplayRecording(const Recording& recording, const Submission& submission) {
    for (const auto& op : recording.operations) {
        if (op.nested) {
            ReplayRecording(*op.nested, submission);
            continue;
        }
        switch (op.replay) {
        case ReplayOp::SetRenderTargets: {
            colors.fill(std::nullopt);
            color_views.fill({});
            depth_texture.reset();
            depth_view = {};
            if (op.args.empty() || op.args[0] > colors.size()) {
                break;
            }
            const size_t count = op.args[0];
            if (op.textures.size() != count + 1 || op.args.size() != 2 * count + 3 ||
                op.payload.size() != 16 * (count + 1)) {
                break;
            }
            const auto attachment = [&](const Texture& texture, std::span<const u8> view,
                                        bool has_view) -> AttachmentView {
                const auto& desc = texture.description;
                const u32 format = has_view ? ViewFormat(view, desc.format) : desc.format;
                const auto host = FindColorFormat(format);
                if (!host || !HasCompatibleFormat(format, desc.format)) {
                    return {};
                }
                const u8 flags = has_view ? view[7] : 0;
                const u32 level = flags & 1 ? u32{view[4]} : 0;
                if (level >= desc.levels) {
                    return {};
                }
                const u32 available = desc.target == 2 ? std::max(1U, desc.depth >> level)
                                                       : texture.layout.layers;
                const u32 first = flags & 2 ? u32{view[0]} | (u32{view[1]} << 8) : 0;
                const u32 count = flags & 2 ? u32{view[2]} | (u32{view[3]} << 8) : 0;
                if (first >= available) {
                    return {};
                }
                return {
                    .format = host->vk,
                    .level = level,
                    .layer = first,
                    .layers = count ? count : available - first,
                };
            };
            for (size_t i = 0; i < count; ++i) {
                if (op.textures[i].serial) {
                    colors[i] = op.textures[i];
                    color_views[i] = attachment(*colors[i], std::span{op.payload}.subspan(16 * i, 16),
                                                op.args[2 + count + i] != 0);
                }
            }
            if (const auto& texture = op.textures.back(); texture.serial) {
                depth_texture = texture;
                depth_view = attachment(texture, std::span{op.payload}.last(16), op.args.back() != 0);
            }
            break;
        }
        case ReplayOp::ClearColor:
        case ReplayOp::ClearColorSigned:
        case ReplayOp::ClearColorUnsigned:
            ClearColor(op);
            break;
        case ReplayOp::ClearDepthStencil:
            ClearDepthStencil(op);
            break;
        case ReplayOp::BindDepthStencil:
            if (op.payload.size() == sizeof(u64)) {
                u64_le value;
                std::memcpy(&value, op.payload.data(), sizeof(value));
                raster_state.depth_stencil = value & 0xffffffff000000f7ULL;
            }
            break;
        case ReplayOp::BindPolygon:
            if (op.payload.size() == sizeof(u32)) {
                u32_le value;
                std::memcpy(&value, op.payload.data(), sizeof(value));
                raster_state.polygon = value & 31;
                // Compact NVN polygon state stores offset enables above the mode fields.
                const u32 offset_enables = (value >> 15) & 7;
                const u32 mode = (value >> 3) & 3;
                raster_state.depth_bias = mode < 3 && (offset_enables & (1U << mode)) != 0;
            }
            break;
        case ReplayOp::SetDepthClamp:
        case ReplayOp::SetRasterizerDiscard:
            if (op.args.size() == 1) {
                bool& enabled = op.replay == ReplayOp::SetDepthClamp ? raster_state.depth_clamp
                                                                     : raster_state.discard;
                enabled = (op.args[0] & 255) != 0;
            }
            break;
        case ReplayOp::SetPolygonOffsetClamp:
            if (op.payload.size() == sizeof(depth_bias)) {
                std::memcpy(depth_bias.data(), op.payload.data(), sizeof(depth_bias));
            }
            break;
        case ReplayOp::SetDepthRange:
            if (op.payload.size() == sizeof(depth_range)) {
                std::memcpy(depth_range.data(), op.payload.data(), sizeof(depth_range));
            }
            break;
        case ReplayOp::SetStencilRef:
        case ReplayOp::SetStencilMask:
        case ReplayOp::SetStencilValueMask:
            if (op.args.size() == 2) {
                auto& values = op.replay == ReplayOp::SetStencilRef ? stencil_ref
                               : op.replay == ReplayOp::SetStencilMask ? stencil_write_mask
                                                                      : stencil_compare_mask;
                for (u32 i = 0; i < values.size(); ++i) {
                    if (op.args[0] & (1U << i)) {
                        values[i] = static_cast<u32>(op.args[1]) & 255;
                    }
                }
            }
            break;
        case ReplayOp::CopyTexture:
            CopyTexture(op, submission);
            break;
        case ReplayOp::CopyBufferToTexture:
            CopyBufferToTexture(op, submission);
            break;
        case ReplayOp::SetCopyRowStride:
            if (op.args.size() == 1) {
                copy_row_stride = op.args[0];
            }
            break;
        case ReplayOp::Barrier:
            Barrier();
            break;
        case ReplayOp::BindProgram:
            if (op.args.size() == 2) {
                const auto bind = [&](std::shared_ptr<const Program>& destination, u32 bit, u32 stage) {
                    if (!(op.args[1] & bit)) {
                        return;
                    }
                    if (op.program && std::ranges::any_of(op.program->shaders, [stage](const auto& shader) {
                            return shader.stage == stage;
                        })) {
                        destination = op.program;
                    } else {
                        destination.reset();
                    }
                };
                bind(graphics_program, 1, 0);
                bind(fragment_program, 2, 4);
                bind(geometry_program, 4, 3);
                bind(compute_program, 0x20, 5);
            }
            break;
        case ReplayOp::BindVertexBuffer:
            if (op.args.size() == 3 && op.args[0] < vertex_buffers.size()) {
                vertex_buffers[op.args[0]] = {op.args[1], op.args[2]};
            }
            break;
        case ReplayOp::BindUniformBuffer:
        case ReplayOp::BindStorageBuffer: {
            auto& ranges = op.replay == ReplayOp::BindUniformBuffer ? uniform_buffers
                                                                   : storage_buffers;
            if (op.args.size() == 4 && op.args[0] < ranges.size() && op.args[1] < 32) {
                ranges[op.args[0]][op.args[1]] = {op.args[2], op.args[3]};
            }
            break;
        }
        case ReplayOp::BindUniformBuffers:
        case ReplayOp::BindStorageBuffers: {
            auto& ranges = op.replay == ReplayOp::BindUniformBuffers ? uniform_buffers
                                                                    : storage_buffers;
            if (op.args.size() == 3 && op.args[0] < ranges.size() && op.args[1] < 32 &&
                op.args[2] <= 32 - op.args[1] && op.payload.size() == 16 * op.args[2]) {
                for (size_t i = 0; i < op.args[2]; ++i) {
                    std::array<u64_le, 2> range{};
                    std::memcpy(range.data(), op.payload.data() + i * 16, 16);
                    ranges[op.args[0]][op.args[1] + i] = {range[0], range[1]};
                }
            }
            break;
        }
        case ReplayOp::BindImage:
            if (op.args.size() == 3 && op.args[0] < image_handles.size() && op.args[1] < 32) {
                image_handles[op.args[0]][op.args[1]] = op.args[2];
            }
            break;
        case ReplayOp::BindTexture:
            if (op.args.size() == 3 && op.args[0] < texture_handles.size() && op.args[1] < 64) {
                texture_handles[op.args[0]][op.args[1]] = op.args[2];
            }
            break;
        case ReplayOp::BindTextures:
        case ReplayOp::BindTexturesDeferred:
            if (op.args.size() >= 3 && op.args[0] < texture_handles.size() && op.args[1] < 64) {
                auto& handles = texture_handles[op.args[0]];
                const u64 first = op.args[1];
                const u64 count = op.args[2];
                if (!count || count > handles.size() - first) {
                    break;
                }
                std::span<const u8> data = op.payload;
                if (op.replay == ReplayOp::BindTexturesDeferred && op.args.size() == 4) {
                    const auto found = submission.deferred_data.find({op.args[3], count * 8});
                    if (found != submission.deferred_data.end()) {
                        data = found->second;
                    }
                }
                if (data.size() == count * 8) {
                    std::memcpy(handles.data() + first, data.data(), data.size());
                }
            }
            break;
        case ReplayOp::SetTexturePool:
            if (op.args.size() == 1) {
                texture_pool = op.args[0];
            }
            break;
        case ReplayOp::SetSamplerPool:
            if (op.args.size() == 1) {
                sampler_pool = op.args[0];
            }
            break;
        case ReplayOp::PushGraphicsConstants:
        case ReplayOp::PushComputeConstants: {
            auto& constants = op.replay == ReplayOp::PushGraphicsConstants ? graphics_constants
                                                                          : compute_constants;
            if (op.args.size() == 2 && op.args[0] < graphics_constants.size() &&
                op.args[1] <= 256 && op.payload.size() <= 256 - op.args[1]) {
                std::ranges::copy(op.payload, constants[op.args[0]].begin() + op.args[1]);
            }
            break;
        }
        case ReplayOp::UpdateUniformBuffer:
            if (op.args.size() == 4 && !op.payload.empty() && op.payload.size() == op.args[3] &&
                op.args[0] <= std::numeric_limits<u64>::max() - op.args[1] &&
                op.args[2] <= op.args[1] && op.args[3] <= op.args[1] - op.args[2]) {
                const u64 address = op.args[0] + op.args[2];
                const u64 end = address + op.payload.size();
                buffer_updates.push_back({address, op.payload});
                std::erase_if(uploaded_ranges, [address, end](const auto& entry) {
                    const auto [start, size] = entry.first;
                    return start < end && address < start + size;
                });
            }
            break;
        case ReplayOp::BindBlendState:
            if (op.payload.size() == sizeof(u64)) {
                u64_le packed;
                std::memcpy(&packed, op.payload.data(), sizeof(packed));
                color_state.blend[packed & 7] = packed & 0x7f7f7f7f77003f07ULL;
            }
            break;
        case ReplayOp::BindColorState:
        case ReplayOp::BindChannelMask:
        case ReplayOp::SetAlphaRef:
            if (op.payload.size() >= sizeof(u32)) {
                u32_le packed;
                std::memcpy(&packed, op.payload.data(), sizeof(packed));
                if (op.replay == ReplayOp::BindColorState) {
                    color_state.control = packed & 0x0fffff00;
                } else if (op.replay == ReplayOp::BindChannelMask) {
                    color_state.channel_mask = packed;
                } else {
                    color_state.alpha_ref = packed;
                }
            }
            break;
        case ReplayOp::SetBlendColor:
            if (op.payload.size() == sizeof(blend_constants)) {
                std::memcpy(blend_constants.data(), op.payload.data(), sizeof(blend_constants));
            }
            break;
        case ReplayOp::StencilCullCriteria:
            break;
        case ReplayOp::BindVertexAttribState:
            if (op.args.size() == 1 && op.args[0] <= vertex_attributes.size() &&
                op.payload.size() == op.args[0] * 4) {
                vertex_attributes.fill(0);
                std::memcpy(vertex_attributes.data(), op.payload.data(), op.payload.size());
            }
            break;
        case ReplayOp::BindVertexStreamState:
            if (op.args.size() == 1 && op.args[0] <= vertex_strides.size() &&
                op.payload.size() == op.args[0] * 8) {
                vertex_strides.fill(0);
                vertex_divisors.fill(0);
                for (size_t i = 0; i < op.args[0]; ++i) {
                    std::memcpy(&vertex_strides[i], op.payload.data() + i * 8, 4);
                    std::memcpy(&vertex_divisors[i], op.payload.data() + i * 8 + 4, 4);
                }
            }
            break;
        case ReplayOp::SetViewport:
            if (op.args.size() == 4) {
                viewport = {static_cast<float>(static_cast<s32>(op.args[0])),
                            static_cast<float>(static_cast<s32>(op.args[1])),
                            static_cast<float>(static_cast<s32>(op.args[2])),
                            static_cast<float>(static_cast<s32>(op.args[3])), 0.0f, 1.0f};
            }
            break;
        case ReplayOp::SetViewports:
            if (op.args.size() == 2 && op.args[0] == 0 && op.payload.size() >= 4 * sizeof(float)) {
                std::array<float, 4> rect{};
                std::memcpy(rect.data(), op.payload.data(), sizeof(rect));
                if (std::ranges::all_of(rect, [](float value) { return std::isfinite(value); })) {
                    viewport = {rect[0], rect[1], rect[2], rect[3], 0.0f, 1.0f};
                }
            }
            break;
        case ReplayOp::SetScissor:
            if (op.args.size() == 4) {
                scissor = {{static_cast<s32>(op.args[0]), static_cast<s32>(op.args[1])},
                           {static_cast<u32>(op.args[2]), static_cast<u32>(op.args[3])}};
                has_scissor = true;
            }
            break;
        case ReplayOp::SetScissors:
            if (op.args.size() == 2 && op.args[0] == 0 && op.payload.size() >= 4 * sizeof(u32)) {
                std::array<s32_le, 4> rect{};
                std::memcpy(rect.data(), op.payload.data(), sizeof(rect));
                scissor = {{rect[0], rect[1]},
                           {static_cast<u32>(std::max<s32>(0, rect[2])),
                            static_cast<u32>(std::max<s32>(0, rect[3]))}};
                has_scissor = true;
            }
            break;
        case ReplayOp::SetDepthRanges:
            if (op.args.size() == 2 && op.args[0] == 0 && op.payload.size() >= sizeof(depth_range)) {
                std::memcpy(depth_range.data(), op.payload.data(), sizeof(depth_range));
            }
            break;
        case ReplayOp::Draw:
        case ReplayOp::DrawIndexed:
        case ReplayOp::DrawIndexedBaseVertex:
        case ReplayOp::DrawIndexedInstanced:
        case ReplayOp::DrawInstanced:
        case ReplayOp::DrawIndirect:
        case ReplayOp::DrawIndexedIndirect:
            Draw(op, submission);
            break;
        case ReplayOp::Dispatch:
            Dispatch(op, submission);
            break;
        case ReplayOp::SignalEvent:
        case ReplayOp::WaitEvent:
            ReplayEvent(op, submission);
            break;
        case ReplayOp::SetRenderEnable:
            if (op.args.size() == 1) {
                render_enabled = (op.args[0] & 255) != 0;
            }
            break;
        case ReplayOp::SetRenderEnableConditional:
        case ReplayOp::ReportCounter:
        case ReplayOp::ResetCounter:
            ReplayCounter(op, submission);
            break;
        case ReplayOp::Unsupported:
            break;
        }
    }
}

void BananzaNvn2Replay::CollectOcclusion() {
    if (!occlusion_queries) {
        return;
    }
    scheduler.Finish();
    std::array<u64, OcclusionQueryCount> results{};
    vk::Check(device.GetLogical().GetQueryResults(*occlusion_pool, 0, occlusion_queries,
                                                  sizeof(results), results.data(), sizeof(u64),
                                                  VK_QUERY_RESULT_64_BIT));
    occlusion_samples = std::accumulate(results.begin(), results.begin() + occlusion_queries,
                                        occlusion_samples);
    occlusion_queries = 0;
}

void BananzaNvn2Replay::ReplayCounter(const Operation& op, const Submission& submission) {
    constexpr u64 SamplesPassed = 1;
    if (op.replay == ReplayOp::ResetCounter) {
        if (op.args.size() == 1 && op.args[0] == SamplesPassed) {
            if (!occlusion_pool) {
                occlusion_pool = device.GetLogical().CreateQueryPool({
                    .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    .queryType = VK_QUERY_TYPE_OCCLUSION,
                    .queryCount = OcclusionQueryCount,
                });
                precise_queries = device.GetPhysical().GetFeatures().occlusionQueryPrecise;
            }
            CollectOcclusion();
            occlusion_samples = 0;
            count_samples = true;
        }
        return;
    }
    if (!submission.guest_memory || op.args.size() != 2) {
        return;
    }
    const u64 type = op.args[0];
    const auto address = submission.event_addresses.find(op.args[1]);
    if (address == submission.event_addresses.end()) {
        return;
    }
    auto& memory = *submission.guest_memory;
    if (op.replay == ReplayOp::SetRenderEnableConditional && type <= 1) {
        std::array<u64_le, 3> data{};
        if (memory.ReadBlockUnsafe(address->second, data.data(), sizeof(data))) {
            const bool equal = data[0] == data[2];
            render_enabled = equal == (type == 0);
        }
    } else if (op.replay == ReplayOp::ReportCounter) {
        if (type == SamplesPassed) {
            CollectOcclusion();
        } else if (type == 0) {
            scheduler.Finish();
        } else if (type != 16) {
            if (unsupported_counters.insert(type).second) {
                LOG_WARNING(Render_Vulkan, "Bananza NVN 2 counter type {} needs a Vulkan query", type);
            }
            return;
        }
        const std::array<u64_le, 2> report{type == SamplesPassed ? occlusion_samples : 0,
                                           submission.submitted_ns};
        memory.WriteBlockUnsafe(address->second, report.data(), sizeof(report));
    }
}

void BananzaNvn2Replay::ReplayEvent(const Operation& op, const Submission& submission) {
    const bool signal = op.replay == ReplayOp::SignalEvent;
    if (!submission.guest_memory || op.args.size() != (signal ? 5 : 3) || op.args[1] > 1) {
        return;
    }
    const auto address = submission.event_addresses.find(op.args[0]);
    if (address == submission.event_addresses.end()) {
        return;
    }
    auto& memory = *submission.guest_memory;
    if (signal) {
        const auto value = static_cast<u32>(op.args[4]);
        if (op.args[2] != 0) {
            scheduler.Finish();
        }
        memory.Write32(address->second, op.args[1] == 1 ? memory.Read32(address->second) + value
                                                        : value);
        return;
    }
    const auto value = static_cast<u32>(op.args[2]);
    const auto current = memory.Read32(address->second);
    const bool ready = op.args[1] == 0 ? current == value : static_cast<s32>(current - value) >= 0;
    if (!ready && missing_event_waits.size() < 16 && missing_event_waits.insert(op.args[0]).second) {
        LOG_WARNING(Render_Vulkan, "Bananza NVN 2 event wait at {:#x}: current {}, expected {}, mode {}",
                    op.args[0], current, value, op.args[1]);
    }
}

void BananzaNvn2Replay::Replay(const Core::BananzaNvn2Commands::Submission& submission) {
    origin_lower_left = submission.origin_lower_left;
    CollectStorageSpans(submission);
    const u64 previous_work = clears + copies + draws + dispatches + texture_uploads;
    for (const auto& recording : submission.recordings) {
        ReplayRecording(*recording, submission);
    }
    if (clears + copies + draws + dispatches + texture_uploads != previous_work) {
        if (!storage_uploads.empty()) {
            scheduler.Record([](vk::CommandBuffer cmdbuf) {
                const VkMemoryBarrier barrier{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
                };
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                       VK_PIPELINE_STAGE_HOST_BIT, 0, barrier);
            });
        }
        scheduler.Finish();
        WriteBackStorage(submission);
        if (!replay_logged) {
            replay_logged = true;
            LOG_INFO(Render_Vulkan, "Bananza NVN 2 Vulkan replay completed {} clears and {} copies",
                     clears, copies);
        }
    }
    uploaded_ranges.clear();
    storage_uploads.clear();
    buffer_updates.clear();
    reset_indirect_draws.clear();
    uploads.clear();
    descriptor_pools.clear();
    samplers.clear();
}

std::optional<BananzaNvn2Replay::Source> BananzaNvn2Replay::PresentSource(const Texture& texture) {
    captured_inputs.clear();
    TraceImage("present", texture);
    const auto* image = FindImage(texture, false);
    if (!image || !image->initialized) {
        if (!missing_source_logged) {
            missing_source_logged = true;
            LOG_WARNING(Render_Vulkan, "Bananza NVN 2 present image has no replayed writes (format {:#x}, {}x{})",
                        texture.description.format, texture.description.width,
                        texture.description.height);
        }
        return std::nullopt;
    }
    ++present_count;
    const auto now = std::chrono::steady_clock::now();
    if (now >= next_replay_report) {
        const auto presents = present_count - last_reported_present;
        LOG_INFO(Render_Vulkan, "Bananza NVN 2 Vulkan replay: {} draws in {} presents, {} dispatches and {} texture uploads total, {} MiB images",
                 draws - last_reported_draws, presents, dispatches, texture_uploads, image_bytes >> 20);
        if (trace_images) {
            for (const auto& [reason, count] : draw_skips) {
                LOG_INFO(Render_Vulkan, "NVN 2 graphics skipped {} in {} presents: {}", count,
                         presents, reason);
            }
            for (const auto& [reason, count] : dispatch_skips) {
                LOG_INFO(Render_Vulkan, "NVN 2 compute skipped {} in {} presents: {}", count,
                         presents, reason);
            }
        }
        draw_skips.clear();
        dispatch_skips.clear();
        last_reported_draws = draws;
        last_reported_present = present_count;
        next_replay_report = now + std::chrono::seconds{10};
    }
    Barrier();
    return Source{
        .texture = {*image->image, *image->view, image->extent.width, image->extent.height,
                    image->extent.width, image->extent.height},
        .format = image->surface_format,
        .address = texture.storage.gpu_address + texture.offset,
    };
}

class BananzaNvn2Replay::ComputeEnvironment final : public Shader::Environment {
public:
    ComputeEnvironment(const Core::BananzaNvn2Hle::ResourceShaderInfo& shader_,
                       const Submission& submission_, u64 texture_pool_, const ComputeLayout& layout_,
                       std::span<const u8> driver_, const std::array<Range, 32>& uniforms_,
                       const std::array<Range, 32>& storage_)
        : shader{shader_}, submission{submission_}, pool{texture_pool_}, layout{layout_},
          driver{driver_}, uniforms{uniforms_}, storage{storage_} {
        stage = Shader::Stage::Compute;
        start_address = sizeof(Shader::ProgramHeader);
    }

    u64 ReadInstruction(u32) override { return 0; }
    bool IsNativeProgram() const override { return true; }
    bool HasHLEMacroState() const override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return texture_bank; }
    u32 LocalMemorySize() const override { return ControlWord(0x68) + ControlWord(0x6c); }
    u32 SharedMemorySize() const override { return (ControlWord(0x64) + 255) & ~255U; }
    std::array<u32, 3> WorkgroupSize() const override {
        return {ControlWord(0x58), ControlWord(0x5c), ControlWord(0x60)};
    }
    void Dump(u64, u64) override {}
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    bool IsCbufStorageBufferPointer(u32 bank, u32 offset) override {
        return StorageRange(bank, offset).first != 0;
    }
    bool IsCbufIndirectStorageBufferPointer(u32, u32, u32) override { return false; }

    Range StorageRange(u32 bank, u32 offset) const {
        if (bank == 0) {
            if (const auto slot = Slot(offset, layout.storage_base, layout.storage, 16)) {
                return storage[*slot];
            }
            if (const auto slot = Slot(offset, layout.uniform_base, layout.uniforms, 16)) {
                return uniforms[*slot];
            }
        }
        return {};
    }

    u32 ReadCbufValue(u32 bank, u32 offset) override {
        const auto data = Bank(bank);
        return Word(data, offset);
    }

    Shader::TextureType ReadTextureType(u32 handle) override {
        const auto* descriptor = TextureDescriptor(handle);
        if (!descriptor) {
            missing_texture = true;
            return Shader::TextureType::Color2D;
        }
        const auto& view = descriptor->view;
        const u32 target = view && ((*view)[7] & 0x20) ? (*view)[6]
                                                       : descriptor->texture.description.target;
        switch (target) {
        case 0: return Shader::TextureType::Color1D;
        case 1: return Shader::TextureType::Color2D;
        case 2: return Shader::TextureType::Color3D;
        case 3: return Shader::TextureType::ColorArray1D;
        case 4: return Shader::TextureType::ColorArray2D;
        case 5: return Shader::TextureType::Color2D;
        case 6: return Shader::TextureType::ColorArray2D;
        case 7: return Shader::TextureType::Color2DRect;
        case 8: return Shader::TextureType::ColorCube;
        case 9: return Shader::TextureType::ColorArrayCube;
        case 10: return Shader::TextureType::Buffer;
        default:
            missing_texture = true;
            return Shader::TextureType::Color2D;
        }
    }

    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override {
        const auto* descriptor = TextureDescriptor(handle);
        if (descriptor) {
            const u32 base = descriptor->texture.description.format;
            const auto format = FindColorFormat(descriptor->view ? ViewFormat(*descriptor->view, base)
                                                                  : base);
            if (format) {
                return static_cast<Shader::TexturePixelFormat>(format->surface);
            }
        }
        missing_texture = true;
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }

    bool IsTexturePixelFormatInteger(u32 handle) override {
        return VideoCore::Surface::IsPixelFormatInteger(
            static_cast<PixelFormat>(ReadTexturePixelFormat(handle)));
    }

    bool Matches(const VideoCommon::RecordedEnvQueries& queries) {
        missing_texture = false;
        const auto handle_at = [&](const Shader::HandleKey& at) {
            const u32 first = ReadCbufValue(at.index, at.offset) << at.shift_left;
            return at.has_secondary
                       ? first | (ReadCbufValue(at.sec_index, at.sec_offset) << at.sec_shift_left)
                       : first;
        };
        for (const auto& [key, value] : queries.cbuf_values) {
            if (ReadCbufValue(static_cast<u32>(key >> 32), static_cast<u32>(key)) != value) {
                return false;
            }
        }
        for (const auto& [key, mapped] : queries.storage_pointer_types) {
            const bool current = !(key >> 63) &&
                                 IsCbufStorageBufferPointer(static_cast<u32>(key >> 32),
                                                            static_cast<u32>(key));
            if (current != bool(mapped)) {
                return false;
            }
        }
        for (const auto& [handle, type] : queries.texture_types) {
            if (ReadTextureType(handle) != type) {
                return false;
            }
        }
        for (const auto& [handle, format] : queries.texture_pixel_formats) {
            if (ReadTexturePixelFormat(handle) != format) {
                return false;
            }
        }
        for (const auto& [at, type] : queries.location_texture_types) {
            if (ReadTextureType(handle_at(at)) != type) {
                return false;
            }
        }
        for (const auto& [at, format] : queries.location_texture_formats) {
            if (ReadTexturePixelFormat(handle_at(at)) != format) {
                return false;
            }
        }
        for (const auto& [at, is_integer] : queries.location_integer_formats) {
            if (IsTexturePixelFormatInteger(handle_at(at)) != is_integer) {
                return false;
            }
        }
        return !missing_texture;
    }

    std::span<const u8> Bank(u32 bank) const {
        if (bank == 0 || bank == 2) {
            return driver;
        }
        if (bank == 1) {
            return std::span{shader.code}.subspan(shader.constant_offset, shader.constant_size);
        }
        if (bank - 3 < uniforms.size()) {
            const auto data = submission.buffer_snapshots.find(uniforms[bank - 3]);
            if (data != submission.buffer_snapshots.end()) {
                return data->second;
            }
        }
        return {};
    }

    static std::optional<u32> Slot(u32 offset, u32 base, u32 count, u32 stride) {
        if (offset < base || (offset - base) % stride || (offset - base) / stride >= count) {
            return std::nullopt;
        }
        return (offset - base) / stride;
    }

    const Core::BananzaNvn2Hle::ResourceShaderInfo& shader;
    u32 texture_bank{};
    bool missing_texture{};

private:
    static u32 Word(std::span<const u8> bytes, u32 offset) {
        if (offset > bytes.size() || bytes.size() - offset < sizeof(u32)) {
            return 0;
        }
        u32_le value;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        return value;
    }
    u32 ControlWord(u32 offset) const { return Word(shader.control, offset); }

    const Core::BananzaNvn2Hle::ResourceTextureDescriptor* TextureDescriptor(u32 handle) const {
        const auto descriptors = submission.texture_descriptors.find(pool);
        if (descriptors == submission.texture_descriptors.end()) {
            return nullptr;
        }
        const auto found = descriptors->second.find(handle & 0xfffff);
        return found == descriptors->second.end() ? nullptr : &found->second;
    }

    const Submission& submission;
    u64 pool;
    const ComputeLayout& layout;
    std::span<const u8> driver;
    const std::array<Range, 32>& uniforms;
    const std::array<Range, 32>& storage;
};

std::optional<BananzaNvn2Replay::ComputePipeline> BananzaNvn2Replay::CreateComputePipeline(
    ComputeEnvironment& environment) {
    Shader::Backend::Bindings bindings;
    const auto& profile = pipeline_cache.ShaderProfile();
    auto shader = TranslateBananzaNvn2Shader(environment.shader, profile,
                                            pipeline_cache.ShaderHostInfo(), {}, bindings,
                                            &environment);
    if (shader.status == BananzaNvn2ShaderStatus::BindingMismatch &&
        shader.texture_bound != environment.texture_bank) {
        environment.texture_bank = shader.texture_bound;
        shader = TranslateBananzaNvn2Shader(environment.shader, profile,
                                            pipeline_cache.ShaderHostInfo(), {}, bindings,
                                            &environment);
    }
    const u64 serial = compute_program->serial;
    if (shader.status != BananzaNvn2ShaderStatus::Success || environment.missing_texture) {
        LOG_WARNING(Render_Vulkan, "Bananza NVN 2 compute program {} at {:#x}: {}",
                    serial, environment.shader.data_address,
                    environment.missing_texture ? "missing texture metadata" : shader.error);
        return std::nullopt;
    }
    const auto& info = *shader.info;
    if (!profile.unified_descriptor_binding || info.uses_global_memory ||
        info.stores_global_memory || !info.image_buffer_descriptors.empty() ||
        !info.texture_buffer_descriptors.empty() || !info.handle_programs.empty()) {
        LOG_WARNING(Render_Vulkan, "Bananza NVN 2 compute program {} ({:016x}) needs unsupported resources (global {} / {}, texel {}, image buffer {}, dynamic handles {})",
                    serial, shader.native_hash, info.uses_global_memory, info.stores_global_memory,
                    info.texture_buffer_descriptors.size(), info.image_buffer_descriptors.size(),
                    info.handle_programs.size());
        return std::nullopt;
    }
    ComputePipeline result{.queries = std::move(shader.queries), .texture_bank = shader.texture_bound,
                           .workgroup = shader.workgroup_size};
    std::vector<VkDescriptorSetLayoutBinding> descriptors;
    const auto add = [&](VkDescriptorType type) {
        const u32 index = static_cast<u32>(descriptors.size());
        descriptors.push_back({index, type, 1, VK_SHADER_STAGE_COMPUTE_BIT});
        return index;
    };
    for (const auto& uniform : info.constant_buffer_descriptors) {
        if (uniform.count != 1 || uniform.index >= 32) {
            return std::nullopt;
        }
        result.state.uniforms.push_back({5, uniform.index, add(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),
                                       info.constant_buffer_used_sizes[uniform.index]});
    }
    for (const auto& storage : info.storage_buffers_descriptors) {
        if (storage.count != 1 || storage.indirect_offset != ~0U) {
            LOG_WARNING(Render_Vulkan, "Bananza NVN 2 compute program {} needs indirect storage pointers", serial);
            return std::nullopt;
        }
        result.storage.emplace_back(add(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER), storage);
    }
    for (const auto& texture : info.texture_descriptors) {
        if (texture.count != 1 || texture.has_secondary || texture.shift_left ||
            texture.is_multisample || texture.is_integer ||
            (texture.type != Shader::TextureType::Color2D &&
             texture.type != Shader::TextureType::ColorArray2D &&
             texture.type != Shader::TextureType::Color3D &&
             texture.type != Shader::TextureType::ColorCube &&
             texture.type != Shader::TextureType::ColorArrayCube)) {
            return std::nullopt;
        }
        result.state.textures.emplace_back(5, add(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
                                           texture);
    }
    for (const auto& image : info.image_descriptors) {
        if (image.count != 1 || image.is_byte_addressed ||
            (image.type != Shader::TextureType::Color2D &&
             image.type != Shader::TextureType::ColorArray2D)) {
            return std::nullopt;
        }
        result.images.emplace_back(add(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE), image);
    }
    if (descriptors.size() != bindings.unified) {
        return std::nullopt;
    }
    const auto& logical = device.GetLogical();
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                  sizeof(Shader::Backend::SPIRV::RescalingLayout)};
    if (!descriptors.empty()) {
        result.state.set_layout = logical.CreateDescriptorSetLayout({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = static_cast<u32>(descriptors.size()),
            .pBindings = descriptors.data(),
        });
    }
    const VkDescriptorSetLayout set_layout = result.state.set_layout ? *result.state.set_layout
                                                                    : VK_NULL_HANDLE;
    result.state.layout = logical.CreatePipelineLayout({
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = set_layout ? 1U : 0U,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push,
    });
    result.state.pipeline = CreateBananzaNvn2ComputePipeline(device, shader, *result.state.layout);
    LOG_INFO(Render_Vulkan, "Bananza NVN 2 compute program {} ({:016x}) built: {} uniforms, {} storage buffers, {} textures, {} images, group {}x{}x{}",
             serial, shader.native_hash, result.state.uniforms.size(), result.storage.size(),
             result.state.textures.size(), result.images.size(), shader.workgroup_size[0],
             shader.workgroup_size[1], shader.workgroup_size[2]);
    return result;
}

std::optional<VkDescriptorImageInfo> BananzaNvn2Replay::BindStorageImage(
    const Submission& submission, u64 handle, const Shader::ImageDescriptor& binding,
    Image*& image) {
    const auto descriptors = submission.texture_descriptors.find(texture_pool);
    if (descriptors == submission.texture_descriptors.end()) {
        return std::nullopt;
    }
    const auto found = descriptors->second.find(handle & 0xfffff);
    if (found == descriptors->second.end()) {
        return std::nullopt;
    }
    const auto& texture = found->second.texture;
    const auto& desc = texture.description;
    const auto& view = found->second.view;
    const u32 format = view ? ViewFormat(*view, desc.format) : desc.format;
    const u32 level = view && ((*view)[7] & 1) ? (*view)[4] : 0;
    const bool layer_view = view && ((*view)[7] & 2);
    const u32 first = layer_view ? u32{(*view)[0]} | (u32{(*view)[1]} << 8) : 0;
    const u32 layers = layer_view ? u32{(*view)[2]} | (u32{(*view)[3]} << 8)
                                  : texture.layout.layers;
    const bool array = binding.type == Shader::TextureType::ColorArray2D;
    const auto color = FindColorFormat(format);
    if (!color || !HasCompatibleFormat(format, desc.format) ||
        !device.IsFormatSupported(color->vk, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT,
                                  FormatType::Optimal) ||
        level >= desc.levels || first >= texture.layout.layers || !layers ||
        layers > texture.layout.layers - first || (!array && layers != 1) ||
        binding.format != Shader::ImageFormat::Typeless ||
        binding.is_integer != VideoCore::Surface::IsPixelFormatInteger(color->surface)) {
        return std::nullopt;
    }
    image = FindImage(texture, true);
    if (!image || !image->storage_image ||
        (binding.is_read && !PrepareSampledImage(*image, texture, submission))) {
        return std::nullopt;
    }
    Initialize(*image);
    const std::array<u32, 4> identity{};
    const auto view_type = array ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    const auto key = std::tuple{color->vk, level, 1U, first, layers, view_type,
                                VkImageAspectFlags{VK_IMAGE_ASPECT_COLOR_BIT}, identity};
    auto target = image->sampled_views.find(key);
    if (target == image->sampled_views.end()) {
        const VkImageViewUsageCreateInfo usage{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT,
        };
        auto host_view = device.GetLogical().CreateImageView({
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = &usage,
            .image = *image->image,
            .viewType = array ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
            .format = color->vk,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level, 1, first, layers},
        });
        target = image->sampled_views.emplace(key, std::move(host_view)).first;
    }
    TraceImage(binding.is_written ? "compute write" : "compute read", texture, compute_program->serial);
    return VkDescriptorImageInfo{VK_NULL_HANDLE, *target->second, VK_IMAGE_LAYOUT_GENERAL};
}

void BananzaNvn2Replay::Dispatch(const Operation& op, const Submission& submission) {
    if (!compute_program) {
        return;
    }
    const auto& program = *compute_program;
    const auto skip = [&](std::string_view reason) {
        ++dispatch_skips[reason];
        if (dispatch_diagnostics.emplace(program.serial, reason).second) {
            LOG_WARNING(Render_Vulkan, "Bananza NVN 2 dispatch skipped (program {}): {}",
                        program.serial, reason);
        }
    };
    if (op.args.size() != 3) {
        skip("indirect dispatch");
        return;
    }
    const auto& limits = device.GetPhysical().GetProperties().limits;
    std::array<u32, 3> groups{};
    for (size_t i = 0; i < groups.size(); ++i) {
        if (!op.args[i] || op.args[i] > limits.maxComputeWorkGroupCount[i]) {
            return;
        }
        groups[i] = static_cast<u32>(op.args[i]);
    }
    const auto shader = std::ranges::find(program.shaders, 5U,
                                          &Core::BananzaNvn2Hle::ResourceShaderInfo::stage);
    if (shader == program.shaders.end() || shader->control.size() < 0xb7) {
        skip("compute program control missing");
        return;
    }
    constexpr u32 stage = 5;
    ComputeLayout layout{};
    const auto& control = shader->control;
    std::ranges::copy(std::span{control}.subspan(0xa9, layout.roots.size()), layout.roots.begin());
    layout.uniforms = control[0xb1];
    layout.storage = control[0xb2];
    layout.textures = control[0xb3];
    layout.images = control[0xb6];
    if (layout.uniforms > uniform_buffers[stage].size() ||
        layout.storage > storage_buffers[stage].size() ||
        layout.textures > texture_handles[stage].size() ||
        layout.images > image_handles[stage].size() ||
        std::ranges::any_of(layout.roots, [](u32 words) { return words > 64; })) {
        skip("compute parameter block exceeds bound state");
        return;
    }
    // NVN 2 computes this parameter block layout from the shader control (DKB Recomp #20).
    std::array<u8, 4096> driver{};
    u32 at = 16;
    for (size_t i = 0; i < layout.roots.size(); ++i) {
        const u32 bytes = layout.roots[i] * sizeof(u32);
        std::memcpy(driver.data() + at, compute_constants[i].data(), bytes);
        at += (bytes + 15) & ~15U;
    }
    const auto pack_buffers = [&](const auto& ranges, u32 count) {
        const u32 base = at;
        for (const auto& [address, size] : std::span{ranges}.first(count)) {
            const u64_le gpu = address;
            const u32_le bytes = static_cast<u32>(size);
            std::memcpy(driver.data() + at, &gpu, sizeof(gpu));
            std::memcpy(driver.data() + at + 8, &bytes, sizeof(bytes));
            at += 16;
        }
        return base;
    };
    const auto pack_handles = [&](const auto& handles, u32 count) {
        const u32 base = at;
        for (const u64_le handle : std::span{handles}.first(count)) {
            std::memcpy(driver.data() + at, &handle, sizeof(handle));
            at += 8;
        }
        return base;
    };
    layout.uniform_base = pack_buffers(uniform_buffers[stage], layout.uniforms);
    layout.storage_base = pack_buffers(storage_buffers[stage], layout.storage);
    layout.texture_base = pack_handles(texture_handles[stage], layout.textures);
    layout.image_base = pack_handles(image_handles[stage], layout.images);
    layout.size = at;
    ComputeEnvironment environment{*shader, submission, texture_pool, layout,
                                   std::span{driver}.first(at), uniform_buffers[stage],
                                   storage_buffers[stage]};
    auto& variants = compute_pipelines[program.serial];
    auto cached = std::ranges::find_if(variants, [&](const ComputePipeline& pipeline) {
        environment.texture_bank = pipeline.texture_bank;
        return environment.Matches(pipeline.queries);
    });
    if (cached == variants.end()) {
        constexpr size_t MaxVariants = 32;
        if (variants.size() >= MaxVariants) {
            skip("compute translation variant limit");
            return;
        }
        std::optional<ComputePipeline> compiled;
        environment.missing_texture = false;
        try {
            compiled = CreateComputePipeline(environment);
        } catch (const vk::Exception& e) {
            if (e.GetResult() == VK_ERROR_DEVICE_LOST) {
                throw;
            }
            LOG_WARNING(Render_Vulkan, "Bananza NVN 2 compute pipeline {}: {}", program.serial, e.what());
        } catch (const std::invalid_argument& e) {
            LOG_WARNING(Render_Vulkan, "Bananza NVN 2 compute pipeline {}: {}", program.serial, e.what());
        }
        if (!compiled) {
            skip("compute pipeline unavailable");
            return;
        }
        variants.push_back(std::move(*compiled));
        cached = std::prev(variants.end());
    }
    const auto& pipeline = *cached;
    std::vector<VkDescriptorBufferInfo> uniforms;
    std::vector<VkDescriptorBufferInfo> storage;
    std::vector<VkDescriptorImageInfo> textures;
    std::vector<VkDescriptorImageInfo> images_info;
    std::vector<Image*> writes_to_images;
    std::vector<std::pair<StorageUpload*, u64>> writes_to_buffers;
    for (const auto& binding : pipeline.state.uniforms) {
        const auto bytes = environment.Bank(binding.bank);
        if (bytes.empty() || bytes.size() > limits.maxUniformBufferRange) {
            if (dispatch_diagnostics.emplace(program.serial, "uniform binding detail").second) {
                LOG_WARNING(Render_Vulkan, "NVN 2 compute program {} uniform bank {}: {} bytes bound, {} bytes addressable",
                            program.serial, binding.bank, bytes.size(), binding.size);
            }
            skip("compute uniform range missing or too large");
            return;
        }
        const auto buffer = binding.bank >= 3
                                ? UploadRange(submission, uniform_buffers[stage][binding.bank - 3])
                                : Upload(bytes);
        if (!buffer) {
            return;
        }
        uniforms.push_back({buffer, 0, bytes.size()});
    }
    for (const auto& [index, descriptor] : pipeline.storage) {
        auto range = environment.StorageRange(descriptor.cbuf_index, descriptor.cbuf_offset);
        if (range.first && !range.second) {
            const auto size = submission.unsized_storage_sizes.find(range.first);
            if (size != submission.unsized_storage_sizes.end()) {
                range.second = size->second;
            }
        }
        if (!range.first && !range.second) {
            if (!unbound_storage) {
                unbound_storage = allocator.CreateBuffer({
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = sizeof(u32),
                    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                }, MemoryUsage::Upload);
                std::ranges::fill(unbound_storage.Mapped(), 0);
                unbound_storage.Flush();
            }
            // A zero-length storage slot cannot contain any shader elements.
            storage.push_back({*unbound_storage, 0, 1});
            continue;
        }
        if (!range.second || range.second > limits.maxStorageBufferRange) {
            if (dispatch_diagnostics.emplace(program.serial, "storage range detail").second) {
                LOG_WARNING(Render_Vulkan, "NVN 2 compute program {} storage bank {} offset {:#x} at {:#x}, {} bytes",
                            program.serial, descriptor.cbuf_index, descriptor.cbuf_offset,
                            range.first, range.second);
            }
            skip("compute storage buffer range invalid");
            return;
        }
        VkBuffer buffer{};
        u64 offset{};
        if (descriptor.is_written) {
            if (const auto [upload, start] = WritableStorage(submission, range); upload) {
                buffer = *upload->buffer;
                offset = start;
                const u64 end = Shader::ResolveStorageWriteExtent(descriptor.write_extent,
                                                                  pipeline.workgroup, groups)
                                    .value_or(range.second);
                writes_to_buffers.emplace_back(upload, offset + std::min(end, range.second));
            }
        } else {
            std::tie(buffer, offset) = UploadBufferRange(submission, range);
        }
        if (!buffer || offset % limits.minStorageBufferOffsetAlignment) {
            if (dispatch_diagnostics.emplace(program.serial, "missing storage backing").second) {
                LOG_WARNING(Render_Vulkan,
                            "NVN 2 compute program {} storage at {:#x}, {} bytes, write {}, snapshot {}, guest mapping {}",
                            program.serial, range.first, range.second, descriptor.is_written,
                            submission.buffer_snapshots.contains(range),
                            submission.storage_addresses.contains(range));
            }
            skip("compute storage buffer range missing");
            return;
        }
        storage.push_back({buffer, offset, range.second});
    }
    for (const auto& [slot, index, descriptor] : pipeline.state.textures) {
        const auto handle = environment.ReadCbufValue(descriptor.cbuf_index, descriptor.cbuf_offset);
        const auto info = BindTexture(submission, slot, descriptor, {}, handle);
        if (!info) {
            if (dispatch_diagnostics.emplace(program.serial, "sample binding detail").second) {
                LOG_WARNING(Render_Vulkan, "NVN 2 compute program {} sampled bank {} offset {:#x} handle {:#x}",
                            program.serial, descriptor.cbuf_index, descriptor.cbuf_offset, handle);
            }
            skip("compute sampled texture unavailable");
            return;
        }
        textures.push_back(*info);
    }
    for (const auto& [index, descriptor] : pipeline.images) {
        const auto handle = environment.ReadCbufValue(descriptor.cbuf_index, descriptor.cbuf_offset);
        Image* image{};
        const auto info = BindStorageImage(submission, handle, descriptor, image);
        if (!info) {
            skip("compute storage image unavailable");
            return;
        }
        images_info.push_back(*info);
        if (descriptor.is_written) {
            writes_to_images.push_back(image);
        }
    }
    VkDescriptorSet set{};
    if (pipeline.state.set_layout) {
        const auto& logical = device.GetLogical();
        std::vector<VkDescriptorPoolSize> sizes;
        const auto reserve = [&](size_t count, VkDescriptorType type) {
            if (count) {
                sizes.push_back({type, static_cast<u32>(count)});
            }
        };
        reserve(uniforms.size(), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        reserve(storage.size(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        reserve(textures.size(), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        reserve(images_info.size(), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        auto pool = logical.CreateDescriptorPool({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = static_cast<u32>(sizes.size()),
            .pPoolSizes = sizes.data(),
        });
        const VkDescriptorSetLayout set_layout = *pipeline.state.set_layout;
        set = pool.Allocate({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = *pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &set_layout,
        })[0];
        std::vector<VkWriteDescriptorSet> writes;
        const auto write_buffers = [&](const auto& infos, VkDescriptorType type) {
            for (const auto& info : infos) {
                writes.push_back({.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                  .dstSet = set, .dstBinding = static_cast<u32>(writes.size()),
                                  .descriptorCount = 1, .descriptorType = type, .pBufferInfo = &info});
            }
        };
        const auto write_images = [&](const auto& infos, VkDescriptorType type) {
            for (const auto& info : infos) {
                writes.push_back({.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                  .dstSet = set, .dstBinding = static_cast<u32>(writes.size()),
                                  .descriptorCount = 1, .descriptorType = type, .pImageInfo = &info});
            }
        };
        write_buffers(uniforms, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        write_buffers(storage, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        write_images(textures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        write_images(images_info, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        logical.UpdateDescriptorSets(writes, {});
        descriptor_pools.push_back(std::move(pool));
    }
    Barrier();
    scheduler.Record([groups, handle = *pipeline.state.pipeline, pipeline_layout = *pipeline.state.layout,
                      set](vk::CommandBuffer cmdbuf) {
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, handle);
        if (set) {
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, {set}, {});
        }
        constexpr Shader::Backend::SPIRV::RescalingLayout scale{.down_factor = 1.0f};
        cmdbuf.PushConstants(pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, scale);
        cmdbuf.Dispatch(groups[0], groups[1], groups[2]);
    });
    for (auto* image : writes_to_images) {
        image->gpu_written = true;
        ++image->write_version;
    }
    for (const auto& [upload, bytes] : writes_to_buffers) {
        upload->written_bytes = std::max(upload->written_bytes, bytes);
    }
    ++dispatches;
}

} // namespace Vulkan
