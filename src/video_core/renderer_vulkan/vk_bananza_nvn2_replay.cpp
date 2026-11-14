// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>

#include "common/logging.h"
#include "common/swap.h"
#include "core/arm/bananza_nvn2_commands.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
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
    case 0x05:
        return ColorFormat{VK_FORMAT_R16_SFLOAT, PixelFormat::R16_FLOAT, 2};
    case 0x06:
        return ColorFormat{VK_FORMAT_R16_UNORM, PixelFormat::R16_UNORM, 2};
    case 0x0a:
        return ColorFormat{VK_FORMAT_R32_SFLOAT, PixelFormat::R32_FLOAT, 4};
    case 0x0d:
        return ColorFormat{VK_FORMAT_R8G8_UNORM, PixelFormat::R8G8_UNORM, 2};
    case 0x11:
        return ColorFormat{VK_FORMAT_R16G16_SFLOAT, PixelFormat::R16G16_FLOAT, 4};
    case 0x12:
        return ColorFormat{VK_FORMAT_R16G16_UNORM, PixelFormat::R16G16_UNORM, 4};
    case 0x16:
        return ColorFormat{VK_FORMAT_R32G32_SFLOAT, PixelFormat::R32G32_FLOAT, 8};
    case 0x25:
        return ColorFormat{VK_FORMAT_R8G8B8A8_UNORM, PixelFormat::A8B8G8R8_UNORM, 4};
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
    case 0x2e:
        return ColorFormat{VK_FORMAT_R32G32B32A32_SFLOAT, PixelFormat::R32G32B32A32_FLOAT, 16};
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

bool IsBaseView(std::span<const u8> data, const Core::BananzaNvn2Hle::TextureDescription& d,
                bool allow_format_view = false) {
    if (data.size() != 16) {
        return false;
    }
    const u8 flags = data[7];
    const u32 format = ViewFormat(data, d.format);
    return (!(flags & 1) || data[4] == 0) &&
           (!(flags & 2) || (data[0] == 0 && data[1] == 0)) &&
           (format == d.format || (allow_format_view && HasCompatibleFormat(format, d.format))) &&
           (!(flags & 8) || std::ranges::equal(data.subspan(12, 4), d.swizzle)) &&
           (!(flags & 0x10) || (flags >> 6) == d.depth_mode) &&
           (!(flags & 0x20) || data[6] == d.target);
}

struct CopyRegion {
    u32 x;
    u32 y;
    u32 z;
    u32 width;
    u32 height;
    u32 depth;
};

bool Fits(CopyRegion region, VkExtent2D extent) {
    return !region.z && region.depth == 1 && region.width && region.height &&
           region.x <= extent.width && region.width <= extent.width - region.x &&
           region.y <= extent.height && region.height <= extent.height - region.y;
}

VkFormat VertexFormat(u32 format) {
    switch (format) {
    case 0x01: return VK_FORMAT_R8_UNORM;
    case 0x0a: return VK_FORMAT_R32_SFLOAT;
    case 0x0d: return VK_FORMAT_R8G8_UNORM;
    case 0x11: return VK_FORMAT_R16G16_SFLOAT;
    case 0x16: return VK_FORMAT_R32G32_SFLOAT;
    case 0x19: return VK_FORMAT_R8G8B8_UNORM;
    case 0x1d: return VK_FORMAT_R16G16B16_SFLOAT;
    case 0x22: return VK_FORMAT_R32G32B32_SFLOAT;
    case 0x25: return VK_FORMAT_R8G8B8A8_UNORM;
    case 0x29: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case 0x2e: return VK_FORMAT_R32G32B32A32_SFLOAT;
    default: return VK_FORMAT_UNDEFINED;
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
    if (!texture.serial || !texture.storage.gpu_address || (d.target != 1 && d.target != 4) ||
        !layers || layers > device.GetPhysical().GetProperties().limits.maxImageArrayLayers ||
        d.samples > 1 ||
        !d.width || !d.height || d.width > 4096 || d.height > 4096 || !d.levels ||
        d.levels > static_cast<u32>(std::bit_width(std::max(d.width, d.height)))) {
        return nullptr;
    }
    const ImageKey key{texture.storage.gpu_address + texture.offset, d.width, d.height, d.format,
                       d.levels, d.target, layers};
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
    const u64 bytes = std::max<u64>(texture.layout.size,
                                    u64{d.width} * d.height * format->bytes * layers);
    constexpr u64 MaxImageBytes = 384ULL << 20;
    if (bytes > MaxImageBytes - image_bytes) {
        return nullptr;
    }
    const VkExtent2D extent{d.width, d.height};
    const bool mutable_format = d.format == 0x25 || d.format == 0x38 ||
                                d.format == 0x9a || d.format == 0x9c;
    auto image = allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .flags = mutable_format ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0U,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format->vk,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = d.levels,
        .arrayLayers = layers,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                 VK_IMAGE_USAGE_SAMPLED_BIT | (attachment ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0U) |
                 (depth_stencil ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : 0U),
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    auto view = device.GetLogical().CreateImageView({
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = *image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format->vk,
        .subresourceRange = {format->aspects, 0, 1, 0, 1},
    });
    image_bytes += bytes;
    return &images.emplace(key, Image{std::move(image), std::move(view), extent, format->vk,
                                      format->surface, d.levels, layers, attachment,
                                      format->aspects})
                .first->second;
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

void BananzaNvn2Replay::ClearColor(const Operation& op) {
    if (op.args.size() != 2 || op.payload.size() != 16 || op.args[0] >= colors.size() ||
        op.args[1] != 0xf || !colors[op.args[0]] || !base_color_views[op.args[0]] ||
        op.replay != ReplayOp::ClearColor) {
        return;
    }
    auto* image = FindImage(*colors[op.args[0]], true);
    if (!image || !image->color_attachment) {
        return;
    }
    Initialize(*image);
    Barrier();
    VkClearColorValue value{};
    std::memcpy(value.float32, op.payload.data(), op.payload.size());
    TraceImage("clear", *colors[op.args[0]]);
    scheduler.Record([handle = *image->image, value](vk::CommandBuffer cmdbuf) {
        cmdbuf.ClearColorImage(handle, VK_IMAGE_LAYOUT_GENERAL, value, {BaseColorRange});
    });
    ++clears;
}

void BananzaNvn2Replay::ClearDepthStencil(const Operation& op) {
    if (!depth_texture || op.args.size() != 3 || op.payload.size() != sizeof(float)) {
        return;
    }
    auto* image = FindImage(*depth_texture, true);
    if (!image || image->aspects == VK_IMAGE_ASPECT_COLOR_BIT) {
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
    VkRect2D rect{{}, image->extent};
    if (has_scissor) {
        const s64 left = std::clamp<s64>(scissor.offset.x, 0, rect.extent.width);
        const s64 top = std::clamp<s64>(scissor.offset.y, 0, rect.extent.height);
        const s64 right = std::clamp<s64>(s64{scissor.offset.x} + scissor.extent.width,
                                          left, rect.extent.width);
        const s64 bottom = std::clamp<s64>(s64{scissor.offset.y} + scissor.extent.height,
                                           top, rect.extent.height);
        rect = {{static_cast<s32>(left), static_cast<s32>(top)},
                {static_cast<u32>(right - left), static_cast<u32>(bottom - top)}};
    }
    if (!rect.extent.width || !rect.extent.height) {
        return;
    }
    if (origin_lower_left) {
        rect.offset.y = static_cast<s32>(image->extent.height) - rect.offset.y -
                        static_cast<s32>(rect.extent.height);
    }
    const bool initialized = image->initialized;
    Initialize(*image);
    Barrier();
    scheduler.Record([view = *image->view, rect, aspects, image_aspects = image->aspects, value,
                      initialized, extent = image->extent](vk::CommandBuffer cmdbuf) {
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
            .layerCount = 1,
            .pDepthAttachment = image_aspects & VK_IMAGE_ASPECT_DEPTH_BIT ? &target : nullptr,
            .pStencilAttachment = image_aspects & VK_IMAGE_ASPECT_STENCIL_BIT ? &target : nullptr,
        });
        const VkClearAttachment clear{.aspectMask = aspects, .clearValue = {.depthStencil = value}};
        const VkClearRect clear_rect{.rect = rect, .layerCount = 1};
        cmdbuf.ClearAttachments({clear}, {clear_rect});
        cmdbuf.EndRendering();
    });
    ++clears;
}

void BananzaNvn2Replay::CopyTexture(const Operation& op) {
    if (op.textures.size() != 2 || op.args.size() != 5 ||
        op.payload.size() != 32 + 2 * sizeof(CopyRegion) ||
        (op.args[2] & ~3ULL) ||
        (op.args[3] && !IsBaseView(std::span{op.payload}.first(16), op.textures[0].description)) ||
        (op.args[4] && !IsBaseView(std::span{op.payload}.subspan(16, 16), op.textures[1].description))) {
        return;
    }
    const auto source_address = op.textures[0].storage.gpu_address + op.textures[0].offset;
    TraceImage("copy request", op.textures[1], source_address);
    auto* source = FindImage(op.textures[0], false);
    if (!source || !source->initialized || source->aspects != VK_IMAGE_ASPECT_COLOR_BIT) {
        TraceImage("copy source unavailable", op.textures[1], source_address);
        return;
    }
    auto* dest = FindImage(op.textures[1], true);
    if (!dest || dest == source || dest->aspects != VK_IMAGE_ASPECT_COLOR_BIT) {
        return;
    }
    CopyRegion from{}, to{};
    std::memcpy(&from, op.payload.data() + 32, sizeof(from));
    std::memcpy(&to, op.payload.data() + 32 + sizeof(from), sizeof(to));
    if (!Fits(from, source->extent) || !Fits(to, dest->extent)) {
        return;
    }
    const bool blit = source->format != dest->format || from.width != to.width ||
                      from.height != to.height;
    const auto filter = op.args[2] & 1 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    if (blit) {
        const VkFormatFeatureFlags source_features =
            VK_FORMAT_FEATURE_BLIT_SRC_BIT |
            (filter == VK_FILTER_LINEAR ? VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT : 0);
        if (!device.IsFormatSupported(source->format, source_features, FormatType::Optimal) ||
            !device.IsFormatSupported(dest->format, VK_FORMAT_FEATURE_BLIT_DST_BIT,
                                      FormatType::Optimal)) {
            return;
        }
    }
    Initialize(*dest);
    Barrier();
    TraceImage("copy", op.textures[1], source_address);
    if (blit) {
        const VkImageBlit region{
            .srcSubresource = BaseColorLayer,
            .srcOffsets = {{static_cast<s32>(from.x), static_cast<s32>(from.y), 0},
                           {static_cast<s32>(from.x + from.width),
                            static_cast<s32>(from.y + from.height), 1}},
            .dstSubresource = BaseColorLayer,
            .dstOffsets = {{static_cast<s32>(to.x), static_cast<s32>(to.y), 0},
                           {static_cast<s32>(to.x + to.width),
                            static_cast<s32>(to.y + to.height), 1}},
        };
        scheduler.Record([src = *source->image, dst = *dest->image, region, filter](
                             vk::CommandBuffer cmdbuf) {
            cmdbuf.BlitImage(src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL,
                             {region}, filter);
        });
        ++copies;
        return;
    }
    const VkImageCopy copy{
        .srcSubresource = BaseColorLayer,
        .srcOffset = {static_cast<s32>(from.x), static_cast<s32>(from.y), 0},
        .dstSubresource = BaseColorLayer,
        .dstOffset = {static_cast<s32>(to.x), static_cast<s32>(to.y), 0},
        .extent = {from.width, from.height, 1},
    };
    scheduler.Record([src = *source->image, dst = *dest->image, copy](vk::CommandBuffer cmdbuf) {
        cmdbuf.CopyImage(src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, {copy});
    });
    ++copies;
}

VkBuffer BananzaNvn2Replay::Upload(std::span<const u8> bytes) {
    if (bytes.empty()) {
        return VK_NULL_HANDLE;
    }
    auto buffer = allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = bytes.size(),
            .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
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
    };
    std::vector<Level> levels;
    std::vector<VkBufferImageCopy> copies;
    const bool decode_astc = d.format >= 0x79 && d.format <= 0x94;
    size_t upload_size{};
    for (u32 level = 0; level < d.levels; ++level) {
        const auto offset = Core::BananzaNvn2Hle::TextureLevelOffset(d, level);
        const u32 width = std::max(d.width >> level, 1U);
        const u32 height = std::max(d.height >> level, 1U);
        const u32 columns = (width + block->width - 1) / block->width;
        const u32 rows = (height + block->height - 1) / block->height;
        u32 tile = texture.layout.tile_height;
        while (tile && (8U << (tile - 1)) >= rows) {
            --tile;
        }
        const size_t bytes = texture.layout.stride
                                 ? texture.layout.stride * (rows - 1) + columns * block->bytes
                                 : Tegra::Texture::CalculateSize(true, block->bytes, columns,
                                                                  rows, 1, tile, 0);
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
            .imageExtent = {width, height, 1},
        });
        const auto upload_stride = decode_astc ? size_t{width} * height * 4
                                               : size_t{columns} * rows * block->bytes;
        levels.push_back({*offset, texture.layout.layer_size, upload_stride, columns, rows, tile});
        upload_size += upload_stride * image.layers;
    }
    auto upload = CreateWrappedBuffer(allocator, upload_size, MemoryUsage::Upload);
    for (u32 level = 0; level < levels.size(); ++level) {
        const auto& src = levels[level];
        const size_t row = size_t{src.width} * block->bytes;
        const auto extent = copies[level].imageExtent;
        std::vector<u8> astc(decode_astc ? row * src.height : 0);
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
                                                src.width, src.height, 1, src.tile_height, 0, 0, true);
            }
            if (decode_astc) {
                Tegra::Texture::ASTC::Decompress(astc, extent.width, extent.height, 1, block->width,
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
    return true;
}

std::optional<VkDescriptorImageInfo> BananzaNvn2Replay::BindTexture(
    const Submission& submission, u32 stage, const Shader::TextureDescriptor& binding,
    std::span<Image* const> destinations) {
    const auto& program = graphics_program->shaders;
    const auto staged = std::ranges::find(program, stage == 0 ? 0U : 4U,
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
    if (binding.cbuf_index == (stage == 0 ? 0U : 2U) && offset >= HandleBase &&
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
    const auto sampler_table = submission.sampler_descriptors.find(sampler_pool);
    if (textures == submission.texture_descriptors.end() ||
        sampler_table == submission.sampler_descriptors.end()) {
        NoteSkip("missing texture or sampler pool snapshot");
        return std::nullopt;
    }
    const auto texture = textures->second.find(texture_id);
    const auto sampler = sampler_table->second.find(sampler_id);
    if (texture == textures->second.end() || sampler == sampler_table->second.end()) {
        NoteSkip("missing texture or sampler descriptor");
        return std::nullopt;
    }
    const auto& tex = texture->second.texture;
    const auto& desc = sampler->second;
    const auto& d = tex.description;
    u32 format = d.format;
    u32 first_level{};
    u32 levels = d.levels;
    u32 first_layer{};
    u32 layers = tex.layout.layers;
    u32 target = d.target;
    auto swizzle = d.swizzle;
    if (const auto& v = texture->second.view) {
        const u8 flags = (*v)[7];
        format = ViewFormat(*v, d.format);
        if ((flags & 0x10 && (flags >> 6) != d.depth_mode) ||
            !HasCompatibleFormat(format, d.format)) {
            NoteSkip("unsupported sampled texture view");
            return std::nullopt;
        }
        if (flags & 1) {
            first_level = (*v)[4];
            levels = (*v)[5];
        }
        if (flags & 2) {
            first_layer = u32{(*v)[0]} | (u32{(*v)[1]} << 8);
            layers = u32{(*v)[2]} | (u32{(*v)[3]} << 8);
        }
        if (flags & 0x20) {
            target = (*v)[6];
        }
        if (flags & 8) {
            std::ranges::copy(std::span{*v}.subspan(12, 4), swizzle.begin());
        }
    }
    constexpr std::array components{VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ONE,
                                    VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
                                    VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
    const bool array = binding.type == Shader::TextureType::ColorArray2D;
    if ((target != 1 && target != 4) || (array ? target != 4 : layers != 1) ||
        first_layer >= tex.layout.layers || !layers || layers > tex.layout.layers - first_layer ||
        first_level >= d.levels || !levels || levels > d.levels - first_level ||
        std::ranges::any_of(swizzle, [&](u32 s) { return s >= components.size(); })) {
        NoteSkip("unsupported sampled texture view");
        return std::nullopt;
    }
    if (desc.compare_mode || desc.reduction) {
        NoteSkip("unsupported sampler comparison or reduction");
        return std::nullopt;
    }
    auto* image = FindImage(tex, true);
    if (!image || image->aspects != VK_IMAGE_ASPECT_COLOR_BIT ||
        std::ranges::find(destinations, image) != destinations.end()) {
        NoteSkip(image ? "sampled texture aliases destination" : "unsupported sampled texture format");
        return std::nullopt;
    }
    if (!image->initialized) {
        const auto data = submission.texture_snapshots.find(tex.serial);
        if (data == submission.texture_snapshots.end()) {
            TraceImage("missing sampled snapshot", tex, graphics_program->serial);
            if (trace_images && draw_diagnostics.emplace(graphics_program->serial,
                                                          "sample snapshot detail").second) {
                LOG_INFO(Render_Vulkan,
                         "NVN 2 sample program {}, bank {}, offset {:#x}, handle {:#x}, format {:#x}, flags {:#x}, address {:#x}, {}x{}, {} bytes, {} levels",
                         graphics_program->serial, binding.cbuf_index, offset, handle, d.format,
                         d.flags, tex.storage.gpu_address + tex.offset, d.width, d.height,
                         tex.layout.size, d.levels);
            }
            NoteSkip("sampled texture has no replayed write or CPU snapshot");
            return std::nullopt;
        }
        if (!UploadTexture(*image, tex, data->second)) {
            NoteSkip("sampled texture snapshot does not cover its mip levels");
            return std::nullopt;
        }
        ++texture_uploads;
    }
    TraceImage("sampled", tex, graphics_program->serial);
    const auto address = tex.storage.gpu_address + tex.offset;
    if (frame_dump.CapturingSampled() && d.width >= 900 && first_level == 0 &&
        captured_inputs.insert(address).second) {
        frame_dump.CaptureNvn2Input(*image->image, image->surface_format, d.width, d.height,
                                    address);
    }
    const auto vk_format = FindColorFormat(format)->vk;
    const auto key = std::tuple{vk_format, first_level, levels, first_layer,
                                array ? layers : 0U, swizzle};
    auto sampled_view = image->sampled_views.find(key);
    if (sampled_view == image->sampled_views.end()) {
        auto view = device.GetLogical().CreateImageView({
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = *image->image,
            .viewType = array ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
            .format = vk_format,
            .components = {components[swizzle[0]], components[swizzle[1]],
                           components[swizzle[2]], components[swizzle[3]]},
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, first_level, levels, first_layer, layers},
        });
        sampled_view = image->sampled_views.emplace(key, std::move(view)).first;
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
    auto host_sampler = device.GetLogical().CreateSampler({
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = desc.mag_filter == 1 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .minFilter = desc.min_filter & 1 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .mipmapMode = desc.min_filter >= 4 ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                          : VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = modes[desc.wrap[0]],
        .addressModeV = modes[desc.wrap[1]],
        .addressModeW = modes[desc.wrap[2]],
        .mipLodBias = std::clamp(desc.lod_bias, -device.GetPhysical().GetProperties().limits.maxSamplerLodBias,
                                device.GetPhysical().GetProperties().limits.maxSamplerLodBias),
        .minLod = desc.min_filter < 2 ? 0.0f : std::clamp(desc.min_lod, 0.0f, float(levels - 1)),
        .maxLod = desc.min_filter < 2 ? 0.0f : std::clamp(desc.max_lod, 0.0f, float(levels - 1)),
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
    });
    const VkSampler handle_sampler = *host_sampler;
    samplers.push_back(std::move(host_sampler));
    return VkDescriptorImageInfo{handle_sampler, *sampled_view->second, VK_IMAGE_LAYOUT_GENERAL};
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
    const auto fragment = std::ranges::find(shaders, 4U,
                                            &Core::BananzaNvn2Hle::ResourceShaderInfo::stage);
    if (shaders.size() != 2 || vertex == shaders.end() || fragment == shaders.end() ||
        !device.IsDynamicRenderingSupported()) {
        return std::nullopt;
    }
    Shader::RuntimeInfo runtime;
    runtime.generic_input_types.fill(Shader::AttributeType::Float);
    runtime.previous_stage_stores.mask.set();
    runtime.convert_depth_mode = raster.negative_one_to_one;
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
    const u32 fragment_binding = bindings.unified;
    runtime.previous_stage_stores = vs.info->stores;
    const auto fs = TranslateBananzaNvn2Shader(*fragment, profile, host_info, runtime, bindings);
    if (fs.status != BananzaNvn2ShaderStatus::Success) {
        LOG_WARNING(Render_Vulkan, "Bananza NVN 2 fragment shader at {:#x}: {}",
                    fragment->data_address, fs.error);
        return std::nullopt;
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
        save("fs", fs, *fragment);
    }
    const auto can_bind = [](const BananzaNvn2ShaderResult& shader,
                             const Core::BananzaNvn2Hle::ResourceShaderInfo& staged) {
        const auto& info = *shader.info;
        const auto& queries = shader.queries;
        return info.storage_buffers_descriptors.empty() &&
               info.image_descriptors.empty() && info.texture_buffer_descriptors.empty() &&
               info.image_buffer_descriptors.empty() && info.handle_programs.empty() &&
               !info.uses_global_memory && !info.stores_global_memory &&
               queries.storage_pointer_types.empty() && queries.cbuf_replacements.empty() &&
               std::ranges::all_of(info.texture_descriptors, [](const auto& descriptor) {
                   return (descriptor.type == Shader::TextureType::Color2D ||
                           descriptor.type == Shader::TextureType::ColorArray2D) &&
                          !descriptor.is_depth &&
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
    if (!can_bind(vs, *vertex) || !can_bind(fs, *fragment)) {
        NoteSkip("translation needs unsupported descriptors or constant values");
        return std::nullopt;
    }
    std::vector<VkVertexInputAttributeDescription> attributes;
    std::vector<VkVertexInputBindingDescription> streams;
    Pipeline result;
    result.uses_render_area = vs.info->uses_render_area || fs.info->uses_render_area;
    for (u32 i = 0; i < vertex_attributes.size(); ++i) {
        if (!vs.info->loads.Generic(i)) {
            continue;
        }
        const u32 attribute = vertex_attributes[i];
        const u32 stream = attribute & 31;
        const auto vf = VertexFormat((attribute >> 21) & 0x3ff);
        if (!(attribute & 0x40) || !vertex_strides[stream] || vertex_divisors[stream] > 1 ||
            vf == VK_FORMAT_UNDEFINED ||
            !device.IsFormatSupported(vf, VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT, FormatType::Buffer)) {
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
        for (const auto& descriptor : info.texture_descriptors) {
            layout_bindings.push_back({binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, flags});
            result.textures.emplace_back(stage, binding++, descriptor);
        }
        return true;
    };
    if (!describe(*vs.info, *vertex, 0, 0, VK_SHADER_STAGE_VERTEX_BIT) ||
        !describe(*fs.info, *fragment, 1, fragment_binding, VK_SHADER_STAGE_FRAGMENT_BIT) ||
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
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
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
    const auto fragment_module = make_shader(fs.spirv);
    const std::array stages{
        VkPipelineShaderStageCreateInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                        .stage = VK_SHADER_STAGE_VERTEX_BIT,
                                        .module = *vertex_module, .pName = "main"},
        VkPipelineShaderStageCreateInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                                        .module = *fragment_module, .pName = "main"},
    };
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
    // Vulkan's framebuffer-space winding is opposite NVN's clip-space winding.
    const bool clockwise = (raster.polygon & 4) != 0;
    const VkPipelineRasterizationStateCreateInfo rasterization{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = static_cast<VkPolygonMode>(2 - polygon),
        .cullMode = raster.polygon & 3,
        .frontFace = clockwise ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE,
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
        attachment.colorWriteMask = (color_state.channel_mask >> (slot * 4)) & 15;
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
        .stageCount = static_cast<u32>(stages.size()),
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
    if (!trace_images || d.width < 900 || traced_images.size() >= 300 ||
        !traced_images.emplace(operation, address, detail).second) {
        return;
    }
    LOG_INFO(Render_Vulkan, "NVN 2 image {} {:#x} {}x{} format {:#x}, detail {:#x}",
             operation, address, d.width, d.height, d.format, detail);
}

void BananzaNvn2Replay::Draw(const Operation& op, const Submission& submission) {
    if (graphics_program && colors[0]) {
        TraceImage("draw request", *colors[0], graphics_program->serial);
    }
    if (op.args.empty() || !graphics_program || !colors[0]) {
        return;
    }
    const auto topology = Topology(op.args[0]);
    auto draw = DecodeDraw(op);
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
    if (!draw->count || !draw->instances) {
        return;
    }
    ColorFormats formats{};
    std::array<Image*, 8> targets{};
    std::array<VkImageView, 8> views{};
    u32 color_count{};
    for (u32 slot = 0; slot < colors.size(); ++slot) {
        if (!colors[slot] || !colors[slot]->serial) {
            continue;
        }
        if (!base_color_views[slot]) {
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
        const VkFormat format = color_view_formats[slot];
        if (format == target->format) {
            views[slot] = *target->view;
        } else {
            auto view = target->attachment_views.find(format);
            if (view == target->attachment_views.end()) {
                if (!device.IsFormatSupported(format, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT,
                                              FormatType::Optimal)) {
                    NoteSkip("unsupported color attachment view format");
                    return;
                }
                view = target->attachment_views
                           .emplace(format, CreateWrappedImageView(device, target->image, format))
                           .first;
            }
            views[slot] = *view->second;
        }
        formats[slot] = format;
        targets[slot] = target;
        color_count = slot + 1;
    }
    const auto* image = targets[0];
    if (!image || !std::ranges::all_of(targets, [image](const auto* target) {
            return !target || (target->extent.width >= image->extent.width &&
                               target->extent.height >= image->extent.height);
        })) {
        NoteSkip("incompatible color attachment sizes");
        return;
    }
    auto* depth = depth_texture ? FindImage(*depth_texture, true) : nullptr;
    if (depth && (depth->aspects == VK_IMAGE_ASPECT_COLOR_BIT ||
                  depth->extent.width < image->extent.width ||
                  depth->extent.height < image->extent.height)) {
        depth = nullptr;
    }
    auto raster = raster_state;
    raster.depth_format = depth ? depth->format : VK_FORMAT_UNDEFINED;
    raster.depth_aspects = depth ? depth->aspects : 0;
    raster.negative_one_to_one = submission.depth_minus_one_to_one;
    raster.origin_lower_left = submission.origin_lower_left;
    const PipelineKey key{graphics_program->serial, formats, topology, vertex_attributes,
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
        index_buffer = UploadRange(submission, draw->index_range);
        if (!index_buffer) {
            NoteSkip("index buffer range missing at submit");
            return;
        }
    }
    VkDescriptorSet set{};
    if (!pipeline.uniforms.empty() || !pipeline.textures.empty()) {
        std::vector<VkDescriptorBufferInfo> infos;
        infos.reserve(pipeline.uniforms.size());
        for (const auto& binding : pipeline.uniforms) {
            const auto& bound = uniform_buffers[binding.stage];
            // NVN user uniform slots follow the driver's three internal banks.
            const auto range = binding.bank >= 3 ? bound[binding.bank - 3] : Range{};
            const auto literal = binding.bank == 17
                                     ? std::span<const u8>{graphics_constants[0]}
                                     : binding.bank >= 24
                                           ? std::span<const u8>{graphics_constants[binding.bank - 24]}
                                           : std::span<const u8>{binding.literal};
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
    const VkExtent2D extent = image->extent;
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
    VkRect2D draw_scissor{{}, extent};
    if (has_scissor) {
        const s64 left = std::clamp<s64>(scissor.offset.x, 0, extent.width);
        const s64 top = std::clamp<s64>(scissor.offset.y, 0, extent.height);
        const s64 right = std::clamp<s64>(s64{scissor.offset.x} + scissor.extent.width,
                                          left, extent.width);
        const s64 bottom = std::clamp<s64>(s64{scissor.offset.y} + scissor.extent.height,
                                           top, extent.height);
        draw_scissor = {{static_cast<s32>(left), static_cast<s32>(top)},
                        {static_cast<u32>(right - left), static_cast<u32>(bottom - top)}};
    }
    if (submission.origin_lower_left) {
        draw_viewport.y = static_cast<float>(extent.height) - draw_viewport.y;
        draw_viewport.height = -draw_viewport.height;
        draw_scissor.offset.y = static_cast<s32>(extent.height) - draw_scissor.offset.y -
                                static_cast<s32>(draw_scissor.extent.height);
    }
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
        }
    }
    VkRenderingAttachmentInfo depth_attachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = depth ? *depth->view : VK_NULL_HANDLE,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        .loadOp = depth && depth->initialized ? VK_ATTACHMENT_LOAD_OP_LOAD
                                              : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };
    if (depth) {
        Initialize(*depth);
    }
    Barrier();
    std::optional<u32> probe;
    if (frame_dump.CapturingSampled() && extent.width >= 1900) {
        const std::array shaders{graphics_program->serial};
        frame_dump.NoteDraw(shaders);
        probe = frame_dump.BeginProbe();
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
    scheduler.Record([attachments, color_count, depth_attachment, depth_aspects = raster.depth_aspects,
                      graphics = *pipeline.pipeline, layout = *pipeline.layout,
                      extent, draw_viewport, draw_scissor, command = *draw, set, index_buffer,
                      blend_color = blend_constants, uses_render_area = pipeline.uses_render_area,
                      refs = stencil_ref, compare_masks = stencil_compare_mask,
                      write_masks = stencil_write_mask,
                      vertices = std::move(vertices)](
                         vk::CommandBuffer cmdbuf) {
        cmdbuf.BeginRendering(VkRenderingInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = {{}, extent},
            .layerCount = 1,
            .colorAttachmentCount = color_count,
            .pColorAttachments = attachments.data(),
            .pDepthAttachment = depth_aspects & VK_IMAGE_ASPECT_DEPTH_BIT ? &depth_attachment : nullptr,
            .pStencilAttachment = depth_aspects & VK_IMAGE_ASPECT_STENCIL_BIT ? &depth_attachment : nullptr,
        });
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, graphics);
        cmdbuf.SetViewport(0, {draw_viewport});
        cmdbuf.SetScissor(0, {draw_scissor});
        cmdbuf.SetBlendConstants(blend_color.data());
        for (u32 i = 0; i < 2; ++i) {
            const VkStencilFaceFlags face = i == 0 ? VK_STENCIL_FACE_FRONT_BIT : VK_STENCIL_FACE_BACK_BIT;
            cmdbuf.SetStencilReference(face, refs[i]);
            cmdbuf.SetStencilCompareMask(face, compare_masks[i]);
            cmdbuf.SetStencilWriteMask(face, write_masks[i]);
        }
        using namespace Shader::Backend::SPIRV;
        constexpr auto stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
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
        if (index_buffer) {
            cmdbuf.BindIndexBuffer(index_buffer, 0, command.index_type);
            cmdbuf.DrawIndexed(command.count, command.instances, 0, command.base_vertex,
                               command.base_instance);
        } else {
            cmdbuf.Draw(command.count, command.instances, command.first, command.base_instance);
        }
        cmdbuf.EndRendering();
    });
    if (probe) {
        frame_dump.EndProbe(*probe, {.first = draw->first, .elements = draw->count,
                                     .instances = draw->instances,
                                     .color0 = (color_state.channel_mask & 15) != 0});
    }
    TraceImage("draw", *colors[0], graphics_program->serial);
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
            base_color_views.fill(false);
            depth_texture.reset();
            if (op.args.empty() || op.args[0] > colors.size()) {
                break;
            }
            const size_t count = op.args[0];
            if (op.textures.size() != count + 1 || op.args.size() != 2 * count + 3 ||
                op.payload.size() != 16 * (count + 1)) {
                break;
            }
            for (size_t i = 0; i < count; ++i) {
                colors[i] = op.textures[i];
                const auto& description = op.textures[i].description;
                const auto view = std::span{op.payload}.subspan(16 * i, 16);
                const bool has_view = op.args[2 + count + i] != 0;
                const auto format = FindColorFormat(has_view ? ViewFormat(view, description.format)
                                                              : description.format);
                base_color_views[i] = format && (!has_view || IsBaseView(view, description, true));
                color_view_formats[i] = format ? format->vk : VK_FORMAT_UNDEFINED;
            }
            if (const auto& texture = op.textures.back(); texture.serial) {
                const auto view = std::span{op.payload}.last(16);
                if (!op.args.back() || IsBaseView(view, texture.description)) {
                    depth_texture = texture;
                }
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
            CopyTexture(op);
            break;
        case ReplayOp::Barrier:
            Barrier();
            break;
        case ReplayOp::BindProgram:
            if (op.args.size() == 2 && (op.args[1] & 3)) {
                graphics_program = (op.args[1] & 3) == 3 ? op.program : nullptr;
            }
            break;
        case ReplayOp::BindVertexBuffer:
            if (op.args.size() == 3 && op.args[0] < vertex_buffers.size()) {
                vertex_buffers[op.args[0]] = {op.args[1], op.args[2]};
            }
            break;
        case ReplayOp::BindUniformBuffer:
            if (op.args.size() == 4 && op.args[0] < uniform_buffers.size() && op.args[1] < 32) {
                uniform_buffers[op.args[0]][op.args[1]] = {op.args[2], op.args[3]};
            }
            break;
        case ReplayOp::BindUniformBuffers:
            if (op.args.size() == 3 && op.args[0] < uniform_buffers.size() && op.args[1] < 32 &&
                op.args[2] <= 32 - op.args[1] && op.payload.size() == 16 * op.args[2]) {
                for (size_t i = 0; i < op.args[2]; ++i) {
                    std::array<u64_le, 2> range{};
                    std::memcpy(range.data(), op.payload.data() + i * 16, 16);
                    uniform_buffers[op.args[0]][op.args[1] + i] = {range[0], range[1]};
                }
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
            if (op.args.size() == 2 && op.args[0] < graphics_constants.size() &&
                op.args[1] <= 256 && op.payload.size() <= 256 - op.args[1]) {
                std::ranges::copy(op.payload, graphics_constants[op.args[0]].begin() + op.args[1]);
            }
            break;
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
        case ReplayOp::SetScissor:
            if (op.args.size() == 4) {
                scissor = {{static_cast<s32>(op.args[0]), static_cast<s32>(op.args[1])},
                           {static_cast<u32>(op.args[2]), static_cast<u32>(op.args[3])}};
                has_scissor = true;
            }
            break;
        case ReplayOp::Draw:
        case ReplayOp::DrawIndexed:
        case ReplayOp::DrawIndexedBaseVertex:
        case ReplayOp::DrawIndexedInstanced:
        case ReplayOp::DrawInstanced:
            Draw(op, submission);
            break;
        case ReplayOp::Dispatch:
            if (!missing_dispatch_logged) {
                missing_dispatch_logged = true;
                LOG_WARNING(Render_Vulkan, "Bananza NVN 2 Vulkan replay: guest dispatches need compute pipelines");
            }
            break;
        case ReplayOp::Unsupported:
            break;
        }
    }
}

void BananzaNvn2Replay::Replay(const Core::BananzaNvn2Commands::Submission& submission) {
    origin_lower_left = submission.origin_lower_left;
    if (!frame_dump.CapturingSampled()) {
        captured_inputs.clear();
    }
    const u64 previous_work = clears + copies + draws + texture_uploads;
    for (const auto& recording : submission.recordings) {
        ReplayRecording(*recording, submission);
    }
    if (clears + copies + draws + texture_uploads != previous_work) {
        scheduler.Finish();
        if (!replay_logged) {
            replay_logged = true;
            LOG_INFO(Render_Vulkan, "Bananza NVN 2 Vulkan replay completed {} clears and {} copies",
                     clears, copies);
        }
    }
    uploaded_ranges.clear();
    buffer_updates.clear();
    uploads.clear();
    descriptor_pools.clear();
    samplers.clear();
}

std::optional<BananzaNvn2Replay::Source> BananzaNvn2Replay::PresentSource(const Texture& texture) {
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
    if (++present_count % 600 == 0) {
        LOG_INFO(Render_Vulkan, "Bananza NVN 2 Vulkan replay: {} draws in 600 presents, {} texture uploads total",
                 draws - last_reported_draws, texture_uploads);
        last_reported_draws = draws;
    }
    Barrier();
    return Source{
        .texture = {*image->image, *image->view, image->extent.width, image->extent.height,
                    image->extent.width, image->extent.height},
        .format = image->surface_format,
        .address = texture.storage.gpu_address + texture.offset,
    };
}

} // namespace Vulkan
