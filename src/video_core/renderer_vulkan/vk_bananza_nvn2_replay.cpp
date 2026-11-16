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

#include <boost/container/small_vector.hpp>

#include "common/alignment.h"
#include "common/logging.h"
#include "common/cityhash.h"
#include "common/fs/path_util.h"
#include "common/scope_exit.h"
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
constexpr u64 DrawsBeforeFlush = 256;
constexpr u32 ScopesBeforeFlush = 6;
constexpr auto ScopeFlushInterval = std::chrono::microseconds{250};
constexpr auto ReplayFlushInterval = std::chrono::microseconds{700};
constexpr u64 MaxFreeStorageBytes = 128ULL << 20;
constexpr u64 StorageArenaBytes = 8ULL << 20;
constexpr u64 DeviceArenaBytes = 1ULL << 20;
constexpr u64 MaxFreeDeviceBytes = 32ULL << 20;
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
      trace_images{std::getenv("EDEN_NX2_NVN_TRACE_IMAGES") != nullptr},
      profile_replay{std::getenv("EDEN_NX2_NVN_CALL_PROFILE") != nullptr} {
    device_limits = device.GetPhysical().GetProperties().limits;
    device_features = device.GetPhysical().GetFeatures();
    upload_alignment = std::max<u64>({device_limits.minUniformBufferOffsetAlignment,
                                      device_limits.minStorageBufferOffsetAlignment, 64});
    constexpr float max_point_size = 2047.0f;
    std::memcpy(driver_constants.data() + 0x610, &max_point_size, sizeof(max_point_size));
    if (std::getenv("EDEN_NX2_NVN_NO_AHEAD_PIPELINES") == nullptr) {
        const auto& properties = device.GetPhysical().GetProperties();
        pipeline_list.emplace(Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir) /
                                  "bananza-nvn2-pipelines.bin",
                              properties.vendorID, properties.deviceID);
    }
    Core::BananzaNvn2Commands::GetRetirement().Start();
    retire_thread = std::jthread([this](std::stop_token token) { RetireLoop(token); });
}

BananzaNvn2Replay::~BananzaNvn2Replay() {
    retire_thread.request_stop();
    if (retire_thread.joinable()) {
        retire_thread.join();
    }
    Core::BananzaNvn2Commands::GetRetirement().Stop();
}

namespace {
// Writes the runs of mapped that differ from the submit-time snapshots; returns the bytes
// written and compared.
std::pair<u64, u64> WriteChangedBytes(Core::Memory::Memory& memory, u64 guest, u64 base,
                                      std::span<const u8> mapped,
                                      std::span<const std::pair<u64, u64>> snapshots,
                                      const Core::BananzaNvn2Commands::Submission& submission,
                                      const u8* whole = nullptr,
                                      std::span<const std::pair<u64, u64>> written_intervals = {}) {
    if (snapshots.empty() && !whole) {
        memory.WriteBlockUnsafe(guest, mapped.data(), mapped.size());
        return {mapped.size(), 0};
    }
    u64 written{};
    u64 compared{};
    const auto compare = [&](u64 offset, const u8* before, size_t size) {
        const u8* const now = mapped.data() + offset;
        compared += size;
        // Most ranges come back as they went in.
        if (std::memcmp(now, before, size) == 0) {
            return;
        }
        size_t i = 0;
        while (i < size) {
            const auto first = std::mismatch(now + i, now + size, before + i);
            i = static_cast<size_t>(first.first - now);
            if (i == size) {
                break;
            }
            // Merge runs separated by fewer than 64 equal bytes.
            size_t end = i + 1;
            size_t same = 0;
            for (size_t j = end; j < size && same < 64; ++j) {
                if (now[j] != before[j]) {
                    end = j + 1;
                    same = 0;
                } else {
                    ++same;
                }
            }
            memory.WriteBlockUnsafe(guest + offset + i, now + i, end - i);
            written += end - i;
            i = end;
        }
    };
    if (whole) {
        // Guest memory holds the span as it is now, not as it went in: outside what a shader
        // could write, a difference is the game's newer data and must stay.
        std::vector<std::pair<u64, u64>> intervals(written_intervals.begin(),
                                                   written_intervals.end());
        std::ranges::sort(intervals);
        constexpr size_t Piece = 64ULL << 10;
        u64 done{};
        for (const auto& [first, last] : intervals) {
            const u64 begin = std::max(first, done);
            const u64 end = std::min<u64>(last, mapped.size());
            for (u64 offset = begin; offset < end; offset += Piece) {
                compare(offset, whole + offset, static_cast<size_t>(std::min<u64>(Piece, end - offset)));
            }
            done = std::max(done, end);
        }
        return {written, compared};
    }
    for (const auto& snapshot : snapshots) {
        const u64 offset = snapshot.first - base;
        if (offset >= mapped.size()) {
            continue;
        }
        const auto before = submission.BufferSnapshot(snapshot);
        compare(offset, before.data(), std::min<size_t>(before.size(), mapped.size() - offset));
    }
    return {written, compared};
}
} // Anonymous namespace

void BananzaNvn2Replay::RetireLoop(std::stop_token token) {
    while (true) {
        std::unique_lock lock{retire_mutex};
        if (!retire_cv.wait(lock, token, [&] { return !retire_jobs.empty(); })) {
            return;
        }
        RetireJob job = std::move(retire_jobs.front());
        retire_jobs.pop_front();
        retire_running = true;
        lock.unlock();
        const u64 dequeued_ns = Core::BananzaNvn2Commands::HostNs();
        const auto write_back = [this, &job](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                const auto& write = job.writes[i];
                const auto [written, compared] = WriteChangedBytes(
                    *job.submission->guest_memory, write.guest_address, write.base,
                    std::span<const u8>{write.data, write.bytes}, write.snapshots,
                    *job.submission, write.before, write.written);
                writeback_bytes.fetch_add(written, std::memory_order_relaxed);
                writeback_compared.fetch_add(compared, std::memory_order_relaxed);
            }
        };
        static const bool serial_copies = std::getenv("EDEN_NX2_NVN_SERIAL_COPIES") != nullptr;
        // Batches of about 1 MiB; the last one runs here.
        const auto compare = [&](size_t first, size_t last) {
            for (const auto* buffer : job.written_buffers) {
                buffer->Invalidate();
            }
            size_t begin = first;
            u64 batch_bytes = 0;
            for (size_t i = first; i < last && !serial_copies; ++i) {
                batch_bytes += job.writes[i].bytes;
                if (batch_bytes >= (1ULL << 20) && i + 1 < last) {
                    retire_workers.QueueWork(
                        [write_back, begin, end = i + 1] { write_back(begin, end); });
                    begin = i + 1;
                    batch_bytes = 0;
                }
            }
            write_back(begin, last);
            retire_workers.WaitForRequests();
        };
        // Spans the GPU wrote for the last time in an earlier command buffer are compared
        // while it finishes the later ones.
        std::ranges::stable_sort(job.writes, {}, &WriteBack::tick);
        size_t compared_writes = 0;
        while (compared_writes < job.writes.size() &&
               job.writes[compared_writes].tick < job.tick) {
            const u64 tick = job.writes[compared_writes].tick;
            size_t end = compared_writes;
            while (end < job.writes.size() && job.writes[end].tick == tick) {
                ++end;
            }
            scheduler.GetMasterSemaphore().Wait(tick);
            compare(compared_writes, end);
            compared_writes = end;
        }
        // What the game can see of the submission: its event signals and counters, then the
        // sync. The caller holds retire_mutex.
        const auto finish = [&] {
            std::span<const u64> results;
            if (job.query_results) {
                job.query_results->Invalidate();
                results = {reinterpret_cast<const u64*>(job.query_results->Mapped().data()),
                           job.queries};
            }
            ApplyCounterEvents(results, job.counters, job.queries);
            for (const auto& event : job.counters) {
                if (event.kind == CounterEvent::Kind::Report) {
                    if (const auto it = pending_reports.find(event.gpu_address);
                        it != pending_reports.end() && --it->second == 0) {
                        pending_reports.erase(it);
                    }
                }
            }
            if (!job.counters.empty()) {
                pending_counter_jobs.fetch_sub(1, std::memory_order_relaxed);
            }
        };
        // The last command buffers usually hold the post-processing passes only. With every
        // written range back and no counter to read, the game has nothing to wait for in them.
        static const bool late_retire = std::getenv("EDEN_NX2_NVN_LATE_RETIRE") != nullptr;
        // EDEN_NX2_NVN_EARLY_SYNC=1: also when the last ranges are still to come back; they
        // stay pending for the next scan that reads them (Retirement::WritesDone).
        static const bool early_sync = std::getenv("EDEN_NX2_NVN_EARLY_SYNC") != nullptr;
        const bool writes_done = compared_writes == job.writes.size();
        const bool early = !late_retire && (writes_done || early_sync) && !job.query_results &&
                           std::ranges::all_of(job.counters, [](const CounterEvent& event) {
                               return event.kind == CounterEvent::Kind::Signal;
                           });
        u64 retired_ns{};
        if (early) {
            lock.lock();
            finish();
            lock.unlock();
            Core::BananzaNvn2Commands::GetRetirement().Retire(job.sequence, writes_done);
            retired_ns = Core::BananzaNvn2Commands::HostNs();
        }
        static std::atomic<u64> retired_jobs{};
        static std::atomic<u64> retired_early{};
        static std::atomic<u64> late_writes{};
        static std::atomic<u64> late_write_bytes{};
        static std::atomic<u64> late_counters{};
        static std::atomic<u64> late_queries{};
        static std::atomic<u64> late_ranges{};
        static std::atomic<u64> late_changed{};
        late_ranges.fetch_add(job.writes.size() - compared_writes, std::memory_order_relaxed);
        retired_early.fetch_add(early, std::memory_order_relaxed);
        if (compared_writes < job.writes.size()) {
            late_writes.fetch_add(1, std::memory_order_relaxed);
            for (size_t i = compared_writes; i < job.writes.size(); ++i) {
                late_write_bytes.fetch_add(job.writes[i].bytes, std::memory_order_relaxed);
            }
        }
        late_queries.fetch_add(job.query_results != nullptr, std::memory_order_relaxed);
        late_counters.fetch_add(
            std::ranges::any_of(job.counters, [](const CounterEvent& event) {
                return event.kind != CounterEvent::Kind::Signal;
            }),
            std::memory_order_relaxed);
        if (const u64 count = retired_jobs.fetch_add(1, std::memory_order_relaxed) + 1;
            count % 2400 == 0) {
            LOG_INFO(Render_Vulkan,
                     "NVN 2 retire: {} of the last 2400 submissions retired before the GPU "
                     "finished them; kept to the end: {} by ranges written in the last command "
                     "buffer ({} KiB in {} ranges, {} bytes of them changed), {} by counters, "
                     "{} by query results",
                     retired_early.exchange(0, std::memory_order_relaxed),
                     late_writes.exchange(0, std::memory_order_relaxed),
                     late_write_bytes.exchange(0, std::memory_order_relaxed) >> 10,
                     late_ranges.exchange(0, std::memory_order_relaxed),
                     late_changed.exchange(0, std::memory_order_relaxed),
                     late_counters.exchange(0, std::memory_order_relaxed),
                     late_queries.exchange(0, std::memory_order_relaxed));
        }
        scheduler.GetMasterSemaphore().Wait(job.tick);
        const u64 gpu_done_ns = Core::BananzaNvn2Commands::HostNs();
        if (!job.pass_labels.empty()) {
            ReportPassTimes(job);
            pass_trace_busy.store(false, std::memory_order_release);
        }
        if (compared_writes < job.writes.size()) {
            const u64 before = writeback_bytes.load(std::memory_order_relaxed);
            compare(compared_writes, job.writes.size());
            late_changed.fetch_add(writeback_bytes.load(std::memory_order_relaxed) - before,
                                   std::memory_order_relaxed);
        }
        lock.lock();
        if (!early) {
            finish();
        }
        retire_running = false;
        lock.unlock();
        if (!early) {
            Core::BananzaNvn2Commands::GetRetirement().Retire(job.sequence);
            retired_ns = Core::BananzaNvn2Commands::HostNs();
        } else if (!writes_done) {
            Core::BananzaNvn2Commands::GetRetirement().RetireWrites(job.sequence);
        }
        retire_idle_cv.notify_all();
        if (profile_replay && Core::BananzaNvn2Commands::InTimelineWindow(job.sequence)) {
            const u64 base = job.submission->host_submit_ns;
            const auto at = [&](u64 ns) { return (static_cast<double>(ns) - base) / 1e6; };
            const auto [first_begin, first_end] = scheduler.SubmitTimes(job.first_tick);
            const auto [last_begin, last_end] = scheduler.SubmitTimes(job.tick);
            LOG_INFO(Render_Vulkan,
                     "NVN 2 timeline: seq {} draws {} submit at {:.3f} ms, queued +{:.3f}, replay "
                     "+{:.3f} to +{:.3f}, ticks {} to {}, first vkQueueSubmit +{:.3f}, last "
                     "vkQueueSubmit +{:.3f} to +{:.3f}, retire thread free +{:.3f}, gpu done "
                     "+{:.3f}, retired +{:.3f}",
                     job.sequence, job.draws, base / 1e6, at(job.submission->host_queued_ns),
                     at(job.replay_start_ns), at(job.push_ns), job.first_tick, job.tick,
                     job.tick >= job.first_tick ? at(first_begin) : 0.0, at(last_begin),
                     at(last_end), at(dequeued_ns), at(gpu_done_ns), at(retired_ns));
        }
    }
}

void BananzaNvn2Replay::PassTimestamp(PassLabel label, bool top) {
    if (!pass_trace_active || pass_trace_labels.size() + 1 >= PassTraceQueries) {
        return;
    }
    Record([pool = *pass_trace_pool, query = static_cast<u32>(pass_trace_labels.size()),
            top](vk::CommandBuffer cmdbuf) {
        cmdbuf.WriteTimestamp(top ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                  : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                              pool, query);
    });
    label.draws = draws;
    pass_trace_labels.push_back(label);
}

void BananzaNvn2Replay::ReportPassTimes(const RetireJob& job) {
    const auto& labels = job.pass_labels;
    std::vector<u64> stamps(labels.size() + 1);
    const auto result = device.GetLogical().GetQueryResults(
        *pass_trace_pool, 0, static_cast<u32>(stamps.size()), stamps.size() * sizeof(u64),
        stamps.data(), sizeof(u64), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (result != VK_SUCCESS) {
        LOG_WARNING(Render_Vulkan, "NVN 2 GPU passes: query result {}", static_cast<int>(result));
        return;
    }
    const double period = device.GetPhysical().GetProperties().limits.timestampPeriod / 1e6;
    struct Total {
        double ms{};
        u32 scopes{};
        u64 draws{};
    };
    std::map<std::tuple<u64, u32, u32>, Total> by_target;
    std::array<Total, 7> by_reason{};
    struct Scope {
        double ms;
        size_t index;
        u64 draws;
    };
    std::vector<Scope> slowest;
    std::array<u32, 3> fresh{};
    std::map<std::pair<u64, u64>, u32> storage_ranges;
    std::map<u64, u32> storage_programs;
    std::array<u32, 9> fresh_slots{};
    std::map<u64, u32> fresh_programs;
    double idle_ms{};
    u32 flushes{};
    for (size_t i = 0; i < labels.size(); ++i) {
        const double ms = static_cast<double>(static_cast<s64>(stamps[i + 1] - stamps[i])) * period;
        const u64 scope_draws = (i + 1 < labels.size() ? labels[i + 1].draws : job.pass_end_draws) -
                                labels[i].draws;
        if (labels[i].reason & 32) {
            // From the end of the work before a flush to the start of the next command buffer.
            idle_ms += ms;
            ++flushes;
            continue;
        }
        auto& total = by_target[{labels[i].target, labels[i].width, labels[i].height}];
        total.ms += ms;
        total.draws += scope_draws;
        ++total.scopes;
        for (size_t bit = 0; bit < by_reason.size(); ++bit) {
            if (labels[i].reason & (1U << bit)) {
                by_reason[bit].ms += ms;
                ++by_reason[bit].scopes;
            }
        }
        slowest.push_back({ms, i, scope_draws});
        fresh[0] += (labels[i].fresh & 1) != 0;
        fresh[1] += (labels[i].fresh & 2) != 0;
        fresh[2] += (labels[i].fresh & 4) != 0;
        if (labels[i].fresh & 2) {
            ++fresh_slots[std::min<u32>(labels[i].fresh_slot, 8)];
            ++fresh_programs[labels[i].program];
        }
        if (labels[i].reason & 8) {
            ++storage_ranges[{labels[i].storage, labels[i].storage_size}];
            ++storage_programs[labels[i].program];
        }
    }
    const double span = static_cast<double>(stamps.back() - stamps.front()) * period;
    {
        // What the GPU is left with once the replay has handed over its last command buffer.
        size_t tail = 0;
        for (size_t i = 0; i < labels.size(); ++i) {
            if (labels[i].reason & 32) {
                tail = i + 1;
            }
        }
        std::string passes;
        for (size_t i = tail; i < labels.size(); ++i) {
            const double ms = static_cast<double>(static_cast<s64>(stamps[i + 1] - stamps[i])) * period;
            if (ms >= 0.05) {
                passes += fmt::format(" {}x{} program {} {:.3f} ms;", labels[i].width,
                                      labels[i].height, labels[i].program, ms);
            }
        }
        // The GPU's last 2 ms, newest first: passes, and gaps where it waited for a command buffer.
        std::string last;
        for (size_t i = labels.size(); i-- > 0;) {
            const double ms = static_cast<double>(static_cast<s64>(stamps[i + 1] - stamps[i])) * period;
            const double before_end =
                static_cast<double>(static_cast<s64>(stamps.back() - stamps[i + 1])) * period;
            if (before_end > 2.0) {
                break;
            }
            if (labels[i].reason & 32) {
                last += fmt::format(" [waited {:.3f} ms]", ms);
            } else if (ms >= 0.03) {
                last += fmt::format(" {}x{} p{} {:.3f};", labels[i].width, labels[i].height,
                                    labels[i].program, ms);
            }
        }
        LOG_INFO(Render_Vulkan, "NVN 2 GPU passes: last 2 ms on the GPU, newest first:{}", last);
        LOG_INFO(Render_Vulkan,
                 "NVN 2 GPU passes: after the last flush {:.3f} ms of GPU work in {} scopes:{}",
                 tail < stamps.size()
                     ? static_cast<double>(static_cast<s64>(stamps.back() - stamps[tail])) * period
                     : 0.0,
                 labels.size() - tail, passes);
    }
    LOG_INFO(Render_Vulkan,
             "NVN 2 GPU passes: seq {} with {} draws, {} scopes on {} targets, GPU span {:.3f} ms of "
             "which {:.3f} ms between command buffers ({} flushes)",
             job.sequence, job.pass_end_draws - labels.front().draws, slowest.size(),
             by_target.size(), span, idle_ms, flushes);
    LOG_INFO(Render_Vulkan,
             "NVN 2 GPU passes: scopes begun because targets changed {} ({:.3f} ms), sampled a "
             "written image {} ({:.3f} ms), storage written before {} ({:.3f} ms), storage written "
             "by the draw {} ({:.3f} ms), after other work {} ({:.3f} ms), after a flush {} ({:.3f} ms)",
             by_reason[0].scopes, by_reason[0].ms, by_reason[1].scopes, by_reason[1].ms,
             by_reason[2].scopes, by_reason[2].ms, by_reason[3].scopes, by_reason[3].ms,
             by_reason[4].scopes, by_reason[4].ms, by_reason[6].scopes, by_reason[6].ms);
    {
        std::vector<std::pair<u32, u32>> sites(job.scope_end_sites.begin(), job.scope_end_sites.end());
        std::ranges::sort(sites, std::greater{}, [](const auto& entry) { return entry.second; });
        std::string top;
        for (size_t i = 0; i < sites.size() && i < 10; ++i) {
            top += fmt::format(" line {} x{}", sites[i].first, sites[i].second);
        }
        LOG_INFO(Render_Vulkan, "NVN 2 GPU passes: scopes ended by:{}", top);
    }
    LOG_INFO(Render_Vulkan,
             "NVN 2 GPU passes: sampled a written depth target {}, colour target {}, other image {}; "
             "storage written by {} programs to {} ranges",
             fresh[0], fresh[1], fresh[2], storage_programs.size(), storage_ranges.size());
    {
        std::vector<std::pair<u64, u32>> programs(fresh_programs.begin(), fresh_programs.end());
        std::ranges::sort(programs, std::greater{}, [](const auto& entry) { return entry.second; });
        std::string top;
        for (size_t i = 0; i < programs.size() && i < 8; ++i) {
            top += fmt::format(" {}x{}", programs[i].first, programs[i].second);
        }
        LOG_INFO(Render_Vulkan,
                 "NVN 2 GPU passes: sampled colour slot 0: {}, 1: {}, 2: {}, 3: {}, 4+: {}; {} "
                 "programs (program x scopes):{}",
                 fresh_slots[0], fresh_slots[1], fresh_slots[2], fresh_slots[3],
                 fresh_slots[4] + fresh_slots[5] + fresh_slots[6] + fresh_slots[7],
                 programs.size(), top);
    }
    {
        std::vector<std::pair<std::pair<u64, u64>, u32>> ranges(storage_ranges.begin(),
                                                                storage_ranges.end());
        std::ranges::sort(ranges, std::greater{}, [](const auto& entry) { return entry.second; });
        for (size_t i = 0; i < ranges.size() && i < 6; ++i) {
            LOG_INFO(Render_Vulkan, "NVN 2 GPU passes: storage {:#x} of {} bytes written by {} scopes",
                     ranges[i].first.first, ranges[i].first.second, ranges[i].second);
        }
        std::vector<std::pair<u64, u32>> programs(storage_programs.begin(), storage_programs.end());
        std::ranges::sort(programs, std::greater{}, [](const auto& entry) { return entry.second; });
        for (size_t i = 0; i < programs.size() && i < 6; ++i) {
            LOG_INFO(Render_Vulkan, "NVN 2 GPU passes: program {} writes storage in {} scopes",
                     programs[i].first, programs[i].second);
        }
    }
    std::vector<std::pair<std::tuple<u64, u32, u32>, Total>> targets(by_target.begin(),
                                                                    by_target.end());
    std::ranges::sort(targets, std::greater{}, [](const auto& entry) { return entry.second.ms; });
    for (size_t i = 0; i < targets.size() && i < 16; ++i) {
        const auto& [key, total] = targets[i];
        LOG_INFO(Render_Vulkan,
                 "NVN 2 GPU passes: target {:#x} {}x{}: {:.3f} ms in {} scopes, {} draws",
                 std::get<0>(key), std::get<1>(key), std::get<2>(key), total.ms, total.scopes,
                 total.draws);
    }
    std::ranges::sort(slowest, std::greater{}, &Scope::ms);
    for (size_t i = 0; i < slowest.size() && i < 12; ++i) {
        const auto& label = labels[slowest[i].index];
        LOG_INFO(Render_Vulkan,
                 "NVN 2 GPU passes: scope {} took {:.3f} ms: target {:#x} {}x{}, first program {}, "
                 "{} draws, reason {:#x}",
                 slowest[i].index, slowest[i].ms, label.target, label.width, label.height,
                 label.program, slowest[i].draws, label.reason);
    }
}

void BananzaNvn2Replay::ProbeHostImport(const u8* pointer, u64 size) {
    const auto& dld = device.GetDispatchLoader();
    const VkDevice handle = *device.GetLogical();
    const auto host_properties = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
        dld.vkGetDeviceProcAddr(handle, "vkGetMemoryHostPointerPropertiesEXT"));
    if (!host_properties) {
        LOG_INFO(Render_Vulkan, "NVN 2 import probe: VK_EXT_external_memory_host is not enabled");
        return;
    }
    const auto ms_since = [](std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
            .count();
    };
    constexpr std::array types{VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                               VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT};
    for (const auto type : types) {
        for (const u64 alignment : {u64{4096}, u64{65536}}) {
            const u64 address = reinterpret_cast<u64>(pointer);
            const u64 low = address & ~(alignment - 1);
            const u64 high = (address + size + alignment - 1) & ~(alignment - 1);
            const u64 bytes = high - low;
            VkMemoryHostPointerPropertiesEXT properties{
                .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
            auto start = std::chrono::steady_clock::now();
            VkResult result = host_properties(handle, type, reinterpret_cast<void*>(low), &properties);
            LOG_INFO(Render_Vulkan,
                     "NVN 2 import probe: handle type {:#x}, alignment {}, {} KiB at {:#x}: "
                     "properties result {}, memory types {:#x}, {:.3f} ms",
                     static_cast<u32>(type), alignment, bytes >> 10, low, static_cast<int>(result),
                     properties.memoryTypeBits, ms_since(start));
            if (result != VK_SUCCESS || !properties.memoryTypeBits) {
                continue;
            }
            const VkExternalMemoryBufferCreateInfo external{
                .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                .handleTypes = static_cast<VkExternalMemoryHandleTypeFlags>(type)};
            const VkBufferCreateInfo buffer_info{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = &external,
                .size = bytes,
                .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };
            VkBuffer buffer{};
            result = dld.vkCreateBuffer(handle, &buffer_info, nullptr, &buffer);
            if (result != VK_SUCCESS) {
                LOG_INFO(Render_Vulkan, "NVN 2 import probe: vkCreateBuffer result {}",
                         static_cast<int>(result));
                continue;
            }
            const VkMemoryRequirements requirements =
                device.GetLogical().GetBufferMemoryRequirements(buffer);
            const u32 usable = requirements.memoryTypeBits & properties.memoryTypeBits;
            VkDeviceMemory memory{};
            if (usable && requirements.size <= bytes) {
                const VkImportMemoryHostPointerInfoEXT import{
                    .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
                    .handleType = type,
                    .pHostPointer = reinterpret_cast<void*>(low),
                };
                const VkMemoryAllocateInfo allocate{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    .pNext = &import,
                    .allocationSize = bytes,
                    .memoryTypeIndex = static_cast<u32>(std::countr_zero(usable)),
                };
                start = std::chrono::steady_clock::now();
                result = dld.vkAllocateMemory(handle, &allocate, nullptr, &memory);
                const double allocate_ms = ms_since(start);
                VkResult bound = VK_ERROR_UNKNOWN;
                if (result == VK_SUCCESS) {
                    bound = dld.vkBindBufferMemory(handle, buffer, memory, 0);
                }
                LOG_INFO(Render_Vulkan,
                         "NVN 2 import probe: memory type {} of {:#x}, vkAllocateMemory result {} in "
                         "{:.3f} ms, vkBindBufferMemory result {}",
                         std::countr_zero(usable), usable, static_cast<int>(result), allocate_ms,
                         static_cast<int>(bound));
            } else {
                LOG_INFO(Render_Vulkan,
                         "NVN 2 import probe: no usable memory type (buffer {:#x}, host {:#x}, size "
                         "{} of {})",
                         requirements.memoryTypeBits, properties.memoryTypeBits, requirements.size,
                         bytes);
            }
            dld.vkDestroyBuffer(handle, buffer, nullptr);
            if (memory) {
                dld.vkFreeMemory(handle, memory, nullptr);
            }
        }
    }
}

void BananzaNvn2Replay::DispatchCopies() {
    if (!copy_jobs.empty()) {
        copy_workers.QueueWork([jobs = std::move(copy_jobs), late = late_copies] {
            if (late) {
                late->Wait();
            }
            for (const auto& job : jobs) {
                std::memcpy(job.dest, job.source, job.size);
            }
        });
        copy_jobs.clear();
        copy_job_bytes = 0;
    }
}

void BananzaNvn2Replay::AwaitGuestWrites(const u8* data, u64 size) {
    if (pending_writebacks.empty() && !pending_unknown_writes) {
        return;
    }
    auto& retirement = Core::BananzaNvn2Commands::GetRetirement();
    const auto prune = [&] {
        std::erase_if(pending_writebacks, [&](const PendingWriteback& pending) {
            return retirement.WritesDone(pending.sequence);
        });
        if (pending_unknown_writes && retirement.WritesDone(pending_unknown_writes)) {
            pending_unknown_writes = 0;
        }
    };
    prune();
    u64 sequence = pending_unknown_writes;
    for (const PendingWriteback& pending : pending_writebacks) {
        if (pending.begin < data + size && data < pending.end) {
            sequence = std::max(sequence, pending.sequence);
        }
    }
    if (!sequence) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    retirement.WaitWrites(sequence);
    guest_wait_ns += static_cast<u64>((std::chrono::steady_clock::now() - start).count());
    ++guest_waits;
    prune();
}

std::span<const u8> BananzaNvn2Replay::ReadSnapshot(const Submission& submission, Range range) {
    const auto* const snapshot = submission.FindSnapshot(range);
    if (!snapshot) {
        return {};
    }
    if (!snapshot->owned) {
        AwaitGuestWrites(snapshot->data, range.second);
    }
    submission.WaitLate(*snapshot);
    return {snapshot->data, range.second};
}

void BananzaNvn2Replay::WaitCopies() {
    const auto pending = [](const StorageArena& arena) { return arena.used > arena.flushed; };
    const bool unflushed = std::ranges::any_of(storage_arenas, pending) ||
                           std::ranges::any_of(device_arenas, pending) || !kept_fills.empty();
    if (copy_jobs.empty() && !unflushed) {
        return;
    }
    if (late_copies && !copy_jobs.empty()) {
        late_copies->Wait();
    }
    for (const auto& job : copy_jobs) {
        std::memcpy(job.dest, job.source, job.size);
    }
    copy_jobs.clear();
    copy_job_bytes = 0;
    copy_workers.WaitForRequests();
    for (const auto& [buffer, size] : kept_fills) {
        buffer->Flush(0, size);
    }
    kept_fills.clear();
    for (auto* arenas : {&storage_arenas, &device_arenas}) {
        for (auto& arena : *arenas) {
            if (arena.used > arena.flushed) {
                arena.buffer.Flush(arena.flushed, arena.used - arena.flushed);
                arena.flushed = arena.used;
            }
        }
    }
}

std::pair<VkBuffer, u64> BananzaNvn2Replay::AllocateDeviceStorage(u64 aligned, u8*& mapped) {
    if (device_arenas.empty() ||
        device_arenas.back().capacity - device_arenas.back().used < aligned) {
        const u64 capacity = std::max(DeviceArenaBytes, std::bit_ceil(aligned));
        auto& free = free_device_storage[capacity];
        vk::Buffer buffer;
        if (free.empty()) {
            buffer = allocator.CreateBuffer({
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = capacity,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            }, MemoryUsage::Stream);
        } else {
            buffer = std::move(free.back());
            free.pop_back();
            free_device_bytes -= capacity;
        }
        device_arenas.push_back({std::move(buffer), capacity});
    }
    auto& arena = device_arenas.back();
    const u64 base = arena.used;
    arena.used += aligned;
    mapped = arena.buffer.Mapped().data() + base;
    return {*arena.buffer, base};
}

void BananzaNvn2Replay::MoveStorageToDevice(StorageUpload& upload) {
    run_stats.DrawCut(BananzaNvn2RunStats::Cut::StorageMove);
    u8* mapped{};
    const auto [buffer, base] =
        AllocateDeviceStorage(Common::AlignUp(upload.size, upload_alignment), mapped);
    // What the CPU filled and the GPU wrote so far goes along; the fills are waited for
    // before the command buffer is submitted.
    Barrier();
    Record([source = upload.buffer, target = buffer,
            region = VkBufferCopy{upload.base, base, upload.size}](vk::CommandBuffer cmdbuf) {
        cmdbuf.CopyBuffer(source, target, region);
    });
    Barrier();
    upload.readback_buffer = upload.buffer;
    upload.readback_base = upload.base;
    upload.buffer = buffer;
    upload.base = base;
    device_uploads.push_back(&upload);
    // A vertex stream over the same range must not keep the host copy.
    vertex_uploads.fill({});
    ++upload_generation;
    ++device_moves;
    ++device_spans;
    device_span_bytes += upload.size;
}

void BananzaNvn2Replay::RecordStorageReadback(bool flush) {
    device_readback_pending = false;
    boost::container::small_vector<std::tuple<VkBuffer, VkBuffer, VkBufferCopy>, 8> copies;
    boost::container::small_vector<StorageUpload*, 8> copied;
    for (auto* upload : device_uploads) {
        if (upload->dirty && upload->written_bytes && upload->readback_stale) {
            copies.emplace_back(upload->buffer, upload->readback_buffer,
                                VkBufferCopy{upload->base, upload->readback_base,
                                             upload->written_bytes});
            copied.push_back(upload);
        }
    }
    if (copies.empty()) {
        return;
    }
    Barrier();
    for (auto* upload : copied) {
        upload->readback_stale = false;
        upload->readback_tick = scheduler.CurrentTick();
    }
    Record([copies = std::move(copies)](vk::CommandBuffer cmdbuf) {
        for (const auto& [source, target, region] : copies) {
            cmdbuf.CopyBuffer(source, target, region);
        }
    });
    if (flush && !frame_dump.CapturingSampled()) {
        // The copy's command buffer ends here, so its tick does not wait for later passes.
        PassTimestamp({.reason = 32});
        WaitCopies();
        scheduler.Flush();
        PassTimestamp({.reason = 64}, true);
        ++replay_flushes;
        flush_scopes = 0;
        next_flush_draw = draws + DrawsBeforeFlush;
        next_replay_flush = std::chrono::steady_clock::now() + ReplayFlushInterval;
    }
}

void BananzaNvn2Replay::WaitRetireIdle() {
    std::unique_lock lock{retire_mutex};
    retire_idle_cv.wait(lock, [&] { return retire_jobs.empty() && !retire_running; });
}

void BananzaNvn2Replay::ApplyCounterEvents(std::span<const u64> results,
                                           std::span<const CounterEvent> events, u32 queries) {
    const auto sum = [&](u32 begin, u32 end) {
        end = std::min<u32>(end, static_cast<u32>(results.size()));
        return begin < end ? std::accumulate(results.begin() + begin, results.begin() + end, u64{})
                           : u64{};
    };
    u32 position = 0;
    for (const auto& event : events) {
        switch (event.kind) {
        case CounterEvent::Kind::Reset:
            counted_samples = 0;
            position = event.query;
            break;
        case CounterEvent::Kind::Report: {
            counted_samples += sum(position, event.query);
            position = event.query;
            const std::array<u64_le, 2> data{counted_samples, event.submitted_ns};
            event.memory->WriteBlockUnsafe(event.guest_address, data.data(), sizeof(data));
            break;
        }
        case CounterEvent::Kind::Signal:
            event.memory->Write32(event.guest_address,
                                  event.add ? event.memory->Read32(event.guest_address) + event.value
                                            : event.value);
            break;
        }
    }
    counted_samples += sum(position, queries);
}

template <typename T>
void BananzaNvn2Replay::Record(T&& command, std::source_location location) {
    if (pass_trace_active && rendering_scope) {
        ++scope_end_sites[location.line()];
    }
    EndRendering();
    scheduler.Record(std::forward<T>(command));
}

void BananzaNvn2Replay::EndRendering() {
    if (rendering_scope) {
        scheduler.Record([](vk::CommandBuffer cmdbuf) { cmdbuf.EndRendering(); });
        rendering_scope.reset();
        scope_pipeline = VK_NULL_HANDLE;
        scope_descriptor_set = VK_NULL_HANDLE;
        scope_wrote_storage = false;
    }
}

VkFormatFeatureFlags2 BananzaNvn2Replay::FormatFeatures(VkFormat format) {
    const auto [entry, inserted] = format_features.try_emplace(format);
    if (inserted) {
        entry->second = device.GetPhysical().GetFormatProperties3(format).optimalTilingFeatures;
    }
    return entry->second;
}

bool BananzaNvn2Replay::ReserveImageBytes(u64 bytes) {
    const u64 limit = std::clamp<u64>(device.GetDeviceLocalMemory() / 4,
                                     384ULL << 20, 4ULL << 30);
    if (image_bytes <= limit && bytes <= limit - image_bytes) {
        return true;
    }
    if (bytes > limit) {
        return false;
    }
    std::vector<decltype(images)::iterator> cold;
    for (auto it = images.begin(); it != images.end(); ++it) {
        const auto& image = it->second;
        if (!image.gpu_written && image.cpu_serial && present_count - image.last_present >= 4) {
            cold.push_back(it);
        }
    }
    if (cold.empty()) {
        return false;
    }
    std::ranges::sort(cold, {}, [](const auto& it) { return it->second.last_present; });
    // Finish outstanding transfers and presentation reads before releasing Vulkan objects.
    EndRendering();
    WaitCopies();
    scheduler.Finish();
    const u64 previous_bytes = image_bytes;
    // Leave room for the next resources in the same load transition.
    const u64 target = std::min(limit - bytes, limit - limit / 8);
    for (const auto it : cold) {
        const auto& image = it->second;
        if (const auto feedback = feedback_images.find(&image); feedback != feedback_images.end()) {
            image_bytes -= feedback->second.allocation_bytes;
            feedback_images.erase(feedback);
        }
        if (const auto depth = depth_comparison_images.find(&image);
            depth != depth_comparison_images.end()) {
            image_bytes -= depth->second.allocation_bytes;
            depth_comparison_images.erase(depth);
        }
        image_bytes -= image.allocation_bytes;
        images.erase(it);
        image_memo.fill({});
        if (image_bytes <= target) {
            break;
        }
    }
    if (trace_images) {
        LOG_INFO(Render_Vulkan, "NVN 2 reclaimed {} MiB of cold texture allocations",
                 (previous_bytes - image_bytes) >> 20);
    }
    return image_bytes <= limit - bytes;
}

BananzaNvn2Replay::Image* BananzaNvn2Replay::FindImage(const Texture& texture, bool create) {
    const auto& d = texture.description;
    const u32 layers = texture.layout.layers;
    const bool volume = d.target == 2;
    const u32 depth = volume ? d.depth : 1;
    const auto& limits = device_limits;
    if (!texture.serial || !texture.storage.gpu_address ||
        (d.target != 1 && !volume && d.target != 4 && d.target != 8 && d.target != 9) ||
        !layers || layers > device_limits.maxImageArrayLayers ||
        d.samples > 1 || !depth || !d.width || !d.height ||
        d.width > (volume ? limits.maxImageDimension3D : limits.maxImageDimension2D) ||
        d.height > (volume ? limits.maxImageDimension3D : limits.maxImageDimension2D) ||
        (volume && depth > limits.maxImageDimension3D) || !d.levels ||
        d.levels > static_cast<u32>(std::bit_width(std::max({d.width, d.height, depth})))) {
        return nullptr;
    }
    const ImageKey key{texture.storage.gpu_address + texture.offset, d.width, d.height, d.format,
                       d.levels, d.target, layers, depth};
    const u64 address = std::get<0>(key);
    auto& memo = image_memo[(address >> 12 ^ address >> 22 ^ d.width ^ d.format << 4 ^
                             d.levels << 7 ^ layers << 9) % image_memo.size()];
    if (memo.image && memo.key == key) {
        memo.image->last_present = present_count;
        return memo.image;
    }
    if (const auto it = images.find(key); it != images.end()) {
        it->second.last_present = present_count;
        memo = {key, &it->second};
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
    ++image_creates;
    const u64 bytes = image.GetMemoryBlock().size;
    if (!ReserveImageBytes(bytes)) {
        if (!image_budget_logged) {
            image_budget_logged = true;
            LOG_WARNING(Render_Vulkan, "Bananza NVN 2 image limit reached at {} MiB for format {:#x}, {}x{}",
                        image_bytes >> 20, d.format, d.width, d.height);
        }
        return nullptr;
    }
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
    auto& result = images.emplace(key, Image{std::move(image), std::move(view), extent, format->vk,
                                            format->surface, d.levels, layers, depth, volume,
                                            attachment, storage, format->aspects}).first->second;
    result.allocation_bytes = bytes;
    result.last_present = present_count;
    return &result;
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
    Record([handle = *image.image, levels = image.levels, layers = image.layers,
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

void BananzaNvn2Replay::Barrier(std::source_location location) {
    if (device_readback_pending) {
        // Now and not at the submission's end: the game can then be told about the bytes
        // before the GPU has drawn the passes that follow.
        RecordStorageReadback(true);
    }
    ++barrier_serial;
    ++barriers;
    scheduler.RequestOutsideRenderPassOperationContext();
    Record([](vk::CommandBuffer cmdbuf) {
        const VkMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, barrier);
    }, location);
    // Give the GPU completed passes while replay records the rest of the submission.
    // Passes with few draws (post-processing) count too, or all of them wait for the end.
    static const bool pass_flush = std::getenv("EDEN_NX2_NVN_NO_PASS_FLUSH") == nullptr;
    const bool many_scopes = pass_flush && flush_scopes >= ScopesBeforeFlush;
    if ((draws >= next_flush_draw || many_scopes) && !frame_dump.CapturingSampled()) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_replay_flush ||
            (many_scopes && now + (ReplayFlushInterval - ScopeFlushInterval) >= next_replay_flush)) {
            PassTimestamp({.reason = 32});
            WaitCopies();
            scheduler.Flush();
            PassTimestamp({.reason = 64}, true);
            ++replay_flushes;
            flush_scopes = 0;
            next_flush_draw = draws + DrawsBeforeFlush;
            next_replay_flush = std::chrono::steady_clock::now() + ReplayFlushInterval;
        }
    }
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
    if (!render_enabled || op.args.size() != 2 || op.Payload().size() != 16 || op.args[0] >= colors.size() ||
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
    std::memcpy(&value, op.Payload().data(), op.Payload().size());
    TraceImage("clear", *colors[op.args[0]]);
    image->gpu_written = true;
    ++image->write_version;
    Record([attachment, value, extent, rect, whole, initialized,
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
    if (!render_enabled || !depth_texture || op.args.size() != 3 || op.Payload().size() != sizeof(float)) {
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
    std::memcpy(&value.depth, op.Payload().data(), sizeof(value.depth));
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
    Record([view = attachment, rect, aspects, image_aspects = image->aspects, value,
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
    if (op.Textures().size() != 2 || op.args.size() != 5 ||
        op.Payload().size() != 32 + 2 * sizeof(CopyRegion)) {
        return;
    }
    const auto source_address = op.Textures()[0].storage.gpu_address + op.Textures()[0].offset;
    const auto reject = [&](const char* reason) {
        TraceImage(reason, op.Textures()[1], source_address);
    };
    TraceImage("copy request", op.Textures()[1], source_address);
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
        const auto& texture = op.Textures()[index];
        const auto& d = texture.description;
        const auto view = std::span{op.Payload()}.subspan(16 * index, 16);
        const u8 flags = op.args[3 + index] ? view[7] : 0;
        const u32 level = flags & 1 ? view[4] : 0;
        const u32 layer = flags & 2 ? u32{view[0]} | (u32{view[1]} << 8) : 0;
        if (level >= d.levels || ((flags & 4) && ViewFormat(view, d.format) != d.format) ||
            ((flags & 0x20) && view[6] != d.target)) {
            return std::nullopt;
        }
        CopyRegion region{};
        std::memcpy(&region, op.Payload().data() + 32 + index * sizeof(region), sizeof(region));
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
    auto* source = FindImage(op.Textures()[0], true);
    if (!source || (source->aspects != VK_IMAGE_ASPECT_COLOR_BIT &&
                    source->aspects != VK_IMAGE_ASPECT_DEPTH_BIT) ||
        !PrepareSampledImage(*source, op.Textures()[0], submission)) {
        reject("copy source unavailable");
        return;
    }
    auto* dest = FindImage(op.Textures()[1], true);
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
            submission.texture_snapshots.contains(op.Textures()[1].serial)) {
            if (!PrepareSampledImage(*dest, op.Textures()[1], submission)) {
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
            Record([src = *source->image, dst = *dest->image, buffer = *scratch,
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
        TraceImage("depth copy", op.Textures()[1], source_address);
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
        for (size_t index = 0; index < op.Textures().size(); ++index) {
            const auto& texture = op.Textures()[index];
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
    TraceImage("copy", op.Textures()[1], source_address);
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
            Record([src = *source->image, dst = *dest->image, region, filter](
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
            Record([src = *source->image, dst = *dest->image, copy](vk::CommandBuffer cmdbuf) {
                cmdbuf.CopyImage(src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, {copy});
            });
        }
    }
    ++copies;
}

void BananzaNvn2Replay::CopyBufferToTexture(const Operation& op, const Submission& submission) {
    if (op.args.size() != 3 || op.args[1] & ~2ULL || op.Textures().size() != 1 ||
        op.Payload().size() != 16 + sizeof(CopyRegion)) {
        return;
    }
    const auto& texture = op.Textures()[0];
    const auto& desc = texture.description;
    const auto block = Core::BananzaNvn2Hle::GetTextureFormatBlock(desc.format);
    const auto view = std::span{op.Payload()}.first(16);
    const u8 flags = op.args[2] ? view[7] : 0;
    const u32 level = flags & 1 ? view[4] : 0;
    const u32 base_layer = flags & 2 ? u32{view[0]} | (u32{view[1]} << 8) : 0;
    if (!block || level >= desc.levels || base_layer >= texture.layout.layers ||
        ((flags & 4) && ViewFormat(view, desc.format) != desc.format)) {
        return;
    }
    CopyRegion region{};
    std::memcpy(&region, op.Payload().data() + 16, sizeof(region));
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
    const auto data = submission.BufferSnapshot(
        {op.args[0], stride * (rows * region.depth - 1) + row_bytes});
    if (data.empty()) {
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
                        data.data() + (z * rows + row) * stride, row_bytes);
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
    Record([buffer = *upload, target = *image->image, copy](vk::CommandBuffer cmdbuf) {
        cmdbuf.CopyBufferToImage(buffer, target, VK_IMAGE_LAYOUT_GENERAL, {copy});
    });
    uploads.push_back(std::move(upload));
    image->gpu_written = true;
    ++image->write_version;
    TraceImage("buffer copy", texture, op.args[0]);
    ++copies;
}

VkBuffer BananzaNvn2Replay::Upload(std::span<const u8> bytes, VkBufferUsageFlags extra_usage,
                                  std::optional<u64> patch_address) {
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
    if (patch_address) {
        ApplyBufferUpdates(*patch_address, buffer.Mapped().first(bytes.size()));
    }
    buffer.Flush();
    const VkBuffer handle = *buffer;
    uploads.push_back(std::move(buffer));
    return handle;
}

BananzaNvn2Replay::BufferRef BananzaNvn2Replay::UploadStream(std::span<const u8> bytes,
                                                            std::optional<u64> patch_address) {
    constexpr u64 ChunkSize = 32ULL << 20;
    if (bytes.empty()) {
        return {};
    }
    if (bytes.size() > ChunkSize / 4) {
        return {Upload(bytes, 0, patch_address), 0};
    }
    while (true) {
        auto& chunk = active_chunks.empty() ? NextUploadChunk() : upload_chunks[active_chunks.back()];
        // An early submit can leave earlier slices of this chunk in use on the GPU.
        const u64 alignment = chunk.buffer.IsHostCoherent()
                                  ? upload_alignment
                                  : std::max(upload_alignment, device_limits.nonCoherentAtomSize);
        const u64 offset = Common::AlignUp(chunk.used, alignment);
        if (offset + bytes.size() <= ChunkSize) {
            auto mapped = chunk.buffer.Mapped().subspan(offset, bytes.size());
            std::memcpy(mapped.data(), bytes.data(), bytes.size());
            if (patch_address) {
                ApplyBufferUpdates(*patch_address, mapped);
            }
            chunk.buffer.Flush(offset, bytes.size());
            chunk.used = offset + bytes.size();
            stream_bytes += bytes.size();
            ++stream_uploads;
            return {*chunk.buffer, offset};
        }
        NextUploadChunk();
    }
}

BananzaNvn2Replay::UploadChunk& BananzaNvn2Replay::NextUploadChunk() {
    constexpr u64 ChunkSize = 32ULL << 20;
    for (size_t i = 0; i < upload_chunks.size(); ++i) {
        auto& chunk = upload_chunks[i];
        if (!chunk.active && scheduler.IsFree(chunk.tick)) {
            chunk.active = true;
            chunk.used = 0;
            active_chunks.push_back(i);
            return chunk;
        }
    }
    upload_chunks.push_back({allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = ChunkSize,
            .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        },
        MemoryUsage::Stream)});
    upload_chunks.back().active = true;
    active_chunks.push_back(upload_chunks.size() - 1);
    return upload_chunks.back();
}

void BananzaNvn2Replay::ReclaimBusyResources() {
    constexpr size_t MaxFreeDescriptorPools = 12;
    auto& retirement = Core::BananzaNvn2Commands::GetRetirement();
    std::erase_if(busy_resources, [&](BusyResources& busy) {
        if (!scheduler.IsFree(busy.tick) || !retirement.IsRetired(busy.sequence)) {
            return false;
        }
        for (auto& buffer : busy.query_results) {
            free_query_results.push_back(std::move(buffer));
        }
        for (auto& [capacity, buffer] : busy.storage) {
            if (capacity <= MaxFreeStorageBytes - free_storage_bytes) {
                free_storage_buffers[capacity].push_back(std::move(buffer));
                free_storage_bytes += capacity;
            }
        }
        for (auto& [capacity, buffer] : busy.device_storage) {
            if (capacity <= MaxFreeDeviceBytes - free_device_bytes) {
                free_device_storage[capacity].push_back(std::move(buffer));
                free_device_bytes += capacity;
            }
        }
        for (auto& [reusable, pool] : busy.pools) {
            if (reusable && free_descriptor_pools.size() < MaxFreeDescriptorPools) {
                pool.Reset();
                free_descriptor_pools.push_back(std::move(pool));
            }
        }
        for (auto& block : busy.pushed) {
            if (free_push_blocks.size() < 16) {
                free_push_blocks.push_back(std::move(block));
            }
        }
        return true;
    });
}

VkDescriptorSet BananzaNvn2Replay::AllocateSet(VkDescriptorSetLayout layout, size_t uniforms,
                                               size_t storage, size_t textures, size_t images) {
    constexpr u32 Sets = 1024;
    for (int attempt = 0; attempt < 2; ++attempt) {
        const bool fits = !descriptor_chunks.empty() && descriptor_chunks.back().sets &&
                          descriptor_chunks.back().uniforms >= uniforms &&
                          descriptor_chunks.back().storage >= storage &&
                          descriptor_chunks.back().textures >= textures &&
                          descriptor_chunks.back().images >= images;
        if (!fits || attempt) {
            auto& counts = descriptor_chunks.emplace_back();
            counts.sets = Sets;
            counts.uniforms = static_cast<u32>(std::max<size_t>(Sets * 8, uniforms));
            counts.storage = static_cast<u32>(std::max<size_t>(Sets * 4, storage));
            counts.textures = static_cast<u32>(std::max<size_t>(Sets * 16, textures));
            counts.images = static_cast<u32>(std::max<size_t>(Sets, images));
            counts.reusable = counts.uniforms == Sets * 8 && counts.storage == Sets * 4 &&
                              counts.textures == Sets * 16 && counts.images == Sets;
            if (counts.reusable && !free_descriptor_pools.empty()) {
                counts.pool = std::move(free_descriptor_pools.back());
                free_descriptor_pools.pop_back();
            } else {
                const std::array sizes{
                    VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, counts.uniforms},
                    VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, counts.storage},
                    VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, counts.textures},
                    VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, counts.images},
                };
                counts.pool = device.GetLogical().CreateDescriptorPool({
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                    .maxSets = Sets,
                    .poolSizeCount = static_cast<u32>(sizes.size()),
                    .pPoolSizes = sizes.data(),
                });
            }
        }
        auto& chunk = descriptor_chunks.back();
        const VkDescriptorSet set = chunk.pool.AllocateOne(layout);
        if (!set) {
            continue;
        }
        --chunk.sets;
        chunk.uniforms -= static_cast<u32>(uniforms);
        chunk.storage -= static_cast<u32>(storage);
        chunk.textures -= static_cast<u32>(textures);
        chunk.images -= static_cast<u32>(images);
        return set;
    }
    return VK_NULL_HANDLE;
}

BananzaNvn2Replay::BufferRef BananzaNvn2Replay::UploadRange(const Submission& submission,
                                                            Range range) {
    // A storage upload made for a range takes that range out of uploaded_ranges.
    if (const auto cached = uploaded_ranges.find(range); cached != uploaded_ranges.end()) {
        run_stats.Count(BananzaNvn2RunStats::Note::RangeMemo);
        return cached->second;
    }
    static const bool fixed_cache_on = std::getenv("EDEN_NX2_NVN_NO_FIXED_CACHE") == nullptr;
    const auto* const snapshot = submission.FindSnapshot(range);
    // A uniform range whose pages went to its pool's mirror at the start of the replay.
    if (snapshot && snapshot->staged_pool && staged_ranges_ready && !HasBufferUpdates(range)) {
        if (const auto mirror = pool_mirrors.find(snapshot->staged_pool);
            mirror != pool_mirrors.end() && mirror->second.buffer) {
            const BufferRef buffer{*mirror->second.buffer, snapshot->fixed_offset};
            uploaded_ranges.emplace(range, buffer);
            run_stats.Count(BananzaNvn2RunStats::Note::RangeStaged);
            return buffer;
        }
    }
    // A range the GPU never writes is not a storage upload: no need to look for one.
    if (!(fixed_cache_on && snapshot && snapshot->fixed_pool)) {
        if (const auto storage = storage_upload_of.find(range);
            storage != storage_upload_of.end()) {
            run_stats.Count(BananzaNvn2RunStats::Note::RangeStorage);
            return {storage->second->buffer, storage->second->base};
        }
    }
    if (!snapshot || !range.second) {
        return {};
    }
    submission.WaitLate(*snapshot);
    if (!snapshot->owned) {
        AwaitGuestWrites(snapshot->data, range.second);
    }
    const std::span<const u8> bytes{snapshot->data, range.second};
    const bool patched = HasBufferUpdates(range);
    if (fixed_cache_on && !patched) {
        if (const auto& entry = *snapshot; entry.fixed_pool) {
            // On by default since r129; EDEN_NX2_NVN_NO_POOL_MIRROR=1 is the per-range cache.
            static const bool mirror_on =
                std::getenv("EDEN_NX2_NVN_NO_POOL_MIRROR") == nullptr;
            if (mirror_on) {
                if (const auto buffer = MirrorUpload(submission, entry, range.second);
                    buffer.first) {
                    uploaded_ranges.emplace(range, buffer);
                    return buffer;
                }
            }
            if (const auto buffer = FixedUpload(bytes, entry.fixed_pool, entry.fixed_stamp);
                buffer.first) {
                uploaded_ranges.emplace(range, buffer);
                run_stats.Count(BananzaNvn2RunStats::Note::RangeFixed);
                return buffer;
            }
        }
    }
    const auto buffer = UploadStream(bytes, patched ? std::optional{range.first} : std::nullopt);
    uploaded_ranges.emplace(range, buffer);
    run_stats.Count(BananzaNvn2RunStats::Note::RangeStream);
    return buffer;
}

BananzaNvn2Replay::PoolMirror* BananzaNvn2Replay::GetMirror(u64 pool, u64 size) {
    constexpr u64 MaxMirrorBytes = 3ULL << 30;
    auto& mirror = pool_mirrors[pool];
    if (mirror.buffer) {
        return &mirror;
    }
    if (!size || pool_mirror_bytes + size > MaxMirrorBytes) {
        return nullptr;
    }
    mirror.buffer = allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        },
        MemoryUsage::DeviceLocal);
    mirror.size = size;
    mirror.synced.assign((size + 0xffff) >> 16, 0);
    pool_mirror_bytes += size;
    if ((pool_mirrors.size() & 63) == 1) {
        LOG_INFO(Render_Vulkan, "NVN 2 pool mirror: {} pools, {} MiB in all", pool_mirrors.size(),
                 pool_mirror_bytes >> 20);
    }
    return &mirror;
}

bool BananzaNvn2Replay::SyncMirrorPages(const Submission& submission) {
    if (submission.mirror_pages.empty()) {
        return true;
    }
    if (submission.mirror_pages_incomplete) {
        return false;
    }
    if (submission.late_copies) {
        submission.late_copies->Wait();
    }
    // One recorded command for all runs: a copy per pair of buffers, a region per run.
    struct Copies {
        VkBuffer from;
        VkBuffer target;
        std::vector<VkBufferCopy> regions;
    };
    std::vector<Copies> copies;
    for (const auto& page : submission.mirror_pages) {
        PoolMirror* const mirror = GetMirror(page.pool, page.pool_size);
        const u64 begin = page.offset;
        if (!mirror || begin > mirror->size || page.bytes > mirror->size - begin) {
            return false;
        }
        const auto staging = UploadStream({page.data, page.bytes});
        if (!staging.first) {
            return false;
        }
        if (copies.empty() || copies.back().from != staging.first ||
            copies.back().target != *mirror->buffer) {
            copies.push_back({staging.first, *mirror->buffer, {}});
        }
        copies.back().regions.push_back({staging.second, begin, page.bytes});
        ++pool_mirror_pages;
    }
    Record([copies = std::move(copies)](vk::CommandBuffer cmdbuf) {
        for (const auto& copy : copies) {
            cmdbuf.CopyBuffer(copy.from, copy.target, copy.regions);
        }
    });
    Barrier();
    return true;
}

BananzaNvn2Replay::BufferRef BananzaNvn2Replay::MirrorUpload(
    const Submission& submission, const Core::BananzaNvn2Commands::Submission::BufferBytes& entry,
    u64 size) {
    constexpr u64 PageBits = 16;
    constexpr u64 PageSize = 1ULL << PageBits;
    constexpr u64 MaxMirrorBytes = 3ULL << 30;
    const u64 offset = entry.fixed_offset;
    // The offset is the game's own: it keeps the console's alignments, which are not finer
    // than this device's.
    if (!entry.fixed_pool_size || !size || !submission.guest_memory) {
        return {};
    }
    auto& mirror = pool_mirrors[entry.fixed_pool];
    if (!mirror.buffer) {
        if (pool_mirror_bytes + entry.fixed_pool_size > MaxMirrorBytes) {
            return {};
        }
        mirror.buffer = allocator.CreateBuffer(
            VkBufferCreateInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = entry.fixed_pool_size,
                .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            },
            MemoryUsage::DeviceLocal);
        mirror.size = entry.fixed_pool_size;
        mirror.synced.assign((mirror.size + PageSize - 1) >> PageBits, 0);
        pool_mirror_bytes += mirror.size;
        LOG_INFO(Render_Vulkan,
                 "NVN 2 pool mirror: pool serial {} of {} KiB mirrored; {} pools, {} MiB in all",
                 entry.fixed_pool, mirror.size >> 10, pool_mirrors.size(),
                 pool_mirror_bytes >> 20);
    }
    if (offset > mirror.size || size > mirror.size - offset) {
        return {};
    }
    bool copied = false;
    for (u64 page = offset >> PageBits; page <= (offset + size - 1) >> PageBits; ++page) {
        if (mirror.synced[page] > entry.fixed_stamp) {
            continue;
        }
        const u64 begin = page << PageBits;
        const u64 bytes = std::min(PageSize, mirror.size - begin);
        const u8* const source =
            submission.guest_memory->GetSpan(entry.fixed_pool_guest + begin, bytes);
        if (!source) {
            return {};
        }
        const auto staging = UploadStream({source, bytes});
        if (!staging.first) {
            return {};
        }
        Record([from = staging.first, from_offset = staging.second, target = *mirror.buffer, begin,
                bytes](vk::CommandBuffer cmdbuf) {
            const VkBufferCopy copy{from_offset, begin, bytes};
            cmdbuf.CopyBuffer(from, target, copy);
        });
        mirror.synced[page] = entry.fixed_stamp + 1;
        ++pool_mirror_pages;
        copied = true;
    }
    if (copied) {
        run_stats.DrawCut(BananzaNvn2RunStats::Cut::MirrorPage);
        Barrier();
    } else {
        run_stats.Count(BananzaNvn2RunStats::Note::RangeMirror);
    }
    return {*mirror.buffer, offset};
}

BananzaNvn2Replay::BufferRef BananzaNvn2Replay::FixedUpload(std::span<const u8> bytes, u64 pool,
                                                           u64 stamp) {
    constexpr u64 ChunkSize = 32ULL << 20;
    constexpr u64 MaxChunks = 24;
    static const bool host_chunks = std::getenv("EDEN_NX2_NVN_FIXED_HOST") != nullptr;
    if (bytes.size() > ChunkSize / 4) {
        return {};
    }
    const Range key{reinterpret_cast<u64>(bytes.data()), bytes.size()};
    const auto found = fixed_cache.find(key);
    if (found != fixed_cache.end() && found->second.pool == pool && found->second.stamp == stamp) {
        // Now and then compare a kept range with guest memory: a change without a flush would
        // mean the game does not follow the pool's flush rule.
        bool same = true;
        if ((key.first / 64 + replayed_submissions) % 256 == 0 && fixed_check_budget >= bytes.size()) {
            fixed_check_budget -= bytes.size();
            ++fixed_checks;
            same = Common::CityHash64(reinterpret_cast<const char*>(bytes.data()), bytes.size()) ==
                   found->second.hash;
            if (!same && ++fixed_mismatches <= 16) {
                LOG_WARNING(Render_Vulkan,
                            "NVN 2 kept range of {} bytes changed without a flush (pool serial {}, "
                            "stamp {})",
                            bytes.size(), pool, stamp);
            }
        }
        if (same) {
            ++fixed_hits;
            return found->second.buffer;
        }
    }
    if (fixed_chunks.empty() ||
        Common::AlignUp(fixed_chunks.back().used,
                        std::max(upload_alignment, device_limits.nonCoherentAtomSize)) +
                bytes.size() > ChunkSize) {
        if (fixed_chunks.size() >= MaxChunks) {
            // Start over; the old chunks live until this submission's work is done.
            for (auto& chunk : fixed_chunks) {
                uploads.push_back(std::move(chunk.buffer));
            }
            fixed_chunks.clear();
            fixed_cache.clear();
        }
        fixed_chunks.push_back({allocator.CreateBuffer(
            VkBufferCreateInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = ChunkSize,
                .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            },
            host_chunks ? MemoryUsage::Upload : MemoryUsage::DeviceLocal)});
    }
    auto& chunk = fixed_chunks.back();
    const u64 alignment = std::max(upload_alignment, device_limits.nonCoherentAtomSize);
    const u64 offset = Common::AlignUp(chunk.used, alignment);
    if (offset + bytes.size() > ChunkSize) {
        return {};
    }
    if (host_chunks) {
        auto mapped = chunk.buffer.Mapped().subspan(offset, bytes.size());
        std::memcpy(mapped.data(), bytes.data(), bytes.size());
        chunk.buffer.Flush(offset, bytes.size());
    } else {
        // The chunk is device memory: the bytes go through the upload stream once.
        const auto staging = UploadStream(bytes);
        if (!staging.first) {
            return {};
        }
        Record([source = staging.first, source_offset = staging.second, target = *chunk.buffer,
                offset, size = bytes.size()](vk::CommandBuffer cmdbuf) {
            const VkBufferCopy copy{source_offset, offset, size};
            cmdbuf.CopyBuffer(source, target, copy);
        });
        Barrier();
    }
    chunk.used = offset + bytes.size();
    ++fixed_uploads;
    fixed_upload_bytes += bytes.size();
    const BufferRef buffer{*chunk.buffer, offset};
    fixed_cache.insert_or_assign(
        key, FixedEntry{buffer, pool, stamp,
                        Common::CityHash64(reinterpret_cast<const char*>(bytes.data()),
                                           bytes.size())});
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
            NoteStorageRead(entry->second, offset, range.second);
            return entry;
        }
    }
    return storage_uploads.end();
}

void BananzaNvn2Replay::NoteStorageRead(const StorageUpload& upload, u64 offset, u64 size) {
    // Whether the bytes are fresh is decided where the draw is issued.
    draw_storage_reads.push_back({&upload, offset, size});
}

BananzaNvn2Replay::StorageSpan* BananzaNvn2Replay::BoundSpan(Range range) {
    if (!storage_spans_disjoint) {
        return nullptr;
    }
    // The spans do not overlap: the one that starts at or before the range.
    const auto next = storage_spans.upper_bound({range.first, std::numeric_limits<u64>::max()});
    if (next == storage_spans.begin()) {
        return nullptr;
    }
    auto& span = std::prev(next)->second;
    const u64 offset = range.first - span.range.first;
    if (offset >= span.range.second || range.second > span.range.second - offset) {
        return nullptr;
    }
    // A span filled in one piece holds every byte in it; another one only its ranges.
    if (!span.host && !std::ranges::binary_search(span.snapshots, range)) {
        return nullptr;
    }
    return &span;
}

std::pair<VkBuffer, u64> BananzaNvn2Replay::UploadBufferRange(const Submission& submission,
                                                            Range range) {
    // A range the GPU never writes is in no storage upload: UploadRange has it in VRAM.
    static const bool fixed_cache_on = std::getenv("EDEN_NX2_NVN_NO_FIXED_CACHE") == nullptr;
    BoundRange bound;
    if (const auto [known, fresh] = bound_ranges.try_emplace(range); fresh) {
        const auto* const snapshot = fixed_cache_on ? submission.FindSnapshot(range) : nullptr;
        bound.fixed = snapshot && snapshot->fixed_pool;
        bound.span = bound.fixed ? nullptr : BoundSpan(range);
        known->second = bound;
    } else {
        bound = known->second;
    }
    if (bound.fixed) {
        return UploadRange(submission, range);
    }
    if (const auto* span = bound.span) {
        // No other span holds these bytes: its buffer, or a new one.
        if (span->upload) {
            const u64 offset = range.first - span->range.first;
            NoteStorageRead(*span->upload, offset, range.second);
            return {span->upload->buffer, span->upload->base + offset};
        }
        if (const auto [upload, offset] = CreateStorageUpload(submission, range); upload) {
            return {upload->buffer, upload->base + offset};
        }
        return UploadRange(submission, range);
    }
    if (const auto storage = FindStorageRange(range); storage != storage_uploads.end()) {
        return {storage->second.buffer,
                storage->second.base + range.first - storage->first.first};
    }
    if (const auto [upload, offset] = CreateStorageUpload(submission, range); upload) {
        return {upload->buffer, upload->base + offset};
    }
    return UploadRange(submission, range);
}

void BananzaNvn2Replay::CollectStorageSpans(const Submission& submission) {
    storage_spans.clear();
    bound_ranges.clear();
    const u64 alignment =
        device_limits.minStorageBufferOffsetAlignment;
    StorageSpan* span{};
    for (const auto& [range, stored] : submission.storage_addresses) {
        const auto [address, size] = range;
        const u64 guest = stored.guest;
        // Read from guest memory: not a copy of the scan's.
        const u8* const direct = stored.owned ? nullptr : stored.data;
        // Only share Vulkan storage when both addresses describe the same aligned bytes.
        // Slices packed one after the other share it too.
        constexpr u64 MergeGap = 4096;
        constexpr u64 MaxSpanBytes = 32ULL << 20;
        if (!span || address > span->range.first + span->range.second + MergeGap ||
            address - span->range.first + size > MaxSpanBytes ||
            address - span->range.first != guest - span->guest_address ||
            (address - span->range.first) % alignment) {
            span = &storage_spans.emplace(range, StorageSpan{range, guest, {range}}).first->second;
            span->bytes.push_back(stored);
            span->host = direct;
            span->pool = stored.pool;
            span->stamp = stored.stamp;
            continue;
        }
        if (stored.pool != span->pool) {
            span->pool = 0;
        }
        span->stamp = std::max(span->stamp, stored.stamp);
        span->range.second = std::max(span->range.second, address - span->range.first + size);
        span->snapshots.push_back(range);
        span->bytes.push_back(stored);
        if (!direct || !span->host || direct != span->host + (address - span->range.first)) {
            span->host = nullptr;
        }
    }
    static const bool use_maps = std::getenv("EDEN_NX2_NVN_STORAGE_MAPS") != nullptr;
    storage_span_of.clear();
    storage_spans_disjoint = !use_maps;
    storage_guest_disjoint = !use_maps;
    if (use_maps) {
        return;
    }
    std::vector<Range> guest_ranges;
    guest_ranges.reserve(storage_spans.size());
    u64 end{};
    for (auto& [key, entry] : storage_spans) {
        storage_spans_disjoint &= entry.range.first >= end;
        end = std::max(end, entry.range.first + entry.range.second);
        guest_ranges.emplace_back(entry.guest_address, entry.guest_address + entry.range.second);
    }
    std::ranges::sort(guest_ranges);
    for (size_t i = 1; i < guest_ranges.size(); ++i) {
        storage_guest_disjoint &= guest_ranges[i].first >= guest_ranges[i - 1].second;
    }
}

void BananzaNvn2Replay::AddBufferUpdate(u64 address, std::span<const u8> bytes) {
    while (!bytes.empty()) {
        const u64 page = Common::AlignDown(address, BufferUpdatePageSize);
        const size_t size = std::min<u64>(bytes.size(), BufferUpdatePageSize - (address - page));
        buffer_updates[page].push_back({address, bytes.first(size)});
        bytes = bytes.subspan(size);
        address += size;
    }
}

bool BananzaNvn2Replay::HasBufferUpdates(Range range) const {
    const auto page = buffer_updates.lower_bound(Common::AlignDown(range.first, BufferUpdatePageSize));
    return page != buffer_updates.end() && page->first < range.first + range.second;
}

void BananzaNvn2Replay::ApplyBufferUpdates(u64 address, std::span<u8> bytes) const {
    const u64 range_end = address + bytes.size();
    for (auto page = buffer_updates.lower_bound(Common::AlignDown(address, BufferUpdatePageSize));
         page != buffer_updates.end() && page->first < range_end; ++page) {
        for (const auto& write : page->second) {
            const auto begin = std::max(address, write.address);
            const auto end = std::min(range_end, write.address + write.bytes.size());
            if (begin < end) {
                std::memcpy(bytes.data() + (begin - address),
                            write.bytes.data() + (begin - write.address), end - begin);
            }
        }
    }
}

std::pair<BananzaNvn2Replay::StorageUpload*, u64> BananzaNvn2Replay::WritableStorage(
    const Submission& submission, Range range) {
    const auto align = device_limits.minStorageBufferOffsetAlignment;
    if (const auto* span = BoundSpan(range)) {
        if (!span->upload) {
            return CreateStorageUpload(submission, range);
        }
        const u64 offset = range.first - span->range.first;
        NoteStorageRead(*span->upload, offset, range.second);
        return offset % align == 0 ? std::pair{span->upload, offset}
                                   : std::pair<StorageUpload*, u64>{};
    }
    if (const auto found = FindStorageRange(range); found != storage_uploads.end()) {
        const u64 offset = range.first - found->first.first;
        return offset % align == 0 ? std::pair{&found->second, offset}
                                   : std::pair<StorageUpload*, u64>{};
    }
    return CreateStorageUpload(submission, range);
}

std::pair<BananzaNvn2Replay::StorageUpload*, u64> BananzaNvn2Replay::CreateStorageUpload(
    const Submission& submission, Range range) {
    run_stats.DrawCut(BananzaNvn2RunStats::Cut::StorageCreate);
    StorageSpan* bound{};
    if (storage_spans_disjoint) {
        bound = BoundSpan(range);
        if (!bound) {
            return {};
        }
    } else {
        if (!submission.storage_addresses.contains(range)) {
            return {};
        }
        auto entry = storage_spans.upper_bound({range.first, std::numeric_limits<u64>::max()});
        if (entry == storage_spans.begin()) {
            return {};
        }
        bound = &std::prev(entry)->second;
    }
    auto& span = *bound;
    const u64 offset = range.first - span.range.first;
    if (offset >= span.range.second || range.second > span.range.second - offset ||
        (!storage_spans_disjoint && !std::ranges::binary_search(span.snapshots, range))) {
        return {};
    }
    if (profile_replay) {
        const std::pair<u64, u64> state{span.pool, span.stamp};
        const auto [known, fresh] =
            span_flush_states.try_emplace({span.guest_address, span.range.second}, state);
        span_state_bytes[!span.pool ? 3 : fresh ? 2 : known->second == state ? 0 : 1] +=
            span.range.second;
        known->second = state;
    }
    const u64 end = span.guest_address + span.range.second;
    // Two spans over the same guest bytes get one buffer: the first one made.
    auto next = storage_guest_ends.end();
    if (!storage_guest_disjoint) {
        next = storage_guest_ends.lower_bound(span.guest_address);
        if ((next != storage_guest_ends.end() && next->first < end) ||
            (next != storage_guest_ends.begin() &&
             std::prev(next)->second > span.guest_address)) {
            return {};
        }
    }
    static const bool per_span = std::getenv("EDEN_NX2_NVN_STORAGE_PER_SPAN") != nullptr;
    const u64 aligned = Common::AlignUp(span.range.second, upload_alignment);
    // A span the game must flush after writing keeps its buffer from replay to replay: made
    // again with the same flush state, it holds the bytes as the GPU left them.
    static const bool keep_storage = std::getenv("EDEN_NX2_NVN_NO_KEPT_STORAGE") == nullptr;
    const bool keepable = keep_storage && span.pool && span.host && submission.guest_memory;
    BananzaNvn2KeptStorage::Buffer kept;
    if (keepable) {
        kept = kept_storage.Find(span.guest_address, span.range.second, span.pool, span.stamp,
                                 replay_count);
    }
    const bool reused = kept != nullptr;
    if (!reused) {
        // Whatever was kept over these guest bytes is not their newest state any more.
        kept_storage.Evict(span.guest_address, span.range.second, evicted_kept);
    }
    if (keepable && !reused) {
        kept = std::make_shared<vk::Buffer>(allocator.CreateBuffer({
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = aligned,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        }, MemoryUsage::Download));
        kept_storage.Keep(span.guest_address, span.range.second, span.pool, span.stamp, kept,
                          replay_count);
    }
    if (!kept && (per_span || storage_arenas.empty() ||
        storage_arenas.back().capacity - storage_arenas.back().used < aligned)) {
        const u64 capacity = per_span ? std::bit_ceil(aligned)
                                      : std::max(StorageArenaBytes, std::bit_ceil(aligned));
        auto& free = free_storage_buffers[capacity];
        vk::Buffer buffer;
        if (free.empty()) {
            buffer = allocator.CreateBuffer({
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = capacity,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            }, MemoryUsage::Download);
            ++storage_creates;
        } else {
            buffer = std::move(free.back());
            free.pop_back();
            free_storage_bytes -= capacity;
            ++storage_reuses;
        }
        storage_arenas.push_back({std::move(buffer), capacity});
    }
    u64 base{};
    std::span<u8> host;
    VkBuffer host_buffer{};
    if (kept) {
        host = kept->Mapped().first(span.range.second);
        host_buffer = **kept;
        used_kept.push_back(kept);
    } else {
        auto& arena = storage_arenas.back();
        base = arena.used;
        arena.used += aligned;
        host = arena.buffer.Mapped().subspan(base, span.range.second);
        host_buffer = *arena.buffer;
    }
    storage_bytes += span.range.second;
    std::span<u8> mapped = host;
    VkBuffer bound_buffer = host_buffer;
    u64 bound_base = base;
    static const bool device_storage = std::getenv("EDEN_NX2_NVN_NO_DEVICE_STORAGE") == nullptr;
    // A kept span is filled in its host buffer and moved below.
    const bool in_device = device_storage && storage_for_fragment && !kept;
    if (in_device) {
        // A fragment shader touches the span for every pixel: keep it next to the GPU.
        u8* device_mapped{};
        std::tie(bound_buffer, bound_base) = AllocateDeviceStorage(aligned, device_mapped);
        mapped = {device_mapped, span.range.second};
        ++device_spans;
        device_span_bytes += span.range.second;
    }
    static const bool serial_copies = std::getenv("EDEN_NX2_NVN_SERIAL_COPIES") != nullptr;
    const bool fill_later = !serial_copies && !HasBufferUpdates(span.range) && !reused;
    if (reused) {
        kept_reused_bytes += span.range.second;
    } else if (kept) {
        kept_filled_bytes += span.range.second;
        kept_fills.emplace_back(kept, span.range.second);
    }
    if (span.host && !reused) {
        AwaitGuestWrites(span.host, span.range.second);
        // One run of guest memory: one copy, with nothing to look up.
        if (fill_later) {
            copy_jobs.push_back({mapped.data(), span.host, span.range.second});
            copy_job_bytes += span.range.second;
        } else {
            std::memcpy(mapped.data(), span.host, span.range.second);
        }
    }
    for (size_t index = 0; index < span.snapshots.size() && !span.host; ++index) {
        const auto& snapshot = span.snapshots[index];
        // A fill that runs later waits for the bytes itself (DispatchCopies, WaitCopies).
        std::span<const u8> bytes;
        if (const auto& stored = span.bytes[index]; stored.data) {
            if (!fill_later && stored.late && submission.late_copies) {
                submission.late_copies->Wait();
            }
            if (!stored.owned) {
                AwaitGuestWrites(stored.data, snapshot.second);
            }
            bytes = {stored.data, snapshot.second};
        } else {
            bytes = submission.BufferSnapshot(snapshot, !fill_later);
        }
        u8* const dest = mapped.data() + (snapshot.first - span.range.first);
        if (fill_later) {
            copy_jobs.push_back({dest, bytes.data(), bytes.size()});
            copy_job_bytes += bytes.size();
        } else {
            std::memcpy(dest, bytes.data(), bytes.size());
        }
    }
    if (!fill_later && !reused) {
        ApplyBufferUpdates(span.range.first, mapped);
    }
    const auto upload = storage_uploads.emplace(
        span.range, StorageUpload{bound_buffer, bound_base, mapped.data(), span.guest_address,
                                  host.data(), in_device ? host_buffer : VkBuffer{}, base,
                                  span.range.second});
    if (in_device) {
        device_uploads.push_back(&upload.first->second);
    }
    if (reused && HasBufferUpdates(span.range)) {
        // What the replay changed in these bytes before the span was made (indirect resets)
        // goes into the kept buffer on the GPU, after what the GPU wrote there before.
        std::vector<u8> blob;
        std::vector<VkBufferCopy> regions;
        const u64 first = span.range.first;
        const u64 last = first + span.range.second;
        for (auto page = buffer_updates.lower_bound(Common::AlignDown(first, BufferUpdatePageSize));
             page != buffer_updates.end() && page->first < last; ++page) {
            for (const auto& write : page->second) {
                const u64 begin = std::max(first, write.address);
                const u64 stop = std::min(last, write.address + write.bytes.size());
                if (begin >= stop) {
                    continue;
                }
                regions.push_back({blob.size(), bound_base + (begin - first), stop - begin});
                blob.insert(blob.end(), write.bytes.begin() + (begin - write.address),
                            write.bytes.begin() + (stop - write.address));
            }
        }
        if (!regions.empty()) {
            const VkBuffer source = Upload(blob, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            Barrier();
            Record([source, target = bound_buffer,
                    regions = std::move(regions)](vk::CommandBuffer cmdbuf) {
                for (const VkBufferCopy& region : regions) {
                    cmdbuf.CopyBuffer(source, target, region);
                }
            });
            Barrier();
        }
    }
    if (kept && device_storage && storage_for_fragment) {
        MoveStorageToDevice(upload.first->second);
    }
    if (fill_later && copy_job_bytes >= (512ULL << 10)) {
        DispatchCopies();
    }
    // An exact range match now resolves to the writable allocation.
    for (size_t stream = 0; stream < vertex_uploads.size(); ++stream) {
        if (vertex_buffers[stream] == span.range) {
            vertex_uploads[stream] = {};
        }
    }
    if (!storage_guest_disjoint) {
        storage_guest_ends.emplace_hint(next, span.guest_address, end);
    }
    span.upload = &upload.first->second;
    storage_upload_of.insert_or_assign(span.range, span.upload);
    uploaded_ranges.erase(span.range);
    ForgetUploads(span.range.first, span.range.first + span.range.second);
    return {&upload.first->second, offset};
}

void BananzaNvn2Replay::WriteBackStorage(const Submission& submission) {
    if (!submission.guest_memory) {
        return;
    }
    for (const auto& arena : storage_arenas) {
        arena.buffer.Invalidate();
    }
    for (auto& [range, upload] : storage_uploads) {
        if (!upload.dirty) {
            continue;
        }
        upload.dirty = false;
        // Spans are keyed by their first snapshot, which starts where the upload starts.
        const auto span = storage_spans.lower_bound({range.first, 0});
        const bool found = span != storage_spans.end() && span->first.first == range.first;
        const auto [written, compared] = WriteChangedBytes(
            *submission.guest_memory, upload.guest_address, range.first,
            std::span<const u8>{upload.readback, upload.written_bytes},
            found ? std::span<const Range>{span->second.snapshots} : std::span<const Range>{},
            submission, found ? span->second.host : nullptr, upload.written);
        upload.written.clear();
        writeback_bytes.fetch_add(written, std::memory_order_relaxed);
        writeback_compared.fetch_add(compared, std::memory_order_relaxed);
    }
}

bool BananzaNvn2Replay::UploadTexture(Image& image, const Texture& texture,
                                     std::span<const u8> data) {
    const auto fill = BananzaNvn2TextureFill::Plan(texture, image.layers, image.depth,
                                                   image.aspects, data.size());
    if (!fill) {
        return false;
    }
    auto upload = CreateWrappedBuffer(allocator, fill->Size(), MemoryUsage::Upload);
    fill->Run(data, upload.Mapped());
    FillImage(image, std::move(upload), fill->Copies());
    return true;
}

void BananzaNvn2Replay::FillImage(Image& image, vk::Buffer upload,
                                  std::vector<VkBufferImageCopy> copies) {
    upload.Flush();
    Initialize(image);
    Barrier();
    Record([buffer = *upload, target = *image.image, copies = std::move(copies)](
                         vk::CommandBuffer cmdbuf) {
        cmdbuf.CopyBufferToImage(buffer, target, VK_IMAGE_LAYOUT_GENERAL, copies);
    });
    uploads.push_back(std::move(upload));
    ++image.write_version;
}

void BananzaNvn2Replay::StartTextureFills(const Submission& submission) {
    static const bool off = std::getenv("EDEN_NX2_NVN_NO_TEXTURE_WORKERS") != nullptr;
    if (off) {
        return;
    }
    for (const auto& [serial, bytes] : submission.texture_snapshots) {
        const auto info = submission.texture_infos.find(serial);
        const auto source = submission.texture_sources.find(serial);
        if (info == submission.texture_infos.end() ||
            source == submission.texture_sources.end()) {
            continue;
        }
        // The same tests as where a draw samples the texture.
        Image* const image = FindImage(info->second, true);
        if (!image || image->gpu_written ||
            (image->aspects != VK_IMAGE_ASPECT_COLOR_BIT && image->format != VK_FORMAT_D16_UNORM &&
             image->format != VK_FORMAT_D32_SFLOAT) ||
            (image->initialized && image->cpu_serial == serial &&
             image->cpu_hash == source->second.hash)) {
            continue;
        }
        auto fill = BananzaNvn2TextureFill::Plan(info->second, image->layers, image->depth,
                                                 image->aspects, bytes.size());
        if (!fill) {
            continue;
        }
        auto pending = std::make_unique<PendingFill>();
        pending->fill = std::move(*fill);
        pending->buffer = CreateWrappedBuffer(allocator, pending->fill.Size(), MemoryUsage::Upload);
        texture_workers.QueueWork([entry = pending.get(), data = std::span<const u8>{bytes}] {
            entry->fill.Run(data, entry->buffer.Mapped());
            entry->done.store(true, std::memory_order_release);
            entry->done.notify_all();
        });
        texture_fills.emplace(serial, std::move(pending));
        ++worker_fills;
    }
}

bool BananzaNvn2Replay::TakeTextureFill(Image& image, u64 serial) {
    const auto found = texture_fills.find(serial);
    if (found == texture_fills.end()) {
        return false;
    }
    const auto pending = std::move(found->second);
    texture_fills.erase(found);
    pending->done.wait(false, std::memory_order_acquire);
    FillImage(image, std::move(pending->buffer), pending->fill.Copies());
    return true;
}

void BananzaNvn2Replay::FinishTextureFills(const Submission& submission) {
    // The bytes are not sent again, so a texture no draw of this replay sampled is filled
    // all the same.
    for (auto& [serial, pending] : texture_fills) {
        pending->done.wait(false, std::memory_order_acquire);
        const auto info = submission.texture_infos.find(serial);
        const auto source = submission.texture_sources.find(serial);
        Image* const image =
            info != submission.texture_infos.end() ? FindImage(info->second, false) : nullptr;
        if (image && !image->gpu_written && source != submission.texture_sources.end()) {
            FillImage(*image, std::move(pending->buffer), pending->fill.Copies());
            image->cpu_hash = source->second.hash;
            image->cpu_serial = serial;
            ++texture_uploads;
        }
    }
    texture_fills.clear();
}

VkImage BananzaNvn2Replay::PrepareDepthComparisonImage(const Image& source, VkFormat depth_format) {
    const auto found = depth_comparison_images.find(&source);
    if (found != depth_comparison_images.end() && found->second.write_version == source.write_version) {
        return *found->second.image;
    }
    const auto features = FormatFeatures(depth_format);
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
    if (bytes > (64ULL << 20)) {
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
        const u64 allocation_bytes = image.GetMemoryBlock().size;
        if (!ReserveImageBytes(allocation_bytes)) {
            return VK_NULL_HANDLE;
        }
        entry = depth_comparison_images.emplace(&source, DepthComparisonImage{std::move(image)}).first;
        entry->second.allocation_bytes = allocation_bytes;
        image_bytes += allocation_bytes;
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
    Record([src = *source.image, target, buffer = *scratch,
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
        Record([handle = *image, subresource](vk::CommandBuffer cmdbuf) {
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
        const auto bytes = ReadSnapshot(submission, uniform_buffers[stage][binding.cbuf_index - 3]);
        if (offset > bytes.size() || bytes.size() - offset < 4) {
            NoteSkip("texture handle buffer range missing at submit");
            return std::nullopt;
        }
        u32_le word;
        std::memcpy(&word, bytes.data() + offset, 4);
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
    const u64 view_kind = u64{static_cast<u32>(binding.type)} * 4 +
                          u64{binding.is_depth} * 2 + u64{binding.is_integer};
    const SampledBindingKey binding_key{texture_pool, sampler_pool,
                                        (view_kind << 32) | static_cast<u32>(handle)};
    const bool capture_sampled = frame_dump.CapturingSampled();
    if (!capture_sampled) {
        if (const auto found = sampled_bindings.find(binding_key); found != sampled_bindings.end()) {
            const auto& entry = found->second;
            if (std::ranges::find(destinations, entry.image) == destinations.end()) {
                entry.image->last_present = present_count;
                draw_sampled.push_back(entry.image);
                if (PrepareSampledImage(*entry.image, *entry.texture, submission)) {
                    TraceImage("sampled", *entry.texture, bound_program->serial);
                    run_stats.Count(BananzaNvn2RunStats::Note::TextureHit);
                    return entry.descriptor;
                }
            }
        }
    }
    run_stats.DrawCut(BananzaNvn2RunStats::Cut::TextureSlow);
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
            device_features.imageCubeArray) {
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
    draw_sampled.push_back(image);
    if (!PrepareSampledImage(*image, tex, submission)) {
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
    if (capture_sampled && (d.width >= 900 || aspect == VK_IMAGE_ASPECT_DEPTH_BIT) &&
        !image->volume &&
        (color || aspect == VK_IMAGE_ASPECT_DEPTH_BIT) &&
        captured_inputs.emplace(address, first_level, first_layer).second) {
        Barrier();
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
        !(FormatFeatures(vk_format) & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT)) {
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
    // Sampling a bound target the draw does not write needs no copy.
    for (size_t i = 0; i < destinations.size(); ++i) {
        feedback |= overlaps(i) && ((draw_color_writes >> i) & 1);
    }
    auto* views = &image->sampled_views;
    if (feedback && !comparison_copy) {
        const auto [found, inserted] = feedback_images.try_emplace(image);
        auto& copy = found->second;
        if (inserted) {
            auto feedback_image = allocator.CreateImage(VkImageCreateInfo{
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
            const u64 allocation_bytes = feedback_image.GetMemoryBlock().size;
            if (!ReserveImageBytes(allocation_bytes)) {
                feedback_images.erase(found);
                NoteSkip("feedback image exceeds replay budget");
                return std::nullopt;
            }
            copy.image = std::move(feedback_image);
            copy.allocation_bytes = allocation_bytes;
            image_bytes += allocation_bytes;
        }
        sampled_image = *copy.image;
        views = &copy.views;
        Barrier();
        Record([source = *image->image, sampled_image, inserted, first_level,
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
    const bool point_depth = aspect == VK_IMAGE_ASPECT_DEPTH_BIT && !binding.is_depth;
    const bool use_reduction = desc.reduction && !binding.is_depth &&
        device.IsExtSamplerFilterMinmaxSupported() &&
        (FormatFeatures(vk_format) & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_MINMAX_BIT);
    const bool anisotropic = desc.anisotropy > 1.0f && desc.mag_filter == 1 &&
                             (desc.min_filter & 1) && !point_depth && !binding.is_depth &&
                             !use_reduction && device_features.samplerAnisotropy;
    const auto& limits = device_limits;
    VkBorderColor border = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    if (std::ranges::find(desc.wrap, 5U) != desc.wrap.end() && desc.border_kind == 0) {
        const auto rgba = std::bit_cast<std::array<float, 4>>(desc.border);
        if (rgba == std::array{1.0f, 1.0f, 1.0f, 1.0f}) {
            border = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        } else if (rgba == std::array{0.0f, 0.0f, 0.0f, 1.0f}) {
            border = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        }
    }
    const VkSamplerCreateInfo sampler_info{
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
    };
    // The fields from flags to unnormalizedCoordinates are sixteen 32-bit words.
    static_assert(offsetof(VkSamplerCreateInfo, unnormalizedCoordinates) -
                      offsetof(VkSamplerCreateInfo, flags) == 15 * sizeof(u32));
    std::array<u32, 17> sampler_key{};
    std::memcpy(sampler_key.data(), &sampler_info.flags, 16 * sizeof(u32));
    sampler_key[16] = use_reduction ? desc.reduction : 0;
    auto host_sampler = sampler_cache.find(sampler_key);
    if (host_sampler == sampler_cache.end()) {
        host_sampler = sampler_cache.emplace(sampler_key,
                                             device.GetLogical().CreateSampler(sampler_info)).first;
    }
    const VkDescriptorImageInfo descriptor{*host_sampler->second, *sampled_view->second,
                                            VK_IMAGE_LAYOUT_GENERAL};
    // Feedback and comparison copies depend on writes within this submission.
    if (!capture_sampled && !feedback && !comparison_copy) {
        sampled_bindings.try_emplace(binding_key, SampledBinding{descriptor, image, &tex});
    }
    return descriptor;
}

bool BananzaNvn2Replay::PrepareSampledImage(Image& image, const Texture& texture,
                                           const Submission& submission) {
    if (image.gpu_written) {
        if (image.confirmed_replay_submission != submission.sequence &&
            image.stateless_submission != submission.sequence) {
            if (const auto state = submission.texture_replay_states.find(texture.serial);
                state != submission.texture_replay_states.end()) {
                state->second->gpu_written.store(true, std::memory_order_relaxed);
                image.confirmed_replay_submission = submission.sequence;
            } else {
                image.stateless_submission = submission.sequence;
            }
        }
        return image.initialized;
    }
    if (image.aspects != VK_IMAGE_ASPECT_COLOR_BIT && image.format != VK_FORMAT_D16_UNORM &&
        image.format != VK_FORMAT_D32_SFLOAT) {
        return image.initialized;
    }
    if (image.checked_submission != submission.sequence) {
        const auto source = submission.texture_sources.find(texture.serial);
        if (source != submission.texture_sources.end() &&
            (!image.initialized || image.cpu_serial != texture.serial ||
             image.cpu_hash != source->second.hash)) {
            const auto upload_start = std::chrono::steady_clock::now();
            const auto data = submission.texture_snapshots.find(texture.serial);
            std::vector<u8> fetched;
            std::span<const u8> bytes;
            if (data != submission.texture_snapshots.end()) {
                bytes = data->second;
            } else if (submission.guest_memory) {
                // The bytes were sent with an earlier submission and the image is gone since.
                fetched.resize(texture.layout.size);
                bool read = true;
                u64 offset{};
                for (const auto& [guest, size] : source->second.segments) {
                    read = read && offset + size <= fetched.size() &&
                           submission.guest_memory->ReadBlockUnsafe(guest, fetched.data() + offset,
                                                                    size);
                    offset += size;
                }
                if (read && offset == fetched.size()) {
                    bytes = fetched;
                    hash_bytes += fetched.size();
                }
            }
            if (!bytes.empty()) {
                run_stats.DrawCut(BananzaNvn2RunStats::Cut::TextureUpload);
                const bool uploaded = TakeTextureFill(image, texture.serial) ||
                                      UploadTexture(image, texture, bytes);
                upload_ns += static_cast<u64>(
                    (std::chrono::steady_clock::now() - upload_start).count());
                if (!uploaded) {
                    return false;
                }
                image.cpu_hash = source->second.hash;
                image.cpu_serial = texture.serial;
                ++texture_uploads;
            }
        }
        image.checked_submission = submission.sequence;
    }
    return image.initialized && image.cpu_serial == texture.serial;
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

void BananzaNvn2Replay::NoteProgram(const std::shared_ptr<const Program>& program) {
    if (!pipeline_list || !pipeline_list->Uses(program->content_hash)) {
        return;
    }
    listed_programs.insert_or_assign(program->content_hash, program);
    pipeline_list->ForProgram(
        program->content_hash, [this](u64 hash) { return listed_programs.contains(hash); },
        [this](const BananzaNvn2PipelineList::Entry& entry) { BuildAhead(entry); });
}

void BananzaNvn2Replay::BuildAhead(const BananzaNvn2PipelineList::Entry& entry) {
    const PipelineKey key{entry.programs[0], entry.programs[1], entry.programs[2],
                          entry.formats, entry.topology, entry.vertex_attributes,
                          entry.vertex_strides, entry.vertex_divisors, entry.color_state,
                          entry.raster};
    const auto [slot, inserted] = pipelines.try_emplace(key);
    if (!inserted) {
        return;
    }
    const auto program = [this](u64 hash) {
        return hash ? listed_programs.at(hash) : std::shared_ptr<const Program>{};
    };
    PipelineInputs inputs{program(entry.programs[0]), program(entry.programs[1]),
                          program(entry.programs[2]), entry.vertex_attributes,
                          entry.vertex_strides, entry.vertex_divisors, entry.color_state,
                          entry.color_formats, entry.formats, entry.color_count, entry.topology,
                          entry.raster};
    ahead_pipelines.insert(&slot->second);
    ++ahead_queued;
    pipeline_builder.Queue(&slot->second, [this, inputs = std::move(inputs)] {
        return BuildPipeline(inputs);
    });
}

void BananzaNvn2Replay::CollectPipelines() {
    pipeline_builder.Collect([this](std::optional<Pipeline>* slot, BuiltPipeline&& built) {
        if (building_pipelines.erase(slot)) {
            ++worker_pipelines;
            TakePipeline(*slot, built);
        } else if (ahead_pipelines.erase(slot)) {
            // No draw waits for it, so its skip reasons are not a draw's.
            *slot = std::move(built.pipeline);
            ++ahead_built;
        } else {
            // A draw needed it first and built its own.
            ++ahead_dropped;
        }
    });
}

BananzaNvn2Replay::BuiltPipeline BananzaNvn2Replay::BuildPipeline(
    const PipelineInputs& inputs) const {
    BuiltPipeline built;
    const auto start = std::chrono::steady_clock::now();
    try {
        built.pipeline = CreatePipeline(inputs, built.skips);
    } catch (const vk::Exception& error) {
        LOG_WARNING(Render_Vulkan, "Bananza NVN 2 graphics pipeline {}: {}",
                    inputs.graphics_program->serial, error.what());
    }
    built.ns = static_cast<u64>((std::chrono::steady_clock::now() - start).count());
    return built;
}

void BananzaNvn2Replay::TakePipeline(std::optional<Pipeline>& slot, BuiltPipeline& built) {
    slot = std::move(built.pipeline);
    ++pipeline_creates;
    pipeline_create_ns += built.ns;
    for (const auto reason : built.skips) {
        NoteSkip(reason);
    }
}

std::optional<BananzaNvn2Replay::Pipeline> BananzaNvn2Replay::CreatePipeline(
    const PipelineInputs& inputs, std::vector<std::string_view>& skips) const {
    // May run on a worker thread: everything it reads of the binding state is in the inputs.
    const auto& graphics_program = inputs.graphics_program;
    const auto& fragment_program = inputs.fragment_program;
    const auto& geometry_program = inputs.geometry_program;
    const auto& vertex_attributes = inputs.vertex_attributes;
    const auto& vertex_strides = inputs.vertex_strides;
    const auto& vertex_divisors = inputs.vertex_divisors;
    const auto& color_state = inputs.color_state;
    const auto& color_formats = inputs.color_formats;
    const auto& formats = inputs.formats;
    const u32 color_count = inputs.color_count;
    const auto topology = inputs.topology;
    const auto& raster = inputs.raster;
    const auto NoteSkip = [&skips](std::string_view reason) { skips.push_back(reason); };
    std::set<std::pair<u64, std::string_view>> draw_diagnostics;
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
        (geometry_program && (!geometry || !device_features.geometryShader)) ||
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
        if (color_formats[i] != BananzaNvn2PipelineList::NoColor) {
            const auto format = FindColorFormat(color_formats[i]);
            if (format && VideoCore::Surface::IsPixelFormatInteger(format->surface)) {
                runtime.color_output_types[i] = attribute_type(color_formats[i]);
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
            const auto features = device_features;
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
        static const bool push_off = std::getenv("EDEN_NX2_NVN_NO_PUSH_DESCRIPTORS") != nullptr;
        result.push_descriptors = !push_off && device.IsKhrPushDescriptorSupported() &&
                                  layout_bindings.size() <= device.MaxPushDescriptors();
        result.set_layout = logical.CreateDescriptorSetLayout({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .flags = result.push_descriptors
                         ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR
                         : VkDescriptorSetLayoutCreateFlags{},
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
    if (!layout_bindings.empty()) {
        std::vector<VkDescriptorUpdateTemplateEntry> entries;
        entries.reserve(layout_bindings.size());
        for (const auto& binding : layout_bindings) {
            entries.push_back({
                .dstBinding = binding.binding,
                .descriptorCount = 1,
                .descriptorType = binding.descriptorType,
                .offset = binding.binding * sizeof(DescriptorData),
            });
        }
        result.update_template = logical.CreateDescriptorUpdateTemplate({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO,
            .descriptorUpdateEntryCount = static_cast<u32>(entries.size()),
            .pDescriptorUpdateEntries = entries.data(),
            .templateType = result.push_descriptors
                                ? VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR
                                : VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET,
            .descriptorSetLayout = *result.set_layout,
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .pipelineLayout = *result.layout,
            .set = 0,
        });
    }
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
    if (polygon > 2 || (polygon < 2 && !device_features.fillModeNonSolid)) {
        NoteSkip("unsupported polygon mode");
        return std::nullopt;
    }
    const bool clockwise = ((raster.polygon & 4) == 0) != raster.origin_lower_left;
    const auto features = device_features;
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
    result.uses_stencil = depth_stencil.stencilTestEnable;
    result.depth_write = depth_stencil.depthWriteEnable;
    std::array<VkPipelineColorBlendAttachmentState, 8> attachments{};
    constexpr std::array blend_ops{VK_BLEND_OP_MAX_ENUM, VK_BLEND_OP_ADD, VK_BLEND_OP_SUBTRACT,
                                   VK_BLEND_OP_REVERSE_SUBTRACT, VK_BLEND_OP_MIN, VK_BLEND_OP_MAX};
    for (u32 slot = 0; slot < color_count; ++slot) {
        auto& attachment = attachments[slot];
        const u32 shader_mask = fs.info ? fs.info->frag_color_components[slot] : 0;
        attachment.colorWriteMask = ((color_state.channel_mask >> (slot * 4)) & 15) & shader_mask;
        result.color_writes |= attachment.colorWriteMask ? 1U << slot : 0U;
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
        const auto uses_constant = [](VkBlendFactor factor) {
            return factor >= VK_BLEND_FACTOR_CONSTANT_COLOR &&
                   factor <= VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
        };
        result.uses_blend_constants |= uses_constant(attachment.srcColorBlendFactor) ||
                                       uses_constant(attachment.dstColorBlendFactor) ||
                                       uses_constant(attachment.srcAlphaBlendFactor) ||
                                       uses_constant(attachment.dstAlphaBlendFactor);
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
    }, device.StaticPipelineCache());
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
    const auto snapshot = submission.BufferSnapshot({list->address, list->bytes});
    if (snapshot.empty()) {
        return false;
    }
    std::vector<std::pair<u64, u64>> records;
    std::vector<u64> changed;
    records.reserve(draw.count);
    changed.reserve(draw.count);
    for (size_t offset = 0; offset < list->bytes; offset += sizeof(u64)) {
        u64_le record;
        std::memcpy(&record, snapshot.data() + offset, sizeof(record));
        if (!record) {
            continue;
        }
        const auto guest = submission.indirect_reset_addresses.find(record);
        if (record % alignof(u32) || guest == submission.indirect_reset_addresses.end() ||
            (record < list->address + list->bytes && list->address < record + sizeof(Reset::Record))) {
            return false;
        }
        records.emplace_back(record, guest->second);
        changed.push_back(record);
    }
    static constexpr std::array<u32_le, 5> reset_words{0, 1, 0, 0, 0};
    const std::span<const u8> reset_bytes{reinterpret_cast<const u8*>(reset_words.data()),
                                         sizeof(reset_words)};
    std::ranges::sort(changed);
    std::vector<std::pair<VkBuffer, VkBufferCopy>> copies_to_storage;
    for (const auto& [record, guest] : records) {
        submission.guest_memory->WriteBlockUnsafe(guest, reset_bytes.data(), reset_bytes.size());
        AddBufferUpdate(record, reset_bytes);
        if (const auto storage = FindStorageRange({record, reset_bytes.size()});
            storage != storage_uploads.end()) {
            copies_to_storage.emplace_back(
                storage->second.buffer,
                VkBufferCopy{0, storage->second.base + record - storage->first.first,
                             reset_bytes.size()});
        }
        reset_indirect_draws.insert(record);
    }
    erase_if(uploaded_ranges, [&](const auto& entry) {
        const auto [start, size] = entry.first;
        const auto next = std::ranges::lower_bound(changed, start);
        return (next != changed.end() && *next < start + size) ||
               (next != changed.begin() && *std::prev(next) + reset_bytes.size() > start);
    });
    vertex_uploads.fill({});
    ++upload_generation;
    if (!copies_to_storage.empty()) {
        const auto source = Upload(reset_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        Barrier();
        bool overlaps{};
        if (!changed.empty()) {
            auto last = changed.begin();
            for (auto next = std::next(last); next != changed.end(); last = next++) {
                overlaps |= *next - *last < reset_bytes.size();
            }
        }
        Record([source, destinations = std::move(copies_to_storage), overlaps](vk::CommandBuffer cmdbuf) {
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
    // Experiment: EDEN_NX2_NVN_DRAW_STRIDE=N replays every Nth draw only (wrong image).
    static const u64 draw_stride = [] {
        const char* const value = std::getenv("EDEN_NX2_NVN_DRAW_STRIDE");
        return value ? std::max<u64>(1, std::strtoull(value, nullptr, 10)) : u64{1};
    }();
    static u64 stride_draws{};
    if (draw_stride > 1 && ++stride_draws % draw_stride != 0) {
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
    sampled_fresh = false;
    storage_fresh = false;
    draw_sampled.clear();
    draw_storage_reads.clear();
    run_stats.BeginDraw();
    const struct RunDraw {
        BananzaNvn2RunStats& stats;
        ~RunDraw() {
            stats.EndDraw();
        }
    } run_draw{run_stats};
    if (op.replay == ReplayOp::DrawIndirect || op.replay == ReplayOp::DrawIndexedIndirect) {
        run_stats.DrawCut(BananzaNvn2RunStats::Cut::Indirect);
    }
    auto phase_start = profile_replay ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
    const auto mark = [&](size_t phase) {
        if (!profile_replay) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        draw_phase_ns[phase] += static_cast<u64>((now - phase_start).count());
        phase_start = now;
    };
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
        if (!gpu_written && reset_indirect_draws.contains(arguments.first)) {
            return;
        }
        // Only a draw from arguments the CPU provided reads them and needs the slice.
        const auto snapshot = gpu_written ? std::span<const u8>{}
                                          : ReadSnapshot(submission, arguments);
        const auto slice = snapshot.empty() ? submission.indirect_index_slices.end()
                                            : submission.indirect_index_slices.find(
                                                  {op.args[3], op.args[2], op.args[1]});
        if (!gpu_written && !snapshot.empty()) {
            VkDrawIndexedIndirectCommand command{};
            std::memcpy(&command, snapshot.data(), sizeof(command));
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
                             gpu_written, !snapshot.empty(),
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
            const auto args = ReadSnapshot(submission, draw->indirect_range);
            if (args.empty()) {
                return;
            }
            VkDrawIndexedIndirectCommand command{};
            std::memcpy(&command, args.data(), sizeof(command));
            const u64 element_size = u64{1} << op.args[1];
            const u64 end = (u64{command.firstIndex} + command.indexCount) * element_size;
            if (end > draw->index_range.second) {
                NoteSkip("indirect index range exceeds snapshot");
                return;
            }
        }
    }
    mark(0);
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
        const auto& limits = device_limits;
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
    mark(1);
    auto raster = raster_state;
    raster.depth_format = depth ? depth->format : VK_FORMAT_UNDEFINED;
    raster.depth_aspects = depth ? depth->aspects : 0;
    raster.negative_one_to_one = submission.depth_minus_one_to_one;
    raster.origin_lower_left = submission.origin_lower_left;
    const bool biased = raster.depth_bias;
    const auto bias = depth_bias;
    const float bias_clamp = device_features.depthBiasClamp ? bias[2] : 0.0f;
    const PipelineKey key{graphics_program->content_hash,
                          fragment_program ? fragment_program->content_hash : 0,
                          geometry_program ? geometry_program->content_hash : 0,
                          formats, topology, vertex_attributes,
                          vertex_strides, vertex_divisors, color_state, raster};
    if (last_pipeline == pipelines.end() || last_pipeline->first != key) {
        const auto [entry, inserted] = pipelines.try_emplace(key);
        if (!inserted && ahead_pipelines.contains(&entry->second)) {
            CollectPipelines();
        }
        // One a worker has not finished is built here: its draw is not left out, and the
        // worker's pipeline is dropped when it comes.
        const bool unfinished = !inserted && ahead_pipelines.erase(&entry->second) != 0;
        if (inserted || unfinished) {
            run_stats.DrawCut(BananzaNvn2RunStats::Cut::PipelineCreate);
            std::array<u32, 8> color_formats;
            for (size_t slot = 0; slot < colors.size(); ++slot) {
                color_formats[slot] = colors[slot] ? colors[slot]->description.format
                                                   : BananzaNvn2PipelineList::NoColor;
            }
            PipelineInputs inputs{graphics_program, fragment_program, geometry_program,
                                  vertex_attributes, vertex_strides, vertex_divisors,
                                  color_state, color_formats, formats, color_count, topology,
                                  raster};
            ++draw_built;
            if (pipeline_list && inserted) {
                pipeline_list->Add({{std::get<0>(key), std::get<1>(key), std::get<2>(key)},
                                    formats, color_formats, color_count, topology,
                                    vertex_attributes, vertex_strides, vertex_divisors,
                                    color_state, raster});
            }
            // Built here by default. With EDEN_NX2_NVN_ASYNC_PIPELINES=1 a worker builds it
            // and the draw is left out meanwhile (a few vertices without indices, a screen
            // pass or a UI quad, are still built here): that made the level's intro black,
            // because frames without their draws spoil what the game keeps between frames.
            static const bool build_here = std::getenv("EDEN_NX2_NVN_ASYNC_PIPELINES") == nullptr;
            const bool screen_pass = draw->index_type == VK_INDEX_TYPE_MAX_ENUM &&
                                     !draw->indirect_range.second && draw->count <= 6;
            if (build_here || screen_pass) {
                auto built = BuildPipeline(inputs);
                TakePipeline(entry->second, built);
            } else {
                building_pipelines.insert(&entry->second);
                pipeline_builder.Queue(&entry->second, [this, inputs = std::move(inputs)] {
                    return BuildPipeline(inputs);
                });
            }
        }
        last_pipeline = entry;
    }
    if (!last_pipeline->second) {
        if (building_pipelines.contains(&last_pipeline->second)) {
            ++pipeline_wait_draws;
            return;
        }
        NoteSkip("graphics pipeline unavailable");
        if (!missing_draw_logged) {
            missing_draw_logged = true;
            LOG_WARNING(Render_Vulkan, "Bananza NVN 2 Vulkan replay skipped a draw with unsupported pipeline state");
        }
        return;
    }
    auto& pipeline = *last_pipeline->second;
    static const bool copy_all_feedback = std::getenv("EDEN_NX2_NVN_ALWAYS_WRITTEN") != nullptr;
    draw_color_writes = copy_all_feedback ? ~0U : pipeline.color_writes;
    mark(2);
    boost::container::small_vector<std::tuple<u32, VkBuffer, u64>, 4> vertices;
    vertices.reserve(pipeline.streams.size());
    for (const u32 stream : pipeline.streams) {
        auto& upload = vertex_uploads[stream];
        if (!upload.first) {
            upload = UploadRange(submission, vertex_buffers[stream]);
        }
        const auto [buffer, buffer_offset] = upload;
        if (!buffer) {
            NoteSkip("vertex buffer range missing at submit");
            return;
        }
        vertices.emplace_back(stream, buffer, buffer_offset);
    }
    VkBuffer index_buffer{};
    u64 index_offset{};
    if (op.args[0] == 7 || op.args[0] == 8) {
        run_stats.DrawCut(BananzaNvn2RunStats::Cut::Quads);
        const bool strip = op.args[0] == 8;
        const u32 quads = draw->count < 4 ? 0 : strip ? (draw->count - 2) / 2 : draw->count / 4;
        if (!quads) {
            return;
        }
        std::span<const u8> source;
        if (draw->index_type != VK_INDEX_TYPE_MAX_ENUM) {
            source = ReadSnapshot(submission, draw->index_range);
            if (source.empty()) {
                NoteSkip("quad index buffer range missing at submit");
                return;
            }
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
        std::tie(index_buffer, index_offset) =
            UploadStream({reinterpret_cast<const u8*>(triangles.data()),
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
    mark(3);
    VkDescriptorSet set{};
    const DescriptorData* pushed{};
    VkDescriptorUpdateTemplate push_template{};
    boost::container::small_vector<std::pair<StorageUpload*, u64>, 4> written_storage;
    boost::container::small_vector<std::tuple<StorageUpload*, u64, u64>, 4> written_intervals;
    if (!pipeline.uniforms.empty() || !pipeline.storage.empty() || !pipeline.textures.empty()) {
        const auto count = pipeline.uniforms.size() + pipeline.storage.size() +
                           pipeline.textures.size();
        boost::container::small_vector<DescriptorData, 24> descriptors(count);
        for (const auto& binding : pipeline.uniforms) {
            const auto& bound = uniform_buffers[binding.stage];
            // NVN user uniform slots follow the driver's three internal banks.
            const auto range = binding.bank >= 3 ? bound[binding.bank - 3] : Range{};
            std::span<const u8> literal{binding.literal};
            BufferRef* cached_upload{};
            if (binding.bank == 0) {
                const auto [buffer, offset] = UploadDriverConstants(draw->base_instance);
                descriptors[binding.binding].buffer = {buffer, offset, driver_constants.size()};
                continue;
            } else if (binding.bank == 17 || binding.bank >= 24) {
                const u32 slot = binding.bank == 17 ? 0 : binding.bank - 24;
                literal = graphics_constants[slot];
                cached_upload = &graphics_constant_uploads[slot];
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
            BufferRef upload;
            if (cached_upload) {
                if (!cached_upload->first) {
                    *cached_upload = UploadStream(literal);
                }
                upload = *cached_upload;
            } else if (literal.empty()) {
                auto& memo = uniform_upload_memo[binding.stage % uniform_upload_memo.size()]
                                                [(binding.bank - 3) % 32];
                if (memo.generation != upload_generation || memo.range != range) {
                    const auto ref = UploadRange(submission, range);
                    memo = {range, ref, upload_generation};
                }
                upload = memo.ref;
            } else {
                upload = UploadStream(literal);
            }
            if (!upload.first) {
                NoteSkip("uniform buffer range missing at submit");
                return;
            }
            descriptors[binding.binding].buffer = {upload.first, upload.second, size};
        }
        mark(4);
        const auto& storage_limits = device_limits;
        struct FragmentStorage {
            bool& flag;
            ~FragmentStorage() {
                flag = false;
            }
        } fragment_storage{storage_for_fragment};
        static const bool move_storage = std::getenv("EDEN_NX2_NVN_NO_DEVICE_STORAGE") == nullptr;
        for (const auto& [stage, binding, slot, writable] : pipeline.storage) {
            if (stage != 1 || !move_storage) {
                continue;
            }
            if (const auto* span = BoundSpan(storage_buffers[stage][slot]);
                span && span->upload && !span->upload->readback_buffer) {
                MoveStorageToDevice(*span->upload);
            }
        }
        for (const auto& [stage, binding, slot, writable] : pipeline.storage) {
            const auto range = storage_buffers[stage][slot];
            storage_for_fragment = stage == 1;
            if (trace_images && stage == 2 &&
                draw_diagnostics.emplace(geometry_program->serial,
                                         slot == 0 ? "geometry storage 0" : "geometry storage 1").second) {
                LOG_INFO(Render_Vulkan,
                         "NVN 2 geometry program {} storage slot {} at {:#x}, {} bytes, write {}, snapshot {}, mapping {}",
                         geometry_program->serial, slot, range.first, range.second, writable,
                         submission.HasSnapshot(range),
                         submission.storage_addresses.contains(range));
            }
            if (!range.first || !range.second || range.second > storage_limits.maxStorageBufferRange) {
                NoteSkip("graphics storage range missing or exceeds host limit");
                return;
            }
            VkBuffer buffer{};
            u64 offset{};
            run_stats.Count(writable ? BananzaNvn2RunStats::Note::WritableStorage : BananzaNvn2RunStats::Note::ReadStorage);
            if (writable) {
                if (const auto [upload, start] = WritableStorage(submission, range); upload) {
                    buffer = upload->buffer;
                    offset = upload->base + start;
                    written_storage.emplace_back(upload, start + range.second);
                    written_intervals.emplace_back(upload, start, start + range.second);
                    upload->NoteWritten(start, start + range.second);
                    traced_storage = range;
                }
            } else {
                std::tie(buffer, offset) = UploadBufferRange(submission, range);
            }
            if (!buffer || offset % storage_limits.minStorageBufferOffsetAlignment) {
                NoteSkip("graphics storage range missing at submit");
                return;
            }
            descriptors[binding].buffer = {buffer, offset, range.second};
        }
        mark(5);
        for (const auto& [stage, binding, descriptor] : pipeline.textures) {
            const auto info = BindTexture(submission, stage, descriptor, targets);
            if (!info) {
                return;
            }
            descriptors[binding].image = *info;
        }
        mark(6);
        const auto matches_buffer = [&](u32 binding) {
            const auto& a = pipeline.last_descriptors[binding].buffer;
            const auto& b = descriptors[binding].buffer;
            return std::tie(a.buffer, a.offset, a.range) == std::tie(b.buffer, b.offset, b.range);
        };
        const auto matches_image = [&](u32 binding) {
            const auto& a = pipeline.last_descriptors[binding].image;
            const auto& b = descriptors[binding].image;
            return std::tie(a.sampler, a.imageView, a.imageLayout) ==
                   std::tie(b.sampler, b.imageView, b.imageLayout);
        };
        if (!pipeline.push_descriptors) {
            run_stats.DrawCut(BananzaNvn2RunStats::Cut::DescriptorSet);
        }
        if (pipeline.push_descriptors) {
            if (PushBlockEntries - push_used < count) {
                if (free_push_blocks.empty()) {
                    push_blocks.push_back(
                        std::make_unique_for_overwrite<DescriptorData[]>(PushBlockEntries));
                } else {
                    push_blocks.push_back(std::move(free_push_blocks.back()));
                    free_push_blocks.pop_back();
                }
                push_used = 0;
            }
            DescriptorData* const data = push_blocks.back().get() + push_used;
            push_used += count;
            std::memcpy(data, descriptors.data(), count * sizeof(DescriptorData));
            pushed = data;
            push_template = *pipeline.update_template;
        } else if (pipeline.descriptor_submission == replayed_submissions && pipeline.last_set &&
            std::ranges::all_of(pipeline.uniforms, [&](const auto& b) { return matches_buffer(b.binding); }) &&
            std::ranges::all_of(pipeline.storage, [&](const auto& b) { return matches_buffer(b.binding); }) &&
            std::ranges::all_of(pipeline.textures, [&](const auto& b) { return matches_image(std::get<1>(b)); })) {
            set = pipeline.last_set;
        } else {
            set = AllocateSet(*pipeline.set_layout, pipeline.uniforms.size(), pipeline.storage.size(),
                              pipeline.textures.size(), 0);
            if (!set) {
                NoteSkip("descriptor set unavailable");
                return;
            }
            device.GetLogical().UpdateDescriptorSet(set, *pipeline.update_template, descriptors.data());
            pipeline.last_descriptors.assign(descriptors.begin(), descriptors.end());
            pipeline.last_set = set;
            pipeline.descriptor_submission = replayed_submissions;
        }
        mark(7);
    }

    VkViewport draw_viewport = viewport;
    if (!(draw_viewport.width > 0.0f && draw_viewport.height > 0.0f)) {
        draw_viewport = {0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height),
                         0.0f, 1.0f};
    }
    draw_viewport.minDepth = depth_range[0];
    draw_viewport.maxDepth = depth_range[1];
    const auto& limits = device_limits;
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
        upload->dirty = true;
        // Set again once the draw is recorded; a draw that is dropped below writes nothing.
        upload->write_tick = scheduler.CurrentTick();
        if (upload->readback_buffer) {
            upload->readback_stale = true;
            device_readback_pending = true;
        }
    }
    if (occlusion_queries == OcclusionQueryCount) {
        CollectOcclusion();
    }
    const VkQueryPool query_pool = count_samples ? *occlusion_pool : VK_NULL_HANDLE;
    if (query_pool && occlusion_queries == 0) {
        Record([query_pool](vk::CommandBuffer cmdbuf) {
            cmdbuf.ResetQueryPool(query_pool, 0, OcclusionQueryCount);
        });
    }
    const bool capture_draw = frame_dump.CapturingSampled() && colors[0] && extent.width >= 1900;
    const RenderingScope scope{views, depth_attachment_view, color_count, extent.width,
                               extent.height, layers};
    // An image the draw samples, or storage it reads, that a draw wrote since the last barrier.
    for (const Image* const sampled : draw_sampled) {
        if (sampled->written_barrier == barrier_serial) {
            sampled_fresh = true;
            fresh_image = sampled;
        }
    }
    for (const StorageRead& read : draw_storage_reads) {
        const StorageUpload& upload = *read.upload;
        if (upload.written_barrier == barrier_serial) {
            const auto next = upload.fresh.upper_bound(read.offset);
            storage_fresh |=
                (next != upload.fresh.begin() && std::prev(next)->second > read.offset) ||
                (next != upload.fresh.end() && next->first < read.offset + read.size);
        }
    }
    u32 fresh_kind{};
    u32 fresh_slot{8};
    if (sampled_fresh && fresh_image) {
        const auto slot = std::ranges::find(targets, fresh_image);
        fresh_kind = fresh_image == depth ? 1U : slot != targets.end() ? 2U : 4U;
        fresh_slot = static_cast<u32>(slot - targets.begin());
    }
    const Range scope_storage = written_storage.empty() ? Range{} : traced_storage;
    static const bool old_storage_rule = std::getenv("EDEN_NX2_NVN_ALWAYS_WRITTEN") != nullptr;
    const bool storage_barrier = old_storage_rule
                                     ? scope_wrote_storage || !written_storage.empty()
                                     : storage_fresh;
    const u32 scope_reason = (rendering_scope && rendering_scope != scope ? 1U : 0U) |
                             (sampled_fresh ? 2U : 0U) | (storage_barrier ? 4U : 0U) |
                             (!written_storage.empty() ? 8U : 0U) | (!rendering_scope ? 16U : 0U);
    if (rendering_scope != scope || sampled_fresh || storage_barrier || capture_draw) {
        run_stats.Count(BananzaNvn2RunStats::Note::ScopeBarrier);
        Barrier();
    }
    std::optional<u32> probe;
    if (!rendering_scope) {
        if (pass_trace_active) {
            const u64 target = colors[0] ? colors[0]->storage.gpu_address + colors[0]->offset
                               : depth_texture
                                     ? depth_texture->storage.gpu_address + depth_texture->offset
                                     : 0;
            PassTimestamp({.target = target,
                           .width = extent.width,
                           .height = extent.height,
                           .program = graphics_program->serial,
                           .reason = scope_reason,
                           .fresh = fresh_kind,
                           .fresh_slot = fresh_slot,
                           .storage = scope_storage.first,
                           .storage_size = scope_storage.second});
        }
        scheduler.Record([scope, extent, attachments, depth_attachment, aspects = raster.depth_aspects](
                             vk::CommandBuffer cmdbuf) {
            cmdbuf.BeginRendering({
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .renderArea = {{}, extent},
                .layerCount = scope.layers,
                .colorAttachmentCount = scope.color_count,
                .pColorAttachments = attachments.data(),
                .pDepthAttachment = aspects & VK_IMAGE_ASPECT_DEPTH_BIT ? &depth_attachment : nullptr,
                .pStencilAttachment = aspects & VK_IMAGE_ASPECT_STENCIL_BIT ? &depth_attachment : nullptr,
            });
        });
        rendering_scope = scope;
        scope_vertices.fill({});
        scope_index = {};
        ++rendering_scopes;
        ++flush_scopes;
    }
    if (capture_draw) {
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
    scope_wrote_storage = !written_storage.empty();
    for (const auto& [upload, begin, end] : written_intervals) {
        if (upload->written_barrier != barrier_serial) {
            upload->written_barrier = barrier_serial;
            upload->fresh.clear();
        }
        // Merge with the intervals it touches.
        u64 first = begin;
        u64 last = end;
        auto it = upload->fresh.upper_bound(begin);
        if (it != upload->fresh.begin() && std::prev(it)->second >= begin) {
            --it;
            first = it->first;
        }
        while (it != upload->fresh.end() && it->first <= last) {
            last = std::max(last, it->second);
            it = upload->fresh.erase(it);
        }
        upload->fresh.emplace(first, last);
    }
    // Only a draw that can write a target makes a later sample of it need a barrier.
    static const bool always_written = std::getenv("EDEN_NX2_NVN_ALWAYS_WRITTEN") != nullptr;
    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i] && (always_written || ((pipeline.color_writes >> i) & 1))) {
            targets[i]->written_barrier = barrier_serial;
        }
    }
    if (depth && (always_written || pipeline.depth_write || pipeline.uses_stencil)) {
        depth->written_barrier = barrier_serial;
    }
    const u32 query = count_samples ? occlusion_queries++ : 0;
    const VkQueryControlFlags query_flags = precise_queries ? VK_QUERY_CONTROL_PRECISE_BIT : 0;
    const auto graphics = *pipeline.pipeline;
    const bool bind_pipeline = std::exchange(scope_pipeline, graphics) != graphics;
    const bool bind_set = set && (bind_pipeline || scope_descriptor_set != set);
    scope_descriptor_set = set;
    const auto unchanged = [this](const auto& vertex) {
        const auto [stream, buffer, offset] = vertex;
        const BufferRef binding{buffer, offset};
        return std::exchange(scope_vertices[stream], binding) == binding;
    };
    vertices.erase(std::remove_if(vertices.begin(), vertices.end(), unchanged), vertices.end());
    const auto index_binding = std::tuple{index_buffer, index_offset, draw->index_type};
    const bool bind_index = index_buffer && std::exchange(scope_index, index_binding) != index_binding;
    mark(8);
    scheduler.Record([graphics, bind_pipeline, layout = *pipeline.layout,
                      extent, draw_viewport, draw_scissor, command = *draw, set, bind_set, pushed,
                      push_template, index_buffer,
                      index_offset, bind_index,
                      indirect_buffer, indirect_offset, biased, bias, bias_clamp,
                      query_pool, query, query_flags,
                      blend_color = blend_constants, uses_render_area = pipeline.uses_render_area,
                      uses_blend_constants = pipeline.uses_blend_constants,
                      uses_stencil = pipeline.uses_stencil,
                      stages = pipeline.stages,
                      refs = stencil_ref, compare_masks = stencil_compare_mask,
                      write_masks = stencil_write_mask,
                      vertices = std::move(vertices)](
                         vk::CommandBuffer cmdbuf) {
        if (query_pool) {
            cmdbuf.BeginQuery(query_pool, query, query_flags);
        }
        if (bind_pipeline) {
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, graphics);
            using namespace Shader::Backend::SPIRV;
            constexpr RescalingLayout rescale{.down_factor = std::bit_cast<u32>(1.0f)};
            cmdbuf.PushConstants(layout, stages, rescale);
            if (uses_render_area) {
                const RenderAreaLayout area{{static_cast<float>(extent.width),
                                             static_cast<float>(extent.height), 0, 0}};
                cmdbuf.PushConstants(layout, stages, RENDERAREA_LAYOUT_OFFSET, sizeof(area), &area);
            }
        }
        cmdbuf.SetViewport(0, {draw_viewport});
        cmdbuf.SetScissor(0, {draw_scissor});
        if (uses_blend_constants) {
            cmdbuf.SetBlendConstants(blend_color.data());
        }
        if (biased) {
            cmdbuf.SetDepthBias(bias[1], bias_clamp, bias[0]);
        }
        if (uses_stencil) {
            for (u32 i = 0; i < 2; ++i) {
                const VkStencilFaceFlags face =
                    i == 0 ? VK_STENCIL_FACE_FRONT_BIT : VK_STENCIL_FACE_BACK_BIT;
                cmdbuf.SetStencilReference(face, refs[i]);
                cmdbuf.SetStencilCompareMask(face, compare_masks[i]);
                cmdbuf.SetStencilWriteMask(face, write_masks[i]);
            }
        }
        if (bind_set) {
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, {set}, {});
        }
        if (pushed) {
            cmdbuf.PushDescriptorSetWithTemplateKHR(push_template, layout, 0, pushed);
        }
        for (const auto& [stream, buffer, buffer_offset] : vertices) {
            cmdbuf.BindVertexBuffer(stream, buffer, buffer_offset);
        }
        if (bind_index) {
            cmdbuf.BindIndexBuffer(index_buffer, index_offset, command.index_type);
        }
        if (command.indexed_indirect) {
            cmdbuf.DrawIndexedIndirect(indirect_buffer, indirect_offset, 1, sizeof(VkDrawIndexedIndirectCommand));
        } else if (indirect_buffer) {
            cmdbuf.DrawIndirect(indirect_buffer, indirect_offset, 1, sizeof(VkDrawIndirectCommand));
        } else if (index_buffer) {
            cmdbuf.DrawIndexed(command.count, command.instances, 0, command.base_vertex,
                               command.base_instance);
        } else {
            cmdbuf.Draw(command.count, command.instances, command.first, command.base_instance);
        }
        if (query_pool) {
            cmdbuf.EndQuery(query_pool, query);
        }
    });
    for (const auto& [upload, size] : written_storage) {
        // The draw is in the command buffer with this tick.
        upload->write_tick = scheduler.CurrentTick();
    }
    mark(9);
    if (probe) {
        frame_dump.EndProbe(*probe, {.first = draw->first, .elements = draw->count,
                                     .instances = draw->instances,
                                     .color0 = (color_state.channel_mask & 15) != 0});
        EndRendering();
    }
    if (colors[0]) {
        TraceImage("draw", *colors[0], graphics_program->serial);
    }
    if (++draws == 1) {
        LOG_INFO(Render_Vulkan, "Bananza NVN 2 Vulkan replay recorded its first guest draw");
    }
}

void BananzaNvn2Replay::BindBufferRange(bool storage, u32 stage, u32 slot, Range range) {
    auto& current = storage ? storage_buffers[stage][slot] : uniform_buffers[stage][slot];
    driver_storage_dirty |= storage && stage < GraphicsStorageBase.size() && slot < 16 &&
                            current != range;
    current = range;
}

BananzaNvn2Replay::BufferRef BananzaNvn2Replay::UploadDriverConstants(u32 base_instance) {
    if (driver_storage_dirty) {
        for (u32 stage = 0; stage < GraphicsStorageBase.size(); ++stage) {
            const u32 base = GraphicsStorageBase[stage];
            for (u32 slot = 0; slot < 16; ++slot) {
                const auto& range = storage_buffers[stage][slot];
                const u64_le address = range.first;
                const u32_le size = static_cast<u32>(range.second);
                std::memcpy(driver_constants.data() + base + slot * 16, &address, sizeof(address));
                std::memcpy(driver_constants.data() + base + slot * 16 + 8, &size, sizeof(size));
            }
        }
        driver_storage_dirty = false;
        driver_upload = {};
    }
    if (driver_base_instance != base_instance) {
        driver_base_instance = base_instance;
        const u32_le word = base_instance;
        std::memcpy(driver_constants.data() + 0x644, &word, sizeof(word));
        driver_upload = {};
    }
    if (!driver_upload.first) {
        driver_upload = UploadStream(driver_constants);
    }
    return driver_upload;
}

void BananzaNvn2Replay::ReplayRecording(const std::vector<const Operation*>& flat,
                                        const Submission& submission) {
    // Every operation that runs, in order: the called recordings' are in the list already.
    for (size_t op_index = 0; op_index < flat.size(); ++op_index) {
        Core::BananzaNvn2Commands::PrefetchAhead(flat, op_index);
        const auto& op = *flat[op_index];
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
            if (op.Textures().size() != count + 1 || op.args.size() != 2 * count + 3 ||
                op.Payload().size() != 16 * (count + 1)) {
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
                if (op.Textures()[i].serial) {
                    colors[i] = op.Textures()[i];
                    color_views[i] = attachment(*colors[i], std::span{op.Payload()}.subspan(16 * i, 16),
                                                op.args[2 + count + i] != 0);
                }
            }
            if (const auto& texture = op.Textures().back(); texture.serial) {
                depth_texture = texture;
                depth_view = attachment(texture, std::span{op.Payload()}.last(16), op.args.back() != 0);
            }
            break;
        }
        case ReplayOp::ClearColor:
        case ReplayOp::ClearColorSigned:
        case ReplayOp::ClearColorUnsigned:
            run_stats.OpCut(BananzaNvn2RunStats::Cut::Clear);
            ClearColor(op);
            break;
        case ReplayOp::ClearDepthStencil:
            run_stats.OpCut(BananzaNvn2RunStats::Cut::Clear);
            ClearDepthStencil(op);
            break;
        case ReplayOp::BindDepthStencil:
            if (op.Payload().size() == sizeof(u64)) {
                u64_le value;
                std::memcpy(&value, op.Payload().data(), sizeof(value));
                raster_state.depth_stencil = value & 0xffffffff000000f7ULL;
            }
            break;
        case ReplayOp::BindPolygon:
            if (op.Payload().size() == sizeof(u32)) {
                u32_le value;
                std::memcpy(&value, op.Payload().data(), sizeof(value));
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
            if (op.Payload().size() == sizeof(depth_bias)) {
                std::memcpy(depth_bias.data(), op.Payload().data(), sizeof(depth_bias));
            }
            break;
        case ReplayOp::SetDepthRange:
            if (op.Payload().size() == sizeof(depth_range)) {
                std::memcpy(depth_range.data(), op.Payload().data(), sizeof(depth_range));
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
            run_stats.OpCut(BananzaNvn2RunStats::Cut::Copy);
            CopyTexture(op, submission);
            break;
        case ReplayOp::CopyBufferToTexture:
            run_stats.OpCut(BananzaNvn2RunStats::Cut::Copy);
            CopyBufferToTexture(op, submission);
            break;
        case ReplayOp::SetCopyRowStride:
            if (op.args.size() == 1) {
                copy_row_stride = op.args[0];
            }
            break;
        case ReplayOp::Barrier:
            run_stats.OpCut(BananzaNvn2RunStats::Cut::BarrierOp);
            Barrier();
            break;
        case ReplayOp::BindProgram:
            if (op.args.size() == 2) {
                const auto bind = [&](std::shared_ptr<const Program>& destination, u32 bit, u32 stage) {
                    if (!(op.args[1] & bit)) {
                        return;
                    }
                    if (op.Program() && std::ranges::any_of(op.Program()->shaders, [stage](const auto& shader) {
                            return shader.stage == stage;
                        })) {
                        destination = op.Program();
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
                const u64 slot = op.args[0];
                const Range range{op.args[1], op.args[2]};
                if (vertex_buffers[slot] != range) {
                    vertex_buffers[slot] = range;
                    vertex_uploads[slot] = {};
                }
            }
            break;
        case ReplayOp::BindUniformBuffer:
        case ReplayOp::BindStorageBuffer: {
            if (op.args.size() == 4 && op.args[0] < uniform_buffers.size() && op.args[1] < 32) {
                BindBufferRange(op.replay == ReplayOp::BindStorageBuffer, static_cast<u32>(op.args[0]),
                                static_cast<u32>(op.args[1]), {op.args[2], op.args[3]});
            }
            break;
        }
        case ReplayOp::BindUniformBuffers:
        case ReplayOp::BindStorageBuffers: {
            if (op.args.size() == 3 && op.args[0] < uniform_buffers.size() && op.args[1] < 32 &&
                op.args[2] <= 32 - op.args[1] && op.Payload().size() == 16 * op.args[2]) {
                for (size_t i = 0; i < op.args[2]; ++i) {
                    std::array<u64_le, 2> range{};
                    std::memcpy(range.data(), op.Payload().data() + i * 16, 16);
                    BindBufferRange(op.replay == ReplayOp::BindStorageBuffers,
                                    static_cast<u32>(op.args[0]),
                                    static_cast<u32>(op.args[1] + i), {range[0], range[1]});
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
                std::span<const u8> data = op.Payload();
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
                op.args[1] <= 256 && op.Payload().size() <= 256 - op.args[1]) {
                std::ranges::copy(op.Payload(), constants[op.args[0]].begin() + op.args[1]);
                if (op.replay == ReplayOp::PushGraphicsConstants) {
                    graphics_constant_uploads[op.args[0]] = {};
                }
            }
            break;
        }
        case ReplayOp::UpdateUniformBuffer:
            if (op.args.size() == 4 && !op.Payload().empty() && op.Payload().size() == op.args[3] &&
                op.args[0] <= std::numeric_limits<u64>::max() - op.args[1] &&
                op.args[2] <= op.args[1] && op.args[3] <= op.args[1] - op.args[2]) {
                const u64 address = op.args[0] + op.args[2];
                const u64 end = address + op.Payload().size();
                run_stats.OpCut(BananzaNvn2RunStats::Cut::BufferUpdate);
                AddBufferUpdate(address, op.Payload());
                erase_if(uploaded_ranges, [address, end](const auto& entry) {
                    const auto [start, size] = entry.first;
                    return start < end && address < start + size;
                });
                vertex_uploads.fill({});
                ForgetUploads(address, end);
            }
            break;
        case ReplayOp::BindBlendState:
            if (op.Payload().size() == sizeof(u64)) {
                u64_le packed;
                std::memcpy(&packed, op.Payload().data(), sizeof(packed));
                color_state.blend[packed & 7] = packed & 0x7f7f7f7f77003f07ULL;
            }
            break;
        case ReplayOp::BindColorState:
        case ReplayOp::BindChannelMask:
        case ReplayOp::SetAlphaRef:
            if (op.Payload().size() >= sizeof(u32)) {
                u32_le packed;
                std::memcpy(&packed, op.Payload().data(), sizeof(packed));
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
            if (op.Payload().size() == sizeof(blend_constants)) {
                std::memcpy(blend_constants.data(), op.Payload().data(), sizeof(blend_constants));
            }
            break;
        case ReplayOp::StencilCullCriteria:
            break;
        case ReplayOp::BindVertexAttribState: {
            if (op.args.size() != 2 || op.args[0] > vertex_attributes.size()) {
                break;
            }
            const u64 id = op.args[1];
            if (id == vertex_attributes_id) {
                break;
            }
            auto& known = known_attributes[(id >> 8) % known_attributes.size()];
            if (known.id != id) {
                if (op.Payload().size() != op.args[0] * 4) {
                    break;
                }
                known.attributes.fill(0);
                std::memcpy(known.attributes.data(), op.Payload().data(), op.Payload().size());
                known.id = id;
            }
            vertex_attributes = known.attributes;
            vertex_attributes_id = id;
            break;
        }
        case ReplayOp::BindVertexStreamState: {
            if (op.args.size() != 2 || op.args[0] > vertex_strides.size()) {
                break;
            }
            const u64 id = op.args[1];
            if (id == vertex_streams_id) {
                break;
            }
            auto& known = known_streams[(id >> 8) % known_streams.size()];
            if (known.id != id) {
                if (op.Payload().size() != op.args[0] * 8) {
                    break;
                }
                known.strides.fill(0);
                known.divisors.fill(0);
                for (size_t i = 0; i < op.args[0]; ++i) {
                    std::memcpy(&known.strides[i], op.Payload().data() + i * 8, 4);
                    std::memcpy(&known.divisors[i], op.Payload().data() + i * 8 + 4, 4);
                }
                known.id = id;
            }
            vertex_strides = known.strides;
            vertex_divisors = known.divisors;
            vertex_streams_id = id;
            break;
        }
        case ReplayOp::SetViewport:
            if (op.args.size() == 4) {
                viewport = {static_cast<float>(static_cast<s32>(op.args[0])),
                            static_cast<float>(static_cast<s32>(op.args[1])),
                            static_cast<float>(static_cast<s32>(op.args[2])),
                            static_cast<float>(static_cast<s32>(op.args[3])), 0.0f, 1.0f};
            }
            break;
        case ReplayOp::SetViewports:
            if (op.args.size() == 2 && op.args[0] == 0 && op.Payload().size() >= 4 * sizeof(float)) {
                std::array<float, 4> rect{};
                std::memcpy(rect.data(), op.Payload().data(), sizeof(rect));
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
            if (op.args.size() == 2 && op.args[0] == 0 && op.Payload().size() >= 4 * sizeof(u32)) {
                std::array<s32_le, 4> rect{};
                std::memcpy(rect.data(), op.Payload().data(), sizeof(rect));
                scissor = {{rect[0], rect[1]},
                           {static_cast<u32>(std::max<s32>(0, rect[2])),
                            static_cast<u32>(std::max<s32>(0, rect[3]))}};
                has_scissor = true;
            }
            break;
        case ReplayOp::SetDepthRanges:
            if (op.args.size() == 2 && op.args[0] == 0 && op.Payload().size() >= sizeof(depth_range)) {
                std::memcpy(depth_range.data(), op.Payload().data(), sizeof(depth_range));
            }
            break;
        case ReplayOp::Draw:
        case ReplayOp::DrawIndexed:
        case ReplayOp::DrawIndexedBaseVertex:
        case ReplayOp::DrawIndexedInstanced:
        case ReplayOp::DrawInstanced:
        case ReplayOp::DrawIndirect:
        case ReplayOp::DrawIndexedIndirect:
        {
            const auto start = profile_replay ? std::chrono::steady_clock::now()
                                              : std::chrono::steady_clock::time_point{};
            Draw(op, submission);
            if (profile_replay) {
                draw_ns += static_cast<u64>((std::chrono::steady_clock::now() - start).count());
            }
            break;
        }
        case ReplayOp::Dispatch: {
            const auto start = profile_replay ? std::chrono::steady_clock::now()
                                              : std::chrono::steady_clock::time_point{};
            run_stats.OpCut(BananzaNvn2RunStats::Cut::Dispatch);
            Dispatch(op, submission);
            if (profile_replay) {
                dispatch_ns += static_cast<u64>((std::chrono::steady_clock::now() - start).count());
            }
            break;
        }
        case ReplayOp::SignalEvent:
        case ReplayOp::WaitEvent:
            run_stats.OpCut(BananzaNvn2RunStats::Cut::Event);
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
            run_stats.OpCut(BananzaNvn2RunStats::Cut::Counter);
            ReplayCounter(op, submission);
            break;
        case ReplayOp::Unsupported:
            break;
        }
    }
}

void BananzaNvn2Replay::CollectOcclusion() {
    WaitRetireIdle();
    std::vector<u64> results(occlusion_queries);
    if (occlusion_queries) {
        EndRendering();
        WaitCopies();
        scheduler.Finish();
        ++occlusion_flushes;
        vk::Check(device.GetLogical().GetQueryResults(*occlusion_pool, 0, occlusion_queries,
                                                      results.size() * sizeof(u64), results.data(),
                                                      sizeof(u64), VK_QUERY_RESULT_64_BIT));
    }
    {
        std::scoped_lock lock{retire_mutex};
        ApplyCounterEvents(results, counter_events, occlusion_queries);
    }
    counter_events.clear();
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
                precise_queries = device_features.occlusionQueryPrecise;
            }
            counter_events.push_back({.kind = CounterEvent::Kind::Reset, .query = occlusion_queries});
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
        if (std::ranges::any_of(counter_events, [&](const CounterEvent& event) {
                return event.kind == CounterEvent::Kind::Report && event.gpu_address == op.args[1];
            })) {
            // The condition reads a report of this submission.
            CollectOcclusion();
        }
        {
            std::unique_lock lock{retire_mutex};
            retire_idle_cv.wait(lock, [&] { return !pending_reports.contains(op.args[1]); });
        }
        std::array<u64_le, 3> data{};
        if (memory.ReadBlockUnsafe(address->second, data.data(), sizeof(data))) {
            const bool equal = data[0] == data[2];
            render_enabled = equal == (type == 0);
        }
    } else if (op.replay == ReplayOp::ReportCounter) {
        if (type == SamplesPassed) {
            if (!count_samples) {
                cumulative_reports = true;
            }
            // Written once the queries are read, by the retire thread at the latest.
            counter_events.push_back({.kind = CounterEvent::Kind::Report,
                                      .query = occlusion_queries,
                                      .memory = &memory,
                                      .guest_address = address->second,
                                      .gpu_address = op.args[1],
                                      .submitted_ns = submission.submitted_ns});
            count_samples = cumulative_reports;
            return;
        }
        if (type != 0 && type != 16) {
            if (unsupported_counters.insert(type).second) {
                LOG_WARNING(Render_Vulkan, "Bananza NVN 2 counter type {} needs a Vulkan query", type);
            }
            return;
        }
        static const bool frame_stamps = std::getenv("EDEN_NX2_NVN_FRAME_TIMESTAMPS") != nullptr;
        static const bool trace_counters = std::getenv("EDEN_NX2_NVN_COUNTER_TRACE") != nullptr;
        static u64 stamp_present = std::numeric_limits<u64>::max();
        static u64 frame_stamp_ns{};
        static u64 frame_reports{};
        if (stamp_present != present_count) {
            stamp_present = present_count;
            frame_stamp_ns = submission.submitted_ns;
            frame_reports = 0;
        }
        const u64 stamp = frame_stamps ? frame_stamp_ns + 1000 * frame_reports
                                       : submission.submitted_ns;
        ++frame_reports;
        if (trace_counters && present_count % 600 < 3) {
            LOG_INFO(Render_Vulkan,
                     "NVN 2 timestamp report: present {}, submission {}, draw {}, type {}, at "
                     "{:#x}, value {} us, {} us after the frame's first submit",
                     present_count, submission.sequence, draws, type, op.args[1], stamp / 1000,
                     (stamp - frame_stamp_ns) / 1000);
        }
        const std::array<u64_le, 2> report{0, stamp};
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
            if (profile_replay) {
                ++event_signals;
                last_event_signal = {draws, dispatches};
            }
            // The GPU reads only host copies of guest memory, so the guest may see the
            // event before the GPU gets there unless it reads a result back.
            static const bool sync_events = std::getenv("EDEN_NX2_NVN_SYNC_EVENTS") != nullptr;
            const bool dirty = std::ranges::any_of(
                storage_uploads, [](const auto& entry) { return entry.second.dirty; });
            if (!sync_events && (!counter_events.empty() ||
                                 pending_counter_jobs.load(std::memory_order_relaxed) != 0)) {
                // After a pending report: written in order with it.
                counter_events.push_back({.kind = CounterEvent::Kind::Signal,
                                          .query = occlusion_queries,
                                          .memory = &memory,
                                          .guest_address = address->second,
                                          .value = value,
                                          .add = op.args[1] == 1});
                return;
            }
            if (!sync_events) {
                memory.Write32(address->second, op.args[1] == 1
                                                    ? memory.Read32(address->second) + value
                                                    : value);
                return;
            }
            ++event_signal_waits;
            const auto start = std::chrono::steady_clock::now();
            EndRendering();
            if (dirty) {
                RecordStorageReadback();
                Record([](vk::CommandBuffer cmdbuf) {
                    const VkMemoryBarrier barrier{
                        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
                    };
                    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                           VK_PIPELINE_STAGE_HOST_BIT, 0, barrier);
                });
            }
            if (scheduler.RecordedSinceSubmit()) {
                WaitCopies();
                scheduler.Finish();
            } else {
                scheduler.Wait(scheduler.CurrentTick() - 1);
            }
            event_finish_ns += static_cast<u64>((std::chrono::steady_clock::now() - start).count());
            if (dirty) {
                WriteBackStorage(submission);
            }
            next_flush_draw = draws + DrawsBeforeFlush;
            next_replay_flush = std::chrono::steady_clock::now() + ReplayFlushInterval;
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
    const auto replay_start = std::chrono::steady_clock::now();
    struct BusyTime {
        std::chrono::steady_clock::time_point start;
        ~BusyTime() {
            Core::BananzaNvn2Commands::g_replay_busy_ns.fetch_add(
                static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now() - start)
                                     .count()),
                std::memory_order_relaxed);
        }
    } busy_time{replay_start};
    // The sync of the submission may signal when this returns.
    struct Replayed {
        u64 sequence;
        ~Replayed() {
            Core::BananzaNvn2Commands::GetRetirement().MarkReplayed(sequence);
        }
    } replayed{submission.sequence};
    static const bool wait_at_start = std::getenv("EDEN_NX2_NVN_WAIT_AT_START") != nullptr;
    ++replay_count;
    if ((replay_count & 511) == 0 && replay_count > 2048) {
        kept_storage.EvictUnused(replay_count - 2048, evicted_kept);
    }
    if (wait_at_start && submission.after_writes) {
        // Ranges read from guest memory below may be written by that submission's GPU work.
        auto& retirement = Core::BananzaNvn2Commands::GetRetirement();
        if (!retirement.WritesDone(submission.after_writes)) {
            retirement.WaitWrites(submission.after_writes);
            static std::atomic<u64> waits{};
            static std::atomic<u64> wait_ns{};
            wait_ns.fetch_add(static_cast<u64>((std::chrono::steady_clock::now() - replay_start)
                                                   .count()),
                              std::memory_order_relaxed);
            if (const u64 count = waits.fetch_add(1, std::memory_order_relaxed) + 1;
                count % 300 == 0) {
                LOG_INFO(Render_Vulkan,
                         "NVN 2 replay: the last 300 waits for an earlier submission's GPU "
                         "writes took {:.3f} ms each",
                         wait_ns.exchange(0, std::memory_order_relaxed) / 300e6);
            }
        }
    }
    late_copies = submission.late_copies;
    const u64 draws_before = draws;
    const std::array slow_before{pipeline_creates, pipeline_create_ns, image_creates,
                                 texture_uploads, upload_ns, fixed_uploads, fixed_upload_bytes,
                                 storage_creates, finishes, stream_bytes, hash_ns, device_moves};
    const u64 tick_before = scheduler.CurrentTick();
    fixed_check_budget = 256ULL << 10;
    static const bool import_probe = std::getenv("EDEN_NX2_NVN_IMPORT_PROBE") != nullptr;
    if (import_probe && import_probes < 2) {
        const u8* pointer{};
        u64 size{};
        for (const auto& shard : submission.buffer_snapshots) {
            for (const auto& [range, bytes] : shard) {
                if (!bytes.owned && range.second > size) {
                    pointer = bytes.data;
                    size = range.second;
                }
            }
        }
        if (size >= (256ULL << 10)) {
            ++import_probes;
            ProbeHostImport(pointer, size);
        }
    }
    static const bool pass_trace_enabled = std::getenv("EDEN_NX2_NVN_GPU_PASSES") != nullptr;
    pass_trace = pass_trace_enabled;
    if (pass_trace && replay_start >= pass_trace_next &&
        !pass_trace_busy.load(std::memory_order_acquire)) {
        if (!pass_trace_pool) {
            pass_trace_pool = device.GetLogical().CreateQueryPool({
                .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                .queryType = VK_QUERY_TYPE_TIMESTAMP,
                .queryCount = PassTraceQueries,
            });
        }
        pass_trace_labels.clear();
        Record([pool = *pass_trace_pool](vk::CommandBuffer cmdbuf) {
            cmdbuf.ResetQueryPool(pool, 0, PassTraceQueries);
        });
        pass_trace_active = true;
    }
    if (profile_replay) {
        scheduler.trace_submit_times.store(true, std::memory_order_relaxed);
    }
    next_flush_draw = draws + DrawsBeforeFlush;
    next_replay_flush = replay_start + ReplayFlushInterval;
    flush_scopes = 0;
    ++replayed_submissions;
    for (const auto& program : submission.new_programs) {
        NoteProgram(program);
    }
    CollectPipelines();
    ReclaimBusyResources();
    origin_lower_left = submission.origin_lower_left;
    const u64 first_tick = scheduler.CurrentTick();
    const u64 first_record = scheduler.RecordingSerial();
    CollectStorageSpans(submission);
    staged_ranges_ready = SyncMirrorPages(submission);
    StartTextureFills(submission);
    // The workers read the submission's bytes: none of them outlives this replay.
    SCOPE_EXIT {
        for (auto& [serial, pending] : texture_fills) {
            pending->done.wait(false, std::memory_order_acquire);
        }
        texture_fills.clear();
    };
    for (const auto& flat : submission.flats) {
        ReplayRecording(flat, submission);
    }
    FinishTextureFills(submission);
    run_stats.EndSubmission();
    if (last_event_signal) {
        event_tail_draws += draws - last_event_signal->first;
        event_tail_dispatches += dispatches - last_event_signal->second;
        last_event_signal.reset();
    }
    EndRendering();
    const bool wrote_storage = std::ranges::any_of(storage_uploads, [](const auto& entry) {
        return entry.second.dirty;
    });
    if (wrote_storage) {
        RecordStorageReadback();
    }
    if (wrote_storage) {
        Record([](vk::CommandBuffer cmdbuf) {
            const VkMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   VK_PIPELINE_STAGE_HOST_BIT, 0, barrier);
        });
    }
    RetireJob job{.sequence = submission.sequence,
                  .submission = submission.shared_from_this(),
                  .queries = occlusion_queries};
    vk::Buffer query_results;
    if (occlusion_queries) {
        if (free_query_results.empty()) {
            query_results = allocator.CreateBuffer({
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = OcclusionQueryCount * sizeof(u64),
                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            }, MemoryUsage::Download);
        } else {
            query_results = std::move(free_query_results.back());
            free_query_results.pop_back();
        }
        Record([pool = *occlusion_pool, buffer = *query_results,
                count = occlusion_queries](vk::CommandBuffer cmdbuf) {
            cmdbuf.CopyQueryPoolResults(pool, 0, count, buffer, 0, sizeof(u64),
                                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            const VkMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_MEMORY_READ_BIT |
                                 VK_ACCESS_MEMORY_WRITE_BIT,
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   0, barrier);
        });
        occlusion_queries = 0;
    }
    if (pass_trace_active) {
        // Small submissions are not worth a sample; the next one starts over.
        if (pass_trace_labels.size() >= 32) {
            job.pass_end_draws = draws;
            Record([pool = *pass_trace_pool, query = static_cast<u32>(pass_trace_labels.size())](
                       vk::CommandBuffer cmdbuf) {
                cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, query);
            });
            job.pass_labels = std::move(pass_trace_labels);
            job.scope_end_sites = std::move(scope_end_sites);
            pass_trace_busy.store(true, std::memory_order_release);
            pass_trace_next = replay_start + std::chrono::seconds{5};
        }
        pass_trace_labels.clear();
        scope_end_sites.clear();
        pass_trace_active = false;
    }
    if (scheduler.RecordingSerial() != first_record && scheduler.RecordedSinceSubmit()) {
        WaitCopies();
        scheduler.Flush();
        ++replay_flush_ends;
    }
    if (!replay_logged && first_tick != scheduler.CurrentTick()) {
        replay_logged = true;
        LOG_INFO(Render_Vulkan, "Bananza NVN 2 Vulkan replay completed {} clears and {} copies",
                 clears, copies);
    }
    next_flush_draw = std::numeric_limits<u64>::max();
    uploaded_ranges.clear();
    bound_ranges.clear();
    vertex_uploads.fill({});
    ++upload_generation;
    sampled_bindings.clear();
    WaitCopies();
    auto& busy = busy_resources.emplace_back();
    busy.tick = scheduler.CurrentTick() - 1;
    busy.sequence = submission.sequence;
    if (query_results) {
        busy.query_results.push_back(std::move(query_results));
    }
    job.tick = busy.tick;
    job.query_results = busy.query_results.empty() ? nullptr : &busy.query_results.front();
    busy.storage.reserve(storage_arenas.size());
    for (auto& arena : storage_arenas) {
        busy.storage.emplace_back(arena.capacity, std::move(arena.buffer));
        if (wrote_storage && submission.guest_memory) {
            job.written_buffers.push_back(&busy.storage.back().second);
        }
    }
    storage_arenas.clear();
    for (const auto& buffer : used_kept) {
        if (wrote_storage && submission.guest_memory) {
            job.written_buffers.push_back(buffer.get());
        }
    }
    job.kept = std::move(used_kept);
    used_kept.clear();
    busy.kept = std::move(evicted_kept);
    evicted_kept.clear();
    busy.device_storage.reserve(device_arenas.size());
    for (auto& arena : device_arenas) {
        busy.device_storage.emplace_back(arena.capacity, std::move(arena.buffer));
    }
    device_arenas.clear();
    device_uploads.clear();
    for (auto& [range, upload] : storage_uploads) {
        if (wrote_storage && upload.dirty && submission.guest_memory) {
            // The spans are rebuilt for the next submission: this one's lists can move.
            const auto span = storage_spans.lower_bound({range.first, 0});
            const bool found = span != storage_spans.end() && span->first.first == range.first;
            // A span in device memory is copied back by the last command buffer.
            static const bool late = std::getenv("EDEN_NX2_NVN_LATE_WRITEBACK") != nullptr;
            // A span in device memory is back once the command buffer with its copy ran.
            // One that moved there after its last write has those bytes in the host buffer
            // since that write.
            const u64 done = !upload.readback_buffer ? upload.write_tick
                             : upload.readback_stale ? 0
                             : upload.readback_tick  ? upload.readback_tick
                                                     : upload.write_tick;
            const u64 tick = late || !done ? job.tick : std::min(done, job.tick);
            job.writes.push_back({upload.guest_address, range.first, upload.written_bytes,
                                  upload.readback,
                                  found ? std::move(span->second.snapshots)
                                        : std::vector<Range>{},
                                  tick, found ? span->second.host : nullptr,
                                  std::move(upload.written)});
        }
    }
    storage_uploads.clear();
    storage_upload_of.clear();
    storage_guest_ends.clear();
    buffer_updates.clear();
    reset_indirect_draws.clear();
    busy.uploads = std::move(uploads);
    uploads.clear();
    for (auto& chunk : descriptor_chunks) {
        busy.pools.emplace_back(chunk.reusable, std::move(chunk.pool));
    }
    descriptor_chunks.clear();
    busy.pushed = std::move(push_blocks);
    push_blocks.clear();
    push_used = PushBlockEntries;
    for (const size_t index : active_chunks) {
        upload_chunks[index].tick = busy.tick;
        upload_chunks[index].active = false;
    }
    active_chunks.clear();
    job.counters = std::move(counter_events);
    counter_events.clear();
    if (!job.submission->guest_memory) {
        job.writes.clear();
        job.counters.clear();
    }
    for (const WriteBack& write : job.writes) {
        if (write.before) {
            pending_writebacks.push_back({write.before, write.before + write.bytes,
                                          submission.sequence});
        } else {
            pending_unknown_writes = submission.sequence;
        }
    }
    {
        std::scoped_lock lock{retire_mutex};
        for (const auto& event : job.counters) {
            if (event.kind == CounterEvent::Kind::Report) {
                ++pending_reports[event.gpu_address];
            }
        }
        if (!job.counters.empty()) {
            pending_counter_jobs.fetch_add(1, std::memory_order_relaxed);
        }
        job.replay_start_ns = static_cast<u64>(replay_start.time_since_epoch().count());
        job.push_ns = Core::BananzaNvn2Commands::HostNs();
        job.first_tick = tick_before;
        job.draws = draws - draws_before;
        retire_jobs.push_back(std::move(job));
    }
    retire_cv.notify_one();
    static const bool sync_replay = std::getenv("EDEN_NX2_NVN_SYNC_REPLAY") != nullptr;
    if (sync_replay) {
        const auto finish_start = std::chrono::steady_clock::now();
        WaitRetireIdle();
        finish_ns += static_cast<u64>((std::chrono::steady_clock::now() - finish_start).count());
        ++finishes;
    }
    ReclaimBusyResources();
    graphics_constant_uploads.fill({});
    driver_upload = {};
    const u64 replay_elapsed =
        static_cast<u64>((std::chrono::steady_clock::now() - replay_start).count());
    replay_ns += replay_elapsed;
    // With the call profile every replay that is late for a 60 Hz frame; without it only
    // one the player sees as a hitch.
    if (replay_elapsed > (profile_replay ? 13'500'000ULL : 50'000'000ULL)) {
        LOG_INFO(Render_Vulkan,
                 "NVN 2 slow replay: seq {} took {:.3f} ms for {} draws: {} pipelines made in "
                 "{:.3f} ms, {} images made, {} texture uploads in {:.3f} ms, {} ranges with {:.2f} "
                 "MiB newly kept in VRAM, {} storage buffers made, {} finishes, {:.2f} MiB "
                 "streamed, {:.3f} ms hashing, {} storage spans moved to device memory",
                 submission.sequence, replay_elapsed / 1e6, draws - draws_before,
                 pipeline_creates - slow_before[0], (pipeline_create_ns - slow_before[1]) / 1e6,
                 image_creates - slow_before[2], texture_uploads - slow_before[3],
                 (upload_ns - slow_before[4]) / 1e6, fixed_uploads - slow_before[5],
                 (fixed_upload_bytes - slow_before[6]) / 1048576.0,
                 storage_creates - slow_before[7], finishes - slow_before[8],
                 (stream_bytes - slow_before[9]) / 1048576.0, (hash_ns - slow_before[10]) / 1e6,
                 device_moves - slow_before[11]);
    }
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
        if (pipeline_list && ahead_queued + draw_built != reported_ahead) {
            LOG_INFO(Render_Vulkan,
                     "NVN 2 pipelines ahead: {} on the list, {} given to the workers, {} built "
                     "by them, {} still to build, {} a draw built first; {} built by draws in all",
                     pipeline_list->Size(), ahead_queued, ahead_built, ahead_pipelines.size(),
                     ahead_dropped, draw_built);
            reported_ahead = ahead_queued + draw_built;
        }
        if (worker_fills != reported_worker_fills) {
            LOG_INFO(Render_Vulkan, "NVN 2 texture fills: {} made by workers so far", worker_fills);
            reported_worker_fills = worker_fills;
        }
        if (pipeline_wait_draws != reported_pipeline_wait_draws) {
            LOG_INFO(Render_Vulkan,
                     "NVN 2 pipelines: {} built by workers so far, {} being built, {} draws left "
                     "out since the last report while theirs was being built",
                     worker_pipelines, pipeline_builder.Pending(),
                     pipeline_wait_draws - reported_pipeline_wait_draws);
            reported_pipeline_wait_draws = pipeline_wait_draws;
        }
        if (profile_replay && presents) {
            const auto ms = [&](u64 nanoseconds) { return double(nanoseconds) / presents / 1e6; };
            LOG_INFO(Render_Vulkan, "Bananza NVN 2 replay profile: replay_ms={:.3f} of which finish_ms={:.3f} upload_ms={:.3f}, submissions={:.1f} finishes={:.1f} refetched_MiB={:.2f} (per present)",
                     ms(replay_ns), ms(finish_ns), ms(upload_ns),
                     double(replayed_submissions) / presents, double(finishes) / presents,
                     double(hash_bytes) / presents / 1048576.0);
            LOG_INFO(Render_Vulkan, "Bananza NVN 2 replay profile: draw_ms={:.3f} dispatch_ms={:.3f}; draw phases decode={:.3f} targets={:.3f} pipeline={:.3f} vertex_index={:.3f} uniforms={:.3f} storage={:.3f} textures={:.3f} descriptors={:.3f} state={:.3f} record={:.3f} (ms per present)",
                     ms(draw_ns), ms(dispatch_ns), ms(draw_phase_ns[0]), ms(draw_phase_ns[1]),
                     ms(draw_phase_ns[2]), ms(draw_phase_ns[3]), ms(draw_phase_ns[4]),
                     ms(draw_phase_ns[5]), ms(draw_phase_ns[6]), ms(draw_phase_ns[7]),
                     ms(draw_phase_ns[8]), ms(draw_phase_ns[9]));
        }
        if (profile_replay && presents) {
            const double frames = static_cast<double>(presents);
            LOG_INFO(Render_Vulkan, "Bananza NVN 2 replay profile: stream uploads {:.0f} with {:.2f} MiB, storage buffers {:.0f} new and {:.0f} reused with {:.2f} MiB, occlusion waits {:.1f}, rendering scopes {:.0f}, barriers {:.0f}, replay flushes {:.1f}, event waits {:.3f} ms, {:.1f} signals ({:.1f} waited), {:.0f} draws and {:.0f} dispatches after their submission's last signal, write-back {:.2f} MiB of {:.2f} MiB compared, {:.1f} submissions ended without waiting, {} upload chunks (per present)",
                     stream_uploads / frames, stream_bytes / frames / 1048576.0,
                     storage_creates / frames, storage_reuses / frames, storage_bytes / frames / 1048576.0,
                     occlusion_flushes / frames, rendering_scopes / frames, barriers / frames,
                     replay_flushes / frames,
                     event_finish_ns / frames / 1e6, event_signals / frames,
                     event_signal_waits / frames,
                     event_tail_draws / frames, event_tail_dispatches / frames,
                     writeback_bytes.load() / frames / 1048576.0,
                     writeback_compared.load() / frames / 1048576.0,
                     replay_flush_ends / frames, upload_chunks.size());
        LOG_INFO(Render_Vulkan,
                 "Bananza NVN 2 replay profile: kept in VRAM: {:.0f} ranges reused, {:.0f} uploaded "
                 "with {:.2f} MiB (per present); {} chunks, {} entries, {} checked against guest "
                 "memory, {} changed without a flush",
                 fixed_hits / frames, fixed_uploads / frames, fixed_upload_bytes / frames / 1048576.0,
                 fixed_chunks.size(), fixed_cache.size(), fixed_checks, fixed_mismatches);
        LOG_INFO(Render_Vulkan,
                 "Bananza NVN 2 replay profile: {:.1f} storage spans with {:.1f} KiB in device "
                 "memory for fragment shaders, {:.1f} of them moved there (per present)",
                 device_spans / frames, device_span_bytes / frames / 1024.0, device_moves / frames);
        LOG_INFO(Render_Vulkan,
                 "Bananza NVN 2 replay profile: storage spans made: {:.2f} MiB with the flush "
                 "state they had last time, {:.2f} flushed since, {:.2f} new, {:.2f} in pools "
                 "without flush information (per present); {} spans known",
                 span_state_bytes[0] / frames / 1048576.0, span_state_bytes[1] / frames / 1048576.0,
                 span_state_bytes[2] / frames / 1048576.0, span_state_bytes[3] / frames / 1048576.0,
                 span_flush_states.size());
        LOG_INFO(Render_Vulkan,
                 "Bananza NVN 2 replay profile: kept storage: {:.2f} MiB of spans used as the GPU "
                 "left them, {:.2f} MiB filled from guest memory (per present), {} spans with "
                 "{:.1f} MiB kept; {:.2f} waits a present for an earlier submission's GPU writes, "
                 "{:.3f} ms a present",
                 kept_reused_bytes / frames / 1048576.0, kept_filled_bytes / frames / 1048576.0,
                 kept_storage.Count(), kept_storage.Bytes() / 1048576.0, guest_waits / frames,
                 guest_wait_ns / frames / 1e6);
        }
        span_state_bytes = {};
        kept_reused_bytes = kept_filled_bytes = guest_waits = guest_wait_ns = 0;
        fixed_hits = fixed_uploads = fixed_upload_bytes = fixed_checks = 0;
        device_spans = device_span_bytes = device_moves = 0;
        stream_uploads = stream_bytes = storage_creates = storage_reuses = storage_bytes = occlusion_flushes = 0;
        rendering_scopes = barriers = replay_flushes = event_finish_ns = 0;
        event_signals = event_tail_draws = event_tail_dispatches = 0;
        writeback_bytes.store(0);
        writeback_compared.store(0);
        event_signal_waits = replay_flush_ends = 0;
        draw_phase_ns = {};
        draw_ns = dispatch_ns = 0;
        replay_ns = finish_ns = hash_ns = hash_bytes = upload_ns = 0;
        replayed_submissions = finishes = 0;
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

    /// What of a variant's recorded reads the environment answers differently now; empty
    /// when it matches.
    std::string Mismatch(const VideoCommon::RecordedEnvQueries& queries) {
        missing_texture = false;
        std::string text;
        size_t listed{};
        const auto note = [&](std::string line) {
            if (listed++ < 6) {
                text += text.empty() ? "" : "; ";
                text += line;
            }
        };
        const auto handle_at = [&](const Shader::HandleKey& at) {
            const u32 first = ReadCbufValue(at.index, at.offset) << at.shift_left;
            return at.has_secondary
                       ? first | (ReadCbufValue(at.sec_index, at.sec_offset) << at.sec_shift_left)
                       : first;
        };
        for (const auto& [key, value] : queries.cbuf_values) {
            const u32 bank = static_cast<u32>(key >> 32);
            const u32 offset = static_cast<u32>(key);
            if (const u32 now = ReadCbufValue(bank, offset); now != value) {
                note(fmt::format("constant bank {} offset {:#x} was {:#x}, is {:#x}", bank, offset,
                                 value, now));
            }
        }
        for (const auto& [key, mapped] : queries.storage_pointer_types) {
            const bool current = !(key >> 63) &&
                                 IsCbufStorageBufferPointer(static_cast<u32>(key >> 32),
                                                            static_cast<u32>(key));
            if (current != bool(mapped)) {
                note(fmt::format("storage pointer at bank {} offset {:#x}",
                                 static_cast<u32>(key >> 32), static_cast<u32>(key)));
            }
        }
        for (const auto& [handle, type] : queries.texture_types) {
            if (ReadTextureType(handle) != type) {
                note(fmt::format("type of texture handle {:#x}", handle));
            }
        }
        for (const auto& [handle, format] : queries.texture_pixel_formats) {
            if (ReadTexturePixelFormat(handle) != format) {
                note(fmt::format("format of texture handle {:#x}", handle));
            }
        }
        for (const auto& [at, type] : queries.location_texture_types) {
            if (ReadTextureType(handle_at(at)) != type) {
                note(fmt::format("texture type at bank {} offset {:#x}", at.index, at.offset));
            }
        }
        for (const auto& [at, format] : queries.location_texture_formats) {
            if (ReadTexturePixelFormat(handle_at(at)) != format) {
                note(fmt::format("texture format at bank {} offset {:#x}", at.index, at.offset));
            }
        }
        for (const auto& [at, is_integer] : queries.location_integer_formats) {
            if (IsTexturePixelFormatInteger(handle_at(at)) != is_integer) {
                note(fmt::format("integer format at bank {} offset {:#x}", at.index, at.offset));
            }
        }
        if (missing_texture) {
            note("a texture is missing");
        }
        if (listed > 6) {
            text += fmt::format("; {} more", listed - 6);
        }
        return text;
    }

    std::span<const u8> Bank(u32 bank) const {
        if (bank == 0 || bank == 2) {
            return driver;
        }
        if (bank == 1) {
            return std::span{shader.code}.subspan(shader.constant_offset, shader.constant_size);
        }
        if (bank - 3 < uniforms.size()) {
            return submission.BufferSnapshot(uniforms[bank - 3]);
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
    // A dispatch decides its barriers itself; the lists are a draw's.
    draw_sampled.clear();
    draw_storage_reads.clear();
    if (!compute_program) {
        return;
    }
    const auto& program = *compute_program;
    draw_color_writes = ~0U;
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
    const auto& limits = device_limits;
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
    // By the program's contents: the game makes many program objects of one shader.
    auto& variants = compute_pipelines[program.content_hash];
    auto cached = std::ranges::find_if(variants, [&](const ComputePipeline& pipeline) {
        environment.texture_bank = pipeline.texture_bank;
        return environment.Matches(pipeline.queries);
    });
    if (cached == variants.end()) {
        if (!variants.empty() && compute_variant_notes < 60) {
            ++compute_variant_notes;
            environment.texture_bank = variants.back().texture_bank;
            LOG_INFO(Render_Vulkan,
                     "NVN 2 compute variant: program {} ({:016x}) has {} and needs another: {}",
                     program.serial, program.content_hash, variants.size(),
                     environment.Mismatch(variants.back().queries));
        }
        constexpr size_t MaxVariants = 32;
        if (variants.size() >= MaxVariants) {
            skip("compute translation variant limit");
            return;
        }
        std::optional<ComputePipeline> compiled;
        environment.missing_texture = false;
        try {
            const auto create_start = std::chrono::steady_clock::now();
            compiled = CreateComputePipeline(environment);
            ++pipeline_creates;
            pipeline_create_ns += static_cast<u64>(
                (std::chrono::steady_clock::now() - create_start).count());
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
        const auto [buffer, buffer_offset] =
            binding.bank >= 3 ? UploadRange(submission, uniform_buffers[stage][binding.bank - 3])
                              : UploadStream(bytes);
        if (!buffer) {
            return;
        }
        uniforms.push_back({buffer, buffer_offset, bytes.size()});
    }
    for (const auto& [index, descriptor] : pipeline.storage) {
        auto range = environment.StorageRange(descriptor.cbuf_index, descriptor.cbuf_offset);
        if (range.first && !range.second) {
            const auto size = submission.unsized_storage_sizes.find(range.first);
            if (size != submission.unsized_storage_sizes.end()) {
                range.second = size->second;
                if (!range.second && (device.HasNullDescriptor() || device.IsRobustBufferAccessEnabled())) {
                    range.first = 0;
                }
            }
        }
        if (!range.first && !range.second) {
            if (device.HasNullDescriptor()) {
                storage.push_back({VK_NULL_HANDLE, 0, VK_WHOLE_SIZE});
                continue;
            }
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
                buffer = upload->buffer;
                offset = upload->base + start;
                const u64 end = Shader::ResolveStorageWriteExtent(descriptor.write_extent,
                                                                  pipeline.workgroup, groups)
                                    .value_or(range.second);
                writes_to_buffers.emplace_back(upload, start + std::min(end, range.second));
                upload->NoteWritten(start, start + std::min(end, range.second));
            }
        } else {
            std::tie(buffer, offset) = UploadBufferRange(submission, range);
        }
        if (!buffer || offset % limits.minStorageBufferOffsetAlignment) {
            if (dispatch_diagnostics.emplace(program.serial, "missing storage backing").second) {
                LOG_WARNING(Render_Vulkan,
                            "NVN 2 compute program {} storage at {:#x}, {} bytes, write {}, snapshot {}, guest mapping {}",
                            program.serial, range.first, range.second, descriptor.is_written,
                            submission.HasSnapshot(range),
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
        set = AllocateSet(*pipeline.state.set_layout, uniforms.size(), storage.size(),
                          textures.size(), images_info.size());
        if (!set) {
            skip("descriptor set unavailable");
            return;
        }
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
    }
    Barrier();
    Record([groups, handle = *pipeline.state.pipeline, pipeline_layout = *pipeline.state.layout,
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
        upload->dirty = true;
        upload->write_tick = scheduler.CurrentTick();
        if (upload->readback_buffer) {
            upload->readback_stale = true;
            device_readback_pending = true;
        }
    }
    ++dispatches;
}

} // namespace Vulkan
