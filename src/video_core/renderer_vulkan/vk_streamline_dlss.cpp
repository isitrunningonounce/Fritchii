// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS (through NVIDIA Streamline) in place of a game's DLSS evaluation.

#include <array>
#include <atomic>
#include <cstdlib>
#include <string_view>
#include <fmt/ranges.h>

#include "common/frame_gen_stats.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "video_core/dlss_evaluation.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_streamline_dlss.h"
#include "video_core/vulkan_common/streamline.h"
#include "video_core/vulkan_common/streamline_api.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan {
namespace {

constexpr VkFormat OUTPUT_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
// Streamline clears the output before DLSS writes it.
constexpr VkImageUsageFlags OUTPUT_USAGE = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_DST_BIT;
constexpr VkFormat EXPOSURE_FORMAT = VK_FORMAT_R32_SFLOAT;
constexpr VkImageUsageFlags EXPOSURE_USAGE =
    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
constexpr VkExtent2D EXPOSURE_SIZE{1, 1};

bool InRenderRange(VkExtent2D size, VkExtent2D minimum, VkExtent2D maximum) {
    return minimum.width && minimum.height && minimum.width <= size.width &&
           minimum.height <= size.height && size.width <= maximum.width &&
           size.height <= maximum.height;
}

int EnvInt(const char* name, int fallback) {
    const char* const value = std::getenv(name);
    return value ? std::atoi(value) : fallback;
}

// DLSS validates the render size against the mode's range: pick the mode for this ratio.
sl::DLSSMode ModeFor(VkExtent2D render, VkExtent2D upscale) {
    const f32 ratio = static_cast<f32>(render.width) / static_cast<f32>(upscale.width);
    if (ratio >= 0.99f) {
        return sl::DLSSMode::eDLAA;
    }
    if (ratio >= 0.65f) {
        return sl::DLSSMode::eMaxQuality;
    }
    if (ratio >= 0.57f) {
        return sl::DLSSMode::eBalanced;
    }
    if (ratio >= 0.49f) {
        return sl::DLSSMode::eMaxPerformance;
    }
    return sl::DLSSMode::eUltraPerformance;
}

// Preset K for every mode: NVIDIA's defaults for Performance/Ultra Performance (M, L) draw thin
// horizontal lines in Fire Emblem at 2x resolution scale. EDEN_NX2_DLSS_PRESET=J|K|L|M picks
// another model, EDEN_NX2_DLSS_PRESET=default leaves the choice to NVIDIA.
sl::DLSSPreset Preset() {
    const char* const value = std::getenv("EDEN_NX2_DLSS_PRESET");
    if (value && value[0] >= 'J' && value[0] <= 'M') {
        return static_cast<sl::DLSSPreset>(static_cast<u32>(sl::DLSSPreset::ePresetJ) +
                                           static_cast<u32>(value[0] - 'J'));
    }
    if (value && std::string_view{value} == "default") {
        return sl::DLSSPreset::eDefault;
    }
    // Graphics > DLSS preset.
    switch (Settings::values.nvidia_dlss_preset.GetValue()) {
    case Settings::DlssPreset::Default:
        return sl::DLSSPreset::eDefault;
    case Settings::DlssPreset::J:
        return sl::DLSSPreset::ePresetJ;
    case Settings::DlssPreset::L:
        return sl::DLSSPreset::ePresetL;
    case Settings::DlssPreset::M:
        return sl::DLSSPreset::ePresetM;
    case Settings::DlssPreset::K:
    default:
        return sl::DLSSPreset::ePresetK;
    }
}

sl::float4x4 Identity() {
    sl::float4x4 matrix;
    matrix.setRow(0, {1.0f, 0.0f, 0.0f, 0.0f});
    matrix.setRow(1, {0.0f, 1.0f, 0.0f, 0.0f});
    matrix.setRow(2, {0.0f, 0.0f, 1.0f, 0.0f});
    matrix.setRow(3, {0.0f, 0.0f, 0.0f, 1.0f});
    return matrix;
}

VkImageMemoryBarrier Barrier(const StreamlineDlssUpscaler::Image& image, VkAccessFlags src_access,
                             VkAccessFlags dst_access, VkImageLayout old_layout,
                             VkImageLayout new_layout) {
    VkImageSubresourceRange range = image.range;
    range.aspectMask = image.barrier_aspect;
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image.image,
        .subresourceRange = range,
    };
}

sl::Resource Resource(VkImage image, VkImageView view, VkFormat format, VkImageLayout layout,
                      VkExtent2D size, VkImageUsageFlags usage) {
    sl::Resource resource{sl::ResourceType::eTex2d, image, nullptr, view,
                          static_cast<uint32_t>(layout)};
    resource.width = size.width;
    resource.height = size.height;
    resource.nativeFormat = static_cast<uint32_t>(format);
    resource.mipLevels = 1;
    resource.arrayLayers = 1;
    resource.flags = 0;
    resource.usage = usage;
    return resource;
}

// LICHII_DLSS_FG_COPY=0: let Streamline copy frame generation's inputs (eOnlyValidNow).
bool FrameGenOwnCopies() {
    static const bool own = [] {
        const char* const value = std::getenv("LICHII_DLSS_FG_COPY");
        return !value || value[0] != '0';
    }();
    return own;
}

constexpr VkImageUsageFlags FG_COPY_USAGE =
    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

} // Anonymous namespace

StreamlineDlssUpscaler::FrameGenCopy StreamlineDlssUpscaler::CreateFrameGenCopy(const Inputs& in) {
    const auto create = [&](VkFormat format, VkImageAspectFlags aspect, vk::Image& image,
                            vk::ImageView& view) {
        image = memory_allocator.CreateImage(VkImageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = format,
            .extent = {in.render_size.width, in.render_size.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = FG_COPY_USAGE,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });
        view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = *image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .components{},
            .subresourceRange{
                .aspectMask = aspect,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        });
    };
    FrameGenCopy copy;
    create(in.depth.format, VK_IMAGE_ASPECT_DEPTH_BIT, copy.depth, copy.depth_view);
    create(in.motion_vectors.format, VK_IMAGE_ASPECT_COLOR_BIT, copy.motion, copy.motion_view);
    return copy;
}

void StreamlineDlssUpscaler::CreateFrameGenCopies(const Inputs& in) {
    if (fg_copies[0].depth) {
        scheduler.Finish(); // frame generation may still read the old ones
    }
    for (auto& slot : fg_copies) {
        auto copy = CreateFrameGenCopy(in);
        std::swap(slot, copy);
    }
    fg_copy_extent = in.render_size;
    fg_depth_format = in.depth.format;
    fg_motion_format = in.motion_vectors.format;
    LOG_INFO(Render_Vulkan, "DLSS Frame Generation: {} copies of depth and motion at {}x{}",
             FRAME_GEN_COPIES, in.render_size.width, in.render_size.height);
}

StreamlineDlssUpscaler::BananzaFrameGenCopy* StreamlineDlssUpscaler::AcquireBananzaFrameGenCopy(
    const Inputs& in) {
    scheduler.GetMasterSemaphore().Refresh();
    for (size_t offset = 0; offset < bananza_fg_copies.size(); ++offset) {
        const size_t index = (bananza_fg_copy_next + offset) % bananza_fg_copies.size();
        auto& slot = bananza_fg_copies[index];
        if (!scheduler.IsFree(slot.tick) || !slot.fence.Ready()) {
            continue;
        }
        if (!slot.images.depth || slot.extent.width != in.render_size.width ||
            slot.extent.height != in.render_size.height || slot.depth_format != in.depth.format ||
            slot.motion_format != in.motion_vectors.format) {
            auto copy = CreateFrameGenCopy(in);
            std::swap(slot.images, copy);
            slot.extent = in.render_size;
            slot.depth_format = in.depth.format;
            slot.motion_format = in.motion_vectors.format;
        }
        slot.tick = scheduler.CurrentTick();
        slot.fence.Reserve();
        bananza_fg_copy_next = (index + 1) % bananza_fg_copies.size();
        return &slot;
    }
    return nullptr;
}

StreamlineDlssUpscaler::StreamlineDlssUpscaler(const Device& device_, Scheduler& scheduler_,
                                               MemoryAllocator& memory_allocator_)
    : device{device_}, scheduler{scheduler_}, memory_allocator{memory_allocator_} {
    if (!Streamline::IsDlssAvailable()) {
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    const sl::Result result = Streamline::GetApi().get_feature_function(
        sl::kFeatureDLSS, "slDLSSSetOptions", set_options);
    if (result != sl::Result::eOk || !set_options) {
        LOG_ERROR(Render_Vulkan, "Streamline's DLSS has no slDLSSSetOptions (result {})",
                  static_cast<int>(result));
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    point_sampler = device.GetLogical().CreateSampler(VkSamplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_NEVER,
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    });
}

StreamlineDlssUpscaler::~StreamlineDlssUpscaler() {
    // Not scheduler.Finish(): this runs when the renderer goes away, after the game's channels are
    // released, and a submit would touch their state. Nothing more gets recorded by then.
    void(device.GetLogical().WaitIdle());
    if (Streamline::IsDlssAvailable()) {
        Streamline::GetApi().free_resources(sl::kFeatureDLSS, sl::ViewportHandle{0u});
    }
}

void StreamlineDlssUpscaler::CreateOutput(VkExtent2D upscale_size) {
    if (output_image) {
        scheduler.Finish(); // the old output may still be in use
    }
    output_image = memory_allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = OUTPUT_FORMAT,
        .extent = {upscale_size.width, upscale_size.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = OUTPUT_USAGE,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    output_view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *output_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = OUTPUT_FORMAT,
        .components{},
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    });
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image = *output_image](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                               VkImageMemoryBarrier{
                                   .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                   .pNext = nullptr,
                                   .srcAccessMask = 0,
                                   .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                   .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                   .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = image,
                                   .subresourceRange{
                                       .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                       .baseMipLevel = 0,
                                       .levelCount = 1,
                                       .baseArrayLayer = 0,
                                       .layerCount = 1,
                                   },
                               });
    });
}

void StreamlineDlssUpscaler::CreateBananzaExposure() {
    bananza_exposure = memory_allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = EXPOSURE_FORMAT,
        .extent = {EXPOSURE_SIZE.width, EXPOSURE_SIZE.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = EXPOSURE_USAGE,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    constexpr VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    bananza_exposure_view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *bananza_exposure,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = EXPOSURE_FORMAT,
        .components{},
        .subresourceRange = range,
    });
    const Image exposure{*bananza_exposure_view, *bananza_exposure, EXPOSURE_FORMAT, range,
                         VK_IMAGE_ASPECT_COLOR_BIT};
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([exposure](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
            Barrier(exposure, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
        constexpr VkClearColorValue neutral{.float32 = {1.0f, 0.0f, 0.0f, 0.0f}};
        cmdbuf.ClearColorImage(exposure.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, neutral,
                               exposure.range);
        cmdbuf.PipelineBarrier(
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
            Barrier(exposure, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
    });
}

bool StreamlineDlssUpscaler::Upscale(const Inputs& in) {
    if (failed) {
        return false;
    }
    const bool render_changed = in.render_size.width != render_extent.width ||
                                in.render_size.height != render_extent.height;
    const bool extent_changed = !output_image || in.upscale_size.width != upscale_extent.width ||
                                in.upscale_size.height != upscale_extent.height ||
                                (!in.bananza_dynamic && render_changed);
    const bool options_changed = extent_changed || in.hdr != hdr_input;
    const bool reset = in.reset || options_changed;
    if (extent_changed) {
        CreateOutput(in.upscale_size);
        upscale_extent = in.upscale_size;
    }
    render_extent = in.render_size;
    hdr_input = in.hdr;
    if (in.bananza_dynamic && !bananza_exposure) {
        CreateBananzaExposure();
    }

    const bool fg_enabled = Streamline::IsFrameGenAvailable() &&
        Common::FrameGenStats::user_enabled.load(std::memory_order_relaxed);
    const bool bananza_own_copies = in.bananza_dynamic && Streamline::BananzaFrameGenOwnCopies();
    BananzaFrameGenCopy* bananza_copy = nullptr;
    if (fg_enabled && bananza_own_copies) {
        if (in.depth_transfer_src) {
            try {
                bananza_copy = AcquireBananzaFrameGenCopy(in);
            } catch (const vk::Exception& error) {
                if (!std::exchange(bananza_fg_fallback_logged, true)) {
                    LOG_WARNING(Render_Vulkan, "Bananza temporal: FG copy allocation failed ({})",
                                error.what());
                }
            }
        }
        if (!bananza_copy && !std::exchange(bananza_fg_fallback_logged, true)) {
            LOG_WARNING(Render_Vulkan, "Bananza temporal: FG copy unavailable ({})",
                        in.depth_transfer_src ? "all slots busy, using Streamline copies"
                                              : "scene depth lacks transfer-source usage");
        }
    }
    const FrameGenCopy* fg_copy = nullptr;
    if (!in.bananza_dynamic && fg_enabled && FrameGenOwnCopies()) {
        if (!fg_copies[0].depth || fg_copy_extent.width != in.render_size.width ||
            fg_copy_extent.height != in.render_size.height ||
            fg_depth_format != in.depth.format || fg_motion_format != in.motion_vectors.format) {
            CreateFrameGenCopies(in);
        }
        fg_copy = &fg_copies[fg_copy_next++ % FRAME_GEN_COPIES];
    }

    // Keep the existing HDR controls for linear input such as Fortune's Weave. Tonemapped
    // input such as Ocarina's already has its exposure applied.
    sl::DLSSOptions options{};
    options.mode = ModeFor(in.render_size, in.upscale_size);
    options.outputWidth = in.upscale_size.width;
    options.outputHeight = in.upscale_size.height;
    options.colorBuffersHDR =
        in.hdr && EnvInt("EDEN_NX2_DLSS_HDR", 1) != 0 ? sl::eTrue : sl::eFalse;
    options.useAutoExposure =
        in.hdr && EnvInt("EDEN_NX2_DLSS_AUTO_EXPOSURE", 1) != 0 ? sl::eTrue : sl::eFalse;
    const sl::DLSSPreset preset = Preset();
    options.dlaaPreset = preset;
    options.qualityPreset = preset;
    options.balancedPreset = preset;
    options.performancePreset = preset;
    options.ultraPerformancePreset = preset;
    options.ultraQualityPreset = preset;
    if (!in.bananza_dynamic && options_changed) {
        LOG_INFO(Render_Vulkan, "NVIDIA DLSS (Streamline): {}x{} -> {}x{}, mode {}, preset {}, {}",
                 in.render_size.width, in.render_size.height, in.upscale_size.width,
                 in.upscale_size.height, static_cast<int>(options.mode), static_cast<int>(preset),
                 options.colorBuffersHDR == sl::eTrue ? "HDR" : "LDR");
    }

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([this, in, reset, initial_options = options, options_changed, render_changed,
                      index = frame_index++, fg_copy, bananza_copy, bananza_own_copies,
                      output_image_handle = *output_image,
                      output_view_handle = *output_view](vk::CommandBuffer cmdbuf) {
        bool copy_handed_off = false;
        SCOPE_EXIT {
            if (bananza_copy && !copy_handed_off) {
                bananza_copy->fence.Release();
            }
        };
        const Streamline::Api& api = Streamline::GetApi();
        const sl::ViewportHandle viewport{0u};
        // The caller draws the output this frame even when DLSS writes nothing to it: clear it
        // then, rather than show undefined contents (FSR 3.1 takes over from the next frame).
        const auto clear_output = [&] {
            const VkImageSubresourceRange range{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            };
            const auto barrier = [&](VkAccessFlags src_access, VkAccessFlags dst_access) {
                return VkImageMemoryBarrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = src_access,
                    .dstAccessMask = dst_access,
                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = output_image_handle,
                    .subresourceRange = range,
                };
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                   barrier(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                           VK_ACCESS_TRANSFER_WRITE_BIT));
            cmdbuf.ClearColorImage(output_image_handle, VK_IMAGE_LAYOUT_GENERAL,
                                   VkClearColorValue{}, range);
            cmdbuf.PipelineBarrier(
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT));
        };
        auto frame_options = initial_options;
        if (in.bananza_dynamic) {
            if (failed.load()) {
                return;
            }
            if (!bananza_mode) {
                void* address{};
                const auto result = api.get_feature_function(
                    sl::kFeatureDLSS, "slDLSSGetOptimalSettings", address);
                if (result == sl::Result::eOk && address) {
                    const auto get = reinterpret_cast<PFun_slDLSSGetOptimalSettings*>(address);
                    constexpr std::array modes{
                        sl::DLSSMode::eMaxQuality, sl::DLSSMode::eBalanced,
                        sl::DLSSMode::eMaxPerformance, sl::DLSSMode::eUltraQuality,
                        sl::DLSSMode::eUltraPerformance, sl::DLSSMode::eDLAA,
                    };
                    constexpr VkExtent2D captured_minimum{2095, 1179};
                    for (const auto mode : modes) {
                        frame_options.mode = mode;
                        sl::DLSSOptimalSettings settings{};
                        if (get(frame_options, settings) != sl::Result::eOk) {
                            continue;
                        }
                        const VkExtent2D minimum{settings.renderWidthMin, settings.renderHeightMin};
                        const VkExtent2D maximum{settings.renderWidthMax, settings.renderHeightMax};
                        if (InRenderRange(captured_minimum, minimum, maximum) &&
                            InRenderRange(in.upscale_size, minimum, maximum) &&
                            InRenderRange(in.render_size, minimum, maximum)) {
                            bananza_mode = DynamicMode{static_cast<u32>(mode), minimum, maximum};
                            LOG_INFO(Render_Vulkan,
                                     "Bananza temporal: DLSS mode {} render range {}x{} to {}x{}",
                                     bananza_mode->mode, minimum.width, minimum.height,
                                     maximum.width, maximum.height);
                            break;
                        }
                    }
                }
                if (!bananza_mode) {
                    LOG_WARNING(Render_Vulkan,
                                "Bananza temporal: no DLSS mode covers 2095x1179 through {}x{} "
                                "and the current {}x{} scene (optimal settings result {})",
                                in.upscale_size.width, in.upscale_size.height, in.render_size.width,
                                in.render_size.height, static_cast<int>(result));
                    failed.store(true);
                    return;
                }
            }
            const auto& mode = *bananza_mode;
            if (!InRenderRange(in.render_size, mode.minimum, mode.maximum)) {
                LOG_WARNING(Render_Vulkan,
                            "Bananza temporal: scene {}x{} falls outside the DLSS render range",
                            in.render_size.width, in.render_size.height);
                failed.store(true);
                return;
            }
            frame_options.mode = static_cast<sl::DLSSMode>(mode.mode);
            if (options_changed || render_changed) {
                LOG_INFO(Render_Vulkan,
                         "NVIDIA DLSS (Streamline): {}x{} -> {}x{}, mode {}, preset {}, LDR",
                         in.render_size.width, in.render_size.height, in.upscale_size.width,
                         in.upscale_size.height, mode.mode, static_cast<int>(frame_options.qualityPreset));
            }
        }
        sl::FrameToken* frame{};

        constexpr VkAccessFlags WRITES =
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        // The inputs come from the game's draws and dispatches and stay in GENERAL otherwise;
        // DLSS reads them as sampled images.
        const std::array to_read{
            Barrier(in.color, WRITES, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(in.depth, WRITES, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(in.motion_vectors, WRITES, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        };

        // Keep the existing defaults for callers without a measured camera contract.
        const f32 rw = static_cast<f32>(in.render_size.width);
        const f32 rh = static_cast<f32>(in.render_size.height);
        sl::Constants constants{};
        constants.cameraViewToClip = Identity();
        constants.clipToCameraView = Identity();
        constants.clipToLensClip = Identity();
        constants.clipToPrevClip = Identity();
        constants.prevClipToClip = Identity();
        constants.jitterOffset = {in.jitter[0], in.jitter[1]};
        // Streamline wants motion in [-1,1] screen units; ours turns into render pixels.
        constants.mvecScale = {in.motion_vector_scale[0] / rw, in.motion_vector_scale[1] / rh};
        constants.cameraPinholeOffset = {0.0f, 0.0f};
        constants.cameraPos = {0.0f, 0.0f, 0.0f};
        constants.cameraUp = {0.0f, 1.0f, 0.0f};
        constants.cameraRight = {1.0f, 0.0f, 0.0f};
        constants.cameraFwd = {0.0f, 0.0f, 1.0f};
        constants.cameraNear = 0.1f;
        constants.cameraFar = 1000.0f;
        constants.cameraFOV = 1.0f;
        constants.cameraAspectRatio = rw / rh;
        constants.depthInverted = in.depth_inverted ? sl::eTrue : sl::eFalse;
        constants.cameraMotionIncluded = sl::eTrue;
        constants.motionVectors3D = sl::eFalse;
        constants.reset = reset ? sl::eTrue : sl::eFalse;
        constants.motionVectorsJittered = sl::eFalse;
        if (in.camera) {
            const auto& camera = *in.camera;
            const auto matrix = [](const std::array<f32, 16>& values) {
                sl::float4x4 result;
                for (u32 row = 0; row < 4; ++row) {
                    result.setRow(row, {values[row * 4], values[row * 4 + 1],
                                        values[row * 4 + 2], values[row * 4 + 3]});
                }
                return result;
            };
            const auto vector = [](const std::array<f32, 3>& values) {
                return sl::float3{values[0], values[1], values[2]};
            };
            constants.cameraViewToClip = matrix(camera.view_to_clip);
            constants.clipToCameraView = matrix(camera.clip_to_view);
            constants.clipToPrevClip = matrix(camera.clip_to_previous_clip);
            constants.prevClipToClip = matrix(camera.previous_clip_to_clip);
            constants.cameraPos = vector(camera.position);
            constants.cameraRight = vector(camera.right);
            constants.cameraUp = vector(camera.up);
            constants.cameraFwd = vector(camera.forward);
            constants.cameraNear = camera.near_plane;
            constants.cameraFar = camera.far_plane;
            constants.cameraFOV = camera.vertical_fov;
            constants.cameraAspectRatio = camera.aspect_ratio;
            static const bool trace = std::getenv("EDEN_NX2_DLSS_CONTRACT_TRACE") != nullptr;
            if (trace || index == 0 || reset || index % 120 == 0) {
                LOG_INFO(Render_Vulkan,
                         "{} temporal: FG camera frame={} render={}x{} "
                         "near={} far={} fov={} aspect={} position=({},{},{}) "
                         "projection={:g} clip_to_previous={:g} "
                         "jitter=({},{}) mv_scale=({},{}) depth_inverted={}",
                         in.bananza_dynamic ? "Bananza" : "Mario Kart World",
                         index, in.render_size.width, in.render_size.height,
                         camera.near_plane, camera.far_plane, camera.vertical_fov, camera.aspect_ratio,
                         camera.position[0], camera.position[1], camera.position[2],
                         fmt::join(camera.view_to_clip, ","),
                         fmt::join(camera.clip_to_previous_clip, ","),
                         in.jitter[0], in.jitter[1], constants.mvecScale.x, constants.mvecScale.y,
                         in.depth_inverted);
            }
        }

        sl::Resource color = Resource(in.color.image, in.color.view, in.color.format,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, in.render_size,
                                      VK_IMAGE_USAGE_SAMPLED_BIT);
        const VkExtent2D depth_size = in.bananza_dynamic ? in.depth_allocation_size : in.render_size;
        const VkImageUsageFlags scene_usage = in.bananza_dynamic
                                                 ? VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                                                 : VK_IMAGE_USAGE_SAMPLED_BIT;
        sl::Resource depth = Resource(in.depth.image, in.depth.view, in.depth.format,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, depth_size,
                                      scene_usage);
        sl::Resource motion = Resource(in.motion_vectors.image, in.motion_vectors.view,
                                       in.motion_vectors.format,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, in.render_size,
                                       scene_usage);
        sl::Resource output = Resource(output_image_handle, output_view_handle, OUTPUT_FORMAT,
                                       VK_IMAGE_LAYOUT_GENERAL, in.upscale_size, OUTPUT_USAGE);
        const sl::Extent render_area{0, 0, in.render_size.width, in.render_size.height};
        const sl::Extent output_area{0, 0, in.upscale_size.width, in.upscale_size.height};
        // Local tags: everything stays put until the evaluation is recorded.
        sl::ResourceTag color_tag{&color, sl::kBufferTypeScalingInputColor,
                                  sl::ResourceLifecycle::eValidUntilEvaluate, &render_area};
        sl::ResourceTag depth_tag{&depth, sl::kBufferTypeDepth,
                                  sl::ResourceLifecycle::eValidUntilEvaluate, &render_area};
        sl::ResourceTag motion_tag{&motion, sl::kBufferTypeMotionVectors,
                                   sl::ResourceLifecycle::eValidUntilEvaluate, &render_area};
        sl::ResourceTag output_tag{&output, sl::kBufferTypeScalingOutputColor,
                                   sl::ResourceLifecycle::eValidUntilEvaluate, &output_area};
        sl::Resource exposure{};
        if (in.bananza_dynamic) {
            exposure = Resource(*bananza_exposure, *bananza_exposure_view, EXPOSURE_FORMAT,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, EXPOSURE_SIZE,
                                EXPOSURE_USAGE);
        }
        const sl::Extent exposure_area{0, 0, EXPOSURE_SIZE.width, EXPOSURE_SIZE.height};
        const sl::ResourceTag exposure_tag{&exposure, sl::kBufferTypeExposure,
                                           sl::ResourceLifecycle::eValidUntilEvaluate,
                                           &exposure_area};
        const sl::BaseStructure* inputs[] = {&viewport, &color_tag, &depth_tag, &motion_tag,
                                             &output_tag, &exposure_tag};
        const auto input_count = std::size(inputs) - (in.bananza_dynamic ? 0u : 1u);
        const std::array to_general{
            Barrier(in.color, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(in.depth, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(in.motion_vectors, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        };
        const VkMemoryBarrier output_written{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        const auto copy_and_tag = [&](const FrameGenCopy& copy, Streamline::FrameGenCopyFence* fence) {
            // Copy depth and motion (read-only for DLSS right now) into this frame's slot.
            const auto copy_image = [](VkImage image, VkImageAspectFlags aspect) {
                return StreamlineDlssUpscaler::Image{
                    .view = VK_NULL_HANDLE,
                    .image = image,
                    .format = VK_FORMAT_UNDEFINED,
                    .range = {aspect, 0, 1, 0, 1},
                    .barrier_aspect = aspect,
                };
            };
            const auto depth_copy = copy_image(*copy.depth, in.depth.barrier_aspect);
            const auto motion_copy = copy_image(*copy.motion, VK_IMAGE_ASPECT_COLOR_BIT);
            constexpr VkImageLayout READ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            constexpr VkImageLayout SRC = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            constexpr VkImageLayout DST = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            const std::array before_copy{
                Barrier(in.depth, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                        READ, SRC),
                Barrier(in.motion_vectors, VK_ACCESS_SHADER_READ_BIT,
                        VK_ACCESS_TRANSFER_READ_BIT, READ, SRC),
                Barrier(depth_copy, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_IMAGE_LAYOUT_UNDEFINED, DST),
                Barrier(motion_copy, VK_ACCESS_SHADER_READ_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, DST),
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, {}, {}, before_copy);
            const auto region = [&](const StreamlineDlssUpscaler::Image& source,
                                    VkImageAspectFlags aspect) {
                return VkImageCopy{
                    .srcSubresource = {aspect, source.range.baseMipLevel,
                                       source.range.baseArrayLayer, 1},
                    .srcOffset = {0, 0, 0},
                    .dstSubresource = {aspect, 0, 0, 1},
                    .dstOffset = {0, 0, 0},
                    .extent = {in.render_size.width, in.render_size.height, 1},
                };
            };
            cmdbuf.CopyImage(in.depth.image, SRC, *copy.depth, DST,
                             region(in.depth, VK_IMAGE_ASPECT_DEPTH_BIT));
            cmdbuf.CopyImage(in.motion_vectors.image, SRC, *copy.motion, DST,
                             region(in.motion_vectors, VK_IMAGE_ASPECT_COLOR_BIT));
            const std::array after_copy{
                Barrier(in.depth, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                        SRC, READ),
                Barrier(in.motion_vectors, VK_ACCESS_TRANSFER_READ_BIT,
                        VK_ACCESS_SHADER_READ_BIT, SRC, READ),
                Barrier(depth_copy, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                        DST, READ),
                Barrier(motion_copy, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                        DST, READ),
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, {}, {}, after_copy);
            const sl::Resource depth_tagged =
                Resource(*copy.depth, *copy.depth_view, in.depth.format, READ,
                         in.render_size, FG_COPY_USAGE);
            const sl::Resource motion_tagged =
                Resource(*copy.motion, *copy.motion_view, in.motion_vectors.format,
                         READ, in.render_size, FG_COPY_USAGE);
            Streamline::TagFrameGen(*frame, depth_tagged, motion_tagged, render_area, *cmdbuf,
                                    sl::ResourceLifecycle::eValidUntilPresent, fence);
        };
        sl::Result api_result = sl::Result::eOk;
        const auto result = VideoCommon::RecordDlssEvaluation(
            [&] {
                if (!options_changed) {
                    return true;
                }
                const auto set = reinterpret_cast<PFun_slDLSSSetOptions*>(set_options);
                api_result = set(viewport, frame_options);
                return api_result == sl::Result::eOk;
            },
            [&] {
                api_result = api.get_new_frame_token(frame, &index);
                return api_result == sl::Result::eOk && frame;
            },
            [&] {
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {}, to_read);
            },
            [&] {
                api_result = api.set_constants(constants, *frame, viewport);
                if (api_result != sl::Result::eOk) {
                    return false;
                }
                if (in.bananza_dynamic) {
                    return true;
                }
                // DLSS Frame Generation (when on) presents from this frame's depth and motion.
                if (!fg_copy) {
                    Streamline::TagFrameGen(*frame, depth, motion, render_area, *cmdbuf,
                                            sl::ResourceLifecycle::eOnlyValidNow);
                    return true;
                }
                copy_and_tag(*fg_copy, nullptr);
                return true;
            },
            [&] {
                api_result = api.evaluate_feature(sl::kFeatureDLSS, *frame, inputs, input_count,
                                                  reinterpret_cast<sl::CommandBuffer*>(*cmdbuf));
                const bool evaluated = api_result == sl::Result::eOk ||
                                       api_result == sl::Result::eWarnOutOfVRAM;
                if (evaluated && in.bananza_dynamic) {
                    if (bananza_copy) {
                        copy_and_tag(bananza_copy->images, &bananza_copy->fence);
                        copy_handed_off = true;
                    } else if (!bananza_own_copies || in.depth_transfer_src) {
                        Streamline::TagFrameGen(*frame, depth, motion, render_area, *cmdbuf,
                                                sl::ResourceLifecycle::eOnlyValidNow);
                    }
                }
                // eWarnOutOfVRAM is a warning: the evaluation was recorded. Treating it as a
                // failure turned DLSS off for the session on an 8 GB card with other apps open.
                if (api_result == sl::Result::eWarnOutOfVRAM) {
                    static std::atomic<bool> warned{};
                    if (!warned.exchange(true)) {
                        LOG_WARNING(Render_Vulkan, "NVIDIA DLSS: video memory is over budget "
                                                   "(Streamline warning); DLSS keeps running");
                    }
                    return true;
                }
                return api_result == sl::Result::eOk;
            },
            [&] {
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, output_written, {},
                                       to_general);
            });
        if (result != VideoCommon::DlssEvaluationResult::Succeeded && !failed.exchange(true)) {
            LOG_ERROR(Render_Vulkan,
                      "NVIDIA DLSS recording failed at stage {} (Streamline result {}); using {}",
                      static_cast<int>(result), static_cast<int>(api_result),
                      in.bananza_dynamic ? "the game's pass" : "FSR 3.1");
        }
        if (!in.bananza_dynamic && result != VideoCommon::DlssEvaluationResult::Succeeded) {
            clear_output();
        }
    });
    // DLSS binds its own pipelines and descriptors on the command buffer.
    scheduler.InvalidateState();
    return true;
}

} // namespace Vulkan
