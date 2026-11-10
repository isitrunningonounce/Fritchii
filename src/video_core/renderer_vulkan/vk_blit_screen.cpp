// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2024 Torzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <vulkan/vulkan_core.h>
#include <cstdlib>
#include <exception>
#include "common/frame_gen_stats.h"
#include "common/game_presets.h"
#include "common/logging.h"
#include "video_core/mkw_hudless.h"
#ifdef HAS_RESHADE
#include "video_core/post_processing/fx_chain.h"
#endif
#ifdef EGLANTINE_STREAMLINE
#include "video_core/vulkan_common/streamline.h"
#endif
#include "common/settings.h"
#include "video_core/framebuffer_config.h"
#include "video_core/present.h"
#include "video_core/renderer_vulkan/present/filters.h"
#include "video_core/renderer_vulkan/present/layer.h"
#include "video_core/renderer_vulkan/vk_blit_screen.h"
#include "video_core/renderer_vulkan/vk_present_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

BlitScreen::BlitScreen(Tegra::MaxwellDeviceMemoryManager& device_memory_, const Device& device_, MemoryAllocator& memory_allocator_, PresentManager& present_manager_, Scheduler& scheduler_, const PresentFilters& filters_)
    : device_memory{device_memory_}
    , memory_allocator{memory_allocator_}
    , present_manager{present_manager_}
    , scheduler{scheduler_}
    , filters{filters_}
    , image_count{1}
    , image_index{0}
    , swapchain_view_format{VK_FORMAT_B8G8R8A8_UNORM}
{}

BlitScreen::~BlitScreen() = default;

void BlitScreen::WaitIdle(const Device& device) {
    present_manager.WaitPresent();
    scheduler.Finish();
    device.GetLogical().WaitIdle();
}

void BlitScreen::SetWindowAdaptPass(const Device& device) {
    layers.clear();
    hudless_layers.clear();
    hudless_failed = false;
    scaling_filter = filters.get_scaling_filter();

    switch (scaling_filter) {
    case Settings::ScalingFilter::NearestNeighbor:
        window_adapt = MakeNearestNeighbor(device, swapchain_view_format);
        break;
    case Settings::ScalingFilter::Bicubic:
        window_adapt = MakeBicubic(device, swapchain_view_format, VK_CUBIC_FILTER_WEIGHTS_CATMULL_ROM_QCOM);
        break;
    case Settings::ScalingFilter::ZeroTangent:
        window_adapt = MakeBicubic(device, swapchain_view_format, VK_CUBIC_FILTER_WEIGHTS_ZERO_TANGENT_CARDINAL_QCOM);
        break;
    case Settings::ScalingFilter::BSpline:
        window_adapt = MakeBicubic(device, swapchain_view_format, VK_CUBIC_FILTER_WEIGHTS_B_SPLINE_QCOM);
        break;
    case Settings::ScalingFilter::Mitchell:
        window_adapt = MakeBicubic(device, swapchain_view_format, VK_CUBIC_FILTER_WEIGHTS_MITCHELL_NETRAVALI_QCOM);
        break;
    case Settings::ScalingFilter::Spline1:
        window_adapt = MakeSpline1(device, swapchain_view_format);
        break;
    case Settings::ScalingFilter::Gaussian:
        window_adapt = MakeGaussian(device, swapchain_view_format);
        break;
    case Settings::ScalingFilter::Lanczos:
        window_adapt = MakeLanczos(device, swapchain_view_format);
        break;
    case Settings::ScalingFilter::ScaleForce:
        window_adapt = MakeScaleForce(device, swapchain_view_format);
        break;
    case Settings::ScalingFilter::Area:
        window_adapt = MakeArea(device, swapchain_view_format);
        break;
    case Settings::ScalingFilter::Mmpx:
        window_adapt = MakeMmpx(device, swapchain_view_format);
        break;
    case Settings::ScalingFilter::Fsr:
    case Settings::ScalingFilter::Fsr3:
    case Settings::ScalingFilter::Sgsr:
    case Settings::ScalingFilter::SgsrEdge:
    case Settings::ScalingFilter::Bilinear:
    default:
        window_adapt = MakeBilinear(device, swapchain_view_format);
        break;
    }
}

void BlitScreen::PrepareFrame(const Device& device, Frame* frame,
                              const Layout::FramebufferLayout& layout) {
    if (!window_adapt) {
        return;
    }

    if (frame->width != layout.width || frame->height != layout.height) {
        WaitIdle(device);
    } else if (!present_manager.NeedsStorage(frame, true)) {
        return;
    }

    present_manager.RecreateFrame(frame, layout.width, layout.height, swapchain_view_format,
                                  window_adapt->GetRenderPass(), true);
}

void BlitScreen::DrawToFrame(const Device& device, RasterizerVulkan& rasterizer, Frame* frame,
                             std::span<const Tegra::FramebufferConfig> framebuffers,
                             const Layout::FramebufferLayout& layout,
                             size_t current_swapchain_image_count,
                             VkFormat current_swapchain_view_format, bool prepare_hudless) {
    bool resource_update_required = false;
    bool presentation_recreate_required = false;

    if (!window_adapt || scaling_filter != filters.get_scaling_filter()) {
        resource_update_required = true;
    }

    if (image_count != current_swapchain_image_count) {
        resource_update_required = true;
        image_count = current_swapchain_image_count;
    }

    if (swapchain_view_format != current_swapchain_view_format ||
        layout.width != frame->width || layout.height != frame->height) {
        resource_update_required = true;
        presentation_recreate_required = true;
        swapchain_view_format = current_swapchain_view_format;
    }

    const bool storage_required = Settings::FrameGenActive();
    if (resource_update_required) {
        WaitIdle(device);
        SetWindowAdaptPass(device);

        if (presentation_recreate_required) {
            present_manager.RecreateFrame(frame, layout.width, layout.height, swapchain_view_format,
                                          window_adapt->GetRenderPass(), storage_required);
        }

        image_index = 0;
    } else if (present_manager.NeedsStorage(frame, storage_required)) {
        present_manager.RecreateFrame(frame, layout.width, layout.height, swapchain_view_format,
                                      window_adapt->GetRenderPass(), true);
    }

    const VkExtent2D window_size{
        .width = layout.screen.GetWidth(),
        .height = layout.screen.GetHeight(),
    };

    if (layers.size() != framebuffers.size()) {
        layers.clear();
        for (size_t i = 0; i < framebuffers.size(); ++i) {
            layers.emplace_back(device, memory_allocator, scheduler, device_memory, image_count,
                                window_size, window_adapt->GetDescriptorSetLayout(), filters);
        }
    }

    bool hudless_wanted = false;
#ifdef EGLANTINE_STREAMLINE
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_MKW_FG_HUDLESS");
        return !value || value[0] != '0';
    }();
    hudless_wanted = prepare_hudless && frame->hudless_available && !hudless_failed && enabled &&
        Settings::GetCurrentProgramID() == Settings::MarioKartWorldProgramId &&
        Streamline::IsFrameGenAvailable() &&
        Common::FrameGenStats::user_enabled.load(std::memory_order_relaxed);
#endif
    frame->hudless_valid = false;
    FramebufferTextureInfo source;
    window_adapt->Draw(device, rasterizer, scheduler, image_index, layers, framebuffers, layout, frame,
                       {}, hudless_wanted ? &source : nullptr);
    if (hudless_wanted) {
        bool effects = false;
#ifdef HAS_RESHADE
        effects = Settings::values.post_shader_enabled.GetValue() && VideoCore::FxChain::Instance().Size() != 0;
#endif
        const bool supported = !framebuffers.empty() && source.hudless_image && source.hudless_view &&
            VideoCommon::CanComposeMarioKartHudless(framebuffers.front(), source.hudless_scene_width,
                source.hudless_scene_height, framebuffers.size(), effects,
                filters.get_scaling_filter() == Settings::ScalingFilter::Fsr3);
        const char* reason = supported ? nullptr : "missing scene or unsupported layers/crop/effects";
        if (reason != hudless_reason) {
            LOG_INFO(Render_Vulkan, "Mario Kart World temporal: HUD-less compositor {}",
                     reason ? reason : "enabled");
            hudless_reason = reason;
        }
        if (supported) {
            try {
                if (!frame->hudless) {
                    frame->hudless = std::make_unique<Frame>();
                }
                if (!frame->hudless->image || frame->hudless->width != layout.width ||
                    frame->hudless->height != layout.height ||
                    frame->hudless_format != current_swapchain_view_format || presentation_recreate_required) {
                    present_manager.RecreateFrame(frame->hudless.get(), layout.width, layout.height,
                        current_swapchain_view_format, window_adapt->GetRenderPass(), false);
                    frame->hudless_format = current_swapchain_view_format;
                }
                if (hudless_layers.empty()) {
                    hudless_layers.emplace_back(device, memory_allocator, scheduler, device_memory,
                        image_count, window_size, window_adapt->GetDescriptorSetLayout(), filters);
                }
                source.image = source.hudless_image;
                source.image_view = source.hudless_view;
                scheduler.RequestOutsideRenderPassOperationContext();
                window_adapt->Draw(device, rasterizer, scheduler, image_index, hudless_layers,
                    framebuffers, layout, frame->hudless.get(), std::span{&source, 1});
                frame->hudless_valid = true;
            } catch (const std::exception& error) {
                hudless_failed = true;
                LOG_WARNING(Render_Vulkan, "Mario Kart World temporal: HUD-less compositor unavailable ({})", error.what());
            }
        }
    }

    if (++image_index >= image_count) {
        image_index = 0;
    }
}

vk::Framebuffer BlitScreen::CreateFramebuffer(const Device& device, const Layout::FramebufferLayout& layout,
                                              VkImageView image_view,
                                              VkFormat current_view_format) {
    bool format_updated = swapchain_view_format != current_view_format;
    swapchain_view_format = current_view_format;

    if (!window_adapt || scaling_filter != filters.get_scaling_filter() || format_updated) {
        WaitIdle(device);
        SetWindowAdaptPass(device);
        image_index = 0;
    }

    const VkExtent2D extent{
        .width = layout.width,
        .height = layout.height,
    };

    return CreateFramebuffer(device, image_view, extent, window_adapt->GetRenderPass());
}

vk::Framebuffer BlitScreen::CreateFramebuffer(const Device& device, const VkImageView& image_view, VkExtent2D extent,
                                              VkRenderPass render_pass) {
    return device.GetLogical().CreateFramebuffer(VkFramebufferCreateInfo{
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .renderPass = render_pass,
        .attachmentCount = 1,
        .pAttachments = &image_view,
        .width = extent.width,
        .height = extent.height,
        .layers = 1,
    });
}

} // namespace Vulkan
