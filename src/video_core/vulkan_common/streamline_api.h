// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Streamline's API as loaded from the user's sl.interposer.dll. Only for the
// Streamline sources: it pulls in <windows.h> and Streamline's (MIT-licensed) headers.

#pragma once

#include <windows.h>

#include "video_core/vulkan_common/vulkan_wrapper.h"

#include <sl.h>
#include <sl_dlss.h>
#include <sl_dlss_g.h>
#include <sl_helpers_vk.h>
#include <sl_pcl.h>
#include <sl_reflex.h>

namespace Vulkan::Streamline {

class FrameGenCopyFence;

struct Api {
    PFun_slInit* init{};
    PFun_slShutdown* shutdown{};
    PFun_slIsFeatureSupported* is_feature_supported{};
    PFun_slGetFeatureRequirements* get_feature_requirements{};
    PFun_slGetFeatureVersion* get_feature_version{};
    PFun_slGetFeatureFunction* get_feature_function{};
    PFun_slGetNewFrameToken* get_new_frame_token{};
    PFun_slSetConstants* set_constants{};
    PFun_slEvaluateFeature* evaluate_feature{};
    PFun_slFreeResources* free_resources{};
    PFun_slSetVulkanInfo* set_vulkan_info{};
    PFun_slSetTagForFrame* set_tag_for_frame{};
    // DLSS Frame Generation (LICHII_DLSS_FG), from slGetFeatureFunction once attached.
    PFun_slDLSSGSetOptions* dlssg_set_options{};
    PFun_slDLSSGGetState* dlssg_get_state{};
    PFun_slReflexSetOptions* reflex_set_options{};
    PFun_slReflexSleep* reflex_sleep{};
    PFun_slPCLSetMarker* pcl_set_marker{};
};

/// The loaded functions; only valid while IsDlssAvailable().
[[nodiscard]] const Api& GetApi();

/// DLSS Frame Generation, on the Vulkan worker inside a DLSS evaluation's command buffer (after
/// slSetConstants on the same token): tags the frame's depth and motion and hands the token to
/// the next present. eOnlyValidNow has Streamline copy them on cmdbuf; eValidUntilPresent is for
/// the renderer's own copies.
void TagFrameGen(const sl::FrameToken& frame, const sl::Resource& depth, const sl::Resource& motion,
                 const sl::Extent& area, VkCommandBuffer cmdbuf, sl::ResourceLifecycle lifecycle,
                 FrameGenCopyFence* copy_fence = nullptr);

} // namespace Vulkan::Streamline
