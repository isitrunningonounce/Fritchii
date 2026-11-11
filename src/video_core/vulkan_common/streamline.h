// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS through NVIDIA Streamline. Only Streamline's MIT-licensed
// headers are built in: NVIDIA's DLLs are loaded at run time from the user's PC, and none ship
// with the emulator.

#pragma once

#include <atomic>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan::Streamline {

/// NVIDIA's files DLSS needs, as found on this PC. Paths are empty when missing.
struct DlssFiles {
    std::filesystem::path folder;     ///< Where the loader is, or where it should go
    std::filesystem::path interposer; ///< sl.interposer.dll (Streamline's loader)
    std::filesystem::path model;      ///< nvngx_dlss.dll next to the loader
    std::filesystem::path app_model;  ///< Newest DLSS the NVIDIA app downloaded
    std::string interposer_version;
    std::string model_version;
    std::string app_model_version;
};

[[nodiscard]] DlssFiles FindDlssFiles();

/// The user folder's "streamline" folder.
[[nodiscard]] std::filesystem::path UserFolder();

/// Streamline's own modules carry a second, Streamline-only signature.
[[nodiscard]] bool IsStreamlineModule(const std::filesystem::path& path);

/// A valid Authenticode signature from NVIDIA Corporation (DLSS models).
[[nodiscard]] bool IsSignedByNvidia(const std::filesystem::path& path);

/// When the NVIDIA DLSS setting is on, loads the user's sl.interposer.dll (checking NVIDIA's
/// signature) and starts Streamline for DLSS. Call before creating the renderer's instance.
bool Start();

/// Whether Streamline is running and waiting for the renderer's device.
[[nodiscard]] bool WantsDevice();

/// Instance and device extensions DLSS asks for (empty when Streamline isn't running).
[[nodiscard]] std::vector<std::string> InstanceExtensions();
[[nodiscard]] std::vector<std::string> DeviceExtensions();

/// LICHII_DLSS_FG=1: DLSS Frame Generation is wanted (with Reflex and PCL, which it needs).
/// LICHII_DLSS_FG_MULTIPLIER=2..6 picks its multiplier, within what the GPU allows.
[[nodiscard]] bool FrameGenRequested();
[[nodiscard]] bool BananzaFrameGenOwnCopies();

/// The queues DLSS Frame Generation asks the device for, beyond the renderer's own: compute and
/// graphics queues in the graphics family (after its first queue), optical flow queues in the
/// optical flow family. All zero when frame generation is off or unavailable.
struct ExtraQueues {
    u32 compute{};
    u32 graphics{};
    u32 optical_flow{};
};
[[nodiscard]] ExtraQueues ExtraQueuesWanted();

/// Device extensions frame generation adds (enabled only when the device has them all; else
/// frame generation is dropped and DLSS keeps running).
[[nodiscard]] std::vector<std::string> FrameGenDeviceExtensions();
/// The device lacks some of them: frame generation is off for this session.
void DropFrameGen(const char* reason);

/// Hands the renderer's device to Streamline and routes the swapchain calls through it, as
/// Streamline's manual hooking needs. Stops Streamline and returns false when DLSS can't run.
/// optical_flow_family is the family the optical flow queues were created in (frame generation).
bool AttachDevice(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
                  u32 graphics_family, u32 optical_flow_family, vk::DeviceDispatch& dld);

/// A device queue fetched through Streamline's vkGetDeviceQueue, so DLSS-G accepts presents on
/// it ("Invalid VK app queue - not created by the application" otherwise). Null without it.
[[nodiscard]] VkQueue FrameGenDeviceQueue(VkDevice device, u32 family, u32 index);

/// Whether DLSS Frame Generation can run on the attached device.
[[nodiscard]] bool IsFrameGenAvailable();

/// With frame generation wanted: Streamline's proxy for a Vulkan function the renderer must call
/// through it (surface creation, before the device exists). Null otherwise.
[[nodiscard]] void* FrameGenProcAddress(const char* name);

struct FrameGenInputFence {
    VkSemaphore semaphore{};
    u64 value{};
    bool pending{};
    bool timed_out{};

    template <typename Wait>
    bool Retire(Wait&& wait) {
        if (!pending) {
            return true;
        }
        if (!semaphore || !wait(semaphore, value)) {
            return false;
        }
        *this = {};
        return true;
    }
};

class FrameGenCopyFence {
public:
    [[nodiscard]] bool Ready();
    template <typename Wait>
    bool Ready(Wait&& wait) {
        return completed.load(std::memory_order_acquire) && fence.Retire(wait);
    }

    void Reserve() noexcept {
        completed.store(false, std::memory_order_relaxed);
    }
    void Release(FrameGenInputFence result = {}, u64 swapchain_generation = 0) noexcept {
        fence = result;
        generation = swapchain_generation;
        completed.store(true, std::memory_order_release);
    }

private:
    std::atomic<bool> completed{true};
    FrameGenInputFence fence;
    u64 generation{};
};

/// Keep SDK-owned completion semaphores from being polled while their swapchain is replaced.
[[nodiscard]] std::unique_lock<std::mutex> LockFrameGenCopyFencesForSwapchainReset();

struct FrameGenSubmission {
    void* token{};
    FrameGenCopyFence* copy_fence{};
};

struct FrameGenUiOptionsState {
    std::optional<bool> requested;
    bool active{};

    bool NeedsUpdate(bool desired) const {
        return requested != desired;
    }
    static bool Desired(bool enabled, bool hudless_tagged, bool alpha_tagged) {
        return enabled && hudless_tagged && alpha_tagged;
    }
    void Applied(bool desired, bool actual) {
        requested = desired;
        active = actual;
    }
};

struct FrameGenHudlessImage {
    VkImage image{};
    VkImageView view{};
    VkFormat format{};
    VkExtent2D extent{};
};

bool WaitFrameGenInputs(FrameGenInputFence& fence);

/// Frame generation per presented frame. The present manager takes the token on the Vulkan worker
/// right after the frame's commands (so it is the token of the DLSS evaluation in that frame, or
/// null without one), then wraps the frame's vkQueuePresentKHR with it.
[[nodiscard]] FrameGenSubmission TakeFrameGenSubmission();
/// game_* is the rectangle the game image covers in the backbuffer: frame generation runs inside it
/// (a little inset) and copies the rest as is (else, scaled to fullscreen, it smeared the edge row).
void FrameGenBeforePresent(void* token, u32 game_left, u32 game_top, u32 game_right,
                           u32 game_bottom, VkExtent2D frame_extent, VkExtent2D swapchain_extent,
                           const FrameGenHudlessImage* hudless = nullptr,
                           const FrameGenHudlessImage* ui_alpha = nullptr);
void FrameGenAfterPresent(void* token, FrameGenInputFence* inputs = nullptr,
                          FrameGenCopyFence* copy_fence = nullptr);
/// Time the present thread spent acquiring the swapchain image (for the frame generation log).
void NoteFrameGenAcquire(u64 ns);
/// Time the renderer waited for a free present frame (for the frame generation log).
void NoteFrameGenFreeWait(u64 ns);
/// With frame generation available: present without VkPresentIdKHR and without the host pacer's
/// present waits (LICHII_DLSS_FG_PRESENT_ID=1 keeps them).
[[nodiscard]] bool FrameGenDropsPresentIds();
/// LICHII_DLSS_FG_IMAGES=n with frame generation available: at least n swapchain images.
[[nodiscard]] u32 FrameGenSwapchainImages();

/// Whether DLSS can run on the attached device.
[[nodiscard]] bool IsDlssAvailable();

/// Shuts Streamline down; call before the device is destroyed.
void Stop();

} // namespace Vulkan::Streamline
