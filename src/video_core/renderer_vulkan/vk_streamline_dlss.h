// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS (through NVIDIA Streamline) in place of a game's DLSS
// evaluation. NVIDIA's DLLs come from the user's PC; see vulkan_common/streamline.h.

#pragma once

#include <array>
#include <atomic>
#include <optional>

#include "common/common_types.h"
#include "video_core/temporal_camera.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class MemoryAllocator;
class Scheduler;

class StreamlineDlssUpscaler {
public:
    struct Image {
        VkImageView view;
        VkImage image;
        VkFormat format;
        VkImageSubresourceRange range;     ///< The view's level and layer, with the view's aspect.
        VkImageAspectFlags barrier_aspect; ///< Every aspect of the image's format.
    };

    struct Inputs {
        Image color;          ///< Jittered colour at the render size.
        Image depth;          ///< Device depth (reversed) at the render size.
        Image motion_vectors; ///< Motion to the previous frame at the render size.
        VkExtent2D render_size;
        VkExtent2D upscale_size;
        /// This frame's sample offset in render pixels, as the game hands its own DLSS.
        std::array<f32, 2> jitter;
        /// Turns a motion vector into render pixels (the render size for UV offsets).
        std::array<f32, 2> motion_vector_scale;
        bool reset;
        bool depth_inverted = true;
        bool hdr = true; ///< Linear HDR colour. False for tonemapped, perceptually encoded colour.
        std::optional<VideoCommon::TemporalCameraConstants> camera;
    };

    /// Throws vk::Exception when Streamline has no DLSS for this device.
    explicit StreamlineDlssUpscaler(const Device& device, Scheduler& scheduler,
                                    MemoryAllocator& memory_allocator);
    ~StreamlineDlssUpscaler();

    /// Records one evaluation into the output image (RGBA16F at the upscale size, GENERAL
    /// layout). Returns false when DLSS can't run, so the caller can fall back to FSR 3.1.
    bool Upscale(const Inputs& inputs);

    [[nodiscard]] bool Failed() const noexcept {
        return failed.load();
    }

    [[nodiscard]] VkImage OutputImage() const noexcept {
        return *output_image;
    }
    [[nodiscard]] VkImageView OutputView() const noexcept {
        return *output_view;
    }
    /// Nearest-neighbour clamp sampler (normalized coordinates), for reading the output.
    [[nodiscard]] VkSampler PointSampler() const noexcept {
        return *point_sampler;
    }

private:
    /// DLSS Frame Generation reads a frame's depth and motion at its (paced) present, after
    /// the game has drawn over them: they are copied into a ring and tagged valid until present.
    /// Streamline's own copies (eOnlyValidNow) held the Vulkan worker until frame generation had
    /// consumed the previous frame, which halved the real frame rate.
    struct FrameGenCopy {
        vk::Image depth;
        vk::ImageView depth_view;
        vk::Image motion;
        vk::ImageView motion_view;
    };
    static constexpr size_t FRAME_GEN_COPIES = 4;

    void CreateOutput(VkExtent2D upscale_size);
    void CreateFrameGenCopies(const Inputs& inputs);

    const Device& device;
    Scheduler& scheduler;
    MemoryAllocator& memory_allocator;
    void* set_options{}; ///< slDLSSSetOptions
    VkExtent2D render_extent{};
    VkExtent2D upscale_extent{};
    bool hdr_input = true;
    u32 frame_index{};
    vk::Image output_image;
    vk::ImageView output_view;
    vk::Sampler point_sampler;
    std::array<FrameGenCopy, FRAME_GEN_COPIES> fg_copies;
    VkExtent2D fg_copy_extent{};
    VkFormat fg_depth_format{};
    VkFormat fg_motion_format{};
    u32 fg_copy_next{};
    std::atomic<bool> failed{};
};

} // namespace Vulkan
