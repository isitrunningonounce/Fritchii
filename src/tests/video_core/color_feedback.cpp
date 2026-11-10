// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include "common/settings.h"
#include "video_core/renderer_vulkan/color_feedback.h"
#include "video_core/texture_cache/color_feedback.h"

namespace {

VkImageView View(uintptr_t value) {
    return reinterpret_cast<VkImageView>(value);
}

Vulkan::DescriptorUpdateEntry Sampled(uintptr_t value) {
    return VkDescriptorImageInfo{
        .sampler = VK_NULL_HANDLE,
        .imageView = View(value),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
}

} // namespace

TEST_CASE("Color feedback selects active color images, not null or depth", "[video_core]") {
    using namespace VideoCommon;
    const std::array<ImageId, NUM_RT> targets{ImageId{4}, ImageId{5}, NULL_IMAGE_ID, ImageId{},
                                              ImageId{},  ImageId{},  ImageId{},     ImageId{9}};
    const u32 active = 1U | (1U << 7) | (1U << NUM_RT);
    REQUIRE(IsColorRenderTarget(ImageId{4}, active, targets));
    REQUIRE(IsColorRenderTarget(ImageId{9}, active, targets));
    REQUIRE_FALSE(IsColorRenderTarget(ImageId{5}, active, targets));
    REQUIRE_FALSE(IsColorRenderTarget(ImageId{10}, active, targets));
    REQUIRE_FALSE(IsColorRenderTarget(NULL_IMAGE_ID, ~0U, targets));
    REQUIRE_FALSE(IsColorRenderTarget(ImageId{}, ~0U, targets));
    REQUIRE_FALSE(IsColorRenderTarget(ImageId{4}, 0, targets));
}

TEST_CASE("Color feedback copies only supported native color sampling", "[video_core]") {
    using namespace VideoCommon;
    REQUIRE(CanCopyColorFeedback(true, true, true, ImageViewType::e2D, 1, {1, 1}));
    REQUIRE_FALSE(CanCopyColorFeedback(false, true, true, ImageViewType::e2D, 1, {1, 1}));
    REQUIRE_FALSE(CanCopyColorFeedback(true, false, true, ImageViewType::e2D, 1, {1, 1}));
    REQUIRE_FALSE(CanCopyColorFeedback(true, true, false, ImageViewType::e2D, 1, {1, 1}));
    REQUIRE_FALSE(CanCopyColorFeedback(true, true, true, ImageViewType::e3D, 1, {1, 1}));
    REQUIRE_FALSE(CanCopyColorFeedback(true, true, true, ImageViewType::e2D, 4, {1, 1}));
    REQUIRE_FALSE(CanCopyColorFeedback(true, true, true, ImageViewType::e2D, 1, {2, 1}));
    REQUIRE_FALSE(CanCopyColorFeedback(true, true, true, ImageViewType::e2D, 1, {1, 2}));
}

TEST_CASE("Color feedback copies the host mip extent at each resolution scale", "[video_core]") {
    using VideoCommon::ColorFeedbackExtent;
    REQUIRE(ColorFeedbackExtent({640, 360, 1}, 0) == VideoCommon::Extent3D{640, 360, 1});
    Settings::ResolutionScalingInfo resolution;
    resolution.up_scale = 2;
    REQUIRE(ColorFeedbackExtent({resolution.ScaleUp(640U), resolution.ScaleUp(360U), 1}, 0) ==
            VideoCommon::Extent3D{1280, 720, 1});
    // Scale the backing image before selecting its mip, including odd dimensions.
    REQUIRE(ColorFeedbackExtent({resolution.ScaleUp(641U), resolution.ScaleUp(361U), 1}, 1) ==
            VideoCommon::Extent3D{641, 361, 1});
    resolution.up_scale = 1;
    resolution.down_shift = 1;
    REQUIRE(ColorFeedbackExtent({resolution.ScaleUp(641U), resolution.ScaleUp(361U), 1}, 1) ==
            VideoCommon::Extent3D{160, 90, 1});
    REQUIRE(ColorFeedbackExtent({1, 1, 1}, 4) == VideoCommon::Extent3D{1, 1, 1});
}

TEST_CASE("Color feedback replaces current payload on memo misses and hits", "[video_core]") {
    using namespace VideoCommon;
    const std::array textures{
        Shader::TextureDescriptor{.type = Shader::TextureType::Color2D, .count = 2},
        Shader::TextureDescriptor{
            .type = Shader::TextureType::Color2D, .is_depth = true, .count = 1},
        Shader::TextureDescriptor{.type = Shader::TextureType::Color3D, .count = 1},
        Shader::TextureDescriptor{
            .type = Shader::TextureType::Color2D, .is_multisample = true, .count = 1}};
    const std::array views{
        ImageViewInOut{.id = ImageViewId{4}}, ImageViewInOut{.id = ImageViewId{5}},
        ImageViewInOut{.id = ImageViewId{6}}, ImageViewInOut{.id = ImageViewId{7}},
        ImageViewInOut{.id = ImageViewId{8}}};
    const std::array memo{Sampled(4), Sampled(5), Sampled(6), Sampled(7), Sampled(8)};
    unsigned copies{};
    const auto copy = [&](ImageViewId id) -> VkImageView {
        REQUIRE((id == ImageViewId{4} || id == ImageViewId{5}));
        if (id == ImageViewId{5}) {
            return VK_NULL_HANDLE;
        }
        return View(100 + ++copies);
    };
    auto miss = memo; // Original descriptors captured before the per-draw replacements.
    Vulkan::ApplyColorFeedbackCopies(textures, views, miss, copy);
    REQUIRE(copies == 1);
    REQUIRE(miss[0].image.imageView == View(101));
    REQUIRE(memo[0].image.imageView == View(4));
    auto hit = memo;
    Vulkan::ApplyColorFeedbackCopies(textures, views, hit, copy);
    REQUIRE(copies == 2);
    REQUIRE(hit[0].image.imageView == View(102));
    REQUIRE(miss[0].image.imageView == View(101));
    REQUIRE(memo[0].image.imageView == View(4));
    for (size_t index = 1; index < memo.size(); ++index) {
        REQUIRE(hit[index].image.imageView == memo[index].image.imageView);
    }
    REQUIRE(hit[0].image.sampler == memo[0].image.sampler);
    REQUIRE(hit[0].image.imageLayout == memo[0].image.imageLayout);
}
