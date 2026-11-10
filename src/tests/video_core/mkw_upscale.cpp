// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <bit>
#include <limits>

#include <catch2/catch_test_macros.hpp>
#include "video_core/mkw_upscale.h"
#include "video_core/mkw_hudless.h"
#include "video_core/renderer_vulkan/present/present_push_constants.h"
#include "video_core/vulkan_common/streamline.h"

namespace {

// Fragment cbuf 4, frames 2910 and 2911 of the captured single-player race.
constexpr std::array<u32, 268> Camera2910Words{{
    0xbf0448ebU, 0x00000000U, 0x3f5b2c3eU, 0xc562e579U, 0x3dc77c25U, 0x3f7e5678U, 0x3d70cdcbU, 0xc1a59a76U,
    0xbf59bfedU, 0x3de90106U, 0xbf036d08U, 0xc5382ac3U, 0xbee999eeU, 0x00000000U, 0x3f4184aaU, 0xc5485675U,
    0x3e9cac0bU, 0x3fc4fe6dU, 0x3e3d1f8dU, 0x43f8f437U, 0x3f59cb14U, 0xbde90cf5U, 0x3f0373c3U, 0x45381430U,
    0x3f59bfedU, 0xbde90106U, 0x3f036d08U, 0x45382ac3U, 0x3f6208e3U, 0x00000000U, 0x00000000U, 0x00000000U,
    0x00000000U, 0x3fc8eb74U, 0xbe3851ecU, 0x00000000U, 0x00000000U, 0x00000000U, 0xbf80068eU, 0xc0000347U,
    0x00000000U, 0x00000000U, 0xbf800000U, 0x00000000U, 0xbf0448ebU, 0x3dc77c25U, 0xbf59bfedU, 0xc588e281U,
    0x2f544500U, 0x3f7e5678U, 0x3de90107U, 0x43b1e81cU, 0x3f5b2c3dU, 0x3d70cdcbU, 0xbf036d08U, 0x44c79157U,
    0x3f800000U, 0x461c4000U, 0x38d1b717U, 0x3f7ff972U, 0x38d1bc76U, 0x38d1bc76U, 0x3fe38e39U, 0x3f100000U,
    0x461c3c00U, 0x3f800347U, 0x00000000U, 0x3f6208e3U, 0x3f90f809U, 0x3f23170aU, 0x3f91361eU, 0x3fc8eb74U,
    0x45200000U, 0x44b40000U, 0x39cccccdU, 0x3a360b61U, 0x44a00000U, 0x44340000U, 0x3a4ccccdU, 0x3ab60b61U,
    0x3c088889U, 0x3c088889U, 0x3c088889U, 0x3c088889U, 0xbf0393e5U, 0x00000000U, 0x3f5b991bU, 0xc5624e39U,
    0x3dc7e888U, 0x3f7e5652U, 0x3d6f8f6bU, 0xc19a25fbU, 0xbf5a2bf3U, 0x3de90bdfU, 0xbf02b91aU, 0xc538ea62U,
    0xbee85a43U, 0x00000000U, 0x3f41e4c9U, 0xc547d0e9U, 0x3e9cfd77U, 0x3fc4fe31U, 0x3e3c2110U, 0x43fb27d8U,
    0x3f5a371fU, 0xbde917ceU, 0x3f02bfccU, 0x4538d3d9U, 0x3f5a2bf3U, 0xbde90bdfU, 0x3f02b91aU, 0x4538ea62U,
    0x3f6208e3U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x3fc8eb74U, 0xbe3851ecU, 0x00000000U,
    0x00000000U, 0x00000000U, 0xbf80068eU, 0xc0000347U, 0x00000000U, 0x00000000U, 0xbf800000U, 0x00000000U,
    0xbf0393e4U, 0x3dc7e886U, 0xbf5a2bf3U, 0xc588e501U, 0xb1981a2fU, 0x3f7e5650U, 0x3de90bdeU, 0x43b1e840U,
    0x3f5b991aU, 0x3d6f8f69U, 0xbf02b91aU, 0x44c78b20U, 0x3f800000U, 0x00000000U, 0x00000000U, 0x00000000U,
    0x00000000U, 0x3f800000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x3f800000U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0x3f800000U, 0x00000001U, 0x00000000U, 0x00000001U, 0x00000000U,
    0x81da48f0U, 0x00000000U, 0x8208bea0U, 0x00000000U, 0x3f800000U, 0x3f800000U, 0x00000000U, 0x00000000U,
    0x3f3acaf7U, 0x3e07dce3U, 0xbe40239dU, 0x45579f60U, 0x3faa9f2bU, 0xbee09cbaU, 0x3f1ed328U, 0x459c5a40U,
    0x3eee631fU, 0xbf30699dU, 0xbe261cd7U, 0x451f1ecaU, 0x3f5e6763U, 0xbe9263d5U, 0x3ecf06d5U, 0x454c0638U,
    0x3f3b1d31U, 0x3e094dc1U, 0xbe422d45U, 0x4558957cU, 0x3fabe9c4U, 0xbee24ff0U, 0x3f2006e6U, 0x459d5986U,
    0x3eed0532U, 0xbf30fe7aU, 0xbe29aa7dU, 0x451f7c60U, 0x3f5e6763U, 0xbe9263d5U, 0x3ecf06d5U, 0x454cdf0cU,
    0x3f38e8aeU, 0x3dfef3caU, 0xbe34474bU, 0x455801a8U, 0x3fa35d4eU, 0xbed70efaU, 0x3f1811c0U, 0x45947fb4U,
    0x3ef5e316U, 0xbf2cf62eU, 0xbe11e14bU, 0x4525fdf6U, 0x3f5e6763U, 0xbe9263d5U, 0x3ecf06d5U, 0x45525f15U,
    0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x3f800000U,
    0x40c00000U, 0x420c0000U, 0x437a0000U, 0x7f7fffffU, 0x39aaaaabU, 0x39aaaaabU, 0x00000000U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x3f800000U,
    0x44800000U, 0x40800000U, 0x3f800000U, 0x3f800000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U,
}};
constexpr auto Camera2910 = std::bit_cast<std::array<f32, 268>>(Camera2910Words);

constexpr std::array<u32, 268> Camera2911Words{{
    0xbf04ea7fU, 0x00000000U, 0x3f5aca66U, 0xc5636c18U, 0x3dc71724U, 0x3f7e56acU, 0x3d71e567U, 0xc1b01971U,
    0xbf595ee5U, 0x3de8f30bU, 0xbf040dabU, 0xc5377e58U, 0xbeeab743U, 0x00000000U, 0x3f412e46U, 0xc548cd52U,
    0x3e9c617bU, 0x3fc4febeU, 0x3e3e00efU, 0x43f6f454U, 0x3f596a07U, 0xbde8fef9U, 0x3f04146eU, 0x453767bcU,
    0x3f595ee5U, 0xbde8f30bU, 0x3f040dabU, 0x45377e58U, 0x3f6208e3U, 0x00000000U, 0x00000000U, 0x00000000U,
    0x00000000U, 0x3fc8eb74U, 0xbe3851ecU, 0x00000000U, 0x00000000U, 0x00000000U, 0xbf80068eU, 0xc0000347U,
    0x00000000U, 0x00000000U, 0xbf800000U, 0x00000000U, 0xbf04ea7fU, 0x3dc71722U, 0xbf595ee2U, 0xc588dff6U,
    0x30e5242eU, 0x3f7e56aaU, 0x3de8f308U, 0x43b1e800U, 0x3f5aca65U, 0x3d71e565U, 0xbf040da9U, 0x44c7976cU,
    0x3f800000U, 0x461c4000U, 0x38d1b717U, 0x3f7ff972U, 0x38d1bc76U, 0x38d1bc76U, 0x3fe38e39U, 0x3f100000U,
    0x461c3c00U, 0x3f800347U, 0x00000000U, 0x3f6208e3U, 0x3f90f809U, 0x3f23170aU, 0x3f91361eU, 0x3fc8eb74U,
    0x45200000U, 0x44b40000U, 0x39cccccdU, 0x3a360b61U, 0x44a00000U, 0x44340000U, 0x3a4ccccdU, 0x3ab60b61U,
    0x3c888889U, 0x3c888889U, 0x3c888889U, 0x3c888889U, 0xbf0448ebU, 0x00000000U, 0x3f5b2c3eU, 0xc562e579U,
    0x3dc77c25U, 0x3f7e5678U, 0x3d70cdcbU, 0xc1a59a76U, 0xbf59bfedU, 0x3de90106U, 0xbf036d08U, 0xc5382ac3U,
    0xbee999eeU, 0x00000000U, 0x3f4184aaU, 0xc5485675U, 0x3e9cac0bU, 0x3fc4fe6dU, 0x3e3d1f8dU, 0x43f8f437U,
    0x3f59cb14U, 0xbde90cf5U, 0x3f0373c3U, 0x45381430U, 0x3f59bfedU, 0xbde90106U, 0x3f036d08U, 0x45382ac3U,
    0x3f6208e3U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x3fc8eb74U, 0xbe3851ecU, 0x00000000U,
    0x00000000U, 0x00000000U, 0xbf80068eU, 0xc0000347U, 0x00000000U, 0x00000000U, 0xbf800000U, 0x00000000U,
    0xbf0448ebU, 0x3dc77c25U, 0xbf59bfedU, 0xc588e281U, 0x2f544500U, 0x3f7e5678U, 0x3de90107U, 0x43b1e81cU,
    0x3f5b2c3dU, 0x3d70cdcbU, 0xbf036d08U, 0x44c79157U, 0x3f800000U, 0x00000000U, 0x00000000U, 0x00000000U,
    0x00000000U, 0x3f800000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x3f800000U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0x3f800000U, 0x00000003U, 0x00000000U, 0x00000002U, 0x00000000U,
    0xab9d0b30U, 0x00000010U, 0x804878d8U, 0x00000000U, 0x3f800000U, 0x3f800000U, 0x00000000U, 0x00000000U,
    0x3f3af013U, 0x3e06d750U, 0xbe3eb1b0U, 0x4557a58eU, 0x3faa4c1cU, 0xbee17ef1U, 0x3f1f731cU, 0x459be2dbU,
    0x3eee85f8U, 0xbf30bc74U, 0xbe24e400U, 0x451f1852U, 0x3f5e15afU, 0xbe9308daU, 0x3ecff034U, 0x454b82eaU,
    0x3f3b4258U, 0x3e084eb0U, 0xbe40c48cU, 0x45589c90U, 0x3fab98c5U, 0xbee3376dU, 0x3f20aa95U, 0x459ce3d0U,
    0x3eed24f6U, 0xbf315470U, 0xbe2881a8U, 0x451f7690U, 0x3f5e15afU, 0xbe9308daU, 0x3ecff034U, 0x454c5bfaU,
    0x3f39097aU, 0x3dfcf414U, 0xbe32dd76U, 0x45580481U, 0x3fa31287U, 0xbed7eddaU, 0x3f18af58U, 0x4594127fU,
    0x3ef60732U, 0xbf2d4962U, 0xbe109802U, 0x4525f97aU, 0x3f5e15afU, 0xbe9308daU, 0x3ecff034U, 0x4551dd4cU,
    0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x3f800000U,
    0x40c00000U, 0x420c0000U, 0x437a0000U, 0x7f7fffffU, 0x39aaaaabU, 0x39aaaaabU, 0x00000000U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x3f800000U,
    0x44800000U, 0x40800000U, 0x3f800000U, 0x3f800000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U,
}};
constexpr auto Camera2911 = std::bit_cast<std::array<f32, 268>>(Camera2911Words);

using VideoCore::Surface::PixelFormat;
using VideoCommon::MarioKartBlendImage;
constexpr std::array Images{
    MarioKartBlendImage{1, PixelFormat::B10G11R11_FLOAT, 2560, 1440},
    MarioKartBlendImage{2, PixelFormat::A8B8G8R8_UNORM, 2560, 1440},
};
constexpr MarioKartBlendImage Target{3, PixelFormat::B10G11R11_FLOAT, 2560, 1440};

} // namespace

TEST_CASE("MKW temporal backend is independent of Ocarina anti-aliasing", "[video_core][mkw]") {
    using enum VideoCommon::OcarinaUpscaleBackend;
    using VideoCommon::SelectMarioKartUpscaler;
    CHECK(SelectMarioKartUpscaler(false, true, "") == Native);
    CHECK(SelectMarioKartUpscaler(true, false, "") == Native);
    CHECK(SelectMarioKartUpscaler(true, true, "") == Dlss);
    CHECK(SelectMarioKartUpscaler(true, true, "native") == Native);
    CHECK(SelectMarioKartUpscaler(false, false, "fsr") == Fsr);
    CHECK(SelectMarioKartUpscaler(true, true, "fsr") == Fsr);
    CHECK(SelectMarioKartUpscaler(false, true, "dlss") == Dlss);
    CHECK(SelectMarioKartUpscaler(false, false, "dlss") == Native);
    CHECK(SelectMarioKartUpscaler(false, true, "unknown") == Native);
}

TEST_CASE("MKW blend recognition requires the captured image structure", "[video_core][mkw]") {
    using VideoCommon::IsMarioKartBlend;
    constexpr u64 hash = VideoCommon::MarioKartBlendFragment;
    CHECK(IsMarioKartBlend(hash, false, false, 3, 1, Images, Target, 2560, 1440));
    auto reversed = Images;
    std::swap(reversed[0], reversed[1]);
    CHECK(IsMarioKartBlend(hash, false, false, 3, 1, reversed, Target, 2560, 1440));
    CHECK_FALSE(IsMarioKartBlend(hash + 1, false, false, 3, 1, Images, Target, 2560, 1440));
    CHECK_FALSE(IsMarioKartBlend(hash, true, false, 3, 1, Images, Target, 2560, 1440));
    CHECK_FALSE(IsMarioKartBlend(hash, false, true, 3, 1, Images, Target, 2560, 1440));
    CHECK_FALSE(IsMarioKartBlend(hash, false, false, 6, 1, Images, Target, 2560, 1440));
    CHECK_FALSE(IsMarioKartBlend(hash, false, false, 3, 1,
                                 std::span{Images}.first(1), Target, 2560, 1440));
    CHECK_FALSE(IsMarioKartBlend(hash, false, false, 3, 1, Images, Target, 1280, 720));
    auto images = Images;
    SECTION("Wrong sampled format") { images[1].format = PixelFormat::R8G8_UNORM; }
    SECTION("Half-sized sampled image") { images[1].width = 1280; }
    SECTION("Feedback into scene input") { images[0].identity = Target.identity; }
    SECTION("Feedback into weights") { images[1].identity = Target.identity; }
    SECTION("Missing sampled image") { images[0].identity = 0; }
    SECTION("Aliased inputs") { images[1].identity = images[0].identity; }
    CHECK_FALSE(IsMarioKartBlend(hash, false, false, 3, 1, images, Target, 2560, 1440));
}

TEST_CASE("MKW blend uses the effective direct-draw instance count", "[video_core][mkw]") {
    // Draw's argument is one even when the captured draw-state counter is zero.
    using VideoCommon::IsMarioKartBlend;
    const auto accepts = [](u32 effective_instances) {
        return IsMarioKartBlend(VideoCommon::MarioKartBlendFragment, false, false, 3,
                                effective_instances, Images, Target, 2560, 1440);
    };
    CHECK(accepts(1));
    CHECK_FALSE(accepts(0));
    CHECK_FALSE(accepts(2));
}

TEST_CASE("MKW captured camera blocks preserve previous-frame matrices", "[video_core][mkw]") {
    const auto first = VideoCommon::ReadMarioKartCamera(Camera2910, 2560, 1440, false);
    const auto second = VideoCommon::ReadMarioKartCamera(Camera2911, 2560, 1440, false);
    REQUIRE(first);
    REQUIRE(second);
    CHECK(std::bit_cast<std::array<u32, 16>>(first->current) ==
          std::bit_cast<std::array<u32, 16>>(second->previous));
    CHECK(first->current != second->current);
    CHECK(Camera2910[0xa8 / 4] == -1.0f);
    CHECK(Camera2910[0xac / 4] == 0.0f);
}

TEST_CASE("MKW camera rejects other views and unavailable camera data", "[video_core][mkw]") {
    using VideoCommon::ReadMarioKartCamera;
    CHECK_FALSE(ReadMarioKartCamera(Camera2910, 1280, 720, false));
    CHECK_FALSE(ReadMarioKartCamera(Camera2910, 2560, 1440, true));
    CHECK_FALSE(ReadMarioKartCamera(std::span{Camera2910}.first(0x1bc / 4), 2560, 1440, false));
    CHECK(ReadMarioKartCamera(std::span{Camera2910}.first(0x1c0 / 4), 2560, 1440, false));
    auto block = Camera2910;
    SECTION("Non-perspective projection") { block[0xa8 / 4] = 1.0f; }
    SECTION("Invalid projection w") { block[0xac / 4] = 1.0f; }
    SECTION("Mismatched width") { block[0x120 / 4] = 1920.0f; }
    SECTION("Mismatched height") { block[0x124 / 4] = 1080.0f; }
    SECTION("Nonfinite current matrix") { block[0x30 / 4] = std::numeric_limits<f32>::quiet_NaN(); }
    SECTION("Nonfinite previous matrix") { block[0x180 / 4] = std::numeric_limits<f32>::infinity(); }
    CHECK_FALSE(ReadMarioKartCamera(block, 2560, 1440, false));
}

namespace {

std::array<f64, 4> RowTransform(const std::array<f64, 4>& vector,
                               const std::array<f32, 16>& matrix) {
    std::array<f64, 4> result{};
    for (size_t col = 0; col < 4; ++col) {
        for (size_t row = 0; row < 4; ++row) {
            result[col] += vector[row] * matrix[row * 4 + col];
        }
    }
    return result;
}

std::array<f64, 4> ColumnTransform(const std::array<f32, 16>& matrix,
                                  const std::array<f64, 4>& vector) {
    std::array<f64, 4> result{};
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 4; ++col) {
            result[row] += matrix[row * 4 + col] * vector[col];
        }
    }
    return result;
}

} // namespace

TEST_CASE("MKW FG camera uses captured pose and shifted perspective", "[video_core][mkw]") {
    for (const auto& block : {Camera2910, Camera2911}) {
        const auto camera = VideoCommon::ReadMarioKartCamera(block, 2560, 1440, false);
        REQUIRE(camera);
        const auto constants = VideoCommon::MakeMarioKartTemporalCamera(*camera, 2560, 1440);
        REQUIRE(constants);
        CHECK(std::abs(constants->near_plane - 1.0f) < 0.00001f);
        CHECK(std::abs(constants->far_plane - 10000.0f) < 1.0f);
        CHECK(std::abs(constants->vertical_fov - 1.13446408f) < 0.000001f);
        CHECK(std::abs(constants->aspect_ratio - 16.0f / 9.0f) < 0.000001f);
        CHECK(constants->view_to_clip[1 * 4 + 1] == block[0x84 / 4]);
        CHECK(constants->view_to_clip[2 * 4 + 1] == block[0x88 / 4]);
        CHECK(constants->view_to_clip[2 * 4 + 1] == -0.18f);
        CHECK(constants->view_to_clip[2 * 4 + 3] == -1.0f);
        for (size_t i = 0; i < 3; ++i) {
            CHECK(constants->position[i] == block[0xb0 / 4 + i * 4 + 3]);
            CHECK(std::abs(constants->right[i] - camera->view[i]) < 0.000001f);
            CHECK(std::abs(constants->up[i] - camera->view[4 + i]) < 0.000001f);
            CHECK(std::abs(constants->forward[i] + camera->view[8 + i]) < 0.000001f);
        }
        const std::array axes{constants->right, constants->up, constants->forward};
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                f64 dot{};
                for (size_t k = 0; k < 3; ++k) {
                    dot += static_cast<f64>(axes[i][k]) * axes[j][k];
                }
                CHECK(std::abs(dot - (i == j ? 1.0 : 0.0)) < 0.000001);
            }
        }
    }
}

TEST_CASE("MKW FG projection maps near and far to device depth", "[video_core][mkw]") {
    const auto camera = VideoCommon::ReadMarioKartCamera(Camera2911, 2560, 1440, false);
    REQUIRE(camera);
    const auto constants = VideoCommon::MakeMarioKartTemporalCamera(*camera, 2560, 1440);
    REQUIRE(constants);
    for (const f64 distance : {1.0, 10.0, 100.0, 10000.0}) {
        const std::array<f64, 4> view{0.25, -0.5, -distance, 1.0};
        const auto gl_clip = ColumnTransform(camera->projection, view);
        const auto device_clip = RowTransform(view, constants->view_to_clip);
        CHECK(std::abs(device_clip[0] - gl_clip[0]) < 1e-6);
        CHECK(std::abs(device_clip[1] - gl_clip[1]) < 1e-6);
        CHECK(std::abs(device_clip[2] - (gl_clip[2] + gl_clip[3]) * 0.5) < 1e-6);
        CHECK(device_clip[3] == gl_clip[3]);
        const f64 depth = device_clip[2] / device_clip[3];
        if (distance == 1.0) {
            CHECK(std::abs(depth) < 1e-7);
        } else if (distance == 10000.0) {
            CHECK(std::abs(depth - 1.0) < 1e-6);
        }
        const auto roundtrip = RowTransform(device_clip, constants->clip_to_view);
        for (size_t i = 0; i < 4; ++i) {
            CHECK(std::abs(roundtrip[i] - view[i]) < 0.002);
        }
    }
}

TEST_CASE("MKW FG reprojection agrees with captured camera motion", "[video_core][mkw]") {
    const auto camera = VideoCommon::ReadMarioKartCamera(Camera2911, 2560, 1440, false);
    REQUIRE(camera);
    const auto constants = VideoCommon::MakeMarioKartTemporalCamera(*camera, 2560, 1440);
    REQUIRE(constants);
    struct Sample { f64 u, v, depth, motion_x, motion_y; };
    // Independent double-precision evaluation of prevVP * inverse(VP), then clip-to-pixel.
    constexpr std::array samples{
        Sample{0.0, 0.0, 0.99, 12.2196466934, 3.1533631130},
        Sample{0.5, 0.5, 0.99, 3.1683319465, -0.9800911542},
        Sample{1.0, 1.0, 0.99, 2.3126639271, -0.4973171023},
        Sample{0.05, 0.9, 0.999, 6.8290175181, -2.8038837076},
        Sample{0.05, 0.9, 0.5, 183.3525894026, -131.1608600647},
    };
    for (const auto& sample : samples) {
        const std::array<f64, 4> clip{sample.u * 2.0 - 1.0, 1.0 - sample.v * 2.0,
                                     sample.depth, 1.0};
        const auto previous = RowTransform(clip, constants->clip_to_previous_clip);
        REQUIRE(previous[3] > 0);
        const f64 motion_x = ((previous[0] / previous[3] + 1.0) * 0.5 - sample.u) * 2560;
        const f64 motion_y = ((1.0 - previous[1] / previous[3]) * 0.5 - sample.v) * 1440;
        CHECK(std::abs(motion_x - sample.motion_x) < 0.001);
        CHECK(std::abs(motion_y - sample.motion_y) < 0.001);
        const auto roundtrip = RowTransform(previous, constants->previous_clip_to_clip);
        for (size_t i = 0; i < 4; ++i) {
            CHECK(std::abs(roundtrip[i] - clip[i]) < 1e-6);
        }
    }
}

TEST_CASE("MKW FG stationary camera keeps reprojection identity", "[video_core][mkw]") {
    auto camera = VideoCommon::ReadMarioKartCamera(Camera2910, 2560, 1440, false);
    REQUIRE(camera);
    camera->previous = camera->current;
    const auto constants = VideoCommon::MakeMarioKartTemporalCamera(*camera, 2560, 1440);
    REQUIRE(constants);
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 4; ++col) {
            CHECK(std::abs(constants->clip_to_previous_clip[row * 4 + col] -
                           (row == col ? 1.0 : 0.0)) < 1e-6);
        }
    }
    CHECK(constants->view_to_clip[2 * 4 + 1] == -0.18f);
}

TEST_CASE("MKW FG rejects inconsistent camera contracts", "[video_core][mkw]") {
    auto camera = VideoCommon::ReadMarioKartCamera(Camera2911, 2560, 1440, false);
    REQUIRE(camera);
    REQUIRE(VideoCommon::MakeMarioKartTemporalCamera(*camera, 2560, 1440));
    SECTION("Projection does not match the captured VP") { camera->projection[6] = 0.0f; }
    SECTION("View does not match the captured VP") { camera->view[3] += 10.0f; }
    SECTION("Pose is not the inverse view") { camera->camera_to_world[7] += 10.0f; }
    SECTION("Nonfinite projection") { camera->projection[5] = std::numeric_limits<f32>::infinity(); }
    SECTION("Nonfinite pose") { camera->camera_to_world[0] = std::numeric_limits<f32>::quiet_NaN(); }
    SECTION("Singular previous camera") { camera->previous.fill(0.0f); }
    SECTION("Nonfinite previous camera") { camera->previous[0] = std::numeric_limits<f32>::quiet_NaN(); }
    CHECK_FALSE(VideoCommon::MakeMarioKartTemporalCamera(*camera, 2560, 1440));
}

TEST_CASE("MKW FG camera validation tolerates distant-world float rounding", "[video_core][mkw]") {
    auto camera = VideoCommon::ReadMarioKartCamera(Camera2911, 2560, 1440, false);
    REQUIRE(camera);
    camera->camera_to_world[3] += 1000000.0f;
    camera->camera_to_world[7] -= 2000000.0f;
    camera->camera_to_world[11] += 3000000.0f;
    for (size_t row = 0; row < 3; ++row) {
        f64 translation{};
        for (size_t k = 0; k < 3; ++k) {
            translation -= static_cast<f64>(camera->view[row * 4 + k]) *
                           camera->camera_to_world[k * 4 + 3];
        }
        camera->view[row * 4 + 3] = static_cast<f32>(translation);
    }
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 4; ++col) {
            f64 value{};
            for (size_t k = 0; k < 4; ++k) {
                value += static_cast<f64>(camera->projection[row * 4 + k]) *
                         camera->view[k * 4 + col];
            }
            camera->current[row * 4 + col] = static_cast<f32>(value);
        }
    }
    camera->previous = camera->current;
    REQUIRE(VideoCommon::MakeMarioKartTemporalCamera(*camera, 2560, 1440));
    SECTION("A mismatched far-away pose still fails") { camera->camera_to_world[3] += 1000.0f; }
    SECTION("Linear checks remain tight") { camera->camera_to_world[0] += 0.01f; }
    CHECK_FALSE(VideoCommon::MakeMarioKartTemporalCamera(*camera, 2560, 1440));
}

namespace {
using std::pow;
#include "video_core/host_shaders/mkw_hudless_srgb.glsl"
} // namespace

TEST_CASE("MKW HUD-less recognizes the captured final copy", "[video_core][mkw]") {
    using VideoCore::Surface::PixelFormat;
    const VideoCommon::MarioKartBlendImage scene{0x8122aff000, PixelFormat::B10G11R11_FLOAT, 2560, 1440};
    VideoCommon::MarioKartHudlessScene saved{2910, scene};
    auto source = scene;
    VideoCommon::MarioKartBlendImage target{0x2109d23000, PixelFormat::A8B8G8R8_SRGB, 3840, 2160};
    u64 frame = 2910, vertex = VideoCommon::MarioKartPresentVertex;
    u64 fragment = VideoCommon::MarioKartPresentFragment;
    u32 elements = 6, instances = 1;
    REQUIRE(saved.MatchCopy(frame, vertex, fragment, elements, instances, source, target));
    SECTION("Previous scene") { ++frame; }
    SECTION("Different scene allocation") { ++source.identity; }
    SECTION("Different vertex shader") { ++vertex; }
    SECTION("Different fragment shader") { ++fragment; }
    SECTION("Only one triangle") { elements = 3; }
    SECTION("No actual work") { instances = 0; }
    SECTION("Instanced copy") { instances = 2; }
    SECTION("Unencoded destination") { target.format = PixelFormat::A8B8G8R8_UNORM; }
    SECTION("Copy is scaled") { source.width = 1920; }
    SECTION("Scene does not fit") { target.height = 1080; }
    CHECK_FALSE(saved.MatchCopy(frame, vertex, fragment, elements, instances, source, target));
}

TEST_CASE("MKW HUD-less never associates another present or frame", "[video_core][mkw]") {
    VideoCommon::MarioKartHudlessScene saved{
        2910, {0x8122aff000, VideoCore::Surface::PixelFormat::B10G11R11_FLOAT, 2560, 1440}};
    CHECK_FALSE(saved.MatchesPresent(2910, 0));
    saved.present = 0x2109d23000;
    CHECK(saved.MatchesPresent(2910, 0x2109d23000));
    CHECK_FALSE(saved.MatchesPresent(2911, 0x2109d23000));
    CHECK_FALSE(saved.MatchesPresent(2910, 0x210bd03000));
    saved.present = 0;
    CHECK_FALSE(saved.MatchesPresent(2910, 0x2109d23000));
}

TEST_CASE("MKW HUD-less supports only reproducible presentation", "[video_core][mkw]") {
    Tegra::FramebufferConfig config{};
    config.width = 3840;
    config.height = 2160;
    config.crop_rect = {0, 0, 2560, 1440};
    config.blending = Tegra::BlendMode::Opaque;
    size_t layers = 1;
    bool effects = false, temporal = false;
    REQUIRE(VideoCommon::CanComposeMarioKartHudless(config, 2560, 1440, layers, effects, temporal));
    SECTION("Extra layer") { layers = 2; }
    SECTION("Real post effect") { effects = true; }
    SECTION("Temporal presentation filter") { temporal = true; }
    SECTION("Blended layer") { config.blending = Tegra::BlendMode::Coverage; }
    SECTION("Rotation") { config.transform_flags = Service::android::BufferTransformFlags::Rotate90; }
    SECTION("Crop includes guest padding") { config.crop_rect.right = 2561; }
    SECTION("Empty crop defaults to full padded allocation") { config.crop_rect = {}; }
    CHECK_FALSE(VideoCommon::CanComposeMarioKartHudless(config, 2560, 1440, layers, effects, temporal));
}

TEST_CASE("MKW HUD-less shares crop flip and letterbox coordinates", "[video_core][mkw]") {
    Tegra::FramebufferConfig config{};
    config.width = 3840;
    config.height = 2160;
    config.crop_rect = {0, 0, 2560, 1440};
    const Common::Rectangle<u32> screen{240, 135, 3600, 2025};
    for (const auto flags : {Service::android::BufferTransformFlags::Unset,
                            Service::android::BufferTransformFlags::FlipH,
                            Service::android::BufferTransformFlags::FlipV,
                            Service::android::BufferTransformFlags::Rotate180}) {
        config.transform_flags = flags;
        REQUIRE(VideoCommon::CanComposeMarioKartHudless(config, 2560, 1440, 1, false, false));
        const auto crop = Tegra::NormalizeCrop(config, 3840, 2160);
        const auto vertices = Vulkan::MakeScreenVertices(screen, crop);
        CHECK((vertices[0].position == std::array<f32, 2>{240, 135}));
        CHECK((vertices[3].position == std::array<f32, 2>{3600, 2025}));
        const bool flip_h = (static_cast<u32>(flags) & 1) != 0;
        const bool flip_v = (static_cast<u32>(flags) & 2) != 0;
        CHECK(vertices[0].tex_coord[0] == (flip_h ? 2.0f / 3 : 0));
        CHECK(vertices[3].tex_coord[0] == (flip_h ? 0 : 2.0f / 3));
        CHECK(vertices[0].tex_coord[1] == (flip_v ? 2.0f / 3 : 0));
        CHECK(vertices[3].tex_coord[1] == (flip_v ? 0 : 2.0f / 3));
    }
}

TEST_CASE("MKW HUD-less shader encoding matches captured present bytes", "[video_core][mkw]") {
    // Exact B10G11R11 samples and corresponding pass119 bytes in captured frame2910.
    struct Sample { std::array<f32, 3> linear; std::array<u8, 3> encoded; };
    const std::array samples{
        Sample{{0.197265625f, 0.7421875f, 0.828125f}, {123, 224, 235}},
        Sample{{0.181640625f, 0.193359375f, 0.1484375f}, {118, 122, 108}},
        Sample{{0.072265625f, 0.0673828125f, 0.0556640625f}, {76, 74, 67}},
        Sample{{0.328125f, 0.30078125f, 0.24609375f}, {155, 149, 136}},
        Sample{{0.1875f, 0.1640625f, 0.115234375f}, {120, 113, 95}},
        Sample{{0.109375f, 0.1484375f, 0.1328125f}, {93, 108, 102}},
    };
    for (const auto& sample : samples) {
        for (size_t channel = 0; channel < 3; ++channel) {
            const auto encoded = std::lround(MarioKartHudlessSrgb(sample.linear[channel]) * 255);
            CHECK(std::abs(encoded - sample.encoded[channel]) <= 1);
        }
    }
    CHECK(MarioKartHudlessSrgb(-1) == 0);
    CHECK(std::abs(MarioKartHudlessSrgb(2) - 1) < 1e-6);
    CHECK(std::abs(MarioKartHudlessSrgb(0.0031308f) - 0.0404499f) < 1e-6);
}

TEST_CASE("MKW HUD-less diagnostic counts pixels and ignores alpha", "[video_core][mkw]") {
    const std::array<u8, 16> final_color{10, 20, 30, 255, 40, 50, 60, 255,
                                        70, 80, 90, 255, 255, 0, 0, 255};
    const std::array<u8, 16> hudless{10, 20, 30, 0, 42, 48, 62, 0,
                                    70, 83, 90, 255, 0, 0, 0, 255};
    const auto result = VideoCommon::CompareMarioKartHudless(final_color, hudless);
    CHECK(result.pixels == 4);
    CHECK(result.changed == 2);
    CHECK(result.maximum == 255);
}

TEST_CASE("MKW HUD-less fence retirement preserves unconsumed resources", "[video_core][mkw]") {
    using Vulkan::Streamline::FrameGenInputFence;
    auto semaphore = reinterpret_cast<VkSemaphore>(uintptr_t{0x1234});
    FrameGenInputFence fence{semaphore, 42, true};
    size_t waits{};
    bool consumed = false;
    const auto wait = [&](VkSemaphore actual, u64 value) {
        ++waits;
        CHECK(actual == semaphore);
        CHECK(value == 42);
        return consumed;
    };
    CHECK_FALSE(fence.Retire(wait));
    CHECK(fence.pending);
    CHECK(fence.semaphore == semaphore);
    CHECK(fence.value == 42);
    consumed = true;
    CHECK(fence.Retire(wait));
    CHECK_FALSE(fence.pending);
    CHECK(fence.Retire(wait));
    CHECK(waits == 2);
    fence = {.pending = true};
    CHECK_FALSE(fence.Retire(wait));
    CHECK(fence.pending);
    CHECK(waits == 2);
}

TEST_CASE("MKW final-copy bindings are tracked without frame dumping", "[video_core][mkw]") {
    using VideoCommon::ShouldTrackMarioKartSampledImages;
    using VideoCommon::MarioKartBlendFragment;
    using VideoCommon::MarioKartPresentFragment;
    using VideoCommon::MarioKartPresentVertex;
    CHECK(ShouldTrackMarioKartSampledImages(false, true, MarioKartPresentVertex, MarioKartPresentFragment));
    CHECK(ShouldTrackMarioKartSampledImages(false, true, 0, MarioKartBlendFragment));
    CHECK_FALSE(ShouldTrackMarioKartSampledImages(false, false, MarioKartPresentVertex, MarioKartPresentFragment));
    CHECK_FALSE(ShouldTrackMarioKartSampledImages(false, false, 0, MarioKartBlendFragment));
    CHECK_FALSE(ShouldTrackMarioKartSampledImages(false, true, MarioKartPresentVertex + 1, MarioKartPresentFragment));
    CHECK_FALSE(ShouldTrackMarioKartSampledImages(false, true, MarioKartPresentVertex, MarioKartPresentFragment + 1));
    CHECK(ShouldTrackMarioKartSampledImages(true, false, 0, 0));
}

TEST_CASE("MKW FG rectangle follows the final image blit or copy", "[video_core][mkw]") {
    using VideoCommon::MapMarioKartPresentRect;
    const auto full = MapMarioKartPresentRect({0, 0, 3839, 1986}, 3839, 1986, 3840, 1986, true);
    REQUIRE(full);
    CHECK(full->left == 0);
    CHECK(full->top == 0);
    CHECK(full->right == 3840);
    CHECK(full->bottom == 1986);
    const auto letterbox = MapMarioKartPresentRect({240, 135, 3600, 2025}, 3840, 2160, 2560, 1440, true);
    REQUIRE(letterbox);
    CHECK(letterbox->left == 160);
    CHECK(letterbox->top == 90);
    CHECK(letterbox->right == 2400);
    CHECK(letterbox->bottom == 1350);
    const auto coverage = MapMarioKartPresentRect({1, 1, 3, 3}, 4, 4, 7, 7, true);
    REQUIRE(coverage);
    CHECK(coverage->left == 1);
    CHECK(coverage->top == 1);
    CHECK(coverage->right == 6);
    CHECK(coverage->bottom == 6);
    const auto copy = MapMarioKartPresentRect({240, 135, 3600, 2025}, 3840, 2160, 2560, 1440, false);
    REQUIRE(copy);
    CHECK(copy->left == 240);
    CHECK(copy->top == 135);
    CHECK(copy->right == 2560);
    CHECK(copy->bottom == 1440);
    const auto padded_copy = MapMarioKartPresentRect({0, 0, 3839, 1986}, 3839, 1986, 3840, 1986, false);
    REQUIRE(padded_copy);
    CHECK(padded_copy->right == 3839);
    const auto bounded = MapMarioKartPresentRect({0, 0, 4000, 2200}, 3840, 2160, 2560, 1440, true);
    REQUIRE(bounded);
    CHECK(bounded->right == 2560);
    CHECK(bounded->bottom == 1440);
    CHECK_FALSE(MapMarioKartPresentRect({1, 0, 1, 4}, 4, 4, 7, 7, true));
    CHECK_FALSE(MapMarioKartPresentRect({5, 0, 6, 4}, 4, 4, 7, 7, true));
    CHECK_FALSE(MapMarioKartPresentRect({3, 0, 4, 4}, 4, 4, 2, 2, false));
    CHECK_FALSE(MapMarioKartPresentRect({0, 0, 4, 4}, 0, 4, 7, 7, true));
    CHECK_FALSE(MapMarioKartPresentRect({0, 0, 4, 4}, 4, 4, 0, 7, true));
}

namespace {
using std::abs;
using std::round;
#include "video_core/host_shaders/mkw_ui_alpha.glsl"
} // namespace

TEST_CASE("MKW UI alpha compares encoded bytes at the exact two-byte boundary",
          "[video_core][mkw]") {
    for (int value = 0; value <= 255; ++value) {
        CHECK(MkwEncodedByte(static_cast<float>(value) / 255.0f) == value);
        for (const int delta : {-3, -2, 0, 2, 3}) {
            const int other = value + delta;
            if (other < 0 || other > 255) {
                continue;
            }
            const int a = MkwEncodedByte(static_cast<float>(value) / 255.0f);
            const int b = MkwEncodedByte(static_cast<float>(other) / 255.0f);
            CHECK(MkwUiAlphaPixel(0, 0, 0, 0, 1, 1, a, 0, 0, b, 0, 0) == (std::abs(delta) > 2));
            CHECK(MkwUiAlphaPixel(0, 0, 0, 0, 1, 1, 0, a, 0, 0, b, 0) == (std::abs(delta) > 2));
            CHECK(MkwUiAlphaPixel(0, 0, 0, 0, 1, 1, 0, 0, a, 0, 0, b) == (std::abs(delta) > 2));
        }
    }
}

TEST_CASE("MKW UI alpha excludes letterbox pixels and ignores color alpha", "[video_core][mkw]") {
    const auto alpha = [](int x, int y, std::array<u8, 4> a, std::array<u8, 4> b) {
        return MkwUiAlphaPixel(x, y, 160, 90, 2400, 1350, a[0], a[1], a[2], b[0], b[1], b[2]);
    };
    CHECK_FALSE(alpha(160, 90, {10, 20, 30, 0}, {10, 20, 30, 255}));
    CHECK(alpha(160, 90, {10, 20, 30, 0}, {13, 20, 30, 255}));
    CHECK(alpha(2399, 1349, {10, 20, 30, 0}, {10, 20, 33, 255}));
    CHECK_FALSE(alpha(159, 90, {0, 0, 0, 0}, {255, 255, 255, 255}));
    CHECK_FALSE(alpha(160, 89, {0, 0, 0, 0}, {255, 255, 255, 255}));
    CHECK_FALSE(alpha(2400, 90, {0, 0, 0, 0}, {255, 255, 255, 255}));
    CHECK_FALSE(alpha(160, 1350, {0, 0, 0, 0}, {255, 255, 255, 255}));
    CHECK_FALSE(MkwUiAlphaPixel(0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0));
}

TEST_CASE("MKW recomposition follows successful tags and option fallback", "[video_core][mkw]") {
    using Vulkan::Streamline::FrameGenUiOptionsState;
    FrameGenUiOptionsState state;
    CHECK(state.NeedsUpdate(false));
    state.Applied(false, false);
    CHECK_FALSE(state.NeedsUpdate(false));
    for (bool enabled : {false, true}) {
        for (bool hudless : {false, true}) {
            for (bool alpha : {false, true}) {
                CHECK(FrameGenUiOptionsState::Desired(enabled, hudless, alpha) ==
                      (enabled && hudless && alpha));
            }
        }
    }
    CHECK(state.NeedsUpdate(true));
    state.Applied(true, true);
    CHECK(state.active);
    CHECK_FALSE(state.NeedsUpdate(true));
    // Input loss updates options even when FG mode and multiplier have not changed.
    CHECK(state.NeedsUpdate(FrameGenUiOptionsState::Desired(true, true, false)));
    state.Applied(false, false);
    CHECK_FALSE(state.active);
    CHECK(state.NeedsUpdate(true));
    // A failed enable with a successful off fallback must not claim recomposition is active.
    state.Applied(true, false);
    CHECK_FALSE(state.active);
    CHECK_FALSE(state.NeedsUpdate(true));
    CHECK(state.NeedsUpdate(false));
}
