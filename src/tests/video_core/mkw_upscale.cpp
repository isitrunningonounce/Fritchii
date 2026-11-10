// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <bit>
#include <limits>

#include <catch2/catch_test_macros.hpp>
#include "video_core/mkw_upscale.h"

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
