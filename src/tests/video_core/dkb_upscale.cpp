// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <bit>
#include <limits>

#include <catch2/catch_test_macros.hpp>
#include "video_core/dkb_upscale.h"

namespace {

using VideoCommon::BananzaFrameShape;
using VideoCommon::Extent2D;
using VideoCommon::MarioKartBlendImage;
using VideoCore::Surface::PixelFormat;

// Fragment cbuf 3 in the tunnel (frames 2580 and 2581) and the town (frame 2940).
constexpr std::array<u32, 200> Camera2580Words{
    0xbf800000U, 0x00000000U, 0x35ebbbcfU, 0x390bf783U, 0x34e41dd2U, 0x3f78654eU, 0x3e77ba5eU,
    0x418048c9U, 0xb5e4bb3aU, 0x3e77ba5eU, 0xbf78654eU, 0xc2b4a836U, 0xbfadd2c9U, 0x00000000U,
    0x36201004U, 0x393e1305U, 0x3589ae33U, 0x4015eb8aU, 0x3f15845eU, 0x421ada55U, 0x35e4c6f1U,
    0xbe77c70eU, 0x3f787206U, 0x42b0b15cU, 0x35e4bb3aU, 0xbe77ba5eU, 0x3f78654eU, 0x42b4a836U,
    0x3fadd2c9U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x401a8279U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0xbf80068eU, 0xc0000347U, 0x00000000U, 0x00000000U,
    0xbf800000U, 0x00000000U, 0xbf800000U, 0x34e41dd0U, 0xb5e4bb37U, 0xb7e4bb36U, 0xa8162007U,
    0x3f78654cU, 0x3e77ba5cU, 0x40c96227U, 0x35ebbbcfU, 0x3e77ba5cU, 0xbf78654cU, 0xc2b70caaU,
    0x3f800000U, 0x461c4000U, 0x38d1b717U, 0x3f7ff972U, 0x38d1bc76U, 0x38d1bc76U, 0x3fe38e39U,
    0x3f100000U, 0x461c3c00U, 0x3f800347U, 0x00000000U, 0x3fadd2c9U, 0x3f3c8361U, 0x3ed413cdU,
    0x3f490fdbU, 0x401a8279U, 0x45340000U, 0x44ca8000U, 0x39b60b61U, 0x3a21d13aU, 0x44a00000U,
    0x44340000U, 0x3a4ccccdU, 0x3ab60b61U, 0x3ea22222U, 0x3ea22222U, 0x3ea22222U, 0x3ea22222U,
    0xbf800000U, 0x00000000U, 0x35ebbbcfU, 0x390bf783U, 0x34e41dd2U, 0x3f78654eU, 0x3e77ba5eU,
    0x418048ceU, 0xb5e4bb3aU, 0x3e77ba5eU, 0xbf78654eU, 0xc2b4a835U, 0xbfadd2c9U, 0x00000000U,
    0x36201004U, 0x393e1305U, 0x3589ae33U, 0x4015eb8aU, 0x3f15845eU, 0x421ada5bU, 0x35e4c6f1U,
    0xbe77c70eU, 0x3f787206U, 0x42b0b15bU, 0x35e4bb3aU, 0xbe77ba5eU, 0x3f78654eU, 0x42b4a835U,
    0x3fadd2c9U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x401a8279U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0xbf80068eU, 0xc0000347U, 0x00000000U, 0x00000000U,
    0xbf800000U, 0x00000000U, 0xbf800000U, 0x34e41dd0U, 0xb5e4bb37U, 0xb7e4bb37U, 0xa8162007U,
    0x3f78654cU, 0x3e77ba5cU, 0x40c9620fU, 0x35ebbbcfU, 0x3e77ba5cU, 0xbf78654cU, 0xc2b70ca9U,
    0x3f800000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x3f800000U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0x3f800000U, 0x00000000U, 0x00000000U, 0x00000000U,
    0x00000000U, 0x3f800000U, 0x88685098U, 0x00000021U, 0x81ce7044U, 0x00000000U, 0x81ce7044U,
    0x00000000U, 0x81ce7044U, 0x00000000U, 0x3f800000U, 0x3f800000U, 0x00000000U, 0x00000000U,
    0xbc8b0f0aU, 0xb309a6b1U, 0x32bb75a8U, 0x3f00001eU, 0xb374133aU, 0x3cac2bd4U, 0xbc5a86c0U,
    0xbeb1f164U, 0xb100c796U, 0xbc5ee1eeU, 0xbcaf9a73U, 0xbf5f119aU, 0x00000000U, 0x00000000U,
    0x00000000U, 0x3f800000U, 0x3f27f1f2U, 0xbed825b5U, 0x3e892bd6U, 0x42ef0871U, 0x36757230U,
    0xbfad2364U, 0x3f5bc0f6U, 0x426be8fcU, 0x360435b8U, 0xbfa45c6cU, 0xbeb4ed2bU, 0x43340a0dU,
    0x3619357eU, 0xbf5825e0U, 0x3f092bf3U, 0x436f087cU,
};
constexpr auto Camera2580 = std::bit_cast<std::array<f32, 200>>(Camera2580Words);

constexpr std::array<u32, 200> Camera2581Words{
    0xbf800000U, 0x00000000U, 0x35ebbbcfU, 0x390bf783U, 0x34e41dd0U, 0x3f78654eU, 0x3e77ba5cU,
    0x418048c4U, 0xb5e4bb3aU, 0x3e77ba5cU, 0xbf78654eU, 0xc2b4a836U, 0xbfadd2c9U, 0x00000000U,
    0x36201004U, 0x393e1305U, 0x3589ae32U, 0x4015eb8aU, 0x3f15845dU, 0x421ada4fU, 0x35e4c6f1U,
    0xbe77c70cU, 0x3f787206U, 0x42b0b15cU, 0x35e4bb3aU, 0xbe77ba5cU, 0x3f78654eU, 0x42b4a836U,
    0x3fadd2c9U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x401a8279U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0xbf80068eU, 0xc0000347U, 0x00000000U, 0x00000000U,
    0xbf800000U, 0x00000000U, 0xbf800000U, 0x34e41dceU, 0xb5e4bb37U, 0xb7e4bb36U, 0xa7ef6a7eU,
    0x3f78654cU, 0x3e77ba5aU, 0x40c96235U, 0x35ebbbcfU, 0x3e77ba5aU, 0xbf78654cU, 0xc2b70caaU,
    0x3f800000U, 0x461c4000U, 0x38d1b717U, 0x3f7ff972U, 0x38d1bc76U, 0x38d1bc76U, 0x3fe38e39U,
    0x3f100000U, 0x461c3c00U, 0x3f800347U, 0x00000000U, 0x3fadd2c9U, 0x3f3c8361U, 0x3ed413cdU,
    0x3f490fdbU, 0x401a8279U, 0x45340000U, 0x44ca8000U, 0x39b60b61U, 0x3a21d13aU, 0x44a00000U,
    0x44340000U, 0x3a4ccccdU, 0x3ab60b61U, 0x3eaeeeefU, 0x3eaeeeefU, 0x3eaeeeefU, 0x3eaeeeefU,
    0xbf800000U, 0x00000000U, 0x35ebbbcfU, 0x390bf783U, 0x34e41dd2U, 0x3f78654eU, 0x3e77ba5eU,
    0x418048c9U, 0xb5e4bb3aU, 0x3e77ba5eU, 0xbf78654eU, 0xc2b4a836U, 0xbfadd2c9U, 0x00000000U,
    0x36201004U, 0x393e1305U, 0x3589ae33U, 0x4015eb8aU, 0x3f15845eU, 0x421ada55U, 0x35e4c6f1U,
    0xbe77c70eU, 0x3f787206U, 0x42b0b15cU, 0x35e4bb3aU, 0xbe77ba5eU, 0x3f78654eU, 0x42b4a836U,
    0x3fadd2c9U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x401a8279U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0xbf80068eU, 0xc0000347U, 0x00000000U, 0x00000000U,
    0xbf800000U, 0x00000000U, 0xbf800000U, 0x34e41dd0U, 0xb5e4bb37U, 0xb7e4bb36U, 0xa8162007U,
    0x3f78654cU, 0x3e77ba5cU, 0x40c96227U, 0x35ebbbcfU, 0x3e77ba5cU, 0xbf78654cU, 0xc2b70caaU,
    0x3f800000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x3f800000U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0x3f800000U, 0x00000000U, 0x00000000U, 0x00000000U,
    0x00000000U, 0x3f800000U, 0x8867d778U, 0x00000021U, 0x81ce7044U, 0x00000000U, 0x81ce7044U,
    0x00000000U, 0x81ce7044U, 0x00000000U, 0x3f800000U, 0x3f800000U, 0x00000000U, 0x00000000U,
    0xbc8b0f0aU, 0xb309a6b2U, 0x32bb75aaU, 0x3f00001eU, 0xb374133cU, 0x3cac2bd4U, 0xbc5a86c0U,
    0xbeb1f168U, 0xb100c794U, 0xbc5ee1ebU, 0xbcaf9a71U, 0xbf5f1194U, 0x00000000U, 0x00000000U,
    0x00000000U, 0x3f800000U, 0x3f27f1f0U, 0xbed825b4U, 0x3e892bd5U, 0x42ef086fU, 0x36757230U,
    0xbfad2364U, 0x3f5bc0f6U, 0x426be8feU, 0x360435b8U, 0xbfa45c6bU, 0xbeb4ed29U, 0x43340a0bU,
    0x3619357eU, 0xbf5825e0U, 0x3f092bf3U, 0x436f087aU,
};
constexpr auto Camera2581 = std::bit_cast<std::array<f32, 200>>(Camera2581Words);

constexpr std::array<u32, 200> Camera2940Words{
    0x3f51b3f2U, 0x80000000U, 0x3f12d5e9U, 0x4309c4dfU, 0x00000000U, 0x3f800000U, 0x00000000U,
    0x42c60000U, 0xbf12d5e9U, 0x00000000U, 0x3f51b3f2U, 0x4299ec2cU, 0x3f8e6345U, 0x00000000U,
    0x3f4766eaU, 0x433b16f5U, 0x00000000U, 0x401a8279U, 0x00000000U, 0x436f01d3U, 0x3f12e4f3U,
    0x00000000U, 0xbf51c96cU, 0xc29dfc24U, 0x3f12d5e9U, 0x00000000U, 0xbf51b3f2U, 0xc299ec2cU,
    0x3fadd2c9U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x401a8279U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0xbf800d1cU, 0xc000068eU, 0x00000000U, 0x00000000U,
    0xbf800000U, 0x00000000U, 0x3f51b3f2U, 0x00000000U, 0xbf12d5e9U, 0xc2896bd3U, 0x80000000U,
    0x3f800000U, 0x80000000U, 0xc2c60000U, 0x3f12d5e9U, 0x80000000U, 0x3f51b3f2U, 0xc30e1066U,
    0x3f800000U, 0x459c4000U, 0x3951b717U, 0x3f7ff2e5U, 0x3951c1d5U, 0x3951c1d5U, 0x3fe38e39U,
    0x3f100000U, 0x459c3800U, 0x3f80068eU, 0x00000000U, 0x3fadd2c9U, 0x3f3c8361U, 0x3ed413cdU,
    0x3f490fdbU, 0x401a8279U, 0x4502f000U, 0x44936000U, 0x39fa41beU, 0x3a5e5828U, 0x44a00000U,
    0x44340000U, 0x3a4ccccdU, 0x3ab60b61U, 0x3f400000U, 0x3f400000U, 0x3f400000U, 0x3f400000U,
    0x3f51b3f2U, 0x80000000U, 0x3f12d5e9U, 0x4309c4dfU, 0x00000000U, 0x3f800000U, 0x00000000U,
    0x42c60000U, 0xbf12d5e9U, 0x00000000U, 0x3f51b3f2U, 0x4299ec2cU, 0x3f8e6345U, 0x00000000U,
    0x3f4766eaU, 0x433b16f5U, 0x00000000U, 0x401a8279U, 0x00000000U, 0x436f01d3U, 0x3f12e4f3U,
    0x00000000U, 0xbf51c96cU, 0xc29dfc24U, 0x3f12d5e9U, 0x00000000U, 0xbf51b3f2U, 0xc299ec2cU,
    0x3fadd2c9U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x401a8279U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0xbf800d1cU, 0xc000068eU, 0x00000000U, 0x00000000U,
    0xbf800000U, 0x00000000U, 0x3f51b3f2U, 0x00000000U, 0xbf12d5e9U, 0xc2896bd3U, 0x80000000U,
    0x3f800000U, 0x80000000U, 0xc2c60000U, 0x3f12d5e9U, 0x80000000U, 0x3f51b3f2U, 0xc30e1066U,
    0x3f800000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x3f800000U, 0x00000000U,
    0x00000000U, 0x00000000U, 0x00000000U, 0x3f800000U, 0x00000000U, 0x00000000U, 0x00000000U,
    0x00000000U, 0x3f800000U, 0x0b802778U, 0x00000021U, 0x815b3778U, 0x00000000U, 0x88e80060U,
    0x00000021U, 0x00000000U, 0x00000000U, 0x3f800000U, 0x3f800000U, 0x00000000U, 0x00000000U,
    0x3c28ad88U, 0xbc005ad0U, 0x3bec3813U, 0x3fba5be9U, 0xbc2fa3d8U, 0x3b733606U, 0x3c9e755dU,
    0x4058d745U, 0xbc17457dU, 0xbc6a889eU, 0xbb1b5c11U, 0xbfe08914U, 0x00000000U, 0x00000000U,
    0x00000000U, 0x3f800000U, 0xbea0d2e7U, 0x3ead6b2bU, 0xbf526870U, 0xc221e84cU, 0x3f313274U,
    0xbe755dfcU, 0xbf9fdcfbU, 0xc34a0630U, 0xbe443ffcU, 0xbf5a8c14U, 0xbf276faaU, 0xc29d7599U,
    0x3ef4c492U, 0xbe297775U, 0xbf5cd2f3U, 0x41af1e20U,
};
constexpr auto Camera2940 = std::bit_cast<std::array<f32, 200>>(Camera2940Words);

// Compute cbuf 0 in the town (frame 2940).
constexpr std::array<u32, 24> Upscale2940Words{
    0x00000080U, 0x00000042U, 0x00000032U, 0x00000001U, 0x00000000U, 0x00000000U, 0x0000082eU,
    0x0000049aU, 0x00000000U, 0x00000000U, 0x00000b3fU, 0x00000653U, 0x0c072380U, 0x00000022U,
    0x00005ac0U, 0x00000000U, 0x895ad840U, 0x00000021U, 0x000009f0U, 0x00000000U, 0x162657d0U,
    0x00000021U, 0x00003000U, 0x00000000U,
};


constexpr Extent2D ReducedScene{2095, 1179};
constexpr Extent2D FullScene{2880, 1620};

} // namespace

TEST_CASE("Bananza upscaler follows the DLSS setting and explicit overrides", "[video_core][dkb]") {
    using VideoCommon::OcarinaUpscaleBackend;
    using VideoCommon::SelectBananzaUpscaler;
    CHECK(SelectBananzaUpscaler(true, true, "") == OcarinaUpscaleBackend::Dlss);
    CHECK(SelectBananzaUpscaler(false, true, "") == OcarinaUpscaleBackend::Native);
    CHECK(SelectBananzaUpscaler(true, false, "") == OcarinaUpscaleBackend::Native);
    CHECK(SelectBananzaUpscaler(true, true, "native") == OcarinaUpscaleBackend::Native);
    CHECK(SelectBananzaUpscaler(true, true, "fsr") == OcarinaUpscaleBackend::Fsr);
    CHECK(SelectBananzaUpscaler(false, false, "fsr") == OcarinaUpscaleBackend::Fsr);
}

TEST_CASE("Bananza recognizes both docked frame shapes", "[video_core][dkb]") {
    using VideoCommon::ClassifyBananzaBlend;
    CHECK(ClassifyBananzaBlend(ReducedScene, {2096, 1179}) == BananzaFrameShape::Upscale);
    CHECK(ClassifyBananzaBlend(FullScene, FullScene) == BananzaFrameShape::NativeSize);
    CHECK(ClassifyBananzaBlend({2879, 1619}, {2880, 1619}) == BananzaFrameShape::Upscale);
    CHECK_FALSE(ClassifyBananzaBlend(ReducedScene, FullScene));
    CHECK_FALSE(ClassifyBananzaBlend(FullScene, {2881, 1620}));
    CHECK_FALSE(ClassifyBananzaBlend({2095, 1620}, {2096, 1620}));
    CHECK_FALSE(ClassifyBananzaBlend(ReducedScene, {2159, 1179}));
}

TEST_CASE("Bananza recognizes the padded SMAA blend target before native upscale", "[video_core][dkb]") {
    using VideoCommon::MarioKartBlendFragment;
    using VideoCommon::RecognizeBananzaBlend;
    std::array inputs{MarioKartBlendImage{0x8112d17000, PixelFormat::B10G11R11_FLOAT, 2095, 1179},
                      MarioKartBlendImage{0x8113754000, PixelFormat::A8B8G8R8_UNORM, 2095, 1179}};
    auto target = MarioKartBlendImage{0x8114bce000, PixelFormat::B10G11R11_FLOAT, 2096, 1179};
    bool indexed = false;
    bool indirect = false;
    u32 vertices = 3;
    u32 instances = 1;
    u64 fragment = MarioKartBlendFragment;
    REQUIRE(RecognizeBananzaBlend(fragment, indexed, indirect, vertices, instances, inputs,
                                  target, ReducedScene) == BananzaFrameShape::Upscale);
    SECTION("Different shader") { ++fragment; }
    SECTION("Target is also sampled") { inputs[0].identity = target.identity; }
    SECTION("Texture descriptor includes the padded column") { ++inputs[0].width; }
    SECTION("Wrong target height") { ++target.height; }
    SECTION("Wrong weights format") { inputs[1].format = PixelFormat::B10G11R11_FLOAT; }
    SECTION("Indexed") { indexed = true; }
    SECTION("Indirect") { indirect = true; }
    SECTION("Extra vertex") { ++vertices; }
    SECTION("Instanced") { ++instances; }
    CHECK_FALSE(RecognizeBananzaBlend(fragment, indexed, indirect, vertices, instances, inputs,
                                      target, ReducedScene));
}

TEST_CASE("Bananza recognizes an in-place SMAA blend at native size", "[video_core][dkb]") {
    using VideoCommon::MarioKartBlendFragment;
    using VideoCommon::RecognizeBananzaBlend;
    const MarioKartBlendImage target{0x811a247000, PixelFormat::B10G11R11_FLOAT, 2880, 1620};
    std::array inputs{target,
                      MarioKartBlendImage{0x8114e7f000, PixelFormat::A8B8G8R8_UNORM, 2880, 1620}};
    REQUIRE(RecognizeBananzaBlend(MarioKartBlendFragment, false, false, 3, 1, inputs, target,
                                  FullScene) == BananzaFrameShape::NativeSize);
    SECTION("Color comes from a separate image") { ++inputs[0].identity; }
    SECTION("Weights alias the target") { inputs[1].identity = target.identity; }
    CHECK_FALSE(RecognizeBananzaBlend(MarioKartBlendFragment, false, false, 3, 1, inputs, target,
                                      FullScene));
}

TEST_CASE("Bananza camera captures preserve scene extent and clip planes", "[video_core][dkb]") {
    using VideoCommon::ReadBananzaCamera;
    const auto first = ReadBananzaCamera(Camera2580, FullScene, false);
    const auto second = ReadBananzaCamera(Camera2581, FullScene, false);
    const auto reduced = ReadBananzaCamera(Camera2940, {2096, 1179}, false);
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(reduced);
    CHECK(first->extent == FullScene);
    CHECK(first->far_plane == 10000.0f);
    CHECK(first->matrices.current == second->matrices.previous);
    CHECK(reduced->extent == ReducedScene);
    CHECK(reduced->near_plane == 1.0f);
    CHECK(reduced->far_plane == 5000.0f);
    CHECK_FALSE(ReadBananzaCamera(Camera2940, {2880, 1179}, false));
    CHECK_FALSE(ReadBananzaCamera(Camera2940, {2094, 1179}, false));
    CHECK_FALSE(ReadBananzaCamera(Camera2940, {2096, 1180}, false));
    CHECK_FALSE(ReadBananzaCamera(Camera2940, {2096, 1179}, true));
    CHECK_FALSE(ReadBananzaCamera(std::span{Camera2940}.first(111), {2096, 1179}, false));
}

TEST_CASE("Bananza rejects malformed camera fields", "[video_core][dkb]") {
    auto block = Camera2940;
    SECTION("Nonfinite matrix") { block[0x30 / 4] = std::numeric_limits<f32>::infinity(); }
    SECTION("Nonfinite scene width") { block[0x120 / 4] = std::numeric_limits<f32>::quiet_NaN(); }
    SECTION("Fractional scene width") { block[0x120 / 4] += 0.5f; }
    SECTION("Invalid clip planes") { block[0xe4 / 4] = block[0xe0 / 4]; }
    SECTION("Different projection") { block[0xa8 / 4] = 0.0f; }
    CHECK_FALSE(VideoCommon::ReadBananzaCamera(block, {2096, 1179}, false));
}

TEST_CASE("Bananza validates the captured native upscale dispatch", "[video_core][dkb]") {
    using VideoCommon::IsBananzaUpscaleDispatch;
    constexpr u64 blend = 0x8114bce000;
    constexpr u64 tonemapped = 0x8112d17000;
    constexpr u64 depth = 0x8000;
    auto hash = VideoCommon::BananzaUpscaleShader;
    bool indirect = false;
    const std::array inputs{MarioKartBlendImage{blend, PixelFormat::B10G11R11_FLOAT, 2095, 1179},
                            MarioKartBlendImage{tonemapped, PixelFormat::B10G11R11_FLOAT, 2095, 1179}};
    auto sampled = std::span{inputs}.first(1);
    std::array written{MarioKartBlendImage{0x811a247000, PixelFormat::B10G11R11_FLOAT, 2880, 1620}};
    auto words = Upscale2940Words;
    auto output = std::span<const MarioKartBlendImage>{written};
    auto constants = std::span<const u32>{words};
    REQUIRE(IsBananzaUpscaleDispatch(hash, indirect, ReducedScene, blend, tonemapped, depth,
                                     sampled, output, constants));
    SECTION("Other shader") { ++hash; }
    SECTION("Indirect dispatch") { indirect = true; }
    SECTION("Extra sampled image") { sampled = inputs; }
    SECTION("Sampled image is not the blend output") { sampled = std::span{inputs}.last(1); }
    SECTION("No written image") { output = {}; }
    SECTION("Wrong output format") { written[0].format = PixelFormat::A8B8G8R8_UNORM; }
    SECTION("Written image aliases tonemapped scene") { written[0].identity = tonemapped; }
    SECTION("Written image aliases scene depth") { written[0].identity = depth; }
    SECTION("Written image is not 1620p") { --written[0].width; }
    SECTION("Wrong source last column") { ++words[6]; }
    SECTION("Wrong output last row") { ++words[11]; }
    SECTION("Short constant buffer") { constants = constants.first(11); }
    CHECK_FALSE(IsBananzaUpscaleDispatch(hash, indirect, ReducedScene, blend, tonemapped, depth,
                                         sampled, output, constants));
}

TEST_CASE("Bananza recalculates jitter phases for each scene width", "[video_core][dkb]") {
    using VideoCommon::BananzaJitterPhaseCount;
    CHECK(BananzaJitterPhaseCount(2095, false) == 15);
    CHECK(BananzaJitterPhaseCount(2880, false) == 8);
    CHECK(BananzaJitterPhaseCount(2095, true) == 16);
    CHECK(BananzaJitterPhaseCount(2880, true) == 16);
    CHECK(BananzaJitterPhaseCount(1920, true) == 18);
}
