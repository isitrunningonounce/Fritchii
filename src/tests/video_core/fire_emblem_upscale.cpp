// SPDX-License-Identifier: GPL-3.0-or-later

#include <type_traits>
#include <bit>
#include <limits>
#include <catch2/catch_test_macros.hpp>
#include "shader_recompiler/exception.h"
#include "video_core/fire_emblem_upscale.h"

namespace {
using namespace Tegra::Texture;
namespace FE = VideoCommon::FireEmblemUpscale;

TICEntry Image(u32 address, u32 width, u32 height) {
    TICEntry image{};
    image.address_low = address;
    image.width_minus_one.Assign(width - 1);
    image.height_minus_1.Assign(height - 1);
    image.header_version.Assign(TICHeaderVersion::BlockLinear);
    image.texture_type.Assign(TextureType::Texture2D);
    image.format.Assign(TextureFormat::B10G11R11);
    return image;
}
} // namespace

TEST_CASE("FE upscale substitute only selects the captured output pass", "[fe_upscale]") {
    REQUIRE(FE::UseReplacement(FE::ProgramId, FE::OutputProgramKey, nullptr));
    REQUIRE(FE::UseReplacement(FE::ProgramId, FE::OutputProgramKey, "fsr"));
    for (const char* option : {"native", "standin", "0", "", "10", "invalid"}) {
        REQUIRE_FALSE(FE::UseReplacement(FE::ProgramId, FE::OutputProgramKey, option));
    }
    for (u64 title : {0ULL, 0x040074A01BF12000ULL, 0x0400C3F00006E000ULL}) {
        REQUIRE_FALSE(FE::UseReplacement(title, FE::OutputProgramKey, nullptr));
    }
    for (unsigned bit = 0; bit < 64; ++bit) {
        REQUIRE_FALSE(FE::UseReplacement(FE::ProgramId, FE::OutputProgramKey ^ (1ULL << bit),
                                          nullptr));
        REQUIRE_FALSE(FE::UseReplacement(FE::ProgramId ^ (1ULL << bit), FE::OutputProgramKey,
                                          nullptr));
    }
}

TEST_CASE("FE upscale substitute validates both surface descriptors", "[fe_upscale]") {
    const auto input = Image(0x100000, 1280, 720);
    const auto output = Image(0x800000, 2560, 1440);
    REQUIRE(FE::ValidPair(input, output));
    REQUIRE_FALSE(FE::ValidPair(output, input));
    REQUIRE_FALSE(FE::ValidPair(input, input));
    REQUIRE_FALSE(FE::ValidPair(input, Image(0, 2560, 1440)));
    REQUIRE_FALSE(FE::ValidPair(input, Image(0x800000, 1280, 1440)));
    REQUIRE_FALSE(FE::ValidPair(input, Image(0x800000, 2560, 720)));
    REQUIRE_FALSE(FE::ValidPair(input, Image(0x100000, 2560, 1440)));
    for (bool alter_input : {false, true}) {
        const auto rejected = [&](auto mutate) {
            auto from = input;
            auto to = output;
            mutate(alter_input ? from : to);
            REQUIRE_FALSE(FE::ValidPair(from, to));
        };
        rejected([](auto& image) { image.texture_type.Assign(TextureType::Texture2DArray); });
        rejected([](auto& image) { image.header_version.Assign(TICHeaderVersion::Pitch); });
        rejected([](auto& image) { image.format.Assign(TextureFormat::A8B8G8R8); });
        rejected([](auto& image) { image.depth_minus_1.Assign(1); });
        rejected([](auto& image) { image.max_mip_level.Assign(1); });
        rejected([](auto& image) { image.layer_base_0_2.Assign(1); });
        rejected([](auto& image) { image.address_low = 0; });
    }
}

TEST_CASE("FE substitute completion is not a cached shader rejection", "[fe_upscale]") {
    static_assert(!std::is_base_of_v<Shader::Exception, FE::DispatchReplaced>);
    bool completed = false;
    try {
        try {
            throw FE::DispatchReplaced{};
        } catch (const Shader::Exception&) {
            FAIL("Substitute completion must not enter shader rejection handling");
        }
    } catch (const FE::DispatchReplaced&) {
        completed = true;
    }
    REQUIRE(completed);
}

TEST_CASE("FE camera jitter belongs to one output attempt", "[fe_upscale]") {
    FE::FrameJitter state;
    REQUIRE_FALSE(state.Take());
    state.Capture(0x1000, 1280, 720, {0.25f, -0.375f});
    const auto frame = state.Take();
    REQUIRE(frame);
    REQUIRE(frame->color == 0x1000);
    REQUIRE(frame->width == 1280);
    REQUIRE(frame->height == 720);
    REQUIRE(frame->offset == std::array<f32, 2>{0.25f, -0.375f});
    REQUIRE_FALSE(state.Take());
    state.Capture(0x1000, 1280, 720, {0.25f, -0.375f});
    state.Capture(0x2000, 1920, 1080, {-0.5f, 0.5f});
    const auto next = state.Take();
    REQUIRE(next);
    REQUIRE(next->color == 0x2000);
    REQUIRE(next->offset == std::array<f32, 2>{-0.5f, 0.5f});
}

TEST_CASE("FE invalid camera jitter clears the earlier frame", "[fe_upscale]") {
    FE::FrameJitter state;
    for (f32 invalid : {std::numeric_limits<f32>::quiet_NaN(),
                        std::numeric_limits<f32>::infinity(), -0.51f, 0.51f}) {
        for (size_t component = 0; component < 2; ++component) {
            state.Capture(0x1000, 1280, 720, {0.25f, -0.375f});
            std::array<f32, 2> jitter{};
            jitter[component] = invalid;
            state.Capture(0x1000, 1280, 720, jitter);
            REQUIRE_FALSE(state.Take());
        }
    }
    state.Capture(0, 1280, 720, {});
    REQUIRE_FALSE(state.Take());
    state.Capture(0x1000, 0, 720, {});
    REQUIRE_FALSE(state.Take());
    state.Capture(0x1000, 1280, 0, {});
    REQUIRE_FALSE(state.Take());
}
