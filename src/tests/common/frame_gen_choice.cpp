// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include "common/settings.h"

TEST_CASE("NVIDIA frame generation runs in Streamline, not the emulator's present pass",
          "[common][settings]") {
    using Settings::FrameGenType;
    const bool frame_gen = Settings::values.frame_gen.GetValue();
    const FrameGenType type = Settings::values.frame_gen_type.GetValue();

    const auto choose = [](bool on, FrameGenType chosen) {
        Settings::values.frame_gen.SetValue(on);
        Settings::values.frame_gen_type.SetValue(chosen);
    };
    choose(false, FrameGenType::Nvidia);
    CHECK_FALSE(Settings::FrameGenActive());
    CHECK_FALSE(Settings::NvidiaFrameGenChosen());
    choose(true, FrameGenType::Lsfg);
    CHECK(Settings::FrameGenActive());
    CHECK_FALSE(Settings::NvidiaFrameGenChosen());
    choose(true, FrameGenType::Fsr3);
    CHECK(Settings::FrameGenActive());
    CHECK_FALSE(Settings::NvidiaFrameGenChosen());
    choose(true, FrameGenType::Nvidia);
    CHECK_FALSE(Settings::FrameGenActive());
    CHECK(Settings::NvidiaFrameGenChosen());
    CHECK(Settings::FrameGenGenerations() == 0);

    // A launcher's LICHII_DLSS_FG=1 picks NVIDIA's at the next game start, whatever is chosen.
    Settings::SetSessionOverride("LICHII_DLSS_FG", "1");
    Settings::UpdateRescalingInfo();
    choose(true, FrameGenType::Lsfg);
    CHECK_FALSE(Settings::FrameGenActive());
    CHECK(Settings::NvidiaFrameGenChosen());
    Settings::SetSessionOverride("LICHII_DLSS_FG", nullptr);
    Settings::UpdateRescalingInfo();
    CHECK(Settings::FrameGenActive());

    choose(frame_gen, type);
}
