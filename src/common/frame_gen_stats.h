// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// What DLSS Frame Generation puts on screen, for the frontend's FPS counters. The renderer's
// present thread writes it once per second; the game's own FPS stays in PerfStats.

#pragma once

#include <atomic>

namespace Common::FrameGenStats {

/// The running game's renderer was set up for DLSS Frame Generation (it can be switched now).
inline std::atomic<bool> available{};
/// The player's switch (status bar): off keeps the setup but generates nothing.
inline std::atomic<bool> user_enabled{true};
/// Frame generation is on (its frames are counted below).
inline std::atomic<bool> active{};
/// Frames shown per second over the last second, generated ones included.
inline std::atomic<double> shown_fps{};
/// Of those, the share with a generated frame in front of them (0..1).
inline std::atomic<double> generated_share{};

} // namespace Common::FrameGenStats
