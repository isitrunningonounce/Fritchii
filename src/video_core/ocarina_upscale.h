// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>
#include <string_view>

#include "common/settings_enums.h"

namespace VideoCommon {

enum class OcarinaUpscaleBackend { Native, Fsr, Dlss };

constexpr OcarinaUpscaleBackend SelectOcarinaUpscaler(bool fsr_enabled, bool dlss_enabled,
                                                    bool dlss_available,
                                                    std::string_view override) {
    if (override == "native") {
        return OcarinaUpscaleBackend::Native;
    }
    if (override == "fsr") {
        return OcarinaUpscaleBackend::Fsr;
    }
    if ((dlss_enabled || override == "dlss") && dlss_available) {
        return OcarinaUpscaleBackend::Dlss;
    }
    return fsr_enabled ? OcarinaUpscaleBackend::Fsr : OcarinaUpscaleBackend::Native;
}

/// Whether the "OoT anti-aliasing" choice asks for AMD FSR (3.1, or 4 with FSR 3.1 as fallback).
constexpr bool OcarinaFsrChosen(Settings::OotAntiAliasing choice) {
    return choice == Settings::OotAntiAliasing::Fsr3 || choice == Settings::OotAntiAliasing::Fsr4;
}

/// The backend for the "OoT anti-aliasing" choice and the NVIDIA DLSS setting. An FSR choice wins
/// over DLSS; with Game or No sharpening, DLSS replaces the game's pass when it's on and available
/// (as before FSR came in).
constexpr OcarinaUpscaleBackend SelectOcarinaUpscaler(Settings::OotAntiAliasing choice,
                                                    bool dlss_enabled, bool dlss_available,
                                                    std::string_view override) {
    const bool fsr = OcarinaFsrChosen(choice);
    return SelectOcarinaUpscaler(fsr, dlss_enabled && !fsr, dlss_available, override);
}

/// DLSS gets the camera jitter and the texture LOD bias the FSR path uses: DLSS is made for
/// jittered input, and without it a still camera gives it nothing to reconstruct (2026-11-10).
/// NVIDIA's guide (3.7.3) gives the jitter in render pixels in the motion vectors' direction, as
/// FSR takes it. EDEN_NX2_DLSS_JITTER (boot): 1 = on (default), 0 = off, 2 = vertical sign
/// flipped, to check the sign in play (right is sharper, wrong shakes).
inline int OcarinaDlssJitterMode() {
    static const int mode = [] {
        const char* value = std::getenv("EDEN_NX2_DLSS_JITTER");
        return value != nullptr && value[0] != '\0' ? std::atoi(value) : 1;
    }();
    return mode;
}

/// Texture LOD bias added to Ocarina's mipmapped samplers while FSR or jittered DLSS upscales.
/// AMD and NVIDIA (DLSS guide 3.5) both recommend log2(render / output) - 1: -1.58 for docked
/// 1440p -> 4K.
inline float OcarinaFsrLodBias(OcarinaUpscaleBackend backend,
                               bool dlss_jitter = OcarinaDlssJitterMode() != 0) {
    if (backend == OcarinaUpscaleBackend::Fsr ||
        (backend == OcarinaUpscaleBackend::Dlss && dlss_jitter)) {
        return -1.58f;
    }
    return 0.0f;
}

/// Whether a sampler with that bias keeps it for an image: only docked (handheld isn't measured),
/// not for images the GPU wrote (render targets would read a mip too sharp), and not once the
/// upscaler failed this session (the game's own pass runs again).
constexpr bool KeepOcarinaFsrLodBias(bool docked, bool upscaler_failed, bool gpu_written) {
    return docked && !upscaler_failed && !gpu_written;
}

} // namespace VideoCommon
