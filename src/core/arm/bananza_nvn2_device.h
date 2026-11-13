// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>

#include "common/common_types.h"

namespace Core::BananzaNvn2Device {
namespace Detail {

// Bananza 3.2.0 SDK 18265DAEA2105B31F25346ADFBBD09B5EDF1207F.
// NSO SHA-256: 3abe3c58afb318ac6964d2ee418398223de0d7894e3157ad759f248971d39d91
// GetInteger +0x5D5240 calls +0x5D9960: base switch +0x5C0A20, jump tables
// +0xB9A2B0 (properties 2..0xB0) and +0xB9A53C (overrides 0..0xA9).
// Constants were extracted from every jump target's MOV/MOVK and branch path.
// -1 excludes device-dependent results and unrecognized properties, not a native return value.
inline constexpr std::array<s32, 0xB1> DeviceIntegers{
    231, 30, 14, 65536, 256, 8, 32, 4, // 0x00
    16, 32, 16, 4, 4, 4, 32, 32, // 0x08
    32, 256, 256, 4, 8, 512, 256, -1, // 0x10
    192, 4096, 16384, 16, 1024, 1024, 64, 1024, // 0x18, measured through the native SDK
    2147483647, 65535, 65535, 8, 1048576, 4096, 16, 4096, // 0x20
    1, 1, 1, -1, 16, -1, 201, 201, // 0x28
    17, 0, 1, 8, 8, 4, 32, -1, // 0x30
    -1, 128, -1, 1, 4, -1, -1, -1, // 0x38
    -1, -1, 1, 1, 1, 1, 5, 1, // 0x40
    1, 1, 1, 1, 1, 16, 2048, 32, // 0x48
    4096, 65536, 65536, 256, 16384, 262144, 4096, 1, // 0x50
    4, 2, 0, 4096, 16384, 4096, 128, 32, // 0x58
    65535, 1, -1, 256, 16, 65535, 65535, 256, // 0x60
    1, 16, 16, 16, 1, 1, 1, 256, // 0x68
    -1, 16, 8, 16, 32, 31, 536870912, 16777216, // 0x70
    16777216, 1073741824, 32, 1, 64, -1, 1, 2147483647, // 0x78
    32, 32, 4194303, 512, 256, 32, 16383, -1, // 0x80
    8, -1, -1, 4, 256, 16, -1, 1, // 0x88
    31, -1, 64, 1, 58368, 24576, 0, 256, // 0x90
    262144, -1, -1, 0, 201, 14, 201, 17, // 0x98
    163840, 8, 4, 256, 256, 4, 65536, 201, // 0xA0
    5, 1, 0, 63, 1, 0, 1, 0, // 0xA8
    4096, // 0xB0
};

} // namespace Detail

constexpr std::optional<s32> GetDeviceInteger(u32 property) {
    if (property >= Detail::DeviceIntegers.size() || Detail::DeviceIntegers[property] < 0) {
        return std::nullopt;
    }
    return Detail::DeviceIntegers[property];
}

} // namespace Core::BananzaNvn2Device
