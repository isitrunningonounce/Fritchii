// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>

#include "common/common_types.h"
#include "common/literals.h"

namespace VideoCommon {

constexpr u32 NativeStorageBufferSize(u64 pool_size, u64 following_word) {
    using namespace Common::Literals;

    const u64 default_size = (std::min)(pool_size, 8_MiB);
    if ((following_word >> 32) == 0 && following_word > default_size &&
        following_word <= pool_size && (following_word & 15) == 0) {
        return static_cast<u32>(following_word);
    }
    return static_cast<u32>(default_size);
}

} // namespace VideoCommon
