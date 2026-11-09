// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/buffer_cache/native_storage_size.h"

using VideoCommon::NativeStorageBufferSize;
using namespace Common::Literals;

TEST_CASE("Native storage bindings extend to the Mario Kart World probe data",
          "[video_core][native-storage-size]") {
    REQUIRE(NativeStorageBufferSize(32_MiB, 0x102edc0) == 0x102edc0);
}

TEST_CASE("Native storage bindings ignore smaller parameters and invalid lengths",
          "[video_core][native-storage-size]") {
    CHECK(NativeStorageBufferSize(32_MiB, 266) == 8_MiB);
    CHECK(NativeStorageBufferSize(32_MiB, 0) == 8_MiB);
    CHECK(NativeStorageBufferSize(32_MiB, 0x102edc4) == 8_MiB);
    CHECK(NativeStorageBufferSize(32_MiB, 32_MiB + 16) == 8_MiB);
    CHECK(NativeStorageBufferSize(32_MiB, (u64{1} << 32) | 0x102edc0) == 8_MiB);
    CHECK(NativeStorageBufferSize(4_MiB, 2_MiB) == 4_MiB);
}

TEST_CASE("Native storage binding lengths respect mapping and alignment boundaries",
          "[video_core][native-storage-size]") {
    CHECK(NativeStorageBufferSize(32_MiB, 8_MiB) == 8_MiB);
    CHECK(NativeStorageBufferSize(32_MiB, 8_MiB + 16) == 8_MiB + 16);
    CHECK(NativeStorageBufferSize(32_MiB, 32_MiB) == 32_MiB);
    CHECK(NativeStorageBufferSize(4_MiB, 0x102edc0) == 4_MiB);
    CHECK(NativeStorageBufferSize(0, 0x102edc0) == 0);
    CHECK(NativeStorageBufferSize(8_GiB, (u64{1} << 32) | 0x102edc0) == 8_MiB);
}
