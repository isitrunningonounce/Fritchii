// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;
constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0x000fea00}};
constexpr Instruction MinMax{{0x0b037817, 0x1fffffff, 0x03800000, 0x001fc800}};
constexpr Instruction Reduction{{0x0500759f, 0x20c00206, 0x0010e000, 0x000fe200}};
}

TEST_CASE("Yoshi captured IMNMX preserves full-width immediates", "[ampere][yoshi]") {
    for (const u32 value : {0x80000U, 0x1fffffffU, 0x7fffffffU, 0x80000000U, 0xfff7ffffU}) {
        for (const u32 guard : {0U, 7U, 9U}) {
            for (const u32 mode : {0U, 0x200U, 0x04000000U, 0x04000200U}) {
                auto instruction = MinMax;
                instruction.words[0] = (instruction.words[0] & ~0xf000U) | (guard << 12);
                instruction.words[1] = value;
                instruction.words[2] |= mode;
                const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                INFO(lowered.error);
                REQUIRE(lowered);
                REQUIRE(lowered.instructions.size() == 3);
                CHECK(lowered.instructions[0] ==
                      (0x010000000000f0fbULL | (u64{guard} << 16) | (u64{value} << 20)));
                CHECK(lowered.instructions[1] ==
                      (0x5c2000000fb00b03ULL | (u64{guard} << 16) |
                       (u64{instruction.Bits(87, 4)} << 39) |
                       (u64{instruction.Bits(73, 1)} << 48)));
                CHECK(lowered.source_size == 32);
            }
        }
    }
}

TEST_CASE("Ampere IMNMX retains its signed twenty-bit immediate encoding", "[ampere][yoshi]") {
    for (const u32 value : {0U, 0x7ffffU, 0xfff80000U, 0xffffffffU}) {
        auto instruction = MinMax;
        instruction.words[1] = value;
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        REQUIRE(lowered.instructions.size() == 2);
        CHECK(lowered.instructions[0] ==
              (0x3820038000070b03ULL | (u64{value & 0x7ffffU} << 20) |
               (u64{value >> 31} << 56)));
    }
    auto unsupported = MinMax;
    unsupported.words[2] |= 0x400;
    REQUIRE_FALSE(LowerToMaxwell(std::array{unsupported, Exit}));
}

TEST_CASE("Yoshi captured SURED uses a buffer coordinate and descriptor", "[ampere][yoshi]") {
    for (const u32 dimension : {0U, 1U, 2U, 3U, 4U, 5U}) {
        for (const u32 guard : {0U, 7U, 9U}) {
            auto instruction = Reduction;
            instruction.words[0] = (instruction.words[0] & ~0xf000U) | (guard << 12);
            instruction.words[1] = (instruction.words[1] & 0x1fffffffU) | (dimension << 29);
            const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
            INFO(lowered.error);
            REQUIRE(lowered);
            REQUIRE(lowered.instructions.size() == 2);
            CHECK(lowered.required_texture_buffer == 3);
            CHECK(lowered.instructions[0] ==
                  (0xea000020006005ffULL | (u64{guard} << 16) | (u64{dimension} << 33)));
        }
    }
    for (const u32 dimension : {6U, 7U}) {
        auto instruction = Reduction;
        instruction.words[1] = (instruction.words[1] & 0x1fffffffU) | (dimension << 29);
        REQUIRE_FALSE(LowerToMaxwell(std::array{instruction, Exit}));
    }
    auto unsupported = Reduction;
    unsupported.words[2] |= 0x400;
    REQUIRE_FALSE(LowerToMaxwell(std::array{unsupported, Exit}));
}

TEST_CASE("Yoshi LDG load predicates zero results without accessing memory", "[ampere][yoshi]") {
    // Adapted from PR #762: an aliased address must survive until a taken load.
    constexpr Instruction captured{{0x2e1c7381, 0x00000800, 0x001e890e, 0x000ea200}};
    REQUIRE(LowerToMaxwell(std::array{captured, Exit}));
    for (const u32 size : {0U, 1U, 2U, 3U, 4U, 5U, 6U}) {
        for (const u32 execution : {0U, 8U, 7U}) {
            for (const u32 condition : {0U, 1U, 9U, 7U, 15U}) {
                for (const bool p0 : {false, true}) {
                    for (const bool p1 : {false, true}) {
                        CAPTURE(size, execution, condition, p0, p1);
                        const u32 encoded = (7U - (condition & 7)) | (condition & 8);
                        const Instruction load{{0x08080381U | (execution << 12), 0x00000800,
                                                0x001e8100U | (size << 9) | encoded, 0}};
                        const auto lowered = LowerToMaxwell(std::array{load, Exit});
                        REQUIRE(lowered);
                        const auto active = [&](u32 pred) {
                            const bool value = (pred & 7) == 7 ? true
                                             : (pred & 7) == 0 ? p0 : p1;
                            return value != bool(pred & 8);
                        };
                        std::array<u32, 256> registers{};
                        registers.fill(0x12345678);
                        registers[9] = 0;
                        const auto before = registers;
                        const u32 count = size <= 4 ? 1 : 1U << (size - 4);
                        u32 loads = 0;
                        const auto address = [](u32 i) { return 0x50U + i * 8 + ((i + 1) / 3) * 8; };
                        for (u32 pc = 0, steps = 0; pc < lowered.instructions.size(); ++pc) {
                            REQUIRE(++steps < 32);
                            const u64 word = lowered.instructions[pc];
                            if (!active((word >> 16) & 15)) continue;
                            if ((word >> 52) == 0xe30) break;
                            if ((word >> 52) == 0xe24) {
                                const s32 offset = static_cast<s32>((word >> 20) << 8) >> 8;
                                const u32 target = address(pc) + 8 + offset;
                                u32 next = 0;
                                while (next < lowered.instructions.size() && address(next) != target) ++next;
                                REQUIRE(next < lowered.instructions.size());
                                pc = next - 1;
                            } else if ((word >> 52) == 0x010) {
                                registers[word & 255] = static_cast<u32>(word >> 20);
                            } else if ((word >> 56) == 0xee) {
                                CHECK(((word >> 8) & 255) == 8);
                                CHECK(((word >> 20) & 0xffffff) == 8);
                                CHECK(((word >> 48) & 7) == size);
                                CHECK(registers[8] == before[8]);
                                CHECK(registers[9] == before[9]);
                                ++loads;
                                for (u32 component = 0; component < count; ++component)
                                    registers[8 + component] = 0xabcdef00U + component;
                            } else {
                                FAIL("Unexpected instruction in LDG predicate test");
                            }
                        }
                        CHECK(loads == (active(execution) && active(condition) ? 1U : 0U));
                        for (u32 component = 0; component < count; ++component) {
                            CHECK(registers[8 + component] ==
                                  (!active(execution) ? before[8 + component]
                                   : active(condition) ? 0xabcdef00U + component : 0U));
                        }
                        for (u32 reg = 0; reg < registers.size(); ++reg)
                            if (reg < 8 || reg >= 8 + count) CHECK(registers[reg] == before[reg]);
                    }
                }
            }
        }
    }
    for (const auto [word, mask] : std::array{
             std::array{0U, 1U << 24}, std::array{1U, 1U},
             std::array{2U, 1U << 4}, std::array{2U, 1U << 6},
             std::array{2U, 3U << 9}, std::array{2U, 1U << 14}}) {
        auto invalid = captured;
        invalid.words[word] |= mask;
        REQUIRE_FALSE(LowerToMaxwell(std::array{invalid, Exit}));
    }
}

TEST_CASE("Yoshi immediate I2F preserves F16 size signedness rounding and guard", "[ampere][yoshi]") {
    constexpr Instruction captured{{0x00197906, 0xffffffff, 0x00200c00, 0x000fe200}};
    for (const u32 guard : {0U, 7U, 9U}) {
        for (const u32 rounding : {0U, 1U, 2U, 3U}) {
            for (const u32 signed_source : {0U, 1U}) {
                for (const u32 value : {0U, 0xffffffffU, 0x80000000U, 0x1000001U}) {
                    auto instruction = captured;
                    instruction.words[0] = (instruction.words[0] & ~0xf000U) | (guard << 12);
                    instruction.words[1] = value;
                    instruction.words[2] = 0x00200800U | (signed_source << 10) | (rounding << 14);
                    const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                    REQUIRE(lowered);
                    REQUIRE(lowered.instructions.size() == 3);
                    CHECK(lowered.instructions[0] ==
                          (0x010000000000f0faULL | (u64{guard} << 16) | (u64{value} << 20)));
                    CHECK(lowered.instructions[1] ==
                          (0x5cb800000fa00919ULL | (u64{guard} << 16) |
                           (u64{signed_source} << 13) | (u64{rounding} << 39)));
                }
            }
        }
    }
    for (const auto [word, mask] : std::array{
             std::array{0U, 1U << 24}, std::array{2U, 1U},
             std::array{2U, 1U << 12}, std::array{3U, 1U}}) {
        auto invalid = captured;
        invalid.words[word] |= mask;
        REQUIRE_FALSE(LowerToMaxwell(std::array{invalid, Exit}));
    }
    auto invalid = captured;
    invalid.words[2] &= ~(3U << 11);
    REQUIRE_FALSE(LowerToMaxwell(std::array{invalid, Exit}));
    invalid = captured;
    invalid.words[0] = (invalid.words[0] & ~0xfffU) | 0x845U;
    REQUIRE_FALSE(LowerToMaxwell(std::array{invalid, Exit}));
}

TEST_CASE("Yoshi 3D SURED preserves all coordinates and the OR operation", "[ampere][yoshi]") {
    constexpr Instruction captured{{0x0800759f, 0xa0c00201, 0x0310e000, 0x000fe200}};
    for (const u32 guard : {0U, 7U, 9U}) {
        for (const u32 operation : {0U, 1U, 2U, 5U, 6U, 7U}) {
            auto instruction = captured;
            instruction.words[0] = (instruction.words[0] & ~0xf000U) | (guard << 12);
            instruction.words[2] = 0x0010e000U | (operation << 23);
            const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
            REQUIRE(lowered);
            REQUIRE(lowered.instructions.size() == 2);
            CHECK(lowered.required_texture_buffer == 3);
            CHECK(lowered.instructions[0] ==
                  (0xea00002a001008ffULL | (u64{guard} << 16) | (u64{operation} << 29)));
        }
    }
    for (const u32 operation : {3U, 4U, 8U}) {
        auto invalid = captured;
        invalid.words[2] = 0x0010e000U | (operation << 23);
        REQUIRE_FALSE(LowerToMaxwell(std::array{invalid, Exit}));
    }
    for (const u32 coordinate : {253U, 254U, 255U}) {
        auto invalid = captured;
        invalid.words[0] = (invalid.words[0] & 0xffffffU) | (coordinate << 24);
        REQUIRE_FALSE(LowerToMaxwell(std::array{invalid, Exit}));
    }
    for (const u32 coordinate : {214U, 218U}) {
        auto crossing = captured;
        crossing.words[0] = (crossing.words[0] & 0xffffffU) | (coordinate << 24);
        const auto pressure = Shader::Ampere::PrepareRegisterPressure(crossing, 216);
        REQUIRE(pressure.valid);
        const u32 mapped = pressure.instruction.Bits(24, 8);
        CHECK(mapped + 3 <= 216);
        for (u32 component = 0; component < 3; ++component) {
            const u32 original = coordinate + component;
            const u32 bank = original >= 216 ? 1U : 0U;
            CHECK(std::ranges::find(pressure.prefix,
                Shader::Ampere::NativeRegisterCopy(0, mapped + component, bank, original)) !=
                pressure.prefix.end());
        }
    }
    auto high = captured;
    high.words[0] = (high.words[0] & 0xffffffU) | (216U << 24);
    const auto pressure = Shader::Ampere::PrepareRegisterPressure(high, 216);
    REQUIRE(pressure.valid);
    CHECK(pressure.instruction.Bits(24, 8) + 3 <= 216);
    CHECK_FALSE(pressure.prefix.empty());
}
