// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Host code ranges of JIT-compiled guest blocks, so the Windows sampler (nx2_sampler) can name
// the guest code a guest CPU thread runs instead of "(jit)". Dynarmic's perf map feeds it on
// Windows (on Linux the same names go to perf's /tmp/perf-<pid>.map). It records only while the
// debug server is configured (EDEN_NX2_DEBUG_SERVER) or EDEN_NX2_JIT_SYMBOLS=1, and never with
// EDEN_NX2_JIT_SYMBOLS=0.
namespace Common::JitSymbols {

bool Enabled();

void Register(const void* start, const void* end, std::string_view name);

/// Forgets every range (the JIT cleared its code cache).
void Clear();

/// The name of the block holding the host address, if one is registered.
std::optional<std::string> Lookup(const void* address);

struct Block {
    std::uintptr_t start;
    std::uintptr_t end;
    std::string name;
};

/// Every registered block, for a profile file.
std::vector<Block> Snapshot();

/// A JIT's code space and the host address of guest address zero. Both are kept without a lock,
/// so a sampler can ask about them while the thread it stopped holds any lock.
void AddCodeRange(const void* start, std::size_t size);
void SetGuestBase(const void* base);
[[nodiscard]] bool InCode(std::uintptr_t address);
[[nodiscard]] std::uintptr_t GuestBase();

} // namespace Common::JitSymbols
