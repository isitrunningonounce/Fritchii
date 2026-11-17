// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>

#include "common/jit_symbols.h"

namespace Common::JitSymbols {
namespace {

struct Range {
    std::uintptr_t end;
    std::string name;
};

std::mutex mutex;
std::map<std::uintptr_t, Range> ranges;

constexpr std::size_t MaxCodeRanges = 64;
std::array<std::atomic<std::uintptr_t>, MaxCodeRanges> code_starts;
std::array<std::atomic<std::uintptr_t>, MaxCodeRanges> code_ends;
std::atomic<std::size_t> num_code_ranges{};
std::atomic<std::uintptr_t> guest_base{};

} // Anonymous namespace

bool Enabled() {
    static const bool enabled = [] {
        // Opt-in only (Fritchii): the owner's setup always sets EDEN_NX2_DEBUG_SERVER, and every
        // JIT block compile would otherwise take this lock and store a name nobody reads.
        const char* const value = std::getenv("EDEN_NX2_JIT_SYMBOLS");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

void Register(const void* start, const void* end, std::string_view name) {
    if (!Enabled() || start == end) {
        return;
    }
    const auto first = reinterpret_cast<std::uintptr_t>(start);
    std::scoped_lock lock{mutex};
    ranges.insert_or_assign(first, Range{reinterpret_cast<std::uintptr_t>(end), std::string{name}});
}

void Clear() {
    if (!Enabled()) {
        return;
    }
    std::scoped_lock lock{mutex};
    ranges.clear();
}

std::optional<std::string> Lookup(const void* address) {
    if (!Enabled()) {
        return std::nullopt;
    }
    const auto value = reinterpret_cast<std::uintptr_t>(address);
    std::scoped_lock lock{mutex};
    auto it = ranges.upper_bound(value);
    if (it == ranges.begin()) {
        return std::nullopt;
    }
    --it;
    if (value >= it->second.end) {
        return std::nullopt;
    }
    return it->second.name;
}

std::vector<Block> Snapshot() {
    std::vector<Block> blocks;
    std::scoped_lock lock{mutex};
    blocks.reserve(ranges.size());
    for (const auto& [start, range] : ranges) {
        blocks.push_back({start, range.end, range.name});
    }
    return blocks;
}

void AddCodeRange(const void* start, std::size_t size) {
    if (!Enabled()) {
        return;
    }
    std::scoped_lock lock{mutex};
    const std::size_t index = num_code_ranges.load(std::memory_order_relaxed);
    if (index == MaxCodeRanges) {
        return;
    }
    const auto first = reinterpret_cast<std::uintptr_t>(start);
    code_starts[index].store(first, std::memory_order_relaxed);
    code_ends[index].store(first + size, std::memory_order_relaxed);
    num_code_ranges.store(index + 1, std::memory_order_release);
}

void SetGuestBase(const void* base) {
    guest_base.store(reinterpret_cast<std::uintptr_t>(base), std::memory_order_relaxed);
}

bool InCode(std::uintptr_t address) {
    const std::size_t count = num_code_ranges.load(std::memory_order_acquire);
    for (std::size_t index = 0; index < count; ++index) {
        if (address >= code_starts[index].load(std::memory_order_relaxed) &&
            address < code_ends[index].load(std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

std::uintptr_t GuestBase() {
    return guest_base.load(std::memory_order_relaxed);
}

} // namespace Common::JitSymbols
