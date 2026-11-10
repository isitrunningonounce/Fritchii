// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// LICHII_HOOK=main+0x43c870,sdk+0x7d74a0,...: log the arguments of guest functions whose first
// instruction is PACIBSP (a no-op here): the code fetch hands the JIT a BRK at that address, the
// breakpoint logs x0-x3 and the strings they point to, and execution continues after it.

#pragma once

#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/common_types.h"

namespace Core::LichiiHooks {

constexpr u32 Pacibsp = 0xD503237F;
constexpr u32 Brk = 0xD4200000;

inline std::mutex mutex;
inline std::unordered_map<std::string, u64> module_bases; ///< name -> load base
inline std::unordered_set<u64> hooks;                     ///< resolved guest addresses
inline std::unordered_map<u64, std::string> hook_names;
inline size_t resolved_with = 0;                          ///< module count hooks were resolved with

inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("LICHII_HOOK");
        return value != nullptr && value[0] != '\0';
    }();
    return enabled;
}

inline void SetModuleBase(const std::string& name, u64 base) {
    if (!Enabled()) {
        return;
    }
    std::scoped_lock lock{mutex};
    module_bases[name] = base;
}

/// Resolves LICHII_HOOK against the module bases known so far (lock held).
inline void ResolveLocked() {
    if (resolved_with == module_bases.size()) {
        return;
    }
    resolved_with = module_bases.size();
    hooks.clear();
    hook_names.clear();
    const std::string spec = std::getenv("LICHII_HOOK");
    size_t pos = 0;
    while (pos < spec.size()) {
        const size_t comma = spec.find(',', pos);
        const std::string item =
            spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        const size_t plus = item.find('+');
        if (plus != std::string::npos) {
            const auto it = module_bases.find(item.substr(0, plus));
            if (it != module_bases.end()) {
                const u64 address = it->second + std::strtoull(item.c_str() + plus + 1, nullptr, 16);
                hooks.insert(address);
                hook_names[address] = item;
            }
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
}

/// The instruction the JIT gets at address: a BRK at a hooked PACIBSP.
inline u32 InstructionFor(u64 address, u32 instruction) {
    if (instruction != Pacibsp || !Enabled()) {
        return instruction;
    }
    std::scoped_lock lock{mutex};
    ResolveLocked();
    return hooks.contains(address) ? Brk : instruction;
}

inline bool IsHook(u64 address, std::string* name) {
    if (!Enabled()) {
        return false;
    }
    std::scoped_lock lock{mutex};
    ResolveLocked();
    const auto it = hook_names.find(address);
    if (it == hook_names.end()) {
        return false;
    }
    if (name) {
        *name = it->second;
    }
    return true;
}

/// Whether this (hook, text) combination is new (logs each once, up to a limit).
inline bool FirstTime(const std::string& line) {
    static std::unordered_set<std::string> seen;
    std::scoped_lock lock{mutex};
    if (seen.size() >= 20000) {
        return false;
    }
    return seen.insert(line).second;
}

} // namespace Core::LichiiHooks
