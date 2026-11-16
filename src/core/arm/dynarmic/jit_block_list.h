// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"

namespace Core {

/// The blocks the JITs compiled in earlier runs of a program, each with the cores that
/// compiled it, kept in a file: the next run compiles them before the game runs them. A block
/// is in the list once; the file is appended to while the game runs and written again
/// without repeats when it is read.
class JitBlockList {
public:
    struct Block {
        u64 location;
        u32 first_instruction;
        // One bit per core that compiled the block.
        u32 cores;
    };
    static_assert(sizeof(Block) == 16);

    /// The list of the program, read when it is first asked for.
    static JitBlockList& For(u64 program_id);

    /// The blocks a core compiles ahead: those it compiled in an earlier run, and those
    /// three or more cores did (code of jobs that move between the cores).
    std::vector<Block> ForCore(size_t core) const;

    /// Adds the blocks, or the cores of blocks it holds, that are new to the list, and
    /// appends them to the file.
    void Add(std::span<const Block> compiled);

    size_t Size() const;

private:
    explicit JitBlockList(std::filesystem::path file);

    std::filesystem::path file;
    mutable std::mutex mutex;
    std::unordered_map<u64, Block> blocks;
};

} // namespace Core
