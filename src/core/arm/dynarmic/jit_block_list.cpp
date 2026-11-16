// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <fstream>
#include <map>
#include <memory>
#include <system_error>
#include <utility>

#include <fmt/format.h>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "core/arm/dynarmic/jit_block_list.h"

namespace Core {

JitBlockList& JitBlockList::For(u64 program_id) {
    static std::mutex lists_mutex;
    static std::map<u64, std::unique_ptr<JitBlockList>> lists;
    std::scoped_lock lock{lists_mutex};
    auto& list = lists[program_id];
    if (!list) {
        list.reset(new JitBlockList{Common::FS::GetEdenPath(Common::FS::EdenPath::CacheDir) /
                                    "lichii_jit" / fmt::format("{:016X}.bin", program_id)});
    }
    return *list;
}

JitBlockList::JitBlockList(std::filesystem::path file_) : file{std::move(file_)} {
    u64 records{};
    if (std::ifstream in{file, std::ios::binary}) {
        std::vector<Block> chunk(1 << 16);
        while (in.read(reinterpret_cast<char*>(chunk.data()),
                       static_cast<std::streamsize>(chunk.size() * sizeof(Block))) ||
               in.gcount() >= static_cast<std::streamsize>(sizeof(Block))) {
            const size_t count = static_cast<size_t>(in.gcount()) / sizeof(Block);
            records += count;
            for (size_t i = 0; i < count; ++i) {
                // A record without cores is of the older list, which did not know them.
                if (chunk[i].cores == 0) {
                    continue;
                }
                const auto [held, inserted] = blocks.try_emplace(chunk[i].location, chunk[i]);
                if (!inserted) {
                    held->second.cores |= chunk[i].cores;
                    held->second.first_instruction = chunk[i].first_instruction;
                }
            }
        }
    }
    if (records == blocks.size()) {
        LOG_INFO(Core_ARM, "JIT block list: {} blocks in {}", blocks.size(), file.string());
        return;
    }
    // Repeats, or records of the older list: the file is written again.
    std::error_code error;
    std::filesystem::create_directories(file.parent_path(), error);
    auto fresh = file;
    fresh += ".new";
    {
        std::ofstream out{fresh, std::ios::binary | std::ios::trunc};
        for (const auto& [location, block] : blocks) {
            out.write(reinterpret_cast<const char*>(&block), sizeof(block));
        }
    }
    std::filesystem::rename(fresh, file, error);
    LOG_INFO(Core_ARM, "JIT block list: {} written again with {} blocks ({} records before)",
             file.string(), blocks.size(), records);
}

std::vector<JitBlockList::Block> JitBlockList::ForCore(size_t core) const {
    const u32 bit = 1U << core;
    std::vector<Block> result;
    std::scoped_lock lock{mutex};
    for (const auto& [location, block] : blocks) {
        if ((block.cores & bit) != 0 || std::popcount(block.cores) >= 3) {
            result.push_back(block);
        }
    }
    // In the order of the game's code (the low 56 bits of a location are its address), so
    // blocks that run one after the other lie together in the code cache. In the order the
    // table holds them they are scattered, and the game ran a tenth slower.
    constexpr u64 AddressMask = 0x00FF'FFFF'FFFF'FFFFULL;
    std::ranges::sort(result, [](const Block& a, const Block& b) {
        return std::pair{a.location & AddressMask, a.location} <
               std::pair{b.location & AddressMask, b.location};
    });
    return result;
}

void JitBlockList::Add(std::span<const Block> compiled) {
    std::vector<Block> fresh;
    std::scoped_lock lock{mutex};
    for (const Block& block : compiled) {
        const auto [held, inserted] = blocks.try_emplace(block.location, block);
        if (inserted) {
            fresh.push_back(block);
        } else if ((held->second.cores | block.cores) != held->second.cores) {
            // Only the new cores: reading the file joins the records of a block.
            fresh.push_back({block.location, block.first_instruction,
                             block.cores & ~held->second.cores});
            held->second.cores |= block.cores;
        }
    }
    if (fresh.empty()) {
        return;
    }
    std::error_code error;
    std::filesystem::create_directories(file.parent_path(), error);
    std::ofstream out{file, std::ios::binary | std::ios::app};
    out.write(reinterpret_cast<const char*>(fresh.data()),
              static_cast<std::streamsize>(fresh.size() * sizeof(Block)));
}

size_t JitBlockList::Size() const {
    std::scoped_lock lock{mutex};
    return blocks.size();
}

} // namespace Core
