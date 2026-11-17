// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace Core::BananzaNvn2Commands {

/// Objects made one at a time and freed together, in chunks that grow from 4 to 64 objects:
/// one allocation for many objects, and an object never moves.
template <typename T>
class ChunkPool {
public:
    ChunkPool() = default;
    ChunkPool(ChunkPool&& other) noexcept
        : chunks{std::move(other.chunks)}, size{std::exchange(other.size, 0)},
          used{std::exchange(other.used, 0)} {
        other.chunks.clear();
    }
    ChunkPool& operator=(ChunkPool&& other) noexcept {
        if (this != &other) {
            chunks = std::move(other.chunks);
            other.chunks.clear();
            size = std::exchange(other.size, 0);
            used = std::exchange(other.used, 0);
        }
        return *this;
    }

    T* Make() {
        if (used == size) {
            size = std::min<size_t>(MaxChunk, size ? size * 2 : MinChunk);
            chunks.push_back(std::make_unique<T[]>(size));
            used = 0;
        }
        return &chunks.back()[used++];
    }

    void clear() {
        chunks.clear();
        size = 0;
        used = 0;
    }

private:
    static constexpr size_t MinChunk = 4;
    static constexpr size_t MaxChunk = 64;

    std::vector<std::unique_ptr<T[]>> chunks;
    // The newest chunk's size and how many of its objects are handed out.
    size_t size{};
    size_t used{};
};

} // namespace Core::BananzaNvn2Commands
