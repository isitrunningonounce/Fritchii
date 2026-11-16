// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <iterator>
#include <map>
#include <memory>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

/// The host buffers of storage spans that stay as the GPU left them from one replay to the
/// next, so a span is not copied in from guest memory again for every submission.
/// An entry holds its span's bytes while the game has not flushed the pool over them and no
/// other buffer was made for the same guest bytes; entries never overlap in guest memory.
class BananzaNvn2KeptStorage {
public:
    using Buffer = std::shared_ptr<vk::Buffer>;

    /// The buffer of the span when it still holds the span's bytes.
    Buffer Find(u64 guest, u64 size, u64 pool, u64 stamp, u64 now) {
        const auto found = entries.find(guest);
        if (found == entries.end() || found->second.size != size ||
            found->second.pool != pool || found->second.stamp != stamp) {
            return {};
        }
        found->second.used = now;
        return found->second.buffer;
    }

    /// Takes out every entry that overlaps the guest bytes. The caller keeps the buffers until
    /// the GPU is done with them.
    void Evict(u64 guest, u64 size, std::vector<Buffer>& out) {
        auto entry = entries.lower_bound(guest);
        if (entry != entries.begin() &&
            std::prev(entry)->first + std::prev(entry)->second.size > guest) {
            --entry;
        }
        while (entry != entries.end() && entry->first < guest + size) {
            bytes -= entry->second.size;
            out.push_back(std::move(entry->second.buffer));
            entry = entries.erase(entry);
        }
    }

    /// Enters a buffer that was just filled with the span's bytes. Evict the span first.
    void Keep(u64 guest, u64 size, u64 pool, u64 stamp, Buffer buffer, u64 now) {
        bytes += size;
        entries.insert_or_assign(guest, Entry{std::move(buffer), size, pool, stamp, now});
    }

    /// Takes out the entries no replay has used since `before`.
    void EvictUnused(u64 before, std::vector<Buffer>& out) {
        for (auto entry = entries.begin(); entry != entries.end();) {
            if (entry->second.used >= before) {
                ++entry;
                continue;
            }
            bytes -= entry->second.size;
            out.push_back(std::move(entry->second.buffer));
            entry = entries.erase(entry);
        }
    }

    u64 Bytes() const {
        return bytes;
    }

    size_t Count() const {
        return entries.size();
    }

private:
    struct Entry {
        Buffer buffer;
        u64 size;
        u64 pool;
        u64 stamp;
        // The count of replays when a span last used the entry.
        u64 used;
    };

    std::map<u64, Entry> entries;
    u64 bytes{};
};

} // namespace Vulkan
