// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "common/thread_worker.h"

namespace Vulkan {

/// Builds things on worker threads and hands each result back to the thread that asked.
/// Queue and Collect are called by that one thread; the jobs must not touch what it changes.
template <class Key, class Result>
class BananzaNvn2AsyncBuilder {
public:
    explicit BananzaNvn2AsyncBuilder(size_t threads, std::string name)
        : workers{threads, std::move(name)} {}

    /// Runs job() on a worker; its result waits for Collect.
    template <class Job>
    void Queue(Key key, Job job) {
        ++pending;
        workers.QueueWork([this, key = std::move(key), job = std::move(job)]() mutable {
            Result result = job();
            std::scoped_lock lock{mutex};
            finished.emplace_back(std::move(key), std::move(result));
        });
    }

    /// Calls take(key, result) for every job finished since the last call.
    template <class Take>
    void Collect(Take&& take) {
        if (!pending) {
            return;
        }
        std::vector<std::pair<Key, Result>> done;
        {
            std::scoped_lock lock{mutex};
            done.swap(finished);
        }
        pending -= done.size();
        for (auto& [key, result] : done) {
            take(key, std::move(result));
        }
    }

    [[nodiscard]] size_t Pending() const {
        return pending;
    }

private:
    std::mutex mutex;
    std::vector<std::pair<Key, Result>> finished;
    // Jobs queued and not yet collected.
    size_t pending{};
    // Declared last: the threads stop before the results go.
    Common::ThreadWorker workers;
};

} // namespace Vulkan
