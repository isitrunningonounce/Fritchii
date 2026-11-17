// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <utility>

#include "common/common_types.h"
#include "common/thread_worker.h"

namespace Core::BananzaNvn2Hle {

/// Runs the slow part of the game's submits (the scan of what a submission reads) and the
/// presents that follow them on one thread, in the order the game made the calls, so the
/// game's thread goes on with its next frame meanwhile.
class SubmitLane {
public:
    template <class Task>
    void Queue(Task task) {
        queued.fetch_add(1, std::memory_order_acq_rel);
        worker.QueueWork([this, task = std::move(task)]() mutable {
            task();
            {
                std::scoped_lock lock{mutex};
                done.fetch_add(1, std::memory_order_acq_rel);
            }
            finished.notify_all();
        });
    }

    /// Waits for everything queued before this call.
    void Wait() {
        const u64 target = queued.load(std::memory_order_acquire);
        if (done.load(std::memory_order_acquire) >= target) {
            return;
        }
        std::unique_lock lock{mutex};
        finished.wait(lock, [&] { return done.load(std::memory_order_acquire) >= target; });
    }

private:
    std::atomic<u64> queued{};
    std::atomic<u64> done{};
    std::mutex mutex;
    std::condition_variable finished;
    // Declared last: the thread stops before the counters go.
    Common::ThreadWorker worker{1, "NvnSubmit"};
};

} // namespace Core::BananzaNvn2Hle
