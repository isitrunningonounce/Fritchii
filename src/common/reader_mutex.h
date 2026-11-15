// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <mutex>
#include <thread>

#include "common/common_types.h"

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Common {

/// A shared mutex for many readers on several threads and rare, short writers.
/// A reader counts itself in a slot of its own thread, so readers do not all write to one cache
/// line, and nobody sleeps: a writer waits for the slots to empty, readers wait for the writer.
/// The thread that locked unlocks. As with std::shared_mutex, a thread must not take the shared
/// lock again while it holds it.
class ReaderMutex {
public:
    void lock_shared() noexcept {
        auto& count = slots[Slot()].count;
        for (;;) {
            count.fetch_add(1, std::memory_order_seq_cst);
            if (!writer.load(std::memory_order_seq_cst)) [[likely]] {
                return;
            }
            count.fetch_sub(1, std::memory_order_seq_cst);
            Wait([this] { return !writer.load(std::memory_order_relaxed); });
        }
    }

    void unlock_shared() noexcept {
        slots[Slot()].count.fetch_sub(1, std::memory_order_release);
    }

    void lock() noexcept {
        writers.lock();
        writer.store(true, std::memory_order_seq_cst);
        for (auto& slot : slots) {
            Wait([&slot] { return slot.count.load(std::memory_order_seq_cst) == 0; });
        }
    }

    void unlock() noexcept {
        writer.store(false, std::memory_order_release);
        writers.unlock();
    }

private:
    static constexpr size_t Slots = 64;

    struct alignas(64) SlotData {
        std::atomic<u32> count{};
    };

    static size_t Slot() noexcept {
        static std::atomic<size_t> next{};
        thread_local const size_t slot = next.fetch_add(1, std::memory_order_relaxed) % Slots;
        return slot;
    }

    template <typename Done>
    static void Wait(Done&& done) noexcept {
        for (u32 spins = 0; !done(); ++spins) {
            if (spins < 64) {
#if defined(_M_X64) || defined(__x86_64__)
                _mm_pause();
#endif
            } else {
                std::this_thread::yield();
            }
        }
    }

    std::array<SlotData, Slots> slots{};
    alignas(64) std::atomic<bool> writer{};
    std::mutex writers;
};

} // namespace Common
