// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>

#include "common/common_types.h"

namespace Core::BananzaNvn2Trace {

bool CountEnabled();

class Session {
public:
    static std::unique_ptr<Session> Create(std::span<const u8> sdk, u64 base);

    std::optional<size_t> Find(u64 address) const;
    void Count(size_t index, u64 thread, u64 arg1);
    void Present();
    void Finish();

private:
    enum class Kind { Other, Draw, Begin, End, Submit };
    static constexpr size_t EntryCount = 698;
    struct Entry {
        u64 address{};
        std::string name;
        Kind kind{};
        std::atomic<u64> calls{};
        u64 reported_calls{};
    };

    void Report(bool final);

    std::array<Entry, EntryCount> entries{};
    std::mutex report_mutex;
    std::set<u64> recording_threads;
    std::set<u64> reported_threads;
    std::atomic<u64> submitted_handles{};
    u64 reported_handles{};
    u64 presents{};
    u64 reported_presents{};
    std::chrono::steady_clock::time_point next_report =
        std::chrono::steady_clock::now() + std::chrono::seconds{10};
};

} // namespace Core::BananzaNvn2Trace
