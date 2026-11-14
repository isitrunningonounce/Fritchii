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
#include <string_view>

#include "common/common_types.h"

namespace Core::BananzaNvn2Trace {

bool CountEnabled();
bool HleEnabled();

struct SdkLayout {
    std::array<u8, 32> build_id;
    size_t names;
    size_t name_offsets;
    size_t targets;
    size_t text_end;
    size_t entry_count;
    size_t bootstrap;
    size_t internal_bootstrap;
    size_t callback_return{};
};

const SdkLayout* FindSdk(const std::array<u8, 32>& build_id);

class Session {
public:
    static std::unique_ptr<Session> Create(const SdkLayout& layout, std::span<const u8> sdk,
                                           u64 base);

    std::optional<size_t> Find(u64 address) const;
    std::string_view Name(size_t index) const;
    size_t Size() const { return entry_count; }
    u64 Resolve(std::string_view name) const;
    void Count(size_t index, u64 thread, u64 arg1);
    void Present();
    void Finish();

private:
    enum class Kind { Other, Draw, Begin, End, Submit };
    static constexpr size_t MaxEntries = 726;
    struct Entry {
        u64 address{};
        std::string name;
        Kind kind{};
        std::atomic<u64> calls{};
        u64 reported_calls{};
    };

    void Report(bool final);

    std::array<Entry, MaxEntries> entries{};
    size_t entry_count{};
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
