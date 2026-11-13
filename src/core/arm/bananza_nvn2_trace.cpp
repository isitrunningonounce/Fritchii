// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string_view>

#include <fmt/format.h>

#include "common/logging.h"
#include "common/swap.h"
#include "core/arm/bananza_nvn2_trace.h"

namespace Core::BananzaNvn2Trace {
namespace {

// The base and 3.2.0 update packages have separate SDK procedure tables.
constexpr std::array SupportedSdks{
    SdkLayout{
        .build_id = {0xB7, 0xFF, 0xF6, 0x14, 0xF2, 0xBC, 0xDD, 0xD3, 0x8E, 0x80,
                     0xD5, 0x71, 0x61, 0x9C, 0x73, 0xF3, 0x00, 0xF4, 0x8E, 0x1A},
        .names = 0xB74C26,
        .name_offsets = 0xB79B90,
        .targets = 0xD95168,
        .text_end = 0x776A70,
        .entry_count = 698,
    },
    SdkLayout{
        .build_id = {0x18, 0x26, 0x5D, 0xAE, 0xA2, 0x10, 0x5B, 0x31, 0xF2, 0x53,
                     0x46, 0xAD, 0xFB, 0xBD, 0x09, 0xB5, 0xED, 0xF1, 0x20, 0x7F},
        .names = 0xB9A5EA,
        .name_offsets = 0xB9F94C,
        .targets = 0xE056E0,
        .text_end = 0x7B65A0,
        .entry_count = 726,
    },
};

} // Anonymous namespace

bool CountEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_NVN_HLE");
        return value && std::string_view{value} == "count";
    }();
    return enabled;
}

const SdkLayout* FindSdk(const std::array<u8, 32>& build_id) {
    const auto it = std::ranges::find(SupportedSdks, build_id, &SdkLayout::build_id);
    return it == SupportedSdks.end() ? nullptr : &*it;
}

std::unique_ptr<Session> Session::Create(const SdkLayout& layout, std::span<const u8> sdk,
                                         u64 base) {
    if (layout.entry_count == 0 || layout.entry_count > MaxEntries ||
        sdk.size() < layout.targets + layout.entry_count * sizeof(u64_le) ||
        base + layout.text_end < base) {
        return {};
    }
    std::array<size_t, MaxEntries> order{};
    std::array<u64, MaxEntries> targets{};
    for (size_t i = 0; i < layout.entry_count; ++i) {
        u64_le target{};
        std::memcpy(&target, sdk.data() + layout.targets + i * sizeof(target), sizeof(target));
        if (target == 0 || target >= layout.text_end || target % sizeof(u32) != 0) {
            return {};
        }
        targets[i] = target;
        order[i] = i;
    }
    std::ranges::sort(std::span{order}.first(layout.entry_count), {},
                      [&](size_t index) { return targets[index]; });

    auto session = std::make_unique<Session>();
    session->entry_count = layout.entry_count;
    for (size_t i = 0; i < layout.entry_count; ++i) {
        const size_t index = order[i];
        if (i != 0 && targets[index] == targets[order[i - 1]]) {
            return {};
        }
        u32_le offset{};
        std::memcpy(&offset, sdk.data() + layout.name_offsets + index * sizeof(offset),
                    sizeof(offset));
        if (offset >= layout.name_offsets - layout.names) {
            return {};
        }
        const std::string_view name_storage{
            reinterpret_cast<const char*>(sdk.data() + layout.names + offset),
            layout.name_offsets - layout.names - offset};
        const size_t length = name_storage.find('\0');
        if (length == 0 || length == std::string_view::npos || length > 96) {
            return {};
        }
        const std::string_view name = name_storage.substr(0, length);
        auto& entry = session->entries[i];
        entry.address = base + targets[index];
        entry.name = "nvn";
        entry.name.append(name);
        if (name == "CommandBufferBeginRecording") {
            entry.kind = Kind::Begin;
        } else if (name == "CommandBufferEndRecording") {
            entry.kind = Kind::End;
        } else if (name == "QueueSubmitCommands") {
            entry.kind = Kind::Submit;
        } else if (name.starts_with("CommandBufferDraw") ||
                   name.starts_with("CommandBufferMultiDraw")) {
            entry.kind = Kind::Draw;
        }
    }
    return session;
}

std::optional<size_t> Session::Find(u64 address) const {
    const auto active_entries = std::span{entries}.first(entry_count);
    if (address < active_entries.front().address || address > active_entries.back().address) {
        return std::nullopt;
    }
    const auto it = std::ranges::lower_bound(active_entries, address, {}, &Entry::address);
    if (it == active_entries.end() || it->address != address) {
        return std::nullopt;
    }
    return static_cast<size_t>(it - active_entries.begin());
}

void Session::Count(size_t index, u64 thread, u64 arg1) {
    auto& entry = entries[index];
    entry.calls.fetch_add(1, std::memory_order_relaxed);
    if (entry.kind == Kind::Submit) {
        const s32 handles = static_cast<s32>(arg1);
        if (handles > 0) {
            submitted_handles.fetch_add(handles, std::memory_order_relaxed);
        }
    } else if (entry.kind == Kind::Begin) {
        std::scoped_lock lock{report_mutex};
        recording_threads.insert(thread);
    }
}

void Session::Present() {
    std::scoped_lock lock{report_mutex};
    ++presents;
    const auto now = std::chrono::steady_clock::now();
    if (now >= next_report) {
        Report(false);
        next_report = now + std::chrono::seconds{10};
    }
}

void Session::Finish() {
    std::scoped_lock lock{report_mutex};
    Report(true);
}

void Session::Report(bool final) {
    std::array<u64, MaxEntries> calls{};
    u64 total{};
    u64 draws{};
    u64 recorded{};
    u64 submits{};
    for (size_t i = 0; i < entry_count; ++i) {
        auto& entry = entries[i];
        const u64 current = entry.calls.load(std::memory_order_relaxed);
        calls[i] = final ? current : current - entry.reported_calls;
        entry.reported_calls = current;
        total += calls[i];
        if (entry.kind == Kind::Draw) {
            draws += calls[i];
        } else if (entry.kind == Kind::End) {
            recorded += calls[i];
        } else if (entry.kind == Kind::Submit) {
            submits += calls[i];
        }
    }

    const u64 frames = final ? presents : presents - reported_presents;
    const u64 handles = submitted_handles.load(std::memory_order_relaxed);
    const u64 submitted = final ? handles : handles - reported_handles;
    reported_presents = presents;
    reported_handles = handles;
    reported_threads.insert(recording_threads.begin(), recording_threads.end());
    const auto& threads = final ? reported_threads : recording_threads;
    const auto per_frame = [&](u64 count) {
        return frames ? static_cast<double>(count) / frames : 0.0;
    };
    LOG_INFO(Core_ARM,
             "NVN 2 count {}: {} presents, {} calls, {} draw commands, {} end-recording calls "
             "({:.2f}/frame), {} submits, {} submitted handles ({:.2f}/frame), "
             "recording guest threads [{}]",
             final ? "session" : "window", frames, total, draws, recorded, per_frame(recorded),
             submits, submitted, per_frame(submitted), fmt::join(threads, ", "));
    recording_threads.clear();
    std::string line;
    for (size_t i = 0; i < entry_count; ++i) {
        if (calls[i] == 0) {
            continue;
        }
        fmt::format_to(std::back_inserter(line), "{}={} ({:.3f}/frame, {:.3f}/draw command); ",
                       entries[i].name, calls[i], per_frame(calls[i]),
                       draws ? static_cast<double>(calls[i]) / draws : 0.0);
        if (line.size() >= 768) {
            LOG_INFO(Core_ARM, "NVN 2 count {}: {}", final ? "session" : "window", line);
            line.clear();
        }
    }
    if (!line.empty()) {
        LOG_INFO(Core_ARM, "NVN 2 count {}: {}", final ? "session" : "window", line);
    }
}

} // namespace Core::BananzaNvn2Trace
