// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/lichii_telemetry.h"

#include <fmt/format.h>
#include "common/logging.h"
#include "core/file_sys/vfs/vfs.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <set>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#endif

namespace Core::Telemetry {

namespace {

const char* Path() {
    static const char* const path = [] {
        const char* value = std::getenv("LICHII_TELEMETRY");
        return value != nullptr && value[0] != '\0' ? value : nullptr;
    }();
    return path;
}

struct Window {
    std::int64_t frames = 0;
    std::int64_t frame_us = 0;
    std::int64_t max_frame_us = 0;
    std::int64_t over_20ms = 0;
    std::int64_t over_34ms = 0;
    std::int64_t fence_us = 0;
    std::int64_t composite_us = 0;
    std::int64_t pace_us = 0;
};

std::mutex window_mutex;
Window window;
std::atomic<std::int64_t> cpu_read_flushes{};
std::atomic<std::int64_t> cpu_read_wait_us{};
std::atomic<std::int64_t> trap_lock_wait_us{};
std::atomic<std::int64_t> guest_read_bytes{};
std::atomic<std::int64_t> guest_reads{};

/// Guest reads per file or storage over the last 10 s, logged as a summary (whole-run repeats).
struct ReadSource {
    std::int64_t bytes = 0;
    std::int64_t reads = 0;
    std::int64_t repeated_reads = 0;
    std::int64_t repeated_bytes = 0;
    std::map<std::pair<std::int64_t, std::int64_t>, std::int64_t> seen; ///< (offset, size) -> reads
    /// Repeated (offset, size) ranges in this window and how often each was read again.
    std::map<std::pair<std::int64_t, std::int64_t>, std::int64_t> window_repeats;
    std::weak_ptr<FileSys::VfsFile> storage;
};
std::mutex read_mutex;
std::map<std::string, ReadSource> read_sources;

/// Files of a RomFS by their offset in it, to name the ranges a game reads again.
struct RomfsFile {
    std::int64_t offset;
    std::int64_t size;
    std::string path;
};
std::vector<RomfsFile> romfs_list; ///< parsed once, from the first RomFS storage read
bool romfs_parsed = false;

/// Reads the RomFS file table directly (offsets of the data in the storage, paths from the
/// directory table). No RTTI here (/GR-), so the VFS tree from ExtractRomFS can't be inspected.
void CollectFiles(const std::shared_ptr<FileSys::VfsFile>& storage, std::vector<RomfsFile>& out) {
    struct Header {
        std::uint64_t header_size;
        std::uint64_t dir_hash_offset, dir_hash_size, dir_meta_offset, dir_meta_size;
        std::uint64_t file_hash_offset, file_hash_size, file_meta_offset, file_meta_size;
        std::uint64_t data_offset;
    } header{};
    static_assert(sizeof(Header) == 0x50);
    if (storage->ReadBytes(&header, sizeof(header), 0) != sizeof(header) ||
        header.header_size != sizeof(header) || header.dir_meta_size > (64u << 20) ||
        header.file_meta_size > (256u << 20)) {
        return;
    }
    const std::vector<std::uint8_t> dirs = storage->ReadBytes(header.dir_meta_size,
                                                              header.dir_meta_offset);
    const std::vector<std::uint8_t> files = storage->ReadBytes(header.file_meta_size,
                                                               header.file_meta_offset);
    auto u32_at = [](const std::vector<std::uint8_t>& table, size_t offset) {
        std::uint32_t value = 0;
        if (offset + 4 <= table.size()) {
            std::memcpy(&value, table.data() + offset, 4);
        }
        return value;
    };
    // Directory path by its entry offset: parent chain, entry layout parent, sibling, child_dir,
    // child_file, hash, name_length, name.
    std::map<std::uint32_t, std::string> dir_paths;
    std::function<std::string(std::uint32_t, int)> dir_path = [&](std::uint32_t entry,
                                                                 int depth) -> std::string {
        if (entry == 0 || depth > 64 || entry + 0x18 > dirs.size()) {
            return "";
        }
        if (const auto it = dir_paths.find(entry); it != dir_paths.end()) {
            return it->second;
        }
        const std::uint32_t parent = u32_at(dirs, entry);
        const std::uint32_t length = u32_at(dirs, entry + 0x14);
        std::string name(reinterpret_cast<const char*>(dirs.data() + entry + 0x18),
                         std::min<size_t>(length, dirs.size() - entry - 0x18));
        std::string path = dir_path(parent, depth + 1) + "/" + name;
        dir_paths.emplace(entry, path);
        return path;
    };
    // File entries: parent, sibling, offset (u64), size (u64), hash, name_length, name (4-aligned).
    for (size_t entry = 0; entry + 0x20 <= files.size();) {
        const std::uint32_t parent = u32_at(files, entry);
        std::uint64_t offset = 0, size = 0;
        std::memcpy(&offset, files.data() + entry + 8, 8);
        std::memcpy(&size, files.data() + entry + 16, 8);
        const std::uint32_t length = u32_at(files, entry + 0x1c);
        if (entry + 0x20 + length > files.size()) {
            break;
        }
        std::string name(reinterpret_cast<const char*>(files.data() + entry + 0x20), length);
        out.push_back({static_cast<std::int64_t>(offset + header.data_offset),
                       static_cast<std::int64_t>(size), dir_path(parent, 0) + "/" + name});
        entry += 0x20 + ((length + 3) & ~size_t{3});
    }
}

std::string NameRange(std::int64_t offset) {
    const auto& files = romfs_list;
    if (files.empty()) {
        return "?";
    }
    auto it = std::upper_bound(files.begin(), files.end(), offset,
                               [](std::int64_t value, const RomfsFile& file) {
                                   return value < file.offset;
                               });
    if (it == files.begin()) {
        return "?";
    }
    --it;
    if (offset >= it->offset + it->size) {
        return "? (between files)";
    }
    return fmt::format("{} +{:#x} (file {:.1f} MiB)", it->path, offset - it->offset,
                       it->size / 1048576.0);
}

void LogReadSummary() {
    std::vector<std::pair<std::int64_t, std::string>> ranked;
    std::vector<std::string> repeats;
    {
        std::scoped_lock lock{read_mutex};
        for (auto& [name, source] : read_sources) {
            if (source.reads != 0) {
                ranked.emplace_back(source.bytes, fmt::format(
                    "{} {:.1f} MiB in {} reads ({} reads / {:.1f} MiB of ranges read before)",
                    name, source.bytes / 1048576.0, source.reads, source.repeated_reads,
                    source.repeated_bytes / 1048576.0));
            }
            // The ranges read again most (by bytes), named by RomFS file.
            std::vector<std::pair<std::int64_t, std::pair<std::int64_t, std::int64_t>>> top;
            for (const auto& [range, count] : source.window_repeats) {
                top.emplace_back(count * range.second, range);
            }
            std::sort(top.rbegin(), top.rend());
            for (size_t i = 0; i < top.size() && i < 12; ++i) {
                const auto [offset, size] = top[i].second;
                repeats.push_back(fmt::format("{} @{:#x} size {:#x} read again {}x: {}", name,
                                              offset, size, top[i].first / size,
                                              NameRange(offset)));
            }
            source.window_repeats.clear();
            source.bytes = source.reads = source.repeated_reads = source.repeated_bytes = 0;
        }
    }
    std::sort(ranked.rbegin(), ranked.rend());
    for (size_t i = 0; i < ranked.size() && i < 8; ++i) {
        LOG_INFO(Core, "LICHII_TELEMETRY guest reads (10 s): {}", ranked[i].second);
    }
    for (const std::string& line : repeats) {
        LOG_INFO(Core, "LICHII_TELEMETRY read again: {}", line);
    }
}
std::once_flag start_flag;

#ifdef _WIN32
// Minimal NVML, loaded at run time (nvml.dll ships with the NVIDIA driver).
struct NvmlUtilization {
    unsigned int gpu;
    unsigned int memory;
};
struct NvmlMemory {
    unsigned long long total, free, used;
};
using NvmlDevice = void*;
struct Nvml {
    int (*init)() = nullptr;
    int (*handle_by_index)(unsigned int, NvmlDevice*) = nullptr;
    int (*utilization)(NvmlDevice, NvmlUtilization*) = nullptr;
    int (*clock)(NvmlDevice, int, unsigned int*) = nullptr;
    int (*memory)(NvmlDevice, NvmlMemory*) = nullptr;
    NvmlDevice device = nullptr;

    bool Load() {
        HMODULE module = LoadLibraryA("nvml.dll");
        if (module == nullptr) {
            return false;
        }
        const auto proc = [module](const char* name) {
            return reinterpret_cast<void*>(GetProcAddress(module, name));
        };
        init = reinterpret_cast<int (*)()>(proc("nvmlInit_v2"));
        handle_by_index = reinterpret_cast<int (*)(unsigned int, NvmlDevice*)>(
            proc("nvmlDeviceGetHandleByIndex_v2"));
        utilization = reinterpret_cast<int (*)(NvmlDevice, NvmlUtilization*)>(
            proc("nvmlDeviceGetUtilizationRates"));
        clock = reinterpret_cast<int (*)(NvmlDevice, int, unsigned int*)>(
            proc("nvmlDeviceGetClockInfo"));
        memory = reinterpret_cast<int (*)(NvmlDevice, NvmlMemory*)>(
            proc("nvmlDeviceGetMemoryInfo"));
        if (!init || !handle_by_index || !utilization || init() != 0) {
            return false;
        }
        return handle_by_index(0, &device) == 0;
    }
};

std::uint64_t ToU64(FILETIME time) {
    return (static_cast<std::uint64_t>(time.dwHighDateTime) << 32) | time.dwLowDateTime;
}

/// CPU time (100 ns units) of each named thread of this process, summed per column.
constexpr std::array<const char*, 11> thread_columns{
    "CPUCore_0", "CPUCore_1", "CPUCore_2", "CPUCore_3", "CPUCore_4", "CPUCore_5",
    "GPU",       "GPUBack",   "VulkanWorker", "HostTiming", "other",
};

std::map<DWORD, std::pair<size_t, std::uint64_t>> SampleThreads() {
    std::map<DWORD, std::pair<size_t, std::uint64_t>> result;
    const DWORD pid = GetCurrentProcessId();
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    THREADENTRY32 entry{sizeof(entry)};
    for (BOOL ok = Thread32First(snap, &entry); ok; ok = Thread32Next(snap, &entry)) {
        if (entry.th32OwnerProcessID != pid) {
            continue;
        }
        HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
        if (thread == nullptr) {
            continue;
        }
        FILETIME created, exited, kernel, user;
        if (GetThreadTimes(thread, &created, &exited, &kernel, &user)) {
            size_t column = thread_columns.size() - 1;
            PWSTR description = nullptr;
            if (SUCCEEDED(GetThreadDescription(thread, &description)) && description != nullptr) {
                char name[64]{};
                for (int i = 0; i < 63 && description[i] != 0; ++i) {
                    name[i] = description[i] < 128 ? static_cast<char>(description[i]) : '?';
                }
                LocalFree(description);
                for (size_t i = 0; i + 1 < thread_columns.size(); ++i) {
                    if (std::strcmp(name, thread_columns[i]) == 0) {
                        column = i;
                        break;
                    }
                }
            }
            result[entry.th32ThreadID] = {column, ToU64(kernel) + ToU64(user)};
        }
        CloseHandle(thread);
    }
    CloseHandle(snap);
    return result;
}
#endif

void Run() {
    std::FILE* file = std::fopen(Path(), "w");
    if (file == nullptr) {
        return;
    }
    std::fprintf(file, "t_s,fps,frame_avg_ms,frame_max_ms,frames_over_20ms,frames_over_34ms,"
                       "fence_ms,composite_ms,pace_ms,cpu_read_flushes,guest_read_mb,guest_reads,cpu_read_wait_ms,trap_lock_wait_ms");
#ifdef _WIN32
    for (const char* column : thread_columns) {
        std::fprintf(file, ",%s", column);
    }
    std::fprintf(file, ",gpu_util,gpu_mhz,vram_mb,io_read_mb,io_reads");
    Nvml nvml;
    const bool has_nvml = nvml.Load();
    auto previous = SampleThreads();
    IO_COUNTERS previous_io{};
    GetProcessIoCounters(GetCurrentProcess(), &previous_io);
#endif
    std::fprintf(file, "\n");
    std::fflush(file);
    const auto start = std::chrono::steady_clock::now();
    auto last = start;
    unsigned ticks = 0;
    while (true) {
        std::this_thread::sleep_until(last + std::chrono::seconds{1});
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - last).count();
        last = now;
        Window taken;
        {
            std::scoped_lock lock{window_mutex};
            taken = std::exchange(window, {});
        }
        const double frames = static_cast<double>(std::max<std::int64_t>(taken.frames, 1));
        std::fprintf(file, "%.1f,%.1f,%.2f,%.2f,%lld,%lld,%.2f,%.2f,%.2f,%lld,%.2f,%lld,%.1f,%.1f",
                     std::chrono::duration<double>(now - start).count(), taken.frames / seconds,
                     taken.frame_us / frames / 1000.0, taken.max_frame_us / 1000.0,
                     static_cast<long long>(taken.over_20ms), static_cast<long long>(taken.over_34ms),
                     taken.fence_us / frames / 1000.0, taken.composite_us / frames / 1000.0,
                     taken.pace_us / frames / 1000.0,
                     static_cast<long long>(cpu_read_flushes.exchange(0)),
                     guest_read_bytes.exchange(0) / 1048576.0,
                     static_cast<long long>(guest_reads.exchange(0)),
                     cpu_read_wait_us.exchange(0) / 1000.0, trap_lock_wait_us.exchange(0) / 1000.0);
#ifdef _WIN32
        auto current = SampleThreads();
        std::array<double, thread_columns.size()> busy{};
        for (const auto& [id, sample] : current) {
            const auto it = previous.find(id);
            const std::uint64_t before = it != previous.end() ? it->second.second : 0;
            busy[sample.first] += (sample.second - before) / (seconds * 1e5);
        }
        previous = std::move(current);
        for (double percent : busy) {
            std::fprintf(file, ",%.0f", percent);
        }
        unsigned int util = 0, mhz = 0;
        unsigned long long vram = 0;
        if (has_nvml) {
            NvmlUtilization rates{};
            if (nvml.utilization(nvml.device, &rates) == 0) {
                util = rates.gpu;
            }
            if (nvml.clock != nullptr) {
                nvml.clock(nvml.device, 0, &mhz); // NVML_CLOCK_GRAPHICS
            }
            NvmlMemory memory{};
            if (nvml.memory != nullptr && nvml.memory(nvml.device, &memory) == 0) {
                vram = memory.used >> 20;
            }
        }
        IO_COUNTERS io{};
        GetProcessIoCounters(GetCurrentProcess(), &io);
        std::fprintf(file, ",%u,%u,%llu,%.1f,%llu", util, mhz, vram,
                     (io.ReadTransferCount - previous_io.ReadTransferCount) / 1048576.0,
                     io.ReadOperationCount - previous_io.ReadOperationCount);
        previous_io = io;
#endif
        std::fprintf(file, "\n");
        std::fflush(file);
        if (++ticks % 10 == 0) {
            LogReadSummary();
        }
    }
}

void EnsureStarted() {
    std::call_once(start_flag, [] { std::thread(Run).detach(); });
}

} // namespace

bool Enabled() {
    return Path() != nullptr;
}

void NotePresent(std::int64_t frame_us, std::int64_t fence_us, std::int64_t composite_us,
                 std::int64_t pace_us) {
    if (!Enabled()) {
        return;
    }
    EnsureStarted();
    std::scoped_lock lock{window_mutex};
    ++window.frames;
    window.frame_us += frame_us;
    window.max_frame_us = std::max(window.max_frame_us, frame_us);
    window.over_20ms += frame_us > 20'000 ? 1 : 0;
    window.over_34ms += frame_us > 34'000 ? 1 : 0;
    window.fence_us += fence_us;
    window.composite_us += composite_us;
    window.pace_us += pace_us;
}

void NoteCpuReadFlush() {
    if (Enabled()) {
        cpu_read_flushes.fetch_add(1, std::memory_order_relaxed);
    }
}

std::int64_t NoteGuestRead(std::int64_t bytes, const std::string& source, std::int64_t offset,
                           const std::shared_ptr<FileSys::VfsFile>& storage) {
    if (!Enabled()) {
        return 0;
    }
    guest_read_bytes.fetch_add(bytes, std::memory_order_relaxed);
    guest_reads.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{read_mutex};
    ReadSource& entry = read_sources[source];
    if (storage && !romfs_parsed && source == "romfs") {
        romfs_parsed = true;
        CollectFiles(storage, romfs_list);
        std::sort(romfs_list.begin(), romfs_list.end(),
                  [](const RomfsFile& a, const RomfsFile& b) { return a.offset < b.offset; });
        LOG_INFO(Core, "LICHII_TELEMETRY: {} RomFS files indexed", romfs_list.size());
    }
    entry.bytes += bytes;
    entry.reads += 1;
    // Same (offset, size) read before in this run: data the game loads again.
    const std::int64_t before = entry.seen[{offset, bytes}]++;
    if (before != 0) {
        entry.repeated_reads += 1;
        entry.repeated_bytes += bytes;
        entry.window_repeats[{offset, bytes}] += 1;
    }
    return before;
}

} // namespace Core::Telemetry

namespace Core::Telemetry {

namespace {
std::atomic<int> ipc_backtraces{};
std::atomic<int> traced_reads{};
} // namespace

void NoteTracedRead(std::int64_t offset) {
    static const std::int64_t target = [] {
        const char* value = std::getenv("LICHII_TRACE_READ");
        return value != nullptr ? static_cast<std::int64_t>(std::strtoull(value, nullptr, 16))
                                : std::int64_t{-1};
    }();
    if (offset == target && traced_reads.fetch_add(1) < 4) {
        LOG_INFO(Core, "LICHII_TRACE_READ: read of {:#x}, logging the next IPC call stacks", offset);
        ipc_backtraces.store(16);
    }
}

bool TakeIpcBacktrace() {
    int left = ipc_backtraces.load(std::memory_order_relaxed);
    while (left > 0) {
        if (ipc_backtraces.compare_exchange_weak(left, left - 1)) {
            return true;
        }
    }
    return false;
}

} // namespace Core::Telemetry

namespace Core::Telemetry {
void NoteCpuReadWait(std::int64_t microseconds) {
    if (Enabled()) {
        cpu_read_wait_us.fetch_add(microseconds, std::memory_order_relaxed);
    }
}
} // namespace Core::Telemetry

namespace Core::Telemetry {
namespace {
struct ReadPartStats {
    std::int64_t count = 0;
    std::int64_t queue_us = 0;
    std::int64_t flush_us = 0;
    std::int64_t max_us = 0;
    std::uint64_t size = 0;
};
std::mutex read_parts_mutex;
std::map<std::uint64_t, ReadPartStats> read_parts; ///< by 4 KiB page of the read
std::chrono::steady_clock::time_point read_parts_last = std::chrono::steady_clock::now();
} // namespace

void NoteCpuReadParts(std::uint64_t address, std::uint64_t size, std::int64_t queue_us,
                      std::int64_t flush_us) {
    if (!Enabled()) {
        return;
    }
    std::vector<std::string> lines;
    {
        std::scoped_lock lock{read_parts_mutex};
        ReadPartStats& stats = read_parts[address & ~std::uint64_t{0xfff}];
        ++stats.count;
        stats.queue_us += queue_us;
        stats.flush_us += flush_us;
        stats.max_us = std::max(stats.max_us, queue_us + flush_us);
        stats.size = size;
        const auto now = std::chrono::steady_clock::now();
        if (now - read_parts_last < std::chrono::seconds{10}) {
            return;
        }
        read_parts_last = now;
        std::vector<std::pair<std::int64_t, std::uint64_t>> ranked;
        std::int64_t total_queue = 0, total_flush = 0, total_count = 0;
        for (const auto& [page, s] : read_parts) {
            ranked.emplace_back(s.queue_us + s.flush_us, page);
            total_queue += s.queue_us;
            total_flush += s.flush_us;
            total_count += s.count;
        }
        std::sort(ranked.rbegin(), ranked.rend());
        lines.push_back(fmt::format("{} flushes, {:.0f} ms queued behind the GPU thread, {:.0f} ms "
                                    "flushing (10 s)",
                                    total_count, total_queue / 1000.0, total_flush / 1000.0));
        for (size_t i = 0; i < ranked.size() && i < 10; ++i) {
            const auto& s = read_parts[ranked[i].second];
            lines.push_back(fmt::format("  page {:#x}: {} reads, flush area {:#x}, queue {:.2f} ms "
                                        "+ flush {:.2f} ms each (max {:.1f} ms)",
                                        ranked[i].second, s.count, s.size,
                                        s.queue_us / 1000.0 / s.count,
                                        s.flush_us / 1000.0 / s.count, s.max_us / 1000.0));
        }
        read_parts.clear();
    }
    for (const auto& line : lines) {
        LOG_INFO(Core, "LICHII_TELEMETRY readbacks: {}", line);
    }
}
} // namespace Core::Telemetry

namespace Core::Telemetry {
namespace {
struct GuestThreadStats {
    std::int64_t run_ns = 0;
    std::int64_t read_wait_us = 0;
    int priority = 0;
    std::array<std::int64_t, 8> core_ns{};
    std::map<std::uint32_t, std::int64_t> blocked_ns; ///< off-core time by last svc (0xff: none)
};
std::map<std::uint64_t, std::int64_t> switched_out_at;
std::map<std::uint64_t, std::uint32_t> last_svc;
std::mutex guest_threads_mutex;
std::map<std::uint64_t, GuestThreadStats> guest_threads;
std::array<std::int64_t, 8> core_last_switch{};
std::chrono::steady_clock::time_point guest_threads_last = std::chrono::steady_clock::now();

std::int64_t NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void MaybeLogGuestThreads(std::unique_lock<std::mutex>& lock) {
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - guest_threads_last).count();
    if (seconds < 10.0) {
        return;
    }
    guest_threads_last = now;
    std::vector<std::pair<std::int64_t, std::string>> ranked;
    for (const auto& [id, stats] : guest_threads) {
        int main_core = 0;
        for (int c = 1; c < 8; ++c) {
            if (stats.core_ns[c] > stats.core_ns[main_core]) {
                main_core = c;
            }
        }
        std::vector<std::pair<std::int64_t, std::uint32_t>> blocked;
        for (const auto& [svc, ns] : stats.blocked_ns) {
            blocked.emplace_back(ns, svc);
        }
        std::sort(blocked.rbegin(), blocked.rend());
        std::string waits;
        for (size_t b = 0; b < blocked.size() && b < 4; ++b) {
            waits += fmt::format(" svc{:#x}={:.0f}", blocked[b].second,
                                 blocked[b].first / 1e6 / seconds);
        }
        ranked.emplace_back(stats.run_ns, fmt::format(
            "t{} prio {} core {}: runs {:.0f} ms/s (readback waits {:.0f}), off-core ms/s by last "
            "svc:{}",
            id, stats.priority, main_core, stats.run_ns / 1e6 / seconds,
            stats.read_wait_us / 1e3 / seconds, waits));
    }
    guest_threads.clear();
    lock.unlock();
    std::sort(ranked.rbegin(), ranked.rend());
    for (size_t i = 0; i < ranked.size() && i < 40; ++i) {
        LOG_INFO(Core, "LICHII_TELEMETRY guest thread {}", ranked[i].second);
    }
}
} // namespace

void NoteThreadSwitch(int core, std::uint64_t thread_id, int priority, std::uint64_t next_id) {
    if (core < 0 || core >= 8) {
        return;
    }
    const std::int64_t now = NowNs();
    std::unique_lock lock{guest_threads_mutex};
    const std::int64_t last = std::exchange(core_last_switch[core], now);
    if (last != 0) {
        auto& stats = guest_threads[thread_id];
        stats.run_ns += now - last;
        stats.core_ns[core] += now - last;
        stats.priority = priority;
    }
    switched_out_at[thread_id] = now;
    // The next thread was off a core since it was switched out: blocked in (or preempted after)
    // its last system call.
    if (const auto it = switched_out_at.find(next_id); it != switched_out_at.end()) {
        const std::int64_t off = now - it->second;
        switched_out_at.erase(it);
        const auto svc = last_svc.find(next_id);
        guest_threads[next_id].blocked_ns[svc != last_svc.end() ? svc->second : 0xff] += off;
    }
    MaybeLogGuestThreads(lock);
}

void NoteThreadReadWait(std::uint64_t thread_id, std::int64_t microseconds) {
    std::scoped_lock lock{guest_threads_mutex};
    guest_threads[thread_id].read_wait_us += microseconds;
}
} // namespace Core::Telemetry

namespace Core::Telemetry {
void NoteSvc(std::uint64_t thread_id, std::uint32_t svc) {
    if (!Enabled()) {
        return;
    }
    std::scoped_lock lock{guest_threads_mutex};
    last_svc[thread_id] = svc;
}
} // namespace Core::Telemetry

namespace Core::Telemetry {
void NoteTrapLockWait(std::int64_t microseconds) {
    if (Enabled()) {
        trap_lock_wait_us.fetch_add(microseconds, std::memory_order_relaxed);
    }
}
} // namespace Core::Telemetry
