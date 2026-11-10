// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace FileSys {
class VfsFile;
}

namespace Core::Telemetry {

/// LICHII_TELEMETRY=<csv path>: once a second, one CSV line with the game's frame rate and frame
/// times (from its presents), where the present thread waited, the CPU use of the emulator's main
/// threads and the host GPU's load (NVML). For measuring changes without reading the overlay.
[[nodiscard]] bool Enabled();

/// One game present: microseconds since the previous one, and how long the present waited for
/// the GPU fence, composited and was paced to the vblank.
void NotePresent(std::int64_t frame_us, std::int64_t fence_us, std::int64_t composite_us,
                 std::int64_t pace_us);

/// A guest CPU read that waited for the GPU thread (reactive flush).
void NoteCpuReadFlush();
/// How long the guest CPU thread waited for that flush (microseconds).
void NoteCpuReadWait(std::int64_t microseconds);
/// A CPU write trap waited this long for the buffer cache lock (held by the GPU thread).
void NoteTrapLockWait(std::int64_t microseconds);

/// The guest scheduler switched a thread out of `core`: the time since the core's previous switch
/// is that thread's run time. Summarized per thread every 10 s with its readback waits.
void NoteThreadSwitch(int core, std::uint64_t thread_id, int priority, std::uint64_t next_id);
/// A guest thread made a system call (its blocked time is attributed to the last one).
void NoteSvc(std::uint64_t thread_id, std::uint32_t svc);
/// A readback wait of the guest thread `thread_id` (microseconds).
void NoteThreadReadWait(std::uint64_t thread_id, std::int64_t microseconds);
/// Where a flush's wait went: queued until the GPU thread reached it, then the flush itself
/// (downloads, host GPU idle). Summarized every 10 s by the page read.
void NoteCpuReadParts(std::uint64_t address, std::uint64_t size, std::int64_t queue_us,
                      std::int64_t flush_us);

/// A guest file or storage read through the FS service, of `bytes` requested.
/// `storage`: the storage read (IStorage), so repeated ranges of a RomFS can be named by file.
/// Returns how often this (offset, size) was read before in this run.
[[maybe_unused]] std::int64_t NoteGuestRead(std::int64_t bytes, const std::string& source, std::int64_t offset,
                   const std::shared_ptr<FileSys::VfsFile>& storage = nullptr);

/// LICHII_TRACE_READ=<storage offset> (hex): when the game reads that offset, the next IPC calls
/// (svcSendSyncRequest) log the guest call stack of the thread making them. Returns whether one
/// should be logged now (consumes one).
void NoteTracedRead(std::int64_t offset);
[[nodiscard]] bool TakeIpcBacktrace();

} // namespace Core::Telemetry
