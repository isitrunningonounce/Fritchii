// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// LICHII_ZSTD_HLE (Ocarina of Time): the game's zstd decoder (ZSTD_decompressMultiFrame at
// main+0xeb380, statically linked) runs on the host instead of in the JIT. The game decompresses
// every model, texture and sound it streams (~45% of the guest cores during area loads); the host
// library does the same work several times faster.
//   LICHII_ZSTD_HLE=1      replace the guest function
//   LICHII_ZSTD_HLE=verify keep the guest function and compare its output with the host's
// Anything the host decoder rejects runs the guest function as before.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <zstd.h>

#include "common/cityhash.h"
#include "common/common_types.h"
#include "common/logging.h"

namespace Core::ZstdHle {

constexpr u64 OcarinaProgramId = 0x0400486024DEC000ULL;
constexpr u64 MultiFrameOffset = 0xeb380;
// pacibsp; stp x29, x30, [sp, #-0x60]!; stp x28, x27, [sp, #0x10]; stp x26, x25, [sp, #0x20]
constexpr std::array<u32, 4> MultiFramePrologue{0xD503237F, 0xA9BA7BFD, 0xA9016FFC, 0xA90267FA};
constexpr u32 Brk = 0xD4200000;

enum class Mode { Off, Replace, Verify };

inline Mode GetMode() {
    static const Mode mode = [] {
        const char* value = std::getenv("LICHII_ZSTD_HLE");
        if (value == nullptr || value[0] == '\0' || value[0] == '0') {
            return Mode::Off;
        }
        return std::string_view{value} == "verify" ? Mode::Verify : Mode::Replace;
    }();
    return mode;
}

inline std::atomic<u64> main_base{0};
/// 0: not checked yet, 1: the prologue matches this build, 2: it does not (left alone).
inline std::atomic<int> checked{0};

inline void SetModuleBase(std::string_view name, u64 base) {
    if (GetMode() != Mode::Off && name == "main") {
        main_base.store(base);
        checked.store(0);
    }
}

inline u64 Target() {
    const u64 base = main_base.load(std::memory_order_relaxed);
    return base != 0 ? base + MultiFrameOffset : 0;
}

/// The instruction the JIT gets: a BRK on the decoder's entry once its prologue is verified.
template <typename Memory>
inline bool IsHookedEntry(u64 vaddr, u64 program_id, Memory& memory) {
    // The one-shot decoder only with LICHII_ZSTD_HLE_ONESHOT=1: no measurable gain, and its
    // verify could not tell the game's reuse of the output buffers from a wrong decode.
    static const bool oneshot = [] {
        const char* value = std::getenv("LICHII_ZSTD_HLE_ONESHOT");
        return value != nullptr && value[0] == '1';
    }();
    if (!oneshot || GetMode() == Mode::Off || program_id != OcarinaProgramId ||
        vaddr != Target()) {
        return false;
    }
    int state = checked.load();
    if (state == 0) {
        bool match = true;
        for (size_t i = 0; i < MultiFramePrologue.size(); ++i) {
            match &= memory.Read32(vaddr + i * 4) == MultiFramePrologue[i];
        }
        state = match ? 1 : 2;
        checked.store(state);
        if (match) {
            LOG_INFO(Core_ARM, "LICHII_ZSTD_HLE: zstd decoder at {:#x} runs on the host ({})", vaddr,
                     GetMode() == Mode::Verify ? "verify" : "replace");
        } else {
            LOG_WARNING(Core_ARM, "LICHII_ZSTD_HLE: unexpected code at {:#x}: left alone", vaddr);
        }
    }
    return state == 1;
}

struct Stats {
    std::mutex mutex;
    u64 calls{}, replaced{}, fallbacks{}, in_bytes{}, out_bytes{};
    u64 verified{}, mismatches{}, bad{};
    double host_ms{};
    std::chrono::steady_clock::time_point last_report = std::chrono::steady_clock::now();
};
inline Stats stats;

struct DictEntry {
    u64 hash{};
    ZSTD_DDict* ddict{};
};
inline std::mutex dict_mutex;
inline std::unordered_map<u64, DictEntry> dicts; ///< by guest content address ^ size

/// Verify mode: the host's output for a call, compared with guest memory at the next call.
struct PendingCheck {
    u64 dst{};
    u64 size{};
    u64 hash{};
    std::vector<u8> bytes; ///< The host's output, to count differing bytes
};
inline std::mutex pending_mutex;
inline std::vector<PendingCheck> pending;

inline void Report() {
    const auto now = std::chrono::steady_clock::now();
    if (now - stats.last_report < std::chrono::seconds{10}) {
        return;
    }
    stats.last_report = now;
    LOG_INFO(Core_ARM,
             "LICHII_ZSTD_HLE: {} calls, {} on the host, {} by the guest; {:.1f} MiB -> {:.1f} MiB "
             "in {:.1f} ms of host time; verified {} ({} differ, {} of them over 20% of the bytes)",
             stats.calls, stats.replaced, stats.fallbacks, stats.in_bytes / 1048576.0,
             stats.out_bytes / 1048576.0, stats.host_ms, stats.verified, stats.mismatches,
             stats.bad);
}

/// Runs the decoder call on the host. Returns true when the guest function must be skipped (its
/// result is in x0); false runs the guest function (verify mode, or the host could not decode).
template <typename Jit, typename Memory>
inline bool Run(Jit& jit, Memory& memory) {
    const u64 dst = jit.GetRegister(1);
    const u64 capacity = jit.GetRegister(2);
    const u64 src = jit.GetRegister(3);
    const u64 src_size = jit.GetRegister(4);
    const u64 dict = jit.GetRegister(5);
    const u64 dict_size = jit.GetRegister(6);
    const u64 guest_ddict = jit.GetRegister(7);

    if (GetMode() == Mode::Verify) {
        // The previous calls' outputs are final by now: compare them.
        std::vector<PendingCheck> checks;
        {
            std::scoped_lock lock{pending_mutex};
            checks.swap(pending);
        }
        for (const auto& check : checks) {
            std::vector<u8> guest(check.size);
            memory.ReadBlock(check.dst, guest.data(), guest.size());
            const bool same = Common::CityHash64(reinterpret_cast<const char*>(guest.data()),
                                                 guest.size()) == check.hash;
            // The game relocates loaded resources in place (offsets become pointers) before the
            // next call: a few percent of the bytes then differ. A wrong decode differs nearly
            // everywhere.
            size_t differing = 0;
            if (!same) {
                for (size_t i = 0; i < guest.size(); ++i) {
                    differing += guest[i] != check.bytes[i] ? 1 : 0;
                }
            }
            const double percent = guest.empty() ? 0.0 : 100.0 * differing / guest.size();
            std::scoped_lock lock{stats.mutex};
            ++stats.verified;
            if (!same) {
                ++stats.mismatches;
                stats.bad += percent > 20.0 ? 1 : 0;
                if (percent > 20.0 || stats.mismatches < 8) {
                    LOG_WARNING(Core_ARM,
                                "LICHII_ZSTD_HLE verify: output at {:#x} ({} bytes): {:.2f}% of the "
                                "bytes differ{}",
                                check.dst, check.size, percent,
                                percent > 20.0 ? " (WRONG DECODE?)" : " (relocated by the game)");
                }
            }
        }
    }

    const auto fallback = [&] {
        std::scoped_lock lock{stats.mutex};
        ++stats.calls;
        ++stats.fallbacks;
        Report();
        return false;
    };
    if (src_size == 0 || src_size > (u64{256} << 20) || capacity > (u64{1} << 30) ||
        !memory.IsValidVirtualAddressRange(src, src_size) ||
        (capacity != 0 && !memory.IsValidVirtualAddressRange(dst, capacity))) {
        return fallback();
    }
    // The dictionary: the DDict's content (dictContent at +8, dictSize at +16) or the raw one.
    u64 content = 0, content_size = 0;
    if (guest_ddict != 0) {
        content = memory.Read64(guest_ddict + 8);
        content_size = memory.Read64(guest_ddict + 16);
    } else if (dict != 0 && dict_size != 0) {
        content = dict;
        content_size = dict_size;
    }
    if (content_size > (u64{32} << 20) ||
        (content_size != 0 && !memory.IsValidVirtualAddressRange(content, content_size))) {
        return fallback();
    }

    const auto begin = std::chrono::steady_clock::now();
    thread_local std::vector<u8> input;
    thread_local std::unique_ptr<u8[]> output;
    thread_local size_t output_size = 0;
    thread_local ZSTD_DCtx* dctx = ZSTD_createDCtx();
    input.resize(src_size);
    memory.ReadBlock(src, input.data(), src_size);
    if (output_size < capacity) {
        output_size = capacity;
        output = std::make_unique_for_overwrite<u8[]>(output_size);
    }

    ZSTD_DDict* host_ddict = nullptr;
    if (content_size != 0) {
        std::vector<u8> bytes(content_size);
        memory.ReadBlock(content, bytes.data(), content_size);
        const u64 hash =
            Common::CityHash64(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        std::scoped_lock lock{dict_mutex};
        auto& entry = dicts[content ^ (content_size << 40)];
        if (entry.ddict == nullptr || entry.hash != hash) {
            if (entry.ddict != nullptr) {
                ZSTD_freeDDict(entry.ddict);
            }
            entry.ddict = ZSTD_createDDict(bytes.data(), bytes.size());
            entry.hash = hash;
        }
        host_ddict = entry.ddict;
        if (host_ddict == nullptr) {
            return fallback();
        }
    }
    const size_t result =
        host_ddict != nullptr
            ? ZSTD_decompress_usingDDict(dctx, output.get(), capacity, input.data(), src_size,
                                         host_ddict)
            : ZSTD_decompressDCtx(dctx, output.get(), capacity, input.data(), src_size);
    if (ZSTD_isError(result)) {
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1) < 8) {
            LOG_WARNING(Core_ARM, "LICHII_ZSTD_HLE: host decoder: {} (src {} bytes, dict {} bytes): "
                                  "the guest decodes it",
                        ZSTD_getErrorName(result), src_size, content_size);
        }
        return fallback();
    }
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();

    if (GetMode() == Mode::Verify) {
        std::scoped_lock lock{pending_mutex};
        pending.push_back({dst, result,
                           Common::CityHash64(reinterpret_cast<const char*>(output.get()), result),
                           std::vector<u8>(output.get(), output.get() + result)});
        std::scoped_lock stats_lock{stats.mutex};
        ++stats.calls;
        ++stats.fallbacks;
        stats.in_bytes += src_size;
        stats.out_bytes += result;
        stats.host_ms += ms;
        Report();
        return false;
    }
    memory.WriteBlock(dst, output.get(), result);
    jit.SetRegister(0, result);
    std::scoped_lock lock{stats.mutex};
    ++stats.calls;
    ++stats.replaced;
    stats.in_bytes += src_size;
    stats.out_bytes += result;
    stats.host_ms += ms;
    Report();
    return true;
}

// ---------------------------------------------------------------------------------------------
// The game's own streaming decoder (main+0xf5c80), what its background loader runs: a context
// (x0) fed one read buffer per call (x1 = input, w2 = bytes, x3 = the DDict on the first call),
// built on zstd's buffer-less API. Context: +0 state (1 begin a frame, 2 running, 3 failed,
// 4 done), +8 output position, +0x10 output capacity left (s32), +0x14 frame bytes still to
// produce (s32). It returns the bytes still to produce (0: the frame is complete), negative on
// error. LICHII_ZSTD_HLE=1 decodes a frame begun on the host entirely on the host (a zstd stream
// per context); =verify keeps the guest decoder and compares what it produced at every call.
constexpr u64 StreamOffset = 0xf5c80;
// pacibsp; sub sp, sp, #0x90; stp x29, x30, [sp, #0x30]; stp x28, x27, [sp, #0x40]
constexpr std::array<u32, 4> StreamPrologue{0xD503237F, 0xD10243FF, 0xA9037BFD, 0xA9046FFC};
inline std::atomic<int> stream_checked{0};

inline u64 StreamTarget() {
    const u64 base = main_base.load(std::memory_order_relaxed);
    return base != 0 ? base + StreamOffset : 0;
}

template <typename Memory>
inline bool IsHookedStreamEntry(u64 vaddr, u64 program_id, Memory& memory) {
    if (GetMode() == Mode::Off || program_id != OcarinaProgramId || vaddr != StreamTarget()) {
        return false;
    }
    int state = stream_checked.load();
    if (state == 0) {
        bool match = true;
        for (size_t i = 0; i < StreamPrologue.size(); ++i) {
            match &= memory.Read32(vaddr + i * 4) == StreamPrologue[i];
        }
        state = match ? 1 : 2;
        stream_checked.store(state);
        LOG_INFO(Core_ARM, "LICHII_ZSTD_HLE: streaming decoder at {:#x} {}", vaddr,
                 match ? "runs on the host" : "has unexpected code: left alone");
    }
    return state == 1;
}

struct HostStream {
    ZSTD_DStream* stream{};
    bool on_host{};         ///< This frame is decoded by the host (else by the guest)
    u64 dst_start{};        ///< Verify: where the frame's output begins
    std::vector<u8> shadow; ///< Verify: the host's output so far
    u64 checked{};          ///< Verify: guest bytes compared so far
    bool failed{};          ///< Verify: the host stream errored for this frame
};
inline std::mutex streams_mutex;
inline std::unordered_map<u64, HostStream> streams; ///< by guest context address

struct StreamStats {
    std::mutex mutex;
    u64 frames{}, calls{}, in_bytes{}, out_bytes{}, compared{}, mismatches{}, errors{};
    double host_ms{};
    std::chrono::steady_clock::time_point last_report = std::chrono::steady_clock::now();
};
inline StreamStats stream_stats;

inline void ReportStreams() {
    const auto now = std::chrono::steady_clock::now();
    if (now - stream_stats.last_report < std::chrono::seconds{10}) {
        return;
    }
    stream_stats.last_report = now;
    LOG_INFO(Core_ARM,
             "LICHII_ZSTD_HLE stream: {} frames, {} calls, {:.1f} MiB -> {:.1f} MiB in {:.1f} ms of "
             "host time, {} errors; verify compared {:.1f} MiB, {} mismatches",
             stream_stats.frames, stream_stats.calls, stream_stats.in_bytes / 1048576.0,
             stream_stats.out_bytes / 1048576.0, stream_stats.host_ms, stream_stats.errors,
             stream_stats.compared / 1048576.0, stream_stats.mismatches);
}

/// The host DDict for a guest DDict (its content at +8, size at +16); null for none or failure.
template <typename Memory>
inline ZSTD_DDict* HostDDictFor(u64 guest_ddict, Memory& memory) {
    if (guest_ddict == 0) {
        return nullptr;
    }
    const u64 content = memory.Read64(guest_ddict + 8);
    const u64 size = memory.Read64(guest_ddict + 16);
    if (size == 0 || size > (u64{32} << 20) || !memory.IsValidVirtualAddressRange(content, size)) {
        return nullptr;
    }
    std::vector<u8> bytes(size);
    memory.ReadBlock(content, bytes.data(), size);
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    std::scoped_lock lock{dict_mutex};
    auto& entry = dicts[content ^ (size << 40)];
    if (entry.ddict == nullptr || entry.hash != hash) {
        if (entry.ddict != nullptr) {
            ZSTD_freeDDict(entry.ddict);
        }
        entry.ddict = ZSTD_createDDict(bytes.data(), bytes.size());
        entry.hash = hash;
    }
    return entry.ddict;
}

/// One call of the game's streaming decoder. True: decoded on the host, the result is in x0 and
/// the guest function is skipped. False: the guest function runs (verify mode, or a frame the
/// host did not begin).
template <typename Jit, typename Memory>
inline bool RunStream(Jit& jit, Memory& memory) {
    const u64 ctx = jit.GetRegister(0);
    const u64 input = jit.GetRegister(1);
    const s64 input_size = static_cast<s32>(jit.GetRegister(2));
    const u64 guest_ddict = jit.GetRegister(3);
    if (!memory.IsValidVirtualAddressRange(ctx, 0x38) || input_size < 0 ||
        input_size > (s64{256} << 20) ||
        (input_size > 0 && !memory.IsValidVirtualAddressRange(input, input_size))) {
        return false;
    }
    const u8 state = memory.Read8(ctx);
    const bool verify = GetMode() == Mode::Verify;
    std::scoped_lock lock{streams_mutex};
    HostStream& host = streams[ctx];
    if (state == 1) {
        // A new frame: its content size from the header, as the guest reads it.
        host.on_host = false;
        host.failed = false;
        host.shadow.clear();
        host.checked = 0;
        host.dst_start = memory.Read64(ctx + 8);
        std::vector<u8> head(static_cast<size_t>((std::min<s64>)(input_size, 32)));
        memory.ReadBlock(input, head.data(), head.size());
        const unsigned long long content = ZSTD_getFrameContentSize(head.data(), head.size());
        if (content == ZSTD_CONTENTSIZE_UNKNOWN || content == ZSTD_CONTENTSIZE_ERROR ||
            content > 0x7FFFFFFFULL) {
            return false; // the guest handles (and reports) it
        }
        if (host.stream == nullptr) {
            host.stream = ZSTD_createDStream();
        }
        ZSTD_DCtx_reset(host.stream, ZSTD_reset_session_and_parameters);
        ZSTD_DCtx_setParameter(host.stream, ZSTD_d_windowLogMax, 31);
        if (guest_ddict != 0) {
            ZSTD_DDict* const ddict = HostDDictFor(guest_ddict, memory);
            if (ddict == nullptr) {
                return false;
            }
            ZSTD_DCtx_refDDict(host.stream, ddict);
        }
        if (!verify) {
            host.on_host = true;
            memory.Write32(ctx + 0x14, static_cast<u32>(content));
            memory.Write8(ctx, 2);
        }
        std::scoped_lock stats_lock{stream_stats.mutex};
        ++stream_stats.frames;
    } else if (!verify && (state != 2 || !host.on_host)) {
        return false; // a frame the guest began, or a finished/failed one: the guest answers
    }

    if (verify && state == 2 && !host.failed) {
        // What the guest produced since the last call, against the host's output.
        const u64 produced = memory.Read64(ctx + 8) - host.dst_start;
        const u64 upto = (std::min<u64>)(produced, host.shadow.size());
        if (upto > host.checked) {
            std::vector<u8> guest(upto - host.checked);
            memory.ReadBlock(host.dst_start + host.checked, guest.data(), guest.size());
            const bool same = std::memcmp(guest.data(), host.shadow.data() + host.checked,
                                          guest.size()) == 0;
            std::scoped_lock stats_lock{stream_stats.mutex};
            stream_stats.compared += guest.size();
            if (!same && stream_stats.mismatches++ < 16) {
                LOG_WARNING(Core_ARM,
                            "LICHII_ZSTD_HLE stream verify: context {:#x} differs in [{:#x}, {:#x})",
                            ctx, host.checked, upto);
            }
            host.checked = upto;
        }
    }

    // Decode this read buffer.
    const auto begin = std::chrono::steady_clock::now();
    thread_local std::vector<u8> in_bytes;
    thread_local std::vector<u8> out_bytes;
    in_bytes.resize(static_cast<size_t>(input_size));
    if (input_size > 0) {
        memory.ReadBlock(input, in_bytes.data(), in_bytes.size());
    }
    const s32 capacity = static_cast<s32>(memory.Read32(ctx + 0x10));
    const s32 left = static_cast<s32>(memory.Read32(ctx + 0x14));
    // Replace: at most what the guest may still write. Verify: no limit (the shadow grows).
    const size_t room = verify ? SIZE_MAX
                               : static_cast<size_t>((std::max)(0, (std::min)(capacity, left)));
    constexpr size_t Chunk = size_t{1} << 20;
    ZSTD_inBuffer in{in_bytes.data(), in_bytes.size(), 0};
    size_t produced_total = 0;
    size_t ret = 1;
    bool error = false;
    for (;;) {
        const size_t space = (std::min)(Chunk, room - produced_total);
        if (space == 0) {
            break;
        }
        out_bytes.resize(produced_total + space);
        ZSTD_outBuffer out{out_bytes.data() + produced_total, space, 0};
        ret = ZSTD_decompressStream(host.stream, &out, &in);
        if (ZSTD_isError(ret)) {
            error = true;
            break;
        }
        produced_total += out.pos;
        const bool out_full = out.pos == out.size;
        if ((in.pos == in.size || ret == 0) && !out_full) {
            break;
        }
    }
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    {
        std::scoped_lock stats_lock{stream_stats.mutex};
        ++stream_stats.calls;
        stream_stats.in_bytes += in.pos;
        stream_stats.out_bytes += produced_total;
        stream_stats.host_ms += ms;
        stream_stats.errors += error ? 1 : 0;
        ReportStreams();
    }
    if (verify) {
        if (error) {
            host.failed = true;
        } else {
            host.shadow.insert(host.shadow.end(), out_bytes.begin(),
                               out_bytes.begin() + static_cast<std::ptrdiff_t>(produced_total));
        }
        return false;
    }
    if (error || in.pos < in.size) {
        // As the guest's error path: failed state, -1.
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1) < 8) {
            LOG_WARNING(Core_ARM, "LICHII_ZSTD_HLE stream: context {:#x}: {}", ctx,
                        error ? ZSTD_getErrorName(ret) : "output full with input left");
        }
        host.on_host = false;
        memory.Write8(ctx, 3);
        jit.SetRegister(0, static_cast<u64>(static_cast<u32>(-1)));
        return true;
    }
    const u64 dst = memory.Read64(ctx + 8);
    if (produced_total != 0) {
        memory.WriteBlock(dst, out_bytes.data(), produced_total);
    }
    const s32 produced = static_cast<s32>(produced_total);
    memory.Write64(ctx + 8, dst + produced_total);
    memory.Write32(ctx + 0x10, static_cast<u32>(capacity - produced));
    const s32 remaining = left - produced;
    memory.Write32(ctx + 0x14, static_cast<u32>(remaining));
    if (ret == 0) {
        // The frame is complete (the guest's "done" state).
        host.on_host = false;
        memory.Write8(ctx, 4);
    }
    jit.SetRegister(0, static_cast<u64>(static_cast<u32>(remaining)));
    return true;
}

} // namespace Core::ZstdHle
