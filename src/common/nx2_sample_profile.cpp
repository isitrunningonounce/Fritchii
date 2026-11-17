// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "common/nx2_sample_profile.h"

#ifdef _WIN32
#include <array>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <thread>

#include <windows.h>
#include <tlhelp32.h>

#include "common/fs/path_util.h"
#include "common/jit_symbols.h"
#include "common/logging.h"
#endif

namespace Common::Nx2SampleProfile {

namespace {

struct Config {
    unsigned hz{};
    unsigned delay_seconds{};
    unsigned seconds{20};
    std::vector<std::string> threads;
};

unsigned ReadUnsigned(const char* variable, unsigned fallback) {
    const char* const value = std::getenv(variable);
    return value != nullptr ? static_cast<unsigned>(std::strtoul(value, nullptr, 10)) : fallback;
}

const Config& GetConfig() {
    static const Config config = [] {
        Config result{
            .hz = ReadUnsigned("EDEN_NX2_SAMPLE_PROFILE", 0),
            .delay_seconds = ReadUnsigned("EDEN_NX2_SAMPLE_DELAY", 0),
            .seconds = ReadUnsigned("EDEN_NX2_SAMPLE_SECONDS", 20),
        };
        const char* const list = std::getenv("EDEN_NX2_SAMPLE_THREADS");
        std::string_view rest = list != nullptr ? list : "GPU,GPUBack,VulkanWorker";
        while (!rest.empty()) {
            const size_t comma = rest.find(',');
            if (const std::string_view item = rest.substr(0, comma); !item.empty()) {
                result.threads.emplace_back(item);
            }
            rest = comma == std::string_view::npos ? std::string_view{} : rest.substr(comma + 1);
        }
        return result;
    }();
    return config;
}

bool Selected(std::string_view name) {
    for (const std::string& pattern : GetConfig().threads) {
        if (!pattern.empty() && pattern.back() == '*'
                ? name.starts_with(std::string_view{pattern}.substr(0, pattern.size() - 1))
                : name == pattern) {
            return true;
        }
    }
    return false;
}

#ifdef _WIN32

// Record kinds of the sample file (tools/nx2/sample_report.py parses it).
constexpr std::uint8_t KindSample = 1;
constexpr std::uint8_t KindThread = 2;
constexpr std::uint8_t KindModule = 3;
// With EDEN_NX2_JIT_SYMBOLS=1: a JIT block (host start, host end, name), written at the end.
constexpr std::uint8_t KindJitBlock = 4;
// With EDEN_NX2_JIT_SYMBOLS=1: the guest call chain of the sample before it (x30, then the
// return address of each frame record from x29 up).
constexpr std::uint8_t KindGuestStack = 5;
constexpr std::uint8_t KindEnd = 0xff;
constexpr size_t MaxThreads = 32;
constexpr size_t MaxDepth = 32;
constexpr size_t MaxGuestDepth = 20;

struct SampledThread {
    HANDLE handle{};
    std::string name;
};

std::mutex threads_mutex;
std::array<SampledThread, MaxThreads> sampled_threads;
std::atomic<size_t> num_threads{};

template <typename T>
void Put(std::ofstream& file, T value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void PutString(std::ofstream& file, std::string_view text) {
    Put(file, static_cast<std::uint16_t>(text.size()));
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
}

std::string Utf8(const wchar_t* text) {
    const int size = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    std::string result(size > 0 ? size - 1 : 0, '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, -1, result.data(), size, nullptr, nullptr);
    return result;
}

void WriteModules(std::ofstream& file) {
    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        return;
    }
    MODULEENTRY32W entry{.dwSize = sizeof(MODULEENTRY32W)};
    for (BOOL more = Module32FirstW(snapshot, &entry); more; more = Module32NextW(snapshot, &entry)) {
        Put(file, KindModule);
        Put(file, static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(entry.modBaseAddr)));
        Put(file, static_cast<std::uint32_t>(entry.modBaseSize));
        PutString(file, Utf8(entry.szExePath));
    }
    CloseHandle(snapshot);
}

/// Unwinds a suspended thread's stack. Allocates nothing: the thread may hold the heap lock.
size_t WalkUnguarded(CONTEXT& context, std::array<std::uint64_t, MaxDepth>& frames,
                     size_t& depth, std::uint64_t& jit_state) {
    while (depth < MaxDepth && context.Rip != 0) {
        if (jit_state == 0 && JitSymbols::InCode(context.Rip)) {
            // JIT code keeps the guest state's address in r15.
            jit_state = context.R15;
        }
        frames[depth++] = context.Rip;
        DWORD64 image_base{};
        const PRUNTIME_FUNCTION function = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
        if (function == nullptr) {
            // A leaf without unwind data has its return address on top of the stack. Past the
            // first frame this is code without unwind data (JIT): stop there.
            if (depth > 1) {
                break;
            }
            context.Rip = *reinterpret_cast<const DWORD64*>(context.Rsp);
            context.Rsp += sizeof(DWORD64);
            continue;
        }
        PVOID handler_data{};
        DWORD64 establisher_frame{};
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, function, &context,
                         &handler_data, &establisher_frame, nullptr);
    }
    return depth;
}

/// A thread stopped in driver or JIT code can leave the unwinder a garbage stack pointer: it then
/// faults inside ntdll and took the whole emulator down (twice in Ocarina of Time sessions). Keep
/// the frames found up to the fault instead.
size_t Walk(CONTEXT& context, std::array<std::uint64_t, MaxDepth>& frames,
            std::uint64_t& jit_state) {
    size_t depth = 0;
#ifdef _MSC_VER
    __try {
        WalkUnguarded(context, frames, depth, jit_state);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
#else
    WalkUnguarded(context, frames, depth, jit_state);
#endif
    return depth;
}

/// The guest's call chain from the A64 state a JIT frame runs on. Reads through
/// ReadProcessMemory, which fails on an unmapped address instead of faulting.
size_t WalkGuest(std::uint64_t jit_state, std::array<std::uint64_t, MaxGuestDepth>& chain) {
    const std::uintptr_t base = JitSymbols::GuestBase();
    const HANDLE process = GetCurrentProcess();
    // x29 and x30: the state starts with the 31 general registers.
    std::uint64_t registers[2]{};
    if (base == 0 || jit_state == 0 ||
        !ReadProcessMemory(process, reinterpret_cast<LPCVOID>(jit_state + 29 * sizeof(std::uint64_t)),
                           registers, sizeof(registers), nullptr)) {
        return 0;
    }
    size_t depth = 0;
    chain[depth++] = registers[1];
    std::uint64_t frame = registers[0];
    while (depth < MaxGuestDepth && frame != 0 && (frame & 15) == 0 && frame < (1ULL << 48)) {
        std::uint64_t record[2]{};
        if (!ReadProcessMemory(process, reinterpret_cast<LPCVOID>(base + frame), record,
                               sizeof(record), nullptr)) {
            break;
        }
        chain[depth++] = record[1];
        if (record[0] <= frame) {
            break;
        }
        frame = record[0];
    }
    return depth;
}

void SamplerMain() {
    const Config& config = GetConfig();
    Sleep(config.delay_seconds * 1000);

    const auto path = Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) /
                      ("nx2_samples_" + std::to_string(GetCurrentProcessId()) + ".bin");
    std::ofstream file{path, std::ios::binary | std::ios::trunc};
    if (!file) {
        LOG_ERROR(Common, "Sample profile: can't open {}", path.string());
        return;
    }
    LARGE_INTEGER frequency{};
    QueryPerformanceFrequency(&frequency);
    file.write("NX2SMPL1", 8);
    Put(file, std::uint32_t{1});
    Put(file, static_cast<std::uint32_t>(config.hz));
    Put(file, static_cast<std::uint64_t>(frequency.QuadPart));
    WriteModules(file);

    // A high resolution timer reaches 2 kHz; without it (before Windows 10 1803) Sleep(1) runs
    // at the system timer resolution.
    constexpr DWORD HighResolution = 0x2; // CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
    const HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, HighResolution, TIMER_ALL_ACCESS);
    const LONGLONG period_100ns = -static_cast<LONGLONG>(10'000'000 / config.hz);

    LARGE_INTEGER start{};
    QueryPerformanceCounter(&start);
    const LONGLONG end = start.QuadPart + static_cast<LONGLONG>(config.seconds) * frequency.QuadPart;
    size_t threads_written = 0;
    std::uint64_t samples = 0;
    std::array<std::uint64_t, MaxDepth> frames{};
    std::array<std::uint64_t, MaxGuestDepth> chain{};
    for (LARGE_INTEGER now = start; now.QuadPart < end; QueryPerformanceCounter(&now)) {
        const size_t count = num_threads.load(std::memory_order_acquire);
        for (; threads_written < count; ++threads_written) {
            Put(file, KindThread);
            Put(file, static_cast<std::uint8_t>(threads_written));
            PutString(file, sampled_threads[threads_written].name);
        }
        for (size_t index = 0; index < count; ++index) {
            const HANDLE thread = sampled_threads[index].handle;
            if (SuspendThread(thread) == static_cast<DWORD>(-1)) {
                continue;
            }
            alignas(16) CONTEXT context{};
            context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
            std::uint64_t jit_state{};
            const size_t depth =
                GetThreadContext(thread, &context) ? Walk(context, frames, jit_state) : 0;
            const size_t guest_depth = jit_state != 0 ? WalkGuest(jit_state, chain) : 0;
            ResumeThread(thread);
            if (depth == 0) {
                continue;
            }
            Put(file, KindSample);
            Put(file, static_cast<std::uint8_t>(index));
            Put(file, static_cast<std::uint8_t>(depth));
            Put(file, static_cast<std::uint8_t>(0));
            Put(file, static_cast<std::uint64_t>(now.QuadPart));
            file.write(reinterpret_cast<const char*>(frames.data()),
                       static_cast<std::streamsize>(depth * sizeof(std::uint64_t)));
            if (guest_depth != 0) {
                Put(file, KindGuestStack);
                Put(file, static_cast<std::uint8_t>(index));
                Put(file, static_cast<std::uint8_t>(guest_depth));
                file.write(reinterpret_cast<const char*>(chain.data()),
                           static_cast<std::streamsize>(guest_depth * sizeof(std::uint64_t)));
            }
            ++samples;
        }
        if (timer != nullptr) {
            LARGE_INTEGER due{.QuadPart = period_100ns};
            SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
            WaitForSingleObject(timer, INFINITE);
        } else {
            Sleep(1);
        }
    }
    for (const JitSymbols::Block& block : JitSymbols::Snapshot()) {
        Put(file, KindJitBlock);
        Put(file, static_cast<std::uint64_t>(block.start));
        Put(file, static_cast<std::uint64_t>(block.end));
        PutString(file, block.name);
    }
    Put(file, KindEnd);
    file.close();
    if (timer != nullptr) {
        CloseHandle(timer);
    }
    LOG_INFO(Common, "Sample profile: {} samples of {} threads at {} Hz written to {}", samples,
             threads_written, config.hz, path.string());
}

#endif

} // namespace

bool Requested() {
    return GetConfig().hz != 0;
}

void OnThreadNamed(const char* name) {
    if (!Requested() || !Selected(name)) {
        return;
    }
#ifdef _WIN32
    HANDLE handle{};
    if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
                         THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                         FALSE, 0)) {
        return;
    }
    {
        std::scoped_lock lock{threads_mutex};
        const size_t index = num_threads.load(std::memory_order_relaxed);
        if (index == MaxThreads) {
            CloseHandle(handle);
            return;
        }
        sampled_threads[index] = {handle, name};
        num_threads.store(index + 1, std::memory_order_release);
    }
    static std::once_flag started;
    std::call_once(started, [] { std::thread{SamplerMain}.detach(); });
#endif
}

} // namespace Common::Nx2SampleProfile
