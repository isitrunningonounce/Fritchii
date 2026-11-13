// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "core/arm/bananza_nvn2_commands.h"
#include "core/arm/bananza_nvn2_resources.h"

namespace Core::Memory {
class Memory;
}

namespace Core::BananzaNvn2Trace {
class Session;
}

namespace Core::BananzaNvn2Hle {

struct GuestCall {
    std::array<u64, 8> x;
    std::array<u64, 8> v;
};

struct GuestCallback {
    u64 address{};
    std::array<u64, 4> args{};
};

struct CallbackContinuation {
    size_t procedure{};
    GuestCall call{};
    u64 link{};
};

class Session {
public:
    Session(u64 bootstrap, u64 internal_bootstrap, u64 callback_return);
    bool IsBootstrap(u64 pc) const;
    bool IsCallbackReturn(u64 pc) const;
    u64 CallbackReturn() const;
    static bool UseGuestStateHelper(std::string_view name);
    static bool IsFloatReturn(std::string_view name);
    std::optional<GuestCallback> BeginCallback(u64 thread, size_t procedure,
                                               std::string_view name, const GuestCall& call,
                                               u64 link);
    std::optional<CallbackContinuation> EndCallback(u64 thread);
    u64 Dispatch(const BananzaNvn2Trace::Session& procs, Memory::Memory& memory, size_t index,
                 const GuestCall& call);
    void Finish();

private:
    struct Device {
        u32 flags{};
        u32 origin{};
        u32 depth_mode{};
    };

    struct Pool {
        u64 device{};
        u64 storage{};
        u64 size{};
        u64 gpu_address{};
        u32 flags{};
    };

    struct Buffer {
        u64 device{};
        u64 pool{};
        u64 offset{};
        u64 size{};
        u64 pool_address{};
    };

    struct Queue {
        u64 device{};
        u64 command_size{};
        u64 compute_size{};
        u64 control_size{};
        u64 storage{};
        u64 storage_size{};
        u64 primary{};
        u64 flush_threshold{};
        u32 flags{};
        bool rejected_submit{};
    };

    enum class SyncStatus { Unsignaled, Signaled, Failed };

    struct Sync {
        u64 device{};
        SyncStatus status{};
    };

    struct Event {
        u64 pool{};
        u64 offset{};
        u64 pool_address{};
    };

    struct Window {
        u64 device{};
        u64 native{};
        std::vector<u64> textures;
        std::array<u32, 4> crop{};
        u32 active{};
        u32 interval{1};
        u32 next{};
        std::array<bool, 4> acquired{};
    };

    struct VirtualPage {
        u64 physical_pool{};
        u64 physical_address{};
        u64 offset{};
        u32 storage_class{};
    };

    struct CommandBuffer {
        u64 callback{};
        u64 callback_data{};
        u64 command_storage{};
        u64 gpu_address{};
        u32 recording_mode{};
        bool command_callback{true};
        bool control_callback{true};
    };

    struct MissingCall {
        u64 calls{};
        u64 reported{};
    };

    std::optional<u64> HandleDevice(Memory::Memory& memory, std::string_view name,
                                    const GuestCall& call);
    std::optional<u64> HandlePool(Memory::Memory& memory, std::string_view name,
                                  const GuestCall& call);
    std::optional<u64> HandleBuffer(Memory::Memory& memory, std::string_view name,
                                    const GuestCall& call);
    std::optional<u64> HandleQueue(Memory::Memory& memory, std::string_view name,
                                   const GuestCall& call);
    std::optional<u64> HandleWindow(Memory::Memory& memory, std::string_view name,
                                    const GuestCall& call);
    std::optional<u64> HandleSyncEvent(Memory::Memory& memory, std::string_view name,
                                       const GuestCall& call);
    bool MapVirtual(Memory::Memory& memory, const GuestCall& call);
    std::optional<ResourcePoolInfo> GetPool(u64 address) const;
    std::optional<u64> ResolveGpuRange(u64 address, u64 size) const;
    std::optional<u64> HandleCommands(Memory::Memory& memory, size_t index,
                                      std::string_view name, const GuestCall& call);
    std::optional<u64> HandleCommandPayload(Memory::Memory& memory, size_t index,
                                            std::string_view name, const GuestCall& call);
    void UpdateCommandCursor(Memory::Memory& memory, u64 address, const CommandBuffer& object);
    void UpdateCommandBuffer(Memory::Memory& memory, u64 address, const CommandBuffer& object,
                             const BananzaNvn2Commands::State& state);
    void NoteMissing(std::string_view name);
    void ReportMissing(bool final);

    const u64 bootstrap;
    const u64 internal_bootstrap;
    const u64 callback_return;
    std::mutex mutex;
    std::unordered_map<u64, Device> devices;
    std::unordered_map<u64, Pool> pools;
    std::map<u64, u64> gpu_pools;
    std::unordered_map<u64, VirtualPage> virtual_pages;
    std::unordered_map<u64, Buffer> buffers;
    std::unordered_map<u64, Queue> queue_builders;
    std::unordered_map<u64, Queue> queues;
    std::unordered_map<u64, Window> window_builders;
    std::unordered_map<u64, Window> windows;
    std::unordered_map<u64, Sync> syncs;
    std::unordered_map<u64, Event> event_builders;
    std::unordered_map<u64, Event> events;
    std::unordered_map<u64, CommandBuffer> command_buffers;
    std::unordered_map<u64, std::vector<CallbackContinuation>> callback_continuations;
    BananzaNvn2Commands::Recorder recorder;
    Resources resources;
    std::map<std::string, MissingCall, std::less<>> missing;
    std::chrono::steady_clock::time_point next_report =
        std::chrono::steady_clock::now() + std::chrono::seconds{10};
    u64 next_gpu_address = 0x1000000000ULL;
    u64 accepted_submissions{};
    u64 accepted_commands{};
    u64 rejected_submissions{};
    bool logged_submit{};
};

} // namespace Core::BananzaNvn2Hle
