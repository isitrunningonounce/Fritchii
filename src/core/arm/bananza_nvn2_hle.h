// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <map>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/reader_mutex.h"
#include "common/common_types.h"
#include "common/thread_worker.h"
#include "core/arm/bananza_nvn2_commands.h"
#include "core/arm/bananza_nvn2_resources.h"
#include "core/arm/bananza_nvn2_submit_lane.h"
#include "core/arm/nvn2_sdk_layout.h"
#include "core/hle/service/nx2_present_pacer.h"

namespace Core {
class System;
}

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
    Session(System& system, const BananzaNvn2Trace::Session& procedures, u64 bootstrap,
            u64 internal_bootstrap, u64 callback_return);
    ~Session();
    bool IsBootstrap(u64 pc) const;
    bool IsCallbackReturn(u64 pc) const;
    u64 CallbackReturn() const;
    bool UsesGuestStateHelper(size_t index) const;
    std::optional<std::pair<size_t, size_t>> DirectCallArguments(size_t index) const;
    bool ReturnsFloat(size_t index) const;
    bool ProfilesCalls() const;
    void ProfileCall(size_t index, std::string_view name, u64 nanoseconds, bool continuation,
                     u64 thread);
    std::optional<GuestCallback> BeginCallback(u64 thread, size_t procedure, const GuestCall& call,
                                               u64 link);
    std::optional<CallbackContinuation> EndCallback(u64 thread);
    u64 Dispatch(const BananzaNvn2Trace::Session& procs, Memory::Memory& memory, size_t index,
                 const GuestCall& call, std::chrono::steady_clock::time_point* present_deadline = nullptr);
    void Finish();

private:
    enum class Handler : u8 { Resource, Device, Pool, Buffer, Queue, Window, SyncEvent, Command,
                              GetProcAddress, GetTextureHandle, GetPoolFlags, GetBufferAddress,
                              MapBuffer };
    struct Procedure {
        std::string_view name;
        Handler handler{};
        bool guest_helper{};
        bool float_return{};
        bool begins_recording{};
        // Touches only its own command buffer, so it records without the session mutex.
        bool per_buffer_command{};
        bool ends_recording{};
        bool appends_handles{};
        bool calls_handles{};
        bool binds_deferred_textures{};
        const void* scalar_command{};
        const void* state_binding{};
    };
    static bool UseGuestStateHelper(std::string_view name);

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
        u64 serial{};
        // Newest FlushMappedRange over each 64 KiB of the pool; empty until the first flush.
        mutable std::vector<u64> flush_stamps;
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
        u64 fence{};
        u64 sequence{};
        u64 submit_host_ns{};
        u64 texture_pool{};
        u64 sampler_pool{};
        u64 copy_row_stride{};
        std::array<std::pair<u64, u64>, 32> vertex_ranges{};
        std::array<std::array<std::pair<u64, u64>, 32>, 6> uniform_ranges{};
        std::array<std::array<std::pair<u64, u64>, 32>, 6> storage_ranges{};
        std::array<std::array<u64, 64>, 6> texture_handles{};
        std::array<std::array<u64, 32>, 6> image_handles{};
        // Serials of the bound render targets, colors then depth.
        std::array<u64, 9> target_serials{};
        bool targets_noted{};
        bool resets_indirect_draws{};
        u32 flags{};
        bool rejected_submit{};
    };

    enum class SyncStatus { Unsignaled, Signaled, Failed };

    struct Sync {
        u64 device{};
        u64 fence{};
        u64 sequence{};
        u64 submit_host_ns{};
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
        // The recorder's entry for it, from Initialize to Finalize.
        BananzaNvn2Commands::Recorder::Buffer* recording{};
        u64 callback{};
        u64 callback_data{};
        u64 command_storage{};
        u64 gpu_address{};
        u32 recording_mode{};
        bool command_callback{true};
        bool control_callback{true};
    };

    struct TextureCheck {
        u64 hash{};
        u64 present{};
        u64 map_generation{};
        // Pool serial and newest flush over the texture when it was hashed; 0 without the flush rule.
        u64 flush_key{};
    };
    struct TextureCopyStat {
        u64 count{};
        u64 bytes{};
    };

    struct MissingCall {
        u64 calls{};
        u64 reported{};
    };

    struct RecordingStats {
        std::unordered_set<u64> submitted;
        std::unordered_set<u64> nested;
        std::unordered_set<u64> command_buffers;
        std::unordered_set<u64> used;
        std::unordered_set<u64> recorded_buffers;
        u64 presents{};
        u64 submissions{};
        u64 recorded{};
        u64 rerecorded{};
        u64 commands{};
        u64 draws{};
        u64 called_commands{};
        u64 called_draws{};
        u64 copied_commands{};
        u64 copied_draws{};
        u64 recording_uses{};
        u64 repeated_uses{};
        u64 prior_window_uses{};
    };

    struct CallStats {
        std::string_view name;
        u64 calls{};
        u64 nanoseconds{};
        u64 lock_wait{};
    };

    struct ThreadStats {
        u64 calls{};
        u64 nanoseconds{};
    };

    // Written under the session mutex.
    struct SubmitStats {
        u64 submits{};
        u64 waits{};
        u64 scan_ns{};
        u64 texture_ns{};
        u64 copy_wait_ns{};
        u64 wait_ns{};
        u64 buffer_bytes{};
        u64 texture_bytes{};
        u64 buffer_ranges{};
        u64 textures{};
        u64 overlap_waits{};
        u64 overlap_wait_ns{};
        u64 present_waits{};
        u64 present_wait_ns{};
        u64 referenced_bytes{};
        u64 referenced_ranges{};
        u64 texture_hashes{};
        u64 texture_hash_bytes{};
        u64 texture_flush_skips{};
        u64 fixed_bytes{};
        u64 fixed_ranges{};
        std::map<u32, u64> referenced_by_flags;
        u64 sync_waits{};
        u64 sync_timeouts{};
        u64 sync_age_ns{};
        u64 sync_fence_ns{};
        u64 sync_retire_ns{};
    };

    struct CallProfile {
        std::mutex mutex;
        std::array<CallStats, Nvn2::SdkLayout::MaxEntries> entries;
        std::map<u64, ThreadStats> threads;
        SubmitStats submit;
        u64 presents{};
    };

    std::optional<u64> HandleDevice(Memory::Memory& memory, std::string_view name,
                                    const GuestCall& call);
    std::optional<u64> HandlePool(Memory::Memory& memory, std::string_view name,
                                  const GuestCall& call);
    std::optional<u64> HandleBuffer(Memory::Memory& memory, std::string_view name,
                                    const GuestCall& call);
    std::optional<u64> HandleQueue(Memory::Memory& memory, std::string_view name,
                                  const GuestCall& call, std::unique_lock<std::mutex>& lock,
                                  std::chrono::steady_clock::time_point* present_deadline);
    std::optional<u64> HandleWindow(Memory::Memory& memory, std::string_view name,
                                    const GuestCall& call);
    std::optional<u64> HandleSyncEvent(Memory::Memory& memory, std::string_view name,
                                       const GuestCall& call, std::unique_lock<std::mutex>& lock);
    bool MapVirtual(Memory::Memory& memory, const GuestCall& call);
    std::optional<ResourcePoolInfo> GetPool(u64 address) const;
    std::optional<u64> ResolveGpuRange(u64 address, u64 size,
                                       const Pool** backing = nullptr) const;
    // The pool that backs a GPU address: the physical pool behind a virtual one.
    const Pool* BackingPool(u64 address) const;
    void SnapshotResources(Memory::Memory& memory, BananzaNvn2Commands::Submission& submission,
                           Queue& queue);
    // The part of a submit after the recordings are taken: the scan and the hand-over to the
    // GPU thread. Runs with the session lock held, in the submit call or on the submit lane.
    void FinishSubmit(Memory::Memory& memory, Queue& queue,
                      BananzaNvn2Commands::Submission& submission,
                      std::unique_lock<std::mutex>& lock);
    // nvnQueuePresentTexture without the session lock: the present itself goes to the submit
    // lane, behind the scans of the submits before it.
    u64 PresentAsync(const GuestCall& call, std::chrono::steady_clock::time_point* present_deadline);
    // nvnQueueFenceSync without the session lock; false when the queue or the sync is unknown.
    bool FenceSyncAsync(const GuestCall& call);
    // nvnBufferInitialize: under pool_buffer_mutex alone.
    u64 InitializeBuffer(Memory::Memory& memory, const GuestCall& call);
    // nvnWindowAcquireTexture without the session lock; false when it cannot hand out an
    // image (the call then goes the usual way and is reported).
    bool AcquireAsync(Memory::Memory& memory, const GuestCall& call);
    // Waits for the pending submissions that may write back into [guest, guest + size).
    void WaitForPendingWrites(u64 guest, u64 size, const char* reason);
    std::optional<u64> HandleCommands(Memory::Memory& memory, size_t index,
                                      std::string_view name, const GuestCall& call);
    std::optional<bool> RecordCommand(Memory::Memory& memory, size_t index, const GuestCall& call);
    std::optional<u64> HandleCommandPayload(Memory::Memory& memory, size_t index,
                                            std::string_view name, const GuestCall& call);
    void UpdateCommandBuffer(Memory::Memory& memory, u64 address, const CommandBuffer& object,
                             const BananzaNvn2Commands::State& state);
    void NoteMissing(std::string_view name);
    void ReportMissing(bool final);
    void CountRecordings(const BananzaNvn2Commands::Submission& batch);
    void ReportRecordingStats();
    void ReportCallProfile();

    const u64 bootstrap;
    const u64 internal_bootstrap;
    const u64 callback_return;
    std::vector<Procedure> procedures;
    System& system;
    // Copies of bound buffer ranges made while a submit scans its commands.
    // When the game thread began the frame it is working on (the end of its last pacing wait).
    std::chrono::steady_clock::time_point perf_frame_begin{};
    Common::ThreadWorker copy_workers{8, "NvnCopy"};
    // Copies that may finish after the replay of their submission has started.
    Common::ThreadWorker late_copy_workers{4, "NvnLate"};
    std::mutex mutex;
    Common::ReaderMutex pool_buffer_mutex;
    // Guards resources and events against per-buffer commands that run without the session
    // mutex. Writers also hold the session mutex.
    Common::ReaderMutex resource_mutex;
    std::mutex missing_mutex;
    // The continuations of memory callbacks, by guest thread.
    std::mutex callback_mutex;
    u64 pending_storage_fence{};
    // Guest ranges a submission may write back once its replay ends, start to end, disjoint.
    struct PendingWrites {
        u64 fence;
        u64 sequence;
        // Start and end of each, in order and apart: the list is copied for every scan.
        std::vector<std::pair<u64, u64>> ranges;
    };
    std::vector<PendingWrites> pending_writes;
    std::vector<u64> present_fences;
    // Guards the windows, the syncs, the set of queues and each queue's fence, sequence and
    // submit time, for the present and the fence made without the session lock. Taken after
    // the session lock by the calls that hold both, and never held across a wait.
    std::mutex frame_mutex;
    // The newest sequence submitted when the last present was made, and the present before.
    std::array<u64, 2> present_marks{};
    // Flushes of mapped ranges the game made without waiting for the session lock: applied
    // by the next call that holds it, a scan included.
    struct PendingFlush {
        u64 pool;
        u64 offset;
        u64 size;
    };
    std::mutex flush_mutex;
    std::vector<PendingFlush> pending_flushes;
    std::atomic<bool> flushes_pending{};
    void ApplyPendingFlushes();
    // Presents handed to the submit lane that the GPU thread has not shown yet.
    std::atomic<u32> presents_outstanding{};
    std::mutex present_wait_mutex;
    std::condition_variable present_done;
    u32 logged_overlaps{};
    Service::Nx2PresentPacer present_pacer;
    std::chrono::nanoseconds present_period{};
    std::unordered_map<u64, Device> devices;
    std::unordered_map<u64, Pool> pools;
    std::map<u64, u64> gpu_pools;
    // The pools with CPU storage by GPU address, for the submit scan; rebuilt when stale.
    struct PoolSpan {
        u64 begin;
        u64 size;
        u64 storage;
        const Pool* pool;
    };
    std::vector<PoolSpan> pool_spans;
    bool pool_spans_stale{true};
    std::unordered_map<u64, VirtualPage> virtual_pages;
    std::unordered_map<u64, Buffer> buffers;
    std::unordered_map<u64, Queue> queue_builders;
    std::unordered_map<u64, Queue> queues;
    std::unordered_map<u64, Window> window_builders;
    std::unordered_map<u64, Window> windows;
    std::unordered_map<u64, Sync> syncs;
    std::unordered_map<u64, Event> event_builders;
    std::unordered_map<u64, Event> events;
    // The command buffers, in parts by address. If a call also needs session state, acquire
    // the session mutex first. A part's lock is shared for recording into one command buffer
    // and exclusive for everything else done to a buffer of that part.
    struct CommandShard {
        Common::ReaderMutex mutex;
        std::unordered_map<u64, CommandBuffer> buffers;
    };
    static constexpr size_t CommandShards = 16;
    std::array<CommandShard, CommandShards> command_shards;
    CommandShard& CommandShardOf(u64 command_buffer) {
        return command_shards[(command_buffer * 0x9E3779B97F4A7C15ULL) >> 60];
    }
    std::unordered_map<u64, std::vector<CallbackContinuation>> callback_continuations;
    BananzaNvn2Commands::Recorder recorder;
    Resources resources;
    // Textures an earlier submission drew, cleared or copied into: their bytes come from replay.
    std::unordered_set<u64> gpu_written_textures;
    // Hash of each CPU-filled texture when it was last sent, and the frame it was checked in.
    std::unordered_map<u64, TextureCheck> texture_checks;
    using TextureReplayKey = std::tuple<u64, u32, u32, u32, u32, u32, u32, u32, u64>;
    std::map<TextureReplayKey, std::shared_ptr<BananzaNvn2Commands::TextureReplayState>> texture_replay_states;
    std::map<u64, TextureCopyStat> texture_copy_stats;
    std::vector<u8> texture_scratch;
    // The textures the last submit hashed: the next one starts their hashes ahead.
    struct HashedTexture {
        u64 serial;
        u64 guest;
        u64 size;
    };
    std::vector<HashedTexture> hashed_textures;
    u64 present_serial{1};
    // Counts nvnMemoryPoolMapVirtual calls: the way a pool without CPU access gets new bytes.
    u64 virtual_map_generation{};
    std::map<std::string, MissingCall, std::less<>> missing;
    std::optional<RecordingStats> recording_stats;
    std::optional<CallProfile> call_profile;
    u64 last_recording_serial{};
    u64 last_reported_recording_serial{};
    std::chrono::steady_clock::time_point next_report =
        std::chrono::steady_clock::now() + std::chrono::seconds{10};
    u64 next_gpu_address = 0x1000000000ULL;
    u64 pool_serial{};
    u64 flush_generation{1};
    // Guest ranges the GPU has written: their bytes change without a flush.
    std::map<u64, u64> gpu_written_guest;
    void NoteFlush(const Pool& pool, u64 offset, u64 size);
    u64 accepted_submissions{};
    u64 accepted_commands{};
    u64 rejected_submissions{};
    bool logged_submit{};
    // Declared last: its thread works on this object and stops before anything else goes.
    std::unique_ptr<SubmitLane> submit_lane;
};

} // namespace Core::BananzaNvn2Hle
