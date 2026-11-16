// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#if defined(_M_X64) || defined(__x86_64__)
#include <iterator>
#include <cstring>
#include <immintrin.h>
#endif
#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <chrono>
#include <mutex>
#include <shared_mutex>
#include <optional>
#include <set>
#include <span>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "common/reader_mutex.h"
#include "common/thread_worker.h"
#include "common/common_types.h"
#include "common/container/unordered_map.h"
#include "core/arm/bananza_nvn2_resources.h"

namespace Core::Memory {
class Memory;
}

namespace Core::BananzaNvn2Commands {

struct Recording;

enum class ReplayOp : u8 {
    Unsupported,
    SetRenderTargets,
    ClearColor,
    ClearColorSigned,
    ClearColorUnsigned,
    ClearDepthStencil,
    CopyTexture,
    Barrier,
    BindProgram,
    BindUniformBuffer,
    BindUniformBuffers,
    BindTextures,
    BindTexture,
    BindTexturesDeferred,
    SetTexturePool,
    SetSamplerPool,
    PushGraphicsConstants,
    UpdateUniformBuffer,
    BindBlendState,
    BindColorState,
    BindChannelMask,
    BindDepthStencil,
    BindPolygon,
    SetBlendColor,
    SetAlphaRef,
    SetDepthRange,
    SetStencilRef,
    SetStencilMask,
    SetStencilValueMask,
    StencilCullCriteria,
    BindVertexBuffer,
    BindVertexAttribState,
    BindVertexStreamState,
    SetViewport,
    SetScissor,
    Draw,
    DrawIndexed,
    DrawIndexedBaseVertex,
    DrawIndexedInstanced,
    DrawInstanced,
    DrawIndirect,
    DrawIndexedIndirect,
    Dispatch,
    SignalEvent,
    WaitEvent,
    ReportCounter,
    ResetCounter,
    SetRenderEnable,
    SetRenderEnableConditional,
    BindStorageBuffer,
    BindStorageBuffers,
    BindImage,
    PushComputeConstants,
    CopyBufferToTexture,
    SetCopyRowStride,
    SetDepthClamp,
    SetRasterizerDiscard,
    SetPolygonOffsetClamp,
    SetViewports,
    SetScissors,
    SetDepthRanges,
};

// Time the GPU thread spent replaying since the last present, for the status bar.
inline std::atomic<u64> g_replay_busy_ns{};

// The arguments of an operation: up to four in place, more on the heap.
class OpArgs {
public:
    OpArgs() = default;
    OpArgs(OpArgs&& other) noexcept : count{other.count} {
        std::memcpy(&storage, &other.storage, sizeof(storage));
        other.count = 0;
    }
    OpArgs& operator=(OpArgs&& other) noexcept {
        if (this != &other) {
            Free();
            count = other.count;
            std::memcpy(&storage, &other.storage, sizeof(storage));
            other.count = 0;
        }
        return *this;
    }
    OpArgs(const OpArgs&) = delete;
    OpArgs& operator=(const OpArgs&) = delete;
    ~OpArgs() {
        Free();
    }

    template <typename It>
    void assign(It first, It last) {
        Free();
        const size_t size = static_cast<size_t>(std::distance(first, last));
        u64* const target = size > Inline ? (storage.heap = new u64[size]) : storage.values;
        std::copy(first, last, target);
        count = static_cast<u32>(size);
    }
    size_t size() const {
        return count;
    }
    bool empty() const {
        return count == 0;
    }
    const u64* data() const {
        return count > Inline ? storage.heap : storage.values;
    }
    const u64& operator[](size_t index) const {
        return data()[index];
    }
    const u64* begin() const {
        return data();
    }
    const u64* end() const {
        return data() + count;
    }
    const u64& front() const {
        return data()[0];
    }
    const u64& back() const {
        return data()[count - 1];
    }

private:
    static constexpr size_t Inline = 4;

    void Free() {
        if (count > Inline) {
            delete[] storage.heap;
        }
        count = 0;
    }

    union {
        u64 values[Inline];
        u64* heap;
    } storage{};
    u32 count{};
};

// What few operations carry.
struct OperationExtra {
    std::vector<u8> payload;
    std::vector<BananzaNvn2Hle::ResourceTextureInfo> textures;
    std::shared_ptr<const BananzaNvn2Hle::ResourceProgramInfo> program;
};

// One cache line: a frame of Bananza records about 40,000 of these, and the submit scan and
// the replay each read them all.
struct Operation {
    u32 id{};
    ReplayOp replay{};
    OpArgs args;
    // The called recording; Recording::called keeps it alive.
    const Recording* nested{};
    std::unique_ptr<OperationExtra> extra;

    const std::vector<u8>& Payload() const {
        static const std::vector<u8> none;
        return extra ? extra->payload : none;
    }
    const std::vector<BananzaNvn2Hle::ResourceTextureInfo>& Textures() const {
        static const std::vector<BananzaNvn2Hle::ResourceTextureInfo> none;
        return extra ? extra->textures : none;
    }
    const std::shared_ptr<const BananzaNvn2Hle::ResourceProgramInfo>& Program() const {
        static const std::shared_ptr<const BananzaNvn2Hle::ResourceProgramInfo> none;
        return extra ? extra->program : none;
    }
};
static_assert(sizeof(Operation) == 64);

// What the submit scan needs of a recording, in order: made once, when the recording ends.
struct ScanEntry {
    enum class Kind : u8 {
        Vertex,  // slot, a address, b size
        Uniform, // stage, slot, a address, b size
        Storage, // the same
        Texture, // stage, slot, a handle
        Image,   // the same
        Index,   // a address, b bytes: the index data of a draw
        Draw,    // an indexed draw after which targets, buffers or descriptors may be pending
        Op,      // a: the Operation, for the general code
        Call,    // a: the Recording called here
    };
    Kind kind{};
    u8 stage{};
    u16 slot{};
    u64 a{};
    u64 b{};
};
static_assert(sizeof(ScanEntry) == 24);
// The largest range the scan takes; an indexed draw over more is left to the general code.
inline constexpr u64 ScanMaxRange = 8ULL << 20;

struct Recording;
struct Submission;
// Fills recording.scan from its own operations, with a Call entry for each recording it
// calls, and the totals of its flattened lists; the called recordings have theirs already.
void BuildScanEntries(Recording& recording);
// Fills submission.flats: for each recording submitted, its operations in the order they
// run, with the called recordings' in place of the calls.
void FlattenRecordings(Submission& submission);

struct Recording {
    std::vector<Operation> operations;
    // Its own operations as the submit scan reads them; a called recording is a Call entry.
    std::vector<ScanEntry> scan;
    // How many entries and operations it has once the called recordings are put in place.
    u64 scan_total{};
    u64 flat_total{};
    // The recordings its operations call: a called recording outlives its handle this way.
    std::vector<std::shared_ptr<const Recording>> called;
    u64 serial{};
    u64 command_buffer{};
    u64 draw_count{};
    u64 called_commands{};
    u64 called_draws{};
    u64 copied_commands{};
    u64 copied_draws{};
    u64 command_count{};
    u64 payload_bytes{};
    u32 nesting_depth{1};
};

// For a loop over operations: starts loading the recordings the next two operations call. They
// are small and scattered, so without this each one is a cache miss when the loop reaches it.
inline void PrefetchCalled(const std::vector<Operation>& operations, size_t index) {
#if defined(_M_X64) || defined(__x86_64__)
    if (index + 2 < operations.size()) {
        if (const Recording* const ahead = operations[index + 2].nested) {
            _mm_prefetch(reinterpret_cast<const char*>(ahead), _MM_HINT_T0);
        }
    }
    if (index + 1 < operations.size()) {
        if (const Recording* const next = operations[index + 1].nested;
            next && !next->operations.empty()) {
            const char* const data = reinterpret_cast<const char*>(next->operations.data());
            _mm_prefetch(data, _MM_HINT_T0);
            _mm_prefetch(data + 64, _MM_HINT_T0);
            _mm_prefetch(data + 128, _MM_HINT_T0);
        }
    }
#else
    static_cast<void>(operations);
    static_cast<void>(index);
#endif
}

// For a loop over a recording's flat list: starts loading the operation eight steps ahead and
// the rare part of the one four steps ahead (its operation is loaded by then).
inline void PrefetchAhead(const std::vector<const Operation*>& flat, size_t index) {
#if defined(_M_X64) || defined(__x86_64__)
    if (index + 8 < flat.size()) {
        _mm_prefetch(reinterpret_cast<const char*>(flat[index + 8]), _MM_HINT_T0);
    }
    if (index + 4 < flat.size()) {
        if (const OperationExtra* const extra = flat[index + 4]->extra.get()) {
            _mm_prefetch(reinterpret_cast<const char*>(extra), _MM_HINT_T0);
        }
    }
#else
    static_cast<void>(flat);
    static_cast<void>(index);
#endif
}

struct State {
    bool recording{};
    u64 command_size{};
    u64 command_used{};
    u64 control_base{};
    u64 control_size{};
    u64 control_used{};
};

// Where the bytes of a CPU-filled texture are in guest memory, and their hash at submit.
struct TextureSource {
    u64 hash{};
    std::vector<std::pair<u64, u64>> segments;
};

struct TextureReplayState {
    std::atomic<bool> gpu_written{};
};

// Submissions in replay order. A submission retires once its GPU work and the results the
// guest reads back (storage write-back, counter reports, the events behind them) are done.
class Retirement {
public:
    u64 NextSequence() {
        return next.fetch_add(1, std::memory_order_relaxed) + 1;
    }
    bool IsRetired(u64 sequence) const {
        return retired.load(std::memory_order_acquire) >= sequence;
    }
    // The sync of the submission may signal. With writes_done false its last storage ranges
    // are still on their way to guest memory; RetireWrites follows.
    void Retire(u64 sequence, bool writes_done = true);
    void Wait(u64 sequence);
    // The replay of the submission has ended: guest memory is not read for it any more.
    bool IsReplayed(u64 sequence) const {
        return replayed.load(std::memory_order_acquire) >= sequence;
    }
    void MarkReplayed(u64 sequence);
    void WaitReplayed(u64 sequence);
    // Every range the GPU wrote for the submission is back in guest memory.
    bool WritesDone(u64 sequence) const {
        return written.load(std::memory_order_acquire) >= sequence;
    }
    void RetireWrites(u64 sequence);
    void WaitWrites(u64 sequence);
    // A renderer that retires submissions exists; earlier submissions count as retired.
    void Start();
    // Releases waiters when the renderer that retires submissions goes away.
    void Stop();
    // Called by Stop before anything else: work that still hands submissions to the renderer
    // finishes first.
    void SetDrain(std::function<void()> drain_);

private:
    std::atomic<u64> next{};
    std::atomic<u64> retired{};
    std::atomic<u64> written{};
    std::atomic<u64> replayed{};
    std::mutex mutex;
    std::condition_variable cv;
    std::function<void()> drain;
    bool stopped{};
};

Retirement& GetRetirement();

// The submissions the timeline trace follows: 32 in a row out of every 1024.
inline bool InTimelineWindow(u64 sequence) {
    return (sequence & 0x3ff) < 32;
}

inline u64 HostNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

// Memory for the copies one submission makes of bound ranges. The blocks go back to a shared
// pool when the submission is destroyed, so their pages stay mapped from frame to frame.
class SnapshotBlocks {
public:
    static constexpr u64 BlockBytes = 2ULL << 20;

    SnapshotBlocks() = default;
    SnapshotBlocks(SnapshotBlocks&& other) noexcept
        : blocks{std::move(other.blocks)}, large{std::move(other.large)}, used{other.used} {
        other.blocks.clear();
        other.used = BlockBytes;
    }
    SnapshotBlocks& operator=(SnapshotBlocks&& other) noexcept {
        if (this != &other) {
            Release();
            blocks = std::move(other.blocks);
            large = std::move(other.large);
            used = other.used;
            other.blocks.clear();
            other.used = BlockBytes;
        }
        return *this;
    }
    ~SnapshotBlocks() {
        Release();
    }

    // Uninitialized, 64-byte aligned, valid for the life of this object.
    u8* Allocate(u64 size);

private:
    void Release();

    std::vector<std::unique_ptr<u8[]>> blocks;
    std::vector<std::unique_ptr<u8[]>> large;
    u64 used{BlockBytes};
};

struct Submission : std::enable_shared_from_this<Submission> {
    static constexpr bool ExecutesGpuWork = false;

    u64 sequence{};
    u64 queue{};
    u64 submitted_ns{};
    // Host steady clock at the submit call and when the replay was queued (timeline trace).
    u64 host_submit_ns{};
    u64 host_queued_ns{};
    // The newest earlier submission whose GPU work may write a range this one passes by
    // reference: the replay waits for its writes before it reads guest memory.
    u64 after_writes{};
    bool depth_minus_one_to_one{};
    bool origin_lower_left{};
    std::vector<u64> handles;
    std::vector<std::shared_ptr<const Recording>> recordings;
    // For each of the recordings: the operations in the order they run, the called
    // recordings' in place of their calls (FlattenRecordings).
    std::vector<std::vector<const Operation*>> flats;
    u64 command_count{};
    u64 payload_bytes{};
    // Bytes of a bound range: a copy made at submit (in snapshot_blocks), or guest memory
    // itself for ranges the game must keep until the submission's sync signals.
    struct BufferBytes {
        const u8* data{};
        bool owned{};
        // Nonzero when the bytes change only with these: the backing pool's serial and the
        // newest flush over the range's pages. The replay keeps such a range in VRAM.
        u64 fixed_pool{};
        u64 fixed_stamp{};
        // The copy may still run when the replay starts: wait for late_copies before reading.
        bool late{};
        // With fixed_pool: where the range lies in that pool and the pool's storage, for the
        // replay's mirror of the pool. A size of zero means the range cannot be mirrored.
        u64 fixed_offset{};
        u64 fixed_pool_guest{};
        u64 fixed_pool_size{};
        // Nonzero: a uniform range whose pages are in mirror_pages; the serial of its pool
        // (fixed_offset and the two fields after it hold for it as well).
        u64 staged_pool{};
    };
    // A run of a pool's bytes as they were at submit (staged uniform ranges that touch,
    // merged), for the replay's mirror of the pool.
    struct MirrorPage {
        u64 pool{};
        u64 pool_size{};
        // Where the run starts in the pool.
        u64 offset{};
        const u8* data{};
        u64 bytes{};
    };
    std::vector<MirrorPage> mirror_pages;
    // A page could not be read: the staged ranges are uploaded one by one instead.
    bool mirror_pages_incomplete{};
    // The copies of large ranges that are still running when the replay is queued.
    struct CopyState {
        std::atomic<u32> left{};
        std::mutex mutex;
        std::condition_variable cv;

        void Done() {
            if (left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::scoped_lock lock{mutex};
                cv.notify_all();
            }
        }
        void Wait() {
            if (left.load(std::memory_order_acquire) == 0) {
                return;
            }
            std::unique_lock lock{mutex};
            cv.wait(lock, [this] { return left.load(std::memory_order_acquire) == 0; });
        }
    };
    std::shared_ptr<CopyState> late_copies;
    void WaitLate(const BufferBytes& entry) const {
        if (entry.late && late_copies) {
            late_copies->Wait();
        }
    }
    // In shards, so that several threads can fill the table at submit: a range belongs to one.
    static constexpr size_t SnapshotShards = 8;
    std::array<Common::unordered_map<std::pair<u64, u64>, BufferBytes>, SnapshotShards>
        buffer_snapshots;
    std::array<SnapshotBlocks, SnapshotShards> snapshot_blocks;
    static size_t SnapshotShard(std::pair<u64, u64> range) {
        return static_cast<size_t>((range.first >> 4) ^ (range.first >> 14) ^ range.second) %
               SnapshotShards;
    }
    const BufferBytes* FindSnapshot(std::pair<u64, u64> range) const {
        const auto& shard = buffer_snapshots[SnapshotShard(range)];
        const auto found = shard.find(range);
        return found != shard.end() ? &found->second : nullptr;
    }
    BufferBytes* FindSnapshot(std::pair<u64, u64> range) {
        auto& shard = buffer_snapshots[SnapshotShard(range)];
        const auto found = shard.find(range);
        return found != shard.end() ? &found->second : nullptr;
    }
    bool HasSnapshot(std::pair<u64, u64> range) const {
        return FindSnapshot(range) != nullptr;
    }
    size_t SnapshotCount() const {
        size_t count = 0;
        for (const auto& shard : buffer_snapshots) {
            count += shard.size();
        }
        return count;
    }
    // A storage range's guest address and its bytes as buffer_snapshots had them when the
    // range was entered (the table's entries move, so not a pointer to one).
    struct StorageAddress {
        u64 guest{};
        const u8* data{};
        // A copy of the scan's; otherwise guest memory itself.
        bool owned{};
        // The copy may still run: wait for late_copies before reading.
        bool late{};
    };
    std::map<std::pair<u64, u64>, StorageAddress> storage_addresses;
    // Zero denotes an unsized binding whose first byte is unmapped at submit.
    std::map<u64, u64> unsized_storage_sizes;
    std::map<u64, u64> indirect_index_sizes;
    std::map<std::tuple<u64, u64, u64>, std::pair<u64, u64>> indirect_index_slices;
    Common::unordered_map<u64, u64> indirect_reset_addresses;
    std::map<std::pair<u64, u64>, std::vector<u8>> deferred_data;
    std::unordered_map<u64, std::unordered_map<u32, BananzaNvn2Hle::ResourceTextureDescriptor>>
        texture_descriptors;
    std::unordered_map<u64, std::unordered_map<u32, BananzaNvn2Hle::ResourceSamplerInfo>>
        sampler_descriptors;
    // Bytes of the textures that are new or changed since they were last sent.
    std::map<u64, std::vector<u8>> texture_snapshots;
    std::map<u64, TextureSource> texture_sources;
    std::unordered_map<u64, std::shared_ptr<TextureReplayState>> texture_replay_states;
    Memory::Memory* guest_memory{};
    std::map<u64, u64> event_addresses;

    // wait false: the caller reads the bytes later and waits for late_copies itself.
    std::span<const u8> BufferSnapshot(std::pair<u64, u64> range, bool wait = true) const {
        const auto* const found = FindSnapshot(range);
        if (!found) {
            return {};
        }
        if (wait) {
            WaitLate(*found);
        }
        return {found->data, range.second};
    }
};

// One instance belongs to one guest process. Guest pointers are opaque addresses, never host pointers.
class Recorder {
public:
    bool Initialize(u64 command_buffer);
    bool Finalize(u64 command_buffer);
    bool SetCommandMemory(u64 command_buffer, u64 size);
    bool SetControlMemory(u64 command_buffer, u64 guest_base, u64 size);
    bool Begin(u64 command_buffer);

    // Memory charges come from the caller's ABI model, not the host record representation.
    bool Record(u64 command_buffer, u32 op, std::span<const u64> args = {},
                std::span<const u8> payload = {}, u64 command_bytes = 0,
                std::span<const BananzaNvn2Hle::ResourceTextureInfo> textures = {},
                std::shared_ptr<const BananzaNvn2Hle::ResourceProgramInfo> program = {},
                ReplayOp replay = ReplayOp::Unsupported);
    bool AppendHandles(u64 command_buffer, std::span<const u64> handles, u64 command_bytes = 0,
                       bool call = true);
    std::optional<u64> End(u64 command_buffer, u64 control_bytes, u64 guest_handle_offset = 0);
    bool FinalizeHandle(u64 handle);
    std::optional<State> GetState(u64 command_buffer) const;

    // Validates the entire batch and retains its snapshots; it does not execute or complete GPU work.
    std::optional<Submission> Submit(u64 queue, std::span<const u64> handles);

private:
    struct Buffer {
        State state;
        std::vector<Operation> operations;
        std::vector<std::shared_ptr<const Recording>> called;
        u64 command_count{};
        u64 payload_bytes{};
        u64 argument_bytes{};
        u64 draw_count{};
        u64 called_commands{};
        u64 called_draws{};
        u64 copied_commands{};
        u64 copied_draws{};
        u32 nesting_depth{1};
    };

    struct Handle {
        std::shared_ptr<const Recording> recording;
        u64 control_begin{};
        u64 control_end{};
    };

    // For the handles and what follows them below: shared for a lookup (a command buffer
    // that calls another one, thousands of times a frame from every recording thread).
    mutable std::shared_mutex handles_mutex;
    // Frees recordings whose handles are gone: not work for a thread of the game.
    Common::ThreadWorker garbage{1, "NvnFree"};
    // The command buffers, in parts by address. A part's lock is shared while one thread
    // works on its own buffer (NVN allows one thread per command buffer) and exclusive for
    // the calls that add or remove a buffer of that part: the game makes and drops about a
    // hundred a frame, which must not stop every recording thread.
    struct BufferShard {
        mutable Common::ReaderMutex mutex;
        std::unordered_map<u64, Buffer> buffers;
    };
    static constexpr size_t BufferShards = 16;
    mutable std::array<BufferShard, BufferShards> buffer_shards;
    BufferShard& ShardOf(u64 command_buffer) const {
        return buffer_shards[(command_buffer * 0x9E3779B97F4A7C15ULL) >> 60];
    }
    std::unordered_map<u64, Handle> live_handles;
    std::set<std::pair<u64, u64>> handles_by_control_begin;
    u64 max_control_record_bytes{};
    u64 next_handle{1};
    u64 next_recording{1};
    u64 next_submission{1};
};

} // namespace Core::BananzaNvn2Commands
