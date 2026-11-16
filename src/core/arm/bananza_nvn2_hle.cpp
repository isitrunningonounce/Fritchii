// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>
#include <thread>
#include <memory>
#include <set>
#include <unordered_map>
#include <span>

#include "common/alignment.h"
#include "common/cityhash.h"
#include "common/container/unordered_set.h"
#include "common/logging.h"
#include "common/display_hdr.h"
#include "common/settings.h"
#include "common/swap.h"
#include "core/arm/bananza_nvn2_device.h"
#include "core/arm/bananza_nvn2_hle.h"
#include "core/arm/bananza_nvn2_trace.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/memory.h"
#include "core/perf_stats.h"
#include "video_core/gpu.h"
#include "video_core/native_indirect_reset.h"
#include "video_core/nx2_perf.h"
#include "video_core/renderer_base.h"

namespace Core::BananzaNvn2Hle {
namespace {

constexpr u32 CpuNoAccess = 1;
constexpr u32 PhysicalPool = 0x100;
constexpr u32 VirtualPool = 0x200;
constexpr u64 PoolAlignment = 0x10000;

// The hash of a texture's bytes. A large one is hashed in pieces, with the threads of
// `helpers` when given; the result does not depend on who hashed which piece.
u64 TextureHash(const u8* data, u64 size, Common::ThreadWorker* helpers) {
    static constexpr u64 Piece = 1ULL << 20;
    if (size < 2 * Piece) {
        return Common::CityHash64(reinterpret_cast<const char*>(data), size);
    }
    struct Job {
        const u8* data;
        u64 size;
        std::vector<u64> hashes;
        std::atomic<size_t> next{};
        std::atomic<size_t> done{};
    };
    const auto job = std::make_shared<Job>();
    job->data = data;
    job->size = size;
    job->hashes.resize((size + Piece - 1) / Piece);
    // A thread that starts late finds no piece left and does not touch the bytes.
    const auto work = [job] {
        for (;;) {
            const size_t piece = job->next.fetch_add(1, std::memory_order_relaxed);
            if (piece >= job->hashes.size()) {
                return;
            }
            const u64 offset = piece * Piece;
            job->hashes[piece] = Common::CityHash64(
                reinterpret_cast<const char*>(job->data + offset),
                std::min(Piece, job->size - offset));
            job->done.fetch_add(1, std::memory_order_release);
        }
    };
    if (helpers) {
        const size_t count = std::min<size_t>(3, job->hashes.size() - 1);
        for (size_t i = 0; i < count; ++i) {
            // A copy of its own: a task made from `work` itself would only refer to it, and
            // a thread that starts late would find it gone (crashed r118 and r119).
            helpers->QueueWork(decltype(work){work});
        }
    }
    work();
    while (job->done.load(std::memory_order_acquire) < job->hashes.size()) {
        std::this_thread::yield();
    }
    return Common::CityHash64(reinterpret_cast<const char*>(job->hashes.data()),
                              job->hashes.size() * sizeof(u64));
}

// EDEN_NX2_NVN_DIRTY_STATS=1: what a mirror fed by dirty pages would have to move.
struct DirtyStats {
    struct Page {
        u64 hash;
        u64 stamp;
        u64 seen;
    };
    struct Class {
        u64 pages{};
        u64 fresh{};
        u64 changed{};
        u64 changed_unflushed{};
        u64 changed_gpu_written{};
    };
    std::unordered_map<u64, Page> pages;
    // [copied, referenced][CPU_CACHED, CPU_UNCACHED, CPU_NO_ACCESS, no pool]
    std::array<std::array<Class, 4>, 2> classes{};
    u64 submissions{};
    std::mutex mutex;
};
constexpr u64 CommandBufferSize = 0xb0;
constexpr u64 ControlRecordBytes = 64;
constexpr u64 ControlHandleOffset = 16;
constexpr u64 AcquireError = 1;
constexpr u64 SyncAlreadySignaled = 0;
constexpr u64 SyncTimeout = 2;
constexpr u64 SyncFailed = 3;

using BananzaNvn2Commands::ReplayOp;

bool Fits(u64 offset, u64 size, u64 capacity) {
    return offset <= capacity && size <= capacity - offset;
}

bool ValidRange(Memory::Memory& memory, u64 address, u64 size) {
    return address && Fits(address, size, std::numeric_limits<u64>::max()) &&
           memory.IsValidVirtualAddressRange(address, size);
}

std::optional<std::vector<u64>> ReadHandles(Memory::Memory& memory, u64 address, u64 count) {
    if (count > 0x10000 || (count && !ValidRange(memory, address, count * sizeof(u64)))) {
        return std::nullopt;
    }
    std::vector<u64> handles(count);
    if (count && !memory.ReadBlock(address, handles.data(), handles.size() * sizeof(u64))) {
        return std::nullopt;
    }
    if constexpr (std::endian::native == std::endian::big) {
        for (auto& handle : handles) {
            handle = std::bit_cast<u64_le>(handle);
        }
    }
    return handles;
}

bool AppendGuestBytes(Memory::Memory& memory, std::vector<u8>& out, u64 address, u64 size) {
    if (!size) {
        return true;
    }
    if (!ValidRange(memory, address, size)) {
        return false;
    }
    const size_t previous_size = out.size();
    out.resize(previous_size + size);
    return memory.ReadBlock(address, out.data() + previous_size, size);
}

struct ScalarCommand {
    std::string_view name;
    size_t args;
    size_t float_args;
    ReplayOp replay{};
    u64 command_bytes{};
};

constexpr std::array ScalarCommands{
    ScalarCommand{"nvnCommandBufferBarrier", 1, 0, ReplayOp::Barrier},
    ScalarCommand{"nvnCommandBufferBindImage", 3, 0, ReplayOp::BindImage},
    ScalarCommand{"nvnCommandBufferBindStorageBuffer", 4, 0, ReplayOp::BindStorageBuffer},
    ScalarCommand{"nvnCommandBufferBindTexture", 3, 0, ReplayOp::BindTexture},
    ScalarCommand{"nvnCommandBufferBindUniformBuffer", 4, 0, ReplayOp::BindUniformBuffer},
    ScalarCommand{"nvnCommandBufferBindVertexBuffer", 3, 0, ReplayOp::BindVertexBuffer, 24},
    ScalarCommand{"nvnCommandBufferClearDepthStencil", 3, 1, ReplayOp::ClearDepthStencil},
    ScalarCommand{"nvnCommandBufferCopyBufferToBuffer", 4, 0},
    ScalarCommand{"nvnCommandBufferDispatchCompute", 3, 0, ReplayOp::Dispatch},
    ScalarCommand{"nvnCommandBufferDispatchComputeIndirect", 1, 0, ReplayOp::Dispatch},
    ScalarCommand{"nvnCommandBufferDrawArrays", 3, 0, ReplayOp::Draw},
    ScalarCommand{"nvnCommandBufferDrawArraysIndirect", 2, 0, ReplayOp::DrawIndirect},
    ScalarCommand{"nvnCommandBufferDrawArraysInstanced", 5, 0, ReplayOp::DrawInstanced},
    ScalarCommand{"nvnCommandBufferDrawElements", 4, 0, ReplayOp::DrawIndexed, 48},
    ScalarCommand{"nvnCommandBufferDrawElementsBaseVertex", 5, 0, ReplayOp::DrawIndexedBaseVertex, 44},
    ScalarCommand{"nvnCommandBufferDrawElementsIndirect", 4, 0, ReplayOp::DrawIndexedIndirect},
    ScalarCommand{"nvnCommandBufferDrawElementsInstanced", 7, 0, ReplayOp::DrawIndexedInstanced, 44},
    ScalarCommand{"nvnCommandBufferFenceSync", 3, 0},
    ScalarCommand{"nvnCommandBufferReportCounter", 2, 0, ReplayOp::ReportCounter},
    ScalarCommand{"nvnCommandBufferResetCounter", 1, 0, ReplayOp::ResetCounter},
    ScalarCommand{"nvnCommandBufferSetComputeOverlapMode", 1, 0},
    ScalarCommand{"nvnCommandBufferSetCopyRowStride", 1, 0, ReplayOp::SetCopyRowStride},
    ScalarCommand{"nvnCommandBufferSetAlphaRef", 0, 1, ReplayOp::SetAlphaRef},
    ScalarCommand{"nvnCommandBufferSetConservativeRasterEnable", 1, 0},
    ScalarCommand{"nvnCommandBufferSetDepthBounds", 0, 2},
    ScalarCommand{"nvnCommandBufferSetDepthClamp", 1, 0, ReplayOp::SetDepthClamp},
    ScalarCommand{"nvnCommandBufferSetDepthRange", 0, 2, ReplayOp::SetDepthRange},
    ScalarCommand{"nvnCommandBufferSetGeometryProcessingMode", 1, 0},
    ScalarCommand{"nvnCommandBufferSetLineWidth", 0, 1},
    ScalarCommand{"nvnCommandBufferSetPointSize", 0, 1},
    ScalarCommand{"nvnCommandBufferSetPolygonOffsetClamp", 0, 3, ReplayOp::SetPolygonOffsetClamp},
    ScalarCommand{"nvnCommandBufferSetRasterizerDiscard", 1, 0, ReplayOp::SetRasterizerDiscard},
    ScalarCommand{"nvnCommandBufferSetRenderEnable", 1, 0, ReplayOp::SetRenderEnable},
    ScalarCommand{"nvnCommandBufferSetRenderEnableConditional", 2, 0,
                  ReplayOp::SetRenderEnableConditional, 16},
    ScalarCommand{"nvnCommandBufferSetScissor", 4, 0, ReplayOp::SetScissor},
    ScalarCommand{"nvnCommandBufferSetShaderScratchMemory", 3, 0},
    ScalarCommand{"nvnCommandBufferSetStencilMask", 2, 0, ReplayOp::SetStencilMask},
    ScalarCommand{"nvnCommandBufferSetStencilCullCriteria", 3, 0, ReplayOp::StencilCullCriteria},
    ScalarCommand{"nvnCommandBufferSetStencilRef", 2, 0, ReplayOp::SetStencilRef},
    ScalarCommand{"nvnCommandBufferSetStencilValueMask", 2, 0, ReplayOp::SetStencilValueMask},
    ScalarCommand{"nvnCommandBufferSetViewport", 4, 0, ReplayOp::SetViewport},
    ScalarCommand{"nvnCommandBufferSetSamplerPool", 1, 0, ReplayOp::SetSamplerPool},
    ScalarCommand{"nvnCommandBufferSetTexturePool", 1, 0, ReplayOp::SetTexturePool},
};

struct StateBinding {
    std::string_view name;
    size_t bytes;
    ReplayOp replay{};
};

constexpr std::array StateBindings{
    StateBinding{"nvnCommandBufferBindBlendState", 8, ReplayOp::BindBlendState},
    StateBinding{"nvnCommandBufferBindChannelMaskState", 4, ReplayOp::BindChannelMask},
    StateBinding{"nvnCommandBufferBindColorState", 4, ReplayOp::BindColorState},
    StateBinding{"nvnCommandBufferBindDepthStencilState", 8, ReplayOp::BindDepthStencil},
    StateBinding{"nvnCommandBufferBindMultisampleState", 24},
    StateBinding{"nvnCommandBufferBindPolygonState", 4, ReplayOp::BindPolygon},
};

// Time the current call waited for the session mutex, for the call profile.
thread_local u64 profile_lock_wait{};

u64 ElapsedNs(std::chrono::steady_clock::time_point since) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - since)
                                .count());
}

u64 CommandBytes(std::string_view name, const GuestCall& call) {
    if (name == "nvnCommandBufferBindVertexAttribState" ||
        name == "nvnCommandBufferBindVertexStreamState") {
        const u64 count = static_cast<u32>(call.x[1]);
        if (count > 32) {
            return 0;
        }
        const u64 padded = count + (count % 2);
        return name == "nvnCommandBufferBindVertexAttribState" ? 4 * (2 + padded)
                                                             : 4 * (4 + 2 * padded);
    }
    return 0;
}

} // Anonymous namespace

namespace {
bool SyncSubmits();
} // Anonymous namespace

Session::~Session() {
    BananzaNvn2Commands::GetRetirement().SetDrain({});
}

Session::Session(System& system_, const BananzaNvn2Trace::Session& trace, u64 bootstrap_,
                 u64 internal_bootstrap_, u64 callback_return_)
    : bootstrap{bootstrap_}, internal_bootstrap{internal_bootstrap_},
      callback_return{callback_return_}, procedures(trace.Size()), system{system_} {
    // EDEN_NX2_NVN_SYNC_SCAN=1: the scan runs inside the game's submit call, as before.
    if (std::getenv("EDEN_NX2_NVN_SYNC_SCAN") == nullptr && !SyncSubmits()) {
        submit_lane = std::make_unique<SubmitLane>();
        BananzaNvn2Commands::GetRetirement().SetDrain([this] { submit_lane->Wait(); });
    }
    for (size_t index = 0; index < procedures.size(); ++index) {
        const auto name = trace.Name(index);
        auto& procedure = procedures[index];
        procedure.name = name;
        procedure.guest_helper = UseGuestStateHelper(name);
        procedure.float_return = Resources::IsFloatReturn(name);
        if (name == "nvnDeviceGetProcAddress") {
            procedure.handler = Handler::GetProcAddress;
        } else if (name == "nvnDeviceGetTextureHandle") {
            procedure.handler = Handler::GetTextureHandle;
        } else if (name == "nvnMemoryPoolGetFlags") {
            procedure.handler = Handler::GetPoolFlags;
        } else if (name == "nvnBufferGetAddress") {
            procedure.handler = Handler::GetBufferAddress;
        } else if (name == "nvnBufferMap") {
            procedure.handler = Handler::MapBuffer;
        } else if (name.starts_with("nvnDevice")) {
            procedure.handler = Handler::Device;
        } else if (name.starts_with("nvnMemoryPool")) {
            procedure.handler = Handler::Pool;
        } else if (name.starts_with("nvnBuffer")) {
            procedure.handler = Handler::Buffer;
        } else if (name.starts_with("nvnQueue")) {
            procedure.handler = Handler::Queue;
        } else if (name.starts_with("nvnWindow")) {
            procedure.handler = Handler::Window;
        } else if (name.starts_with("nvnSync") || name.starts_with("nvnEvent")) {
            procedure.handler = Handler::SyncEvent;
        } else if (name.starts_with("nvnCommandBuffer")) {
            procedure.handler = Handler::Command;
            procedure.begins_recording = name == "nvnCommandBufferBeginRecording";
            procedure.per_buffer_command = name != "nvnCommandBufferInitialize" &&
                                           name != "nvnCommandBufferFinalize";
            procedure.ends_recording = name == "nvnCommandBufferEndRecording";
            procedure.calls_handles = name == "nvnCommandBufferCallCommands";
            procedure.appends_handles = procedure.calls_handles ||
                                         name == "nvnCommandBufferCopyCommands";
            procedure.binds_deferred_textures = name == "nvnCommandBufferBindTexturesDeferred";
            const auto scalar = std::ranges::find(ScalarCommands, name, &ScalarCommand::name);
            const auto state = std::ranges::find(StateBindings, name, &StateBinding::name);
            if (scalar != ScalarCommands.end()) {
                procedure.scalar_command = &*scalar;
            }
            if (state != StateBindings.end()) {
                procedure.state_binding = &*state;
            }
        }
    }
    if (const char* value = std::getenv("EDEN_NX2_NVN_RECORDING_STATS"); value && *value == '1') {
        recording_stats.emplace();
    }
    if (const char* value = std::getenv("EDEN_NX2_NVN_CALL_PROFILE"); value && *value == '1') {
        call_profile.emplace();
    }
}

bool Session::IsBootstrap(u64 pc) const {
    return pc == bootstrap || pc == internal_bootstrap;
}

bool Session::IsCallbackReturn(u64 pc) const {
    return pc == callback_return;
}

u64 Session::CallbackReturn() const {
    return callback_return;
}

bool Session::UseGuestStateHelper(std::string_view name) {
    // These SDK routines only encode or decode their compact guest state structures.
    if (name.starts_with("nvnDeviceBuilder")) {
        return name == "nvnDeviceBuilderSetDefaults" || name == "nvnDeviceBuilderSetFlags" ||
               name == "nvnDeviceBuilderGetFlags";
    }
    if (name.starts_with("nvnBufferBuilder")) {
        return name == "nvnBufferBuilderSetDefaults" || name == "nvnBufferBuilderSetDevice" ||
               name == "nvnBufferBuilderSetStorage" || name == "nvnBufferBuilderGetDevice" ||
               name == "nvnBufferBuilderGetMemoryPool" ||
               name == "nvnBufferBuilderGetMemoryOffset" || name == "nvnBufferBuilderGetSize";
    }
    if (name.starts_with("nvnMemoryPoolBuilder")) {
        return name == "nvnMemoryPoolBuilderSetDefaults" ||
               name == "nvnMemoryPoolBuilderSetDevice" ||
               name == "nvnMemoryPoolBuilderSetFlags" ||
               name == "nvnMemoryPoolBuilderSetStorage" ||
               name == "nvnMemoryPoolBuilderGetDevice" ||
               name == "nvnMemoryPoolBuilderGetFlags" ||
               name == "nvnMemoryPoolBuilderGetMemory" ||
               name == "nvnMemoryPoolBuilderGetSize";
    }
    if (name == "nvnMemoryPoolGetBufferAddress") {
        return true;
    }
    constexpr std::array prefixes{
        std::string_view{"nvnBlendState"}, std::string_view{"nvnChannelMaskState"},
        std::string_view{"nvnColorState"}, std::string_view{"nvnDepthStencilState"},
        std::string_view{"nvnMultisampleState"}, std::string_view{"nvnPolygonState"},
        std::string_view{"nvnVertexAttribState"}, std::string_view{"nvnVertexStreamState"},
    };
    return std::ranges::any_of(prefixes, [&](auto prefix) { return name.starts_with(prefix); });
}

bool Session::UsesGuestStateHelper(size_t index) const {
    return procedures[index].guest_helper;
}

std::optional<std::pair<size_t, size_t>> Session::DirectCallArguments(size_t index) const {
    const auto& procedure = procedures[index];
    const auto* scalar = static_cast<const ScalarCommand*>(procedure.scalar_command);
    if (scalar && !scalar->command_bytes) {
        return std::pair{scalar->args + 1, scalar->float_args};
    }
    if (procedure.appends_handles) {
        return std::pair<size_t, size_t>{3, 0};
    }
    if (procedure.binds_deferred_textures) {
        return std::pair<size_t, size_t>{5, 0};
    }
    if (procedure.state_binding) {
        return std::pair<size_t, size_t>{2, 0};
    }
    return std::nullopt;
}

bool Session::ReturnsFloat(size_t index) const {
    return procedures[index].float_return;
}

bool Session::ProfilesCalls() const {
    return call_profile.has_value();
}

void Session::ProfileCall(size_t index, std::string_view name, u64 nanoseconds, bool continuation,
                          u64 thread) {
    auto& profile = *call_profile;
    std::scoped_lock lock{profile.mutex};
    auto& entry = profile.entries[index];
    entry.name = name;
    entry.calls += !continuation;
    entry.nanoseconds += nanoseconds;
    entry.lock_wait += std::exchange(profile_lock_wait, 0);
    auto& stats = profile.threads[thread];
    stats.calls += !continuation;
    stats.nanoseconds += nanoseconds;
}

std::optional<GuestCallback> Session::BeginCallback(u64 thread, size_t procedure,
                                                   const GuestCall& call, u64 link) {
    const auto& route = procedures[procedure];
    if (route.handler != Handler::Command) {
        return std::nullopt;
    }
    const auto name = route.name;
    const auto* scalar = static_cast<const ScalarCommand*>(route.scalar_command);
    const u64 command_bytes = scalar ? scalar->command_bytes : CommandBytes(name, call);
    const bool beginning = route.begins_recording;
    const bool ending = route.ends_recording;
    if (!beginning && !ending && !command_bytes) {
        return std::nullopt;
    }
    const u64 address = call.x[0];
    auto& shard = CommandShardOf(address);
    std::shared_lock lock{shard.mutex};
    const auto buffer = shard.buffers.find(address);
    const auto state = recorder.GetState(address);
    if (buffer == shard.buffers.end() || !state || !buffer->second.callback) {
        return std::nullopt;
    }
    if (command_bytes) {
        if (!state->recording || !buffer->second.command_callback ||
            Fits(state->command_used, command_bytes, state->command_size)) {
            return std::nullopt;
        }
    } else {
        const u64 required = ending ? ControlRecordBytes : 16;
        if (state->recording != ending || !buffer->second.control_callback ||
            Fits(state->control_used, required, state->control_size)) {
            return std::nullopt;
        }
    }
    const u64 callback = buffer->second.callback;
    const u64 callback_data = buffer->second.callback_data;
    lock.unlock();
    std::unique_lock exclusive{callback_mutex};
    auto& continuations = callback_continuations[thread];
    if (continuations.size() >= 8) {
        exclusive.unlock();
        std::scoped_lock session_lock{mutex};
        NoteMissing("nvnCommandBuffer control callback nesting");
        return std::nullopt;
    }
    if ((callback & 3) != 0) {
        exclusive.unlock();
        std::scoped_lock session_lock{mutex};
        NoteMissing("nvnCommandBuffer unaligned memory callback");
        return std::nullopt;
    }
    continuations.push_back({procedure, call, link});
    return GuestCallback{callback,
                         {address, command_bytes ? 0U : 1U, command_bytes ? 512U : 256U,
                          callback_data}};
}

std::optional<CallbackContinuation> Session::EndCallback(u64 thread) {
    std::scoped_lock lock{callback_mutex};
    const auto it = callback_continuations.find(thread);
    if (it == callback_continuations.end()) {
        return std::nullopt;
    }
    const auto continuation = it->second.back();
    it->second.pop_back();
    if (it->second.empty()) {
        callback_continuations.erase(it);
    }
    return continuation;
}

namespace {
bool WaitInvalidate();
} // Anonymous namespace

u64 Session::Dispatch(const BananzaNvn2Trace::Session& procs, Memory::Memory& memory, size_t index,
                      const GuestCall& call,
                      std::chrono::steady_clock::time_point* present_deadline) {
    const auto& procedure = procedures[index];
    const std::string_view name = procedure.name;
    const auto handler = procedure.handler;
    const auto& x = call.x;
    if (handler == Handler::GetProcAddress) {
        return x[1] ? procs.Resolve(memory.ReadCString(x[1], 96)) : 0;
    }
    if (handler == Handler::GetTextureHandle) {
        return (x[1] & 0xfffff) | ((x[2] & 0xfff) << 20);
    }
    if (handler == Handler::GetPoolFlags || handler == Handler::GetBufferAddress ||
        handler == Handler::MapBuffer) {
        const auto lock_start = call_profile ? std::chrono::steady_clock::now()
                                             : std::chrono::steady_clock::time_point{};
        std::shared_lock lock{pool_buffer_mutex};
        if (call_profile) {
            profile_lock_wait += ElapsedNs(lock_start);
        }
        if (handler == Handler::GetPoolFlags) {
            if (const auto pool = pools.find(x[0]); pool != pools.end()) {
                return pool->second.flags;
            }
        } else if (const auto buffer = buffers.find(x[0]); buffer != buffers.end()) {
            if (const auto pool = pools.find(buffer->second.pool);
                pool != pools.end() && pool->second.gpu_address == buffer->second.pool_address) {
                if (handler == Handler::GetBufferAddress) {
                    return buffer->second.pool_address + buffer->second.offset;
                }
                return pool->second.flags & CpuNoAccess ? 0 : pool->second.storage + buffer->second.offset;
            }
        }
    }
    if (procedure.scalar_command || procedure.appends_handles || procedure.binds_deferred_textures ||
        procedure.state_binding) {
        const auto start = call_profile ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
        std::shared_lock lock{CommandShardOf(x[0]).mutex};
        if (call_profile) {
            profile_lock_wait += ElapsedNs(start);
        }
        const auto result = RecordCommand(memory, index, call);
        if (result && *result) {
            return 0;
        }
        lock.unlock();
        std::scoped_lock session_lock{mutex};
        if (result) {
            NoteMissing(fmt::format("{} recording or memory state", name));
        } else {
            NoteMissing(name);
        }
        return 0;
    }
    if (handler == Handler::Command && procedure.per_buffer_command && !recording_stats) {
        const auto start = call_profile ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
        std::shared_lock lock{CommandShardOf(x[0]).mutex};
        std::shared_lock resource_lock{resource_mutex};
        std::shared_lock pool_lock{pool_buffer_mutex};
        if (call_profile) {
            profile_lock_wait += ElapsedNs(start);
        }
        if (const auto result = HandleCommands(memory, index, name, call)) {
            return *result;
        }
        NoteMissing(name);
        return 0;
    }
    if (handler == Handler::Command && !recording_stats) {
        // Initialize and Finalize: the command buffers of one part change, nothing else of
        // the session, so they do not wait for a submit that holds the session lock.
        std::scoped_lock command_lock{CommandShardOf(x[0]).mutex};
        if (const auto result = HandleCommands(memory, index, name, call)) {
            return *result;
        }
        NoteMissing(name);
        return 0;
    }
    // Calls that touch only the buffer table, or nothing: they do not wait for a submit that
    // holds the session lock (the game's worker threads make about 80 of them a frame while
    // the camera moves).
    if (handler == Handler::Buffer && name == "nvnBufferFinalize") {
        std::unique_lock buffer_lock{pool_buffer_mutex};
        buffers.erase(x[0]);
        return 0;
    }
    if (handler == Handler::Buffer && name == "nvnBufferInitialize") {
        return InitializeBuffer(memory, call);
    }
    if ((handler == Handler::Buffer || handler == Handler::Pool) && !WaitInvalidate() &&
        (name == "nvnBufferInvalidateMappedRange" ||
         name == "nvnMemoryPoolInvalidateMappedRange")) {
        return 0;
    }
    if (submit_lane) {
        if (handler == Handler::Queue && name == "nvnQueuePresentTexture") {
            return PresentAsync(call, present_deadline);
        }
        // A fence names the last submission, which is known when its submit returns; a call
        // with an unknown queue or sync goes on to be reported below.
        if (handler == Handler::Queue && name == "nvnQueueFenceSync" && FenceSyncAsync(call)) {
            return 0;
        }
        // A flush only stamps pages for a later scan: noted here, applied under the session
        // lock by whoever takes it next. The game makes dozens a frame while the camera
        // turns, from the threads that work on the next frame while the scan runs.
        if (handler == Handler::Pool && name == "nvnMemoryPoolFlushMappedRange") {
            std::scoped_lock flush_lock{flush_mutex};
            pending_flushes.push_back({x[0], x[1], x[2]});
            flushes_pending.store(true, std::memory_order_release);
            return 0;
        }
        if (handler == Handler::Buffer && name == "nvnBufferFlushMappedRange") {
            std::shared_lock buffer_lock{pool_buffer_mutex};
            if (const auto buffer = buffers.find(x[0]);
                buffer != buffers.end() && Fits(x[1], x[2], buffer->second.size)) {
                std::scoped_lock flush_lock{flush_mutex};
                pending_flushes.push_back({buffer->second.pool, buffer->second.offset + x[1], x[2]});
                flushes_pending.store(true, std::memory_order_release);
                return 0;
            }
        }
        // The game acquires the next image right after it presents one.
        if (handler == Handler::Window && name == "nvnWindowAcquireTexture" &&
            AcquireAsync(memory, call)) {
            return 0;
        }
        if (handler == Handler::Queue &&
            (name == "nvnQueueFlush" || name == "nvnQueueSetDebugLabel")) {
            std::scoped_lock frame_lock{frame_mutex};
            if (queues.contains(x[0])) {
                return 0;
            }
        }
        // These read or change what a submit leaves or what its scan reads (pending GPU
        // writes, pools and textures, the queue's sequence): after the scans queued before.
        if (handler == Handler::Pool || handler == Handler::Buffer ||
            handler == Handler::Resource || handler == Handler::Queue ||
            (handler == Handler::SyncEvent && name.starts_with("nvnSync"))) {
            static const bool no_sync_wait = std::getenv("EDEN_NX2_NVN_NO_SYNC_WAIT") != nullptr;
            if (!(no_sync_wait && name == "nvnSyncWait")) {
                const auto lane_start = call_profile ? std::chrono::steady_clock::now()
                                                     : std::chrono::steady_clock::time_point{};
                submit_lane->Wait();
                if (call_profile) {
                    profile_lock_wait += ElapsedNs(lane_start);
                }
            }
        }
    }
    const auto lock_start = call_profile ? std::chrono::steady_clock::now()
                                         : std::chrono::steady_clock::time_point{};
    std::unique_lock lock{mutex};
    if (call_profile) {
        profile_lock_wait += ElapsedNs(lock_start);
    }
    ApplyPendingFlushes();
    if (const auto now = std::chrono::steady_clock::now(); now >= next_report) {
        ReportMissing(false);
        ReportRecordingStats();
        ReportCallProfile();
        next_report = now + std::chrono::seconds{10};
    }
    std::optional<u64> result;
    switch (handler) {
    case Handler::Device:
        result = HandleDevice(memory, name, call);
        break;
    case Handler::Pool:
        result = HandlePool(memory, name, call);
        break;
    case Handler::Buffer:
        result = HandleBuffer(memory, name, call);
        break;
    case Handler::Queue: {
        // The set of queues changes: not while a present reads it.
        std::unique_lock frame_lock{frame_mutex, std::defer_lock};
        if (name == "nvnQueueInitialize" || name == "nvnQueueFinalize") {
            frame_lock.lock();
        }
        result = HandleQueue(memory, name, call, lock, present_deadline);
        break;
    }
    case Handler::Window: {
        std::scoped_lock frame_lock{frame_mutex};
        result = HandleWindow(memory, name, call);
        break;
    }
    case Handler::SyncEvent:
        result = HandleSyncEvent(memory, name, call, lock);
        break;
    case Handler::Command: {
        std::scoped_lock command_lock{CommandShardOf(x[0]).mutex};
        result = HandleCommands(memory, index, name, call);
        break;
    }
    case Handler::Resource: {
        std::unique_lock resource_lock{resource_mutex};
        result = resources.Dispatch(
            memory, name, call, [&](u64 device) { return devices.contains(device); },
            [&](u64 address) { return GetPool(address); },
            [&](u64 address, u64 size) { return ResolveGpuRange(address, size); });
        break;
    }
    case Handler::GetProcAddress:
    case Handler::GetTextureHandle:
    case Handler::GetPoolFlags:
    case Handler::GetBufferAddress:
    case Handler::MapBuffer:
        break;
    }
    if (result) {
        return *result;
    }
    NoteMissing(name);
    if (name == "nvnWindowAcquireTexture" || name == "nvnQueueAcquireTexture") {
        return AcquireError;
    }
    return name == "nvnSyncWait" ? SyncFailed : 0;
}

std::optional<u64> Session::HandleDevice(Memory::Memory& memory, std::string_view name,
                                        const GuestCall& call) {
    const auto& x = call.x;
    if (name == "nvnDeviceGetInteger") {
        const auto value = BananzaNvn2Device::GetDeviceInteger(static_cast<u32>(x[1]));
        if (!value) {
            NoteMissing(fmt::format("{} property {:#x}", name, static_cast<u32>(x[1])));
        }
        if (ValidRange(memory, x[2], sizeof(u32))) {
            memory.Write32(x[2], static_cast<u32>(value.value_or(0)));
        }
    } else if (name == "nvnDeviceInitialize") {
        if (!ValidRange(memory, x[0], 12) || !ValidRange(memory, x[1], 8)) {
            return 0;
        }
        const u32 flags = memory.Read32(x[1] + 4);
        devices.insert_or_assign(x[0], Device{.flags = flags});
        memory.Write64(x[0], 0);
        memory.Write32(x[0] + 8, flags);
        LOG_INFO(Core_ARM, "Bananza NVN 2 HLE initialized a device");
        return 1;
    } else if (name == "nvnDeviceFinalize") {
        devices.erase(x[0]);
    } else if (name == "nvnDeviceSetWindowOriginMode") {
        if (const auto it = devices.find(x[0]); it != devices.end()) {
            it->second.origin = static_cast<u32>(x[1]);
            memory.Write32(x[0], it->second.origin);
        }
    } else if (name == "nvnDeviceSetDepthMode") {
        if (const auto it = devices.find(x[0]); it != devices.end()) {
            it->second.depth_mode = static_cast<u32>(x[1]);
            memory.Write32(x[0] + 4, it->second.depth_mode);
        }
    } else if (name == "nvnDeviceGetWindowOriginMode" || name == "nvnDeviceGetDepthMode" ||
               name == "nvnDeviceGetFlags") {
        const auto device = devices.find(x[0]);
        if (device == devices.end()) {
            return std::nullopt;
        }
        if (name == "nvnDeviceGetFlags") {
            return device->second.flags;
        }
        return name == "nvnDeviceGetWindowOriginMode" ? device->second.origin
                                                      : device->second.depth_mode;
    } else if (name == "nvnDeviceFinalizeCommandHandle") {
        recorder.FinalizeHandle(x[1]);
    } else if (name == "nvnDeviceGetSeparateSamplerHandle") {
        return static_cast<u64>(static_cast<s64>(static_cast<s32>(x[1]))) << 20;
    } else if (name == "nvnDeviceGetImageHandle" || name == "nvnDeviceGetSeparateTextureHandle" ||
               name == "nvnDeviceGetTexelFetchHandle") {
        return static_cast<u64>(static_cast<s64>(static_cast<s32>(x[1])));
    } else if (name == "nvnDeviceGetTimestampInNanoseconds") {
        // The emulator's NVN 2 counter records already carry nanoseconds.
        if (!ValidRange(memory, x[1], 2 * sizeof(u64))) {
            return std::nullopt;
        }
        return memory.Read64(x[1] + sizeof(u64));
    } else if (name == "nvnDeviceRegisterFastClearColor" ||
               name == "nvnDeviceRegisterFastClearColori" ||
               name == "nvnDeviceRegisterFastClearColorui" ||
               name == "nvnDeviceRegisterFastClearDepth") {
        // Registration is optional. The normal clear remains in the recording.
        return 0;
    } else {
        return std::nullopt;
    }
    return 0;
}

namespace {
bool WaitInvalidate();
} // Anonymous namespace

std::optional<u64> Session::HandlePool(Memory::Memory& memory, std::string_view name,
                                      const GuestCall& call) {
    const auto& x = call.x;
    if (name == "nvnMemoryPoolInitialize") {
        if (!ValidRange(memory, x[0], 0x100) || !ValidRange(memory, x[1], 0x24)) {
            return 0;
        }
        Pool pool{
            .device = memory.Read64(x[1] + 0x08),
            .storage = memory.Read64(x[1] + 0x10),
            .size = memory.Read64(x[1] + 0x18),
            .flags = memory.Read32(x[1] + 0x04),
        };
        if (!devices.contains(pool.device)) {
            return 0;
        }
        if (const u32 external = memory.Read32(x[1] + 0x20); external) {
            NoteMissing("nvnMemoryPoolInitialize external storage");
            return 0;
        }
        if (pool.size == 0 || pool.size > (1ULL << 40) ||
            (pool.storage && !ValidRange(memory, pool.storage, pool.size)) ||
            (!pool.storage && !(pool.flags & (CpuNoAccess | VirtualPool)))) {
            NoteMissing(fmt::format("nvnMemoryPoolInitialize invalid storage (flags={:#x}, size={:#x}, mapped={})",
                                    pool.flags, pool.size, pool.storage != 0));
            return 0;
        }
        const u64 span = Common::AlignUp(pool.size, PoolAlignment);
        if (!Fits(next_gpu_address, span, std::numeric_limits<u64>::max())) {
            return 0;
        }
        if (pools.contains(x[0])) {
            NoteMissing("nvnMemoryPoolInitialize already initialized");
            return 0;
        }
        pool.gpu_address = next_gpu_address;
        next_gpu_address += span;
        {
            std::unique_lock lock{pool_buffer_mutex};
            pool.serial = ++pool_serial;
            pools.emplace(x[0], pool);
        }
        gpu_pools.emplace(pool.gpu_address, x[0]);
        pool_spans_stale = true;
        constexpr std::array<u8, 0x100> empty{};
        memory.WriteBlock(x[0], empty.data(), empty.size());
        memory.Write64(x[0] + 0x08, pool.device);
        memory.Write32(x[0] + 0x10, pool.flags);
        memory.Write64(x[0] + 0x18, pool.gpu_address);
        memory.Write64(x[0] + 0x30, pool.storage);
        memory.Write64(x[0] + 0x38, pool.size);
        return 1;
    } else if (name == "nvnMemoryPoolFinalize") {
        if (const auto it = pools.find(x[0]); it != pools.end()) {
            const Pool& pool = it->second;
            std::erase_if(virtual_pages, [&](const auto& page) {
                return (page.first >= pool.gpu_address &&
                        page.first - pool.gpu_address < pool.size) ||
                       page.second.physical_address == pool.gpu_address;
            });
            gpu_pools.erase(pool.gpu_address);
            pool_spans_stale = true;
            std::unique_lock lock{pool_buffer_mutex};
            pools.erase(it);
        }
    } else if (name == "nvnMemoryPoolMapVirtual") {
        if (!MapVirtual(memory, call)) {
            NoteMissing("nvnMemoryPoolMapVirtual invalid mapping");
            return 0;
        }
        ++virtual_map_generation;
        return 1;
    } else {
        const auto it = pools.find(x[0]);
        if (it == pools.end()) {
            return std::nullopt;
        }
        const Pool& pool = it->second;
        if (name == "nvnMemoryPoolMap") {
            return pool.flags & CpuNoAccess ? 0 : pool.storage;
        }
        if (name == "nvnMemoryPoolGetSize") {
            return pool.size;
        }
        if (name == "nvnMemoryPoolFlushMappedRange" ||
            name == "nvnMemoryPoolInvalidateMappedRange") {
            if (!pool.storage || !Fits(x[1], x[2], pool.size)) {
                NoteMissing(fmt::format("{} invalid range", name));
            } else if (name == "nvnMemoryPoolFlushMappedRange") {
                NoteFlush(pool, x[1], x[2]);
            } else if (name == "nvnMemoryPoolInvalidateMappedRange" && WaitInvalidate()) {
                WaitForPendingWrites(pool.storage + x[1], x[2], "pool invalidate");
            }
            return 0;
        }
        if (name != "nvnMemoryPoolSetDebugLabel") {
            return std::nullopt;
        }
    }
    return 0;
}

std::optional<ResourcePoolInfo> Session::GetPool(u64 address) const {
    const auto it = pools.find(address);
    if (it == pools.end()) {
        return std::nullopt;
    }
    const auto& pool = it->second;
    return ResourcePoolInfo{pool.device, pool.storage, pool.size, pool.gpu_address, pool.flags};
}

void Session::ApplyPendingFlushes() {
    if (!flushes_pending.load(std::memory_order_acquire)) {
        return;
    }
    std::vector<PendingFlush> flushes;
    {
        std::scoped_lock flush_lock{flush_mutex};
        flushes.swap(pending_flushes);
        flushes_pending.store(false, std::memory_order_release);
    }
    for (const PendingFlush& flush : flushes) {
        if (const auto pool = pools.find(flush.pool);
            pool != pools.end() && pool->second.storage &&
            Fits(flush.offset, flush.size, pool->second.size)) {
            NoteFlush(pool->second, flush.offset, flush.size);
        }
    }
}

void Session::NoteFlush(const Pool& pool, u64 offset, u64 size) {
    if (!size) {
        return;
    }
    auto& stamps = pool.flush_stamps;
    if (stamps.empty()) {
        stamps.resize((pool.size >> 16) + 1);
    }
    const u64 generation = ++flush_generation;
    for (u64 page = offset >> 16; page <= (offset + size - 1) >> 16; ++page) {
        stamps[page] = generation;
    }
}

const Session::Pool* Session::BackingPool(u64 address) const {
    const auto upper = gpu_pools.upper_bound(address);
    if (upper == gpu_pools.begin()) {
        return nullptr;
    }
    const Pool& pool = pools.at(std::prev(upper)->second);
    if (!(pool.flags & VirtualPool)) {
        return &pool;
    }
    const auto page = virtual_pages.find(Common::AlignDown(address, PoolAlignment));
    if (page == virtual_pages.end()) {
        return nullptr;
    }
    const auto source = pools.find(page->second.physical_pool);
    return source != pools.end() ? &source->second : nullptr;
}

std::optional<u64> Session::ResolveGpuRange(u64 address, u64 size, const Pool** backing) const {
    const auto upper = gpu_pools.upper_bound(address);
    if (upper == gpu_pools.begin() || !size) {
        return std::nullopt;
    }
    const auto it = std::prev(upper);
    const Pool& pool = pools.at(it->second);
    const u64 offset = address - pool.gpu_address;
    if (!Fits(offset, size, pool.size)) {
        return std::nullopt;
    }
    if (!(pool.flags & VirtualPool)) {
        if (backing) {
            *backing = &pool;
        }
        return pool.storage ? std::optional{pool.storage + offset} : std::nullopt;
    }
    const u64 page_address = Common::AlignDown(address, PoolAlignment);
    const auto first = virtual_pages.find(page_address);
    if (first == virtual_pages.end()) {
        return std::nullopt;
    }
    const auto& page = first->second;
    const auto source = pools.find(page.physical_pool);
    const u64 physical_offset = page.offset + address - page_address;
    if (source == pools.end() || source->second.gpu_address != page.physical_address ||
        !source->second.storage || !Fits(physical_offset, size, source->second.size)) {
        return std::nullopt;
    }
    const u64 end = Common::AlignDown(address + size - 1, PoolAlignment);
    for (u64 current = page_address; current < end;) {
        current += PoolAlignment;
        const auto next = virtual_pages.find(current);
        if (next == virtual_pages.end() || next->second.physical_address != page.physical_address ||
            next->second.offset != page.offset + current - page_address) {
            return std::nullopt;
        }
    }
    if (backing) {
        *backing = &source->second;
    }
    return source->second.storage + physical_offset;
}

namespace {
bool SyncSubmits() {
    static const bool sync = std::getenv("EDEN_NX2_NVN_SYNC_SUBMITS") != nullptr;
    return sync;
}

void AddRange(std::map<u64, u64>& ranges, u64 start, u64 end) {
    auto it = ranges.upper_bound(start);
    if (it != ranges.begin() && std::prev(it)->second >= start) {
        --it;
        start = it->first;
        end = std::max(end, it->second);
        it = ranges.erase(it);
    }
    while (it != ranges.end() && it->first <= end) {
        end = std::max(end, it->second);
        it = ranges.erase(it);
    }
    ranges.emplace(start, end);
}

// The game reads back GPU results right after submitting without a sync; the hardware does
// not make it wait either.
bool WaitInvalidate() {
    static const bool wait = std::getenv("EDEN_NX2_NVN_WAIT_INVALIDATE") != nullptr;
    return wait;
}

bool SerialCopies() {
    static const bool serial = std::getenv("EDEN_NX2_NVN_SERIAL_COPIES") != nullptr;
    return serial;
}

bool CopyVertices() {
    static const bool copy = std::getenv("EDEN_NX2_NVN_COPY_VERTICES") != nullptr;
    return copy;
}

bool Overlaps(const std::map<u64, u64>& ranges, u64 start, u64 end) {
    if (ranges.empty() || end <= ranges.begin()->first) {
        return false;
    }
    const auto it = ranges.upper_bound(start);
    return (it != ranges.end() && it->first < end) ||
           (it != ranges.begin() && std::prev(it)->second > start);
}
} // Anonymous namespace

void Session::WaitForPendingWrites(u64 guest, u64 size, const char* reason) {
    if (pending_writes.empty() || !size) {
        return;
    }
    auto& retirement = BananzaNvn2Commands::GetRetirement();
    const auto retire = [&] {
        std::erase_if(pending_writes,
                      [&](const PendingWrites& p) { return retirement.WritesDone(p.sequence); });
    };
    // Asking takes a lock; a submission that retired meanwhile is found below when it overlaps.
    static u32 calls{};
    if ((++calls & 63) == 0) {
        retire();
    }
    if (std::ranges::none_of(pending_writes, [&](const PendingWrites& p) {
            return Overlaps(p.ranges, guest, guest + size);
        })) {
        return;
    }
    retire();
    u64 fence{};
    u64 sequence{};
    for (const auto& pending : pending_writes) {
        if (Overlaps(pending.ranges, guest, guest + size)) {
            fence = std::max(fence, pending.fence);
            sequence = std::max(sequence, pending.sequence);
        }
    }
    if (!sequence) {
        return;
    }
    if (call_profile && logged_overlaps < 48) {
        ++logged_overlaps;
        LOG_INFO(Core_ARM, "Bananza NVN 2 {} read {:#x}+{:#x} waits for submission fence {}",
                 reason, guest, size, fence);
    }
    const auto start = std::chrono::steady_clock::now();
    system.GPU().WaitForSyncOperation(fence);
    retirement.WaitWrites(sequence);
    retire();
    if (call_profile) {
        ++call_profile->submit.overlap_waits;
        call_profile->submit.overlap_wait_ns += ElapsedNs(start);
    }
}

void Session::SnapshotResources(Memory::Memory& memory, BananzaNvn2Commands::Submission& batch,
                                Queue& queue) {
    batch.guest_memory = &memory;
    static DirtyStats* const dirty_stats =
        std::getenv("EDEN_NX2_NVN_DIRTY_STATS") ? new DirtyStats : nullptr;
    std::unique_lock<std::mutex> dirty_lock;
    if (dirty_stats) {
        dirty_lock = std::unique_lock{dirty_stats->mutex};
        ++dirty_stats->submissions;
    }
    const auto note_dirty = [&](u64 guest, u64 size, const u8* source, const Pool* pool,
                                size_t kind) {
        if (!dirty_stats) {
            return;
        }
        const u32 flags = pool ? pool->flags : 0;
        auto& stats = dirty_stats->classes[kind][!pool ? 3 : flags & 2 ? 1 : flags & 4 ? 0
                                                                      : flags & 1 ? 2 : 3];
        const u64 first = guest & ~0xfffULL;
        const u8* data = source - (guest - first);
        for (u64 page = first; page < guest + size; page += 0x1000, data += 0x1000) {
            const auto [entry, fresh] = dirty_stats->pages.try_emplace(page);
            if (!fresh && entry->second.seen == dirty_stats->submissions) {
                continue;
            }
            const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(data), 0x1000);
            u64 stamp{};
            if (pool && !pool->flush_stamps.empty() && page >= pool->storage) {
                if (const u64 index = (page - pool->storage) >> 16;
                    index < pool->flush_stamps.size()) {
                    stamp = pool->flush_stamps[index];
                }
            }
            ++stats.pages;
            if (fresh) {
                ++stats.fresh;
            } else if (entry->second.hash != hash) {
                ++stats.changed;
                stats.changed_unflushed += entry->second.stamp == stamp;
                stats.changed_gpu_written += Overlaps(gpu_written_guest, page, page + 0x1000);
            }
            entry->second = {hash, stamp, dirty_stats->submissions};
        }
    };
    const auto scan_start = std::chrono::steady_clock::now();
    constexpr u64 MaxBufferBytes = 96ULL << 20;
    constexpr u64 MaxTextureBytes = 128ULL << 20;
    constexpr u64 MaxRange = 8ULL << 20;
    std::atomic<u64> buffer_bytes{};
    u64 texture_bytes{};
    std::vector<const ResourceTextureInfo*> sampled_textures;
    Common::unordered_set<std::tuple<u64, u64, u32>> seen_descriptors;
    using Range = std::pair<u64, u64>;
    Common::unordered_map<u64, u64> indirect_index_requests;
    Common::unordered_set<Range> indirect_arg_requests;
    const auto read_gpu = [&](u64 address, std::span<u8> dest) {
        const auto read = [&](u64 gpu, std::span<u8> bytes) {
            const auto guest = ResolveGpuRange(gpu, bytes.size());
            if (!guest || !ValidRange(memory, *guest, bytes.size())) {
                return false;
            }
            WaitForPendingWrites(*guest, bytes.size(), "snapshot");
            if (const auto* source = memory.GetSpan(*guest, bytes.size())) {
                std::memcpy(bytes.data(), source, bytes.size());
                return true;
            }
            return memory.ReadBlockUnsafe(*guest, bytes.data(), bytes.size());
        };
        if (read(address, dest)) {
            return true;
        }
        if (dest.size() <= PoolAlignment - address % PoolAlignment) {
            return false;
        }
        while (!dest.empty()) {
            const auto bytes = std::min<u64>(dest.size(), PoolAlignment - address % PoolAlignment);
            if (!read(address, dest.first(bytes))) {
                return false;
            }
            address += bytes;
            dest = dest.subspan(bytes);
        }
        return true;
    };
    // Copies run on worker threads while the scan goes on; drain() before anything reads them.
    struct CopyJob {
        u8* dest;
        const u8* source;
        u64 size;
    };
    std::vector<CopyJob> copy_jobs;
    u64 copy_job_bytes{};
    u64 copy_wait_ns{};
    const auto dispatch = [&] {
        if (!copy_jobs.empty()) {
            copy_workers.QueueWork([jobs = std::move(copy_jobs)] {
                for (const auto& job : jobs) {
                    std::memcpy(job.dest, job.source, job.size);
                }
            });
            copy_jobs.clear();
            copy_job_bytes = 0;
        }
    };
    const auto drain = [&] {
        for (const auto& job : copy_jobs) {
            std::memcpy(job.dest, job.source, job.size);
        }
        copy_jobs.clear();
        copy_job_bytes = 0;
        const auto wait_start = std::chrono::steady_clock::now();
        copy_workers.WaitForRequests();
        copy_wait_ns += ElapsedNs(wait_start);
    };
    using Submission = BananzaNvn2Commands::Submission;
    struct FixedCandidate {
        std::pair<u64, u64> key;
        u64 guest;
    };
    // kind: 0 a copy, 1 a copy that is also a graphics storage range, 2 a reference.
    struct Request {
        u64 address;
        u64 size;
        u32 kind;
    };
    if (pool_spans_stale) {
        pool_spans_stale = false;
        pool_spans.clear();
        for (const auto& [gpu_address, handle] : gpu_pools) {
            if (const auto pool = pools.find(handle);
                pool != pools.end() && !(pool->second.flags & VirtualPool) &&
                pool->second.storage && pool->second.gpu_address == gpu_address) {
                pool_spans.push_back({gpu_address, pool->second.size, pool->second.storage,
                                      &pool->second});
            }
        }
    }
    struct LateCopy {
        u8* dest;
        const u8* source;
        u64 size;
    };
    static const bool late_copies_on = std::getenv("EDEN_NX2_NVN_NO_LATE_COPIES") == nullptr;
    constexpr u64 LateCopyBytes = 16ULL << 10;
    struct ScanContext {
        // False on the threads that share the ranges of a scan: what needs the session
        // (a wait for pending writes, a read through the page table) is left in `later`.
        bool whole{true};
        u64 referenced_bytes{};
        u64 referenced_ranges{};
        u64 fixed_bytes{};
        std::vector<FixedCandidate> fixed;
        struct Storage {
            Range range;
            Submission::StorageAddress address;
        };
        std::vector<Storage> storage;
        // The staged uniform ranges: pool serial, pool storage, pool size, begin and end in
        // the pool.
        std::vector<std::array<u64, 5>> pages;
        // Those ranges as runs with their staging blocks, made by the thread of the shard.
        std::vector<Submission::MirrorPage> runs;
        std::vector<Request> later;
        // Large copies, made after the tables are complete.
        std::vector<LateCopy> late;
        // The newest submission that may still write a range passed by reference.
        u64 after_writes{};
    };
    constexpr u32 CpuUncached = 2;
    constexpr u32 CpuCached = 4;
    const auto resolve = [&](ScanContext& context, u64 address, u64 size,
                             const Pool** backing) -> std::optional<u64> {
        static_cast<void>(context);
        if (size) {
            // The pool that starts at or before the address, as ResolveGpuRange picks it.
            const auto next = std::ranges::upper_bound(pool_spans, address, {}, &PoolSpan::begin);
            if (next != pool_spans.begin()) {
                const PoolSpan& span = *std::prev(next);
                if (const u64 offset = address - span.begin; Fits(offset, size, span.size)) {
                    *backing = span.pool;
                    return span.storage + offset;
                }
            }
        }
        // A virtual pool, or no pool.
        return ResolveGpuRange(address, size, backing);
    };
    // The range threads look at a copy: the game thread may retire entries meanwhile.
    const std::vector<PendingWrites>* pending_view = &pending_writes;
    const auto pending_overlap = [&](u64 guest, u64 size) {
        return std::ranges::any_of(*pending_view, [&](const PendingWrites& pending) {
            return Overlaps(pending.ranges, guest, guest + size);
        });
    };
    // A range the replay reads from guest memory is waited for there (Submission::after_writes),
    // not by the scan: the scan of a frame runs while the GPU finishes the frame before.
    // EDEN_NX2_NVN_SCAN_WAITS=1 waits here as before r155.
    static const bool scan_waits = std::getenv("EDEN_NX2_NVN_SCAN_WAITS") != nullptr;
    const auto note_pending = [&](ScanContext& context, u64 guest, u64 size) {
        for (const PendingWrites& pending : *pending_view) {
            if (pending.sequence > context.after_writes &&
                Overlaps(pending.ranges, guest, guest + size)) {
                context.after_writes = pending.sequence;
            }
        }
    };
    static const bool reference_storage = std::getenv("EDEN_NX2_NVN_COPY_STORAGE") == nullptr;
    // Uniform ranges go to their pool's mirror as merged runs (r133); on by default since
    // r150, EDEN_NX2_NVN_NO_STAGED_UNIFORMS=1 copies each range as before.
    static const bool stage_uniforms =
        std::getenv("EDEN_NX2_NVN_NO_STAGED_UNIFORMS") == nullptr;
    // kind: 0 a copy, 1 storage of a graphics stage, 3 storage of a dispatch, 4 a uniform
    // range (a copy unless its pages can be staged for the pool's mirror).
    const auto copy_range = [&](ScanContext& context, u64 address, u64 size, u32 kind) {
        if (!size || size > MaxRange || !Fits(buffer_bytes.load(), size, MaxBufferBytes)) {
            return;
        }
        const auto key = std::pair{address, size};
        const size_t shard = Submission::SnapshotShard(key);
        auto& snapshots = batch.buffer_snapshots[shard];
        const auto found = snapshots.find(key);
        const bool owned = found != snapshots.end() && found->second.owned;
        const Pool* pool{};
        const auto guest = resolve(context, address, size, &pool);
        const u8* const source = guest ? memory.GetSpan(*guest, size) : nullptr;
        const bool referenced = !scan_waits && reference_storage && (kind == 1 || kind == 3) &&
                                source && !owned;
        if (!scan_waits && kind == 4 && source && !owned) {
            const u64 before = context.after_writes;
            context.after_writes = 0;
            note_pending(context, *guest, size);
            const bool pending = context.after_writes != 0;
            context.after_writes = std::max(context.after_writes, before);
            if (pending) {
                // GPU work of an earlier submission may still write the range: the replay
                // reads it from guest memory once those writes are back.
                if (found == snapshots.end()) {
                    snapshots.emplace(key, Submission::BufferBytes{source, false});
                }
                return;
            }
        }
        if (!context.whole && (!source || (!referenced && pending_overlap(*guest, size)))) {
            context.later.push_back({address, size, kind});
            return;
        }
        // The range's bytes as the table has them now, for the storage table.
        const auto stored_as = [&](const Submission::BufferBytes& bytes) {
            return Submission::StorageAddress{*guest, bytes.data, bytes.owned, bytes.late};
        };
        std::optional<Submission::StorageAddress> stored;
        if (found != snapshots.end() && guest) {
            stored = stored_as(found->second);
        }
        // A uniform range is not copied by itself: the pages it lies on go to the replay's
        // mirror of its pool once per submission, and the entry says where it is in the pool.
        if (stage_uniforms && kind == 4 && source && !owned && pool && pool->storage &&
            size <= 65536 && *guest >= pool->storage &&
            Fits(*guest - pool->storage, size, pool->size)) {
            if (context.whole) {
                WaitForPendingWrites(*guest, size, "uniform");
                note_dirty(*guest, size, source, pool, 0);
            }
            const u64 offset = *guest - pool->storage;
            auto& entry = found != snapshots.end()
                              ? found->second
                              : snapshots.emplace(key, Submission::BufferBytes{source, false})
                                    .first->second;
            entry.staged_pool = pool->serial;
            entry.fixed_offset = offset;
            entry.fixed_pool_guest = pool->storage;
            entry.fixed_pool_size = pool->size;
            const std::array<u64, 5> wanted{pool->serial, pool->storage, pool->size, offset,
                                            offset + size};
            if (context.pages.empty() || context.pages.back() != wanted) {
                context.pages.push_back(wanted);
            }
            return;
        }
        // Storage is read by the replay from guest memory, as vertex data is: the game keeps
        // it until the sync, and what the GPU wrote before is back (waited for above or here).
        if (reference_storage && (kind == 1 || kind == 3) && source && !owned) {
            if (referenced) {
                note_pending(context, *guest, size);
            } else if (context.whole) {
                WaitForPendingWrites(*guest, size, "storage");
            }
            if (context.whole) {
                note_dirty(*guest, size, source, pool, 1);
            }
            if (!stored) {
                stored = stored_as(
                    snapshots.emplace(key, Submission::BufferBytes{source, false}).first->second);
                context.referenced_bytes += size;
                ++context.referenced_ranges;
            }
            if (kind == 1) {
                context.storage.push_back({key, *stored});
            }
            return;
        }
        bool copied = owned;
        bool late = false;
        if (!owned) {
            u8* const data = batch.snapshot_blocks[shard].Allocate(size);
            if (source) {
                if (context.whole) {
                    WaitForPendingWrites(*guest, size, "snapshot");
                    note_dirty(*guest, size, source, pool, 0);
                }
                buffer_bytes += size;
                const bool later = context.whole && !SerialCopies() && size >= 1024;
                if (later) {
                    copy_jobs.push_back({data, source, size});
                    copy_job_bytes += size;
                } else if (!context.whole && late_copies_on && size >= LateCopyBytes) {
                    context.late.push_back({data, source, size});
                    late = true;
                } else {
                    // Not worth a job: most uniform ranges are a few hundred bytes.
                    std::memcpy(data, source, size);
                }
                copied = true;
                if (later && copy_job_bytes >= (256ULL << 10)) {
                    dispatch();
                }
            } else if (read_gpu(address, {data, size})) {
                buffer_bytes += size;
                copied = true;
            }
            if (copied) {
                // A range first bound for vertices is copied once something else needs it fixed.
                auto& entry = found != snapshots.end() ? found->second
                                                       : snapshots.try_emplace(key).first->second;
                entry = {data, true};
                entry.late = late;
                if (guest) {
                    stored = stored_as(entry);
                }
            }
        }
        if (kind == 1 && stored && guest && (source || ValidRange(memory, *guest, size))) {
            context.storage.push_back({key, *stored});
        }
    };
    // Vertex and index data: the replay reads guest memory, as the GPU does at execution.
    const auto reference_range = [&](ScanContext& context, u64 address, u64 size) {
        if (CopyVertices() || !size || size > MaxRange) {
            copy_range(context, address, size, 0);
            return;
        }
        const auto key = std::pair{address, size};
        auto& snapshots = batch.buffer_snapshots[Submission::SnapshotShard(key)];
        if (snapshots.contains(key)) {
            return;
        }
        const Pool* pool{};
        const auto guest = resolve(context, address, size, &pool);
        const u8* const source = guest ? memory.GetSpan(*guest, size) : nullptr;
        if (!source) {
            copy_range(context, address, size, 0);
            return;
        }
        if (!scan_waits) {
            note_pending(context, *guest, size);
            if (context.whole) {
                note_dirty(*guest, size, source, pool, 1);
            }
        } else if (context.whole) {
            WaitForPendingWrites(*guest, size, "vertices");
            note_dirty(*guest, size, source, pool, 1);
        } else if (pending_overlap(*guest, size)) {
            context.later.push_back({address, size, 2});
            return;
        }
        Submission::BufferBytes bytes{source, false};
        // The game must flush a CPU_CACHED pool after writing it, and cannot write a
        // CPU_NO_ACCESS one. A range the GPU writes changes without a flush.
        if (pool && (pool->flags & (CpuNoAccess | CpuCached)) && !(pool->flags & CpuUncached) &&
            !Overlaps(gpu_written_guest, *guest, *guest + size)) {
            u64 stamp{};
            if (!pool->flush_stamps.empty()) {
                const u64 offset = *guest - pool->storage;
                for (u64 page = offset >> 16; page <= (offset + size - 1) >> 16; ++page) {
                    stamp = std::max(stamp, pool->flush_stamps[page]);
                }
            }
            bytes.fixed_pool = pool->serial;
            bytes.fixed_stamp = stamp;
            if (*guest >= pool->storage && Fits(*guest - pool->storage, size, pool->size)) {
                bytes.fixed_offset = *guest - pool->storage;
                bytes.fixed_pool_guest = pool->storage;
                bytes.fixed_pool_size = pool->size;
            }
            context.fixed.push_back({key, *guest});
            context.fixed_bytes += size;
        }
        snapshots.emplace(key, std::move(bytes));
        context.referenced_bytes += size;
        ++context.referenced_ranges;
    };
    ScanContext main_context;
    auto& fixed_candidates = main_context.fixed;
    u64& fixed_bytes = main_context.fixed_bytes;
    u64& referenced_bytes = main_context.referenced_bytes;
    u64& referenced_ranges = main_context.referenced_ranges;
    const auto note_storage = [&](ScanContext& context) {
        for (const auto& [range, address] : context.storage) {
            batch.storage_addresses.emplace(range, address);
        }
        context.storage.clear();
    };
    // While the commands are walked the ranges are only collected; see the loop below.
    static const bool serial_scan = std::getenv("EDEN_NX2_NVN_SERIAL_SCAN") != nullptr;
    bool deferring = !serial_scan && !dirty_stats && !SerialCopies();
    // Kept between submits (one at a time, under the session lock): no new pages every frame.
    static std::array<std::vector<Request>, Submission::SnapshotShards> request_storage;
    auto& requests = request_storage;
    for (auto& list : requests) {
        list.clear();
        if (deferring) {
            list.reserve(8192);
        }
    }
    size_t request_count{};
    // The last requests by address: one that repeats address, size and kind adds nothing.
    static std::array<Request, 2048> recent_requests;
    if (deferring) {
        recent_requests.fill({});
    }
    const auto add_request = [&](u64 address, u64 size, u32 kind) {
        Request& recent = recent_requests[(address * 0x9E3779B97F4A7C15ULL) >> 53];
        if (recent.address == address && recent.size == size && recent.kind == kind && size) {
            return;
        }
        recent = {address, size, kind};
        requests[Submission::SnapshotShard({address, size})].push_back(recent);
        ++request_count;
    };
    const auto snapshot = [&](u64 address, u64 size) {
        if (deferring) {
            add_request(address, size, 0);
        } else {
            copy_range(main_context, address, size, 0);
        }
    };
    const auto snapshot_uniform = [&](u64 address, u64 size) {
        if (deferring) {
            add_request(address, size, 4);
        } else {
            copy_range(main_context, address, size, 4);
        }
    };
    const auto reference = [&](u64 address, u64 size) {
        if (deferring) {
            add_request(address, size, 2);
        } else {
            reference_range(main_context, address, size);
        }
    };
    const auto descriptor = [&](u64 handle) {
        if (!seen_descriptors.emplace(queue.texture_pool, queue.sampler_pool,
                                      static_cast<u32>(handle)).second) {
            return;
        }
        const u32 texture_id = static_cast<u32>(handle) & 0xfffff;
        const u32 sampler_id = static_cast<u32>(handle >> 20) & 0xfff;
        if (const auto* samplers = resources.GetSamplerPool(queue.sampler_pool)) {
            if (const auto it = samplers->find(sampler_id); it != samplers->end()) {
                batch.sampler_descriptors[queue.sampler_pool].try_emplace(sampler_id, it->second);
            }
        }
        if (const auto* textures = resources.GetTexturePool(queue.texture_pool)) {
            const auto it = textures->find(texture_id);
            if (it == textures->end()) {
                return;
            }
            auto& descriptors = batch.texture_descriptors[queue.texture_pool];
            const auto [saved, inserted] = descriptors.try_emplace(texture_id, it->second);
            if (inserted) {
                sampled_textures.push_back(&saved->second.texture);
            }
        }
    };
    // The bindings left by earlier submissions are collected at the first draw or dispatch;
    // after that each range and handle is collected when it is bound.
    bool buffers_pending = true;
    bool storage_pending = true;
    bool descriptors_pending = true;
    std::vector<u64> written_textures;
    // Reads of ranges the frame before may still write, made when the rest of the scan is
    // done: the lists of indirect reset draws and the arguments of indexed indirect draws.
    std::vector<Range> reset_lists;
    std::vector<std::tuple<u64, u64, u64>> wanted_slices;
    const auto snapshot_graphics_storage = [&](std::pair<u64, u64> range) {
        if (deferring) {
            add_request(range.first, range.second, 1);
            return;
        }
        copy_range(main_context, range.first, range.second, 1);
        note_storage(main_context);
    };
    const auto bound_handle = [&](u64 handle) {
        if (!descriptors_pending && handle) {
            descriptor(handle);
        }
    };
    const auto note_targets = [&] {
        if (!queue.targets_noted) {
            queue.targets_noted = true;
            for (const u64 serial : queue.target_serials) {
                if (serial) {
                    written_textures.push_back(serial);
                }
            }
        }
    };
    // What a draw or dispatch finds bound from before: collected at the first one.
    const auto flush_buffers = [&] {
        if (!buffers_pending) {
            return;
        }
        buffers_pending = false;
        for (const auto& [address, size] : queue.vertex_ranges) {
            reference(address, size);
        }
        for (u32 stage = 0; stage < 3; ++stage) {
            for (const auto& range : queue.storage_ranges[stage]) {
                snapshot_graphics_storage(range);
            }
        }
        for (const auto& stage : queue.uniform_ranges) {
            for (const auto& [address, size] : stage) {
                snapshot_uniform(address, size);
            }
        }
    };
    const auto flush_descriptors = [&] {
        if (!descriptors_pending) {
            return;
        }
        descriptors_pending = false;
        for (const auto& stage : queue.texture_handles) {
            for (const u64 handle : stage) {
                if (handle) {
                    descriptor(handle);
                }
            }
        }
        for (const auto& stage : queue.image_handles) {
            for (const u64 handle : stage) {
                if (handle) {
                    descriptor(handle);
                }
            }
        }
    };
    // One operation that is not a call of another recording.
    const auto scan_op = [&](const BananzaNvn2Commands::Operation& op) {
        if (op.replay == ReplayOp::CopyTexture && op.Textures().size() == 2) {
            sampled_textures.push_back(&op.Textures()[0]);
            written_textures.push_back(op.Textures()[1].serial);
        } else if (op.replay == ReplayOp::CopyBufferToTexture && op.Textures().size() == 1) {
            written_textures.push_back(op.Textures()[0].serial);
        } else if (op.replay == ReplayOp::SetRenderTargets) {
            queue.target_serials = {};
            for (size_t i = 0; i < op.Textures().size() && i < queue.target_serials.size(); ++i) {
                queue.target_serials[i] = op.Textures()[i].serial;
            }
            queue.targets_noted = false;
        } else if (op.replay >= ReplayOp::ClearColor && op.replay <= ReplayOp::ClearDepthStencil) {
            note_targets();
        }
        if ((op.replay == ReplayOp::SignalEvent || op.replay == ReplayOp::WaitEvent) &&
                   !op.args.empty() && !batch.event_addresses.contains(op.args[0])) {
            if (const auto address = ResolveGpuRange(op.args[0], sizeof(u32));
                address && ValidRange(memory, *address, sizeof(u32))) {
                batch.event_addresses.emplace(op.args[0], *address);
            }
        } else if ((op.replay == ReplayOp::ReportCounter ||
                    op.replay == ReplayOp::SetRenderEnableConditional) && op.args.size() == 2) {
            const u64 size = op.replay == ReplayOp::ReportCounter ? 16 : 24;
            const u64 gpu_address = op.args[1];
            if (const auto address = ResolveGpuRange(gpu_address, size);
                address && ValidRange(memory, *address, size)) {
                batch.event_addresses.emplace(gpu_address, *address);
            }
        } else if (op.replay == ReplayOp::SetTexturePool && op.args.size() == 1) {
            queue.texture_pool = op.args[0];
            descriptors_pending = true;
        } else if (op.replay == ReplayOp::BindProgram && op.args.size() == 2 && (op.args[1] & 1)) {
            queue.resets_indirect_draws = op.Program() && op.Program()->resets_indirect_draws;
        } else if (op.replay == ReplayOp::SetSamplerPool && op.args.size() == 1) {
            queue.sampler_pool = op.args[0];
            descriptors_pending = true;
        } else if (op.replay == ReplayOp::SetCopyRowStride && op.args.size() == 1) {
            queue.copy_row_stride = op.args[0];
        } else if (op.replay == ReplayOp::CopyBufferToTexture && op.Textures().size() == 1 &&
                   op.args.size() == 3 && op.Payload().size() == 40) {
            const auto block = GetTextureFormatBlock(op.Textures()[0].description.format);
            std::array<u32_le, 6> region{};
            std::memcpy(region.data(), op.Payload().data() + 16, 24);
            if (block && region[3] && region[4] && region[5] && region[3] <= 16384 &&
                region[4] <= 16384 && region[5] <= 2048) {
                const u64 bytes = (u64{region[3]} + block->width - 1) / block->width *
                                  block->bytes;
                const u64 rows = (u64{region[4]} + block->height - 1) / block->height;
                const u64 stride = queue.copy_row_stride ? queue.copy_row_stride : bytes;
                if (stride >= bytes && stride <= MaxRange / (rows * region[5])) {
                    snapshot(op.args[0], stride * (rows * region[5] - 1) + bytes);
                }
            }
        } else if (op.replay == ReplayOp::BindTexture && op.args.size() == 3) {
            if (op.args[0] < queue.texture_handles.size() && op.args[1] < 64) {
                queue.texture_handles[op.args[0]][op.args[1]] = op.args[2];
                bound_handle(op.args[2]);
            }
        } else if (op.replay == ReplayOp::BindImage && op.args.size() == 3) {
            if (op.args[0] < queue.image_handles.size() && op.args[1] < 32) {
                queue.image_handles[op.args[0]][op.args[1]] = op.args[2];
                bound_handle(op.args[2]);
            }
        } else if (op.replay == ReplayOp::BindTextures && op.args.size() == 3) {
            for (size_t offset = 0; offset + 8 <= op.Payload().size(); offset += 8) {
                u64_le handle;
                std::memcpy(&handle, op.Payload().data() + offset, 8);
                const u64 slot = op.args[1] + offset / 8;
                if (op.args[0] < queue.texture_handles.size() && slot < 64) {
                    queue.texture_handles[op.args[0]][slot] = handle;
                    bound_handle(handle);
                }
            }
        } else if (op.replay == ReplayOp::BindVertexBuffer && op.args.size() == 3) {
            if (op.args[0] < queue.vertex_ranges.size()) {
                queue.vertex_ranges[op.args[0]] = {op.args[1], op.args[2]};
                if (!buffers_pending) {
                    reference(op.args[1], op.args[2]);
                }
            }
        } else if ((op.replay == ReplayOp::BindUniformBuffer ||
                    op.replay == ReplayOp::BindStorageBuffer) && op.args.size() == 4) {
            const bool uniform = op.replay == ReplayOp::BindUniformBuffer;
            auto& ranges = uniform ? queue.uniform_ranges : queue.storage_ranges;
            if (op.args[0] < ranges.size() && op.args[1] < 32) {
                ranges[op.args[0]][op.args[1]] = {op.args[2], op.args[3]};
                storage_pending |= op.args[0] == 5;
                if (!buffers_pending && uniform) {
                    snapshot_uniform(op.args[2], op.args[3]);
                } else if (!buffers_pending && op.args[0] < 3) {
                    snapshot_graphics_storage({op.args[2], op.args[3]});
                }
            }
        } else if ((op.replay == ReplayOp::BindUniformBuffers ||
                    op.replay == ReplayOp::BindStorageBuffers) && op.args.size() == 3) {
            storage_pending |= op.args[0] == 5;
            const bool uniform = op.replay == ReplayOp::BindUniformBuffers;
            auto& ranges = uniform ? queue.uniform_ranges : queue.storage_ranges;
            for (size_t offset = 0; offset + 16 <= op.Payload().size(); offset += 16) {
                std::array<u64_le, 2> range{};
                std::memcpy(range.data(), op.Payload().data() + offset, 16);
                const u64 slot = op.args[1] + offset / 16;
                if (op.args[0] < ranges.size() && slot < 32) {
                    ranges[op.args[0]][slot] = {range[0], range[1]};
                    if (buffers_pending) {
                        continue;
                    }
                    if (uniform) {
                        snapshot_uniform(range[0], range[1]);
                    } else if (op.args[0] < 3) {
                        snapshot_graphics_storage({range[0], range[1]});
                    }
                }
            }
        } else if (op.replay == ReplayOp::BindTexturesDeferred && op.args.size() == 4) {
            const u64 size = op.args[2] * sizeof(u64);
            const auto key = std::pair{op.args[3], size};
            if (size && Fits(buffer_bytes.load(), size, MaxBufferBytes) && !batch.deferred_data.contains(key) &&
                ValidRange(memory, key.first, key.second)) {
                WaitForPendingWrites(key.first, key.second, "deferred textures");
                std::vector<u8> data(size);
                if (memory.ReadBlockUnsafe(key.first, data.data(), size)) {
                    batch.deferred_data.emplace(key, std::move(data));
                    buffer_bytes += size;
                }
            }
            if (const auto it = batch.deferred_data.find(key); it != batch.deferred_data.end()) {
                for (size_t offset = 0; offset + 8 <= it->second.size(); offset += 8) {
                    u64_le handle;
                    std::memcpy(&handle, it->second.data() + offset, 8);
                    const u64 slot = op.args[1] + offset / 8;
                    if (op.args[0] < queue.texture_handles.size() && slot < 64) {
                        queue.texture_handles[op.args[0]][slot] = handle;
                        bound_handle(handle);
                    }
                }
            }
        } else if ((op.replay == ReplayOp::DrawIndexed ||
                    op.replay == ReplayOp::DrawIndexedBaseVertex ||
                    op.replay == ReplayOp::DrawIndexedInstanced) && op.args.size() >= 4 &&
                   op.args[1] <= 2 && op.args[2] <= MaxRange / 4) {
            reference(op.args[3], op.args[2] << op.args[1]);
        }
        if (op.replay == ReplayOp::DrawIndirect && op.args.size() == 2) {
            indirect_arg_requests.emplace(op.args[1], 4 * sizeof(u32));
        } else if (op.replay == ReplayOp::DrawIndexedIndirect && op.args.size() == 4 &&
                   op.args[1] <= 2) {
            const u64 address = op.args[2];
            indirect_arg_requests.emplace(op.args[3], 5 * sizeof(u32));
            wanted_slices.emplace_back(op.args[3], address, op.args[1]);
            if (!indirect_index_requests.contains(address)) {
                constexpr u64 wanted = 16ULL << 10;
                const auto pool = gpu_pools.upper_bound(address);
                if (pool != gpu_pools.begin()) {
                    const auto& storage = pools.at(std::prev(pool)->second);
                    const u64 offset = address - storage.gpu_address;
                    if (offset < storage.size) {
                        u64 available = std::min(wanted, storage.size - offset);
                        if (storage.flags & VirtualPool) {
                            available = std::min(available,
                                                 PoolAlignment - address % PoolAlignment);
                        }
                        indirect_index_requests.try_emplace(address, available);
                    }
                }
            }
        }
        const bool draw = op.replay >= ReplayOp::Draw && op.replay <= ReplayOp::DrawIndexedIndirect;
        const bool dispatch = op.replay == ReplayOp::Dispatch;
        if (draw) {
            note_targets();
        }
        if (queue.resets_indirect_draws && op.replay == ReplayOp::Draw &&
            op.args.size() == 3 && op.args[0] == 0) {
            const auto& [address, size] = queue.storage_ranges[0][0];
            const auto list = VideoCommon::NativeIndirectReset::GetListRange(
                address, static_cast<u32>(size), static_cast<u32>(op.args[1]),
                static_cast<u32>(op.args[2]));
            if (list && std::ranges::find(reset_lists, Range{list->address, list->bytes}) ==
                            reset_lists.end()) {
                reset_lists.emplace_back(list->address, list->bytes);
            }
        }
        if (dispatch && op.args.size() == 1) {
            snapshot(op.args[0], 3 * sizeof(u32));
        }
        if (draw || dispatch) {
            flush_buffers();
        }
        if (dispatch && storage_pending) {
            storage_pending = false;
            const auto snapshot_storage = [&](std::pair<u64, u64> range) {
                const bool unsized = range.first && !range.second;
                if (unsized) {
                    const auto guest = ResolveGpuRange(range.first, 1);
                    if (!guest || !ValidRange(memory, *guest, 1)) {
                        batch.unsized_storage_sizes.try_emplace(range.first, 0);
                        return;
                    }
                    const auto& pool = pools.at(std::prev(gpu_pools.upper_bound(range.first))->second);
                    const u64 offset = range.first - pool.gpu_address;
                    range.second = std::min(MaxRange, pool.size - offset);
                    if (pool.flags & VirtualPool) {
                        range.second = std::min(range.second, PoolAlignment - range.first % PoolAlignment);
                    }
                    const auto mapped_range = [&](u64 size) {
                        return ResolveGpuRange(range.first, size) && ValidRange(memory, *guest, size);
                    };
                    if (!mapped_range(range.second)) {
                        // Keep the mapped prefix when an unsized binding crosses a hole.
                        u64 mapped = 1;
                        u64 end = range.second;
                        while (mapped < end) {
                            const u64 size = mapped + (end - mapped + 1) / 2;
                            if (mapped_range(size)) {
                                mapped = size;
                            } else {
                                end = size - 1;
                            }
                        }
                        range.second = mapped;
                    }
                }
                copy_range(main_context, range.first, range.second, 3);
                if (const auto* const bytes = batch.FindSnapshot(range)) {
                    if (const auto guest = ResolveGpuRange(range.first, range.second);
                        guest && ValidRange(memory, *guest, range.second)) {
                        batch.storage_addresses.try_emplace(
                            range, Submission::StorageAddress{*guest, bytes->data,
                                                              bytes->owned, bytes->late});
                        if (unsized) {
                            batch.unsized_storage_sizes.try_emplace(range.first, range.second);
                        }
                    }
                }
            };
            for (const auto& range : queue.storage_ranges[5]) {
                snapshot_storage(range);
            }
            for (const auto& range : queue.uniform_ranges[5]) {
                snapshot_storage(range);
            }
        }
        if (draw || dispatch) {
            flush_descriptors();
        }
    };
    static_assert(MaxRange == BananzaNvn2Commands::ScanMaxRange);
    // The operations themselves, with EDEN_NX2_NVN_WALK_OPS=1.
    const auto walk_ops = [&](auto&& self, const BananzaNvn2Commands::Recording& recording) -> void {
        for (size_t op_index = 0; op_index < recording.operations.size(); ++op_index) {
            const auto& op = recording.operations[op_index];
            BananzaNvn2Commands::PrefetchCalled(recording.operations, op_index);
            if (op.nested) {
                self(self, *op.nested);
            } else {
                scan_op(op);
            }
        }
    };
    static const bool walk_entries = std::getenv("EDEN_NX2_NVN_WALK_OPS") == nullptr;
    using ScanKind = BananzaNvn2Commands::ScanEntry::Kind;
    // The entries a recording made of its own operations when it ended: the same steps as
    // scan_op takes for a bind or an indexed draw. A called recording's entries are walked in
    // place of its call.
    const auto walk_scan = [&](auto&& self,
                               const BananzaNvn2Commands::Recording& scanned) -> void {
        const auto& entries = scanned.scan;
        for (size_t entry_index = 0; entry_index < entries.size(); ++entry_index) {
            const BananzaNvn2Commands::ScanEntry& entry = entries[entry_index];
#if defined(_M_X64) || defined(__x86_64__)
            // The called recordings are small and scattered: start loading the next one.
            if (entry_index + 2 < entries.size() &&
                entries[entry_index + 2].kind == ScanKind::Call) {
                _mm_prefetch(reinterpret_cast<const char*>(entries[entry_index + 2].a),
                             _MM_HINT_T0);
            }
#endif
            switch (entry.kind) {
            case ScanKind::Vertex:
                if (entry.slot < queue.vertex_ranges.size()) {
                    queue.vertex_ranges[entry.slot] = {entry.a, entry.b};
                    if (!buffers_pending) {
                        reference(entry.a, entry.b);
                    }
                }
                break;
            case ScanKind::Uniform:
            case ScanKind::Storage: {
                const bool uniform = entry.kind == ScanKind::Uniform;
                auto& ranges = uniform ? queue.uniform_ranges : queue.storage_ranges;
                if (entry.stage < ranges.size() && entry.slot < 32) {
                    ranges[entry.stage][entry.slot] = {entry.a, entry.b};
                    storage_pending |= entry.stage == 5;
                    if (!buffers_pending && uniform) {
                        snapshot_uniform(entry.a, entry.b);
                    } else if (!buffers_pending && entry.stage < 3) {
                        snapshot_graphics_storage({entry.a, entry.b});
                    }
                }
                break;
            }
            case ScanKind::Texture:
                if (entry.stage < queue.texture_handles.size() && entry.slot < 64) {
                    queue.texture_handles[entry.stage][entry.slot] = entry.a;
                    bound_handle(entry.a);
                }
                break;
            case ScanKind::Image:
                if (entry.stage < queue.image_handles.size() && entry.slot < 32) {
                    queue.image_handles[entry.stage][entry.slot] = entry.a;
                    bound_handle(entry.a);
                }
                break;
            case ScanKind::Index:
                reference(entry.a, entry.b);
                break;
            case ScanKind::Draw:
                note_targets();
                flush_buffers();
                flush_descriptors();
                break;
            case ScanKind::Op:
                scan_op(*reinterpret_cast<const BananzaNvn2Commands::Operation*>(entry.a));
                break;
            case ScanKind::Call:
                self(self, *reinterpret_cast<const BananzaNvn2Commands::Recording*>(entry.a));
                break;
            }
        }
    };
    for (const auto& recording : batch.recordings) {
        if (!walk_entries) {
            walk_ops(walk_ops, *recording);
            continue;
        }
        walk_scan(walk_scan, *recording);
    }
    // The backing bytes of the sampled textures: on the game thread, while the range
    // threads work when there are any.
    u64 texture_ns{};
    bool textures_done{};
    const auto texture_stage = [&] {
    textures_done = true;
    const auto texture_start = std::chrono::steady_clock::now();
    const auto prioritize = [](const ResourceTextureInfo* texture) {
        const auto block = GetTextureFormatBlock(texture->description.format);
        if (texture->layout.size <= (1ULL << 20) || !(texture->storage.flags & VirtualPool)) {
            return 0;
        }
        if (block && block->width > 1) {
            return 1;
        }
        // Depth and single-channel float inputs can be read before their first replayed write.
        switch (texture->description.format) {
        case 0x05: // R16_FLOAT
        case 0x0a: // R32_FLOAT
        case 0x32: // D16_UNORM
        case 0x34: // D32_FLOAT
            return 2;
        default:
            return 3;
        }
    };
    // Small CPU textures, font atlases and compressed assets need their backing bytes.
    // Large render targets usually get their contents from replayed work.
    std::ranges::stable_sort(sampled_textures, {}, prioritize);
    // The textures hashed last time are usually hashed again: start those on other threads.
    struct PreHash {
        u64 serial;
        u64 size;
        const u8* source;
        u64 hash;
    };
    std::vector<PreHash> pre_hashes;
    // Off unless asked for: it was slower on both machines (r116).
    static const bool pre_hash_on = std::getenv("EDEN_NX2_NVN_PRE_HASH") != nullptr;
    if (pre_hash_on) {
        for (const HashedTexture& texture : hashed_textures) {
            if (const u8* source = memory.GetSpan(texture.guest, texture.size);
                source && !pending_overlap(texture.guest, texture.size)) {
                pre_hashes.push_back({texture.serial, texture.size, source, 0});
            }
        }
    }
    hashed_textures.clear();
    bool pre_hashes_ready = pre_hashes.empty();
    for (PreHash& pre : pre_hashes) {
        late_copy_workers.QueueWork([&pre] {
            pre.hash = TextureHash(pre.source, pre.size, nullptr);
        });
    }
    static const bool trace_images = std::getenv("EDEN_NX2_NVN_TRACE_IMAGES") != nullptr;
    static thread_local std::set<std::pair<u64, bool>> texture_snapshot_failures;
    static const bool trace_texture_copies = std::getenv("EDEN_NX2_NVN_TRACE_TEXTURE_COPIES") != nullptr;
    const auto segments_of = [&](u64 address, u64 size) {
        std::vector<std::pair<u64, u64>> segments;
        const auto add = [&](u64 gpu, u64 bytes) {
            const auto guest = ResolveGpuRange(gpu, bytes);
            if (!guest || !ValidRange(memory, *guest, bytes)) {
                return false;
            }
            WaitForPendingWrites(*guest, bytes, "texture");
            segments.emplace_back(*guest, bytes);
            return true;
        };
        if (add(address, size)) {
            return segments;
        }
        if (size <= PoolAlignment - address % PoolAlignment) {
            return segments;
        }
        while (size) {
            const u64 bytes = std::min<u64>(size, PoolAlignment - address % PoolAlignment);
            if (!add(address, bytes)) {
                segments.clear();
                return segments;
            }
            address += bytes;
            size -= bytes;
        }
        return segments;
    };
    for (const auto* texture : sampled_textures) {
        const auto size = texture->layout.size;
        if (!size || batch.texture_sources.contains(texture->serial) ||
            gpu_written_textures.contains(texture->serial)) {
            continue;
        }
        const u64 address = texture->storage.gpu_address + texture->offset;
        if (texture->storage.flags & CpuNoAccess) {
            const auto& desc = texture->description;
            const TextureReplayKey key{address, desc.width, desc.height, desc.format, desc.levels,
                                       desc.target, texture->layout.layers,
                                       desc.target == 2 ? desc.depth : 1U, size};
            auto& state = texture_replay_states[key];
            if (!state) {
                state = std::make_shared<BananzaNvn2Commands::TextureReplayState>();
            }
            batch.texture_replay_states.try_emplace(texture->serial, state);
            if (state->gpu_written.load(std::memory_order_relaxed)) {
                continue;
            }
        }
        const auto note_failure = [&](bool budget) {
            if (trace_images && texture_snapshot_failures.size() < 100 &&
                texture_snapshot_failures.emplace(address, budget).second) {
                LOG_INFO(Core_ARM, "NVN 2 texture snapshot {} at {:#x}, {} bytes, format {:#x}, pool flags {:#x}, {} MiB collected",
                         budget ? "over budget" : "unmapped", address, size,
                         texture->description.format, texture->storage.flags, texture_bytes >> 20);
            }
        };
        if (size > (32ULL << 20)) {
            note_failure(true);
            continue;
        }
        auto segments = segments_of(address, size);
        if (segments.empty()) {
            note_failure(false);
            continue;
        }
        const auto known = texture_checks.find(texture->serial);
        // The game must flush a CPU_CACHED pool after writing it: without a newer flush over
        // the texture its bytes are the ones hashed last time.
        u64 flush_key{};
        static const bool hash_always = std::getenv("EDEN_NX2_NVN_HASH_TEXTURES") != nullptr;
        if (!hash_always && segments.size() == 1 && (texture->storage.flags & 4) &&
            !(texture->storage.flags & 2)) {
            if (const Pool* pool = BackingPool(address); pool && pool->storage) {
                const u64 guest = segments.front().first;
                u64 stamp{};
                if (!pool->flush_stamps.empty() && guest >= pool->storage) {
                    const u64 offset = guest - pool->storage;
                    for (u64 page = offset >> 16;
                         page <= (offset + size - 1) >> 16 && page < pool->flush_stamps.size(); ++page) {
                        stamp = std::max(stamp, pool->flush_stamps[page]);
                    }
                }
                flush_key = (pool->serial << 40) ^ stamp ^ (1ULL << 63);
            }
        }
        if (known != texture_checks.end() && flush_key && known->second.flush_key == flush_key &&
            (present_serial + texture->serial) % 128 != 0) {
            if (call_profile) {
                ++call_profile->submit.texture_flush_skips;
            }
            batch.texture_sources.emplace(
                texture->serial,
                BananzaNvn2Commands::TextureSource{known->second.hash, std::move(segments)});
            continue;
        }
        // A pool without CPU access changes through page remaps. Compare such a texture once
        // a frame while pages move, and now and then otherwise.
        if (known != texture_checks.end() && (texture->storage.flags & CpuNoAccess) &&
            (known->second.present == present_serial ||
             (known->second.map_generation == virtual_map_generation &&
              (present_serial + texture->serial) % 128 != 0))) {
            batch.texture_sources.emplace(
                texture->serial,
                BananzaNvn2Commands::TextureSource{known->second.hash, std::move(segments)});
            continue;
        }
        if (call_profile) {
            ++call_profile->submit.texture_hashes;
            call_profile->submit.texture_hash_bytes += size;
        }
        static const bool trace_hashes =
            std::getenv("EDEN_NX2_NVN_TRACE_TEXTURE_HASHES") != nullptr;
        if (trace_hashes && present_serial % 600 == 0) {
            const auto& desc = texture->description;
            const bool known_texture = known != texture_checks.end();
            LOG_INFO(Core_ARM,
                     "NVN 2 texture hashed: serial {} at {:#x}, {} bytes, format {:#x}, {}x{}, "
                     "{} levels, pool flags {:#x}, {} segments, {}, flush key {:#x} (was {:#x}), "
                     "map generation {} (was {}), checked at present {} of {}",
                     texture->serial, address, size, desc.format, desc.width, desc.height,
                     desc.levels, texture->storage.flags, segments.size(),
                     known_texture ? "known" : "new", flush_key,
                     known_texture ? known->second.flush_key : 0, virtual_map_generation,
                     known_texture ? known->second.map_generation : 0,
                     known_texture ? known->second.present : 0, present_serial);
        }
        const u8* source = segments.size() == 1 ? memory.GetSpan(segments.front().first, size)
                                                : nullptr;
        if (!source) {
            if (texture_scratch.size() < size) {
                texture_scratch.resize(size);
            }
            bool read = true;
            u64 offset{};
            for (const auto& [guest, bytes] : segments) {
                read = read && memory.ReadBlockUnsafe(guest, texture_scratch.data() + offset, bytes);
                offset += bytes;
            }
            if (!read) {
                note_failure(false);
                continue;
            }
            source = texture_scratch.data();
        }
        u64 hash{};
        const bool direct = source != texture_scratch.data();
        const auto pre = direct ? std::ranges::find_if(pre_hashes,
                                                       [&](const PreHash& entry) {
                                                           return entry.serial == texture->serial &&
                                                                  entry.source == source &&
                                                                  entry.size == size;
                                                       })
                                : pre_hashes.end();
        if (pre != pre_hashes.end()) {
            if (!pre_hashes_ready) {
                late_copy_workers.WaitForRequests();
                pre_hashes_ready = true;
            }
            hash = pre->hash;
        } else {
            static const bool serial_hash = std::getenv("EDEN_NX2_NVN_SERIAL_HASH") != nullptr;
            // The copy threads for late ranges are idle at this point of a submit.
            hash = TextureHash(source, size, serial_hash ? nullptr : &late_copy_workers);
        }
        if (direct && pre_hash_on) {
            hashed_textures.push_back({texture->serial, segments.front().first, size});
        }
        if (known == texture_checks.end() || known->second.hash != hash) {
            if (!Fits(texture_bytes, size, MaxTextureBytes)) {
                note_failure(true);
                continue;
            }
            texture_bytes += size;
            if (trace_texture_copies && size >= (1ULL << 20)) {
                auto& stat = texture_copy_stats[address];
                ++stat.count;
                stat.bytes += size;
            }
            std::vector<u8> snapshot(source, source + size);
            if (source != texture_scratch.data()) {
                hash = TextureHash(snapshot.data(), size, nullptr);
            }
            batch.texture_snapshots.emplace(texture->serial, std::move(snapshot));
        }
        texture_checks.insert_or_assign(
            texture->serial,
            TextureCheck{hash, present_serial, virtual_map_generation, flush_key});
        batch.texture_sources.emplace(
            texture->serial, BananzaNvn2Commands::TextureSource{hash, std::move(segments)});
    }
    if (!pre_hashes_ready) {
        late_copy_workers.WaitForRequests();
    }
    texture_ns = ElapsedNs(texture_start);
    };
    // The staged uniform ranges as runs: ranges of a pool that touch or lie within 256 bytes
    // go as one. With `later` the copy is left to the late-copy threads.
    const auto stage_runs = [&](std::vector<std::array<u64, 5>>& wanted,
                                std::vector<LateCopy>* later, size_t shard,
                                std::vector<Submission::MirrorPage>& runs, bool& incomplete) {
        std::ranges::sort(wanted);
        for (size_t i = 0; i < wanted.size();) {
            const u64 serial = wanted[i][0];
            const u64 storage = wanted[i][1];
            const u64 pool_size = wanted[i][2];
            const u64 begin = wanted[i][3];
            u64 end = wanted[i][4];
            for (++i; i < wanted.size() && wanted[i][0] == serial && wanted[i][3] <= end + 256;
                 ++i) {
                end = std::max(end, wanted[i][4]);
            }
            const u64 bytes = end - begin;
            const u8* const host = memory.GetSpan(storage + begin, bytes);
            if (!host) {
                incomplete = true;
                continue;
            }
            u8* const dest = batch.snapshot_blocks[shard].Allocate(bytes);
            if (later) {
                later->push_back({dest, host, bytes});
            } else {
                std::memcpy(dest, host, bytes);
            }
            runs.push_back({serial, pool_size, begin, dest, bytes});
        }
    };
    std::array<bool, Submission::SnapshotShards> runs_incomplete{};
    if (deferring) {
        deferring = false;
        std::array<ScanContext, Submission::SnapshotShards> contexts;
        const auto run = [&](size_t shard) {
            auto& context = contexts[shard];
            context.whole = false;
            for (const Request& request : requests[shard]) {
                if (request.kind == 2) {
                    reference_range(context, request.address, request.size);
                } else {
                    copy_range(context, request.address, request.size, request.kind);
                }
            }
            stage_runs(context.pages, &context.late, shard, context.runs, runs_incomplete[shard]);
        };
        if (request_count >= 512) {
            const std::vector<PendingWrites> pending_copy = pending_writes;
            pending_view = &pending_copy;
            for (size_t shard = 0; shard < contexts.size(); ++shard) {
                copy_workers.QueueWork([&run, shard] { run(shard); });
            }
            texture_stage();
            copy_workers.WaitForRequests();
            pending_view = &pending_writes;
        } else {
            for (size_t shard = 0; shard < contexts.size(); ++shard) {
                run(shard);
            }
        }
        for (auto& context : contexts) {
            note_storage(context);
            batch.after_writes = std::max(batch.after_writes, context.after_writes);
            fixed_candidates.insert(fixed_candidates.end(), context.fixed.begin(),
                                    context.fixed.end());
            fixed_bytes += context.fixed_bytes;
            referenced_bytes += context.referenced_bytes;
            referenced_ranges += context.referenced_ranges;
        }
        // The large copies start now and may end after the replay has begun.
        std::vector<LateCopy> pieces;
        constexpr u64 PieceBytes = 256ULL << 10;
        for (const auto& context : contexts) {
            for (const LateCopy& copy : context.late) {
                for (u64 offset = 0; offset < copy.size; offset += PieceBytes) {
                    pieces.push_back({copy.dest + offset, copy.source + offset,
                                      std::min(PieceBytes, copy.size - offset)});
                }
            }
        }
        for (size_t shard = 0; shard < contexts.size(); ++shard) {
            batch.mirror_pages.insert(batch.mirror_pages.end(), contexts[shard].runs.begin(),
                                      contexts[shard].runs.end());
            batch.mirror_pages_incomplete |= runs_incomplete[shard];
        }
        if (!pieces.empty()) {
            const u32 tasks = static_cast<u32>(std::min<size_t>(4, pieces.size()));
            auto state = std::make_shared<Submission::CopyState>();
            state->left.store(tasks, std::memory_order_release);
            batch.late_copies = state;
            for (u32 task = 0; task < tasks; ++task) {
                std::vector<LateCopy> part;
                for (size_t i = task; i < pieces.size(); i += tasks) {
                    part.push_back(pieces[i]);
                }
                late_copy_workers.QueueWork([state, part = std::move(part)] {
                    for (const LateCopy& copy : part) {
                        std::memcpy(copy.dest, copy.source, copy.size);
                    }
                    state->Done();
                });
            }
        }
        // What the other threads left: ranges with pending GPU writes or without one host span.
        for (const auto& context : contexts) {
            for (const Request& request : context.later) {
                if (request.kind == 2) {
                    reference_range(main_context, request.address, request.size);
                } else {
                    copy_range(main_context, request.address, request.size, request.kind);
                }
            }
        }
        note_storage(main_context);
    }
    stage_runs(main_context.pages, nullptr, 0, batch.mirror_pages,
               batch.mirror_pages_incomplete);
    main_context.pages.clear();
    for (const auto& [list_address, list_bytes] : reset_lists) {
        copy_range(main_context, list_address, list_bytes, 0);
        drain();
        const auto bytes = batch.BufferSnapshot({list_address, list_bytes});
        if (bytes.empty()) {
            continue;
        }
        for (size_t offset = 0; offset < list_bytes; offset += sizeof(u64)) {
            u64_le record;
            std::memcpy(&record, bytes.data() + offset, sizeof(record));
            if (batch.indirect_reset_addresses.contains(record)) {
                continue;
            }
            constexpr size_t RecordSize = sizeof(VideoCommon::NativeIndirectReset::Record);
            if (const auto guest = ResolveGpuRange(record, RecordSize);
                guest && ValidRange(memory, *guest, RecordSize)) {
                batch.indirect_reset_addresses.emplace(record, *guest);
            }
        }
    }
    for (const auto& key : wanted_slices) {
        if (batch.indirect_index_slices.contains(key)) {
            continue;
        }
        const auto& [arguments, address, type] = key;
        std::array<u32_le, 5> words{};
        const bool read = read_gpu(arguments, {reinterpret_cast<u8*>(words.data()),
                                               sizeof(words)});
        const u64 start = u64{words[2]} << type;
        const u64 bytes = u64{words[0]} << type;
        const bool valid = read && bytes && words[1] && bytes <= MaxRange &&
                           address <= std::numeric_limits<u64>::max() - start;
        batch.indirect_index_slices.try_emplace(key, valid ? address + start : 0,
                                               valid ? bytes : 0);
    }
    // Keep the order of indirect snapshots when they compete for the remaining byte budget.
    std::vector<Range> indirect_args(indirect_arg_requests.begin(), indirect_arg_requests.end());
    std::ranges::sort(indirect_args);
    for (const auto& [address, size] : indirect_args) {
        snapshot(address, size);
    }
    std::erase_if(batch.indirect_index_slices, [&](const auto& entry) {
        const auto& [address, size] = entry.second;
        reference(address, size);
        return !batch.HasSnapshot(entry.second);
    });
    std::vector<Range> indirect_indices(indirect_index_requests.begin(), indirect_index_requests.end());
    std::ranges::sort(indirect_indices);
    for (const auto& [address, size] : indirect_indices) {
        reference(address, size);
        if (batch.HasSnapshot({address, size})) {
            batch.indirect_index_sizes.try_emplace(address, size);
        }
    }
    // A range the GPU writes is never kept: its bytes change without a flush.
    constexpr u64 ResetRecordSize = sizeof(VideoCommon::NativeIndirectReset::Record);
    bool written_grew{};
    const auto add_written = [&](u64 start, u64 end) {
        const auto next = gpu_written_guest.upper_bound(start);
        if (next != gpu_written_guest.begin() && std::prev(next)->second >= end) {
            return;
        }
        AddRange(gpu_written_guest, start, end);
        written_grew = true;
    };
    for (const auto& [range, stored] : batch.storage_addresses) {
        add_written(stored.guest, stored.guest + range.second);
    }
    for (const auto& [record, guest] : batch.indirect_reset_addresses) {
        add_written(guest, guest + ResetRecordSize);
    }
    // Ranges tagged before this submission's first GPU write over them was seen.
    if (written_grew) {
        for (const auto& candidate : fixed_candidates) {
            if (!Overlaps(gpu_written_guest, candidate.guest,
                          candidate.guest + candidate.key.second)) {
                continue;
            }
            if (auto* const found = batch.FindSnapshot(candidate.key)) {
                found->fixed_pool = 0;
                fixed_bytes -= candidate.key.second;
            }
        }
    }
    if (call_profile) {
        call_profile->submit.fixed_bytes += fixed_bytes;
        call_profile->submit.fixed_ranges += fixed_candidates.size();
    }
    drain();
    if (!textures_done) {
        texture_stage();
    }
    batch.after_writes = std::max(batch.after_writes, main_context.after_writes);
    for (const u64 serial : written_textures) {
        if (serial) {
            gpu_written_textures.insert(serial);
        }
    }
    if (call_profile) {
        auto& stats = call_profile->submit;
        stats.scan_ns += ElapsedNs(scan_start) - texture_ns;
        stats.texture_ns += texture_ns;
        stats.copy_wait_ns += copy_wait_ns;
        stats.buffer_bytes += buffer_bytes;
        stats.texture_bytes += texture_bytes;
        stats.buffer_ranges += batch.SnapshotCount();
        stats.referenced_bytes += referenced_bytes;
        stats.referenced_ranges += referenced_ranges;
        stats.textures += batch.texture_snapshots.size();
    }
    if (dirty_stats && dirty_stats->submissions % 1200 == 0) {
        static constexpr std::array kinds{"copied", "referenced"};
        static constexpr std::array names{"CPU_CACHED", "CPU_UNCACHED", "CPU_NO_ACCESS", "no pool"};
        for (size_t kind = 0; kind < 2; ++kind) {
            for (size_t i = 0; i < 4; ++i) {
                auto& stats = dirty_stats->classes[kind][i];
                if (stats.pages) {
                    LOG_INFO(Core_ARM,
                             "NVN 2 dirty stats: {} {}: per submission {:.1f} pages looked at "
                             "({:.2f} MiB), {:.1f} new, {:.1f} changed ({:.3f} MiB), of them {:.1f} "
                             "without a newer flush and {:.1f} in ranges the GPU writes",
                             kinds[kind], names[i], stats.pages / 1200.0,
                             stats.pages / 1200.0 / 256.0, stats.fresh / 1200.0,
                             stats.changed / 1200.0, stats.changed / 1200.0 / 256.0,
                             stats.changed_unflushed / 1200.0,
                             stats.changed_gpu_written / 1200.0);
                }
                stats = {};
            }
        }
    }
}

u64 Session::InitializeBuffer(Memory::Memory& memory, const GuestCall& call) {
    const auto& x = call.x;
    if (!ValidRange(memory, x[0], 0x28) || !ValidRange(memory, x[1], 0x28)) {
        return 0;
    }
    Buffer buffer{
        .device = memory.Read64(x[1] + 0x08),
        .pool = memory.Read64(x[1] + 0x10),
        .offset = memory.Read64(x[1] + 0x18),
        .size = memory.Read64(x[1] + 0x20),
    };
    {
        // The pools and the buffer table: no other session state, so no wait for a scan.
        std::unique_lock lock{pool_buffer_mutex};
        const auto pool = pools.find(buffer.pool);
        if (pool == pools.end() || pool->second.device != buffer.device ||
            pool->second.flags & PhysicalPool ||
            !Fits(buffer.offset, buffer.size, pool->second.size)) {
            return 0;
        }
        buffer.pool_address = pool->second.gpu_address;
        buffers.insert_or_assign(x[0], buffer);
    }
    memory.Write64(x[0], 0);
    memory.Write64(x[0] + 0x08, buffer.device);
    memory.Write64(x[0] + 0x10, buffer.pool);
    memory.Write64(x[0] + 0x18, buffer.offset);
    memory.Write64(x[0] + 0x20, buffer.size);
    return 1;
}

std::optional<u64> Session::HandleBuffer(Memory::Memory& memory, std::string_view name,
                                        const GuestCall& call) {
    const auto& x = call.x;
    if (name == "nvnBufferInitialize") {
        return InitializeBuffer(memory, call);
    } else if (name == "nvnBufferFinalize") {
        std::unique_lock lock{pool_buffer_mutex};
        buffers.erase(x[0]);
    } else {
        const auto it = buffers.find(x[0]);
        if (it == buffers.end()) {
            return std::nullopt;
        }
        const Buffer& buffer = it->second;
        const auto pool = pools.find(buffer.pool);
        if (pool == pools.end() || pool->second.gpu_address != buffer.pool_address) {
            return std::nullopt;
        }
        if (name == "nvnBufferGetSize") {
            return buffer.size;
        }
        if (name == "nvnBufferGetMemoryPool") {
            return buffer.pool;
        }
        if (name == "nvnBufferGetMemoryOffset") {
            return buffer.offset;
        }
        if (name == "nvnBufferFlushMappedRange" || name == "nvnBufferInvalidateMappedRange") {
            if (!pool->second.storage || !Fits(x[1], x[2], buffer.size)) {
                NoteMissing(fmt::format("{} invalid range", name));
            } else if (name == "nvnBufferFlushMappedRange") {
                NoteFlush(pool->second, buffer.offset + x[1], x[2]);
            } else if (name == "nvnBufferInvalidateMappedRange" && WaitInvalidate()) {
                WaitForPendingWrites(pool->second.storage + buffer.offset + x[1], x[2],
                                     "buffer invalidate");
            }
            return 0;
        }
        if (name != "nvnBufferSetDebugLabel") {
            return std::nullopt;
        }
    }
    return 0;
}

bool Session::MapVirtual(Memory::Memory& memory, const GuestCall& call) {
    struct MappingRequest {
        u64_le physical_pool;
        u64_le physical_offset;
        u64_le virtual_offset;
        u64_le size;
        u32_le storage_class;
        u32_le reserved;
    };
    static_assert(sizeof(MappingRequest) == 40);

    const auto& x = call.x;
    const auto target = pools.find(x[0]);
    const u64 count = static_cast<u32>(x[1]);
    if (target == pools.end() || !(target->second.flags & VirtualPool) || count > 0x10000 ||
        (count && !ValidRange(memory, x[2], count * sizeof(MappingRequest)))) {
        return false;
    }
    std::vector<MappingRequest> requests(count);
    if (count && !memory.ReadBlock(x[2], requests.data(), count * sizeof(MappingRequest))) {
        return false;
    }
    const Pool& pool = target->second;
    u64 total_pages = 0;
    for (const auto& request : requests) {
        if ((request.physical_offset | request.virtual_offset | request.size) % PoolAlignment ||
            !Fits(request.virtual_offset, request.size, pool.size) ||
            !Fits(total_pages, request.size / PoolAlignment, 1ULL << 20)) {
            return false;
        }
        total_pages += request.size / PoolAlignment;
        if (!request.physical_pool) {
            continue;
        }
        const auto source = pools.find(request.physical_pool);
        if (source == pools.end() || !(source->second.flags & PhysicalPool) ||
            source->second.device != pool.device ||
            !Fits(request.physical_offset, request.size, source->second.size)) {
            return false;
        }
    }
    for (const auto& request : requests) {
        const u64 start = pool.gpu_address + request.virtual_offset;
        const u64 end = start + request.size;
        if (!request.physical_pool) {
            for (u64 address = start; address < end; address += PoolAlignment) {
                virtual_pages.erase(address);
            }
            continue;
        }
        const Pool& source = pools.at(request.physical_pool);
        for (u64 address = start; address < end; address += PoolAlignment) {
            virtual_pages.insert_or_assign(address,
                VirtualPage{request.physical_pool, source.gpu_address,
                            request.physical_offset + (address - start), request.storage_class});
        }
    }
    return true;
}

void Session::UpdateCommandBuffer(Memory::Memory& memory, u64 address,
                                  const CommandBuffer& object,
                                  const BananzaNvn2Commands::State& state) {
    // Keep fields read by inlined SDK helpers coherent with the host recorder.
    const u64 command_cursor = object.command_storage + state.command_used;
    const u64 command_end = object.command_storage + state.command_size;
    const u64 control_cursor = state.control_base + state.control_used;
    const u64 control_end = state.control_base + state.control_size;
    memory.Write64(address + 0x00, command_cursor);
    memory.Write64(address + 0x08, command_end);
    memory.Write8(address + 0x11, state.recording);
    memory.Write8(address + 0x14, object.command_callback);
    memory.Write8(address + 0x15, object.control_callback);
    memory.Write32(address + 0x18, object.recording_mode);
    memory.Write64(address + 0x20, command_end);
    memory.Write64(address + 0x28, control_cursor);
    memory.Write64(address + 0x30, state.control_size >= 16 ? control_end - 16 : control_end);
    memory.Write64(address + 0x38, object.command_storage);
    memory.Write64(address + 0x40, state.control_base);
    memory.Write64(address + 0x60, object.gpu_address);
    memory.Write64(address + 0x78, object.callback);
    memory.Write64(address + 0x80, object.callback_data);
}

void Session::UpdateCommandCursor(Memory::Memory& memory, u64 address,
                                  const CommandBuffer& object) {
    if (const auto state = recorder.GetState(address)) {
        memory.Write64(address, object.command_storage + state->command_used);
    }
}

void Session::FinishSubmit(Memory::Memory& memory, Queue& queue,
                           BananzaNvn2Commands::Submission& submission,
                           std::unique_lock<std::mutex>& lock) {
    ApplyPendingFlushes();
    // The operations in the order they run are only read by the replay: a helper thread
    // lists them while the scan runs.
    const auto flattened = std::make_shared<BananzaNvn2Commands::Submission::CopyState>();
    flattened->left.store(1, std::memory_order_release);
    late_copy_workers.QueueWork([flattened, &submission] {
        BananzaNvn2Commands::FlattenRecordings(submission);
        flattened->Done();
    });
    SnapshotResources(memory, submission, queue);
    flattened->Wait();
    ++accepted_submissions;
    accepted_commands += submission.command_count;
    if (recording_stats) {
        CountRecordings(submission);
    }
    if (!logged_submit) {
        logged_submit = true;
        LOG_INFO(Core_ARM,
                 "Bananza NVN 2 HLE accepted submission {} with {} handles and {} commands",
                 submission.sequence, submission.handles.size(), submission.command_count);
    }
    const bool may_write_storage = !submission.storage_addresses.empty() ||
                                   !submission.indirect_reset_addresses.empty();
    std::map<u64, u64> writes;
    const u64 sequence = submission.sequence;
    submission.host_queued_ns = BananzaNvn2Commands::HostNs();
    if (may_write_storage && !SyncSubmits()) {
        // Sorted and merged first: thousands of ranges become a few map entries.
        std::vector<std::pair<u64, u64>> spans;
        spans.reserve(submission.storage_addresses.size() +
                      submission.indirect_reset_addresses.size());
        for (const auto& [range, stored] : submission.storage_addresses) {
            spans.emplace_back(stored.guest, stored.guest + range.second);
        }
        constexpr u64 RecordSize = sizeof(VideoCommon::NativeIndirectReset::Record);
        for (const auto& [record, guest] : submission.indirect_reset_addresses) {
            spans.emplace_back(guest, guest + RecordSize);
        }
        std::ranges::sort(spans);
        for (size_t i = 0; i < spans.size();) {
            auto [start, end] = spans[i];
            for (++i; i < spans.size() && spans[i].first <= end; ++i) {
                end = std::max(end, spans[i].second);
            }
            writes.emplace_hint(writes.end(), start, end);
        }
    }
    const auto late_copies = submission.late_copies;
    const u64 replay_fence = system.GPU().SubmitRendererWork(
        [batch = std::make_shared<const BananzaNvn2Commands::Submission>(
             std::move(submission))](VideoCore::RendererBase& renderer) {
            renderer.ReplayBananzaNvn2(*batch);
        });
    {
        std::scoped_lock frame_lock{frame_mutex};
        queue.fence = replay_fence;
    }
    // The replay runs already; the game gets its memory back once the copies are made.
    if (late_copies) {
        late_copies->Wait();
    }
    {
        std::scoped_lock frame_lock{frame_mutex};
        queue.sequence = sequence;
        queue.submit_host_ns = static_cast<u64>(
            std::chrono::steady_clock::now().time_since_epoch().count());
    }
    if (call_profile) {
        ++call_profile->submit.submits;
    }
    if (may_write_storage && !SyncSubmits()) {
        if (call_profile && logged_overlaps < 48) {
            LOG_INFO(Core_ARM, "Bananza NVN 2 submission {} is fence {}", sequence,
                     queue.fence);
        }
        pending_writes.push_back({queue.fence, sequence, std::move(writes)});
    } else if (may_write_storage) {
        const u64 fence = queue.fence;
        pending_storage_fence = fence;
        if (call_profile) {
            ++call_profile->submit.waits;
        }
        const auto wait_start = std::chrono::steady_clock::now();
        lock.unlock();
        system.GPU().WaitForSyncOperation(fence);
        BananzaNvn2Commands::GetRetirement().WaitWrites(sequence);
        lock.lock();
        if (pending_storage_fence == fence) {
            pending_storage_fence = 0;
        }
        if (call_profile) {
            call_profile->submit.wait_ns += ElapsedNs(wait_start);
        }
    }
}

bool Session::AcquireAsync(Memory::Memory& memory, const GuestCall& call) {
    const auto& x = call.x;
    std::scoped_lock frame_lock{frame_mutex};
    const auto found = windows.find(x[0]);
    if (found == windows.end()) {
        return false;
    }
    Window& window = found->second;
    const auto sync = syncs.find(x[1]);
    if ((x[1] && (sync == syncs.end() || sync->second.device != window.device)) ||
        !ValidRange(memory, x[2], sizeof(u32))) {
        return false;
    }
    for (u32 i = 0; i < window.active; ++i) {
        const u32 image = (window.next + i) % window.active;
        if (window.acquired[image]) {
            continue;
        }
        window.next = (image + 1) % window.active;
        window.acquired[image] = true;
        memory.Write32(x[2], image);
        if (sync != syncs.end()) {
            sync->second.status = SyncStatus::Signaled;
            sync->second.fence = 0;
            sync->second.sequence = 0;
        }
        return true;
    }
    return false;
}

bool Session::FenceSyncAsync(const GuestCall& call) {
    const auto& x = call.x;
    std::scoped_lock frame_lock{frame_mutex};
    const auto queue = queues.find(x[0]);
    const auto sync = syncs.find(x[1]);
    if (queue == queues.end() || sync == syncs.end() ||
        queue->second.device != sync->second.device) {
        return false;
    }
    sync->second.status =
        queue->second.rejected_submit ? SyncStatus::Failed : SyncStatus::Signaled;
    // The fence may still be an earlier submission's: the sequence is what a wait goes by.
    sync->second.fence = queue->second.fence;
    sync->second.sequence = queue->second.sequence;
    sync->second.submit_host_ns = queue->second.submit_host_ns;
    return true;
}

u64 Session::PresentAsync(const GuestCall& call,
                          std::chrono::steady_clock::time_point* present_deadline) {
    const auto& x = call.x;
    std::unique_lock frame_lock{frame_mutex};
    const auto queue = queues.find(x[0]);
    const auto window = windows.find(x[1]);
    const u32 image = static_cast<u32>(x[2]);
    const bool valid = queue != queues.end() && window != windows.end() &&
                       queue->second.device == window->second.device &&
                       image < window->second.textures.size() && window->second.acquired[image];
    std::optional<ResourceTextureInfo> texture;
    if (valid) {
        window->second.acquired[image] = false;
        std::shared_lock resource_lock{resource_mutex};
        if (const auto found = resources.GetTexture(window->second.textures[image])) {
            texture = *found;
        }
    }
    if (!texture) {
        frame_lock.unlock();
        std::scoped_lock session_lock{mutex};
        NoteMissing(valid ? "nvnQueuePresentTexture invalid texture"
                          : "nvnQueuePresentTexture invalid queue or image");
        return 0;
    }
    const auto& description = texture->description;
    Tegra::FramebufferConfig config{
        .width = description.width,
        .height = description.height,
        .stride = description.width,
        .pixel_format = Service::android::PixelFormat::Rgba8888,
    };
    // The colour space follows the format, as for a native present.
    if (description.format == 0x29) {
        config.pixel_format = Service::android::PixelFormat::RgbaFp16;
        config.color_space = Tegra::ColorSpace::RGB709_Linear;
    } else if (description.format == 0x3d) {
        config.pixel_format = Service::android::PixelFormat::Rgba1010102;
        if (Common::DisplayHdrMonitor::Instance().GuestToldHdr()) {
            config.color_space = Tegra::ColorSpace::RGB2020_PQ;
        }
    }
    const auto& crop = window->second.crop;
    if (crop[2] && crop[3] && Fits(crop[0], crop[2], config.width) &&
        Fits(crop[1], crop[3], config.height)) {
        config.crop_rect = {static_cast<int>(crop[0]), static_cast<int>(crop[1]),
                            static_cast<int>(crop[0] + crop[2]),
                            static_cast<int>(crop[1] + crop[3])};
    }
    const auto present_call = std::chrono::steady_clock::now();
    const auto game_stage = perf_frame_begin.time_since_epoch().count()
                                ? present_call - perf_frame_begin
                                : std::chrono::steady_clock::duration::zero();
    perf_frame_begin = present_call;
    presents_outstanding.fetch_add(1, std::memory_order_acq_rel);
    present_marks = {queue->second.sequence, present_marks[0]};
    // Behind the scans of the submits made before this call, on the same thread.
    submit_lane->Queue([this, handle = x[0], shown = std::move(*texture), config, game_stage] {
        std::scoped_lock session_lock{mutex};
        ++present_serial;
        if (recording_stats) {
            ++recording_stats->presents;
        }
        if (call_profile) {
            std::scoped_lock profile_lock{call_profile->mutex};
            ++call_profile->presents;
        }
        const u64 fence = system.GPU().SubmitRendererWork(
            [this, shown, config, game_stage,
             perf = &system.GetPerfStats()](VideoCore::RendererBase& renderer) {
                const auto start = std::chrono::steady_clock::now();
                renderer.PresentBananzaNvn2(shown, config);
                const std::chrono::nanoseconds replay_stage =
                    std::chrono::nanoseconds{BananzaNvn2Commands::g_replay_busy_ns.exchange(
                        0, std::memory_order_relaxed)} +
                    (std::chrono::steady_clock::now() - start);
                perf->AddSystemFrame(std::max<std::chrono::nanoseconds>(game_stage, replay_stage));
                {
                    std::scoped_lock wait_lock{present_wait_mutex};
                    presents_outstanding.fetch_sub(1, std::memory_order_acq_rel);
                }
                present_done.notify_all();
            });
        std::scoped_lock frame_lock{frame_mutex};
        if (const auto found = queues.find(handle); found != queues.end()) {
            found->second.fence = fence;
        }
    });
    const u16 speed = Settings::SpeedLimitEnabled() ? Settings::SpeedLimit() : 0;
    const auto interval = window->second.interval;
    // Experiment: EDEN_NX2_NVN_UNPACED=1 never sleeps to the display's grid.
    static const bool unpaced = std::getenv("EDEN_NX2_NVN_UNPACED") != nullptr;
    static u32 unpaced_interval{};
    if (unpaced && interval != unpaced_interval) {
        unpaced_interval = interval;
        LOG_INFO(Core_ARM, "Bananza NVN 2 present interval {}, not paced", interval);
    }
    const auto period = speed && interval && !unpaced
        ? std::chrono::nanoseconds{100'000'000'000LL * std::min(interval, 4U) / 60 / speed}
        : std::chrono::nanoseconds::zero();
    if (period != present_period) {
        LOG_INFO(Core_ARM, "Bananza NVN 2 present interval {}, pacing period {} us", interval,
                 period.count() / 1000);
        present_pacer = {};
        present_period = period;
        VideoCommon::Nx2Perf::g_guest_frame_interval_ns.store(period.count(),
                                                             std::memory_order_relaxed);
    }
    if (period.count() && present_deadline) {
        *present_deadline =
            present_pacer.Advance(std::chrono::steady_clock::now(), period).deadline;
    }
    if (present_deadline && *present_deadline > perf_frame_begin) {
        perf_frame_begin = *present_deadline;
    }
    frame_lock.unlock();
    // Nothing else keeps the game near the replay.
    constexpr u32 PresentsAhead = 2;
    if (presents_outstanding.load(std::memory_order_acquire) > PresentsAhead) {
        std::unique_lock wait_lock{present_wait_mutex};
        present_done.wait_for(wait_lock, std::chrono::milliseconds{250}, [&] {
            return presents_outstanding.load(std::memory_order_acquire) <= PresentsAhead;
        });
    }
    return 0;
}

std::optional<u64> Session::HandleQueue(Memory::Memory& memory, std::string_view name,
                                       const GuestCall& call, std::unique_lock<std::mutex>& lock,
                                       std::chrono::steady_clock::time_point* present_deadline) {
    const auto& x = call.x;
    if (name == "nvnQueueBuilderSetDefaults") {
        const auto device = queue_builders[x[0]].device;
        queue_builders[x[0]] = {.device = device, .command_size = 65536,
                               .compute_size = 262144, .control_size = 16384};
    } else if (name == "nvnQueueBuilderSetDevice") {
        queue_builders[x[0]].device = x[1];
    } else if (name == "nvnQueueBuilderSetFlags") {
        queue_builders[x[0]].flags = static_cast<u32>(x[1]);
    } else if (name == "nvnQueueBuilderSetCommandMemorySize") {
        queue_builders[x[0]].command_size = x[1];
    } else if (name == "nvnQueueBuilderSetComputeMemorySize") {
        queue_builders[x[0]].compute_size = x[1];
    } else if (name == "nvnQueueBuilderSetControlMemorySize") {
        queue_builders[x[0]].control_size = x[1];
    } else if (name == "nvnQueueBuilderSetCommandFlushThreshold") {
        queue_builders[x[0]].flush_threshold = x[1];
    } else if (name == "nvnQueueBuilderSetPrimaryQueue") {
        queue_builders[x[0]].primary = x[1];
    } else if (name == "nvnQueueBuilderSetQueueMemory") {
        auto& builder = queue_builders[x[0]];
        builder.storage = x[1];
        builder.storage_size = x[2];
    } else if (name == "nvnQueueBuilderGetQueueMemorySize") {
        const auto builder = queue_builders.find(x[0]);
        if (builder == queue_builders.end()) {
            return 0;
        }
        const Queue& q = builder->second;
        constexpr u64 MaxSize = 1ULL << 40;
        if (q.command_size > MaxSize || q.compute_size > MaxSize || q.control_size > MaxSize) {
            return 0;
        }
        return Common::AlignUp(q.command_size + q.compute_size + q.control_size, u64{4096});
    } else if (name == "nvnQueueInitialize") {
        const auto builder = queue_builders.find(x[1]);
        if (!x[0] || builder == queue_builders.end() || !devices.contains(builder->second.device)) {
            return 0;
        }
        queues.insert_or_assign(x[0], builder->second);
        return 1;
    } else if (name == "nvnQueueFinalize") {
        queues.erase(x[0]);
    } else if (name == "nvnQueueSubmitCommands") {
        // A new snapshot may read data written back by the previous submit.
        while (SyncSubmits() && pending_storage_fence) {
            const u64 fence = pending_storage_fence;
            lock.unlock();
            system.GPU().WaitForSyncOperation(fence);
            lock.lock();
            if (pending_storage_fence == fence) {
                pending_storage_fence = 0;
            }
        }
        const auto queue = queues.find(x[0]);
        const auto handles = ReadHandles(memory, x[2], static_cast<u32>(x[1]));
        auto submission = queue != queues.end() && handles
                              ? recorder.Submit(x[0], *handles)
                              : std::nullopt;
        if (submission) {
            submission->submitted_ns = static_cast<u64>(system.CoreTiming().GetGlobalTimeNs().count());
            submission->host_submit_ns = BananzaNvn2Commands::HostNs();
            if (const auto device = devices.find(queue->second.device); device != devices.end()) {
                submission->depth_minus_one_to_one = device->second.depth_mode == 0;
                submission->origin_lower_left = device->second.origin == 0;
            }
            if (submit_lane) {
                // The sequence is the queue's at once: a sync fenced now waits for it.
                {
                    std::scoped_lock frame_lock{frame_mutex};
                    queue->second.sequence = submission->sequence;
                }
                submit_lane->Queue([this, &memory, handle = x[0],
                                    batch = std::move(*submission)]() mutable {
                    std::unique_lock scan_lock{mutex};
                    if (const auto found = queues.find(handle); found != queues.end()) {
                        FinishSubmit(memory, found->second, batch, scan_lock);
                    }
                });
            } else {
                FinishSubmit(memory, queue->second, *submission, lock);
            }
        } else {
            ++rejected_submissions;
            if (queue != queues.end()) {
                queue->second.rejected_submit = true;
            }
            NoteMissing("nvnQueueSubmitCommands invalid recording handles");
        }
    } else if (name == "nvnQueueGetError") {
        const auto queue = queues.find(x[0]);
        if (queue != queues.end() && !queue->second.rejected_submit) {
            return 0;
        }
        if (ValidRange(memory, x[1], 16)) {
            constexpr std::array<u8, 16> info{};
            memory.WriteBlock(x[1], info.data(), info.size());
        }
        return 1;
    } else if (name == "nvnQueueFenceSync") {
        std::scoped_lock frame_lock{frame_mutex};
        const auto queue = queues.find(x[0]);
        const auto sync = syncs.find(x[1]);
        if (queue == queues.end() || sync == syncs.end() ||
            queue->second.device != sync->second.device) {
            NoteMissing("nvnQueueFenceSync invalid queue or sync");
        } else {
            sync->second.status =
                queue->second.rejected_submit ? SyncStatus::Failed : SyncStatus::Signaled;
            sync->second.fence = queue->second.fence;
            sync->second.sequence = queue->second.sequence;
            sync->second.submit_host_ns = queue->second.submit_host_ns;
        }
    } else if (name == "nvnQueueWaitSync") {
        std::scoped_lock frame_lock{frame_mutex};
        const auto queue = queues.find(x[0]);
        const auto sync = syncs.find(x[1]);
        if (queue == queues.end() || sync == syncs.end() ||
            queue->second.device != sync->second.device ||
            sync->second.status != SyncStatus::Signaled) {
            NoteMissing("nvnQueueWaitSync unresolved sync");
            return 0;
        }
        return 1;
    } else if (name == "nvnQueuePresentTexture") {
        const auto queue = queues.find(x[0]);
        const auto window = windows.find(x[1]);
        const u32 image = static_cast<u32>(x[2]);
        if (queue == queues.end() || window == windows.end() ||
            queue->second.device != window->second.device ||
            image >= window->second.textures.size() || !window->second.acquired[image]) {
            NoteMissing("nvnQueuePresentTexture invalid queue or image");
        } else {
            window->second.acquired[image] = false;
            const auto texture = resources.GetTexture(window->second.textures[image]);
            if (!texture) {
                NoteMissing("nvnQueuePresentTexture invalid texture");
                return 0;
            }
            const auto& description = texture->description;
            ++present_serial;
            if (recording_stats) {
                ++recording_stats->presents;
            }
            if (call_profile) {
                std::scoped_lock profile_lock{call_profile->mutex};
                ++call_profile->presents;
            }
            if (call_profile && BananzaNvn2Commands::InTimelineWindow(queue->second.sequence)) {
                LOG_INFO(Core_ARM, "NVN 2 timeline: seq {} present call at {:.3f} ms",
                         queue->second.sequence, BananzaNvn2Commands::HostNs() / 1e6);
            }
            Tegra::FramebufferConfig config{
                .width = description.width,
                .height = description.height,
                .stride = description.width,
                .pixel_format = Service::android::PixelFormat::Rgba8888,
            };
            // The colour space follows the format, as for a native present.
            if (description.format == 0x29) {
                config.pixel_format = Service::android::PixelFormat::RgbaFp16;
                config.color_space = Tegra::ColorSpace::RGB709_Linear;
            } else if (description.format == 0x3d) {
                config.pixel_format = Service::android::PixelFormat::Rgba1010102;
                if (Common::DisplayHdrMonitor::Instance().GuestToldHdr()) {
                    config.color_space = Tegra::ColorSpace::RGB2020_PQ;
                }
            }
            const auto& crop = window->second.crop;
            if (crop[2] && crop[3] && Fits(crop[0], crop[2], config.width) &&
                Fits(crop[1], crop[3], config.height)) {
                config.crop_rect = {static_cast<int>(crop[0]), static_cast<int>(crop[1]),
                                    static_cast<int>(crop[0] + crop[2]),
                                    static_cast<int>(crop[1] + crop[3])};
            }
            // The status bar's frame time is the longer stage of this frame: the game thread
            // up to here, or the GPU thread in its replays and the present.
            const auto present_call = std::chrono::steady_clock::now();
            const auto game_stage = perf_frame_begin.time_since_epoch().count()
                                        ? present_call - perf_frame_begin
                                        : std::chrono::steady_clock::duration::zero();
            perf_frame_begin = present_call;
            queue->second.fence = system.GPU().SubmitRendererWork(
                [texture = *texture, config, game_stage,
                 perf = &system.GetPerfStats()](VideoCore::RendererBase& renderer) {
                    const auto start = std::chrono::steady_clock::now();
                    renderer.PresentBananzaNvn2(texture, config);
                    const std::chrono::nanoseconds replay_stage =
                        std::chrono::nanoseconds{BananzaNvn2Commands::g_replay_busy_ns.exchange(
                            0, std::memory_order_relaxed)} +
                        (std::chrono::steady_clock::now() - start);
                    perf->AddSystemFrame(std::max<std::chrono::nanoseconds>(game_stage,
                                                                           replay_stage));
                });
            const u16 speed = Settings::SpeedLimitEnabled() ? Settings::SpeedLimit() : 0;
            const auto interval = window->second.interval;
            // Experiment: EDEN_NX2_NVN_UNPACED=1 never sleeps to the display's grid.
            static const bool unpaced = std::getenv("EDEN_NX2_NVN_UNPACED") != nullptr;
            static u32 unpaced_interval{};
            if (unpaced && interval != unpaced_interval) {
                unpaced_interval = interval;
                LOG_INFO(Core_ARM, "Bananza NVN 2 present interval {}, not paced", interval);
            }
            const auto period = speed && interval && !unpaced
                ? std::chrono::nanoseconds{100'000'000'000LL * std::min(interval, 4U) / 60 / speed}
                : std::chrono::nanoseconds::zero();
            if (period != present_period) {
                LOG_INFO(Core_ARM, "Bananza NVN 2 present interval {}, pacing period {} us", interval,
                         period.count() / 1000);
                present_pacer = {};
                present_period = period;
                VideoCommon::Nx2Perf::g_guest_frame_interval_ns.store(period.count(),
                                                                     std::memory_order_relaxed);
            }
            if (period.count() && present_deadline) {
                *present_deadline =
                    present_pacer.Advance(std::chrono::steady_clock::now(), period).deadline;
            }
            if (present_deadline && *present_deadline > perf_frame_begin) {
                perf_frame_begin = *present_deadline;
            }
            // Without the submit waits nothing else keeps the game near the replay.
            constexpr size_t PresentsAhead = 2;
            present_fences.push_back(queue->second.fence);
            if (present_fences.size() > PresentsAhead) {
                const u64 fence = present_fences.front();
                present_fences.erase(present_fences.begin());
                if (system.GPU().CurrentSyncRequestFence() < fence) {
                    const auto wait_start = std::chrono::steady_clock::now();
                    lock.unlock();
                    system.GPU().WaitForSyncOperation(fence);
                    if (call_profile) {
                        std::scoped_lock profile_lock{call_profile->mutex};
                        ++call_profile->submit.present_waits;
                        call_profile->submit.present_wait_ns += ElapsedNs(wait_start);
                    }
                }
            }
        }
    } else if (name == "nvnQueueFinish") {
        if (const auto queue = queues.find(x[0]); queue != queues.end() && queue->second.fence) {
            const u64 fence = queue->second.fence;
            const u64 sequence = queue->second.sequence;
            lock.unlock();
            system.GPU().WaitForSyncOperation(fence);
            BananzaNvn2Commands::GetRetirement().WaitWrites(sequence);
        }
    } else if (name == "nvnQueueFlush" ||
               name == "nvnQueueSetDebugLabel") {
        if (!queues.contains(x[0])) {
            NoteMissing(fmt::format("{} invalid queue", name));
        }
    } else {
        return std::nullopt;
    }
    return 0;
}

std::optional<u64> Session::HandleWindow(Memory::Memory& memory, std::string_view name,
                                        const GuestCall& call) {
    const auto& x = call.x;
    if (name == "nvnWindowBuilderSetDefaults") {
        const u64 device = window_builders[x[0]].device;
        window_builders[x[0]] = {.device = device};
    } else if (name == "nvnWindowBuilderSetDevice") {
        window_builders[x[0]].device = x[1];
    } else if (name == "nvnWindowBuilderSetNativeWindow") {
        window_builders[x[0]].native = x[1];
    } else if (name == "nvnWindowBuilderSetTextures") {
        const auto textures = x[1] <= 4 ? ReadHandles(memory, x[2], x[1]) : std::nullopt;
        if (!textures) {
            NoteMissing("nvnWindowBuilderSetTextures invalid array");
        } else {
            window_builders[x[0]].textures = std::move(*textures);
        }
    } else if (name == "nvnWindowBuilderSetNumActiveTextures") {
        window_builders[x[0]].active = static_cast<u32>(x[1]);
    } else if (name == "nvnWindowBuilderSetPresentInterval") {
        window_builders[x[0]].interval = static_cast<u32>(x[1]);
    } else if (name == "nvnWindowInitialize") {
        const auto builder = window_builders.find(x[1]);
        if (!x[0] || builder == window_builders.end()) {
            return 0;
        }
        Window window = builder->second;
        if (!window.active) {
            window.active = static_cast<u32>(window.textures.size());
        }
        if (!devices.contains(window.device) || window.active < 2 ||
            window.active > window.textures.size() ||
            !std::ranges::all_of(window.textures, [&](u64 texture) {
                return resources.HasTexture(window.device, texture);
            })) {
            NoteMissing("nvnWindowInitialize invalid device or textures");
            return 0;
        }
        windows.insert_or_assign(x[0], std::move(window));
        return 1;
    } else if (name == "nvnWindowFinalize") {
        windows.erase(x[0]);
    } else {
        const bool builder_query = name.starts_with("nvnWindowBuilder");
        auto& objects = builder_query ? window_builders : windows;
        const auto it = objects.find(x[0]);
        if (it == objects.end()) {
            return std::nullopt;
        }
        Window& window = it->second;
        if (name == "nvnWindowGetCrop") {
            if (!ValidRange(memory, x[1], window.crop.size() * sizeof(u32))) {
                return std::nullopt;
            }
            for (size_t i = 0; i < window.crop.size(); ++i) {
                memory.Write32(x[1] + i * sizeof(u32), window.crop[i]);
            }
            return 0;
        }
        if (name == "nvnWindowAcquireTexture") {
            const auto sync = syncs.find(x[1]);
            if ((x[1] && (sync == syncs.end() || sync->second.device != window.device)) ||
                !ValidRange(memory, x[2], sizeof(u32))) {
                NoteMissing("nvnWindowAcquireTexture invalid sync or output");
                return AcquireError;
            }
            for (u32 i = 0; i < window.active; ++i) {
                const u32 image = (window.next + i) % window.active;
                if (window.acquired[image]) {
                    continue;
                }
                window.next = (image + 1) % window.active;
                window.acquired[image] = true;
                memory.Write32(x[2], image);
                if (sync != syncs.end()) {
                    sync->second.status = SyncStatus::Signaled;
                    sync->second.fence = 0;
                    sync->second.sequence = 0;
                }
                return 0;
            }
            NoteMissing("nvnWindowAcquireTexture no available image");
            return AcquireError;
        }
        if (name == "nvnWindowGetNumActiveTextures" ||
            name == "nvnWindowBuilderGetNumActiveTextures") {
            return window.active ? window.active : window.textures.size();
        }
        if (name == "nvnWindowGetNumTextures" || name == "nvnWindowBuilderGetNumTextures") {
            return window.textures.size();
        }
        if (name == "nvnWindowBuilderGetTexture") {
            return x[1] < window.textures.size() ? window.textures[x[1]] : 0;
        }
        if (name == "nvnWindowGetNativeWindow" || name == "nvnWindowBuilderGetNativeWindow") {
            return window.native;
        }
        if (name == "nvnWindowGetPresentInterval" ||
            name == "nvnWindowBuilderGetPresentInterval") {
            return window.interval;
        }
        if (name == "nvnWindowSetPresentInterval") {
            window.interval = static_cast<u32>(x[1]);
        } else if (name == "nvnWindowSetNumActiveTextures") {
            if (x[1] < 2 || x[1] > window.textures.size()) {
                NoteMissing(fmt::format("{} texture count {}", name, x[1]));
            } else {
                window.active = static_cast<u32>(x[1]);
            }
        } else if (name == "nvnWindowSetCrop") {
            for (size_t i = 0; i < window.crop.size(); ++i) {
                window.crop[i] = static_cast<u32>(x[i + 1]);
            }
        } else if (name != "nvnWindowSetDebugLabel") {
            return std::nullopt;
        }
    }
    return 0;
}

std::optional<u64> Session::HandleSyncEvent(Memory::Memory& memory, std::string_view name,
                                           const GuestCall& call,
                                           std::unique_lock<std::mutex>& lock) {
    const auto& x = call.x;
    if (name == "nvnSyncInitialize") {
        if (!x[0] || !devices.contains(x[1])) {
            return 0;
        }
        std::scoped_lock frame_lock{frame_mutex};
        syncs.insert_or_assign(x[0], Sync{.device = x[1]});
        return 1;
    }
    if (name == "nvnSyncFinalize") {
        std::scoped_lock frame_lock{frame_mutex};
        syncs.erase(x[0]);
    } else if (name == "nvnSyncWait") {
        Sync waited;
        u64 earlier_mark{};
        {
            std::scoped_lock frame_lock{frame_mutex};
            const auto sync = syncs.find(x[0]);
            if (sync == syncs.end() || sync->second.status == SyncStatus::Failed) {
                return SyncFailed;
            }
            if (sync->second.status != SyncStatus::Signaled) {
                return SyncTimeout;
            }
            waited = sync->second;
            earlier_mark = present_marks[1];
        }
        // Experiment: EDEN_NX2_NVN_NO_SYNC_WAIT=1 never makes the game wait for the GPU side.
        static const bool no_sync_wait = std::getenv("EDEN_NX2_NVN_NO_SYNC_WAIT") != nullptr;
        if (no_sync_wait) {
            return SyncAlreadySignaled;
        }
        const u64 fence = waited.fence;
        const u64 sequence = waited.sequence;
        auto& retirement = BananzaNvn2Commands::GetRetirement();
        // What a sync waits for, EDEN_NX2_NVN_SYNC:
        // unset: what was submitted before the present before the last one is finished. The
        //   game prepares and submits a frame while the one before is replayed, as it does
        //   on the console while the GPU draws; never much more than a frame is outstanding.
        // "replay": also the end of the submission's replay, when the game's memory has been
        //   read for the last time (r155).
        // "full": the GPU has finished the submission and its results are back (before r155).
        // Without "full" the game can find counters and ranges the GPU wrote a frame later.
        static const int sync_mode = [] {
            const char* const value = std::getenv("EDEN_NX2_NVN_SYNC");
            const std::string_view mode{value ? value : ""};
            return mode == "full" ? 0 : mode == "replay" ? 1 : 2;
        }();
        const bool early = sync_mode != 0 && submit_lane;
        const bool at_replay_end = early && sync_mode == 1;
        const u64 finished = early ? std::min(sequence, earlier_mark) : sequence;
        const bool reached = at_replay_end ? retirement.IsReplayed(sequence)
                             : early       ? true
                                           : system.GPU().CurrentSyncRequestFence() >= fence;
        // Without a submission the sync is signalled as it is; one fenced before its
        // submission reached the GPU thread has the sequence and an older fence.
        if ((!fence && !sequence) || (reached && retirement.IsRetired(finished))) {
            return SyncAlreadySignaled;
        }
        if (x[1] == std::numeric_limits<u64>::max()) {
            const u64 submitted = waited.submit_host_ns;
            lock.unlock();
            const auto start = std::chrono::steady_clock::now();
            if (at_replay_end) {
                retirement.WaitReplayed(sequence);
            } else if (!early) {
                system.GPU().WaitForSyncOperation(fence);
            }
            const auto replayed = std::chrono::steady_clock::now();
            retirement.Wait(finished);
            if (call_profile && BananzaNvn2Commands::InTimelineWindow(sequence)) {
                LOG_INFO(Core_ARM, "NVN 2 timeline: seq {} sync wait {:.3f} to {:.3f} ms", sequence,
                         start.time_since_epoch().count() / 1e6,
                         BananzaNvn2Commands::HostNs() / 1e6);
            }
            if (call_profile) {
                const auto end = std::chrono::steady_clock::now();
                const u64 now = static_cast<u64>(start.time_since_epoch().count());
                std::scoped_lock profile_lock{call_profile->mutex};
                auto& stats = call_profile->submit;
                ++stats.sync_waits;
                stats.sync_age_ns += submitted && now > submitted ? now - submitted : 0;
                stats.sync_fence_ns += static_cast<u64>((replayed - start).count());
                stats.sync_retire_ns += static_cast<u64>((end - replayed).count());
            }
            return 1;
        }
        if (call_profile) {
            ++call_profile->submit.sync_timeouts;
        }
        return SyncTimeout;
    } else if (name == "nvnEventBuilderSetDefaults") {
        event_builders[x[0]] = {};
    } else if (name == "nvnEventBuilderSetStorage") {
        event_builders[x[0]] = {.pool = x[1], .offset = x[2]};
    } else if (name == "nvnEventInitialize") {
        const auto builder = event_builders.find(x[1]);
        if (!x[0] || builder == event_builders.end()) {
            return 0;
        }
        Event event = builder->second;
        const auto pool = pools.find(event.pool);
        if (pool == pools.end() || !pool->second.storage || event.offset % alignof(u32) ||
            !Fits(event.offset, sizeof(u32), pool->second.size)) {
            NoteMissing("nvnEventInitialize invalid pool storage");
            return 0;
        }
        event.pool_address = pool->second.gpu_address;
        std::unique_lock resource_lock{resource_mutex};
        events.insert_or_assign(x[0], event);
        return 1;
    } else if (name == "nvnEventFinalize") {
        std::unique_lock resource_lock{resource_mutex};
        events.erase(x[0]);
    } else if (name == "nvnEventGetValue" || name == "nvnEventSignal") {
        const auto event = events.find(x[0]);
        if (event == events.end()) {
            return std::nullopt;
        }
        const auto pool = pools.find(event->second.pool);
        if (pool == pools.end() || pool->second.gpu_address != event->second.pool_address) {
            return std::nullopt;
        }
        const u64 address = pool->second.storage + event->second.offset;
        if (!ValidRange(memory, address, sizeof(u32))) {
            return std::nullopt;
        }
        if (name == "nvnEventGetValue") {
            return memory.Read32(address);
        }
        const u32 mode = static_cast<u32>(x[1]);
        const u32 value = static_cast<u32>(x[2]);
        if (mode == 0 || mode == 1) {
            memory.Write32(address, mode == 1 ? memory.Read32(address) + value : value);
        } else {
            NoteMissing(fmt::format("{} mode {:#x}", name, mode));
        }
    } else if (name != "nvnSyncSetDebugLabel") {
        return std::nullopt;
    }
    return 0;
}

std::optional<u64> Session::HandleCommandPayload(Memory::Memory& memory, size_t index,
                                                 std::string_view name, const GuestCall& call) {
    constexpr u64 TextureViewBytes = 16;
    constexpr u64 CopyRegionBytes = 6 * sizeof(u32);
    const auto& x = call.x;
    std::vector<u64> args;
    std::vector<u8> payload;
    std::vector<ResourceTextureInfo> texture_snapshots;
    std::shared_ptr<const ResourceProgramInfo> program;
    ReplayOp replay{};
    bool valid = true;
    const auto copy = [&](u64 address, u64 size) {
        return AppendGuestBytes(memory, payload, address, size);
    };
    const auto view = [&](u64 address) {
        args.push_back(address != 0);
        if (!address) {
            payload.resize(payload.size() + TextureViewBytes);
            return true;
        }
        return copy(address, TextureViewBytes);
    };
    const auto pin_texture = [&](u64 address) {
        if (!address) {
            texture_snapshots.emplace_back();
            return true;
        }
        if (auto texture = resources.GetTexture(address)) {
            texture_snapshots.push_back(std::move(*texture));
            return true;
        }
        return false;
    };
    if (name == "nvnCommandBufferBindProgram") {
        replay = ReplayOp::BindProgram;
        args = {x[1], static_cast<u32>(x[2])};
        program = resources.GetProgram(x[1]);
        valid = !x[1] || program != nullptr;
    } else if (name == "nvnCommandBufferBindVertexAttribState" ||
               name == "nvnCommandBufferBindVertexStreamState") {
        replay = name == "nvnCommandBufferBindVertexAttribState" ? ReplayOp::BindVertexAttribState
                                                                 : ReplayOp::BindVertexStreamState;
        const u64 count = static_cast<u32>(x[1]);
        const u64 element_size = name == "nvnCommandBufferBindVertexAttribState" ? 4 : 8;
        args = {count};
        valid = count <= 32 && copy(x[2], count * element_size);
    } else if (name == "nvnCommandBufferPushGraphicsConstants" ||
               name == "nvnCommandBufferPushComputeConstants") {
        const u32 offset = static_cast<u32>(x[2]);
        const u32 size = std::min<u32>(static_cast<u32>(x[3]), 256) & ~3U;
        replay = name == "nvnCommandBufferPushComputeConstants" ? ReplayOp::PushComputeConstants
                                                                : ReplayOp::PushGraphicsConstants;
        args = {x[1] & 7, offset};
        valid = offset % 4 == 0 && Fits(offset, size, 256) && copy(x[4], size);
    } else if (name == "nvnCommandBufferBindUniformBuffers" ||
               name == "nvnCommandBufferBindUniformBuffersDeferred" ||
               name == "nvnCommandBufferBindStorageBuffers" ||
               name == "nvnCommandBufferBindStorageBuffersDeferred" ||
               name == "nvnCommandBufferBindTextures") {
        const u64 count = static_cast<u32>(x[3]);
        const u64 element_size = name == "nvnCommandBufferBindTextures" ? 8 : 16;
        if (name.starts_with("nvnCommandBufferBindUniformBuffers")) {
            replay = ReplayOp::BindUniformBuffers;
        } else if (name == "nvnCommandBufferBindTextures") {
            replay = ReplayOp::BindTextures;
        } else {
            replay = ReplayOp::BindStorageBuffers;
        }
        args = {static_cast<u32>(x[1]), static_cast<u32>(x[2]), count};
        valid = count <= 64 && copy(x[4], count * element_size);
    } else if (name == "nvnCommandBufferSetRenderTargets") {
        replay = ReplayOp::SetRenderTargets;
        const u64 count = static_cast<u32>(x[1]);
        const auto textures = count <= 8
                                  ? (x[2] ? ReadHandles(memory, x[2], count)
                                          : std::optional{std::vector<u64>(count)})
                                  : std::nullopt;
        const auto views = count <= 8
                               ? (x[3] ? ReadHandles(memory, x[3], count)
                                       : std::optional{std::vector<u64>(count)})
                               : std::nullopt;
        valid = textures.has_value() && views.has_value();
        if (valid) {
            args = {count, x[4]};
            args.insert(args.end(), textures->begin(), textures->end());
            for (const u64 address : *textures) {
                valid = valid && pin_texture(address);
            }
            valid = valid && pin_texture(x[4]);
            for (const u64 address : *views) {
                valid = valid && view(address);
            }
            valid = valid && view(x[5]);
        }
    } else if (name == "nvnCommandBufferCopyTextureToTexture") {
        replay = ReplayOp::CopyTexture;
        args = {x[1], x[4], static_cast<u32>(x[7])};
        valid = x[1] && x[4] && pin_texture(x[1]) && pin_texture(x[4]) &&
                view(x[2]) && view(x[5]) && copy(x[3], CopyRegionBytes) &&
                copy(x[6], CopyRegionBytes);
    } else if (name == "nvnCommandBufferCopyBufferToTexture") {
        replay = ReplayOp::CopyBufferToTexture;
        args = {x[1], static_cast<u32>(x[5])};
        valid = x[1] && x[2] && pin_texture(x[2]) && view(x[3]) &&
                copy(x[4], CopyRegionBytes);
    } else if (name == "nvnCommandBufferClearColor" || name == "nvnCommandBufferClearColori" ||
               name == "nvnCommandBufferClearColorui") {
        replay = name.ends_with("ui") ? ReplayOp::ClearColorUnsigned
                 : name.ends_with("i") ? ReplayOp::ClearColorSigned
                                        : ReplayOp::ClearColor;
        args = {static_cast<u32>(x[1]), static_cast<u32>(x[3])};
        valid = copy(x[2], 4 * sizeof(u32));
    } else if (name == "nvnCommandBufferSetBlendColor") {
        replay = ReplayOp::SetBlendColor;
        valid = copy(x[1], 4 * sizeof(u32));
    } else if (name == "nvnCommandBufferSetScissors" || name == "nvnCommandBufferSetViewports") {
        replay = name == "nvnCommandBufferSetScissors" ? ReplayOp::SetScissors
                                                         : ReplayOp::SetViewports;
        const u64 first = static_cast<u32>(x[1]);
        const u64 count = static_cast<u32>(x[2]);
        args = {first, count};
        valid = Fits(first, count, 16) && copy(x[3], count * 4 * sizeof(u32));
    } else if (name == "nvnCommandBufferSetDepthRanges") {
        replay = ReplayOp::SetDepthRanges;
        const u64 first = static_cast<u32>(x[1]);
        const u64 count = static_cast<u32>(x[2]);
        args = {first, count};
        valid = Fits(first, count, 16) && copy(x[3], count * 2 * sizeof(float));
    } else if (name == "nvnCommandBufferUpdateUniformBuffer") {
        replay = ReplayOp::UpdateUniformBuffer;
        args = {x[1], x[2], x[3], x[4]};
        valid = x[4] <= (16ULL << 20) && Fits(x[3], x[4], x[2]) && copy(x[5], x[4]);
    } else if (name == "nvnCommandBufferSignalEvent" || name == "nvnCommandBufferWaitEvent") {
        const bool signal = name == "nvnCommandBufferSignalEvent";
        replay = signal ? ReplayOp::SignalEvent : ReplayOp::WaitEvent;
        const auto event = events.find(x[1]);
        valid = event != events.end();
        if (valid) {
            args = {event->second.pool_address + event->second.offset, static_cast<u32>(x[2]),
                    static_cast<u32>(x[3])};
            if (signal) {
                args.push_back(static_cast<u32>(x[4]));
                args.push_back(static_cast<u32>(x[5]));
            }
        }
    } else {
        return std::nullopt;
    }
    if (memory.Read8(x[0] + 0x11)) {
        recorder.Begin(x[0]);
    }
    const u64 bytes = CommandBytes(name, call);
    if (!valid || !recorder.Record(x[0], static_cast<u32>(index), args, payload, bytes,
                                   texture_snapshots, std::move(program), replay)) {
        NoteMissing(fmt::format("{} invalid payload or recording state", name));
    } else if (bytes) {
        UpdateCommandCursor(memory, x[0], CommandShardOf(x[0]).buffers.at(x[0]));
    }
    return 0;
}

std::optional<u64> Session::HandleCommands(Memory::Memory& memory, size_t index,
                                          std::string_view name, const GuestCall& call) {
    const auto& x = call.x;
    bool ok = true;
    if (name == "nvnCommandBufferInitialize") {
        if (!devices.contains(x[1]) || !ValidRange(memory, x[0], CommandBufferSize) ||
            !recorder.Initialize(x[0])) {
            return 0;
        }
        constexpr std::array<u8, CommandBufferSize> empty{};
        memory.WriteBlock(x[0], empty.data(), empty.size());
        memory.Write64(x[0] + 0x70, x[1]);
        auto& object =
            CommandShardOf(x[0]).buffers.insert_or_assign(x[0], CommandBuffer{}).first->second;
        UpdateCommandBuffer(memory, x[0], object, {});
        return 1;
    }
    auto& command_buffers = CommandShardOf(x[0]).buffers;
    const auto it = command_buffers.find(x[0]);
    if (it == command_buffers.end() || !ValidRange(memory, x[0], CommandBufferSize)) {
        return std::nullopt;
    }
    auto& object = it->second;
    if (name == "nvnCommandBufferFinalize") {
        ok = recorder.Finalize(x[0]);
        command_buffers.erase(it);
        return 0;
    } else if (name == "nvnCommandBufferSetMemoryCallback") {
        if (x[1] && !ValidRange(memory, x[1], sizeof(u32))) {
            NoteMissing("nvnCommandBufferSetMemoryCallback invalid address");
        } else {
            object.callback = x[1];
        }
    } else if (name == "nvnCommandBufferSetMemoryCallbackData") {
        object.callback_data = x[1];
    } else if (name == "nvnCommandBufferGetMemoryCallback") {
        return object.callback;
    } else if (name == "nvnCommandBufferGetMemoryCallbackData") {
        return object.callback_data;
    } else if (name == "nvnCommandBufferSetCommandMemoryCallbackEnabled") {
        object.command_callback = static_cast<u8>(x[1]) != 0;
    } else if (name == "nvnCommandBufferSetControlMemoryCallbackEnabled") {
        object.control_callback = static_cast<u8>(x[1]) != 0;
    } else if (name == "nvnCommandBufferSetRecordingMode") {
        object.recording_mode = static_cast<u32>(x[1]);
    } else if (name == "nvnCommandBufferAddCommandMemory") {
        const auto pool = pools.find(x[1]);
        if (pool == pools.end() || x[2] > pool->second.size) {
            NoteMissing("nvnCommandBufferAddCommandMemory invalid pool or offset");
            return 0;
        }
        // A callback can request its minimum block size near the end of a pool. The host
        // recorder only needs the actual in-pool bytes for its guest-visible accounting.
        const u64 capacity = std::min(x[3], pool->second.size - x[2]);
        ok = recorder.SetCommandMemory(x[0], capacity);
        if (ok) {
            object.command_storage = pool->second.storage ? pool->second.storage + x[2] : 0;
            object.gpu_address = pool->second.gpu_address + x[2];
        }
    } else if (name == "nvnCommandBufferAddControlMemory") {
        ok = (!x[2] || ValidRange(memory, x[1], x[2])) &&
             recorder.SetControlMemory(x[0], x[1], x[2]);
    } else if (name == "nvnCommandBufferBeginRecording") {
        ok = recorder.Begin(x[0]);
    } else if (name == "nvnCommandBufferEndRecording") {
        if (const auto state = recorder.GetState(x[0]);
            state && !state->recording && memory.Read8(x[0] + 0x11)) {
            recorder.Begin(x[0]);
        }
        if (const auto handle = recorder.End(x[0], ControlRecordBytes, ControlHandleOffset)) {
            if (recording_stats) {
                ++last_recording_serial;
                ++recording_stats->recorded;
                recording_stats->rerecorded += !recording_stats->recorded_buffers.insert(x[0]).second;
            }
            // A guest-visible handle points into its control allocation.
            constexpr std::array<u8, ControlRecordBytes> header{1};
            memory.WriteBlock(*handle - ControlHandleOffset, header.data(), header.size());
            memory.Write8(*handle, 2);
            UpdateCommandBuffer(memory, x[0], object, *recorder.GetState(x[0]));
            return *handle;
        }
        ok = false;
    } else if (const auto payload = HandleCommandPayload(memory, index, name, call)) {
        return *payload;
    } else if (const auto state = recorder.GetState(x[0])) {
        if (name == "nvnCommandBufferIsRecording") {
            return state->recording;
        }
        if (name == "nvnCommandBufferGetCommandMemorySize") {
            return state->command_size;
        }
        if (name == "nvnCommandBufferGetCommandMemoryUsed") {
            return state->command_used;
        }
        if (name == "nvnCommandBufferGetCommandMemoryFree") {
            return state->command_size - state->command_used;
        }
        if (name == "nvnCommandBufferGetControlMemorySize") {
            return state->control_size;
        }
        if (name == "nvnCommandBufferGetControlMemoryUsed") {
            return state->control_used;
        }
        if (name == "nvnCommandBufferGetControlMemoryFree") {
            return state->control_size - state->control_used;
        }
        return std::nullopt;
    } else {
        return std::nullopt;
    }
    if (!ok) {
        NoteMissing(fmt::format("{} recording or memory state", name));
    } else {
        UpdateCommandBuffer(memory, x[0], object, *recorder.GetState(x[0]));
    }
    return 0;
}

std::optional<bool> Session::RecordCommand(Memory::Memory& memory, size_t index,
                                           const GuestCall& call) {
    const auto& x = call.x;
    auto& command_buffers = CommandShardOf(x[0]).buffers;
    const auto buffer = command_buffers.find(x[0]);
    if (buffer == command_buffers.end() || !ValidRange(memory, x[0], CommandBufferSize)) {
        return std::nullopt;
    }
    if (memory.Read8(x[0] + 0x11)) {
        recorder.Begin(x[0]);
    }
    const auto& procedure = procedures[index];
    const auto* scalar = static_cast<const ScalarCommand*>(procedure.scalar_command);
    if (const auto* state = static_cast<const StateBinding*>(procedure.state_binding)) {
        std::array<u8, 24> data;
        const auto payload = std::span{data}.first(state->bytes);
        return ValidRange(memory, x[1], payload.size()) &&
               memory.ReadBlock(x[1], payload.data(), payload.size()) &&
               recorder.Record(x[0], static_cast<u32>(index), {}, payload, 0, {}, {}, state->replay);
    }
    if (procedure.binds_deferred_textures) {
        const u32 count = static_cast<u32>(x[3]);
        const std::array<u64, 4> args{static_cast<u32>(x[1]), static_cast<u32>(x[2]), count, x[4]};
        return count <= 64 && (!count || ValidRange(memory, x[4], count * sizeof(u64))) &&
               recorder.Record(x[0], static_cast<u32>(index), args, {}, 0, {}, {},
                               ReplayOp::BindTexturesDeferred);
    }
    if (!scalar) {
        const auto handles = ReadHandles(memory, x[2], static_cast<u32>(x[1]));
        if (!handles || !recorder.AppendHandles(x[0], *handles, 0, procedure.calls_handles)) {
            return false;
        }
        UpdateCommandBuffer(memory, x[0], buffer->second, *recorder.GetState(x[0]));
        return true;
    }
    const auto& command = *scalar;
    const auto args = std::span{x}.subspan(1, command.args);
    std::array<u32_le, 8> float_words{};
    auto floats = std::span{float_words}.first(command.float_args);
    std::ranges::transform(std::span{call.v}.first(command.float_args), floats.begin(),
                           [](u64 reg) { return static_cast<u32>(reg); });
    const auto bytes = std::span<const u8>{reinterpret_cast<const u8*>(floats.data()),
                                           floats.size_bytes()};
    if (!recorder.Record(x[0], static_cast<u32>(index), args, bytes, command.command_bytes,
                         {}, {}, command.replay)) {
        return false;
    }
    if (command.command_bytes) {
        UpdateCommandCursor(memory, x[0], buffer->second);
    }
    return true;
}

void Session::NoteMissing(std::string_view name) {
    std::scoped_lock lock{missing_mutex};
    if (const auto it = missing.find(name); it != missing.end()) {
        ++it->second.calls;
        return;
    }
    missing.emplace(name, MissingCall{.calls = 1});
    LOG_WARNING(Core_ARM, "Bananza NVN 2 HLE unsupported {}", name);
}

void Session::ReportMissing(bool final) {
    LOG_INFO(Core_ARM, "Bananza NVN 2 HLE recorder totals: {} submissions, {} commands, {} invalid submissions",
             accepted_submissions, accepted_commands, rejected_submissions);
    for (const auto& [address, stat] : texture_copy_stats) {
        LOG_INFO(Core_ARM, "NVN 2 texture copy at {:#x}: {} snapshots, {:.1f} MiB",
                 address, stat.count, stat.bytes / 1048576.0);
    }
    texture_copy_stats.clear();
    std::vector<std::pair<std::string_view, u64>> totals;
    std::scoped_lock lock{missing_mutex};
    for (auto& [name, count] : missing) {
        const u64 calls = final ? count.calls : count.calls - count.reported;
        if (calls) {
            totals.emplace_back(name, calls);
        }
        count.reported = count.calls;
    }
    std::ranges::sort(totals, std::greater{}, [](const auto& entry) { return entry.second; });
    for (const auto& [name, calls] : totals) {
        LOG_INFO(Core_ARM, "Bananza NVN 2 HLE unsupported {} {} calls in {}",
                 name, calls, final ? "session" : "window");
    }
}

void Session::Finish() {
    std::scoped_lock lock{mutex};
    ReportMissing(true);
    ReportRecordingStats();
    ReportCallProfile();
}

void Session::CountRecordings(const BananzaNvn2Commands::Submission& batch) {
    auto& stats = *recording_stats;
    ++stats.submissions;
    stats.commands += batch.command_count;
    const auto walk = [&](auto&& self, const BananzaNvn2Commands::Recording& recording,
                          bool nested) -> void {
        ++stats.recording_uses;
        stats.repeated_uses += !stats.used.insert(recording.serial).second;
        stats.prior_window_uses += recording.serial <= last_reported_recording_serial;
        stats.command_buffers.insert(recording.command_buffer);
        (nested ? stats.nested : stats.submitted).insert(recording.serial);
        for (const auto& op : recording.operations) {
            if (op.nested) {
                self(self, *op.nested, true);
            }
        }
    };
    for (const auto& recording : batch.recordings) {
        stats.draws += recording->draw_count;
        stats.called_commands += recording->called_commands;
        stats.called_draws += recording->called_draws;
        stats.copied_commands += recording->copied_commands;
        stats.copied_draws += recording->copied_draws;
        walk(walk, *recording, false);
    }
}

void Session::ReportRecordingStats() {
    if (!recording_stats) {
        return;
    }
    const auto& stats = *recording_stats;
    LOG_INFO(Core_ARM,
             "Bananza NVN 2 recordings: presents={} submits={} commands={} draws={} through_call={}/{} through_copy={}/{} distinct_submitted={} distinct_nested={} used_buffers={} new_recordings={} rerecorded={} uses={} repeats={} from_prior_windows={}",
             stats.presents, stats.submissions, stats.commands, stats.draws,
             stats.called_commands, stats.called_draws, stats.copied_commands, stats.copied_draws,
             stats.submitted.size(), stats.nested.size(), stats.command_buffers.size(), stats.recorded,
             stats.rerecorded, stats.recording_uses, stats.repeated_uses, stats.prior_window_uses);
    last_reported_recording_serial = last_recording_serial;
    recording_stats.emplace();
}

void Session::ReportCallProfile() {
    if (!call_profile) {
        return;
    }
    auto& profile = *call_profile;
    std::array<CallStats, 726> entries;
    std::map<u64, ThreadStats> threads;
    u64 presents;
    {
        std::scoped_lock lock{profile.mutex};
        entries = std::exchange(profile.entries, {});
        threads = std::exchange(profile.threads, {});
        presents = std::exchange(profile.presents, 0);
    }
    const SubmitStats submit = std::exchange(profile.submit, {});
    u64 calls{};
    u64 elapsed{};
    u64 waited{};
    for (const auto& entry : entries) {
        calls += entry.calls;
        elapsed += entry.nanoseconds;
        waited += entry.lock_wait;
    }
    if (!calls) {
        return;
    }
    const double frames = presents ? double(presents) : 1.0;
    const auto ms = [&](u64 nanoseconds) { return double(nanoseconds) / frames / 1e6; };
    LOG_INFO(Core_ARM,
             "Bananza NVN 2 call profile: calls={} presents={} host_ns={} ns/call={:.1f} ms/present={:.3f} lock_wait_ms/present={:.3f}",
             calls, presents, elapsed, double(elapsed) / calls, ms(elapsed), ms(waited));
    LOG_INFO(Core_ARM,
             "Bananza NVN 2 submit profile: submits/present={:.1f} scan_ms={:.3f} texture_copy_ms={:.3f} wait_ms={:.3f} waits/present={:.1f} buffer_MiB={:.2f} ranges={:.0f} texture_MiB={:.2f} textures={:.0f} copy_wait_ms={:.3f} (per present)",
             submit.submits / frames, ms(submit.scan_ns), ms(submit.texture_ns), ms(submit.wait_ns),
             submit.waits / frames, submit.buffer_bytes / frames / 1048576.0,
             submit.buffer_ranges / frames, submit.texture_bytes / frames / 1048576.0,
             submit.textures / frames, ms(submit.copy_wait_ns));
    LOG_INFO(Core_ARM,
             "Bananza NVN 2 submit profile: overlap waits {:.1f} for {:.3f} ms, present waits {:.1f} for {:.3f} ms (per present)",
             submit.overlap_waits / frames, ms(submit.overlap_wait_ns),
             submit.present_waits / frames, ms(submit.present_wait_ns));
    LOG_INFO(Core_ARM,
             "Bananza NVN 2 submit profile: {:.0f} vertex and index ranges with {:.2f} MiB passed by "
             "reference, not counted in buffer_MiB (per present)",
             submit.referenced_ranges / frames, submit.referenced_bytes / frames / 1048576.0);
    LOG_INFO(Core_ARM,
             "Bananza NVN 2 submit profile: of those, {:.0f} ranges with {:.2f} MiB can stay in VRAM "
             "until flushed (per present)",
             submit.fixed_ranges / frames, submit.fixed_bytes / frames / 1048576.0);
    LOG_INFO(Core_ARM,
             "Bananza NVN 2 submit profile: textures hashed {:.0f} with {:.2f} MiB, {:.0f} not "
             "hashed because no flush touched them (per present)",
             submit.texture_hashes / frames, submit.texture_hash_bytes / frames / 1048576.0,
             submit.texture_flush_skips / frames);
    LOG_INFO(Core_ARM,
             "Bananza NVN 2 submit profile: sync waits {:.1f}, awaited submission submitted {:.3f} ms "
             "before the wait, then {:.3f} ms until replayed and {:.3f} ms until retired; {:.1f} "
             "timeouts (per present, times per wait)",
             submit.sync_waits / frames,
             submit.sync_waits ? submit.sync_age_ns / 1e6 / submit.sync_waits : 0.0,
             submit.sync_waits ? submit.sync_fence_ns / 1e6 / submit.sync_waits : 0.0,
             submit.sync_waits ? submit.sync_retire_ns / 1e6 / submit.sync_waits : 0.0,
             submit.sync_timeouts / frames);
    for (const auto& [thread, stats] : threads) {
        LOG_INFO(Core_ARM, "Bananza NVN 2 call profile: thread {} calls/present={:.0f} ms/present={:.3f}",
                 thread, stats.calls / frames, ms(stats.nanoseconds));
    }
    constexpr size_t Listed = 20;
    std::ranges::partial_sort(entries, entries.begin() + Listed, std::greater{},
                              &CallStats::nanoseconds);
    for (const auto& entry : std::span{entries}.first<Listed>()) {
        if (entry.calls) {
            LOG_INFO(Core_ARM,
                     "Bananza NVN 2 call profile: {} calls/present={:.0f} ns/call={:.1f} ms/present={:.3f} lock_wait_ms/present={:.3f}",
                     entry.name, entry.calls / frames, double(entry.nanoseconds) / entry.calls,
                     ms(entry.nanoseconds), ms(entry.lock_wait));
        }
    }
    // The calls that waited most for a lock or for the submit lane.
    constexpr size_t Waiting = 10;
    std::ranges::partial_sort(entries, entries.begin() + Waiting, std::greater{},
                              &CallStats::lock_wait);
    for (const auto& entry : std::span{entries}.first<Waiting>()) {
        if (entry.calls && entry.lock_wait) {
            LOG_INFO(Core_ARM,
                     "Bananza NVN 2 call waits: {} calls/present={:.1f} waited {:.3f} ms a present, "
                     "{:.1f} us a call",
                     entry.name, entry.calls / frames, ms(entry.lock_wait),
                     double(entry.lock_wait) / entry.calls / 1e3);
        }
    }
}

} // namespace Core::BananzaNvn2Hle
