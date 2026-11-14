// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <limits>
#include <span>

#include "common/alignment.h"
#include "common/logging.h"
#include "common/swap.h"
#include "core/arm/bananza_nvn2_device.h"
#include "core/arm/bananza_nvn2_hle.h"
#include "core/arm/bananza_nvn2_trace.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/gpu.h"
#include "video_core/renderer_base.h"

namespace Core::BananzaNvn2Hle {
namespace {

constexpr u32 CpuNoAccess = 1;
constexpr u32 PhysicalPool = 0x100;
constexpr u32 VirtualPool = 0x200;
constexpr u64 PoolAlignment = 0x10000;
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
    std::vector<u64_le> wire(count);
    if (count && !memory.ReadBlock(address, wire.data(), wire.size() * sizeof(u64_le))) {
        return std::nullopt;
    }
    return std::vector<u64>{wire.begin(), wire.end()};
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
};

constexpr std::array ScalarCommands{
    ScalarCommand{"nvnCommandBufferBarrier", 1, 0},
    ScalarCommand{"nvnCommandBufferBindStorageBuffer", 4, 0},
    ScalarCommand{"nvnCommandBufferBindTexture", 3, 0},
    ScalarCommand{"nvnCommandBufferBindUniformBuffer", 4, 0},
    ScalarCommand{"nvnCommandBufferBindVertexBuffer", 3, 0},
    ScalarCommand{"nvnCommandBufferClearDepthStencil", 3, 1},
    ScalarCommand{"nvnCommandBufferCopyBufferToBuffer", 4, 0},
    ScalarCommand{"nvnCommandBufferDispatchCompute", 3, 0},
    ScalarCommand{"nvnCommandBufferDispatchComputeIndirect", 1, 0},
    ScalarCommand{"nvnCommandBufferDrawArrays", 3, 0},
    ScalarCommand{"nvnCommandBufferDrawArraysIndirect", 2, 0},
    ScalarCommand{"nvnCommandBufferDrawArraysInstanced", 5, 0},
    ScalarCommand{"nvnCommandBufferDrawElements", 4, 0},
    ScalarCommand{"nvnCommandBufferDrawElementsBaseVertex", 5, 0},
    ScalarCommand{"nvnCommandBufferDrawElementsIndirect", 4, 0},
    ScalarCommand{"nvnCommandBufferDrawElementsInstanced", 7, 0},
    ScalarCommand{"nvnCommandBufferFenceSync", 3, 0},
    ScalarCommand{"nvnCommandBufferReportCounter", 2, 0},
    ScalarCommand{"nvnCommandBufferResetCounter", 1, 0},
    ScalarCommand{"nvnCommandBufferSetComputeOverlapMode", 1, 0},
    ScalarCommand{"nvnCommandBufferSetAlphaRef", 0, 1},
    ScalarCommand{"nvnCommandBufferSetConservativeRasterEnable", 1, 0},
    ScalarCommand{"nvnCommandBufferSetDepthBounds", 0, 2},
    ScalarCommand{"nvnCommandBufferSetDepthClamp", 1, 0},
    ScalarCommand{"nvnCommandBufferSetDepthRange", 0, 2},
    ScalarCommand{"nvnCommandBufferSetGeometryProcessingMode", 1, 0},
    ScalarCommand{"nvnCommandBufferSetLineWidth", 0, 1},
    ScalarCommand{"nvnCommandBufferSetPointSize", 0, 1},
    ScalarCommand{"nvnCommandBufferSetPolygonOffsetClamp", 0, 3},
    ScalarCommand{"nvnCommandBufferSetRasterizerDiscard", 1, 0},
    ScalarCommand{"nvnCommandBufferSetRenderEnable", 1, 0},
    ScalarCommand{"nvnCommandBufferSetRenderEnableConditional", 2, 0},
    ScalarCommand{"nvnCommandBufferSetScissor", 4, 0},
    ScalarCommand{"nvnCommandBufferSetShaderScratchMemory", 3, 0},
    ScalarCommand{"nvnCommandBufferSetStencilMask", 2, 0},
    ScalarCommand{"nvnCommandBufferSetStencilCullCriteria", 3, 0},
    ScalarCommand{"nvnCommandBufferSetStencilRef", 2, 0},
    ScalarCommand{"nvnCommandBufferSetStencilValueMask", 2, 0},
    ScalarCommand{"nvnCommandBufferSetViewport", 4, 0},
    ScalarCommand{"nvnCommandBufferSetSamplerPool", 1, 0},
    ScalarCommand{"nvnCommandBufferSetTexturePool", 1, 0},
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

u64 CommandBytes(std::string_view name, const GuestCall& call) {
    if (name == "nvnCommandBufferDrawElements") {
        return 48;
    }
    if (name == "nvnCommandBufferDrawElementsBaseVertex" ||
        name == "nvnCommandBufferDrawElementsInstanced") {
        return 44;
    }
    if (name == "nvnCommandBufferBindVertexBuffer") {
        return 24;
    }
    if (name == "nvnCommandBufferSetRenderEnableConditional") {
        return 16;
    }
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

Session::Session(System& system_, u64 bootstrap_, u64 internal_bootstrap_, u64 callback_return_)
    : bootstrap{bootstrap_}, internal_bootstrap{internal_bootstrap_},
      callback_return{callback_return_}, system{system_} {}

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

bool Session::IsFloatReturn(std::string_view name) {
    return Resources::IsFloatReturn(name);
}

std::optional<GuestCallback> Session::BeginCallback(u64 thread, size_t procedure,
                                                   std::string_view name, const GuestCall& call,
                                                   u64 link) {
    const u64 command_bytes = CommandBytes(name, call);
    const bool beginning = name == "nvnCommandBufferBeginRecording";
    const bool ending = name == "nvnCommandBufferEndRecording";
    if (!beginning && !ending && !command_bytes) {
        return std::nullopt;
    }
    std::scoped_lock lock{mutex};
    const u64 address = call.x[0];
    const auto buffer = command_buffers.find(address);
    const auto state = recorder.GetState(address);
    if (buffer == command_buffers.end() || !state || !buffer->second.callback) {
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
    auto& continuations = callback_continuations[thread];
    if (continuations.size() >= 8) {
        NoteMissing("nvnCommandBuffer control callback nesting");
        return std::nullopt;
    }
    if ((buffer->second.callback & 3) != 0) {
        NoteMissing("nvnCommandBuffer unaligned memory callback");
        return std::nullopt;
    }
    continuations.push_back({procedure, call, link});
    return GuestCallback{buffer->second.callback,
                         {address, command_bytes ? 0U : 1U, command_bytes ? 512U : 256U,
                          buffer->second.callback_data}};
}

std::optional<CallbackContinuation> Session::EndCallback(u64 thread) {
    std::scoped_lock lock{mutex};
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

u64 Session::Dispatch(const BananzaNvn2Trace::Session& procs, Memory::Memory& memory, size_t index,
                      const GuestCall& call) {
    const std::string_view name = procs.Name(index);
    const auto& x = call.x;
    if (name == "nvnDeviceGetProcAddress") {
        return x[1] ? procs.Resolve(memory.ReadCString(x[1], 96)) : 0;
    }
    std::scoped_lock lock{mutex};
    if (const auto now = std::chrono::steady_clock::now(); now >= next_report) {
        ReportMissing(false);
        next_report = now + std::chrono::seconds{10};
    }
    std::optional<u64> result;
    if (name.starts_with("nvnDevice")) {
        result = HandleDevice(memory, name, call);
    } else if (name.starts_with("nvnMemoryPool")) {
        result = HandlePool(memory, name, call);
    } else if (name.starts_with("nvnBuffer")) {
        result = HandleBuffer(memory, name, call);
    } else if (name.starts_with("nvnQueue")) {
        result = HandleQueue(memory, name, call);
    } else if (name.starts_with("nvnWindow")) {
        result = HandleWindow(memory, name, call);
    } else if (name.starts_with("nvnSync") || name.starts_with("nvnEvent")) {
        result = HandleSyncEvent(memory, name, call);
    } else if (name.starts_with("nvnCommandBuffer")) {
        result = HandleCommands(memory, index, name, call);
    } else {
        result = resources.Dispatch(
            memory, name, call, [&](u64 device) { return devices.contains(device); },
            [&](u64 address) { return GetPool(address); },
            [&](u64 address, u64 size) { return ResolveGpuRange(address, size); });
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
    } else if (name == "nvnDeviceGetTextureHandle") {
        return (x[1] & 0xfffff) | ((x[2] & 0xfff) << 20);
    } else if (name == "nvnDeviceGetSeparateSamplerHandle") {
        return static_cast<u64>(static_cast<s64>(static_cast<s32>(x[1]))) << 20;
    } else if (name == "nvnDeviceGetImageHandle" || name == "nvnDeviceGetSeparateTextureHandle") {
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
        pools.emplace(x[0], pool);
        gpu_pools.emplace(pool.gpu_address, x[0]);
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
            pools.erase(it);
        }
    } else if (name == "nvnMemoryPoolMapVirtual") {
        if (!MapVirtual(memory, call)) {
            NoteMissing("nvnMemoryPoolMapVirtual invalid mapping");
            return 0;
        }
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
        if (name == "nvnMemoryPoolGetFlags") {
            return pool.flags;
        }
        if (name == "nvnMemoryPoolFlushMappedRange" ||
            name == "nvnMemoryPoolInvalidateMappedRange") {
            if (!pool.storage || !Fits(x[1], x[2], pool.size)) {
                NoteMissing(fmt::format("{} invalid range", name));
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

std::optional<u64> Session::ResolveGpuRange(u64 address, u64 size) const {
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
    return source->second.storage + physical_offset;
}

void Session::SnapshotResources(Memory::Memory& memory, BananzaNvn2Commands::Submission& batch,
                                Queue& queue) {
    constexpr u64 MaxBufferBytes = 48ULL << 20;
    constexpr u64 MaxTextureBytes = 96ULL << 20;
    constexpr u64 MaxRange = 8ULL << 20;
    u64 buffer_bytes{};
    u64 texture_bytes{};
    std::vector<const ResourceTextureInfo*> sampled_textures;
    const auto read_gpu = [&](u64 address, std::span<u8> dest) {
        const auto read = [&](u64 gpu, std::span<u8> bytes) {
            const auto guest = ResolveGpuRange(gpu, bytes.size());
            return guest && ValidRange(memory, *guest, bytes.size()) &&
                   memory.ReadBlockUnsafe(*guest, bytes.data(), bytes.size());
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
    const auto snapshot = [&](u64 address, u64 size) {
        if (!size || size > MaxRange || !Fits(buffer_bytes, size, MaxBufferBytes)) {
            return;
        }
        const auto key = std::pair{address, size};
        if (batch.buffer_snapshots.contains(key)) {
            return;
        }
        std::vector<u8> data(size);
        if (read_gpu(address, data)) {
            buffer_bytes += size;
            batch.buffer_snapshots.emplace(key, std::move(data));
        }
    };
    const auto descriptor = [&](u64 handle) {
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
    bool buffers_pending = true;
    bool descriptors_pending = true;
    const auto scan = [&](auto&& self, const BananzaNvn2Commands::Recording& recording) -> void {
        for (const auto& op : recording.operations) {
            if (op.nested) {
                self(self, *op.nested);
            } else if (op.replay == ReplayOp::SetTexturePool && op.args.size() == 1) {
                queue.texture_pool = op.args[0];
                descriptors_pending = true;
            } else if (op.replay == ReplayOp::SetSamplerPool && op.args.size() == 1) {
                queue.sampler_pool = op.args[0];
                descriptors_pending = true;
            } else if (op.replay == ReplayOp::BindTexture && op.args.size() == 3) {
                if (op.args[0] < queue.texture_handles.size() && op.args[1] < 64) {
                    queue.texture_handles[op.args[0]][op.args[1]] = op.args[2];
                }
                descriptors_pending = true;
            } else if (op.replay == ReplayOp::BindTextures && op.args.size() == 3) {
                for (size_t offset = 0; offset + 8 <= op.payload.size(); offset += 8) {
                    u64_le handle;
                    std::memcpy(&handle, op.payload.data() + offset, 8);
                    const u64 slot = op.args[1] + offset / 8;
                    if (op.args[0] < queue.texture_handles.size() && slot < 64) {
                        queue.texture_handles[op.args[0]][slot] = handle;
                    }
                }
                descriptors_pending = true;
            } else if (op.replay == ReplayOp::BindVertexBuffer && op.args.size() == 3) {
                if (op.args[0] < queue.vertex_ranges.size()) {
                    queue.vertex_ranges[op.args[0]] = {op.args[1], op.args[2]};
                    buffers_pending = true;
                }
            } else if (op.replay == ReplayOp::BindUniformBuffer && op.args.size() == 4) {
                if (op.args[0] < queue.uniform_ranges.size() && op.args[1] < 32) {
                    queue.uniform_ranges[op.args[0]][op.args[1]] = {op.args[2], op.args[3]};
                    buffers_pending = true;
                }
                buffers_pending = true;
            } else if (op.replay == ReplayOp::BindUniformBuffers && op.args.size() == 3) {
                for (size_t offset = 0; offset + 16 <= op.payload.size(); offset += 16) {
                    std::array<u64_le, 2> range{};
                    std::memcpy(range.data(), op.payload.data() + offset, 16);
                    const u64 slot = op.args[1] + offset / 16;
                    if (op.args[0] < queue.uniform_ranges.size() && slot < 32) {
                        queue.uniform_ranges[op.args[0]][slot] = {range[0], range[1]};
                    }
                }
            } else if (op.replay == ReplayOp::BindTexturesDeferred && op.args.size() == 4) {
                const u64 size = op.args[2] * sizeof(u64);
                const auto key = std::pair{op.args[3], size};
                if (size && Fits(buffer_bytes, size, MaxBufferBytes) && !batch.deferred_data.contains(key) &&
                    ValidRange(memory, key.first, key.second)) {
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
                        }
                    }
                    descriptors_pending = true;
                }
            } else if ((op.replay == ReplayOp::DrawIndexed ||
                        op.replay == ReplayOp::DrawIndexedBaseVertex ||
                        op.replay == ReplayOp::DrawIndexedInstanced) && op.args.size() >= 4 &&
                       op.args[1] <= 2 && op.args[2] <= MaxRange / 4) {
                snapshot(op.args[3], op.args[2] << op.args[1]);
            }
            const bool draw = op.replay >= ReplayOp::Draw && op.replay <= ReplayOp::DrawInstanced;
            if (draw && buffers_pending) {
                buffers_pending = false;
                for (const auto& [address, size] : queue.vertex_ranges) {
                    snapshot(address, size);
                }
                for (const auto& stage : queue.uniform_ranges) {
                    for (const auto& [address, size] : stage) {
                        snapshot(address, size);
                    }
                }
            }
            if (draw && descriptors_pending) {
                descriptors_pending = false;
                for (const auto& stage : queue.texture_handles) {
                    for (const u64 handle : stage) {
                        if (handle) {
                            descriptor(handle);
                        }
                    }
                }
            }
        }
    };
    for (const auto& recording : batch.recordings) {
        scan(scan, *recording);
    }
    const auto compressed = [](const ResourceTextureInfo* texture) {
        const auto block = GetTextureFormatBlock(texture->description.format);
        return block && block->width > 1;
    };
    // The replay path creates render targets on demand. Give CPU-backed compressed assets
    // first use of the snapshot budget, since they cannot be recovered from replayed draws.
    std::stable_partition(sampled_textures.begin(), sampled_textures.end(), compressed);
    for (const auto* texture : sampled_textures) {
        const auto size = texture->layout.size;
        if (!size || size > (32ULL << 20) || !Fits(texture_bytes, size, MaxTextureBytes) ||
            batch.texture_snapshots.contains(texture->serial)) {
            continue;
        }
        std::vector<u8> data(size);
        if (read_gpu(texture->storage.gpu_address + texture->offset, data)) {
            texture_bytes += size;
            batch.texture_snapshots.emplace(texture->serial, std::move(data));
        }
    }
}

std::optional<u64> Session::HandleBuffer(Memory::Memory& memory, std::string_view name,
                                        const GuestCall& call) {
    const auto& x = call.x;
    if (name == "nvnBufferInitialize") {
        if (!ValidRange(memory, x[0], 0x28) || !ValidRange(memory, x[1], 0x28)) {
            return 0;
        }
        Buffer buffer{
            .device = memory.Read64(x[1] + 0x08),
            .pool = memory.Read64(x[1] + 0x10),
            .offset = memory.Read64(x[1] + 0x18),
            .size = memory.Read64(x[1] + 0x20),
        };
        const auto pool = pools.find(buffer.pool);
        if (pool == pools.end() || pool->second.device != buffer.device ||
            pool->second.flags & PhysicalPool || !Fits(buffer.offset, buffer.size, pool->second.size)) {
            return 0;
        }
        buffer.pool_address = pool->second.gpu_address;
        buffers.insert_or_assign(x[0], buffer);
        memory.Write64(x[0], 0);
        memory.Write64(x[0] + 0x08, buffer.device);
        memory.Write64(x[0] + 0x10, buffer.pool);
        memory.Write64(x[0] + 0x18, buffer.offset);
        memory.Write64(x[0] + 0x20, buffer.size);
        return 1;
    } else if (name == "nvnBufferFinalize") {
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
        if (name == "nvnBufferMap") {
            return pool->second.flags & CpuNoAccess ? 0 : pool->second.storage + buffer.offset;
        }
        if (name == "nvnBufferGetAddress") {
            return buffer.pool_address + buffer.offset;
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

std::optional<u64> Session::HandleQueue(Memory::Memory& memory, std::string_view name,
                                       const GuestCall& call) {
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
        const auto queue = queues.find(x[0]);
        const auto handles = ReadHandles(memory, x[2], static_cast<u32>(x[1]));
        auto submission = queue != queues.end() && handles
                              ? recorder.Submit(x[0], *handles)
                              : std::nullopt;
        if (submission) {
            if (const auto device = devices.find(queue->second.device); device != devices.end()) {
                submission->depth_minus_one_to_one = device->second.depth_mode == 0;
                submission->origin_lower_left = device->second.origin == 0;
            }
            SnapshotResources(memory, *submission, queue->second);
            ++accepted_submissions;
            accepted_commands += submission->command_count;
            if (!logged_submit) {
                logged_submit = true;
                LOG_INFO(Core_ARM,
                         "Bananza NVN 2 HLE accepted submission {} with {} handles and {} commands",
                         submission->sequence, submission->handles.size(), submission->command_count);
            }
            queue->second.fence = system.GPU().SubmitRendererWork(
                [batch = std::move(*submission)](VideoCore::RendererBase& renderer) {
                    renderer.ReplayBananzaNvn2(batch);
                });
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
        const auto queue = queues.find(x[0]);
        const auto sync = syncs.find(x[1]);
        if (queue == queues.end() || sync == syncs.end() ||
            queue->second.device != sync->second.device) {
            NoteMissing("nvnQueueFenceSync invalid queue or sync");
        } else {
            sync->second.status =
                queue->second.rejected_submit ? SyncStatus::Failed : SyncStatus::Signaled;
            sync->second.fence = queue->second.fence;
        }
    } else if (name == "nvnQueueWaitSync") {
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
            Tegra::FramebufferConfig config{
                .width = description.width,
                .height = description.height,
                .stride = description.width,
                .pixel_format = Service::android::PixelFormat::Rgba8888,
            };
            const auto& crop = window->second.crop;
            if (crop[2] && crop[3] && Fits(crop[0], crop[2], config.width) &&
                Fits(crop[1], crop[3], config.height)) {
                config.crop_rect = {static_cast<int>(crop[0]), static_cast<int>(crop[1]),
                                    static_cast<int>(crop[0] + crop[2]),
                                    static_cast<int>(crop[1] + crop[3])};
            }
            queue->second.fence = system.GPU().SubmitRendererWork(
                [texture = *texture, config](VideoCore::RendererBase& renderer) {
                    renderer.PresentBananzaNvn2(texture, config);
                });
        }
    } else if (name == "nvnQueueFinish") {
        if (const auto queue = queues.find(x[0]); queue != queues.end() && queue->second.fence) {
            system.GPU().WaitForSyncOperation(queue->second.fence);
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
                                           const GuestCall& call) {
    const auto& x = call.x;
    if (name == "nvnSyncInitialize") {
        if (!x[0] || !devices.contains(x[1])) {
            return 0;
        }
        syncs.insert_or_assign(x[0], Sync{.device = x[1]});
        return 1;
    }
    if (name == "nvnSyncFinalize") {
        syncs.erase(x[0]);
    } else if (name == "nvnSyncWait") {
        const auto sync = syncs.find(x[0]);
        if (sync == syncs.end() || sync->second.status == SyncStatus::Failed) {
            return SyncFailed;
        }
        if (sync->second.status != SyncStatus::Signaled) {
            return SyncTimeout;
        }
        const u64 fence = sync->second.fence;
        if (!fence || system.GPU().CurrentSyncRequestFence() >= fence) {
            return SyncAlreadySignaled;
        }
        if (x[1] == std::numeric_limits<u64>::max()) {
            system.GPU().WaitForSyncOperation(fence);
            return 1;
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
        events.insert_or_assign(x[0], event);
        return 1;
    } else if (name == "nvnEventFinalize") {
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
    if (const auto binding = std::ranges::find(StateBindings, name, &StateBinding::name);
        binding != StateBindings.end()) {
        replay = binding->replay;
        valid = copy(x[1], binding->bytes);
    } else if (name == "nvnCommandBufferBindProgram") {
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
    } else if (name == "nvnCommandBufferBindTexturesDeferred") {
        const u32 count = static_cast<u32>(x[3]);
        replay = ReplayOp::BindTexturesDeferred;
        args = {static_cast<u32>(x[1]), static_cast<u32>(x[2]), count, x[4]};
        valid = count <= 64 && (!count || ValidRange(memory, x[4], count * sizeof(u64)));
    } else if (name == "nvnCommandBufferPushGraphicsConstants") {
        const u32 offset = static_cast<u32>(x[2]);
        const u32 size = std::min<u32>(static_cast<u32>(x[3]), 256) & ~3U;
        replay = ReplayOp::PushGraphicsConstants;
        args = {x[1] & 7, offset};
        valid = offset % 4 == 0 && Fits(offset, size, 256) && copy(x[4], size);
    } else if (name == "nvnCommandBufferBindUniformBuffers" ||
               name == "nvnCommandBufferBindUniformBuffersDeferred" ||
               name == "nvnCommandBufferBindStorageBuffers" ||
               name == "nvnCommandBufferBindTextures") {
        const u64 count = static_cast<u32>(x[3]);
        const u64 element_size = name == "nvnCommandBufferBindTextures" ? 8 : 16;
        if (name.starts_with("nvnCommandBufferBindUniformBuffers")) {
            replay = ReplayOp::BindUniformBuffers;
        } else if (name == "nvnCommandBufferBindTextures") {
            replay = ReplayOp::BindTextures;
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
        const u64 first = static_cast<u32>(x[1]);
        const u64 count = static_cast<u32>(x[2]);
        args = {first, count};
        valid = Fits(first, count, 16) && copy(x[3], count * 4 * sizeof(u32));
    } else if (name == "nvnCommandBufferUpdateUniformBuffer") {
        replay = ReplayOp::UpdateUniformBuffer;
        args = {x[1], x[2], x[3], x[4]};
        valid = x[4] <= (16ULL << 20) && Fits(x[3], x[4], x[2]) && copy(x[5], x[4]);
    } else if (name == "nvnCommandBufferSignalEvent" || name == "nvnCommandBufferWaitEvent") {
        const auto event = events.find(x[1]);
        valid = event != events.end();
        if (valid) {
            args = {event->second.pool_address + event->second.offset, static_cast<u32>(x[2]),
                    static_cast<u32>(x[3])};
            if (name == "nvnCommandBufferSignalEvent") {
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
        UpdateCommandCursor(memory, x[0], command_buffers.at(x[0]));
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
        auto& object = command_buffers.insert_or_assign(x[0], CommandBuffer{}).first->second;
        UpdateCommandBuffer(memory, x[0], object, {});
        return 1;
    }
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
            // A guest-visible handle points into its control allocation.
            constexpr std::array<u8, ControlRecordBytes> header{1};
            memory.WriteBlock(*handle - ControlHandleOffset, header.data(), header.size());
            memory.Write8(*handle, 2);
            UpdateCommandBuffer(memory, x[0], object, *recorder.GetState(x[0]));
            return *handle;
        }
        ok = false;
    } else if (name == "nvnCommandBufferCallCommands" || name == "nvnCommandBufferCopyCommands") {
        if (memory.Read8(x[0] + 0x11)) {
            recorder.Begin(x[0]);
        }
        const auto handles = ReadHandles(memory, x[2], static_cast<u32>(x[1]));
        ok = handles && recorder.AppendHandles(x[0], *handles);
    } else if (const auto command = std::ranges::find(ScalarCommands, name, &ScalarCommand::name);
               command != ScalarCommands.end()) {
        if (memory.Read8(x[0] + 0x11)) {
            recorder.Begin(x[0]);
        }
        const auto args = std::span{x}.subspan(1, command->args);
        const auto floats = std::span{call.v}.first(command->float_args);
        const auto bytes = std::span<const u8>{reinterpret_cast<const u8*>(floats.data()),
                                               floats.size_bytes()};
        const u64 charge = CommandBytes(name, call);
        ReplayOp replay{};
        if (name == "nvnCommandBufferBarrier") {
            replay = ReplayOp::Barrier;
        } else if (name == "nvnCommandBufferSetAlphaRef") {
            replay = ReplayOp::SetAlphaRef;
        } else if (name == "nvnCommandBufferClearDepthStencil") {
            replay = ReplayOp::ClearDepthStencil;
        } else if (name == "nvnCommandBufferSetDepthRange") {
            replay = ReplayOp::SetDepthRange;
        } else if (name == "nvnCommandBufferSetStencilRef") {
            replay = ReplayOp::SetStencilRef;
        } else if (name == "nvnCommandBufferSetStencilMask") {
            replay = ReplayOp::SetStencilMask;
        } else if (name == "nvnCommandBufferSetStencilValueMask") {
            replay = ReplayOp::SetStencilValueMask;
        } else if (name == "nvnCommandBufferBindUniformBuffer") {
            replay = ReplayOp::BindUniformBuffer;
        } else if (name == "nvnCommandBufferBindTexture") {
            replay = ReplayOp::BindTexture;
        } else if (name == "nvnCommandBufferBindVertexBuffer") {
            replay = ReplayOp::BindVertexBuffer;
        } else if (name == "nvnCommandBufferSetTexturePool") {
            replay = ReplayOp::SetTexturePool;
        } else if (name == "nvnCommandBufferSetSamplerPool") {
            replay = ReplayOp::SetSamplerPool;
        } else if (name == "nvnCommandBufferSetStencilCullCriteria") {
            replay = ReplayOp::StencilCullCriteria;
        } else if (name == "nvnCommandBufferSetViewport") {
            replay = ReplayOp::SetViewport;
        } else if (name == "nvnCommandBufferSetScissor") {
            replay = ReplayOp::SetScissor;
        } else if (name == "nvnCommandBufferDrawArrays") {
            replay = ReplayOp::Draw;
        } else if (name == "nvnCommandBufferDrawArraysInstanced") {
            replay = ReplayOp::DrawInstanced;
        } else if (name == "nvnCommandBufferDrawElements") {
            replay = ReplayOp::DrawIndexed;
        } else if (name == "nvnCommandBufferDrawElementsBaseVertex") {
            replay = ReplayOp::DrawIndexedBaseVertex;
        } else if (name == "nvnCommandBufferDrawElementsInstanced") {
            replay = ReplayOp::DrawIndexedInstanced;
        } else if (name.starts_with("nvnCommandBufferDispatch")) {
            replay = ReplayOp::Dispatch;
        }
        if (!recorder.Record(x[0], static_cast<u32>(index), args, bytes, charge, {}, {}, replay)) {
            NoteMissing(fmt::format("{} recording or memory state", name));
        } else if (charge) {
            UpdateCommandCursor(memory, x[0], object);
        }
        return 0;
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

void Session::NoteMissing(std::string_view name) {
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
    std::vector<std::pair<std::string_view, u64>> totals;
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
}

} // namespace Core::BananzaNvn2Hle
