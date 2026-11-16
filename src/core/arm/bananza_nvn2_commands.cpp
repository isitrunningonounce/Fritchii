// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>
#include <algorithm>
#include <limits>
#include <utility>

#include "core/arm/bananza_nvn2_commands.h"

namespace Core::BananzaNvn2Commands {
namespace {

constexpr u64 MaxCommands = 1ULL << 20;
constexpr u64 MaxPayloadBytes = 64ULL << 20;
constexpr u32 MaxNestingDepth = 64;

bool Fits(u64 used, u64 added, u64 limit) {
    return used <= limit && added <= limit - used;
}

} // Anonymous namespace

Retirement& GetRetirement() {
    static Retirement retirement;
    return retirement;
}

namespace {
std::mutex snapshot_pool_mutex;
std::vector<std::unique_ptr<u8[]>> snapshot_pool;
constexpr size_t MaxPooledSnapshotBlocks = 48;
} // Anonymous namespace

u8* SnapshotBlocks::Allocate(u64 size) {
    if (size > BlockBytes / 2) {
        large.push_back(std::make_unique_for_overwrite<u8[]>(size));
        return large.back().get();
    }
    u64 offset = (used + 63) & ~u64{63};
    if (offset + size > BlockBytes) {
        std::unique_ptr<u8[]> block;
        {
            std::scoped_lock lock{snapshot_pool_mutex};
            if (!snapshot_pool.empty()) {
                block = std::move(snapshot_pool.back());
                snapshot_pool.pop_back();
            }
        }
        if (!block) {
            block = std::make_unique_for_overwrite<u8[]>(BlockBytes);
        }
        blocks.push_back(std::move(block));
        offset = 0;
    }
    used = offset + size;
    return blocks.back().get() + offset;
}

void SnapshotBlocks::Release() {
    if (!blocks.empty()) {
        std::scoped_lock lock{snapshot_pool_mutex};
        for (auto& block : blocks) {
            if (snapshot_pool.size() < MaxPooledSnapshotBlocks) {
                snapshot_pool.push_back(std::move(block));
            }
        }
    }
    blocks.clear();
    large.clear();
    used = BlockBytes;
}

void Retirement::Retire(u64 sequence, bool writes_done) {
    {
        std::scoped_lock lock{mutex};
        if (retired.load(std::memory_order_relaxed) < sequence) {
            retired.store(sequence, std::memory_order_release);
        }
        if (writes_done && written.load(std::memory_order_relaxed) < sequence) {
            written.store(sequence, std::memory_order_release);
        }
    }
    cv.notify_all();
}

void Retirement::Wait(u64 sequence) {
    if (IsRetired(sequence)) {
        return;
    }
    std::unique_lock lock{mutex};
    cv.wait(lock, [&] { return IsRetired(sequence) || stopped; });
}

void Retirement::MarkReplayed(u64 sequence) {
    {
        std::scoped_lock lock{mutex};
        if (replayed.load(std::memory_order_relaxed) < sequence) {
            replayed.store(sequence, std::memory_order_release);
        }
    }
    cv.notify_all();
}

void Retirement::WaitReplayed(u64 sequence) {
    if (IsReplayed(sequence)) {
        return;
    }
    std::unique_lock lock{mutex};
    cv.wait(lock, [&] { return IsReplayed(sequence) || stopped; });
}

void Retirement::RetireWrites(u64 sequence) {
    {
        std::scoped_lock lock{mutex};
        if (written.load(std::memory_order_relaxed) < sequence) {
            written.store(sequence, std::memory_order_release);
        }
    }
    cv.notify_all();
}

void Retirement::WaitWrites(u64 sequence) {
    if (WritesDone(sequence)) {
        return;
    }
    std::unique_lock lock{mutex};
    cv.wait(lock, [&] { return WritesDone(sequence) || stopped; });
}

void Retirement::Start() {
    std::scoped_lock lock{mutex};
    stopped = false;
    retired.store(next.load(std::memory_order_relaxed), std::memory_order_release);
    written.store(next.load(std::memory_order_relaxed), std::memory_order_release);
    replayed.store(next.load(std::memory_order_relaxed), std::memory_order_release);
}

void Retirement::Stop() {
    std::function<void()> wait;
    {
        std::scoped_lock lock{mutex};
        wait = drain;
    }
    if (wait) {
        wait();
    }
    {
        std::scoped_lock lock{mutex};
        stopped = true;
    }
    cv.notify_all();
}

void Retirement::SetDrain(std::function<void()> drain_) {
    std::scoped_lock lock{mutex};
    drain = std::move(drain_);
}

namespace {
// The sizes of the last recording this thread ended: the next one starts with that room.
thread_local size_t operations_hint = 64;
thread_local size_t called_hint = 16;
constexpr size_t MaxOperationsHint = 16384;
constexpr size_t MaxCalledHint = 4096;

// A list that ended far smaller than its room gives the room back: the game keeps small
// command lists for good.
template <typename List>
void TrimList(List& list) {
    if (list.capacity() > 4 * list.size() + 16) {
        list.shrink_to_fit();
    }
}
} // Anonymous namespace

bool Recorder::Initialize(u64 command_buffer) {
    if (command_buffer == 0) {
        return false;
    }
    auto& shard = ShardOf(command_buffer);
    std::scoped_lock lock{shard.mutex};
    shard.buffers.insert_or_assign(command_buffer, Buffer{});
    return true;
}

bool Recorder::Finalize(u64 command_buffer) {
    auto& shard = ShardOf(command_buffer);
    std::scoped_lock lock{shard.mutex};
    return shard.buffers.erase(command_buffer) != 0;
}

bool Recorder::SetCommandMemory(u64 command_buffer, u64 size) {
    auto& shard = ShardOf(command_buffer);
    std::shared_lock lock{shard.mutex};
    const auto it = shard.buffers.find(command_buffer);
    if (it == shard.buffers.end()) {
        return false;
    }
    it->second.state.command_size = size;
    it->second.state.command_used = 0;
    return true;
}

bool Recorder::SetControlMemory(u64 command_buffer, u64 guest_base, u64 size) {
    if ((size != 0 && guest_base == 0) || !Fits(guest_base, size, std::numeric_limits<u64>::max())) {
        return false;
    }
    auto& shard = ShardOf(command_buffer);
    std::shared_lock lock{shard.mutex};
    const auto it = shard.buffers.find(command_buffer);
    if (it == shard.buffers.end()) {
        return false;
    }
    const u64 end = guest_base + size;
    if (size != 0) {
        std::vector<std::shared_ptr<const Recording>> dead;
        {
            std::scoped_lock handles_lock{handles_mutex};
            const u64 min_begin = guest_base - std::min(guest_base, max_control_record_bytes);
            auto record = handles_by_control_begin.lower_bound({min_begin, 0});
            while (record != handles_by_control_begin.end() && record->first < end) {
                const auto handle = live_handles.find(record->second);
                if (guest_base < handle->second.control_end) {
                    dead.push_back(std::move(handle->second.recording));
                    live_handles.erase(handle);
                    record = handles_by_control_begin.erase(record);
                } else {
                    ++record;
                }
            }
        }
        if (!dead.empty()) {
            // The last reference to a recording frees its lists: on the worker.
            garbage.QueueWork([dead = std::move(dead)] {});
        }
    }
    auto& state = it->second.state;
    state.control_base = guest_base;
    state.control_size = size;
    state.control_used = 0;
    return true;
}

bool Recorder::Begin(u64 command_buffer) {
    auto& shard = ShardOf(command_buffer);
    std::shared_lock lock{shard.mutex};
    const auto it = shard.buffers.find(command_buffer);
    if (it == shard.buffers.end() || it->second.state.recording) {
        return false;
    }
    auto& buffer = it->second;
    buffer.operations.clear();
    buffer.called.clear();
    buffer.operations.reserve(std::min(operations_hint, MaxOperationsHint));
    buffer.called.reserve(std::min(called_hint, MaxCalledHint));
    buffer.command_count = 0;
    buffer.payload_bytes = 0;
    buffer.argument_bytes = 0;
    buffer.draw_count = 0;
    buffer.called_commands = 0;
    buffer.called_draws = 0;
    buffer.copied_commands = 0;
    buffer.copied_draws = 0;
    buffer.nesting_depth = 1;
    buffer.state.recording = true;
    return true;
}

bool Recorder::Record(u64 command_buffer, u32 op, std::span<const u64> args,
                      std::span<const u8> payload, u64 command_bytes,
                      std::span<const BananzaNvn2Hle::ResourceTextureInfo> textures,
                      std::shared_ptr<const BananzaNvn2Hle::ResourceProgramInfo> program,
                      ReplayOp replay) {
    if (args.size() > MaxPayloadBytes / sizeof(u64) || payload.size() > MaxPayloadBytes ||
        textures.size() > 9) {
        return false;
    }
    auto& shard = ShardOf(command_buffer);
    std::shared_lock lock{shard.mutex};
    const auto it = shard.buffers.find(command_buffer);
    if (it == shard.buffers.end() || !it->second.state.recording) {
        return false;
    }
    auto& buffer = it->second;
    const u64 argument_bytes = args.size_bytes();
    if (buffer.operations.size() >= MaxCommands || buffer.command_count >= MaxCommands ||
        !Fits(buffer.payload_bytes, payload.size(), MaxPayloadBytes) ||
        !Fits(buffer.argument_bytes, argument_bytes, MaxPayloadBytes) ||
        !Fits(buffer.state.command_used, command_bytes, buffer.state.command_size)) {
        return false;
    }
    Operation operation;
    operation.id = op;
    operation.replay = replay;
    operation.args.assign(args.begin(), args.end());
    if (!payload.empty() || !textures.empty() || program) {
        operation.extra = std::make_unique<OperationExtra>();
        operation.extra->payload.assign(payload.begin(), payload.end());
        operation.extra->textures.assign(textures.begin(), textures.end());
        operation.extra->program = std::move(program);
    }
    buffer.operations.push_back(std::move(operation));
    ++buffer.command_count;
    buffer.draw_count += replay >= ReplayOp::Draw && replay <= ReplayOp::DrawIndexedIndirect;
    buffer.payload_bytes += payload.size();
    buffer.argument_bytes += argument_bytes;
    buffer.state.command_used += command_bytes;
    return true;
}

bool Recorder::AppendHandles(u64 command_buffer, std::span<const u64> handles, u64 command_bytes,
                             bool call) {
    auto& shard = ShardOf(command_buffer);
    std::shared_lock lock{shard.mutex};
    const auto it = shard.buffers.find(command_buffer);
    if (it == shard.buffers.end() || !it->second.state.recording) {
        return false;
    }
    auto& buffer = it->second;
    if (!Fits(buffer.operations.size(), handles.size(), MaxCommands) ||
        !Fits(buffer.state.command_used, command_bytes, buffer.state.command_size)) {
        return false;
    }
    u64 count = buffer.command_count;
    u64 bytes = buffer.payload_bytes;
    u32 depth = buffer.nesting_depth;
    // Each handle is looked up once; the entries stay where they are under the shared lock.
    std::array<const Handle*, 16> few;
    std::vector<const Handle*> many;
    if (handles.size() > few.size()) {
        many.resize(handles.size());
    }
    const Handle** const entries = handles.size() > few.size() ? many.data() : few.data();
    std::shared_lock handles_lock{handles_mutex};
    for (size_t i = 0; i < handles.size(); ++i) {
        const auto found = live_handles.find(handles[i]);
        if (found == live_handles.end() ||
            found->second.recording->nesting_depth >= MaxNestingDepth ||
            !Fits(count, found->second.recording->command_count, MaxCommands) ||
            !Fits(bytes, found->second.recording->payload_bytes, MaxPayloadBytes)) {
            return false;
        }
        count += found->second.recording->command_count;
        bytes += found->second.recording->payload_bytes;
        depth = std::max(depth, found->second.recording->nesting_depth + 1);
        entries[i] = &found->second;
    }
    for (size_t i = 0; i < handles.size(); ++i) {
        Operation operation;
        const auto& recording = entries[i]->recording;
        operation.nested = recording.get();
        buffer.called.push_back(recording);
        buffer.draw_count += operation.nested->draw_count;
        if (call) {
            buffer.called_commands += operation.nested->command_count;
            buffer.called_draws += operation.nested->draw_count;
        } else {
            buffer.copied_commands += operation.nested->command_count;
            buffer.copied_draws += operation.nested->draw_count;
        }
        buffer.operations.push_back(std::move(operation));
    }
    buffer.command_count = count;
    buffer.payload_bytes = bytes;
    buffer.nesting_depth = depth;
    buffer.state.command_used += command_bytes;
    return true;
}

void BuildScanEntries(Recording& recording) {
    using Kind = ScanEntry::Kind;
    auto& scan = recording.scan;
    // Counted first: a list that grows while it is filled is copied again and again.
    size_t scan_size{};
    u64 called_entries{};
    u64 calls{};
    for (const Operation& op : recording.operations) {
        if (op.nested) {
            ++scan_size;
            ++calls;
            called_entries += op.nested->scan_total;
            recording.flat_total += op.nested->flat_total;
        } else {
            // A bind of several slots makes an entry for each; a draw may make two.
            scan_size += 2 + (op.extra ? op.extra->payload.size() / 8 : 0);
            ++recording.flat_total;
        }
    }
    scan.reserve(scan_size);
    // An indexed draw needs its own entry only when something before it may have left
    // targets, buffers or descriptors pending.
    bool draw_entry = true;
    const auto general = [&](const Operation& op) {
        scan.push_back({Kind::Op, 0, 0, reinterpret_cast<u64>(&op), 0});
        draw_entry = true;
    };
    for (const Operation& op : recording.operations) {
        if (op.nested) {
            // The called recording's entries take this place when a submission is flattened.
            scan.push_back({Kind::Call, 0, 0, reinterpret_cast<u64>(op.nested), 0});
            draw_entry = true;
            continue;
        }
        const auto& args = op.args;
        switch (op.replay) {
        case ReplayOp::BindVertexBuffer:
            if (args.size() == 3 && args[0] <= 0xffff) {
                scan.push_back({Kind::Vertex, 0, static_cast<u16>(args[0]), args[1], args[2]});
            }
            break;
        case ReplayOp::BindUniformBuffer:
        case ReplayOp::BindStorageBuffer:
            if (args.size() == 4 && args[0] <= 0xff && args[1] <= 0xffff) {
                scan.push_back({op.replay == ReplayOp::BindUniformBuffer ? Kind::Uniform
                                                                          : Kind::Storage,
                                static_cast<u8>(args[0]), static_cast<u16>(args[1]), args[2],
                                args[3]});
            }
            break;
        case ReplayOp::BindUniformBuffers:
        case ReplayOp::BindStorageBuffers: {
            if (args.size() != 3) {
                break;
            }
            const auto& payload = op.Payload();
            const u64 count = payload.size() / 16;
            // The compute stage marks its storage pending whatever the slots are.
            if (args[0] == 5 || args[0] > 0xff || args[1] > 0xffff - count) {
                general(op);
                break;
            }
            const Kind kind = op.replay == ReplayOp::BindUniformBuffers ? Kind::Uniform
                                                                        : Kind::Storage;
            for (u64 i = 0; i < count; ++i) {
                u64 range[2];
                std::memcpy(range, payload.data() + i * 16, 16);
                scan.push_back({kind, static_cast<u8>(args[0]), static_cast<u16>(args[1] + i),
                                range[0], range[1]});
            }
            break;
        }
        case ReplayOp::BindTexture:
        case ReplayOp::BindImage:
            if (args.size() == 3 && args[0] <= 0xff && args[1] <= 0xffff) {
                scan.push_back({op.replay == ReplayOp::BindTexture ? Kind::Texture : Kind::Image,
                                static_cast<u8>(args[0]), static_cast<u16>(args[1]), args[2], 0});
            }
            break;
        case ReplayOp::BindTextures: {
            if (args.size() != 3) {
                break;
            }
            const auto& payload = op.Payload();
            const u64 count = payload.size() / 8;
            if (args[0] > 0xff || args[1] > 0xffff - count) {
                general(op);
                break;
            }
            for (u64 i = 0; i < count; ++i) {
                u64 handle;
                std::memcpy(&handle, payload.data() + i * 8, 8);
                scan.push_back({Kind::Texture, static_cast<u8>(args[0]),
                                static_cast<u16>(args[1] + i), handle, 0});
            }
            break;
        }
        case ReplayOp::DrawIndexed:
        case ReplayOp::DrawIndexedBaseVertex:
        case ReplayOp::DrawIndexedInstanced:
            if (args.size() >= 4 && args[1] <= 2 && args[2] <= ScanMaxRange / 4) {
                scan.push_back({Kind::Index, 0, 0, args[3], args[2] << args[1]});
            }
            if (draw_entry) {
                scan.push_back({Kind::Draw, 0, 0, 0, 0});
                draw_entry = false;
            }
            break;
        case ReplayOp::BindProgram:
            // Only a graphics program changes what the scan does.
            if (args.size() == 2 && (args[1] & 1)) {
                general(op);
            }
            break;
        case ReplayOp::SetRenderTargets:
        case ReplayOp::CopyTexture:
        case ReplayOp::CopyBufferToTexture:
        case ReplayOp::SignalEvent:
        case ReplayOp::WaitEvent:
        case ReplayOp::ReportCounter:
        case ReplayOp::SetRenderEnableConditional:
        case ReplayOp::SetTexturePool:
        case ReplayOp::SetSamplerPool:
        case ReplayOp::SetCopyRowStride:
        case ReplayOp::BindTexturesDeferred:
        case ReplayOp::Dispatch:
            general(op);
            break;
        default:
            if ((op.replay >= ReplayOp::ClearColor && op.replay <= ReplayOp::ClearDepthStencil) ||
                (op.replay >= ReplayOp::Draw && op.replay <= ReplayOp::DrawIndexedIndirect)) {
                general(op);
            }
            break;
        }
    }
    recording.scan_total = scan.size() - calls + called_entries;
}

namespace {
void FlattenOperations(const Recording& recording, std::vector<const Operation*>& flat) {
    for (const Operation& op : recording.operations) {
        if (op.nested) {
            FlattenOperations(*op.nested, flat);
        } else {
            flat.push_back(&op);
        }
    }
}
} // Anonymous namespace

void FlattenRecordings(Submission& submission) {
    const size_t count = submission.recordings.size();
    submission.flats.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const Recording& recording = *submission.recordings[i];
        submission.flats[i].reserve(recording.flat_total);
        FlattenOperations(recording, submission.flats[i]);
    }
}

std::optional<u64> Recorder::End(u64 command_buffer, u64 control_bytes, u64 guest_handle_offset) {
    auto& shard = ShardOf(command_buffer);
    std::shared_lock lock{shard.mutex};
    const auto it = shard.buffers.find(command_buffer);
    if (it == shard.buffers.end() || !it->second.state.recording || control_bytes == 0) {
        return std::nullopt;
    }
    auto& buffer = it->second;
    auto& state = buffer.state;
    if (!Fits(state.control_used, control_bytes, state.control_size)) {
        return std::nullopt;
    }
    const u64 begin = state.control_base + state.control_used;
    if (guest_handle_offset >= control_bytes) {
        return std::nullopt;
    }
    auto snapshot = std::make_shared<Recording>();
    u64 handle{};
    {
        std::scoped_lock handles_lock{handles_mutex};
        if (next_handle == 0 || next_recording == 0) {
            return std::nullopt;
        }
        handle = guest_handle_offset ? begin + guest_handle_offset : next_handle;
        if (live_handles.contains(handle)) {
            return std::nullopt;
        }
        snapshot->serial = next_recording++;
        if (!guest_handle_offset) {
            ++next_handle;
        }
    }
    snapshot->command_buffer = command_buffer;
    snapshot->draw_count = buffer.draw_count;
    snapshot->called_commands = buffer.called_commands;
    snapshot->called_draws = buffer.called_draws;
    snapshot->copied_commands = buffer.copied_commands;
    snapshot->copied_draws = buffer.copied_draws;
    snapshot->operations = std::move(buffer.operations);
    snapshot->called = std::move(buffer.called);
    operations_hint = snapshot->operations.size() + snapshot->operations.size() / 8 + 16;
    called_hint = snapshot->called.size() + snapshot->called.size() / 8 + 8;
    TrimList(snapshot->operations);
    TrimList(snapshot->called);
    snapshot->command_count = buffer.command_count;
    snapshot->payload_bytes = buffer.payload_bytes;
    snapshot->nesting_depth = buffer.nesting_depth;
    state.control_used += control_bytes;
    state.recording = false;
    // Nobody knows the handle yet: the entries are made without a lock.
    lock.unlock();
    BuildScanEntries(*snapshot);
    std::scoped_lock handles_lock{handles_mutex};
    live_handles.emplace(handle, Handle{std::move(snapshot), begin, begin + control_bytes});
    handles_by_control_begin.emplace(begin, handle);
    max_control_record_bytes = std::max(max_control_record_bytes, control_bytes);
    return handle;
}

std::optional<State> Recorder::GetState(u64 command_buffer) const {
    auto& shard = ShardOf(command_buffer);
    std::shared_lock lock{shard.mutex};
    const auto it = shard.buffers.find(command_buffer);
    return it != shard.buffers.end() ? std::optional{it->second.state} : std::nullopt;
}

bool Recorder::FinalizeHandle(u64 handle) {
    std::shared_ptr<const Recording> dead;
    {
        std::scoped_lock lock{handles_mutex};
        const auto found = live_handles.find(handle);
        if (found == live_handles.end()) {
            return false;
        }
        handles_by_control_begin.erase({found->second.control_begin, handle});
        dead = std::move(found->second.recording);
        live_handles.erase(found);
    }
    garbage.QueueWork([dead = std::move(dead)] {});
    return true;
}

std::optional<Submission> Recorder::Submit(u64 queue, std::span<const u64> handles) {
    if (queue == 0 || handles.size() > MaxCommands) {
        return std::nullopt;
    }
    std::scoped_lock lock{handles_mutex};
    if (next_submission == 0) {
        return std::nullopt;
    }
    Submission submission;
    submission.queue = queue;
    for (const u64 handle : handles) {
        const auto found = live_handles.find(handle);
        if (found == live_handles.end() ||
            !Fits(submission.command_count, found->second.recording->command_count,
                  std::numeric_limits<u64>::max()) ||
            !Fits(submission.payload_bytes, found->second.recording->payload_bytes,
                  std::numeric_limits<u64>::max())) {
            return std::nullopt;
        }
        submission.command_count += found->second.recording->command_count;
        submission.payload_bytes += found->second.recording->payload_bytes;
    }
    submission.handles.assign(handles.begin(), handles.end());
    submission.recordings.reserve(handles.size());
    for (const u64 handle : handles) {
        submission.recordings.push_back(live_handles.at(handle).recording);
    }
    ++next_submission;
    submission.sequence = GetRetirement().NextSequence();
    return submission;
}

} // namespace Core::BananzaNvn2Commands
