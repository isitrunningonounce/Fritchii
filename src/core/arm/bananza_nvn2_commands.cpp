// SPDX-License-Identifier: GPL-3.0-or-later

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

bool Recorder::Initialize(u64 command_buffer) {
    if (command_buffer == 0) {
        return false;
    }
    std::scoped_lock lock{mutex};
    buffers.insert_or_assign(command_buffer, Buffer{});
    return true;
}

bool Recorder::Finalize(u64 command_buffer) {
    std::scoped_lock lock{mutex};
    return buffers.erase(command_buffer) != 0;
}

bool Recorder::SetCommandMemory(u64 command_buffer, u64 size) {
    std::scoped_lock lock{mutex};
    const auto it = buffers.find(command_buffer);
    if (it == buffers.end()) {
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
    std::scoped_lock lock{mutex};
    const auto it = buffers.find(command_buffer);
    if (it == buffers.end()) {
        return false;
    }
    const u64 end = guest_base + size;
    if (size != 0) {
        for (auto handle = live_handles.begin(); handle != live_handles.end();) {
            if (guest_base < handle->second.control_end && handle->second.control_begin < end) {
                handle = live_handles.erase(handle);
            } else {
                ++handle;
            }
        }
    }
    auto& state = it->second.state;
    state.control_base = guest_base;
    state.control_size = size;
    state.control_used = 0;
    return true;
}

bool Recorder::Begin(u64 command_buffer) {
    std::scoped_lock lock{mutex};
    const auto it = buffers.find(command_buffer);
    if (it == buffers.end() || it->second.state.recording) {
        return false;
    }
    auto& buffer = it->second;
    buffer.operations.clear();
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
    std::scoped_lock lock{mutex};
    const auto it = buffers.find(command_buffer);
    if (it == buffers.end() || !it->second.state.recording) {
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
    operation.payload.assign(payload.begin(), payload.end());
    operation.textures.assign(textures.begin(), textures.end());
    operation.program = std::move(program);
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
    std::scoped_lock lock{mutex};
    const auto it = buffers.find(command_buffer);
    if (it == buffers.end() || !it->second.state.recording) {
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
    for (const u64 handle : handles) {
        const auto found = live_handles.find(handle);
        if (found == live_handles.end() ||
            found->second.recording->nesting_depth >= MaxNestingDepth ||
            !Fits(count, found->second.recording->command_count, MaxCommands) ||
            !Fits(bytes, found->second.recording->payload_bytes, MaxPayloadBytes)) {
            return false;
        }
        count += found->second.recording->command_count;
        bytes += found->second.recording->payload_bytes;
        depth = std::max(depth, found->second.recording->nesting_depth + 1);
    }
    buffer.operations.reserve(buffer.operations.size() + handles.size());
    for (const u64 handle : handles) {
        Operation operation;
        operation.nested = live_handles.at(handle).recording;
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

std::optional<u64> Recorder::End(u64 command_buffer, u64 control_bytes, u64 guest_handle_offset) {
    std::scoped_lock lock{mutex};
    const auto it = buffers.find(command_buffer);
    if (it == buffers.end() || !it->second.state.recording || control_bytes == 0 ||
        next_handle == 0 || next_recording == 0) {
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
    const u64 handle = guest_handle_offset ? begin + guest_handle_offset : next_handle;
    if (live_handles.contains(handle)) {
        return std::nullopt;
    }
    auto snapshot = std::make_shared<Recording>();
    snapshot->serial = next_recording++;
    snapshot->command_buffer = command_buffer;
    snapshot->draw_count = buffer.draw_count;
    snapshot->called_commands = buffer.called_commands;
    snapshot->called_draws = buffer.called_draws;
    snapshot->copied_commands = buffer.copied_commands;
    snapshot->copied_draws = buffer.copied_draws;
    snapshot->operations = std::move(buffer.operations);
    snapshot->command_count = buffer.command_count;
    snapshot->payload_bytes = buffer.payload_bytes;
    snapshot->nesting_depth = buffer.nesting_depth;
    live_handles.emplace(handle, Handle{std::move(snapshot), begin, begin + control_bytes});
    if (!guest_handle_offset) {
        ++next_handle;
    }
    state.control_used += control_bytes;
    state.recording = false;
    return handle;
}

std::optional<State> Recorder::GetState(u64 command_buffer) const {
    std::scoped_lock lock{mutex};
    const auto it = buffers.find(command_buffer);
    return it != buffers.end() ? std::optional{it->second.state} : std::nullopt;
}

bool Recorder::FinalizeHandle(u64 handle) {
    std::scoped_lock lock{mutex};
    return live_handles.erase(handle) != 0;
}

std::optional<Submission> Recorder::Submit(u64 queue, std::span<const u64> handles) {
    if (queue == 0 || handles.size() > MaxCommands) {
        return std::nullopt;
    }
    std::scoped_lock lock{mutex};
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
    submission.sequence = next_submission++;
    return submission;
}

} // namespace Core::BananzaNvn2Commands
