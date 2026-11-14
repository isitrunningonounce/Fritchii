// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "core/arm/bananza_nvn2_resources.h"

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
    Dispatch,
};

struct Operation {
    u32 id{};
    ReplayOp replay{};
    std::vector<u64> args;
    std::vector<u8> payload;
    std::vector<BananzaNvn2Hle::ResourceTextureInfo> textures;
    std::shared_ptr<const BananzaNvn2Hle::ResourceProgramInfo> program;
    // A nested recording owns its snapshot even after the original handle is retired.
    std::shared_ptr<const Recording> nested;
};

struct Recording {
    std::vector<Operation> operations;
    u64 command_count{};
    u64 payload_bytes{};
    u32 nesting_depth{1};
};

struct State {
    bool recording{};
    u64 command_size{};
    u64 command_used{};
    u64 control_base{};
    u64 control_size{};
    u64 control_used{};
};

struct Submission {
    static constexpr bool ExecutesGpuWork = false;

    u64 sequence{};
    u64 queue{};
    bool depth_minus_one_to_one{};
    bool origin_lower_left{};
    std::vector<u64> handles;
    std::vector<std::shared_ptr<const Recording>> recordings;
    u64 command_count{};
    u64 payload_bytes{};
    std::map<std::pair<u64, u64>, std::vector<u8>> buffer_snapshots;
    std::map<std::pair<u64, u64>, std::vector<u8>> deferred_data;
    std::unordered_map<u64, std::unordered_map<u32, BananzaNvn2Hle::ResourceTextureDescriptor>>
        texture_descriptors;
    std::unordered_map<u64, std::unordered_map<u32, BananzaNvn2Hle::ResourceSamplerInfo>>
        sampler_descriptors;
    std::map<u64, std::vector<u8>> texture_snapshots;
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
    bool AppendHandles(u64 command_buffer, std::span<const u64> handles, u64 command_bytes = 0);
    std::optional<u64> End(u64 command_buffer, u64 control_bytes, u64 guest_handle_offset = 0);
    bool FinalizeHandle(u64 handle);
    std::optional<State> GetState(u64 command_buffer) const;

    // Validates the entire batch and retains its snapshots; it does not execute or complete GPU work.
    std::optional<Submission> Submit(u64 queue, std::span<const u64> handles);

private:
    struct Buffer {
        State state;
        std::vector<Operation> operations;
        u64 command_count{};
        u64 payload_bytes{};
        u64 argument_bytes{};
        u32 nesting_depth{1};
    };

    struct Handle {
        std::shared_ptr<const Recording> recording;
        u64 control_begin{};
        u64 control_end{};
    };

    mutable std::mutex mutex;
    std::unordered_map<u64, Buffer> buffers;
    std::unordered_map<u64, Handle> live_handles;
    u64 next_handle{1};
    u64 next_submission{1};
};

} // namespace Core::BananzaNvn2Commands
