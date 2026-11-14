// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/common_types.h"
#include "core/arm/bananza_nvn2_texture.h"

namespace Core::Memory {
class Memory;
}

namespace Core::BananzaNvn2Hle {

struct GuestCall;

struct ResourcePoolInfo {
    u64 device{};
    u64 storage{};
    u64 size{};
    u64 gpu_address{};
    u32 flags{};
};

struct ResourceTextureInfo {
    u64 device{};
    u64 pool{};
    u64 offset{};
    u64 serial{};
    ResourcePoolInfo storage;
    TextureDescription description;
    TextureLayout layout;
};

struct ResourceShaderInfo {
    u64 data_address{};
    u64 control_address{};
    u32 stage{}; // Vertex, tess-control, tess-eval, geometry, fragment, compute, task, mesh.
    u32 code_size{}; // Native +0x38: includes a graphics SPH, excludes trailing constants.
    u32 header_size{};
    u32 constant_offset{};
    u32 constant_size{};
    std::vector<u8> control;
    std::vector<u8> code; // Full storage extent, including bank 1 constants.
};

struct ResourceProgramInfo {
    static constexpr bool Executable = false;
    u64 device{};
    u64 serial{};
    std::vector<ResourceShaderInfo> shaders;
    bool resets_indirect_draws{};
};

struct ResourceSamplerInfo {
    u64 device{};
    u32 min_filter{4};
    u32 mag_filter{1};
    std::array<u32, 3> wrap{1, 1, 1};
    u32 compare_mode{};
    u32 compare_func{2};
    u32 reduction{};
    std::array<u32, 4> border{};
    u32 border_kind{};
    float min_lod{-1000.0f};
    float max_lod{1000.0f};
    float lod_bias{};
    float anisotropy{1.0f};
    float lod_snap{};
};

struct ResourceTextureDescriptor {
    ResourceTextureInfo texture;
    std::optional<std::array<u8, 16>> view;
};

// The owning Session serializes access. No guest callbacks are executed here.
class Resources {
public:
    using DeviceLookup = std::function<bool(u64)>;
    using PoolLookup = std::function<std::optional<ResourcePoolInfo>(u64)>;
    using GpuAddressLookup = std::function<std::optional<u64>(u64, u64)>;

    std::optional<u64> Dispatch(Memory::Memory& memory, std::string_view name,
                               const GuestCall& call, const DeviceLookup& has_device,
                               const PoolLookup& find_pool,
                               const GpuAddressLookup& resolve_gpu = {});
    static bool IsFloatReturn(std::string_view name);
    bool HasTexture(u64 device, u64 texture) const;
    std::optional<ResourceTextureInfo> GetTexture(u64 texture) const;
    std::shared_ptr<const ResourceProgramInfo> GetProgram(u64 program) const;
    const std::unordered_map<u32, ResourceTextureDescriptor>* GetTexturePool(u64 pool) const;
    const std::unordered_map<u32, ResourceSamplerInfo>* GetSamplerPool(u64 pool) const;

private:
    struct TextureBuilder {
        u64 device{};
        u64 pool{};
        u64 offset{};
        u64 packaged_data{};
        u16 layout_word{};
        TextureDescription description;
    };

    using Sampler = ResourceSamplerInfo;
    using TextureDescriptor = ResourceTextureDescriptor;

    struct DescriptorPool {
        u64 pool{};
        u64 offset{};
        u32 count{};
        ResourcePoolInfo storage;
        std::unordered_map<u32, TextureDescriptor> textures;
        std::unordered_map<u32, Sampler> samplers;
    };

    struct Shader {
        u64 data{};
        u64 control{};
        u64 reserved{};
    };

    struct Program {
        u64 device{};
        std::vector<Shader> shaders;
        std::shared_ptr<const ResourceProgramInfo> snapshot;
    };

    std::optional<u64> HandleTexture(Memory::Memory& memory, std::string_view name,
                                    const GuestCall& call, const DeviceLookup& has_device,
                                    const PoolLookup& find_pool,
                                    const GpuAddressLookup& resolve_gpu);
    std::optional<u64> HandleView(Memory::Memory& memory, std::string_view name,
                                 const GuestCall& call);
    std::optional<u64> HandleSampler(Memory::Memory& memory, std::string_view name,
                                    const GuestCall& call, const DeviceLookup& has_device);
    std::optional<u64> HandleDescriptorPool(Memory::Memory& memory, std::string_view name,
                                           const GuestCall& call, const PoolLookup& find_pool);
    std::optional<u64> HandleProgram(Memory::Memory& memory, std::string_view name,
                                    const GuestCall& call, const DeviceLookup& has_device,
                                    const GpuAddressLookup& resolve_gpu);
    static std::optional<std::vector<Shader>> ReadShaders(Memory::Memory& memory, u64 address,
                                                         u64 count, bool extended);
    static std::optional<TextureBuilder> ReadTextureBuilder(Memory::Memory& memory, u64 address);
    static void WriteTextureBuilder(Memory::Memory& memory, u64 address,
                                    const TextureBuilder& builder);
    void DiagnoseTexture(std::string_view name, const GuestCall& call, std::string_view reason,
                         const TextureBuilder* builder = nullptr);
    void NoteDiagnostic(std::string key, std::string message);
    std::shared_ptr<const ResourceProgramInfo> StageProgram(Memory::Memory& memory,
                                                           const Program& program,
                                                           const GpuAddressLookup& resolve_gpu);

    std::unordered_map<u64, ResourceTextureInfo> textures;
    std::unordered_map<u64, Sampler> sampler_builders;
    std::unordered_map<u64, Sampler> samplers;
    std::unordered_map<u64, DescriptorPool> texture_pools;
    std::unordered_map<u64, DescriptorPool> sampler_pools;
    std::unordered_map<u64, Program> program_builders;
    std::unordered_map<u64, Program> programs;
    u64 next_serial{1};
    bool reported_shader_staging{};
    std::unordered_set<std::string> diagnostics;
};

} // namespace Core::BananzaNvn2Hle
