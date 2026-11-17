// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/shader_environment.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Core::BananzaNvn2Hle {
struct ResourceShaderInfo;
}

namespace Shader {
struct HostTranslateInfo;
struct Profile;
struct RuntimeInfo;
}

namespace Vulkan {

class Device;

enum class BananzaNvn2ShaderStatus {
    InvalidInput,
    UnsupportedStage,
    LoweringFailed,
    TranslationFailed,
    BindingMismatch,
    EmissionFailed,
    Success,
};

enum class BananzaNvn2ResourceAuthority {
    Speculative,
    ResourceIndependent,
    Validated,
};

struct BananzaNvn2ShaderResult {
    BananzaNvn2ShaderStatus status{BananzaNvn2ShaderStatus::InvalidInput};
    BananzaNvn2ResourceAuthority resource_authority{BananzaNvn2ResourceAuthority::Speculative};
    Shader::Stage stage{};
    u64 native_hash{};
    u64 lowered_hash{};
    u32 source_size{};
    u32 texture_bound{};
    std::array<u32, 3> workgroup_size{};
    u32 shared_memory_size{};
    u32 local_memory_size{};
    std::vector<u32> spirv;
    std::unique_ptr<Shader::Info> info;
    VideoCommon::RecordedEnvQueries queries;
    std::string error;
};

// Translation only: no Vulkan objects or guest execution. Bindings advance only on success.
// RuntimeInfo must describe the actual interface and render state before using the result.
// Vertex translation assumes viewport-transform=1, also exposed in queries. ResourceIndependent
// covers bindings only; bank 1 descriptors still need the staged literal bytes uploaded.
// A live environment must describe this same program; all recorded
// answers must match before resource_authority becomes Validated. No environment is retained.
[[nodiscard]] BananzaNvn2ShaderResult TranslateBananzaNvn2Shader(
    const Core::BananzaNvn2Hle::ResourceShaderInfo& shader, const Shader::Profile& profile,
    const Shader::HostTranslateInfo& host_info, const Shader::RuntimeInfo& runtime_info,
    Shader::Backend::Bindings& bindings, Shader::Environment* authoritative_bindings = nullptr,
    const std::array<u32, 8>* geometry_skip_mask = nullptr);

[[nodiscard]] vk::Pipeline CreateBananzaNvn2ComputePipeline(
    const Device& device, const BananzaNvn2ShaderResult& shader, VkPipelineLayout layout,
    VkPipelineCache cache = {});

} // namespace Vulkan
