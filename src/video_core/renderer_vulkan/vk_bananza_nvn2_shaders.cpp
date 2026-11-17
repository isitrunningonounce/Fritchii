// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <exception>
#include <span>
#include <stdexcept>

#include "core/arm/bananza_nvn2_resources.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "video_core/renderer_vulkan/vk_bananza_nvn2_shaders.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {
namespace {

constexpr u32 HeaderSize = 128;
constexpr u32 MaxCodeSize = HeaderSize + Shader::Ampere::ProgramCaptureState::MaximumInstructions * 16;
constexpr size_t MaxLoweredWords = (16U << 20) / sizeof(u64);
constexpr size_t MaxSpirvWords = (16U << 20) / sizeof(u32);

u32 ReadWord(std::span<const u8> bytes, size_t offset) {
    u32 value;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

bool HasCompleteCode(std::span<const u8> bytes, u32 header_size) {
    Shader::Ampere::ProgramCaptureState capture;
    for (size_t offset = header_size; offset < bytes.size(); offset += 16) {
        Shader::Ampere::Instruction instruction;
        std::memcpy(&instruction, bytes.data() + offset, sizeof(instruction));
        const auto status = capture.Observe(instruction, static_cast<u32>((offset - header_size) / 16));
        if (status == Shader::Ampere::ProgramCaptureState::Result::InvalidTarget) return false;
        if (status == Shader::Ampere::ProgramCaptureState::Result::Complete) return true;
    }
    return false;
}

bool HasIndependentResources(const Shader::Info& info,
                             const VideoCommon::RecordedEnvQueries& queries,
                             std::span<const u32> literals) {
    if (!info.storage_buffers_descriptors.empty() || !info.texture_buffer_descriptors.empty() ||
        !info.image_buffer_descriptors.empty() || !info.texture_descriptors.empty() ||
        !info.image_descriptors.empty() || !info.handle_programs.empty() ||
        info.uses_global_memory || info.stores_global_memory || info.uses_cbuf_indirect ||
        (info.constant_buffer_mask & ~(1U << 1)) ||
        !queries.texture_types.empty() || !queries.texture_pixel_formats.empty() ||
        !queries.location_texture_types.empty() || !queries.location_texture_formats.empty() ||
        !queries.location_integer_formats.empty() || !queries.storage_pointer_types.empty() ||
        !queries.cbuf_replacements.empty() || queries.has_hle_engine_state ||
        queries.has_unbound_instructions || queries.texture_bound != ~0U) return false;
    for (const auto& descriptor : info.constant_buffer_descriptors) {
        if (descriptor.index != 1 || descriptor.count != 1 ||
            info.constant_buffer_used_sizes[1] > literals.size_bytes()) return false;
    }
    for (const auto& [key, value] : queries.cbuf_values) {
        const u32 bank = static_cast<u32>(key >> 32);
        const u32 offset = static_cast<u32>(key);
        if (bank != 1 || offset % 4 || offset / 4 >= literals.size() ||
            literals[offset / 4] != value) return false;
    }
    return true;
}

} // Anonymous namespace

BananzaNvn2ShaderResult TranslateBananzaNvn2Shader(
    const Core::BananzaNvn2Hle::ResourceShaderInfo& shader, const Shader::Profile& profile,
    const Shader::HostTranslateInfo& host_info, const Shader::RuntimeInfo& runtime_info,
    Shader::Backend::Bindings& bindings, Shader::Environment* authoritative_bindings,
    const std::array<u32, 8>* geometry_skip_mask) {
    BananzaNvn2ShaderResult result;
    const bool compute = shader.stage == 5;
    const bool geometry = shader.stage == 3 && geometry_skip_mask;
    if (!compute && !geometry && shader.stage != 0 && shader.stage != 4) {
        result.status = BananzaNvn2ShaderStatus::UnsupportedStage;
        result.error = "Unsupported NVN 2 shader stage";
        return result;
    }
    result.stage = compute ? Shader::Stage::Compute : geometry ? Shader::Stage::Geometry
                           : shader.stage == 0 ? Shader::Stage::VertexB : Shader::Stage::Fragment;
    const u32 header_size = compute ? 0 : HeaderSize;
    if (shader.header_size != header_size || shader.code_size < header_size + 16 ||
        shader.code_size > MaxCodeSize - HeaderSize + header_size || shader.code_size % 16 ||
        shader.code_size > shader.code.size() || shader.code.size() > (16U << 20) ||
        shader.control.size() < (compute ? 0x70 : 0x48) || shader.control.size() > (64U << 10) ||
        shader.constant_size > (64U << 10) || shader.constant_size % 4 ||
        shader.constant_offset > shader.code.size() ||
        shader.constant_size > shader.code.size() - shader.constant_offset ||
        (shader.constant_size && shader.constant_offset < shader.code_size) ||
        runtime_info.mesh_as_compute) {
        result.error = "Invalid or oversized staged shader";
        return result;
    }

    if (compute) {
        result.workgroup_size = {ReadWord(shader.control, 0x58), ReadWord(shader.control, 0x5c),
                                 ReadWord(shader.control, 0x60)};
        const auto& group = result.workgroup_size;
        const u64 shared = (u64{ReadWord(shader.control, 0x64)} + 255) & ~u64{255};
        const u64 local = u64{ReadWord(shader.control, 0x68)} + ReadWord(shader.control, 0x6c);
        if (!group[0] || !group[1] || !group[2] || group[0] > 1024 || group[1] > 1024 ||
            group[2] > 64 || u64{group[0]} * group[1] * group[2] > 1024 ||
            shared > (128U << 10) || local > (1U << 20)) {
            result.error = "Invalid compute workgroup or memory size";
            return result;
        }
        // SDK 18265D: +0x60D1FC rounds the shared allocation to 256-byte units.
        result.shared_memory_size = static_cast<u32>(shared);
        result.local_memory_size = static_cast<u32>(local);
    }
    const u32 header = ReadWord(shader.code, 0);
    const u32 expected_type = shader.stage == 4 ? 2 : 1;
    const u32 expected_stage = shader.stage + 1;
    if ((compute ? shader.control[0x56] != 5
                 : ((header & 31) != expected_type || ((header >> 5) & 31) != 4 ||
                    ((header >> 10) & 15) != expected_stage)) ||
        ReadWord(shader.control, 0) != 0x98761234 || ReadWord(shader.control, 4) != 201 ||
        ReadWord(shader.control, 0x38) != shader.code_size ||
        ReadWord(shader.control, 0x40) != shader.constant_offset ||
        ReadWord(shader.control, 0x44) != shader.constant_offset + shader.constant_size) {
        result.error = "Staged control, SPH and shader metadata disagree";
        return result;
    }

    try {
        result.status = BananzaNvn2ShaderStatus::LoweringFailed;
        const auto code = std::span{shader.code}.first(shader.code_size);
        if (!HasCompleteCode(code, header_size)) {
            result.error = "Native program has no complete bounded capture";
            return result;
        }
        std::vector<u8> compute_code;
        std::span<const u8> native = code;
        if (compute) {
            compute_code.resize(HeaderSize);
            compute_code.insert(compute_code.end(), code.begin(), code.end());
            native = compute_code;
        }
        const auto lowered = VideoCommon::LowerNativeProgramBytes(
            native, result.stage, geometry ? shader.data_address + header_size : 0);
        if (!lowered || lowered->code.size() < sizeof(Shader::ProgramHeader) / sizeof(u64) + 1 ||
            lowered->code.size() > MaxLoweredWords || lowered->source_size > native.size() ||
            lowered->source_size < HeaderSize + 16) {
            result.error = "Native lowering failed or exceeded limits";
            return result;
        }
        result.native_hash = lowered->native_program_key;
        result.lowered_hash = VideoCommon::NativeProgramHash(lowered->code);
        result.source_size = static_cast<u32>(lowered->source_size) - (compute ? HeaderSize : 0);
        result.texture_bound = lowered->texture_bound != ~0U ? lowered->texture_bound :
            authoritative_bindings ? authoritative_bindings->TextureBoundBuffer() :
            shader.stage == 4 ? 2 : 0;

        std::vector<u32> literals(shader.constant_size / sizeof(u32));
        if (!literals.empty()) {
            std::memcpy(literals.data(), shader.code.data() + shader.constant_offset,
                        shader.constant_size);
        }
        Shader::ProgramHeader sph{};
        static_assert(sizeof(sph) <= HeaderSize);
        if (!compute) {
            std::memcpy(&sph, shader.code.data(), sizeof(sph));
        }
        VideoCommon::FileEnvironment environment;
        environment.InitializeFromArchive(lowered->code, sph, result.stage,
                                          result.texture_bound, literals);
        if (compute) {
            environment.SetComputeState(result.workgroup_size, result.shared_memory_size,
                                        result.local_memory_size);
        }
        if (geometry) {
            environment.SetGeometryPassthroughMask(*geometry_skip_mask);
        }
        if (authoritative_bindings) {
            environment.SetArchiveQuerySource(*authoritative_bindings);
        }
        ShaderPools pools;
        result.status = BananzaNvn2ShaderStatus::TranslationFailed;
        Shader::Maxwell::Flow::CFG cfg(environment, pools.flow_block,
                                      compute ? environment.StartAddress()
                                              : environment.StartAddress() + sizeof(Shader::ProgramHeader),
                                      false);
        auto program = Shader::Maxwell::TranslateProgram(pools.inst, pools.block, environment,
                                                        cfg, host_info);
        if (!compute) {
            Shader::Maxwell::ConvertLegacyToGeneric(program, runtime_info);
        }
        result.queries = environment.RecordQueries();
        result.info = std::make_unique<Shader::Info>(program.info);
        if (authoritative_bindings) {
            result.status = BananzaNvn2ShaderStatus::BindingMismatch;
            const char* reason = nullptr;
            if (!environment.AnswersHoldFor(*authoritative_bindings, &reason)) {
                result.error = reason ? reason : "Recorded binding answers differ";
                return result;
            }
            result.resource_authority = BananzaNvn2ResourceAuthority::Validated;
        } else if (HasIndependentResources(program.info, result.queries, literals)) {
            result.resource_authority = BananzaNvn2ResourceAuthority::ResourceIndependent;
        }

        result.status = BananzaNvn2ShaderStatus::EmissionFailed;
        auto next_bindings = bindings;
        auto spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, next_bindings);
        if (spirv.size() < 5 || spirv.size() > MaxSpirvWords || spirv[0] != 0x07230203) {
            result.error = "Invalid or oversized emitted SPIR-V";
            return result;
        }
        *result.info = program.info;
        result.spirv = std::move(spirv);
        bindings = next_bindings;
        result.status = BananzaNvn2ShaderStatus::Success;
    } catch (const std::exception& exception) {
        result.error = exception.what();
    }
    return result;
}

vk::Pipeline CreateBananzaNvn2ComputePipeline(const Device& device,
                                             const BananzaNvn2ShaderResult& shader,
                                             VkPipelineLayout layout, VkPipelineCache cache) {
    if (shader.status != BananzaNvn2ShaderStatus::Success || shader.stage != Shader::Stage::Compute ||
        !shader.info || !layout || shader.spirv.size() < 5 ||
        shader.spirv.size() > MaxSpirvWords || shader.spirv[0] != 0x07230203) {
        throw std::invalid_argument{"Invalid Bananza compute shader or pipeline layout"};
    }
    const auto limits = device.GetPhysical().GetProperties().limits;
    const auto& group = shader.workgroup_size;
    u64 invocations = 1;
    for (size_t i = 0; i < group.size(); ++i) {
        if (!group[i] || group[i] > limits.maxComputeWorkGroupSize[i] ||
            group[i] > limits.maxComputeWorkGroupInvocations / invocations) {
            throw std::invalid_argument{"Bananza compute workgroup exceeds host limits"};
        }
        invocations *= group[i];
    }
    if (shader.shared_memory_size > limits.maxComputeSharedMemorySize) {
        throw std::invalid_argument{"Bananza compute shared memory exceeds host limits"};
    }
    const auto& info = *shader.info;
    const bool uses_warp_ops = info.uses_subgroup_vote || info.uses_subgroup_invocation_id ||
                               info.uses_subgroup_shuffles || info.uses_subgroup_mask ||
                               info.uses_quad_shuffles || info.uses_fswzadd ||
                               info.uses_cooperative_matrix;
    const VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
        .requiredSubgroupSize = GuestWarpSize,
    };
    const auto& logical = device.GetLogical();
    const auto module = logical.CreateShaderModule({
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = shader.spirv.size() * sizeof(u32),
        .pCode = shader.spirv.data(),
    });
    return logical.CreateComputePipeline({
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = uses_warp_ops && device.IsExtSubgroupSizeControlSupported() &&
                             device.IsGuestWarpSizeSupported(VK_SHADER_STAGE_COMPUTE_BIT)
                         ? &subgroup : nullptr,
            .flags = info.uses_cooperative_matrix
                         ? VkPipelineShaderStageCreateFlags{
                               VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT}
                         : VkPipelineShaderStageCreateFlags{},
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = *module,
            .pName = "main",
        },
        .layout = layout,
        .basePipelineIndex = -1,
    }, cache);
}

} // namespace Vulkan
