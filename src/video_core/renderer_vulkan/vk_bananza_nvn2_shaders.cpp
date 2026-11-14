// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <exception>
#include <span>

#include "core/arm/bananza_nvn2_resources.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "video_core/renderer_vulkan/vk_bananza_nvn2_shaders.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

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

bool HasCompleteCode(std::span<const u8> bytes) {
    Shader::Ampere::ProgramCaptureState capture;
    for (size_t offset = HeaderSize; offset < bytes.size(); offset += 16) {
        Shader::Ampere::Instruction instruction;
        std::memcpy(&instruction, bytes.data() + offset, sizeof(instruction));
        const auto status = capture.Observe(instruction, static_cast<u32>((offset - HeaderSize) / 16));
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
    Shader::Backend::Bindings& bindings, Shader::Environment* authoritative_bindings) {
    BananzaNvn2ShaderResult result;
    if (shader.stage != 0 && shader.stage != 4) {
        result.status = BananzaNvn2ShaderStatus::UnsupportedStage;
        result.error = "Only VertexB and Fragment are supported";
        return result;
    }
    result.stage = shader.stage == 0 ? Shader::Stage::VertexB : Shader::Stage::Fragment;
    if (shader.header_size != HeaderSize || shader.code_size < HeaderSize + 16 ||
        shader.code_size > MaxCodeSize || shader.code_size % 16 ||
        shader.code_size > shader.code.size() || shader.code.size() > (16U << 20) ||
        shader.control.size() < 0x48 || shader.control.size() > (64U << 10) ||
        shader.constant_size > (64U << 10) || shader.constant_size % 4 ||
        shader.constant_offset > shader.code.size() ||
        shader.constant_size > shader.code.size() - shader.constant_offset ||
        (shader.constant_size && shader.constant_offset < shader.code_size) ||
        runtime_info.mesh_as_compute) {
        result.error = "Invalid or oversized staged graphics shader";
        return result;
    }
    const u32 header = ReadWord(shader.code, 0);
    const u32 expected_type = shader.stage == 4 ? 2 : 1;
    const u32 expected_stage = shader.stage == 4 ? 5 : 1;
    if ((header & 31) != expected_type || ((header >> 5) & 31) != 4 ||
        ((header >> 10) & 15) != expected_stage ||
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
        if (!HasCompleteCode(code)) {
            result.error = "Native program has no complete bounded capture";
            return result;
        }
        const auto lowered = VideoCommon::LowerNativeProgramBytes(code, result.stage);
        if (!lowered || lowered->code.size() < sizeof(Shader::ProgramHeader) / sizeof(u64) + 1 ||
            lowered->code.size() > MaxLoweredWords || lowered->source_size > code.size()) {
            result.error = "Native lowering failed or exceeded limits";
            return result;
        }
        result.native_hash = lowered->native_program_key;
        result.lowered_hash = VideoCommon::NativeProgramHash(lowered->code);
        result.source_size = static_cast<u32>(lowered->source_size);
        result.texture_bound = lowered->texture_bound != ~0U ? lowered->texture_bound :
            authoritative_bindings ? authoritative_bindings->TextureBoundBuffer() :
            shader.stage == 0 ? 0 : 2;

        std::vector<u32> literals(shader.constant_size / sizeof(u32));
        if (!literals.empty()) {
            std::memcpy(literals.data(), shader.code.data() + shader.constant_offset,
                        shader.constant_size);
        }
        Shader::ProgramHeader sph;
        static_assert(sizeof(sph) <= HeaderSize);
        std::memcpy(&sph, shader.code.data(), sizeof(sph));
        VideoCommon::FileEnvironment environment;
        environment.InitializeFromArchive(lowered->code, sph, result.stage,
                                          result.texture_bound, literals);
        ShaderPools pools;
        result.status = BananzaNvn2ShaderStatus::TranslationFailed;
        Shader::Maxwell::Flow::CFG cfg(environment, pools.flow_block,
                                      environment.StartAddress() + sizeof(Shader::ProgramHeader), false);
        auto program = Shader::Maxwell::TranslateProgram(pools.inst, pools.block, environment,
                                                        cfg, host_info);
        Shader::Maxwell::ConvertLegacyToGeneric(program, runtime_info);
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

} // namespace Vulkan
