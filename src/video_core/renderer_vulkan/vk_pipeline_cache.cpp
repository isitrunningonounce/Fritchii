// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <unordered_map>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <unordered_set>
#include <span>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>
#include <bit>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <numeric>
#include <streambuf>
#include "common/cityhash.h"
#include "common/polyfill_thread.h"
#include "common/scm_rev.h"
#include "common/scope_exit.h"
#include "common/fs/fs.h"
#include "common/fs/fs_util.h"
#include "common/fs/path_util.h"
#include "common/host_memory.h"
#include "common/memory_detect.h"
#include "common/shutdown_progress.h"
#include "common/thread.h"
#include "common/thread_worker.h"
#include "core/core.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/info_serialize.h"
#include "shader_recompiler/translate_profile.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/ir_opt/oot_visual_workaround.h"
#include "shader_recompiler/program_header.h"
#include "video_core/async_translate.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_precompile.h"
#include "video_core/pipeline_cache_files.h"
#include "video_core/pipeline_cache_policy.h"
#include "video_core/renderer_vulkan/async_draw_policy.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/mesh_fallback_limits.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/lichii_dynamic_rendering.h"
#include "video_core/renderer_vulkan/lichii_shader_object_cache.h"
#include "video_core/renderer_vulkan/pipeline_shader_emission.h"
#include "video_core/renderer_vulkan/pipeline_statistics.h"
#include "video_core/renderer_vulkan/vertex_repack.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_cooperative_hmma_test.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/texture_cache/formatter.h"
#include "video_core/renderer_vulkan/vk_pipeline_translate_queue.h"
#include "video_core/renderer_vulkan/vk_precompiled_kept.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_cache.h"
#include "video_core/shader_environment.h"
#include "video_core/renderer_vulkan/lichii_precompile_io.h"
#include "common/zstd_compression.h"
#include "video_core/shader_notify.h"
#include "video_core/shader_warm/shader_warm_builder.h"
#include "video_core/shader_warm/warm_records.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/gpu_logging/gpu_logging.h"
#ifdef _WIN32
#include <windows.h> // already in through vulkan.h, for the disk cache reader
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef __ANDROID__
#include "../../android/app/src/main/jni/android_settings.h"
#endif

// Xlib.h (via the Vulkan XLIB platform headers) defines Status as int, which eats the
// PipelineCacheMergeResult::Status member below. Nothing here uses the X11 type.
#undef Status

namespace Vulkan {

namespace {
using Shader::Backend::SPIRV::EmitSPIRV;
using Shader::Maxwell::ConvertLegacyToGeneric;
using Shader::Maxwell::GenerateGeometryPassthrough;
using Shader::Maxwell::MergeDualVertexPrograms;
using Shader::Maxwell::TranslateProgram;
using VideoCommon::ComputeEnvironment;
using VideoCommon::FileEnvironment;
using VideoCommon::GenericEnvironment;
using VideoCommon::GuessMissing;
using VideoCommon::GraphicsEnvironment;

constexpr u32 CACHE_VERSION = VideoCommon::VulkanPipelineCacheVersion;
// The warm-up builder writes these keys byte for byte (shader_warm/warm_records.h).
static_assert(sizeof(GraphicsPipelineCacheKey) == VideoCommon::ShaderWarm::GraphicsKeySize);
static_assert(sizeof(ComputePipelineCacheKey) == sizeof(VideoCommon::ShaderWarm::ComputeKey));
constexpr size_t VULKAN_CACHE_FLUSH_PIPELINES = 128;
constexpr size_t VULKAN_CACHE_FLUSH_MIN_SECONDS = 30;
constexpr std::array<char, 8> VULKAN_CACHE_MAGIC_NUMBER{'y', 'u', 'z', 'u', 'v', 'k', 'c', 'h'};

template <typename Container>
auto MakeSpan(Container& container) {
    return std::span(container.data(), container.size());
}

Shader::OutputTopology MaxwellToOutputTopology(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        return Shader::OutputTopology::PointList;
    case Maxwell::PrimitiveTopology::LineStrip:
        return Shader::OutputTopology::LineStrip;
    default:
        return Shader::OutputTopology::TriangleStrip;
    }
}

Shader::CompareFunction MaxwellToCompareFunction(Maxwell::ComparisonOp comparison) {
    switch (comparison) {
    case Maxwell::ComparisonOp::Never_D3D:
    case Maxwell::ComparisonOp::Never_GL:
        return Shader::CompareFunction::Never;
    case Maxwell::ComparisonOp::Less_D3D:
    case Maxwell::ComparisonOp::Less_GL:
        return Shader::CompareFunction::Less;
    case Maxwell::ComparisonOp::Equal_D3D:
    case Maxwell::ComparisonOp::Equal_GL:
        return Shader::CompareFunction::Equal;
    case Maxwell::ComparisonOp::LessEqual_D3D:
    case Maxwell::ComparisonOp::LessEqual_GL:
        return Shader::CompareFunction::LessThanEqual;
    case Maxwell::ComparisonOp::Greater_D3D:
    case Maxwell::ComparisonOp::Greater_GL:
        return Shader::CompareFunction::Greater;
    case Maxwell::ComparisonOp::NotEqual_D3D:
    case Maxwell::ComparisonOp::NotEqual_GL:
        return Shader::CompareFunction::NotEqual;
    case Maxwell::ComparisonOp::GreaterEqual_D3D:
    case Maxwell::ComparisonOp::GreaterEqual_GL:
        return Shader::CompareFunction::GreaterThanEqual;
    case Maxwell::ComparisonOp::Always_D3D:
    case Maxwell::ComparisonOp::Always_GL:
        return Shader::CompareFunction::Always;
    }
    UNIMPLEMENTED_MSG("Unimplemented comparison op={}", comparison);
    return {};
}

Shader::AttributeType CastAttributeType(const FixedPipelineState::VertexAttribute& attr) {
    if (attr.enabled == 0) {
        return Shader::AttributeType::Disabled;
    }
    switch (attr.Type()) {
    case Maxwell::VertexAttribute::Type::UnusedEnumDoNotUseBecauseItWillGoAway:
        ASSERT_MSG(false, "Invalid vertex attribute type!");
        return Shader::AttributeType::Disabled;
    case Maxwell::VertexAttribute::Type::SNorm:
    case Maxwell::VertexAttribute::Type::UNorm:
    case Maxwell::VertexAttribute::Type::Float:
        return Shader::AttributeType::Float;
    case Maxwell::VertexAttribute::Type::SInt:
        return Shader::AttributeType::SignedInt;
    case Maxwell::VertexAttribute::Type::UInt:
        return Shader::AttributeType::UnsignedInt;
    case Maxwell::VertexAttribute::Type::UScaled:
        return Shader::AttributeType::UnsignedScaled;
    case Maxwell::VertexAttribute::Type::SScaled:
        return Shader::AttributeType::SignedScaled;
    }
    return Shader::AttributeType::Float;
}

Shader::AttributeType AttributeType(const FixedPipelineState& state, size_t index) {
    switch (state.DynamicAttributeType(index)) {
    case 0:
        return Shader::AttributeType::Disabled;
    case 1:
        return Shader::AttributeType::Float;
    case 2:
        return Shader::AttributeType::SignedInt;
    case 3:
        return Shader::AttributeType::UnsignedInt;
    }
    return Shader::AttributeType::Disabled;
}

Shader::RuntimeInfo MakeRuntimeInfo(std::span<const Shader::IR::Program> programs,
                                    const GraphicsPipelineCacheKey& key,
                                    const Shader::IR::Program& program,
                                    const Shader::IR::Program* previous_program,
                                    const Vulkan::Device& device) {
    Shader::RuntimeInfo info;
    if (previous_program) {
        info.previous_stage_stores = previous_program->info.stores;
        info.previous_stage_legacy_stores_mapping = previous_program->info.legacy_stores_mapping;
        if (previous_program->is_geometry_passthrough) {
            info.previous_stage_stores.mask |= previous_program->info.passthrough.mask;
        }
    } else {
        info.previous_stage_stores.mask.set();
    }
    if (previous_program && previous_program->stage == Shader::Stage::Mesh) {
        for (const u32 line : previous_program->mesh.primitive_lines) {
            const auto attribute{static_cast<Shader::IR::Attribute>(line)};
            if (Shader::IR::IsGeneric(attribute)) {
                info.per_primitive_generics.set(Shader::IR::GenericAttributeIndex(attribute));
            }
        }
    }
    const Shader::Stage stage{program.stage};
    const bool has_geometry{key.unique_hashes[4] != 0 && !programs[4].is_geometry_passthrough};
    const bool gl_ndc{key.state.ndc_minus_one_to_one != 0};
    const float point_size{std::bit_cast<float>(key.state.point_size)};
    switch (stage) {
    case Shader::Stage::VertexB:
        if (!has_geometry) {
            if (key.state.topology == Maxwell::PrimitiveTopology::Points) {
                info.fixed_state_point_size = point_size;
            }
            if (key.state.xfb_enabled) {
                if (device.IsExtTransformFeedbackSupported()) {
                    auto [varyings, count] =
                        VideoCommon::MakeTransformFeedbackVaryings(key.state.xfb_state);
                    info.xfb_varyings = varyings;
                    info.xfb_count = count;
                } else {
                    LOG_WARNING(Render_Vulkan, "XFB requested in pipeline key but device lacks VK_EXT_transform_feedback; ignoring XFB decorations");
                }
            }
            info.convert_depth_mode = gl_ndc;
        }
        if (key.state.dynamic_vertex_input) {
            for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
                info.generic_input_types[index] = AttributeType(key.state, index);
            }
        } else {
            std::ranges::transform(key.state.attributes, info.generic_input_types.begin(),
                                   &CastAttributeType);
        }
        break;
    case Shader::Stage::TessellationEval:
        info.tess_clockwise = key.state.tessellation_clockwise != 0;
        info.tess_primitive = [&key] {
            const u32 raw{key.state.tessellation_primitive.Value()};
            switch (static_cast<Maxwell::Tessellation::DomainType>(raw)) {
            case Maxwell::Tessellation::DomainType::Isolines:
                return Shader::TessPrimitive::Isolines;
            case Maxwell::Tessellation::DomainType::Triangles:
                return Shader::TessPrimitive::Triangles;
            case Maxwell::Tessellation::DomainType::Quads:
                return Shader::TessPrimitive::Quads;
            }
            ASSERT(false);
            return Shader::TessPrimitive::Triangles;
        }();
        info.tess_spacing = [&] {
            const u32 raw{key.state.tessellation_spacing};
            switch (static_cast<Maxwell::Tessellation::Spacing>(raw)) {
            case Maxwell::Tessellation::Spacing::Integer:
                return Shader::TessSpacing::Equal;
            case Maxwell::Tessellation::Spacing::FractionalOdd:
                return Shader::TessSpacing::FractionalOdd;
            case Maxwell::Tessellation::Spacing::FractionalEven:
                return Shader::TessSpacing::FractionalEven;
            }
            ASSERT(false);
            return Shader::TessSpacing::Equal;
        }();
        break;
    case Shader::Stage::Mesh:
        info.convert_depth_mode = gl_ndc;
        if (program.mesh.topology == 0) {
            info.fixed_state_point_size = point_size;
        }
        break;
    case Shader::Stage::Geometry:
        if (program.output_topology == Shader::OutputTopology::PointList) {
            info.fixed_state_point_size = point_size;
        }
        if (key.state.xfb_enabled != 0) {
            if (device.IsExtTransformFeedbackSupported()) {
                auto [varyings, count] =
                    VideoCommon::MakeTransformFeedbackVaryings(key.state.xfb_state);
                info.xfb_varyings = varyings;
                info.xfb_count = count;
            } else {
                LOG_WARNING(Render_Vulkan, "XFB requested in pipeline key but device lacks VK_EXT_transform_feedback; ignoring XFB decorations");
            }
        }
        info.convert_depth_mode = gl_ndc;
        break;
    case Shader::Stage::Fragment: {
        info.alpha_test_func = MaxwellToCompareFunction(
            key.state.UnpackComparisonOp(key.state.alpha_test_func.Value()));
        info.alpha_test_reference = std::bit_cast<float>(key.state.alpha_test_ref);

        info.dual_source_blend = key.state.attachment0_dual_source_blend != 0;

        if (device.IsMoltenVK()) {
            for (size_t i = 0; i < 8; ++i) {
                const auto format = static_cast<Tegra::RenderTargetFormat>(key.state.color_formats[i]);
                if (format == Tegra::RenderTargetFormat::NONE) {
                    info.color_output_types[i] = Shader::AttributeType::Float;
                    continue;
                }
                const auto pixel_format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(format);
                if (VideoCore::Surface::IsPixelFormatInteger(pixel_format)) {
                    if (VideoCore::Surface::IsPixelFormatSignedInteger(pixel_format)) {
                        info.color_output_types[i] = Shader::AttributeType::SignedInt;
                    } else {
                        info.color_output_types[i] = Shader::AttributeType::UnsignedInt;
                    }
                } else {
                    info.color_output_types[i] = Shader::AttributeType::Float;
                }
            }
        }
        break;
    }
    default:
        break;
    }
    switch (key.state.topology) {
    case Maxwell::PrimitiveTopology::Points:
        info.input_topology = Shader::InputTopology::Points;
        break;
    case Maxwell::PrimitiveTopology::Lines:
    case Maxwell::PrimitiveTopology::LineLoop:
    case Maxwell::PrimitiveTopology::LineStrip:
        info.input_topology = Shader::InputTopology::Lines;
        break;
    case Maxwell::PrimitiveTopology::Triangles:
    case Maxwell::PrimitiveTopology::TriangleStrip:
    case Maxwell::PrimitiveTopology::TriangleFan:
    case Maxwell::PrimitiveTopology::Quads:
    case Maxwell::PrimitiveTopology::QuadStrip:
    case Maxwell::PrimitiveTopology::Polygon:
    case Maxwell::PrimitiveTopology::Patches:
        info.input_topology = Shader::InputTopology::Triangles;
        break;
    case Maxwell::PrimitiveTopology::LinesAdjacency:
    case Maxwell::PrimitiveTopology::LineStripAdjacency:
        info.input_topology = Shader::InputTopology::LinesAdjacency;
        break;
    case Maxwell::PrimitiveTopology::TrianglesAdjacency:
    case Maxwell::PrimitiveTopology::TriangleStripAdjacency:
        info.input_topology = Shader::InputTopology::TrianglesAdjacency;
        break;
    }
    info.force_early_z = key.state.early_z != 0;
    info.y_negate = key.state.y_negate != 0;
    return info;
}

std::array<std::atomic<std::chrono::nanoseconds::rep>, 2> runtime_spirv_ns{};
/// Set while a translation thread builds (EDEN_NX2_ASYNC_TRANSLATE).
thread_local bool t_translate_worker{};

// Draws skipped because their pipeline was still building, reported with the runtime totals.
std::atomic<u32> discarded_pending_draws{};
// Skipped and deferred draws since the presenter last asked.
std::atomic<u32> skipped_since_frame{};
// Draws that could be skipped but whose pipeline finished within the frame's wait budget.
std::atomic<u32> waited_pending_draws{};
std::atomic<std::chrono::nanoseconds::rep> pending_wait_ns{};
// What is left of this frame's wait budget (LICHII_DISCARD_WAIT_MS), refilled every frame.
std::atomic<std::chrono::nanoseconds::rep> frame_wait_left_ns{};

// LICHII_SPIRV_REUSE: pipelines built from an earlier translation, and reuse attempts whose
// environments answered differently.
std::atomic<u32> reused_pipelines{};
std::atomic<u32> reuse_rejected{};
std::atomic<u32> reuse_state_misses{};

bool SpirvReuseEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("LICHII_SPIRV_REUSE");
        const bool on = value != nullptr && *value != '\0' && *value != '0';
        LOG_INFO(Render_Vulkan, "LICHII_SPIRV_REUSE: {}", on ? "on" : "off");
        return on;
    }();
    return enabled;
}

// New pipelines deferred to a later frame, and what is left of this frame's translation budget.
std::atomic<u32> deferred_new_pipelines{};
/// Set by the archive precompiler around CreateGraphicsPipeline: translate and keep for reuse only.
thread_local bool t_translate_only{};
/// Set by the archive precompiler around CreateComputePipeline: receives the SPIR-V and info.
thread_local std::pair<std::vector<u32>, Shader::Info>* t_capture_compute{};
/// Graphics pipelines built at runtime: mesh (never stored), in the disk cache, new.
std::array<std::atomic<u32>, 3> runtime_built_kind{};
std::atomic<std::chrono::nanoseconds::rep> frame_translate_left_ns{};

std::chrono::nanoseconds GetTranslateBudget() {
    static const std::chrono::nanoseconds budget = [] {
        const char* value = std::getenv("LICHII_TRANSLATE_BUDGET_MS");
        const int ms = std::clamp(value ? std::atoi(value) : 0, 0, 50);
        LOG_INFO(Render_Vulkan,
                 "LICHII_TRANSLATE_BUDGET_MS: new pipelines for skippable draws are translated up "
                 "to {} ms per frame{}",
                 ms, ms == 0 ? " (off)" : "");
        return std::chrono::nanoseconds{std::chrono::milliseconds{ms}};
    }();
    return budget;
}

std::chrono::nanoseconds GetDiscardWaitBudget() {
    static const std::chrono::nanoseconds budget = [] {
        const char* value = std::getenv("LICHII_DISCARD_WAIT_MS");
        int ms = value ? std::atoi(value) : 0;
        ms = std::clamp(ms, 0, 50);
        LOG_INFO(Render_Vulkan,
                 "LICHII_DISCARD_WAIT_MS: a skippable draw waits for its pipeline up to {} ms "
                 "per frame",
                 ms);
        return std::chrono::nanoseconds{std::chrono::milliseconds{ms}};
    }();
    return budget;
}

NativeDiscardMode GetNativeDiscardMode() {
    static const NativeDiscardMode mode = [] {
        const char* value = std::getenv("LICHII_NATIVE_DISCARD");
        const std::string_view name = value ? value : "";
        NativeDiscardMode result = NativeDiscardMode::Legacy;
        if (name == "depth") {
            result = NativeDiscardMode::Depth;
        } else if (name == "never") {
            result = NativeDiscardMode::Never;
        } else if (name == "legacy") {
            result = NativeDiscardMode::Legacy;
        } else if (name == "always") {
            result = NativeDiscardMode::Always;
        } else if (!name.empty() && name != "legacy") {
            LOG_WARNING(Render_Vulkan, "LICHII_NATIVE_DISCARD={} is unknown; use never, depth, "
                                       "legacy or always",
                        name);
        }
        static constexpr std::array names{"never", "depth", "legacy", "always"};
        LOG_INFO(Render_Vulkan, "LICHII_NATIVE_DISCARD: native draws with unbuilt pipelines: {}",
                 names[static_cast<size_t>(result)]);
        return result;
    }();
    return mode;
}

void AddRuntimeSpirvTime(size_t thread, std::chrono::steady_clock::time_point begin) {
    if (t_translate_worker) {
        // SPIR-V a translation thread emits itself is the workers' time, not the GPU thread's.
        thread = 1;
    }
    const auto elapsed = std::chrono::steady_clock::now() - begin;
    runtime_spirv_ns[thread].fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count(),
        std::memory_order::relaxed);
}

// The GPU thread's split of its own runtime graphics builds: environments (GetGraphicsEnvironments,
// a native program's first lowering included) and translation (control flow, TranslateProgram and
// the IR passes, with the environment's lazy reads). Written and read on the GPU thread only.
struct RuntimeBuildSplit {
    std::chrono::steady_clock::duration environments{};
    std::chrono::steady_clock::duration translation{};
};
thread_local bool t_runtime_build{};
thread_local RuntimeBuildSplit t_build_split{};

// What the GPU thread spent on pipelines in the current frame, for the worst frame line
// (ReportWorstFrame). GPU thread only.
struct FrameTally {
    std::chrono::steady_clock::duration builds{};
    RuntimeBuildSplit split{};
    u32 count{};
    /// EDEN_NX2_ASYNC_TRANSLATE: translations queued (their environments and snapshots).
    std::chrono::steady_clock::duration queued_time{};
    u32 queued{};
};
FrameTally frame_tally;
/// Pipeline lookups skipped for unbuilt pipelines this session (BuiltPipeline), and for
/// pipelines still translating (EDEN_NX2_ASYNC_TRANSLATE).
std::atomic<u64> skipped_lookups{};

void NoteSkippedLookup() {
    if ((skipped_lookups.fetch_add(1, std::memory_order_relaxed) & 255) == 0) {
        // Counts lookups, not draws: a draw state can look its pipeline up more than once.
        LOG_INFO(Render_Vulkan, "Async shaders: {} pipeline lookups skipped for unbuilt pipelines",
                 skipped_lookups.load(std::memory_order_relaxed));
    }
}

/// EDEN_NX2_ASYNC_TRANSLATE: translations in flight at most (each holds its constant buffer
/// copies until translated), how long a draw that must not be skipped waits for one before
/// building its pipeline itself, unusable translations of a key before it is built on the GPU
/// thread for good, and the age at which a translation is taken for wedged.
constexpr size_t MaxTranslationsInFlight = 64;
constexpr std::chrono::milliseconds TranslateWaitCap{50};
constexpr u32 MaxTranslateFailures = 2;
constexpr std::chrono::seconds TranslateSweepAge{5};

/// EDEN_NX2_ASYNC_TRANSLATE_THREADS: translation threads (2 by default, 1 to 8).
size_t TranslateThreads() {
    const char* const value = std::getenv("EDEN_NX2_ASYNC_TRANSLATE_THREADS");
    const int threads = value != nullptr ? std::atoi(value) : 2;
    return static_cast<size_t>(std::clamp(threads, 1, 8));
}

/// EDEN_NX2_SPIRV_SELFCHECK=1: throws (the pipeline is dropped, as one the translation rejects)
/// when the emitted SPIR-V uses an id it never defines or its own result, before the driver gets
/// it. Off: nothing.
void SelfCheckSpirv(std::span<const u32> code, u64 shader_hash, const char* stage) {
    if (!SpirvSelfCheck()) {
        return;
    }
    if (std::string problem = FindSpirvIdProblem(code); !problem.empty()) {
        LOG_ERROR(Render_Vulkan,
                  "SPIR-V self-check (EDEN_NX2_SPIRV_SELFCHECK=1): shader {:016x} {}: {}; its "
                  "pipeline is dropped instead of handed to the driver",
                  shader_hash, stage, problem);
        throw Shader::LogicError("SPIR-V self-check failed for shader {:016x} {}", shader_hash,
                                 stage);
    }
}

/// The shader module of a stage from its finished SPIR-V.
vk::ShaderModule BuildStageModule(const Device& device, std::span<const u32> code,
                                  u64 shader_hash, size_t stage_index, u128* spirv_hash) {
    static constexpr std::array stage_names{"vertex", "tess_control", "tess_eval", "geometry",
                                           "fragment"};
    SelfCheckSpirv(code, shader_hash, stage_names[stage_index]);
    if (spirv_hash) {
        // Graphics pipeline libraries are keyed by the SPIR-V the stage was emitted to.
        *spirv_hash = Common::CityHash128(reinterpret_cast<const char*>(code.data()),
                                          code.size() * sizeof(u32));
    }
    device.SaveShader(code);
    vk::ShaderModule module{BuildShader(device, code)};

    // Text log + .spv dump. Text log is gated by gpu_log_level != Off; .spv dump
    // is independent and gated only by gpu_log_shader_dumps.
    const bool should_log = GPU::Logging::IsActive();
    const bool should_dump = Settings::values.gpu_log_shader_dumps.GetValue();
    if (should_log || should_dump) {
        const std::string shader_name =
            fmt::format("shader_{:016x}_{}", shader_hash, stage_names[stage_index]);
        if (should_log) {
            const std::string shader_info = fmt::format(
                "SPIR-V size: {} bytes, hash: {:016x}", code.size() * sizeof(u32), shader_hash);
            GPU::Logging::GPULogger::GetInstance().LogShaderCompilation(shader_name, shader_info);
        }
        if (should_dump) {
            GPU::Logging::DumpSpirvShader(shader_hash,
                                         std::span<const u32>(code.data(), code.size()));
        }
    }

    if (device.HasDebuggingToolAttached()) {
        const std::string name{fmt::format("Shader {:016x}", shader_hash)};
        module.SetObjectNameEXT(name.c_str());
    }
    return module;
}

vk::ShaderModule EmitShaderModule(const Device& device, const Shader::Profile& profile,
                                 const Shader::RuntimeInfo& runtime_info,
                                 Shader::IR::Program& program, Shader::Backend::Bindings& binding,
                                 u64 shader_hash, size_t stage_index,
                                 std::span<const u32> reference = {},
                                 u128* spirv_hash = nullptr,
                                 std::vector<u32>* code_out = nullptr) {
    const auto code =
        EmitPipelineShader(profile, runtime_info, program, binding, shader_hash, reference);
    if (code_out) {
        // LICHII_SPIRV_REUSE: the SPIR-V is kept with the translation.
        *code_out = code;
    }
    return BuildStageModule(device, code, shader_hash, stage_index, spirv_hash);
}

// Graphics pipeline libraries (EDEN_NX2_GPL): each stage's resources go to its own descriptor
// set (set index = stage index) with bindings from 0, and the fragment stage's rescaling bits
// start at fixed indices, so a stage's SPIR-V does not depend on the other stages' resources.
void PrepareStageBindings(Shader::Backend::Bindings& binding, size_t stage_index, bool libraries) {
    if (!libraries) {
        return;
    }
    binding.unified = 0;
    binding.uniform_buffer = 0;
    binding.storage_buffer = 0;
    binding.texture = 0;
    binding.image = 0;
    binding.descriptor_set = static_cast<u32>(stage_index);
    if (stage_index + 1 == Maxwell::MaxShaderStage) {
        binding.texture_scaling_index = GPL_FRAGMENT_TEXTURE_SCALING_BASE;
        binding.image_scaling_index = GPL_FRAGMENT_IMAGE_SCALING_BASE;
    }
}

struct DeferredStages {
    std::unique_ptr<EmissionBudget::Reservation> reservation;
    std::shared_ptr<ShaderPools> pools;
    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
    std::array<std::optional<Shader::RuntimeInfo>, Maxwell::MaxShaderStage> runtime_infos;
    std::array<std::vector<u32>, Maxwell::MaxShaderStage> reference_spirv;
    /// SPIR-V emitted already (EDEN_NX2_ASYNC_TRANSLATE=2 records it on the translating thread):
    /// such a stage is built from these words, never emitted from its program a second time.
    std::array<std::vector<u32>, Maxwell::MaxShaderStage> emitted_spirv;
};

size_t GetTotalPipelineWorkers() {
    const size_t max_core_threads =
        std::max<size_t>(static_cast<size_t>(std::thread::hardware_concurrency()), 2ULL) - 1ULL;
#ifdef __ANDROID__
    const int configured = AndroidSettings::values.pipeline_worker_count.GetValue();
    const int clamped = std::clamp(configured, 2, 8);
    const size_t desired = static_cast<size_t>(clamped);
    if (desired == 0) {
        return 1ULL;
    }
    return std::min(max_core_threads, desired);
#else
    return max_core_threads;
#endif
}

/// How many of a program's storage descriptors have a proven read extent
/// (Shader::Info::storage_read_extents). Nothing uses them yet: the preparation counts them.
u32 ProvenReadExtents(const Shader::Info& info) {
    if (info.storage_read_extents.size() != info.storage_buffers_descriptors.size()) {
        return 0;
    }
    return static_cast<u32>(
        std::ranges::count_if(info.storage_read_extents,
                              [](const Shader::StorageWriteExtent& extent) { return extent.bounded; }));
}

} // Anonymous namespace

size_t ComputePipelineCacheKey::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), sizeof *this);
    return static_cast<size_t>(hash);
}

bool ComputePipelineCacheKey::operator==(const ComputePipelineCacheKey& rhs) const noexcept {
    return std::memcmp(&rhs, this, sizeof *this) == 0;
}

size_t GraphicsPipelineCacheKey::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), Size());
    return static_cast<size_t>(hash);
}

bool GraphicsPipelineCacheKey::operator==(const GraphicsPipelineCacheKey& rhs) const noexcept {
    if (unique_hashes != rhs.unique_hashes) {
        return false;
    }
    return std::memcmp(reinterpret_cast<const char*>(this) + sizeof(unique_hashes),
                       reinterpret_cast<const char*>(&rhs) + sizeof(unique_hashes),
                       state.Size()) == 0;
}

namespace {
// NVIDIA's and AMD's drivers keep a disk cache of built pipelines of their own. Eden's copy on top
// only kept every pipeline the title ever built in host memory (title 1: a 245 MB file, ~0.35 GB
// of RAM for the session) and copied it whole at each save, which stalled frames; pipelines built
// no slower without it.
/// The driver's own disk cache folder (NVIDIA's GLCache, AMD's VkCache) on Windows: files, size
/// and the newest write, so a "shaders build again" report shows whether the driver kept what the
/// preparation built (fable52_answer_3: an AMD player's cache kept ~44% and nothing after a
/// restart). Logged at boot, after a preparation and at exit.
void LogDriverCacheFolder(const Device& device, std::string_view when) {
#ifdef _WIN32
    const char* const local = std::getenv("LOCALAPPDATA");
    if (local == nullptr) {
        return;
    }
    std::filesystem::path folder{local};
    switch (device.GetDriverID()) {
    case VK_DRIVER_ID_NVIDIA_PROPRIETARY:
        folder /= "NVIDIA/GLCache";
        break;
    case VK_DRIVER_ID_AMD_PROPRIETARY:
        folder /= "AMD/VkCache";
        break;
    default:
        return;
    }
    std::error_code ec;
    if (!std::filesystem::is_directory(folder, ec)) {
        LOG_INFO(Render_Vulkan, "Driver cache folder {} ({}): not there", folder.string(), when);
        return;
    }
    u64 files = 0;
    u64 bytes = 0;
    std::filesystem::file_time_type newest{};
    for (std::filesystem::recursive_directory_iterator it{folder, ec}, end; !ec && it != end;
         it.increment(ec)) {
        std::error_code entry_ec;
        if (!it->is_regular_file(entry_ec)) {
            continue;
        }
        ++files;
        bytes += it->file_size(entry_ec);
        newest = (std::max)(newest, it->last_write_time(entry_ec));
    }
    const auto age = std::chrono::duration_cast<std::chrono::seconds>(
        std::filesystem::file_time_type::clock::now() - newest);
    const char* const amd_path = std::getenv("AMD_VK_PIPELINE_CACHE_PATH");
    LOG_INFO(Render_Vulkan,
             "Driver cache folder {} ({}): {} files, {} MiB, newest written {} s ago{}{}",
             folder.string(), when, files, bytes >> 20, files != 0 ? age.count() : -1,
             amd_path != nullptr ? "; AMD_VK_PIPELINE_CACHE_PATH=" : "",
             amd_path != nullptr ? amd_path : "");
#endif
}

/// A marker that lives while a pipeline cache does: still there at boot means the last session
/// never reached ~PipelineCache (killed, crashed or its emulation thread terminated), so the
/// driver may not have finished writing its cache either.
std::filesystem::path SessionMarkerPath() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) / "pipeline_cache_open.flag";
}

/// EDEN_NX2_AMD_APP_CACHE=1 (boot, off): on AMD's own driver keep our VkPipelineCache file too,
/// for what play builds (boot rebuilds of earlier sessions' pipelines included), not the
/// preparation's ~120k. A player's driver cache kept ~44% of a preparation and nothing visible
/// across a restart (fable52_answer_3, fix 2).
bool AmdAppCache(const Device& device) {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_AMD_APP_CACHE");
        return value != nullptr && value[0] == '1';
    }();
    return enabled && device.GetDriverID() == VK_DRIVER_ID_AMD_PROPRIETARY;
}

bool DriverCachesPipelines(const Device& device) {
    // Lichii: the NVIDIA and AMD drivers keep their own cache, but per executable: a new build of
    // the emulator starts it empty (every pipeline a 25-100 ms compile again). Ours, in the
    // shader folder, survives that. Opt-in (LICHII_APP_PIPELINE_CACHE=1): it skips the prepared
    // precompiled pipelines, which rely on the driver cache.
    static const bool app_cache = [] {
        const char* value = std::getenv("LICHII_APP_PIPELINE_CACHE");
        return value != nullptr && value[0] == '1';
    }();
    if (app_cache) {
        return false;
    }
    switch (device.GetDriverID()) {
    case VK_DRIVER_ID_NVIDIA_PROPRIETARY:
    case VK_DRIVER_ID_AMD_PROPRIETARY:
        return true;
    default:
        return false;
    }
}
} // Anonymous namespace

PipelineCache::PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_,
                             const Device& device_, Scheduler& scheduler_,
                             DescriptorPool& descriptor_pool_,
                             GuestDescriptorQueue& guest_descriptor_queue_,
                             DescriptorBufferRing& descriptor_buffer_ring_,
                             RenderPassCache& render_pass_cache_, BufferCache& buffer_cache_,
                             TextureCache& texture_cache_, VideoCore::ShaderNotify& shader_notify_)
    : VideoCommon::ShaderCache{device_memory_}, device{device_}, scheduler{scheduler_},
      descriptor_pool{descriptor_pool_}, guest_descriptor_queue{guest_descriptor_queue_},
      descriptor_buffer_ring{descriptor_buffer_ring_},
      render_pass_cache{render_pass_cache_}, buffer_cache{buffer_cache_},
      texture_cache{texture_cache_}, shader_notify{shader_notify_},
      use_asynchronous_shaders{Settings::values.use_asynchronous_shaders.GetValue()},
      use_vulkan_pipeline_cache{Settings::values.use_vulkan_driver_pipeline_cache.GetValue() &&
                                (!DriverCachesPipelines(device_) || AmdAppCache(device_))},
      oot_visual_workaround{[] {
          const char* value = std::getenv("EDEN_NX2_OOT_VISUAL_WORKAROUND");
          return value != nullptr && std::string_view{value} == "1";
      }()},
      num_pipeline_workers{device.HasBrokenParallelShaderCompiling() ? 1ULL
                                                                   : GetTotalPipelineWorkers()},
      emission_budget{num_pipeline_workers},
      workers(num_pipeline_workers,
              "VkPipelineBuilder", {}, Common::ThreadPlacement::Background),
      serialization_thread(1, "VkPipelineSerialization", {},
                           Common::ThreadPlacement::Background) {
    const auto& float_control{device.FloatControlProperties()};
    const VkDriverId driver_id{device.GetDriverID()};
    const VkShaderStageFlags subgroup_stages{device.GetSubgroupSupportedStages()};
    const auto subgroup_stage_bit{[subgroup_stages](VkShaderStageFlags flag, Shader::Stage stage) {
        return (subgroup_stages & flag) != 0 ? (1u << static_cast<u32>(stage)) : 0u;
    }};
    u32 supported_subgroup_stages{
        subgroup_stage_bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexA) |
        subgroup_stage_bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexB) |
        subgroup_stage_bit(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                           Shader::Stage::TessellationControl) |
        subgroup_stage_bit(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                           Shader::Stage::TessellationEval) |
        subgroup_stage_bit(VK_SHADER_STAGE_GEOMETRY_BIT, Shader::Stage::Geometry) |
        subgroup_stage_bit(VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Stage::Fragment) |
        subgroup_stage_bit(VK_SHADER_STAGE_COMPUTE_BIT, Shader::Stage::Compute) |
        subgroup_stage_bit(VK_SHADER_STAGE_TASK_BIT_EXT, Shader::Stage::Task) |
        subgroup_stage_bit(VK_SHADER_STAGE_MESH_BIT_EXT, Shader::Stage::Mesh)};
    if (driver_id == VK_DRIVER_ID_MESA_TURNIP) {
        supported_subgroup_stages &= ~(1u << static_cast<u32>(Shader::Stage::Geometry));
    }
    profile = Shader::Profile{
        .supported_spirv = device.SupportedSpirvVersion(),
        .unified_descriptor_binding = true,
        .support_descriptor_aliasing = device.IsDescriptorAliasingSupported(),
        .support_int8 = device.IsInt8Supported(),
        .support_uniform_and_storage_buffer_8bit =
            device.IsUniformAndStorageBuffer8BitAccessSupported(),
        .support_storage_buffer_8bit = device.IsStorageBuffer8BitAccessSupported(),
        .support_int16 = device.IsShaderInt16Supported(),
        .support_uniform_and_storage_buffer_16bit =
            device.IsUniformAndStorageBuffer16BitAccessSupported(),
        .support_storage_buffer_16bit = device.IsStorageBuffer16BitAccessSupported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .support_vertex_instance_id = false,
        .support_float_controls = device.IsKhrShaderFloatControlsSupported(),
        .support_separate_denorm_behavior =
            float_control.denormBehaviorIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_separate_rounding_mode =
            float_control.roundingModeIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_fp16_denorm_preserve = float_control.shaderDenormPreserveFloat16 != VK_FALSE,
        .support_fp32_denorm_preserve = float_control.shaderDenormPreserveFloat32 != VK_FALSE,
        .support_fp16_denorm_flush = float_control.shaderDenormFlushToZeroFloat16 != VK_FALSE,
        .support_fp32_denorm_flush = float_control.shaderDenormFlushToZeroFloat32 != VK_FALSE,
        .support_fp16_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat16 != VK_FALSE,
        .support_fp32_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat32 != VK_FALSE,
        .support_fp64_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat64 != VK_FALSE,
        .support_explicit_workgroup_layout = device.IsKhrWorkgroupMemoryExplicitLayoutSupported(),
        .support_workgroup_layout_8bit_access =
            device.IsWorkgroupMemoryExplicitLayout8BitAccessSupported(),
        .support_workgroup_layout_16bit_access =
            device.IsWorkgroupMemoryExplicitLayout16BitAccessSupported(),
        .support_shader_quad_control = device.IsKhrShaderQuadControlSupported(),
        .support_quad_shuffles = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_QUAD_BIT),
        .support_vote = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_VOTE_BIT),
        .support_shuffle_relative =
            device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT),
        .supported_subgroup_stages = supported_subgroup_stages,
        .support_viewport_index_layer_non_geometry =
            device.IsExtShaderViewportIndexLayerSupported(),
        .support_viewport_mask = device.IsNvViewportArray2Supported(),
        .support_typeless_image_loads = device.IsFormatlessImageLoadSupported(),
        .support_demote_to_helper_invocation =
            device.IsExtShaderDemoteToHelperInvocationSupported(),
        .support_int64_atomics = device.IsExtShaderAtomicInt64Supported(),
        .support_shared_int64_atomics = device.IsSharedInt64AtomicsSupported(),
        .support_derivative_control = true,
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_native_ndc = device.IsExtDepthClipControlSupported(),
        .support_scaled_attributes = !device.MustEmulateScaledFormats(),
        .support_multi_viewport = device.SupportsMultiViewport(),
        .support_geometry_streams = device.AreTransformFeedbackGeometryStreamsSupported(),
        .support_sampled_image_array_nonuniform_indexing =
            device.IsSampledImageArrayNonUniformIndexingSupported(),
        .support_storage_image_array_nonuniform_indexing =
            device.IsStorageImageArrayNonUniformIndexingSupported(),
        .support_uniform_texel_buffer_array_nonuniform_indexing =
            device.IsUniformTexelBufferArrayNonUniformIndexingSupported(),
        .support_storage_texel_buffer_array_nonuniform_indexing =
            device.IsStorageTexelBufferArrayNonUniformIndexingSupported(),

        .warp_size_potentially_larger_than_guest = device.IsWarpSizePotentiallyBiggerThanGuest(),

        .lower_left_origin_mode = false,
        .need_declared_frag_colors = false,
        .need_gather_subpixel_offset = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                       driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                       driver_id == VK_DRIVER_ID_MESA_RADV ||
                                       driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS ||
                                       driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA ||
                                       driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY ||
                                       driver_id == VK_DRIVER_ID_MESA_TURNIP,

        .has_broken_spirv_clamp = driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS,
        .has_broken_spirv_position_input = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .has_broken_unsigned_image_offsets = false,
        .has_broken_signed_operations = false,
        .has_broken_fp16_float_controls = driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY,
        .max_compute_shared_memory_size = device.GetMaxComputeSharedMemorySize(),
        .has_broken_fp32_denorm_flush = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .ignore_nan_fp_comparisons = false,
        .has_broken_spirv_subgroup_mask_vector_extract_dynamic =
            driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY &&
            device.GetDriverVersion() < VK_MAKE_VERSION(512, 672, 0),
        .has_broken_spirv_subgroup_shuffle = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .has_broken_spirv_fragment_shuffle = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY,
        .max_subgroup_size = device.GetMaxSubgroupSize(),
        .has_broken_robust =
            device.IsNvidia() && device.GetNvidiaArch() <= NvidiaArchitecture::Arch_Pascal,
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_user_clip_distances = device.GetMaxUserClipDistances()
    };
    profile.support_cooperative_hmma =
        device.IsCooperativeHmmaSupported() && TestCooperativeHmma(device);
    profile.support_cooperative_hmma_16x16x16 =
        !profile.support_cooperative_hmma && device.IsCooperativeHmma16x16x16Supported();
    profile.support_cooperative_hmma_8x16x16 = !profile.support_cooperative_hmma &&
                                               !profile.support_cooperative_hmma_16x16x16 &&
                                               device.IsCooperativeHmma8x16x16Supported();
    if (profile.support_cooperative_hmma_16x16x16) {
        LOG_INFO(Render_Vulkan, "Guest tensor operations run on 16x16x16 cooperative matrices");
    } else if (profile.support_cooperative_hmma_8x16x16) {
        LOG_INFO(Render_Vulkan, "Guest tensor operations run on 8x16x16 cooperative matrices");
    }

    host_info = Shader::HostTranslateInfo{
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_per_stage_descriptor_sampled_images = device.GetMaxPerStageDescriptorSampledImages(),
        .max_per_stage_resources = device.GetMaxPerStageResources(),
        .max_per_stage_descriptor_storage_buffers = device.GetMaxPerStageDescriptorStorageBuffers(),
        .max_descriptor_set_samplers = device.GetMaxDescriptorSetSamplers(),
        .max_descriptor_set_uniform_buffers = device.GetMaxDescriptorSetUniformBuffers(),
        .max_descriptor_set_uniform_buffers_dynamic = device.GetMaxDescriptorSetUniformBuffersDynamic(),
        .max_descriptor_set_storage_buffers = device.GetMaxDescriptorSetStorageBuffers(),
        .max_descriptor_set_storage_buffers_dynamic = device.GetMaxDescriptorSetStorageBuffersDynamic(),
        .max_descriptor_set_sampled_images = device.GetMaxDescriptorSetSampledImages(),
        .max_descriptor_set_storage_images = device.GetMaxDescriptorSetStorageImages(),
        .max_descriptor_set_input_attachements = device.GetMaxDescriptorSetInputAttachments(),
        .support_float64 = device.IsFloat64Supported(),
        .support_float16 = device.IsFloat16Supported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .needs_demote_reorder = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                driver_id == VK_DRIVER_ID_SAMSUNG_PROPRIETARY,
        .support_snorm_render_buffer = true,
        .support_viewport_index_layer = device.IsExtShaderViewportIndexLayerSupported(),
        .support_viewport_mask = device.IsNvViewportArray2Supported(),
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_conditional_barrier = device.SupportsConditionalBarriers(),
        .single_lane_geometry_subgroups = !profile.SupportsSubgroupStage(Shader::Stage::Geometry),
        .support_byte_addressed_r16 = true,
    };
    host_info.ApplyDescriptorLimitPolicy();

    if (device.GetMaxVertexInputAttributes() < Maxwell::NumVertexAttributes) {
        LOG_WARNING(Render_Vulkan, "maxVertexInputAttributes is too low: {} < {}",
                    device.GetMaxVertexInputAttributes(), Maxwell::NumVertexAttributes);
    }
    if (device.GetMaxVertexInputBindings() < Maxwell::NumVertexArrays) {
        LOG_WARNING(Render_Vulkan, "maxVertexInputBindings is too low: {} < {}",
                    device.GetMaxVertexInputBindings(), Maxwell::NumVertexArrays);
    }

    LOG_INFO(Render_Vulkan, "DynamicState setting value: {}", u32(EffectiveDynaState()));

    dynamic_features = {};
    dynamic_features.driver_id = device.GetDriverID();
    dynamic_features.driver_version = device.GetDriverVersion();

    // User granularity enforced in vulkan_device.cpp switch statement:
    //   Level 0: Core Dynamic States only
    //   Level 1: Core + EDS1
    //   Level 2: Core + EDS1 + EDS2 (accumulative)
    //   Level 3: Core + EDS1 + EDS2 + EDS3 (accumulative)
    // Here we only verify if extensions were successfully loaded by the device

    dynamic_features.has_extended_dynamic_state =
        device.IsExtExtendedDynamicStateSupported();

    dynamic_features.has_extended_dynamic_state_2 =
        device.IsExtExtendedDynamicState2Supported();
    dynamic_features.has_extended_dynamic_state_2_logic_op =
        device.IsExtExtendedDynamicState2ExtrasSupported();
    dynamic_features.has_extended_dynamic_state_2_patch_control_points = false;

    dynamic_features.has_extended_dynamic_state_3_blend =
        device.IsExtExtendedDynamicState3BlendingSupported();
    dynamic_features.has_extended_dynamic_state_3_enables =
        device.IsExtExtendedDynamicState3EnablesSupported();
    dynamic_features.has_color_write_enable =
        device.IsExtColorWriteEnableSupported();
    dynamic_features.has_dynamic_state3_depth_clamp_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3DepthClampEnable();
    dynamic_features.has_dynamic_state3_logic_op_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3LogicOpEnable();
    dynamic_features.has_dynamic_state3_line_stipple_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3LineStippleEnable();

    // VIDS: Independent toggle (not affected by dyna_state levels)
    dynamic_features.has_dynamic_vertex_input =
        device.IsExtVertexInputDynamicStateSupported() &&
        Settings::values.vertex_input_dynamic_state.GetValue();
    dynamic_features.has_static_vertex_repack =
        StaticVertexRepackEnabled(device.GetMinVertexInputBindingStrideAlignment());

    dynamic_features.has_provoking_vertex = device.IsExtProvokingVertexSupported();
    dynamic_features.has_provoking_vertex_first_mode =
        device.SupportsProvokingVertexFirstMode();
    dynamic_features.has_provoking_vertex_last_mode =
        device.SupportsProvokingVertexLastMode();
    dynamic_features.has_provoking_vertex_tf_preserve =
        device.SupportsTransformFeedbackProvokingVertexPreservation();

    // EDEN_NX2_ASYNC_TRANSLATE=1/2: new pipelines of draws that may be skipped are translated on
    // threads of their own. It relies on the asynchronous shaders' draw skipping.
    if (VideoCommon::AsyncTranslateWorkers()) {
        if (use_asynchronous_shaders) {
            const size_t threads = TranslateThreads();
            translate_queue = std::make_unique<TranslateQueue>(
                threads, MaxTranslationsInFlight,
                [this](TranslationJob& job) { TranslateOnWorker(job); },
                [this](std::unique_ptr<GraphicsPipeline> pipeline) {
                    RetireTranslatedPipeline(std::move(pipeline));
                });
            LOG_INFO(Render_Vulkan,
                     "EDEN_NX2_ASYNC_TRANSLATE={}: new pipelines of draws that may be skipped are "
                     "translated on {} threads{}",
                     VideoCommon::AsyncTranslateMode(), threads,
                     VideoCommon::AsyncTranslateMode() == 2
                         ? ", and on the GPU thread too to compare (verify mode)"
                         : "");
        } else {
            LOG_WARNING(Render_Vulkan, "EDEN_NX2_ASYNC_TRANSLATE needs asynchronous shaders; "
                                       "translating on the GPU thread");
        }
    } else if (VideoCommon::AsyncTranslateMode() == 3) {
        LOG_INFO(Render_Vulkan, "EDEN_NX2_ASYNC_TRANSLATE=3: GPU-thread translations use "
                                "environment snapshots, checked against the live state");
    }
}

namespace {
// <title>/vulkan_used.bin: the graphics pipelines recent sessions used, as key hash + sessions
// since the last use.
constexpr u32 UsedPipelinesMagic = 0x31505545; // "EUP1"
// Pipelines the last two sessions used are built at boot. Each holds ~0.45 MB of driver memory:
// title 1 built 4,800 from eight sessions, 1.5 GB more than building them on first use (but that
// stalled frames 4x as often), while one session of its title screen uses ~1,500.
// MoltenVK keeps the previous eight sessions: a pipeline built while playing goes through
// SPIR-V -> MSL -> Metal, far more than the 2-3 ms the shorter window assumes, so each one
// built at boot is a stutter avoided.
// Ocarina still built pipelines for 1-2 s of every 10 s early in play with two sessions kept:
// keep sixteen (#161).
constexpr u8 UsedPipelinesMaxAge = 2;
constexpr u8 UsedPipelinesMaxAgePerf = 16;
constexpr u8 UsedPipelinesMaxAgeMoltenVK = 8;

u8 MaxUsedPipelineAge(const Device& device) {
    if (device.IsMoltenVK()) {
        return UsedPipelinesMaxAgeMoltenVK;
    }
    return VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::PipelineRetention) ? UsedPipelinesMaxAgePerf : UsedPipelinesMaxAge;
}

std::unordered_map<u64, u8> LoadUsedPipelines(const std::filesystem::path& path, u8 max_age) {
    std::unordered_map<u64, u8> used;
    std::ifstream file{path, std::ios::binary};
    u32 magic{}, count{};
    if (!file.read(reinterpret_cast<char*>(&magic), sizeof(magic)) || magic != UsedPipelinesMagic ||
        !file.read(reinterpret_cast<char*>(&count), sizeof(count))) {
        return used;
    }
    used.reserve(count);
    for (u32 i = 0; i < count; ++i) {
        u64 hash{};
        u8 age{};
        if (!file.read(reinterpret_cast<char*>(&hash), sizeof(hash)) ||
            !file.read(reinterpret_cast<char*>(&age), sizeof(age))) {
            break;
        }
        if (age < max_age) { // lists saved with a longer window
            used.emplace(hash, age);
        }
    }
    return used;
}

/// Keeps only the vertex attribute types the programs of `envs` read (dynamic vertex input keys).
void MaskKeyedAttributeTypes(GraphicsPipelineCacheKey& key, std::span<const FileEnvironment> envs) {
    std::array<u64, Maxwell::MaxShaderProgram> attribute_masks;
    attribute_masks.fill(~u64{0});
    size_t env_index = 0;
    for (size_t stage = 0; stage < Maxwell::MaxShaderProgram; ++stage) {
        if (key.unique_hashes[stage] != 0 && env_index < envs.size()) {
            attribute_masks[stage] = VideoCommon::VertexAttributeMaskOf(envs[env_index++]);
        }
    }
    key.state.attribute_types = VideoCommon::KeyedAttributeTypes(
        key.state.attribute_types, key.unique_hashes, attribute_masks);
}

/// EDEN_NX2_CANONICAL_KEYS=1 (off by default; read on first use, after nx2_switches.txt): native
/// graphics keys without the state that cannot change the pipeline (zortos293/Lichii#895's
/// FixedPipelineState::CanonicalizeNative), at the draw, at disk cache load and in the
/// precompiled preparation, so a precompiled pipeline is created with the draw's create info.
bool CanonicalKeys() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_CANONICAL_KEYS");
        // On by default since 2026-11-09: the precompiled fast path needs the draws' keys in the
        // recipe's canonical form (without it 425-566 OoT keys a window missed). =0 turns it off.
        const bool on = value == nullptr || value[0] != '0';
        if (on) {
            LOG_INFO(Render_Vulkan, "EDEN_NX2_CANONICAL_KEYS=1: native pipeline keys are canonical "
                                    "(blend state of attachments without blending, attachments "
                                    "past the last colour target, depth bounds)");
        }
        return on;
    }();
    return enabled;
}

/// EDEN_NX2_PRECOMPILED_FAST (zortos293/Lichii#896; off by default, read on first use): 0 off; 1
/// the precompiled preparation (EDEN_NX2_PRECOMPILE) keeps each pipeline's shader infos and
/// SPIR-V, and a draw of a precompiled pipeline whose recorded answers hold is created from them
/// instead of translated; 2 the same, and each such pipeline is translated from the game's state
/// too and compared (the audit): one that differs is never used, the normal path builds it.
int PrecompiledFastMode() {
    static const int mode = [] {
        const char* value = std::getenv("EDEN_NX2_PRECOMPILED_FAST");
        // 1 by default since 2026-11-09 (owner: extra disk OK if no fps or RAM cost): OoT empty
        // cache, precompile + fast path vs precompile only: GPU-thread pipeline time 3133 -> 44 ms,
        // translation 1884 -> 2 ms, load 75 -> 58 s; audit 989 compared, 0 differ; index 10.2 MB,
        // working set 11.6 -> 11.3 GB. =0 off, =2 audit.
        const int parsed = value == nullptr ? 1 : value[0] == '0' ? 0 : value[0] == '2' ? 2 : 1;
        if (parsed != 0) {
            LOG_INFO(Render_Vulkan,
                     "EDEN_NX2_PRECOMPILED_FAST={}: precompiled pipelines are drawn from the "
                     "infos and SPIR-V their preparation keeps{}",
                     parsed,
                     parsed == 2 ? ", each translated from the game's state too and compared "
                                   "(audit, slow, not for play)"
                                 : "");
        }
        return parsed;
    }();
    return mode;
}

/// The canonical form of a native key's state (EDEN_NX2_CANONICAL_KEYS=1). The attribute types
/// are left as they are: draws and the loader already keep only those the programs read
/// (VideoCommon::KeyedAttributeTypes, the rule #895's mask follows). Idempotent.
void CanonicalizeKeyState(FixedPipelineState& state, const DynamicFeatures& features) noexcept {
    // With a dynamic stipple enable the key never says whether lines are stippled (Refresh3
    // leaves line_stipple_enable 0), while the factor and pattern stay static in the pipeline:
    // clearing them would give a game that enables stippling a factor of 0. Kept then.
    const u32 stipple_factor = state.line_stipple_factor;
    const u32 stipple_pattern = state.line_stipple_pattern;
    state.CanonicalizeNative(~0U);
    if (features.has_dynamic_state3_line_stipple_enable) {
        state.line_stipple_factor = stipple_factor;
        state.line_stipple_pattern = stipple_pattern;
    }
    // The logic operation counts only with logic op on. Without a dynamic logic op (AMD's
    // Windows driver) the key held the register's operation even with logic op off, so keys
    // recorded where it is dynamic (NVIDIA, the OoT recipe) never matched. Only while the enable
    // itself is static: a dynamic enable leaves logic_op_enable 0 in every key.
    if (!features.has_dynamic_state3_logic_op_enable && !state.dynamic_state.logic_op_enable) {
        state.dynamic_state.logic_op.Assign(0);
    }
}

// EDEN_NX2_REKEY_FOREIGN_PIPELINES=0: cached graphics pipelines another GPU driver (or driver
// version) stored are skipped at boot instead of loaded under this driver (see load_graphics).
bool RekeyForeignPipelines() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_REKEY_FOREIGN_PIPELINES");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

/// Reads one record as VideoCommon::LoadPipelines does and returns the key it is loaded under
/// (graphics keys masked as load_graphics masks them, compared over GraphicsPipelineCacheKey::Size
/// bytes), for merging another build's cache. Counts and sizes are checked against the bytes
/// left first, so a corrupt record cannot make an environment allocate more than the file holds.
std::optional<std::string> ReadPipelineRecordKey(std::ifstream& file, std::streamoff end) {
    u32 num_envs{};
    file.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));
    if (num_envs == 0 || num_envs > Maxwell::MaxShaderProgram) {
        return std::nullopt;
    }
    std::vector<FileEnvironment> envs(num_envs);
    for (FileEnvironment& env : envs) {
        const auto begin = static_cast<std::streamoff>(file.tellg());
        std::array<u64, 6> counts{}; // code size, then the entry count of each table
        file.read(reinterpret_cast<char*>(counts.data()), sizeof(counts));
        file.seekg(begin);
        const u64 left = static_cast<u64>(end - begin);
        if (std::ranges::any_of(counts, [left](u64 count) { return count > left; })) {
            return std::nullopt;
        }
        env.Deserialize(file);
        if (static_cast<u32>(env.ShaderStage()) > static_cast<u32>(Shader::Stage::Mesh)) {
            return std::nullopt;
        }
    }
    const bool compute = envs.front().ShaderStage() == Shader::Stage::Compute;
    const auto compute_envs = std::ranges::count_if(envs, [](const FileEnvironment& env) {
        return env.ShaderStage() == Shader::Stage::Compute;
    });
    if (compute ? num_envs != 1 : compute_envs != 0) {
        return std::nullopt;
    }
    std::string bytes(1, compute ? 'c' : 'g');
    if (compute) {
        ComputePipelineCacheKey key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));
        bytes.append(reinterpret_cast<const char*>(&key), sizeof(key));
        return bytes;
    }
    GraphicsPipelineCacheKey key;
    file.read(reinterpret_cast<char*>(&key), sizeof(key));
    if (key.state.dynamic_vertex_input != 0) {
        MaskKeyedAttributeTypes(key, envs);
    }
    // Loading puts every record under this driver (see load_graphics): a pipeline cached by
    // another driver is the same pipeline.
    key.state.driver_id = 0;
    key.state.driver_version = 0;
    bytes.append(reinterpret_cast<const char*>(&key), (std::min)(key.Size(), sizeof(key)));
    return bytes;
}

/// Pipeline caches of zortos293/Lichii builds whose records this build reads, newest first.
/// From #194 on (cache version 53) they write the same records as this build under another
/// version number; version 54 (#408) only masks the vertex attribute types in its keys, as
/// loading does here. Community packs often keep the legacy names, so those are taken when
/// their header says 53 or 54.
void MergeCompatiblePipelineCaches(const std::filesystem::path& base_dir,
                                   const std::filesystem::path& own_cache,
                                   const std::filesystem::path& own_used) {
    constexpr std::array<char, 8> PipelineCacheMagic{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};
    // 55: zortos293/Lichii#408/#804 (same records, keys masked as ours already are).
    constexpr std::array<u32, 3> CompatibleVersions{55, 54, 53};
    struct Candidate {
        std::filesystem::path cache;
        std::filesystem::path used;
        std::optional<u32> version; // nullopt: legacy name, read from the header
    };
    std::vector<Candidate> candidates;
    for (const u32 version : CompatibleVersions) {
        candidates.push_back({VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan", version),
                              VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan_used",
                                                                      version),
                              version});
    }
    candidates.push_back({base_dir / "vulkan.bin", base_dir / "vulkan_used.bin", std::nullopt});
    candidates.push_back({base_dir / "vulkan_pipelines.bin", {}, std::nullopt});

    auto log_path = own_cache;
    log_path += ".merged";
    std::error_code error;
    // Merge everything again into a cache that was deleted.
    auto merge_log = std::filesystem::exists(own_cache, error)
                         ? VideoCommon::ReadPipelineMergeLog(log_path)
                         : VideoCommon::PipelineMergeLog{};
    bool log_changed = false;
    std::optional<VideoCommon::PipelineKeySet> own_keys;
    using Status = VideoCommon::PipelineCacheMergeResult::Status;
    for (const Candidate& candidate : candidates) {
        const auto stamp = VideoCommon::StampPipelineCacheFile(candidate.cache);
        if (!stamp) {
            continue;
        }
        const std::string name = Common::FS::PathToUTF8String(candidate.cache.filename());
        const std::string path = Common::FS::PathToUTF8String(candidate.cache);
        std::optional<u32> version = candidate.version;
        if (!version) {
            for (const u32 compatible : CompatibleVersions) {
                if (VideoCommon::PipelineCacheHeaderMatches(candidate.cache, PipelineCacheMagic,
                                                            compatible)) {
                    version = compatible;
                    break;
                }
            }
            if (!version) {
                continue; // another build's or a driver cache
            }
        }
        const auto merged = merge_log.find(name);
        bool cache_merged = merged != merge_log.end() && merged->second == *stamp;
        if (!cache_merged) {
            const auto result = VideoCommon::MergePipelineCache(
                candidate.cache, own_cache, PipelineCacheMagic, *version, CACHE_VERSION,
                ReadPipelineRecordKey, own_keys);
            switch (result.status) {
            case Status::Merged:
                LOG_INFO(Render_Vulkan,
                         "Merged {} new pipelines ({} records) from {} (cache version {}); {} "
                         "were already cached",
                         result.added_pipelines, result.added_records, path, *version,
                         result.known_pipelines);
                cache_merged = true;
                break;
            case Status::Missing:
                continue;
            case Status::WrongHeader:
                LOG_WARNING(Render_Vulkan,
                            "Skipped pipeline cache {}: it is not a cache version {} file", path,
                            *version);
                break;
            case Status::BadSource:
                LOG_WARNING(Render_Vulkan,
                            "Skipped pipeline cache {} (cache version {}): {}; nothing was "
                            "merged, it is tried again when it changes",
                            path, *version, result.detail);
                break;
            case Status::BadDestination:
                LOG_WARNING(Render_Vulkan, "Not merging pipeline cache {} into {}: {}", path,
                            Common::FS::PathToUTF8String(own_cache), result.detail);
                return;
            case Status::NoSpace:
            case Status::WriteFailed:
                LOG_WARNING(Render_Vulkan,
                            "Not merging pipeline cache {}: {}; the cache is unchanged", path,
                            result.detail);
                continue; // tried again next boot
            }
            // A source that does not parse is recorded too, so it is not scanned every boot.
            merge_log.insert_or_assign(name, *stamp);
            log_changed = true;
        }
        if (!cache_merged || candidate.used.empty()) {
            continue;
        }
        const auto used_stamp = VideoCommon::StampPipelineCacheFile(candidate.used);
        const std::string used_name = Common::FS::PathToUTF8String(candidate.used.filename());
        const auto used_merged = merge_log.find(used_name);
        if (!used_stamp || (used_merged != merge_log.end() && used_merged->second == *used_stamp)) {
            continue;
        }
        if (const auto added =
                VideoCommon::MergeUsedPipelineLists(candidate.used, own_used, UsedPipelinesMagic)) {
            LOG_INFO(Render_Vulkan, "Merged {} used pipelines from {}", *added,
                     Common::FS::PathToUTF8String(candidate.used));
        } else {
            LOG_WARNING(Render_Vulkan,
                        "Skipped used pipelines list {}: it or {} does not parse",
                        Common::FS::PathToUTF8String(candidate.used),
                        Common::FS::PathToUTF8String(own_used));
        }
        merge_log.insert_or_assign(used_name, *used_stamp);
        log_changed = true;
    }
    if (log_changed && !VideoCommon::WritePipelineMergeLog(log_path, merge_log)) {
        LOG_WARNING(Render_Vulkan, "Failed to write {}", Common::FS::PathToUTF8String(log_path));
    }
}

// Without a list of the pipelines recent sessions used (the first boot with this cache, or a session
// that ended before saving it), this many cached graphics pipelines at most are built at boot and the
// rest when first needed. A cache that grew over many builds held 48k pipelines, ~21 GB of driver
// memory built before the game started; the list is only saved at exit or after two minutes, so a
// boot that ran out of memory never saved one and the next boot did the same.
constexpr size_t MaxPipelinesWithoutUsedList = 3000;

// EDEN_NX2_BACKGROUND_PIPELINES=0: cached pipelines boot skips are built when first needed, on
// the GPU thread, as before. EDEN_NX2_BACKGROUND_PIPELINE_THREADS sets how many threads build
// them (default a quarter of the host's, 1 to 6).
bool BackgroundPipelinesEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_BACKGROUND_PIPELINES");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

size_t BackgroundPipelineThreads() {
    if (const char* value = std::getenv("EDEN_NX2_BACKGROUND_PIPELINE_THREADS")) {
        return std::clamp<size_t>(static_cast<size_t>(std::strtoull(value, nullptr, 10)), 1, 16);
    }
    return std::clamp<size_t>(std::thread::hardware_concurrency() / 4, 1, 6);
}

/// Built graphics pipelines past which EvictUnusedPipelines destroys unused ones. By default a
/// sixteenth of the RAM at ~0.45 MB of driver memory each, as BackgroundPipelineLimit (16 GB:
/// ~2,280, 32 GB: ~4,560), at least 2,000 and at most 8,000 (the fixed default before, ~3.6 GB).
/// Ocarina of Time held 3,657 built 6 minutes in and its working set reached ~17.5 GB after an
/// hour on a 32 GB PC with the fixed 8,000. EDEN_NX2_PIPELINE_EVICT_ABOVE sets it (any value).
size_t EvictAbovePipelines() {
    static const size_t evict_above = [] {
        if (const char* value = std::getenv("EDEN_NX2_PIPELINE_EVICT_ABOVE")) {
            const size_t set = static_cast<size_t>(std::strtoull(value, nullptr, 10));
            LOG_INFO(Render_Vulkan,
                     "Graphics pipelines are evicted past {} (EDEN_NX2_PIPELINE_EVICT_ABOVE)", set);
            return set;
        }
        constexpr u64 PipelineBytes = u64{460} << 10;
        constexpr size_t MinPipelines = 2000;
        constexpr size_t MaxPipelines = 8000;
        const u64 ram = Common::GetMemInfo().TotalPhysicalMemory;
        const size_t by_ram = static_cast<size_t>(ram / 16 / PipelineBytes);
        // Only where RAM is short: an evicted pipeline is rebuilt on the GPU thread when its area
        // comes back (a run that evicted 3,400 in 25 minutes paid 14 s of stalls), so 24 GB and
        // more keep the old 8,000; under that, paging costs more than those rebuilds.
        constexpr u64 ShortRam = u64{24} << 30;
        const size_t chosen =
            ram >= ShortRam ? MaxPipelines : std::clamp<size_t>(by_ram, MinPipelines, MaxPipelines);
        LOG_INFO(Render_Vulkan,
                 "Graphics pipelines are evicted past {} ({} MB of RAM; "
                 "EDEN_NX2_PIPELINE_EVICT_ABOVE sets it)",
                 chosen, ram >> 20);
        return chosen;
    }();
    return evict_above;
}

/// How many of the cached graphics pipelines boot skipped the background builds, newest first;
/// the rest are built when a draw first needs them. A cache merged from a community pack held
/// 60k pipelines: building all of them took 15 minutes on 6 threads and, at ~0.45 MB each, held
/// ~25 GB of driver memory in pipelines no draw had asked for (none is evicted before a draw
/// takes it), which shutting down then had to destroy one by one.
/// EDEN_NX2_BACKGROUND_PIPELINE_LIMIT sets it.
size_t BackgroundPipelineLimit(size_t built_at_boot) {
    if (const char* value = std::getenv("EDEN_NX2_BACKGROUND_PIPELINE_LIMIT")) {
        return static_cast<size_t>(std::strtoull(value, nullptr, 10));
    }
    // ~0.45 MB of driver memory each: at most a sixteenth of the RAM (16 GB: ~2,200), and no more
    // than the eviction threshold leaves past what boot built.
    constexpr u64 PipelineBytes = u64{460} << 10;
    const u64 ram = Common::GetMemInfo().TotalPhysicalMemory;
    const size_t by_ram = static_cast<size_t>(ram / 16 / PipelineBytes);
    const size_t evict_above = EvictAbovePipelines();
    const size_t by_eviction = evict_above > built_at_boot ? evict_above - built_at_boot : 0;
    constexpr size_t MinPipelines = 500;
    constexpr size_t MaxPipelines = 6000;
    return std::clamp<size_t>((std::min)(by_ram, by_eviction), MinPipelines, MaxPipelines);
}

// EXPERIMENT EDEN_NX2_EXP5 (live, read by each background build): background pipeline builds
// competed with the game's few busy threads (Ocarina of Time 49 instead of 60 fps for the 45-80 s
// they took). 0 = as before (below normal priority, any core). 1 = idle priority, so every game
// thread and the GPU thread's own pipeline builders run first, and off the cores pinned to the GPU
// threads (and off their L3 where the CPU has two), on efficiency cores on hybrid CPUs. 2..16 = as
// 1, with at most that many building at once.
int QuietPrebuildMode() {
    return VideoCore::Nx2Debug::Experiment(5);
}

void PlacePrebuildThread(bool quiet) {
    thread_local bool placed_quiet = false;
    thread_local u64 placed_mask = 0;
    // The GPU threads pin themselves when they start, which can be after the first builds.
    const u64 mask = quiet ? Common::BackgroundWorkCpuMask() : 0;
    if (quiet == placed_quiet && mask == placed_mask) {
        return;
    }
    if (quiet) {
        Common::SetCurrentThreadIdlePriority();
        Common::SetCurrentThreadToEfficiencyCores();
    } else {
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
        Common::SetCurrentThreadToBackgroundWork();
    }
    Common::SetCurrentThreadCpuMask(mask);
    placed_quiet = quiet;
    placed_mask = mask;
    static std::atomic<u64> logged{~u64{0}};
    const u64 state = quiet ? mask | (u64{1} << 63) : 0;
    if (logged.exchange(state, std::memory_order_relaxed) != state) {
        LOG_INFO(Render_Vulkan, "Background pipeline threads: {} priority, CPU mask {:#x}",
                 quiet ? "idle" : "below normal", mask);
    }
}

// Precompiled pipeline workers: idle priority on the background cores beside the game, or every
// core at normal priority while the frontend holds the game for them (ShaderNotify full speed).
// Set on precompiled pipeline workers: their builds stay out of the "Building: N shaders"
// counter, which would climb by the thousand during a preparation the frontend already shows.
thread_local bool t_precompile_build = false;

void PlacePrecompileThread(bool full) {
    if (full) {
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Normal);
        Common::SetCurrentThreadToAllCores();
        Common::SetCurrentThreadCpuMask(0);
    } else {
        Common::SetCurrentThreadIdlePriority();
        Common::SetCurrentThreadToEfficiencyCores();
        Common::SetCurrentThreadCpuMask(Common::BackgroundWorkCpuMask());
    }
}

// EDEN_NX2_PRELOAD_ALL_PIPELINES=1 builds every cached pipeline at boot, as before.
// LICHII_MESH_KEY=0 keeps the vertex attribute state in task/mesh pipeline keys (old behaviour).
bool MeshKeyNormalized() {
    static const bool enabled = [] {
        const char* value = std::getenv("LICHII_MESH_KEY");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

bool PreloadAllPipelines() {
    static const bool all = [] {
        const char* value = std::getenv("EDEN_NX2_PRELOAD_ALL_PIPELINES");
        return value != nullptr && value[0] == '1';
    }();
    return all;
}

/// Memory log (LogMemoryUse): builds of disk cache records queued (at boot or in the background)
/// and not done yet, the builds done, and the file bytes of the records the queued builds hold in
/// memory. Globals, as a build dropped at shutdown can end after the cache's members.
std::atomic<size_t> g_prebuild_pending{};
std::atomic<size_t> g_prebuild_done{};
std::atomic<u64> g_prebuild_held_bytes{};
/// When the last LoadDiskResources began, for the prebuild times.
std::chrono::steady_clock::time_point g_disk_load_start{};

/// Held by a queued build of disk cache records: counts the build as pending and its records'
/// bytes as held until the build is done or dropped (the task holding it is destroyed).
class PrebuildTicket {
public:
    explicit PrebuildTicket(u64 bytes_) : bytes{bytes_} {
        g_prebuild_pending.fetch_add(1, std::memory_order_relaxed);
        g_prebuild_held_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    PrebuildTicket(PrebuildTicket&& other) noexcept
        : bytes{std::exchange(other.bytes, 0)}, live{std::exchange(other.live, false)} {}
    PrebuildTicket& operator=(PrebuildTicket&&) = delete;
    PrebuildTicket(const PrebuildTicket&) = delete;
    PrebuildTicket& operator=(const PrebuildTicket&) = delete;
    ~PrebuildTicket() {
        if (live) {
            g_prebuild_pending.fetch_sub(1, std::memory_order_relaxed);
            g_prebuild_held_bytes.fetch_sub(bytes, std::memory_order_relaxed);
        }
    }
    /// Records read later and held from then on (EDEN_NX2_PREBUILD_STREAM): released with the rest.
    void Adopt(u64 more) noexcept {
        bytes += more;
    }
    void Done() noexcept {
        g_prebuild_done.fetch_add(1, std::memory_order_relaxed);
    }

private:
    u64 bytes;
    bool live{true};
};

/// The guest memory section's commit (EDEN_NX2_SEC_RESERVE, Common::HostMemory::Commit).
std::string GuestMemoryText() {
    const auto stats = Common::HostMemory::GetCommitStats();
    if (!stats.reserved) {
        return "guest memory committed whole";
    }
    std::string text = fmt::format("guest memory {} of {} MiB committed (reserved), {} fault "
                                   "commits",
                                   stats.committed_bytes >> 20, stats.section_bytes >> 20,
                                   stats.net_commits);
    if (stats.net_commits != 0) {
        // Each is a page some path touched before the kernel handed it out: a missed commit.
        const bool in_virtual = (stats.first_net_offset >> 63) != 0;
        text += fmt::format(" (first at {} offset {:#x})", in_virtual ? "virtual" : "backing",
                            stats.first_net_offset & ~(u64{1} << 63));
    }
    return text;
}

/// One line every 10 s: this process's working set and commit, the system's commit, what the
/// disk cache prebuild still holds and the guest memory's commit. One GetProcessMemoryInfo and
/// one GlobalMemoryStatusEx.
void LogMemoryUse() {
    constexpr double MiB = 1024.0 * 1024.0;
    constexpr double GiB = MiB * 1024.0;
    const Common::ProcessMemoryUsage usage = Common::GetProcessMemoryUsage();
    LOG_INFO(Render_Vulkan,
             "Memory: working set {:.0f} MiB (peak {:.0f}), private commit {:.0f} MiB (peak "
             "{:.0f}), system commit {:.1f} of {:.1f} GiB, {:.1f} GiB RAM free; prebuild {} "
             "builds queued, {} done, {:.0f} MiB of records held; {}",
             static_cast<double>(usage.working_set) / MiB,
             static_cast<double>(usage.peak_working_set) / MiB,
             static_cast<double>(usage.private_bytes) / MiB,
             static_cast<double>(usage.peak_private_bytes) / MiB,
             static_cast<double>(usage.system_commit) / GiB,
             static_cast<double>(usage.system_commit_limit) / GiB,
             static_cast<double>(usage.available_physical) / GiB,
             g_prebuild_pending.load(std::memory_order_relaxed),
             g_prebuild_done.load(std::memory_order_relaxed),
             static_cast<double>(g_prebuild_held_bytes.load(std::memory_order_relaxed)) / MiB,
             GuestMemoryText());
}

double SecondsSinceDiskLoad() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - g_disk_load_start)
        .count();
}

// EDEN_NX2_PREBUILD_STREAM (on by default, =0 off): the disk cache records of queued builds are not
// kept in memory from boot until a worker builds them (minutes for the background builds, over
// a gigabyte for a large cache); each keeps its place in the file and is read again just before
// its build (RecordStream). What is built, in which order, and which record is tried first stay
// as without it.
bool PrebuildStreamEnabled() {
    static const bool enabled = [] {
        // On by default (2026-11-08: 4 fresh launches, background prebuild 21.4 -> 14.8 s, workers
        // waited 0.03 s in total, no more slow frames); EDEN_NX2_PREBUILD_STREAM=0 turns it off.
        const char* const stream_value = std::getenv("EDEN_NX2_PREBUILD_STREAM");
        const bool on = stream_value == nullptr || stream_value[0] != '0';
        LOG_INFO(Render_Vulkan, "EDEN_NX2_PREBUILD_STREAM: disk cache records {}",
                 on ? "read again for each build" : "kept in memory until built");
        return on;
    }();
    return enabled;
}

/// Where a disk cache record is, to read it again (EDEN_NX2_PREBUILD_STREAM).
struct RecordRef {
    u64 offset{};   ///< Its first byte (the environment count)
    u32 bytes{};    ///< Its size, key included
    u64 key_hash{}; ///< CityHash64 of the key as stored, to check the read found this record
};

/// The disk cache, read at record offsets (EDEN_NX2_PREBUILD_STREAM). Opened before the loader
/// reads the file and shared for writing and deleting: the GPU thread appends records meanwhile
/// (appends never move a record), and a loader that deletes the file (a record it cannot read)
/// leaves this handle reading the records found before.
class RecordFile {
public:
    explicit RecordFile(const std::filesystem::path& path) {
#ifdef _WIN32
        handle = CreateFileW(path.c_str(), GENERIC_READ,
                             FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
#else
        fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
#endif
    }
    ~RecordFile() {
#ifdef _WIN32
        if (handle != INVALID_HANDLE_VALUE) {
            CloseHandle(handle);
        }
#else
        if (fd >= 0) {
            close(fd);
        }
#endif
    }
    RecordFile(const RecordFile&) = delete;
    RecordFile& operator=(const RecordFile&) = delete;

    [[nodiscard]] bool IsOpen() const {
#ifdef _WIN32
        return handle != INVALID_HANDLE_VALUE;
#else
        return fd >= 0;
#endif
    }

    /// Reads out.size() bytes at offset; false when the file ends first or the read fails.
    [[nodiscard]] bool ReadAt(u64 offset, std::span<char> out) const {
        while (!out.empty()) {
#ifdef _WIN32
            OVERLAPPED position{};
            position.Offset = static_cast<DWORD>(offset);
            position.OffsetHigh = static_cast<DWORD>(offset >> 32);
            const DWORD wanted = static_cast<DWORD>((std::min<size_t>)(out.size(), 1u << 30));
            DWORD read = 0;
            if (!ReadFile(handle, out.data(), wanted, &read, &position) || read == 0) {
                return false;
            }
#else
            const ssize_t read = pread(fd, out.data(), out.size(), static_cast<off_t>(offset));
            if (read <= 0) {
                return false;
            }
#endif
            offset += static_cast<u64>(read);
            out = out.subspan(static_cast<size_t>(read));
        }
        return true;
    }

private:
#ifdef _WIN32
    HANDLE handle{INVALID_HANDLE_VALUE};
#else
    int fd{-1};
#endif
};

/// An input stream over bytes in memory, for FileEnvironment::Deserialize and RecordFits.
class MemoryStreamBuf final : public std::streambuf {
public:
    MemoryStreamBuf(char* data, size_t size) {
        setg(data, data, data + size);
    }

protected:
    pos_type seekoff(off_type offset, std::ios_base::seekdir dir,
                     std::ios_base::openmode) override {
        char* const base = eback();
        const off_type from = dir == std::ios_base::beg   ? 0
                              : dir == std::ios_base::cur ? gptr() - base
                                                          : egptr() - base;
        const off_type target = from + offset;
        if (target < 0 || target > egptr() - base) {
            return pos_type(off_type(-1));
        }
        setg(base, base + target, egptr());
        return pos_type(target);
    }
    pos_type seekpos(pos_type position, std::ios_base::openmode which) override {
        return seekoff(off_type(position), std::ios_base::beg, which);
    }
};

/// Reads the record at `ref` again into `envs` (key_size: the size of its key), as
/// VideoCommon::LoadPipelines read it. False when it cannot be read or is no longer the record
/// that was there (its size or key differ).
bool ReadRecord(const RecordFile& file, const RecordRef& ref, size_t key_size,
                std::vector<FileEnvironment>& envs, std::vector<char>& buffer) {
    buffer.resize(ref.bytes);
    if (ref.bytes < sizeof(u32) + key_size || !file.ReadAt(ref.offset, buffer)) {
        return false;
    }
    if (Common::CityHash64(buffer.data() + ref.bytes - key_size, key_size) != ref.key_hash) {
        return false;
    }
    MemoryStreamBuf memory{buffer.data(), buffer.size() - key_size};
    std::istream stream{&memory};
    stream.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    try {
        u32 num_envs{};
        stream.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));
        if (num_envs == 0 || num_envs > Maxwell::MaxShaderProgram) {
            return false;
        }
        envs.clear();
        envs.resize(num_envs);
        for (FileEnvironment& env : envs) {
            const u64 left = ref.bytes - key_size - static_cast<u64>(stream.tellg());
            if (!FileEnvironment::RecordFits(stream, left)) {
                return false;
            }
            env.Deserialize(stream);
        }
        // The environments end where the key begins.
        return static_cast<u64>(stream.tellg()) == ref.bytes - key_size;
    } catch (const std::exception&) {
        return false;
    }
}

/// Reader-side totals of the records read again, for the prebuild times.
struct StreamTotals {
    std::atomic<u64> records{};
    std::atomic<u64> failed{};
    std::atomic<u64> bytes{};
    std::atomic<u64> read_ns{};
    std::atomic<u64> wait_ns{}; ///< Time workers waited for their records
    std::atomic<u64> waits{};
};

/// EDEN_NX2_PREBUILD_STREAM: the records of queued builds, read again from the disk cache by one
/// thread in the order the builds were queued, up to `window` builds ahead of the workers. A
/// worker finds its records read already instead of waiting on the disk, and only the window is
/// held instead of every record from boot until its build. The workers take builds in queue
/// order (ThreadWorker is first in, first out), so at most one per worker is ahead of a waiting
/// one, and a window over the worker count always reaches it.
class RecordStream {
public:
    RecordStream(std::shared_ptr<const RecordFile> file_, size_t window_, bool low_priority_)
        : file{std::move(file_)}, window{window_}, low_priority{low_priority_} {
        reader = std::jthread([this](std::stop_token stop) { Run(stop); });
    }
    ~RecordStream() {
        reader.request_stop();
        if (reader.joinable()) {
            reader.join();
        }
        for (const Entry& entry : entries) {
            if (entry.state == State::Ready) {
                g_prebuild_held_bytes.fetch_sub(entry.bytes, std::memory_order_relaxed);
            }
        }
    }
    RecordStream(const RecordStream&) = delete;
    RecordStream& operator=(const RecordStream&) = delete;

    [[nodiscard]] const StreamTotals& Totals() const noexcept {
        return totals;
    }

    /// Adds a build's records (tried in this order); returns its index for Take and Drop. Builds
    /// are added in the order they are queued.
    size_t Add(std::vector<RecordRef> refs, size_t key_size) {
        std::scoped_lock lock{mutex};
        entries.push_back(Entry{std::move(refs), key_size});
        changed.notify_all();
        return entries.size() - 1;
    }
    /// No more builds come: the reader ends (and closes the file) once it has read them all.
    void Seal() {
        std::scoped_lock lock{mutex};
        sealed = true;
        changed.notify_all();
    }

    /// The records of build `index`, waiting for them if the reader is behind (counted in
    /// wait_ns). Records that could not be read again are left out. `bytes` is what the caller
    /// holds from now on (PrebuildTicket::Adopt).
    std::vector<std::vector<FileEnvironment>> Take(size_t index, u64& bytes,
                                                   const std::atomic_bool* stop) {
        std::unique_lock lock{mutex};
        Entry& entry = entries[index];
        if (entry.state != State::Ready && !finished) {
            const auto begin = std::chrono::steady_clock::now();
            while (entry.state != State::Ready && !finished &&
                   !(stop && stop->load(std::memory_order_relaxed))) {
                changed.wait_for(lock, std::chrono::milliseconds{20});
            }
            const auto waited = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - begin);
            totals.wait_ns.fetch_add(static_cast<u64>(waited.count()), std::memory_order_relaxed);
            totals.waits.fetch_add(1, std::memory_order_relaxed);
        }
        std::vector<std::vector<FileEnvironment>> records;
        bytes = 0;
        if (entry.state == State::Ready) {
            records = std::move(entry.records);
            bytes = entry.bytes;
            --ready;
        }
        entry.state = State::Taken;
        lock.unlock();
        changed.notify_all();
        return records;
    }
    /// A build that does not run (claimed by the GPU thread, or stopping): its records are freed,
    /// or not read at all if the reader has not got to them.
    void Drop(size_t index) {
        std::unique_lock lock{mutex};
        Entry& entry = entries[index];
        if (entry.state == State::Ready) {
            g_prebuild_held_bytes.fetch_sub(entry.bytes, std::memory_order_relaxed);
            entry.records = std::vector<std::vector<FileEnvironment>>{};
            --ready;
        }
        entry.state = State::Taken;
        lock.unlock();
        changed.notify_all();
    }

private:
    enum class State { Queued, Ready, Taken };
    struct Entry {
        std::vector<RecordRef> refs;
        size_t key_size{};
        State state{State::Queued};
        std::vector<std::vector<FileEnvironment>> records;
        u64 bytes{};
    };

    void Run(std::stop_token stop) {
        Common::SetCurrentThreadName("VkRecordReader");
        if (low_priority) {
            Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
        }
        std::vector<char> buffer;
        size_t next = 0;
        while (!stop.stop_requested()) {
            std::vector<RecordRef> refs;
            size_t key_size{};
            {
                std::unique_lock lock{mutex};
                changed.wait(lock, stop, [&] {
                    // Builds dropped before the reader got to them are skipped.
                    while (next < entries.size() && entries[next].state == State::Taken) {
                        ++next;
                    }
                    return (next < entries.size() && ready < window) ||
                           (sealed && next >= entries.size());
                });
                if (stop.stop_requested() || next >= entries.size()) {
                    break;
                }
                refs = std::move(entries[next].refs);
                key_size = entries[next].key_size;
            }
            const auto begin = std::chrono::steady_clock::now();
            std::vector<std::vector<FileEnvironment>> records;
            records.reserve(refs.size());
            u64 bytes = 0;
            for (const RecordRef& ref : refs) {
                std::vector<FileEnvironment> envs;
                if (ReadRecord(*file, ref, key_size, envs, buffer)) {
                    records.push_back(std::move(envs));
                    bytes += ref.bytes;
                } else {
                    totals.failed.fetch_add(1, std::memory_order_relaxed);
                }
            }
            totals.records.fetch_add(refs.size(), std::memory_order_relaxed);
            totals.bytes.fetch_add(bytes, std::memory_order_relaxed);
            totals.read_ns.fetch_add(
                static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now() - begin)
                                     .count()),
                std::memory_order_relaxed);
            {
                std::scoped_lock lock{mutex};
                Entry& entry = entries[next];
                if (entry.state == State::Queued) {
                    entry.records = std::move(records);
                    entry.bytes = bytes;
                    entry.state = State::Ready;
                    g_prebuild_held_bytes.fetch_add(bytes, std::memory_order_relaxed);
                    ++ready;
                }
                ++next;
            }
            changed.notify_all();
        }
        {
            std::scoped_lock lock{mutex};
            finished = true;
        }
        changed.notify_all();
        // Nothing more to read: the file is not held open for the rest of the session.
        file.reset();
    }

    std::shared_ptr<const RecordFile> file; ///< The reader thread's
    const size_t window;
    const bool low_priority;
    StreamTotals totals;
    std::mutex mutex;
    std::condition_variable_any changed;
    std::deque<Entry> entries;
    size_t ready{}; ///< Entries read and not taken
    bool sealed{};
    bool finished{};
    std::jthread reader; ///< Last: started once the members above exist
};

/// "workers waited 0.12 s on 3 reads, 812 records (41 MiB) read in 0.40 s, 0 failed"
std::string StreamTotalsText(const StreamTotals& totals) {
    return fmt::format("workers waited {:.2f} s on {} reads, {} records ({:.0f} MiB) read in "
                       "{:.2f} s, {} failed",
                       static_cast<double>(totals.wait_ns.load()) / 1e9, totals.waits.load(),
                       totals.records.load(),
                       static_cast<double>(totals.bytes.load()) / (1024.0 * 1024.0),
                       static_cast<double>(totals.read_ns.load()) / 1e9, totals.failed.load());
}
} // Anonymous namespace

namespace {
/// Destroys the pipelines of `pipelines` (a map or a deque of pairs holding unique_ptrs), telling
/// the frontend every few hundred that the teardown still moves: tens of thousands of them took
/// longer than the frontend waited for the emulation thread.
template <typename Container>
void DestroyPipelinesWithProgress(Container& pipelines) {
    constexpr size_t ProgressEvery = 256;
    size_t destroyed = 0;
    for (auto& entry : pipelines) {
        entry.second.reset();
        if (++destroyed % ProgressEvery == 0) {
            Common::NoteShutdownProgress();
        }
    }
    pipelines.clear();
    Common::NoteShutdownProgress();
}
} // Anonymous namespace

PipelineCache::~PipelineCache() {
    // Stop the warm-ups and background builds first: their builds use the members below. The
    // stop callback goes first, as it writes the flags below.
    warm_stop_callback.reset();
    warm_cancel = true;
    background_stop.store(true, std::memory_order_relaxed);
    graphics_warm_threads.clear();
    compute_warm_threads.clear();
    precompile_thread = {};
    // Its threads build with the members below (a translation in progress finishes).
    translate_queue.reset();
    StopBackgroundPipelines();
    Common::NoteShutdownProgress();
    workers.WaitForRequests();
    serialization_thread.WaitForRequests();
    Common::NoteShutdownProgress();
    // The caches are written before anything slow is torn down, so a teardown cut short still
    // leaves them on disk.
    SaveUsedPipelines();
    if (spirv_spool) {
        // The first boot's temporary module file: the store holds its modules now.
        spirv_spool->file.close();
        std::error_code ec;
        std::filesystem::remove(spirv_spool->path, ec);
    }
    if (use_vulkan_pipeline_cache && !vulkan_pipeline_cache_filename.empty()) {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
    }
    Common::NoteShutdownProgress();
    if (background_taken != 0 || !background_ready.empty()) {
        // Prebuilt pipelines a draw never took cost driver memory and build time for nothing.
        LOG_INFO(Render_Vulkan,
                 "Background pipelines this session: {} taken by draws, {} never used",
                 background_taken, background_ready.size());
    }
    DestroyPipelinesWithProgress(background_ready);
    LogDriverCacheFolder(device, "at exit, before the device goes");
    {
        std::error_code ec;
        std::filesystem::remove(SessionMarkerPath(), ec);
    }
    DestroyPipelinesWithProgress(graphics_cache);
    DestroyPipelinesWithProgress(evicted_pipelines);
    DestroyPipelinesWithProgress(retired_translations);
    // The shared shader objects go with the pipelines that used them.
    ShaderObjectCache::Instance().Clear();
}

void PipelineCache::NoteDiskRecordFailure(std::string_view kind, u64 hash, const char* reason) {
    constexpr size_t Shown = 4;
    if (disk_record_failures.fetch_add(1, std::memory_order_relaxed) < Shown) {
        LOG_INFO(Render_Vulkan,
                 "{} pipeline {:016x}: a cached record this build cannot translate ({}), built "
                 "when first needed instead",
                 kind, hash, reason);
    } else {
        LOG_DEBUG(Render_Vulkan, "{} pipeline {:016x}: cached record not translated: {}", kind,
                  hash, reason);
    }
}

void PipelineCache::ReportDiskRecordFailures() {
    const size_t failures = disk_record_failures.exchange(0, std::memory_order_relaxed);
    if (failures != 0) {
        LOG_INFO(Render_Vulkan,
                 "{} cached pipeline records could not be translated by this build (stored by "
                 "another or an older build); their pipelines are built when first needed",
                 failures);
    }
}

void PipelineCache::SaveUsedPipelines() {
    used_pipelines_saved = std::chrono::steady_clock::now();
    // A session too short to say what the game uses (a boot to the menu) keeps the old list.
    constexpr size_t MinimumUsed = 64;
    if (used_pipelines_filename.empty() ||
        used_this_session.size() + used_and_evicted.size() < MinimumUsed) {
        return;
    }
    std::unordered_map<u64, u8> next;
    next.reserve(used_pipelines.size() + used_this_session.size());
    for (const auto& [key, pipeline] : graphics_cache) {
        if (used_this_session.contains(pipeline.get())) {
            next.emplace(key.Hash(), u8{0});
        }
    }
    for (const u64 hash : used_and_evicted) {
        next.emplace(hash, u8{0});
    }
    for (const auto& [hash, age] : used_pipelines) {
        if (age + 1 < MaxUsedPipelineAge(device)) {
            next.try_emplace(hash, static_cast<u8>(age + 1));
        }
    }
    std::ofstream file{used_pipelines_filename, std::ios::binary | std::ios::trunc};
    const u32 count = static_cast<u32>(next.size());
    file.write(reinterpret_cast<const char*>(&UsedPipelinesMagic), sizeof(UsedPipelinesMagic));
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& [hash, age] : next) {
        file.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
        file.write(reinterpret_cast<const char*>(&age), sizeof(age));
    }
}

namespace {
// EDEN_NX2_PIPELINE_CLEAN=1 or EDEN_NX2_EXP19=1 (live; default off): see
// CurrentGraphicsPipelineClean. EXP19=2: with the patches of zortos293/Lichii#641
// (EDEN_NX2_PIPE_SKIP, EDEN_NX2_PIPE_PATCH).
int PipelineCleanMode() {
    static const bool env = [] {
        const char* value = std::getenv("EDEN_NX2_PIPELINE_CLEAN");
        return value != nullptr && value[0] != '0';
    }();
    const int experiment = VideoCore::Nx2Debug::Experiment(19);
    return experiment != 0 ? experiment : (env ? 1 : 0);
}

/// EXP52=2: where a programs-only key (`fast`) differs from the full Refresh's (`full`).
std::string DescribeProgramsKeyDiff(const GraphicsPipelineCacheKey& fast,
                                    const GraphicsPipelineCacheKey& full) {
    const FixedPipelineState& a = fast.state;
    const FixedPipelineState& b = full.state;
    std::string diffs;
    if (fast.unique_hashes != full.unique_hashes) {
        diffs += " unique_hashes";
    }
    if (a.raw1 != b.raw1) {
        diffs += fmt::format(" raw1^{:#x}", a.raw1 ^ b.raw1);
    }
    if (a.raw2 != b.raw2) {
        diffs += fmt::format(" raw2^{:#x}", a.raw2 ^ b.raw2);
    }
    if (a.attribute_types != b.attribute_types) {
        diffs += fmt::format(" attribute_types {:#x} (programs only) vs {:#x} (Refresh)",
                             a.attribute_types, b.attribute_types);
    }
    if (std::memcmp(a.attachments.data(), b.attachments.data(), sizeof(a.attachments)) != 0) {
        diffs += " attachments";
    }
    if (std::memcmp(&a.dynamic_state, &b.dynamic_state, sizeof(a.dynamic_state)) != 0) {
        diffs += " dynamic_state";
    }
    // Any byte at all, hashed or not (the state past Size() still goes to the driver).
    const auto* const bytes_a = reinterpret_cast<const u8*>(&a);
    const auto* const bytes_b = reinterpret_cast<const u8*>(&b);
    size_t first = sizeof(a);
    size_t last = 0;
    for (size_t offset = 0; offset < sizeof(a); ++offset) {
        if (bytes_a[offset] != bytes_b[offset]) {
            first = (std::min)(first, offset);
            last = offset;
        }
    }
    if (first < sizeof(a)) {
        diffs += fmt::format(" state bytes {}..{} (hashed size {})", first, last, b.Size());
    }
    return diffs;
}
} // Anonymous namespace

void PipelineCache::BuildCleanWatch(bool patch, bool blend_by_value, bool patch_vertex) {
    using Range = CleanLookup::Range;
    const Maxwell& regs = maxwell3d->regs;
    const DynamicFeatures& features = dynamic_features;
    const auto watch_into = [&regs](std::vector<Range>& ranges, const auto& field) {
        const auto offset = static_cast<size_t>(reinterpret_cast<const char*>(&field) -
                                                reinterpret_cast<const char*>(&regs));
        const size_t first = offset / sizeof(u32);
        const size_t last = (offset + sizeof(field) + sizeof(u32) - 1) / sizeof(u32);
        ranges.push_back({static_cast<u32>(first), static_cast<u32>(last - first)});
    };
    std::vector<Range> ranges;
    const auto watch = [&](const auto& field) { watch_into(ranges, field); };
    std::vector<Range> blend_ranges;
    const auto watch_blend = [&](const auto& field) { watch_into(blend_ranges, field); };

    // Every register FixedPipelineState::Refresh (and what it calls) reads with these features.
    // With `patch`, the blend registers are left out (the dual-source bit is compared instead,
    // and the attachments follow Dirty::Blending), and so are the vertex attribute formats with
    // dynamic vertex input (the types follow Dirty::VertexInput), as Refresh itself does.
    // Refresh itself, always:
    watch(regs.transform_feedback_enabled);
    watch(regs.depth_mode);
    watch(regs.polygon_mode_front);
    watch(regs.tessellation);
    watch(regs.patch_vertices);
    watch(regs.anti_alias_samples_mode);
    if (blend_by_value) {
        // EXP50: the blend registers are compared by the state they make when they moved.
        watch_blend(regs.blend);                    // ComputeAttachment0DualSourceBlend
        watch_blend(regs.blend_per_target_enabled); // ComputeAttachment0DualSourceBlend
        watch_blend(regs.blend_per_target[0]);      // ComputeAttachment0DualSourceBlend
    } else if (!patch) {
        watch(regs.blend);                    // ComputeAttachment0DualSourceBlend
        watch(regs.blend_per_target_enabled); // ComputeAttachment0DualSourceBlend
        watch(regs.blend_per_target[0]);      // ComputeAttachment0DualSourceBlend
    }
    watch(regs.alpha_test_enabled);
    watch(regs.alpha_test_func);
    watch(regs.alpha_test_ref);
    watch(regs.mandated_early_z);
    watch(regs.zeta_enable);
    watch(regs.zeta.format);
    watch(regs.window_origin);
    watch(regs.provoking_vertex);
    watch(regs.conservative_raster_enable);
    watch(regs.line_anti_alias_enable);
    watch(regs.anti_alias_alpha_control);
    watch(regs.depth_bounds);
    watch(regs.line_stipple_params);
    for (const auto& rt : regs.rt) {
        watch(rt.format);
    }
    watch(regs.point_size);
    // Dirty::VertexInput (both branches) and the static vertex repack:
    if ((!patch && !patch_vertex) || !features.has_dynamic_vertex_input) {
        watch(regs.vertex_attrib_format);
    }
    const bool repack = !features.has_dynamic_vertex_input && features.has_static_vertex_repack;
    if (!features.has_dynamic_vertex_input) {
        watch(regs.vertex_stream_instances);
        for (const auto& stream : regs.vertex_streams) {
            watch(stream.frequency);
        }
    }
    if (!features.has_extended_dynamic_state || repack) {
        // Strides (DynamicState::Refresh's vertex_strides, the repack): the first word of each
        // stream, VertexStream's leading union of stride and enable.
        for (const auto& stream : regs.vertex_streams) {
            const auto offset = static_cast<size_t>(reinterpret_cast<const char*>(&stream) -
                                                    reinterpret_cast<const char*>(&regs));
            ranges.push_back({static_cast<u32>(offset / sizeof(u32)), 1});
        }
    }
    // Dirty::ViewportSwizzles:
    for (const auto& viewport : regs.viewport_transform) {
        watch(viewport.swizzle);
    }
    // DynamicState::Refresh (no extended dynamic state):
    if (!features.has_extended_dynamic_state) {
        watch(regs.gl_front_face);
        watch(regs.stencil_front_op);
        watch(regs.stencil_two_side_enable);
        watch(regs.stencil_back_op);
        watch(regs.stencil_enable);
        watch(regs.depth_write_enabled);
        watch(regs.depth_bounds_enable);
        watch(regs.depth_test_enable);
        watch(regs.depth_test_func);
        watch(regs.gl_cull_face);
        watch(regs.gl_cull_test_enabled);
    }
    // DynamicState::Refresh2:
    if (!features.has_extended_dynamic_state_2_logic_op) {
        watch(regs.logic_op);
        if (!features.has_extended_dynamic_state_2) {
            watch(regs.polygon_offset_point_enable);
            watch(regs.polygon_offset_line_enable);
            watch(regs.polygon_offset_fill_enable);
            watch(regs.rasterize_enable);
            watch(regs.primitive_restart);
        }
    }
    // Dirty::Blending, BlendingAttachment::Refresh (no extended dynamic state 3 blending):
    if (!features.has_extended_dynamic_state_3_blend) {
        if (blend_by_value) {
            watch_blend(regs.color_mask_common);
            watch_blend(regs.color_mask);
            watch_blend(regs.blend_per_target);
            watch_blend(regs.iterated_blend);
        } else {
            if (!patch) {
                watch(regs.color_mask_common);
                watch(regs.color_mask);
                watch(regs.blend_per_target);
            }
            watch(regs.iterated_blend); // No dirty flag
        }
    }
    // DynamicState::Refresh3:
    if (!features.has_dynamic_state3_logic_op_enable) {
        watch(regs.logic_op);
    }
    if (!features.has_dynamic_state3_depth_clamp_enable) {
        watch(regs.viewport_clip_control);
    }
    if (!features.has_dynamic_state3_line_stipple_enable) {
        watch(regs.line_stipple_enable);
    }

    // Sorted and merged: contiguous words compare as one block.
    const auto merge = [](std::vector<Range> input) {
        std::ranges::sort(input, {}, &Range::first);
        std::vector<Range> merged;
        for (const Range& range : input) {
            if (!merged.empty() && range.first <= merged.back().first + merged.back().count) {
                const u32 end = (std::max)(merged.back().first + merged.back().count,
                                           range.first + range.count);
                merged.back().count = end - merged.back().first;
            } else {
                merged.push_back(range);
            }
        }
        return merged;
    };
    clean.ranges = merge(std::move(ranges));
    // RefreshXfbState, while transform_feedback_enabled (watched above) is set:
    std::vector<Range> xfb_ranges;
    watch_into(xfb_ranges, regs.transform_feedback.controls);
    watch_into(xfb_ranges, regs.stream_out_layout);
    clean.xfb_ranges = merge(std::move(xfb_ranges));

    size_t words = 0;
    for (const Range& range : clean.ranges) {
        words += range.count;
    }
    clean.snapshot.assign(words, 0);
    size_t xfb_words = 0;
    for (const Range& range : clean.xfb_ranges) {
        xfb_words += range.count;
    }
    clean.xfb_snapshot.assign(xfb_words, 0);
    clean.blend_ranges = merge(std::move(blend_ranges));
    size_t blend_words = 0;
    for (const Range& range : clean.blend_ranges) {
        blend_words += range.count;
    }
    clean.blend_snapshot.assign(blend_words, 0);
    clean.built = true;
    clean.patch = patch;
    clean.patch_vertex = patch_vertex;
    clean.blend_by_value = blend_by_value;
    clean.pipeline = nullptr;
    LOG_INFO(Render_Vulkan,
             "EDEN_NX2_PIPELINE_CLEAN: watching {} register words in {} ranges{}{}", words,
             clean.ranges.size(), patch ? " (with patches)" : "",
             blend_by_value ? fmt::format(", {} blend words compared by value (EXP50)",
                                          blend_words)
                            : std::string{});
}

bool PipelineCache::CleanBlendUnchanged() const {
    const u32* const words = maxwell3d->regs.reg_array.data();
    const u32* saved = clean.blend_snapshot.data();
    for (const CleanLookup::Range& range : clean.blend_ranges) {
        if (std::memcmp(words + range.first, saved, range.count * sizeof(u32)) != 0) {
            return false;
        }
        saved += range.count;
    }
    return true;
}

void PipelineCache::TakeCleanBlendSnapshot() {
    const u32* const words = maxwell3d->regs.reg_array.data();
    u32* saved = clean.blend_snapshot.data();
    for (const CleanLookup::Range& range : clean.blend_ranges) {
        std::memcpy(saved, words + range.first, range.count * sizeof(u32));
        saved += range.count;
    }
}

bool PipelineCache::CleanInputsUnchanged() const {
    const u32* const words = maxwell3d->regs.reg_array.data();
    const u32* saved = clean.snapshot.data();
    for (const CleanLookup::Range& range : clean.ranges) {
        if (std::memcmp(words + range.first, saved, range.count * sizeof(u32)) != 0) {
            return false;
        }
        saved += range.count;
    }
    if (maxwell3d->regs.transform_feedback_enabled != 0) {
        saved = clean.xfb_snapshot.data();
        for (const CleanLookup::Range& range : clean.xfb_ranges) {
            if (std::memcmp(words + range.first, saved, range.count * sizeof(u32)) != 0) {
                return false;
            }
            saved += range.count;
        }
    }
    return true;
}

void PipelineCache::TakeCleanSnapshot() {
    const u32* const words = maxwell3d->regs.reg_array.data();
    u32* saved = clean.snapshot.data();
    for (const CleanLookup::Range& range : clean.ranges) {
        std::memcpy(saved, words + range.first, range.count * sizeof(u32));
        saved += range.count;
    }
    if (maxwell3d->regs.transform_feedback_enabled != 0) {
        saved = clean.xfb_snapshot.data();
        for (const CleanLookup::Range& range : clean.xfb_ranges) {
            std::memcpy(saved, words + range.first, range.count * sizeof(u32));
            saved += range.count;
        }
    }
    TakeCleanBlendSnapshot(); // No ranges unless blend_by_value
    clean.hashes = graphics_key.unique_hashes;
    clean.maxwell3d = maxwell3d;
    clean.topology = maxwell3d->draw_manager.draw_state.topology;
    clean.engine_state = maxwell3d->engine_state;
}

// The key is (unique_hashes, FixedPipelineState). Refresh's output depends only on: the
// registers in clean.ranges (and clean.xfb_ranges while transform feedback is on), the dirty
// flags VertexInput, ViewportSwizzles and Blending (it recomputes those groups only while set,
// and clears some), dynamic_features (set once at construction), the draw's topology and the
// engine hint. When all of these equal their values at the last Refresh and the flags are
// clear, Refresh would rewrite the same key and Next(graphics_key) would return
// current_pipeline (whose key it is): the lookup returns it as that fast path does.
// With `patch` (zortos293/Lichii#641): the dual-source blend bit is compared instead of the blend
// registers, and a raised VertexInput (dynamic vertex input) or Blending flag redoes only that
// part of the key, which Next then looks up.
// With EXP50 (blend by value, after the-cashman's fix in zortos293/Lichii#717): the blend
// registers (clean.blend_ranges) leave the word compare and Dirty::Blending is not looked at.
// While those words equal the ones the key's attachments and dual-source bit were made from (or
// last proved equal to), the key stands. When they moved, the attachments RefreshBlending would
// build (canonical as RefreshGraphicsKey makes them) and the dual-source bit are compared with
// the key's: equal is a hit, different patches only that part of the key, which Next looks up.
// The compare reads the registers, so blend writes that never raise the flag (#679) are seen.
// With EXP52 (fable52_answer_1 section 1): when the programs changed but every other input above
// stands, the state part of the key is still what Refresh would build, except the one field the
// programs feed: with dynamic vertex input, the attribute types are masked by what the bound
// programs read (RefreshGraphicsKey's KeyedAttributeTypes over vertex_attribute_types, the
// unmasked types, which stay current: Dirty::VertexInput either misses or patches them). Nothing
// else Refresh or CanonicalizeKeyState writes reads the programs (CanonicalizeNative runs with
// the full mask). Those types are keyed again for the new programs and Next (or the slow path)
// looks the new key up. EXP52=2 also runs the full Refresh after it, compares the whole key and
// keeps Refresh's ("programs only (EXP52)" mismatches in the per-second line, first 16 logged).
GraphicsPipeline* PipelineCache::CurrentGraphicsPipelineClean(bool patch) {
    if (!RefreshStages(graphics_key.unique_hashes)) {
        current_pipeline = nullptr;
        clean.pipeline = nullptr;
        return nullptr;
    }
    const int programs_mode = VideoCore::Nx2Debug::Experiment(52);
    const bool by_value = VideoCore::Nx2Debug::Experiment(50) != 0;
    // EDEN_NX2_EXP55 (live, 0 = off): patch the key's vertex attribute types when only the vertex
    // input moved (dynamic vertex input), as EXP19=2 does, without its blend patching. In
    // Castle Town every "other state" refresh is vertex input (~20k/s). 2 = also run the full
    // Refresh after it and compare the keys (the verified/differ counts of the line).
    const int vertex_mode = VideoCore::Nx2Debug::Experiment(55);
    if (!clean.built || clean.patch != patch || clean.blend_by_value != by_value ||
        clean.patch_vertex != (vertex_mode != 0)) {
        BuildCleanWatch(patch, by_value, vertex_mode != 0);
    }
    auto& flags = maxwell3d->dirty.flags;
    const bool dynamic_blend = dynamic_features.has_extended_dynamic_state_3_blend;
    const bool blending_dirty = !dynamic_blend && flags[Dirty::Blending];
    const bool vertex_input_dirty = flags[Dirty::VertexInput];
    const bool patch_blending = patch && !by_value && blending_dirty;
    const bool patch_vertex_input = (patch || vertex_mode != 0) && vertex_input_dirty &&
                                    dynamic_features.has_dynamic_vertex_input;
    // The first reason the last Refresh's key may not stand (index into clean.miss_reasons),
    // EDEN_NX2_EXP54 (live, 0 = off): with extended dynamic state 1 and 2 the key holds only the
    // topology's class (FixedPipelineState::KeyTopology) and the draw sets the topology itself, so
    // a strip after a list keeps the pipeline. Off: any raw topology change refreshes.
    const auto topology_differs = [&] {
        const auto now = maxwell3d->draw_manager.draw_state.topology;
        if (clean.topology == now) {
            return false;
        }
        const bool collapse = graphics_key.state.extended_dynamic_state != 0 &&
                              graphics_key.state.extended_dynamic_state_2 != 0;
        if (!collapse || VideoCore::Nx2Debug::Experiment(54) == 0) {
            return true;
        }
        return FixedPipelineState::KeyTopology(clean.topology, true) !=
               FixedPipelineState::KeyTopology(now, true);
    };
    // -1 = none.
    int miss = -1;
    // With EXP52 a change of programs is no reason by itself: the other checks decide.
    const bool programs_changed = clean.hashes != graphics_key.unique_hashes;
    if (clean.pipeline == nullptr || clean.pipeline != current_pipeline ||
        clean.maxwell3d != maxwell3d) {
        miss = 0;
    } else if (programs_changed && programs_mode == 0) {
        miss = 1;
    } else if (blending_dirty && !patch_blending && !by_value) {
        miss = 2;
    } else if ((vertex_input_dirty && !patch_vertex_input) || flags[Dirty::ViewportSwizzles] ||
               topology_differs() ||
               clean.engine_state != maxwell3d->engine_state ||
               (patch && !by_value &&
                FixedPipelineState::DualSourceBlend(maxwell3d->regs) !=
                    (graphics_key.state.attachment0_dual_source_blend != 0))) {
        miss = 3;
        // Which part of "other state" (fable52_answer_2 Q3: the key holds the topology's class
        // under EDS1+2, this compares the raw topology): counted only, the first that differs.
        const size_t part =
            (vertex_input_dirty && !patch_vertex_input) ? 0
            : flags[Dirty::ViewportSwizzles]           ? 1
            : topology_differs()                                             ? 2
            : clean.engine_state != maxwell3d->engine_state                 ? 3
                                                                            : 4;
        ++clean.other_parts[part];
    } else if (!CleanInputsUnchanged()) {
        miss = 4;
    }
    bool patch_blend_value = false;
    // Only reached with EXP52 on: miss 1 catches a change of programs otherwise.
    const bool programs_only = miss < 0 && programs_changed;
    if (miss < 0) {
        ++clean.hits;
        if (by_value) {
            if (!CleanBlendUnchanged()) {
                const Maxwell& regs = maxwell3d->regs;
                const bool canonical =
                    canonical_keys && maxwell3d->GetNativeMacroEngine().native_turing;
                if (FixedPipelineState::DualSourceBlend(regs) ==
                        (graphics_key.state.attachment0_dual_source_blend != 0) &&
                    (dynamic_blend || graphics_key.state.BlendingMatches(regs, canonical))) {
                    // Moved, but to registers that make the same state: the key stands for them.
                    TakeCleanBlendSnapshot();
                    ++clean.blend_saved;
                } else {
                    patch_blend_value = true;
                }
            } else if (blending_dirty) {
                ++clean.blend_saved;
            }
            if (blending_dirty && !patch_blend_value) {
                // As RefreshBlending: the key's attachments are what the registers make.
                flags[Dirty::Blending] = false;
            }
        }
    } else {
        ++clean.misses;
        ++clean.miss_reasons[static_cast<size_t>(miss)];
        clean.pipeline = nullptr;
    }
    if (((clean.hits + clean.misses) & 0xfff) == 0) {
        const auto now = std::chrono::steady_clock::now();
        if (clean.last_log == std::chrono::steady_clock::time_point{}) {
            clean.last_log = now;
        } else if (now - clean.last_log >= std::chrono::seconds{10}) {
            const double seconds = std::chrono::duration<double>(now - clean.last_log).count();
            const auto per_second = [seconds](u64 count) {
                return static_cast<u64>(static_cast<double>(count) / seconds);
            };
            const auto& reasons = clean.miss_reasons;
            LOG_INFO(Render_Vulkan,
                     "EDEN_NX2_PIPELINE_CLEAN per second: lookups without Refresh {} (patched "
                     "{}, saved by the blend value check (EXP50{}) {}, programs only (EXP52={}) "
                     "{}{}), with Refresh {} (pipeline changed {}, programs {}, blend flag {}, "
                     "other state {} [vertex input {}, swizzles {}, topology {}, engine state {}, "
                     "dual source {}], registers {})",
                     per_second(clean.hits), per_second(clean.patches), by_value ? " on" : " off",
                     per_second(clean.blend_saved), programs_mode,
                     per_second(clean.programs_only),
                     programs_mode == 2 || vertex_mode == 2
                         ? fmt::format(", verified {} in {:.0f} s, {} differ",
                                       clean.programs_verified, seconds,
                                       clean.programs_mismatches)
                         : std::string{},
                     per_second(clean.misses), per_second(reasons[0]), per_second(reasons[1]),
                     per_second(reasons[2]), per_second(reasons[3]),
                     per_second(clean.other_parts[0]), per_second(clean.other_parts[1]),
                     per_second(clean.other_parts[2]), per_second(clean.other_parts[3]),
                     per_second(clean.other_parts[4]), per_second(reasons[4]));
            clean.other_parts = {};
            clean.last_log = now;
            clean.hits = 0;
            clean.patches = 0;
            clean.misses = 0;
            clean.blend_saved = 0;
            clean.programs_only = 0;
            clean.programs_verified = 0;
            clean.programs_mismatches = 0;
            clean.miss_reasons = {};
        }
    }
    if (clean.pipeline != nullptr) {
        const bool patching = patch_blending || patch_vertex_input || patch_blend_value;
        if (!patching && !programs_only) {
            current_pipeline->MarkUsed(frame_count);
            return BuiltPipeline(current_pipeline);
        }
        // Every other Refresh input is unchanged, so the snapshot stands. As RefreshGraphicsKey
        // under the raised flags; VertexInput stays raised for the command buffer update.
        if (patching) {
            ++clean.patches;
        }
        clean.pipeline = nullptr;
        if (patch_vertex_input) {
            vertex_attribute_types = FixedPipelineState::DynamicAttributeTypes(maxwell3d->regs);
        }
        if (patch_vertex_input || (programs_only && dynamic_features.has_dynamic_vertex_input)) {
            // New programs read other inputs: their types are keyed from the unmasked ones.
            graphics_key.state.attribute_types = VideoCommon::KeyedAttributeTypes(
                vertex_attribute_types, graphics_key.unique_hashes, vertex_attribute_masks);
        }
        if (programs_only) {
            ++clean.programs_only;
            clean.hashes = graphics_key.unique_hashes; // The snapshot's programs from now on
        }
        if (patch_blending || patch_blend_value) {
            // EXP19=2 compared the dual-source bit above; EXP50 may find it moved.
            graphics_key.state.attachment0_dual_source_blend.Assign(
                FixedPipelineState::DualSourceBlend(maxwell3d->regs) ? 1 : 0);
            if (!dynamic_blend) {
                graphics_key.state.RefreshBlending(*maxwell3d);
            }
            if (canonical_keys && maxwell3d->GetNativeMacroEngine().native_turing) {
                CanonicalizeKeyState(graphics_key.state, dynamic_features);
            }
            TakeCleanBlendSnapshot(); // No ranges unless by_value
        }
        if ((programs_only && programs_mode == 2) || (patch_vertex_input && vertex_mode == 2)) {
            // Verify: the full Refresh must build the same key, byte for byte. Its key is kept
            // (and its flag clears, as a miss would do them).
            const GraphicsPipelineCacheKey fast = graphics_key;
            RefreshGraphicsKey();
            ++clean.programs_verified;
            if (std::memcmp(&fast, &graphics_key, sizeof(fast)) != 0) {
                ++clean.programs_mismatches;
                if (clean.programs_mismatches_logged < 16) {
                    ++clean.programs_mismatches_logged;
                    LOG_WARNING(Render_Vulkan,
                                "Patched key (EXP52=2 / EXP55=2) differs from Refresh{} (programs "
                                "{:016x}):{}",
                                fast == graphics_key ? " outside the hashed bytes" : "",
                                fmt::join(graphics_key.unique_hashes, " "),
                                DescribeProgramsKeyDiff(fast, graphics_key));
                }
            }
        }
    } else {
        TakeCleanSnapshot();
        RefreshGraphicsKey();
    }
    if (current_pipeline) {
        GraphicsPipeline* const next{current_pipeline->Next(graphics_key)};
        if (next) {
            current_pipeline = next;
            clean.pipeline = next;
            current_pipeline->MarkUsed(frame_count);
            return BuiltPipeline(current_pipeline);
        }
    }
    // The slow path makes graphics_cache[graphics_key] current, unless it has no pipeline.
    GraphicsPipeline* const previous = current_pipeline;
    GraphicsPipeline* const result = CurrentGraphicsPipelineSlowPath();
    if (current_pipeline != previous) {
        clean.pipeline = current_pipeline;
    }
    return result;
}

void PipelineCache::RefreshGraphicsKey() {
    if (!dynamic_features.has_dynamic_vertex_input) {
        graphics_key.state.Refresh(*maxwell3d, dynamic_features);
    } else {
        // Refresh only rewrites the types when the vertex input changes, so the masked copy in
        // the key must not carry over to the next programs. Types of attributes the programs
        // don't read would split their pipelines by whatever the game left in those registers
        // (Ocarina of Time: up to 20 keys for the same shaders, each built again on the GPU
        // thread).
        graphics_key.state.attribute_types = vertex_attribute_types;
        graphics_key.state.Refresh(*maxwell3d, dynamic_features);
        vertex_attribute_types = graphics_key.state.attribute_types;
        graphics_key.state.attribute_types = VideoCommon::KeyedAttributeTypes(
            vertex_attribute_types, graphics_key.unique_hashes, vertex_attribute_masks);
    }
    // Everything CanonicalizeNative clears is rewritten by every Refresh (the attachments too:
    // without dynamic blend state Refresh always rebuilds them), so it can be cleared in place.
    if (canonical_keys && maxwell3d->GetNativeMacroEngine().native_turing) {
        CanonicalizeKeyState(graphics_key.state, dynamic_features);
    }
}

GraphicsPipeline* PipelineCache::CurrentGraphicsPipeline() {
    ++lookup_serial;
    if (const int clean_mode = PipelineCleanMode(); clean_mode != 0) {
        return CurrentGraphicsPipelineClean(clean_mode == 2);
    }
    // Switched off live: the next switch on starts from a full Refresh.
    clean.pipeline = nullptr;

    if (!RefreshStages(graphics_key.unique_hashes)) {
        current_pipeline = nullptr;
        return nullptr;
    }
    RefreshGraphicsKey();
    if (maxwell3d->GetNativeMacroEngine().native_turing && maxwell3d->regs.IsMeshEnabled() &&
        MeshKeyNormalized()) {
        // Task and mesh pipelines have no vertex input: the vertex attribute registers left by
        // earlier draws would key copies of the same pipeline. The next refresh rebuilds them.
        auto& state = graphics_key.state;
        state.attribute_types = 0;
        state.attributes = {};
        state.binding_divisors = {};
        maxwell3d->dirty.flags[Dirty::VertexInput] = true;
    }

    if (current_pipeline) {
        GraphicsPipeline* const next{current_pipeline->Next(graphics_key)};
        if (next) {
            current_pipeline = next;
            current_pipeline->MarkUsed(frame_count);
            return BuiltPipeline(current_pipeline);
        }
    }
    return CurrentGraphicsPipelineSlowPath();
}

ComputePipeline* ComputePipelineCacheEntry::Get(bool native, GPUVAddr address,
                                                const ComputePipelineCacheKey& key) const {
    if (!pipeline && native) {
        throw Shader::RuntimeError(
            "Native compute pipeline unavailable: address={:#x} shader={:#016x} pipeline={:#016x}: {}",
            address, key.unique_hash, key.Hash(),
            failure.empty() ? "No compiler diagnostic was recorded" : failure);
    }
    return pipeline.get();
}

constexpr u32 MaxComputeRebuilds = 4;

namespace {
/// Which pointer slots a native program's arguments fill depends on the area (Ocarina of Time's
/// foliage culling fills different list slots in Hyrule Field and Castle Town), so a translation
/// rebuilt in one area may lack pointers the first cached one binds. Try first the translations
/// that bind every pointer the first one binds, those binding the most first and the newest of
/// those binding as many (an older one can be from a translator that asked about fewer slots, and
/// fail), then the others in cache order. `pointer_keys(candidate)` gives a candidate's
/// MappedStoragePointerKeys (the candidates are environments, or with EDEN_NX2_PREBUILD_STREAM
/// records to read again with the keys taken at load).
template <typename Candidate, typename PointerKeys>
void OrderComputeTranslations(std::vector<Candidate>& candidates, PointerKeys&& pointer_keys) {
    if (candidates.size() < 2) {
        return;
    }
    std::vector<std::vector<u64>> keys;
    keys.reserve(candidates.size());
    for (const Candidate& candidate : candidates) {
        keys.push_back(pointer_keys(candidate));
    }
    std::vector<size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), size_t{0});
    const auto keeps_first = [&](size_t index) {
        return std::ranges::includes(keys[index], keys.front());
    };
    std::ranges::stable_sort(order, [&](size_t lhs, size_t rhs) {
        const bool lhs_keeps{keeps_first(lhs)};
        if (lhs_keeps != keeps_first(rhs)) {
            return lhs_keeps;
        }
        if (!lhs_keeps) {
            return false;
        }
        if (keys[lhs].size() != keys[rhs].size()) {
            return keys[lhs].size() > keys[rhs].size();
        }
        return lhs > rhs;
    });
    std::vector<Candidate> ordered;
    ordered.reserve(candidates.size());
    for (const size_t index : order) {
        ordered.push_back(std::move(candidates[index]));
    }
    candidates = std::move(ordered);
}
} // Anonymous namespace

bool PipelineCache::StaleNullPointerSlot(const ComputePipelineCacheEntry& entry) const {
    const auto& qmd{kepler_compute->launch_description};
    for (const auto& [index, offset] : entry.null_pointer_slots) {
        if (index >= qmd.const_buffer_config.size() ||
            ((qmd.const_buffer_enable_mask >> index) & 1) == 0) {
            continue;
        }
        const auto& cbuf{qmd.const_buffer_config[index]};
        if (offset + sizeof(u64) > cbuf.size) {
            continue; // The translator reads past the bound size as zero, too
        }
        const GPUVAddr slot{cbuf.Address() + offset};
        if (!gpu_memory->IsFullyMappedRange(slot, sizeof(u64))) {
            continue;
        }
        const GPUVAddr address{gpu_memory->Read<u64>(slot)};
        if (address != 0 && gpu_memory->IsWithinGPUAddressRange(address) &&
            gpu_memory->IsFullyMappedRange(address, 1)) {
            LOG_INFO(Render_Vulkan,
                     "Rebuilding compute program {:#x}: c{}[{:#x}] was null at translation and "
                     "now points to {:#x}",
                     kepler_compute->regs.code_loc.Address() + qmd.program_start, index, offset, address);
            return true;
        }
    }
    return false;
}

ComputePipeline* PipelineCache::CurrentComputePipeline() {

    const ShaderInfo* const shader{ComputeShader()};
    if (!shader) {
        return nullptr;
    }
    const auto& qmd{kepler_compute->launch_description};
    const ComputePipelineCacheKey key{
        .unique_hash = shader->unique_hash,
        .shared_memory_size = qmd.shared_alloc,
        .workgroup_size{qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z},
    };
    const auto [pair, is_new]{compute_cache.try_emplace(key)};
    auto& entry{pair->second};
    if (is_new) {
        if (!precompiled_compute_programs.empty()) {
            // Which archive program this is, if any, and what it was precompiled with.
            static u32 logged{};
            const auto known = precompiled_compute_programs.find(key.unique_hash);
            if (logged++ < 60) {
                LOG_INFO(Render_Vulkan,
                         "LICHII_PRECOMPILE compute miss: {:016x} shared {} workgroup {}x{}x{} local "
                         "{}; {}",
                         key.unique_hash, key.shared_memory_size, key.workgroup_size[0],
                         key.workgroup_size[1], key.workgroup_size[2],
                         qmd.local_pos_alloc + qmd.local_crs_alloc,
                         known == precompiled_compute_programs.end()
                             ? std::string{"program not in the archives"}
                             : fmt::format("archive program precompiled with shared {} workgroup "
                                           "{}x{}x{}",
                                           known->second.shared_memory_size,
                                           known->second.workgroup_size[0],
                                           known->second.workgroup_size[1],
                                           known->second.workgroup_size[2]));
            }
        }
        if (const auto evicted = evicted_compute.find(key); evicted != evicted_compute.end()) {
            // EXP51 evicted it: translated binding the pointers it bound, from the driver's
            // pipeline cache; its disk cache records are there already.
            const EvictedCompute learned{std::move(evicted->second)};
            evicted_compute.erase(evicted);
            entry = CreateComputePipeline(key, shader, learned.mapped_pointers, false);
            entry.rebuilds = (std::max)(entry.rebuilds, learned.rebuilds);
            ++compute_rebuilt_after_evict;
        } else {
            entry = CreateComputePipeline(key, shader);
        }
    } else if (entry.pending) {
        // A rebuild (below) compiles on a worker; switch once it is built (zortos293/Lichii#969:
        // the next dispatch used to wait for it, up to 2.8 s in Ocarina of Time).
        if (entry.pending->pipeline->IsBuilt()) {
            auto rebuilt{std::move(entry.pending)};
            if (!rebuilt->pipeline->BuildFailed()) {
                retired_compute_pipelines.emplace_back(scheduler.CurrentTick(),
                                                       std::move(entry.pipeline));
                entry = std::move(*rebuilt);
            } else {
                LOG_WARNING(Render_Vulkan, "Keeping compute shader {:016x}: rebuild failed",
                            key.unique_hash);
                retired_compute_pipelines.emplace_back(scheduler.CurrentTick(),
                                                       std::move(rebuilt->pipeline));
                entry.rebuilds = MaxComputeRebuilds;
            }
        }
    } else if (entry.pipeline && entry.pipeline->IsPrecompiled() && entry.pipeline->BuildFailed()) {
        // EDEN_NX2_PRECOMPILED_FAST: its kept SPIR-V did not build. Translated from the game's
        // state instead, from now on.
        precompiled_rejected.insert(key.Hash());
        ++precompiled_stats.failed;
        const u32 rebuilds = entry.rebuilds;
        retired_compute_pipelines.emplace_back(scheduler.CurrentTick(), std::move(entry.pipeline));
        entry = CreateComputePipeline(key, shader, entry.mapped_pointers);
        entry.rebuilds = rebuilds;
    } else if (!entry.null_pointer_slots.empty() && entry.rebuilds < MaxComputeRebuilds &&
               StaleNullPointerSlot(entry)) {
        // The translation dropped accesses through a pointer that was null back then. Which slots
        // the game fills depends on the area, so the new one keeps binding every pointer this one
        // binds, null now or not: a null pointer binds nothing.
        auto rebuilt{CreateComputePipeline(key, shader, entry.mapped_pointers)};
        if (rebuilt.pipeline) {
            rebuilt.rebuilds = entry.rebuilds + 1;
            if (entry.pipeline && !entry.pipeline->BuildFailed()) {
                // The current translation keeps dispatching meanwhile: for a few frames it misses
                // the accesses through the newly filled pointer, instead of a freeze.
                entry.pending = std::make_unique<ComputePipelineCacheEntry>(std::move(rebuilt));
            } else {
                if (entry.pipeline) {
                    retired_compute_pipelines.emplace_back(scheduler.CurrentTick(),
                                                           std::move(entry.pipeline));
                }
                entry = std::move(rebuilt);
            }
        } else {
            // Keep the working translation when the new one cannot be built (for example a
            // narrow access that now stays global across two buffers), and stop retrying.
            LOG_WARNING(Render_Vulkan, "Keeping compute shader {:016x}: rebuild failed: {}",
                        key.unique_hash, rebuilt.failure);
            entry.rebuilds = MaxComputeRebuilds;
        }
    } else if (!precompiled_compute.empty()) {
        // A precompiled pipeline, used for the first time: is its translation valid here?
        if (const auto it = precompiled_compute.find(key); it != precompiled_compute.end()) {
            const PrecompiledCompute precompiled = std::move(it->second);
            precompiled_compute.erase(it);
            const GPUVAddr program_base{kepler_compute->regs.code_loc.Address()};
            ComputeEnvironment env{*kepler_compute, *gpu_memory, program_base, qmd.program_start,
                                   shader->lowered_native};
            const u32 local = qmd.local_pos_alloc + qmd.local_crs_alloc;
            static const bool disabled = std::getenv("LICHII_PRECOMPILE_NO_COMPUTE") != nullptr;
            const char* reason = disabled ? "disabled (LICHII_PRECOMPILE_NO_COMPUTE)"
                                 : local > precompiled.local_memory_size
                                     ? "local memory"
                                     : env.ReplayQueries(precompiled.queries);
            if (reason) {
                ++precompiled_compute_rebuilt;
                entry = CreateComputePipeline(key, shader);
            } else {
                ++precompiled_compute_used;
            }
            static u32 logged{};
            if (reason != nullptr || logged++ < 20) {
                LOG_INFO(Render_Vulkan,
                         "LICHII_PRECOMPILE compute {:016x}: {} ({} used, {} rebuilt so far)",
                         key.unique_hash, reason ? reason : "precompiled pipeline used",
                         precompiled_compute_used, precompiled_compute_rebuilt);
            }
        }
    }
    entry.last_use_frame = frame_count;
    entry.dispatched = true;
    return entry.Get(kepler_compute->IsNativeMode(),
                     kepler_compute->regs.code_loc.Address() + qmd.program_start, key);
}

u32 TakeSkippedDrawsSinceLastFrame() noexcept {
    return skipped_since_frame.exchange(0, std::memory_order::relaxed);
}

namespace {
// LICHII_KEY_STATS: for the cached pipelines that share their shaders, counts which fixed-state
// fields differ, to tell how many pipelines exist only because of baked state.
void ReportCachedKeyStats(const std::vector<GraphicsPipelineCacheKey>& keys) {
    std::unordered_map<u64, std::vector<const GraphicsPipelineCacheKey*>> groups;
    for (const auto& key : keys) {
        u64 shaders = 0;
        for (const u64 hash : key.unique_hashes) {
            shaders = shaders * 0x100000001b3ULL ^ hash;
        }
        groups[shaders].push_back(&key);
    }
    std::map<std::string, u32> fields;
    u32 variants = 0;
    u32 groups_with_variants = 0;
    const auto note = [&](bool differs, const char* name) {
        if (differs) {
            ++fields[name];
        }
    };
    for (const auto& [shaders, members] : groups) {
        if (members.size() < 2) {
            continue;
        }
        ++groups_with_variants;
        const FixedPipelineState& a = members.front()->state;
        for (size_t index = 1; index < members.size(); ++index) {
            const FixedPipelineState& b = members[index]->state;
            ++variants;
            const u32 r1 = a.raw1 ^ b.raw1;
            const u32 r2 = a.raw2 ^ b.raw2;
            note(r1 & 0x3Fu, "raw1.dynamic_feature_flags");
            note(r1 & (1u << 6), "raw1.xfb_enabled");
            note(r1 & (1u << 7), "raw1.ndc_minus_one_to_one");
            note(r1 & (3u << 8), "raw1.polygon_mode");
            note(r1 & (0x7Fu << 10), "raw1.tessellation");
            note(r1 & (0x1Fu << 15), "raw1.patch_control_points");
            note(r1 & (1u << 20), "raw1.color_write_enable_dynamic");
            note(r1 & (1u << 21), "raw1.dual_source_blend");
            note(r1 & (0xFu << 24), "raw1.topology");
            note(r1 & (0xFu << 28), "raw1.msaa_mode");
            note(r2 & (7u << 1), "raw2.alpha_test_func");
            note(r2 & (1u << 4), "raw2.early_z");
            note(r2 & (1u << 5), "raw2.depth_enabled");
            note(r2 & (0x1Fu << 6), "raw2.depth_format");
            note(r2 & (1u << 11), "raw2.y_negate");
            note(r2 & (1u << 12), "raw2.provoking_vertex_last");
            note(r2 & (1u << 13), "raw2.conservative_raster");
            note(r2 & (1u << 14), "raw2.smooth_lines");
            note(r2 & (3u << 15), "raw2.alpha_to_coverage_one");
            note(r2 & (7u << 17), "raw2.app_stage");
            note(a.color_formats != b.color_formats, "color_formats");
            note(a.alpha_test_ref != b.alpha_test_ref, "alpha_test_ref");
            note(a.point_size != b.point_size, "point_size");
            note(a.viewport_swizzles != b.viewport_swizzles, "viewport_swizzles");
            note(a.attribute_types != b.attribute_types, "attribute_types/divisors");
            note(std::memcmp(&a.dynamic_state, &b.dynamic_state, sizeof(a.dynamic_state)) != 0,
                 "dynamic_state");
            note(std::memcmp(&a.attachments, &b.attachments, sizeof(a.attachments)) != 0,
                 "attachments(blend)");
            note(std::memcmp(&a.attributes, &b.attributes, sizeof(a.attributes)) != 0,
                 "vertex_attributes");
            note(a.binding_divisors != b.binding_divisors, "binding_divisors");
            note(a.vertex_strides != b.vertex_strides, "vertex_strides");
            note(a.depth_bounds_min != b.depth_bounds_min || a.depth_bounds_max != b.depth_bounds_max,
                 "depth_bounds");
            note(a.line_stipple_factor != b.line_stipple_factor ||
                     a.line_stipple_pattern != b.line_stipple_pattern,
                 "line_stipple");
        }
    }
    LOG_INFO(Render_Vulkan,
             "LICHII_KEY_STATS: {} cached pipelines, {} shader sets, {} sets with variants, {} "
             "variant pipelines",
             keys.size(), groups.size(), groups_with_variants, variants);
    std::vector<std::pair<u32, std::string>> sorted;
    for (const auto& [name, count] : fields) {
        sorted.emplace_back(count, name);
    }
    std::ranges::sort(sorted, std::greater{});
    for (const auto& [count, name] : sorted) {
        LOG_INFO(Render_Vulkan, "LICHII_KEY_STATS:   {:5} variants differ in {}", count, name);
    }
}
} // Anonymous namespace

void PipelineCache::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                      const VideoCore::DiskResourceLoadCallback& callback) {
    // Read here, after nx2_switches.txt, and kept for the draws' lookups.
    canonical_keys = CanonicalKeys();
    static_cast<void>(EvictAbovePipelines()); // Chosen and logged at boot
    visual_workaround_title = title_id;
    static const bool key_stats = std::getenv("LICHII_KEY_STATS") != nullptr;
    std::vector<GraphicsPipelineCacheKey> stat_keys;
    static const bool state_stats = std::getenv("LICHII_STATE_STATS") != nullptr;
    std::unordered_map<u64, u32> state_counts;
    std::unordered_map<u64, FixedPipelineState> state_examples;
    std::unordered_map<u64, u32> attribute_counts;
    std::unordered_map<u64, std::unordered_set<u64>> fragment_states;
    if (title_id == 0) {
        return;
    }
    g_disk_load_start = std::chrono::steady_clock::now();
    {
        std::error_code ec;
        const auto marker = SessionMarkerPath();
        if (std::filesystem::exists(marker, ec)) {
            LOG_WARNING(Render_Vulkan, "The previous session ended without a clean stop (its "
                                       "pipeline cache was never closed): the driver may have "
                                       "lost what it built then");
        }
        std::ofstream{marker} << "open";
        LogDriverCacheFolder(device, "at boot");
        LOG_INFO(Render_Vulkan, "Our VkPipelineCache file: {}",
                 !use_vulkan_pipeline_cache  ? "off (the driver's own cache only)"
                 : AmdAppCache(device)        ? "on for pipelines built in play "
                                                "(EDEN_NX2_AMD_APP_CACHE, up to 512 MiB)"
                                              : "on");
        LOG_INFO(Render_Vulkan, "VK_KHR_pipeline_binary: {}",
                 device.GetAvailableExtensions().contains("VK_KHR_pipeline_binary")
                     ? "offered by the driver"
                     : "not offered");
    }
    if (!pipeline_cache_filename.empty()) {
        // A later load (another title or program) stops the earlier warm-ups and waits for them,
        // so StartComputeWarm and StartGraphicsWarm can start this title's and warm_cancel is
        // only cleared once no thread of the earlier load is left.
        warm_cancel = true;
        graphics_warm_threads.clear();
        compute_warm_threads.clear();
        precompile_thread = {};
        warm_stop_callback.reset();
        compute_warm_done = true;
        // The earlier load's background builds too: its pipelines no draw took are dropped
        // (never bound, so no GPU work uses them).
        StopBackgroundPipelines();
        {
            std::scoped_lock lock{background_mutex};
            background_ready.clear();
            background_claimed.clear();
        }
        background_active.store(false, std::memory_order_release);
        background_stop.store(false, std::memory_order_relaxed);
        background_slots.store(0, std::memory_order_relaxed);
        serialization_thread.WaitForRequests();
        if (use_vulkan_pipeline_cache && !vulkan_pipeline_cache_filename.empty()) {
            SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                         CACHE_VERSION);
        }
    }

    const auto shader_dir{Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir)};
    const auto base_dir{shader_dir / fmt::format("{:016x}", title_id)};
    if (!Common::FS::CreateDir(shader_dir) || !Common::FS::CreateDir(base_dir)) {
        LOG_ERROR(Common_Filesystem, "Failed to create pipeline cache directories");
        return;
    }
    // This fork's files: its records carry task/mesh state other builds of the same cache version
    // cannot read, and their loader deletes a file it fails to read. The shared files of earlier
    // builds are copied once.
    pipeline_cache_filename =
        VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan-lichiiv4", CACHE_VERSION);
    used_pipelines_filename =
        VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan_used-lichiiv4", CACHE_VERSION);
    // This fork's files of an earlier version with the same records (52 before 2026-11-08) are
    // copied to the current version's names, header rewritten, used list and merge log with them.
    // Copied, not renamed: older builds (a rollback to the previous release) keep their cache.
    for (const u32 previous : VideoCommon::VulkanOwnPreviousVersions) {
        const auto old_cache =
            VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan-lichiiv4", previous);
        // The shader cache's records file magic (the driver blob file uses VULKAN_CACHE_MAGIC_NUMBER).
        constexpr std::array<char, 8> RecordsMagic{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};
        if (VideoCommon::CopyCompatiblePipelineCache(old_cache, pipeline_cache_filename,
                                                     RecordsMagic, previous, CACHE_VERSION)) {
            VideoCommon::CopyOwnUnversionedFile(
                VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan_used-lichiiv4", previous),
                used_pipelines_filename);
            auto old_log = old_cache;
            old_log += ".merged";
            auto new_log = pipeline_cache_filename;
            new_log += ".merged";
            VideoCommon::CopyOwnUnversionedFile(old_log, new_log);
            LOG_INFO(Render_Vulkan, "Pipeline cache of version {} copied as version {}: {}",
                     previous, CACHE_VERSION, Common::FS::PathToUTF8String(pipeline_cache_filename));
            break;
        }
    }
    for (const auto& [shared, own] :
         {std::pair{VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan", CACHE_VERSION),
                    pipeline_cache_filename},
          std::pair{VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan_used", CACHE_VERSION),
                    used_pipelines_filename}}) {
        std::error_code error;
        if (!std::filesystem::exists(own, error) && std::filesystem::exists(shared, error)) {
            std::filesystem::copy_file(shared, own, error);
            LOG_INFO(Render_Vulkan, "Copied pipeline cache {} to {}{}",
                     Common::FS::PathToUTF8String(shared), Common::FS::PathToUTF8String(own),
                     error ? fmt::format(" (failed: {})", error.message()) : std::string{});
        }
    }
    // Pipelines of zortos293/Lichii caches this cache lacks are appended each time such a cache
    // changes, and their used lists merged, so they are built like this build's own.
    MergeCompatiblePipelineCaches(base_dir, pipeline_cache_filename, used_pipelines_filename);
    used_pipelines = LoadUsedPipelines(used_pipelines_filename, MaxUsedPipelineAge(device));
    used_pipelines_saved = std::chrono::steady_clock::now();
    size_t skipped_pipelines = 0;
    size_t queued_graphics = 0;
    using GraphicsRecords = std::vector<std::vector<FileEnvironment>>;
    /// The records of one pipeline and their size in the file (for the memory log): in memory, or
    /// with EDEN_NX2_PREBUILD_STREAM where to read them again.
    struct DiskRecords {
        GraphicsRecords envs;
        std::vector<RecordRef> refs;
        u64 bytes{};
    };
    /// A compute record to read again, with what OrderComputeTranslations orders it by.
    struct StreamedTranslation {
        RecordRef ref;
        std::vector<u64> pointer_keys;
    };
    struct ComputeCandidates {
        std::vector<FileEnvironment> envs;
        std::vector<StreamedTranslation> refs;
        u64 bytes{};
    };
    std::vector<std::pair<GraphicsPipelineCacheKey, DiskRecords>> deferred_graphics;
    // Keys of graphics pipelines only another driver stored (see load_graphics).
    std::unordered_set<u64> rekeyed_pipelines;
    std::unordered_set<u64> native_pipelines;
    size_t foreign_records = 0;
    size_t canonical_rekeyed = 0; ///< EDEN_NX2_CANONICAL_KEYS=1: records whose key hash changed

    if (use_vulkan_pipeline_cache) {
        vulkan_pipeline_cache_filename =
            VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan_pipelines-lichiiv4", CACHE_VERSION);
        vulkan_pipeline_cache =
            LoadVulkanPipelineCache(vulkan_pipeline_cache_filename, CACHE_VERSION);
    }
    // Lichii: the shader objects' binaries, kept beside the pipeline cache (LICHII_SHADER_OBJECTS).
    if (UseShaderObjects(device)) {
        ShaderObjectCache::Instance().Open(
            device, pipeline_cache_filename.parent_path() / "lichii_shader_objects_v5.bin");
    }
    // LICHII_PRECOMPILE: the effect shaders built after the disk cache share its loading bar; the
    // last boot left their count here so the bar knows its full length from the start.
    loading_extra = 0;
    if (const char* dir = std::getenv("LICHII_PRECOMPILE_DIR"); dir != nullptr && *dir != 0) {
        std::ifstream hint{pipeline_cache_filename.parent_path() / "lichii_precompile_effects.count"};
        hint >> loading_extra;
    }

    struct {
        std::mutex mutex;
        size_t total{};
        size_t built{};
        bool has_loaded{};
        std::unique_ptr<PipelineStatistics> statistics;
        std::chrono::steady_clock::time_point last_save{std::chrono::steady_clock::now()};
    } state;
    // A rebuilt translation is cached beside the stale one. Choose between them before building, so
    // loading only ever adds entries (see OrderComputeTranslations), and take the next when a
    // translation fails.
    ::Common::unordered_map<ComputePipelineCacheKey, ComputeCandidates> compute_envs;
    // The same for graphics: a pipeline whose record failed to load is stored again when the GPU
    // thread builds it, and its records are tried newest first.
    std::vector<std::pair<GraphicsPipelineCacheKey, DiskRecords>> graphics_records;
    ::Common::unordered_map<GraphicsPipelineCacheKey, size_t> graphics_record_index;
    // Each record's place in the file: records follow the header (magic and version, 12 bytes)
    // back to back, and both loaders end at the record's end (after its key).
    std::streamoff record_begin = 12;
    const auto record_extent{[&record_begin](std::ifstream& file) {
        const std::streamoff end = file.tellg();
        const RecordRef extent{static_cast<u64>(record_begin),
                               static_cast<u32>(end - record_begin)};
        record_begin = end;
        return extent;
    }};
    // EDEN_NX2_PREBUILD_STREAM: the records are dropped once loading has decided what to build,
    // and read again for each build. The file is opened before the loader reads it (see
    // RecordFile); without a file there is nothing to read.
    std::shared_ptr<RecordFile> record_file;
    if (PrebuildStreamEnabled()) {
        record_file = std::make_shared<RecordFile>(pipeline_cache_filename);
        if (!record_file->IsOpen()) {
            record_file.reset();
        }
    }
    const bool streaming = record_file != nullptr;
    std::shared_ptr<RecordStream> boot_stream;

    if (device.IsKhrPipelineExecutablePropertiesEnabled()) {
        state.statistics = std::make_unique<PipelineStatistics>(device);
    }
    const auto load_compute{[&](std::ifstream& file, FileEnvironment env) {
        ComputePipelineCacheKey key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));
        RecordRef ref = record_extent(file);
        ComputeCandidates& candidates = compute_envs[key];
        candidates.bytes += ref.bytes;
        if (streaming) {
            ref.key_hash = Common::CityHash64(reinterpret_cast<const char*>(&key), sizeof(key));
            candidates.refs.push_back({ref, env.MappedStoragePointerKeys()});
        } else {
            candidates.envs.push_back(std::move(env));
        }
    }};
    const auto build_compute{[this, &state, &callback](const ComputePipelineCacheKey& key,
                                                       std::vector<FileEnvironment>& envs,
                                                       PrebuildTicket& ticket) {
        ShaderPools pools;
        ComputePipelineCacheEntry entry;
        for (auto& env : envs) {
            pools.ReleaseContents();
            entry = CreateComputePipeline(pools, key, env, state.statistics.get(), false);
            if (entry.pipeline) {
                break;
            }
        }
        ticket.Done();
        std::scoped_lock lock{state.mutex};
        if (entry.pipeline) {
            compute_cache.emplace(key, std::move(entry));
        }
        ++state.built;
        if (state.has_loaded) {
            callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
        }
    }};
    const auto queue_compute{[&](const ComputePipelineCacheKey& key,
                                 ComputeCandidates candidates) {
        if (boot_stream) {
            std::vector<RecordRef> refs;
            refs.reserve(candidates.refs.size());
            for (const StreamedTranslation& candidate : candidates.refs) {
                refs.push_back(candidate.ref);
            }
            const size_t index = boot_stream->Add(std::move(refs), sizeof(ComputePipelineCacheKey));
            workers.QueueWork([key, index, stream_ = boot_stream, ticket_ = PrebuildTicket{0},
                               build_compute]() mutable {
                u64 bytes{};
                auto records = stream_->Take(index, bytes, nullptr);
                ticket_.Adopt(bytes);
                std::vector<FileEnvironment> envs;
                envs.reserve(records.size());
                for (auto& record : records) {
                    envs.push_back(std::move(record.front()));
                }
                records.clear();
                build_compute(key, envs, ticket_);
            });
        } else {
            workers.QueueWork([key, envs_ = std::move(candidates.envs),
                               ticket_ = PrebuildTicket{candidates.bytes},
                               build_compute]() mutable { build_compute(key, envs_, ticket_); });
        }
        ++state.total;
    }};
    const auto load_graphics{[&](std::ifstream& file, std::vector<FileEnvironment> envs) {
        GraphicsPipelineCacheKey key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));
        if (key_stats) {
            stat_keys.push_back(key);
        }
        if (state_stats) {
            // LICHII_STATE_STATS: the shader-relevant fixed state of the cached pipelines.
            FixedPipelineState reuse = ReuseState(key.state);
            const u64 attributes = reuse.dynamic_vertex_input ? reuse.attribute_types : 0;
            if (reuse.dynamic_vertex_input) {
                reuse.attribute_types = 0;
            }
            const u64 state_hash =
                Common::CityHash64(reinterpret_cast<const char*>(&reuse), reuse.Size());
            ++state_counts[state_hash];
            state_examples.try_emplace(state_hash, reuse);
            ++attribute_counts[attributes];
            fragment_states[key.unique_hashes[5]].insert(state_hash);
        }

        // Task/mesh pipelines translate with their mesh state: entries of builds that did not store
        // it, or any on a device without mesh shaders, are skipped.
        if (std::ranges::any_of(envs, [this](const FileEnvironment& env) {
                return (env.ShaderStage() == Shader::Stage::Task ||
                        env.ShaderStage() == Shader::Stage::Mesh) &&
                       (!device.IsExtMeshShaderSupported() || !env.HasMeshState());
            })) {
            return;
        }
        RecordRef ref = record_extent(file);
        if (streaming) {
            // The key as stored, before the rekeying and masking below.
            ref.key_hash = Common::CityHash64(reinterpret_cast<const char*>(&key), sizeof(key));
        }
        // The mesh states and mesh pipeline states the archive precompiler translates with.
        for (const FileEnvironment& env : envs) {
            if (env.ShaderStage() != Shader::Stage::Mesh) {
                continue;
            }
            std::array<u32, 23> mesh_state{};
            std::memcpy(mesh_state.data(), env.MeshRegisters().data(), 3 * sizeof(u32));
            std::memcpy(mesh_state.data() + 3, &env.MeshPrimitiveHeader(), 20 * sizeof(u32));
            if (std::ranges::find(observed_mesh_states, mesh_state) == observed_mesh_states.end()) {
                observed_mesh_states.push_back(mesh_state);
            }
            const FixedPipelineState reuse = ReuseState(key.state);
            const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(&reuse), reuse.Size());
            auto& entry = observed_mesh_templates[hash];
            if (entry.first++ == 0) {
                entry.second = key.state;
            }
        }

        if ((key.state.extended_dynamic_state != 0) !=
                dynamic_features.has_extended_dynamic_state ||
            (key.state.extended_dynamic_state_2 != 0) !=
                dynamic_features.has_extended_dynamic_state_2 ||
            (key.state.extended_dynamic_state_2_logic_op != 0) !=
                dynamic_features.has_extended_dynamic_state_2_logic_op ||
            (key.state.extended_dynamic_state_3_blend != 0) !=
                dynamic_features.has_extended_dynamic_state_3_blend ||
            (key.state.extended_dynamic_state_3_enables != 0) !=
                dynamic_features.has_extended_dynamic_state_3_enables ||
            (key.state.color_write_enable_dynamic != 0) !=
                dynamic_features.has_color_write_enable ||
            (key.state.dynamic_vertex_input != 0) !=
                dynamic_features.has_dynamic_vertex_input) {
            return;
        }

        // The key holds the driver that stored it, so a driver update (or a cache shared by
        // someone on another driver) left every record a key no draw makes: built at boot or in
        // the background all the same, never used, while the game translated each pipeline
        // again. Nothing builds from those two fields, so the record is the same pipeline under
        // this driver; its used-list age moves to that key.
        const bool foreign = key.state.driver_id != dynamic_features.driver_id ||
                             key.state.driver_version != dynamic_features.driver_version;
        if (foreign) {
            ++foreign_records;
            if (!RekeyForeignPipelines()) {
                return;
            }
            const u64 stored_hash = key.Hash();
            key.state.driver_id = dynamic_features.driver_id;
            key.state.driver_version = dynamic_features.driver_version;
            if (const auto used = used_pipelines.find(stored_hash); used != used_pipelines.end()) {
                const u8 age = used->second;
                const auto [entry, inserted] = used_pipelines.try_emplace(key.Hash(), age);
                entry->second = (std::min)(entry->second, age);
            }
        }

        // Keys stored before they kept only the attribute types the programs read (and imported
        // cache version 53 ones) are looked up masked. The pipeline is the same: a type is only
        // dropped for an attribute no program reads. Records that now share a key are tried
        // newest first, like those of a pipeline stored again.
        if (key.state.dynamic_vertex_input != 0) {
            const u64 stored_hash = key.Hash();
            MaskKeyedAttributeTypes(key, envs);
            if (const auto used = used_pipelines.find(stored_hash); used != used_pipelines.end()) {
                const u8 age = used->second;
                const auto [entry, inserted] = used_pipelines.try_emplace(key.Hash(), age);
                entry->second = (std::min)(entry->second, age);
            }
        }

        // EDEN_NX2_CANONICAL_KEYS=1: native records are looked up as draws key them now. Records
        // stored before (or while the switch was off) are rekeyed here and their used-list age
        // moves to the canonical key, as for the masking above; records stored with it on are
        // canonical already (it is idempotent).
        if (CanonicalKeys() && std::ranges::all_of(envs, [](const FileEnvironment& env) {
                return env.IsNativeProgram();
            })) {
            const u64 stored_hash = key.Hash();
            CanonicalizeKeyState(key.state, dynamic_features);
            if (const u64 hash = key.Hash(); hash != stored_hash) {
                ++canonical_rekeyed;
                if (const auto used = used_pipelines.find(stored_hash);
                    used != used_pipelines.end()) {
                    const u8 age = used->second;
                    const auto [entry, inserted] = used_pipelines.try_emplace(hash, age);
                    entry->second = (std::min)(entry->second, age);
                }
            }
        }

        const bool key_requests_provoking_last = key.state.provoking_vertex_last != 0;
        if (key_requests_provoking_last && !dynamic_features.has_provoking_vertex_last_mode) {
            return;
        }

        const bool key_uses_transform_feedback = key.state.xfb_enabled != 0;
        if (key_uses_transform_feedback && key_requests_provoking_last &&
            !dynamic_features.has_provoking_vertex_tf_preserve) {
            return;
        }

        (foreign ? rekeyed_pipelines : native_pipelines).insert(key.Hash());
        const auto [it, inserted] = graphics_record_index.try_emplace(key, graphics_records.size());
        if (inserted) {
            graphics_records.emplace_back(key, DiskRecords{});
        }
        DiskRecords& records = graphics_records[it->second].second;
        if (streaming) {
            records.refs.push_back(ref);
        } else {
            records.envs.push_back(std::move(envs));
        }
        records.bytes += ref.bytes;
    }};
    const auto build_graphics{[this, &state, &callback](const GraphicsPipelineCacheKey& key,
                                                        GraphicsRecords& records,
                                                        PrebuildTicket& ticket) {
        auto pipeline{CreateCachedGraphicsPipeline(key, records, state.statistics.get())};
        ticket.Done();

        std::scoped_lock lock{state.mutex};
        if (pipeline) {
            graphics_cache.emplace(key, std::move(pipeline));
        }
        // A boot that builds tens of thousands of pipelines takes minutes: what it built so
        // far is written now and then, so a boot that ends early (a crash, the app killed)
        // does not leave the next one to compile everything again.
        if (VulkanPipelineCacheSaveDue(state.last_save)) {
            QueueVulkanPipelineCacheSave(false);
        }
        ++state.built;
        if (state.has_loaded) {
            callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
        }
    }};
    const auto queue_graphics{[&](const GraphicsPipelineCacheKey& key, DiskRecords records) {
        const u64 hash = key.Hash();
        disk_pipelines.insert(hash);
        if (!PreloadAllPipelines() &&
            (used_pipelines.empty() ? queued_graphics >= MaxPipelinesWithoutUsedList
                                    : !used_pipelines.contains(hash))) {
            ++skipped_pipelines;
            if (BackgroundPipelinesEnabled()) {
                deferred_graphics.emplace_back(key, std::move(records));
            }
            return;
        }
        ++queued_graphics;

        if (boot_stream) {
            const size_t index =
                boot_stream->Add(std::move(records.refs), sizeof(GraphicsPipelineCacheKey));
            workers.QueueWork([key, index, stream_ = boot_stream, ticket_ = PrebuildTicket{0},
                               build_graphics]() mutable {
                u64 bytes{};
                GraphicsRecords records_ = stream_->Take(index, bytes, nullptr);
                ticket_.Adopt(bytes);
                build_graphics(key, records_, ticket_);
            });
        } else {
            workers.QueueWork([key, records_ = std::move(records.envs),
                               ticket_ = PrebuildTicket{records.bytes}, build_graphics]() mutable {
                build_graphics(key, records_, ticket_);
            });
        }
        ++state.total;
    }};
    VideoCommon::LoadPipelines(stop_loading, pipeline_cache_filename, CACHE_VERSION, load_compute,
                               load_graphics);
    if (streaming) {
        // Read again by one thread, a few builds ahead of the boot workers.
        boot_stream = std::make_shared<RecordStream>(record_file, 2 * num_pipeline_workers + 8,
                                                     false);
    }
    for (auto& [key, records] : graphics_records) {
        std::ranges::reverse(records.envs);
        std::ranges::reverse(records.refs);
        queue_graphics(key, std::move(records));
    }
    graphics_records.clear();
    graphics_record_index.clear();
    for (auto& [key, candidates] : compute_envs) {
        OrderComputeTranslations(candidates.envs, [](const FileEnvironment& env) {
            return env.MappedStoragePointerKeys();
        });
        OrderComputeTranslations(candidates.refs, [](const StreamedTranslation& candidate) {
            return candidate.pointer_keys;
        });
        queue_compute(key, std::move(candidates));
    }
    compute_envs.clear();
    if (boot_stream) {
        boot_stream->Seal();
    }

    LOG_INFO(Render_Vulkan, "Total Pipeline Count: {}", state.total);
    if (state_stats) {
        std::vector<std::pair<u32, u64>> order;
        for (const auto& [hash, count] : state_counts) order.push_back({count, hash});
        std::sort(order.rbegin(), order.rend());
        size_t fragments_one = 0, fragments_total = fragment_states.size(), max_states = 0;
        for (const auto& [fragment, states] : fragment_states) {
            fragments_one += states.size() == 1 ? 1 : 0;
            max_states = (std::max)(max_states, states.size());
        }
        LOG_INFO(Render_Vulkan,
                 "LICHII_STATE_STATS: {} distinct shader-relevant states (vertex attribute types "
                 "aside), {} distinct attribute type sets; {} of {} fragment programs use one "
                 "state, at most {}",
                 state_counts.size(), attribute_counts.size(), fragments_one, fragments_total,
                 max_states);
        for (size_t i = 0; i < order.size() && i < 12; ++i) {
            const FixedPipelineState& s = state_examples.at(order[i].second);
            LOG_INFO(Render_Vulkan,
                     "LICHII_STATE_STATS: state {:016x} in {} pipelines: raw1 {:#x} raw2 {:#x} "
                     "topology {} msaa {} alpha_test {} ref {:#x} point {:#x} swizzles {:04x}",
                     order[i].second, order[i].first, s.raw1, s.raw2,
                     static_cast<u32>(s.topology.Value()), static_cast<u32>(s.msaa_mode.Value()),
                     static_cast<u32>(s.alpha_test_func.Value()), s.alpha_test_ref, s.point_size,
                     fmt::join(s.viewport_swizzles, ""));
        }
        std::vector<std::pair<u32, u64>> attribute_order;
        for (const auto& [types, count] : attribute_counts) attribute_order.push_back({count, types});
        std::sort(attribute_order.rbegin(), attribute_order.rend());
        for (size_t i = 0; i < attribute_order.size() && i < 8; ++i) {
            LOG_INFO(Render_Vulkan, "LICHII_STATE_STATS: attribute types {:016x} in {} pipelines",
                     attribute_order[i].second, attribute_order[i].first);
        }
    }
    if (key_stats) {
        ReportCachedKeyStats(stat_keys);
        }
    if (foreign_records != 0) {
        LOG_INFO(Render_Vulkan,
                 "{} cached graphics pipeline records were stored by another GPU driver{}",
                 foreign_records,
                 RekeyForeignPipelines() ? ", loaded under this one" : " and are skipped");
    }
    if (CanonicalKeys()) {
        LOG_INFO(Render_Vulkan,
                 "EDEN_NX2_CANONICAL_KEYS: {} cached graphics pipeline records loaded under a new "
                 "(canonical) key",
                 canonical_rekeyed);
    }
    if (PreloadAllPipelines() && queued_graphics > 10000) {
        LOG_WARNING(Render_Vulkan,
                    "EDEN_NX2_PRELOAD_ALL_PIPELINES: building and keeping all {} cached graphics "
                    "pipelines holds ~{} GB of driver memory all session",
                    queued_graphics, queued_graphics * 460 / (1024 * 1024));
    }
    if (skipped_pipelines != 0) {
        LOG_INFO(Render_Vulkan,
                 "{} cached graphics pipelines are built when first needed",
                 skipped_pipelines);
    }

    std::unique_lock lock{state.mutex};
    callback(VideoCore::LoadCallbackStage::Build, 0, state.total + loading_extra);
    state.has_loaded = true;
    lock.unlock();

    workers.WaitForRequests(stop_loading);
    LOG_INFO(Render_Vulkan, "Prebuild: {} boot builds done {:.1f} s after the disk cache load began{}",
             state.total, SecondsSinceDiskLoad(),
             boot_stream ? "; records streamed: " + StreamTotalsText(boot_stream->Totals())
                         : std::string{"; records in memory"});
    boot_stream.reset();
    // The effect shaders continue the same bar after these.
    loading_base = state.total;
    PrecompileArchives(stop_loading, callback);
    // The status bar counts pipelines built since it was last idle: what loading built is not
    // "building shaders" once the game runs.
    shader_notify.ForgetBuilt();

    const size_t background_count =
        (std::min)(deferred_graphics.size(), BackgroundPipelineLimit(queued_graphics));
    // Built newest first (from the end): this driver's own records before those only another
    // driver stored, which a merged pack appends after them.
    std::ranges::stable_partition(deferred_graphics, [&](const auto& deferred) {
        const u64 hash = deferred.first.Hash();
        return rekeyed_pipelines.contains(hash) && !native_pipelines.contains(hash);
    });
    if (background_count == 0 && !deferred_graphics.empty()) {
        LOG_INFO(Render_Vulkan,
                 "No background pipeline builds: {} cached graphics pipelines are built when first "
                 "needed",
                 deferred_graphics.size());
    }
    if (background_count != 0 && !stop_loading.stop_requested()) {
        const size_t threads = BackgroundPipelineThreads();
        LOG_INFO(Render_Vulkan,
                 "Building {} of {} cached graphics pipelines in the background ({} threads), the "
                 "rest when first needed",
                 background_count, deferred_graphics.size(), threads);
        background_left.store(background_count, std::memory_order_relaxed);
        background_start = std::chrono::steady_clock::now();
        background_last_save = background_start;
        background_active.store(true, std::memory_order_release);
        background_workers.reset(new Common::ThreadWorker(threads, "VkPipelinePrebuild", {},
                                                          Common::ThreadPlacement::Background));
        std::shared_ptr<RecordStream> background_stream;
        if (streaming) {
            // At low priority like the builds, a few builds ahead of them.
            background_stream = std::make_shared<RecordStream>(record_file, 2 * threads + 8, true);
        }
        // Newest first: the cache file appends, so the end holds what the latest sessions
        // found, which the game is likelier to need soon than its oldest entries.
        for (auto& deferred :
             std::views::reverse(deferred_graphics) | std::views::take(background_count)) {
            const size_t index =
                background_stream ? background_stream->Add(std::move(deferred.second.refs),
                                                           sizeof(GraphicsPipelineCacheKey))
                                  : 0;
            background_workers->QueueWork([this, key = deferred.first,
                                           records_ = std::move(deferred.second.envs),
                                           ticket_ = PrebuildTicket{streaming ? 0
                                                                              : deferred.second.bytes},
                                           stream_ = background_stream, index]() mutable {
                thread_local const size_t slot =
                    background_slots.fetch_add(1, std::memory_order_relaxed);
                for (;;) {
                    const int mode = QuietPrebuildMode();
                    PlacePrebuildThread(mode != 0);
                    if (mode < 2 || slot < static_cast<size_t>(mode) ||
                        background_stop.load(std::memory_order_relaxed)) {
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds{100});
                }
                bool build = !background_stop.load(std::memory_order_relaxed);
                if (build) {
                    std::scoped_lock claim_lock{background_mutex};
                    build = !background_claimed.contains(key);
                }
                if (stream_ && build) {
                    u64 bytes{};
                    records_ = stream_->Take(index, bytes, &background_stop);
                    ticket_.Adopt(bytes);
                    // Stopped while waiting for them: not a record that failed to load.
                    build = !records_.empty() || !background_stop.load(std::memory_order_relaxed);
                } else if (stream_) {
                    stream_->Drop(index);
                }
                std::unique_ptr<GraphicsPipeline> pipeline;
                if (build) {
                    pipeline = CreateCachedGraphicsPipeline(key, records_, nullptr);
                }
                ticket_.Done();
                std::unique_lock ready_lock{background_mutex};
                if (pipeline && !background_claimed.contains(key)) {
                    background_ready.emplace(key, std::move(pipeline));
                }
                // These builds fill the driver cache too, but only GPU thread builds queued its
                // writes: a session stopped before its exit save kept none of them.
                if (build && VulkanPipelineCacheSaveDue(background_last_save)) {
                    QueueVulkanPipelineCacheSave(false);
                }
                if (background_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    const auto seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - background_start).count();
                    LOG_INFO(Render_Vulkan,
                             "Background pipelines done in {:.1f} s: {} ready, {} taken by draws, "
                             "{} built by the GPU thread first",
                             seconds, background_ready.size() + background_taken, background_taken,
                             background_claimed.size());
                    LOG_INFO(Render_Vulkan,
                             "Prebuild: background done {:.1f} s after the disk cache load began{}",
                             SecondsSinceDiskLoad(),
                             stream_ ? "; records streamed: " + StreamTotalsText(stream_->Totals())
                                     : std::string{"; records in memory"});
                    background_claimed.clear();
                    background_active.store(false, std::memory_order_release);
                    ReportDiskRecordFailures();
                    if (!background_stop.load(std::memory_order_relaxed)) {
                        QueueVulkanPipelineCacheSave(true);
                    }
                }
                ready_lock.unlock();
                // A pipeline dropped here was never bound, so no GPU work can still use it.
                pipeline.reset();
            });
        }
        if (background_stream) {
            background_stream->Seal();
        }
    }
    // The streams hold the file while they still have records to read.
    record_file.reset();

    if (use_vulkan_pipeline_cache) {
        // On the serialization thread: for a cache of tens of thousands of pipelines, reading
        // the driver data (gigabytes) and writing it can hold the boot at 100% long enough to
        // look frozen. It follows any write the boot build queued.
        QueueVulkanPipelineCacheSave(true);
        last_flush = std::chrono::steady_clock::now();
    }

    if (state.statistics) {
        state.statistics->Report();
    }
    ReportDiskRecordFailures();
    if (!stop_loading.stop_requested()) {
        warm_cancel = false;
        // Stopping emulation also stops the background builds at once: they would otherwise
        // compile on (minutes for a large cache) until the cache is destroyed, competing with
        // the teardown for the CPU.
        warm_stop_callback = std::make_unique<std::stop_callback<std::function<void()>>>(
            stop_loading, std::function<void()>{[this] {
                warm_cancel = true;
                background_stop.store(true, std::memory_order_relaxed);
            }});
        StartComputeWarm(title_id);
        StartGraphicsWarm(title_id);
        StartPrecompiled(title_id, base_dir);
    }
}

namespace {
/// The warm-up file: the environment variable when set, else the title's file when it exists.
/// The warm file to load: the environment override (native narrow encoding, as the environment
/// gives it) or the title's own file, empty when neither exists.
std::filesystem::path WarmFile(const char* variable, const std::filesystem::path& title_file) {
    if (const char* const path = std::getenv(variable); path != nullptr && *path != '\0') {
        return std::filesystem::path{path};
    }
    std::error_code ec;
    if (std::filesystem::exists(title_file, ec)) {
        return title_file;
    }
    return {};
}

/// Threads for each warm-up (compute and graphics): EDEN_NX2_WARM_THREADS=N (1 to 32), else the
/// shader_warm_threads setting. They always run at low priority.
size_t WarmThreads() {
    const size_t setting =
        std::clamp<size_t>(Settings::values.shader_warm_threads.GetValue(), 1, 32);
    static const size_t forced = [] {
        constexpr size_t Default = 0;
        const char* const value = std::getenv("EDEN_NX2_WARM_THREADS");
        if (value == nullptr || *value == '\0') {
            return Default;
        }
        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || parsed < 1 || parsed > 32) {
            LOG_WARNING(Render_Vulkan,
                        "Nx2Perf: EDEN_NX2_WARM_THREADS={} is not a number from 1 to 32, using "
                        "the setting",
                        value);
            return Default;
        }
        return static_cast<size_t>(parsed);
    }();
    return forced != 0 ? forced : setting;
}

/// The "already warmed" marker: warm file size and last-write time, entry count, GPU driver
/// version and device ID, pipeline cache version and the SPIR-V settings below. Next to the warm
/// file, or in the title's shader folder when that folder is not writable.
struct WarmStamp {
    u64 size{};
    s64 write_time{};
    u64 entries{};
    u32 driver{};
    u32 device_id{};
    u32 cache_version{};
    u64 spirv_settings{};
};

/// The settings that change the SPIR-V the warm-up emits (so the driver cache entries it made),
/// packed: resolution scaling (the rescaling pass bakes its factors in), shader loop safety
/// checks, the bloom fix and renderer debug. GPU accuracy is not here: no shader emission reads
/// it. Markers written before this field no longer match, so those warm-ups run once more.
u64 SpirvSettingsStamp() {
    const auto& resolution = Settings::values.resolution_info;
    u64 bits = static_cast<u64>(resolution.up_scale & 0xFF);
    bits |= static_cast<u64>(resolution.down_shift & 0xFF) << 8;
    bits |= static_cast<u64>(resolution.active ? 1 : 0) << 16;
    bits |= static_cast<u64>(Settings::values.disable_shader_loop_safety_checks.GetValue() ? 1 : 0)
            << 17;
    bits |= static_cast<u64>(Settings::values.fix_bloom_effects.GetValue() ? 1 : 0) << 18;
    bits |= static_cast<u64>(Settings::values.renderer_debug.GetValue() ? 1 : 0) << 19;
    return bits;
}

std::optional<WarmStamp> CurrentStamp(const std::filesystem::path& file, const Device& device,
                                      u64 entries) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(file, ec);
    if (ec) {
        return std::nullopt;
    }
    const auto time = std::filesystem::last_write_time(file, ec);
    if (ec) {
        return std::nullopt;
    }
    return WarmStamp{size,
                     static_cast<s64>(time.time_since_epoch().count()),
                     entries,
                     device.GetDriverVersion(),
                     device.GetDeviceID(),
                     CACHE_VERSION,
                     SpirvSettingsStamp()};
}

std::array<std::filesystem::path, 2> MarkerPaths(const std::filesystem::path& file,
                                                 const std::filesystem::path& title_file) {
    const auto marker = VideoCommon::ShaderWarm::WarmMarkerPath(file);
    return {marker, title_file.parent_path() / marker.filename()};
}

/// Whether `marker` holds a stamp of the same file and device as `stamp` (the entry count is
/// informative).
bool StampMatches(const std::filesystem::path& marker, const WarmStamp& stamp) {
    std::ifstream in(marker);
    WarmStamp saved;
    if (!(in >> saved.size >> saved.write_time >> saved.entries >> saved.driver >>
          saved.device_id >> saved.cache_version >> saved.spirv_settings)) {
        return false;
    }
    return saved.size == stamp.size && saved.write_time == stamp.write_time &&
           saved.driver == stamp.driver && saved.device_id == stamp.device_id &&
           saved.cache_version == stamp.cache_version &&
           saved.spirv_settings == stamp.spirv_settings;
}

/// Writes `stamp` to `marker`; false when it could not be written.
bool WriteStamp(const std::filesystem::path& marker, const WarmStamp& stamp) {
    std::error_code ec;
    std::filesystem::create_directories(marker.parent_path(), ec);
    std::ofstream out(marker, std::ios::trunc);
    out << stamp.size << ' ' << stamp.write_time << ' ' << stamp.entries << ' ' << stamp.driver
        << ' ' << stamp.device_id << ' ' << stamp.cache_version << ' ' << stamp.spirv_settings
        << '\n';
    out.close();
    return static_cast<bool>(out);
}

/// True when a marker matches the file and the device (the entry count is informative).
bool WarmAlreadyDone(const std::filesystem::path& file, const std::filesystem::path& title_file,
                     const Device& device) {
    const auto stamp = CurrentStamp(file, device, 0);
    if (!stamp) {
        return false;
    }
    for (const auto& marker : MarkerPaths(file, title_file)) {
        if (StampMatches(marker, *stamp)) {
            return true;
        }
    }
    return false;
}

void WriteWarmMarker(const std::filesystem::path& file, const std::filesystem::path& title_file,
                     const Device& device, u64 entries) {
    const auto stamp = CurrentStamp(file, device, entries);
    if (!stamp) {
        return;
    }
    for (const auto& marker : MarkerPaths(file, title_file)) {
        if (WriteStamp(marker, *stamp)) {
            return;
        }
    }
}

/// The "in progress" marker (warm-compute.bin.running), next to the warm file or in the title's
/// shader folder like the done marker. Written before a warm-up's first build and removed when
/// the warm-up ends, so one found at boot means the process died during a warm-up (a driver
/// compiler crash or a recompiler assert on a guessed entry).
std::array<std::filesystem::path, 2> RunningMarkerPaths(const std::filesystem::path& file,
                                                        const std::filesystem::path& title_file) {
    const auto marker = VideoCommon::ShaderWarm::WarmRunningPath(file);
    return {marker, title_file.parent_path() / marker.filename()};
}

/// True when a warm-up of this file died on this driver: it stays skipped until the warm file
/// is rebuilt or the driver changes. A marker of another file or driver is removed.
bool WarmDiedBefore(const std::filesystem::path& file, const std::filesystem::path& title_file,
                    const Device& device) {
    const auto stamp = CurrentStamp(file, device, 0);
    for (const auto& marker : RunningMarkerPaths(file, title_file)) {
        std::error_code ec;
        if (!std::filesystem::exists(marker, ec)) {
            continue;
        }
        if (stamp && StampMatches(marker, *stamp)) {
            return true;
        }
        std::filesystem::remove(marker, ec);
    }
    return false;
}

void WriteRunningMarker(const std::filesystem::path& file, const std::filesystem::path& title_file,
                        const Device& device) {
    const auto stamp = CurrentStamp(file, device, 0);
    if (!stamp) {
        return;
    }
    for (const auto& marker : RunningMarkerPaths(file, title_file)) {
        if (WriteStamp(marker, *stamp)) {
            return;
        }
    }
}

void RemoveRunningMarkers(const std::filesystem::path& file,
                          const std::filesystem::path& title_file) {
    for (const auto& marker : RunningMarkerPaths(file, title_file)) {
        std::error_code ec;
        std::filesystem::remove(marker, ec);
    }
}

/// Lets one caller through every 250 ms (status bar updates from the warm-up threads).
struct WarmProgressThrottle {
    std::atomic<s64> last{};

    bool Due() {
        const s64 now = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
        s64 previous = last.load(std::memory_order_relaxed);
        return now - previous >= 250 &&
               last.compare_exchange_strong(previous, now, std::memory_order_relaxed);
    }
};
} // Anonymous namespace

void PipelineCache::StartComputeWarm(u64 title_id) {
    if (!compute_warm_threads.empty()) {
        return;
    }
    const std::filesystem::path warm_file =
        WarmFile("EDEN_NX2_COMPUTE_WARM", VideoCommon::ShaderWarm::ComputeWarmPath(title_id));
    if (warm_file.empty()) {
        return;
    }
    const std::string file_path = Common::FS::PathToUTF8String(warm_file); // for the log
    const char* const path = file_path.c_str();
    const std::filesystem::path title_file = VideoCommon::ShaderWarm::ComputeWarmPath(title_id);
    if (WarmAlreadyDone(warm_file, title_file, device)) {
        LOG_INFO(Render_Vulkan, "Nx2Perf compute_warm: already done for this file and driver");
        return;
    }
    if (WarmDiedBefore(warm_file, title_file, device)) {
        LOG_WARNING(Render_Vulkan,
                    "Nx2Perf compute_warm: skipped, the last one did not end (the emulator "
                    "stopped during it, a driver crash?); it runs again once {} is rebuilt or "
                    "the driver changes",
                    path);
        return;
    }
    size_t file_entries = 0;
    bool read_ok = false;
    // Read the whole file here (its own reader: the disk cache loader deletes files it rejects).
    struct WarmEntry {
        ComputePipelineCacheKey key;
        FileEnvironment env;
    };
    auto entries = std::make_shared<std::vector<WarmEntry>>();
    try {
        std::ifstream file(warm_file, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            LOG_WARNING(Render_Vulkan, "Nx2Perf compute_warm: cannot open {}", path);
            return;
        }
        file.exceptions(std::ifstream::failbit);
        const auto end{file.tellg()};
        file.seekg(0, std::ios::beg);
        std::array<char, 8> magic{};
        u32 version{};
        file.read(magic.data(), magic.size())
            .read(reinterpret_cast<char*>(&version), sizeof(version));
        constexpr std::array<char, 8> expected_magic{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};
        if (magic != expected_magic || version != CACHE_VERSION) {
            LOG_WARNING(Render_Vulkan, "Nx2Perf compute_warm: {} is not a version {} file", path,
                        CACHE_VERSION);
            return;
        }
        while (file.tellg() != end) {
            u32 num_envs{};
            file.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));
            if (num_envs != 1) {
                LOG_WARNING(Render_Vulkan, "Nx2Perf compute_warm: unexpected entry in {}", path);
                break;
            }
            if (!FileEnvironment::RecordFits(file, static_cast<u64>(end - file.tellg()))) {
                LOG_WARNING(Render_Vulkan,
                            "Nx2Perf compute_warm: entry {} of {} is larger than the file, "
                            "skipping the rest",
                            file_entries + 1, path);
                break;
            }
            WarmEntry entry;
            entry.env.Deserialize(file);
            if (entry.env.ShaderStage() != Shader::Stage::Compute) {
                LOG_WARNING(Render_Vulkan, "Nx2Perf compute_warm: non-compute entry in {}", path);
                break;
            }
            file.read(reinterpret_cast<char*>(&entry.key), sizeof(entry.key));
            ++file_entries;
            // Programs already in the disk cache were built at boot; leave them out.
            if (!compute_cache.contains(entry.key)) {
                entries->push_back(std::move(entry));
            }
        }
        read_ok = file.tellg() == end;
    } catch (const std::exception& e) {
        LOG_WARNING(Render_Vulkan, "Nx2Perf compute_warm: reading {} failed: {}", path, e.what());
    }
    if (entries->empty()) {
        if (read_ok && file_entries != 0) {
            WriteWarmMarker(warm_file, title_file, device, file_entries);
        }
        return;
    }
    struct Progress {
        std::atomic<size_t> next{};
        std::atomic<size_t> built{};
        std::atomic<size_t> failed{};
        std::atomic<size_t> running{};
        std::once_flag running_marker; ///< Written before the first build
        WarmProgressThrottle throttle;
        std::chrono::steady_clock::time_point begin{std::chrono::steady_clock::now()};
    };
    auto progress = std::make_shared<Progress>();
    static const GuessMissing guess{};
    const size_t NumThreads = WarmThreads();
    progress->running = NumThreads;
    compute_warm_done = false;
    shader_notify.BeginWarm();
    shader_notify.SetWarmProgress(1, 0, entries->size());
    LOG_INFO(Render_Vulkan, "Nx2Perf compute_warm: {} programs to build on {} low priority threads",
             entries->size(), NumThreads);
    for (size_t thread = 0; thread < NumThreads; ++thread) {
        compute_warm_threads.emplace_back([this, entries, progress, warm_file, title_file,
                                           file_entries, read_ok](std::stop_token stop) {
            Common::SetCurrentThreadName("VkComputeWarm");
            Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
            while (WaitForWarmTurn(stop)) {
                const size_t index = progress->next.fetch_add(1);
                if (index >= entries->size()) {
                    break;
                }
                std::call_once(progress->running_marker, [this, &warm_file, &title_file] {
                    WriteRunningMarker(warm_file, title_file, device);
                });
                WarmEntry& entry = (*entries)[index];
                entry.env.SetGuessMissing(&guess);
                bool ok = false;
                try {
                    ShaderPools pools;
                    // Built and destroyed here: only the driver's cache keeps it.
                    ok = CreateComputePipeline(pools, entry.key, entry.env, nullptr, false, true)
                             .pipeline != nullptr;
                } catch (const std::exception& e) {
                    LOG_DEBUG(Render_Vulkan, "compute_warm {:016x}: {}", entry.key.unique_hash,
                              e.what());
                }
                (ok ? progress->built : progress->failed).fetch_add(1);
                entry.env = FileEnvironment{};
                const size_t done = progress->built + progress->failed;
                if (progress->throttle.Due()) {
                    shader_notify.SetWarmProgress(1, done, entries->size());
                }
                if (done % 500 == 0) {
                    LOG_INFO(Render_Vulkan, "Nx2Perf compute_warm: {} of {} done", done,
                             entries->size());
                }
            }
            if (progress->running.fetch_sub(1) == 1) {
                // Ended (finished, stopped or cancelled): no build is in flight any more.
                RemoveRunningMarkers(warm_file, title_file);
                shader_notify.SetWarmProgress(0, 0, 0);
                if (read_ok && !stop.stop_requested() && !warm_cancel &&
                    progress->built + progress->failed >= entries->size()) {
                    WriteWarmMarker(warm_file, title_file, device, file_entries);
                }
                compute_warm_done = true;
                shader_notify.EndWarm();
                LOG_INFO(Render_Vulkan,
                         "Nx2Perf compute_warm: finished, {} built, {} failed, {} not reached, "
                         "{:.0f} s",
                         progress->built.load(), progress->failed.load(),
                         entries->size() - std::min(entries->size(),
                                                    progress->built + progress->failed),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                       progress->begin)
                             .count());
            }
        });
    }
}

void PipelineCache::StartGraphicsWarm(u64 title_id) {
    if (!graphics_warm_threads.empty() || !device.IsExtGraphicsPipelineLibrarySupported()) {
        return;
    }
    const std::filesystem::path warm_file =
        WarmFile("EDEN_NX2_GRAPHICS_WARM", VideoCommon::ShaderWarm::GraphicsWarmPath(title_id));
    if (warm_file.empty()) {
        return;
    }
    const std::string file_path = Common::FS::PathToUTF8String(warm_file); // for the log
    const char* const path = file_path.c_str();
    const std::filesystem::path title_file = VideoCommon::ShaderWarm::GraphicsWarmPath(title_id);
    if (WarmAlreadyDone(warm_file, title_file, device)) {
        LOG_INFO(Render_Vulkan, "Nx2Perf graphics_warm: already done for this file and driver");
        return;
    }
    if (WarmDiedBefore(warm_file, title_file, device)) {
        LOG_WARNING(Render_Vulkan,
                    "Nx2Perf graphics_warm: skipped, the last one did not end (the emulator "
                    "stopped during it, a driver crash?); it runs again once {} is rebuilt or "
                    "the driver changes",
                    path);
        return;
    }
    // Streamed: the file is 1.5 GB; workers take the next entry under the reader's lock.
    struct Reader {
        std::mutex mutex;
        std::ifstream file;
        std::streampos end{};
        bool failed{};
        std::atomic<size_t> built{};
        std::atomic<size_t> failed_builds{};
        /// Builds the driver refused: worth another warm-up, unlike entries that cannot be built.
        std::atomic<size_t> driver_failures{};
        std::atomic<size_t> running{};
        std::once_flag running_marker; ///< Written before the first build
        size_t read{}; ///< Entries taken from the file (under the mutex)
        size_t known_total{}; ///< Entry count recorded by the builder, 0 when unknown
        std::streamoff position{};
        WarmProgressThrottle throttle;
        std::chrono::steady_clock::time_point begin{};
    };
    auto reader = std::make_shared<Reader>();
    try {
        reader->file.open(warm_file, std::ios::binary | std::ios::ate);
        if (!reader->file.is_open()) {
            LOG_WARNING(Render_Vulkan, "Nx2Perf graphics_warm: cannot open {}", path);
            return;
        }
        reader->file.exceptions(std::ifstream::failbit);
        reader->end = reader->file.tellg();
        reader->known_total =
            static_cast<size_t>(VideoCommon::ShaderWarm::ReadWarmCount(warm_file));
        reader->file.seekg(0, std::ios::beg);
        std::array<char, 8> magic{};
        u32 version{};
        reader->file.read(magic.data(), magic.size())
            .read(reinterpret_cast<char*>(&version), sizeof(version));
        constexpr std::array<char, 8> expected_magic{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};
        if (magic != expected_magic || version != CACHE_VERSION) {
            LOG_WARNING(Render_Vulkan, "Nx2Perf graphics_warm: {} is not a version {} file",
                        path, CACHE_VERSION);
            return;
        }
    } catch (const std::exception& e) {
        LOG_WARNING(Render_Vulkan, "Nx2Perf graphics_warm: opening {} failed: {}", path,
                    e.what());
        return;
    }
    const size_t NumThreads = WarmThreads();
    reader->running = NumThreads;
    shader_notify.BeginWarm();
    LOG_INFO(Render_Vulkan,
             "Nx2Perf graphics_warm: {} ({} MB) queued on {} low priority threads, starts after "
             "the compute warm-up",
             path, static_cast<long long>(reader->end) / (1024 * 1024), NumThreads);
    for (size_t thread = 0; thread < NumThreads; ++thread) {
        graphics_warm_threads.emplace_back([this, reader, warm_file,
                                            title_file](std::stop_token stop) {
            Common::SetCurrentThreadName("VkGraphicsWarm");
            Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
            while (!stop.stop_requested() && !warm_cancel && !compute_warm_done) {
                std::this_thread::sleep_for(std::chrono::milliseconds{200});
            }
            {
                std::scoped_lock lock{reader->mutex};
                if (reader->begin == std::chrono::steady_clock::time_point{}) {
                    reader->begin = std::chrono::steady_clock::now();
                    shader_notify.SetWarmProgress(2, 0, reader->known_total);
                }
            }
            static const GuessMissing guess{};
            GraphicsPipelineLibraryCache::SetNoRetainOnThisThread(true);
            ShaderPools pools;
            while (WaitForWarmTurn(stop)) {
                FileEnvironment env;
                GraphicsPipelineCacheKey key{};
                {
                    std::scoped_lock lock{reader->mutex};
                    if (reader->failed || reader->file.tellg() == reader->end) {
                        break;
                    }
                    try {
                        u32 num_envs{};
                        reader->file.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));
                        if (num_envs != 1) {
                            LOG_WARNING(Render_Vulkan,
                                        "Nx2Perf graphics_warm: unexpected entry, stopping");
                            reader->failed = true;
                            break;
                        }
                        if (!FileEnvironment::RecordFits(
                                reader->file,
                                static_cast<u64>(reader->end - reader->file.tellg()))) {
                            LOG_WARNING(Render_Vulkan,
                                        "Nx2Perf graphics_warm: entry {} is larger than the "
                                        "file, skipping the rest",
                                        reader->read + 1);
                            reader->failed = true;
                            break;
                        }
                        env.Deserialize(reader->file);
                        reader->file.read(reinterpret_cast<char*>(&key), sizeof(key));
                        ++reader->read;
                        reader->position = reader->file.tellg();
                    } catch (const std::exception& e) {
                        LOG_WARNING(Render_Vulkan, "Nx2Perf graphics_warm: read failed: {}",
                                    e.what());
                        reader->failed = true;
                        break;
                    }
                }
                std::call_once(reader->running_marker, [this, &warm_file, &title_file] {
                    WriteRunningMarker(warm_file, title_file, device);
                });
                env.SetGuessMissing(&guess);
                WarmResult result = WarmResult::Unbuildable;
                try {
                    Shader::Environment* const env_ptr = &env;
                    result = WarmGraphicsLibraries(pools, key, std::span{&env_ptr, 1});
                } catch (const vk::Exception& e) {
                    LOG_DEBUG(Render_Vulkan, "graphics_warm: {}", e.what());
                    result = WarmResult::DriverFailed;
                } catch (const std::exception& e) {
                    LOG_DEBUG(Render_Vulkan, "graphics_warm: {}", e.what());
                }
                const bool ok = result == WarmResult::Built;
                (ok ? reader->built : reader->failed_builds).fetch_add(1);
                if (result == WarmResult::DriverFailed) {
                    reader->driver_failures.fetch_add(1);
                }
                const size_t done = reader->built + reader->failed_builds;
                if (reader->throttle.Due()) {
                    // The builder's recorded count, or else an estimate from the bytes read.
                    size_t total{reader->known_total};
                    if (total == 0) {
                        std::scoped_lock lock{reader->mutex};
                        const std::streamoff header = 12;
                        const std::streamoff end = reader->end;
                        if (reader->position > header) {
                            total = static_cast<size_t>(
                                static_cast<double>(reader->read) *
                                static_cast<double>(end - header) /
                                static_cast<double>(reader->position - header));
                        }
                    }
                    shader_notify.SetWarmProgress(2, done, std::max(total, done));
                }
                if (done % 2000 == 0) {
                    LOG_INFO(Render_Vulkan, "Nx2Perf graphics_warm: {} done", done);
                }
            }
            if (reader->running.fetch_sub(1) == 1) {
                // Ended (finished, stopped or cancelled): no build is in flight any more.
                RemoveRunningMarkers(warm_file, title_file);
                shader_notify.SetWarmProgress(0, 0, 0);
                bool complete = false;
                size_t entries_read = 0;
                {
                    std::scoped_lock lock{reader->mutex};
                    complete = !reader->failed && reader->end == reader->position;
                    entries_read = reader->read;
                }
                // A few builds the driver refuses every time (1% of the entries, at least 50)
                // must not keep the warm-up pending forever; more is worth another warm-up.
                const size_t driver_failures = reader->driver_failures.load();
                const size_t allowed_failures = std::max<size_t>(50, entries_read / 100);
                if (complete && driver_failures > allowed_failures) {
                    complete = false;
                    LOG_WARNING(Render_Vulkan,
                                "Nx2Perf graphics_warm: the driver refused {} builds (over {}), "
                                "the warm-up runs again next time",
                                driver_failures, allowed_failures);
                }
                if (complete && !stop.stop_requested() && !warm_cancel) {
                    WriteWarmMarker(warm_file, title_file, device, entries_read);
                }
                shader_notify.EndWarm();
                LOG_INFO(Render_Vulkan,
                         "Nx2Perf graphics_warm: finished, {} built, {} failed ({} refused by the "
                         "driver), {:.0f} s",
                         reader->built.load(), reader->failed_builds.load(), driver_failures,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                       reader->begin)
                             .count());
            }
        });
    }
}

bool PipelineCache::WaitForWarmTurn(std::stop_token stop) {
    while (!stop.stop_requested() && !warm_cancel && !shader_notify.WarmMayBuild()) {
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
    // Idle priority and off the cores pinned to the GPU threads, like the quiet prebuild.
    PlacePrebuildThread(true);
    return !stop.stop_requested() && !warm_cancel;
}

void PipelineCache::StartPrecompiled(u64 title_id, const std::filesystem::path& base_dir) {
    precompile_thread = {};
    shader_notify.SetPrecompileAvailable(false);
    const int mode = VideoCommon::Nx2Precompile::Mode();
    if (mode == 0) {
        return;
    }
    // The prepared pipelines are kept by the driver's own cache. Without one they would go to the
    // emulator's pipeline cache file, which is loaded whole at boot: only on request there.
    if (!DriverCachesPipelines(device) && mode < 2) {
        LOG_INFO(Render_Vulkan, "Precompiled pipelines skipped: the driver keeps no pipeline cache "
                                "of its own (EDEN_NX2_PRECOMPILE=2 prepares them anyway)");
        return;
    }
    // Taken here, before the GPU thread starts adding to it.
    precompile_disk_keys = std::make_shared<const std::unordered_set<u64>>(disk_pipelines);
    // Pending until the first preparation ends (or there is nothing to prepare); set before the
    // thread starts so a frontend asking right after loading sees it.
    shader_notify.SetPrecompilePending(true);
    precompile_thread = std::jthread([this, title_id, base_dir](std::stop_token stop) {
        Common::SetCurrentThreadName("VkPrecompile");
        PlacePrebuildThread(true);
        struct PendingGuard {
            VideoCore::ShaderNotify& notify;
            ~PendingGuard() {
                notify.SetPrecompilePending(false);
            }
        } pending_guard{shader_notify};
        const auto file = VideoCommon::Nx2Precompile::EnsurePrecompiledFile(title_id, base_dir,
                                                                            CACHE_VERSION, stop);
        if (file.empty() || stop.stop_requested() || warm_cancel) {
            return;
        }
        shader_notify.SetPrecompileAvailable(true);
        bool force = false;
        for (;;) {
            shader_notify.PrecompileStarted();
            PreparePrecompiled(base_dir, stop, {}, file, force);
            shader_notify.SetWarmProgress(0, 0, 0);
            shader_notify.SetPrecompilePending(false);
            shader_notify.PrecompileEnded();
            // Prepare Shaders Now prepares the file once more, whatever the stamp says.
            while (!stop.stop_requested() && !warm_cancel && !shader_notify.PrecompileRequested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds{250});
            }
            if (stop.stop_requested() || warm_cancel) {
                break;
            }
            LOG_INFO(Render_Vulkan, "Precompiled pipelines: prepared again (Prepare Shaders Now)");
            force = true;
        }
        if (shader_notify.PrecompileRequested()) {
            // A request that came with the stop: it was counted as a warm-up.
            shader_notify.PrecompileStarted();
            shader_notify.PrecompileEnded();
        }
        shader_notify.SetPrecompileAvailable(false);
    });
}

void PipelineCache::CanonicalizeNativeKey(
    GraphicsPipelineCacheKey& key,
    std::span<const Shader::Environment* const> envs) const noexcept {
    if (envs.empty() || !std::ranges::all_of(envs, [](const Shader::Environment* env) {
            return env->IsNativeProgram();
        })) {
        return;
    }
    u32 mask = ~0U;
    for (const Shader::Environment* const env : envs) {
        if (env->ShaderStage() == Shader::Stage::Task || env->ShaderStage() == Shader::Stage::Mesh) {
            mask = 0;
            break;
        }
        if (env->ShaderStage() == Shader::Stage::VertexB &&
            !IsUnmaskedVertexProgram(key.unique_hashes[1])) {
            mask = VideoCommon::GenericReadMask(env->SPH());
        }
    }
    key.state.CanonicalizeNative(mask);
}

void PipelineCache::PreparePrecompiled(const std::filesystem::path& base_dir,
                                       std::stop_token stop_loading,
                                       const VideoCore::DiskResourceLoadCallback& callback,
                                       const std::filesystem::path& file, bool force) {
    // Ported from #895 for Fritchii: it runs beside the game (precompile_thread) instead of
    // before the first frame, on threads of its own at idle priority instead of the boot's
    // pipeline workers, and reads each record from the file when it is built instead of
    // holding all of them (Ocarina of Time: 132,606 records, ~140 MB of keys).
    if (VideoCommon::Nx2Precompile::Mode() == 0) {
        return;
    }
    // Layout (tools/nx2-precompile/README.md): "NX2PREP1", u32 cache version, u32 environment
    // count, u32 record count, u32 reserved; per environment u64 offset and u64 size of a
    // GenericEnvironment::Serialize blob; per record u8 kind (0 graphics, 1 compute), u8
    // environment count, u16 reserved, u32 environment indices in stage order, then the key.
    const auto path = file.empty() ? VideoCommon::Nx2Precompile::PrecompiledPath(base_dir,
                                                                                 CACHE_VERSION)
                                   : file;
    const std::string path_text = Common::FS::PathToUTF8String(path);
    std::error_code error;
    const auto file_size = std::filesystem::file_size(path, error);
    if (error) {
        // A file built for an older cache version is not read: it has to be built again.
        for (u32 version = CACHE_VERSION; version-- > 40;) {
            const auto older = VideoCommon::Nx2Precompile::PrecompiledPath(base_dir, version);
            if (std::filesystem::exists(older, error)) {
                LOG_WARNING(Render_Vulkan,
                            "Precompiled pipelines skipped: {} is for cache version {}, this "
                            "build uses {}",
                            older.filename().string(), version, CACHE_VERSION);
                break;
            }
        }
        return;
    }
    const auto modified = std::filesystem::last_write_time(path, error).time_since_epoch().count();
    // The driver keeps what it compiled; the step runs again for another file, driver, GPU or
    // release. The release version, not the commit (2026-11-09, owner): every dev build re-ran the
    // ~20 minute preparation. A translator change inside a release only costs a few runtime
    // compiles, which the game handles anyway.
    const auto stamp_path =
        base_dir / fmt::format("precompiled_prepared.fork-v{}.txt", CACHE_VERSION);
    const std::string stamp = fmt::format("{} {} {:#x} {} {}\n", file_size, modified,
                                          device.GetDriverVersion(), Common::g_build_version,
                                          device.GetModelName());
    // EDEN_NX2_PRECOMPILED_FAST: the preparation keeps each pipeline's infos and SPIR-V too
    // (vk_precompiled_kept.h), for this release, its settings and GPU. Whether this build
    // translates them the same is checked on a sample at boot (CheckKeptSample), not stamped.
    const bool keep = PrecompiledFastMode() != 0;
    const auto kept_path =
        base_dir / fmt::format("precompiled_infos.fork-v{}.bin", CACHE_VERSION);
    auto kept_part = kept_path;
    kept_part += ".part";
    const std::string kept_stamp = keep ? KeptStamp(stamp) : std::string{};
    if (!force && !VideoCommon::Nx2Precompile::Forced()) {
        std::ifstream stamp_file(stamp_path, std::ios::binary);
        const std::string previous{std::istreambuf_iterator<char>(stamp_file), {}};
        if (previous == stamp) {
            if (!keep || LoadKept(kept_path, kept_part, path, kept_stamp, true)) {
                LOG_INFO(Render_Vulkan, "Precompiled pipelines already prepared for this driver");
                return;
            }
            LOG_INFO(Render_Vulkan,
                     "Precompiled pipelines were prepared without keeping their infos and SPIR-V "
                     "for this release and its settings, or this build translates them otherwise "
                     "(EDEN_NX2_PRECOMPILED_FAST): preparing them again");
        }
    }
    // Published again once this preparation has written its file.
    {
        std::scoped_lock lock{kept_mutex};
        kept_index.reset();
        kept_available = false;
    }

    // The environment index and where each record starts; records are read again when built.
    std::vector<std::pair<u64, u64>> env_index;
    std::vector<u64> record_offsets;
    try {
        std::ifstream in(path, std::ios::binary);
        in.exceptions(std::ifstream::failbit);
        std::array<char, 8> magic{};
        u32 version{}, env_count{}, record_count{}, reserved{};
        in.read(magic.data(), magic.size())
            .read(reinterpret_cast<char*>(&version), sizeof(version))
            .read(reinterpret_cast<char*>(&env_count), sizeof(env_count))
            .read(reinterpret_cast<char*>(&record_count), sizeof(record_count))
            .read(reinterpret_cast<char*>(&reserved), sizeof(reserved));
        if (std::string_view{magic.data(), magic.size()} != "NX2PREP1" ||
            version != CACHE_VERSION) {
            LOG_WARNING(Render_Vulkan, "Precompiled pipeline file {} has another format or version",
                        path_text);
            return;
        }
        if (u64{env_count} * 16 > file_size || u64{record_count} * 8 > file_size) {
            LOG_ERROR(Render_Vulkan, "Precompiled pipeline file {} is damaged", path_text);
            return;
        }
        env_index.resize(env_count);
        in.read(reinterpret_cast<char*>(env_index.data()),
                static_cast<std::streamsize>(env_count * sizeof(env_index[0])));
        if (std::ranges::any_of(env_index, [&](const auto& entry) {
                return entry.first > file_size || entry.second > file_size - entry.first;
            })) {
            LOG_ERROR(Render_Vulkan, "Precompiled pipeline file {} is damaged", path_text);
            return;
        }
        record_offsets.reserve(record_count);
        std::array<u32, Maxwell::MaxShaderProgram> ids{};
        for (u32 i = 0; i < record_count; ++i) {
            record_offsets.push_back(static_cast<u64>(in.tellg()));
            u8 kind{}, count{};
            u16 pad{};
            in.read(reinterpret_cast<char*>(&kind), 1)
                .read(reinterpret_cast<char*>(&count), 1)
                .read(reinterpret_cast<char*>(&pad), 2);
            const bool compute = kind == 1;
            if (count == 0 || count > ids.size() || kind > 1 || (compute && count != 1)) {
                LOG_ERROR(Render_Vulkan, "Precompiled pipeline file {} is damaged", path_text);
                return;
            }
            in.read(reinterpret_cast<char*>(ids.data()), count * sizeof(u32));
            if (std::any_of(ids.begin(), ids.begin() + count,
                            [&](u32 id) { return id >= env_count; })) {
                LOG_ERROR(Render_Vulkan, "Precompiled pipeline file {} is damaged", path_text);
                return;
            }
            in.seekg(compute ? sizeof(ComputePipelineCacheKey) : sizeof(GraphicsPipelineCacheKey),
                     std::ios::cur);
        }
    } catch (const std::ios_base::failure& e) {
        LOG_ERROR(Render_Vulkan, "Cannot read precompiled pipelines: {}", e.what());
        return;
    }

    const size_t total = record_offsets.size();
    // A checkpoint lets an interrupted step (closed, or a crash) continue where it stopped. A boot
    // that starts twice from the same record without getting further skips past it.
    const auto progress_path =
        base_dir / fmt::format("precompiled_progress.fork-v{}.txt", CACHE_VERSION);
    const size_t threads = VideoCommon::Nx2Precompile::Threads();
    size_t start_index = 0;
    u32 restarts = 0;
    if (!force) {
        std::ifstream progress_file(progress_path, std::ios::binary);
        std::string line;
        size_t next{};
        u32 count{};
        if (std::getline(progress_file, line) && line + '\n' == stamp &&
            (progress_file >> next >> count)) {
            start_index = std::min(next, total);
            restarts = count + 1;
        }
    }
    if (restarts >= 2) {
        const size_t jump = std::min(total - start_index, threads * 8);
        LOG_WARNING(Render_Vulkan,
                    "Precompiled pipelines stopped twice near record {}: skipping {} records",
                    start_index, jump);
        start_index += jump;
        restarts = 0;
    }
    // The kept file is written as the records are built. A preparation resumed from a
    // checkpoint continues the one it left; without one (it was made without keeping, or for
    // another build) the preparation starts from the first record, so every record is kept.
    PrecompiledKept::Writer writer;
    bool keeping = false;
    if (keep) {
        constexpr u64 KeptSpaceNeeded = u64{1536} << 20; // Ocarina of Time: 758 MiB
        const auto space = std::filesystem::space(base_dir, error);
        if (!error && space.available < KeptSpaceNeeded) {
            LOG_WARNING(Render_Vulkan,
                        "Precompiled pipelines: {} MiB free in {}, the kept infos and SPIR-V "
                        "need about {} MiB: not kept (EDEN_NX2_PRECOMPILED_FAST has nothing to "
                        "draw from)",
                        space.available >> 20, Common::FS::PathToUTF8String(base_dir),
                        KeptSpaceNeeded >> 20);
        } else if (!writer.Open(kept_part, kept_stamp, start_index > 0, total)) {
            LOG_WARNING(Render_Vulkan, "Precompiled pipelines: cannot write {}",
                        Common::FS::PathToUTF8String(kept_part));
        } else {
            keeping = true;
            if (start_index > 0 && writer.Resumed() == 0) {
                LOG_INFO(Render_Vulkan,
                         "Precompiled pipelines: the checkpoint at record {} has no kept infos "
                         "with it, starting from the first record",
                         start_index);
                start_index = 0;
                restarts = 0;
                keeping = writer.Open(kept_part, kept_stamp, false, total);
            } else if (writer.Resumed() != 0) {
                // The earlier run can have been another build: its entries are kept only when
                // a sample of them translates the same with this one, as for a finished file.
                bool same = false;
                if (const auto part_index =
                        writer.PartIndex(kept_part, path, env_index, record_offsets)) {
                    same = CheckKeptSample(*part_index, kept_part.filename().string());
                }
                if (same) {
                    LOG_INFO(Render_Vulkan,
                             "Precompiled pipelines: {} kept pipelines resumed from an earlier run",
                             writer.Resumed());
                } else {
                    LOG_INFO(Render_Vulkan,
                             "Precompiled pipelines: the {} pipelines kept by an earlier run are "
                             "not this build's translation, starting from the first record",
                             writer.Resumed());
                    start_index = 0;
                    restarts = 0;
                    keeping = writer.Open(kept_part, kept_stamp, false, total);
                }
            }
        }
    }
    const auto write_progress = [&](size_t next, u32 count) {
        if (keeping) {
            // The kept entries of every record before the checkpoint are on disk first.
            writer.Flush();
        }
        std::ofstream progress_file(progress_path, std::ios::binary | std::ios::trunc);
        progress_file << stamp << next << ' ' << count << '\n';
    };
    write_progress(start_index, restarts);

    struct State {
        std::mutex mutex;
        std::atomic<size_t> next{};
        std::atomic<size_t> built{};
        std::atomic<size_t> fast{};      ///< built in under FastBuild: most likely a driver cache hit
        std::atomic<size_t> failed{};
        std::atomic<size_t> other_device{};
        /// Of other_device, per dynamic state the record and this device disagree on (recorded
        /// dynamic / this device dynamic): EDS, EDS2, EDS2 logic op, EDS3 blend, EDS3 enables,
        /// color write enable, vertex input, provoking vertex last.
        std::array<std::atomic<size_t>, 16> other_device_by{};
        std::atomic<size_t> known{};     ///< graphics keys the disk cache holds
        std::atomic<size_t> finished{};
        /// Read extents of the kept translations (Shader::Info::storage_read_extents): storage
        /// descriptors and those proven, compute programs with storage descriptors and those
        /// proven in full.
        std::atomic<size_t> compute_storage{};
        std::atomic<size_t> compute_storage_proven{};
        std::atomic<size_t> compute_programs{};
        std::atomic<size_t> compute_programs_proven{};
        std::atomic<size_t> graphics_storage{};
        std::atomic<size_t> graphics_storage_proven{};
        size_t low_water{};              ///< first record not finished yet (under mutex)
        size_t checkpoint{};
        std::vector<u8> done;
        WarmProgressThrottle throttle;
    } state;
    state.done.assign(total, 0);
    state.next = start_index;
    state.finished = start_index;
    state.low_water = state.checkpoint = start_index;
    // A pipeline the driver finds in its cache builds in a few ms here (translation included);
    // a compile takes tens to hundreds.
    constexpr auto FastBuild = std::chrono::milliseconds{20};
    const auto disk_keys = precompile_disk_keys;
    const auto begin = std::chrono::steady_clock::now();
    // Threads() workers beside the game; at full speed (the frontend holds the game for the
    // preparation) one per host thread, at normal priority.
    const size_t full_threads =
        std::max<size_t>(threads, std::max(1U, std::thread::hardware_concurrency()));
    const auto complete = [&](size_t index) {
        const size_t finished = state.finished.fetch_add(1) + 1;
        {
            std::scoped_lock lock{state.mutex};
            state.done[index] = 1;
            while (state.low_water < total && state.done[state.low_water]) {
                ++state.low_water;
            }
            if (state.low_water >= state.checkpoint + 512) {
                state.checkpoint = state.low_water;
                write_progress(state.low_water, 0);
            }
        }
        if (state.throttle.Due()) {
            shader_notify.SetWarmProgress(3, finished, total);
        }
        if (callback && (finished % 64 == 0 || finished == total)) {
            callback(VideoCore::LoadCallbackStage::Build, finished, total);
        }
        if (finished % 5000 == 0) {
            // The mode too: a player's preparation ran at 66 pipelines a second while the game
            // was held for it, and the log never said whether it was at full speed
            // (fable52_answer_3).
            const bool full = shader_notify.PrecompileFullSpeed();
            LOG_INFO(Render_Vulkan,
                     "Precompiled pipelines: {} of {} done, {} built ({} fast), {} failed, {:.0f} s "
                     "({}, {} threads)",
                     finished, total, state.built.load(), state.fast.load(), state.failed.load(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
                         .count(),
                     full ? "full speed" : "beside the game", full ? full_threads : threads);
        }
    };
    const auto work = [&](std::stop_token stop, size_t worker) {
        t_precompile_build = true;
        Common::SetCurrentThreadName("VkPrecompile");
        std::optional<bool> placed_full;
        // Shader libraries built here go to the driver's cache only, like the graphics warm-up.
        GraphicsPipelineLibraryCache::SetNoRetainOnThisThread(true);
        std::ifstream in(path, std::ios::binary);
        in.exceptions(std::ifstream::failbit);
        while (!stop.stop_requested() && !stop_loading.stop_requested() && !warm_cancel) {
            const bool full = shader_notify.PrecompileFullSpeed();
            if (worker >= (full ? full_threads : threads)) {
                // Parked until full speed; done once every record is taken.
                if (state.next.load() >= total) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds{100});
                continue;
            }
            if (placed_full != full) {
                PlacePrecompileThread(full);
                placed_full = full;
            }
            const size_t index = state.next.fetch_add(1);
            if (index >= total) {
                break;
            }
            bool ok = false;
            bool skipped = false;
            const auto build_begin = std::chrono::steady_clock::now();
            try {
                in.clear();
                in.seekg(static_cast<std::streamoff>(record_offsets[index]));
                u8 kind{}, count{};
                u16 pad{};
                std::array<u32, Maxwell::MaxShaderProgram> ids{};
                in.read(reinterpret_cast<char*>(&kind), 1)
                    .read(reinterpret_cast<char*>(&count), 1)
                    .read(reinterpret_cast<char*>(&pad), 2)
                    .read(reinterpret_cast<char*>(ids.data()), count * sizeof(u32));
                const bool compute = kind == 1;
                ComputePipelineCacheKey compute_key{};
                GraphicsPipelineCacheKey record_key{};
                if (compute) {
                    in.read(reinterpret_cast<char*>(&compute_key), sizeof(compute_key));
                } else {
                    in.read(reinterpret_cast<char*>(&record_key), sizeof(record_key));
                    RecordedKeyForThisDevice(record_key);
                    const auto& key_state = record_key.state;
                    const bool other_device =
                        (key_state.extended_dynamic_state != 0) !=
                            dynamic_features.has_extended_dynamic_state ||
                        (key_state.extended_dynamic_state_2 != 0) !=
                            dynamic_features.has_extended_dynamic_state_2 ||
                        (key_state.extended_dynamic_state_2_logic_op != 0) !=
                            dynamic_features.has_extended_dynamic_state_2_logic_op ||
                        (key_state.extended_dynamic_state_3_blend != 0) !=
                            dynamic_features.has_extended_dynamic_state_3_blend ||
                        (key_state.extended_dynamic_state_3_enables != 0) !=
                            dynamic_features.has_extended_dynamic_state_3_enables ||
                        (key_state.color_write_enable_dynamic != 0) !=
                            dynamic_features.has_color_write_enable ||
                        (key_state.dynamic_vertex_input != 0) !=
                            dynamic_features.has_dynamic_vertex_input ||
                        (key_state.provoking_vertex_last != 0 &&
                         !dynamic_features.has_provoking_vertex_last_mode);
                    // Made for a device with other dynamic state, or in the disk cache, which
                    // builds it at boot or when a draw needs it (from the driver's cache then).
                    // Kept like the others when keeping: a draw of one the boot and background
                    // builds have not reached would translate it on the GPU thread.
                    if (other_device) {
                        ++state.other_device;
                        const std::array<std::pair<bool, bool>, 8> flags{{
                            {key_state.extended_dynamic_state != 0,
                             dynamic_features.has_extended_dynamic_state},
                            {key_state.extended_dynamic_state_2 != 0,
                             dynamic_features.has_extended_dynamic_state_2},
                            {key_state.extended_dynamic_state_2_logic_op != 0,
                             dynamic_features.has_extended_dynamic_state_2_logic_op},
                            {key_state.extended_dynamic_state_3_blend != 0,
                             dynamic_features.has_extended_dynamic_state_3_blend},
                            {key_state.extended_dynamic_state_3_enables != 0,
                             dynamic_features.has_extended_dynamic_state_3_enables},
                            {key_state.color_write_enable_dynamic != 0,
                             dynamic_features.has_color_write_enable},
                            {key_state.dynamic_vertex_input != 0,
                             dynamic_features.has_dynamic_vertex_input},
                            {key_state.provoking_vertex_last != 0,
                             key_state.provoking_vertex_last == 0 ||
                                 dynamic_features.has_provoking_vertex_last_mode},
                        }};
                        for (size_t flag = 0; flag < flags.size(); ++flag) {
                            if (flags[flag].first != flags[flag].second) {
                                ++state.other_device_by[flag * 2 + (flags[flag].first ? 0 : 1)];
                            }
                        }
                        skipped = true;
                    } else if (!keeping && disk_keys && disk_keys->contains(record_key.Hash())) {
                        ++state.known;
                        skipped = true;
                    }
                }
                // A compute record is built even when the disk cache holds its key: a key has a
                // record for each storage pointer state the program can be translated in.
                if (!skipped) {
                    std::vector<FileEnvironment> envs(count);
                    for (size_t i = 0; i < count; ++i) {
                        const auto [offset, size] = env_index[ids[i]];
                        in.clear();
                        in.seekg(static_cast<std::streamoff>(offset));
                        if (!FileEnvironment::RecordFits(in, size)) {
                            throw std::ios_base::failure("environment larger than its entry");
                        }
                        envs[i].Deserialize(in);
                    }
                    ShaderPools pools;
                    const std::span<const u32> env_ids{ids.data(), count};
                    if (compute) {
                        ComputeKeep kept;
                        ok = CreateComputePipeline(pools, compute_key, envs.front(), nullptr,
                                                   false, true, keeping ? &kept : nullptr)
                                 .pipeline != nullptr;
                        if (ok && keeping && !kept.spirv.empty()) {
                            std::array<std::vector<u8>, 5> infos;
                            std::array<std::vector<u32>, 5> spirv;
                            infos[0] = std::move(kept.info);
                            spirv[0] = std::move(kept.spirv);
                            writer.Keep(index, 1, 0, compute_key.Hash(), env_ids, infos, spirv);
                            if (kept.storage_descriptors != 0) {
                                state.compute_storage += kept.storage_descriptors;
                                state.compute_storage_proven += kept.proven_read_extents;
                                ++state.compute_programs;
                                if (kept.proven_read_extents == kept.storage_descriptors) {
                                    ++state.compute_programs_proven;
                                }
                            }
                        }
                    } else {
                        boost::container::static_vector<Shader::Environment*,
                                                        Maxwell::MaxShaderProgram>
                            env_ptrs;
                        for (auto& env : envs) {
                            env_ptrs.push_back(&env);
                        }
                        // Keeping: each stage emitted once, its words both the record and the
                        // module (EmitSPIRV changes the IR it emits), its info serialized then.
                        TranslationRecord kept;
                        kept.keep_infos = true;
                        const auto pipeline{CreateGraphicsPipeline(
                            pools, record_key, MakeSpan(env_ptrs), nullptr, false, nullptr,
                            nullptr, false, keeping ? &kept : nullptr)};
                        ok = pipeline && !pipeline->Failed();
                        // A pipeline drawn through the task/mesh compute fallback is not kept:
                        // it needs the fallback's setup besides its SPIR-V, so the normal path
                        // builds it (CreatePrecompiledGraphicsPipeline declines task/mesh
                        // records on a device without VK_EXT_mesh_shader).
                        if (ok && keeping && kept.complete && !kept.mesh_emulated) {
                            writer.Keep(index, 0,
                                        kept.libraries ? PrecompiledKept::FlagLibraries : u8{0},
                                        record_key.Hash(), env_ids, kept.infos, kept.spirv);
                            state.graphics_storage += kept.storage_descriptors;
                            state.graphics_storage_proven += kept.proven_read_extents;
                        }
                    }
                }
            } catch (const std::exception& e) {
                LOG_DEBUG(Render_Vulkan, "Precompiled pipeline {} failed: {}", index, e.what());
            }
            if (!skipped) {
                ++(ok ? state.built : state.failed);
                if (ok && std::chrono::steady_clock::now() - build_begin < FastBuild) {
                    ++state.fast;
                }
            }
            complete(index);
        }
    };
    LOG_INFO(Render_Vulkan,
             "Preparing {} precompiled pipelines from {}, from record {}, on {} idle threads ({} "
             "at full speed)",
             total, path_text, start_index, threads, full_threads);
    shader_notify.SetWarmProgress(3, start_index, total);
    if (callback) {
        callback(VideoCore::LoadCallbackStage::Build, start_index, total);
    }
    {
        std::vector<std::jthread> pool;
        for (size_t i = 0; i < full_threads; ++i) {
            pool.emplace_back(work, i);
        }
        // Joined here: a jthread's destructor would ask it to stop first. They stop on their
        // own with stop_loading (the precompile thread's) and warm_cancel.
        for (auto& thread : pool) {
            thread.join();
        }
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    const size_t reached = std::min(state.finished.load(), total);
    LOG_INFO(Render_Vulkan,
             "Precompiled pipelines: {} prepared ({} under {} ms each, most likely already in the "
             "driver cache), {} failed, {} skipped ({} in the disk cache, {} for another device), "
             "{} not reached, in {:.1f} s",
             state.built.load(), state.fast.load(), FastBuild.count(), state.failed.load(),
             state.known.load() + state.other_device.load(), state.known.load(),
             state.other_device.load(), total - reached, seconds);
    LogDriverCacheFolder(device, "after the preparation");
    if (state.other_device.load() != 0) {
        // Which dynamic states made records "for another device" (recorded on a GPU that has
        // them dynamic and this one static, or the reverse): AMD players skipped most of the
        // NVIDIA-recorded recipe (2026-11-09).
        static constexpr std::array<const char*, 8> names{
            "EDS", "EDS2", "EDS2 logic op", "EDS3 blend", "EDS3 enables",
            "color write enable", "vertex input", "provoking vertex last"};
        std::string by;
        for (size_t flag = 0; flag < names.size(); ++flag) {
            const size_t recorded_dynamic = state.other_device_by[flag * 2].load();
            const size_t device_dynamic = state.other_device_by[flag * 2 + 1].load();
            if (recorded_dynamic + device_dynamic != 0) {
                by += fmt::format("{}{}: {} recorded dynamic, {} dynamic here", by.empty() ? "" : "; ",
                                  names[flag], recorded_dynamic, device_dynamic);
            }
        }
        LOG_WARNING(Render_Vulkan, "Precompiled pipelines for another device: {} ({})",
                    state.other_device.load(), by);
    }
    const bool finished = !stop_loading.stop_requested() && !warm_cancel && reached >= total;
    if (keeping && !finished) {
        // Resumed with the checkpoint next time.
        writer.Close();
    } else if (keeping) {
        const auto totals = writer.GetTotals();
        if (writer.Finish(record_offsets)) {
            std::filesystem::rename(kept_part, kept_path, error);
            if (error) {
                // A pipeline built from the previous file can still hold it open: this one is
                // read where it is, and moved over it at the next boot (LoadKept).
                LOG_INFO(Render_Vulkan, "Precompiled pipelines: {} stays {} until the next boot",
                         kept_path.filename().string(), kept_part.filename().string());
            }
            LOG_INFO(Render_Vulkan,
                     "Precompiled shader infos and SPIR-V kept for {} pipelines: {} distinct "
                     "infos, {} distinct stages of SPIR-V ({} MiB compressed), {} MiB in all",
                     totals.records, totals.infos, totals.spirv, totals.spirv_bytes >> 20,
                     totals.file_bytes >> 20);
            // Recorded, not used yet: what a sync limited to the reads could cover.
            LOG_INFO(Render_Vulkan,
                     "Read extents: {} of {} compute storage descriptors proven ({} of {} "
                     "programs fully), {} of {} graphics, in the pipelines kept by this run",
                     state.compute_storage_proven.load(), state.compute_storage.load(),
                     state.compute_programs_proven.load(), state.compute_programs.load(),
                     state.graphics_storage_proven.load(), state.graphics_storage.load());
        } else {
            LOG_WARNING(Render_Vulkan, "Precompiled pipelines: writing {} failed",
                        kept_part.filename().string());
        }
    }
    if (finished) {
        std::ofstream stamp_file(stamp_path, std::ios::binary | std::ios::trunc);
        stamp_file << stamp;
        stamp_file.close();
        std::filesystem::remove(progress_path, error);
        if (keeping) {
            // Written by this build: its translation, nothing to check.
            LoadKept(kept_path, kept_part, path, kept_stamp, false);
        }
    }
}

void PipelineCache::RecordedKeyForThisDevice(GraphicsPipelineCacheKey& key) const noexcept {
    auto& key_state = key.state;
    // The driver fields only tell cache generations apart; the pipeline does not use them.
    // Records made before a driver update are built for this driver, as draws will key them.
    key_state.driver_id = dynamic_features.driver_id;
    key_state.driver_version = dynamic_features.driver_version;
    // Recorded with a dynamic logic op (NVIDIA: the OoT recipe), built where it is static (AMD's
    // Windows driver, 116373 of the recipe's 120461 records skipped, 2026-11-09): with logic op
    // off the operation does not count (CanonicalizeKeyState clears it), so the record is the
    // key a draw gives here. With logic op on the operation is unknown: left for another device.
    if (key_state.extended_dynamic_state_2_logic_op != 0 &&
        !dynamic_features.has_extended_dynamic_state_2_logic_op && CanonicalKeys() &&
        !dynamic_features.has_dynamic_state3_logic_op_enable &&
        key_state.dynamic_state.logic_op_enable == 0) {
        key_state.extended_dynamic_state_2_logic_op.Assign(0);
        key_state.dynamic_state.logic_op.Assign(0);
    }
    // Canonical as draws key them (EDEN_NX2_CANONICAL_KEYS=1), so the pipeline is created with
    // the create info a draw gives it (depth bounds included, which the tool's keys keep as
    // recorded) and the driver finds it compiled.
    if (CanonicalKeys()) {
        CanonicalizeKeyState(key_state, dynamic_features);
    }
}

std::string PipelineCache::KeptStamp(const std::string& prepared_stamp) const {
    // What is cheap to compare and decides a translation of the same environments: beyond the
    // prepared stamp (the precompiled file, driver, release and GPU), the cache version, the
    // settings SPIR-V depends on, the graphics pipeline library split (stages are emitted with
    // other bindings), the OoT visual workaround and canonical keys (the key hashes records are
    // found by). Not the exe nor the EDEN_NX2_ switches (2026-11-09, owner: every dev build and
    // every test switch prepared again): whether this build and its switches translate the
    // kept pipelines the same is checked at boot on a sample of them (CheckKeptSample).
    // NX2KEPT4: the SPIR-V emitter changed under kept modules (zortos293/Lichii#902 clip
    // distances; #820 noperspective varyings, then its revert), which the sample only sees if it hits one.
    // NX2KEPT5 (v5.6): F32->S32 conversions saturate in range (zortos293/Lichii#1051), and the
    // fragment output masks are kept with the infos (Fritchii #29, AMD horizon rainbow).
    return fmt::format("{}kept NX2KEPT5 cache {} settings {:#x} gpl {} oot {} {:016x} canonical "
                       "{}\n",
                       prepared_stamp, CACHE_VERSION, SpirvSettingsStamp(),
                       device.IsExtGraphicsPipelineLibrarySupported() ? 1 : 0,
                       oot_visual_workaround ? 1 : 0, visual_workaround_title,
                       CanonicalKeys() ? 1 : 0);
}

std::shared_ptr<const PrecompiledKept::Index> PipelineCache::KeptIndex() const {
    std::scoped_lock lock{kept_mutex};
    return kept_index;
}

bool PipelineCache::LoadKept(const std::filesystem::path& kept, const std::filesystem::path& part,
                             const std::filesystem::path& precompiled, const std::string& stamp,
                             bool check) {
    std::string why;
    std::shared_ptr<const PrecompiledKept::Index> index;
    std::error_code error;
    std::filesystem::path used = kept;
    if (std::filesystem::exists(part, error)) {
        // A finished file the preparation could not move over the previous one (still open
        // then): moved now. An unfinished one is left for the preparation to resume.
        if (PrecompiledKept::Index::Load(part, precompiled, stamp, CACHE_VERSION, why)) {
            std::filesystem::rename(part, kept, error);
            if (error) {
                used = part;
            }
        }
    }
    index = PrecompiledKept::Index::Load(used, precompiled, stamp, CACHE_VERSION, why);
    if (!index) {
        LOG_INFO(Render_Vulkan, "Precompiled fast path: no kept infos and SPIR-V to draw from ({})",
                 why);
        return false;
    }
    // Before the GPU thread can see it: a file written by another build is used only when a
    // sample of it translates the same with this one. One that differs is deleted, so the
    // preparation keeps them again (from its first record) instead of checking it every boot.
    if (check && !CheckKeptSample(*index, used.filename().string())) {
        index.reset();
        std::filesystem::remove(used, error);
        LOG_INFO(Render_Vulkan, "Precompiled fast path: no kept infos and SPIR-V to draw from "
                                "(they differ from this build's translation)");
        return false;
    }
    const u64 file_bytes = std::filesystem::file_size(used, error);
    LOG_INFO(Render_Vulkan,
             "Precompiled fast path: kept infos and SPIR-V of {} graphics and {} compute "
             "pipelines; index {:.1f} MB in RAM, {} ({} MiB) read per pipeline",
             index->GraphicsCount(), index->ComputeCount(),
             static_cast<double>(index->MemoryBytes()) / (1024.0 * 1024.0),
             used.filename().string(), file_bytes >> 20);
    {
        std::scoped_lock lock{kept_mutex};
        kept_index = std::move(index);
        kept_available = true;
    }
    return true;
}

void PipelineCache::NotePrecompiledMiss(const char* reason) {
    auto& misses = precompiled_stats.misses;
    const auto it = std::ranges::find_if(misses, [reason](const auto& entry) {
        return std::string_view{entry.first} == reason;
    });
    if (it != misses.end()) {
        ++it->second;
    } else {
        misses.emplace_back(reason, 1);
    }
}

namespace {
/// The pixel format traits a translation used: integer formats always (sampled textures), the
/// others only with descriptors that use them (#896).
u32 UsedFormatTraits(std::span<const Shader::Info* const> infos) {
    u32 traits = FileEnvironment::FormatTraitInteger;
    for (const Shader::Info* const info : infos) {
        if (!info) {
            continue;
        }
        if (std::ranges::any_of(info->image_descriptors,
                                [](const auto& desc) { return desc.is_byte_addressed; })) {
            traits |= FileEnvironment::FormatTraitByteAddressed;
        }
        if (!info->texture_buffer_descriptors.empty()) {
            traits |= FileEnvironment::FormatTraitSnorm;
        }
    }
    return traits;
}

std::atomic<u32> precompiled_worker_builds{};
std::atomic<u32> precompiled_audit_logged{};

/// What differs between a translation's stages (SPIR-V and serialized infos, per pipeline
/// stage; a compute pipeline's in stage 0) and a kept record, empty when nothing does. `stage`
/// is set to the stage that differs.
std::string KeptDifference(const PrecompiledKept::Index& index,
                           const PrecompiledKept::Record& record,
                           const std::array<std::vector<u32>, Maxwell::MaxShaderStage>& spirv,
                           const std::array<std::vector<u8>, Maxwell::MaxShaderStage>& infos,
                           size_t& stage) {
    std::vector<u8> kept_info;
    std::vector<u32> kept_spirv;
    for (stage = 0; stage < spirv.size(); ++stage) {
        const bool kept = record.spirv[stage] != PrecompiledKept::NoBlob;
        if (kept != !spirv[stage].empty()) {
            return "the stages";
        }
        if (!kept) {
            continue;
        }
        if (!index.ReadInfoBytes(record.infos[stage], kept_info) ||
            !index.ReadSpirv(record.spirv[stage], kept_spirv)) {
            return "the kept files (unreadable)";
        }
        if (kept_info != infos[stage]) {
            return "the infos";
        }
        if (kept_spirv != spirv[stage]) {
            const auto [mismatch, unused] = std::ranges::mismatch(kept_spirv, spirv[stage]);
            return fmt::format("the SPIR-V words ({} kept, {} translated, first at word {})",
                               kept_spirv.size(), spirv[stage].size(),
                               std::distance(kept_spirv.begin(), mismatch));
        }
    }
    stage = 0;
    return {};
}
} // Anonymous namespace

bool PipelineCache::AuditPrecompiled(const PrecompiledKept::Index& index,
                                     const PrecompiledKept::Record& record,
                                     std::span<Shader::Environment* const> envs, bool compute,
                                     const ComputePipelineCacheKey* compute_key) {
    const u64 key_hash = compute ? compute_key->Hash() : graphics_key.Hash();
    const auto differ = [&](size_t stage, std::string_view what) {
        if (precompiled_audit_logged.fetch_add(1, std::memory_order_relaxed) < 32) {
            LOG_ERROR(Render_Vulkan,
                      "Precompiled audit: {} pipeline {:016x} stage {}: {} differ from its "
                      "translation of the game's state; not used, the normal path builds it",
                      compute ? "compute" : "graphics", key_hash, stage, what);
        }
        return false;
    };
    // Pools of its own: nothing else on the GPU thread holds IR in them. The live programs are
    // translated and emitted once here (EmitSPIRV changes the IR it emits); the normal path,
    // when it runs after a difference, translates them again from scratch.
    if (!audit_pools) {
        audit_pools = std::make_unique<ShaderPools>();
    }
    audit_pools->ReleaseContents();
    std::array<std::vector<u32>, Maxwell::MaxShaderStage> spirv;
    std::array<std::vector<u8>, Maxwell::MaxShaderStage> infos;
    try {
        if (compute) {
            ComputeKeep keep;
            keep.translate_only = true;
            [[maybe_unused]] const auto none =
                CreateComputePipeline(*audit_pools, *compute_key, *envs[0], nullptr, false, true,
                                      &keep);
            spirv[0] = std::move(keep.spirv);
            infos[0] = std::move(keep.info);
        } else {
            TranslationRecord translated;
            translated.translate_only = true;
            translated.keep_infos = true;
            [[maybe_unused]] const auto none =
                CreateGraphicsPipeline(*audit_pools, graphics_key, envs, nullptr, false, nullptr,
                                       nullptr, false, &translated);
            if (!translated.complete) {
                return differ(0, "the pipeline (its translation failed)");
            }
            if (translated.libraries != ((record.flags & PrecompiledKept::FlagLibraries) != 0)) {
                return differ(0, "the library split");
            }
            spirv = std::move(translated.spirv);
            infos = std::move(translated.infos);
        }
    } catch (const std::exception& exception) {
        return differ(0, exception.what());
    }
    size_t stage{};
    const std::string difference = KeptDifference(index, record, spirv, infos, stage);
    if (!difference.empty()) {
        return differ(stage, difference);
    }
    return true;
}

bool PipelineCache::CheckKeptSample(const PrecompiledKept::Index& index,
                                    std::string_view file_name) {
    // 64 graphics pipelines and up to 16 compute ones, one in each equal stretch of the index
    // (sorted by key hash, so spread over the programs too). About 6 ms of translation each.
    constexpr size_t GraphicsSample = 64;
    constexpr size_t ComputeSample = 16;
    const auto begin = std::chrono::steady_clock::now();
    const u64 seed = PrecompiledKept::SampleSeed();
    std::vector<std::pair<bool, size_t>> items; // compute, position
    for (const size_t position :
         PrecompiledKept::SamplePositions(index.GraphicsCount(), GraphicsSample, seed)) {
        items.emplace_back(false, position);
    }
    for (const size_t position : PrecompiledKept::SamplePositions(
             index.ComputeCount(), ComputeSample, seed ^ 0x6B65707443505554ULL)) {
        items.emplace_back(true, position);
    }
    // What differs in one sampled pipeline, empty when it is equal. Translated from the
    // precompiled file's key and environments as the preparation did, into pools of its own,
    // and each program emitted once (EmitSPIRV changes the IR it emits).
    const auto check_one = [&](bool compute, size_t position, u64& key_hash,
                               size_t& stage) -> std::string {
        stage = 0;
        const PrecompiledKept::Record& record =
            compute ? index.AllCompute()[position] : index.AllGraphics()[position];
        key_hash = record.key_hash;
        u64 offset{};
        std::array<u8, 4> head{};
        if (!index.RecordOffset(compute, position, offset) || !index.ReadPrecompiled(offset, head)) {
            return "its precompiled record (unreadable)";
        }
        const u8 count = head[1];
        std::array<u32, 5> ids{};
        if (head[0] != (compute ? 1 : 0) || count != record.env_count || count > ids.size() ||
            !index.ReadPrecompiled(offset + 4, std::span(reinterpret_cast<u8*>(ids.data()),
                                                         count * sizeof(u32))) ||
            !std::equal(ids.begin(), ids.begin() + count, record.envs.begin())) {
            return "its precompiled record (other environments)";
        }
        const u64 key_offset = offset + 4 + count * sizeof(u32);
        ComputePipelineCacheKey compute_key{};
        GraphicsPipelineCacheKey record_key{};
        if (compute) {
            if (!index.ReadPrecompiled(key_offset,
                                       std::span(reinterpret_cast<u8*>(&compute_key),
                                                 sizeof(compute_key)))) {
                return "its precompiled record (unreadable)";
            }
            if (compute_key.Hash() != record.key_hash) {
                return "the key";
            }
        } else {
            if (!index.ReadPrecompiled(key_offset,
                                       std::span(reinterpret_cast<u8*>(&record_key),
                                                 sizeof(record_key)))) {
                return "its precompiled record (unreadable)";
            }
            RecordedKeyForThisDevice(record_key);
            if (record_key.Hash() != record.key_hash) {
                return "the key (as this device and EDEN_NX2_CANONICAL_KEYS key it)";
            }
        }
        std::vector<FileEnvironment> envs(count);
        for (size_t i = 0; i < count; ++i) {
            if (!index.ReadEnvironment(record.envs[i], envs[i])) {
                return "its recorded environments (unreadable)";
            }
        }
        // Without VK_EXT_mesh_shader a task/mesh record translates to the compute fallback's
        // form (or to no pipeline) and is never used (CreatePrecompiledGraphicsPipeline):
        // passed, so testing the fallback does not prepare every pipeline again.
        if (!compute && !device.IsExtMeshShaderSupported() &&
            std::ranges::any_of(envs, [](const FileEnvironment& env) {
                return env.ShaderStage() == Shader::Stage::Task ||
                       env.ShaderStage() == Shader::Stage::Mesh;
            })) {
            return {};
        }
        ShaderPools pools;
        std::array<std::vector<u32>, Maxwell::MaxShaderStage> spirv;
        std::array<std::vector<u8>, Maxwell::MaxShaderStage> infos;
        try {
            if (compute) {
                ComputeKeep keep;
                keep.translate_only = true;
                [[maybe_unused]] const auto none = CreateComputePipeline(
                    pools, compute_key, envs.front(), nullptr, false, true, &keep);
                spirv[0] = std::move(keep.spirv);
                infos[0] = std::move(keep.info);
            } else {
                boost::container::static_vector<Shader::Environment*, Maxwell::MaxShaderProgram>
                    env_ptrs;
                for (auto& env : envs) {
                    env_ptrs.push_back(&env);
                }
                TranslationRecord translated;
                translated.translate_only = true;
                translated.keep_infos = true;
                [[maybe_unused]] const auto none =
                    CreateGraphicsPipeline(pools, record_key, MakeSpan(env_ptrs), nullptr, false,
                                           nullptr, nullptr, false, &translated);
                if (!translated.complete) {
                    return "the pipeline (its translation failed)";
                }
                if (translated.libraries !=
                    ((record.flags & PrecompiledKept::FlagLibraries) != 0)) {
                    return "the library split";
                }
                spirv = std::move(translated.spirv);
                infos = std::move(translated.infos);
            }
        } catch (const std::exception& exception) {
            return fmt::format("the pipeline (its translation failed: {})", exception.what());
        }
        return KeptDifference(index, record, spirv, infos, stage);
    };
    struct Difference {
        bool compute;
        u64 key_hash;
        size_t stage;
        std::string what;
    };
    std::mutex mutex;
    std::vector<Difference> differences;
    std::atomic<size_t> next{};
    std::atomic<size_t> equal{};
    std::atomic<bool> stop{};
    // A few threads at normal priority: the frontend holds the game while this runs, like the
    // stamp check before it, and its "Preparing shaders" screen shows after 0.8 s.
    const size_t threads = std::clamp<size_t>(std::thread::hardware_concurrency() / 2, 1, 8);
    {
        std::vector<std::jthread> pool;
        for (size_t i = 0; i < std::min(threads, items.size()); ++i) {
            pool.emplace_back([&] {
                Common::SetCurrentThreadName("VkKeptCheck");
                PlacePrecompileThread(true);
                while (!stop.load(std::memory_order_relaxed)) {
                    const size_t item = next.fetch_add(1);
                    if (item >= items.size()) {
                        break;
                    }
                    const auto [compute, position] = items[item];
                    u64 key_hash{};
                    size_t stage{};
                    std::string what = check_one(compute, position, key_hash, stage);
                    if (what.empty()) {
                        equal.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }
                    std::scoped_lock lock{mutex};
                    differences.push_back({compute, key_hash, stage, std::move(what)});
                    // The answer is known; a few more are only for the log.
                    if (differences.size() >= 4) {
                        stop.store(true, std::memory_order_relaxed);
                    }
                }
            });
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - begin)
                          .count();
    const size_t graphics_items = std::min(index.GraphicsCount(), GraphicsSample);
    const size_t compute_items = items.size() - graphics_items;
    if (differences.empty()) {
        LOG_INFO(Render_Vulkan,
                 "Kept precompiled pipelines checked: {} of {} equal, kept file valid ({} "
                 "graphics of {} and {} compute of {} translated again by this build, {}, {:.0f} "
                 "ms on {} threads)",
                 equal.load(), items.size(), graphics_items, index.GraphicsCount(), compute_items,
                 index.ComputeCount(), file_name, ms, threads);
        return true;
    }
    for (const Difference& difference : differences) {
        LOG_WARNING(Render_Vulkan,
                    "Kept precompiled pipelines: {} pipeline {:016x} stage {}: {} differ from "
                    "this build's translation",
                    difference.compute ? "compute" : "graphics", difference.key_hash,
                    difference.stage, difference.what);
    }
    LOG_WARNING(Render_Vulkan,
                "Kept precompiled pipelines checked: {} of {} equal, {} differ before the check "
                "stopped: {} is not valid for this build, its pipelines are kept again ({:.0f} "
                "ms on {} threads)",
                equal.load(), items.size(), differences.size(), file_name, ms, threads);
    return false;
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreatePrecompiledGraphicsPipeline(
    const VideoCommon::EvictedPointerSlots::Slots* keep_pointers) {
    const int mode = PrecompiledFastMode();
    if (mode == 0 || !kept_available.load(std::memory_order_acquire)) {
        return nullptr;
    }
    const auto index = KeptIndex();
    if (!index) {
        return nullptr;
    }
    const u64 hash = graphics_key.Hash();
    const auto records = index->Graphics(hash);
    if (records.empty()) {
        ++precompiled_stats.not_precompiled;
        return nullptr;
    }
    if (precompiled_rejected.contains(hash)) {
        NotePrecompiledMiss("rejected earlier (failed build or audit)");
        return nullptr;
    }
    const auto begin = std::chrono::steady_clock::now();
    // A vertex A program is merged into vertex B: never one of these.
    if (graphics_key.unique_hashes[0] != 0) {
        NotePrecompiledMiss("vertex A");
        return nullptr;
    }
    GraphicsEnvironments environments;
    GetGraphicsEnvironments(environments, graphics_key.unique_hashes);
    KeepPointers(environments.envs, graphics_key, keep_pointers);
    const auto live = environments.Span();
    const auto stages_match = [&](const PrecompiledKept::Record& record) {
        size_t programs = 0;
        for (size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
            const bool has = graphics_key.unique_hashes[stage + 1] != 0;
            programs += has ? 1 : 0;
            if (has != (record.infos[stage] != PrecompiledKept::NoBlob) ||
                has != (record.spirv[stage] != PrecompiledKept::NoBlob)) {
                return false;
            }
        }
        return programs == record.env_count && live.size() == record.env_count;
    };
    auto infos = std::make_unique<std::array<Shader::Info, Maxwell::MaxShaderStage>>();
    std::array<const Shader::Info*, Maxwell::MaxShaderStage> info_ptrs{};
    const PrecompiledKept::Record* matched = nullptr;
    const char* reason = "stages";
    for (const PrecompiledKept::Record& record : records) {
        if (!stages_match(record)) {
            reason = "stages";
            continue;
        }
        info_ptrs = {};
        bool readable = true;
        for (size_t stage = 0; stage < Maxwell::MaxShaderStage && readable; ++stage) {
            if (record.infos[stage] == PrecompiledKept::NoBlob) {
                continue;
            }
            (*infos)[stage] = Shader::Info{};
            readable = index->ReadInfo(record.infos[stage], (*infos)[stage]);
            info_ptrs[stage] = &(*infos)[stage];
        }
        if (!readable) {
            reason = "infos unreadable";
            continue;
        }
        if (std::ranges::any_of(info_ptrs, [](const Shader::Info* info) {
                return info && info->requires_layer_emulation;
            })) {
            reason = "layer emulation";
            continue;
        }
        const u32 traits = UsedFormatTraits(info_ptrs);
        bool holds = true;
        for (size_t i = 0; i < live.size() && holds; ++i) {
            FileEnvironment recorded;
            if (!index->ReadEnvironment(record.envs[i], recorded)) {
                reason = "environments unreadable";
                holds = false;
                break;
            }
            const char* why = "";
            try {
                holds = recorded.AnswersHoldFor(*live[i], &why, traits);
            } catch (const std::exception&) {
                why = "the game's state unreadable";
                holds = false;
            }
            if (!holds) {
                reason = why;
            }
        }
        if (holds) {
            matched = &record;
            break;
        }
    }
    if (!matched) {
        NotePrecompiledMiss(reason);
        return nullptr;
    }
    // Without VK_EXT_mesh_shader (or with EDEN_NX2_FORCE_MESH_FALLBACK) a task/mesh pipeline is
    // drawn through the compute fallback, which needs the translated programs: the kept SPIR-V
    // is the native mesh form, so the normal path builds it.
    if (!device.IsExtMeshShaderSupported() &&
        std::ranges::any_of(info_ptrs, [](const Shader::Info* info) {
            return info && (info->is_mesh || info->is_task);
        })) {
        NotePrecompiledMiss("task/mesh fallback");
        return nullptr;
    }
    const bool libraries = (matched->flags & PrecompiledKept::FlagLibraries) != 0;
    if (libraries && !device.IsExtGraphicsPipelineLibrarySupported()) {
        NotePrecompiledMiss("graphics pipeline libraries");
        return nullptr;
    }
    const auto checked = std::chrono::steady_clock::now();
    if (mode == 2) {
        ++precompiled_stats.audited;
        if (!AuditPrecompiled(*index, *matched, live, false, nullptr)) {
            ++precompiled_stats.audit_differ;
            precompiled_rejected.insert(hash);
            NotePrecompiledMiss("audit differed");
            return nullptr;
        }
    }
    const auto audit_end = std::chrono::steady_clock::now();
    // The worker builds the SPIR-V the preparation kept: this build's translation of the
    // recorded environments, which the check found to give the game state's.
    GraphicsPipeline::EmitStagesFunc emit_stages =
        [this, index, blobs = matched->spirv, hashes = graphics_key.unique_hashes,
         libraries](std::array<u128, Maxwell::MaxShaderStage>& stage_hashes) {
            std::array<vk::ShaderModule, Maxwell::MaxShaderStage> modules;
            std::vector<u32> words;
            for (size_t stage = 0; stage < modules.size(); ++stage) {
                if (blobs[stage] == PrecompiledKept::NoBlob) {
                    continue;
                }
                if (!index->ReadSpirv(blobs[stage], words)) {
                    throw Shader::LogicError("Precompiled pipeline: kept SPIR-V of stage {} "
                                             "unreadable",
                                             stage);
                }
                modules[stage] = BuildStageModule(device, words, hashes[stage + 1], stage,
                                                  libraries ? &stage_hashes[stage] : nullptr);
            }
            precompiled_worker_builds.fetch_add(1, std::memory_order_relaxed);
            return modules;
        };
    GraphicsPipeline::LibraryParams library_params{
        .library_cache = libraries ? &graphics_libraries : nullptr,
    };
    auto pipeline = std::make_unique<GraphicsPipeline>(
        scheduler, buffer_cache, texture_cache, PlayPipelineCache(), &shader_notify, device,
        descriptor_pool, guest_descriptor_queue, descriptor_buffer_ring, &workers, nullptr,
        render_pass_cache, graphics_key, std::array<vk::ShaderModule, Maxwell::MaxShaderStage>{},
        info_ptrs, std::move(emit_stages), library_params);
    // As a translation's (the live environments answered the same questions): kept if evicted.
    VideoCommon::EvictedPointerSlots::Slots mapped_pointers;
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (graphics_key.unique_hashes[index] != 0) {
            mapped_pointers[index] = environments.envs[index].MappedStoragePointerKeys();
        }
    }
    pipeline->SetMappedPointers(std::move(mapped_pointers));
    precompiled_pending.push_back(graphics_key);
    ++precompiled_stats.graphics;
    // The GPU thread's own time: the check and the creation, not the audit.
    const auto elapsed = (checked - begin) + (std::chrono::steady_clock::now() - audit_end);
    precompiled_stats.gpu_time += elapsed;
    precompiled_stats.gpu_worst = (std::max)(precompiled_stats.gpu_worst, elapsed);
    // The live environments answered every recorded question (AnswersHoldFor), as a
    // translation of them would have: stored like a translated pipeline's.
    if (!pipeline_cache_filename.empty()) {
        StoreRuntimeGraphicsPipeline(graphics_key, environments.envs);
    }
    return pipeline;
}

ComputePipelineCacheEntry PipelineCache::CreatePrecompiledComputePipeline(
    const ComputePipelineCacheKey& key, Shader::Environment& env) {
    const int mode = PrecompiledFastMode();
    if (mode == 0 || !kept_available.load(std::memory_order_acquire)) {
        return {};
    }
    const auto index = KeptIndex();
    if (!index) {
        return {};
    }
    const u64 hash = key.Hash();
    const auto records = index->Compute(hash);
    if (records.empty()) {
        ++precompiled_stats.not_precompiled;
        return {};
    }
    if (precompiled_rejected.contains(hash)) {
        NotePrecompiledMiss("rejected earlier (failed build or audit)");
        return {};
    }
    const auto begin = std::chrono::steady_clock::now();
    auto info = std::make_unique<Shader::Info>();
    const PrecompiledKept::Record* matched = nullptr;
    const char* reason = "stages";
    // One record for each storage pointer state the program was prepared in.
    for (const PrecompiledKept::Record& record : records) {
        if (record.env_count != 1 || record.infos[0] == PrecompiledKept::NoBlob ||
            record.spirv[0] == PrecompiledKept::NoBlob) {
            reason = "stages";
            continue;
        }
        *info = Shader::Info{};
        if (!index->ReadInfo(record.infos[0], *info)) {
            reason = "infos unreadable";
            continue;
        }
        FileEnvironment recorded;
        if (!index->ReadEnvironment(record.envs[0], recorded)) {
            reason = "environments unreadable";
            continue;
        }
        const std::array<const Shader::Info*, 1> info_ptrs{info.get()};
        const char* why = "";
        bool holds = false;
        try {
            holds = recorded.AnswersHoldFor(env, &why, UsedFormatTraits(info_ptrs));
        } catch (const std::exception&) {
            why = "the game's state unreadable";
        }
        if (holds) {
            matched = &record;
            break;
        }
        reason = why;
    }
    if (!matched) {
        NotePrecompiledMiss(reason);
        return {};
    }
    const auto checked = std::chrono::steady_clock::now();
    if (mode == 2) {
        ++precompiled_stats.audited;
        Shader::Environment* const env_ptr = &env;
        if (!AuditPrecompiled(*index, *matched, std::span(&env_ptr, 1), true, &key)) {
            ++precompiled_stats.audit_differ;
            precompiled_rejected.insert(hash);
            NotePrecompiledMiss("audit differed");
            return {};
        }
    }
    const auto audit_end = std::chrono::steady_clock::now();
    auto make_module = [this, index, blob = matched->spirv[0], shader = key.unique_hash] {
        std::vector<u32> words;
        if (!index->ReadSpirv(blob, words)) {
            throw Shader::LogicError("Precompiled compute shader {:016x}: kept SPIR-V unreadable",
                                     shader);
        }
        SelfCheckSpirv(words, shader, "compute");
        device.SaveShader(words);
        precompiled_worker_builds.fetch_add(1, std::memory_order_relaxed);
        return BuildShader(device, words);
    };
    ComputePipelineCacheEntry entry{
        .pipeline = std::make_unique<ComputePipeline>(
            device, scheduler, PlayPipelineCache(), descriptor_pool, guest_descriptor_queue,
            descriptor_buffer_ring, &workers, nullptr, &shader_notify, *info, vk::ShaderModule{},
            key.unique_hash, std::move(make_module)),
        // The live environment answered the same questions, as the translation would have.
        .null_pointer_slots = env.IsNativeProgram() ? env.NullStoragePointerSlots()
                                                    : std::vector<std::pair<u32, u32>>{},
        .mapped_pointers =
            env.IsNativeProgram() ? env.MappedStoragePointerKeys() : std::vector<u64>{},
    };
    entry.pipeline->MarkPrecompiled();
    ++precompiled_stats.compute;
    const auto elapsed = (checked - begin) + (std::chrono::steady_clock::now() - audit_end);
    precompiled_stats.gpu_time += elapsed;
    precompiled_stats.gpu_worst = (std::max)(precompiled_stats.gpu_worst, elapsed);
    return entry;
}

void PipelineCache::CheckPrecompiledBuilds() {
    if (precompiled_pending.empty()) {
        return;
    }
    bool dropped = false;
    std::erase_if(precompiled_pending, [&](const GraphicsPipelineCacheKey& key) {
        const auto it = graphics_cache.find(key);
        if (it == graphics_cache.end() || !it->second) {
            return true;
        }
        GraphicsPipeline* const pipeline = it->second.get();
        if (pipeline->Failed()) {
            // Its kept SPIR-V did not build (logged by the worker): the next draw of the key
            // translates it from the game's state.
            precompiled_rejected.insert(key.Hash());
            ++precompiled_stats.failed;
            used_this_session.erase(pipeline);
            evicted_pipelines.emplace_back(scheduler.CurrentTick(), std::move(it->second));
            graphics_cache.erase(it);
            dropped = true;
            return true;
        }
        if (!pipeline->IsBuilt()) {
            return false;
        }
        // Built: an evicted key's kept pointer slots are done with. A failed one keeps them for
        // the translation that replaces it.
        if (evicted_pointer_slots.Size() != 0) {
            evicted_pointer_slots.Forget(key.Hash());
        }
        return true;
    });
    if (dropped) {
        for (auto& [key, pipeline] : graphics_cache) {
            if (pipeline) {
                pipeline->ClearTransitions();
            }
        }
        current_pipeline = nullptr;
        clean.pipeline = nullptr;
    }
}

void PipelineCache::ReportPrecompiled() {
    if (PrecompiledFastMode() == 0) {
        return;
    }
    auto& stats = precompiled_stats;
    const auto now = std::chrono::steady_clock::now();
    if (stats.window == std::chrono::steady_clock::time_point{}) {
        stats.window = now;
        stats.bytes_at_window = PrecompiledKept::BytesRead().load(std::memory_order_relaxed);
        return;
    }
    if (now - stats.window < std::chrono::seconds{10}) {
        return;
    }
    const u64 bytes_total = PrecompiledKept::BytesRead().load(std::memory_order_relaxed);
    const u64 bytes = bytes_total - stats.bytes_at_window;
    const u32 built = precompiled_worker_builds.exchange(0, std::memory_order_relaxed);
    if (stats.graphics + stats.compute + stats.failed + stats.not_precompiled + built != 0 ||
        !stats.misses.empty()) {
        const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
        std::string misses;
        for (const auto& [reason, count] : stats.misses) {
            misses += fmt::format("{}{} {}", misses.empty() ? "" : ", ", count, reason);
        }
        const auto index = KeptIndex();
        LOG_INFO(Render_Vulkan,
                 "Precompiled fast path in the last 10 s: {} graphics and {} compute pipelines "
                 "created from kept infos ({:.1f} ms on the GPU thread, {:.2f} ms the worst), {} "
                 "built from kept SPIR-V on workers, {} failed builds replaced, {} new keys not "
                 "precompiled; checks sent back: {}; {} KB read from the kept and precompiled "
                 "files; index {:.1f} MB in RAM",
                 stats.graphics, stats.compute, ms(stats.gpu_time), ms(stats.gpu_worst), built,
                 stats.failed, stats.not_precompiled, misses.empty() ? "none" : misses,
                 bytes / 1024,
                 index ? static_cast<double>(index->MemoryBytes()) / (1024.0 * 1024.0) : 0.0);
        if (PrecompiledFastMode() == 2) {
            LOG_INFO(Render_Vulkan,
                     "Precompiled audit in the last 10 s (EDEN_NX2_PRECOMPILED_FAST=2): {} "
                     "compared with their translation of the game's state, {} differ (never "
                     "used, built by the normal path)",
                     stats.audited, stats.audit_differ);
        }
    }
    stats.graphics = stats.compute = stats.failed = stats.not_precompiled = 0;
    stats.audited = stats.audit_differ = 0;
    stats.gpu_time = stats.gpu_worst = {};
    stats.misses.clear();
    stats.bytes_at_window = bytes_total;
    stats.window = now;
}

vk::PipelineCache& PipelineCache::PlayPipelineCache() {
    // The preparation's builds only go to the driver's own cache: the app cache would grow by
    // gigabytes. With EDEN_NX2_AMD_APP_CACHE, past 512 MiB no new pipeline goes into it either
    // (a VkPipelineCache can't evict; the file is read into RAM at every boot).
    constexpr size_t AmdAppCacheCap = size_t{512} << 20;
    if (t_precompile_build && AmdAppCache(device)) {
        return no_pipeline_cache;
    }
    if (AmdAppCache(device) &&
        last_cache_size.load(std::memory_order_relaxed) >= AmdAppCacheCap) {
        return no_pipeline_cache;
    }
    return vulkan_pipeline_cache;
}

vk::PipelineCache& PipelineCache::WarmPipelineCache() {
    static const bool app_cache = [] {
        const char* value = std::getenv("EDEN_NX2_WARM_APP_CACHE");
        return value != nullptr && value[0] == '1';
    }();
    return app_cache ? vulkan_pipeline_cache : no_pipeline_cache;
}

void PipelineCache::QueueVulkanPipelineCacheFlush() {
    if (!use_vulkan_pipeline_cache || vulkan_pipeline_cache_filename.empty()) {
        return;
    }
    if (++pipelines_since_flush < VULKAN_CACHE_FLUSH_PIPELINES) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto megabytes = last_cache_size.load(std::memory_order_relaxed) / (1024 * 1024);
    const std::chrono::seconds interval{
        std::max<size_t>(VULKAN_CACHE_FLUSH_MIN_SECONDS, megabytes)};
    if (last_flush.time_since_epoch().count() != 0 && now - last_flush < interval) {
        return;
    }
    if (flush_in_flight.load(std::memory_order_acquire)) {
        return;
    }
    pipelines_since_flush = 0;
    last_flush = now;
    QueueVulkanPipelineCacheSave(false);
}

bool PipelineCache::VulkanPipelineCacheSaveDue(std::chrono::steady_clock::time_point& last) const {
    // As QueueVulkanPipelineCacheFlush: at most once a second per MiB the cache holds, as each
    // write copies all of it.
    const auto now = std::chrono::steady_clock::now();
    const auto megabytes = last_cache_size.load(std::memory_order_relaxed) / (1024 * 1024);
    const std::chrono::seconds interval{
        std::max<size_t>(VULKAN_CACHE_FLUSH_MIN_SECONDS, megabytes)};
    if (now - last < interval) {
        return false;
    }
    last = now;
    return true;
}

void PipelineCache::QueueVulkanPipelineCacheSave(bool again_if_busy) {
    if (!use_vulkan_pipeline_cache || vulkan_pipeline_cache_filename.empty()) {
        return;
    }
    if (flush_in_flight.exchange(true, std::memory_order_acq_rel)) {
        // The write under way may have read the driver data already: one more follows it.
        if (again_if_busy) {
            save_again.store(true, std::memory_order_release);
        }
        return;
    }
    serialization_thread.QueueWork([this] {
        do {
            SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                         CACHE_VERSION);
            size_t size = 0;
            vulkan_pipeline_cache.Read(&size, nullptr);
            last_cache_size.store(size, std::memory_order_relaxed);
        } while (save_again.exchange(false, std::memory_order_acq_rel));
        flush_in_flight.store(false, std::memory_order_release);
    });
}

namespace {
// EDEN_NX2_PIPELINE_DIFF=1: for a new graphics pipeline whose shaders an earlier one uses too,
// log which parts of the fixed state differ from that one (finds state that splits the same
// draws into many pipelines, each translated on the GPU thread).
void ReportNewPipelineKey(const GraphicsPipelineCacheKey& key) {
    static const bool enabled = std::getenv("EDEN_NX2_PIPELINE_DIFF") != nullptr;
    if (!enabled) {
        return;
    }
    static std::unordered_map<u64, GraphicsPipelineCacheKey> last_by_shaders;
    static u32 logged{};
    const u64 shaders = Common::CityHash64(reinterpret_cast<const char*>(key.unique_hashes.data()),
                                           sizeof(key.unique_hashes));
    const auto [it, first] = last_by_shaders.try_emplace(shaders, key);
    if (first) {
        return;
    }
    const FixedPipelineState& now = key.state;
    const FixedPipelineState& before = it->second.state;
    std::string diffs;
    if (now.raw1 != before.raw1) diffs += fmt::format(" raw1^{:#x}", now.raw1 ^ before.raw1);
    if (now.raw2 != before.raw2) diffs += fmt::format(" raw2^{:#x}", now.raw2 ^ before.raw2);
    if (now.color_formats != before.color_formats) {
        diffs += fmt::format(" color_formats {:02x} -> {:02x}", fmt::join(before.color_formats, ""),
                             fmt::join(now.color_formats, ""));
    }
    if (now.alpha_test_ref != before.alpha_test_ref) {
        diffs += fmt::format(" alpha_test_ref {:#x} -> {:#x} (func {})", before.alpha_test_ref,
                             now.alpha_test_ref, now.alpha_test_func.Value());
    }
    if (now.point_size != before.point_size) {
        diffs += fmt::format(" point_size {:#x} -> {:#x}", before.point_size, now.point_size);
    }
    if (now.viewport_swizzles != before.viewport_swizzles) diffs += " viewport_swizzles";
    if (now.attribute_types != before.attribute_types) {
        diffs += fmt::format(" attribute_types^{:#x}", now.attribute_types ^ before.attribute_types);
    }
    if (std::memcmp(&now.dynamic_state, &before.dynamic_state, sizeof(now.dynamic_state)) != 0) {
        diffs += " dynamic_state";
    }
    if (now.driver_id != before.driver_id || now.driver_version != before.driver_version) {
        diffs += " driver";
    }
    const size_t size = now.Size();
    if (diffs.empty() && size > offsetof(FixedPipelineState, attachments)) {
        diffs = fmt::format(" past attachments (key size {})", size);
    }
    it->second = key;
    if (logged++ < 400) {
        LOG_INFO(Render_Vulkan, "New pipeline for known shaders {:016x} differs in:{}", shaders,
                 diffs.empty() ? std::string{" nothing in the key prefix"} : diffs);
    }
}

// Without VK_EXT_mesh_shader: logs, once per shader set, what a skipped task/mesh pipeline would
// need from an emulation (task stage, per-primitive outputs, topology, output sizes).
template <typename Programs>
void ReportUnsupportedMeshPipeline(const GraphicsPipelineCacheKey& key, const Programs& programs) {
    static std::mutex mutex;
    static std::unordered_set<u64> seen;
    static u32 with_task{}, per_primitive{}, non_triangle{}, total{};
    u64 shaders{};
    for (const u64 hash : key.unique_hashes) {
        shaders = shaders * 31 + hash;
    }
    std::scoped_lock lock{mutex};
    if (!seen.insert(shaders).second) {
        return;
    }
    const Shader::IR::Program* task{};
    const Shader::IR::Program* mesh{};
    bool has_pixel{};
    for (size_t index = 0; index < programs.size(); ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        const auto& program{programs[index]};
        if (program.stage == Shader::Stage::Task) {
            task = &program;
        } else if (program.stage == Shader::Stage::Mesh) {
            mesh = &program;
        } else if (program.stage == Shader::Stage::Fragment) {
            has_pixel = true;
        }
    }
    ++total;
    if (mesh == nullptr) {
        LOG_WARNING(Render_Vulkan, "Mesh emulation survey: pipeline {:016x} without a mesh stage",
                    shaders);
        return;
    }
    const auto& info{mesh->mesh};
    u32 position{}, generic{}, other{};
    std::string others;
    for (const u32 line : info.vertex_lines) {
        const auto attribute{static_cast<Shader::IR::Attribute>(line)};
        if (attribute >= Shader::IR::Attribute::PositionX &&
            attribute <= Shader::IR::Attribute::PositionW) {
            ++position;
        } else if (Shader::IR::IsGeneric(attribute)) {
            ++generic;
        } else {
            ++other;
            others += fmt::format(" {}", line);
        }
    }
    with_task += task != nullptr;
    per_primitive += !info.primitive_lines.empty();
    non_triangle += info.topology != 4;
    LOG_WARNING(Render_Vulkan,
                "Mesh emulation survey #{}: task={} (payload {} words, wg {}x{}x{}) mesh wg "
                "{}x{}x{} max_vertices={} max_primitives={} topology={} vertex_lines={} "
                "(position {} generic {} other {}:{}) primitive_lines={} pixel={} | totals: "
                "task {} per-primitive {} non-triangle {}",
                total, task != nullptr, info.payload_words, task ? task->workgroup_size[0] : 0,
                task ? task->workgroup_size[1] : 0, task ? task->workgroup_size[2] : 0,
                mesh->workgroup_size[0], mesh->workgroup_size[1], mesh->workgroup_size[2],
                info.max_vertices, info.max_primitives, info.topology, info.vertex_lines.size(),
                position, generic, other, others, info.primitive_lines.size(), has_pixel,
                with_task, per_primitive, non_triangle);
}
} // Anonymous namespace

namespace {
u64 HashPrograms(const std::array<u64, Maxwell::MaxShaderProgram>& unique_hashes);
} // Anonymous namespace

GraphicsPipeline* PipelineCache::CurrentGraphicsPipelineSlowPath() {
    // Translating a new pipeline runs on this thread, so once a frame has spent its translation
    // budget (LICHII_TRANSLATE_BUDGET_MS), skippable draws that need one wait for a later frame.
    if (GetTranslateBudget().count() > 0 &&
        frame_translate_left_ns.load(std::memory_order::relaxed) <= 0 &&
        graphics_cache.find(graphics_key) == graphics_cache.end() && CanSkipCurrentDraw()) {
        deferred_new_pipelines.fetch_add(1, std::memory_order::relaxed);
        skipped_since_frame.fetch_add(1, std::memory_order::relaxed);
        return nullptr;
    }
    const auto [pair, is_new]{graphics_cache.try_emplace(graphics_key)};
    auto& pipeline{pair->second};
    if (is_new) {
        ReportNewPipelineKey(graphics_key);
        pipeline = TakeBackgroundPipeline(graphics_key);
        // A key evicted earlier (EvictUnusedPipelines) is translated binding the storage buffer
        // pointers it bound, whatever their slots hold now: one null or unmapped around an area
        // load would otherwise read zeros for the rest of the session (from zortos293/Lichii#1143).
        // Kept until a pipeline of the key is made: a queued translation may be queued again.
        const auto* const keep_pointers = pipeline ? nullptr : KeptPointers();
        if (keep_pointers) {
            for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
                if (graphics_key.unique_hashes[index] != 0) {
                    for (const u64 slot : (*keep_pointers)[index]) {
                        WarnKeptPointer(index, slot);
                    }
                }
            }
        }
        bool precompiled = false;
        if (!pipeline) {
            // EDEN_NX2_PRECOMPILED_FAST: a precompiled pipeline whose recorded answers hold is
            // created from its kept infos (no translation) and built from its kept SPIR-V on a
            // worker; its draws wait or are skipped by the usual rule (BuiltPipeline).
            pipeline = CreatePrecompiledGraphicsPipeline(keep_pointers);
            precompiled = pipeline != nullptr;
        }
        if (!pipeline) {
            // EDEN_NX2_ASYNC_TRANSLATE=1/2: a draw that may be skipped has its new pipeline
            // translated on a translation thread, and is skipped (as for an unbuilt pipeline)
            // until it is ready. The entry stays empty meanwhile.
            // Lichii: a key the archive precompiler (or an earlier translation) already holds is
            // reused at once on this thread instead of queued (TryReuseGraphicsPipeline).
            bool reusable = false;
            {
                std::scoped_lock lock{reusable_stages_mutex};
                reusable = reusable_stages.contains(HashPrograms(graphics_key.unique_hashes));
            }
            if (!reusable && translate_queue && EnqueueTranslation(keep_pointers)) {
                NoteSkippedLookup();
                ++translate_stats.skipped;
                return nullptr;
            }
            const auto begin = std::chrono::steady_clock::now();
            pipeline = CreateGraphicsPipeline(keep_pointers);
            frame_translate_left_ns.fetch_sub(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - begin)
                    .count(),
                std::memory_order::relaxed);
        }
        // A precompiled one's are forgotten once it builds (CheckPrecompiledBuilds)
        if (pipeline && !precompiled && evicted_pointer_slots.Size() != 0) {
            evicted_pointer_slots.Forget(graphics_key.Hash());
        }
    } else if (!pipeline && translate_queue) {
        bool pending{};
        pipeline = AdoptTranslation(pending);
        if (pending) {
            return nullptr;
        }
        if (pipeline && evicted_pointer_slots.Size() != 0) {
            evicted_pointer_slots.Forget(graphics_key.Hash());
        }
    }
    if (!pipeline) {
        return nullptr;
    }
    const auto save_interval = VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::PipelineRetention)
                                   ? std::chrono::steady_clock::duration{std::chrono::seconds{15}}
                                   : std::chrono::steady_clock::duration{std::chrono::minutes{2}};
    if (used_this_session.insert(pipeline.get()).second &&
        std::chrono::steady_clock::now() - used_pipelines_saved > save_interval) {
        SaveUsedPipelines();
    }
    if (current_pipeline) {
        current_pipeline->AddTransition(pipeline.get());
    }
    current_pipeline = pipeline.get();
    current_pipeline->MarkUsed(frame_count);
    return BuiltPipeline(current_pipeline);
}

PipelineCache::WarmResult PipelineCache::WarmGraphicsLibraries(
    ShaderPools& pools, const GraphicsPipelineCacheKey& key,
    std::span<Shader::Environment* const> envs) {
    if (!device.IsExtGraphicsPipelineLibrarySupported()) {
        return WarmResult::Unbuildable;
    }
    pools.ReleaseContents();
    const auto pipeline{
        CreateGraphicsPipeline(pools, key, envs, nullptr, false, nullptr, nullptr, true)};
    if (!pipeline) {
        return WarmResult::Unbuildable;
    }
    return pipeline->Failed() ? WarmResult::DriverFailed : WarmResult::Built;
}

namespace {
// The worst frame of the last 10 s (TickFrame runs once per presented frame, on the GPU thread)
// and what the GPU thread spent on pipelines in it, so a hitch reads as "2 builds, 14 ms
// translation, 0 held" without a profiler. Held draws are counted when the worker waited for them
// (they belong to the frame being recorded, close enough). EDEN_NX2_PIPELINE_TIMING=1 also logs
// every frame over 33 ms.
void ReportWorstFrame() {
    struct Frame {
        std::chrono::steady_clock::duration time{};
        FrameTally tally{};
        std::chrono::nanoseconds held{};
        u64 skipped{};
    };
    static const bool every_slow_frame = [] {
        const char* value = std::getenv("EDEN_NX2_PIPELINE_TIMING");
        return value != nullptr && value[0] == '1';
    }();
    static std::chrono::steady_clock::time_point last_tick{};
    static std::chrono::steady_clock::time_point window{};
    static std::chrono::nanoseconds last_held{};
    static u64 last_skipped{};
    static Frame worst{};
    static u32 frames{};
    static u32 slow_frames{};
    static u32 logged_slow{};
    const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
    const auto describe = [&](const Frame& frame) {
        return fmt::format("{} pipelines built on the GPU thread in {:.1f} ms ({:.1f} ms "
                           "environments, {:.1f} ms translation), {:.1f} ms of draws held, {} "
                           "lookups skipped, {} translations queued in {:.1f} ms",
                           frame.tally.count, ms(frame.tally.builds),
                           ms(frame.tally.split.environments), ms(frame.tally.split.translation),
                           ms(frame.held), frame.skipped, frame.tally.queued,
                           ms(frame.tally.queued_time));
    };
    const auto now = std::chrono::steady_clock::now();
    const auto held_total = GraphicsPipeline::HoldTimeTotal();
    const u64 skipped_total = skipped_lookups.load(std::memory_order_relaxed);
    if (last_tick == std::chrono::steady_clock::time_point{}) {
        window = now;
    } else {
        const Frame frame{now - last_tick, frame_tally, held_total - last_held,
                          skipped_total - last_skipped};
        ++frames;
        if (frame.time > std::chrono::milliseconds{33}) {
            ++slow_frames;
            if (every_slow_frame && logged_slow < 2000) {
                ++logged_slow;
                LOG_INFO(Render_Vulkan, "Slow frame: {:.1f} ms; {}", ms(frame.time),
                         describe(frame));
            }
        }
        if (frame.time > worst.time) {
            worst = frame;
        }
    }
    last_tick = now;
    last_held = held_total;
    last_skipped = skipped_total;
    frame_tally = {};
    if (now - window >= std::chrono::seconds{10}) {
        LOG_INFO(Render_Vulkan,
                 "Worst frame in the last 10 s: {:.1f} ms ({} of {} frames over 33 ms); in it {}",
                 ms(worst.time), slow_frames, frames, describe(worst));
        worst = {};
        frames = 0;
        slow_frames = 0;
        window = now;
    }
}
} // Anonymous namespace

namespace {
/// EDEN_NX2_TRANSLATE_PROFILE=1 (zortos293/Lichii#896's profiler, off by default): every 10 s the
/// time each translation step took on all threads, the largest first.
void ReportTranslateProfile() {
    if (!Shader::TranslateProfile::Enabled()) {
        return;
    }
    static auto window = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - window < std::chrono::seconds{10}) {
        return;
    }
    window = now;
    namespace Profile = Shader::TranslateProfile;
    const Profile::Totals totals = Profile::Take();
    std::array<size_t, Profile::NumSteps> order{};
    std::iota(order.begin(), order.end(), size_t{0});
    std::ranges::sort(order, [&](size_t lhs, size_t rhs) {
        return totals.nanoseconds[lhs] > totals.nanoseconds[rhs];
    });
    std::string steps;
    for (const size_t step : order) {
        if (totals.counts[step] == 0) {
            continue;
        }
        steps += fmt::format("{}{} {:.1f} ms ({})", steps.empty() ? "" : ", ",
                             Profile::Name(static_cast<Profile::Step>(step)),
                             static_cast<double>(totals.nanoseconds[step]) / 1e6,
                             totals.counts[step]);
    }
    if (!steps.empty()) {
        LOG_INFO(Render_Vulkan, "Translation profile in the last 10 s (all threads): {}", steps);
    }
}
} // Anonymous namespace

void PipelineCache::TickFrame() {
    ++frame_count;
    frame_wait_left_ns.store(GetDiscardWaitBudget().count(), std::memory_order::relaxed);
    frame_translate_left_ns.store(GetTranslateBudget().count(), std::memory_order::relaxed);
    ReportWorstFrame();
    ReportAsyncDiscard();
    ReportTranslateProfile();
    if (PrecompiledFastMode() != 0) {
        CheckPrecompiledBuilds();
        ReportPrecompiled();
    }
    if (translate_queue) {
        TickTranslations();
    }
    if (frame_count % 60 == 0) {
        static std::chrono::steady_clock::time_point last_memory_log{};
        const auto now = std::chrono::steady_clock::now();
        if (now - last_memory_log >= std::chrono::seconds{10}) {
            last_memory_log = now;
            LogMemoryUse();
        }
    }
    if (device.IsExtGraphicsPipelineLibrarySupported()) {
        graphics_libraries.Report();
    }
    while (!evicted_pipelines.empty() && scheduler.IsFree(evicted_pipelines.front().first)) {
        evicted_pipelines.pop_front();
    }
    // A retired translation's build may still be queued on the pipeline workers: destroying it
    // now would wait for that build, or run it here on the GPU thread.
    std::erase_if(retired_translations, [this](const auto& entry) {
        return scheduler.IsFree(entry.first) && entry.second->IsBuilt();
    });
    while (!retired_compute_pipelines.empty() &&
           scheduler.IsFree(retired_compute_pipelines.front().first) &&
           retired_compute_pipelines.front().second->IsBuildTaskDone()) {
        retired_compute_pipelines.pop_front();
    }
    constexpr u64 EvictCheckFrames = 600;
    if (frame_count % EvictCheckFrames == 0) {
        EvictUnusedPipelines();
        if (const int mode = VideoCore::Nx2Debug::Experiment(51); mode != 0) {
            EvictUnusedComputePipelines(mode);
        }
        if (VideoCore::Nx2Debug::Experiment(23) == 1 &&
            device.IsExtGraphicsPipelineLibrarySupported()) {
            // EXP23 (off by default, unmeasured): a library goes unused once every pipeline
            // linked from it is evicted and destroyed.
            constexpr size_t MaxLibraries = 4096;
            graphics_libraries.TrimUnused(MaxLibraries);
        }
    }
}

PipelineCache::MemoryOwnerCounts PipelineCache::MemoryOwners() {
    MemoryOwnerCounts counts{};
    counts.graphics_entries = graphics_cache.size();
    for (const auto& [key, pipeline] : graphics_cache) {
        if (pipeline != nullptr && pipeline->IsBuilt()) {
            ++counts.graphics_built;
        }
    }
    for (const auto& [key, entry] : compute_cache) {
        counts.compute += entry.pipeline != nullptr ? 1 : 0;
        counts.compute += entry.pending != nullptr && entry.pending->pipeline != nullptr ? 1 : 0;
    }
    counts.awaiting_destruction = evicted_pipelines.size() + retired_translations.size() +
                                  retired_compute_pipelines.size();
    // The VkPipelineCache object keeps every binary built this session in memory. Its size comes
    // from the saves, and between them from a query once a minute on the serialization thread
    // (some drivers serialize the whole cache to answer it), never while a save is under way.
    if (use_vulkan_pipeline_cache && vulkan_pipeline_cache) {
        const auto now = std::chrono::steady_clock::now();
        if (now - last_cache_size_query >= std::chrono::minutes{1} &&
            !flush_in_flight.load(std::memory_order_acquire)) {
            last_cache_size_query = now;
            serialization_thread.QueueWork([this] {
                size_t size = 0;
                if (vulkan_pipeline_cache.Read(&size, nullptr) == VK_SUCCESS) {
                    last_cache_size.store(size, std::memory_order_relaxed);
                }
            });
        }
    }
    counts.cache_data_bytes = last_cache_size.load(std::memory_order_relaxed);
    return counts;
}

void PipelineCache::RetireTranslatedPipeline(std::unique_ptr<GraphicsPipeline> pipeline) {
    if (pipeline) {
        retired_translations.emplace_back(scheduler.CurrentTick(), std::move(pipeline));
    }
}

void PipelineCache::EvictUnusedPipelines() {
    // Building a pipeline again on MoltenVK means a Metal compile mid-game, far dearer than the
    // 2-3 ms below, so none are evicted there.
    if (device.IsMoltenVK()) {
        return;
    }
    // EDEN_NX2_PRELOAD_ALL_PIPELINES asked for every cached pipeline built: evicting the ones not
    // used yet threw the preload away after 9000 frames (Ocarina: 22,510 of 24,285, then rebuilt
    // one by one in new places).
    if (PreloadAllPipelines()) {
        return;
    }
    // Each built pipeline holds ~0.45 MB of driver memory and none was ever destroyed, so it grew
    // with every place a session went (title 1: 1,500 on its title screen alone). One unused for
    // this long is destroyed; it takes 2-3 ms to build again from the driver's pipeline cache.
    constexpr u64 EvictAfterFrames = 9000;
    if (frame_count <= EvictAfterFrames) {
        return;
    }
    // Only while the cache is large: each one evicted is translated again on the GPU thread when
    // its place is revisited (~10 ms). Ocarina of Time kept ~1,900 of them and evicted 3,400 in
    // 25 minutes of play, then rebuilt 1,850 (14 s of stalls, 3.8 s in one 10 s window). The
    // threshold scales with the RAM (EvictAbovePipelines: 32 GB ~4,560, ~2 GB of driver memory).
    // EDEN_NX2_PIPELINE_EVICT_ABOVE sets it.
    // Past it only the excess goes, least recently used first (from zortos293/Lichii#408), so
    // crossing it doesn't drop every stale pipeline at once.
    const size_t evict_above = EvictAbovePipelines();
    if (graphics_cache.size() <= evict_above) {
        return;
    }
    const u64 cutoff = frame_count - EvictAfterFrames;
    // A pipeline built from the disk cache at boot that no draw has used yet stays, up to a budget
    // (from zortos293/Lichii#550): evicting it throws the prebuild away, and the game translates it
    // again on the GPU thread when it reaches that content. Past the budget (~0.9 GB of driver
    // memory) they go like the rest, so a large cache can't hold their memory all session.
    constexpr size_t MaxUnusedPrebuiltPipelines = 2000;
    size_t kept_prebuilt = 0;
    std::vector<std::pair<u64, GraphicsPipelineCacheKey>> candidates;
    for (const auto& [key, pipeline] : graphics_cache) {
        if (pipeline == nullptr || pipeline.get() == current_pipeline || !pipeline->IsBuilt() ||
            pipeline->LastUse() >= cutoff) {
            continue;
        }
        if (!pipeline->WasUsed() && kept_prebuilt < MaxUnusedPrebuiltPipelines) {
            ++kept_prebuilt;
            continue;
        }
        candidates.emplace_back(pipeline->LastUse(), key);
    }
    const size_t evicted = VideoCommon::GraphicsPipelinesToEvict(graphics_cache.size(),
                                                                 candidates.size(), evict_above);
    if (evicted == 0) {
        return;
    }
    std::nth_element(candidates.begin(), candidates.begin() + (evicted - 1), candidates.end(),
                     [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    const u64 tick = scheduler.CurrentTick();
    for (size_t i = 0; i < evicted; ++i) {
        const auto it = graphics_cache.find(candidates[i].second);
        if (used_this_session.erase(it->second.get()) != 0) {
            used_and_evicted.insert(it->first.Hash());
        }
        evicted_pointer_slots.Remember(it->first.Hash(), it->second->MappedPointers());
        evicted_pipelines.emplace_back(tick, std::move(it->second));
        graphics_cache.erase(it);
    }
    for (auto& [key, pipeline] : graphics_cache) {
        if (pipeline) {
            pipeline->ClearTransitions();
        }
    }
    LOG_INFO(Render_Vulkan, "Evicted {} graphics pipelines unused for {} frames, {} left", evicted,
             EvictAfterFrames, graphics_cache.size());
}

void PipelineCache::EvictUnusedComputePipelines(int mode) {
    // As EvictUnusedPipelines: not on MoltenVK (a Metal compile per rebuild), nor when every
    // cached pipeline was asked for.
    if (device.IsMoltenVK() || PreloadAllPipelines()) {
        return;
    }
    // EXP51 (live, off by default): compute pipelines were never destroyed (Ocarina of Time: 1,803
    // 6 minutes in, beside 3,657 graphics). One unused this long goes, least recently used first,
    // while more than MinKept are held; its next dispatch translates it again on the GPU thread
    // and builds it from the driver's pipeline cache. 1 = only programs dispatched this session,
    // 2 = also boot builds no dispatch used yet (dropped once the session is that old).
    constexpr u64 EvictAfterFrames = 9000;
    constexpr size_t MinKept = 500;
    if (frame_count <= EvictAfterFrames || compute_cache.size() <= MinKept) {
        return;
    }
    const u64 cutoff = frame_count - EvictAfterFrames;
    std::vector<std::pair<u64, ComputePipelineCacheKey>> candidates;
    for (const auto& [key, entry] : compute_cache) {
        // A rebuild in flight (pending) is left to finish; an entry without a pipeline is a
        // failure record, which keeps a native program from being retried every dispatch.
        if (entry.pipeline == nullptr || entry.pending != nullptr || !entry.pipeline->IsBuilt() ||
            entry.pipeline->BuildFailed() || entry.last_use_frame >= cutoff ||
            (!entry.dispatched && mode < 2)) {
            continue;
        }
        candidates.emplace_back(entry.last_use_frame, key);
    }
    const size_t evicted =
        VideoCommon::GraphicsPipelinesToEvict(compute_cache.size(), candidates.size(), MinKept);
    if (evicted == 0) {
        return;
    }
    std::nth_element(candidates.begin(), candidates.begin() + (evicted - 1), candidates.end(),
                     [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    // Destroyed once the GPU is done with them and no build of theirs is queued on a worker
    // (TickFrame frees retired_compute_pipelines from the front).
    const u64 tick = scheduler.CurrentTick();
    for (size_t i = 0; i < evicted; ++i) {
        const auto it = compute_cache.find(candidates[i].second);
        ComputePipelineCacheEntry& entry = it->second;
        evicted_compute.insert_or_assign(
            it->first, EvictedCompute{std::move(entry.mapped_pointers), entry.rebuilds});
        retired_compute_pipelines.emplace_back(tick, std::move(entry.pipeline));
        compute_cache.erase(it);
    }
    compute_evicted_total += evicted;
    LOG_INFO(Render_Vulkan,
             "Evicted {} compute pipelines unused for {} frames (EXP51={}), {} left; this "
             "session {} evicted, {} of those dispatched again",
             evicted, EvictAfterFrames, mode, compute_cache.size(), compute_evicted_total,
             compute_rebuilt_after_evict);
}

std::unique_ptr<GraphicsPipeline> PipelineCache::TakeBackgroundPipeline(
    const GraphicsPipelineCacheKey& key) {
    std::unique_lock lock{background_mutex, std::defer_lock};
    if (QuietPrebuildMode() == 0) {
        lock.lock();
    } else {
        // The background threads hold this briefly, but at idle priority one can be preempted
        // while holding it: the GPU thread then builds the pipeline itself rather than wait. Not
        // claimed, its background build (if any) stays in background_ready, unused.
        const auto give_up = std::chrono::steady_clock::now() + std::chrono::microseconds{200};
        while (!lock.try_lock()) {
            if (std::chrono::steady_clock::now() > give_up) {
                return nullptr;
            }
            std::this_thread::yield();
        }
    }
    if (const auto it = background_ready.find(key); it != background_ready.end()) {
        auto pipeline = std::move(it->second);
        background_ready.erase(it);
        ++background_taken;
        return pipeline;
    }
    if (background_active.load(std::memory_order_acquire)) {
        background_claimed.insert(key);
    }
    return nullptr;
}

void PipelineCache::StopBackgroundPipelines() {
    background_stop.store(true, std::memory_order_relaxed);
    // Joins the threads: a build in progress finishes, queued ones are dropped.
    background_workers.reset();
}

bool PipelineCache::CanSkipCurrentDraw() const noexcept {
    if (!use_asynchronous_shaders) {
        return false;
    }
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    return CanDiscardPendingDraw(maxwell3d->GetNativeMacroEngine().native_turing,
                                 maxwell3d->regs.zeta_enable != 0, draw_state.draw_element_count,
                                 draw_state.draw_element_count, GetNativeDiscardMode());
}

GraphicsPipeline* PipelineCache::BuiltPipeline(GraphicsPipeline* pipeline) const noexcept {
    if (pipeline->IsBuilt()) {
        return pipeline;
    }
    if (!CanSkipCurrentDraw()) {
        return pipeline;
    }
    const bool discard = DrawMayBeSkipped();
    if (discard) {
        NoteSkippedLookup();
    }
    return discard ? nullptr : pipeline;
}

int AsyncDiscardMode() {
    // Read on first use (the first missing pipeline), after nx2_switches.txt was applied.
    static const int mode = [] {
        const char* value = std::getenv("EDEN_NX2_ASYNC_DISCARD");
        // 2 by default since 2026-11-08: with an empty shader cache, mode 1 dropped draws into
        // render-once layers for good (no "Press Start", empty NPC chat bubbles, wrong lighting);
        // with 2 the owner saw perfect lighting and every text, only shader-compile stutter. A warm
        // cache is unchanged. EDEN_NX2_ASYNC_DISCARD=1 restores the legacy rule.
        const int parsed = value == nullptr ? 2 : value[0] == '0' ? 0 : value[0] == '1' ? 1 : 2;
        static constexpr std::array<const char*, 3> what{
            "wait for every unbuilt pipeline",
            "skip unbuilt pipelines by the legacy heuristic",
            "skip unbuilt pipelines only into render targets cleared every frame",
        };
        LOG_INFO(Render_Vulkan, "EDEN_NX2_ASYNC_DISCARD={}: native draws {}", parsed,
                 what[parsed]);
        return parsed;
    }();
    return mode;
}

bool PipelineCache::DrawMayBeSkipped() const {
    // Native draws whose pipeline is still building are skipped instead of waited for (the
    // object is missing for a frame or more rather than the game freezing while the driver
    // compiles). Full-screen passes (no depth, few vertices) still wait, so a lighting or post
    // pass is never dropped. EDEN_NX2_ASYNC_DISCARD=0 waits for every pipeline.
    //
    // The legacy heuristic ignores the target, so a draw into a layer the game renders once
    // (title screen text, a chat bubble, a probe) loses its only draw for good when its
    // pipeline is missing (fable_answer_18 1a). EDEN_NX2_ASYNC_DISCARD=2 skips only into
    // targets fully cleared in this render frame and in a recent earlier one; first-time and
    // persistent targets wait by the paths that wait today (Configure's build wait, the
    // translate queue's capped Wait and synchronous fallback). The frame is the texture cache's
    // render frame, the one last_write_frame uses (the render-frame clock when live, frame_tick
    // otherwise); a tick that lands mid-frame can only make a target look first-time, so it
    // costs a wait and never a skip (ClassifyClearedTarget).
    if (discard_decision.serial == lookup_serial) {
        return discard_decision.skip;
    }
    const int mode = AsyncDiscardMode();
    const bool native = maxwell3d->GetNativeMacroEngine().native_turing;
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    // Classified in every mode, for the counts (with 1, the persistent column is what today's
    // rule drops for good); it runs per missing pipeline, not per draw.
    const VideoCommon::DrawTargets targets = texture_cache.ClassifyDrawTargets();
    const bool recurring =
        mode != 2 || !native || targets.kind == VideoCommon::DrawTargetClass::Recurring;
    const bool skip = CanDiscardPendingDraw(native, maxwell3d->regs.zeta_enable != 0,
                                            draw_state.draw_element_count,
                                            draw_state.draw_element_count, mode != 0, recurring);
    discard_decision = DiscardDecision{.serial = lookup_serial, .skip = skip};
    NoteDiscardDecision(skip, targets);
    return skip;
}

void PipelineCache::NoteDiscardDecision(bool skip, const VideoCommon::DrawTargets& targets) const {
    auto& stats = discard_stats;
    const bool repeat = stats.last_frame == frame_count && stats.last_skip == skip &&
                        stats.last_kind == targets.kind && stats.last_target == targets.gpu_addr &&
                        stats.last_key == graphics_key;
    if (repeat) {
        return;
    }
    stats.last_frame = frame_count;
    stats.last_skip = skip;
    stats.last_kind = targets.kind;
    stats.last_target = targets.gpu_addr;
    stats.last_key = graphics_key;
    const size_t index = static_cast<size_t>(targets.kind);
    ++(skip ? stats.skipped : stats.waited)[index];
    constexpr size_t MaxListedTargets = 32;
    if (targets.kind != VideoCommon::DrawTargetClass::Persistent || targets.gpu_addr == 0 ||
        listed_persistent_targets.size() >= MaxListedTargets ||
        !listed_persistent_targets.insert(targets.gpu_addr).second) {
        return;
    }
    LOG_INFO(Render_Vulkan,
             "Async discard: draw into persistent target {:#x} {}x{} {} {} for its pipeline "
             "{:016x} (EDEN_NX2_ASYNC_DISCARD={}; each target listed once, {} of {})",
             targets.gpu_addr, targets.width, targets.height, targets.format,
             skip ? "skipped" : "waits", graphics_key.Hash(), AsyncDiscardMode(),
             listed_persistent_targets.size(), MaxListedTargets);
}

void PipelineCache::ReportAsyncDiscard() {
    auto& stats = discard_stats;
    const auto now = std::chrono::steady_clock::now();
    if (stats.window == std::chrono::steady_clock::time_point{}) {
        stats.window = now;
        stats.held_at_window = GraphicsPipeline::HoldTimeTotal();
        return;
    }
    if (now - stats.window < std::chrono::seconds{10}) {
        return;
    }
    const auto held_total = GraphicsPipeline::HoldTimeTotal();
    const auto held = held_total - stats.held_at_window;
    const auto& skipped = stats.skipped;
    const auto& waited = stats.waited;
    if (skipped[0] + skipped[1] + skipped[2] + waited[0] + waited[1] + waited[2] != 0) {
        const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
        LOG_INFO(Render_Vulkan,
                 "Async discard in the last 10 s (EDEN_NX2_ASYNC_DISCARD={}): lookups of "
                 "unbuilt pipelines skipped {} recurring, {} first-time, {} persistent; waited "
                 "{} recurring, {} first-time, {} persistent; {:.1f} ms of draws held for "
                 "unbuilt pipelines, {:.1f} ms of GPU thread waiting for translations",
                 AsyncDiscardMode(), skipped[0], skipped[1], skipped[2], waited[0], waited[1],
                 waited[2], ms(held), ms(stats.queue_wait));
    }
    stats.skipped = {};
    stats.waited = {};
    stats.queue_wait = {};
    stats.held_at_window = held_total;
    stats.window = now;
}

namespace {
/// "pipeline=<key hash> s<stage>:<shader hash> ...", as the runtime pipeline lines print them
std::string PipelineShaders(const GraphicsPipelineCacheKey& key) {
    std::string shaders = fmt::format("pipeline={:016x}", key.Hash());
    for (size_t stage = 0; stage < key.unique_hashes.size(); ++stage) {
        if (key.unique_hashes[stage] != 0) {
            shaders += fmt::format(" s{}:{:016x}", stage, key.unique_hashes[stage]);
        }
    }
    return shaders;
}
} // Anonymous namespace

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateCachedGraphicsPipeline(
    const GraphicsPipelineCacheKey& key, std::vector<std::vector<FileEnvironment>>& records,
    PipelineStatistics* statistics) {
    ShaderPools pools;
    for (auto& envs : records) {
        pools.ReleaseContents();
        boost::container::static_vector<Shader::Environment*, 5> env_ptrs;
        for (auto& env : envs) {
            env_ptrs.push_back(&env);
        }
        if (auto pipeline{CreateGraphicsPipeline(pools, key, MakeSpan(env_ptrs), statistics, false,
                                                 nullptr, nullptr, false, nullptr, EnvKind::File)}) {
            return pipeline;
        }
    }
    // A pipeline that keeps failing however often it is stored again is not appended every session.
    constexpr size_t MaxRecords = 3;
    if (records.size() < MaxRecords) {
        std::scoped_lock lock{failed_disk_pipelines_mutex};
        failed_disk_pipelines.insert(key.Hash());
    }
    return nullptr;
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline(
    ShaderPools& pools, const GraphicsPipelineCacheKey& key,
    std::span<Shader::Environment* const> envs, PipelineStatistics* statistics,
    bool build_in_parallel, std::shared_ptr<ShaderPools> emit_pools,
    std::unique_ptr<EmissionBudget::Reservation> reservation, bool libraries_only,
    TranslationRecord* record, EnvKind env_kind) try {
    auto hash = key.Hash();
    LOG_DEBUG(Render_Vulkan, "{:#016x}", hash);
    size_t env_index{0};
    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
    const bool uses_vertex_a{key.unique_hashes[0] != 0};
    const bool uses_vertex_b{key.unique_hashes[1] != 0};

    // Layer passthrough generation for devices without VK_EXT_shader_viewport_index_layer
    Shader::IR::Program* layer_source_program{};
    bool mesh_unsupported{};

    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        const bool is_emulated_stage = layer_source_program != nullptr &&
                                       index == static_cast<u32>(Maxwell::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && is_emulated_stage) {
            auto topology = MaxwellToOutputTopology(key.state.topology);
            programs[index] = GenerateGeometryPassthrough(pools.inst, pools.block, host_info,
                                                          *layer_source_program, topology);
            continue;
        }
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        if (index == static_cast<size_t>(Maxwell::ShaderType::Geometry) &&
            !device.IsGeometryShaderSupported() && maxwell3d != nullptr &&
            maxwell3d->GetNativeMacroEngine().native_turing) {
            // A native (Switch 2) geometry program on a device without geometry shaders: the
            // pipeline cannot run it, so its draws are skipped.
            static std::atomic<bool> logged{};
            if (!logged.exchange(true)) {
                LOG_WARNING(Render_Vulkan, "Skipping native geometry shader pipelines: the device has no geometry shaders");
            }
            return nullptr;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;

        // Without VK_EXT_mesh_shader the driver cannot build a task/mesh pipeline; the draw is
        // skipped (see the pipeline check in the rasterizer). The stages are still translated
        // so the skipped pipeline's needs can be reported below.
        if ((env.ShaderStage() == Shader::Stage::Task ||
             env.ShaderStage() == Shader::Stage::Mesh) &&
            !device.IsExtMeshShaderSupported()) {
            static std::atomic<bool> logged{};
            if (!logged.exchange(true, std::memory_order_relaxed)) {
                LOG_WARNING(Render_Vulkan,
                            "{} has no VK_EXT_mesh_shader: pipelines with task/mesh shaders are "
                            "skipped, so the geometry they draw (such as foliage and props) will "
                            "be missing",
                            device.GetModelName());
            }
            mesh_unsupported = true;
        }

        const auto translate_begin = t_runtime_build ? std::chrono::steady_clock::now()
                                                     : std::chrono::steady_clock::time_point{};
        const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
        Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
        if (!uses_vertex_a || index != 1) {
            // Normal path
            programs[index] = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
        } else {
            // VertexB path when VertexA is present.
            auto& program_va{programs[0]};
            auto program_vb{TranslateProgram(pools.inst, pools.block, env, cfg, host_info)};
            programs[index] = MergeDualVertexPrograms(program_va, program_vb, env, host_info);
        }
        if (t_runtime_build) {
            t_build_split.translation += std::chrono::steady_clock::now() - translate_begin;
        }

        if (const auto edits = Shader::Optimization::ApplyOotVisualWorkaround(
                programs[index], program_id != 0 ? program_id : visual_workaround_title,
                key.unique_hashes[index],
                oot_visual_workaround)) {
            LOG_WARNING(Render_Vulkan,
                        "TEMPORARY OoT visual workaround: shader {:016x}, {} scalar edits",
                        key.unique_hashes[index], *edits);
        }
        if (device.GetDriverID() != VK_DRIVER_ID_NVIDIA_PROPRIETARY && Shader::Optimization::HasBrokenPattern(programs[index])) {
            LOG_WARNING(Render_Vulkan, "0x{:016x}: Skipping shader with known issues", key.unique_hashes[index]);
            return nullptr;
        }

        if (Settings::values.dump_guest_shaders) {
            env.Dump(hash, key.unique_hashes[index]);
        }

        if (programs[index].info.requires_layer_emulation) {
            layer_source_program = &programs[index];
        }
        if (key.state.dynamic_vertex_input != 0 &&
            programs[index].stage == Shader::Stage::VertexB) {
            // The key dropped the types of attributes outside the input map: such a read now gets
            // the default value (EDEN_NX2_KEY_ALL_ATTRIBUTES=1 keeps every type).
            const u64 attribute_mask = VideoCommon::VertexAttributeMaskOf(env);
            for (size_t attribute = 0; attribute < Shader::IR::NUM_GENERICS; ++attribute) {
                if (programs[index].info.loads.Generic(attribute) &&
                    ((attribute_mask >> (attribute * 2)) & 3) == 0) {
                    static std::atomic<u32> warned{};
                    if (warned.fetch_add(1, std::memory_order_relaxed) < 16) {
                        LOG_WARNING(Render_Vulkan,
                                    "Vertex program {:016x} reads attribute {} outside its input "
                                    "map",
                                    key.unique_hashes[index], attribute);
                    }
                    break;
                }
            }
        }
    }
    // Mesh emulation without VK_EXT_mesh_shader (prototype): a mesh program, after an optional
    // task program, drawing triangles without per-primitive outputs, before a pixel program only.
    std::optional<MeshEmulationSetup> mesh_emulation;
    size_t mesh_index{};
    if (mesh_unsupported) {
        ReportUnsupportedMeshPipeline(key, programs);
        const bool has_task{key.unique_hashes[1] != 0 &&
                            programs[1].stage == Shader::Stage::Task};
        for (size_t index = 1; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0 && programs[index].stage == Shader::Stage::Mesh) {
                mesh_index = index;
                break;
            }
        }
        bool eligible{mesh_index != 0 && (mesh_index == 1 || has_task) &&
                      layer_source_program == nullptr};
        const Shader::IR::Program& mesh{programs[mesh_index]};
        eligible = eligible && mesh.mesh.topology == 4 && mesh.mesh.primitive_lines.empty() &&
                   !mesh.mesh.vertex_lines.empty() && mesh.mesh.max_vertices != 0 &&
                   mesh.mesh.max_primitives != 0;
        for (size_t index = 1; index < Maxwell::MaxShaderProgram && eligible; ++index) {
            const bool emulated{index == mesh_index || (has_task && index == 1)};
            eligible = emulated || key.unique_hashes[index] == 0 ||
                       programs[index].stage == Shader::Stage::Fragment;
        }
        if (!eligible) {
            return nullptr;
        }
        // The output buffer's binding follows every program's (unified bindings, one per
        // descriptor element, as the SPIR-V backend numbers them). The task and mesh programs
        // share the compute stage, so their descriptors and the emulation's own buffers count
        // against one set of per-stage limits.
        u32 output_binding{};
        u32 compute_storage_buffers{has_task ? 4U : 1U};
        u32 compute_sampled_images{};
        u32 compute_resources{compute_storage_buffers};
        for (size_t index = 1; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] == 0) {
                continue;
            }
            const Shader::Info& info{programs[index].info};
            const bool emulated{index == mesh_index || (has_task && index == 1)};
            const auto add = [&](const auto& descriptors, u32* category) {
                const u32 count{Shader::NumDescriptors(descriptors)};
                output_binding += count;
                if (emulated) {
                    compute_resources += count;
                    if (category) {
                        *category += count;
                    }
                }
            };
            add(info.constant_buffer_descriptors, nullptr);
            add(info.storage_buffers_descriptors, &compute_storage_buffers);
            add(info.texture_buffer_descriptors, &compute_sampled_images);
            add(info.image_buffer_descriptors, nullptr);
            add(info.texture_descriptors, &compute_sampled_images);
            add(info.image_descriptors, nullptr);
        }
        if (compute_storage_buffers > device.GetMaxPerStageDescriptorStorageBuffers() ||
            compute_sampled_images > device.GetMaxPerStageDescriptorSampledImages() ||
            compute_resources > device.GetMaxPerStageResources()) {
            LOG_WARNING(Render_Vulkan,
                        "Mesh emulation skipped: {} storage buffers, {} sampled images and {} "
                        "resources exceed the compute stage limits",
                        compute_storage_buffers, compute_sampled_images, compute_resources);
            return nullptr;
        }
        mesh_emulation.emplace();
        mesh_emulation->output_binding = output_binding;
        if (has_task) {
            // After the output: the task payload, offset and argument buffers.
            mesh_emulation->has_task = true;
            mesh_emulation->mesh_stage = static_cast<u32>(mesh_index - 1);
            mesh_emulation->payload_stride = (std::max)(programs[1].mesh.payload_words, 2U);
            mesh_emulation->payload_binding = output_binding + 1;
            mesh_emulation->offsets_binding = output_binding + 2;
            mesh_emulation->args_binding = output_binding + 3;
            mesh_emulation->distribution_spirv =
                Shader::Backend::SPIRV::EmitMeshEmulationDistribution(profile);
        }
        mesh_emulation->block_words = Shader::Backend::SPIRV::MeshEmulationBlockWords(mesh.mesh);
        mesh_emulation->max_primitives = mesh.mesh.max_primitives;
        mesh_emulation->max_vertices = mesh.mesh.max_vertices;
        mesh_emulation->vertex_lines = static_cast<u32>(mesh.mesh.vertex_lines.size());
        for (u32 line = 0; line < mesh.mesh.vertex_lines.size(); ++line) {
            const u32 attribute{mesh.mesh.vertex_lines[line]};
            const u32 position_x{static_cast<u32>(Shader::IR::Attribute::PositionX)};
            if (attribute >= position_x && attribute < position_x + 4) {
                mesh_emulation->position_lines[attribute - position_x] = line;
            }
        }
        static std::atomic<u32> emulated{};
        if (const u32 count{++emulated}; count <= 64) {
            // The input region addresses the mesh program reads (0 is its workgroup index).
            std::string reads;
            for (const Shader::IR::Block* block : mesh.blocks) {
                for (const Shader::IR::Inst& inst : block->Instructions()) {
                    if (inst.GetOpcode() != Shader::IR::Opcode::IsbeRead) {
                        continue;
                    }
                    const Shader::IR::Value address{inst.Arg(0)};
                    reads += address.IsImmediate() ? fmt::format(" {:#x}", address.U32())
                                                   : std::string{" dynamic"};
                }
            }
            LOG_INFO(Render_Vulkan,
                     "Mesh emulation #{}: pipeline {:016x}, {} vertices x {} lines, {} triangles "
                     "per workgroup, output binding {}, task {} (payload {} words), input "
                     "reads:{}",
                     count, hash, mesh.mesh.max_vertices, mesh.mesh.vertex_lines.size(),
                     mesh.mesh.max_primitives, output_binding, has_task,
                     mesh_emulation->payload_stride, reads.empty() ? " none" : reads);
        }
    }
    // A tessellation pipeline the driver would refuse is skipped like a missing mesh pipeline:
    // the draw is dropped (see the pipeline check in the rasterizer) instead of failing the build.
    {
        const Shader::IR::Program* control{};
        const Shader::IR::Program* evaluation{};
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] == 0) {
                continue;
            }
            if (programs[index].stage == Shader::Stage::TessellationControl) {
                control = &programs[index];
            } else if (programs[index].stage == Shader::Stage::TessellationEval) {
                evaluation = &programs[index];
            }
        }
        if (control != nullptr || evaluation != nullptr) {
            const u32 max_patch_size = device.GetMaxTessellationPatchSize();
            const char* reason = nullptr;
            if (!device.IsTessellationSupported()) {
                reason = "the device has no tessellation shaders";
            } else if (control == nullptr || evaluation == nullptr) {
                reason = "it has only one of the control and evaluation stages";
            } else if (control->invocations == 0 || control->invocations > max_patch_size) {
                reason = "the control shader's output vertex count is out of range";
            }
            if (reason != nullptr) {
                LOG_WARNING(Render_Vulkan,
                            "Skipping tessellation pipeline: {} (tcs={} tes={}, tcs output "
                            "vertices {}, patch control points {}, device max patch size {})",
                            reason, control != nullptr, evaluation != nullptr,
                            control != nullptr ? control->invocations : 0,
                            key.state.patch_control_points_minus_one.Value() + 1, max_patch_size);
                return nullptr;
            }
        }
    }
    // Every environment has answered its queries now; keep them with the SPIR-V below so later
    // pipelines of these programs can skip translation (LICHII_SPIRV_REUSE).
    std::shared_ptr<ReusableStages> reusable;
    if (SpirvReuseEnabled() && env_kind != EnvKind::Unknown) {
        reusable = std::make_shared<ReusableStages>();
        for (Shader::Environment* const env : envs) {
            if (env_kind == EnvKind::Graphics) {
                reusable->queries.push_back(
                    static_cast<const VideoCommon::GraphicsEnvironment*>(env)->RecordQueries());
            } else {
                reusable->queries.push_back(
                    static_cast<const VideoCommon::FileEnvironment*>(env)->RecordQueries());
            }
        }
    }
    // The mesh fallback binds its output buffer (and, for task draws, payload, offsets and
    // indirect arguments) into the descriptor set every stage shares: with the stages' own storage
    // buffers they must fit the set's limit (zortos293/Lichii#642; only fallback pipelines).
    if (const u32 set_limit = host_info.max_descriptor_set_storage_buffers;
        mesh_emulation && set_limit != 0) {
        size_t storage_buffers{};
        for (size_t index = uses_vertex_a && uses_vertex_b ? 1 : 0;
             index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0) {
                storage_buffers +=
                    Shader::NumDescriptors(programs[index].info.storage_buffers_descriptors);
            }
        }
        const size_t auxiliary_buffers{mesh_emulation->has_task ? 4U : 1U};
        if (!FitsStorageBufferSetLimit(storage_buffers, auxiliary_buffers, set_limit)) {
            throw Shader::NotImplementedException(
                "Binding {} storage buffers ({} for mesh fallback) to a descriptor set limited to {}",
                storage_buffers + auxiliary_buffers, auxiliary_buffers, set_limit);
        }
    }
    // Graphics pipeline libraries (EDEN_NX2_GPL). Left on the monolithic route: task and mesh
    // pipelines, transform feedback, a static rasterizer discard, and stages whose texture or
    // image counts do not fit the split rescaling bits.
    const bool use_libraries = [&] {
        if (!device.IsExtGraphicsPipelineLibrarySupported() || key.state.xfb_enabled != 0) {
            return false;
        }
        // Lichii: LICHII_SHADER_OBJECTS draws these programs as shader objects (their binaries
        // precompiled per driver), so they are not made pipeline libraries.
        if (static const bool shader_objects = [] {
                const char* value = std::getenv("LICHII_SHADER_OBJECTS");
                return value != nullptr && value[0] == '1';
            }();
            shader_objects) {
            return false;
        }
        if (key.state.extended_dynamic_state_2 == 0 &&
            key.state.dynamic_state.rasterize_enable == 0) {
            return false;
        }
        size_t pre_textures{}, pre_images{}, fragment_textures{}, fragment_images{};
        for (size_t index = 1; index < Maxwell::MaxShaderProgram; ++index) {
            const bool is_emulated_stage =
                layer_source_program != nullptr &&
                index == static_cast<u32>(Maxwell::ShaderType::Geometry);
            if (key.unique_hashes[index] == 0 && !is_emulated_stage) {
                continue;
            }
            const Shader::IR::Program& program{programs[index]};
            if (program.stage == Shader::Stage::Task || program.stage == Shader::Stage::Mesh) {
                return false;
            }
            const bool fragment = index + 1 == Maxwell::MaxShaderProgram;
            (fragment ? fragment_textures : pre_textures) += program.info.texture_descriptors.size();
            (fragment ? fragment_images : pre_images) += program.info.image_descriptors.size();
        }
        return pre_textures <= GPL_FRAGMENT_TEXTURE_SCALING_BASE &&
               fragment_textures <= GPL_FRAGMENT_TEXTURE_SCALING_BASE &&
               pre_images <= GPL_FRAGMENT_IMAGE_SCALING_BASE &&
               fragment_images <= GPL_FRAGMENT_IMAGE_SCALING_BASE;
    }();
    if (libraries_only && !use_libraries) {
        return nullptr;
    }
    std::array<u128, Maxwell::MaxShaderStage> spirv_hashes{};

    std::array<const Shader::Info*, Maxwell::MaxShaderStage> infos{};
    std::array<vk::ShaderModule, Maxwell::MaxShaderStage> modules;
    std::array<std::optional<Shader::RuntimeInfo>, Maxwell::MaxShaderStage> deferred_runtime_infos;
    std::array<std::vector<u32>, Maxwell::MaxShaderStage> reference_spirv;
    std::array<std::vector<u32>, Maxwell::MaxShaderStage> deferred_spirv;

    const Shader::IR::Program* previous_stage{};
    Shader::Backend::Bindings binding;
    for (size_t index = uses_vertex_a && uses_vertex_b ? 1 : 0; index < Maxwell::MaxShaderProgram;
         ++index) {
        const bool is_emulated_stage = layer_source_program != nullptr &&
                                       index == static_cast<u32>(Maxwell::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && !is_emulated_stage) {
            continue;
        }
        UNIMPLEMENTED_IF(index == 0);

        Shader::IR::Program& program{programs[index]};
        const size_t stage_index{index - 1};
        infos[stage_index] = &program.info;

        auto runtime_info{MakeRuntimeInfo(programs, key, program, previous_stage, device)};
        if (mesh_emulation && (program.stage == Shader::Stage::Mesh ||
                               program.stage == Shader::Stage::Task)) {
            runtime_info.mesh_as_compute = true;
            runtime_info.mesh_output_binding = mesh_emulation->output_binding;
            if (mesh_emulation->has_task) {
                runtime_info.mesh_task_payload_binding = mesh_emulation->payload_binding;
                runtime_info.mesh_task_offsets_binding = mesh_emulation->offsets_binding;
                runtime_info.mesh_task_args_binding = mesh_emulation->args_binding;
                runtime_info.mesh_task_payload_stride = mesh_emulation->payload_stride;
            }
            if (program.stage == Shader::Stage::Mesh) {
                mesh_emulation->vertex_spirv = Shader::Backend::SPIRV::EmitMeshEmulationVertex(
                    profile, program.mesh, mesh_emulation->output_binding,
                    mesh_emulation->has_task ? mesh_emulation->args_binding : ~0U,
                    runtime_info.convert_depth_mode && !profile.support_native_ndc);
            }
        }
        ConvertLegacyToGeneric(program, runtime_info);
        // EDEN_NX2_MESH_EMULATION_DUMP=<dir>: write the emulation's vertex shader for spirv-val.
        // (Emitting the mesh program a second time here to dump its compute shader crashed the
        // build that followed, so only the vertex shader is written.)
        if (mesh_emulation && program.stage == Shader::Stage::Mesh) {
            if (const char* const dump{std::getenv("EDEN_NX2_MESH_EMULATION_DUMP")}) {
                const auto& code{mesh_emulation->vertex_spirv};
                std::ofstream file{fmt::format("{}/{:016x}.vert.spv", dump, hash),
                                   std::ios::binary};
                file.write(reinterpret_cast<const char*>(code.data()),
                           static_cast<std::streamsize>(code.size() * sizeof(u32)));
            }
        }
        PrepareStageBindings(binding, stage_index, use_libraries);
        if (record) {
            // EDEN_NX2_ASYNC_TRANSLATE=2: this stage's SPIR-V, emitted once and used for the
            // record and for the pipeline alike. EmitSPIRV changes the program it emits: the flag
            // of an operation (GetZeroFromOp, GetSignFromOp, GetCarryFromOp, GetOverflowFromOp,
            // GetSparseFromOp, GetInBoundsFromOp) gets its id from the operation's emitter and is
            // turned into a Void, cut from the operation. A second emission of the same program
            // never defines those flags again, and their users read the ids of the first one:
            // ids the module never defines, "%782 = OpLogicalNot %u1 %782" at spirv-val, and the
            // driver crash in verify mode.
            const auto begin = std::chrono::steady_clock::now();
            record->spirv[stage_index] = EmitSPIRV(profile, runtime_info, program, binding);
            if (record->keep_infos) {
                // The info the pipeline is made with (it copies program.info after this).
                record->infos[stage_index].clear();
                Shader::SerializeInfo(program.info, record->infos[stage_index]);
                record->libraries = use_libraries;
                record->storage_descriptors +=
                    static_cast<u32>(program.info.storage_buffers_descriptors.size());
                record->proven_read_extents += ProvenReadExtents(program.info);
            }
            if (record->translate_only) {
                previous_stage = &program;
                continue;
            }
            if (emit_pools) {
                // The workers build the module from these words (DeferredStages).
                deferred_runtime_infos[stage_index] = runtime_info;
                deferred_spirv[stage_index] = record->spirv[stage_index];
            } else {
                modules[stage_index] = BuildStageModule(
                    device, record->spirv[stage_index], key.unique_hashes[index], stage_index,
                    use_libraries ? &spirv_hashes[stage_index] : nullptr);
            }
            if (build_in_parallel) {
                AddRuntimeSpirvTime(0, begin);
            }
            previous_stage = &program;
            continue;
        }
        if (emit_pools) {
            deferred_runtime_infos[stage_index] = runtime_info;
            if (VideoCommon::Nx2Perf::Verify()) {
                // EDEN_NX2_PERF_VERIFY: the workers' emission is the program's second, so a stage
                // with an operation flag (see the record above) always differs from this one and
                // falls back to it (EmitPipelineShader). This is the SPIR-V that gets used then.
                reference_spirv[stage_index] = EmitSPIRV(profile, runtime_info, program, binding);
            }
        } else {
            const auto begin = std::chrono::steady_clock::now();
            modules[stage_index] = EmitShaderModule(
                device, profile, runtime_info, program, binding, key.unique_hashes[index],
                stage_index, {}, use_libraries ? &spirv_hashes[stage_index] : nullptr,
                reusable ? &reusable->spirv[stage_index] : nullptr);
            if (build_in_parallel) {
                AddRuntimeSpirvTime(0, begin);
            }
        }
        previous_stage = &program;
    }
    if (record) {
        record->complete = true;
        record->mesh_emulated = mesh_emulation.has_value();
        if (record->translate_only) {
            return nullptr;
        }
    }
    GraphicsPipeline::EmitStagesFunc emit_stages;
    if (emit_pools) {
        auto job = std::make_shared<DeferredStages>();
        job->reservation = std::move(reservation);
        job->pools = emit_pools;
        job->programs = std::move(programs);
        job->runtime_infos = std::move(deferred_runtime_infos);
        job->reference_spirv = std::move(reference_spirv);
        job->emitted_spirv = std::move(deferred_spirv);
        for (size_t stage = 0; stage < infos.size(); ++stage) {
            if (infos[stage]) {
                infos[stage] = &job->programs[stage + 1].info;
                if (reusable) {
                    reusable->infos[stage] = InternInfo(*infos[stage]);
                }
            }
        }
        emit_stages = [this, job = std::move(job), hashes = key.unique_hashes, use_libraries,
                       reusable = std::move(reusable), stored_key = key](
                          std::array<u128, Maxwell::MaxShaderStage>& stage_hashes) mutable {
            const auto begin = std::chrono::steady_clock::now();
            std::array<vk::ShaderModule, Maxwell::MaxShaderStage> stage_modules;
            Shader::Backend::Bindings stage_binding;
            for (size_t stage = 0; stage < stage_modules.size(); ++stage) {
                if (!job->emitted_spirv[stage].empty()) {
                    // Emitted already: a program is emitted once (see the record in the loop).
                    stage_modules[stage] =
                        BuildStageModule(device, job->emitted_spirv[stage], hashes[stage + 1],
                                         stage, use_libraries ? &stage_hashes[stage] : nullptr);
                    if (reusable) {
                        reusable->spirv[stage] = job->emitted_spirv[stage];
                    }
                } else if (job->runtime_infos[stage]) {
                    PrepareStageBindings(stage_binding, stage, use_libraries);
                    stage_modules[stage] = EmitShaderModule(
                        device, profile, *job->runtime_infos[stage], job->programs[stage + 1],
                        stage_binding, hashes[stage + 1], stage, job->reference_spirv[stage],
                        use_libraries ? &stage_hashes[stage] : nullptr,
                        reusable ? &reusable->spirv[stage] : nullptr);
                }
            }
            if (reusable) {
                StoreReusableStages(stored_key, std::move(reusable));
            }
            auto job_pools = std::move(job->pools);
            job.reset();
            job_pools->ReleaseContents();
            {
                std::scoped_lock lock{spare_pools_mutex};
                if (spare_pools.size() < 4) {
                    spare_pools.push_back(std::move(job_pools));
                }
            }
            AddRuntimeSpirvTime(1, begin);
            return stage_modules;
        };
    }
    if (reusable && !emit_stages) {
        // Emitted on this thread above: the pipeline copies these infos as they are now.
        for (size_t stage = 0; stage < infos.size(); ++stage) {
            if (infos[stage]) {
                reusable->infos[stage] = InternInfo(*infos[stage]);
            }
        }
        StoreReusableStages(key, std::move(reusable));
    }
    if (t_translate_only) {
        // LICHII_PRECOMPILE_DIR: the translation is kept for reuse; no Vulkan pipeline is built.
        return nullptr;
    }
    Common::ThreadWorker* const thread_worker{build_in_parallel && !libraries_only ? &workers
                                                                                   : nullptr};
    GraphicsPipeline::LibraryParams library_params{
        .library_cache = use_libraries ? &graphics_libraries : nullptr,
        .spirv_hashes = spirv_hashes,
        .libraries_only = libraries_only,
    };
    // The slots each program bound as storage buffer pointers, kept if the pipeline is evicted
    // (evicted_pointer_slots) for its next translation (from zortos293/Lichii#1143).
    static_assert(VideoCommon::EvictedPointerSlots::NumPrograms == Maxwell::MaxShaderProgram);
    VideoCommon::EvictedPointerSlots::Slots mapped_pointers;
    for (size_t index = 0, program_env = 0;
         index < Maxwell::MaxShaderProgram && program_env < envs.size(); ++index) {
        if (key.unique_hashes[index] != 0) {
            mapped_pointers[index] = envs[program_env++]->MappedStoragePointerKeys();
        }
    }
    auto pipeline = std::make_unique<GraphicsPipeline>(
        scheduler, buffer_cache, texture_cache,
        libraries_only ? WarmPipelineCache() : PlayPipelineCache(),
        libraries_only || t_precompile_build ? nullptr : &shader_notify, device, descriptor_pool,
        guest_descriptor_queue, descriptor_buffer_ring, thread_worker, statistics,
        render_pass_cache, key, std::move(modules), infos, std::move(emit_stages),
        library_params, std::move(mesh_emulation));
    pipeline->SetMappedPointers(std::move(mapped_pointers));
    return pipeline;

} catch (const Shader::Exception& exception) {
    auto hash = key.Hash();
    if (libraries_only) {
        // A warm-up guess the translation rejects: no dump and no error per entry (the warm-up
        // counts them and logs the total when it ends).
        LOG_DEBUG(Render_Vulkan, "Warm graphics pipeline {}: {}", PipelineShaders(key),
                  exception.what());
        return nullptr;
    }
    if (record != nullptr && record->translate_only) {
        // The verify mode's GPU-thread translation: the worker's logs and dumps this failure.
        LOG_DEBUG(Render_Vulkan, "Verify translation of {}: {}", PipelineShaders(key),
                  exception.what());
        return nullptr;
    }
    if (!build_in_parallel) {
        // A disk cache record: one another (or an older) build stored can lack what this
        // translator asks for
        // (merged packs: "Uncached storage pointer classification", 144 errors and dumps for
        // 108 pipelines). Not an error: the GPU thread builds the pipeline from the game when a
        // draw needs it, and stores that record (see CreateCachedGraphicsPipeline).
        NoteDiskRecordFailure("Graphics", hash, exception.what());
        return nullptr;
    }
    size_t env_index{0};
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;

        const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
        try {
            // Walks the program for the dump; the one that failed may fail here again, and an
            // exception leaving this handler would end the process.
            Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
        } catch (const Shader::Exception&) {
        }
        env.Dump(hash, key.unique_hashes[index]);
    }
    LOG_ERROR(Render_Vulkan, "Graphics pipeline compilation failed {}: {}", PipelineShaders(key),
              exception.what());
    return nullptr;
}

namespace {
// Pipelines built on the GPU thread at run time (shader translation, and the SPIR-V unless a worker
// emits it; the driver compile is on workers): the first 400 over 2 ms and every one over 100 ms
// are logged, and the totals every 10 s, to tell how much of the stutter on new content is
// translation.
void ReportRuntimePipeline(const char* kind, std::chrono::steady_clock::time_point begin,
                           const std::string& shaders, const RuntimeBuildSplit& split = {}) {
    static std::chrono::steady_clock::duration total{};
    static RuntimeBuildSplit split_total{};
    static u32 count{}, logged{};
    static auto window = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = now - begin;
    total += elapsed;
    split_total.environments += split.environments;
    split_total.translation += split.translation;
    ++count;
    frame_tally.builds += elapsed;
    frame_tally.split.environments += split.environments;
    frame_tally.split.translation += split.translation;
    ++frame_tally.count;
    const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
    // Slow builds are always logged with their shaders, to find the programs that are slow to
    // translate.
    if (elapsed > std::chrono::milliseconds{100} ||
        (elapsed > std::chrono::milliseconds{2} && logged < 400)) {
        ++logged;
        // Graphics builds name their environments and translation time too, after the shaders
        // so readers of the older line keep their fields.
        const std::string parts =
            split.environments.count() != 0 || split.translation.count() != 0
                ? fmt::format(" ({:.1f} ms environments, {:.1f} ms translation)",
                              ms(split.environments), ms(split.translation))
                : std::string{};
        LOG_INFO(Render_Vulkan, "{} pipeline built on the GPU thread in {:.1f} ms: {}{}", kind,
                 ms(elapsed), shaders, parts);
    }
    if (now - window > std::chrono::seconds{10}) {
        const auto take = [](auto& counter) {
            return std::chrono::nanoseconds{counter.exchange(0, std::memory_order::relaxed)};
        };
        // The split goes at the end, so readers of the older line keep their fields.
        LOG_INFO(Render_Vulkan,
                 "Runtime pipelines in the last 10 s: {} built, {:.1f} ms on the GPU thread "
                 "({:.1f} ms of it graphics SPIR-V), {:.1f} ms graphics SPIR-V on workers, "
                 "{:.1f} ms of draws held for unbuilt pipelines; graphics on the GPU thread: "
                 "{:.1f} ms environments, {:.1f} ms translation",
                 count, ms(total), ms(take(runtime_spirv_ns[0])), ms(take(runtime_spirv_ns[1])),
                 ms(GraphicsPipeline::TakeHoldTime()), ms(split_total.environments),
                 ms(split_total.translation));
        total = {};
        split_total = {};
        count = 0;
        window = now;
    }
}

// EDEN_NX2_ASYNC_TRANSLATE=3: GPU-thread builds translate from environment snapshots, and every
// read they recorded is answered again from the live state right after. Any difference is a
// snapshot gap (or a texture descriptor that needed a cache flush): the first 16 are logged with
// their pipeline, and the counts every 10 s. GPU thread only.
void NoteSnapshotCheck(const GraphicsPipelineCacheKey& key, size_t bytes,
                       std::span<Shader::Environment* const> envs) {
    static u32 pipelines{}, logged{};
    static size_t copied{}, reads{}, differing{}, gave_up{};
    static auto window = std::chrono::steady_clock::now();
    ++pipelines;
    copied += bytes;
    for (Shader::Environment* const env : envs) {
        auto& graphics_env = static_cast<GraphicsEnvironment&>(*env);
        const auto check = graphics_env.CheckAgainstLive();
        reads += check.reads;
        differing += check.differing;
        if (graphics_env.GuardedReadFailed()) {
            ++gave_up;
        }
        if (check.differing != 0 && logged < 16) {
            ++logged;
            LOG_WARNING(Render_Vulkan,
                        "Environment snapshot differs from the live state: {} reads of {} in {} "
                        "stage {}, first: {}",
                        check.differing, check.reads, PipelineShaders(key),
                        static_cast<u32>(env->ShaderStage()), check.first);
        }
        graphics_env.ReleaseSnapshotBytes();
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - window > std::chrono::seconds{10}) {
        LOG_INFO(Render_Vulkan,
                 "Environment snapshots in the last 10 s: {} pipelines, {} KB copied, {} reads "
                 "checked, {} differ, {} guarded reads gave up",
                 pipelines, copied / 1024, reads, differing, gave_up);
        pipelines = 0;
        copied = reads = differing = gave_up = 0;
        window = now;
    }
}
} // Anonymous namespace

const VideoCommon::EvictedPointerSlots::Slots* PipelineCache::KeptPointers() const {
    if (evicted_pointer_slots.Size() == 0) {
        return nullptr;
    }
    return evicted_pointer_slots.Find(graphics_key.Hash());
}

void PipelineCache::KeepPointers(std::span<GraphicsEnvironment> envs,
                                 const GraphicsPipelineCacheKey& key,
                                 const VideoCommon::EvictedPointerSlots::Slots* keep_pointers) {
    if (!keep_pointers) {
        return;
    }
    for (size_t index = 0; index < Maxwell::MaxShaderProgram && index < envs.size(); ++index) {
        if (key.unique_hashes[index] != 0) {
            envs[index].KeepStoragePointers((*keep_pointers)[index]);
        }
    }
}

void PipelineCache::WarnKeptPointer(size_t program, u64 slot) const {
    if ((slot >> 63) != 0) {
        // A pointer-table entry: its holder slot is warned about on its own
        return;
    }
    const u32 index = static_cast<u32>(slot >> 32);
    const u32 offset = static_cast<u32>(slot);
    const size_t stage = program == 0 ? 0 : program - 1;
    const auto& cbufs = maxwell3d->state.shader_stages[stage].const_buffers;
    GPUVAddr value = 0;
    const char* reason = nullptr;
    if (index >= cbufs.size() || !cbufs[index].enabled ||
        u64{offset} + sizeof(u64) > cbufs[index].size ||
        !gpu_memory->IsFullyMappedRange(cbufs[index].address + offset, sizeof(u64))) {
        reason = "constant buffer not bound";
    } else {
        value = gpu_memory->Read<u64>(cbufs[index].address + offset);
        if (value == 0) {
            reason = "null";
        } else if (!gpu_memory->IsWithinGPUAddressRange(value)) {
            reason = "outside the GPU address range";
        } else if (!gpu_memory->IsFullyMappedRange(value, 1)) {
            reason = "unmapped";
        }
    }
    if (!reason) {
        return;
    }
    static std::atomic<u64> kept{};
    const u64 count = kept.fetch_add(1, std::memory_order_relaxed) + 1;
    if (count <= 32 || count % 1000 == 0) {
        LOG_WARNING(Render_Vulkan,
                    "Graphics pipeline {:016x} translated again after eviction: s{}:c{}[{:#x}] "
                    "holds {:#x} ({}), kept as the storage buffer pointer it was ({} so far)",
                    graphics_key.Hash(), program, index, offset, value, reason, count);
    }
}

FixedPipelineState PipelineCache::ReuseState(const FixedPipelineState& source) const noexcept {
    // Fields MakeRuntimeInfo and the translation never read are cleared, so pipelines that differ
    // only in them share one translation.
    FixedPipelineState state = source;
    state.depth_format.Assign(0);
    std::memset(&state.dynamic_state, 0, sizeof(state.dynamic_state));
    std::memset(&state.attachments, 0, sizeof(state.attachments));
    state.binding_divisors = {};
    state.vertex_strides = {};
    state.depth_bounds_min = 0;
    state.depth_bounds_max = 0;
    state.line_stipple_factor = 0;
    state.line_stipple_pattern = 0;
    // A driver update changes no translation (the profile depends on the driver, not on its
    // version): the translations made before it stay reusable.
    state.driver_version = 0;
    if (!device.IsMoltenVK()) {
        // Colour output types are taken from the formats only on MoltenVK.
        state.color_formats = {};
    }
    if (!state.dynamic_vertex_input) {
        // The union holds the vertex divisors then, which are vertex input state.
        state.enabled_divisors = 0;
    }
    return state;
}

namespace {
u64 HashPrograms(const std::array<u64, Maxwell::MaxShaderProgram>& unique_hashes) {
    return Common::CityHash64(reinterpret_cast<const char*>(unique_hashes.data()),
                              sizeof(unique_hashes));
}

bool SameReuseState(const FixedPipelineState& a, const FixedPipelineState& b) {
    const size_t size = a.Size();
    return size == b.Size() && std::memcmp(&a, &b, size) == 0;
}
} // Anonymous namespace

s32 PipelineCache::SpoolSpirv(const std::vector<u32>& spirv) {
    const size_t bytes = spirv.size() * sizeof(u32);
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(spirv.data()), bytes) ^
                     (u64{spirv.size()} << 40);
    {
        std::scoped_lock lock{spirv_spool->mutex};
        if (const auto it = spirv_spool->by_hash.find(hash); it != spirv_spool->by_hash.end()) {
            return it->second;
        }
    }
    const std::vector<u8> compressed =
        Common::Compression::CompressDataZSTD(reinterpret_cast<const u8*>(spirv.data()), bytes, 1);
    std::scoped_lock lock{spirv_spool->mutex};
    if (const auto it = spirv_spool->by_hash.find(hash); it != spirv_spool->by_hash.end()) {
        return it->second;
    }
    auto& file = spirv_spool->file;
    file.clear();
    file.seekp(static_cast<std::streamoff>(spirv_spool->end));
    file.write(reinterpret_cast<const char*>(compressed.data()),
               static_cast<std::streamsize>(compressed.size()));
    if (!file) {
        return -1;
    }
    const s32 index = static_cast<s32>(spirv_spool->modules.size());
    spirv_spool->modules.push_back({spirv_spool->end, static_cast<u32>(compressed.size()),
                                    static_cast<u32>(spirv.size())});
    spirv_spool->end += compressed.size();
    spirv_spool->by_hash.emplace(hash, index);
    return index;
}

std::vector<u8> PipelineCache::ReadSpoolFrame(s32 index) {
    if (!spirv_spool || index < 0) {
        return {};
    }
    std::scoped_lock lock{spirv_spool->mutex};
    if (static_cast<size_t>(index) >= spirv_spool->modules.size()) {
        return {};
    }
    const SpirvStore::Module& where = spirv_spool->modules[index];
    std::vector<u8> frame(where.compressed_size);
    auto& file = spirv_spool->file;
    file.clear();
    file.flush();
    file.seekg(static_cast<std::streamoff>(where.offset));
    file.read(reinterpret_cast<char*>(frame.data()), static_cast<std::streamsize>(frame.size()));
    if (!file) {
        LOG_ERROR(Render_Vulkan, "LICHII_PRECOMPILE: could not read spooled SPIR-V module {}",
                  index);
        return {};
    }
    return frame;
}

const Shader::Info& PackedInfo::Get() const {
    std::call_once(once, [this] {
        LichiiIO::Reader r{bytes};
        decoded = std::make_unique<Shader::Info>(LichiiIO::GetInfo(r));
    });
    return *decoded;
}

InfoRef PipelineCache::InternInfo(const Shader::Info& info) {
    // The same program appears in thousands of precompiled pairs with the same info: keep one,
    // packed (a translation made at run time is used at once, so it keeps the expanded copy).
    LichiiIO::Writer w;
    LichiiIO::PutInfo(w, info);
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(w.buffer.data()),
                                        w.buffer.size()) ^
                     (u64{w.buffer.size()} << 44);
    std::scoped_lock lock{info_intern_mutex};
    auto& slot = info_intern[hash];
    if (!slot) {
        auto packed = std::make_shared<PackedInfo>();
        packed->bytes = std::move(w.buffer);
        packed->generic_mask = 0;
        for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
            if (info.loads.Generic(index) || info.passthrough.Generic(index)) {
                packed->generic_mask |= u64{0b11} << (index * 2);
            }
        }
        if (!t_translate_only) {
            std::call_once(packed->once, [&] {
                packed->decoded = std::make_unique<Shader::Info>(info);
            });
        }
        slot = std::move(packed);
    }
    return InfoRef{slot};
}

void PipelineCache::StoreReusableStages(const GraphicsPipelineCacheKey& key,
                                        std::shared_ptr<ReusableStages> stages) {
    constexpr size_t MaxPerPrograms = 32;
    stages->unique_hashes = key.unique_hashes;
    stages->state = ReuseState(key.state);
    if (stages->state.dynamic_vertex_input) {
        // Only the types of the inputs the vertex stage reads or passes on reach its SPIR-V.
        u64 mask = ~0ULL;
        if (const auto& vertex = stages->infos[0]) {
            // Computed when the info was packed: reading it here would expand every vertex info.
            mask = vertex.GenericMask();
        }
        stages->attribute_mask = mask;
        stages->state.attribute_types &= mask;
    }
    if (t_translate_only) {
        // LICHII_PRECOMPILE: hundreds of thousands of translations wait in memory until they
        // are stored. Each distinct module is kept once, compressed to its exact size (the
        // compressor's buffer is sized for the worst case), and shared.
        for (size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
            auto& spirv = stages->spirv[stage];
            if (spirv.empty()) {
                continue;
            }
            stages->spirv_words[stage] = static_cast<u32>(spirv.size());
            if (spirv_spool) {
                // Straight to the spool file: memory keeps only the module's index.
                if (const s32 index = SpoolSpirv(spirv); index >= 0) {
                    stages->spool_spirv[stage] = index;
                    spirv = {};
                    continue;
                }
            }
            const size_t bytes = spirv.size() * sizeof(u32);
            const u64 hash =
                Common::CityHash64(reinterpret_cast<const char*>(spirv.data()), bytes) ^
                (u64{spirv.size()} << 40);
            {
                std::scoped_lock intern_lock{spirv_intern_mutex};
                if (const auto it = spirv_intern.find(hash); it != spirv_intern.end()) {
                    stages->spirv_zstd[stage] = it->second;
                }
            }
            if (!stages->spirv_zstd[stage]) {
                const std::vector<u8> compressed = Common::Compression::CompressDataZSTD(
                    reinterpret_cast<const u8*>(spirv.data()), bytes, 1);
                auto exact = std::make_shared<const std::vector<u8>>(compressed.begin(),
                                                                     compressed.end());
                std::scoped_lock intern_lock{spirv_intern_mutex};
                stages->spirv_zstd[stage] = spirv_intern.try_emplace(hash, std::move(exact))
                                                .first->second;
            }
            spirv = {};
        }
    }
    std::scoped_lock lock{reusable_stages_mutex};
    auto& entries = reusable_stages[HashPrograms(key.unique_hashes)];
    if (entries.size() >= MaxPerPrograms) {
        return;
    }
    for (const auto& entry : entries) {
        if (entry->attribute_mask == stages->attribute_mask &&
            SameReuseState(entry->state, stages->state)) {
            return;
        }
    }
    entries.push_back(std::move(stages));
}

namespace {
// Names the shader-relevant fixed-state fields that differ, for reuse misses on known programs.
std::string DescribeReuseStateDiff(const FixedPipelineState& a, const FixedPipelineState& b) {
    std::string diff;
    const auto note = [&](bool differs, std::string_view name) {
        if (differs) {
            diff += ' ';
            diff += name;
        }
    };
    const u32 r1 = a.raw1 ^ b.raw1;
    const u32 r2 = a.raw2 ^ b.raw2;
    note(r1 & 0x3Fu, "dynamic_feature_flags");
    note(r1 & (1u << 6), "xfb_enabled");
    note(r1 & (1u << 7), "ndc");
    note(r1 & (3u << 8), "polygon_mode");
    note(r1 & (0xFFFu << 10), "tessellation");
    note(r1 & (1u << 20), "color_write_enable_dynamic");
    note(r1 & (1u << 21), "dual_source_blend");
    note(r1 & (0xFu << 24), "topology");
    note(r1 & (0xFu << 28), "msaa_mode");
    note(r2 & (7u << 1), "alpha_test_func");
    note(r2 & (1u << 4), "early_z");
    note(r2 & (1u << 5), "depth_enabled");
    note(r2 & (1u << 11), "y_negate");
    note(r2 & (1u << 12), "provoking_vertex_last");
    note(r2 & (1u << 13), "conservative_raster");
    note(r2 & (1u << 14), "smooth_lines");
    note(r2 & (3u << 15), "alpha_to_coverage_one");
    note(r2 & (7u << 17), "app_stage");
    note(a.color_formats != b.color_formats, "color_formats");
    note(a.driver_id != b.driver_id || a.driver_version != b.driver_version, "driver");
    note(a.alpha_test_ref != b.alpha_test_ref, "alpha_test_ref");
    note(a.point_size != b.point_size, "point_size");
    note(a.viewport_swizzles != b.viewport_swizzles, "viewport_swizzles");
    note(a.attribute_types != b.attribute_types, "attribute_types");
    note(std::memcmp(&a.attributes, &b.attributes, sizeof(a.attributes)) != 0, "vertex_attributes");
    if (a.xfb_enabled && b.xfb_enabled) {
        note(std::memcmp(&a.xfb_state, &b.xfb_state, sizeof(a.xfb_state)) != 0, "xfb_state");
    }
    return diff.empty() ? std::string{" (nothing in the compared fields)"} : diff;
}
} // Anonymous namespace

std::unique_ptr<GraphicsPipeline> PipelineCache::TryReuseGraphicsPipeline(
    GraphicsEnvironments& environments) {
    if (!SpirvReuseEnabled()) {
        return nullptr;
    }
    static std::atomic<u32> logged_misses{};
    static std::atomic<u32> logged_rejections{};
    const FixedPipelineState state = ReuseState(graphics_key.state);
    std::shared_ptr<const ReusableStages> stages;
    {
        std::scoped_lock lock{reusable_stages_mutex};
        const auto it = reusable_stages.find(HashPrograms(graphics_key.unique_hashes));
        if (it == reusable_stages.end()) {
            // A mesh pipeline nothing precompiled matches: is its pixel program known (with
            // which mesh program key), and which mesh registers does the game use?
            static std::atomic<u32> mesh_misses{};
            if (maxwell3d->regs.IsMeshEnabled() && mesh_misses.fetch_add(1) < 40) {
                u64 known_mesh = 0;
                size_t pixel_known = 0;
                for (const auto& [programs_hash, entries] : reusable_stages) {
                    for (const auto& entry : entries) {
                        if (entry->unique_hashes[5] == graphics_key.unique_hashes[5]) {
                            ++pixel_known;
                            known_mesh = entry->unique_hashes[1];
                        }
                    }
                }
                using Regs = Tegra::Engines::Maxwell3D::Regs;
                const auto& regs = maxwell3d->regs;
                // Undo the mesh state fold: which archive programs are the task and mesh ones?
                const u64 state_hash = MeshStateHash();
                const auto archive_stage = [&](u64 key) -> int {
                    if (key == 0) {
                        return -2;
                    }
                    const auto found = precompiled_programs.find(key ^ state_hash);
                    return found != precompiled_programs.end() ? static_cast<int>(found->second) : -1;
                };
                LOG_INFO(Render_Vulkan,
                         "LICHII_PRECOMPILE mesh miss: key {:016x}/{:016x}/{:016x}, pixel known in "
                         "{} entries (e.g. mesh key {:016x}); A={} B={} I={} geometry={}; archive "
                         "stage of slot1 {} slot3 {} (6 VertexB? 4 Fragment, 8 Mesh; -1 none)",
                         graphics_key.unique_hashes[1], graphics_key.unique_hashes[3],
                         graphics_key.unique_hashes[5], pixel_known, known_mesh,
                         regs.reg_array[Regs::MeshShaderA], regs.reg_array[Regs::MeshShaderB],
                         regs.reg_array[Regs::MeshInitShader],
                         regs.IsShaderConfigEnabled(Regs::ShaderType::Geometry),
                         archive_stage(graphics_key.unique_hashes[1]),
                         archive_stage(graphics_key.unique_hashes[3]));
            }
            return nullptr;
        }
        for (const auto& entry : it->second) {
            // LICHII_PRECOMPILE_NO_GRAPHICS_REUSE: translations from the archive store are not used.
            static const bool skip_stored =
                std::getenv("LICHII_PRECOMPILE_NO_GRAPHICS_REUSE") != nullptr;
            if (skip_stored && (std::ranges::any_of(entry->stored_spirv, [](s32 m) { return m >= 0; }) ||
                                std::ranges::any_of(entry->spool_spirv, [](s32 m) { return m >= 0; }) ||
                                std::ranges::any_of(entry->spirv_zstd, [](const auto& z) { return z != nullptr; }))) {
                continue;
            }
            if (entry->unique_hashes != graphics_key.unique_hashes) {
                continue;
            }
            FixedPipelineState masked = state;
            if (masked.dynamic_vertex_input) {
                masked.attribute_types &= entry->attribute_mask;
            }
            if (SameReuseState(masked, entry->state)) {
                stages = entry;
                break;
            }
        }
        if (!stages) {
            reuse_state_misses.fetch_add(1, std::memory_order::relaxed);
            if (!it->second.empty() &&
                logged_misses.fetch_add(1, std::memory_order::relaxed) < 60) {
                const auto& front = it->second.front();
                FixedPipelineState shown = state;
                if (shown.dynamic_vertex_input) {
                    shown.attribute_types &= front->attribute_mask;
                }
                LOG_INFO(Render_Vulkan,
                         "LICHII_SPIRV_REUSE: known programs, new shader state:{}",
                         DescribeReuseStateDiff(front->state, shown));
            }
            return nullptr;
        }
    }
    const auto envs = environments.Span();
    const auto reject = [&](const char* reason) {
        reuse_rejected.fetch_add(1, std::memory_order::relaxed);
        if (logged_rejections.fetch_add(1, std::memory_order::relaxed) < 60) {
            LOG_INFO(Render_Vulkan, "LICHII_SPIRV_REUSE: pipeline {:016x} not reused: {}",
                     graphics_key.Hash(), reason);
        }
        return nullptr;
    };
    if (envs.size() != stages->queries.size()) {
        return reject("environment count");
    }
    for (size_t index = 0; index < envs.size(); ++index) {
        // GetGraphicsEnvironments points every entry of the span into environments.envs.
        auto* const env = static_cast<VideoCommon::GraphicsEnvironment*>(envs[index]);
        if (const char* reason = env->ReplayQueries(stages->queries[index])) {
            return reject(reason);
        }
    }
    std::array<vk::ShaderModule, Maxwell::MaxShaderStage> modules;
    std::array<const Shader::Info*, Maxwell::MaxShaderStage> infos{};
    for (size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
        if (!stages->infos[stage]) {
            continue;
        }
        const std::vector<u32> spirv = ReusableSpirv(*stages, stage);
        if (spirv.empty()) {
            reuse_rejected.fetch_add(1, std::memory_order::relaxed);
            return nullptr;
        }
        modules[stage] = BuildShader(device, spirv);
        infos[stage] = &*stages->infos[stage];
    }
    reused_pipelines.fetch_add(1, std::memory_order::relaxed);
    return std::make_unique<GraphicsPipeline>(
        scheduler, buffer_cache, texture_cache, vulkan_pipeline_cache, nullptr, device,
        descriptor_pool, guest_descriptor_queue, descriptor_buffer_ring, &workers, nullptr,
        render_pass_cache, graphics_key, std::move(modules), infos,
        GraphicsPipeline::EmitStagesFunc{});
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline(
    const VideoCommon::EvictedPointerSlots::Slots* keep_pointers) {
    const auto begin = std::chrono::steady_clock::now();
    t_runtime_build = true;
    t_build_split = {};
    SCOPE_EXIT {
        t_runtime_build = false;
        ReportRuntimePipeline("Graphics", begin, PipelineShaders(graphics_key), t_build_split);
    };
    GraphicsEnvironments environments;
    const bool snapshot_check = VideoCommon::AsyncTranslateMode() == 3;
    const size_t snapshot_bytes =
        GetGraphicsEnvironments(environments, graphics_key.unique_hashes, snapshot_check);
    KeepPointers(environments.envs, graphics_key, keep_pointers);
    t_build_split.environments = std::chrono::steady_clock::now() - begin;
    SCOPE_EXIT {
        // Right after the translation (the SPIR-V and the driver compile may still be queued),
        // before the GPU thread changes any state.
        if (snapshot_check) {
            NoteSnapshotCheck(graphics_key, snapshot_bytes, environments.Span());
        }
    };

    // Kept pointers change the translation: never reused from an earlier one.
    auto pipeline{keep_pointers ? nullptr : TryReuseGraphicsPipeline(environments)};
    if (!pipeline) {
        std::shared_ptr<ShaderPools> emit_pools;
        auto reservation = emission_budget.TryAcquire(
            VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::AsyncPipelines),
            maxwell3d->GetNativeMacroEngine().native_turing);
        if (reservation) {
            {
                std::scoped_lock lock{spare_pools_mutex};
                if (!spare_pools.empty()) {
                    emit_pools = std::move(spare_pools.back());
                    spare_pools.pop_back();
                }
            }
            if (!emit_pools) {
                emit_pools = std::make_shared<ShaderPools>();
            }
        } else {
            main_pools.ReleaseContents();
        }
        ShaderPools& pools = emit_pools ? *emit_pools : main_pools;
        pipeline = CreateGraphicsPipeline(pools, graphics_key, environments.Span(), nullptr, true,
                                          std::move(emit_pools), std::move(reservation), false,
                                          nullptr, EnvKind::Graphics);
    }
    if (!pipeline || pipeline_cache_filename.empty()) {
        return pipeline;
    }
    StoreRuntimeGraphicsPipeline(graphics_key, environments.envs);
    return pipeline;
}

void PipelineCache::StoreRuntimeGraphicsPipeline(
    const GraphicsPipelineCacheKey& key,
    const std::array<GraphicsEnvironment, Maxwell::MaxShaderProgram>& environments) {
    if (disk_pipelines.contains(key.Hash())) {
        bool failed_to_load{};
        {
            std::scoped_lock lock{failed_disk_pipelines_mutex};
            failed_to_load = failed_disk_pipelines.erase(key.Hash()) != 0;
        }
        if (!failed_to_load) {
            // In the disk cache already, only not built at boot.
            runtime_built_kind[1].fetch_add(1, std::memory_order::relaxed);
            QueueVulkanPipelineCacheFlush();
            return;
        }
        // Its records failed to load (see CreateCachedGraphicsPipeline): store this build after
        // them, to be tried first from the next boot on.
    }
    disk_pipelines.insert(key.Hash());
    serialization_thread.QueueWork([this, key, envs = environments] {
        boost::container::static_vector<const GenericEnvironment*, Maxwell::MaxShaderProgram>
            env_ptrs;
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0) {
                env_ptrs.push_back(&envs[index]);
            }
        }
        SerializePipeline(key, env_ptrs, pipeline_cache_filename, CACHE_VERSION);
    });
    QueueVulkanPipelineCacheFlush();
}

namespace {
// EDEN_NX2_ASYNC_TRANSLATE=2: translations compared and how many differed, logged every 10 s by
// TickTranslations; the first 8 differences are logged by the translation thread that found them.
std::atomic<u64> verify_compared{};
std::atomic<u64> verify_differing{};
std::atomic<u32> verify_logged{};

/// The worker's translation against the GPU thread's: the SPIR-V of each stage word for word,
/// then the reads each environment recorded. Empty when they match.
std::string CompareTranslations(const TranslationJob& job) {
    const TranslationRecord& reference = *job.reference;
    const TranslationRecord& record = *job.record;
    if (reference.complete != record.complete) {
        return fmt::format("the translation {} on the GPU thread and {} on its thread",
                           reference.complete ? "completed" : "failed",
                           record.complete ? "completed" : "failed");
    }
    for (size_t stage = 0; stage < record.spirv.size(); ++stage) {
        const auto& expected = reference.spirv[stage];
        const auto& got = record.spirv[stage];
        if (expected == got) {
            continue;
        }
        const auto [mismatch, unused] = std::ranges::mismatch(expected, got);
        return fmt::format("stage {} SPIR-V: {} words on the GPU thread, {} on its thread, first "
                           "difference at word {}",
                           stage, expected.size(), got.size(),
                           std::distance(expected.begin(), mismatch));
    }
    for (size_t index = 0; index < job.envs.size(); ++index) {
        if (job.key.unique_hashes[index] == 0) {
            continue;
        }
        if (std::string difference = job.envs[index].FirstReadDifference(job.reference_envs[index]);
            !difference.empty()) {
            return fmt::format("program {} reads: {}", index, difference);
        }
    }
    return {};
}
} // Anonymous namespace

bool PipelineCache::EnqueueTranslation(
    const VideoCommon::EvictedPointerSlots::Slots* keep_pointers) {
    if (!DrawMayBeSkipped() || translate_sync_only.contains(graphics_key)) {
        return false;
    }
    if (translate_queue->InFlight() >= MaxTranslationsInFlight) {
        ++translate_stats.full;
        return false;
    }
    const auto begin = std::chrono::steady_clock::now();
    auto job = std::make_unique<TranslationJob>();
    job->key = graphics_key;
    job->native = maxwell3d->GetNativeMacroEngine().native_turing;
    job->enqueued = begin;
    GraphicsEnvironments environments;
    translate_stats.snapshot_bytes +=
        GetGraphicsEnvironments(environments, graphics_key.unique_hashes, true);
    KeepPointers(environments.envs, graphics_key, keep_pointers);
    // A copy (environments declare a destructor, so they have no moves); the snapshot is shared.
    job->envs = environments.envs;
    if (VideoCommon::AsyncTranslateMode() == 2) {
        // Verify mode: the same draw translated here from the live state too, as the GPU thread
        // would have, for the translation thread to compare with its own (SPIR-V and reads).
        GraphicsEnvironments live;
        GetGraphicsEnvironments(live, graphics_key.unique_hashes);
        KeepPointers(live.envs, graphics_key, keep_pointers);
        job->reference = std::make_unique<TranslationRecord>();
        job->reference->translate_only = true;
        // Pools of its own: nothing else on the GPU thread holds IR in them.
        if (!verify_pools) {
            verify_pools = std::make_unique<ShaderPools>();
        }
        verify_pools->ReleaseContents();
        [[maybe_unused]] const auto none =
            CreateGraphicsPipeline(*verify_pools, graphics_key, live.Span(), nullptr, true,
                                   nullptr, nullptr, false, job->reference.get());
        job->reference_envs = live.envs;
        job->record = std::make_unique<TranslationRecord>();
    }
    const auto elapsed = std::chrono::steady_clock::now() - begin;
    translate_stats.enqueue_time += elapsed;
    frame_tally.queued_time += elapsed;
    ++frame_tally.queued;
    if (!translate_queue->Enqueue(std::move(job))) {
        ++translate_stats.full;
        return false;
    }
    ++translate_stats.queued;
    return true;
}

std::unique_ptr<GraphicsPipeline> PipelineCache::AdoptTranslation(bool& pending) {
    pending = false;
    if (!translate_queue->Contains(graphics_key)) {
        // Not translating: a key that made no pipeline, which stays empty as before.
        return nullptr;
    }
    auto job = translate_queue->Take(graphics_key);
    if (!job) {
        if (DrawMayBeSkipped()) {
            NoteSkippedLookup();
            ++translate_stats.skipped;
            pending = true;
            return nullptr;
        }
        // A draw that must not be skipped (a full-screen pass) needs a pipeline a skippable draw
        // queued: wait a little for it, then build it here. Never a near match.
        const auto begin = std::chrono::steady_clock::now();
        job = translate_queue->Wait(graphics_key, TranslateWaitCap);
        ++translate_stats.waits;
        const auto waited = std::chrono::steady_clock::now() - begin;
        translate_stats.wait_time += waited;
        discard_stats.queue_wait += waited;
        if (!job) {
            translate_queue->Abandon(graphics_key);
            ++translate_stats.fallbacks;
            return CreateGraphicsPipeline(KeptPointers());
        }
    }
    bool retry{};
    auto pipeline = AdoptJob(std::move(job), retry);
    if (!retry) {
        return pipeline;
    }
    // Not usable: translate again from a fresh snapshot of this draw, or build it here (an
    // evicted key's kept pointer slots stay kept until its pipeline is made).
    if (!translate_sync_only.contains(graphics_key) && DrawMayBeSkipped() &&
        EnqueueTranslation(KeptPointers())) {
        ++translate_stats.requeued;
        NoteSkippedLookup();
        ++translate_stats.skipped;
        pending = true;
        return nullptr;
    }
    ++translate_stats.fallbacks;
    return CreateGraphicsPipeline(KeptPointers());
}

std::unique_ptr<GraphicsPipeline> PipelineCache::AdoptJob(std::unique_ptr<TranslationJob> job,
                                                          bool& retry) {
    retry = false;
    switch (job->outcome) {
    case TranslationJob::Outcome::Built:
        if (std::ranges::all_of(job->envs, [](GraphicsEnvironment& env) {
                return env.TextureReadsStillMatch();
            })) {
            const auto ready = std::chrono::steady_clock::now() - job->enqueued;
            ++translate_stats.adopted;
            translate_stats.translation_time += job->translation;
            translate_stats.ready_total += ready;
            translate_stats.ready_max = (std::max)(translate_stats.ready_max, ready);
            translate_failures.erase(job->key);
            auto pipeline = std::move(job->pipeline);
            if (!pipeline_cache_filename.empty()) {
                StoreRuntimeGraphicsPipeline(job->key, job->envs);
            }
            return pipeline;
        }
        ++translate_stats.stale;
        break;
    case TranslationJob::Outcome::Retry:
        ++translate_stats.gave_up;
        break;
    case TranslationJob::Outcome::Differs:
        // Verify mode: a snapshot gap (logged by the translation thread). Built on the GPU
        // thread from the live state, now and from then on.
        ++translate_stats.differs;
        translate_sync_only.insert(job->key);
        break;
    case TranslationJob::Outcome::NoPipeline:
    case TranslationJob::Outcome::Pending:
        ++translate_stats.no_pipeline;
        // A pipeline made before a failure on its thread is retired, not destroyed here.
        translate_queue->Discard(std::move(job));
        return nullptr;
    }
    retry = true;
    if (++translate_failures[job->key] >= MaxTranslateFailures) {
        translate_sync_only.insert(job->key);
    }
    translate_queue->Discard(std::move(job));
    return nullptr;
}

void PipelineCache::TranslateOnWorker(TranslationJob& job) {
    boost::container::static_vector<Shader::Environment*, Maxwell::MaxShaderProgram> env_ptrs;
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (job.key.unique_hashes[index] != 0) {
            env_ptrs.push_back(&job.envs[index]);
        }
    }
    // As the GPU thread's build: SPIR-V emission and the driver compile on the pipeline workers
    // when the emission budget has room, otherwise the SPIR-V here.
    std::shared_ptr<ShaderPools> emit_pools;
    auto reservation = emission_budget.TryAcquire(
        VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::AsyncPipelines), job.native);
    if (reservation) {
        {
            std::scoped_lock lock{spare_pools_mutex};
            if (!spare_pools.empty()) {
                emit_pools = std::move(spare_pools.back());
                spare_pools.pop_back();
            }
        }
        if (!emit_pools) {
            emit_pools = std::make_shared<ShaderPools>();
        }
    }
    thread_local ShaderPools thread_pools;
    if (!emit_pools) {
        thread_pools.ReleaseContents();
    }
    ShaderPools& pools = emit_pools ? *emit_pools : thread_pools;
    t_translate_worker = true;
    SCOPE_EXIT {
        t_translate_worker = false;
    };
    job.pipeline = CreateGraphicsPipeline(pools, job.key, MakeSpan(env_ptrs), nullptr, true,
                                          std::move(emit_pools), std::move(reservation), false,
                                          job.record.get());
    if (std::ranges::any_of(job.envs, [](const GraphicsEnvironment& env) {
            return env.GuardedReadFailed();
        })) {
        // Translated with zeros where a read gave up: never used, translated again from a fresh
        // snapshot. The pipeline stays in the job: the GPU thread retires it (AdoptJob, Discard)
        // rather than destroying it on this thread.
        job.outcome = TranslationJob::Outcome::Retry;
    } else if (job.reference && job.record) {
        verify_compared.fetch_add(1, std::memory_order_relaxed);
        if (const std::string difference = CompareTranslations(job); !difference.empty()) {
            verify_differing.fetch_add(1, std::memory_order_relaxed);
            if (verify_logged.fetch_add(1, std::memory_order_relaxed) < 8) {
                LOG_WARNING(Render_Vulkan, "Async translate verify: {} differs: {}",
                            PipelineShaders(job.key), difference);
            }
            // Never used: retired by the GPU thread (AdoptJob, Discard), which builds the key
            // itself from the live state.
            job.outcome = TranslationJob::Outcome::Differs;
        }
    }
    // The comparison is done; the copies go with the job's snapshot bytes.
    job.reference.reset();
    job.record.reset();
    for (GraphicsEnvironment& env : job.reference_envs) {
        env = GraphicsEnvironment{};
    }
    for (GraphicsEnvironment& env : job.envs) {
        env.ReleaseSnapshotBytes();
    }
}

void PipelineCache::TickTranslations() {
    translate_queue->RetireAbandoned();
    // Translations whose draws did not come back yet: adopted now, so the next draw finds them.
    for (auto& job : translate_queue->TakeFinished()) {
        const auto it = graphics_cache.find(job->key);
        if (it == graphics_cache.end() || it->second) {
            translate_queue->Discard(std::move(job));
            continue;
        }
        bool retry{};
        if (auto pipeline = AdoptJob(std::move(job), retry)) {
            it->second = std::move(pipeline);
            ++translate_stats.adopted_between_frames;
            if (evicted_pointer_slots.Size() != 0) {
                evicted_pointer_slots.Forget(it->first.Hash());
            }
        } else if (retry) {
            // Not usable, and no draw to take a fresh snapshot from: the next lookup of the key
            // is a new one (queued again, or built on the GPU thread once it failed too often).
            graphics_cache.erase(it);
        }
    }
    if (frame_count % 60 == 0) {
        for (const auto& key : translate_queue->Sweep(TranslateSweepAge)) {
            ++translate_stats.swept;
            translate_sync_only.insert(key);
            if (const auto it = graphics_cache.find(key);
                it != graphics_cache.end() && !it->second) {
                graphics_cache.erase(it);
            }
            LOG_WARNING(Render_Vulkan,
                        "Async translate: {} still translating after {} s; built on the GPU "
                        "thread at its next draw instead",
                        PipelineShaders(key), TranslateSweepAge.count());
        }
    }
    auto& stats = translate_stats;
    const auto now = std::chrono::steady_clock::now();
    if (stats.window == std::chrono::steady_clock::time_point{}) {
        stats.window = now;
        return;
    }
    if (now - stats.window < std::chrono::seconds{10}) {
        return;
    }
    const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
    const u32 adopted = stats.adopted;
    LOG_INFO(Render_Vulkan,
             "Async translate in the last 10 s: {} queued ({:.1f} ms of GPU thread for "
             "environments and snapshots, {} KB copied, {} not queued: full), {} adopted ({} "
             "between frames; first lookup to ready {:.1f} ms average, {:.1f} ms worst; {:.1f} ms "
             "translating on its threads), {} lookups skipped while translating, {} waits "
             "({:.1f} ms), {} built on the GPU thread after all, {} queued again ({} stale "
             "texture descriptors, {} guarded reads gave up), {} made no pipeline, {} swept, {} "
             "in flight",
             stats.queued, ms(stats.enqueue_time), stats.snapshot_bytes / 1024, stats.full,
             adopted, stats.adopted_between_frames,
             adopted != 0 ? ms(stats.ready_total) / adopted : 0.0, ms(stats.ready_max),
             ms(stats.translation_time), stats.skipped, stats.waits, ms(stats.wait_time),
             stats.fallbacks, stats.requeued, stats.stale, stats.gave_up, stats.no_pipeline,
             stats.swept, translate_queue->InFlight());
    if (VideoCommon::AsyncTranslateMode() == 2) {
        LOG_INFO(Render_Vulkan,
                 "Async translate verify: {} compared, {} differ ({} rejected at adoption)",
                 verify_compared.exchange(0, std::memory_order_relaxed),
                 verify_differing.exchange(0, std::memory_order_relaxed), stats.differs);
    }
    stats = {};
    stats.window = now;
}

ComputePipelineCacheEntry PipelineCache::CreateComputePipeline(const ComputePipelineCacheKey& key,
                                                               const ShaderInfo* shader,
                                                               std::span<const u64> keep_pointers,
                                                               bool store) {
    const auto begin = std::chrono::steady_clock::now();
    SCOPE_EXIT {
        ReportRuntimePipeline("Compute", begin,
                              fmt::format("pipeline={:016x} s6:{:016x} address={:#x}", key.Hash(),
                                          key.unique_hash,
                                          kepler_compute->regs.code_loc.Address() +
                                              kepler_compute->launch_description.program_start));
    };
    const GPUVAddr program_base{kepler_compute->regs.code_loc.Address()};
    const auto& qmd{kepler_compute->launch_description};
    ComputeEnvironment env{*kepler_compute, *gpu_memory, program_base, qmd.program_start,
                           shader->lowered_native};
    env.SetCachedSize(shader->size_bytes);
    env.KeepStoragePointers(keep_pointers);

    // EDEN_NX2_PRECOMPILED_FAST: a precompiled program whose recorded answers hold is made from
    // its kept info and SPIR-V. A rebuild too (keep_pointers): its environment answers the kept
    // pointers as pointers before anything is asked, as its translation would see them.
    auto entry{CreatePrecompiledComputePipeline(key, env)};
    if (!entry.pipeline) {
        main_pools.ReleaseContents();
        entry = CreateComputePipeline(main_pools, key, env, nullptr, true);
    }
    if (const char* directory = std::getenv("EDEN_NX2_SHADER_DUMP");
        directory && !entry.pipeline && env.CanBeSerialized()) {
        const auto path = std::filesystem::path{directory} /
                          fmt::format("failed-compute-{:016x}-v{}.env", key.unique_hash, CACHE_VERSION);
        std::ofstream file{path, std::ios::binary};
        if (file) {
            env.Serialize(file);
            LOG_INFO(Render_Vulkan, "Captured failed compute environment: {} written={}",
                     path.string(), file.good());
        }
    }
    if (!entry.pipeline || pipeline_cache_filename.empty() || !store) {
        return entry;
    }
    serialization_thread.QueueWork([this, key, env_ = std::move(env)] {
        SerializePipeline(key, std::array<const GenericEnvironment*, 1>{&env_},
                          pipeline_cache_filename, CACHE_VERSION);
    });
    QueueVulkanPipelineCacheFlush();
    return entry;
}

ComputePipelineCacheEntry PipelineCache::CreateComputePipeline(
    ShaderPools& pools, const ComputePipelineCacheKey& key, Shader::Environment& env,
    PipelineStatistics* statistics, bool build_in_parallel, bool warm, ComputeKeep* keep) try {
    auto hash = key.Hash();
    if (device.HasBrokenCompute()) {
        if (!warm) {
            LOG_ERROR(Render_Vulkan, "Skipping {:#016x}", hash);
        }
        return {.failure = "Device reports broken compute support"};
    }

    LOG_DEBUG(Render_Vulkan, "{:#016x}", hash);

    Shader::Maxwell::Flow::CFG cfg{env, pools.flow_block, env.StartAddress()};

    // Dump it before error.
    if (Settings::values.dump_guest_shaders) {
        env.Dump(hash, key.unique_hash);
    }

    auto program{TranslateProgram(pools.inst, pools.block, env, cfg, host_info)};
    if (const auto edits = Shader::Optimization::ApplyOotVisualWorkaround(
            program, program_id != 0 ? program_id : visual_workaround_title, key.unique_hash,
            oot_visual_workaround)) {
        LOG_WARNING(Render_Vulkan,
                    "TEMPORARY OoT visual workaround: shader {:016x}, {} scalar edits",
                    key.unique_hash, *edits);
    }
    const VkDriverIdKHR driver_id = device.GetDriverID();
    const bool needs_shared_mem_clamp =
        driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY ||
        driver_id == VK_DRIVER_ID_ARM_PROPRIETARY || driver_id == VK_DRIVER_ID_MOLTENVK;
    const u32 max_shared_memory = device.GetMaxComputeSharedMemorySize();
    if (needs_shared_mem_clamp && program.shared_memory_size > max_shared_memory) {
        LOG_WARNING(Render_Vulkan,
                    "Compute shader {:#016x} requests {}KB shared memory but device max is {}KB - clamping",
                    key.unique_hash,
                    program.shared_memory_size / 1024,
                    max_shared_memory / 1024);
        program.shared_memory_size = max_shared_memory;
    }
    std::vector<u32> code{EmitSPIRV(profile, program)};
    if (t_capture_compute) {
        // LICHII_PRECOMPILE: the archive compute translation is stored with its info.
        t_capture_compute->first = code;
        t_capture_compute->second = program.info;
    }
    if (keep) {
        // The precompiled preparation keeps these (EDEN_NX2_PRECOMPILED_FAST); its audit
        // compares them and stops here.
        keep->spirv = code;
        keep->info.clear();
        Shader::SerializeInfo(program.info, keep->info);
        keep->storage_descriptors =
            static_cast<u32>(program.info.storage_buffers_descriptors.size());
        keep->proven_read_extents = ProvenReadExtents(program.info);
        if (keep->translate_only) {
            return {};
        }
    }
    SelfCheckSpirv(code, key.unique_hash, "compute");
    device.SaveShader(code);
    vk::ShaderModule spv_module{BuildShader(device, code)};

    // Text log + .spv dump. Same split as the graphics path.
    const bool should_log = GPU::Logging::IsActive();
    const bool should_dump = Settings::values.gpu_log_shader_dumps.GetValue();
    if (should_log || should_dump) {
        const std::string shader_name = fmt::format("shader_{:016x}_compute", key.unique_hash);
        if (should_log) {
            const std::string shader_info = fmt::format("SPIR-V size: {} bytes, hash: {:016x}",
                code.size() * sizeof(u32), key.unique_hash);
            GPU::Logging::GPULogger::GetInstance().LogShaderCompilation(shader_name, shader_info);
        }
        if (should_dump) {
            GPU::Logging::DumpSpirvShader(key.unique_hash,
                                          std::span<const u32>(code.data(), code.size()));
        }
    }

    if (device.HasDebuggingToolAttached()) {
        const auto name{fmt::format("Shader {:016x}", key.unique_hash)};
        spv_module.SetObjectNameEXT(name.c_str());
    }
    Common::ThreadWorker* const thread_worker{build_in_parallel ? &workers : nullptr};
    const Shader::Info info{std::move(program.info)};
    if (!thread_worker) {
        // The driver compiles on this thread. Precache and boot workers would otherwise hold their
        // IR through the compile, up to hundreds of MiB for one large shader.
        program = {};
        code = std::vector<u32>{};
        pools.block.Free();
        pools.inst.Free();
    }
    return {.pipeline = std::make_unique<ComputePipeline>(
                device, scheduler, warm ? WarmPipelineCache() : PlayPipelineCache(),
                descriptor_pool, guest_descriptor_queue,
                descriptor_buffer_ring, thread_worker, statistics,
                warm || t_capture_compute ? nullptr : &shader_notify, info,
                std::move(spv_module), key.unique_hash),
            .null_pointer_slots = env.IsNativeProgram() ? env.NullStoragePointerSlots()
                                                        : std::vector<std::pair<u32, u32>>{},
            .mapped_pointers =
                env.IsNativeProgram() ? env.MappedStoragePointerKeys() : std::vector<u64>{}};

} catch (const Shader::Exception& exception) {
    if (warm) {
        // A warm-up guess the translation rejects: counted by the warm-up, logged at its end.
        LOG_DEBUG(Render_Vulkan, "Warm compute pipeline shader={:#016x} pipeline={:#016x}: {}",
                  key.unique_hash, key.Hash(), exception.what());
    } else if (!build_in_parallel) {
        // A disk cache record this build cannot translate, as for graphics: built from the game
        // when first dispatched.
        NoteDiskRecordFailure("Compute", key.Hash(), exception.what());
    } else {
        LOG_ERROR(Render_Vulkan,
                  "Compute pipeline compilation failed shader={:#016x} pipeline={:#016x}: {}",
                  key.unique_hash, key.Hash(), exception.what());
    }
    return {.failure = exception.what()};
}

void PipelineCache::SerializeVulkanPipelineCache(const std::filesystem::path& filename,
                                                 const vk::PipelineCache& pipeline_cache,
                                                 u32 cache_version) try {
    // Written beside the cache and renamed over it: the emulator stopped mid-write left the cache
    // empty, and every later boot failed to read it.
    auto temporary = filename;
    temporary += ".tmp";
    std::ofstream file(temporary, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR(Common_Filesystem, "Failed to open Vulkan driver pipeline cache file {}",
                  Common::FS::PathToUTF8String(temporary));
        return;
    }
    file.exceptions(std::ifstream::failbit);
    file.write(VULKAN_CACHE_MAGIC_NUMBER.data(), VULKAN_CACHE_MAGIC_NUMBER.size())
        .write(reinterpret_cast<const char*>(&cache_version), sizeof(cache_version));

    size_t cache_size = 0;
    std::vector<char> cache_data;
    if (pipeline_cache) {
        pipeline_cache.Read(&cache_size, nullptr);
        cache_data.resize(cache_size);
        pipeline_cache.Read(&cache_size, cache_data.data());
    }
    Common::NoteShutdownProgress();
    // In pieces: a cache of tens of thousands of pipelines is gigabytes, and a shutdown waiting
    // on it is told the write still moves.
    constexpr size_t Chunk = size_t{64} << 20;
    for (size_t offset = 0; offset < cache_size; offset += Chunk) {
        file.write(cache_data.data() + offset,
                   static_cast<std::streamsize>((std::min)(Chunk, cache_size - offset)));
        Common::NoteShutdownProgress();
    }
    file.close();
    std::error_code error;
    std::filesystem::rename(temporary, filename, error);
    if (error) {
        LOG_ERROR(Common_Filesystem, "Failed to replace Vulkan driver pipeline cache file {}: {}",
                  Common::FS::PathToUTF8String(filename), error.message());
        std::filesystem::remove(temporary, error);
        return;
    }

    LOG_INFO(Render_Vulkan, "Vulkan driver pipelines cached at: {}",
             Common::FS::PathToUTF8String(filename));

} catch (const std::ios_base::failure& e) {
    // Only the half-written copy goes: the cache on disk is still whole (deleting it on a failed
    // write, e.g. a full disk, made the next boot compile every pipeline again).
    LOG_ERROR(Common_Filesystem, "Writing the Vulkan driver pipeline cache failed: {}", e.what());
    auto temporary = filename;
    temporary += ".tmp";
    std::error_code error;
    std::filesystem::remove(temporary, error);
}

vk::PipelineCache PipelineCache::LoadVulkanPipelineCache(const std::filesystem::path& filename,
                                                         u32 expected_cache_version) {
    const auto create_pipeline_cache = [this](size_t data_size, const void* data) {
        VkPipelineCacheCreateInfo pipeline_cache_ci = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .initialDataSize = data_size,
            .pInitialData = data};
        return device.GetLogical().CreatePipelineCache(pipeline_cache_ci);
    };
    try {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return create_pipeline_cache(0, nullptr);
        }
        file.exceptions(std::ifstream::failbit);
        const auto end{file.tellg()};
        file.seekg(0, std::ios::beg);

        std::array<char, 8> magic_number;
        u32 cache_version;
        // A file cut short (the emulator stopped while writing it) would throw from the reads
        // below, and macOS's libc++ throws an ios_base::failure the handler doesn't match.
        if (static_cast<size_t>(end) < magic_number.size() + sizeof(cache_version)) {
            file.close();
            LOG_ERROR(Common_Filesystem, "Truncated Vulkan driver pipeline cache file, deleting it");
            Common::FS::RemoveFile(filename);
            return create_pipeline_cache(0, nullptr);
        }
        file.read(magic_number.data(), magic_number.size())
            .read(reinterpret_cast<char*>(&cache_version), sizeof(cache_version));
        if (magic_number != VULKAN_CACHE_MAGIC_NUMBER || cache_version != expected_cache_version) {
            file.close();
            if (Common::FS::RemoveFile(filename)) {
                if (magic_number != VULKAN_CACHE_MAGIC_NUMBER) {
                    LOG_ERROR(Common_Filesystem, "Invalid Vulkan driver pipeline cache file");
                }
                if (cache_version != expected_cache_version) {
                    LOG_INFO(Common_Filesystem, "Deleting old Vulkan driver pipeline cache");
                }
            } else {
                LOG_ERROR(Common_Filesystem,
                          "Invalid Vulkan pipeline cache file and failed to delete it in \"{}\"",
                          Common::FS::PathToUTF8String(filename));
            }
            return create_pipeline_cache(0, nullptr);
        }

        static constexpr size_t header_size = magic_number.size() + sizeof(cache_version);
        const size_t cache_size = static_cast<size_t>(end) - header_size;
        std::vector<char> cache_data(cache_size);
        file.read(cache_data.data(), cache_size);

        LOG_INFO(Render_Vulkan,
                 "Loaded Vulkan driver pipeline cache: {}", Common::FS::PathToUTF8String(filename));

        try {
            return create_pipeline_cache(cache_size, cache_data.data());
        } catch (const vk::Exception& e) {
            // The blob came from another driver or driver version (e.g. switching between
            // MoltenVK and KosmicKrisp on macOS): the driver rejects it. Start empty instead of
            // aborting, and drop the stale file so the next run doesn't retry it.
            LOG_ERROR(Render_Vulkan,
                      "Driver rejected the Vulkan pipeline cache file ({}), starting with an empty "
                      "cache and deleting it",
                      e.what());
            Common::FS::RemoveFile(filename);
            return create_pipeline_cache(0, nullptr);
        }

    } catch (const std::ios_base::failure& e) {
        LOG_ERROR(Common_Filesystem, "{}", e.what());
        if (!Common::FS::RemoveFile(filename)) {
            LOG_ERROR(Common_Filesystem, "Failed to delete Vulkan driver pipeline cache file {}",
                      Common::FS::PathToUTF8String(filename));
        }

        return create_pipeline_cache(0, nullptr);
    }
}

void PipelineCache::PrecompileArchives(std::stop_token stop_loading,
                                       const VideoCore::DiskResourceLoadCallback& callback) {
    const char* dir = std::getenv("LICHII_PRECOMPILE_DIR");
    if (dir == nullptr || dir[0] == '\0') {
        return;
    }
    if (!SpirvReuseEnabled()) {
        LOG_WARNING(Render_Vulkan, "LICHII_PRECOMPILE_DIR needs LICHII_SPIRV_REUSE=1");
        return;
    }
    const auto begin = std::chrono::steady_clock::now();
    // The template: the most common shader-relevant state among the pipelines built at boot,
    // every vertex attribute read as a float.
    std::optional<FixedPipelineState> templ;
    {
        std::unordered_map<u64, std::pair<u32, FixedPipelineState>> counts;
        for (const auto& [key, pipeline] : graphics_cache) {
            FixedPipelineState reuse = ReuseState(key.state);
            if (reuse.dynamic_vertex_input) {
                reuse.attribute_types = 0;
            }
            const u64 hash =
                Common::CityHash64(reinterpret_cast<const char*>(&reuse), reuse.Size());
            auto& entry = counts[hash];
            if (entry.first++ == 0) {
                entry.second = key.state;
            }
        }
        u32 best = 0;
        for (const auto& [hash, entry] : counts) {
            if (entry.first > best) {
                best = entry.first;
                templ = entry.second;
            }
        }
    }
    // The template is kept next to the pipeline cache: an emptied cache still has it.
    const auto template_path =
        pipeline_cache_filename.parent_path() / "lichii_precompile_state.bin";
    if (templ) {
        std::ofstream out{template_path, std::ios::binary};
        out.write(reinterpret_cast<const char*>(&*templ), sizeof(*templ));
    } else {
        std::ifstream in{template_path, std::ios::binary};
        FixedPipelineState stored{};
        if (in.read(reinterpret_cast<char*>(&stored), sizeof(stored))) {
            templ = stored;
        }
    }
    if (!templ) {
        LOG_WARNING(Render_Vulkan, "LICHII_PRECOMPILE_DIR: no cached pipeline to take a state from");
        return;
    }
    if (templ->dynamic_vertex_input) {
        templ->attribute_types = 0x5555555555555555ULL;
    }
    // The game draws most of its models with and without the depth test: both are translated.
    std::vector<FixedPipelineState> templates{*templ};
    {
        FixedPipelineState other = *templ;
        other.depth_enabled.Assign(other.depth_enabled ? 0 : 1);
        templates.push_back(other);
    }
    // Vertex attribute types the game binds (bone indices as integers, packed colors...): which
    // vertex program was drawn with which types is learned from the pipelines in the disk cache
    // and kept next to it (lichii_precompile_attributes2.bin). Each pair is translated with float
    // attributes first; the second phase adds the types its vertex program was seen with, or the
    // few most common ones for a program the game has not drawn yet.
    constexpr u64 AllFloatAttributes = 0x5555555555555555ULL;
    constexpr size_t CommonAttributeSets = 3;
    std::vector<std::pair<u64, u64>> attribute_uses; ///< Vertex program hash, attribute types
    std::vector<u64> attribute_sets;                 ///< Distinct, most used first
    if (templ->dynamic_vertex_input) {
        const auto attributes_path =
            pipeline_cache_filename.parent_path() / "lichii_precompile_attributes2.bin";
        std::unordered_set<u64> seen;
        const auto remember = [&](u64 vertex, u64 set) {
            if (seen.insert(Common::CityHash64(reinterpret_cast<const char*>(&vertex), 8) ^ set)
                    .second) {
                attribute_uses.emplace_back(vertex, set);
            }
        };
        if (std::ifstream in{attributes_path, std::ios::binary}) {
            u32 count{};
            in.read(reinterpret_cast<char*>(&count), sizeof(count));
            for (u32 i = 0; in && i < count && i < (1u << 20); ++i) {
                std::pair<u64, u64> use{};
                if (in.read(reinterpret_cast<char*>(&use), sizeof(use))) {
                    remember(use.first, use.second);
                }
            }
        }
        for (const auto& [key, pipeline] : graphics_cache) {
            if (key.state.dynamic_vertex_input && key.state.attribute_types != AllFloatAttributes) {
                remember(key.unique_hashes[1], key.state.attribute_types);
            }
        }
        std::ofstream out{attributes_path, std::ios::binary};
        const u32 count = static_cast<u32>(attribute_uses.size());
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        out.write(reinterpret_cast<const char*>(attribute_uses.data()),
                  static_cast<std::streamsize>(attribute_uses.size() * sizeof(attribute_uses[0])));
        std::unordered_map<u64, u32> uses_per_set;
        for (const auto& [vertex, set] : attribute_uses) {
            ++uses_per_set[set];
        }
        for (const auto& [set, uses] : uses_per_set) {
            attribute_sets.push_back(set);
        }
        std::ranges::sort(attribute_sets, [&](u64 a, u64 b) {
            return uses_per_set[a] != uses_per_set[b] ? uses_per_set[a] > uses_per_set[b] : a < b;
        });
        // The stored translations name their template in a byte.
        if (attribute_sets.size() > 100) {
            attribute_sets.resize(100);
        }
    }
    // Mesh programs translate with registers and a pass-through header the archive does not
    // carry: the few combinations the game uses are learned from the disk cache and kept next to
    // it (lichii_precompile_mesh.bin), with the most common mesh pipeline state.
    std::vector<std::array<u32, 23>> mesh_states;
    std::optional<FixedPipelineState> mesh_templ;
    {
        const auto mesh_path = pipeline_cache_filename.parent_path() / "lichii_precompile_mesh.bin";
        if (std::ifstream in{mesh_path, std::ios::binary}) {
            u32 has_template{}, count{};
            FixedPipelineState stored{};
            in.read(reinterpret_cast<char*>(&has_template), sizeof(has_template));
            in.read(reinterpret_cast<char*>(&stored), sizeof(stored));
            in.read(reinterpret_cast<char*>(&count), sizeof(count));
            if (in && has_template) {
                mesh_templ = stored;
            }
            for (u32 i = 0; in && i < count && i < 4096; ++i) {
                std::array<u32, 23> state{};
                if (in.read(reinterpret_cast<char*>(state.data()), sizeof(state))) {
                    mesh_states.push_back(state);
                }
            }
        }
        for (const auto& state : observed_mesh_states) {
            if (std::ranges::find(mesh_states, state) == mesh_states.end()) {
                mesh_states.push_back(state);
            }
        }
        u32 best = 0;
        for (const auto& [hash, entry] : observed_mesh_templates) {
            if (entry.first > best) {
                best = entry.first;
                mesh_templ = entry.second;
            }
        }
        std::ofstream out{mesh_path, std::ios::binary};
        const u32 has_template = mesh_templ ? 1U : 0U;
        const FixedPipelineState stored = mesh_templ.value_or(FixedPipelineState{});
        const u32 count = static_cast<u32>(mesh_states.size());
        out.write(reinterpret_cast<const char*>(&has_template), sizeof(has_template));
        out.write(reinterpret_cast<const char*>(&stored), sizeof(stored));
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (const auto& state : mesh_states) {
            out.write(reinterpret_cast<const char*>(state.data()), sizeof(state));
        }
    }
    std::vector<FixedPipelineState> mesh_templates;
    if (mesh_templ && !mesh_states.empty()) {
        // Mesh pipelines nearly always keep the depth test of the template (83 of 91 cached).
        mesh_templates.push_back(*mesh_templ);
    }
    // The programs of every BNSH (vertex and pixel; mesh and pixel with the learned mesh states;
    // task and geometry programs need state the archive does not carry).
    struct Program {
        std::vector<u8> bytes;
        std::vector<u32> bank1;
        Shader::Stage stage{};
        std::shared_ptr<const VideoCommon::LoweredNativeCode> lowered;
        u64 hash{};
        std::array<u32, 3> workgroup{}; ///< Compute only
        u32 shared_memory{};            ///< Compute only
    };
    std::vector<Program> programs;
    std::vector<size_t> compute_programs;
    std::unordered_set<size_t> compute_seen;
    std::unordered_map<u64, size_t> by_bytes;
    std::vector<std::pair<size_t, size_t>> pairs;
    struct MeshSet {
        size_t task; ///< SIZE_MAX: none
        size_t mesh;
        size_t pixel;
    };
    std::vector<MeshSet> mesh_pairs;
    std::unordered_set<std::string> mesh_seen;
    std::unordered_set<u64> pair_seen;
    size_t files = 0, variations = 0, skipped_variations = 0;
    const auto add_program = [&](const std::vector<char>& data, size_t base, u64 code_info,
                                 Shader::Stage stage) -> std::optional<size_t> {
        const auto u32_at = [&](size_t o) { u32 v; std::memcpy(&v, data.data() + o, 4); return v; };
        const auto u64_at = [&](size_t o) { u64 v; std::memcpy(&v, data.data() + o, 8); return v; };
        const size_t ci = base + code_info;
        if (ci + 0x20 > data.size()) {
            return std::nullopt;
        }
        const u64 control = u64_at(ci + 0x08);
        const u64 code = u64_at(ci + 0x10);
        const u32 size = u32_at(ci + 0x18);
        if (size < 0x90 || base + code + size > data.size()) {
            return std::nullopt;
        }
        const u8* first = reinterpret_cast<const u8*>(data.data() + base + code);
        u64 h = 1469598103934665603ULL;
        for (u32 i = 0; i < size; ++i) {
            h = (h ^ first[i]) * 1099511628211ULL;
        }
        h ^= static_cast<u64>(stage) << 56;
        if (const auto it = by_bytes.find(h); it != by_bytes.end()) {
            return it->second;
        }
        Program program;
        if (stage == Shader::Stage::Compute) {
            // Compute programs start with code: the runtime lowers them with a zeroed header.
            program.bytes.assign(128, u8{0});
            program.bytes.insert(program.bytes.end(), first, first + size);
            // The launch values NVN puts in the dispatch come with the program's control block.
            if (base + control + 0x68 <= data.size()) {
                program.workgroup = {u32_at(base + control + 0x58), u32_at(base + control + 0x5c),
                                     u32_at(base + control + 0x60)};
                program.shared_memory = u32_at(base + control + 0x64);
            }
        } else {
            program.bytes.assign(first, first + size);
        }
        program.stage = stage;
        if (base + control + 0x48 <= data.size()) {
            const u32 const_size = u32_at(base + control + 0x3c);
            const u32 const_offset = u32_at(base + control + 0x40);
            if (const_size != 0 && const_offset + const_size <= size && const_size < 0x10000) {
                program.bank1.resize(const_size / 4);
                std::memcpy(program.bank1.data(), first + const_offset, program.bank1.size() * 4);
            }
        }
        programs.push_back(std::move(program));
        by_bytes.emplace(h, programs.size() - 1);
        return programs.size() - 1;
    };
    // LICHII_PRECOMPILE_DIR: folders separated by ';' holding the decompressed shader archives
    // (.bfsha, .sarc) and effect resources (.vsetb, each with its own graphics and compute
    // shaders).
    std::vector<std::filesystem::path> archive_files;
    const bool effect_graphics = [] {
        const char* value = std::getenv("LICHII_PRECOMPILE_EFFECT_GRAPHICS");
        return value != nullptr && value[0] == '1';
    }();
    {
        const std::string list{dir};
        size_t start = 0;
        while (start <= list.size()) {
            const size_t end = (std::min)(list.find(';', start), list.size());
            const std::filesystem::path folder{list.substr(start, end - start)};
            start = end + 1;
            std::error_code error;
            if (folder.empty() || !std::filesystem::is_directory(folder, error)) {
                continue;
            }
            for (const auto& entry : std::filesystem::directory_iterator{folder, error}) {
                const auto name = entry.path().filename().string();
                if (entry.is_regular_file() && (name.find(".bfsha") != std::string::npos ||
                                                name.find(".sarc") != std::string::npos ||
                                                name.find(".vsetb") != std::string::npos)) {
                    archive_files.push_back(entry.path());
                }
            }
        }
    }
    // The translations are stored next to the pipeline cache (lichii_precompile_translations.bin)
    // with a fingerprint of what they depend on: archives, the format, the driver. Mesh registers
    // learned later do not invalidate them (their pipelines translate at runtime until the next
    // full translation; LICHII_PRECOMPILE_REBUILD=1 forces one).
    const auto store_path =
        pipeline_cache_filename.parent_path() / "lichii_precompile_translations_v5.bin";
    u64 validity = 0;
    {
        // The driver and the GPU, not the driver's version: an update changes no translation.
        std::string archives;
        for (const auto& path : archive_files) {
            std::error_code error;
            const auto size = std::filesystem::file_size(path, error);
            const auto time = std::filesystem::last_write_time(path, error);
            archives += fmt::format("|{}:{}:{}", path.filename().string(), size,
                                    time.time_since_epoch().count());
        }
        const std::string fingerprint =
            fmt::format("store {} | driver {} | gpu {} | effects {}", PrecompileStoreVersion,
                        static_cast<u32>(device.GetDriverID()), device.GetModelName(),
                        effect_graphics) +
            archives;
        validity = Common::CityHash64(fingerprint.data(), fingerprint.size());
        // Stores written before named the driver's version: one made with this driver and GPU
        // is the same, so its header takes the new fingerprint instead of translating again.
        const std::string legacy =
            fmt::format("store {} | driver {} {} | effects {}", PrecompileStoreVersion,
                        static_cast<u32>(device.GetDriverID()), device.GetDriverVersion(),
                        effect_graphics) +
            archives;
        const u64 legacy_validity = Common::CityHash64(legacy.data(), legacy.size());
        if (std::fstream file{store_path, std::ios::binary | std::ios::in | std::ios::out}) {
            u64 stored = 0;
            file.seekg(8);
            file.read(reinterpret_cast<char*>(&stored), sizeof(stored));
            if (file && stored == legacy_validity) {
                file.seekp(8);
                file.write(reinterpret_cast<const char*>(&validity), sizeof(validity));
                LOG_INFO(Render_Vulkan,
                         "LICHII_PRECOMPILE: stored translations no longer tied to the driver "
                         "version");
            }
        }
    }
    // The driver keeps its cache per executable: another one (an LTO build) builds them again.
    std::string executable;
#ifdef _MSC_VER
    if (char* program = nullptr; _get_pgmptr(&program) == 0 && program != nullptr) {
        executable = std::filesystem::path{program}.filename().string();
    }
#endif
    // LICHII_PRECOMPILE_PIPELINES runs once per store, driver and executable: this file (one per
    // executable) then names them.
    const std::filesystem::path prebuild_marker{
        store_path.string() + (executable.empty() ? "" : "." + executable) + ".prebuilt"};
    const std::string prebuild_marker_value =
        fmt::format("{:016x} {}", validity, device.GetDriverVersion());
    std::vector<FixedPipelineState> store_templates = templates;
    store_templates.insert(store_templates.end(), mesh_templates.begin(), mesh_templates.end());
    // The attribute variants (second phase): one template per learned set and depth variant.
    for (const u64 set : attribute_sets) {
        for (const FixedPipelineState& base : templates) {
            FixedPipelineState variant = base;
            variant.attribute_types = set;
            store_templates.push_back(variant);
        }
    }
    {
        std::vector<GraphicsPipelineCacheKey> stored_keys;
        static const bool rebuild = std::getenv("LICHII_PRECOMPILE_REBUILD") != nullptr;
        if (!rebuild && LoadPrecompileStore(store_path, validity, stored_keys, callback)) {
            LOG_INFO(Render_Vulkan, "LICHII_PRECOMPILE: translations read from {} in {:.1f} s",
                     store_path.string(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
                         .count());
            StartBackgroundPrebuild(std::move(stored_keys), prebuild_marker, prebuild_marker_value,
                                    stop_loading, callback);
            PrecompileShaderObjects(stop_loading, callback, validity);
            return;
        }
    }
    // What the translations below add to reusable_stages is what gets stored.
    std::unordered_set<const ReusableStages*> stores_before;
    {
        std::scoped_lock lock{reusable_stages_mutex};
        for (const auto& [programs_hash, entries] : reusable_stages) {
            for (const auto& entry : entries) {
                stores_before.insert(entry.get());
            }
        }
    }
    std::vector<StoredCompute> stored_compute;
    for (const auto& archive_path : archive_files) {
        std::ifstream file{archive_path, std::ios::binary};
        const std::vector<char> data{std::istreambuf_iterator<char>(file), {}};
        ++files;
        const bool effect = archive_path.filename().string().find(".vsetb") != std::string::npos;
        const auto u32_at = [&](size_t o) { u32 v; std::memcpy(&v, data.data() + o, 4); return v; };
        const auto u64_at = [&](size_t o) { u64 v; std::memcpy(&v, data.data() + o, 8); return v; };
        for (size_t base = 0; base + 0x80 < data.size(); base += 8) {
            if (std::memcmp(data.data() + base, "BNSH", 4) != 0 || u32_at(base + 4) != 0) {
                continue;
            }
            const size_t grsc = base + (static_cast<u8>(data[base + 0x16]) |
                                        (static_cast<u8>(data[base + 0x17]) << 8));
            if (grsc + 0x28 > data.size() || std::memcmp(data.data() + grsc, "grsc", 4) != 0) {
                continue;
            }
            const u32 count = u32_at(grsc + 0x1C);
            const u64 array = u64_at(grsc + 0x20);
            for (u32 v = 0; v < count; ++v) {
                const size_t variation = base + array + size_t{v} * 0x40;
                if (variation + 0x18 > data.size()) {
                    break;
                }
                const u64 program = u64_at(variation + 0x10);
                if (program == 0 || base + program + 0x50 > data.size()) {
                    continue;
                }
                ++variations;
                // Stage slots: vertex, hull, domain, geometry, pixel, compute, -, task, mesh.
                u64 slots[9];
                for (int s = 0; s < 9; ++s) {
                    slots[s] = u64_at(base + program + 0x08 + s * 8);
                }
                if (slots[5] != 0 && std::count(std::begin(slots), std::end(slots), u64{0}) == 8) {
                    // A compute program on its own (effect resources carry these).
                    if (const auto cs = add_program(data, base, slots[5], Shader::Stage::Compute);
                        cs && compute_seen.insert(*cs).second) {
                        compute_programs.push_back(*cs);
                    }
                    continue;
                }
                if (effect && !effect_graphics) {
                    // Each effect carries its own graphics programs (~1,300 sets): translating
                    // them all at once takes more memory than boot can spare, so only on request.
                    ++skipped_variations;
                    continue;
                }
                bool other = false;
                for (int s = 0; s < 9; ++s) {
                    other |= s != 0 && s != 4 && s != 7 && s != 8 && slots[s] != 0;
                }
                if (!other && slots[0] == 0 && slots[8] != 0 && slots[4] != 0) {
                    // (Task +) mesh + pixel.
                    const auto ms = add_program(data, base, slots[8], Shader::Stage::Mesh);
                    const auto ps = add_program(data, base, slots[4], Shader::Stage::Fragment);
                    std::optional<size_t> ts;
                    if (slots[7] != 0) {
                        ts = add_program(data, base, slots[7], Shader::Stage::Task);
                    }
                    const size_t task_index = ts ? *ts : SIZE_MAX;
                    if (ms && ps && (slots[7] == 0 || ts) &&
                        mesh_seen.insert(fmt::format("{}/{}/{}", task_index, *ms, *ps)).second) {
                        mesh_pairs.push_back({task_index, *ms, *ps});
                    }
                    continue;
                }
                if (other || slots[7] != 0 || slots[8] != 0 || slots[0] == 0 || slots[4] == 0) {
                    ++skipped_variations;
                    continue;
                }
                const auto vs = add_program(data, base, slots[0], Shader::Stage::VertexB);
                const auto ps = add_program(data, base, slots[4], Shader::Stage::Fragment);
                if (!vs || !ps) {
                    ++skipped_variations;
                    continue;
                }
                if (pair_seen.insert((u64{*vs} << 32) | *ps).second) {
                    pairs.emplace_back(*vs, *ps);
                }
            }
        }
    }
    // Lower every program as the runtime would (its shader cache key is the hash of the result).
    for (auto& program : programs) {
        workers.QueueWork([&program] {
            try {
                program.lowered =
                    VideoCommon::LowerNativeProgramBytes(program.bytes, program.stage);
            } catch (const std::exception&) {
                program.lowered = nullptr;
            }
            if (program.lowered) {
                program.hash = VideoCommon::NativeProgramHash(program.lowered->code);
            }
        });
    }
    workers.WaitForRequests(stop_loading);
    size_t lowered = 0;
    for (const auto& program : programs) {
        lowered += program.lowered ? 1 : 0;
        if (program.lowered) {
            precompiled_programs.emplace(program.hash, static_cast<u32>(program.stage));
        }
    }
    // Only the 128-byte headers are read from here on (the lowered code has the rest).
    for (auto& program : programs) {
        if (program.bytes.size() > 128) {
            program.bytes.resize(128);
            program.bytes.shrink_to_fit();
        }
    }
    // The translated modules go to a temporary file as they are made (LICHII_PRECOMPILE_SPOOL=0
    // keeps them compressed in memory instead, ~20 GB for the whole game).
    if (!spirv_spool && !spirv_store) {
        const char* spool_env = std::getenv("LICHII_PRECOMPILE_SPOOL");
        if (spool_env == nullptr || spool_env[0] != '0') {
            auto spool = std::make_unique<SpirvSpool>();
            spool->path = pipeline_cache_filename.parent_path() / "lichii_precompile_spool.tmp";
            spool->file.open(spool->path,
                             std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
            if (spool->file) {
                spirv_spool = std::move(spool);
            } else {
                LOG_WARNING(Render_Vulkan,
                            "LICHII_PRECOMPILE: no spool file; translations stay in memory");
            }
        }
    }
    // Translate every pair for reuse: no Vulkan pipeline.
    const size_t limit = [] {
        const char* value = std::getenv("LICHII_PRECOMPILE_MAX");
        return value ? static_cast<size_t>(std::strtoull(value, nullptr, 10)) : size_t{0};
    }();
    std::atomic<size_t> translated{};
    std::atomic<size_t> failed{};
    size_t queued = 0;
    // The loading screen counts these translations.
    size_t total = 0;
    for (const auto& [vs, ps] : pairs) {
        total += programs[vs].lowered && programs[ps].lowered ? templates.size() : 0;
    }
    for (const auto& [ts, ms, ps] : mesh_pairs) {
        total += programs[ms].lowered && programs[ps].lowered &&
                         (ts == SIZE_MAX || programs[ts].lowered)
                     ? mesh_states.size() * mesh_templates.size()
                     : 0;
    }
    for (const size_t cs : compute_programs) {
        total += programs[cs].lowered ? 1 : 0;
    }
    if (limit != 0) {
        total = (std::min)(total, limit);
    }
    std::mutex progress_mutex;
    std::atomic<size_t> done{};
    // One loading bar for both phases: the attribute variants of the second phase are estimated
    // here (each pair gets one per learned set of its vertex program, or the common sets), and
    // the total is corrected to the real count once they are queued.
    size_t phase2_estimate = 0;
    if (!attribute_sets.empty()) {
        std::unordered_map<u64, size_t> sets_per_vertex;
        for (const auto& use : attribute_uses) {
            ++sets_per_vertex[use.first];
        }
        const size_t common = (std::min)(attribute_sets.size(), CommonAttributeSets);
        for (const auto& [vs, ps] : pairs) {
            if (!programs[vs].lowered || !programs[ps].lowered) {
                continue;
            }
            const auto it = sets_per_vertex.find(programs[vs].hash);
            phase2_estimate += (it != sets_per_vertex.end() ? it->second : common) * templates.size();
        }
        // The count of the last translation, when there is one, is exact for the same archives.
        if (std::ifstream hint{pipeline_cache_filename.parent_path() /
                               "lichii_precompile_phase2.count"}) {
            size_t last = 0;
            if (hint >> last && last != 0 && last <= phase2_estimate) {
                phase2_estimate = last;
            }
        }
    }
    std::atomic<size_t> pair_total{total + phase2_estimate};
    total += phase2_estimate;
    std::vector<GraphicsPipelineCacheKey> prebuild_keys;
    callback(VideoCore::LoadCallbackStage::Translate, 0, total);
    const auto queue_pair = [&](const Program& vertex, const Program& pixel,
                                const FixedPipelineState& state) {
        GraphicsPipelineCacheKey key{};
        key.unique_hashes = {0, vertex.hash, 0, 0, 0, pixel.hash};
        key.state = state;
        ++queued;
        prebuild_keys.push_back(key);
        workers.QueueWork([this, key, &vertex, &pixel, &translated, &failed, &done, &pair_total,
                           &progress_mutex, &callback] {
            SCOPE_EXIT {
                const size_t now = done.fetch_add(1) + 1;
                const size_t all = pair_total.load(std::memory_order::relaxed);
                if (now % 64 == 0 || now == all) {
                    std::scoped_lock lock{progress_mutex};
                    callback(VideoCore::LoadCallbackStage::Translate, now, all);
                }
            };
            VideoCommon::FileEnvironment envs[2];
            const auto bound = [](const Program& p, u32 fallback) {
                return p.lowered->texture_bound != ~0U ? p.lowered->texture_bound : fallback;
            };
            Shader::ProgramHeader vertex_header{};
            Shader::ProgramHeader pixel_header{};
            std::memcpy(&vertex_header, vertex.bytes.data(), sizeof(vertex_header));
            std::memcpy(&pixel_header, pixel.bytes.data(), sizeof(pixel_header));
            envs[0].InitializeFromArchive(vertex.lowered->code, vertex_header,
                                          Shader::Stage::VertexB, bound(vertex, 0), vertex.bank1);
            envs[1].InitializeFromArchive(pixel.lowered->code, pixel_header,
                                          Shader::Stage::Fragment, bound(pixel, 2), pixel.bank1);
            Shader::Environment* env_ptrs[2]{&envs[0], &envs[1]};
            ShaderPools pools;
            t_translate_only = true;
            (void)CreateGraphicsPipeline(pools, key, std::span<Shader::Environment* const>{env_ptrs},
                                         nullptr, false, nullptr, nullptr, false, nullptr, EnvKind::File);
            t_translate_only = false;
            std::scoped_lock lock{reusable_stages_mutex};
            const auto it = reusable_stages.find(HashPrograms(key.unique_hashes));
            (it != reusable_stages.end() ? translated : failed).fetch_add(1);
        });
    };
    for (const auto& [vs, ps] : pairs) {
        for (const FixedPipelineState& state : templates) {
            if (limit != 0 && queued >= limit) {
                break;
            }
            const Program& vertex = programs[vs];
            const Program& pixel = programs[ps];
            if (vertex.lowered && pixel.lowered) {
                queue_pair(vertex, pixel, state);
            }
        }
    }
    // Mesh + pixel, once per learned mesh state: the program key is the hash of the lowered code
    // with the state's hash folded in, as ShaderCache::RefreshStages does.
    size_t mesh_queued = 0;
    for (const auto& [ts, ms, ps] : mesh_pairs) {
        const Program& mesh = programs[ms];
        const Program& pixel = programs[ps];
        const Program* const task = ts != SIZE_MAX ? &programs[ts] : nullptr;
        if (!mesh.lowered || !pixel.lowered || (task && !task->lowered)) {
            continue;
        }
        for (const auto& mesh_state : mesh_states) {
            const u64 mesh_hash =
                Common::CityHash64(reinterpret_cast<const char*>(mesh_state.data()),
                                   sizeof(mesh_state)) |
                1;
            for (const FixedPipelineState& state : mesh_templates) {
                if (limit != 0 && queued >= limit) {
                    break;
                }
                // With a task program: task in the vertex B slot, mesh in the tessellation slot.
                GraphicsPipelineCacheKey key{};
                key.unique_hashes = task ? std::array<u64, Maxwell::MaxShaderProgram>{
                                               0, task->hash ^ mesh_hash, 0,
                                               mesh.hash ^ mesh_hash, 0, pixel.hash}
                                         : std::array<u64, Maxwell::MaxShaderProgram>{
                                               0, mesh.hash ^ mesh_hash, 0, 0, 0, pixel.hash};
                key.state = state;
                ++queued;
                prebuild_keys.push_back(key);
                ++mesh_queued;
                workers.QueueWork([this, key, task, &mesh, &pixel, mesh_state, &translated,
                                   &failed, &done, total, &progress_mutex, &callback] {
                    SCOPE_EXIT {
                        const size_t now = done.fetch_add(1) + 1;
                        if (now % 64 == 0 || now == total) {
                            std::scoped_lock lock{progress_mutex};
                            callback(VideoCore::LoadCallbackStage::Translate, now, total);
                        }
                    };
                    VideoCommon::FileEnvironment envs[3];
                    Shader::ProgramHeader mesh_header{};
                    Shader::ProgramHeader pixel_header{};
                    Shader::ProgramHeader passthrough{};
                    std::memcpy(&mesh_header, mesh.bytes.data(), sizeof(mesh_header));
                    std::memcpy(&pixel_header, pixel.bytes.data(), sizeof(pixel_header));
                    std::memcpy(&passthrough, mesh_state.data() + 3, sizeof(passthrough));
                    const auto bound = [](const Program& p, u32 fallback) {
                        return p.lowered->texture_bound != ~0U ? p.lowered->texture_bound
                                                               : fallback;
                    };
                    const std::array<u32, 3> registers{mesh_state[0], mesh_state[1],
                                                       mesh_state[2]};
                    size_t count = 0;
                    if (task) {
                        Shader::ProgramHeader task_header{};
                        std::memcpy(&task_header, task->bytes.data(), sizeof(task_header));
                        envs[count].InitializeFromArchive(task->lowered->code, task_header,
                                                          Shader::Stage::Task, bound(*task, 0),
                                                          task->bank1);
                        // Only mesh programs read the pass-through header.
                        envs[count].SetMeshState(registers, Shader::ProgramHeader{});
                        ++count;
                    }
                    envs[count].InitializeFromArchive(mesh.lowered->code, mesh_header,
                                                      Shader::Stage::Mesh, bound(mesh, 0),
                                                      mesh.bank1);
                    envs[count].SetMeshState(registers, passthrough);
                    ++count;
                    envs[count].InitializeFromArchive(pixel.lowered->code, pixel_header,
                                                      Shader::Stage::Fragment, bound(pixel, 2),
                                                      pixel.bank1);
                    ++count;
                    Shader::Environment* env_ptrs[3]{&envs[0], &envs[1], &envs[2]};
                    ShaderPools pools;
                    t_translate_only = true;
                    (void)CreateGraphicsPipeline(
                        pools, key, std::span<Shader::Environment* const>{env_ptrs, count},
                        nullptr, false, nullptr, nullptr, false, nullptr, EnvKind::File);
                    t_translate_only = false;
                    std::scoped_lock lock{reusable_stages_mutex};
                    const auto it = reusable_stages.find(HashPrograms(key.unique_hashes));
                    (it != reusable_stages.end() ? translated : failed).fetch_add(1);
                });
            }
        }
    }
    // Compute programs: translated and built here (a compute pipeline has no state besides its
    // key), kept with the answers their translation assumed for the check on first use.
    std::mutex compute_mutex;
    std::atomic<size_t> compute_built{};
    std::atomic<size_t> compute_failed{};
    size_t compute_known = 0;
    size_t compute_skipped = 0;
    for (const size_t cs : compute_programs) {
        const Program& program = programs[cs];
        if (!program.lowered) {
            continue;
        }
        const ComputePipelineCacheKey key{
            .unique_hash = program.hash,
            .shared_memory_size = program.shared_memory,
            .workgroup_size = program.workgroup,
        };
        precompiled_compute_programs.emplace(program.hash, key);
        const auto& wg = program.workgroup;
        if (wg[0] == 0 || wg[1] == 0 || wg[2] == 0 || wg[0] > 1024 || wg[1] > 1024 ||
            wg[2] > 64 || u64{wg[0]} * wg[1] * wg[2] > 1024 ||
            program.shared_memory > device.GetMaxComputeSharedMemorySize()) {
            // Not launch values: this control block is laid out differently.
            static u32 logged{};
            if (logged++ < 10) {
                LOG_WARNING(Render_Vulkan,
                            "LICHII_PRECOMPILE compute {:016x}: unexpected workgroup {}x{}x{} / "
                            "shared {}, skipped",
                            program.hash, wg[0], wg[1], wg[2], program.shared_memory);
            }
            ++compute_skipped;
            continue;
        }
        if (compute_cache.find(key) != compute_cache.end()) {
            // Already built from the disk cache.
            ++compute_known;
            continue;
        }
        workers.QueueWork([this, key, &program, &compute_mutex, &stored_compute, &compute_built, &compute_failed,
                           &done, total, &progress_mutex, &callback] {
            SCOPE_EXIT {
                const size_t now = done.fetch_add(1) + 1;
                if (now % 64 == 0 || now == total) {
                    std::scoped_lock lock{progress_mutex};
                    callback(VideoCore::LoadCallbackStage::Translate, now, total);
                }
            };
            // Local memory the dispatch reserves is not in the archive: generous here, and the
            // check on first use rebuilds the pipeline if the game reserves more.
            constexpr u32 local_memory = 0x2000;
            try {
                VideoCommon::FileEnvironment env;
                env.InitializeFromArchive(program.lowered->code, Shader::ProgramHeader{},
                                          Shader::Stage::Compute,
                                          program.lowered->texture_bound != ~0U
                                              ? program.lowered->texture_bound
                                              : 2,
                                          program.bank1);
                env.SetComputeState(program.workgroup, program.shared_memory, local_memory);
                ShaderPools pools;
                std::pair<std::vector<u32>, Shader::Info> captured;
                t_capture_compute = &captured;
                SCOPE_EXIT {
                    t_capture_compute = nullptr;
                };
                auto entry = CreateComputePipeline(pools, key, env, nullptr, false);
                if (!entry.pipeline) {
                    compute_failed.fetch_add(1);
                    return;
                }
                compute_built.fetch_add(1);
                std::scoped_lock lock{compute_mutex};
                compute_cache.emplace(key, std::move(entry));
                auto queries = env.RecordQueries();
                stored_compute.push_back({key, local_memory, std::move(captured.first),
                                          std::move(captured.second), queries});
                precompiled_compute.emplace(key,
                                            PrecompiledCompute{std::move(queries), local_memory});
            } catch (const std::exception& exception) {
                compute_failed.fetch_add(1);
                LOG_WARNING(Render_Vulkan, "LICHII_PRECOMPILE compute {:016x} failed: {}",
                            key.unique_hash, exception.what());
            } catch (...) {
                compute_failed.fetch_add(1);
                LOG_WARNING(Render_Vulkan, "LICHII_PRECOMPILE compute {:016x} failed",
                            key.unique_hash);
            }
        });
    }
    workers.WaitForRequests(stop_loading);
    // Second phase: the attribute types each vertex program reads are known now (its translations
    // keep the mask); the learned sets that differ there give its other variants.
    size_t attribute_queued = 0;
    std::unordered_multimap<u64, u64> sets_by_vertex;
    for (const auto& [vertex_hash, set] : attribute_uses) {
        sets_by_vertex.emplace(vertex_hash, set);
    }
    if (!attribute_sets.empty() && !stop_loading.stop_requested()) {
        for (const auto& [vs, ps] : pairs) {
            const Program& vertex = programs[vs];
            const Program& pixel = programs[ps];
            if (!vertex.lowered || !pixel.lowered) {
                continue;
            }
            const std::array<u64, Maxwell::MaxShaderProgram> hashes{0, vertex.hash, 0,
                                                                    0, 0, pixel.hash};
            std::optional<u64> mask;
            {
                std::scoped_lock lock{reusable_stages_mutex};
                if (const auto it = reusable_stages.find(HashPrograms(hashes));
                    it != reusable_stages.end()) {
                    for (const auto& entry : it->second) {
                        if (entry->unique_hashes == hashes) {
                            mask = entry->attribute_mask;
                            break;
                        }
                    }
                }
            }
            if (!mask || *mask == 0) {
                continue;
            }
            std::vector<u64> candidates;
            for (auto [it, end] = sets_by_vertex.equal_range(vertex.hash); it != end; ++it) {
                candidates.push_back(it->second);
            }
            if (candidates.empty()) {
                candidates.assign(attribute_sets.begin(),
                                  attribute_sets.begin() +
                                      (std::min)(attribute_sets.size(), CommonAttributeSets));
            }
            std::vector<u64> variants;
            for (const u64 set : candidates) {
                const u64 read = set & *mask;
                if (read != (AllFloatAttributes & *mask) &&
                    std::ranges::find(variants, read) == variants.end()) {
                    variants.push_back(read);
                }
            }
            for (const u64 read : variants) {
                for (const FixedPipelineState& base : templates) {
                    FixedPipelineState variant = base;
                    variant.attribute_types = (AllFloatAttributes & ~*mask) | read;
                    queue_pair(vertex, pixel, variant);
                    ++attribute_queued;
                }
            }
        }
        // The estimate becomes the real count (the bar may step back a little, never forward).
        pair_total.store(total - phase2_estimate + attribute_queued, std::memory_order::relaxed);
        if (std::ofstream hint{pipeline_cache_filename.parent_path() /
                                   "lichii_precompile_phase2.count",
                               std::ios::trunc}) {
            hint << attribute_queued << std::endl;
        }
        workers.WaitForRequests(stop_loading);
    }
    LOG_INFO(Render_Vulkan,
             "LICHII_PRECOMPILE: {} learned vertex attribute sets ({} program uses): {} attribute "
             "variants translated",
             attribute_sets.size(), attribute_uses.size(), attribute_queued);
    LOG_INFO(Render_Vulkan,
             "LICHII_PRECOMPILE: {} compute programs: {} pipelines built, {} failed, {} skipped, {} already in "
             "the disk cache",
             compute_programs.size(), compute_built.load(), compute_failed.load(), compute_skipped,
             compute_known);
    LOG_INFO(Render_Vulkan, "LICHII_PRECOMPILE: {} mesh pairs x {} learned mesh states: {} queued",
             mesh_pairs.size(), mesh_states.size(), mesh_queued);
    LOG_INFO(Render_Vulkan,
             "LICHII_PRECOMPILE: {} archives, {} variations ({} skipped: task/mesh/geometry), {} "
             "programs ({} lowered), {} pairs queued: {} translated, {} failed, in {:.1f} s",
             files, variations, skipped_variations, programs.size(), lowered, queued,
             translated.load(), failed.load(),
             std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());

    if (!stop_loading.stop_requested()) {
        SavePrecompileStore(store_path, validity, store_templates, stores_before,
                            std::move(stored_compute));
        // The translations just made stay in memory for this session with their SPIR-V
        // compressed; the next boot reads the store instead.
    }
    StartBackgroundPrebuild(std::move(prebuild_keys), prebuild_marker, prebuild_marker_value,
                            stop_loading, callback);
    PrecompileShaderObjects(stop_loading, callback, validity);
}

bool PipelineCache::PrebuildPrecompiledPipelines(std::stop_token stop,
                                                 std::vector<GraphicsPipelineCacheKey> prebuild_keys,
                                                 std::vector<FixedPipelineState> extra_states,
                                                 const std::function<void(size_t, size_t)>& progress) {
    // LICHII_PRECOMPILE_PIPELINES=1: one Vulkan pipeline per distinct translation, built with the
    // template's full state on the workers and destroyed right away. The driver keeps what it
    // compiled in the pipeline cache (and its own disk cache), so the pipeline the game asks for
    // over the same SPIR-V is a cache hit instead of a driver compile.
    const auto build_begin = std::chrono::steady_clock::now();
    std::vector<std::pair<GraphicsPipelineCacheKey, std::shared_ptr<const ReusableStages>>> jobs;
    {
        std::unordered_set<const ReusableStages*> seen;
        std::scoped_lock lock{reusable_stages_mutex};
        for (const auto& key : prebuild_keys) {
            const auto it = reusable_stages.find(HashPrograms(key.unique_hashes));
            if (it == reusable_stages.end()) {
                continue;
            }
            const FixedPipelineState state = ReuseState(key.state);
            for (const auto& entry : it->second) {
                if (entry->unique_hashes != key.unique_hashes) {
                    continue;
                }
                FixedPipelineState masked = state;
                if (masked.dynamic_vertex_input) {
                    masked.attribute_types &= entry->attribute_mask;
                }
                if (SameReuseState(masked, entry->state)) {
                    if (seen.insert(entry.get()).second) {
                        jobs.emplace_back(key, entry);
                    }
                    break;
                }
            }
        }
    }
    prebuild_keys.clear();
    prebuild_keys.shrink_to_fit();
    // Then the same translations with the game's other common draw states (blending, attachments
    // and the like, learned from the disk cache), so the pipelines it builds with them are driver
    // cache hits too. Every translation gets its first pipeline before any of these.
    std::vector<std::pair<u32, u8>> extras;
    {
        std::vector<FixedPipelineState> extra_reuse;
        extra_reuse.reserve(extra_states.size());
        for (const auto& state : extra_states) {
            extra_reuse.push_back(ReuseState(state));
        }
        for (size_t base = 0; base < jobs.size(); ++base) {
            const auto& [key, entry] = jobs[base];
            for (size_t index = 0; index < extra_states.size() && index < 255; ++index) {
                // The draw state varies; the vertex attribute types stay the translation's.
                FixedPipelineState masked = extra_reuse[index];
                masked.attribute_types = entry->state.attribute_types;
                if (!SameReuseState(masked, entry->state)) {
                    continue;
                }
                FixedPipelineState state = extra_states[index];
                state.attribute_types = key.state.attribute_types;
                if (state == key.state) {
                    continue;
                }
                extras.emplace_back(static_cast<u32>(base), static_cast<u8>(index));
            }
        }
    }
    const size_t build_total = jobs.size() + extras.size();
    LOG_INFO(Render_Vulkan,
             "LICHII_PRECOMPILE_PIPELINES: {} translations, then {} more pipelines with {} common "
             "draw states",
             jobs.size(), extras.size(), extra_states.size());
    // Shared with the queued jobs: after a stop the ones still queued return at once, and they
    // may outlive this function.
    struct Counters {
        std::atomic<size_t> built{}, build_failed{};
        std::atomic<u32> logged_failures{};
    };
    const auto counters = std::make_shared<Counters>();
    auto& built = counters->built;
    auto& build_failed = counters->build_failed;
    // In chunks, so a stop (the game closing) ends the work soon.
    constexpr size_t Chunk = 256;
    for (size_t chunk = 0; chunk < build_total && !stop.stop_requested(); chunk += Chunk) {
      for (size_t index = chunk; index < (std::min)(build_total, chunk + Chunk); ++index) {
        GraphicsPipelineCacheKey job_key;
        std::shared_ptr<const ReusableStages> job_stages;
        if (index < jobs.size()) {
            job_key = jobs[index].first;
            job_stages = jobs[index].second;
        } else {
            const auto& [base, state_index] = extras[index - jobs.size()];
            job_key = jobs[base].first;
            job_key.state = extra_states[state_index];
            job_key.state.attribute_types = jobs[base].first.state.attribute_types;
            job_stages = jobs[base].second;
        }
        prebuild_workers->QueueWork([this, key = job_key, stages = std::move(job_stages), counters,
                                     stop] {
            if (stop.stop_requested()) {
                return;
            }
            auto& built = counters->built;
            auto& build_failed = counters->build_failed;
            auto& logged_failures = counters->logged_failures;
            try {
                std::array<vk::ShaderModule, Maxwell::MaxShaderStage> modules;
                std::array<const Shader::Info*, Maxwell::MaxShaderStage> infos{};
                for (size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
                    if (!stages->infos[stage]) {
                        continue;
                    }
                    const std::vector<u32> spirv = ReusableSpirv(*stages, stage);
                    if (spirv.empty()) {
                        build_failed.fetch_add(1);
                        return;
                    }
                    modules[stage] = BuildShader(device, spirv);
                    infos[stage] = &*stages->infos[stage];
                }
                GraphicsPipeline pipeline{scheduler, buffer_cache, texture_cache,
                                          vulkan_pipeline_cache, nullptr, device, descriptor_pool,
                                          guest_descriptor_queue, descriptor_buffer_ring, nullptr,
                                          nullptr, render_pass_cache, key, std::move(modules),
                                          infos,
                                          GraphicsPipeline::EmitStagesFunc{}};
                built.fetch_add(1);
            } catch (const std::exception& exception) {
                build_failed.fetch_add(1);
                if (logged_failures.fetch_add(1) < 20) {
                    LOG_WARNING(Render_Vulkan, "LICHII_PRECOMPILE_PIPELINES: {:016x} failed: {}",
                                key.Hash(), exception.what());
                }
            }
        });
      }
      prebuild_workers->WaitForRequests(stop);
      if (progress) {
          progress((std::min)(build_total, chunk + Chunk), build_total);
      }
    }
    u64 cache_size = 0;
    if (use_vulkan_pipeline_cache) {
        size_t size = 0;
        vulkan_pipeline_cache.Read(&size, nullptr);
        cache_size = size;
    }
    LOG_INFO(Render_Vulkan,
             "LICHII_PRECOMPILE_PIPELINES: {} distinct translations: {} driver pipelines built, {} "
             "failed, in {:.1f} s; driver pipeline cache {} MiB",
             build_total, built.load(), build_failed.load(),
             std::chrono::duration<double>(std::chrono::steady_clock::now() - build_begin).count(),
             cache_size >> 20);
    return !stop.stop_requested();
}

void PipelineCache::RunPipelineExperiment(const std::vector<GraphicsPipelineCacheKey>& keys) {
    // LICHII_PIPELINE_EXPERIMENT=<offset>: how long the driver takes to build a stored translation's
    // pipeline as stored, then with only the colour formats, only the depth format, only the
    // blending or only the dynamic-state block of the game's own pipelines changed. Tells which
    // changes cost a full driver compile (and what a pipeline library could share).
    const char* env = std::getenv("LICHII_PIPELINE_EXPERIMENT");
    const size_t offset = env ? static_cast<size_t>(std::strtoull(env, nullptr, 10)) : 0;
    constexpr size_t Samples = 120;
    // The game's own render targets, blending and dynamic state, from the disk cache.
    std::map<std::vector<u8>, u32> targets;
    std::map<std::vector<u8>, u32> blends;
    std::map<std::vector<u8>, u32> dynamics;
    const auto bytes = [](const auto& value) {
        const auto* data = reinterpret_cast<const u8*>(&value);
        return std::vector<u8>(data, data + sizeof(value));
    };
    for (const auto& [key, pipeline] : graphics_cache) {
        std::vector<u8> target = bytes(key.state.color_formats);
        target.push_back(static_cast<u8>(key.state.depth_format.Value()));
        ++targets[target];
        ++blends[bytes(key.state.attachments)];
        ++dynamics[bytes(key.state.dynamic_state)];
    }
    const auto ranked = [](const std::map<std::vector<u8>, u32>& counts) {
        std::vector<std::pair<u32, std::vector<u8>>> list;
        for (const auto& [value, count] : counts) {
            list.emplace_back(count, value);
        }
        std::ranges::sort(list, [](const auto& a, const auto& b) { return a.first > b.first; });
        return list;
    };
    const auto target_list = ranked(targets);
    const auto blend_list = ranked(blends);
    const auto dynamic_list = ranked(dynamics);
    LOG_INFO(Render_Vulkan,
             "LICHII_PIPELINE_EXPERIMENT: {} cached pipelines: {} render targets, {} blendings, {} "
             "dynamic-state blocks",
             graphics_cache.size(), target_list.size(), blend_list.size(), dynamic_list.size());

    std::vector<std::pair<GraphicsPipelineCacheKey, std::shared_ptr<const ReusableStages>>> jobs;
    {
        std::scoped_lock lock{reusable_stages_mutex};
        for (size_t index = offset; index < keys.size() && jobs.size() < Samples; index += 97) {
            const auto& key = keys[index];
            const auto it = reusable_stages.find(HashPrograms(key.unique_hashes));
            if (it == reusable_stages.end()) {
                continue;
            }
            FixedPipelineState state = ReuseState(key.state);
            for (const auto& entry : it->second) {
                if (entry->unique_hashes != key.unique_hashes) {
                    continue;
                }
                FixedPipelineState masked = state;
                if (masked.dynamic_vertex_input) {
                    masked.attribute_types &= entry->attribute_mask;
                }
                if (SameReuseState(masked, entry->state)) {
                    jobs.emplace_back(key, entry);
                    break;
                }
            }
        }
    }
    const auto build = [&](const GraphicsPipelineCacheKey& key,
                           const std::shared_ptr<const ReusableStages>& stages) -> double {
        const auto begin = std::chrono::steady_clock::now();
        try {
            std::array<vk::ShaderModule, Maxwell::MaxShaderStage> modules;
            std::array<const Shader::Info*, Maxwell::MaxShaderStage> infos{};
            for (size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
                if (!stages->infos[stage]) {
                    continue;
                }
                const std::vector<u32> spirv = ReusableSpirv(*stages, stage);
                if (spirv.empty()) {
                    return -1.0;
                }
                modules[stage] = BuildShader(device, spirv);
                infos[stage] = &*stages->infos[stage];
            }
            GraphicsPipeline pipeline{scheduler, buffer_cache, texture_cache, vulkan_pipeline_cache,
                                      nullptr, device, descriptor_pool, guest_descriptor_queue,
                                      descriptor_buffer_ring, nullptr, nullptr, render_pass_cache,
                                      key, std::move(modules), infos,
                                      GraphicsPipeline::EmitStagesFunc{}};
        } catch (const std::exception&) {
            return -1.0;
        }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
            .count();
    };
    constexpr std::array<const char*, 6> Names{"as stored (driver cache)", "colour formats",
                                               "depth format", "blending", "dynamic-state block",
                                               "as stored again"};
    std::array<std::vector<double>, Names.size()> times;
    for (const auto& [key, stages] : jobs) {
        std::array<std::optional<GraphicsPipelineCacheKey>, Names.size()> variants;
        variants[0] = key;
        variants[5] = key;
        std::vector<u8> own_target = bytes(key.state.color_formats);
        own_target.push_back(static_cast<u8>(key.state.depth_format.Value()));
        for (const auto& [count, target] : target_list) {
            if (target == own_target) {
                continue;
            }
            const size_t formats = sizeof(key.state.color_formats);
            if (!variants[1] && std::memcmp(target.data(), own_target.data(), formats) != 0) {
                GraphicsPipelineCacheKey variant = key;
                std::memcpy(&variant.state.color_formats, target.data(), formats);
                variants[1] = variant;
            }
            if (!variants[2] && target[formats] != own_target[formats]) {
                GraphicsPipelineCacheKey variant = key;
                variant.state.depth_format.Assign(target[formats]);
                variants[2] = variant;
            }
        }
        for (const auto& [count, blend] : blend_list) {
            if (blend != bytes(key.state.attachments)) {
                GraphicsPipelineCacheKey variant = key;
                std::memcpy(&variant.state.attachments, blend.data(), blend.size());
                variants[3] = variant;
                break;
            }
        }
        for (const auto& [count, dynamic] : dynamic_list) {
            if (dynamic != bytes(key.state.dynamic_state)) {
                GraphicsPipelineCacheKey variant = key;
                std::memcpy(&variant.state.dynamic_state, dynamic.data(), dynamic.size());
                variants[4] = variant;
                break;
            }
        }
        for (size_t v = 0; v < Names.size(); ++v) {
            if (variants[v]) {
                const double ms = build(*variants[v], stages);
                if (ms >= 0.0) {
                    times[v].push_back(ms);
                }
            }
        }
    }
    for (size_t v = 0; v < Names.size(); ++v) {
        auto& list = times[v];
        if (list.empty()) {
            LOG_INFO(Render_Vulkan, "LICHII_PIPELINE_EXPERIMENT: {}: no samples", Names[v]);
            continue;
        }
        std::ranges::sort(list);
        double total = 0;
        size_t slow = 0;
        for (const double ms : list) {
            total += ms;
            slow += ms > 10.0 ? 1 : 0;
        }
        LOG_INFO(Render_Vulkan,
                 "LICHII_PIPELINE_EXPERIMENT: {:26}: {:3} builds, median {:6.2f} ms, mean {:6.2f} "
                 "ms, p90 {:6.2f} ms, {} over 10 ms",
                 Names[v], list.size(), list[list.size() / 2], total / list.size(),
                 list[list.size() * 9 / 10], slow);
    }
}

void PipelineCache::PrecompileShaderObjects(std::stop_token stop_loading,
                                            const VideoCore::DiskResourceLoadCallback& callback,
                                            u64 store_validity) {
    if (!UseShaderObjects(device)) {
        return;
    }
    ShaderObjectCache& cache = ShaderObjectCache::Instance();
    if (cache.Path().empty()) {
        return;
    }
    // Done once per store and driver: this file names them.
    const std::filesystem::path marker{cache.Path().string() + ".precompiled"};
    const std::string marker_value =
        fmt::format("{:016x} {:016x}", store_validity, cache.Identity());
    if (std::ifstream in{marker}; in) {
        std::string stored;
        std::getline(in, stored);
        if (stored == marker_value) {
            LOG_INFO(Render_Vulkan, "LICHII_SHADER_OBJECTS: translations already precompiled");
            return;
        }
    }
    std::vector<std::shared_ptr<const ReusableStages>> entries;
    {
        std::scoped_lock lock{reusable_stages_mutex};
        for (const auto& [hash, list] : reusable_stages) {
            entries.insert(entries.end(), list.begin(), list.end());
        }
    }
    const auto begin = std::chrono::steady_clock::now();
    static constexpr std::array<VkShaderStageFlagBits, Maxwell::MaxShaderStage> StageBits{
        VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, VK_SHADER_STAGE_GEOMETRY_BIT,
        VK_SHADER_STAGE_FRAGMENT_BIT};
    using Shader::Backend::SPIRV::RenderAreaLayout;
    using Shader::Backend::SPIRV::RescalingLayout;
    const VkPushConstantRange push_range{
        .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
        .offset = 0,
        .size = static_cast<u32>(sizeof(RescalingLayout) + sizeof(RenderAreaLayout)),
    };
    // One set layout per distinct layout, kept while precompiling.
    std::mutex layouts_mutex;
    std::unordered_map<u64, vk::DescriptorSetLayout> layouts;
    std::atomic<size_t> done{}, compiled{};
    const unsigned cores = std::thread::hardware_concurrency();
    Common::ThreadWorker precompile_workers(cores > 1 ? cores - 1 : 1, "VkShaderObjects");
    const size_t total = entries.size();
    if (callback) {
        callback(VideoCore::LoadCallbackStage::Pipelines, 0, total);
    }
    constexpr size_t Chunk = 64;
    for (size_t first = 0; first < total; first += Chunk) {
        precompile_workers.QueueWork([&, first] {
            for (size_t index = first; index < (std::min)(total, first + Chunk); ++index) {
                SCOPE_EXIT {
                    done.fetch_add(1, std::memory_order_relaxed);
                };
                if (stop_loading.stop_requested()) {
                    continue;
                }
                const ReusableStages& stages = *entries[index];
                std::array<Shader::Info, Maxwell::MaxShaderStage> infos{};
                bool skip = false;
                for (size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
                    if (stages.infos[stage]) {
                        infos[stage] = *stages.infos[stage];
                        skip |= infos[stage].is_mesh || infos[stage].is_task;
                    }
                }
                if (skip || !stages.infos[0]) {
                    continue;
                }
                // As GraphicsPipeline builds its layout (MakeBuilder).
                DescriptorLayoutBuilder builder{device};
                for (size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
                    builder.Add(infos[stage], StageBits[stage]);
                }
                const bool push_descriptor = builder.CanUsePushDescriptor();
                if (builder.CanUseDescriptorBuffer() && descriptor_buffer_ring.IsValid()) {
                    continue; // such a pipeline stays a pipeline
                }
                const u64 layout_hash = builder.LayoutHash(push_descriptor);
                VkDescriptorSetLayout set_layout{};
                {
                    std::scoped_lock lock{layouts_mutex};
                    auto [it, inserted] = layouts.try_emplace(layout_hash);
                    if (inserted) {
                        it->second = builder.CreateDescriptorSetLayout(push_descriptor);
                    }
                    set_layout = it->second ? *it->second : VK_NULL_HANDLE;
                }
                std::array<std::vector<u32>, Maxwell::MaxShaderStage> code;
                for (size_t stage = 0; stage < Maxwell::MaxShaderStage && !skip; ++stage) {
                    if (stages.infos[stage]) {
                        code[stage] = ReusableSpirv(stages, stage);
                        skip |= code[stage].empty();
                    }
                }
                if (skip) {
                    continue;
                }
                for (size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
                    if (code[stage].empty()) {
                        continue;
                    }
                    VkShaderStageFlags next_stage = 0;
                    for (size_t next = stage + 1; next < Maxwell::MaxShaderStage; ++next) {
                        if (!code[next].empty()) {
                            next_stage = StageBits[next];
                            break;
                        }
                    }
                    const VkShaderCreateInfoEXT ci{
                        .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
                        .pNext = nullptr,
                        .flags = 0,
                        .stage = StageBits[stage],
                        .nextStage = next_stage,
                        .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
                        .codeSize = code[stage].size() * sizeof(u32),
                        .pCode = code[stage].data(),
                        .pName = "main",
                        .setLayoutCount = set_layout != VK_NULL_HANDLE ? 1U : 0U,
                        .pSetLayouts = set_layout != VK_NULL_HANDLE ? &set_layout : nullptr,
                        .pushConstantRangeCount = 1,
                        .pPushConstantRanges = &push_range,
                        .pSpecializationInfo = nullptr,
                    };
                    if (cache.Precompile(device, ci, layout_hash)) {
                        compiled.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        });
    }
    // The bar, from this thread while the workers compile.
    size_t shown = 0;
    while (done.load() < total) {
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
        const size_t now = done.load();
        if (callback && now != shown) {
            shown = now;
            callback(VideoCore::LoadCallbackStage::Pipelines, now, total);
        }
    }
    precompile_workers.WaitForRequests();
    const bool complete = !stop_loading.stop_requested();
    if (complete) {
        std::ofstream out{marker, std::ios::trunc};
        out << marker_value << std::endl;
    }
    std::error_code error;
    LOG_INFO(Render_Vulkan,
             "LICHII_SHADER_OBJECTS: {} translations precompiled: {} new shader binaries, {} set "
             "layouts, in {:.1f} s ({} MiB file){}",
             total, compiled.load(), layouts.size(),
             std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count(),
             std::filesystem::file_size(cache.Path(), error) >> 20,
             complete ? "" : " (stopped)");
}

void PipelineCache::RunShaderObjectExperiment(const std::vector<GraphicsPipelineCacheKey>& keys) {
    // LICHII_SHADER_OBJECT_EXPERIMENT=<offset>: VK_EXT_shader_object compiles each stage of a stored
    // translation on its own, with every draw state dynamic. Times that compile, the size of the
    // driver's binary, and creating the shader again from that binary (what a store of binaries
    // on the SSD would cost per shader).
    if (!device.IsExtShaderObjectSupported()) {
        LOG_INFO(Render_Vulkan, "LICHII_SHADER_OBJECT_EXPERIMENT: VK_EXT_shader_object unavailable");
        return;
    }
    const char* env = std::getenv("LICHII_SHADER_OBJECT_EXPERIMENT");
    const size_t offset = env ? static_cast<size_t>(std::strtoull(env, nullptr, 10)) : 0;
    constexpr size_t Samples = 120;
    std::vector<std::shared_ptr<const ReusableStages>> jobs;
    {
        std::scoped_lock lock{reusable_stages_mutex};
        std::unordered_set<const ReusableStages*> seen;
        for (size_t index = offset; index < keys.size() && jobs.size() < Samples; index += 89) {
            const auto it = reusable_stages.find(HashPrograms(keys[index].unique_hashes));
            if (it == reusable_stages.end() || it->second.empty()) {
                continue;
            }
            const auto& entry = it->second.front();
            bool mesh = false;
            for (const auto& info : entry->infos) {
                mesh |= info && (info->is_mesh || info->is_task);
            }
            if (!mesh && seen.insert(entry.get()).second) {
                jobs.push_back(entry);
            }
        }
    }
    static constexpr std::array<VkShaderStageFlagBits, Maxwell::MaxShaderStage> StageBits{
        VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, VK_SHADER_STAGE_GEOMETRY_BIT,
        VK_SHADER_STAGE_FRAGMENT_BIT};
    static constexpr std::array<VkShaderStageFlags, Maxwell::MaxShaderStage> NextStages{
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_GEOMETRY_BIT |
            VK_SHADER_STAGE_FRAGMENT_BIT,
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
        VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_FRAGMENT_BIT,
        0};
    using Shader::Backend::SPIRV::RenderAreaLayout;
    using Shader::Backend::SPIRV::RescalingLayout;
    const VkPushConstantRange push_range{
        .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
        .offset = 0,
        .size = static_cast<u32>(sizeof(RescalingLayout) + sizeof(RenderAreaLayout)),
    };
    const vk::Device& logical = device.GetLogical();
    std::vector<double> compile_vertex, compile_fragment, reload, sizes;
    size_t failures = 0;
    for (const auto& stages : jobs) {
        std::array<Shader::Info, Maxwell::MaxShaderStage> infos{};
        for (size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
            if (stages->infos[stage]) {
                infos[stage] = *stages->infos[stage];
            }
        }
        DescriptorLayoutBuilder builder{device};
        for (size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
            builder.Add(infos[stage], StageBits[stage]);
        }
        const vk::DescriptorSetLayout set_layout =
            builder.CreateDescriptorSetLayout(builder.CanUsePushDescriptor());
        const VkDescriptorSetLayout set_layout_handle = *set_layout;
        for (size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
            if (!stages->infos[stage]) {
                continue;
            }
            const std::vector<u32> spirv = ReusableSpirv(*stages, stage);
            if (spirv.empty()) {
                continue;
            }
            VkShaderCreateInfoEXT ci{
                .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
                .pNext = nullptr,
                .flags = 0,
                .stage = StageBits[stage],
                .nextStage = NextStages[stage],
                .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
                .codeSize = spirv.size() * sizeof(u32),
                .pCode = spirv.data(),
                .pName = "main",
                .setLayoutCount = set_layout_handle != VK_NULL_HANDLE ? 1U : 0U,
                .pSetLayouts = set_layout_handle != VK_NULL_HANDLE ? &set_layout_handle : nullptr,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &push_range,
                .pSpecializationInfo = nullptr,
            };
            VkShaderEXT shader{};
            auto begin = std::chrono::steady_clock::now();
            if (logical.CreateShaderEXT(ci, &shader) != VK_SUCCESS) {
                ++failures;
                continue;
            }
            const double compile_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                    .count();
            (stage == 0 ? compile_vertex : compile_fragment).push_back(compile_ms);
            size_t size = 0;
            std::vector<u8> binary;
            if (logical.GetShaderBinaryDataEXT(shader, &size, nullptr) == VK_SUCCESS && size != 0) {
                binary.resize(size);
                if (logical.GetShaderBinaryDataEXT(shader, &size, binary.data()) != VK_SUCCESS) {
                    binary.clear();
                }
            }
            logical.DestroyShaderEXT(shader);
            if (binary.empty()) {
                ++failures;
                continue;
            }
            sizes.push_back(static_cast<double>(binary.size()));
            ci.codeType = VK_SHADER_CODE_TYPE_BINARY_EXT;
            ci.codeSize = binary.size();
            ci.pCode = binary.data();
            begin = std::chrono::steady_clock::now();
            if (logical.CreateShaderEXT(ci, &shader) != VK_SUCCESS) {
                ++failures;
                continue;
            }
            reload.push_back(
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                    .count());
            logical.DestroyShaderEXT(shader);
        }
    }
    const auto report = [](const char* name, std::vector<double>& list, const char* unit) {
        if (list.empty()) {
            LOG_INFO(Render_Vulkan, "LICHII_SHADER_OBJECT_EXPERIMENT: {}: no samples", name);
            return;
        }
        std::ranges::sort(list);
        double total = 0;
        for (const double value : list) {
            total += value;
        }
        LOG_INFO(Render_Vulkan,
                 "LICHII_SHADER_OBJECT_EXPERIMENT: {:28}: {:4} samples, median {:9.2f} {}, mean "
                 "{:9.2f} {}, p90 {:9.2f} {}",
                 name, list.size(), list[list.size() / 2], unit, total / list.size(), unit,
                 list[list.size() * 9 / 10], unit);
    };
    LOG_INFO(Render_Vulkan, "LICHII_SHADER_OBJECT_EXPERIMENT: {} translations, {} failures",
             jobs.size(), failures);
    report("vertex compile (SPIR-V)", compile_vertex, "ms");
    report("other stages compile (SPIR-V)", compile_fragment, "ms");
    report("create from binary", reload, "ms");
    report("binary size", sizes, "B");
}

void PipelineCache::StartBackgroundPrebuild(std::vector<GraphicsPipelineCacheKey> prebuild_keys,
                                            std::filesystem::path marker,
                                            std::string marker_value,
                                            std::stop_token stop_loading,
                                            const VideoCore::DiskResourceLoadCallback& callback) {
    if (std::getenv("LICHII_PIPELINE_EXPERIMENT") != nullptr) {
        RunPipelineExperiment(prebuild_keys);
    }
    if (std::getenv("LICHII_SHADER_OBJECT_EXPERIMENT") != nullptr) {
        RunShaderObjectExperiment(prebuild_keys);
    }
    const char* pipelines_env = std::getenv("LICHII_PRECOMPILE_PIPELINES");
    if (pipelines_env == nullptr || pipelines_env[0] == '\0' || pipelines_env[0] == '0') {
        return;
    }
    // The game's common draw states, from the pipelines in the disk cache: grouped by what reaches
    // the shaders, the three most used variations of each group (twelve at most), with the vertex
    // input left out (dynamic). Their set is part of the marker: newly learned ones build again.
    std::vector<FixedPipelineState> extra_states;
    // LICHII_PRECOMPILE_EXTRA_STATES=1: also these (many more pipelines the first time).
    if (std::getenv("LICHII_PRECOMPILE_EXTRA_STATES") != nullptr) {
        struct Variation {
            u32 uses{};
            FixedPipelineState state{};
        };
        std::unordered_map<u64, std::unordered_map<u64, Variation>> groups;
        for (const auto& [key, pipeline] : graphics_cache) {
            FixedPipelineState state = key.state;
            state.attribute_types = 0;
            state.attributes = {};
            state.binding_divisors = {};
            state.vertex_strides = {};
            state.enabled_divisors = 0;
            FixedPipelineState reuse = ReuseState(state);
            const u64 group = Common::CityHash64(reinterpret_cast<const char*>(&reuse), reuse.Size());
            const u64 variation =
                Common::CityHash64(reinterpret_cast<const char*>(&state), state.Size());
            auto& entry = groups[group][variation];
            if (entry.uses++ == 0) {
                entry.state = state;
            }
        }
        std::vector<std::pair<u32, std::vector<Variation>>> ranked;
        for (auto& [group, variations] : groups) {
            std::vector<Variation> list;
            u32 total = 0;
            for (auto& [hash, variation] : variations) {
                total += variation.uses;
                list.push_back(variation);
            }
            std::ranges::sort(list, [](const Variation& a, const Variation& b) { return a.uses > b.uses; });
            if (list.size() > 3) {
                list.resize(3);
            }
            ranked.emplace_back(total, std::move(list));
        }
        std::ranges::sort(ranked, [](const auto& a, const auto& b) { return a.first > b.first; });
        u64 states_hash = 0;
        for (const auto& [total, list] : ranked) {
            for (const auto& variation : list) {
                if (extra_states.size() < 12) {
                    extra_states.push_back(variation.state);
                    states_hash ^= Common::CityHash64(reinterpret_cast<const char*>(&variation.state),
                                                      variation.state.Size()) +
                                   extra_states.size();
                }
            }
        }
        marker_value += fmt::format(" states {:016x}", states_hash);
    }
    // Built once for these translations and this driver: its cache has them since.
    if (std::ifstream in{marker}; in) {
        std::string stored;
        std::getline(in, stored);
        if (stored == marker_value) {
            LOG_INFO(Render_Vulkan, "LICHII_PRECOMPILE_PIPELINES: already built for this store "
                                    "and driver ({})", marker.filename().string());
            return;
        }
    }
    // The first boot builds them behind the loading screen, on every core but one: one long wait
    // once instead of hitches while playing (and a busy CPU while playing makes some spots
    // flicker). LICHII_PREBUILD_BACKGROUND=1: after boot instead, while the game runs.
    if (std::getenv("LICHII_PREBUILD_BACKGROUND") == nullptr) {
        const unsigned cores = std::thread::hardware_concurrency();
        const size_t threads = cores > 1 ? cores - 1 : 1;
        prebuild_workers.reset(new Common::ThreadWorker(threads, "VkPrebuild"));
        if (callback) {
            callback(VideoCore::LoadCallbackStage::Pipelines, 0, 1);
        }
        const bool complete = PrebuildPrecompiledPipelines(
            stop_loading, std::move(prebuild_keys), std::move(extra_states),
            [&callback](size_t done, size_t total) {
                if (callback) {
                    callback(VideoCore::LoadCallbackStage::Pipelines, done, total);
                }
            });
        prebuild_workers->WaitForRequests({});
        prebuild_workers.reset();
        if (complete) {
            std::ofstream out{marker, std::ios::trunc};
            out << marker_value << '\n';
        }
        return;
    }
    // After boot, on low-priority threads of their own: the game does not wait, and the
    // pipelines it asks for meanwhile keep the main builders to themselves.
    // One thread by default (LICHII_PREBUILD_THREADS): with more, the game loses CPU while it
    // runs and timing-sensitive spots can flicker.
    static const size_t prebuild_threads = [] {
        const char* value = std::getenv("LICHII_PREBUILD_THREADS");
        const int count = value ? std::atoi(value) : 1;
        return static_cast<size_t>(std::clamp(count, 1, 8));
    }();
    prebuild_workers.reset(new Common::ThreadWorker(prebuild_threads, "VkPrebuild", {},
                                                    Common::ThreadPlacement::Background));
    prebuild_thread = std::jthread([this, keys = std::move(prebuild_keys), marker, marker_value,
                                    extra_states = std::move(extra_states)](std::stop_token stop) mutable {
        Common::SetCurrentThreadName("VkPrebuildFeed");
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
        if (PrebuildPrecompiledPipelines(stop, std::move(keys), std::move(extra_states))) {
            std::ofstream out{marker, std::ios::trunc};
            out << marker_value << '\n';
        }
    });
}


namespace {
constexpr u32 PrecompileStoreMagic = 0x3154504C; // "LPT1"

/// Indexes equal values once (SPIR-V modules, serialized shader infos).
template <typename T>
class Deduplicator {
public:
    s32 Add(const T& value, std::span<const u8> bytes) {
        const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(bytes.data()),
                                            bytes.size());
        auto& candidates = index[hash];
        for (const s32 candidate : candidates) {
            if (values[candidate] == value) {
                return candidate;
            }
        }
        values.push_back(value);
        candidates.push_back(static_cast<s32>(values.size() - 1));
        return candidates.back();
    }
    std::vector<T> values;

private:
    std::unordered_map<u64, std::vector<s32>> index;
};
} // Anonymous namespace

std::vector<u32> PipelineCache::ReadStoredSpirv(s32 module) {
    if (module < 0 || !spirv_store || static_cast<size_t>(module) >= spirv_store->modules.size()) {
        return {};
    }
    const SpirvStore::Module& where = spirv_store->modules[module];
    std::vector<u8> compressed(where.compressed_size);
    {
        std::scoped_lock lock{spirv_store->mutex};
        auto& file = spirv_store->file;
        file.clear();
        file.seekg(static_cast<std::streamoff>(where.offset));
        file.read(reinterpret_cast<char*>(compressed.data()),
                  static_cast<std::streamsize>(compressed.size()));
        if (!file) {
            LOG_ERROR(Render_Vulkan, "LICHII_PRECOMPILE: could not read SPIR-V module {}", module);
            return {};
        }
    }
    std::vector<u32> spirv(where.words);
    const std::span<u8> output{reinterpret_cast<u8*>(spirv.data()), spirv.size() * sizeof(u32)};
    if (Common::Compression::DecompressDataZSTD(compressed, output) != output.size()) {
        LOG_ERROR(Render_Vulkan, "LICHII_PRECOMPILE: SPIR-V module {} is damaged", module);
        return {};
    }
    return spirv;
}

std::vector<u32> PipelineCache::ReusableSpirv(const ReusableStages& stages, size_t stage) {
    if (!stages.spirv[stage].empty()) {
        return stages.spirv[stage];
    }
    if (stages.spirv_zstd[stage]) {
        std::vector<u32> spirv(stages.spirv_words[stage]);
        const std::span<u8> output{reinterpret_cast<u8*>(spirv.data()),
                                   spirv.size() * sizeof(u32)};
        if (Common::Compression::DecompressDataZSTD(*stages.spirv_zstd[stage], output) !=
            output.size()) {
            return {};
        }
        return spirv;
    }
    if (stages.spool_spirv[stage] >= 0) {
        const std::vector<u8> frame = ReadSpoolFrame(stages.spool_spirv[stage]);
        std::vector<u32> spirv(stages.spirv_words[stage]);
        const std::span<u8> output{reinterpret_cast<u8*>(spirv.data()),
                                   spirv.size() * sizeof(u32)};
        if (frame.empty() || Common::Compression::DecompressDataZSTD(frame, output) !=
                                 output.size()) {
            return {};
        }
        return spirv;
    }
    return ReadStoredSpirv(stages.stored_spirv[stage]);
}

void PipelineCache::SavePrecompileStore(const std::filesystem::path& path, u64 validity,
                                        const std::vector<FixedPipelineState>& templates,
                                        const std::unordered_set<const ReusableStages*>& stores_before,
                                        std::vector<StoredCompute> compute) try {
    const auto begin = std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<const ReusableStages>> entries;
    {
        std::scoped_lock lock{reusable_stages_mutex};
        for (const auto& [programs_hash, list] : reusable_stages) {
            for (const auto& entry : list) {
                if (!stores_before.contains(entry.get())) {
                    entries.push_back(entry);
                }
            }
        }
    }
    // SPIR-V modules as the zstd frames written to the file: the shared compressed copies made at
    // translation go in as they are (one per distinct module), anything else is compressed here.
    // A frame is either in memory or a module of the first boot's spool file (copied from there
    // while the store is written, one at a time).
    std::vector<std::shared_ptr<const std::vector<u8>>> frames;
    std::vector<u32> frame_words;
    std::vector<s32> frame_spool;
    std::vector<u32> frame_sizes;
    std::unordered_map<const void*, s32> frame_by_copy;
    std::unordered_map<u64, s32> frame_by_hash;
    std::unordered_map<s32, s32> frame_by_spool;
    const auto add_frame = [&](std::shared_ptr<const std::vector<u8>> frame, u32 words) {
        frame_sizes.push_back(static_cast<u32>(frame->size()));
        frames.push_back(std::move(frame));
        frame_words.push_back(words);
        frame_spool.push_back(-1);
        return static_cast<s32>(frames.size() - 1);
    };
    const auto add_blob = [&](const std::vector<u32>& spirv) -> s32 {
        if (spirv.empty()) {
            return -1;
        }
        const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(spirv.data()),
                                            spirv.size() * sizeof(u32)) ^
                         (u64{spirv.size()} << 40);
        if (const auto it = frame_by_hash.find(hash); it != frame_by_hash.end()) {
            return it->second;
        }
        const std::vector<u8> compressed = Common::Compression::CompressDataZSTD(
            reinterpret_cast<const u8*>(spirv.data()), spirv.size() * sizeof(u32), 1);
        const s32 index = add_frame(
            std::make_shared<const std::vector<u8>>(compressed.begin(), compressed.end()),
            static_cast<u32>(spirv.size()));
        frame_by_hash.emplace(hash, index);
        return index;
    };
    const auto add_stage = [&](const ReusableStages& entry, size_t stage) -> s32 {
        if (const s32 spooled = entry.spool_spirv[stage]; spooled >= 0 && spirv_spool) {
            const auto [it, inserted] =
                frame_by_spool.try_emplace(spooled, static_cast<s32>(frames.size()));
            if (inserted) {
                frames.push_back(nullptr);
                frame_words.push_back(spirv_spool->modules[spooled].words);
                frame_spool.push_back(spooled);
                frame_sizes.push_back(spirv_spool->modules[spooled].compressed_size);
            }
            return it->second;
        }
        if (const auto& copy = entry.spirv_zstd[stage]) {
            const auto [it, inserted] =
                frame_by_copy.try_emplace(copy.get(), static_cast<s32>(frames.size()));
            if (inserted) {
                add_frame(copy, entry.spirv_words[stage]);
            }
            return it->second;
        }
        return add_blob(ReusableSpirv(entry, stage));
    };
    Deduplicator<std::vector<u8>> infos;
    const auto add_info = [&](const Shader::Info& info) -> s32 {
        LichiiIO::Writer w;
        LichiiIO::PutInfo(w, info);
        return infos.Add(w.buffer, w.buffer);
    };
    // A packed info goes in as its serialization, without expanding it.
    const auto add_packed_info = [&](const InfoRef& info) -> s32 {
        return infos.Add(info.Bytes(), info.Bytes());
    };
    // Which template each translation was made with: the pipeline LICHII_PRECOMPILE_PIPELINES
    // builds for it after a stored boot.
    const auto template_of = [&](const ReusableStages& entry) -> u8 {
        for (size_t i = 0; i < templates.size() && i < 0xFF; ++i) {
            FixedPipelineState masked = ReuseState(templates[i]);
            if (masked.dynamic_vertex_input) {
                masked.attribute_types &= entry.attribute_mask;
            }
            if (SameReuseState(masked, entry.state)) {
                return static_cast<u8>(i);
            }
        }
        return 0xFF;
    };

    LichiiIO::Writer graphics;
    graphics.Put(static_cast<u32>(entries.size()));
    for (const auto& entry : entries) {
        graphics.Put(entry->unique_hashes);
        graphics.Put(entry->state);
        graphics.Put(entry->attribute_mask);
        graphics.Put(template_of(*entry));
        for (size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
            graphics.Put(add_stage(*entry, stage));
            graphics.Put(entry->infos[stage] ? add_packed_info(entry->infos[stage]) : s32{-1});
        }
        graphics.Put(static_cast<u32>(entry->queries.size()));
        for (const auto& queries : entry->queries) {
            LichiiIO::PutQueries(graphics, queries);
        }
    }
    graphics.Put(static_cast<u32>(compute.size()));
    for (const auto& stored : compute) {
        graphics.Put(stored.key);
        graphics.Put(stored.local_memory_size);
        graphics.Put(add_blob(stored.spirv));
        graphics.Put(add_info(stored.info));
        LichiiIO::PutQueries(graphics, stored.queries);
    }
    std::vector<std::pair<u64, u32>> program_stages(precompiled_programs.begin(),
                                                    precompiled_programs.end());
    graphics.PutRange(program_stages);

    // The index (everything but the SPIR-V) is one zstd frame read at boot; each SPIR-V module is
    // a frame of its own, read when a pipeline needs it, so the modules stay on disk.
    LichiiIO::Writer index;
    index.PutRange(templates);
    index.Put(static_cast<u32>(infos.values.size()));
    for (const auto& info : infos.values) {
        index.PutBytes(info.data(), info.size());
    }
    index.PutBytes(graphics.buffer.data(), graphics.buffer.size());
    graphics.buffer = {};
    const std::vector<u8> compressed_index =
        Common::Compression::CompressDataZSTD(index.buffer.data(), index.buffer.size(), 1);

    u64 modules_bytes = 0;
    u64 spirv_bytes = 0;
    std::vector<SpirvStore::Module> table(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
        table[i] = {modules_bytes, frame_sizes[i], frame_words[i]};
        modules_bytes += frame_sizes[i];
        spirv_bytes += u64{frame_words[i]} * sizeof(u32);
    }

    const auto temp_path = path.string() + ".tmp";
    {
        std::ofstream out{temp_path, std::ios::binary | std::ios::trunc};
        const u32 magic = PrecompileStoreMagic;
        const u32 version = PrecompileStoreVersion;
        const u64 index_size = index.buffer.size();
        const u64 index_compressed = compressed_index.size();
        const u32 module_count = static_cast<u32>(table.size());
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&validity), sizeof(validity));
        out.write(reinterpret_cast<const char*>(&index_size), sizeof(index_size));
        out.write(reinterpret_cast<const char*>(&index_compressed), sizeof(index_compressed));
        out.write(reinterpret_cast<const char*>(compressed_index.data()),
                  static_cast<std::streamsize>(compressed_index.size()));
        out.write(reinterpret_cast<const char*>(&module_count), sizeof(module_count));
        out.write(reinterpret_cast<const char*>(table.data()),
                  static_cast<std::streamsize>(table.size() * sizeof(SpirvStore::Module)));
        for (size_t i = 0; i < frames.size(); ++i) {
            if (frame_spool[i] >= 0) {
                const std::vector<u8> frame = ReadSpoolFrame(frame_spool[i]);
                if (frame.size() != frame_sizes[i]) {
                    throw std::runtime_error("spool read failed");
                }
                out.write(reinterpret_cast<const char*>(frame.data()),
                          static_cast<std::streamsize>(frame.size()));
            } else {
                out.write(reinterpret_cast<const char*>(frames[i]->data()),
                          static_cast<std::streamsize>(frames[i]->size()));
            }
        }
        if (!out) {
            throw std::runtime_error("write failed");
        }
    }
    {
        // A store read earlier this session keeps the old file open.
        std::unique_ptr<SpirvStore> old = std::move(spirv_store);
    }
    std::filesystem::rename(temp_path, path);
    LOG_INFO(Render_Vulkan,
             "LICHII_PRECOMPILE: stored {} translations and {} compute pipelines: index {} MiB "
             "({} infos), {} distinct SPIR-V modules {} MiB ({} MiB compressed), in {:.1f} s",
             entries.size(), compute.size(), compressed_index.size() >> 20, infos.values.size(),
             frames.size(), spirv_bytes >> 20, modules_bytes >> 20,
             std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
} catch (const std::exception& exception) {
    LOG_WARNING(Render_Vulkan, "LICHII_PRECOMPILE: could not store the translations: {}",
                exception.what());
}

bool PipelineCache::LoadPrecompileStore(const std::filesystem::path& path, u64 validity,
                                        std::vector<GraphicsPipelineCacheKey>& prebuild_keys,
                                        const VideoCore::DiskResourceLoadCallback& callback) try {
    std::ifstream in{path, std::ios::binary};
    if (!in) {
        return false;
    }
    u32 magic{}, version{};
    u64 stored_validity{}, index_size{}, index_compressed{};
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&stored_validity), sizeof(stored_validity));
    in.read(reinterpret_cast<char*>(&index_size), sizeof(index_size));
    in.read(reinterpret_cast<char*>(&index_compressed), sizeof(index_compressed));
    if (!in || magic != PrecompileStoreMagic || version != PrecompileStoreVersion ||
        stored_validity != validity || index_size > (u64{8} << 30) ||
        index_compressed > (u64{8} << 30)) {
        LOG_INFO(Render_Vulkan,
                 "LICHII_PRECOMPILE: stored translations are outdated (archives, mesh states, "
                 "driver or this build changed): translating again");
        return false;
    }
    std::vector<u8> compressed(index_compressed);
    in.read(reinterpret_cast<char*>(compressed.data()), static_cast<std::streamsize>(compressed.size()));
    std::vector<u8> raw(index_size);
    if (!in || Common::Compression::DecompressDataZSTD(compressed, raw) != index_size) {
        throw std::runtime_error("index decompression failed");
    }
    compressed = {};
    // The SPIR-V module table; the modules follow it and stay in the file.
    u32 module_count{};
    in.read(reinterpret_cast<char*>(&module_count), sizeof(module_count));
    if (!in || module_count > (1u << 24)) {
        throw std::runtime_error("bad module table");
    }
    auto store = std::make_unique<SpirvStore>();
    store->modules.resize(module_count);
    in.read(reinterpret_cast<char*>(store->modules.data()),
            static_cast<std::streamsize>(module_count * sizeof(SpirvStore::Module)));
    if (!in) {
        throw std::runtime_error("truncated module table");
    }
    const u64 modules_base = static_cast<u64>(in.tellg());
    for (auto& module : store->modules) {
        module.offset += modules_base;
    }
    in.close();
    store->path = path;
    store->file.open(path, std::ios::binary);
    if (!store->file) {
        throw std::runtime_error("cannot reopen the store");
    }

    LichiiIO::Reader r{raw};
    std::vector<FixedPipelineState> templates;
    r.GetRange(templates, 256);
    // Each distinct info once, shared by every entry that uses it.
    std::vector<InfoRef> infos(r.Get<u32>());
    for (auto& info : infos) {
        // Kept packed: expanded when a pipeline first needs it.
        const size_t start = r.Offset();
        LichiiIO::SkipInfo(r);
        const auto bytes = r.Since(start);
        auto packed = std::make_shared<PackedInfo>();
        packed->bytes.assign(bytes.begin(), bytes.end());
        info = InfoRef{std::move(packed)};
    }
    const auto module_at = [&](s32 index) -> s32 {
        if (index >= 0 && static_cast<size_t>(index) >= module_count) {
            throw std::runtime_error("bad SPIR-V index");
        }
        return index;
    };
    const auto info_at = [&](s32 index) -> InfoRef {
        if (index < 0) {
            return InfoRef{};
        }
        if (static_cast<size_t>(index) >= infos.size()) {
            throw std::runtime_error("bad info index");
        }
        return infos[index];
    };
    std::vector<std::shared_ptr<ReusableStages>> entries(r.Get<u32>());
    for (auto& entry : entries) {
        entry = std::make_shared<ReusableStages>();
        entry->unique_hashes = r.Get<std::array<u64, Maxwell::MaxShaderProgram>>();
        entry->state = r.Get<FixedPipelineState>();
        entry->state.driver_version = 0; // stores written before ReuseState cleared it
        entry->attribute_mask = r.Get<u64>();
        const u8 templ = r.Get<u8>();
        for (size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
            entry->stored_spirv[stage] = module_at(r.Get<s32>());
            entry->infos[stage] = info_at(r.Get<s32>());
        }
        const u32 query_count = r.Get<u32>();
        if (query_count > Maxwell::MaxShaderProgram) {
            throw std::runtime_error("bad query count");
        }
        for (u32 i = 0; i < query_count; ++i) {
            entry->queries.push_back(LichiiIO::GetQueries(r));
        }
        if (templ < templates.size()) {
            GraphicsPipelineCacheKey key{};
            key.unique_hashes = entry->unique_hashes;
            key.state = templates[templ];
            key.state.driver_version = device.GetDriverVersion();
            prebuild_keys.push_back(key);
        }
    }
    std::vector<StoredCompute> compute(r.Get<u32>());
    std::vector<s32> compute_modules(compute.size());
    for (size_t i = 0; i < compute.size(); ++i) {
        auto& stored = compute[i];
        stored.key = r.Get<ComputePipelineCacheKey>();
        stored.local_memory_size = r.Get<u32>();
        compute_modules[i] = module_at(r.Get<s32>());
        const auto info = info_at(r.Get<s32>());
        if (compute_modules[i] < 0 || !info) {
            throw std::runtime_error("compute entry without SPIR-V");
        }
        stored.info = *info;
        stored.queries = LichiiIO::GetQueries(r);
    }
    std::vector<std::pair<u64, u32>> program_stages;
    r.GetRange(program_stages);
    if (!r.AtEnd()) {
        throw std::runtime_error("trailing data");
    }
    raw = {};
    spirv_store = std::move(store);

    const size_t graphics_count = entries.size();
    {
        std::scoped_lock lock{reusable_stages_mutex};
        for (auto& entry : entries) {
            reusable_stages[HashPrograms(entry->unique_hashes)].push_back(std::move(entry));
        }
    }
    precompiled_programs.insert(program_stages.begin(), program_stages.end());

    // Compute pipelines are built from the stored SPIR-V (the driver's cache has them).
    std::mutex compute_mutex;
    std::atomic<size_t> done{};
    // Usually a second or two, but a minute with a cold driver cache (a new executable): the
    // same loading bar shows how far it is.
    const size_t compute_total = compute.size();
    if (std::ofstream hint{path.parent_path() / "lichii_precompile_effects.count", std::ios::trunc}) {
        hint << compute_total << std::endl;
    }
    if (callback && compute_total != 0) {
        callback(VideoCore::LoadCallbackStage::Build, loading_base, loading_base + compute_total);
    }
    for (size_t i = 0; i < compute.size(); ++i) {
        auto& stored = compute[i];
        const s32 module = compute_modules[i];
        precompiled_compute_programs.emplace(stored.key.unique_hash, stored.key);
        if (compute_cache.find(stored.key) != compute_cache.end()) {
            done.fetch_add(1);
            continue;
        }
        workers.QueueWork([this, &stored, module, &compute_mutex, &done] {
            SCOPE_EXIT {
                done.fetch_add(1);
            };
            try {
                auto pipeline = std::make_unique<ComputePipeline>(
                    device, scheduler, vulkan_pipeline_cache, descriptor_pool,
                    guest_descriptor_queue, descriptor_buffer_ring, nullptr, nullptr,
                    nullptr, stored.info, BuildShader(device, ReadStoredSpirv(module)),
                    stored.key.unique_hash);
                std::scoped_lock lock{compute_mutex};
                compute_cache.emplace(stored.key,
                                      ComputePipelineCacheEntry{.pipeline = std::move(pipeline)});
                precompiled_compute.emplace(
                    stored.key, PrecompiledCompute{std::move(stored.queries),
                                                   stored.local_memory_size});
            } catch (const std::exception& exception) {
                LOG_WARNING(Render_Vulkan, "LICHII_PRECOMPILE compute {:016x} failed: {}",
                            stored.key.unique_hash, exception.what());
            }
        });
    }
    if (callback && compute_total != 0) {
        size_t shown = 0;
        while (done.load() < compute_total) {
            std::this_thread::sleep_for(std::chrono::milliseconds{50});
            const size_t now = done.load();
            if (now != shown) {
                shown = now;
                callback(VideoCore::LoadCallbackStage::Build, loading_base + now,
                         loading_base + compute_total);
            }
        }
    }
    workers.WaitForRequests({});
    LOG_INFO(Render_Vulkan,
             "LICHII_PRECOMPILE: read {} translations and {} compute pipelines ({} infos)",
             graphics_count, compute.size(), infos.size());
    return true;
} catch (const std::exception& exception) {
    LOG_WARNING(Render_Vulkan, "LICHII_PRECOMPILE: could not read the stored translations: {}",
                exception.what());
    return false;
}

} // namespace Vulkan
