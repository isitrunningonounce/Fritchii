// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <map>
#include <optional>
#include <set>
#include <string_view>
#include <tuple>

#include "core/arm/bananza_nvn2_commands.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_vulkan/vk_blit_screen.h"
#include "video_core/shader_environment.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class FrameDump;
class MemoryAllocator;
class PipelineCache;
class Scheduler;

class BananzaNvn2Replay {
public:
    struct Source {
        FramebufferTextureInfo texture;
        VideoCore::Surface::PixelFormat format;
        GPUVAddr address;
    };

    BananzaNvn2Replay(const Device& device, MemoryAllocator& allocator, Scheduler& scheduler,
                       const PipelineCache& pipeline_cache, FrameDump& frame_dump);
    void Replay(const Core::BananzaNvn2Commands::Submission& submission);
    std::optional<Source> PresentSource(const Core::BananzaNvn2Hle::ResourceTextureInfo& texture);

private:
    class ComputeEnvironment;
    using Texture = Core::BananzaNvn2Hle::ResourceTextureInfo;
    using Operation = Core::BananzaNvn2Commands::Operation;
    using Recording = Core::BananzaNvn2Commands::Recording;
    using Submission = Core::BananzaNvn2Commands::Submission;
    using Program = Core::BananzaNvn2Hle::ResourceProgramInfo;
    using Range = std::pair<u64, u64>;
    using ImageKey = std::tuple<GPUVAddr, u32, u32, u32, u32, u32, u32, u32>;
    using ColorFormats = std::array<VkFormat, 8>;
    struct ColorState {
        std::array<u64, 8> blend{};
        u32 control{0x08030000};
        u32 channel_mask{0xffffffff};
        u32 alpha_ref{};
        auto operator<=>(const ColorState&) const = default;
    };
    struct RasterState {
        u64 depth_stencil{0x1118111800000022ULL};
        u32 polygon{0x14};
        VkFormat depth_format{};
        VkImageAspectFlags depth_aspects{};
        bool negative_one_to_one{};
        bool origin_lower_left{};
        bool depth_clamp{};
        bool discard{};
        bool depth_bias{};
        auto operator<=>(const RasterState&) const = default;
    };
    using PipelineKey = std::tuple<u64, u64, u64, ColorFormats, VkPrimitiveTopology, std::array<u32, 32>,
                                   std::array<u32, 32>, std::array<u32, 32>, ColorState, RasterState>;

    struct Image {
        vk::Image image;
        vk::ImageView view;
        VkExtent2D extent;
        VkFormat format;
        VideoCore::Surface::PixelFormat surface_format;
        u32 levels;
        u32 layers;
        u32 depth;
        bool volume;
        bool color_attachment;
        bool storage_image;
        VkImageAspectFlags aspects;
        std::map<std::tuple<VkFormat, u32, u32, u32>, vk::ImageView> attachment_views;
        std::map<std::tuple<VkFormat, u32, u32, u32, u32, VkImageViewType, VkImageAspectFlags,
                            std::array<u32, 4>>,
                 vk::ImageView>
            sampled_views;
        bool initialized{};
        bool gpu_written{};
        u64 cpu_hash{};
        u64 cpu_serial{};
        u64 checked_submission{};
        u64 write_version{};
    };

    struct AttachmentView {
        VkFormat format{};
        u32 level{};
        u32 layer{};
        u32 layers{1};
    };
    struct FeedbackImage {
        vk::Image image;
        std::map<std::tuple<VkFormat, u32, u32, u32, u32, VkImageViewType, VkImageAspectFlags,
                            std::array<u32, 4>>,
                 vk::ImageView> views;
    };

    struct DepthComparisonImage {
        vk::Image image;
        u64 write_version{};
    };

    struct UnboundTexture {
        vk::Image image;
        vk::ImageView view;
    };

    struct UniformBinding {
        u32 stage;
        u32 bank;
        u32 binding;
        u32 size;
        std::vector<u8> literal;
    };

    struct StorageBinding {
        u32 stage;
        u32 binding;
        u32 slot;
        bool writable;
    };

    struct Pipeline {
        vk::DescriptorSetLayout set_layout;
        vk::PipelineLayout layout;
        vk::Pipeline pipeline;
        std::vector<UniformBinding> uniforms;
        std::vector<StorageBinding> storage;
        std::vector<std::tuple<u32, u32, Shader::TextureDescriptor>> textures;
        std::vector<u32> streams;
        VkShaderStageFlags stages{};
        bool uses_render_area{};
    };

    struct DrawState {
        u32 count{};
        u32 instances{1};
        u32 first{};
        u32 base_instance{};
        s32 base_vertex{};
        VkIndexType index_type{VK_INDEX_TYPE_MAX_ENUM};
        Range index_range{};
        Range indirect_range{};
        bool indexed_indirect{};
    };

    struct ComputeLayout {
        std::array<u8, 8> roots;
        u32 uniforms;
        u32 storage;
        u32 textures;
        u32 images;
        u32 uniform_base;
        u32 storage_base;
        u32 texture_base;
        u32 image_base;
        u32 size;
    };

    struct ComputePipeline {
        Pipeline state;
        VideoCommon::RecordedEnvQueries queries;
        std::vector<std::pair<u32, Shader::StorageBufferDescriptor>> storage;
        std::vector<std::pair<u32, Shader::ImageDescriptor>> images;
        u32 texture_bank;
        std::array<u32, 3> workgroup;
    };

    struct StorageUpload {
        vk::Buffer buffer;
        u64 guest_address;
        u64 written_bytes{};
    };

    struct StorageSpan {
        Range range;
        u64 guest_address;
        std::vector<Range> snapshots;
    };

    struct BufferUpdate {
        u64 address;
        std::span<const u8> bytes;
    };

    Image* FindImage(const Texture& texture, bool create);
    VkImageView GetAttachmentView(Image& image, const AttachmentView& view);
    void ReplayRecording(const Recording& recording, const Submission& submission);
    void ClearColor(const Operation& operation);
    void ClearDepthStencil(const Operation& operation);
    void CopyTexture(const Operation& operation, const Submission& submission);
    void CopyBufferToTexture(const Operation& operation, const Submission& submission);
    void Draw(const Operation& operation, const Submission& submission);
    bool ResetIndirectDraws(const DrawState& draw, const Submission& submission);
    void ReplayEvent(const Operation& operation, const Submission& submission);
    void ReplayCounter(const Operation& operation, const Submission& submission);
    void CollectOcclusion();
    void Dispatch(const Operation& operation, const Submission& submission);
    std::optional<ComputePipeline> CreateComputePipeline(ComputeEnvironment& environment);
    std::optional<VkDescriptorImageInfo> BindStorageImage(
        const Submission& submission, u64 handle, const Shader::ImageDescriptor& binding,
        Image*& image);
    std::optional<DrawState> DecodeDraw(const Operation& operation) const;
    std::optional<Pipeline> CreatePipeline(const ColorFormats& formats, u32 color_count,
                                           VkPrimitiveTopology topology, const RasterState& raster);
    VkBuffer UploadRange(const Submission& submission, Range range);
    VkBuffer Upload(std::span<const u8> bytes, VkBufferUsageFlags extra_usage = 0);
    std::map<Range, StorageUpload>::iterator FindStorageRange(Range range);
    void CollectStorageSpans(const Submission& submission);
    std::pair<VkBuffer, u64> UploadBufferRange(const Submission& submission, Range range);
    std::pair<StorageUpload*, u64> WritableStorage(const Submission& submission, Range range);
    void WriteBackStorage(const Submission& submission);
    std::optional<VkDescriptorImageInfo> BindTexture(const Submission& submission, u32 stage,
                                                    const Shader::TextureDescriptor& descriptor,
                                                    std::span<Image* const> destinations,
                                                    std::optional<u64> compute_handle = std::nullopt);
    bool PrepareSampledImage(Image& image, const Texture& texture, const Submission& submission);
    VkImage PrepareDepthComparisonImage(const Image& source, VkFormat depth_format);
    VkDescriptorImageInfo BindUnboundTexture(VkImageViewType type);
    bool UploadTexture(Image& image, const Texture& texture, std::span<const u8> data);
    void NoteSkip(std::string_view reason);
    void TraceImage(std::string_view operation, const Texture& texture, u64 detail = 0);
    VkRect2D ScissorFor(VkExtent2D extent) const;
    void Initialize(Image& image);
    void Barrier();

    const Device& device;
    MemoryAllocator& allocator;
    Scheduler& scheduler;
    const PipelineCache& pipeline_cache;
    FrameDump& frame_dump;
    std::map<ImageKey, Image> images;
    std::map<const Image*, FeedbackImage> feedback_images;
    std::map<const Image*, DepthComparisonImage> depth_comparison_images;
    std::array<std::optional<Texture>, 8> colors;
    std::array<AttachmentView, 8> color_views{};
    std::optional<Texture> depth_texture;
    AttachmentView depth_view{};
    std::array<u32, 32> vertex_attributes{};
    std::array<u32, 32> vertex_strides{};
    std::array<u32, 32> vertex_divisors{};
    std::array<Range, 32> vertex_buffers{};
    std::array<std::array<Range, 32>, 6> uniform_buffers{};
    std::array<std::array<Range, 32>, 6> storage_buffers{};
    std::array<std::array<u64, 64>, 6> texture_handles{};
    std::array<std::array<u64, 32>, 6> image_handles{};
    std::array<std::array<u8, 256>, 8> graphics_constants{};
    std::array<std::array<u8, 256>, 8> compute_constants{};
    ColorState color_state{};
    RasterState raster_state{};
    std::array<u32, 2> stencil_ref{};
    std::array<u32, 2> stencil_compare_mask{255, 255};
    std::array<u32, 2> stencil_write_mask{255, 255};
    std::array<float, 2> depth_range{0.0f, 1.0f};
    std::array<float, 3> depth_bias{};
    std::array<float, 4> blend_constants{};
    u64 texture_pool{};
    u64 sampler_pool{};
    u64 copy_row_stride{};
    std::shared_ptr<const Program> graphics_program;
    std::shared_ptr<const Program> fragment_program;
    std::shared_ptr<const Program> geometry_program;
    std::shared_ptr<const Program> compute_program;
    VkViewport viewport{};
    VkRect2D scissor{};
    bool has_scissor{};
    bool render_enabled{true};
    bool origin_lower_left{};
    std::map<PipelineKey, std::optional<Pipeline>> pipelines;
    std::map<u64, std::vector<ComputePipeline>> compute_pipelines;
    std::map<Range, VkBuffer> uploaded_ranges;
    std::map<Range, StorageUpload> storage_uploads;
    std::map<Range, StorageSpan> storage_spans;
    vk::Buffer unbound_storage;
    std::map<VkImageViewType, UnboundTexture> unbound_textures;
    vk::Sampler unbound_texture_sampler;
    std::vector<BufferUpdate> buffer_updates;
    std::set<u64> reset_indirect_draws;
    std::vector<vk::Buffer> uploads;
    std::vector<vk::DescriptorPool> descriptor_pools;
    std::vector<vk::Sampler> samplers;
    u64 image_bytes{};
    u64 clears{};
    u64 copies{};
    u64 draws{};
    u64 dispatches{};
    u64 texture_uploads{};
    u64 last_reported_draws{};
    u64 last_reported_present{};
    std::chrono::steady_clock::time_point next_replay_report{
        std::chrono::steady_clock::now() + std::chrono::seconds{10}};
    u64 present_count{};
    std::set<std::pair<u64, std::string_view>> draw_diagnostics;
    std::map<std::string_view, u64> draw_skips;
    std::map<std::string_view, u64> dispatch_skips;
    std::set<u64> missing_event_waits;
    std::set<u64> unsupported_counters;
    static constexpr u32 OcclusionQueryCount = 256;
    vk::QueryPool occlusion_pool;
    u32 occlusion_queries{};
    u64 occlusion_samples{};
    bool count_samples{};
    bool precise_queries{};
    std::set<std::tuple<u32, u32, u32>> unsupported_color_formats;
    std::set<std::tuple<std::string_view, GPUVAddr, u64>> traced_images;
    std::set<std::tuple<GPUVAddr, u32, u32>> captured_inputs;
    bool trace_images{};
    bool replay_logged{};
    bool missing_draw_logged{};
    std::set<std::pair<u64, std::string_view>> dispatch_diagnostics;
    bool missing_source_logged{};
    bool image_budget_logged{};
};

} // namespace Vulkan
