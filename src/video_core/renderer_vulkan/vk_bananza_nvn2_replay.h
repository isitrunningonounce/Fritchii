// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <map>
#include <optional>
#include <set>
#include <string_view>
#include <tuple>

#include "core/arm/bananza_nvn2_commands.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_vulkan/vk_blit_screen.h"
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
    using Texture = Core::BananzaNvn2Hle::ResourceTextureInfo;
    using Operation = Core::BananzaNvn2Commands::Operation;
    using Recording = Core::BananzaNvn2Commands::Recording;
    using Submission = Core::BananzaNvn2Commands::Submission;
    using Program = Core::BananzaNvn2Hle::ResourceProgramInfo;
    using Range = std::pair<u64, u64>;
    using ImageKey = std::tuple<GPUVAddr, u32, u32, u32, u32>;
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
        auto operator<=>(const RasterState&) const = default;
    };
    using PipelineKey = std::tuple<u64, ColorFormats, VkPrimitiveTopology, std::array<u32, 32>,
                                   std::array<u32, 32>, std::array<u32, 32>, ColorState, RasterState>;

    struct Image {
        vk::Image image;
        vk::ImageView view;
        VkExtent2D extent;
        VkFormat format;
        VideoCore::Surface::PixelFormat surface_format;
        u32 levels;
        bool color_attachment;
        VkImageAspectFlags aspects;
        std::map<VkFormat, vk::ImageView> attachment_views;
        std::map<std::tuple<VkFormat, u32, u32, std::array<u32, 4>>, vk::ImageView> sampled_views;
        bool initialized{};
    };

    struct UniformBinding {
        u32 stage;
        u32 bank;
        u32 binding;
        u32 size;
        std::vector<u8> literal;
    };

    struct Pipeline {
        vk::DescriptorSetLayout set_layout;
        vk::PipelineLayout layout;
        vk::Pipeline pipeline;
        std::vector<UniformBinding> uniforms;
        std::vector<std::tuple<u32, u32, Shader::TextureDescriptor>> textures;
        std::vector<u32> streams;
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
    };

    struct BufferUpdate {
        u64 address;
        std::span<const u8> bytes;
    };

    Image* FindImage(const Texture& texture, bool create);
    void ReplayRecording(const Recording& recording, const Submission& submission);
    void ClearColor(const Operation& operation);
    void ClearDepthStencil(const Operation& operation);
    void CopyTexture(const Operation& operation);
    void Draw(const Operation& operation, const Submission& submission);
    std::optional<DrawState> DecodeDraw(const Operation& operation) const;
    std::optional<Pipeline> CreatePipeline(const ColorFormats& formats, u32 color_count,
                                           VkPrimitiveTopology topology, const RasterState& raster);
    VkBuffer UploadRange(const Submission& submission, Range range);
    VkBuffer Upload(std::span<const u8> bytes);
    std::optional<VkDescriptorImageInfo> BindTexture(const Submission& submission, u32 stage,
                                                    const Shader::TextureDescriptor& descriptor,
                                                    std::span<Image* const> destinations);
    bool UploadTexture(Image& image, const Texture& texture, std::span<const u8> data);
    void NoteSkip(std::string_view reason);
    void TraceImage(std::string_view operation, const Texture& texture, u64 detail = 0);
    void Initialize(Image& image);
    void Barrier();

    const Device& device;
    MemoryAllocator& allocator;
    Scheduler& scheduler;
    const PipelineCache& pipeline_cache;
    FrameDump& frame_dump;
    std::map<ImageKey, Image> images;
    std::array<std::optional<Texture>, 8> colors;
    std::array<bool, 8> base_color_views{};
    std::array<VkFormat, 8> color_view_formats{};
    std::optional<Texture> depth_texture;
    std::array<u32, 32> vertex_attributes{};
    std::array<u32, 32> vertex_strides{};
    std::array<u32, 32> vertex_divisors{};
    std::array<Range, 32> vertex_buffers{};
    std::array<std::array<Range, 32>, 2> uniform_buffers{};
    std::array<std::array<u64, 64>, 2> texture_handles{};
    std::array<std::array<u8, 256>, 8> graphics_constants{};
    ColorState color_state{};
    RasterState raster_state{};
    std::array<u32, 2> stencil_ref{};
    std::array<u32, 2> stencil_compare_mask{255, 255};
    std::array<u32, 2> stencil_write_mask{255, 255};
    std::array<float, 2> depth_range{0.0f, 1.0f};
    std::array<float, 4> blend_constants{};
    u64 texture_pool{};
    u64 sampler_pool{};
    std::shared_ptr<const Program> graphics_program;
    VkViewport viewport{};
    VkRect2D scissor{};
    bool has_scissor{};
    bool origin_lower_left{};
    std::map<PipelineKey, std::optional<Pipeline>> pipelines;
    std::map<Range, VkBuffer> uploaded_ranges;
    std::vector<BufferUpdate> buffer_updates;
    std::vector<vk::Buffer> uploads;
    std::vector<vk::DescriptorPool> descriptor_pools;
    std::vector<vk::Sampler> samplers;
    u64 image_bytes{};
    u64 clears{};
    u64 copies{};
    u64 draws{};
    u64 texture_uploads{};
    u64 last_reported_draws{};
    u64 present_count{};
    std::set<std::pair<u64, std::string_view>> draw_diagnostics;
    std::set<std::tuple<u32, u32, u32>> unsupported_color_formats;
    std::set<std::tuple<std::string_view, GPUVAddr, u64>> traced_images;
    std::set<GPUVAddr> captured_inputs;
    bool trace_images{};
    bool replay_logged{};
    bool missing_draw_logged{};
    bool missing_dispatch_logged{};
    bool missing_source_logged{};
};

} // namespace Vulkan
