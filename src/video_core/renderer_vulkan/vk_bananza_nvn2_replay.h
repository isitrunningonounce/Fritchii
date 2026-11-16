// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <source_location>
#include <optional>
#include <set>
#include <stop_token>
#include <thread>
#include <string_view>
#include <tuple>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include <boost/container/flat_map.hpp>

#include <boost/container/small_vector.hpp>

#include "common/container/unordered_set.h"
#include "common/thread_worker.h"
#include "core/arm/bananza_nvn2_commands.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_vulkan/vk_bananza_nvn2_async_builder.h"
#include "video_core/renderer_vulkan/vk_bananza_nvn2_kept_storage.h"
#include "video_core/renderer_vulkan/vk_bananza_nvn2_pipeline_list.h"
#include "video_core/renderer_vulkan/vk_bananza_nvn2_pipeline_state.h"
#include "video_core/renderer_vulkan/vk_bananza_nvn2_run_stats.h"
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
    ~BananzaNvn2Replay();
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
    using ColorFormats = BananzaNvn2ColorFormats;
    using ColorState = BananzaNvn2ColorState;
    using RasterState = BananzaNvn2RasterState;
    // The three numbers are the content hashes of the graphics, fragment and geometry
    // programs (zero for none): the same programs make the same pipeline in every run.
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
        u64 confirmed_replay_submission{};
        // The submission that was found to hold no replay state for the image's texture.
        u64 stateless_submission{};
        u64 write_version{};
        // Value of barrier_serial when a draw last wrote the image.
        u64 written_barrier{};
        u64 allocation_bytes{};
        u64 last_present{};
    };

    struct AttachmentView {
        VkFormat format{};
        u32 level{};
        u32 layer{};
        u32 layers{1};
    };
    using SampledBindingKey = std::tuple<u64, u64, u64>;
    struct SampledBinding {
        VkDescriptorImageInfo descriptor;
        Image* image;
        const Texture* texture;
    };
    struct RenderingScope {
        std::array<VkImageView, 8> colors;
        VkImageView depth;
        u32 color_count;
        u32 width;
        u32 height;
        u32 layers;

        bool operator==(const RenderingScope&) const = default;
    };
    struct FeedbackImage {
        vk::Image image;
        u64 allocation_bytes{};
        std::map<std::tuple<VkFormat, u32, u32, u32, u32, VkImageViewType, VkImageAspectFlags,
                            std::array<u32, 4>>,
                 vk::ImageView> views;
    };

    struct DepthComparisonImage {
        vk::Image image;
        u64 write_version{};
        u64 allocation_bytes{};
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

    union DescriptorData {
        VkDescriptorBufferInfo buffer;
        VkDescriptorImageInfo image;
    };

    struct Pipeline {
        vk::DescriptorSetLayout set_layout;
        vk::DescriptorUpdateTemplate update_template;
        vk::PipelineLayout layout;
        vk::Pipeline pipeline;
        std::vector<UniformBinding> uniforms;
        std::vector<StorageBinding> storage;
        std::vector<std::tuple<u32, u32, Shader::TextureDescriptor>> textures;
        std::vector<u32> streams;
        VkShaderStageFlags stages{};
        bool uses_render_area{};
        bool uses_blend_constants{};
        bool uses_stencil{};
        // Colour slots and depth this pipeline can write.
        u32 color_writes{};
        bool depth_write{};
        // The descriptors are pushed with each draw instead of bound as an allocated set.
        bool push_descriptors{};
        VkDescriptorSet last_set{};
        std::vector<DescriptorData> last_descriptors;
        u64 descriptor_submission{};
    };

    /// What a graphics pipeline is made from, as a value a worker thread can keep.
    struct PipelineInputs {
        std::shared_ptr<const Program> graphics_program;
        std::shared_ptr<const Program> fragment_program;
        std::shared_ptr<const Program> geometry_program;
        std::array<u32, 32> vertex_attributes;
        std::array<u32, 32> vertex_strides;
        std::array<u32, 32> vertex_divisors;
        ColorState color_state;
        // The guest format of each bound colour texture, PipelineList::NoColor for none.
        std::array<u32, 8> color_formats;
        ColorFormats formats;
        u32 color_count;
        VkPrimitiveTopology topology;
        RasterState raster;
    };

    struct BuiltPipeline {
        std::optional<Pipeline> pipeline;
        // Reasons a draw with it has to be skipped, reported by the thread that takes it.
        std::vector<std::string_view> skips;
        u64 ns{};
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

    // Host memory the spans of one submission are cut from.
    struct StorageArena {
        vk::Buffer buffer;
        u64 capacity;
        u64 used{};
        u64 flushed{};
    };

    struct StorageUpload {
        // The arena's buffer; the span starts at base.
        VkBuffer buffer;
        u64 base;
        u8* mapped;
        u64 guest_address;
        // Where the write-back reads the span: mapped, or the host region a span in device
        // memory is copied to (readback_buffer at readback_base).
        const u8* readback;
        VkBuffer readback_buffer{};
        u64 readback_base{};
        u64 size{};
        u64 written_bytes{};
        // The intervals bound to a shader with write access since the last write-back.
        std::vector<std::pair<u64, u64>> written;
        void NoteWritten(u64 begin, u64 end) {
            if (!written.empty() && written.back().first <= begin && end <= written.back().second) {
                return;
            }
            written.emplace_back(begin, end);
        }
        // The scheduler tick of the last command buffer with a write to the span.
        u64 write_tick{};
        // Written since the last write-back.
        bool dirty{};
        // A span in device memory: written since its bytes were last copied to the host
        // buffer, and the tick of the command buffer with that copy.
        bool readback_stale{};
        u64 readback_tick{};
        // Offsets draws wrote since the barrier with this serial, merged.
        u64 written_barrier{};
        boost::container::small_flat_map<u64, u64, 2> fresh;
    };

    struct StorageSpan {
        Range range;
        u64 guest_address;
        std::vector<Range> snapshots;
        // The span's buffer once a binding needed it.
        StorageUpload* upload{};
        // The bytes of each snapshot, in the order of `snapshots`.
        std::vector<Core::BananzaNvn2Commands::Submission::StorageAddress> bytes;
        // Set when every snapshot is read from guest memory and they form one host run:
        // where the span's first byte is.
        const u8* host{};
        // The pool every snapshot lies in when the game must flush it after writing, and the
        // newest flush over them; a pool of zero when that does not hold for all of them.
        u64 pool{};
        u64 stamp{};
    };

    struct BufferUpdate {
        u64 address;
        std::span<const u8> bytes;
    };
    static constexpr u64 BufferUpdatePageSize = 4096;

    // Host-visible memory shared by the uploads of one submission.
    struct UploadChunk {
        vk::Buffer buffer;
        u64 used{};
        // Last tick of the submission that used the chunk.
        u64 tick{};
        bool active{};
    };

    // Released by a submission whose commands may still run on the GPU.
    struct BusyResources {
        u64 tick;
        // Retired after the tick: storage and query results are read until then.
        u64 sequence;
        // Zero or one buffer; a vector keeps its address when the entry moves.
        std::vector<vk::Buffer> query_results;
        std::vector<std::pair<u64, vk::Buffer>> storage;
        std::vector<std::pair<u64, vk::Buffer>> device_storage;
        // Kept storage buffers taken out during the replay.
        std::vector<BananzaNvn2KeptStorage::Buffer> kept;
        std::vector<std::pair<bool, vk::DescriptorPool>> pools;
        std::vector<std::unique_ptr<DescriptorData[]>> pushed;
        std::vector<vk::Buffer> uploads;
    };

    // Descriptors left in a pool shared by the draws of one submission.
    struct DescriptorChunk {
        vk::DescriptorPool pool;
        u32 sets{};
        u32 uniforms{};
        u32 storage{};
        u32 textures{};
        u32 images{};
        bool reusable{};
    };

    using BufferRef = std::pair<VkBuffer, u64>;

    Image* FindImage(const Texture& texture, bool create);
    bool ReserveImageBytes(u64 bytes);
    VkImageView GetAttachmentView(Image& image, const AttachmentView& view);
    void ReplayRecording(const std::vector<const Operation*>& flat, const Submission& submission);
    void BindBufferRange(bool storage, u32 stage, u32 slot, Range range);
    BufferRef UploadDriverConstants(u32 base_instance);
    void ClearColor(const Operation& operation);
    void ClearDepthStencil(const Operation& operation);
    void CopyTexture(const Operation& operation, const Submission& submission);
    void CopyBufferToTexture(const Operation& operation, const Submission& submission);
    void Draw(const Operation& operation, const Submission& submission);
    bool ResetIndirectDraws(const DrawState& draw, const Submission& submission);
    void ReplayEvent(const Operation& operation, const Submission& submission);
    void ReplayCounter(const Operation& operation, const Submission& submission);
    // Counter resets and reports, and the event signals behind them, in recording order.
    // Queries before an event's query index count toward the next report.
    struct CounterEvent {
        enum class Kind : u8 { Reset, Report, Signal } kind;
        u32 query;
        Core::Memory::Memory* memory;
        u64 guest_address;
        u64 gpu_address;
        u64 submitted_ns;
        u32 value;
        bool add;
    };
    // Reads this submission's queries now; earlier submissions are retired first.
    void CollectOcclusion();
    void ApplyCounterEvents(std::span<const u64> results, std::span<const CounterEvent> events,
                            u32 queries);
    void RetireLoop(std::stop_token token);
    void WaitRetireIdle();
    void ProbeHostImport(const u8* pointer, u64 size);
    // Storage buffer fills run on worker threads; WaitCopies() before the GPU can use them.
    struct CopyJob {
        u8* dest;
        const u8* source;
        u64 size;
    };
    std::vector<CopyJob> copy_jobs;
    u64 copy_job_bytes{};
    void DispatchCopies();
    void WaitCopies();
    /// Before guest memory is read: waits for the write-back of earlier submissions over it.
    void AwaitGuestWrites(const u8* data, u64 size);
    /// A snapshot's bytes; one that is guest memory itself is waited for first.
    std::span<const u8> ReadSnapshot(const Submission& submission, Range range);
    u32 import_probes{};

    struct WriteBack {
        u64 guest_address;
        u64 base;
        u64 bytes;
        const u8* data;
        std::vector<Range> snapshots;
        // The GPU is done with the span after this tick.
        u64 tick{};
        // The span's bytes in guest memory when it is compared in one piece.
        const u8* before{};
        // With `before`: the intervals a shader could write; only they are compared.
        std::vector<std::pair<u64, u64>> written;
    };
    // One GPU timestamp of the pass trace: taken when all earlier work has finished.
    struct PassLabel {
        u64 target;
        u32 width;
        u32 height;
        u64 program;
        u64 draws;
        // 1 targets changed, 2 samples an image written in the scope, 4 and 8 storage writes,
        // 16 the scope had been ended by other work, 32 before a flush, 64 after a flush.
        u32 reason;
        // What the draw sampled that the scope had written: 1 depth, 2 a colour target, 4 other.
        u32 fresh;
        u32 fresh_slot;
        u64 storage;
        u64 storage_size;
    };
    struct RetireJob {
        u64 tick;
        u64 sequence;
        std::shared_ptr<const Submission> submission;
        std::vector<WriteBack> writes;
        // The arenas the writes read from.
        std::vector<const vk::Buffer*> written_buffers;
        // The kept storage buffers among them stay alive with the job.
        std::vector<BananzaNvn2KeptStorage::Buffer> kept;
        std::vector<CounterEvent> counters;
        u32 queries;
        const vk::Buffer* query_results;
        u64 replay_start_ns;
        u64 push_ns;
        u64 first_tick;
        u64 draws;
        std::vector<PassLabel> pass_labels;
        std::map<u32, u32> scope_end_sites;
        u64 pass_end_draws;
    };
    void ReportPassTimes(const RetireJob& job);
    void Dispatch(const Operation& operation, const Submission& submission);
    std::optional<ComputePipeline> CreateComputePipeline(ComputeEnvironment& environment);
    std::optional<VkDescriptorImageInfo> BindStorageImage(
        const Submission& submission, u64 handle, const Shader::ImageDescriptor& binding,
        Image*& image);
    std::optional<DrawState> DecodeDraw(const Operation& operation) const;
    /// A program the game made: the listed pipelines it completes go to the workers.
    void NoteProgram(const std::shared_ptr<const Program>& program);
    void BuildAhead(const BananzaNvn2PipelineList::Entry& entry);
    /// Takes the pipelines the workers have finished.
    void CollectPipelines();
    std::optional<Pipeline> CreatePipeline(const PipelineInputs& inputs,
                                           std::vector<std::string_view>& skips) const;
    BuiltPipeline BuildPipeline(const PipelineInputs& inputs) const;
    void TakePipeline(std::optional<Pipeline>& slot, BuiltPipeline& built);
    BufferRef UploadRange(const Submission& submission, Range range);
    VkBuffer Upload(std::span<const u8> bytes, VkBufferUsageFlags extra_usage = 0,
                    std::optional<u64> patch_address = std::nullopt);
    BufferRef UploadStream(std::span<const u8> bytes,
                           std::optional<u64> patch_address = std::nullopt);
    VkDescriptorSet AllocateSet(VkDescriptorSetLayout layout, size_t uniforms, size_t storage,
                                size_t textures, size_t images);
    VkFormatFeatureFlags2 FormatFeatures(VkFormat format);
    std::map<Range, StorageUpload>::iterator FindStorageRange(Range range);
    void CollectStorageSpans(const Submission& submission);
    void AddBufferUpdate(u64 address, std::span<const u8> bytes);
    void ApplyBufferUpdates(u64 address, std::span<u8> bytes) const;
    bool HasBufferUpdates(Range range) const;
    std::pair<VkBuffer, u64> UploadBufferRange(const Submission& submission, Range range);
    std::pair<StorageUpload*, u64> WritableStorage(const Submission& submission, Range range);
    std::pair<StorageUpload*, u64> CreateStorageUpload(const Submission& submission, Range range);
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
    void Barrier(std::source_location location = std::source_location::current());
    UploadChunk& NextUploadChunk();
    void ReclaimBusyResources();
    void EndRendering();
    // Draws record directly. Other replay commands end the active rendering scope.
    template <typename T>
    void Record(T&& command, std::source_location location = std::source_location::current());

    const Device& device;
    MemoryAllocator& allocator;
    Scheduler& scheduler;
    const PipelineCache& pipeline_cache;
    FrameDump& frame_dump;
    std::map<ImageKey, Image> images;
    // The images found last, in front of the map.
    struct ImageMemo {
        ImageKey key{};
        Image* image{};
    };
    std::array<ImageMemo, 1024> image_memo{};
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
    std::array<BufferRef, 32> vertex_uploads{};
    std::array<std::array<Range, 32>, 6> uniform_buffers{};
    std::array<std::array<Range, 32>, 6> storage_buffers{};
    std::array<std::array<u64, 64>, 6> texture_handles{};
    std::array<std::array<u64, 32>, 6> image_handles{};
    std::array<std::array<u8, 256>, 8> graphics_constants{};
    std::array<BufferRef, 8> graphics_constant_uploads{};
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
    decltype(pipelines)::iterator last_pipeline = pipelines.end();
    std::map<u64, std::vector<ComputePipeline>> compute_pipelines;
    Common::unordered_map<Range, BufferRef> uploaded_ranges;
    // Ranges kept in VRAM between submissions, by host address and size.
    struct FixedEntry {
        BufferRef buffer;
        u64 pool;
        u64 stamp;
        u64 hash;
    };
    struct FixedChunk {
        vk::Buffer buffer;
        u64 used{};
    };
    Common::unordered_map<Range, FixedEntry> fixed_cache;
    std::vector<FixedChunk> fixed_chunks;
    // A pool's bytes in device memory, kept between frames (EDEN_NX2_NVN_POOL_MIRROR=1).
    struct PoolMirror {
        vk::Buffer buffer;
        u64 size{};
        // Per 64 KiB page: the flush stamp it was synced with, plus one; zero is never.
        std::vector<u64> synced;
    };
    Common::unordered_map<u64, PoolMirror> pool_mirrors;
    u64 pool_mirror_bytes{};
    u64 pool_mirror_pages{};
    // The mirror of a pool, made when it is first needed; nothing past the limit.
    PoolMirror* GetMirror(u64 pool, u64 size);
    // Sends the submission's staged pages to the mirrors; false when its staged ranges have
    // to be uploaded one by one.
    bool SyncMirrorPages(const Submission& submission);
    bool staged_ranges_ready{};
    // The range in its pool's mirror, synced; nothing when the mirror cannot take it.
    BufferRef MirrorUpload(const Submission& submission,
                           const Core::BananzaNvn2Commands::Submission::BufferBytes& entry,
                           u64 size);
    u64 fixed_hits{};
    u64 fixed_uploads{};
    u64 fixed_upload_bytes{};
    u64 fixed_checks{};
    u64 fixed_mismatches{};
    u64 fixed_check_budget{};
    BufferRef FixedUpload(std::span<const u8> bytes, u64 pool, u64 stamp);
    std::map<Range, StorageUpload> storage_uploads;
    // Counts the events after which a range may resolve to another buffer than before.
    u64 upload_generation{1};
    // What each uniform slot bound last, so a slot that repeats a range skips the lookups.
    struct UploadMemo {
        Range range{};
        BufferRef ref{};
        u64 generation{};
    };
    std::array<std::array<UploadMemo, 32>, 8> uniform_upload_memo{};
    // Drops the slots whose last range overlaps [begin, end): the others resolve as before.
    void ForgetUploads(u64 begin, u64 end) {
        for (auto& stage : uniform_upload_memo) {
            for (auto& memo : stage) {
                if (memo.range.first < end && begin < memo.range.first + memo.range.second) {
                    memo.generation = 0;
                }
            }
        }
    }
    // The same uploads by their exact range, for the lookup every bound range makes.
    Common::unordered_map<Range, StorageUpload*> storage_upload_of;
    std::vector<StorageArena> storage_arenas;
    // Device memory for spans a fragment shader binds first, and those spans.
    std::vector<StorageArena> device_arenas;
    std::vector<StorageUpload*> device_uploads;
    std::map<u64, std::vector<vk::Buffer>> free_device_storage;
    u64 free_device_bytes{};
    bool storage_for_fragment{};
    u64 device_spans{};
    u64 device_span_bytes{};
    u64 device_moves{};
    void RecordStorageReadback(bool flush = false);
    // The submission's large copies; the fill jobs wait for them.
    std::shared_ptr<Core::BananzaNvn2Commands::Submission::CopyState> late_copies;
    // A span in device memory was written and not copied back yet.
    bool device_readback_pending{};
    std::pair<VkBuffer, u64> AllocateDeviceStorage(u64 aligned, u8*& mapped);
    void MoveStorageToDevice(StorageUpload& upload);
    // Every storage range of the submission to its span; used while no two spans overlap.
    Common::unordered_map<Range, StorageSpan*> storage_span_of;
    bool storage_spans_disjoint{};
    bool storage_guest_disjoint{};
    void NoteStorageRead(const StorageUpload& upload, u64 offset, u64 size);
    StorageSpan* BoundSpan(Range range);
    std::map<Range, StorageSpan> storage_spans;
    std::map<u64, u64> storage_guest_ends;
    std::map<u64, std::vector<vk::Buffer>> free_storage_buffers;
    u64 free_storage_bytes{};
    vk::Buffer unbound_storage;
    std::map<VkImageViewType, UnboundTexture> unbound_textures;
    vk::Sampler unbound_texture_sampler;
    std::map<u64, std::vector<BufferUpdate>> buffer_updates;
    Common::unordered_set<u64> reset_indirect_draws;
    std::vector<vk::Buffer> uploads;
    std::vector<vk::DescriptorPool> free_descriptor_pools;
    std::vector<UploadChunk> upload_chunks;
    std::vector<size_t> active_chunks;
    std::vector<BusyResources> busy_resources;
    u64 replay_flush_ends{};
    u64 upload_alignment{};
    std::vector<DescriptorChunk> descriptor_chunks;
    // The data of pushed descriptors, read when the draw's commands are recorded.
    static constexpr size_t PushBlockEntries = 16384;
    std::vector<std::unique_ptr<DescriptorData[]>> push_blocks;
    std::vector<std::unique_ptr<DescriptorData[]>> free_push_blocks;
    size_t push_used{PushBlockEntries};
    std::map<std::array<u32, 17>, vk::Sampler> sampler_cache;
    Common::unordered_map<SampledBindingKey, SampledBinding> sampled_bindings;
    std::unordered_map<VkFormat, VkFormatFeatureFlags2> format_features;
    VkPhysicalDeviceLimits device_limits{};
    VkPhysicalDeviceFeatures device_features{};
    std::array<u8, 0x1000> driver_constants{};
    u32 driver_base_instance{};
    BufferRef driver_upload{};
    bool driver_storage_dirty{true};
    u64 image_bytes{};
    u64 clears{};
    u64 copies{};
    u64 draws{};
    u64 dispatches{};
    u64 texture_uploads{};
    u64 last_reported_draws{};
    u64 last_reported_present{};
    // Replay thread time since the last report, in nanoseconds.
    u64 replay_ns{};
    u64 finish_ns{};
    u64 hash_ns{};
    u64 hash_bytes{};
    u64 upload_ns{};
    // Totals for the slow-replay line.
    u64 pipeline_creates{};
    // Pipelines a worker is building, and the draws left out meanwhile.
    std::set<const std::optional<Pipeline>*> building_pipelines;
    // The pipelines of earlier runs, and those of them the workers are building now. A draw
    // that needs one of these builds it itself and the worker's is dropped.
    std::optional<BananzaNvn2PipelineList> pipeline_list;
    std::set<const std::optional<Pipeline>*> ahead_pipelines;
    // The programs the list names that the game has made, by content hash.
    std::unordered_map<u64, std::shared_ptr<const Program>> listed_programs;
    u64 ahead_queued{};
    u64 ahead_built{};
    u64 ahead_dropped{};
    u64 draw_built{};
    u64 reported_ahead{};
    u64 worker_pipelines{};
    u64 pipeline_wait_draws{};
    u64 reported_pipeline_wait_draws{};
    u64 pipeline_create_ns{};
    u64 image_creates{};
    u64 replayed_submissions{};
    u64 finishes{};
    // Draw phases: decode, targets, pipeline, vertex and index data, uniforms, storage,
    // textures, descriptor set, state, record.
    std::array<u64, 10> draw_phase_ns{};
    u64 draw_ns{};
    u64 dispatch_ns{};
    std::chrono::steady_clock::time_point next_replay_report{
        std::chrono::steady_clock::now() + std::chrono::seconds{10}};
    u64 present_count{};
    std::set<std::pair<u64, std::string_view>> draw_diagnostics;
    std::map<std::string_view, u64> draw_skips;
    std::map<std::string_view, u64> dispatch_skips;
    std::set<u64> missing_event_waits;
    std::set<u64> unsupported_counters;
    static constexpr u32 OcclusionQueryCount = 4096;
    vk::QueryPool occlusion_pool;
    static constexpr u32 PassTraceQueries = 8192;
    bool pass_trace{};
    bool pass_trace_active{};
    vk::QueryPool pass_trace_pool;
    std::vector<PassLabel> pass_trace_labels;
    std::atomic<bool> pass_trace_busy{};
    std::chrono::steady_clock::time_point pass_trace_next{};
    void PassTimestamp(PassLabel label, bool top = false);
    const Image* fresh_image{};
    // A binding of this draw overlaps storage a draw wrote since the last barrier.
    bool storage_fresh{};
    // What the draw being looked up samples and reads of written storage; whether that needs
    // a barrier is decided where the draw is issued.
    struct StorageRead {
        const StorageUpload* upload;
        u64 offset;
        u64 size;
    };
    boost::container::small_vector<const Image*, 16> draw_sampled;
    boost::container::small_vector<StorageRead, 8> draw_storage_reads;
    // Colour slots the current draw's pipeline writes.
    u32 draw_color_writes{~0U};
    std::map<u32, u32> scope_end_sites;
    Range traced_storage{};
    u32 occlusion_queries{};
    std::vector<CounterEvent> counter_events;
    std::vector<vk::Buffer> free_query_results;
    bool count_samples{};
    // The game reported twice without a reset in between: keep counting after a report.
    bool cumulative_reports{};
    u64 occlusion_flushes{};
    std::optional<RenderingScope> rendering_scope;
    VkPipeline scope_pipeline{};
    VkDescriptorSet scope_descriptor_set{};
    std::array<BufferRef, 32> scope_vertices{};
    std::tuple<VkBuffer, u64, VkIndexType> scope_index{};
    bool scope_wrote_storage{};
    u64 rendering_scopes{};
    u64 next_flush_draw{~u64{}};
    u64 replay_flushes{};
    std::chrono::steady_clock::time_point next_replay_flush{};
    // Rendering scopes begun since the last flush.
    u32 flush_scopes{};
    u64 barrier_serial{1};
    bool sampled_fresh{};
    u64 barriers{};
    u64 event_finish_ns{};
    std::atomic<u64> writeback_bytes{};
    u64 event_signal_waits{};
    std::atomic<u64> writeback_compared{};
    u64 event_signals{};
    u64 event_tail_draws{};
    u64 event_tail_dispatches{};
    std::optional<std::pair<u64, u64>> last_event_signal;
    u64 storage_bytes{};
    u64 storage_creates{};
    u64 storage_reuses{};
    // With the replay profile: the flush state each span had when it was last made, by guest
    // address and size, and the bytes of spans made since the last report by how that state
    // compares (same, flushed since, new, no flush information).
    std::map<std::pair<u64, u64>, std::pair<u64, u64>> span_flush_states;
    std::array<u64, 4> span_state_bytes{};
    BananzaNvn2KeptStorage kept_storage;
    // Of this replay: the kept buffers its spans use, the ones it filled (flushed before the
    // next submit) and the ones it took out (alive until the GPU is done).
    std::vector<BananzaNvn2KeptStorage::Buffer> used_kept;
    std::vector<std::pair<BananzaNvn2KeptStorage::Buffer, u64>> kept_fills;
    std::vector<BananzaNvn2KeptStorage::Buffer> evicted_kept;
    u64 replay_count{};
    // Host ranges of guest memory whose write-back by a retire job is not done, and the
    // newest job with a write-back whose host range is not known.
    struct PendingWriteback {
        const u8* begin;
        const u8* end;
        u64 sequence;
    };
    std::vector<PendingWriteback> pending_writebacks;
    u64 pending_unknown_writes{};
    u64 kept_reused_bytes{};
    u64 kept_filled_bytes{};
    u64 guest_waits{};
    u64 guest_wait_ns{};
    u64 stream_bytes{};
    u64 stream_uploads{};
    bool precise_queries{};
    std::set<std::tuple<u32, u32, u32>> unsupported_color_formats;
    std::set<std::tuple<std::string_view, GPUVAddr, u64>> traced_images;
    std::set<std::tuple<GPUVAddr, u32, u32>> captured_inputs;
    bool trace_images{};
    bool profile_replay{};
    BananzaNvn2RunStats run_stats;
    bool replay_logged{};
    bool missing_draw_logged{};
    std::set<std::pair<u64, std::string_view>> dispatch_diagnostics;
    bool missing_source_logged{};
    bool image_budget_logged{};

    // Retire thread state, under retire_mutex. Declared last so the thread stops first.
    std::mutex retire_mutex;
    std::condition_variable_any retire_cv;
    std::condition_variable retire_idle_cv;
    std::deque<RetireJob> retire_jobs;
    bool retire_running{};
    // Samples since the last counter reset, as of the last applied query.
    u64 counted_samples{};
    // Reports of queued jobs, by GPU address, that a conditional render may read.
    std::unordered_map<u64, u32> pending_reports;
    std::atomic<u32> pending_counter_jobs{};
    Common::ThreadWorker copy_workers{3, "NvnFill"};
    Common::ThreadWorker retire_workers{3, "NvnWriteBack"};
    std::jthread retire_thread;
    // Declared last: its threads read this object and stop before anything else goes.
    BananzaNvn2AsyncBuilder<std::optional<Pipeline>*, BuiltPipeline> pipeline_builder{
        4, "NvnPipeline"};
};

} // namespace Vulkan
