// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <string>
#include <string_view>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"

namespace Vulkan {

/// Counts how long the runs of draws are that worker threads could resolve without touching
/// shared state, and what ends them (EDEN_NX2_NVN_RUN_STATS=1).
class BananzaNvn2RunStats {
public:
    /// What forces a draw, or the operation before it, onto the GPU thread.
    enum class Cut : u32 {
        Clear,
        Copy,
        BarrierOp,
        BufferUpdate,
        Dispatch,
        Event,
        Counter,
        Indirect,
        Quads,
        PipelineCreate,
        DescriptorSet,
        StorageCreate,
        StorageMove,
        MirrorPage,
        TextureSlow,
        TextureUpload,
        Count,
    };

    /// Work a worker could do itself, counted to know its volume.
    enum class Note : u32 {
        RangeMemo,
        RangeStaged,
        RangeStorage,
        RangeMirror,
        RangeFixed,
        RangeStream,
        TextureHit,
        WritableStorage,
        ReadStorage,
        ScopeBarrier,
        Count,
    };

    enum class ReplayDependency : u32 {
        ConditionalRender,
        NonIndexedIndirect,
        GpuIndexedIndirect,
        ResetIndexedIndirect,
        IndirectResetList,
        Count,
    };

    [[nodiscard]] bool Enabled() const {
        return enabled;
    }

    /// An operation other than a draw that has to run in order on the GPU thread.
    void OpCut(Cut reason) {
        if (enabled) {
            ++cuts[static_cast<size_t>(reason)];
            EndRun();
        }
    }

    /// The draw being resolved needs the GPU thread.
    void DrawCut(Cut reason) {
        if (enabled) {
            ++cuts[static_cast<size_t>(reason)];
            draw_cut = true;
        }
    }

    void Count(Note note) {
        if (enabled) {
            ++notes[static_cast<size_t>(note)];
        }
    }

    void Count(ReplayDependency dependency) {
        if (enabled) {
            ++dependencies[static_cast<size_t>(dependency)];
        }
    }

    void BeginDraw() {
        draw_cut = false;
    }

    void EndDraw() {
        if (!enabled) {
            return;
        }
        ++draws;
        if (draw_cut) {
            ++serial_draws;
            EndRun();
        } else {
            ++run;
        }
    }

    void EndSubmission(u64 present_count) {
        if (!enabled) {
            return;
        }
        EndRun();
        ++submissions;
        const auto now = std::chrono::steady_clock::now();
        if (now < next_report) {
            return;
        }
        Report();
        ReportReplayDependencies(present_count - last_present);
        *this = BananzaNvn2RunStats{};
        last_present = present_count;
        next_report = now + std::chrono::seconds{5};
    }

private:
    static constexpr std::array<std::string_view, static_cast<size_t>(Cut::Count)> CutNames{
        "clear",          "copy",           "barrier op",    "buffer update",
        "dispatch",       "event",          "counter",       "indirect draw",
        "quads",          "pipeline made",  "descriptor set", "storage made",
        "storage moved",  "mirror page",    "texture slow",  "texture upload",
    };
    static constexpr std::array<std::string_view, static_cast<size_t>(Note::Count)> NoteNames{
        "range memo",   "range staged",     "range storage", "range mirror", "range fixed",
        "range stream", "texture hit",      "writable storage", "read storage",
        "scope barrier",
    };
    static constexpr size_t Buckets = 10;

    void EndRun() {
        if (!run) {
            return;
        }
        const size_t bucket = std::min<size_t>(Buckets - 1, std::bit_width(run) - 1);
        ++runs[bucket];
        run_draws[bucket] += run;
        run = 0;
    }

    void Report() const {
        std::string lengths;
        for (size_t i = 0; i < Buckets; ++i) {
            if (runs[i]) {
                lengths += fmt::format(" {}+:{}/{}", u64{1} << i, runs[i], run_draws[i]);
            }
        }
        std::string reasons;
        for (size_t i = 0; i < cuts.size(); ++i) {
            if (cuts[i]) {
                reasons += fmt::format(" {} {},", CutNames[i], cuts[i]);
            }
        }
        std::string volumes;
        for (size_t i = 0; i < notes.size(); ++i) {
            if (notes[i]) {
                volumes += fmt::format(" {} {},", NoteNames[i], notes[i]);
            }
        }
        LOG_INFO(Render_Vulkan,
                 "NVN 2 run stats: {} submissions, {} draws, {} on the GPU thread; runs by "
                 "length (runs/draws):{}; cuts:{} counts:{}",
                 submissions, draws, serial_draws, lengths, reasons, volumes);
    }

    void ReportReplayDependencies(u64 presents) const {
        const auto count = [this](ReplayDependency dependency) {
            return dependencies[static_cast<size_t>(dependency)];
        };
        LOG_INFO(Render_Vulkan,
                 "NVN 2 replay dependencies in {} presents: {} conditional renders, "
                 "{} non-indexed indirect draws, {} GPU-written indexed indirect draws, "
                 "{} reset indexed indirect draws, {} indirect reset lists",
                 presents, count(ReplayDependency::ConditionalRender),
                 count(ReplayDependency::NonIndexedIndirect),
                 count(ReplayDependency::GpuIndexedIndirect),
                 count(ReplayDependency::ResetIndexedIndirect),
                 count(ReplayDependency::IndirectResetList));
    }

    bool enabled{std::getenv("EDEN_NX2_NVN_RUN_STATS") != nullptr};
    bool draw_cut{};
    u64 run{};
    u64 draws{};
    u64 serial_draws{};
    u64 submissions{};
    u64 last_present{};
    std::array<u64, Buckets> runs{};
    std::array<u64, Buckets> run_draws{};
    std::array<u64, static_cast<size_t>(Cut::Count)> cuts{};
    std::array<u64, static_cast<size_t>(Note::Count)> notes{};
    std::array<u64, static_cast<size_t>(ReplayDependency::Count)> dependencies{};
    std::chrono::steady_clock::time_point next_report{};
};

} // namespace Vulkan
