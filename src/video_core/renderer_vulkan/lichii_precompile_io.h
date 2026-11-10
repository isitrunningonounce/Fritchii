// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// LICHII_PRECOMPILE: byte-level writer/reader for the stored archive translations (SPIR-V,
// shader info and the environment answers they were made with).

#pragma once

#include <cstring>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/shader_environment.h"

namespace Vulkan::LichiiIO {

class Writer {
public:
    template <typename T>
    void Put(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        PutBytes(&value, sizeof(T));
    }

    void PutBytes(const void* data, size_t size) {
        const auto* bytes = static_cast<const u8*>(data);
        buffer.insert(buffer.end(), bytes, bytes + size);
    }

    template <typename Container>
    void PutRange(const Container& values) {
        using T = std::remove_cvref_t<decltype(*values.begin())>;
        // Plain data (std::pair of plain fields included): copied as bytes.
        Put(static_cast<u32>(values.size()));
        if (!values.empty()) {
            PutBytes(&*values.begin(), values.size() * sizeof(T));
        }
    }

    std::vector<u8> buffer;
};

class Reader {
public:
    explicit Reader(std::span<const u8> data_) : data{data_} {}

    template <typename T>
    T Get() {
        static_assert(std::is_trivially_copyable_v<T>);
        T value;
        GetBytes(&value, sizeof(T));
        return value;
    }

    void GetBytes(void* out, size_t size) {
        if (size > data.size() - offset) {
            throw std::runtime_error("truncated precompile store");
        }
        std::memcpy(out, data.data() + offset, size);
        offset += size;
    }

    /// Reads a count and that many elements into a container with resize() and data().
    template <typename Container>
    void GetRange(Container& values, u32 limit = 1u << 24) {
        using T = std::remove_cvref_t<decltype(*values.begin())>;
        const u32 count = Get<u32>();
        if (count > limit) {
            throw std::runtime_error("bad count in precompile store");
        }
        values.resize(count);
        if (count != 0) {
            GetBytes(&*values.begin(), count * sizeof(T));
        }
    }

    [[nodiscard]] bool AtEnd() const noexcept {
        return offset == data.size();
    }

    [[nodiscard]] size_t Offset() const noexcept {
        return offset;
    }

    /// Bytes [from, current position).
    [[nodiscard]] std::span<const u8> Since(size_t from) const {
        return data.subspan(from, offset - from);
    }

    void Skip(size_t size) {
        if (size > data.size() - offset) {
            throw std::runtime_error("truncated precompile store");
        }
        offset += size;
    }

    /// Skips what GetRange would read for elements of type T.
    template <typename T>
    void SkipRange(u32 limit = 1u << 24) {
        const u32 count = Get<u32>();
        if (count > limit) {
            throw std::runtime_error("bad count in precompile store");
        }
        Skip(size_t{count} * sizeof(T));
    }

private:
    std::span<const u8> data;
    size_t offset = 0;
};

/// Where Shader::Info's leading plain fields are: [0, map) and [after map, plain); the
/// legacy_stores_mapping std::map between them is written as its entries.
struct InfoLayout {
    size_t map_begin;
    size_t map_end;
    size_t plain;
};

inline InfoLayout LayoutOf(const Shader::Info& info) {
    const auto* base = reinterpret_cast<const u8*>(&info);
    const auto* map = reinterpret_cast<const u8*>(&info.legacy_stores_mapping);
    return {
        .map_begin = static_cast<size_t>(map - base),
        .map_end = static_cast<size_t>(map - base) + sizeof(info.legacy_stores_mapping),
        .plain = static_cast<size_t>(
            reinterpret_cast<const u8*>(&info.constant_buffer_descriptors) - base),
    };
}

/// Shader::Info: its leading plain fields as bytes, then the attribute map and each list.
inline void PutInfo(Writer& w, const Shader::Info& info) {
    const auto* base = reinterpret_cast<const u8*>(&info);
    const InfoLayout layout = LayoutOf(info);
    w.Put(static_cast<u32>(layout.plain));
    w.Put(static_cast<u32>(layout.map_begin));
    w.PutBytes(base, layout.map_begin);
    w.PutBytes(base + layout.map_end, layout.plain - layout.map_end);
    const std::vector<std::pair<Shader::IR::Attribute, Shader::IR::Attribute>> mapping(
        info.legacy_stores_mapping.begin(), info.legacy_stores_mapping.end());
    w.PutRange(mapping);
    w.PutRange(info.constant_buffer_descriptors);
    w.PutRange(info.storage_buffers_descriptors);
    w.PutRange(info.texture_buffer_descriptors);
    w.PutRange(info.image_buffer_descriptors);
    w.PutRange(info.texture_descriptors);
    w.Put(static_cast<u32>(info.handle_programs.size()));
    for (const auto& program : info.handle_programs) {
        w.PutRange(program);
    }
    w.PutRange(info.image_descriptors);
}

inline Shader::Info GetInfo(Reader& r) {
    Shader::Info info{};
    auto* base = reinterpret_cast<u8*>(&info);
    const InfoLayout layout = LayoutOf(info);
    const u32 plain = r.Get<u32>();
    const u32 map_begin = r.Get<u32>();
    if (plain != layout.plain || map_begin != layout.map_begin) {
        throw std::runtime_error("shader info layout changed");
    }
    r.GetBytes(base, layout.map_begin);
    r.GetBytes(base + layout.map_end, layout.plain - layout.map_end);
    std::vector<std::pair<Shader::IR::Attribute, Shader::IR::Attribute>> mapping;
    r.GetRange(mapping, 4096);
    info.legacy_stores_mapping.insert(mapping.begin(), mapping.end());
    r.GetRange(info.constant_buffer_descriptors, Shader::Info::MAX_CBUFS);
    r.GetRange(info.storage_buffers_descriptors, Shader::Info::MAX_SSBOS);
    r.GetRange(info.texture_buffer_descriptors);
    r.GetRange(info.image_buffer_descriptors);
    r.GetRange(info.texture_descriptors);
    const u32 programs = r.Get<u32>();
    if (programs > 4096) {
        throw std::runtime_error("bad handle program count");
    }
    info.handle_programs.resize(programs);
    for (auto& program : info.handle_programs) {
        r.GetRange(program);
    }
    r.GetRange(info.image_descriptors);
    return info;
}

/// Moves past a PutInfo serialization without building the Shader::Info (kept packed).
inline void SkipInfo(Reader& r) {
    static const InfoLayout layout = [] {
        const auto info = std::make_unique<Shader::Info>();
        return LayoutOf(*info);
    }();
    const u32 plain = r.Get<u32>();
    const u32 map_begin = r.Get<u32>();
    if (plain != layout.plain || map_begin != layout.map_begin) {
        throw std::runtime_error("shader info layout changed");
    }
    r.Skip(layout.map_begin);
    r.Skip(layout.plain - layout.map_end);
    r.SkipRange<std::pair<Shader::IR::Attribute, Shader::IR::Attribute>>(4096);
    using Info = Shader::Info;
    r.SkipRange<typename decltype(Info::constant_buffer_descriptors)::value_type>(Info::MAX_CBUFS);
    r.SkipRange<typename decltype(Info::storage_buffers_descriptors)::value_type>(Info::MAX_SSBOS);
    r.SkipRange<typename decltype(Info::texture_buffer_descriptors)::value_type>();
    r.SkipRange<typename decltype(Info::image_buffer_descriptors)::value_type>();
    r.SkipRange<typename decltype(Info::texture_descriptors)::value_type>();
    const u32 programs = r.Get<u32>();
    if (programs > 4096) {
        throw std::runtime_error("bad handle program count");
    }
    for (u32 i = 0; i < programs; ++i) {
        r.SkipRange<u32>();
    }
    r.SkipRange<typename decltype(Info::image_descriptors)::value_type>();
}

inline void PutQueries(Writer& w, const VideoCommon::RecordedEnvQueries& q) {
    w.PutRange(q.texture_types);
    w.PutRange(q.texture_pixel_formats);
    w.PutRange(q.cbuf_values);
    w.PutRange(q.storage_pointer_types);
    w.PutRange(q.cbuf_replacements);
    w.PutRange(q.location_texture_types);
    w.PutRange(q.location_texture_formats);
    w.PutRange(q.location_integer_formats);
    w.Put(q.texture_bound);
    w.Put(q.viewport_transform_state);
    w.Put(q.has_hle_engine_state);
    w.Put(q.has_unbound_instructions);
}

inline VideoCommon::RecordedEnvQueries GetQueries(Reader& r) {
    VideoCommon::RecordedEnvQueries q;
    r.GetRange(q.texture_types);
    r.GetRange(q.texture_pixel_formats);
    r.GetRange(q.cbuf_values);
    r.GetRange(q.storage_pointer_types);
    r.GetRange(q.cbuf_replacements);
    r.GetRange(q.location_texture_types);
    r.GetRange(q.location_texture_formats);
    r.GetRange(q.location_integer_formats);
    q.texture_bound = r.Get<u32>();
    q.viewport_transform_state = r.Get<u32>();
    q.has_hle_engine_state = r.Get<bool>();
    q.has_unbound_instructions = r.Get<bool>();
    return q;
}

} // namespace Vulkan::LichiiIO
