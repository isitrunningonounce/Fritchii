// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstring>
#include <optional>
#include <unordered_map>

#include "common/common_types.h"
#include "core/arm/bananza_nvn2_resources.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/program_header.h"
#include "video_core/shader_environment.h"
#include "video_core/surface.h"

namespace Vulkan {

/// Answers a compute program's translation from the record of an earlier one (the constants
/// it read, which slots held storage pointers, the type and format of each texture), so the
/// same variant can be built before the game binds anything.
/// A texture handle is not in the record, only where it was read and what the texture was:
/// each such place answers with a number of its own, and that number names the texture.
class BananzaNvn2RecordedBindings final : public Shader::Environment {
public:
    BananzaNvn2RecordedBindings(const Core::BananzaNvn2Hle::ResourceShaderInfo& shader_,
                                const VideoCommon::RecordedEnvQueries& queries, u32 texture_bank_)
        : shader{shader_}, texture_bank{texture_bank_} {
        stage = Shader::Stage::Compute;
        start_address = sizeof(Shader::ProgramHeader);
        for (const auto& [key, value] : queries.cbuf_values) {
            constants.emplace(key, value);
        }
        for (const auto& [key, mapped] : queries.storage_pointer_types) {
            storage_pointers.emplace(key, mapped != 0);
        }
        for (const auto& [handle, type] : queries.texture_types) {
            types.emplace(handle, type);
        }
        for (const auto& [handle, format] : queries.texture_pixel_formats) {
            formats.emplace(handle, format);
        }
        for (const auto& [at, type] : queries.location_texture_types) {
            types.insert_or_assign(HandleAt(at), type);
        }
        for (const auto& [at, format] : queries.location_texture_formats) {
            formats.insert_or_assign(HandleAt(at), format);
        }
        for (const auto& [at, is_integer] : queries.location_integer_formats) {
            integers.insert_or_assign(HandleAt(at), is_integer);
        }
    }

    u64 ReadInstruction(u32) override {
        return 0;
    }
    bool IsNativeProgram() const override {
        return true;
    }
    bool HasHLEMacroState() const override {
        return false;
    }
    u32 ReadViewportTransformState() override {
        return 1;
    }
    u32 TextureBoundBuffer() const override {
        return texture_bank;
    }
    u32 LocalMemorySize() const override {
        return ControlWord(0x68) + ControlWord(0x6c);
    }
    u32 SharedMemorySize() const override {
        return (ControlWord(0x64) + 255) & ~255U;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return {ControlWord(0x58), ControlWord(0x5c), ControlWord(0x60)};
    }
    void Dump(u64, u64) override {}
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    bool IsCbufStorageBufferPointer(u32 bank, u32 offset) override {
        const auto found = storage_pointers.find(Key(bank, offset));
        return found != storage_pointers.end() && found->second;
    }
    bool IsCbufIndirectStorageBufferPointer(u32, u32, u32) override {
        return false;
    }

    u32 ReadCbufValue(u32 bank, u32 offset) override {
        if (const auto found = constants.find(Key(bank, offset)); found != constants.end()) {
            return found->second;
        }
        if (const auto found = stand_ins.find(Key(bank, offset)); found != stand_ins.end()) {
            return found->second;
        }
        if (bank == 1 && offset % 4 == 0 && offset < shader.constant_size) {
            u32 value;
            std::memcpy(&value, shader.code.data() + shader.constant_offset + offset, sizeof(value));
            return value;
        }
        return 0;
    }

    Shader::TextureType ReadTextureType(u32 handle) override {
        const auto found = types.find(handle);
        return found != types.end() ? found->second : Shader::TextureType::Color2D;
    }

    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override {
        const auto found = formats.find(handle);
        return found != formats.end() ? found->second
                                      : Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }

    bool IsTexturePixelFormatInteger(u32 handle) override {
        if (const auto found = integers.find(handle); found != integers.end()) {
            return found->second;
        }
        return VideoCore::Surface::IsPixelFormatInteger(
            static_cast<VideoCore::Surface::PixelFormat>(ReadTexturePixelFormat(handle)));
    }

    // What the pipeline's build reads of an environment.
    const Core::BananzaNvn2Hle::ResourceShaderInfo& shader;
    u32 texture_bank;
    bool missing_texture{};

private:
    static u64 Key(u32 bank, u32 offset) {
        return (static_cast<u64>(bank) << 32) | offset;
    }

    u32 ControlWord(size_t offset) const {
        u32 value{};
        if (offset + sizeof(value) <= shader.control.size()) {
            std::memcpy(&value, shader.control.data() + offset, sizeof(value));
        }
        return value;
    }

    /// The handle the translation computes from a place: from the recorded constants where
    /// the place was also read as a constant, else from a stand-in. Stand-ins are odd and
    /// differ, so two places never give the same handle whatever their shifts.
    u32 HandleAt(const Shader::HandleKey& at) {
        const auto part = [this](u32 bank, u32 offset, bool primary) -> u32 {
            if (const auto found = constants.find(Key(bank, offset)); found != constants.end()) {
                return found->second;
            }
            if (!primary) {
                return 0;
            }
            const auto [stand_in, fresh] = stand_ins.try_emplace(Key(bank, offset), 0);
            if (fresh) {
                stand_in->second = static_cast<u32>(stand_ins.size() * 2 - 1);
            }
            return stand_in->second;
        };
        const u32 first = part(at.index, at.offset, true) << at.shift_left;
        return at.has_secondary
                   ? first | (part(at.sec_index, at.sec_offset, false) << at.sec_shift_left)
                   : first;
    }

    std::unordered_map<u64, u32> constants;
    std::unordered_map<u64, u32> stand_ins;
    std::unordered_map<u64, bool> storage_pointers;
    std::unordered_map<u32, Shader::TextureType> types;
    std::unordered_map<u32, Shader::TexturePixelFormat> formats;
    std::unordered_map<u32, bool> integers;
};

} // namespace Vulkan
