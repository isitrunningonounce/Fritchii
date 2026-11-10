// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <fstream>
#include <mutex>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;

/// Lichii: the shader objects of every pipeline, made once per distinct stage (SPIR-V, set layout,
/// next stage) and shared, with the driver's binary of each kept on the SSD
/// (lichii_shader_objects.bin, next to the pipeline cache): the next boot makes a shader from its
/// binary in about a millisecond instead of compiling it. A driver or GPU change starts the file
/// over.
class ShaderObjectCache {
public:
    struct Key {
        u64 code_hash;
        u64 layout_hash;
        u32 stage;
        u32 next_stage;
        u32 code_words;
        u32 push_size;

        bool operator==(const Key&) const noexcept = default;
    };

    static ShaderObjectCache& Instance();

    /// Opens (or starts) the binary file for this device. Shaders made before it are not stored.
    void Open(const Device& device, const std::filesystem::path& path);

    /// The shared shader for a SPIR-V create info (codeType SPIR-V): from memory, from the file,
    /// or compiled now (and stored). Null when it cannot be made.
    VkShaderEXT Get(const Device& device, const VkShaderCreateInfoEXT& spirv_ci, u64 layout_hash);

    /// Compiles a stage and stores its binary without keeping the shader (precompiling the game's
    /// translations). False when it was stored already or could not be made.
    bool Precompile(const Device& device, const VkShaderCreateInfoEXT& spirv_ci, u64 layout_hash);

    /// Identifies the driver and GPU the stored binaries are for (changes with a driver update).
    u64 Identity() const noexcept;

    /// The binary file (empty before Open).
    const std::filesystem::path& Path() const noexcept {
        return file_path;
    }

    /// Destroys every shader (the device is about to go).
    void Clear();

private:
    struct KeyHash {
        size_t operator()(const Key& key) const noexcept {
            return static_cast<size_t>(key.code_hash ^ (key.layout_hash * 31) ^ key.stage ^
                                       (static_cast<u64>(key.next_stage) << 32));
        }
    };
    struct Stored {
        u64 offset;
        u32 size;
    };

    bool ReadBinary(const Stored& where, std::vector<u8>& out);
    void WriteBinary(const Key& key, std::span<const u8> binary);

    std::mutex mutex;
    std::unordered_map<Key, VkShaderEXT, KeyHash> shaders;
    const Device* device{};

    std::mutex file_mutex;
    std::fstream file;
    std::unordered_map<Key, Stored, KeyHash> index;
    std::unordered_set<Key, KeyHash> in_flight;
    std::filesystem::path file_path;
    u64 identity{};
    bool file_ok{};

    u64 compiled{};
    u64 loaded{};
    u64 shared{};
};

} // namespace Vulkan
