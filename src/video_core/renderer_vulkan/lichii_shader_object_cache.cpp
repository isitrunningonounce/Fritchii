// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <optional>

#include "common/cityhash.h"
#include "common/logging.h"
#include "video_core/renderer_vulkan/lichii_shader_object_cache.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

namespace {
constexpr u32 Magic = 0x314F534C; // "LSO1"
constexpr u32 Version = 1;

struct Header {
    u32 magic;
    u32 version;
    std::array<u8, VK_UUID_SIZE> binary_uuid;
    u32 binary_version;
    u32 driver_version;
    u32 device_id;
    u32 vendor_id;
};

Header MakeHeader(const Device& device) {
    VkPhysicalDeviceShaderObjectPropertiesEXT shader_object{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_PROPERTIES_EXT,
        .pNext = nullptr,
        .shaderBinaryUUID = {},
        .shaderBinaryVersion = 0,
    };
    VkPhysicalDeviceProperties2 properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &shader_object,
        .properties = {},
    };
    device.GetPhysical().GetProperties2(properties);
    Header header{};
    header.magic = Magic;
    header.version = Version;
    std::memcpy(header.binary_uuid.data(), shader_object.shaderBinaryUUID, VK_UUID_SIZE);
    header.binary_version = shader_object.shaderBinaryVersion;
    header.driver_version = properties.properties.driverVersion;
    header.device_id = properties.properties.deviceID;
    header.vendor_id = properties.properties.vendorID;
    return header;
}
} // Anonymous namespace

ShaderObjectCache& ShaderObjectCache::Instance() {
    static ShaderObjectCache cache;
    return cache;
}

void ShaderObjectCache::Open(const Device& device_, const std::filesystem::path& path) {
    std::scoped_lock lock{file_mutex};
    device = &device_;
    if (file.is_open()) {
        file.close();
    }
    index.clear();
    file_ok = false;
    const Header expected = MakeHeader(device_);
    // Read what an earlier boot stored, if it was made by this driver and GPU.
    std::error_code error;
    bool reuse = false;
    u64 good_end = sizeof(Header);
    if (std::filesystem::exists(path, error)) {
        std::ifstream in{path, std::ios::binary};
        Header header{};
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
        reuse = in && std::memcmp(&header, &expected, sizeof(Header)) == 0;
        if (reuse) {
            const u64 file_size = std::filesystem::file_size(path, error);
            u64 offset = sizeof(Header);
            while (offset + sizeof(Key) + sizeof(u32) <= file_size) {
                Key key{};
                u32 size{};
                in.seekg(static_cast<std::streamoff>(offset));
                if (!in.read(reinterpret_cast<char*>(&key), sizeof(key)) ||
                    !in.read(reinterpret_cast<char*>(&size), sizeof(size)) || size == 0 ||
                    size > (64U << 20)) {
                    break;
                }
                const u64 data = offset + sizeof(key) + sizeof(size);
                if (data + size > file_size) {
                    break;
                }
                index[key] = Stored{data, size};
                offset = data + size;
                good_end = offset;
            }
        }
    }
    if (reuse) {
        // A record cut short by a crash is dropped; new ones go after the last whole one.
        std::filesystem::resize_file(path, good_end, error);
    } else {
        std::ofstream out{path, std::ios::binary | std::ios::trunc};
        out.write(reinterpret_cast<const char*>(&expected), sizeof(expected));
        index.clear();
    }
    file.open(path, std::ios::binary | std::ios::in | std::ios::out);
    file_ok = file.is_open();
    file_path = path;
    identity = Common::CityHash64(reinterpret_cast<const char*>(&expected), sizeof(expected));
    LOG_INFO(Render_Vulkan, "LICHII_SHADER_OBJECTS: {} stored shader binaries in {}{}",
             index.size(), path.filename().string(),
             reuse ? "" : " (new file: first boot, driver or GPU changed)");
}

bool ShaderObjectCache::ReadBinary(const Stored& where, std::vector<u8>& out) {
    std::scoped_lock lock{file_mutex};
    if (!file_ok) {
        return false;
    }
    out.resize(where.size);
    file.clear();
    file.seekg(static_cast<std::streamoff>(where.offset));
    file.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(where.size));
    return static_cast<bool>(file);
}

void ShaderObjectCache::WriteBinary(const Key& key, std::span<const u8> binary) {
    std::scoped_lock lock{file_mutex};
    if (!file_ok || index.contains(key)) {
        return;
    }
    file.clear();
    file.seekp(0, std::ios::end);
    const u64 offset = static_cast<u64>(file.tellp());
    const u32 size = static_cast<u32>(binary.size());
    file.write(reinterpret_cast<const char*>(&key), sizeof(key));
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    file.write(reinterpret_cast<const char*>(binary.data()),
               static_cast<std::streamsize>(binary.size()));
    file.flush();
    if (file) {
        index[key] = Stored{offset + sizeof(key) + sizeof(size), size};
    } else {
        file_ok = false;
    }
}

VkShaderEXT ShaderObjectCache::Get(const Device& device_, const VkShaderCreateInfoEXT& spirv_ci,
                                   u64 layout_hash) {
    Key key{};
    key.code_hash = Common::CityHash64(static_cast<const char*>(spirv_ci.pCode), spirv_ci.codeSize);
    key.layout_hash = layout_hash;
    key.stage = static_cast<u32>(spirv_ci.stage);
    key.next_stage = static_cast<u32>(spirv_ci.nextStage);
    key.code_words = static_cast<u32>(spirv_ci.codeSize / sizeof(u32));
    key.push_size = spirv_ci.pushConstantRangeCount != 0 ? spirv_ci.pPushConstantRanges[0].size : 0;
    {
        std::scoped_lock lock{mutex};
        if (const auto it = shaders.find(key); it != shaders.end()) {
            ++shared;
            return it->second;
        }
    }
    const vk::Device& logical = device_.GetLogical();
    VkShaderEXT shader{};
    bool from_file = false;
    std::optional<Stored> stored;
    {
        std::scoped_lock lock{file_mutex};
        if (const auto it = index.find(key); it != index.end()) {
            stored = it->second;
        }
    }
    if (stored) {
        std::vector<u8> binary;
        if (ReadBinary(*stored, binary)) {
            VkShaderCreateInfoEXT binary_ci = spirv_ci;
            binary_ci.codeType = VK_SHADER_CODE_TYPE_BINARY_EXT;
            binary_ci.codeSize = binary.size();
            binary_ci.pCode = binary.data();
            from_file = logical.CreateShaderEXT(binary_ci, &shader) == VK_SUCCESS;
        }
    }
    if (!from_file) {
        if (logical.CreateShaderEXT(spirv_ci, &shader) != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }
        size_t size = 0;
        if (logical.GetShaderBinaryDataEXT(shader, &size, nullptr) == VK_SUCCESS && size != 0) {
            std::vector<u8> binary(size);
            if (logical.GetShaderBinaryDataEXT(shader, &size, binary.data()) == VK_SUCCESS) {
                WriteBinary(key, binary);
            }
        }
    }
    std::scoped_lock lock{mutex};
    const auto [it, inserted] = shaders.try_emplace(key, shader);
    if (!inserted) {
        // Another thread made the same one meanwhile.
        logical.DestroyShaderEXT(shader);
        ++shared;
        return it->second;
    }
    from_file ? ++loaded : ++compiled;
    const u64 total = loaded + compiled;
    if (total <= 3 || total % 1000 == 0) {
        LOG_INFO(Render_Vulkan,
                 "LICHII_SHADER_OBJECTS: {} shaders: {} from stored binaries, {} compiled, {} "
                 "uses shared",
                 total, loaded, compiled, shared);
    }
    return shader;
}

bool ShaderObjectCache::Precompile(const Device& device_, const VkShaderCreateInfoEXT& spirv_ci,
                                   u64 layout_hash) {
    Key key{};
    key.code_hash = Common::CityHash64(static_cast<const char*>(spirv_ci.pCode), spirv_ci.codeSize);
    key.layout_hash = layout_hash;
    key.stage = static_cast<u32>(spirv_ci.stage);
    key.next_stage = static_cast<u32>(spirv_ci.nextStage);
    key.code_words = static_cast<u32>(spirv_ci.codeSize / sizeof(u32));
    key.push_size = spirv_ci.pushConstantRangeCount != 0 ? spirv_ci.pPushConstantRanges[0].size : 0;
    {
        // Stored already, or another worker is on it.
        std::scoped_lock lock{file_mutex};
        if (!file_ok || index.contains(key) || !in_flight.insert(key).second) {
            return false;
        }
    }
    const vk::Device& logical = device_.GetLogical();
    VkShaderEXT shader{};
    bool stored = false;
    if (logical.CreateShaderEXT(spirv_ci, &shader) == VK_SUCCESS) {
        size_t size = 0;
        if (logical.GetShaderBinaryDataEXT(shader, &size, nullptr) == VK_SUCCESS && size != 0) {
            std::vector<u8> binary(size);
            if (logical.GetShaderBinaryDataEXT(shader, &size, binary.data()) == VK_SUCCESS) {
                WriteBinary(key, binary);
                stored = true;
            }
        }
        // Only the binary is kept: tens of thousands of live shaders would fill the GPU's memory.
        logical.DestroyShaderEXT(shader);
    }
    std::scoped_lock lock{file_mutex};
    in_flight.erase(key);
    return stored;
}

u64 ShaderObjectCache::Identity() const noexcept {
    return identity;
}

void ShaderObjectCache::Clear() {
    std::scoped_lock lock{mutex};
    if (device != nullptr) {
        for (const auto& [key, shader] : shaders) {
            device->GetLogical().DestroyShaderEXT(shader);
        }
    }
    shaders.clear();
}

} // namespace Vulkan
