// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <vector>
#include <string>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;

vk::ShaderModule BuildShader(const Device& device, std::span<const u32> code);

/// Lichii: the SPIR-V a module was built from, kept while shader objects are on (then dropped).
std::vector<u32> TakeShaderSpirv(VkShaderModule module);

/// Lichii: drops a module's kept SPIR-V when no shader object will be made from it.
void ForgetShaderSpirv(VkShaderModule module);
/// EDEN_NX2_SPIRV_SELFCHECK=1 (off by default, read at the first emitted shader): every emitted
/// shader is scanned with FindSpirvIdProblem before it goes to the driver, and one with a problem
/// is logged and its pipeline dropped (its draws are skipped) instead of reaching the driver.
bool SpirvSelfCheck();

/// A cheap scan of the ids of emitted SPIR-V: an instruction that uses its own result, a value
/// used before its definition (OpPhi aside, whose values may come from a back edge, but those
/// must still be defined somewhere), an id past the bound, or a result defined twice. Only the
/// instructions whose operands are known to be ids are checked, so valid code is never flagged;
/// it is no spirv-val. Empty when nothing was found, else what it found.
[[nodiscard]] std::string FindSpirvIdProblem(std::span<const u32> code);

} // namespace Vulkan
