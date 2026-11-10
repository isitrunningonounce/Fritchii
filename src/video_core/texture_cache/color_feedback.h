// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <span>

#include "video_core/texture_cache/types.h"

namespace VideoCommon {

[[nodiscard]] inline bool IsColorRenderTarget(ImageId image, u32 active_mask,
                                              std::span<const ImageId> targets) {
    if (!image || image == NULL_IMAGE_ID) {
        return false;
    }
    for (size_t index = 0; index < targets.size(); ++index) {
        if ((active_mask & (1U << index)) != 0 && image == targets[index]) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] inline Extent3D ColorFeedbackExtent(Extent3D host_size, s32 level) {
    return {(std::max)(host_size.width >> level, 1U), (std::max)(host_size.height >> level, 1U), 1};
}

[[nodiscard]] inline bool CanCopyColorFeedback(bool native, bool color_target, bool color_format,
                                               ImageViewType type, u32 samples,
                                               SubresourceExtent extent) {
    return native && color_target && color_format && type == ImageViewType::e2D && samples == 1 &&
           extent.levels == 1 && extent.layers == 1;
}

} // namespace VideoCommon
