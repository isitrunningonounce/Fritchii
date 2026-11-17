// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <utility>

#include "core/file_sys/content_archive.h"
#include "core/loader/loader.h"

namespace FileSys {

NCA::NCA(VirtualFile file_, const NCA*, bool) : file(std::move(file_)) {}

NCA::~NCA() = default;

Loader::ResultStatus NCA::GetStatus() const {
    return Loader::ResultStatus::ErrorEncryptedContentUnsupported;
}

std::vector<VirtualFile> NCA::GetFiles() const {
    return {};
}

std::vector<VirtualDir> NCA::GetSubdirectories() const {
    return {};
}

std::string NCA::GetName() const {
    return {};
}

VirtualDir NCA::GetParentDirectory() const {
    return nullptr;
}

NCAContentType NCA::GetType() const {
    return {};
}

u64 NCA::GetTitleId() const {
    return 0;
}

RightsId NCA::GetRightsId() const {
    return {};
}

u32 NCA::GetSDKVersion() const {
    return 0;
}

u8 NCA::GetKeyGeneration() const {
    return 0;
}

bool NCA::IsUpdate() const {
    return false;
}

VirtualFile NCA::GetRomFS() const {
    return nullptr;
}

VirtualDir NCA::GetExeFS() const {
    return nullptr;
}

VirtualFile NCA::GetBaseFile() const {
    return file;
}

VirtualDir NCA::GetLogoPartition() const {
    return nullptr;
}

} // namespace FileSys
