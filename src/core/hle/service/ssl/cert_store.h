// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2024 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>

#include "core/hle/result.h"
#include "core/hle/service/ssl/ssl_types.h"

namespace Core {
class System;
}

namespace Service::SSL {

class CertStore {
public:
    explicit CertStore(Core::System& system);
    ~CertStore();

    Result GetCertificates(u32* out_num_entries, std::span<u8> out_data,
                           std::span<const CaCertificateId> certificate_ids);
    Result GetCertificateBufSize(u32* out_size, u32* out_num_entries,
                                 std::span<const CaCertificateId> certificate_ids);

};

} // namespace Service::SSL
