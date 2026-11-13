// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2024 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>

#include "core/hle/service/ssl/cert_store.h"

namespace Service::SSL {

CertStore::CertStore(Core::System&) {}

CertStore::~CertStore() = default;

Result CertStore::GetCertificates(u32* out_num_entries, std::span<u8> out_data,
                                  std::span<const CaCertificateId>) {
    R_UNLESS(out_data.size_bytes() >= sizeof(BuiltInCertificateInfo), ResultUnknown);
    *out_num_entries = 0;

    const BuiltInCertificateInfo terminator{
        .cert_id = CaCertificateId::All,
        .status = TrustedCertStatus::Invalid,
        .der_size = 0,
        .der_offset = 0,
    };
    std::memcpy(out_data.data(), &terminator, sizeof(terminator));
    R_SUCCEED();
}

Result CertStore::GetCertificateBufSize(u32* out_size, u32* out_num_entries,
                                        std::span<const CaCertificateId>) {
    *out_size = sizeof(BuiltInCertificateInfo);
    *out_num_entries = 0;
    R_SUCCEED();
}

} // namespace Service::SSL
