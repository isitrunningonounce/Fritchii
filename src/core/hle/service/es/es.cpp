// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/hle/service/es/es.h"
#include "core/hle/service/ipc_helpers.h"
#include "core/hle/service/server_manager.h"
#include "core/hle/service/service.h"
#include "frontend_common/firmware_manager.h"

namespace Service::ES {

constexpr Result ERROR_INVALID_ARGUMENT{ErrorModule::ETicket, 2};
constexpr Result ERROR_INVALID_RIGHTS_ID{ErrorModule::ETicket, 3};

class ETicket final : public ServiceFramework<ETicket> {
public:
    explicit ETicket(Core::System& system_) : ServiceFramework{system_, "es"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {1, &ETicket::ImportTicket, "ImportTicket"},
            {2, nullptr, "ImportTicketCertificateSet"},
            {3, nullptr, "DeleteTicket"},
            {4, nullptr, "DeletePersonalizedTicket"},
            {5, nullptr, "DeleteAllCommonTicket"},
            {6, nullptr, "DeleteAllPersonalizedTicket"},
            {7, nullptr, "DeleteAllPersonalizedTicketEx"},
            {8, &ETicket::MissingRightsId, "GetTitleKey"},
            {9, &ETicket::EmptyTicketList, "CountCommonTicket"},
            {10, &ETicket::EmptyTicketList, "CountPersonalizedTicket"},
            {11, &ETicket::EmptyTicketList, "ListCommonTicketRightsIds"},
            {12, &ETicket::EmptyTicketList, "ListPersonalizedTicketRightsIds"},
            {13, nullptr, "ListMissingPersonalizedTicket"},
            {14, &ETicket::MissingRightsId, "GetCommonTicketSize"},
            {15, &ETicket::MissingRightsId, "GetPersonalizedTicketSize"},
            {16, &ETicket::MissingRightsId, "GetCommonTicketData"},
            {17, &ETicket::MissingRightsId, "GetPersonalizedTicketData"},
            {18, nullptr, "OwnTicket"},
            {19, nullptr, "GetTicketInfo"},
            {20, nullptr, "ListLightTicketInfo"},
            {21, nullptr, "SignData"},
            {22, nullptr, "GetCommonTicketAndCertificateSize"},
            {23, nullptr, "GetCommonTicketAndCertificateData"},
            {24, nullptr, "ImportPrepurchaseRecord"},
            {25, nullptr, "DeletePrepurchaseRecord"},
            {26, nullptr, "DeleteAllPrepurchaseRecord"},
            {27, nullptr, "CountPrepurchaseRecord"},
            {28, nullptr, "ListPrepurchaseRecordRightsIds"},
            {29, nullptr, "ListPrepurchaseRecordInfo"},
            {30, nullptr, "CountTicket"},
            {31, nullptr, "ListTicketRightsIds"},
            {32, nullptr, "CountPrepurchaseRecordEx"},
            {33, nullptr, "ListPrepurchaseRecordRightsIdsEx"},
            {34, nullptr, "GetEncryptedTicketSize"},
            {35, nullptr, "GetEncryptedTicketData"},
            {36, nullptr, "DeleteAllInactiveELicenseRequiredPersonalizedTicket"},
            {37, nullptr, "OwnTicket2"},
            {38, nullptr, "OwnTicket3"},
            {39, nullptr, "DeleteAllInactivePersonalizedTicket"},
            {40, nullptr, "DeletePrepurchaseRecordByNintendoAccountId"},
            {101, nullptr, "Unknown101"}, //18.0.0+
            {102, nullptr, "Unknown102"}, //18.0.0+
            {103, nullptr, "Unknown103"}, //18.0.0+
            {104, nullptr, "Unknown104"}, //18.0.0+
            {105, nullptr, "Unknown105"}, //20.0.0+
            {201, nullptr, "Unknown201"}, //18.0.0+
            {202, nullptr, "Unknown202"}, //18.0.0+
            {203, nullptr, "Unknown203"}, //18.0.0+
            {204, nullptr, "Unknown204"}, //18.0.0+
            {205, nullptr, "Unknown205"}, //18.0.0+
            {501, nullptr, "Unknown501"},
            {502, nullptr, "Unknown502"},
            {503, nullptr, "GetTitleKey"},
            {504, nullptr, "Unknown504"},
            {508, nullptr, "Unknown508"},
            {509, nullptr, "Unknown509"},
            {510, nullptr, "Unknown510"},
            {511, nullptr, "Unknown511"},
            {1001, nullptr, "Unknown1001"},
            {1002, nullptr, "Unknown1001"},
            {1003, nullptr, "Unknown1003"},
            {1004, nullptr, "Unknown1004"},
            {1005, nullptr, "Unknown1005"},
            {1006, nullptr, "Unknown1006"},
            {1007, nullptr, "Unknown1007"},
            {1009, nullptr, "Unknown1009"},
            {1010, nullptr, "Unknown1010"},
            {1011, nullptr, "Unknown1011"},
            {1012, nullptr, "Unknown1012"},
            {1013, nullptr, "Unknown1013"},
            {1014, nullptr, "Unknown1014"},
            {1015, nullptr, "Unknown1015"},
            {1016, nullptr, "Unknown1016"},
            {1017, nullptr, "Unknown1017"},
            {1018, nullptr, "Unknown1018"},
            {1019, nullptr, "Unknown1019"},
            {1020, nullptr, "Unknown1020"},
            {1021, nullptr, "Unknown1021"},
            {1501, nullptr, "Unknown1501"},
            {1502, nullptr, "Unknown1502"},
            {1503, nullptr, "Unknown1503"},
            {1504, nullptr, "Unknown1504"},
            {1505, nullptr, "Unknown1505"},
            {1506, nullptr, "Unknown1506"},
            {2000, nullptr, "Unknown2000"},
            {2001, nullptr, "Unknown2001"},
            {2002, nullptr, "Unknown2002"},
            {2003, nullptr, "Unknown2003"},
            {2100, nullptr, "Unknown2100"},
            {2501, nullptr, "Unknown2501"},
            {2502, nullptr, "Unknown2502"},
            {2601, nullptr, "Unknown2601"},
            {3001, nullptr, "Unknown3001"},
            {3002, nullptr, "Unknown3002"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }

private:
    void ImportTicket(HLERequestContext& ctx) {
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ERROR_INVALID_ARGUMENT);
    }

    void MissingRightsId(HLERequestContext& ctx) {
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ERROR_INVALID_RIGHTS_ID);
    }

    void EmptyTicketList(HLERequestContext& ctx) {
        IPC::ResponseBuilder rb{ctx, 3};
        rb.Push(ResultSuccess);
        rb.Push<u32>(0);
    }
};

class NDRM_LU final : public ServiceFramework<NDRM_LU> {
public:
    explicit NDRM_LU(Core::System& system_)
        : ServiceFramework{system_, "ndrm:lu"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {1, nullptr, "Cmd1"},
            {2, nullptr, "Cmd2"},
            {3, nullptr, "Cmd3"},
            {1000, nullptr, "Cmd1000"},
            {8000, nullptr, "Cmd8000"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }
};

class NDRM_LA final : public ServiceFramework<NDRM_LA> {
public:
    explicit NDRM_LA(Core::System& system_)
        : ServiceFramework{system_, "ndrm:la"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {1, nullptr, "Cmd1"},
            {2, nullptr, "Cmd2"},
            {3, nullptr, "Cmd3"},
            {4, nullptr, "Cmd4"},
            {5, nullptr, "Cmd5"},
            {6, nullptr, "Cmd6"},
            {7, nullptr, "Cmd7"},
            {8, nullptr, "Cmd8"},
            {9, nullptr, "Cmd9"},
            {10, nullptr, "Cmd10"},
            {11, nullptr, "Cmd11"},
            {12, nullptr, "Cmd12"},
            {13, nullptr, "Cmd13"},
            {14, nullptr, "Cmd14"},
            {15, nullptr, "Cmd15"},
            {16, nullptr, "Cmd16"},
            {17, nullptr, "Cmd17"},
            {18, nullptr, "Cmd18"},
            {19, nullptr, "Cmd19"},
            {20, nullptr, "Cmd20"},
            {21, nullptr, "Cmd21"},
            {22, nullptr, "Cmd22"},
            {23, nullptr, "Cmd23"},
            {24, nullptr, "Cmd24"},
            {25, nullptr, "Cmd25"},
            {26, nullptr, "Cmd26"},
            {27, nullptr, "Cmd27"},
            {28, nullptr, "Cmd28"},
            {29, nullptr, "Cmd29"},
            {30, nullptr, "Cmd30"},
            {31, nullptr, "Cmd31"},
            {32, nullptr, "Cmd32"},
            {33, nullptr, "Cmd33"},
            {34, nullptr, "Cmd34"},
            {35, nullptr, "Cmd35"},
            {36, nullptr, "Cmd36"},
            {37, nullptr, "Cmd37"},
            {38, nullptr, "Cmd38"},
            {39, nullptr, "Cmd39"},
            {40, nullptr, "Cmd40"},
            {42, nullptr, "Cmd42"},
            {43, nullptr, "Cmd43"},
            {44, nullptr, "Cmd44"},
            {45, nullptr, "Cmd45"},
            {46, nullptr, "Cmd46"},
            {47, nullptr, "Cmd47"},
            {48, nullptr, "Cmd48"},
            {49, nullptr, "Cmd49"},
            {50, nullptr, "Cmd50"},
            {51, nullptr, "Cmd51"},
            {8000, nullptr, "Cmd8000"},
            {8001, nullptr, "Cmd8001"},
            {8002, nullptr, "Cmd8002"},
            {8003, nullptr, "Cmd8003"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }
};

void LoopProcess(Core::System& system) {
    auto server_manager = std::make_unique<ServerManager>(system);

    server_manager->RegisterNamedService("es", std::make_shared<ETicket>(system));
    // +13.0.0
    if (FirmwareManager::GetFirmwareVersion(system).first.major >= 13) {
        server_manager->RegisterNamedService("ndrm:lu", std::make_shared<NDRM_LU>(system));
        server_manager->RegisterNamedService("ndrm:la", std::make_shared<NDRM_LA>(system));
    }
    ServerManager::RunServer(std::move(server_manager));
}

} // namespace Service::ES
