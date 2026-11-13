// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <concepts>
#include <array>
#include "common/concepts.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/string_util.h"
#include "core/file_sys/control_metadata.h"
#include "core/loader/deconstructed_rom_directory.h"
#include "core/loader/kip.h"
#include "core/loader/nro.h"
#include "core/loader/nso.h"
#include "core/loader/nup.h"

namespace Loader {

namespace {

template <std::derived_from<AppLoader> T>
std::optional<FileType> IdentifyFileLoader(FileSys::VirtualFile file) {
    const auto file_type = T::IdentifyType(file);
    if (file_type != FileType::Error) {
        return file_type;
    }
    return std::nullopt;
}

bool IsEncryptedContent(FileType type) {
    switch (type) {
    case FileType::NSP:
    case FileType::NSZ:
    case FileType::XCI:
    case FileType::XCZ:
    case FileType::NCA:
    case FileType::NAX:
        return true;
    default:
        return false;
    }
}

class AppLoader_Unsupported final : public AppLoader {
public:
    AppLoader_Unsupported(FileSys::VirtualFile file, FileType type_)
        : AppLoader(std::move(file)), type{type_} {}

    FileType GetFileType() const override {
        return type;
    }

    LoadResult Load(Kernel::KProcess&, Core::System&) override {
        return {unsupported, std::nullopt};
    }

    ResultStatus VerifyIntegrity(std::function<bool(size_t, size_t)>) override {
        return unsupported;
    }

    ResultStatus ReadCode(std::vector<u8>&) override {
        return unsupported;
    }

    ResultStatus ReadIcon(std::vector<u8>&) override {
        return unsupported;
    }

    ResultStatus ReadBanner(std::vector<u8>&) override {
        return unsupported;
    }

    ResultStatus ReadLogo(std::vector<u8>&) override {
        return unsupported;
    }

    ResultStatus ReadProgramId(u64&) override {
        return unsupported;
    }

    ResultStatus ReadProgramIds(std::vector<u64>&) override {
        return unsupported;
    }

    ResultStatus ReadRomFS(FileSys::VirtualFile&) override {
        return unsupported;
    }

    ResultStatus ReadUpdateRaw(FileSys::VirtualFile&) override {
        return unsupported;
    }

    ResultStatus ReadTitle(std::string&) override {
        return unsupported;
    }

    ResultStatus ReadControlData(FileSys::NACP&) override {
        return unsupported;
    }

    ResultStatus ReadDisplayVersion(std::string&) override {
        return unsupported;
    }

    ResultStatus ReadManualRomFS(FileSys::VirtualFile&) override {
        return unsupported;
    }

    ResultStatus ReadNSOModules(Modules&) override {
        return unsupported;
    }

private:
    static constexpr auto unsupported = ResultStatus::ErrorEncryptedContentUnsupported;
    FileType type;
};

} // namespace

FileType IdentifyFile(FileSys::VirtualFile file) {
    if (!file) {
        return FileType::Unknown;
    }

    const auto name = file->GetName();
    const auto filename_type = GuessFromFilename(name);
    if (IsEncryptedContent(filename_type)) {
        return filename_type;
    }

    if (Common::FS::GetExtensionFromFilename(name).empty()) {
        std::array<char, 4> magic{};
        if (file->ReadObject(&magic) == magic.size() && magic == std::array{'P', 'F', 'S', '0'}) {
            return FileType::NSP;
        }
        if (file->ReadObject(&magic, 0x100) == magic.size() &&
            magic == std::array{'H', 'E', 'A', 'D'}) {
            return FileType::XCI;
        }
    }

    if (const auto nup_type = IdentifyFileLoader<AppLoader_NUP>(file)) {
        return *nup_type;
    } else if (const auto nro_type = IdentifyFileLoader<AppLoader_NRO>(file)) {
        return *nro_type;
    } else if (const auto kip_type = IdentifyFileLoader<AppLoader_KIP>(file)) {
        return *kip_type;
    } else if (const auto nso_type = IdentifyFileLoader<AppLoader_NSO>(file)) {
        return *nso_type;
    } else if (const auto romdir_type =
                   IdentifyFileLoader<AppLoader_DeconstructedRomDirectory>(file)) {
        return *romdir_type;
    }
    return FileType::Unknown;
}

bool IsBootableGameContainer(FileSys::VirtualFile file, FileType type) {
    if (!file) {
        return false;
    }
    if (type == FileType::Unknown) {
        type = IdentifyFile(file);
    }
    if (type != FileType::NUP) {
        return false;
    }

    FileSys::NupContainer nup{file};
    if (nup.GetStatus() != ResultStatus::Success) {
        return false;
    }
    // A standalone update (no base) carries a complete patched tree.
    if (const auto base = nup.FindBasePackage();
        base.has_value() && nup.GetExeFSDir(*base) != nullptr) {
        return true;
    }
    const auto update = nup.FindBestUpdatePackage();
    return update.has_value() && nup.GetExeFSDir(*update) != nullptr;
}

FileType GuessFromFilename(const std::string& name) {
    if (name == "main")
        return FileType::DeconstructedRomDirectory;
    else if (name == "00")
        return FileType::NCA;

    auto const extension =
        Common::ToLower(std::string(Common::FS::GetExtensionFromFilename(name)));
    if (extension == "nro")
        return FileType::NRO;
    else if (extension == "nso")
        return FileType::NSO;
    else if (extension == "nax")
        return FileType::NAX;
    else if (extension == "nca")
        return FileType::NCA;
    else if (extension == "xci")
        return FileType::XCI;
    else if (extension == "xcz")
        return FileType::XCZ;
    else if (extension == "nsp")
        return FileType::NSP;
    else if (extension == "nup")
        return FileType::NUP;
    else if (extension == "nsz")
        return FileType::NSZ;
    else if (extension == "kip")
        return FileType::KIP;
    return FileType::Unknown;
}

std::string GetFileTypeString(FileType type) {
    switch (type) {
    case FileType::NRO:
        return "NRO";
    case FileType::NSO:
        return "NSO";
    case FileType::NCA:
        return "NCA";
    case FileType::XCI:
        return "XCI";
    case FileType::XCZ:
        return "XCZ";
    case FileType::NAX:
        return "NAX";
    case FileType::NSP:
        return "NSP";
    case FileType::NSZ:
        return "NSZ";
    case FileType::NUP:
        return "NUP";
    case FileType::KIP:
        return "KIP";
    case FileType::DeconstructedRomDirectory:
        return "Directory";
    case FileType::Error:
    case FileType::Unknown:
        break;
    }

    return "unknown";
}

constexpr std::array<const char*, 71> RESULT_MESSAGES{
    "The operation completed successfully.",
    "The loader requested to load is already loaded.",
    "The operation is not implemented.",
    "The loader is not initialized properly.",
    "The NPDM file has a bad header.",
    "The NPDM has a bad ACID header.",
    "The NPDM has a bad ACI header,",
    "The NPDM file has a bad file access control.",
    "The NPDM has a bad file access header.",
    "The NPDM has bad kernel capability descriptors.",
    "The PFS/HFS partition has a bad header.",
    "The PFS/HFS partition has incorrect size as determined by the header.",
    "The NCA file has a bad header.",
    "The general keyfile could not be found.",
    "The NCA Header key could not be found.",
    "The NCA Header key is incorrect or the header is invalid.",
    "Support for NCA2-type NCAs is not implemented.",
    "Support for NCA0-type NCAs is not implemented.",
    "The titlekey for this Rights ID could not be found.",
    "The title key encryption key for this crypto revision could not be found.",
    "The Rights ID in the header is invalid.",
    "The key area key for this application type and crypto revision could not be found.",
    "The key area key is incorrect or the section header is invalid.",
    "The title key or its encryption key is incorrect, or the section header is invalid.",
    "The XCI file is missing a Program-type NCA.",
    "The NCA file is not an application.",
    "The Program-type NCA contains no executable. An update may be required.",
    "The XCI file has a bad header.",
    "The XCI file is missing a partition.",
    "The file could not be found or does not exist.",
    "The game is missing a program metadata file (main.npdm).",
    "The game uses the currently-unimplemented 32-bit architecture.",
    "Unable to completely parse the kernel metadata when loading the emulated process",
    "The RomFS could not be found.",
    "The ELF file has incorrect size as determined by the header.",
    "There was a general error loading the NRO into emulated memory.",
    "There was a general error loading the NSO into emulated memory.",
    "There is no icon available.",
    "There is no control data available.",
    "The NAX file has a bad header.",
    "The NAX file has incorrect size as determined by the header.",
    "The HMAC to generated the NAX decryption keys failed.",
    "The HMAC to validate the NAX decryption keys failed.",
    "The NAX key derivation failed.",
    "The NAX file cannot be interpreted as an NCA file.",
    "The NAX file has an incorrect path.",
    "The SD seed could not be found or derived.",
    "The SD KEK Source could not be found.",
    "The AES KEK Generation Source could not be found.",
    "The AES Key Generation Source could not be found.",
    "The SD Save Key Source could not be found.",
    "The SD NCA Key Source could not be found.",
    "The NSP file is missing a Program-type NCA.",
    "The BKTR-type NCA has a bad BKTR header.",
    "The BKTR Subsection entry is not located immediately after the Relocation entry.",
    "The BKTR Subsection entry is not at the end of the media block.",
    "The BKTR-type NCA has a bad Relocation block.",
    "The BKTR-type NCA has a bad Subsection block.",
    "The BKTR-type NCA has a bad Relocation bucket.",
    "The BKTR-type NCA has a bad Subsection bucket.",
    "Game updates cannot be loaded directly. Load the base game instead.",
    "The NSP or XCI does not contain an update in addition to the base game.",
    "The KIP file has a bad header.",
    "The KIP BLZ decompression of the section failed unexpectedly.",
    "The INI file has a bad header.",
    "The INI file contains more than the maximum allowable number of KIP files.",
    "Integrity verification could not be performed for this file.",
    "Integrity verification failed.",
    "The NUP container is malformed or its TOC hash does not match.",
    "The NUP container does not contain a loadable base package.",
    "Encrypted game files are not supported. Use a loose dump or a NUP file.",
};

std::string GetResultStatusString(ResultStatus status) {
    return RESULT_MESSAGES.at(static_cast<std::size_t>(status));
}

std::ostream& operator<<(std::ostream& os, ResultStatus status) {
    os << RESULT_MESSAGES.at(static_cast<std::size_t>(status));
    return os;
}

AppLoader::AppLoader(FileSys::VirtualFile file_) : file(std::move(file_)) {}
AppLoader::~AppLoader() = default;

ResultStatus AppLoader::ReadDisplayVersion(std::string& version) {
    FileSys::NACP control;
    if (const ResultStatus result = ReadControlData(control); result != ResultStatus::Success) {
        return result;
    }
    version = control.GetVersionString();
    return ResultStatus::Success;
}

static std::unique_ptr<AppLoader> GetFileLoader(FileSys::VirtualFile file, FileType type) {
    switch (type) {
    // NX NSO file format.
    case FileType::NSO:
        return std::make_unique<AppLoader_NSO>(std::move(file));

    // NX NRO file format.
    case FileType::NRO:
        return std::make_unique<AppLoader_NRO>(std::move(file));

    case FileType::NCA:
    case FileType::XCI:
    case FileType::XCZ:
    case FileType::NAX:
    case FileType::NSP:
    case FileType::NSZ:
        return std::make_unique<AppLoader_Unsupported>(std::move(file), type);

    // NX KIP (Kernel Internal Process) file format
    case FileType::KIP:
        return std::make_unique<AppLoader_KIP>(std::move(file));

    // NUP (NX Package) container format.
    case FileType::NUP:
        return std::make_unique<AppLoader_NUP>(std::move(file));

    // NX deconstructed ROM directory.
    case FileType::DeconstructedRomDirectory:
        return std::make_unique<AppLoader_DeconstructedRomDirectory>(std::move(file));

    default:
        return nullptr;
    }
}

std::unique_ptr<AppLoader> GetLoader(Core::System&, FileSys::VirtualFile file, u64, std::size_t) {
    if (!file) {
        return nullptr;
    }

    FileType type = IdentifyFile(file);
    const FileType filename_type = GuessFromFilename(file->GetName());

    // A "main" inside an ExeFS directory is also a valid NSO, but loading it alone skips
    // main.npdm, rtld and the sdk/subsdk modules. Load the whole directory instead.
    if (type == FileType::NSO && filename_type == FileType::DeconstructedRomDirectory &&
        IdentifyFileLoader<AppLoader_DeconstructedRomDirectory>(file)) {
        type = FileType::DeconstructedRomDirectory;
    }

    if (type != filename_type && !IsEncryptedContent(type)) {
        LOG_WARNING(Loader, "File {} has a different type ({}) than its extension.",
                    file->GetName(), GetFileTypeString(type));
        if (FileType::Unknown == type) {
            type = filename_type;
        }
    }

    LOG_DEBUG(Loader, "Loading file {} as {}...", file->GetName(), GetFileTypeString(type));

    return GetFileLoader(std::move(file), type);
}

} // namespace Loader
