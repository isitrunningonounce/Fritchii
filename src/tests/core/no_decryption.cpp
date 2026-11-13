// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string_view>

#include <catch2/catch_test_macros.hpp>

#include "common/scope_exit.h"
#include "core/core.h"
#include "core/file_sys/content_archive.h"
#include "core/file_sys/nup_writer.h"
#include "core/file_sys/vfs/vfs_real.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/kernel.h"
#include "core/loader/loader.h"

TEST_CASE("Encrypted game containers cannot be loaded", "[no-decryption]") {
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() /
                     ("lichii-no-decryption-" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    REQUIRE(fs::create_directory(dir));
    SCOPE_EXIT {
        std::error_code error;
        fs::remove_all(dir, error);
    };

    Core::System system;
    Kernel::KProcess process{system.Kernel()};
    SCOPE_EXIT { system.Kernel().UnregisterKernelObject(&process); };
    FileSys::RealVfsFilesystem vfs;

    using Loader::FileType;
    constexpr auto unsupported = Loader::ResultStatus::ErrorEncryptedContentUnsupported;
    struct FileCase {
        std::string_view name;
        FileType type;
    };
    constexpr std::array cases{
        FileCase{"game.nsp", FileType::NSP},
        FileCase{"game.xci", FileType::XCI},
        FileCase{"game.nca", FileType::NCA},
        FileCase{"game.NSZ", FileType::NSZ},
        FileCase{"game.xcz", FileType::XCZ},
        FileCase{"game.nax", FileType::NAX},
        FileCase{"pfs", FileType::NSP},
        FileCase{"card", FileType::XCI},
    };
    for (const auto& [name, type] : cases) {
        CAPTURE(name);
        const auto path = dir / name;
        std::array<char, 0x104> bytes{};
        if (name == "card") {
            std::copy_n("HEAD", 4, bytes.begin() + 0x100);
        } else {
            std::copy_n("PFS0", 4, bytes.begin());
        }
        {
            std::ofstream stream(path, std::ios::binary);
            REQUIRE(stream.write(bytes.data(), bytes.size()).good());
        }
        const auto file = vfs.OpenFile(path.string(), FileSys::OpenMode::Read);
        REQUIRE(file);
        CHECK(Loader::IdentifyFile(file) == type);
        const auto loader = Loader::GetLoader(system, file);
        REQUIRE(loader);
        CHECK(loader->GetFileType() == type);
        CHECK(loader->Load(process, system).first == unsupported);
        u64 id{};
        CHECK(loader->ReadProgramId(id) == unsupported);
        if (name == "game.nsp") {
            CHECK_FALSE(FileSys::Nup::IsLooseDump(dir));
        }
        if (type == FileType::NCA) {
            CHECK(FileSys::NCA{file}.GetStatus() == unsupported);
        }
    }
    CHECK(Loader::GetResultStatusString(unsupported) ==
          "Encrypted game files are not supported. Use a loose dump or a NUP file.");
}
