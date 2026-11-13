// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2024 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/common_types.h"
#include "core/core.h"
#include "core/file_sys/common_funcs.h"
#include "core/file_sys/nca_metadata.h"
#include "core/file_sys/registered_cache.h"
#include "core/hle/service/filesystem/filesystem.h"

namespace ContentManager {

enum class GameVerificationResult {
    Success,
    Failed,
    NotImplemented,
};

/**
 * \brief Removes a single installed DLC
 * \param fs_controller [FileSystemController] reference from the Core::System instance
 * \param title_id Unique title ID representing the DLC which will be removed
 * \return 'true' if successful
 */
inline bool RemoveDLC(const Service::FileSystem::FileSystemController& fs_controller,
                      const u64 title_id) {
    return fs_controller.GetUserNANDContents()->RemoveExistingEntry(title_id) ||
           fs_controller.GetSDMCContents()->RemoveExistingEntry(title_id);
}

/**
 * \brief Removes all DLC for a game
 * \param system Reference to the system instance
 * \param program_id Program ID for the game that will have all of its DLC removed
 * \return Number of DLC removed
 */
inline size_t RemoveAllDLC(Core::System& system, const u64 program_id) {
    size_t count{};
    const auto& fs_controller = system.GetFileSystemController();
    const auto dlc_entries = system.GetContentProvider().ListEntriesFilter(
        FileSys::TitleType::AOC, FileSys::ContentRecordType::Data);
    std::vector<u64> program_dlc_entries;

    for (const auto& entry : dlc_entries) {
        if (FileSys::GetBaseTitleID(entry.title_id) == program_id) {
            program_dlc_entries.push_back(entry.title_id);
        }
    }

    for (const auto& entry : program_dlc_entries) {
        if (RemoveDLC(fs_controller, entry)) {
            ++count;
        }
    }
    return count;
}

/**
 * \brief Removes the installed update for a game
 * \param fs_controller [FileSystemController] reference from the Core::System instance
 * \param program_id Program ID for the game that will have its installed update removed
 * \return 'true' if successful
 */
inline bool RemoveUpdate(const Service::FileSystem::FileSystemController& fs_controller,
                         const u64 program_id) {
    const auto update_id = program_id | 0x800;
    return fs_controller.GetUserNANDContents()->RemoveExistingEntry(update_id) ||
           fs_controller.GetSDMCContents()->RemoveExistingEntry(update_id);
}

/**
 * \brief Removes the base content for a game
 * \param fs_controller [FileSystemController] reference from the Core::System instance
 * \param program_id Program ID for the game that will have its base content removed
 * \return 'true' if successful
 */
inline bool RemoveBaseContent(const Service::FileSystem::FileSystemController& fs_controller,
                              const u64 program_id) {
    return fs_controller.GetUserNANDContents()->RemoveExistingEntry(program_id) ||
           fs_controller.GetSDMCContents()->RemoveExistingEntry(program_id);
}

/**
 * \brief Removes a mod for a game
 * \param fs_controller [FileSystemController] reference from the Core::System instance
 * \param program_id Program ID for the game where [mod_name] will be removed
 * \param mod_name The name of a mod as given by FileSys::PatchManager::GetPatches. This corresponds
 * with the name of the mod's directory in a game's load folder.
 * \return 'true' if successful
 */
inline bool RemoveMod(const Service::FileSystem::FileSystemController& fs_controller,
                      const u64 program_id, const std::string& mod_name) {
    // Check general Mods (LayeredFS and IPS)
    const auto mod_dir = fs_controller.GetModificationLoadRoot(program_id);
    if (mod_dir != nullptr) {
        return mod_dir->DeleteSubdirectoryRecursive(mod_name);
    }

    // Check SDMC mod directory (RomFS LayeredFS)
    const auto sdmc_mod_dir = fs_controller.GetSDMCModificationLoadRoot(program_id);
    if (sdmc_mod_dir != nullptr) {
        return sdmc_mod_dir->DeleteSubdirectoryRecursive(mod_name);
    }

    return false;
}

} // namespace ContentManager
