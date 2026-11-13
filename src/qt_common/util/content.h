// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <QObject>
#include "frontend_common/data_manager.h"

namespace QtCommon::Content {

void VerifyNupIntegrity(const std::string& game_path);

void ClearDataDir(FrontendCommon::DataManager::DataDir dir, const std::string& user_id = "");
void ExportDataDir(FrontendCommon::DataManager::DataDir dir, const std::string& user_id = "",
                   const QString& name = QStringLiteral("export"),
                   std::function<void()> callback = {});
void ImportDataDir(FrontendCommon::DataManager::DataDir dir, const std::string& user_id = "",
                   std::function<void()> callback = {});

// Loader //
void configureFilesystemProvider(const std::string& filepath);

// Profiles //
void FixProfiles();
} // namespace QtCommon::Content
