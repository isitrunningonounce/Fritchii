// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <limits>
#include <random>
#include <openssl/evp.h>
#include "common/assert.h"
#include "common/fs/path_util.h"
#include "common/hex_util.h"
#include "common/logging.h"
#include "common/random.h"
#include "common/string_util.h"
#include "core/file_sys/common_funcs.h"
#include "core/file_sys/content_archive.h"
#include "core/file_sys/nca_metadata.h"
#include "core/file_sys/registered_cache.h"
#include "core/file_sys/vfs/vfs_concat.h"
#include "core/loader/loader.h"

namespace FileSys {

std::string ContentProviderEntry::DebugInfo() const {
    return fmt::format("title_id={:016X}, content_type={:02X}", title_id, static_cast<u8>(type));
}

bool operator<(const ContentProviderEntry& lhs, const ContentProviderEntry& rhs) {
    return (lhs.title_id < rhs.title_id) || (lhs.title_id == rhs.title_id && lhs.type < rhs.type);
}

bool operator==(const ContentProviderEntry& lhs, const ContentProviderEntry& rhs) {
    return std::tie(lhs.title_id, lhs.type) == std::tie(rhs.title_id, rhs.type);
}

bool operator!=(const ContentProviderEntry& lhs, const ContentProviderEntry& rhs) {
    return !operator==(lhs, rhs);
}

static std::string GetRelativePathFromNcaID(const std::array<u8, 16>& nca_id, bool second_hex_upper,
                                            bool within_two_digit, bool cnmt_suffix) {
    const auto nca_str = Common::HexToString(nca_id, second_hex_upper);

    if (!within_two_digit) {
        const auto format_str = fmt::runtime(cnmt_suffix ? "{}.cnmt.nca" : "/{}.nca");
        return fmt::format(format_str, nca_str);
    }

    std::array<u8, 32> hash{};
    u32 hash_len = 0;
    EVP_Digest(nca_id.data(), nca_id.size(), hash.data(), &hash_len, EVP_sha256(), nullptr);

    const auto format_str =
        fmt::runtime(cnmt_suffix ? "/000000{:02X}/{}.cnmt.nca" : "/000000{:02X}/{}.nca");

    LOG_DEBUG(Loader, "Decoded {} bytes, nca id {}", hash_len, nca_str);

    return fmt::format(format_str, hash[0], nca_str);
}

static std::string GetCNMTName(TitleType type, u64 title_id) {
    static constexpr std::array<const char*, 9> TITLE_TYPE_NAMES{
        "SystemProgram",
        "SystemData",
        "SystemUpdate",
        "BootImagePackage",
        "BootImagePackageSafe",
        "Application",
        "Patch",
        "AddOnContent",
        "" ///< Currently unknown 'DeltaTitle'
    };

    auto index = static_cast<std::size_t>(type);
    // If the index is after the jump in TitleType, subtract it out.
    if (index >= static_cast<std::size_t>(TitleType::Application)) {
        index -= static_cast<std::size_t>(TitleType::Application) -
                 static_cast<std::size_t>(TitleType::FirmwarePackageB);
    }
    return fmt::format("{}_{:016x}.cnmt", TITLE_TYPE_NAMES[index], title_id);
}

ContentRecordType GetCRTypeFromNCAType(NCAContentType type) {
    switch (type) {
    case NCAContentType::Program:
        // TODO(DarkLordZach): Differentiate between Program and Patch
        return ContentRecordType::Program;
    case NCAContentType::Meta:
        return ContentRecordType::Meta;
    case NCAContentType::Control:
        return ContentRecordType::Control;
    case NCAContentType::Data:
    case NCAContentType::PublicData:
        return ContentRecordType::Data;
    case NCAContentType::Manual:
        // TODO(DarkLordZach): Peek at NCA contents to differentiate Manual and Legal.
        return ContentRecordType::HtmlDocument;
    default:
        ASSERT_MSG(false, "Invalid NCAContentType={:02X}", type);
        return ContentRecordType{};
    }
}

ContentProvider::~ContentProvider() = default;

bool ContentProvider::HasEntry(ContentProviderEntry entry) const {
    return HasEntry(entry.title_id, entry.type);
}

VirtualFile ContentProvider::GetEntryUnparsed(ContentProviderEntry entry) const {
    return GetEntryUnparsed(entry.title_id, entry.type);
}

VirtualFile ContentProvider::GetEntryRaw(ContentProviderEntry entry) const {
    return GetEntryRaw(entry.title_id, entry.type);
}

std::unique_ptr<NCA> ContentProvider::GetEntry(ContentProviderEntry entry) const {
    return GetEntry(entry.title_id, entry.type);
}

std::vector<ContentProviderEntry> ContentProvider::ListEntries() const {
    return ListEntriesFilter(std::nullopt, std::nullopt, std::nullopt);
}

std::optional<u64> ContentProvider::GetParentApplicationId(u64 program_id) const {
    const auto application_id = GetBaseTitleID(program_id);
    const auto program_index = program_id - application_id;
    if (program_index == 0 || program_index > std::numeric_limits<u8>::max()) {
        return std::nullopt;
    }
    if (!ListEntriesFilter(TitleType::Application, ContentRecordType::Meta, program_id).empty()) {
        return std::nullopt;
    }
    if ((!ListEntriesFilter(TitleType::Application, ContentRecordType::Meta, application_id).empty() && HasEntry(program_id, ContentRecordType::Program))
     || (!ListEntriesFilter(TitleType::Update, ContentRecordType::Meta, GetUpdateTitleID(application_id)).empty() && HasEntry(GetUpdateTitleID(program_id), ContentRecordType::Program))) {
        return application_id;
    }
    return std::nullopt;
}

PlaceholderCache::PlaceholderCache(VirtualDir dir_) : dir(std::move(dir_)) {}

bool PlaceholderCache::Create(const NcaID& id, u64 size) const {
    const auto path = GetRelativePathFromNcaID(id, false, true, false);

    if (dir->GetFileRelative(path) != nullptr) {
        return false;
    }

    std::array<u8, 32> hash{};
    u32 hash_len = 0;
    EVP_Digest(id.data(), id.size(), hash.data(), &hash_len, EVP_sha256(), nullptr);

    LOG_DEBUG(Loader, "Decoded {} bytes, nca id {}", hash_len, id);

    const auto dirname = fmt::format("000000{:02X}", hash[0]);

    const auto dir2 = GetOrCreateDirectoryRelative(dir, dirname);

    if (dir2 == nullptr)
        return false;

    const auto file = dir2->CreateFile(fmt::format("{}.nca", Common::HexToString(id, false)));

    if (file == nullptr)
        return false;

    return file->Resize(size);
}

bool PlaceholderCache::Delete(const NcaID& id) const {
    const auto path = GetRelativePathFromNcaID(id, false, true, false);

    if (dir->GetFileRelative(path) == nullptr) {
        return false;
    }

    std::array<u8, 32> hash{};
    u32 hash_len = 0;
    EVP_Digest(id.data(), id.size(), hash.data(), &hash_len, EVP_sha256(), nullptr);

    LOG_DEBUG(Loader, "Decoded {} bytes, nca id {}", hash_len, id);

    const auto dirname = fmt::format("000000{:02X}", hash[0]);

    const auto dir2 = GetOrCreateDirectoryRelative(dir, dirname);

    const auto res = dir2->DeleteFile(fmt::format("{}.nca", Common::HexToString(id, false)));

    return res;
}

bool PlaceholderCache::Exists(const NcaID& id) const {
    const auto path = GetRelativePathFromNcaID(id, false, true, false);

    return dir->GetFileRelative(path) != nullptr;
}

bool PlaceholderCache::Write(const NcaID& id, u64 offset, const std::vector<u8>& data) const {
    const auto path = GetRelativePathFromNcaID(id, false, true, false);
    const auto file = dir->GetFileRelative(path);

    if (file == nullptr)
        return false;

    return file->WriteBytes(data, offset) == data.size();
}

bool PlaceholderCache::Register(RegisteredCache*, const NcaID&, const NcaID&) const {
    return false;
}

bool PlaceholderCache::CleanAll() const {
    return dir->GetParentDirectory()->CleanSubdirectoryRecursive(dir->GetName());
}

std::optional<std::array<u8, 0x10>> PlaceholderCache::GetRightsID(const NcaID& id) const {
    const auto path = GetRelativePathFromNcaID(id, false, true, false);
    const auto file = dir->GetFileRelative(path);

    if (file == nullptr)
        return std::nullopt;

    NCA nca{file};

    if (nca.GetStatus() != Loader::ResultStatus::Success &&
        nca.GetStatus() != Loader::ResultStatus::ErrorMissingBKTRBaseRomFS) {
        return std::nullopt;
    }

    const auto rights_id = nca.GetRightsId();
    if (rights_id == NcaID{})
        return std::nullopt;

    return rights_id;
}

u64 PlaceholderCache::Size(const NcaID& id) const {
    const auto path = GetRelativePathFromNcaID(id, false, true, false);
    const auto file = dir->GetFileRelative(path);

    if (file == nullptr)
        return 0;

    return file->GetSize();
}

bool PlaceholderCache::SetSize(const NcaID& id, u64 new_size) const {
    const auto path = GetRelativePathFromNcaID(id, false, true, false);
    const auto file = dir->GetFileRelative(path);

    if (file == nullptr)
        return false;

    return file->Resize(new_size);
}

std::vector<NcaID> PlaceholderCache::List() const {
    std::vector<NcaID> out;
    for (const auto& sdir : dir->GetSubdirectories()) {
        for (const auto& file : sdir->GetFiles()) {
            const auto name = file->GetName();
            if (name.length() == 36 && name.ends_with(".nca")) {
                out.push_back(Common::HexStringToArray<0x10>(name.substr(0, 32)));
            }
        }
    }
    return out;
}

NcaID PlaceholderCache::Generate() {
    auto gen = Common::Random::GetMT19937();
    std::uniform_int_distribution<u64> distribution(1, (std::numeric_limits<u64>::max)());
    NcaID out{};
    const auto v1 = distribution(gen);
    const auto v2 = distribution(gen);
    std::memcpy(out.data(), &v1, sizeof(u64));
    std::memcpy(out.data() + sizeof(u64), &v2, sizeof(u64));
    return out;
}

VirtualFile RegisteredCache::OpenFileOrDirectoryConcat(const VirtualDir& open_dir,
                                                       std::string_view path) const {
    if (open_dir == nullptr) {
        return nullptr;
    }
    const auto file = open_dir->GetFileRelative(path);
    if (file != nullptr) {
        return file;
    }

    const auto nca_dir = open_dir->GetDirectoryRelative(path);
    if (nca_dir == nullptr) {
        return nullptr;
    }

    const auto files = nca_dir->GetFiles();
    if (files.size() == 1 && files[0]->GetName() == "00") {
        return files[0];
    }

    std::vector<VirtualFile> concat;
    // Since the files are a two-digit hex number, max is FF.
    for (std::size_t i = 0; i < 0x100; ++i) {
        auto next = nca_dir->GetFile(fmt::format("{:02X}", i));
        if (next != nullptr) {
            concat.push_back(std::move(next));
        } else {
            next = nca_dir->GetFile(fmt::format("{:02x}", i));
            if (next != nullptr) {
                concat.push_back(std::move(next));
            } else {
                break;
            }
        }
    }

    if (concat.empty()) {
        return nullptr;
    }

    auto name = concat.front()->GetName();
    return ConcatenatedVfsFile::MakeConcatenatedFile(std::move(name), std::move(concat));
}

VirtualFile RegisteredCache::GetFileAtID(NcaID id) const {
    VirtualFile file;
    // Try all five relevant modes of file storage:
    // (bit 2 = uppercase/lower, bit 1 = within a two-digit dir, bit 0 = .cnmt suffix)
    // 000: /000000**/{:032X}.nca
    // 010: /{:032X}.nca
    // 100: /000000**/{:032x}.nca
    // 110: /{:032x}.nca
    // 111: /{:032x}.cnmt.nca
    for (u8 i = 0; i < 8; ++i) {
        if ((i % 2) == 1 && i != 7)
            continue;
        const auto path =
            GetRelativePathFromNcaID(id, (i & 0b100) == 0, (i & 0b010) == 0, (i & 0b001) == 0b001);
        file = OpenFileOrDirectoryConcat(dir, path);
        if (file != nullptr)
            return file;
    }
    return file;
}

static std::optional<NcaID> CheckMapForContentRecord(const ::Common::unordered_map<u64, CNMT>& map, u64 title_id, ContentRecordType type) {
    auto cmnt_iter = map.find(title_id);
    u8 id_offset = 0;

    if (cmnt_iter == map.cend()) {
        const auto program_index = title_id & AOC_TITLE_ID_MASK;
        if (program_index == 0) {
            return std::nullopt;
        }

        cmnt_iter = map.find(title_id & ~AOC_TITLE_ID_MASK);
        if (cmnt_iter == map.cend()) {
            return std::nullopt;
        }

        id_offset = static_cast<u8>(program_index);
    }

    const auto& cnmt = cmnt_iter->second;
    const auto& content_records = cnmt.GetContentRecords();
    const auto iter = std::find_if(content_records.cbegin(), content_records.cend(),
                                   [type, id_offset](const ContentRecord& rec) {
                                       return rec.type == type && rec.id_offset == id_offset;
                                   });
    if (iter != content_records.cend()) {
        return std::make_optional(iter->nca_id);
    }

    if (id_offset != 0) {
        return std::nullopt;
    }

    const auto fallback_iter =
        std::find_if(content_records.cbegin(), content_records.cend(),
                     [type](const ContentRecord& rec) { return rec.type == type; });
    if (fallback_iter == content_records.cend()) {
        return std::nullopt;
    }

    return std::make_optional(fallback_iter->nca_id);
}

std::optional<NcaID> RegisteredCache::GetNcaIDFromMetadata(u64 title_id,
                                                           ContentRecordType type) const {
    if (type == ContentRecordType::Meta && meta_id.find(title_id) != meta_id.end())
        return meta_id.at(title_id);

    const auto res1 = CheckMapForContentRecord(yuzu_meta, title_id, type);
    if (res1)
        return res1;
    return CheckMapForContentRecord(meta, title_id, type);
}

void RegisteredCache::Refresh() {}

RegisteredCache::RegisteredCache(VirtualDir dir_) : dir(std::move(dir_)) {}

RegisteredCache::~RegisteredCache() = default;

bool RegisteredCache::HasEntry(u64 title_id, ContentRecordType type) const {
    return GetEntryRaw(title_id, type) != nullptr;
}

VirtualFile RegisteredCache::GetEntryUnparsed(u64 title_id, ContentRecordType type) const {
    const auto id = GetNcaIDFromMetadata(title_id, type);
    return id ? GetFileAtID(*id) : nullptr;
}

std::optional<u32> RegisteredCache::GetEntryVersion(u64 title_id) const {
    const auto meta_iter = meta.find(title_id);
    if (meta_iter != meta.cend()) {
        return meta_iter->second.GetTitleVersion();
    }

    const auto yuzu_meta_iter = yuzu_meta.find(title_id);
    if (yuzu_meta_iter != yuzu_meta.cend()) {
        return yuzu_meta_iter->second.GetTitleVersion();
    }

    return std::nullopt;
}

VirtualFile RegisteredCache::GetEntryRaw(u64, ContentRecordType) const {
    return nullptr;
}

std::unique_ptr<NCA> RegisteredCache::GetEntry(u64, ContentRecordType) const {
    return nullptr;
}

template <typename T>
void RegisteredCache::IterateAllMetadata(
    std::vector<T>& out, std::function<T(const CNMT&, const ContentRecord&)> proc,
    std::function<bool(const CNMT&, const ContentRecord&)> filter) const {
    for (const auto& kv : meta) {
        const auto& cnmt = kv.second;
        if (filter(cnmt, EMPTY_META_CONTENT_RECORD))
            out.push_back(proc(cnmt, EMPTY_META_CONTENT_RECORD));
        for (const auto& rec : cnmt.GetContentRecords()) {
            if (GetFileAtID(rec.nca_id) != nullptr && filter(cnmt, rec)) {
                out.push_back(proc(cnmt, rec));
            }
        }
    }
    for (const auto& kv : yuzu_meta) {
        const auto& cnmt = kv.second;
        for (const auto& rec : cnmt.GetContentRecords()) {
            if (GetFileAtID(rec.nca_id) != nullptr && filter(cnmt, rec)) {
                out.push_back(proc(cnmt, rec));
            }
        }
    }
}

std::vector<ContentProviderEntry> RegisteredCache::ListEntriesFilter(
    std::optional<TitleType> title_type, std::optional<ContentRecordType> record_type,
    std::optional<u64> title_id) const {
    std::vector<ContentProviderEntry> out;
    IterateAllMetadata<ContentProviderEntry>(
        out,
        [](const CNMT& c, const ContentRecord& r) {
            return ContentProviderEntry{c.GetTitleID(), r.type};
        },
        [&title_type, &record_type, &title_id](const CNMT& c, const ContentRecord& r) {
            if (title_type && *title_type != c.GetType())
                return false;
            if (record_type && *record_type != r.type)
                return false;
            if (title_id && *title_id != c.GetTitleID())
                return false;
            return true;
        });
    return out;
}

bool RegisteredCache::RemoveExistingEntry(u64 title_id) const {
    bool removed_data = false;

    const auto delete_nca = [this](const NcaID& id) {
        const auto path = GetRelativePathFromNcaID(id, false, true, false);

        const bool isFile = dir->GetFileRelative(path) != nullptr;
        const bool isDir = dir->GetDirectoryRelative(path) != nullptr;

        if (isFile) {
            return dir->DeleteFile(path);
        } else if (isDir) {
            return dir->DeleteSubdirectoryRecursive(path);
        }

        return false;
    };

    // If an entry exists in the registered cache, remove it
    if (HasEntry(title_id, ContentRecordType::Meta)) {
        LOG_INFO(Loader,
                 "Previously installed entry (v{}) for title_id={:016X} detected! "
                 "Attempting to remove...",
                 GetEntryVersion(title_id).value_or(0), title_id);

        // Get all the ncas associated with the current CNMT and delete them
        const auto meta_old_id =
            GetNcaIDFromMetadata(title_id, ContentRecordType::Meta).value_or(NcaID{});
        const auto program_id =
            GetNcaIDFromMetadata(title_id, ContentRecordType::Program).value_or(NcaID{});
        const auto data_id =
            GetNcaIDFromMetadata(title_id, ContentRecordType::Data).value_or(NcaID{});
        const auto control_id =
            GetNcaIDFromMetadata(title_id, ContentRecordType::Control).value_or(NcaID{});
        const auto html_id =
            GetNcaIDFromMetadata(title_id, ContentRecordType::HtmlDocument).value_or(NcaID{});
        const auto legal_id =
            GetNcaIDFromMetadata(title_id, ContentRecordType::LegalInformation).value_or(NcaID{});

        const auto deleted_meta = delete_nca(meta_old_id);
        const auto deleted_program = delete_nca(program_id);
        const auto deleted_data = delete_nca(data_id);
        const auto deleted_control = delete_nca(control_id);
        const auto deleted_html = delete_nca(html_id);
        const auto deleted_legal = delete_nca(legal_id);

        removed_data |= (deleted_meta || deleted_program || deleted_data || deleted_control ||
                         deleted_html || deleted_legal);
    }

    // If patch entries for any program exist in yuzu meta, remove them
    for (u8 i = 0; i < 0x10; i++) {
        const auto meta_dir = dir->CreateDirectoryRelative("yuzu_meta");
        const auto filename = GetCNMTName(TitleType::Update, title_id + i);
        if (meta_dir->GetFile(filename)) {
            removed_data |= meta_dir->DeleteFile(filename);
        }
    }

    return removed_data;
}

bool RegisteredCache::Delete(const NcaID& id) const {
    const auto path = GetRelativePathFromNcaID(id, false, true, false);

    const bool is_file = dir->GetFileRelative(path) != nullptr;
    const bool is_dir = dir->GetDirectoryRelative(path) != nullptr;

    if (is_file) {
        return dir->DeleteFile(path);
    }
    if (is_dir) {
        return dir->DeleteSubdirectoryRecursive(path);
    }

    return true;
}

ContentProviderUnion::~ContentProviderUnion() = default;

void ContentProviderUnion::SetSlot(ContentProviderUnionSlot slot, ContentProvider* provider) {
    providers[size_t(slot)] = provider;
}

void ContentProviderUnion::Refresh() {
    for (auto e : providers)
        if (e != nullptr)
            e->Refresh();
}

bool ContentProviderUnion::HasEntry(u64 title_id, ContentRecordType type) const {
    for (auto const e : providers)
        if (e && e->HasEntry(title_id, type))
            return true;
    return false;
}

std::optional<u32> ContentProviderUnion::GetEntryVersion(u64 title_id) const {
    for (auto const e : providers) {
        if (e == nullptr)
            continue;
        if (auto const res = e->GetEntryVersion(title_id); res != std::nullopt)
            return res;
    }
    return std::nullopt;
}

VirtualFile ContentProviderUnion::GetEntryUnparsed(u64 title_id, ContentRecordType type) const {
    for (auto const e : providers) {
        if (e == nullptr)
            continue;
        if (auto const res = e->GetEntryUnparsed(title_id, type); res != nullptr)
            return res;
    }
    return nullptr;
}

VirtualFile ContentProviderUnion::GetEntryRaw(u64 title_id, ContentRecordType type) const {
    for (auto const e : providers) {
        if (e == nullptr)
            continue;
        if (auto const res = e->GetEntryRaw(title_id, type); res != nullptr)
            return res;
    }
    return nullptr;
}

std::unique_ptr<NCA> ContentProviderUnion::GetEntry(u64 title_id, ContentRecordType type) const {
    for (auto const e : providers) {
        if (e == nullptr)
            continue;
        if (auto res = e->GetEntry(title_id, type); res != nullptr)
            return res;
    }
    return nullptr;
}

std::vector<ContentProviderEntry> ContentProviderUnion::ListEntriesFilter(std::optional<TitleType> title_type, std::optional<ContentRecordType> record_type, std::optional<u64> title_id) const {
    std::vector<ContentProviderEntry> out;
    for (auto const& e : providers) {
        if (e != nullptr) {
            auto const vec = e->ListEntriesFilter(title_type, record_type, title_id);
            std::copy(vec.begin(), vec.end(), std::back_inserter(out));
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::vector<std::pair<ContentProviderUnionSlot, ContentProviderEntry>> ContentProviderUnion::ListEntriesFilterOrigin(std::optional<ContentProviderUnionSlot> origin, std::optional<TitleType> title_type, std::optional<ContentRecordType> record_type, std::optional<u64> title_id) const {
    std::vector<std::pair<ContentProviderUnionSlot, ContentProviderEntry>> out;

    for (size_t i = 0; i < providers.size(); ++i) {
        auto const& e = providers[i];
        if (e == nullptr)
            continue;
        if (origin.has_value() && *origin != ContentProviderUnionSlot(i))
            continue;
        auto const vec = e->ListEntriesFilter(title_type, record_type, title_id);
        std::transform(vec.begin(), vec.end(), std::back_inserter(out), [i](const ContentProviderEntry& entry) {
            return std::make_pair(ContentProviderUnionSlot(i), entry);
        });
    }

    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::optional<ContentProviderUnionSlot> ContentProviderUnion::GetSlotForEntry(u64 title_id, ContentRecordType type) const {
    for (size_t i = 0; i < providers.size(); ++i) {
        auto const& e = providers[i];
        if (e != nullptr && e->HasEntry(title_id, type))
            return {ContentProviderUnionSlot(i)};
    }
    return std::nullopt;
}

const ExternalContentProvider* ContentProviderUnion::GetExternalProvider() const {
    return static_cast<const ExternalContentProvider*>(providers[size_t(ContentProviderUnionSlot::External)]);
}

ManualContentProvider::~ManualContentProvider() = default;

void ManualContentProvider::AddEntry(TitleType title_type, ContentRecordType content_type, u64 title_id, VirtualFile file) {
    entries.insert_or_assign({title_type, content_type, title_id}, file);
}

void ManualContentProvider::AddEntryWithVersion(TitleType title_type, ContentRecordType content_type,
                                                u64 title_id, u32 version,
                                                const std::string& version_string, VirtualFile file) {
    if (title_type == TitleType::Update) {
        auto it = std::find_if(multi_version_entries.begin(), multi_version_entries.end(), [title_id, version](const ExternalUpdateEntry& entry) {
            return entry.title_id == title_id && entry.version == version;
        });

        if (it != multi_version_entries.end()) {
            // Update existing entry
            it->files[size_t(content_type)] = file;
            if (!version_string.empty()) {
                it->version_string = version_string;
            }
        } else {
            // Add new entry
            ExternalUpdateEntry new_entry;
            new_entry.title_id = title_id;
            new_entry.version = version;
            new_entry.version_string = version_string;
            new_entry.files[size_t(content_type)] = file;
            multi_version_entries.push_back(new_entry);
        }

        auto existing = entries.find({title_type, content_type, title_id});
        if (existing == entries.end()) {
            entries.insert_or_assign({title_type, content_type, title_id}, file);
        } else {
            // Check if this version is higher
            for (const auto& entry : multi_version_entries) {
                if (entry.title_id == title_id && entry.version > version) {
                    return; // Don't replace with lower version
                }
            }
            entries.insert_or_assign({title_type, content_type, title_id}, file);
        }
    } else {
        entries.insert_or_assign({title_type, content_type, title_id}, file);
    }
}

void ManualContentProvider::ClearAllEntries() {
    entries.clear();
    multi_version_entries.clear();
}

void ManualContentProvider::Refresh() {}

bool ManualContentProvider::HasEntry(u64 title_id, ContentRecordType type) const {
    return GetEntryRaw(title_id, type) != nullptr;
}

std::optional<u32> ManualContentProvider::GetEntryVersion(u64 title_id) const {
    return std::nullopt;
}

VirtualFile ManualContentProvider::GetEntryUnparsed(u64 title_id, ContentRecordType type) const {
    return GetEntryRaw(title_id, type);
}

VirtualFile ManualContentProvider::GetEntryRaw(u64 title_id, ContentRecordType type) const {
    const auto iter =
        std::find_if(entries.begin(), entries.end(), [title_id, type](const auto& entry) {
            const auto content_type = std::get<1>(entry.first);
            const auto e_title_id = std::get<2>(entry.first);
            return content_type == type && e_title_id == title_id;
        });
    if (iter == entries.end())
        return nullptr;
    return iter->second;
}

std::unique_ptr<NCA> ManualContentProvider::GetEntry(u64 title_id, ContentRecordType type) const {
    const auto res = GetEntryRaw(title_id, type);
    if (res == nullptr)
        return nullptr;
    return std::make_unique<NCA>(res);
}

std::vector<ContentProviderEntry> ManualContentProvider::ListEntriesFilter(
    std::optional<TitleType> title_type, std::optional<ContentRecordType> record_type,
    std::optional<u64> title_id) const {
    std::vector<ContentProviderEntry> out;

    for (const auto& entry : entries) {
        const auto [e_title_type, e_content_type, e_title_id] = entry.first;
        if ((title_type == std::nullopt || e_title_type == *title_type) &&
            (record_type == std::nullopt || e_content_type == *record_type) &&
            (title_id == std::nullopt || e_title_id == *title_id)) {
            out.emplace_back(ContentProviderEntry{e_title_id, e_content_type});
        }
    }

    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::vector<ExternalUpdateEntry> ManualContentProvider::ListUpdateVersions(u64 title_id) const {
    std::vector<ExternalUpdateEntry> out;

    for (const auto& entry : multi_version_entries) {
        if (entry.title_id == title_id) {
            out.push_back(entry);
        }
    }

    std::sort(out.begin(), out.end(), [](const ExternalUpdateEntry& a, const ExternalUpdateEntry& b) {
        return a.version > b.version;
    });

    return out;
}

VirtualFile ManualContentProvider::GetEntryForVersion(u64 title_id, ContentRecordType type, u32 version) const {
    for (const auto& entry : multi_version_entries) {
        if (entry.title_id == title_id && entry.version == version) {
            if (auto const p = entry.files[size_t(type)])
                return p;
        }
    }
    return nullptr;
}

ExternalContentProvider::ExternalContentProvider(std::vector<VirtualDir> load_directories)
    : load_dirs(std::move(load_directories)) {
    ExternalContentProvider::Refresh();
}

ExternalContentProvider::~ExternalContentProvider() = default;

void ExternalContentProvider::AddDirectory(VirtualDir directory) {
    if (directory != nullptr) {
        load_dirs.push_back(std::move(directory));
    }
}

void ExternalContentProvider::ClearDirectories() {
    load_dirs.clear();
    entries.clear();
    versions.clear();
    multi_version_entries.clear();
}

void ExternalContentProvider::Refresh() {
    entries.clear();
    versions.clear();
    multi_version_entries.clear();
}

bool ExternalContentProvider::HasEntry(u64 title_id, ContentRecordType type) const {
    return GetEntryRaw(title_id, type) != nullptr;
}

std::optional<u32> ExternalContentProvider::GetEntryVersion(u64 title_id) const {
    const auto it = versions.find(title_id);
    if (it != versions.end()) {
        return it->second;
    }
    return std::nullopt;
}

VirtualFile ExternalContentProvider::GetEntryUnparsed(u64 title_id, ContentRecordType type) const {
    return GetEntryRaw(title_id, type);
}

VirtualFile ExternalContentProvider::GetEntryRaw(u64 title_id, ContentRecordType type) const {
    // Try to find in AOC (DLC) entries
    {
        const auto it = entries.find({title_id, type, TitleType::AOC});
        if (it != entries.end()) {
            return it->second;
        }
    }

    // Try to find in Update entries
    {
        const auto it = entries.find({title_id, type, TitleType::Update});
        if (it != entries.end()) {
            return it->second;
        }
    }

    return nullptr;
}

std::unique_ptr<NCA> ExternalContentProvider::GetEntry(u64 title_id,
                                                        ContentRecordType type) const {
    const auto file = GetEntryRaw(title_id, type);
    if (file == nullptr) {
        return nullptr;
    }
    return std::make_unique<NCA>(file);
}

std::vector<ContentProviderEntry> ExternalContentProvider::ListEntriesFilter(
    std::optional<TitleType> title_type, std::optional<ContentRecordType> record_type,
    std::optional<u64> title_id) const {
    std::vector<ContentProviderEntry> out;

    for (const auto& [key, file] : entries) {
        const auto& [e_title_id, e_content_type, e_title_type] = key;

        if ((title_type == std::nullopt || e_title_type == *title_type) &&
            (record_type == std::nullopt || e_content_type == *record_type) &&
            (title_id == std::nullopt || e_title_id == *title_id)) {
            out.emplace_back(ContentProviderEntry{e_title_id, e_content_type});
        }
    }

    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::vector<ExternalUpdateEntry> ExternalContentProvider::ListUpdateVersions(u64 title_id) const {
    std::vector<ExternalUpdateEntry> out;

    for (const auto& entry : multi_version_entries) {
        if (entry.title_id == title_id) {
            out.push_back(entry);
        }
    }

    std::sort(out.begin(), out.end(), [](const ExternalUpdateEntry& a, const ExternalUpdateEntry& b) {
        return a.version > b.version;
    });

    return out;
}

VirtualFile ExternalContentProvider::GetEntryForVersion(u64 title_id, ContentRecordType type, u32 version) const {
    for (const auto& entry : multi_version_entries)
        if (entry.title_id == title_id && entry.version == version)
            if (auto const p = entry.files[size_t(type)])
                return p;
    return nullptr;
}

} // namespace FileSys
