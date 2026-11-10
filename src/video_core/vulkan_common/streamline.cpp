// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS through NVIDIA Streamline, with Streamline's manual hooking.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stop_token>
#include <thread>
#include <utility>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <fmt/format.h>

#include "common/frame_gen_stats.h"
#include "common/fs/fs_util.h"
#include "common/fs/path_util.h"
#include "common/game_presets.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/vulkan_common/streamline.h"
#include "video_core/vulkan_common/streamline_api.h"

#include <sl_security.h>

#include <softpub.h>
#include <wintrust.h>

namespace Vulkan::Streamline {
namespace {

// A custom engine names itself with its own project id; an application id from NVIDIA is only
// needed without an engine type. Keep it fixed: the driver may key behaviour off it.
constexpr const char* PROJECT_ID = "8d3e2f9a-6b41-4c7e-9a52-1f0c3b7d5e64";
constexpr const wchar_t* INTERPOSER = L"sl.interposer.dll";
constexpr const wchar_t* MODEL = L"nvngx_dlss.dll";
// The first DLSS with preset K.
constexpr u32 MIN_DLSS_MAJOR = 310;

using Version = std::array<u32, 4>;

struct State {
    HMODULE module{};
    Api api{};
    std::wstring plugin_dir;
    std::wstring model_dir;
    std::vector<std::string> instance_extensions;
    std::vector<std::string> device_extensions;
    bool started{};
    bool attached{};
    // DLSS Frame Generation (LICHII_DLSS_FG).
    bool fg_wanted{};    ///< Loaded with DLSS and its requirements read
    bool fg_available{}; ///< Supported on the attached device
    ExtraQueues fg_queues{};
    std::vector<std::string> fg_device_extensions;
    // Per frame: the token of the latest DLSS evaluation, until a present takes it (worker).
    std::atomic<const sl::FrameToken*> fg_tagged{};
    bool fg_reflex_set{}; ///< Worker
    bool fg_reflex_on{};  ///< Worker: Reflex set for the player's frame generation switch on
    // eBlockNoClientQueues: frame generation's inputs fence, from the present thread to the worker.
    VkDevice device{};
    PFN_vkWaitSemaphores wait_semaphores{};
    std::atomic<VkSemaphore> fg_inputs_fence{};
    std::atomic<u64> fg_inputs_value{};
    u64 fg_wait_timeouts{};
    // Timing for the periodic log (ns): worker fence waits, present calls, present to present.
    std::atomic<u64> fg_wait_ns{};
    u64 fg_acquire_ns{};
    std::atomic<u64> fg_free_wait_ns{};
    u64 fg_present_ns{};
    u64 fg_interval_ns{};
    std::chrono::steady_clock::time_point fg_present_begin{};
    std::chrono::steady_clock::time_point fg_last_present{};
    bool fg_on{};         ///< Present thread from here on
    u32 fg_frames{};      ///< Generated frames per rendered one while on
    bool fg_dynamic_resolution{};
    std::optional<std::array<u32, 12>> fg_mkw_geometry;
    FrameGenUiOptionsState fg_ui;
    std::optional<std::array<s32, 4>> fg_ui_log;
    u32 fg_max_frames{1}; ///< DLSSGState::numFramesToGenerateMax
    u64 fg_presents{};
    u64 fg_generated_presents{};
    sl::DLSSGStatus fg_last_status{};
    // Frames shown on screen per second (Common::FrameGenStats), present thread.
    std::chrono::steady_clock::time_point fg_window_begin{};
    u64 fg_window_shown{};
    u64 fg_window_generated{};
    u64 fg_window_presents{};
};

State& GetState() {
    static State state;
    return state;
}

int LogLevel() {
    const char* const value = std::getenv("EDEN_NX2_DLSS_LOG");
    return value ? std::atoi(value) : 0;
}

void LogFromStreamline(sl::LogType type, const char* message) {
    std::string_view text{message ? message : ""};
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.remove_suffix(1);
    }
    switch (type) {
    case sl::LogType::eError:
        LOG_ERROR(Render_Vulkan, "Streamline: {}", text);
        break;
    case sl::LogType::eWarn:
        LOG_WARNING(Render_Vulkan, "Streamline: {}", text);
        break;
    default:
        if (LogLevel() > 0) {
            LOG_INFO(Render_Vulkan, "Streamline: {}", text);
        }
        break;
    }
}

/// Where the user may put sl.interposer.dll: EDEN_STREAMLINE_DIR, the user folder's "streamline"
/// folder, then a "streamline" folder next to the executable.
std::vector<std::filesystem::path> SearchDirs() {
    std::vector<std::filesystem::path> dirs;
    if (const char* const dir = std::getenv("EDEN_STREAMLINE_DIR"); dir && *dir) {
        dirs.emplace_back(dir);
    }
    dirs.push_back(UserFolder());
    dirs.push_back(Common::FS::GetExeDirectory() / "streamline");
    return dirs;
}

template <typename T>
bool Resolve(HMODULE module, const char* name, T*& function) {
    function = reinterpret_cast<T*>(reinterpret_cast<void*>(GetProcAddress(module, name)));
    if (!function) {
        LOG_ERROR(Render_Vulkan, "sl.interposer.dll has no {}", name);
    }
    return function != nullptr;
}

std::optional<Version> ReadVersion(const std::filesystem::path& path) {
    DWORD handle = 0;
    const DWORD size = GetFileVersionInfoSizeW(path.c_str(), &handle);
    if (size == 0) {
        return std::nullopt;
    }
    std::vector<u8> data(size);
    VS_FIXEDFILEINFO* info{};
    UINT length = 0;
    if (!GetFileVersionInfoW(path.c_str(), 0, size, data.data()) ||
        !VerQueryValueW(data.data(), L"\\", reinterpret_cast<void**>(&info), &length) || !info) {
        return std::nullopt;
    }
    return Version{HIWORD(info->dwFileVersionMS), LOWORD(info->dwFileVersionMS),
                   HIWORD(info->dwFileVersionLS), LOWORD(info->dwFileVersionLS)};
}

std::string VersionText(const std::filesystem::path& path) {
    const std::optional<Version> version = path.empty() ? std::nullopt : ReadVersion(path);
    if (!version) {
        return {};
    }
    const Version& v = *version;
    return v[3] == 0 ? fmt::format("{}.{}.{}", v[0], v[1], v[2])
                     : fmt::format("{}.{}.{}.{}", v[0], v[1], v[2], v[3]);
}

/// The NVIDIA app keeps every DLSS it downloads under ProgramData; pick the newest.
std::filesystem::path FindAppModel() {
    const wchar_t* const program_data = _wgetenv(L"ProgramData");
    if (!program_data) {
        return {};
    }
    const std::filesystem::path root =
        std::filesystem::path{program_data} / L"NVIDIA" / L"NGX" / L"models" / L"dlss" / L"versions";
    std::filesystem::path best_path;
    Version best{};
    std::error_code ec;
    for (const auto& version_dir : std::filesystem::directory_iterator(root, ec)) {
        std::error_code files_ec;
        for (const auto& file :
             std::filesystem::directory_iterator(version_dir.path() / L"files", files_ec)) {
            if (file.path().extension() != L".bin") {
                continue;
            }
            const std::optional<Version> version = ReadVersion(file.path());
            if (version && (*version)[0] >= MIN_DLSS_MAJOR && *version > best) {
                best = *version;
                best_path = file.path();
            }
        }
    }
    return best_path;
}

/// Copies the NVIDIA app's DLSS into our folder as nvngx_dlss.dll, the name Streamline loads.
std::filesystem::path UseAppModel(const DlssFiles& files) {
    const std::filesystem::path dir = UserFolder() / L"nvidia-app";
    const std::filesystem::path target = dir / MODEL;
    std::error_code ec;
    const bool current = std::filesystem::is_regular_file(target, ec) &&
                         std::filesystem::file_size(target, ec) ==
                             std::filesystem::file_size(files.app_model, ec) &&
                         ReadVersion(target) == ReadVersion(files.app_model);
    if (!current) {
        if (!IsSignedByNvidia(files.app_model)) {
            LOG_WARNING(Render_Vulkan, "The NVIDIA app's DLSS {} is not signed by NVIDIA",
                        Common::FS::PathToUTF8String(files.app_model));
            return {};
        }
        std::filesystem::create_directories(dir, ec);
        std::filesystem::copy_file(files.app_model, target,
                                   std::filesystem::copy_options::overwrite_existing, ec);
        if (ec) {
            LOG_WARNING(Render_Vulkan, "Could not copy the NVIDIA app's DLSS to {}: {}",
                        Common::FS::PathToUTF8String(target), ec.message());
            return {};
        }
    }
    if (!IsSignedByNvidia(target)) {
        LOG_WARNING(Render_Vulkan, "Cached DLSS {} is not signed by NVIDIA",
                    Common::FS::PathToUTF8String(target));
        return {};
    }
    return dir;
}

bool Load(State& state, const std::filesystem::path& path) {
    if (!IsStreamlineModule(path)) {
        LOG_ERROR(Render_Vulkan, "{} is not signed by NVIDIA; not loading it",
                  Common::FS::PathToUTF8String(path));
        return false;
    }
    state.module = LoadLibraryW(path.c_str());
    if (!state.module) {
        LOG_ERROR(Render_Vulkan, "Could not load {} (error {})", Common::FS::PathToUTF8String(path),
                  GetLastError());
        return false;
    }
    Api& api = state.api;
    const HMODULE m = state.module;
    if (Resolve(m, "slInit", api.init) && Resolve(m, "slShutdown", api.shutdown) &&
        Resolve(m, "slIsFeatureSupported", api.is_feature_supported) &&
        Resolve(m, "slGetFeatureRequirements", api.get_feature_requirements) &&
        Resolve(m, "slGetFeatureVersion", api.get_feature_version) &&
        Resolve(m, "slGetFeatureFunction", api.get_feature_function) &&
        Resolve(m, "slGetNewFrameToken", api.get_new_frame_token) &&
        Resolve(m, "slSetConstants", api.set_constants) &&
        Resolve(m, "slEvaluateFeature", api.evaluate_feature) &&
        Resolve(m, "slFreeResources", api.free_resources) &&
        Resolve(m, "slSetVulkanInfo", api.set_vulkan_info)) {
        // Frame generation tags its inputs per frame (optional for DLSS alone).
        api.set_tag_for_frame = reinterpret_cast<PFun_slSetTagForFrame*>(
            reinterpret_cast<void*>(GetProcAddress(m, "slSetTagForFrame")));
        return true;
    }
    FreeLibrary(state.module);
    state.module = nullptr;
    state.api = {};
    return false;
}

std::vector<std::string> Names(uint32_t count, const char** names) {
    std::vector<std::string> result;
    for (uint32_t i = 0; i < count; ++i) {
        result.emplace_back(names[i]);
    }
    return result;
}

} // Anonymous namespace

std::filesystem::path UserFolder() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::StreamlineDir);
}

bool IsStreamlineModule(const std::filesystem::path& path) {
    return sl::security::verifyEmbeddedSignature(path.c_str());
}

bool IsSignedByNvidia(const std::filesystem::path& path) {
    const std::wstring file_path = std::filesystem::absolute(path).wstring();
    WINTRUST_FILE_INFO file{};
    file.cbStruct = sizeof(file);
    file.pcwszFilePath = file_path.c_str();
    WINTRUST_DATA data{};
    data.cbStruct = sizeof(data);
    data.dwUIChoice = WTD_UI_NONE;
    data.fdwRevocationChecks = WTD_REVOKE_NONE;
    data.dwUnionChoice = WTD_CHOICE_FILE;
    data.pFile = &file;
    data.dwStateAction = WTD_STATEACTION_VERIFY;
    data.dwProvFlags = WTD_CACHE_ONLY_URL_RETRIEVAL;
    GUID action = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    bool valid = WinVerifyTrust(nullptr, &action, &data) == ERROR_SUCCESS;
    if (valid) {
        CRYPT_PROVIDER_DATA* const provider = WTHelperProvDataFromStateData(data.hWVTStateData);
        CRYPT_PROVIDER_SGNR* const signer =
            provider ? WTHelperGetProvSignerFromChain(provider, 0, FALSE, 0) : nullptr;
        CRYPT_PROVIDER_CERT* const cert = signer ? WTHelperGetProvCertFromChain(signer, 0) : nullptr;
        wchar_t name[128]{};
        valid = cert &&
                CertGetNameStringW(cert->pCert, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, nullptr, name,
                                   static_cast<DWORD>(std::size(name))) > 1 &&
                std::wstring_view{name} == L"NVIDIA Corporation";
    }
    data.dwStateAction = WTD_STATEACTION_CLOSE;
    WinVerifyTrust(nullptr, &action, &data);
    return valid;
}

DlssFiles FindDlssFiles() {
    DlssFiles files;
    files.folder = UserFolder();
    std::error_code ec;
    for (const std::filesystem::path& dir : SearchDirs()) {
        if (std::filesystem::is_regular_file(dir / INTERPOSER, ec)) {
            files.folder = dir;
            files.interposer = dir / INTERPOSER;
            break;
        }
    }
    if (std::filesystem::is_regular_file(files.folder / MODEL, ec)) {
        files.model = files.folder / MODEL;
    }
    files.app_model = FindAppModel();
    files.interposer_version = VersionText(files.interposer);
    files.model_version = VersionText(files.model);
    files.app_model_version = VersionText(files.app_model);
    return files;
}

bool Start() {
    State& state = GetState();
    if (state.started) {
        return true;
    }
    if (!Settings::NvidiaDlssEnabled()) {
        return false;
    }
    if (!state.module) {
        const DlssFiles files = FindDlssFiles();
        if (!files.model.empty() && !IsSignedByNvidia(files.model)) {
            LOG_WARNING(Render_Vulkan, "DLSS model {} is not signed by NVIDIA; using FSR 3.1",
                        Common::FS::PathToUTF8String(files.model));
            return false;
        }
        if (files.interposer.empty()) {
            LOG_WARNING(Render_Vulkan,
                        "NVIDIA DLSS is on, but sl.interposer.dll (NVIDIA Streamline) is not in "
                        "{} (the Graphics settings can download it); using FSR 3.1",
                        Common::FS::PathToUTF8String(files.folder));
            return false;
        }
        if (!Load(state, files.interposer)) {
            return false;
        }
        state.plugin_dir = files.folder.wstring();
        state.model_dir.clear();
        if (files.model.empty() && !files.app_model.empty()) {
            if (const std::filesystem::path dir = UseAppModel(files); !dir.empty()) {
                state.model_dir = dir.wstring();
                LOG_INFO(Render_Vulkan, "Using the NVIDIA app's DLSS {}", files.app_model_version);
            }
        } else if (files.model.empty()) {
            LOG_WARNING(Render_Vulkan, "No nvngx_dlss.dll in {} and no DLSS from the NVIDIA app",
                        Common::FS::PathToUTF8String(files.folder));
        }
    }

    // Streamline's plugins come from the loader's folder or the NVIDIA app's downloads (OTA); the
    // DLSS model from the loader's folder, else the NVIDIA app's copy.
    const wchar_t* plugin_paths[] = {state.plugin_dir.c_str(), state.model_dir.c_str()};
    // Frame generation needs Reflex and PCL loaded with it.
    const bool want_fg = FrameGenRequested();
    const sl::Feature features[] = {sl::kFeatureDLSS, sl::kFeatureDLSS_G, sl::kFeatureReflex,
                                    sl::kFeaturePCL};
    sl::Preferences preferences{};
    preferences.logLevel = LogLevel() >= 2 ? sl::LogLevel::eVerbose : sl::LogLevel::eDefault;
    preferences.logMessageCallback = LogFromStreamline;
    preferences.pathsToPlugins = plugin_paths;
    preferences.numPathsToPlugins = state.model_dir.empty() ? 1 : 2;
    preferences.flags = sl::PreferenceFlags::eDisableCLStateTracking |
                        sl::PreferenceFlags::eUseManualHooking | sl::PreferenceFlags::eAllowOTA |
                        sl::PreferenceFlags::eLoadDownloadedPlugins;
    if (want_fg) {
        // slSetTagForFrame (frame generation's tags, valid until present) needs it.
        preferences.flags |= sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    }
    preferences.featuresToLoad = features;
    preferences.numFeaturesToLoad = want_fg ? 4 : 1;
    preferences.engine = sl::EngineType::eCustom;
    preferences.engineVersion = "1.0";
    preferences.projectId = PROJECT_ID;
    preferences.renderAPI = sl::RenderAPI::eVulkan;
    const Api& api = state.api;
    if (const sl::Result result = api.init(preferences, sl::kSDKVersion);
        result != sl::Result::eOk) {
        LOG_ERROR(Render_Vulkan, "NVIDIA Streamline failed to start (result {}); using FSR 3.1",
                  static_cast<int>(result));
        return false;
    }
    sl::FeatureRequirements requirements{};
    if (const sl::Result result = api.get_feature_requirements(sl::kFeatureDLSS, requirements);
        result != sl::Result::eOk) {
        LOG_WARNING(Render_Vulkan,
                    "Streamline found no DLSS plugin (result {}); put sl.dlss.dll next to "
                    "sl.interposer.dll or install the NVIDIA app. Using FSR 3.1",
                    static_cast<int>(result));
        api.shutdown();
        return false;
    }
    if (requirements.vkNumComputeQueuesRequired != 0 ||
        requirements.vkNumGraphicsQueuesRequired != 0) {
        LOG_WARNING(Render_Vulkan, "This DLSS wants extra Vulkan queues; using FSR 3.1");
        api.shutdown();
        return false;
    }
    state.instance_extensions =
        Names(requirements.vkNumInstanceExtensions, requirements.vkInstanceExtensions);
    state.device_extensions =
        Names(requirements.vkNumDeviceExtensions, requirements.vkDeviceExtensions);
    // The KHR and EXT buffer device address extensions can't both be enabled.
    const auto& device_extensions = state.device_extensions;
    if (std::ranges::find(device_extensions, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) !=
        device_extensions.end()) {
        std::erase(state.device_extensions, VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    }
    state.fg_wanted = false;
    state.fg_queues = {};
    state.fg_device_extensions.clear();
    if (want_fg) {
        sl::FeatureRequirements fg{};
        if (const sl::Result result = api.get_feature_requirements(sl::kFeatureDLSS_G, fg);
            result != sl::Result::eOk) {
            LOG_WARNING(Render_Vulkan,
                        "DLSS Frame Generation: no plugin (result {}); sl.dlss_g.dll, "
                        "nvngx_dlssg.dll, sl.reflex.dll and sl.pcl.dll go next to "
                        "sl.interposer.dll. Running without it",
                        static_cast<int>(result));
        } else {
            state.fg_wanted = true;
            state.fg_queues = {fg.vkNumComputeQueuesRequired, fg.vkNumGraphicsQueuesRequired,
                               fg.vkNumOpticalFlowQueuesRequired};
            state.fg_device_extensions = Names(fg.vkNumDeviceExtensions, fg.vkDeviceExtensions);
            for (std::string& name :
                 Names(fg.vkNumInstanceExtensions, fg.vkInstanceExtensions)) {
                if (std::ranges::find(state.instance_extensions, name) ==
                    state.instance_extensions.end()) {
                    state.instance_extensions.push_back(std::move(name));
                }
            }
            std::string extensions;
            for (const std::string& name : state.fg_device_extensions) {
                extensions += (extensions.empty() ? "" : ", ") + name;
            }
            std::string features12;
            for (const std::string& name : Names(fg.vkNumFeatures12, fg.vkFeatures12)) {
                features12 += (features12.empty() ? "" : ", ") + name;
            }
            std::string features13;
            for (const std::string& name : Names(fg.vkNumFeatures13, fg.vkFeatures13)) {
                features13 += (features13.empty() ? "" : ", ") + name;
            }
            LOG_INFO(Render_Vulkan,
                     "DLSS Frame Generation asks for {} compute, {} graphics and {} optical flow "
                     "queues; device extensions: {}; Vulkan 1.2 features: {}; 1.3 features: {}",
                     fg.vkNumComputeQueuesRequired, fg.vkNumGraphicsQueuesRequired,
                     fg.vkNumOpticalFlowQueuesRequired, extensions.empty() ? "none" : extensions,
                     features12.empty() ? "none" : features12,
                     features13.empty() ? "none" : features13);
        }
    }
    sl::FeatureVersion version{};
    api.get_feature_version(sl::kFeatureDLSS, version);
    LOG_INFO(Render_Vulkan, "NVIDIA Streamline {} started from {} (DLSS {})",
             version.versionSL.toStr(),
             Common::FS::PathToUTF8String(std::filesystem::path{state.plugin_dir}),
             version.versionNGX.toStr());
    state.started = true;
    return true;
}

bool WantsDevice() {
    const State& state = GetState();
    return state.started && !state.attached;
}

std::vector<std::string> InstanceExtensions() {
    const State& state = GetState();
    return state.started ? state.instance_extensions : std::vector<std::string>{};
}

std::vector<std::string> DeviceExtensions() {
    const State& state = GetState();
    return state.started ? state.device_extensions : std::vector<std::string>{};
}

bool FrameGenRequested() {
    // Read at every game start: the status bar sets it for the next one.
    const char* value = std::getenv("LICHII_DLSS_FG");
    return value != nullptr && value[0] == '1';
}

ExtraQueues ExtraQueuesWanted() {
    const State& state = GetState();
    return state.started && state.fg_wanted ? state.fg_queues : ExtraQueues{};
}

std::vector<std::string> FrameGenDeviceExtensions() {
    const State& state = GetState();
    return state.started && state.fg_wanted ? state.fg_device_extensions
                                            : std::vector<std::string>{};
}

void DropFrameGen(const char* reason) {
    State& state = GetState();
    if (state.fg_wanted) {
        LOG_WARNING(Render_Vulkan, "DLSS Frame Generation is off: {}", reason);
    }
    state.fg_wanted = false;
    state.fg_queues = {};
    state.fg_device_extensions.clear();
}

void* FrameGenProcAddress(const char* name) {
    const State& state = GetState();
    // Only once the device is attached: Streamline's Vulkan proxies need it.
    if (!state.started || !state.attached || !state.fg_available || !state.module) {
        return nullptr;
    }
    return reinterpret_cast<void*>(GetProcAddress(state.module, name));
}

bool IsFrameGenAvailable() {
    const State& state = GetState();
    return state.started && state.attached && state.fg_available;
}

bool AttachDevice(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
                  u32 graphics_family, u32 optical_flow_family, vk::DeviceDispatch& dld) {
    State& state = GetState();
    if (!WantsDevice()) {
        return false;
    }
    sl::VulkanInfo info{};
    info.device = device;
    info.instance = instance;
    info.physicalDevice = physical;
    info.computeQueueFamily = graphics_family;
    info.graphicsQueueFamily = graphics_family;
    if (state.fg_wanted) {
        // The device was created with frame generation's queues after the renderer's first
        // graphics queue (Device::GetDeviceQueueCreateInfos): compute, then graphics, then optical
        // flow (in its own family, or after those in the graphics family).
        info.computeQueueIndex = 1;
        info.graphicsQueueIndex = 1 + state.fg_queues.compute;
        info.opticalFlowQueueFamily = optical_flow_family;
        info.opticalFlowQueueIndex = optical_flow_family == graphics_family
                                         ? 1 + state.fg_queues.compute + state.fg_queues.graphics
                                         : 0;
        info.useNativeOpticalFlowMode = state.fg_queues.optical_flow != 0;
    }
    if (const sl::Result result = state.api.set_vulkan_info(info); result != sl::Result::eOk) {
        LOG_ERROR(Render_Vulkan, "Streamline did not take the Vulkan device (result {})",
                  static_cast<int>(result));
        Stop();
        return false;
    }
    sl::AdapterInfo adapter{};
    adapter.vkPhysicalDevice = physical;
    if (const sl::Result result = state.api.is_feature_supported(sl::kFeatureDLSS, adapter);
        result != sl::Result::eOk) {
        LOG_WARNING(Render_Vulkan,
                    "NVIDIA DLSS is not supported on this GPU or driver (Streamline result {}); "
                    "using FSR 3.1",
                    static_cast<int>(result));
        Stop();
        return false;
    }
    // Manual hooking: Streamline does its per-frame work in these.
    const auto hook = [&](auto& function, const char* name) {
        if (const FARPROC proxy = GetProcAddress(state.module, name)) {
            function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(
                reinterpret_cast<void*>(proxy));
        } else {
            LOG_WARNING(Render_Vulkan, "sl.interposer.dll has no {}", name);
        }
    };
    hook(dld.vkCreateSwapchainKHR, "vkCreateSwapchainKHR");
    hook(dld.vkDestroySwapchainKHR, "vkDestroySwapchainKHR");
    hook(dld.vkGetSwapchainImagesKHR, "vkGetSwapchainImagesKHR");
    hook(dld.vkAcquireNextImageKHR, "vkAcquireNextImageKHR");
    hook(dld.vkQueuePresentKHR, "vkQueuePresentKHR");
    hook(dld.vkDeviceWaitIdle, "vkDeviceWaitIdle");
    state.attached = true;
    state.device = device;
    state.wait_semaphores = dld.vkWaitSemaphores;
    LOG_INFO(Render_Vulkan, "NVIDIA DLSS is available through Streamline");

    state.fg_available = false;
    if (state.fg_wanted) {
        const auto feature_function = [&](sl::Feature feature, const char* name, auto& function) {
            void* address = nullptr;
            if (state.api.get_feature_function(feature, name, address) == sl::Result::eOk) {
                function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(address);
            }
            return function != nullptr;
        };
        Api& api = state.api;
        const sl::Result supported = api.is_feature_supported(sl::kFeatureDLSS_G, adapter);
        const bool functions =
            feature_function(sl::kFeatureDLSS_G, "slDLSSGSetOptions", api.dlssg_set_options) &&
            feature_function(sl::kFeatureDLSS_G, "slDLSSGGetState", api.dlssg_get_state) &&
            feature_function(sl::kFeatureReflex, "slReflexSetOptions", api.reflex_set_options) &&
            feature_function(sl::kFeatureReflex, "slReflexSleep", api.reflex_sleep) &&
            feature_function(sl::kFeaturePCL, "slPCLSetMarker", api.pcl_set_marker) &&
            api.set_tag_for_frame != nullptr;
        if (supported != sl::Result::eOk || !functions) {
            LOG_WARNING(Render_Vulkan,
                        "DLSS Frame Generation is not available (Streamline result {}, functions "
                        "{}); DLSS keeps running without it",
                        static_cast<int>(supported), functions ? "found" : "missing");
        } else {
            sl::DLSSGState fg_state{};
            sl::DLSSGOptions probe{};
            api.dlssg_get_state(sl::ViewportHandle{0u}, fg_state, &probe);
            state.fg_available = true;
            state.fg_max_frames = (std::max)(fg_state.numFramesToGenerateMax, 1u);
            Common::FrameGenStats::max_multiplier.store(state.fg_max_frames + 1,
                                                       std::memory_order_relaxed);
            if (const char* const value = std::getenv("LICHII_DLSS_FG_MULTIPLIER");
                value && value[0] != '\0') {
                Common::FrameGenStats::multiplier.store(
                    static_cast<unsigned>(std::clamp(std::atoi(value), 2, 6)),
                    std::memory_order_relaxed);
            }
            Common::FrameGenStats::available.store(true, std::memory_order_relaxed);
            LOG_INFO(Render_Vulkan,
                     "DLSS Frame Generation is available: up to {} generated frames per rendered "
                     "one, optical flow queue family {} index {}",
                     fg_state.numFramesToGenerateMax, info.opticalFlowQueueFamily,
                     info.opticalFlowQueueIndex);
        }
    }
    return true;
}

bool IsDlssAvailable() {
    const State& state = GetState();
    return state.started && state.attached;
}

// Calls slReflexSleep for the latest frame on its own thread: Reflex counts as active for
// DLSS-G without holding the Vulkan worker.
class ReflexSleeper {
public:
    ReflexSleeper() : thread{[this](std::stop_token stop) { Run(stop); }} {}

    void Post(const sl::FrameToken* token) {
        {
            std::scoped_lock lock{mutex};
            pending = token;
        }
        cv.notify_one();
    }

private:
    void Run(std::stop_token stop) {
        while (true) {
            const sl::FrameToken* token{};
            {
                std::unique_lock lock{mutex};
                if (!cv.wait(lock, stop, [this] { return pending != nullptr; })) {
                    return;
                }
                token = std::exchange(pending, nullptr);
            }
            const State& state = GetState();
            if (IsFrameGenAvailable() && state.api.reflex_sleep) {
                state.api.reflex_sleep(*token);
            }
        }
    }

    std::mutex mutex;
    std::condition_variable_any cv;
    const sl::FrameToken* pending{};
    std::jthread thread;
};

// LICHII_DLSS_FG_WAIT=1: CPU wait for frame generation's inputs fence before tagging. Off: it held
// the worker ~20 ms per frame, and the tagged inputs are copies in a ring (vk_streamline_dlss).
bool FrameGenWaitsForInputs() {
    static const bool waits = [] {
        const char* const value = std::getenv("LICHII_DLSS_FG_WAIT");
        return value && value[0] == '1';
    }();
    return waits;
}

void TagFrameGen(const sl::FrameToken& frame, const sl::Resource& depth, const sl::Resource& motion,
                 const sl::Extent& area, VkCommandBuffer cmdbuf, sl::ResourceLifecycle lifecycle) {
    State& state = GetState();
    if (!IsFrameGenAvailable()) {
        return;
    }
    const Api& api = state.api;
    // The player's switch (status bar): off, no token reaches the present (frame generation off)
    // and Reflex goes back to off, so nothing of it costs.
    const bool user_on = Common::FrameGenStats::user_enabled.load(std::memory_order_relaxed);
    if (!state.fg_reflex_set || user_on != state.fg_reflex_on) {
        // DLSS-G needs Reflex in a low latency mode and slReflexSleep every frame (else it does
        // nothing while still reporting two presented frames: "eDLSSGStatusFailReflexNot-
        // DetectedAtRuntime"). LICHII_DLSS_FG_REFLEX=0|1|2: off, low latency (default), boost.
        sl::ReflexOptions reflex{};
        reflex.mode = sl::ReflexMode::eLowLatency;
        if (const char* const value = std::getenv("LICHII_DLSS_FG_REFLEX")) {
            reflex.mode = value[0] == '0'   ? sl::ReflexMode::eOff
                          : value[0] == '2' ? sl::ReflexMode::eLowLatencyWithBoost
                                            : sl::ReflexMode::eLowLatency;
        }
        if (!user_on) {
            reflex.mode = sl::ReflexMode::eOff;
        }
        const sl::Result result = api.reflex_set_options(reflex);
        LOG_INFO(Render_Vulkan, "DLSS Frame Generation {} by the player; Reflex mode {} (result {})",
                 user_on ? "on" : "off", static_cast<int>(reflex.mode), static_cast<int>(result));
        state.fg_reflex_set = true;
        state.fg_reflex_on = user_on;
    }
    if (!user_on) {
        return;
    }
    // DLSS-G only runs with Reflex active ("eDLSSGStatusFailReflexNotDetectedAtRuntime - sl.reflex
    // must be enabled and active -1 != frame" without slReflexSleep each frame, or with the low
    // latency mode off, while still reporting two presented frames). Inline on the worker the
    // sleep held it ~14 ms a frame (28 real fps), as Reflex keeps a single frame in flight: by
    // default it sleeps on a thread of its own. LICHII_DLSS_FG_REFLEX_SLEEP=0 none, 1 inline.
    static const int reflex_sleep = [] {
        const char* const value = std::getenv("LICHII_DLSS_FG_REFLEX_SLEEP");
        return value ? std::atoi(value) : 2;
    }();
    if (reflex_sleep == 1) {
        const auto sleep_begin = std::chrono::steady_clock::now();
        api.reflex_sleep(frame);
        state.fg_wait_ns.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - sleep_begin)
                                       .count()),
                                   std::memory_order_relaxed);
    } else if (reflex_sleep == 2) {
        static ReflexSleeper sleeper;
        sleeper.Post(&frame);
    }
    api.pcl_set_marker(sl::PCLMarker::eSimulationStart, frame);
    api.pcl_set_marker(sl::PCLMarker::eSimulationEnd, frame);
    api.pcl_set_marker(sl::PCLMarker::eRenderSubmitStart, frame);
    // With eBlockNoClientQueues frame generation keeps running on its own queues after the
    // present: the previous frame's inputs must be consumed before this frame's copies replace
    // them. This command buffer is submitted after the wait, so a CPU wait orders it.
    if (const VkSemaphore fence = state.fg_inputs_fence.load(std::memory_order_acquire);
        fence && state.wait_semaphores && FrameGenWaitsForInputs()) {
        const auto wait_begin = std::chrono::steady_clock::now();
        const u64 value = state.fg_inputs_value.load(std::memory_order_acquire);
        const VkSemaphoreWaitInfo wait{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .pNext = nullptr,
            .flags = 0,
            .semaphoreCount = 1,
            .pSemaphores = &fence,
            .pValues = &value,
        };
        if (state.wait_semaphores(state.device, &wait, 100'000'000) == VK_TIMEOUT &&
            state.fg_wait_timeouts++ == 0) {
            LOG_WARNING(Render_Vulkan, "DLSS Frame Generation: inputs fence wait timed out");
        }
        state.fg_wait_ns.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - wait_begin)
                                       .count()),
                                   std::memory_order_relaxed);
    }
    // The game draws over its depth and motion next frame, before this one is presented.
    sl::ResourceTag tags[] = {
        sl::ResourceTag{const_cast<sl::Resource*>(&depth), sl::kBufferTypeDepth,
                        lifecycle, &area},
        sl::ResourceTag{const_cast<sl::Resource*>(&motion), sl::kBufferTypeMotionVectors,
                        lifecycle, &area},
    };
    const auto tag_begin = std::chrono::steady_clock::now();
    const sl::Result result =
        api.set_tag_for_frame(frame, sl::ViewportHandle{0u}, tags, static_cast<u32>(std::size(tags)),
                              reinterpret_cast<sl::CommandBuffer*>(cmdbuf));
    state.fg_wait_ns.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - tag_begin)
                                   .count()),
                               std::memory_order_relaxed);
    if (result != sl::Result::eOk) {
        static std::atomic<bool> warned{};
        if (!warned.exchange(true)) {
            LOG_WARNING(Render_Vulkan, "DLSS Frame Generation: tagging failed (result {})",
                        static_cast<int>(result));
        }
        return;
    }
    state.fg_tagged.store(&frame, std::memory_order_release);
}

void NoteFrameGenAcquire(u64 ns) {
    State& state = GetState();
    if (state.fg_on) {
        state.fg_acquire_ns += ns;
    }
}

void NoteFrameGenFreeWait(u64 ns) {
    State& state = GetState();
    if (state.fg_on) {
        state.fg_free_wait_ns.fetch_add(ns, std::memory_order_relaxed);
    }
}

VkQueue FrameGenDeviceQueue(VkDevice device, u32 family, u32 index) {
    const State& state = GetState();
    if (!state.module) {
        return VK_NULL_HANDLE;
    }
    const auto get_queue = reinterpret_cast<PFN_vkGetDeviceQueue>(
        reinterpret_cast<void*>(GetProcAddress(state.module, "vkGetDeviceQueue")));
    if (!get_queue) {
        return VK_NULL_HANDLE;
    }
    VkQueue queue{};
    get_queue(device, family, index, &queue);
    return queue;
}

bool FrameGenDropsPresentIds() {
    if (!IsFrameGenAvailable()) {
        return false;
    }
    const char* const value = std::getenv("LICHII_DLSS_FG_PRESENT_ID");
    return !value || value[0] != '1';
}

u32 FrameGenSwapchainImages() {
    if (!IsFrameGenAvailable()) {
        return 0;
    }
    const char* const value = std::getenv("LICHII_DLSS_FG_IMAGES");
    return value ? static_cast<u32>(std::atoi(value)) : 0;
}

void* TakeFrameGenToken() {
    State& state = GetState();
    if (!IsFrameGenAvailable()) {
        return nullptr;
    }
    return const_cast<sl::FrameToken*>(state.fg_tagged.exchange(nullptr, std::memory_order_acq_rel));
}

bool WaitFrameGenInputs(FrameGenInputFence& fence) {
    State& state = GetState();
    const bool retired = fence.Retire([&](VkSemaphore semaphore, u64 value) {
        if (!state.started || !state.device || !semaphore || !state.wait_semaphores) {
            return false;
        }
        const VkSemaphoreWaitInfo wait{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .semaphoreCount = 1,
            .pSemaphores = &semaphore,
            .pValues = &value,
        };
        const auto result = state.wait_semaphores(state.device, &wait,
                                                  fence.timed_out ? 0 : 100'000'000);
        return result == VK_SUCCESS;
    });
    if (!retired) {
        static std::atomic<bool> warned{};
        if (!warned.exchange(true)) {
            LOG_WARNING(Render_Vulkan,
                        "Mario Kart World temporal: HUD-less inputs fence wait timed out or unavailable; image quarantined");
        }
        fence.timed_out = true;
    }
    return retired;
}

void FrameGenBeforePresent(void* token, u32 game_left, u32 game_top, u32 game_right,
                           u32 game_bottom, VkExtent2D frame_extent, VkExtent2D swapchain_extent,
                           const FrameGenHudlessImage* hudless,
                           const FrameGenHudlessImage* ui_alpha) {
    State& state = GetState();
    if (!IsFrameGenAvailable()) {
        return;
    }
    const Api& api = state.api;
    const bool want_on = token != nullptr;
    const u32 multiplier = Common::FrameGenStats::multiplier.load(std::memory_order_relaxed);
    const u32 frames = want_on ? std::clamp<u32>(multiplier, 2, state.fg_max_frames + 1) - 1 : 0;
    const u64 program_id = Settings::GetCurrentProgramID();
    const bool mario_kart = program_id == Settings::MarioKartWorldProgramId;
    const bool bananza = program_id == Settings::DonkeyKongBananzaProgramId;
    const bool dynamic_resolution = want_on && bananza;
    const auto update_options = [&](bool recompose) {
        if (want_on == state.fg_on && frames == state.fg_frames &&
            dynamic_resolution == state.fg_dynamic_resolution &&
            (!mario_kart || !state.fg_ui.NeedsUpdate(recompose))) {
            return;
        }
        // Off for frames without a DLSS evaluation (menus, loading): nothing to generate from.
        sl::DLSSGOptions options{};
        options.mode = want_on ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
        options.numFramesToGenerate = (std::max)(frames, 1u);
        options.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
        if (dynamic_resolution) {
            options.flags |= sl::DLSSGFlags::eDynamicResolutionEnabled;
            options.dynamicResWidth = 2400;
            options.dynamicResHeight = 1350;
        }
        // The default blocks the presenting queue (the emulator's only graphics queue) until
        // frame generation finishes: the real frame rate halved (50 -> 26 fps). LICHII_DLSS_FG_BLOCK=1
        // goes back to it.
        const char* const block = std::getenv("LICHII_DLSS_FG_BLOCK");
        options.queueParallelismMode = block && block[0] == '1'
                                           ? sl::DLSSGQueueParallelismMode::eBlockPresentingClientQueue
                                           : sl::DLSSGQueueParallelismMode::eBlockNoClientQueues;
        options.enableUserInterfaceRecomposition =
            recompose ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        sl::Result result = api.dlssg_set_options(sl::ViewportHandle{0u}, options);
        const auto requested_result = result;
        const bool applied_recomposition = recompose && result == sl::Result::eOk;
        if (mario_kart && recompose && result != sl::Result::eOk) {
            options.enableUserInterfaceRecomposition = sl::Boolean::eFalse;
            result = api.dlssg_set_options(sl::ViewportHandle{0u}, options);
        }
        if (mario_kart) {
            if (result == sl::Result::eOk) {
                state.fg_ui.Applied(recompose, applied_recomposition);
            }
            const std::array log{static_cast<s32>(recompose), static_cast<s32>(state.fg_ui.active),
                                 static_cast<s32>(requested_result), static_cast<s32>(result)};
            if (state.fg_ui_log != log) {
                state.fg_ui_log = log;
                LOG_INFO(Render_Vulkan,
                         "Mario Kart World temporal: FG UI recomposition requested={} active={} "
                         "option_result={} fallback_result={}",
                         recompose, state.fg_ui.active, static_cast<int>(requested_result),
                         static_cast<int>(result));
            }
            if (result != sl::Result::eOk) {
                return;
            }
        }
        if (want_on) {
            LOG_INFO(Render_Vulkan, "DLSS Frame Generation on, {}x (result {})", frames + 1,
                     static_cast<int>(result));
        } else {
            LOG_INFO(Render_Vulkan, "DLSS Frame Generation off (result {})",
                     static_cast<int>(result));
        }
        if (bananza && result != sl::Result::eOk) {
            return;
        }
        state.fg_on = want_on;
        state.fg_frames = frames;
        state.fg_dynamic_resolution = dynamic_resolution;
        Common::FrameGenStats::active.store(want_on, std::memory_order_relaxed);
        state.fg_window_begin = {};
        state.fg_window_shown = 0;
        state.fg_window_generated = 0;
        state.fg_window_presents = 0;
    };
    if (!mario_kart || !token) {
        update_options(false);
    }
    if (!token) {
        return;
    }
    const auto& frame = *static_cast<const sl::FrameToken*>(token);
    api.pcl_set_marker(sl::PCLMarker::eRenderSubmitEnd, frame);
    // The game image's rectangle in the backbuffer, a few pixels in: frame generation runs inside
    // it and copies the edges as they are. Scaled to fullscreen, the last row came out dark and
    // generation smeared it into a black line while running (LICHII_DLSS_FG_INSET=n, default 4).
    static const u32 inset = [] {
        const char* const value = std::getenv("LICHII_DLSS_FG_INSET");
        return value ? static_cast<u32>(std::atoi(value)) : 4u;
    }();
    sl::Extent subrect{};
    if (game_right > game_left + 2 * inset && game_bottom > game_top + 2 * inset) {
        subrect = {game_top + inset, game_left + inset,
                   game_right - game_left - 2 * inset, game_bottom - game_top - 2 * inset};
        sl::ResourceTag backbuffer{nullptr, sl::kBufferTypeBackbuffer,
                                   sl::ResourceLifecycle::eValidUntilPresent, &subrect};
        api.set_tag_for_frame(frame, sl::ViewportHandle{0u}, &backbuffer, 1, nullptr);
    }
    if (mario_kart) {
        const bool valid_hudless = hudless && hudless->image && hudless->view &&
            hudless->extent.width == swapchain_extent.width &&
            hudless->extent.height == swapchain_extent.height && subrect.width && subrect.height &&
            subrect.left + subrect.width <= swapchain_extent.width &&
            subrect.top + subrect.height <= swapchain_extent.height;
        sl::Resource resource{};
        if (valid_hudless) {
            resource = {sl::ResourceType::eTex2d, hudless->image, nullptr, hudless->view,
                        static_cast<u32>(VK_IMAGE_LAYOUT_GENERAL)};
            resource.width = hudless->extent.width;
            resource.height = hudless->extent.height;
            resource.nativeFormat = static_cast<u32>(hudless->format);
            resource.mipLevels = 1;
            resource.arrayLayers = 1;
            resource.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                             VK_IMAGE_USAGE_SAMPLED_BIT;
        }
        sl::ResourceTag tag{valid_hudless ? &resource : nullptr, sl::kBufferTypeHUDLessColor,
                            sl::ResourceLifecycle::eValidUntilPresent,
                            valid_hudless ? &subrect : nullptr};
        const sl::Result result = api.set_tag_for_frame(frame, sl::ViewportHandle{0u}, &tag, 1, nullptr);
        if (state.fg_presents % 120 == 0 || result != sl::Result::eOk) {
            LOG_INFO(
                Render_Vulkan,
                "Mario Kart World temporal: FG HUD-less tagged={} extent={}x{} format={} result={}",
                valid_hudless && result == sl::Result::eOk, swapchain_extent.width,
                swapchain_extent.height, valid_hudless ? static_cast<u32>(hudless->format) : 0,
                static_cast<int>(result));
        }
        const bool valid_alpha = valid_hudless && result == sl::Result::eOk && ui_alpha &&
                                 ui_alpha->image && ui_alpha->view &&
                                 ui_alpha->format == VK_FORMAT_R8_UNORM &&
                                 ui_alpha->extent.width == swapchain_extent.width &&
                                 ui_alpha->extent.height == swapchain_extent.height;
        sl::Resource alpha_resource{};
        if (valid_alpha) {
            alpha_resource = {sl::ResourceType::eTex2d, ui_alpha->image, nullptr, ui_alpha->view,
                              static_cast<u32>(VK_IMAGE_LAYOUT_GENERAL)};
            alpha_resource.width = ui_alpha->extent.width;
            alpha_resource.height = ui_alpha->extent.height;
            alpha_resource.nativeFormat = static_cast<u32>(ui_alpha->format);
            alpha_resource.mipLevels = 1;
            alpha_resource.arrayLayers = 1;
            alpha_resource.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        }
        sl::ResourceTag alpha_tag{valid_alpha ? &alpha_resource : nullptr, sl::kBufferTypeUIAlpha,
                                  sl::ResourceLifecycle::eValidUntilPresent,
                                  valid_alpha ? &subrect : nullptr};
        const auto alpha_result =
            api.set_tag_for_frame(frame, sl::ViewportHandle{0u}, &alpha_tag, 1, nullptr);
        if (valid_alpha && alpha_result != sl::Result::eOk) {
            sl::ResourceTag clear{nullptr, sl::kBufferTypeUIAlpha,
                                  sl::ResourceLifecycle::eValidUntilPresent};
            api.set_tag_for_frame(frame, sl::ViewportHandle{0u}, &clear, 1, nullptr);
        }
        static const bool recompose = [] {
            const char* value = std::getenv("EDEN_NX2_MKW_FG_UI_RECOMPOSITION");
            return value && value[0] == '1';
        }();
        const bool alpha_tagged = valid_alpha && alpha_result == sl::Result::eOk;
        update_options(FrameGenUiOptionsState::Desired(
            recompose, valid_hudless && result == sl::Result::eOk, alpha_tagged));
        if (state.fg_presents % 120 == 0 || alpha_result != sl::Result::eOk) {
            LOG_INFO(Render_Vulkan,
                     "Mario Kart World temporal: FG UI alpha tagged={} extent={}x{} format={} "
                     "result={} recomposition={}",
                     alpha_tagged, swapchain_extent.width, swapchain_extent.height,
                     valid_alpha ? static_cast<u32>(ui_alpha->format) : 0,
                     static_cast<int>(alpha_result), state.fg_ui.active);
        }
        const std::array geometry{frame_extent.width, frame_extent.height,
                                  swapchain_extent.width, swapchain_extent.height,
                                  game_left, game_top, game_right, game_bottom,
                                  subrect.left, subrect.top, subrect.width, subrect.height};
        if (state.fg_mkw_geometry != geometry) {
            state.fg_mkw_geometry = geometry;
            LOG_INFO(Render_Vulkan,
                     "Mario Kart World temporal: FG geometry frame={}x{} swapchain={}x{} "
                     "game_rect=({},{},{},{}) subrect=({},{},{}x{}) inset={}",
                     frame_extent.width, frame_extent.height,
                     swapchain_extent.width, swapchain_extent.height,
                     game_left, game_top, game_right, game_bottom,
                     subrect.left, subrect.top, subrect.width, subrect.height, inset);
        }
    }
    api.pcl_set_marker(sl::PCLMarker::ePresentStart, frame);
    state.fg_present_begin = std::chrono::steady_clock::now();
}

void FrameGenAfterPresent(void* token, FrameGenInputFence* inputs) {
    State& state = GetState();
    if (!IsFrameGenAvailable() || !token) {
        return;
    }
    if (inputs) {
        *inputs = {.pending = true};
    }
    const Api& api = state.api;
    api.pcl_set_marker(sl::PCLMarker::ePresentEnd, *static_cast<const sl::FrameToken*>(token));
    ++state.fg_presents;
    const auto now = std::chrono::steady_clock::now();
    const auto ns = [](auto duration) {
        return static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
    };
    state.fg_present_ns += ns(now - state.fg_present_begin);
    if (state.fg_last_present != std::chrono::steady_clock::time_point{}) {
        state.fg_interval_ns += ns(now - state.fg_last_present);
    }
    state.fg_last_present = now;
    sl::DLSSGState fg_state{};
    if (api.dlssg_get_state(sl::ViewportHandle{0u}, fg_state, nullptr) != sl::Result::eOk) {
        return;
    }
    if (inputs && fg_state.inputsProcessingCompletionFence) {
        inputs->semaphore = static_cast<VkSemaphore>(fg_state.inputsProcessingCompletionFence);
        inputs->value = fg_state.lastPresentInputsProcessingCompletionFenceValue;
    }
    if (fg_state.inputsProcessingCompletionFence) {
        state.fg_inputs_value.store(fg_state.lastPresentInputsProcessingCompletionFenceValue,
                                    std::memory_order_release);
        state.fg_inputs_fence.store(static_cast<VkSemaphore>(fg_state.inputsProcessingCompletionFence),
                                    std::memory_order_release);
    }
    // What reached the screen, for the frontend's FPS counters (one-second windows).
    {
        const u32 shown = (std::max)(fg_state.numFramesActuallyPresented, 1u);
        state.fg_window_shown += shown;
        state.fg_window_generated += shown > 1 ? 1 : 0;
        ++state.fg_window_presents;
        if (state.fg_window_begin == std::chrono::steady_clock::time_point{}) {
            state.fg_window_begin = now;
        } else if (const auto elapsed = now - state.fg_window_begin;
                   elapsed >= std::chrono::seconds{1}) {
            const double seconds = std::chrono::duration<double>(elapsed).count();
            Common::FrameGenStats::shown_fps.store(
                static_cast<double>(state.fg_window_shown) / seconds, std::memory_order_relaxed);
            Common::FrameGenStats::generated_share.store(
                static_cast<double>(state.fg_window_generated) /
                    static_cast<double>(state.fg_window_presents),
                std::memory_order_relaxed);
            state.fg_window_begin = now;
            state.fg_window_shown = 0;
            state.fg_window_generated = 0;
            state.fg_window_presents = 0;
        }
    }
    if (fg_state.numFramesActuallyPresented > 1) {
        ++state.fg_generated_presents;
    }
    if (fg_state.status != state.fg_last_status) {
        LOG_WARNING(Render_Vulkan, "DLSS Frame Generation status {:#x}",
                    static_cast<u32>(fg_state.status));
        state.fg_last_status = fg_state.status;
    }
    if (state.fg_presents % 300 == 1) {
        const u64 window = state.fg_presents == 1 ? 1 : 300;
        const u64 wait_ns = state.fg_wait_ns.exchange(0, std::memory_order_relaxed);
        LOG_INFO(Render_Vulkan,
                 "DLSS Frame Generation: {} of {} rendered frames presented with generated ones "
                 "(last: {} presented); per frame: present call {:.2f} ms, present interval "
                 "{:.2f} ms, worker waits (inputs fence + tagging) {:.2f} ms, render wait + acquire {:.2f} ms, renderer waits "
                 "for a free frame {:.2f} ms",
                 state.fg_generated_presents, state.fg_presents,
                 fg_state.numFramesActuallyPresented,
                 static_cast<double>(state.fg_present_ns) / 1e6 / static_cast<double>(window),
                 static_cast<double>(state.fg_interval_ns) / 1e6 / static_cast<double>(window),
                 static_cast<double>(wait_ns) / 1e6 / static_cast<double>(window),
                 static_cast<double>(state.fg_acquire_ns) / 1e6 / static_cast<double>(window),
                 static_cast<double>(state.fg_free_wait_ns.exchange(0, std::memory_order_relaxed)) /
                     1e6 / static_cast<double>(window));
        state.fg_acquire_ns = 0;
        state.fg_present_ns = 0;
        state.fg_interval_ns = 0;
    }
}

void Stop() {
    State& state = GetState();
    if (!state.started) {
        return;
    }
    const sl::Result result = state.api.shutdown();
    state.started = false;
    state.attached = false;
    state.fg_mkw_geometry.reset();
    state.fg_ui = {};
    state.fg_ui_log.reset();
    Common::FrameGenStats::active.store(false, std::memory_order_relaxed);
    Common::FrameGenStats::available.store(false, std::memory_order_relaxed);
    // The next game start sets frame generation up again.
    state.fg_available = false;
    state.fg_reflex_set = false;
    state.fg_on = false;
    state.fg_frames = 0;
    state.fg_dynamic_resolution = false;
    state.fg_tagged.store(nullptr, std::memory_order_relaxed);
    state.fg_inputs_fence.store(VK_NULL_HANDLE, std::memory_order_relaxed);
    LOG_INFO(Render_Vulkan, "NVIDIA Streamline stopped (result {})", static_cast<int>(result));
}

const Api& GetApi() {
    return GetState().api;
}

} // namespace Vulkan::Streamline
