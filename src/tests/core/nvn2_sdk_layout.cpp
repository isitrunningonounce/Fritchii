// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/alignment.h"
#include "common/hex_util.h"
#include "common/lz4_compression.h"
#include "core/arm/bananza_nvn2_trace.h"
#include "core/arm/nvn2_sdk_layout.h"
#include "core/loader/nso.h"

namespace {

using Core::Nvn2::SdkLayout;
using Core::Nvn2::SdkLayoutFinder;

constexpr size_t TextEnd = 0x1000;
constexpr size_t Bootstrap = 0x100;
constexpr size_t Internal = 0x200;
constexpr size_t Lookup = 0x300;
constexpr size_t Names = 0x1102;
constexpr size_t Symbols = 0x1800;
constexpr size_t Strings = 0x1900;
constexpr size_t Hash = 0x1A00;
constexpr size_t Dynamic = 0x1B00;
constexpr size_t Targets = 0x2008;
constexpr size_t SecondTable = 0x2800;
constexpr size_t ImageSize = 0x3000;
constexpr u32 Ret = 0xD65F03C0;
constexpr std::string_view AnchorName{"DeviceGetProcAddress"};

constexpr std::array<std::string_view, 7> DefaultNames{
    "BufferMap",           "CommandBufferBeginRecording", "DeviceGetInteger",
    "DeviceGetProcAddress", "DeviceInitialize",            "QueueSubmitCommands",
    "WindowSetPresentInterval",
};

void Put32(std::vector<u8>& image, size_t offset, u32 value) {
    std::memcpy(image.data() + offset, &value, sizeof(value));
}

void Put64(std::vector<u8>& image, size_t offset, u64 value) {
    std::memcpy(image.data() + offset, &value, sizeof(value));
}

void PutText(std::vector<u8>& image, size_t offset, std::string_view text) {
    std::memcpy(image.data() + offset, text.data(), text.size());
}

void PutSymbol(std::vector<u8>& image, size_t index, u32 name, u64 value, u64 size) {
    const size_t symbol = Symbols + index * 24;
    Put32(image, symbol, name);
    Put32(image, symbol + 4, 0x12 | 2U << 16); // a global function in section 2
    Put64(image, symbol + 8, value);
    Put64(image, symbol + 16, size);
}

size_t OffsetsOf(std::span<const std::string_view> names) {
    size_t end = Names;
    for (const std::string_view name : names) {
        end += name.size() + 1;
    }
    return Common::AlignUp(end, sizeof(u32));
}

// A small image with the parts of an SDK the finder reads, laid out as the real ones are.
std::vector<u8> BuildSdk(std::span<const std::string_view> names = DefaultNames,
                         bool sysv_hash = false) {
    std::vector<u8> image(ImageSize);
    Put32(image, 4, 8);
    PutText(image, 8, "MOD0");
    Put32(image, 12, static_cast<u32>(Dynamic - 8));

    Put32(image, Bootstrap, Ret);
    Put32(image, Internal, 0xAA0003E1);     // mov x1, x0
    Put32(image, Internal + 4, 0xAA1F03E0); // mov x0, xzr
    Put32(image, Internal + 8, 0x14000000 | static_cast<u32>((Lookup - (Internal + 8)) / 4));
    Put32(image, Lookup, Ret);

    const size_t offsets = OffsetsOf(names);
    size_t name = Names;
    for (size_t i = 0; i < names.size(); ++i) {
        Put32(image, offsets + i * sizeof(u32), static_cast<u32>(name - Names));
        PutText(image, name, names[i]);
        name += names[i].size() + 1;
        Put64(image, Targets + i * sizeof(u64), names[i] == AnchorName ? Lookup : 0x400 + 0x10 * i);
    }
    Put32(image, offsets + names.size() * sizeof(u32), 0x91180002); // other data, not an offset
    // What follows the table in a real SDK: more code pointers, the anchor's among them.
    const size_t after = Targets + names.size() * sizeof(u64);
    Put64(image, after, 0x900);
    Put64(image, after + 8, Lookup);
    Put64(image, after + 16, 0xFFFFFFFF);
    Put64(image, after + 24, 0x101);

    constexpr std::string_view strings{"\0nvnBootstrapLoader\0nvnBootstrapLoaderInternal\0", 47};
    PutText(image, Strings, strings);
    PutSymbol(image, 1, 1, Bootstrap, 8);
    PutSymbol(image, 2, 20, Internal, 12);
    if (sysv_hash) {
        Put32(image, Hash, 1);
        Put32(image, Hash + 4, 3); // chains: one a symbol
    } else {
        Put32(image, Hash, 1);      // buckets
        Put32(image, Hash + 4, 1);  // the first hashed symbol
        Put32(image, Hash + 8, 1);  // bloom words
        Put32(image, Hash + 24, 1); // the bucket's first symbol
        Put32(image, Hash + 28, 2); // symbol 1's hash, the chain goes on
        Put32(image, Hash + 32, 3); // symbol 2's hash, the chain ends
    }
    const std::array<std::pair<u64, u64>, 5> dynamic{{
        {6, Symbols},
        {5, Strings},
        {10, strings.size()},
        {sysv_hash ? 4 : 0x6FFFFEF5, Hash},
        {0, 0},
    }};
    for (size_t i = 0; i < dynamic.size(); ++i) {
        Put64(image, Dynamic + i * 16, dynamic[i].first);
        Put64(image, Dynamic + i * 16 + 8, dynamic[i].second);
    }
    return image;
}

SdkLayout Expected(std::span<const std::string_view> names = DefaultNames) {
    return SdkLayout{
        .names = Names,
        .name_offsets = OffsetsOf(names),
        .targets = Targets,
        .text_end = TextEnd,
        .entry_count = names.size(),
        .bootstrap = Bootstrap,
        .internal_bootstrap = Internal,
        .callback_return = Internal + 12,
    };
}

std::optional<SdkLayout> Find(const std::vector<u8>& image) {
    return SdkLayoutFinder{image, TextEnd}.Find();
}

// The layouts that were written by hand for the two SDK builds the path was made with.
struct KnownSdk {
    std::string_view build_id;
    SdkLayout layout;
};
constexpr std::array KnownSdks{
    KnownSdk{"B7FFF614F2BCDDD38E80D571619C73F300F48E1A",
             {.names = 0xB74C26,
              .name_offsets = 0xB79B90,
              .targets = 0xD95168,
              .text_end = 0x776A70,
              .entry_count = 698,
              .bootstrap = 0x6129C0,
              .internal_bootstrap = 0x612A30,
              // The hand-written layout had none; the image has the same padding as the other.
              .callback_return = 0x612A3C}},
    KnownSdk{"18265DAEA2105B31F25346ADFBBD09B5EDF1207F",
             {.names = 0xB9A5EA,
              .name_offsets = 0xB9F94C,
              .targets = 0xE056E0,
              .text_end = 0x7B65A0,
              .entry_count = 726,
              .bootstrap = 0x632C60,
              .internal_bootstrap = 0x632CC0,
              .callback_return = 0x632CCC}},
};

struct LoadedSdk {
    std::string build_id;
    std::vector<u8> image;
    size_t text_end{};
};

std::optional<LoadedSdk> LoadSdk(const std::string& path) {
    std::ifstream stream{path, std::ios::binary};
    const std::vector<u8> file{std::istreambuf_iterator<char>{stream}, {}};
    Loader::NSOHeader header;
    if (file.size() < sizeof(header)) {
        return std::nullopt;
    }
    std::memcpy(&header, file.data(), sizeof(header));
    if (header.IsZBICCompressed()) {
        return std::nullopt;
    }
    LoadedSdk sdk;
    sdk.build_id = Common::HexToString(std::span{header.build_id}.first(20));
    sdk.text_end = size_t{header.segments[0].location} + header.segments[0].size;
    for (size_t i = 0; i < header.segments.size(); ++i) {
        const auto& segment = header.segments[i];
        const size_t packed = header.segments_compressed_size[i];
        if (segment.offset > file.size() || packed > file.size() - segment.offset) {
            return std::nullopt;
        }
        const auto bytes = std::span{file}.subspan(segment.offset, packed);
        const std::vector<u8> data =
            header.IsSegmentCompressed(i)
                ? Common::Compression::DecompressDataLZ4(bytes, segment.size)
                : std::vector<u8>{bytes.begin(), bytes.end()};
        if (data.size() < segment.size) {
            return std::nullopt;
        }
        sdk.image.resize(std::max<size_t>(sdk.image.size(), size_t{segment.location} + segment.size));
        std::memcpy(sdk.image.data() + segment.location, data.data(), segment.size);
    }
    return sdk;
}

} // Anonymous namespace

TEST_CASE("NVN 2 SDK layout is found from the image", "[nvn2_sdk_layout]") {
    REQUIRE(Find(BuildSdk()) == Expected());
    REQUIRE(Find(BuildSdk(DefaultNames, true)) == Expected());
}

TEST_CASE("NVN 2 SDK layout found is one the call trace reads", "[nvn2_sdk_layout]") {
    const auto image = BuildSdk();
    const auto layout = Find(image);
    REQUIRE(layout);
    constexpr u64 Base = 0x80000000;
    const auto session = Core::BananzaNvn2Trace::Session::Create(*layout, image, Base);
    REQUIRE(session);
    REQUIRE(session->Size() == DefaultNames.size());
    REQUIRE(session->Resolve("nvnDeviceGetProcAddress") == Base + Lookup);
    REQUIRE(session->Resolve("nvnBufferMap") == Base + 0x400);
    REQUIRE(session->Resolve("nvnWindowSetPresentInterval") == Base + 0x460);
}

TEST_CASE("NVN 2 SDK layout needs both bootstrap functions", "[nvn2_sdk_layout]") {
    auto image = BuildSdk();
    Put32(image, Symbols + 2 * 24 + 4, 0x12); // the internal bootstrap is undefined
    REQUIRE_FALSE(Find(image));

    image = BuildSdk();
    Put32(image, Internal + 8, Ret); // it does not jump to a procedure
    REQUIRE_FALSE(Find(image));

    image = BuildSdk();
    PutText(image, 8, "MOD1");
    REQUIRE_FALSE(Find(image));
}

TEST_CASE("NVN 2 SDK layout rejects a table that is not one", "[nvn2_sdk_layout]") {
    constexpr std::array<std::string_view, 7> unsorted{
        "BufferMap",        "DeviceGetProcAddress", "CommandBufferBeginRecording",
        "DeviceGetInteger", "DeviceInitialize",     "QueueSubmitCommands",
        "WindowSetPresentInterval",
    };
    REQUIRE_FALSE(Find(BuildSdk(unsorted)));

    auto image = BuildSdk();
    const size_t offsets = OffsetsOf(DefaultNames);
    u32 offset;
    std::memcpy(&offset, image.data() + offsets + 5 * sizeof(u32), sizeof(offset));
    Put32(image, offsets + 5 * sizeof(u32), offset + 1); // into the middle of a name
    REQUIRE_FALSE(Find(image));

    image = BuildSdk();
    Put64(image, Targets + 3 * sizeof(u64), 0x440); // not what the internal bootstrap jumps to
    REQUIRE_FALSE(Find(image));

    image = BuildSdk();
    Put64(image, Targets + 5 * sizeof(u64), TextEnd + 0x10); // not code
    REQUIRE_FALSE(Find(image));

    image = BuildSdk();
    Put64(image, Targets + 5 * sizeof(u64), 0x400); // two procedures at one address
    REQUIRE_FALSE(Find(image));
}

TEST_CASE("NVN 2 SDK layout ignores the anchor name outside the table", "[nvn2_sdk_layout]") {
    auto image = BuildSdk();
    PutText(image, 0x1010, std::string_view{"\0DeviceGetProcAddress\0", 22});
    REQUIRE(Find(image) == Expected());
    PutText(image, 0x1F00, std::string_view{"\0DeviceGetProcAddress\0", 22});
    REQUIRE(Find(image) == Expected());
}

TEST_CASE("NVN 2 SDK layout accepts copies of the targets only when they agree",
          "[nvn2_sdk_layout]") {
    auto image = BuildSdk();
    const size_t bytes = DefaultNames.size() * sizeof(u64);
    std::memcpy(image.data() + SecondTable, image.data() + Targets, bytes);
    REQUIRE(Find(image) == Expected());
    Put64(image, SecondTable, 0x7F0);
    REQUIRE_FALSE(Find(image));
}

TEST_CASE("NVN 2 SDK layout without padding after the bootstrap has no return slot",
          "[nvn2_sdk_layout]") {
    auto image = BuildSdk();
    Put32(image, Internal + 12, Ret);
    auto expected = Expected();
    expected.callback_return = 0;
    REQUIRE(Find(image) == expected);
}

TEST_CASE("NVN 2 SDK layout finder reads no further than the image", "[nvn2_sdk_layout]") {
    const auto image = BuildSdk();
    const size_t table_end = Targets + DefaultNames.size() * sizeof(u64);
    for (size_t size = 0; size <= image.size(); size += 13) {
        const auto cut = std::span{image}.first(size);
        const auto layout = SdkLayoutFinder{cut, std::min(TextEnd, size)}.Find();
        REQUIRE(layout.has_value() == (size >= table_end));
    }
    REQUIRE_FALSE(SdkLayoutFinder{image, image.size() + 1}.Find());
    REQUIRE_FALSE(Find(std::vector<u8>(ImageSize)));
    REQUIRE_FALSE(Find(std::vector<u8>(ImageSize, 0xFF)));
}

// The SDK modules are game files and stay on the owner's machines:
// NVN2_SDK_FILES=<path of an sdk module>;<another>...
TEST_CASE("NVN 2 SDK layout of real SDK modules", "[nvn2_sdk_layout]") {
    const char* const list = std::getenv("NVN2_SDK_FILES");
    if (list == nullptr || *list == '\0') {
        SKIP("NVN2_SDK_FILES is not set");
    }
    std::string_view rest{list};
    while (!rest.empty()) {
        const size_t separator = rest.find(';');
        const std::string path{rest.substr(0, separator)};
        rest = separator == std::string_view::npos ? std::string_view{} : rest.substr(separator + 1);
        if (path.empty()) {
            continue;
        }
        INFO(path);
        const auto sdk = LoadSdk(path);
        REQUIRE(sdk);
        const auto layout = SdkLayoutFinder{sdk->image, sdk->text_end}.Find();
        REQUIRE(layout);
        INFO("SDK " << sdk->build_id << ": " << layout->entry_count << " entries, names "
                    << layout->names << ", name offsets " << layout->name_offsets << ", targets "
                    << layout->targets << ", bootstrap " << layout->bootstrap << " and "
                    << layout->internal_bootstrap << ", return slot " << layout->callback_return);
        const auto known = std::ranges::find(KnownSdks, std::string_view{sdk->build_id},
                                             &KnownSdk::build_id);
        if (known != KnownSdks.end()) {
            REQUIRE(*layout == known->layout);
        }
        const auto session = Core::BananzaNvn2Trace::Session::Create(*layout, sdk->image, 0);
        REQUIRE(session);
        REQUIRE(session->Resolve("nvnDeviceGetProcAddress") != 0);
        REQUIRE(session->Resolve("nvnDeviceGetInteger") != 0);
        REQUIRE(session->Resolve("nvnDeviceInitialize") != 0);
        REQUIRE(session->Resolve("nvnQueueSubmitCommands") != 0);
    }
}
