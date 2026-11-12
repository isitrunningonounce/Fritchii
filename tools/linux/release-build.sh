#!/bin/sh -e
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# Builds the Linux x86_64 release of a build-N tag on Ubuntu 24.04 (WSL on the release PC, the
# same system as the CI runner) and writes lichii-linux-x86_64.tar.gz and lichii-linux-x86_64.AppImage
# to --out. Configure flags and packaging follow .github/workflows/release-binaries.yml.
# tools/release-local.ps1 runs it; by hand:
#
#   sh tools/linux/release-build.sh --tag build-12 --commit <sha> --repo /mnt/f/src/Fritchii \
#       --out /mnt/f/src/Fritchii/build/release-local/build-12 [--ref refs/heads/<branch>]
#
# The source is a clean clone of --repo at --ref (default the tag), which must be --commit. Clone
# and build directory are in $FRITCHII_RELEASE_HOME (default ~/fritchii-release), on the WSL disk:
# building under /mnt is much slower. Qt is $QT_ROOT_DIR (default ~/Qt/6.8.3/gcc_64, the version
# the CI installs). CMake 3.31 (Ubuntu 24.04 has 3.28), linuxdeploy and its Qt plugin are
# downloaded once into $FRITCHII_RELEASE_HOME/tools.

# Also when started as "sh release-build.sh", which ignores the -e above.
set -e

die() {
    echo "release-build: $*" >&2
    exit 1
}

tag=
commit=
ref=
repo=
out=
while [ $# -gt 0 ]; do
    [ $# -ge 2 ] || die "$1 needs a value"
    case $1 in
        --tag) tag=$2 ;;
        --commit) commit=$2 ;;
        --ref) ref=$2 ;;
        --repo) repo=$2 ;;
        --out) out=$2 ;;
        *) die "unknown argument $1" ;;
    esac
    shift 2
done
expr "$tag" : 'build-[1-9][0-9]*$' > /dev/null || die "--tag must be build-N"
expr "$commit" : '[0-9a-f]\{40\}$' > /dev/null || die "--commit must be a full commit hash"
[ -d "$repo" ] || die "--repo $repo is not a directory"
[ -n "$out" ] || die "--out is missing"
ref=${ref:-refs/tags/$tag}

home=${FRITCHII_RELEASE_HOME:-$HOME/fritchii-release}
src=$home/src
build=$home/build
tools=$home/tools
package=$home/package
qt=${QT_ROOT_DIR:-$HOME/Qt/6.8.3/gcc_64}
platform=linux-x86_64
appimage=lichii-$platform.AppImage
archive=lichii-$platform.tar.gz

[ "$(uname -m)" = x86_64 ] || die "this builds $platform only"
[ -x "$qt/bin/qmake" ] || die "no Qt at $qt; set QT_ROOT_DIR"

# Kitware's build of the CMake that CMakeLists.txt requires, checked against its published hash.
cmake_version=3.31.12
cmake_sha256=0dc2e9a6860f06bf10bd8fadc03e35d9eeb4df46e33763a7e480e987758f385c
cmake_dir=$tools/cmake-$cmake_version-linux-x86_64
cmake=$cmake_dir/bin/cmake
if [ ! -x "$cmake" ]; then
    mkdir -p "$tools"
    archive_part=$tools/cmake-$cmake_version.tar.gz.part
    curl -fsSL -o "$archive_part" \
        "https://github.com/Kitware/CMake/releases/download/v$cmake_version/cmake-$cmake_version-linux-x86_64.tar.gz"
    echo "$cmake_sha256  $archive_part" | sha256sum -c --quiet - || die "CMake $cmake_version download is corrupt"
    rm -rf "$cmake_dir"
    tar -xzf "$archive_part" -C "$tools"
    rm "$archive_part"
fi

# The Windows checkout under /mnt belongs to another user, which git refuses unless trusted.
mkdir -p "$home"
if [ ! -d "$src/.git" ]; then
    git -c safe.directory='*' clone --no-checkout "file://$repo" "$src"
fi
cd "$src"
git remote set-url origin "file://$repo"
git -c safe.directory='*' fetch --force --no-tags origin "+$ref:refs/release-build/target"
fetched=$(git rev-parse "refs/release-build/target^{commit}")
[ "$fetched" = "$commit" ] || die "$ref is $fetched, expected $commit"
git checkout --quiet --force --detach "$commit"
# .cache holds the CPM downloads (CMakeLists.txt puts them in the source tree).
git clean -ffdxq -e /.cache/

ccache=OFF
if command -v ccache > /dev/null; then
    ccache=ON
fi
# The release must not name this machine or its user: __FILE__ would carry the home directory,
# and the build tree's run path the Qt install under it.
prefix_map="-ffile-prefix-map=$home=. -ffile-prefix-map=$qt=qt"
# The CI runner has no libfmt-dev and so builds the bundled fmt 12. Ubuntu's fmt 9 would pass
# cpmfile.json's minimum but is too old for common/logging.h.
"$cmake" -S "$src" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_C_FLAGS=$prefix_map" \
    "-DCMAKE_CXX_FLAGS=$prefix_map" \
    -DCMAKE_SKIP_BUILD_RPATH=ON \
    -DBUILD_TESTING=OFF \
    -DYUZU_TESTS=OFF \
    -DYUZU_CMD=ON \
    -DYUZU_ROOM=OFF \
    -DYUZU_ROOM_STANDALONE=OFF \
    -DYUZU_USE_BUNDLED_QT=OFF \
    "-DCMAKE_PREFIX_PATH=$qt" \
    -DYUZU_USE_BUNDLED_SDL3=ON \
    -DVulkanHeaders_FORCE_BUNDLED=ON \
    -DVulkanUtilityLibraries_FORCE_BUNDLED=ON \
    -Dfmt_FORCE_BUNDLED=ON \
    -DENABLE_UPDATE_CHECKER=ON \
    "-DLICHII_BUILD_TAG=$tag" \
    "-DLICHII_BUILD_PLATFORM=$platform" \
    -DUSE_DISCORD_PRESENCE=OFF \
    "-DUSE_CCACHE=$ccache"
"$cmake" --build "$build" --target yuzu yuzu-cmd --parallel "$(nproc)"

# A stale build directory would ship another version: check what lichii will report.
for name in lichii eden-cli; do
    [ -x "$build/bin/$name" ] || die "no $build/bin/$name"
done
for expected in "$tag" "$platform" isitrunningonounce/Fritchii; do
    tr '\0' '\n' < "$build/bin/lichii" | grep -qE "(^|[^[:alnum:]_-])$expected\$" ||
        die "lichii does not contain $expected"
done

rm -rf "$package"
mkdir -p "$package"
cd "$package"
cp "$build/bin/lichii" "$build/bin/eden-cli" .
strip lichii eden-cli
for name in lichii eden-cli; do
    ! grep -qaF "$HOME" "$name" || die "$name contains $HOME"
done
tar --owner=0 --group=0 --numeric-owner -czf "$archive" lichii eden-cli

mkdir -p "$tools"
for name in linuxdeploy linuxdeploy-plugin-qt; do
    file=$tools/$name-x86_64.AppImage
    if [ ! -x "$file" ]; then
        curl -fsSL -o "$file.part" \
            "https://github.com/linuxdeploy/$name/releases/download/continuous/$name-x86_64.AppImage"
        chmod +x "$file.part"
        mv "$file.part" "$file"
    fi
done

# The desktop file names its icon io.github.zortos293.lichii.
cp "$src/dist/lichii.png" io.github.zortos293.lichii.png
export APPIMAGE_EXTRACT_AND_RUN=1
export QMAKE="$qt/bin/qmake"
export LD_LIBRARY_PATH="$qt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Qt 6's Wayland client plugin and an AppRun that prefers it under Wayland (Frizti/Fritchii#37).
# Qt 6.8.3 has libqwayland-egl.so and libqwayland-generic.so, not the libqwayland.so that PR names.
export EXTRA_QT_MODULES=waylandclient
wayland=
for plugin in libqwayland.so libqwayland-egl.so libqwayland-generic.so; do
    if [ -f "$qt/plugins/platforms/$plugin" ]; then
        wayland=${wayland:+$wayland;}$plugin
    fi
done
[ -n "$wayland" ] || die "no Qt Wayland platform plugin in $qt/plugins/platforms"
export EXTRA_PLATFORM_PLUGINS="$wayland"
version=$tag-$(printf %.7s "$commit")
export LINUXDEPLOY_OUTPUT_VERSION="$version" LDAI_VERSION="$version"
export LDAI_OUTPUT="$appimage" OUTPUT="$appimage" ARCH=x86_64

cat << 'EOF' > AppRun
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
export PATH="${HERE}/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${HERE}/usr/lib:${LD_LIBRARY_PATH:-}"
export QT_PLUGIN_PATH="${HERE}/usr/plugins:${QT_PLUGIN_PATH:-}"
export QML2_IMPORT_PATH="${HERE}/usr/qml:${QML2_IMPORT_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${HERE}/usr/plugins/platforms"

if [ -n "${WAYLAND_DISPLAY:-}" ]; then
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-wayland;xcb}"
fi

exec "${HERE}/usr/bin/lichii" "$@"
EOF
chmod +x AppRun

"$tools/linuxdeploy-x86_64.AppImage" --appdir AppDir \
    --executable lichii \
    --executable eden-cli \
    --desktop-file "$src/dist/io.github.zortos293.lichii.desktop" \
    --icon-file io.github.zortos293.lichii.png \
    --custom-apprun AppRun \
    --plugin qt \
    --output appimage
[ -s "$appimage" ] || die "linuxdeploy wrote no $appimage"
rm -rf squashfs-root
"./$appimage" --appimage-extract > /dev/null
! grep -rqaF "$HOME" squashfs-root || die "$appimage contains $HOME"
rm -rf squashfs-root

mkdir -p "$out"
cp "$archive" "$appimage" "$out/"
ls -l "$out/$archive" "$out/$appimage"
