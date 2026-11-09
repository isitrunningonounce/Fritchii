# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

<#
Builds the Windows release of a build-N tag and adds it to that tag's GitHub release.

Start the release-binaries workflow first: it creates the tag, builds Linux and leaves a draft
release. While it runs, check out the tag and build from the repository root:

    git fetch origin refs/tags/build-12:refs/tags/build-12
    git switch --detach build-12
    pwsh tools/windows/release-pgo.ps1 -Tag build-12 -Publish

The build is clang-cl with ThinLTO and this fork's PGO profile (docs/lto-pgo-build-plan.md), in
build/release-pgo. out/ gets only the runnable build, out/Fritchii-<tag>, with the Visual C++
runtime and without tests; its zip stays in build/release-pgo/release. The script waits for the
draft, uploads the zip and, with -Publish, publishes the release once the Linux files are there.
Without -Publish it stays a draft.
#>
param(
    [Parameter(Mandatory)][ValidatePattern('^build-[1-9][0-9]*$')][string]$Tag,
    [switch]$Publish,
    [string]$Repo = 'isitrunningonounce/Fritchii',
    [string]$PgoProfile = 'build/lto-pgo/fritchii.profdata',
    [string]$InitialCache = 'build/lto-pgo/baseline-options.cmake',
    [string]$BuildDir = 'build/release-pgo',
    [string]$VisualStudio = 'C:/Program Files/Microsoft Visual Studio/2022/Community',
    [int]$Jobs = 12
)

$ErrorActionPreference = 'Stop'
$Platform = 'windows-x86_64-pgo'
$Asset = "lichii-$Platform.zip"

function Invoke-Native([string]$What, [scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}

$root = (git rev-parse --show-toplevel).Trim()
$build = (Join-Path $root $BuildDir) -replace '\\', '/'
$package = Join-Path $root "out/Fritchii-$Tag"
$zip = "$build/release/$Asset"

# Releases are made as the repository owner only, never from another signed-in account.
$owner = $Repo.Split('/')[0]
$login = (gh api user --jq .login).Trim()
if ($login -ne $owner) { throw "gh is signed in as '$login'. Run: gh auth switch -u $owner" }

# The zip has to be the tagged commit: the tag is what the update checker compares.
Invoke-Native 'Fetching the tag' { git -C $root fetch --quiet origin "refs/tags/${Tag}:refs/tags/${Tag}" }
$tagCommit = (git -C $root rev-parse "$Tag^{commit}").Trim()
$head = (git -C $root rev-parse HEAD).Trim()
if ($tagCommit -ne $head) { throw "HEAD is $head, but $Tag is $tagCommit. Run: git switch --detach $Tag" }
git -C $root diff --quiet HEAD
if ($LASTEXITCODE -ne 0) { throw 'Tracked files have local changes; commit or stash them first.' }
if (Test-Path $package) { throw "$package already exists" }

cmd /c "`"$VisualStudio/VC/Auxiliary/Build/vcvarsall.bat`" x64 >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
$env:PATH = "$VisualStudio/VC/Tools/Llvm/x64/bin;$env:PATH"

$profilePath = (Resolve-Path (Join-Path $root $PgoProfile)).Path -replace '\\', '/'
$flags = "/O2 /Ob2 /DNDEBUG -flto=thin -fprofile-use=$profilePath " +
    '-Wno-profile-instr-unprofiled -Wno-profile-instr-out-of-date -Wno-backend-plugin'
Invoke-Native 'Configure' {
    cmake -S $root -B $build -G Ninja -C (Join-Path $root $InitialCache) -UZLIB_CONF_WRITTEN `
        "-DCMAKE_C_FLAGS_RELEASE:STRING=$flags" "-DCMAKE_CXX_FLAGS_RELEASE:STRING=$flags" `
        "-DCMAKE_EXE_LINKER_FLAGS_RELEASE:STRING=/INCREMENTAL:NO /opt:lldltojobs=$Jobs /lldltocache:$build/lto-cache" `
        -DBUILD_TESTING=OFF -DYUZU_TESTS=OFF -DYUZU_ROOM=OFF -DYUZU_ROOM_STANDALONE=OFF `
        -DENABLE_UPDATE_CHECKER=ON "-DLICHII_BUILD_TAG=$Tag" "-DLICHII_BUILD_PLATFORM=$Platform"
}
Invoke-Native 'Build' { cmake --build $build --target yuzu yuzu-cmd --parallel $Jobs }

# A stale build directory would ship another version: check what the executable will report.
$exe = Join-Path $build 'bin/lichii.exe'
$text = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($exe))
foreach ($expected in "$Tag`0", "$Platform`0", "$Repo`0") {
    if (-not $text.Contains($expected)) { throw "lichii.exe does not contain '$($expected.TrimEnd([char]0))'" }
}

$skip = 'tests.exe', 'update_checker_tests.exe', 'event_wait_probe.exe', 'eden-room.exe', '*.pdb', '*.ilk', '*.lib', '*.exp'
New-Item -ItemType Directory $package | Out-Null
Get-ChildItem (Join-Path $build 'bin') | Where-Object {
    $name = $_.Name
    -not ($skip | Where-Object { $name -like $_ })
} | Copy-Item -Destination $package -Recurse

# The Qt deploy step leaves the compiler runtime out; ship it so no redistributable is needed.
$redist = Get-ChildItem "$VisualStudio/VC/Redist/MSVC" -Directory |
    Where-Object Name -Match '^\d+(\.\d+)+$' | Sort-Object { [version]$_.Name } | Select-Object -Last 1
Copy-Item (Join-Path $redist.FullName 'x64/Microsoft.VC143.CRT/*.dll') $package

$profileHash = (Get-FileHash $profilePath -Algorithm SHA256).Hash.ToLowerInvariant()
@(
    "Fritchii $Tag ($Platform)"
    "Commit: $head"
    "Built: $((Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm')) UTC, clang-cl ThinLTO + PGO"
    "Profile: $(Split-Path $profilePath -Leaf) sha256 $profileHash"
    "Visual C++ runtime: $($redist.Name)"
) | Set-Content (Join-Path $package 'build-info.txt')

New-Item -ItemType Directory -Force (Split-Path $zip) | Out-Null
Remove-Item $zip -ErrorAction SilentlyContinue
Compress-Archive -Path (Join-Path $package '*') -DestinationPath $zip -CompressionLevel Optimal
Write-Host "Package: $package"
Write-Host "Zip: $zip ($([math]::Round((Get-Item $zip).Length / 1MB)) MiB)"

# The workflow creates the draft after the Linux build, which can still be running.
$number = [int]$Tag.Substring('build-'.Length)
while ($true) {
    $json = gh release view $Tag -R $Repo --json isDraft,assets 2>$null
    if ($LASTEXITCODE -eq 0) { break }
    $run = gh run list -R $Repo --workflow release-binaries.yml --limit 100 --json number,status,conclusion |
        ConvertFrom-Json | Where-Object number -EQ $number
    if (-not $run) { throw "No release-binaries run $number in $Repo" }
    if ($run.status -eq 'completed') {
        $json = gh release view $Tag -R $Repo --json isDraft,assets 2>$null
        if ($LASTEXITCODE -eq 0) { break }
        throw "release-binaries run $number ended ($($run.conclusion)) without a release for $Tag"
    }
    Write-Host "Waiting for release-binaries run $number ($($run.status))..."
    Start-Sleep -Seconds 60
}
if (-not ($json | ConvertFrom-Json).isDraft) { throw "$Tag is already published; not replacing its files" }

Invoke-Native 'Upload' { gh release upload $Tag $zip -R $Repo --clobber }
if (-not $Publish) {
    Write-Host "Uploaded $Asset to the $Tag draft. Publish it with -Publish or on GitHub."
    return
}
$names = (gh release view $Tag -R $Repo --json assets | ConvertFrom-Json).assets.name
foreach ($name in $Asset, 'lichii-linux-x86_64.AppImage', 'lichii-linux-x86_64.tar.gz') {
    if ($names -notcontains $name) { throw "$Tag has no $name; not publishing" }
}
Invoke-Native 'Publish' { gh release edit $Tag -R $Repo --draft=false --latest }
Write-Host "Published https://github.com/$Repo/releases/tag/$Tag"
