# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

<#
Makes a Fritchii release on this PC from the commit checked out here: the Windows zip with
tools/windows/release-pgo.ps1 (clang-cl, ThinLTO and this fork's PGO profile), then the Linux
x86_64 tar.gz and AppImage in WSL (Ubuntu 24.04) with tools/linux/release-build.sh. From the
repository root:

    pwsh tools/release-local.ps1             # build and package; nothing leaves this PC
    pwsh tools/release-local.ps1 -Draft      # also push the tag and upload a draft release
    pwsh tools/release-local.ps1 -Publish    # also publish it as the latest release

The release notes are src/yuzu/fritchii/whats-new.md, which the app also shows after the update:
write and commit this build's notes there first. -Draft and -Publish refuse notes that are empty
or unchanged since the previous build.

The tag is the build-N already on HEAD, or one past the highest build-N on the remote. It is made
here when the build starts and pushed only with -Draft or -Publish. The files go to
build/release-local/<tag>, and a later run for the same tag and commit reuses them, so -Publish
after a test run uploads exactly what was tested. -Rebuild builds them again; the Windows build
then needs out/Fritchii-<tag> removed first.
#>
param(
    [ValidatePattern('^build-[1-9][0-9]*$')][string]$Tag,
    [switch]$Draft,
    [switch]$Publish,
    [switch]$Rebuild,
    [string]$Repo = 'isitrunningonounce/Fritchii',
    [string]$Remote = 'origin',
    [string]$WslDistro = 'Ubuntu-24.04',
    [int]$Jobs = 12
)

$ErrorActionPreference = 'Stop'
$WindowsAsset = 'lichii-windows-x86_64-pgo.zip'
$LinuxAssets = 'lichii-linux-x86_64.AppImage', 'lichii-linux-x86_64.tar.gz'

function Invoke-Native([string]$What, [scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}

function ConvertTo-WslPath([string]$Path) {
    $converted = wsl.exe -d $WslDistro --exec wslpath -a $Path
    if ($LASTEXITCODE -ne 0) { throw "WSL ($WslDistro) cannot convert $Path" }
    return "$converted".Trim()
}

$root = (git -C $PSScriptRoot rev-parse --show-toplevel).Trim()
$commit = (git -C $root rev-parse HEAD).Trim()
git -C $root diff --quiet HEAD
if ($LASTEXITCODE -ne 0) { throw 'Tracked files have local changes; commit or stash them first.' }
$push = $Draft -or $Publish
if ($push) {
    # Releases are made as the repository owner only, never from another signed-in account.
    $owner = $Repo.Split('/')[0]
    $login = (gh api user --jq .login).Trim()
    if ($login -ne $owner) { throw "gh is signed in as '$login'. Run: gh auth switch -u $owner" }
}

# build-N tags on the remote and here, name -> commit (an annotated tag's ^{} line is its commit).
$remoteTags = @{}
$lines = git -C $root ls-remote --tags $Remote 'refs/tags/build-*'
if ($LASTEXITCODE -ne 0) { throw "Cannot list the tags on $Remote" }
foreach ($line in $lines) {
    $sha, $ref = $line -split "`t"
    if ($ref -match '^refs/tags/(build-[0-9]+)(\^\{\})?$' -and ($matches[2] -or -not $remoteTags.Contains($matches[1]))) {
        $remoteTags[$matches[1]] = $sha
    }
}
$localTags = @{}
foreach ($name in (git -C $root tag --list 'build-*')) {
    $localTags[$name] = (git -C $root rev-parse "$name^{commit}").Trim()
}

if (-not $Tag) {
    $onHead = @(@($remoteTags.Keys) + @($localTags.Keys) | Sort-Object -Unique | Where-Object {
        $remoteTags[$_] -eq $commit -or $localTags[$_] -eq $commit
    })
    if ($onHead.Count -gt 1) { throw "HEAD has several build tags ($($onHead -join ', ')); pass -Tag" }
    if ($onHead.Count -eq 1) {
        $Tag = $onHead[0]
    } else {
        $highest = ($remoteTags.Keys | ForEach-Object { [int]$_.Substring(6) } | Measure-Object -Maximum).Maximum
        $Tag = "build-$([int]$highest + 1)"
    }
}
foreach ($place in @(@{ Name = $Remote; Tags = $remoteTags }, @{ Name = 'here'; Tags = $localTags })) {
    $at = $place.Tags[$Tag]
    if ($at -and $at -ne $commit) {
        throw "$Tag is $at $($place.Name), not HEAD ($commit). An unpublished leftover here goes with: git tag -d $Tag"
    }
}
if (-not $localTags.Contains($Tag)) {
    Invoke-Native 'Tagging' { git -C $root tag $Tag $commit }
}
$number = $Tag.Substring(6)
Write-Host "Release $Tag from $commit"

$stage = Join-Path $root "build/release-local/$Tag"
New-Item -ItemType Directory -Force $stage | Out-Null
$windowsZip = Join-Path $stage $WindowsAsset
$linuxFiles = $LinuxAssets | ForEach-Object { Join-Path $stage $_ }

# The notes go under the download links. Like the app (FritchiiUi::CleanReleaseNotes), drop the
# file's leading comment and a download header above a rule in its first 20 lines.
$notesSource = 'src/yuzu/fritchii/whats-new.md'
$lines = (Get-Content (Join-Path $root $notesSource) -Raw -Encoding utf8) -replace '(?s)^\s*<!--.*?-->', '' -split '\r?\n'
for ($i = 0; $i -lt [math]::Min($lines.Count, 20); $i++) {
    if ($lines[$i].Trim() -ne '---') { continue }
    if ((($lines | Select-Object -First $i) -join "`n") -match 'download') {
        $lines = @($lines | Select-Object -Skip ($i + 1))
    }
    break
}
$changes = ($lines -join "`n").Trim()
# The binaries bundle the same file, so stale notes have to be caught before the build.
$problem = $null
$previous = $remoteTags.Keys | Where-Object { [int]$_.Substring(6) -lt [int]$number } |
    Sort-Object { [int]$_.Substring(6) } | Select-Object -Last 1
if (-not $changes) {
    $problem = "$notesSource has no notes for this build"
} elseif ($previous) {
    git -C $root diff --quiet $remoteTags[$previous] $commit -- $notesSource 2>$null
    if ($LASTEXITCODE -eq 0) { $problem = "$notesSource is unchanged since $previous" }
}
if ($problem) {
    if ($push) { throw "${problem}: write this build's notes there and commit them first." }
    Write-Warning "${problem}. A release needs them."
}

# Files of a finished build carry a stamp with their commit; a build that stopped half way has none.
function Test-Built([string]$Platform, [string[]]$Files) {
    $stamp = Join-Path $stage "$Platform.commit"
    if ($Rebuild -or -not (Test-Path $stamp) -or (Get-Content $stamp -Raw).Trim() -ne $commit) { return $false }
    return -not ($Files | Where-Object { -not (Test-Path $_) })
}

if (Test-Built 'windows' @($windowsZip)) {
    Write-Host "Reusing $windowsZip"
} else {
    Remove-Item (Join-Path $stage 'windows.commit') -ErrorAction SilentlyContinue
    $pwsh = (Get-Process -Id $PID).Path
    Invoke-Native 'Windows build' {
        & $pwsh -NoProfile -File (Join-Path $root 'tools/windows/release-pgo.ps1') -Tag $Tag -BuildOnly -Jobs $Jobs
    }
    Copy-Item (Join-Path $root "build/release-pgo/release/$WindowsAsset") $windowsZip -Force
    Set-Content (Join-Path $stage 'windows.commit') $commit
}

if (Test-Built 'linux' $linuxFiles) {
    Write-Host "Reusing the Linux files in $stage"
} else {
    Remove-Item (Join-Path $stage 'linux.commit') -ErrorAction SilentlyContinue
    $script = ConvertTo-WslPath (Join-Path $root 'tools/linux/release-build.sh')
    $wslRoot = ConvertTo-WslPath $root
    $wslStage = ConvertTo-WslPath $stage
    Invoke-Native 'Linux build' {
        wsl.exe -d $WslDistro --exec sh $script --tag $Tag --commit $commit --repo $wslRoot --out $wslStage
    }
    Set-Content (Join-Path $stage 'linux.commit') $commit
}

$branch = (git -C $root branch --show-current).Trim()
if (-not $branch) { $branch = 'a detached HEAD' } else { $branch = "``$branch``" }
$base = "https://github.com/$Repo/releases/download/$Tag"
$notesFile = Join-Path $stage 'notes.md'
# The update dialog drops everything above the rule, which has to stay within the first 20 lines.
@"
Fritchii build $number, from $branch at $commit. Download the file for your system:

## Windows (x86_64)

[$WindowsAsset]($base/$WindowsAsset): extract it and run ``lichii.exe``, or ``eden-cli.exe`` from a terminal. Built with clang-cl, LTO and this fork's own PGO profile. The Visual C++ runtime is included.

## Linux (x86_64)

- [AppImage]($base/$($LinuxAssets[0])): self-contained, needs glibc 2.39 or newer (Ubuntu 24.04, Fedora 40, Debian 13, Arch). Make it executable (``chmod +x``) and run it.
- [tar.gz]($base/$($LinuxAssets[1])): uses your system libraries, so Qt 6 and the usual libraries must be installed.

---

$changes
"@ | Set-Content $notesFile

$files = @($windowsZip) + $linuxFiles
foreach ($file in $files) {
    Write-Host ("  {0} ({1} MiB)" -f $file, [math]::Round((Get-Item $file).Length / 1MB))
}
if (-not $push) {
    Write-Host "$Tag exists only here. Push and publish it with: pwsh tools/release-local.ps1 -Publish"
    Write-Host "To drop it instead: git tag -d $Tag"
    return
}

$defaultBranch = (gh repo view $Repo --json defaultBranchRef --jq .defaultBranchRef.name).Trim()
Invoke-Native 'Fetching' { git -C $root fetch --quiet $Remote $defaultBranch }
git -C $root merge-base --is-ancestor $commit FETCH_HEAD
if ($LASTEXITCODE -ne 0) { Write-Warning "$commit is not on $Remote/$defaultBranch yet" }

function Get-Release {
    $json = gh release view $Tag -R $Repo --json tagName,name,isDraft,assets,url 2>$null
    if ($LASTEXITCODE -ne 0) { return $null }
    return $json | ConvertFrom-Json
}
$hashes = @{}
foreach ($file in $files) { $hashes[$file] = (Get-FileHash $file -Algorithm SHA256).Hash.ToLowerInvariant() }
function Test-Uploaded($Release, [string]$File) {
    $asset = $Release.assets | Where-Object name -EQ (Split-Path $File -Leaf)
    if (-not $asset) { return $false }
    if (-not $asset.digest) { return $asset.size -eq (Get-Item $File).Length }
    return $asset.digest -eq "sha256:$($hashes[$File])"
}

$release = Get-Release
if ($release -and -not $release.isDraft) { throw "$Tag is already published; not replacing its files" }

Invoke-Native 'Pushing the tag' { git -C $root push --quiet $Remote "refs/tags/${Tag}:refs/tags/${Tag}" }
$title = "Fritchii build $number ($($commit.Substring(0, 7)))"
if ($release) {
    $upload = @($files | Where-Object { -not (Test-Uploaded $release $_) })
    if ($upload) { Invoke-Native 'Upload' { gh release upload $Tag @upload -R $Repo --clobber } }
    Invoke-Native 'Release notes' { gh release edit $Tag -R $Repo --title $title --notes-file $notesFile }
} else {
    Invoke-Native 'Release' {
        gh release create $Tag @files -R $Repo --verify-tag --draft --title $title --notes-file $notesFile
    }
}

# The update checker compares the tag with the build-N baked into the binaries and downloads
# these exact names: check the release before it goes out.
$release = Get-Release
if (-not $release -or $release.tagName -ne $Tag -or $release.name -ne $title) {
    throw "The release is tagged '$($release.tagName)' and titled '$($release.name)'"
}
foreach ($file in $files) {
    if (-not (Test-Uploaded $release $file)) { throw "$Tag does not have this $(Split-Path $file -Leaf)" }
}
if (-not $Publish) {
    Write-Host "Draft ready: $($release.url). Publish with: pwsh tools/release-local.ps1 -Publish"
    return
}
Invoke-Native 'Publish' { gh release edit $Tag -R $Repo --draft=false --latest }
Write-Host "Published https://github.com/$Repo/releases/tag/$Tag"
