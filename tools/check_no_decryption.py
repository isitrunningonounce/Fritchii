#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

import re
import subprocess
import sys


FORBIDDEN = (
    b"KeyManager",
    b"PartitionDataManager",
    b"core/crypto/",
    b"Core::Crypto",
    b"AESCipher",
    b"DecompressNCZ",
    b"prod.keys",
    b"title.keys",
    b"titlekek",
    b"key_area_key",
    b"header_key",
    b"eticket_rsa",
)
EXCLUDED = (
    ":(exclude)docs/",
    ":(exclude)LICENSES/",
    ":(exclude)src/dynarmic/",
    ":(exclude)externals/",
    ":(exclude)tools/check_no_decryption.py",
)
ANDROID_EXCLUDED = (
    ":(exclude,glob)src/android/**/*.kt",
    ":(exclude,glob)src/android/**/strings.xml",
)
# Stage 2 removes these obsolete result codes.
LEGACY_RESULT_CODES = {
    b"ErrorMissingTitlekek,",
    b"ErrorIncorrectTitlekeyOrTitlekek,",
}


def main():
    pattern = "|".join(re.escape(token.decode("ascii")) for token in FORBIDDEN)
    command = ["git", "grep", "--no-textconv", "-n", "-I", "-i", "-E", "-e", pattern, "--", "."]
    result = subprocess.run(command + list(EXCLUDED + ANDROID_EXCLUDED), stdout=subprocess.PIPE,
                            check=False)
    if result.returncode not in (0, 1):
        return result.returncode

    violations = []
    for match in result.stdout.splitlines():
        path, _, source = match.split(b":", 2)
        if path == b"src/core/loader/loader.h" and source.strip() in LEGACY_RESULT_CODES:
            continue
        violations.append(match)
    if violations:
        sys.stderr.buffer.write(b"\n".join(violations) + b"\n")
        print("Game decryption references found.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
