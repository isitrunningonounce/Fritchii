// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

int MkwEncodedByte(float value) {
    return int(round(value * 255.0f));
}

bool MkwUiAlphaPixel(int x, int y, int left, int top, int right, int bottom,
                     int ar, int ag, int ab, int br, int bg, int bb) {
    return x >= left && y >= top && x < right && y < bottom &&
           (abs(ar - br) > 2 || abs(ag - bg) > 2 || abs(ab - bb) > 2);
}
