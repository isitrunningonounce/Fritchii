// SPDX-License-Identifier: GPL-3.0-or-later

float MarioKartHudlessSrgb(float value) {
    value = value < 0.0f ? 0.0f : value > 1.0f ? 1.0f : value;
    return value < 0.0031308f ? 12.92f * value :
                              1.055f * pow(value, 1.0f / 2.4f) - 0.055f;
}
