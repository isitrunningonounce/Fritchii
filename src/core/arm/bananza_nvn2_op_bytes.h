// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include <iterator>

#include "common/common_types.h"

namespace Core::BananzaNvn2Commands {

/// The bytes an operation carries besides its arguments: up to 32 in place (a state object, the
/// float arguments of a call), more on the heap.
class OpBytes {
public:
    OpBytes() = default;
    OpBytes(OpBytes&& other) noexcept : count{other.count} {
        std::memcpy(&storage, &other.storage, sizeof(storage));
        other.count = 0;
    }
    OpBytes& operator=(OpBytes&& other) noexcept {
        if (this != &other) {
            Free();
            count = other.count;
            std::memcpy(&storage, &other.storage, sizeof(storage));
            other.count = 0;
        }
        return *this;
    }
    OpBytes(const OpBytes&) = delete;
    OpBytes& operator=(const OpBytes&) = delete;
    ~OpBytes() {
        Free();
    }

    template <typename It>
    void assign(It first, It last) {
        Free();
        const size_t size = static_cast<size_t>(std::distance(first, last));
        u8* const target = size > Inline ? (storage.heap = new u8[size]) : storage.values;
        std::copy(first, last, target);
        count = static_cast<u32>(size);
    }
    size_t size() const {
        return count;
    }
    bool empty() const {
        return count == 0;
    }
    const u8* data() const {
        return count > Inline ? storage.heap : storage.values;
    }
    const u8& operator[](size_t index) const {
        return data()[index];
    }
    const u8* begin() const {
        return data();
    }
    const u8* end() const {
        return data() + count;
    }

private:
    static constexpr size_t Inline = 32;

    void Free() {
        if (count > Inline) {
            delete[] storage.heap;
        }
        count = 0;
    }

    union {
        u8 values[Inline];
        u8* heap;
    } storage{};
    u32 count{};
};

} // namespace Core::BananzaNvn2Commands
