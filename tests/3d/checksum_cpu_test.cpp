#include "checkpoint_format_3d.h"
#include "surface_format_3d.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

std::uint64_t bit_crc64(std::uint64_t crc, const unsigned char* data, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= std::uint64_t{data[i]} << 56;
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc & (UINT64_C(1) << 63))
                ? (crc << 1) ^ UINT64_C(0x42f0e1eba9ea3693) : crc << 1;
    }
    return crc;
}

std::uint32_t bit_crc32(std::uint32_t crc, const unsigned char* data, std::size_t size) {
    crc = ~crc;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ ((crc & 1u) ? 0xedb88320u : 0u);
    }
    return ~crc;
}

bool check(const unsigned char* data, std::size_t size, std::uint64_t initial) {
    const auto initial32 = static_cast<std::uint32_t>(initial);
    return ckpt3d::crc64_ecma_update(initial, data, size) == bit_crc64(initial, data, size)
        && pf3d::surface::crc32_update(initial32, data, size) == bit_crc32(initial32, data, size);
}

} // namespace

int main() {
    if (ckpt3d::crc64_ecma("123456789", 9) != UINT64_C(0x6c40df5f0b497347)
        || pf3d::surface::crc32_update(0, "123456789", 9) != UINT32_C(0xcbf43926)
        || !check(nullptr, 0, UINT64_C(0x123456789abcdef0))) return 1;
    std::vector<unsigned char> bytes((1u << 20) + 16);
    std::uint64_t random = 29;
    for (auto& byte : bytes) {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        byte = static_cast<unsigned char>(random);
    }
    for (std::size_t offset = 0; offset < 8; ++offset) {
        for (std::size_t size = 0; size <= 521; ++size) {
            if (!check(bytes.data() + offset, size, 0)
                || !check(bytes.data() + offset, size, random)
                || !check(bytes.data() + offset, size, UINT64_MAX)) return 2;
        }
    }
    if (!check(bytes.data() + 3, 1u << 20, random)) return 3;
    const auto full64 = ckpt3d::crc64_ecma_update(random, bytes.data(), bytes.size());
    const auto full32 = pf3d::surface::crc32_update(static_cast<std::uint32_t>(random),
                                                 bytes.data(), bytes.size());
    for (const std::size_t chunk : {1u, 7u, 8u, 9u, 4093u, 65536u}) {
        auto crc64 = random;
        auto crc32 = static_cast<std::uint32_t>(random);
        for (std::size_t offset = 0; offset < bytes.size(); offset += chunk) {
            const auto size = std::min(chunk, bytes.size() - offset);
            crc64 = ckpt3d::crc64_ecma_update(crc64, bytes.data() + offset, size);
            crc32 = pf3d::surface::crc32_update(crc32, bytes.data() + offset, size);
        }
        if (crc64 != full64 || crc32 != full32) return 4;
    }
    for (const unsigned char value : {0, 255}) {
        std::fill(bytes.begin(), bytes.end(), value);
        if (!check(bytes.data(), bytes.size(), random)) return 5;
    }
    std::puts("3D checksum golden, bit-reference, unaligned and streaming tests passed");
    return 0;
}
