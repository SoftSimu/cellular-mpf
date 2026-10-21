#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace pf3d::surface {

// CRC-32/ISO-HDLC, compatible with Python's zlib.crc32. Stored checksum words
// are zero-extended to uint64_t; this is a corruption check, not authentication.
inline std::uint32_t crc32_update(std::uint32_t previous, const void* data, std::size_t size) {
    static const auto table = [] {
        std::array<std::array<std::uint32_t, 256>, 8> result{};
        for (unsigned i = 0; i < 256; ++i) {
            auto value = i;
            for (int bit = 0; bit < 8; ++bit)
                value = (value >> 1) ^ ((value & 1u) ? 0xedb88320u : 0u);
            result[0][i] = value;
        }
        for (std::size_t slice = 1; slice < result.size(); ++slice) {
            for (std::size_t i = 0; i < 256; ++i) {
                const auto value = result[slice - 1][i];
                result[slice][i] = result[0][value & 255u] ^ (value >> 8);
            }
        }
        return result;
    }();
    auto crc = ~previous;
    const auto* bytes = static_cast<const unsigned char*>(data);
    // Eight independent table lookups replace the serial byte-wise dependency.
    while (size >= 8) {
        crc ^= std::uint32_t{bytes[0]} | (std::uint32_t{bytes[1]} << 8)
             | (std::uint32_t{bytes[2]} << 16) | (std::uint32_t{bytes[3]} << 24);
        crc = table[7][crc & 255u] ^ table[6][(crc >> 8) & 255u]
            ^ table[5][(crc >> 16) & 255u] ^ table[4][crc >> 24]
            ^ table[3][bytes[4]] ^ table[2][bytes[5]]
            ^ table[1][bytes[6]] ^ table[0][bytes[7]];
        bytes += 8;
        size -= 8;
    }
    for (std::size_t q = 0; q < size; ++q) crc = table[0][(crc ^ bytes[q]) & 255u] ^ (crc >> 8);
    return ~crc;
}

// Little-endian, lossless samples for the phi=0.5 interface, not a restart.
// The zero shell is an output convention, not a new PDE boundary condition.
struct FileHeader {
    char magic[8];  // "PF3SUR1"
    std::uint32_t version, header_bytes, cells, boundary_flags;
    std::uint32_t codec, vertex_bytes, cell_bytes, frame_bytes;
    std::int64_t nx, ny, nz;
    double dx, dy, dz, dt, tau, level;
    std::uint64_t interval;
    std::int64_t channel_height, channel_padding;
    double wall_kappa, wall_width, lambda;
    std::uint64_t checksum;  // CRC-32 of this header with this word zero.
    // Version1 pins >=0.5 classification and a one-vertex exterior zero shell.
    std::uint64_t classification, shell, reserved;
};

struct FrameHeader {
    char magic[8];  // "P3SFRM1"
    std::uint64_t step;
    double time;
    std::uint64_t cells, vertices, reserved;
};

struct Cell {
    std::int64_t id, origin_x, origin_y, origin_z;  // Unpadded brick origin.
    std::uint32_t brick_edge, reserved;
    std::uint64_t vertices;
    float gamma, active_speed, radius, reserved_float;
    float polarity[3], velocity[3];
    std::uint32_t tumbles, flags;
};

struct Vertex {
    // q = ((z+1)*(B+2) + y+1)*(B+2) + x+1, x,y,z in [-1,B].
    // Two words avoid host/device ABI padding around uint64_t + float.
    std::uint32_t index_lo, index_hi;
    float phi;
};

struct ChunkHeader {
    char magic[8];  // "P3SCHK1"
    std::uint64_t vertices, stored_bytes, checksum;  // CRC of decoded vertices.
};

struct FrameFooter {
    char magic[8];  // "P3SCMT1"
    std::uint64_t step, vertices, checksum;  // CRC of on-disk frame before footer.
};

struct EndMarker {
    char magic[8];  // "P3SEND1"
    std::uint64_t frames, last_step, reserved;
};

static_assert(sizeof(FileHeader) == 192);
static_assert(sizeof(FrameHeader) == 48);
static_assert(sizeof(Cell) == 96);
static_assert(sizeof(Vertex) == 12);
static_assert(sizeof(ChunkHeader) == 32);
static_assert(sizeof(FrameFooter) == 32);
static_assert(sizeof(EndMarker) == 32);

}  // namespace pf3d::surface
