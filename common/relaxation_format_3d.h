/*
 * Copyright (C) 2026 Steven A. Silber and Mikko Karttunen
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */
#pragma once
#include <cstdint>

namespace pf3d::relaxation {

// Little-endian records. Each frame contains N rows in ascending global-id
// order. Rates are independent observations of the current stored field.
constexpr std::uint32_t kValues = 18;
struct Header {
    char magic[8];                    // PF3DRLX1
    std::uint32_t cells, values;
    std::uint64_t start_step;
    double dt;
    std::uint64_t interval, nx, ny, nz;
    std::uint32_t boundary_flags, metadata_bytes;
    std::uint32_t checksum, reserved;  // CRC-32 of header (checksum=0) + metadata
};
struct Row {
    std::int64_t id;
    double centroid[3], volume, mobility, gamma;
    double propulsion[2], advection[2]; // accepted-step cumulative integrals
    double interfacial[3], overlap[3]; // centroid velocities from -M delta F/dphi
    double advection_factor[2];       // sum(phi(x)*phi(x+e))/sum(phi^2)
};
struct Frame {
    char magic[8];                    // P3RLFRM1
    std::uint64_t step;
    double time;
    std::uint32_t cells, checksum;    // CRC-32 of frame (checksum=0) + rows
};
struct End {
    char magic[8];                    // P3RLEND1; absent if output failed
    std::uint64_t frames, last_step;
};
static_assert(sizeof(Header) == 80 && sizeof(Row) == 152 &&
              sizeof(Frame) == 32 && sizeof(End) == 24, "relaxation format layout");

} // namespace pf3d::relaxation
