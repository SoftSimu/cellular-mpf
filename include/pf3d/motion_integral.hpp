/*
 * Copyright (C) 2026 Steven A. Silber and Mikko Karttunen
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */
#pragma once

#ifdef __CUDACC__
#define PF_MOTION_HD __host__ __device__
#else
#define PF_MOTION_HD
#endif

namespace pf3d {

// Integrals of the applied in-plane velocities, not samples at output times.
struct PlanarMotionIntegral {
    double propulsion_x, propulsion_y;
    double advection_x, advection_y;

    PF_MOTION_HD void advance(double dt, float speed, float px, float py,
                              float vx, float vy) {
        // Match the single-precision active term used by the field update.
        propulsion_x += dt * static_cast<double>(speed * px);
        propulsion_y += dt * static_cast<double>(speed * py);
        advection_x += dt * static_cast<double>(vx);
        advection_y += dt * static_cast<double>(vy);
    }
};
static_assert(sizeof(PlanarMotionIntegral) == 32, "planar integral size");

}  // namespace pf3d
#undef PF_MOTION_HD
