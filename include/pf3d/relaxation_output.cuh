/*
 * Copyright (C) 2026 Steven A. Silber and Mikko Karttunen
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */
#pragma once
#include "surface_output.cuh"
#include <memory>
#include <string>

namespace pf3d {

struct RelaxationDeviceView3D {
    SurfaceDeviceView3D fields;
    const std::uint32_t* aggregate;
    SLayout3D layout;
};

// Read-only spatial observer. No solver moments, velocities, RNG state or
// checkpoint fields are modified. Integrate its sampled rates in postprocessing.
class RelaxationOutput3D {
public:
    RelaxationOutput3D();
    ~RelaxationOutput3D();
    RelaxationOutput3D(const RelaxationOutput3D&) = delete;
    RelaxationOutput3D& operator=(const RelaxationOutput3D&) = delete;
    bool open(const std::string& path, const SimParams3D&, std::uint64_t start_step,
              std::uint64_t interval, const std::string& metadata);
    bool capture(const std::vector<RelaxationDeviceView3D>&, std::uint64_t step);
    bool close();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pf3d
