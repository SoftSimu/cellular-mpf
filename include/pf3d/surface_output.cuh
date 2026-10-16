#pragma once

#include "kernels.cuh"
#include "surface_format_3d.h"

#include <memory>
#include <string>
#include <vector>

namespace pf3d {

// A view selects authoritative cells, not stale replicas. All referenced work
// must be ordered on stream; capture completes before either field is reused.
struct SurfaceDeviceView3D {
    int device = 0;
    const float* base_phi = nullptr;
    const float* const* promoted_phi = nullptr;
    const CellState3D* cells = nullptr;
    int base_edge = 0;
    CellFieldStorage3D storage{};
    CellSelection3D selection{};
    cudaStream_t stream = nullptr;
};

class SurfaceOutput3D {
public:
    SurfaceOutput3D();
    ~SurfaceOutput3D();
    SurfaceOutput3D(const SurfaceOutput3D&) = delete;
    SurfaceOutput3D& operator=(const SurfaceOutput3D&) = delete;

    bool open(const std::string& path, const SimParams3D& params,
              std::uint64_t interval, bool compress);
    bool capture(const std::vector<SurfaceDeviceView3D>& views,
                 std::uint64_t step, double time);
    bool close(bool complete = true);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pf3d
