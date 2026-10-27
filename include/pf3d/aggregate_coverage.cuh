#pragma once
#include "kernels.cuh"

namespace pf3d {
// Device-side proof for nonnegative Q5.27 sums. Contributions above the limit
// are deferred until all bounded contributions have been added.
struct AggregateBound3D {
    unsigned max_boxes = 0;
    unsigned contribution_limit = 0;
    unsigned has_exceptions = 0;
    unsigned exception_steps = 0;
};

struct AggregateCoverage3D {
    unsigned *counts = nullptr;
    AggregateBound3D *bound = nullptr;
    int nx = 0, ny = 0, nz = 0, words = 0;
    std::size_t bytes = 0;
};
AggregateCoverage3D *create_aggregate_coverage(const SLayout3D &, int cells, std::size_t available);
void destroy_aggregate_coverage(AggregateCoverage3D *);
bool prepare_aggregate_coverage(AggregateCoverage3D &, const UpdateArgs3D &, cudaStream_t);
bool finish_aggregate_coverage(const UpdateArgs3D &, cudaStream_t);
} // namespace pf3d
