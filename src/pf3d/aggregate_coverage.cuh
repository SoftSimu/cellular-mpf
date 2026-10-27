#include "../../include/pf3d/aggregate_coverage.cuh"
#include <cstdio>
#include <new>

namespace pf3d {
namespace {
constexpr int kCoverageEdge = 32;

__global__ void k_count_coverage(UpdateArgs3D args, AggregateCoverage3D coverage) {
    const CellState3D cell = args.cells[blockIdx.x];
    if (cell_support_edge(cell, args.B) > args.B)
        return;
    const auto map = BoundedZMap::make(cell.origin_z, args.B, args.layout.nz, true);
    if (map.lo >= map.hi)
        return;
    const int ox = wrap_origin(cell.origin_x, args.layout.nx);
    const int oy = wrap_origin(cell.origin_y, args.layout.ny);
    const int zlo = (map.origin + map.lo) / kCoverageEdge;
    const int zhi = (map.origin + map.hi - 1) / kCoverageEdge;
    unsigned maximum = 0;
    for (int ix = 0; ix < 2; ++ix) {
        const auto x = static_cast<std::int64_t>(ox) - ix * args.layout.nx;
        if (x + args.B <= 0)
            continue;
        const int xlo = static_cast<int>(max(std::int64_t{0}, x) / kCoverageEdge);
        const int xhi = static_cast<int>(
            (min(static_cast<std::int64_t>(args.layout.nx), x + args.B) - 1) / kCoverageEdge);
        for (int iy = 0; iy < 2; ++iy) {
            const auto y = static_cast<std::int64_t>(oy) - iy * args.layout.ny;
            if (y + args.B <= 0)
                continue;
            const int ylo = static_cast<int>(max(std::int64_t{0}, y) / kCoverageEdge);
            const int yhi = static_cast<int>(
                (min(static_cast<std::int64_t>(args.layout.ny), y + args.B) - 1) / kCoverageEdge);
            const int sx = xhi - xlo + 1, sy = yhi - ylo + 1;
            const int count = sx * sy * (zhi - zlo + 1);
            for (int i = threadIdx.x; i < count; i += blockDim.x) {
                const int gx = xlo + i % sx, gy = ylo + (i / sx) % sy;
                const int gz = zlo + i / (sx * sy);
                const auto index = (gz * coverage.ny + gy) * coverage.nx + gx;
                const unsigned next = atomicAdd(coverage.counts + index, 1u) + 1;
                maximum = max(maximum, next);
            }
        }
    }
    for (int offset = 16; offset; offset >>= 1)
        maximum = max(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    if ((threadIdx.x & 31) == 0)
        atomicMax(&coverage.bound->max_boxes, maximum);
}

__global__ void k_set_coverage_limit(AggregateBound3D *bound) {
    if (threadIdx.x == 0)
        bound->contribution_limit = bound->max_boxes ? UINT32_MAX / bound->max_boxes : UINT32_MAX;
}

__device__ __forceinline__ void scatter_with_coverage(unsigned *S, std::size_t index, float phi,
                                                      CellState3D *cell, unsigned *flags,
                                                      unsigned limit, AggregateBound3D *bound) {
    if (phi * phi > kQClampPhiSq)
        raise_flag(cell, flags, FLAG3D_Q_CLAMP);
    const unsigned value = q_of(phi);
    if (!value)
        return;
    // At most K bricks cover any site. K*limit <= UINT32_MAX proves every
    // partial sum of these nonnegative contributions safe in any issue order.
    if (value <= limit)
        atomicAdd(S + index, value);
    else
        atomicExch(&bound->has_exceptions, 1u);
}

__global__ void k_scatter_exceptions(UpdateArgs3D args) {
    if (!args.aggregate_bound->has_exceptions)
        return;
    const int cell_id = blockIdx.x / 4, shard = blockIdx.x % 4;
    CellState3D *cell = args.cells + cell_id;
    if (cell_support_edge(*cell, args.B) > args.B || fatal_flags_present(cell, args.global_flags))
        return;
    std::int64_t x0, y0, z0;
    if (!checked_world_coordinate(cell->origin_x, cell->pending_shift_x, &x0) ||
        !checked_world_coordinate(cell->origin_y, cell->pending_shift_y, &y0) ||
        !checked_world_coordinate(cell->origin_z, cell->pending_shift_z, &z0)) {
        if (threadIdx.x == 0)
            raise_flag(cell, args.global_flags, FLAG3D_INVALID_GEOMETRY);
        return;
    }
    const int ox = wrap_origin(x0, args.layout.nx), oy = wrap_origin(y0, args.layout.ny);
    const auto map = BoundedZMap::make(z0, args.B, args.layout.nz, true);
    const unsigned limit = args.aggregate_bound->contribution_limit;
    const float *field =
        args.phi_out + static_cast<std::size_t>(cell_id) * stored_cell_words(args.B, args.storage);
    const std::size_t plane = static_cast<std::size_t>(args.B) * args.B;
    for (std::size_t i = shard * blockDim.x + threadIdx.x;
         i < plane * static_cast<std::size_t>(map.hi - map.lo); i += 4 * blockDim.x) {
        const int x = i % args.B, y = (i / args.B) % args.B;
        const int z = map.lo + i / plane, wz = map.origin + z;
        const float phi = field[field_index_at_world_z(x, y, z, wz, args.B, args.storage)];
        const unsigned value = q_of(phi);
        if (value <= limit)
            continue;
        const auto index = s_index(args.layout, wrap_offset(ox, x, args.layout.nx),
                                   wrap_offset(oy, y, args.layout.ny), wz);
        const unsigned previous = atomicAdd(args.S_out + index, value);
        if (previous > UINT32_MAX - value)
            raise_flag(cell, args.global_flags, FLAG3D_S_OVERFLOW);
    }
}

__global__ void k_count_exception_step(AggregateBound3D *bound) {
    if (threadIdx.x == 0 && bound->has_exceptions)
        ++bound->exception_steps;
}
} // namespace

void destroy_aggregate_coverage(AggregateCoverage3D *coverage) {
    if (!coverage)
        return;
    if (coverage->counts && cudaFree(coverage->counts) != cudaSuccess)
        std::fprintf(stderr, "[3d] cannot free aggregate coverage counts\n");
    if (coverage->bound && cudaFree(coverage->bound) != cudaSuccess)
        std::fprintf(stderr, "[3d] cannot free aggregate coverage bound\n");
    delete coverage;
}

AggregateCoverage3D *create_aggregate_coverage(const SLayout3D &layout, int cells,
                                               std::size_t available) {
    if (!layout.hard_wall_channel() || layout.nx <= 0 || layout.ny <= 0 || layout.nz <= 0 ||
        cells <= 0 || cells > INT_MAX / 4)
        return nullptr;
    auto *coverage = new (std::nothrow) AggregateCoverage3D;
    if (!coverage)
        return nullptr;
    coverage->nx = (layout.nx - 1) / kCoverageEdge + 1;
    coverage->ny = (layout.ny - 1) / kCoverageEdge + 1;
    coverage->nz = (layout.nz - 1) / kCoverageEdge + 1;
    std::size_t count = 0, bytes = 0;
    if (!checked_box_size(coverage->nx, coverage->ny, coverage->nz, &count) || count > INT_MAX ||
        !checked_mul_size(count, sizeof(unsigned), &bytes) ||
        bytes > SIZE_MAX - sizeof(AggregateBound3D)) {
        delete coverage;
        return nullptr;
    }
    coverage->words = static_cast<int>(count);
    coverage->bytes = bytes + sizeof(AggregateBound3D);
    if (coverage->bytes > available || cudaMalloc(&coverage->counts, bytes) != cudaSuccess ||
        cudaMalloc(&coverage->bound, sizeof(AggregateBound3D)) != cudaSuccess ||
        cudaMemset(coverage->bound, 0, sizeof(AggregateBound3D)) != cudaSuccess) {
        destroy_aggregate_coverage(coverage);
        return nullptr;
    }
    return coverage;
}

bool prepare_aggregate_coverage(AggregateCoverage3D &coverage, const UpdateArgs3D &args,
                                cudaStream_t stream) {
    if (!args.S_out || !args.phi_out || !args.cells || args.N <= 0 || args.N > INT_MAX / 4 ||
        args.aggregate_bound != coverage.bound || args.selection.ids ||
        args.tile_pass != UpdateTilePass3D::All || !args.layout.hard_wall_channel() ||
        args.B <= 0 || args.B >= args.layout.nx || args.B >= args.layout.ny)
        return false;
    if (cudaMemsetAsync(coverage.counts, 0, coverage.words * sizeof(unsigned), stream) !=
            cudaSuccess ||
        cudaMemsetAsync(coverage.bound, 0, offsetof(AggregateBound3D, exception_steps), stream) !=
            cudaSuccess)
        return false;
    k_count_coverage<<<args.N, 256, 0, stream>>>(args, coverage);
    k_set_coverage_limit<<<1, 32, 0, stream>>>(coverage.bound);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool finish_aggregate_coverage(const UpdateArgs3D &args, cudaStream_t stream) {
    if (!args.aggregate_bound || args.N <= 0 || args.N > INT_MAX / 4 || args.selection.ids ||
        args.tile_pass != UpdateTilePass3D::All)
        return false;
    k_scatter_exceptions<<<args.N * 4, 256, 0, stream>>>(args);
    k_count_exception_step<<<1, 32, 0, stream>>>(args.aggregate_bound);
    return cudaPeekAtLastError() == cudaSuccess;
}
} // namespace pf3d
