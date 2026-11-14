/*
 * Copyright (C) 2026 Steven A. Silber and Mikko Karttunen
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */
#include "pf3d/relaxation_output.cuh"
#include "output_file.hpp"
#include "relaxation_format_3d.h"
#include "surface_format_3d.h"
#include "device_common.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace pf3d {
namespace {
constexpr int kThreads = 256, kShards = 32, kSums = 14;
struct Partial { double q[kSums]; };

bool fail(const char* message) {
    std::fprintf(stderr, "[relaxation3d] %s\n", message);
    return false;
}
bool gpu_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "[relaxation3d] %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}
#define RELAX_CUDA(call) do { if (!gpu_ok((call), #call)) return false; } while (false)

struct DeviceScope {
    int device;
    explicit DeviceScope(int current) : device(current) {}
    ~DeviceScope() { gpu_ok(cudaSetDevice(device), "restore device"); }
};

template<int FixedEdge = 0>
__global__ __launch_bounds__(kThreads, 3)
void measure(RelaxationDeviceView3D view, double lambda, double kappa, Partial* partials) {
    const auto& fields = view.fields;
    const int id = fields.selection.cell_index(blockIdx.y);
    const auto& state = fields.cells[id];
    const bool promoted = cell_is_promoted(state, fields.base_edge);
    if constexpr (FixedEdge != 0) {
        if (promoted) return;
    } else if (fields.base_edge == 176 && !promoted) return;
    const int B = FixedEdge != 0 ? FixedEdge : cell_support_edge(state, fields.base_edge);
    const float* phi = promoted ? fields.promoted_phi[id]
        : fields.base_phi + static_cast<std::size_t>(id) * fields.storage.words(fields.base_edge);
    Partial local{};
    if (!phi) local.q[0] = nan("");
    const int ox = wrap_origin(state.origin_x, view.layout.nx);
    const int oy = wrap_origin(state.origin_y, view.layout.ny);
    // Sum physical planes only; reflected ghosts enter through the stencil.
    const int begin = static_cast<int>(max(0LL, -static_cast<long long>(state.origin_z)));
    const int end = static_cast<int>(min(static_cast<long long>(B),
        static_cast<long long>(view.layout.nz) - state.origin_z));
    using Index = std::conditional_t<(FixedEdge > 0 && FixedEdge <= 1024),
                                     std::uint32_t, std::size_t>;
    const Index plane = static_cast<Index>(B) * B;
    const Index words = end > begin ? plane * (end - begin) : 0;
    for (Index q = blockIdx.x * blockDim.x + threadIdx.x; phi && q < words;
         q += kShards * blockDim.x) {
        const int x = static_cast<int>(q % B);
        const int y = static_cast<int>((q / B) % B);
        const int z = begin + static_cast<int>(q / plane);
        const float* centre = phi + fields.storage.index(x, y, z, B, state.origin_z);
        const float stored = *centre;
        if (stored == 0.0f) continue;
        const double p = stored, square = p * p;
        double lap = kLapCentreW * p;
        double plus_x, plus_y;
        // Interior planes are contiguous in both cubic and compact storage.
        // Keep the signed ghost-plane mapping only at a physical/support face.
        if (x > 0 && x + 1 < B && y > 0 && y + 1 < B && z > begin && z + 1 < end) {
#pragma unroll
            for (int dz = -1; dz <= 1; ++dz)
#pragma unroll
                for (int dy = -1; dy <= 1; ++dy)
#pragma unroll
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int distance = (dx != 0) + (dy != 0) + (dz != 0);
                        if (distance) lap += (distance == 1 ? kLapFaceW : distance == 2 ? kLapEdgeW : kLapCornerW)
                            * static_cast<double>(centre[(dz * B + dy) * B + dx]);
                    }
            plus_x = centre[1];
            plus_y = centre[B];
        } else {
            std::size_t planes[3];
            int signs[3];
#pragma unroll
            for (int dz = -1; dz <= 1; ++dz) {
                int source_z = 0;
                signs[dz + 1] = phase_fetch_local_z(view.layout, state.origin_z, z + dz, B, &source_z);
                planes[dz + 1] = fields.storage.index(0, 0, source_z, B, state.origin_z);
            }
            const auto sample = [&](int dx, int dy, int dz) {
                const int sx = x + dx, sy = y + dy;
                if (static_cast<unsigned>(sx) >= static_cast<unsigned>(B) ||
                    static_cast<unsigned>(sy) >= static_cast<unsigned>(B) || !signs[dz + 1])
                    return 0.0;
                return signs[dz + 1] * static_cast<double>(
                    phi[planes[dz + 1] + static_cast<std::size_t>(sy) * B + sx]);
            };
#pragma unroll
            for (int dz = -1; dz <= 1; ++dz)
#pragma unroll
                for (int dy = -1; dy <= 1; ++dy)
#pragma unroll
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int distance = (dx != 0) + (dy != 0) + (dz != 0);
                        if (distance) lap += (distance == 1 ? kLapFaceW : distance == 2 ? kLapEdgeW : kLapCornerW)
                            * sample(dx, dy, dz);
                    }
            plus_x = sample(1, 0, 0);
            plus_y = sample(0, 1, 0);
        }
        lap /= kLapDenom;
        const std::uint32_t total = view.aggregate[s_index(view.layout,
            wrap_offset(ox, x, view.layout.nx), wrap_offset(oy, y, view.layout.ny),
            static_cast<int>(state.origin_z + z))];
        const std::uint32_t own = q_of(stored);
        // An observer must not raise the solver's sticky flags on invalid S.
        const double other = total >= own ? static_cast<double>(total - own) * kQInvD
                                          : nan("");
        const double G = -2.0 * state.gamma * lap + (2.0 * kNumerBulk) * state.gamma /
            (lambda * lambda) * p * (1.0 - p) * (1.0 - 2.0 * p);
        const double K = (2.0 * kNumerInteraction) * kappa / (lambda * lambda) * p * other;
        local.q[0] += square;
        local.q[1] += x * square; local.q[2] += y * square; local.q[3] += z * square;
        const double weighted_G = p * G, weighted_K = p * K;
        local.q[4] += weighted_G;
        local.q[5] += x * weighted_G; local.q[6] += y * weighted_G; local.q[7] += z * weighted_G;
        local.q[8] += weighted_K;
        local.q[9] += x * weighted_K; local.q[10] += y * weighted_K; local.q[11] += z * weighted_K;
        local.q[12] += p * plus_x;
        local.q[13] += p * plus_y;
    }
    __shared__ double sums[kSums][kThreads];
#pragma unroll
    for (int j = 0; j < kSums; ++j) sums[j][threadIdx.x] = local.q[j];
    __syncthreads();
    for (int stride = kThreads / 2; stride; stride /= 2) {
        if (threadIdx.x < stride)
#pragma unroll
            for (int j = 0; j < kSums; ++j)
                sums[j][threadIdx.x] += sums[j][threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int j = 0; j < kSums; ++j)
            partials[blockIdx.y * kShards + blockIdx.x].q[j] = sums[j][0];
}

__global__ void finalize(RelaxationDeviceView3D view, double uniform_mobility,
                          int selected, const Partial* partials, relaxation::Row* rows) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= selected) return;
    const auto& state = view.fields.cells[view.fields.selection.cell_index(slot)];
    Partial sum{};
    for (int shard = 0; shard < kShards; ++shard)
        for (int j = 0; j < kSums; ++j) sum.q[j] += partials[slot * kShards + shard].q[j];
    relaxation::Row row{};
    row.id = state.global_id;
    row.volume = sum.q[0];
    row.mobility = state.M_pf == 0.0f ? uniform_mobility : state.M_pf;
    row.gamma = state.gamma;
    const double origin[] = {static_cast<double>(state.origin_x),
                            static_cast<double>(state.origin_y),
                            static_cast<double>(state.origin_z)};
    for (int axis = 0; axis < 3; ++axis) {
        const double centre = sum.q[1 + axis] / row.volume;
        row.centroid[axis] = origin[axis] + centre + (axis == 2 ? 0.5 : 0.0);
        row.interfacial[axis] = -2.0 * row.mobility / row.volume *
            (sum.q[5 + axis] - centre * sum.q[4]);
        row.overlap[axis] = -2.0 * row.mobility / row.volume *
            (sum.q[9 + axis] - centre * sum.q[8]);
    }
    row.propulsion[0] = state.planar_motion.propulsion_x;
    row.propulsion[1] = state.planar_motion.propulsion_y;
    row.advection[0] = state.planar_motion.advection_x;
    row.advection[1] = state.planar_motion.advection_y;
    row.advection_factor[0] = sum.q[12] / row.volume;
    row.advection_factor[1] = sum.q[13] / row.volume;
    if (!state.record_planar_motion) row.volume = nan("");
    rows[slot] = row;
}

struct Workspace {
    int device, capacity = 0;
    Partial* partials = nullptr;
    relaxation::Row* rows = nullptr;
    explicit Workspace(int selected_device) : device(selected_device) {}
    ~Workspace() {
        if (!partials && !rows) return;
        int original = 0;
        if (!gpu_ok(cudaGetDevice(&original), "get device during cleanup")) return;
        DeviceScope restore(original);
        if (!gpu_ok(cudaSetDevice(device), "select workspace device")) return;
        gpu_ok(cudaFree(partials), "free spatial partials");
        gpu_ok(cudaFree(rows), "free spatial rows");
    }
    bool allocate(int cells) {
        if (capacity) return capacity >= cells;
        RELAX_CUDA(cudaMalloc(reinterpret_cast<void**>(&partials), cells * kShards * sizeof(Partial)));
        RELAX_CUDA(cudaMalloc(reinterpret_cast<void**>(&rows), cells * sizeof(relaxation::Row)));
        capacity = cells;
        return true;
    }
};
} // namespace

struct RelaxationOutput3D::Impl {
    SimParams3D params{};
    std::FILE* file = nullptr;
    bool failed = false;
    std::uint64_t frames = 0, last_step = 0, start_step = 0;
    std::vector<std::int64_t> identity;
    std::vector<std::unique_ptr<Workspace>> workspaces;
    ~Impl() { if (file) std::fclose(file); }

    bool write(const void* data, std::size_t size) {
        if (failed || !file) return false;
        if (std::fwrite(data, 1, size, file) == size) return true;
        failed = true;
        return fail("write failed; segment is incomplete");
    }
    bool capture(const std::vector<RelaxationDeviceView3D>& views, std::uint64_t step) {
        if (!file || failed || views.empty() || step < start_step || (frames && step <= last_step))
            return fail("invalid capture state or accepted-step clock");
        int original = 0;
        RELAX_CUDA(cudaGetDevice(&original));
        DeviceScope restore(original);
        std::vector<relaxation::Row> all(params.num_cells);
        std::vector<int> devices;
        struct Pending {
            Workspace* workspace;
            cudaStream_t stream;
            int count;
            std::size_t offset;
        };
        std::vector<Pending> pending;
        std::size_t total = 0;
        for (const auto& view : views) {
            const auto& fields = view.fields;
            if (!fields.cells || !fields.base_phi || !fields.promoted_phi || !view.aggregate ||
                !fields.selection.valid(params.num_cells) || fields.base_edge <= 0 ||
                view.layout.nx != params.Nx || view.layout.ny != params.Ny ||
                view.layout.nz != params.Nz || view.layout.boundary_flags != params.boundary_flags ||
                (fields.storage.z_cap != 0 && fields.storage.z_cap != params.Nz) ||
                std::find(devices.begin(), devices.end(), fields.device) != devices.end())
                return fail("invalid or repeated owner view");
            devices.push_back(fields.device);
            const int selected = fields.selection.selected_count(params.num_cells);
            if (!selected) continue;
            if (selected > 65535) return fail("owner population exceeds grid-y limit");
            RELAX_CUDA(cudaSetDevice(fields.device));
            auto found = std::find_if(workspaces.begin(), workspaces.end(), [&](const auto& item) {
                return item->device == fields.device;
            });
            if (found == workspaces.end()) {
                workspaces.push_back(std::make_unique<Workspace>(fields.device));
                found = workspaces.end() - 1;
            }
            auto& ws = **found;
            if (!ws.allocate(params.num_cells)) return false;
            // Constant base geometry removes repeated integer division per voxel.
            // The two launches write disjoint cell slots with the same reduction order.
            if (fields.base_edge == 176)
                measure<176><<<dim3(kShards, selected), kThreads, 0, fields.stream>>>(
                    view, params.lambda, params.kappa, ws.partials);
            measure<><<<dim3(kShards, selected), kThreads, 0, fields.stream>>>(
                view, params.lambda, params.kappa, ws.partials);
            RELAX_CUDA(cudaGetLastError());
            finalize<<<(selected + kThreads - 1) / kThreads, kThreads, 0, fields.stream>>>(
                view, params.phase_field_mobility, selected, ws.partials, ws.rows);
            RELAX_CUDA(cudaGetLastError());
            pending.push_back({&ws, fields.stream, selected, total});
            total += selected;
        }
        if (total != all.size())
            return fail("owner views must cover all cells exactly once");
        // Launch every owner's reduction before waiting for any host transfer.
        for (const auto& item : pending) {
            RELAX_CUDA(cudaSetDevice(item.workspace->device));
            RELAX_CUDA(cudaMemcpyAsync(all.data() + item.offset, item.workspace->rows,
                                      item.count * sizeof(relaxation::Row),
                                      cudaMemcpyDeviceToHost, item.stream));
            RELAX_CUDA(cudaStreamSynchronize(item.stream));
        }
        std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
        std::vector<std::int64_t> ids;
        for (const auto& row : all) {
            if ((!ids.empty() && row.id <= ids.back()) || row.id < 0 ||
                row.volume <= 0 || row.mobility <= 0 || row.gamma <= 0)
                return fail("invalid spatial observation or cell identity");
            double values[relaxation::kValues];
            std::memcpy(values, &row.centroid, sizeof(values));
            for (double value : values)
                if (!std::isfinite(value)) return fail("non-finite spatial observation");
            ids.push_back(row.id);
        }
        if (frames && ids != identity) return fail("cell identities changed within segment");
        identity = std::move(ids);
        relaxation::Frame frame{};
        std::memcpy(frame.magic, "P3RLFRM1", 8);
        frame.step = step; frame.time = static_cast<double>(step) * params.dt;
        frame.cells = params.num_cells;
        frame.checksum = surface::crc32_update(0, &frame, sizeof(frame));
        frame.checksum = surface::crc32_update(frame.checksum, all.data(), all.size() * sizeof(all[0]));
        if (!write(&frame, sizeof(frame)) || !write(all.data(), all.size() * sizeof(all[0])) ||
            std::fflush(file) != 0) {
            failed = true;
            return fail("could not commit complete frame");
        }
        ++frames; last_step = step;
        return true;
    }
};

RelaxationOutput3D::RelaxationOutput3D() = default;
RelaxationOutput3D::~RelaxationOutput3D() = default;

bool RelaxationOutput3D::open(const std::string& path, const SimParams3D& params,
                             std::uint64_t start_step, std::uint64_t interval,
                             const std::string& metadata) {
    const std::uint16_t endian = 1;
    if (impl_ || path.empty() || !params.substrate_slab() || !interval ||
        params.num_cells <= 0 || params.Nx <= 0 || params.Ny <= 0 || params.Nz <= 0 ||
        !std::isfinite(params.dt) || params.dt <= 0 || metadata.empty() ||
        metadata.size() > 65536 || *reinterpret_cast<const unsigned char*>(&endian) != 1)
        return fail("invalid substrate relaxation-output configuration");
    impl_ = std::make_unique<Impl>();
    impl_->params = params; impl_->start_step = start_step;
    impl_->file = pf::open_new_binary_file(path);
    if (!impl_->file) return fail("cannot create new output; use a distinct path for every restart");
    relaxation::Header header{};
    std::memcpy(header.magic, "PF3DRLX1", 8);
    header.cells = params.num_cells; header.values = relaxation::kValues;
    header.start_step = start_step; header.dt = params.dt; header.interval = interval;
    header.nx = params.Nx; header.ny = params.Ny; header.nz = params.Nz;
    header.boundary_flags = params.boundary_flags;
    header.metadata_bytes = static_cast<std::uint32_t>(metadata.size());
    header.checksum = surface::crc32_update(0, &header, sizeof(header));
    header.checksum = surface::crc32_update(header.checksum, metadata.data(), metadata.size());
    return impl_->write(&header, sizeof(header)) && impl_->write(metadata.data(), metadata.size());
}

bool RelaxationOutput3D::capture(const std::vector<RelaxationDeviceView3D>& views,
                                std::uint64_t step) {
    if (!impl_) return fail("output is not open");
    const bool ok = impl_->capture(views, step);
    if (!ok) impl_->failed = true;
    return ok;
}

bool RelaxationOutput3D::close() {
    if (!impl_ || !impl_->file) return !impl_ || !impl_->failed;
    relaxation::End end{};
    std::memcpy(end.magic, "P3RLEND1", 8);
    end.frames = impl_->frames; end.last_step = impl_->last_step;
    bool ok = impl_->frames != 0 && impl_->write(&end, sizeof(end));
    if (std::fclose(impl_->file) != 0) ok = false;
    impl_->file = nullptr;
    impl_->failed = !ok;
    return ok || fail("segment did not close cleanly");
}

} // namespace pf3d
