#include "pf3d/surface_output.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#include <sys/stat.h>
#else
#include <unistd.h>
#endif
#ifdef PF_BOUNDARY_ZSTD
#include <zstd.h>
#endif

namespace pf3d {
namespace {

constexpr int kThreads = 256;
constexpr int kShards = 64;
constexpr int kTileX = 16, kTileY = 8, kTileZ = 8;
constexpr int kHaloX = kTileX + 2, kHaloY = kTileY + 2;
constexpr int kHaloWords = kHaloX * kHaloY * (kTileZ + 2);
constexpr std::size_t kBatchVertices = (64u * 1024u * 1024u) / sizeof(surface::Vertex);

bool fail(const char* message) {
    std::fprintf(stderr, "[surface3d] %s\n", message);
    return false;
}

bool gpu_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "[surface3d] %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

#define SURFACE_CUDA(call) do { if (!gpu_ok((call), #call)) return false; } while (false)

struct RestoreDevice {
    int device = 0;
    RestoreDevice() { cudaGetDevice(&device); }
    ~RestoreDevice() { cudaSetDevice(device); }
};

std::FILE* create_exclusive(const std::string& path) {
#ifdef _WIN32
    const int fd = _open(path.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY,
                         _S_IREAD | _S_IWRITE);
    if (fd < 0) return nullptr;
    std::FILE* file = _fdopen(fd, "wb");
    if (!file) _close(fd);
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
    if (fd < 0) return nullptr;
    std::FILE* file = fdopen(fd, "wb");
    if (!file) ::close(fd);
#endif
    return file;
}

struct ExtractArgs {
    const float* base;
    const float* const* promoted;
    const CellState3D* cells;
    CellSelection3D selection;
    CellFieldStorage3D storage;
    int base_edge, nz;
    bool bounded_z;
};

__device__ float sample(const float* field, const CellState3D& state,
                         int edge, int x, int y, int z,
                         const ExtractArgs& args) {
    if (x < 0 || y < 0 || z < 0 || x >= edge || y >= edge || z >= edge)
        return 0.0f;
    if (args.bounded_z && (state.origin_z + z < 0 || state.origin_z + z >= args.nz))
        return 0.0f;
    return field[args.storage.index(x, y, z, edge, state.origin_z)];
}

// Count and pack traverse identical disjoint vertex tiles. A vertex belongs to
// a crossing cube iff one of its 26 neighbours is on the other side of 0.5.
// Thus every corner of every crossing cube is saved once, at original precision.
template <bool Pack>
__global__ void extract_vertices(ExtractArgs args, int first_slot,
                                  std::uint64_t* counts,
                                  const std::uint64_t* offsets,
                                  std::uint64_t batch_begin, std::uint64_t batch_end,
                                  surface::Vertex* output, std::uint32_t* invalid) {
    const int slot = first_slot + static_cast<int>(blockIdx.y);
    const int cell_index = args.selection.cell_index(slot);
    const CellState3D state = args.cells[cell_index];
    const int edge = cell_support_edge(state, args.base_edge);
    const int side = edge + 2;
    const bool promoted = cell_is_promoted(state, args.base_edge);
    const float* field = promoted ? (args.promoted ? args.promoted[cell_index] : nullptr)
        : (args.base ? args.base + static_cast<std::size_t>(cell_index) *
            args.storage.words(args.base_edge) : nullptr);
    const std::size_t shard = static_cast<std::size_t>(slot) * kShards + blockIdx.x;
    if (!field) {
        if (threadIdx.x == 0) {
            atomicOr(invalid, 1u);
            if constexpr (!Pack) counts[shard] = 0;
        }
        return;
    }
    const int tiles_x = (side + kTileX - 1) / kTileX;
    const int tiles_y = (side + kTileY - 1) / kTileY;
    int first_z_tile = 0, last_z_tile = (side + kTileZ - 1) / kTileZ;
    if (args.bounded_z) {
        // Bound origin before subtraction, including for malformed extreme
        // coordinates. Only physical planes and their one-vertex shell matter.
        if (state.origin_z >= args.nz || state.origin_z <= -std::int64_t(edge)) {
            last_z_tile = 0;
        } else {
            const std::int64_t begin = state.origin_z < 0 ? -state.origin_z : 0;
            const std::int64_t end = std::min<std::int64_t>(edge, args.nz - state.origin_z);
            first_z_tile = static_cast<int>(begin) / kTileZ;
            last_z_tile = (static_cast<int>(end) + 2 + kTileZ - 1) / kTileZ;
        }
    }
    const int tiles_per_plane = tiles_x * tiles_y;
    const int tile_end = last_z_tile * tiles_per_plane;
    __shared__ unsigned char signs[kHaloWords];
    __shared__ unsigned warp_counts[kThreads / 32];
    __shared__ unsigned warp_offsets[kThreads / 32];
    __shared__ unsigned block_count;
    __shared__ std::uint64_t cursor;
    if (threadIdx.x == 0) cursor = 0;
    __syncthreads();
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x / 32u;
    for (int tile = first_z_tile * tiles_per_plane + static_cast<int>(blockIdx.x);
         tile < tile_end; tile += kShards) {
        const int tx = tile % tiles_x, ty = (tile / tiles_x) % tiles_y;
        const int tz = tile / tiles_per_plane;
        for (int q = threadIdx.x; q < kHaloWords; q += kThreads) {
            const int x = tx * kTileX + q % kHaloX - 2;
            const int y = ty * kTileY + (q / kHaloX) % kHaloY - 2;
            const int z = tz * kTileZ + q / (kHaloX * kHaloY) - 2;
            const float phi = sample(field, state, edge, x, y, z, args);
            if (!isfinite(phi)) atomicOr(invalid, 1u);
            signs[q] = phi >= 0.5f;
        }
        __syncthreads();
        for (int local = threadIdx.x; local < kTileX * kTileY * kTileZ; local += kThreads) {
            const int lx = local % kTileX, ly = (local / kTileX) % kTileY;
            const int lz = local / (kTileX * kTileY);
            const int px = tx * kTileX + lx, py = ty * kTileY + ly;
            const int pz = tz * kTileZ + lz;
            const int at = ((lz + 1) * kHaloY + ly + 1) * kHaloX + lx + 1;
            bool needed = false;
            if (px < side && py < side && pz < side) {
                const unsigned char here = signs[at];
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx)
                            needed |= signs[at + (dz * kHaloY + dy) * kHaloX + dx] != here;
            }
            const unsigned mask = __ballot_sync(0xffffffffu, needed);
            if (lane == 0) warp_counts[warp] = __popc(mask);
            __syncthreads();
            if (threadIdx.x == 0) {
                unsigned prefix = 0;
                for (int w = 0; w < kThreads / 32; ++w) {
                    warp_offsets[w] = prefix;
                    prefix += warp_counts[w];
                }
                block_count = prefix;
            }
            __syncthreads();
            if constexpr (Pack) {
                const std::uint64_t ordinal = offsets[shard] + cursor + warp_offsets[warp]
                    + __popc(mask & ((1u << lane) - 1u));
                if (needed && ordinal >= batch_begin && ordinal < batch_end) {
                    const std::uint64_t index = (static_cast<std::uint64_t>(pz) * side + py) * side + px;
                    surface::Vertex vertex{};
                    vertex.index_lo = static_cast<std::uint32_t>(index);
                    vertex.index_hi = static_cast<std::uint32_t>(index >> 32);
                    vertex.phi = sample(field, state, edge, px - 1, py - 1, pz - 1, args);
                    output[ordinal - batch_begin] = vertex;
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) cursor += block_count;
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        if constexpr (!Pack) counts[shard] = cursor;
        else if (cursor != counts[shard]) atomicOr(invalid, 2u);
    }
}

bool valid_cell(const CellState3D& state, int base_edge) {
    const int edge = cell_support_edge(state, base_edge);
    const auto low = std::numeric_limits<std::int64_t>::min();
    const auto high = std::numeric_limits<std::int64_t>::max();
    return state.global_id >= 0 && edge >= base_edge && edge <= 4096 && edge % 8 == 0
        && state.origin_x > low && state.origin_y > low && state.origin_z > low
        && state.origin_x <= high - edge && state.origin_y <= high - edge
        && state.origin_z <= high - edge && std::isfinite(state.gamma) && state.gamma > 0
        && std::isfinite(state.R_tgt) && state.R_tgt > 0
        && std::isfinite(state.v_A) && state.v_A >= 0
        && std::isfinite(state.polarity_x) && std::isfinite(state.polarity_y) && std::isfinite(state.polarity_z)
        && std::isfinite(state.velocity_x) && std::isfinite(state.velocity_y) && std::isfinite(state.velocity_z);
}

struct DeviceWorkspace {
    int device;
    std::uint64_t *counts = nullptr, *offsets = nullptr;
    std::uint32_t* invalid = nullptr;
    surface::Vertex *packed = nullptr, *host = nullptr;
    std::size_t count_capacity = 0, vertex_capacity = 0;
    explicit DeviceWorkspace(int id) : device(id) {}
    ~DeviceWorkspace() {
        RestoreDevice restore;
        cudaSetDevice(device);
        cudaFree(counts);
        cudaFree(offsets);
        cudaFree(invalid);
        cudaFree(packed);
        if (host) cudaFreeHost(host);
    }
    bool reserve_counts(std::size_t count) {
        if (count <= count_capacity) return true;
        SURFACE_CUDA(cudaFree(counts)); counts = nullptr;
        SURFACE_CUDA(cudaFree(offsets)); offsets = nullptr;
        count_capacity = 0;
        SURFACE_CUDA(cudaMalloc(reinterpret_cast<void**>(&counts), count * sizeof(*counts)));
        SURFACE_CUDA(cudaMalloc(reinterpret_cast<void**>(&offsets), count * sizeof(*offsets)));
        if (!invalid) SURFACE_CUDA(cudaMalloc(reinterpret_cast<void**>(&invalid), sizeof(*invalid)));
        count_capacity = count;
        return true;
    }
    bool reserve_vertices(std::size_t count) {
        if (count <= vertex_capacity) return true;
        SURFACE_CUDA(cudaFree(packed)); packed = nullptr;
        if (host) { SURFACE_CUDA(cudaFreeHost(host)); host = nullptr; }
        vertex_capacity = 0;
        SURFACE_CUDA(cudaMalloc(reinterpret_cast<void**>(&packed), count * sizeof(*packed)));
        SURFACE_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&host), count * sizeof(*host), cudaHostAllocDefault));
        vertex_capacity = count;
        return true;
    }
};

struct CapturePlan {
    SurfaceDeviceView3D view;
    DeviceWorkspace* workspace = nullptr;
    std::vector<CellState3D> states;
    std::vector<int> ids;
    std::vector<std::uint64_t> counts, offsets, cell_offsets;
    ExtractArgs args{};
};

}  // namespace

struct SurfaceOutput3D::Impl {
    std::FILE* file = nullptr;
    SimParams3D params{};
    bool compress = false, failed = false, closed = false;
    std::uint64_t frames = 0, last_step = 0, frame_crc = 0;
    std::vector<std::int64_t> identity;
    std::vector<std::unique_ptr<DeviceWorkspace>> workspaces;
    std::vector<unsigned char> encoded;

    ~Impl() { finish(false); }

    bool write(const void* data, std::size_t bytes, bool frame = false) {
        if (std::fwrite(data, 1, bytes, file) != bytes) {
            std::perror("[surface3d] write"); failed = true; return false;
        }
        if (frame) frame_crc = surface::crc32_update(static_cast<std::uint32_t>(frame_crc), data, bytes);
        return true;
    }

    bool flush() {
        if (std::fflush(file) == 0) return true;
        std::perror("[surface3d] flush"); failed = true; return false;
    }

    bool finish(bool complete) {
        if (closed) return !failed;
        if (file) {
            if (complete && !failed) {
                surface::EndMarker end{};
                std::memcpy(end.magic, "P3SEND1", 8);
                end.frames = frames;
                end.last_step = last_step;
                write(&end, sizeof(end));
            }
            if (std::fclose(file) != 0) { std::perror("[surface3d] close"); failed = true; }
            file = nullptr;
        }
        workspaces.clear();
        closed = true;
        return !failed;
    }

    DeviceWorkspace* workspace(int device) {
        for (auto& item : workspaces) if (item->device == device) return item.get();
        workspaces.emplace_back(std::make_unique<DeviceWorkspace>(device));
        return workspaces.back().get();
    }

    bool prepare(const SurfaceDeviceView3D& view, CapturePlan* plan) {
        if (!view.cells || view.base_edge <= 0 || view.base_edge > 4096 ||
            !view.selection.valid(params.num_cells) ||
            (view.storage.z_cap != 0 &&
             (!params.bounded_z() || view.storage.z_cap != params.Nz)))
            return fail("invalid owner view or storage geometry");
        SURFACE_CUDA(cudaSetDevice(view.device));
        plan->view = view;
        plan->workspace = workspace(view.device);
        const int selected = view.selection.selected_count(params.num_cells);
        if (selected > 65535) return fail("one device view exceeds the CUDA grid-y limit");
        plan->ids.resize(selected);
        if (view.selection.ids)
            SURFACE_CUDA(cudaMemcpyAsync(plan->ids.data(), view.selection.ids,
                selected * sizeof(int), cudaMemcpyDeviceToHost, view.stream));
        else std::iota(plan->ids.begin(), plan->ids.end(), 0);
        plan->states.resize(params.num_cells);
        SURFACE_CUDA(cudaMemcpyAsync(plan->states.data(), view.cells,
            plan->states.size() * sizeof(CellState3D), cudaMemcpyDeviceToHost, view.stream));
        SURFACE_CUDA(cudaStreamSynchronize(view.stream));
        for (int id : plan->ids)
            if (id < 0 || id >= params.num_cells || !valid_cell(plan->states[id], view.base_edge))
                return fail("invalid selected cell identity or geometry");
        plan->cell_offsets.resize(static_cast<std::size_t>(selected) + 1, 0);
        if (selected == 0) return true;
        plan->counts.resize(static_cast<std::size_t>(selected) * kShards);
        plan->offsets.resize(plan->counts.size());
        auto& ws = *plan->workspace;
        if (!ws.reserve_counts(plan->counts.size())) return false;
        plan->args = {view.base_phi, view.promoted_phi, view.cells, view.selection,
                      view.storage, view.base_edge, params.Nz, params.bounded_z()};
        SURFACE_CUDA(cudaMemsetAsync(ws.invalid, 0, sizeof(*ws.invalid), view.stream));
        extract_vertices<false><<<dim3(kShards, selected), kThreads, 0, view.stream>>>(
            plan->args, 0, ws.counts, nullptr, 0, 0, nullptr, ws.invalid);
        SURFACE_CUDA(cudaGetLastError());
        std::uint32_t invalid = 0;
        SURFACE_CUDA(cudaMemcpyAsync(plan->counts.data(), ws.counts,
            plan->counts.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, view.stream));
        SURFACE_CUDA(cudaMemcpyAsync(&invalid, ws.invalid, sizeof(invalid), cudaMemcpyDeviceToHost, view.stream));
        SURFACE_CUDA(cudaStreamSynchronize(view.stream));
        if (invalid) return fail("missing field or non-finite interface input");
        std::uint64_t prefix = 0;
        for (int slot = 0; slot < selected; ++slot) {
            const std::uint64_t side = cell_support_edge(plan->states[plan->ids[slot]], view.base_edge) + 2;
            std::uint64_t cell_count = 0;
            for (int shard = 0; shard < kShards; ++shard) {
                const std::size_t q = static_cast<std::size_t>(slot) * kShards + shard;
                plan->offsets[q] = prefix;
                if (plan->counts[q] > side * side * side - cell_count)
                    return fail("interface vertex count exceeds the padded grid");
                if (plan->counts[q] > std::numeric_limits<std::uint64_t>::max() - prefix)
                    return fail("interface prefix count overflow");
                cell_count += plan->counts[q];
                prefix += plan->counts[q];
            }
            plan->cell_offsets[slot + 1] = prefix;
        }
        SURFACE_CUDA(cudaMemcpyAsync(ws.offsets, plan->offsets.data(),
            plan->offsets.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice, view.stream));
        SURFACE_CUDA(cudaStreamSynchronize(view.stream));
        return true;
    }

    bool write_cell(const CellState3D& state, int base_edge, std::uint64_t count) {
        surface::Cell cell{};
        cell.id = state.global_id;
        cell.origin_x = state.origin_x;
        cell.origin_y = state.origin_y;
        cell.origin_z = state.origin_z;
        cell.brick_edge = cell_support_edge(state, base_edge);
        cell.vertices = count;
        cell.gamma = state.gamma;
        cell.active_speed = state.v_A;
        cell.radius = state.R_tgt;
        cell.polarity[0] = state.polarity_x; cell.polarity[1] = state.polarity_y; cell.polarity[2] = state.polarity_z;
        cell.velocity[0] = state.velocity_x; cell.velocity[1] = state.velocity_y; cell.velocity[2] = state.velocity_z;
        cell.tumbles = state.tumble_ctr; cell.flags = state.flags;
        return write(&cell, sizeof(cell), true);
    }

    bool write_chunk(const surface::Vertex* data, std::size_t count, std::uint64_t* stored_total) {
        const std::size_t raw_bytes = count * sizeof(*data);
        const void* stored = data;
        std::size_t stored_bytes = raw_bytes;
#ifdef PF_BOUNDARY_ZSTD
        if (compress) {
            encoded.resize(ZSTD_compressBound(raw_bytes));
            stored_bytes = ZSTD_compress(encoded.data(), encoded.size(), data, raw_bytes, 1);
            if (ZSTD_isError(stored_bytes)) return fail(ZSTD_getErrorName(stored_bytes));
            stored = encoded.data();
        }
#endif
        surface::ChunkHeader header{};
        std::memcpy(header.magic, "P3SCHK1", 8);
        header.vertices = count;
        header.stored_bytes = stored_bytes;
        header.checksum = surface::crc32_update(0, data, raw_bytes);
        *stored_total += stored_bytes;
        return write(&header, sizeof(header), true) && write(stored, stored_bytes, true);
    }

    bool pack_range(CapturePlan& plan, int first, int end,
                    std::uint64_t begin_vertex, std::uint64_t end_vertex) {
        auto& ws = *plan.workspace;
        SURFACE_CUDA(cudaSetDevice(plan.view.device));
        const auto count = static_cast<std::size_t>(end_vertex - begin_vertex);
        if (count > kBatchVertices || !ws.reserve_vertices(count)) return false;
        extract_vertices<true><<<dim3(kShards, end - first), kThreads, 0, plan.view.stream>>>(
            plan.args, first, ws.counts, ws.offsets, begin_vertex, end_vertex, ws.packed, ws.invalid);
        SURFACE_CUDA(cudaGetLastError());
        SURFACE_CUDA(cudaMemcpyAsync(ws.host, ws.packed, count * sizeof(surface::Vertex),
            cudaMemcpyDeviceToHost, plan.view.stream));
        std::uint32_t invalid = 0;
        SURFACE_CUDA(cudaMemcpyAsync(&invalid, ws.invalid, sizeof(invalid),
            cudaMemcpyDeviceToHost, plan.view.stream));
        SURFACE_CUDA(cudaStreamSynchronize(plan.view.stream));
        return invalid == 0 || fail("interface count/pack disagreement or invalid field");
    }

    bool write_plan(CapturePlan& plan, std::uint64_t* stored_total) {
        const int selected = static_cast<int>(plan.ids.size());
        for (int first = 0; first < selected;) {
            const auto begin = plan.cell_offsets[first];
            const auto cell_end = plan.cell_offsets[first + 1];
            if (cell_end - begin > kBatchVertices) {
                if (!write_cell(plan.states[plan.ids[first]], plan.view.base_edge, cell_end - begin)) return false;
                for (auto at = begin; at < cell_end;) {
                    const auto stop = std::min<std::uint64_t>(cell_end, at + kBatchVertices);
                    if (!pack_range(plan, first, first + 1, at, stop) ||
                        !write_chunk(plan.workspace->host, static_cast<std::size_t>(stop - at), stored_total))
                        return false;
                    at = stop;
                }
                ++first;
                continue;
            }
            int end = first + 1;
            while (end < selected && plan.cell_offsets[end + 1] - begin <= kBatchVertices) ++end;
            const auto stop = plan.cell_offsets[end];
            if (stop > begin && !pack_range(plan, first, end, begin, stop)) return false;
            for (int slot = first; slot < end; ++slot) {
                const auto count = plan.cell_offsets[slot + 1] - plan.cell_offsets[slot];
                if (!write_cell(plan.states[plan.ids[slot]], plan.view.base_edge, count)) return false;
                if (count && !write_chunk(plan.workspace->host + (plan.cell_offsets[slot] - begin),
                                          static_cast<std::size_t>(count), stored_total)) return false;
            }
            first = end;
        }
        return true;
    }

    bool capture(const std::vector<SurfaceDeviceView3D>& views, std::uint64_t step, double time) {
        RestoreDevice restore;
        if (!file || closed || failed || views.empty() || !std::isfinite(time) || time < 0 ||
            time != static_cast<double>(step) * params.dt || (frames && step <= last_step))
            return fail("invalid capture state or accepted-step clock");
        const auto started = std::chrono::steady_clock::now();
        std::vector<CapturePlan> plans(views.size());
        std::unordered_set<int> devices;
        std::vector<std::int64_t> ids;
        std::uint64_t vertices = 0;
        for (std::size_t v = 0; v < views.size(); ++v) {
            if (!devices.insert(views[v].device).second) return fail("duplicate device view");
            if (!prepare(views[v], &plans[v])) return false;
            if (plans[v].cell_offsets.back() > std::numeric_limits<std::uint64_t>::max() - vertices)
                return fail("interface frame count overflow");
            vertices += plans[v].cell_offsets.back();
            for (int id : plans[v].ids) ids.push_back(plans[v].states[id].global_id);
        }
        std::sort(ids.begin(), ids.end());
        if (ids.size() != static_cast<std::size_t>(params.num_cells) ||
            std::adjacent_find(ids.begin(), ids.end()) != ids.end() ||
            (frames && ids != identity)) return fail("owner views must cover every persistent cell exactly once");
        identity = ids;
        surface::FrameHeader header{};
        std::memcpy(header.magic, "P3SFRM1", 8);
        header.step = step; header.time = time; header.cells = ids.size(); header.vertices = vertices;
        frame_crc = 0;
        if (!write(&header, sizeof(header), true)) return false;
        std::uint64_t stored_bytes = 0;
        for (auto& plan : plans) if (!write_plan(plan, &stored_bytes)) return false;
        surface::FrameFooter footer{};
        std::memcpy(footer.magic, "P3SCMT1", 8);
        footer.step = step; footer.vertices = vertices; footer.checksum = frame_crc;
        if (!write(&footer, sizeof(footer)) || !flush()) return false;
        ++frames; last_step = step;
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::printf("[surface3d] step %llu: %llu vertices, %.3f MiB raw, %.3f MiB stored, %.6f s capture+write\n",
            static_cast<unsigned long long>(step), static_cast<unsigned long long>(vertices),
            static_cast<double>(vertices * sizeof(surface::Vertex)) / 1048576.0,
            static_cast<double>(stored_bytes) / 1048576.0, seconds);
        return true;
    }
};

SurfaceOutput3D::SurfaceOutput3D() = default;
SurfaceOutput3D::~SurfaceOutput3D() = default;

bool SurfaceOutput3D::open(const std::string& path, const SimParams3D& params,
                          std::uint64_t interval, bool compress) {
    const std::uint16_t endian = 1;
    if (impl_ || path.empty() || interval == 0 || params.num_cells <= 0 ||
        params.Nx <= 0 || params.Ny <= 0 || params.Nz <= 0 ||
        !(params.periodic_xyz() || params.bounded_z()) ||
        !std::isfinite(params.dt) || params.dt <= 0 ||
        !std::isfinite(params.tau) || params.tau <= 0 ||
        *reinterpret_cast<const unsigned char*>(&endian) != 1)
        return fail("invalid surface output configuration");
#ifndef PF_BOUNDARY_ZSTD
    if (compress) return fail("zstd unavailable; select --surface-compression none");
#endif
    try {
        impl_ = std::make_unique<Impl>();
        auto& out = *impl_;
        out.params = params; out.compress = compress;
        out.file = create_exclusive(path);
        if (!out.file) { out.failed = true; std::perror("[surface3d] create new file"); return false; }
        surface::FileHeader h{};
        std::memcpy(h.magic, "PF3SUR1", 8);
        h.version = 1; h.header_bytes = sizeof(h); h.cells = params.num_cells;
        h.boundary_flags = params.boundary_flags; h.codec = compress ? 1 : 0;
        h.vertex_bytes = sizeof(surface::Vertex); h.cell_bytes = sizeof(surface::Cell);
        h.frame_bytes = sizeof(surface::FrameHeader);
        h.nx = params.Nx; h.ny = params.Ny; h.nz = params.Nz;
        h.dx = params.dx; h.dy = params.dy; h.dz = params.dz;
        h.dt = params.dt; h.tau = params.tau; h.level = 0.5; h.interval = interval;
        h.channel_height = params.channel_height; h.channel_padding = params.channel_padding;
        h.wall_kappa = params.wall_kappa; h.wall_width = params.wall_width; h.lambda = params.lambda;
        h.classification = 1; h.shell = 1;
        h.checksum = surface::crc32_update(0, &h, sizeof(h));
        return out.write(&h, sizeof(h)) && out.flush();
    } catch (const std::exception& error) {
        if (impl_) impl_->failed = true;
        return fail(error.what());
    }
}

bool SurfaceOutput3D::capture(const std::vector<SurfaceDeviceView3D>& views,
                             std::uint64_t step, double time) {
    if (!impl_) return fail("capture requires an open surface stream");
    try { if (impl_->capture(views, step, time)) return true; }
    catch (const std::exception& error) { fail(error.what()); }
    impl_->failed = true;
    return false;
}

bool SurfaceOutput3D::close(bool complete) { return !impl_ || impl_->finish(complete); }

#undef SURFACE_CUDA
}  // namespace pf3d
