/*
 * Cellular MPF
 *
 * Copyright (C) 2026 Steven A. Silber and Mikko Karttunen
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 * See LICENSE.txt and COPYING for the license terms.
 */

#include "pf3d/two_gpu_sim.cuh"
#include "pf3d/bounded_z_map.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <tuple>

namespace pf3d {
namespace {

bool checked(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) return true;
    std::fprintf(stderr, "[two-gpu] %s: %s\n", operation,
                 cudaGetErrorString(result));
    return false;
}

bool fail(const char* message) {
    std::fprintf(stderr, "[two-gpu] %s\n", message);
    return false;
}

// A fixed spatial cut balances live allocated tiles without per-step migration
// decisions. Tile counts include the allocated height inside the domain.
int balanced_y_cut(const std::vector<CellState3D>& cells, int base_edge,
                   const TwoRankSubstrateGeometry& geometry) {
    std::vector<std::pair<int, double>> weights;
    double total = 0.0, original_left = 0.0;
    for (const auto& cell : cells) {
        const int edge = cell_support_edge(cell, base_edge);
        const auto map = BoundedZMap::make(cell.origin_z, edge, geometry.nz, true);
        const int ztiles = (map.hi + kBrickZ - 1) / kBrickZ - map.lo / kBrickZ;
        const double weight = static_cast<double>((edge + kBrickX - 1) / kBrickX) *
            ((edge + kBrickY - 1) / kBrickY) * ztiles;
        const int midpoint = static_cast<int>((cell.origin_y % geometry.ny +
            geometry.ny + edge / 2) % geometry.ny);
        weights.emplace_back(midpoint, weight);
        total += weight;
        if (midpoint < geometry.ny / 2) original_left += weight;
    }
    std::sort(weights.begin(), weights.end());
    const double original_cost = std::max(original_left, total - original_left);
    double left = 0.0, best_cost = original_cost;
    int best = geometry.ny / 2;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        left += weights[i].second;
        if (i + 1 < weights.size() && weights[i].first == weights[i + 1].first)
            continue;
        const int cut = weights[i].first + 1;
        const double cost = std::max(left, total - left);
        if (cut < geometry.ny && cost < best_cost) {
            best_cost = cost;
            best = cut;
        }
    }
    return best_cost < original_cost * 0.97 ? best : geometry.ny / 2;
}

// Quantized contributions are nonnegative integers. Joining disjoint cell
// cohorts therefore preserves the single-device sum exactly unless it overflows.
__global__ void add_rows(std::uint32_t* sum, const std::uint32_t* received,
                         int pitch, int ny, int lo, int rows, int zlo, int planes,
                         std::uint32_t* flags) {
    const std::size_t band_words = static_cast<std::size_t>(pitch / 4) * rows;
    // Walk contiguous rows within each plane without per-word division/modulo.
    for (std::size_t z = blockIdx.y; z < static_cast<std::size_t>(planes); z += gridDim.y) {
        const std::size_t base = ((z + zlo) * ny + lo) * pitch;
        auto* out = reinterpret_cast<uint4*>(sum + base);
        const auto* in = reinterpret_cast<const uint4*>(received + base);
        for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                             threadIdx.x; i < band_words;
             i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
            const uint4 b = in[i];
            if ((b.x | b.y | b.z | b.w) == 0u) continue;
            const uint4 a = out[i];
            if (b.x > 0xffffffffu - a.x) atomicAdd(flags + FLAG3D_S_OVERFLOW, 1u);
            if (b.y > 0xffffffffu - a.y) atomicAdd(flags + FLAG3D_S_OVERFLOW, 1u);
            if (b.z > 0xffffffffu - a.z) atomicAdd(flags + FLAG3D_S_OVERFLOW, 1u);
            if (b.w > 0xffffffffu - a.w) atomicAdd(flags + FLAG3D_S_OVERFLOW, 1u);
            out[i] = make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
        }
    }
}

}  // namespace

TwoGpuSim3D::~TwoGpuSim3D() {
    for (int r = 0; r < 2; ++r) {
        if (!sim_[r]) continue;
        sim_[r]->distributed_step_ = {};
        sim_[r]->gather_state_ = {};
        sim_[r]->prepare_trajectory_ = {};
        sim_[r]->prepare_compaction_ = {};
        sim_[r]->capture_surfaces_ = {};
        sim_[r]->capture_relaxation_ = {};
        if (!select(r)) continue;
        checked(cudaStreamSynchronize(sim_[r]->stream_), "drain owned work");
        if (exchange_streams_[r])
            checked(cudaStreamSynchronize(exchange_streams_[r]), "drain peer copies");
    }
    // Peer copies may reference either allocation. Drain both streams before
    // releasing either rank's buffers or cross-device synchronization events.
    for (int r = 0; r < 2; ++r) {
        if (!select(r)) continue;
        if (exchange_ready_[r])
            checked(cudaEventDestroy(exchange_ready_[r]), "destroy exchange-ready event");
        if (exchange_sent_[r])
            checked(cudaEventDestroy(exchange_sent_[r]), "destroy exchange-sent event");
        if (exchange_joined_[r])
            checked(cudaEventDestroy(exchange_joined_[r]), "destroy exchange-joined event");
        if (exchange_streams_[r])
            checked(cudaStreamDestroy(exchange_streams_[r]), "destroy exchange stream");
        cudaFree(d_ids_[r]);
        cudaFree(d_early_ids_[r]);
        cudaFree(d_deferred_ids_[r]);
        cudaFree(d_receive_[r]);
        cudaFree(d_clear_tile_tops_[r]);
    }
    if (h_clear_tile_tops_)
        checked(cudaFreeHost(h_clear_tile_tops_), "release aggregate bounds");
    for (HostSnapshot* snapshot : {&downloads_[0], &downloads_[1], &upload_})
        if (snapshot->cells)
            checked(cudaFreeHost(snapshot->cells), "release pinned metadata");
}

bool TwoGpuSim3D::select(int rank) const {
    return sim_[rank] && checked(cudaSetDevice(sim_[rank]->options_.device),
                                 "select device");
}

bool TwoGpuSim3D::wait_all() const {
    for (int r = 0; r < 2; ++r)
        if (!select(r) || !checked(cudaStreamSynchronize(sim_[r]->stream_),
                                   "wait for owned work")) return false;
    return select(0);
}

bool TwoGpuSim3D::preflight(const SimParams3D& params, RunOptions3D* options) {
    if (!params.bounded_z())
        return fail("two-device execution requires --geometry slab or channel");
    if (params.hard_wall_channel() &&
        (!params.resolved_wall_channel() || !params.wall_repulsion_active()))
        return fail("two-device channel execution requires positive resolved steric walls");
    if (options->storage_mode != StorageMode3D::Auto &&
        options->storage_mode != StorageMode3D::Throughput)
        return fail("the two-device prototype requires throughput storage");
    int count = 0;
    if (!checked(cudaGetDeviceCount(&count), "count devices")) return false;
    const int first = options->device;
    if (first < 0 || first >= count || peer_device_ < 0 || peer_device_ >= count ||
        first == peer_device_)
        return fail("two distinct visible CUDA devices are required");
    cudaDeviceProp properties[2]{};
    const int devices[2] = {first, peer_device_};
    for (int r = 0; r < 2; ++r) {
        int access = 0;
        if (!checked(cudaGetDeviceProperties(&properties[r], devices[r]), "device properties") ||
            !checked(cudaDeviceCanAccessPeer(&access, devices[r], devices[1-r]), "peer capability"))
            return false;
        if (!access) return fail("bidirectional direct CUDA peer access is required");
        if (!checked(cudaSetDevice(devices[r]), "select peer device")) return false;
        const cudaError_t enabled = cudaDeviceEnablePeerAccess(devices[1-r], 0);
        if (enabled == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        else if (!checked(enabled, "enable peer access")) return false;
    }
    if (properties[0].major != properties[1].major ||
        properties[0].minor != properties[1].minor ||
        properties[0].multiProcessorCount != properties[1].multiProcessorCount)
        return fail("this prototype requires two matching GPUs");
    options->storage_mode = StorageMode3D::Throughput;
    return checked(cudaSetDevice(first), "select primary device");
}

bool TwoGpuSim3D::init_fresh(const SimParams3D& params, const RunOptions3D& options) {
    if (sim_[0]) return fail("construct a new coordinator to initialize another run");
    RunOptions3D resolved = options;
    if (!preflight(params, &resolved)) return false;
    sim_[0] = std::make_unique<Sim3D>();
    return sim_[0]->init_fresh(params, resolved) && initialize();
}

bool TwoGpuSim3D::init_checkpoint(const CheckpointMeta3D& meta,
                                 const std::string& path,
                                 const RunOptions3D& options, double end) {
    if (sim_[0]) return fail("construct a new coordinator to initialize another run");
    RunOptions3D resolved = options;
    if (!preflight(meta.params, &resolved)) return false;
    sim_[0] = std::make_unique<Sim3D>();
    return sim_[0]->init_checkpoint(meta, path, resolved, end) && initialize();
}

bool TwoGpuSim3D::read_states(int rank) {
    auto& sim = *sim_[rank];
    return select(rank) && checked(cudaMemcpyAsync(downloads_[rank].cells, sim.d_cells_,
        canonical_.size() * sizeof(CellState3D), cudaMemcpyDeviceToHost,
        sim.stream_), "read cell metadata") &&
        checked(cudaStreamSynchronize(sim.stream_), "wait for cell metadata");
}

bool TwoGpuSim3D::copy_field(int id, int source, int destination) {
    if (source == destination) return true;
    auto& src = *sim_[source];
    auto& dst = *sim_[destination];
    const int edge = cell_support_edge(canonical_[static_cast<std::size_t>(id)], src.B_);
    std::size_t words = 0, bytes = 0;
    if (src.B_ != dst.B_ || src.field_storage_.z_cap != dst.field_storage_.z_cap ||
        edge < src.B_ || !src.field_storage_.checked_words(edge, &words) ||
        !checked_mul_size(words, sizeof(float), &bytes))
        return fail("invalid owned-cell storage size");
    const bool promoted = edge > src.B_;
    const float* from = promoted
        ? src.h_promoted_phi_[src.current_promoted_phi_index_][id]
        : src.current_phi() + static_cast<std::size_t>(id) * words;
    float* to = promoted
        ? dst.h_promoted_phi_[dst.current_promoted_phi_index_][id]
        : dst.current_phi() + static_cast<std::size_t>(id) * words;
    if (!from || !to) return fail("missing storage for an owned cell");
    return select(source) && checked(cudaMemcpyPeerAsync(to, dst.options_.device,
        from, src.options_.device, bytes, src.stream_),
        "transfer owned phase field");
}

bool TwoGpuSim3D::clone_peer() {
    auto& primary = *sim_[0];
    // Rebuild from the accepted canonical state after adaptive storage growth.
    // The failed timestep is retried only after the replica is ready.
    sim_[1].reset();
    sim_[1] = std::make_unique<Sim3D>();
    auto& peer = *sim_[1];
    RunOptions3D options = primary.options_;
    options.device = peer_device_;
    options.trajectory_path.clear();
    options.boundary_path.clear();
    options.boundary_interval = 0;
    options.boundary_compress = false;
    options.boundary_projection = boundary::Projection::Maximum;
    options.surface_path.clear();
    options.surface_interval = 0;
    options.surface_compress = false;
    options.relaxation_path.clear();
    options.relaxation_interval = 0;
    options.relaxation_dense_every = 0;
    options.checkpoint_dir.clear();
    options.final_checkpoint = false;
    options.measure_shards = primary.measurement_shards_;
    options.measure_shards_supplied = true;
    if (!peer.allocate(primary.params_, options, primary.B_, false,
                       primary.measurement_shards_)) return false;
    CheckpointMeta3D meta{};
    for (const auto& cell : canonical_)
        meta.storage_edges.push_back(cell.storage_edge > 0 ?
            static_cast<int>(cell.storage_edge) : primary.B_);
    if (!peer.install_checkpoint_promotions(meta)) return false;
    // The exchange buffers survive replica reconstruction. allocate() already
    // observes their resident bytes; only this not-yet-allocated ID list needs
    // reserving again after adaptive recovery.
    if (d_receive_[1]) {
        const std::size_t bytes = canonical_.size() * sizeof(int);
        if (bytes > peer.adaptive_budget_remaining_)
            return fail("promoted owner list exceeds the replica memory budget");
        peer.adaptive_budget_remaining_ -= bytes;
        peer.required_device_bytes_ += bytes;
    }
    for (int id = 0; id < primary.params_.num_cells; ++id)
        if (!copy_field(id, 0, 1)) return false;
    if (!wait_all()) return false;
    peer.steps_done_ = primary.steps_done_;
    peer.recovery_events_ = primary.recovery_events_;
    peer.volume_current_ = primary.volume_current_;
    peer.surface_current_ = primary.surface_current_;
    if (!select(1) || !checked(cudaMemcpy(peer.d_step_, &peer.steps_done_,
         sizeof(peer.steps_done_), cudaMemcpyHostToDevice), "replicate accepted step"))
        return false;
    mirrored_recoveries_ = primary.recovery_events_;
    return select(0);
}

bool TwoGpuSim3D::synchronize_peer_growth() {
    auto& primary = *sim_[0];
    auto& peer = *sim_[1];
    if (peer.B_ != primary.B_ ||
        peer.field_storage_.z_cap != primary.field_storage_.z_cap ||
        peer.step_pending_ || primary.step_pending_ ||
        peer.phi_buffers_ != 2 || peer.S_buffers_ != 2)
        return clone_peer();
    if (!read_states(1)) return false;
    std::vector<CellState3D> peer_states(downloads_[1].cells,
                                       downloads_[1].cells + canonical_.size());
    std::vector<std::pair<int, int>> growth;
    std::size_t temporary_bytes = 0;
    for (std::size_t id = 0; id < canonical_.size(); ++id) {
        const int old_edge = cell_support_edge(peer_states[id], peer.B_);
        const int new_edge = cell_support_edge(canonical_[id], primary.B_);
        // Nonowner fields can be stale. Growth preserves them without reading
        // beyond their allocation, but they cannot certify a lossless crop.
        if (old_edge < peer.B_ || new_edge < old_edge ||
            peer_states[id].global_id != canonical_[id].global_id)
            return clone_peer();
        if (new_edge > old_edge) {
            std::size_t bytes = 0, pair_bytes = 0;
            if (!peer.field_storage_.checked_bytes(new_edge, &bytes) ||
                !checked_mul_size(bytes, 2, &pair_bytes))
                return fail("invalid peer replacement storage size");
            // A full rebuild can fit when retaining old and new bricks at
            // once would exceed the configured temporary-allocation budget.
            if (pair_bytes > peer.adaptive_budget_remaining_ - temporary_bytes)
                return clone_peer();
            temporary_bytes += pair_bytes;
            growth.emplace_back(static_cast<int>(id), new_edge);
        }
    }

    peer.selection_ = {};
    peer.owned_promoted_count_ = -1;
    if (!select(1) || !peer.resize_cells(growth, &peer_states)) return false;

    // Recovery may remeasure or retry more than the resized cells. Refresh
    // every future peer owner's accepted field, not merely the growth batch.
    // Other replicas remain unused until an ownership transfer copies them.
    std::size_t refreshed = 0;
    for (std::size_t id = 0; id < canonical_.size(); ++id) {
        const auto& cell = canonical_[id];
        const AllocatedBrick brick{cell.origin_y, cell.origin_z, cell.storage_edge};
        int owner = -1;
        if (!cell_owner_rank(brick, primary.B_, geometry_, &owner))
            return fail("invalid accepted cell ownership during peer growth");
        if (owner == 1) {
            if (!copy_field(static_cast<int>(id), 0, 1)) return false;
            ++refreshed;
        }
    }
    if (!wait_all()) return false;
    peer.steps_done_ = primary.steps_done_;
    peer.recovery_events_ = primary.recovery_events_;
    peer.volume_current_ = primary.volume_current_;
    peer.surface_current_ = primary.surface_current_;
    if (!select(1) ||
        !checked(cudaMemcpy(peer.d_step_, &peer.steps_done_,
                            sizeof(peer.steps_done_), cudaMemcpyHostToDevice),
                 "replicate accepted step after peer growth") ||
        !checked(cudaMemset(peer.d_support_requests_, 0,
                            canonical_.size() * sizeof(std::uint32_t)),
                 "clear resolved peer support requests")) return false;
    mirrored_recoveries_ = primary.recovery_events_;
    std::printf("[two-gpu] synchronized peer growth: %zu resized, %zu accepted "
                "fields refreshed; base/workspace allocations retained\n",
                growth.size(), refreshed);
    return select(0);
}

bool TwoGpuSim3D::initialize() {
    auto& primary = *sim_[0];
    const auto& layout = primary.layout_;
    geometry_ = {layout.nx, layout.ny, layout.nz, layout.pitch_x, layout.boundary_flags};
    if (validate(geometry_) != TwoRankValidation::Ok)
        return fail("unsupported bounded-z exchange geometry");
    canonical_.resize(static_cast<std::size_t>(primary.params_.num_cells));
    const std::size_t cell_bytes = canonical_.size() * sizeof(CellState3D);
    for (HostSnapshot* snapshot : {&downloads_[0], &downloads_[1], &upload_}) {
        if (!checked(cudaHostAlloc(reinterpret_cast<void**>(&snapshot->cells),
                     cell_bytes + sizeof(flags_), cudaHostAllocPortable),
                     "allocate pinned metadata")) return false;
        snapshot->flags = reinterpret_cast<std::uint32_t*>(
            snapshot->cells + canonical_.size());
    }
    if (!read_states(0)) return false;
    std::copy_n(downloads_[0].cells, canonical_.size(), canonical_.begin());
    geometry_.y_cut = balanced_y_cut(canonical_, primary.B_, geometry_);
    std::printf("[two-gpu] work-balanced y partition at %d/%d\n",
                geometry_.y_cut, geometry_.ny);
    if (!clone_peer()) return false;
    const std::size_t receive_bytes = primary.S_words_ * sizeof(std::uint32_t);
    const std::size_t id_bytes = canonical_.size() * sizeof(int);
    if (!primary.options_.strict) {
        clear_tile_count_ = static_cast<std::size_t>(
            (geometry_.pitch_x + kAggregateClearTile - 1) / kAggregateClearTile) *
            ((geometry_.ny + kAggregateClearTile - 1) / kAggregateClearTile);
        if (!checked(cudaHostAlloc(reinterpret_cast<void**>(&h_clear_tile_tops_),
                     clear_tile_count_ * sizeof(int), cudaHostAllocPortable),
                     "allocate pinned aggregate bounds")) return false;
    }
    for (int r = 0; r < 2; ++r) {
        auto& sim = *sim_[r];
        const std::size_t reserved = receive_bytes + 4 * id_bytes +
            clear_tile_count_ * sizeof(int);
        if (reserved > sim.adaptive_budget_remaining_)
            return fail("peer-exchange workspace exceeds the configured memory budget");
        if (!select(r) ||
            !checked(cudaMalloc(reinterpret_cast<void**>(&d_receive_[r]), receive_bytes),
                     "allocate peer receive workspace") ||
            !checked(cudaMalloc(reinterpret_cast<void**>(&d_ids_[r]), id_bytes),
                     "allocate owner list") ||
            !checked(cudaMalloc(reinterpret_cast<void**>(&d_early_ids_[r]), id_bytes),
                     "allocate early measurement IDs") ||
            !checked(cudaMalloc(reinterpret_cast<void**>(&d_deferred_ids_[r]), id_bytes),
                     "allocate deferred measurement IDs") ||
            !checked(cudaEventCreateWithFlags(&exchange_ready_[r], cudaEventDisableTiming),
                     "create exchange-ready event") ||
            !checked(cudaEventCreateWithFlags(&exchange_sent_[r], cudaEventDisableTiming),
                     "create exchange-sent event") ||
            !checked(cudaEventCreateWithFlags(&exchange_joined_[r], cudaEventDisableTiming),
                     "create exchange-joined event") ||
            !checked(cudaStreamCreateWithFlags(&exchange_streams_[r], cudaStreamNonBlocking),
                     "create exchange stream")) return false;
        sim.adaptive_budget_remaining_ -= reserved;
        sim.required_device_bytes_ += reserved;
        if (clear_tile_count_ && !checked(cudaMalloc(
            reinterpret_cast<void**>(&d_clear_tile_tops_[r]), clear_tile_count_ * sizeof(int)),
            "allocate aggregate bounds")) return false;
    }
    if (!select(0) || !checked(cudaMemcpy(flags_.data(), primary.d_flags_,
         sizeof(flags_), cudaMemcpyDeviceToHost), "read initial integrity flags")) return false;
    primary.distributed_step_ = [this] { return distributed_step(); };
    primary.gather_state_ = [this] { return gather_state(); };
    primary.prepare_trajectory_ = [this] { return prepare_trajectory(); };
    primary.prepare_compaction_ = [this] { return prepare_compaction(); };
    primary.capture_surfaces_ = [this](SurfaceOutput3D& output) {
        return capture_surfaces(output);
    };
    primary.capture_relaxation_ = [this](RelaxationOutput3D& output) {
        return capture_relaxation(output);
    };
    std::printf("[two-gpu] devices %d,%d; y-partitioned whole cells; "
                "replicated storage; base measurement shards %d\n",
                primary.options_.device, peer_device_, primary.measurement_shards_);
    return select(0);
}

bool TwoGpuSim3D::enqueue_state_broadcast() {
    // Callers must have drained previous uploads before reusing this pinned
    // buffer. It stays separate from canonical_ while the CPU replans ownership.
    std::copy(canonical_.begin(), canonical_.end(), upload_.cells);
    for (int r = 0; r < 2; ++r)
        if (!select(r) || !checked(cudaMemcpyAsync(sim_[r]->d_cells_, upload_.cells,
             canonical_.size() * sizeof(CellState3D), cudaMemcpyHostToDevice,
             sim_[r]->stream_), "replicate accepted metadata")) return false;
    return select(0);
}

bool TwoGpuSim3D::configure_measurement_order() {
    for (int r = 0; r < 2; ++r) {
        auto& sim = *sim_[r];
        if (!select(r)) return false;
        TileExchangeRegion3D region{};
        const auto& plan = plans_[r];
        region.y_count = plan.bands.rows.count;
        for (int b = 0; b < region.y_count; ++b) {
            region.y_lo[b] = plan.bands.rows.pieces[b].lo;
            region.y_hi[b] = plan.bands.rows.pieces[b].hi;
        }
        region.z_lo = plan.z.lo;
        region.z_hi = plan.z.hi;
        std::vector<int> early, deferred;
        for (int id : ids_[r]) {
            const auto& cell = canonical_[id];
            const int B = cell_support_edge(cell, sim.B_);
            const int oy = static_cast<int>((cell.origin_y % geometry_.ny + geometry_.ny) % geometry_.ny);
            const bool boundary = tile_intersects_exchange_canonical(
                region, geometry_.ny, oy, cell.origin_z, 0, 0, B, B);
            // Keep promoted measurements concurrent with the first base cohort.
            (B > sim.B_ || boundary ? early : deferred).push_back(id);
        }
        if (!early.empty() && !checked(cudaMemcpy(d_early_ids_[r], early.data(),
                early.size() * sizeof(int), cudaMemcpyHostToDevice), "upload early IDs")) return false;
        if (!deferred.empty() && !checked(cudaMemcpy(d_deferred_ids_[r], deferred.data(),
                deferred.size() * sizeof(int), cudaMemcpyHostToDevice), "upload deferred IDs")) return false;
        sim.early_selection_ = {d_early_ids_[r], static_cast<int>(early.size())};
        sim.deferred_selection_ = {d_deferred_ids_[r], static_cast<int>(deferred.size())};
    }
    return select(0);
}

bool TwoGpuSim3D::configure_ownership(bool canonical_fields) {
    if (!canonical_fields && planned_bricks_.size() == canonical_.size()) {
        bool unchanged = true;
        for (std::size_t i = 0; i < canonical_.size(); ++i) {
            const auto& cell = canonical_[i];
            const auto& brick = planned_bricks_[i];
            unchanged = unchanged && cell.origin_y == brick.origin_y &&
                cell.origin_z == brick.origin_z && cell.storage_edge == brick.storage_edge;
        }
        if (unchanged) return select(0);
    }
    std::vector<AllocatedBrick> bricks;
    std::vector<int> next_owners;
    bricks.reserve(canonical_.size());
    next_owners.reserve(canonical_.size());
    bool migrated = false;
    bool storage_changed = planned_bricks_.size() != canonical_.size();
    for (std::size_t i = 0; i < canonical_.size(); ++i) {
        const auto& cell = canonical_[i];
        const AllocatedBrick brick{cell.origin_y, cell.origin_z, cell.storage_edge};
        int rank = -1;
        if (!cell_owner_rank(brick, sim_[0]->B_, geometry_, &rank))
            return fail("invalid accepted cell ownership");
        bricks.push_back(brick);
        next_owners.push_back(rank);
        if (!storage_changed && cell.storage_edge != planned_bricks_[i].storage_edge)
            storage_changed = true;
        if (!owners_.empty() && owners_[i] != rank) {
            if (!copy_field(static_cast<int>(i), canonical_fields ? 0 : owners_[i], rank))
                return false;
            migrated = true;
        }
    }
    for (int r = 0; r < 2; ++r) {
        plans_[r] = build_seam_exchange_plan(geometry_, sim_[r]->B_, r, bricks.data(),
                                             bricks.size(), sizeof(std::uint32_t));
        if (plans_[r].status != SeamPlanStatus::Ok) return fail("invalid seam plan");
        // An owner's subdomain plus the maximum brick overhang encloses both
        // its scatter writes and every incoming seam row, without a field cutoff.
        const auto owned = rank_ownership(geometry_, r);
        // Tightening the transferred rows does not reduce this allocated-edge bound.
        const auto halo = plans_[r].bands.half_width;
        const auto lo = owned.y_lo - halo, hi = owned.y_hi + halo;
        TileExchangeRegion3D rows{};
        rows.z_hi = geometry_.nz;
        if (hi - lo < geometry_.ny) {
            if (lo < 0) {
                rows.y_count = 2;
                rows.y_lo[0] = 0; rows.y_hi[0] = static_cast<int>(hi);
                rows.y_lo[1] = static_cast<int>(lo + geometry_.ny);
                rows.y_hi[1] = geometry_.ny;
            } else if (hi > geometry_.ny) {
                rows.y_count = 2;
                rows.y_lo[0] = 0; rows.y_hi[0] = static_cast<int>(hi - geometry_.ny);
                rows.y_lo[1] = static_cast<int>(lo); rows.y_hi[1] = geometry_.ny;
            } else {
                rows.y_count = 1;
                rows.y_lo[0] = static_cast<int>(lo); rows.y_hi[0] = static_cast<int>(hi);
            }
        }
        sim_[r]->aggregate_write_rows_ = rows;
    }
    if (sim_[0]->layout_.bounded_z()) {
        int planes = 0;
        for (const auto& cell : canonical_) {
            const auto top = cell.origin_z + cell_support_edge(cell, sim_[0]->B_);
            planes = std::max(planes, static_cast<int>(
                std::min<std::int64_t>(geometry_.nz, std::max<std::int64_t>(0, top))));
        }
        // Allocation changes can expand the dirty range. Until the fast path
        // supplies its measured input bound, retain this conservative extent.
        for (int r = 0; r < 2; ++r) {
            auto& sim = *sim_[r];
            if (!select(r)) return false;
            if (sim.aggregate_clear_planes_ == 0 && sim.S_buffers_ == 2)
                launch_clear_S(sim.d_S_[1 - sim.current_S_index_], sim.layout_, sim.stream_);
            sim.aggregate_clear_planes_ = std::max(sim.aggregate_clear_planes_, planes);
        }
    }
    // Recentering changes exchange bounds, not the owner lists. Canonical
    // gathers still require reinstalling the selections they cleared.
    if (!canonical_fields && !migrated && !storage_changed && !owners_.empty()) {
        if (!configure_measurement_order()) return false;
        planned_bricks_ = std::move(bricks);
        return select(0);
    }
    if (!wait_all()) return false;
    owners_ = std::move(next_owners);
    ids_[0].clear(); ids_[1].clear();
    for (std::size_t i = 0; i < owners_.size(); ++i)
        ids_[owners_[i]].push_back(static_cast<int>(i));
    // Neighbouring bricks reuse aggregate cache lines; cell identities and
    // per-cell reduction order are independent of this launch ordering.
    const auto key = [&](int id) {
        const auto& cell = canonical_[id];
        const auto wrapped = [](std::int64_t value, int period) {
            const auto remainder = value % period;
            return remainder < 0 ? remainder + period : remainder;
        };
        const auto x = wrapped(cell.origin_x, geometry_.nx);
        const auto y = wrapped(cell.origin_y, geometry_.ny);
        const int edge = sim_[0]->B_;
        return std::make_tuple(y / edge, cell.origin_z / edge, x / edge,
                               y, cell.origin_z, x);
    };
    for (auto& ids : ids_)
        std::stable_sort(ids.begin(), ids.end(),
                         [&](int a, int b) { return key(a) < key(b); });
    for (int r = 0; r < 2; ++r) {
        auto& sim = *sim_[r];
        if (!select(r)) return false;
        if (!sim.d_owned_promoted_ids_ &&
            !checked(cudaMalloc(reinterpret_cast<void**>(&sim.d_owned_promoted_ids_),
                                canonical_.size() * sizeof(int)), "allocate promoted owner list"))
            return false;
        if (!ids_[r].empty() && !checked(cudaMemcpyAsync(d_ids_[r], ids_[r].data(),
             ids_[r].size() * sizeof(int), cudaMemcpyHostToDevice, sim.stream_),
             "upload cell owners")) return false;
        std::vector<int> promoted;
        for (int id : ids_[r])
            if (canonical_[id].storage_edge > static_cast<std::uint32_t>(sim.B_))
                promoted.push_back(id);
        if (!promoted.empty() && !checked(cudaMemcpyAsync(sim.d_owned_promoted_ids_,
             promoted.data(), promoted.size() * sizeof(int), cudaMemcpyHostToDevice,
             sim.stream_), "upload promoted owners")) return false;
        sim.selection_ = {d_ids_[r], static_cast<int>(ids_[r].size())};
        sim.owned_promoted_count_ = static_cast<int>(promoted.size());
        if (!checked(cudaStreamSynchronize(sim.stream_), "finish ownership upload")) return false;
    }
    if (!configure_measurement_order()) return false;
    if (canonical_fields || migrated) {
        for (int r = 0; r < 2; ++r)
            if (!select(r) || !sim_[r]->reconstruct_current_S()) return false;
        if (!exchange_sum(false) || !wait_all() || !merge_flags()) return false;
        // Reconstruction can raise a flag on either device after the normal
        // commit check. Settle it before a healthy poll can observe this state.
        for (int f = 0; f < FLAG3D_COUNT; ++f)
            if (flag3d_is_fatal(static_cast<Flag3D>(f)) && flags_[f] != 0)
                return fail("integrity failure reconstructing accepted state; durable checkpoint retained");
    }
    if (planned_bricks_.empty()) {
        std::printf("[two-gpu] initial owners %zu/%zu; seam bytes %zu/%zu\n",
                    ids_[0].size(), ids_[1].size(),
                    plans_[0].payload_bytes, plans_[1].payload_bytes);
        std::printf("[two-gpu] split base-update CTAs/cell %d/%d\n",
                    sim_[0]->fast_base_update_shards(UpdateTilePass3D::Boundary),
                    sim_[1]->fast_base_update_shards(UpdateTilePass3D::Boundary));
    }
    planned_bricks_ = std::move(bricks);
    return select(0);
}

bool TwoGpuSim3D::exchange_sum(bool pending) {
    return start_exchange(pending) && finish_exchange(pending);
}

bool TwoGpuSim3D::start_exchange(bool pending) {
    std::uint32_t* sums[2] = {
        pending ? sim_[0]->pending_update_.S_out : sim_[0]->current_S(),
        pending ? sim_[1]->pending_update_.S_out : sim_[1]->current_S()};
    const std::size_t pitch_bytes = static_cast<std::size_t>(geometry_.pitch_x) *
                                     sizeof(std::uint32_t);
    // Each record follows the outgoing boundary tiles and consumption of the
    // previous incoming copy. Interior tiles cannot write the transmitted band.
    // Record both generations before enqueueing cross-device waits.
    for (int r = 0; r < 2; ++r)
        if (!select(r) ||
            !checked(cudaEventRecord(exchange_ready_[r], sim_[r]->stream_),
                     "record exchange readiness")) return false;
    for (int r = 0; r < 2; ++r) {
        const auto& plan = plans_[r];
        if (!select(r) ||
            !checked(cudaStreamWaitEvent(exchange_streams_[r], exchange_ready_[r], 0),
                     "wait for outgoing boundary tiles") ||
            !checked(cudaStreamWaitEvent(exchange_streams_[r], exchange_ready_[1-r], 0),
                     "wait for peer receive workspace")) return false;
        for (int b = 0; plan.payload_bytes != 0 && b < plan.bands.rows.count; ++b) {
            const auto rows = plan.bands.rows.pieces[b];
            cudaMemcpy3DPeerParms copy{};
            copy.srcPtr = make_cudaPitchedPtr(sums[r], pitch_bytes, pitch_bytes, geometry_.ny);
            copy.dstPtr = make_cudaPitchedPtr(d_receive_[1-r], pitch_bytes, pitch_bytes, geometry_.ny);
            copy.srcPos = copy.dstPos = make_cudaPos(0, rows.lo, plan.z.lo);
            copy.srcDevice = sim_[r]->options_.device;
            copy.dstDevice = sim_[1-r]->options_.device;
            copy.extent = make_cudaExtent(pitch_bytes, rows.hi - rows.lo, plan.z.hi - plan.z.lo);
            if (!checked(cudaMemcpy3DPeerAsync(&copy, exchange_streams_[r]), "exchange seam rows"))
                return false;
        }
        if (!checked(cudaEventRecord(exchange_sent_[r], exchange_streams_[r]),
                     "record completed peer copy")) return false;
    }
    return select(0);
}

bool TwoGpuSim3D::finish_exchange(bool pending) {
    std::uint32_t* sums[2] = {
        pending ? sim_[0]->pending_update_.S_out : sim_[0]->current_S(),
        pending ? sim_[1]->pending_update_.S_out : sim_[1]->current_S()};
    // Interior tiles cannot write transmitted rows. Join those rows on the
    // copy stream after both sends, while disjoint interior updates continue.
    for (int r = 0; r < 2; ++r) {
        if (!select(r) ||
            !checked(cudaStreamWaitEvent(exchange_streams_[r], exchange_sent_[r], 0),
                     "wait for outgoing seam copy") ||
            !checked(cudaStreamWaitEvent(exchange_streams_[r], exchange_sent_[1-r], 0),
                     "wait for incoming seam copy")) return false;
        const auto& incoming = plans_[1-r];
        for (int b = 0; incoming.payload_bytes != 0 && b < incoming.bands.rows.count; ++b) {
            const auto rows = incoming.bands.rows.pieces[b];
            const std::size_t band_words = static_cast<std::size_t>(geometry_.pitch_x) *
                                            (rows.hi - rows.lo);
            const unsigned x_blocks = static_cast<unsigned>(
                std::min<std::size_t>(32, (band_words + 255) / 256));
            const dim3 grid(x_blocks, std::min(32, incoming.z.hi - incoming.z.lo));
            add_rows<<<grid, 256, 0, exchange_streams_[r]>>>(sums[r], d_receive_[r],
                geometry_.pitch_x, geometry_.ny, rows.lo, rows.hi - rows.lo,
                incoming.z.lo, incoming.z.hi - incoming.z.lo, sim_[r]->d_flags_);
        }
        if (!checked(cudaGetLastError(), "merge seam rows")) return false;
        if (!checked(cudaEventRecord(exchange_joined_[r], exchange_streams_[r]),
                     "record joined aggregate") ||
            !checked(cudaStreamWaitEvent(sim_[r]->stream_, exchange_joined_[r], 0),
                     "wait for joined aggregate")) return false;
    }
    return select(0);
}

bool TwoGpuSim3D::merge_flags(bool include_states) {
    // Queue both ranks before waiting. After commit the cell snapshots share
    // this barrier; no second round trip is needed to decide ownership.
    for (int r = 0; r < 2; ++r) {
        auto& sim = *sim_[r];
        if (!select(r) || !checked(cudaMemcpyAsync(downloads_[r].flags, sim.d_flags_,
             sizeof(flags_), cudaMemcpyDeviceToHost, sim.stream_),
             "read rank integrity flags")) return false;
        if (include_states && !checked(cudaMemcpyAsync(downloads_[r].cells, sim.d_cells_,
             canonical_.size() * sizeof(CellState3D), cudaMemcpyDeviceToHost,
             sim.stream_), "read committed cell metadata")) return false;
    }
    if (!wait_all()) return false;
    for (std::size_t f = 0; f < flags_.size(); ++f) {
        const std::uint64_t sum = static_cast<std::uint64_t>(flags_[f]) +
            static_cast<std::uint32_t>(downloads_[0].flags[f] - flags_[f]) +
            static_cast<std::uint32_t>(downloads_[1].flags[f] - flags_[f]);
        flags_[f] = static_cast<std::uint32_t>(std::min<std::uint64_t>(sum, 0xffffffffu));
    }
    // The wait above also completes previous uploads from this buffer. The
    // following kernels consume these flags in stream order on each device.
    std::copy(flags_.begin(), flags_.end(), upload_.flags);
    for (int r = 0; r < 2; ++r)
        if (!select(r) || !checked(cudaMemcpyAsync(sim_[r]->d_flags_, upload_.flags,
             sizeof(flags_), cudaMemcpyHostToDevice, sim_[r]->stream_),
             "share integrity flags")) return false;
    return select(0);
}

bool TwoGpuSim3D::gather_state() {
    if (primary_canonical_) return select(0);
    if (!merge_flags(true)) return false;
    for (std::size_t id = 0; id < canonical_.size(); ++id)
        canonical_[id] = downloads_[owners_[id]].cells[id];
    for (int id : ids_[1]) if (!copy_field(id, 1, 0)) return false;
    if (!wait_all() || !enqueue_state_broadcast() || !wait_all()) return false;
    std::vector<std::uint32_t> requests(canonical_.size()), peer_requests(canonical_.size());
    if (!select(0) || !checked(cudaMemcpy(requests.data(), sim_[0]->d_support_requests_,
         requests.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "read support requests") ||
        !select(1) || !checked(cudaMemcpy(peer_requests.data(), sim_[1]->d_support_requests_,
         requests.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "read peer support requests"))
        return false;
    for (std::size_t id = 0; id < requests.size(); ++id) requests[id] |= peer_requests[id];
    auto& primary = *sim_[0];
    if (!select(0) || !checked(cudaMemcpy(primary.d_support_requests_, requests.data(),
         requests.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "join support requests")) return false;
    primary.selection_ = {};
    primary.owned_promoted_count_ = -1;
    if (!primary.reconstruct_current_S() ||
        !checked(cudaStreamSynchronize(primary.stream_), "assemble canonical sum")) return false;
    primary_canonical_ = true;
    return true;
}

void TwoGpuSim3D::measure_aggregate_clear_tiles() {
    std::fill_n(h_clear_tile_tops_, clear_tile_count_, 0);
    constexpr int edge = kAggregateClearTile;
    const int columns = (geometry_.pitch_x + edge - 1) / edge;
    const auto wrap = [](std::int64_t origin, int lo, int hi, int period) {
        WrappedRows result{};
        if (hi < lo) return result;
        if (hi - lo + 1 >= period) {
            result.pieces[0] = {0, period};
            result.count = 1;
            return result;
        }
        const int start = static_cast<int>(
            ((origin % period + lo) % period + period) % period);
        const int end = start + hi - lo + 1;
        result.pieces[0] = {start, std::min(period, end)};
        result.count = 1;
        if (end > period) {
            result.pieces[1] = {0, end - period};
            result.count = 2;
        }
        return result;
    };
    // These are input-field bounds. After the swap, that input aggregate is
    // exactly the next buffer to clear; no bound on future cell growth is needed.
    for (std::size_t id = 0; id < canonical_.size(); ++id) {
        const auto& origin = canonical_[id];
        const auto& measured = downloads_[owners_[id]].cells[id];
        if (measured.bb_hi_z < measured.bb_lo_z) continue;
        const int top = static_cast<int>(std::clamp<std::int64_t>(
            origin.origin_z + measured.bb_hi_z + 1, 0, geometry_.nz));
        const auto xs = wrap(origin.origin_x, measured.bb_lo_x, measured.bb_hi_x, geometry_.nx);
        const auto ys = wrap(origin.origin_y, measured.bb_lo_y, measured.bb_hi_y, geometry_.ny);
        for (int a = 0; a < xs.count; ++a)
            for (int b = 0; b < ys.count; ++b)
                for (int y = ys.pieces[b].lo / edge; y <= (ys.pieces[b].hi - 1) / edge; ++y)
                    for (int x = xs.pieces[a].lo / edge; x <= (xs.pieces[a].hi - 1) / edge; ++x) {
                        auto& height = h_clear_tile_tops_[y * columns + x];
                        height = std::max(height, top);
                    }
    }
}

bool TwoGpuSim3D::distributed_step() {
    auto& primary = *sim_[0];
    if (primary_canonical_) {
        if (!wait_all() || !read_states(0)) return false;
        std::copy_n(downloads_[0].cells, canonical_.size(), canonical_.begin());
        if (primary.recovery_events_ != mirrored_recoveries_ &&
            !synchronize_peer_growth()) return false;
        if (!select(0) || !checked(cudaMemcpy(flags_.data(), primary.d_flags_,
             sizeof(flags_), cudaMemcpyDeviceToHost), "read canonical flags")) return false;
        if (!select(1) || !checked(cudaMemcpy(sim_[1]->d_flags_, flags_.data(),
             sizeof(flags_), cudaMemcpyHostToDevice), "restore peer flags")) return false;
        sim_[1]->volume_current_ = primary.volume_current_;
        sim_[1]->surface_current_ = primary.surface_current_;
        if (!enqueue_state_broadcast() || !configure_ownership(true)) return false;
        primary_canonical_ = false;
    }
    for (int r = 0; r < 2; ++r) {
        const auto& plan = plans_[r];
        TileExchangeRegion3D region{};
        if (plan.payload_bytes != 0) {
            region.y_count = plan.bands.rows.count;
            for (int b = 0; b < region.y_count; ++b) {
                region.y_lo[b] = plan.bands.rows.pieces[b].lo;
                region.y_hi[b] = plan.bands.rows.pieces[b].hi;
            }
            region.z_lo = plan.z.lo;
            region.z_hi = plan.z.hi;
        }
        if (!select(r) || !sim_[r]->begin_step(&region)) return false;
    }
    if (!start_exchange(true)) return false;
    for (int r = 0; r < 2; ++r)
        if (!select(r) ||
            (sim_[r]->pending_update_.tile_pass == UpdateTilePass3D::Boundary &&
             !sim_[r]->enqueue_interior_update())) return false;
    if (!finish_exchange(true) || !merge_flags()) return false;
    bool fatal = false;
    for (int f = 0; f < FLAG3D_COUNT; ++f)
        fatal = fatal || (flag3d_is_fatal(static_cast<Flag3D>(f)) && flags_[f] != 0);
    for (int r = 0; r < 2; ++r)
        if (!select(r) || !sim_[r]->finish_step()) return false;
    if (!merge_flags(true)) return false;
    // Coordinate overflow may be discovered while committing origins, after
    // field repair. Such an unrecoverable commit must not publish any output.
    if (!fatal)
        for (int f = 0; f < FLAG3D_COUNT; ++f)
            if (flag3d_is_fatal(static_cast<Flag3D>(f)) && flags_[f] != 0)
                return fail("integrity failure during commit; durable checkpoint retained");
    if (fatal) {
        // Repair may copy the complete input aggregate into the pending buffer.
        for (auto& sim : sim_)
            sim->aggregate_dirty_rows_[sim->current_S_index_] = {};
        // Every rank rolled back under the same fatal flag. Only the canonical
        // solver performs adaptive recovery; the next step replicates its result.
        return gather_state() && primary.synchronize_and_check("distributed recovery");
    }
    int source_planes = 0;
    if (!primary.options_.strict) {
        // The fast update leaves the input support bounds in the cell record.
        // They enclose every nonzero Q5.27 contribution, including diffuse tails.
        // The old input S is now the buffer that the next step must clear.
        static_assert(kSupportEps * kSupportEps * kQScaleF < 0.5f);
        measure_aggregate_clear_tiles();
        for (std::size_t id = 0; id < canonical_.size(); ++id) {
            const auto& measured = downloads_[owners_[id]].cells[id];
            const auto top = canonical_[id].origin_z + measured.bb_hi_z + 1;
            source_planes = std::max(source_planes, static_cast<int>(
                std::min<std::int64_t>(geometry_.nz, std::max<std::int64_t>(0, top))));
        }
    }
    for (std::size_t id = 0; id < canonical_.size(); ++id)
        canonical_[id] = downloads_[owners_[id]].cells[id];
    if (!enqueue_state_broadcast() || !configure_ownership(false)) return false;
    if (source_planes > 0)
        for (auto& sim : sim_) sim->aggregate_clear_planes_ = source_planes;
    if (clear_tile_count_) {
        for (int r = 0; r < 2; ++r) {
            if (!select(r) || !checked(cudaMemcpyAsync(d_clear_tile_tops_[r], h_clear_tile_tops_,
                 clear_tile_count_ * sizeof(int), cudaMemcpyHostToDevice, sim_[r]->stream_),
                 "upload aggregate bounds")) return false;
            sim_[r]->aggregate_clear_tile_tops_ = d_clear_tile_tops_[r];
        }
    }
    return select(0);
}

bool TwoGpuSim3D::refresh_owned_records(bool compute_surface, bool apply_motion) {
    auto& primary = *sim_[0];
    // Refresh on each owner, then collect records rather than remote fields.
    for (int r = 0; r < 2; ++r)
        if (!select(r) || !sim_[r]->refresh_measurements(false, compute_surface, apply_motion))
            return false;
    if (!merge_flags(true)) return false;
    for (int f = 0; f < FLAG3D_COUNT; ++f)
        if (flag3d_is_fatal(static_cast<Flag3D>(f)) && flags_[f] != 0)
            return gather_state() && primary.synchronize_and_check("owned measurement");
    for (std::size_t id = 0; id < canonical_.size(); ++id)
        canonical_[id] = downloads_[owners_[id]].cells[id];
    return enqueue_state_broadcast();
}

bool TwoGpuSim3D::prepare_trajectory() {
    auto& primary = *sim_[0];
    // Channel wall diagnostics still read the complete field array.
    if (primary_canonical_ || primary.params_.resolved_wall_channel())
        return primary.ensure_measurements(true, "trajectory measurement");
    return refresh_owned_records(true, true);
}

bool TwoGpuSim3D::prepare_compaction() {
    auto& primary = *sim_[0];
    if (primary_canonical_)
        return primary.refresh_measurements(false, primary.surface_current_, false) &&
               primary.synchronize_and_check("compaction measurement");
    if (!refresh_owned_records(primary.surface_current_, false)) return false;
    if (primary_canonical_) return true;
    // A shrink test reads only promoted fields. Base fields remain on their
    // owners unless an accepted crop requires a canonical resize transaction.
    // After a rejected crop, the primary's selection excludes these peer copies.
    for (int id : primary.h_promoted_ids_)
        if (owners_[id] == 1 && !copy_field(id, 1, 0)) return false;
    return wait_all();
}

bool TwoGpuSim3D::capture_surfaces(SurfaceOutput3D& output) {
    // Pack on each owner. No phase-field peer transfer or aggregate rebuild is
    // needed merely to observe interfaces.
    if (!wait_all()) return false;
    // Another observer may already have gathered all fields and cleared the
    // primary selection. Reuse that complete view instead of counting it twice.
    if (primary_canonical_)
        return output.capture({sim_[0]->surface_view()}, sim_[0]->step(), sim_[0]->time());
    return output.capture({sim_[0]->surface_view(), sim_[1]->surface_view()},
                          sim_[0]->step(), sim_[0]->time());
}

bool TwoGpuSim3D::run() { return select(0) && sim_[0]->run(); }

bool TwoGpuSim3D::capture_relaxation(RelaxationOutput3D& output) {
    // Drain ownership migrations before observing the owner's field and S.
    if (!wait_all()) return false;
    if (primary_canonical_)
        return output.capture({sim_[0]->relaxation_view()}, sim_[0]->step());
    return output.capture({sim_[0]->relaxation_view(), sim_[1]->relaxation_view()},
                          sim_[0]->step());
}

bool TwoGpuSim3D::bench(int steps, double* milliseconds_per_step) {
    // GPU0 events include its idle waits for peer/host coordination. The shared
    // benchmark's accepted-step and recovery guards still apply.
    if (!select(0)) return false;
    if (sim_[0]->options_.bench_phases)
        std::printf("[two-gpu] primary-device phases only. Fast update phases "
                    "cover boundary tiles; repair/finalize/scatter includes "
                    "interior tiles, the exchange join, and pre-commit control; "
                    "host/output gap includes post-commit control and ownership.\n");
    return sim_[0]->bench(steps, milliseconds_per_step);
}

}  // namespace pf3d
