#include "pf3d/surface_output.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {
bool checked(cudaError_t status) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    return false;
}

template <class T> void write(std::ofstream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

bool fixture(const std::filesystem::path& root, int test) {
    constexpr int N = 4, B = 24;
    pf3d::SimParams3D p{};
    p.num_cells = N; p.Nx = p.Ny = 96;
    p.Nz = test == 0 ? 96 : test == 3 ? 64 : 16;
    p.boundary_flags = test == 0 ? pf3d::kBoundaryPeriodicXYZ3D : test == 3
        ? pf3d::kBoundarySubstrateSlab3D : pf3d::kBoundaryHardWallChannel3D;
    if (p.hard_wall_channel()) {
        p.channel_height = 10; p.channel_padding = 3; p.wall_kappa = 10; p.wall_width = 1;
    }
    p.dt = 0.001; p.tau = 0.02; p.lambda = 1;
    const pf3d::CellFieldStorage3D storage{test == 1 ? p.Nz : 0};
    std::vector<pf3d::CellState3D> cells(N);
    std::vector<float> base(N * storage.words(B), 0.0f);
    std::vector<std::vector<float>> fields(N);
    const int z_origins[N] = {-4, 4, -12, 8};
    const std::int64_t gids[N] = {11, 29, 103, 501};
    const int permutation[N] = {2, 0, 3, 1};
    for (int n = 0; n < N; ++n) {
        auto& c = cells[n];
        c.global_id = gids[n]; c.origin_x = 3 * n - 18; c.origin_y = 38 * n + 50;
        c.origin_z = test == 0 ? -7 : z_origins[n];
        c.gamma = 1; c.v_A = 0.01f; c.R_tgt = 5; c.polarity_x = 1; c.tumble_ctr = 9 + n;
        c.storage_edge = n % 2 ? 32 : 0;
        const int edge = pf3d::cell_support_edge(c, B);
        fields[n].assign(storage.words(edge), 0.0f);
        const int planes = storage.planes(edge);
        for (int sz = 0; sz < planes; ++sz)
            for (int y = 0; y < edge; ++y)
                for (int x = 0; x < edge; ++x) {
                    const int z = storage.compact(edge) ? sz - static_cast<int>(c.origin_z) : sz;
                    if (z < 0 || z >= edge) continue;
                    float value = 0;
                    if (n == 0) value = (x == 12 && y == 12 && z == 12) ? 0.5f : 0.0f;
                    else if (n == 1) value = (x + y + z) % 2 ? 0.75f : 0.25f;
                    else if (n == 2) value = ((x * 17 + y * 31 + z * 43) % 5) * 0.25f;
                    // Cell3 intentionally has no interface.
                    fields[n][(static_cast<std::size_t>(sz) * edge + y) * edge + x] = value;
                }
        if (edge == B) std::copy(fields[n].begin(), fields[n].end(), base.begin() + n * storage.words(B));
    }
    float* d_base = nullptr;
    float** d_promoted = nullptr;
    pf3d::CellState3D* d_cells = nullptr;
    int* d_ids = nullptr;
    std::vector<float*> promoted(N, nullptr);
    bool ok = checked(cudaMalloc(reinterpret_cast<void**>(&d_base), base.size() * sizeof(float))) &&
        checked(cudaMalloc(reinterpret_cast<void**>(&d_cells), cells.size() * sizeof(cells[0]))) &&
        checked(cudaMalloc(reinterpret_cast<void**>(&d_promoted), N * sizeof(float*))) &&
        checked(cudaMalloc(reinterpret_cast<void**>(&d_ids), sizeof(permutation)));
    for (int n = 0; n < N && ok; ++n)
        if (cells[n].storage_edge) {
            ok = checked(cudaMalloc(reinterpret_cast<void**>(&promoted[n]), fields[n].size() * sizeof(float))) &&
                 checked(cudaMemcpy(promoted[n], fields[n].data(), fields[n].size() * sizeof(float), cudaMemcpyHostToDevice));
        }
    ok = ok && checked(cudaMemcpy(d_base, base.data(), base.size() * sizeof(float), cudaMemcpyHostToDevice)) &&
        checked(cudaMemcpy(d_cells, cells.data(), cells.size() * sizeof(cells[0]), cudaMemcpyHostToDevice)) &&
        checked(cudaMemcpy(d_promoted, promoted.data(), N * sizeof(float*), cudaMemcpyHostToDevice)) &&
        checked(cudaMemcpy(d_ids, permutation, sizeof(permutation), cudaMemcpyHostToDevice));
    const auto name = std::string("fixture-") + std::to_string(test);
    if (ok) {
        pf3d::SurfaceOutput3D output;
        pf3d::SurfaceDeviceView3D view{0, d_base, d_promoted, d_cells, B, storage, {d_ids, N}, nullptr};
        ok = output.open((root / (name + ".pfs3d")).string(), p, 10, false) &&
             output.capture({view}, 0, 0) && output.close();
    }
    if (ok) {
        std::vector<float> downloaded(base.size());
        std::vector<pf3d::CellState3D> downloaded_cells(N);
        ok = checked(cudaMemcpy(downloaded.data(), d_base, downloaded.size() * sizeof(float), cudaMemcpyDeviceToHost)) &&
             checked(cudaMemcpy(downloaded_cells.data(), d_cells, N * sizeof(cells[0]), cudaMemcpyDeviceToHost)) &&
             std::memcmp(base.data(), downloaded.data(), base.size() * sizeof(float)) == 0 &&
             std::memcmp(cells.data(), downloaded_cells.data(), N * sizeof(cells[0])) == 0;
        for (int n = 0; n < N && ok; ++n) if (promoted[n]) {
            downloaded.resize(fields[n].size());
            ok = checked(cudaMemcpy(downloaded.data(), promoted[n], downloaded.size() * sizeof(float), cudaMemcpyDeviceToHost)) &&
                std::memcmp(fields[n].data(), downloaded.data(), fields[n].size() * sizeof(float)) == 0;
        }
    }
    if (ok) {
        std::ofstream oracle(root / (name + ".dense"), std::ios::binary);
        write(oracle, std::uint32_t(N));
        for (int n = 0; n < N; ++n) {
            const auto& c = cells[n];
            const int edge = pf3d::cell_support_edge(c, B), side = edge + 2;
            write(oracle, c.global_id); write(oracle, std::uint32_t(edge));
            write(oracle, c.origin_x); write(oracle, c.origin_y); write(oracle, c.origin_z);
            const std::uint64_t words = std::uint64_t(side) * side * side;
            write(oracle, words);
            for (int z = -1; z <= edge; ++z)
                for (int y = -1; y <= edge; ++y)
                    for (int x = -1; x <= edge; ++x) {
                        float value = 0;
                        if (x >= 0 && y >= 0 && z >= 0 && x < edge && y < edge && z < edge &&
                            (!p.bounded_z() || (c.origin_z + z >= 0 && c.origin_z + z < p.Nz)))
                            value = fields[n][storage.index(x, y, z, edge, c.origin_z)];
                        write(oracle, value);
                    }
        }
        ok = oracle.good();
    }
    if (ok && test == 0) {
        // A corrupt sample must fail the observation without modifying the
        // simulator or writing a successful completion marker.
        const float nan = std::numeric_limits<float>::quiet_NaN();
        ok = checked(cudaMemcpy(d_base + storage.index(10, 10, 10, B, cells[0].origin_z),
                                &nan, sizeof(nan), cudaMemcpyHostToDevice));
        pf3d::SurfaceOutput3D invalid;
        pf3d::SurfaceDeviceView3D view{0, d_base, d_promoted, d_cells, B, storage, {d_ids, N}, nullptr};
        ok = ok && invalid.open((root / "nonfinite.pfs3d").string(), p, 10, false)
             && !invalid.capture({view}, 0, 0) && !invalid.close(false);
    }
    for (float* pointer : promoted) if (pointer) ok = checked(cudaFree(pointer)) && ok;
    if (d_ids) ok = checked(cudaFree(d_ids)) && ok;
    if (d_promoted) ok = checked(cudaFree(d_promoted)) && ok;
    if (d_cells) ok = checked(cudaFree(d_cells)) && ok;
    if (d_base) ok = checked(cudaFree(d_base)) && ok;
    return ok;
}
}  // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    if (argc != 2 || !std::filesystem::create_directory(argv[1])) {
        std::fprintf(stderr, "Pass one new output directory\n"); return 2;
    }
    for (int test = 0; test < 4; ++test)
        if (!fixture(argv[1], test)) return 1;
    std::puts("SURFACE_GPU_FIXTURES_PASS");
    return 0;
}
