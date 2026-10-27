#pragma once

// Load a shared tile of the immutable aggregate field.
// Periodic seams and partial z tiles retain the scalar loader.
namespace pf3d {
namespace {

// Hopper requires the first tensor coordinate to start on a 16-byte boundary.
// Three extra words cover any cell-origin alignment without changing samples.
constexpr int kAggregateRowStride = kBrickX + 4;
constexpr int kAggregateTileWords = kAggregateRowStride * kBrickY * kBrickZ;
constexpr unsigned kAggregateTileBytes = kAggregateTileWords * sizeof(std::uint32_t);

bool aggregate_tensor_map(const std::uint32_t *field, const SLayout3D &layout,
                          CUtensorMap *result) {
    struct Entry {
        const std::uint32_t *field;
        int nx, ny, nz, pitch;
        CUtensorMap map;
    };
    static thread_local std::vector<Entry> entries;
    for (const auto &e : entries) {
        if (e.field == field && e.nx == layout.nx && e.ny == layout.ny && e.nz == layout.nz &&
            e.pitch == layout.pitch_x) {
            *result = e.map;
            return true;
        }
    }
    void *address = nullptr;
    cudaDriverEntryPointQueryResult query{};
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &address, 12000,
                                         cudaEnableDefault, &query) != cudaSuccess ||
        query != cudaDriverEntryPointSuccess || address == nullptr)
        return false;
    const auto encode = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(address);
    const cuuint64_t sizes[3] = {static_cast<cuuint64_t>(layout.nx),
                                 static_cast<cuuint64_t>(layout.ny),
                                 static_cast<cuuint64_t>(layout.nz)};
    const cuuint64_t strides[2] = {static_cast<cuuint64_t>(layout.pitch_x) * sizeof(std::uint32_t),
                                   static_cast<cuuint64_t>(layout.pitch_x) * layout.ny *
                                       sizeof(std::uint32_t)};
    const cuuint32_t box[3] = {kAggregateRowStride, kBrickY, kBrickZ};
    const cuuint32_t steps[3] = {1, 1, 1};
    Entry e{field, layout.nx, layout.ny, layout.nz, layout.pitch_x, {}};
    const CUresult status =
        encode(&e.map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 3, const_cast<std::uint32_t *>(field), sizes,
               strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
               CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        std::fprintf(stderr, "[3d] aggregate tensor-map encoding failed: %d\n", int(status));
        return false;
    }
    entries.push_back(e);
    *result = e.map;
    return true;
}

__device__ __forceinline__ void load_tma_aggregate(const CUtensorMap *map, int x, int y, int z,
                                                   std::uint32_t *destination,
                                                   unsigned long long *barrier) {
    const unsigned target = static_cast<unsigned>(__cvta_generic_to_shared(destination));
    const unsigned ready = static_cast<unsigned>(__cvta_generic_to_shared(barrier));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(ready),
                 "r"(kAggregateTileBytes)
                 : "memory");
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4}], [%5];" ::"r"(target),
                 "l"(map), "r"(x), "r"(y), "r"(z), "r"(ready)
                 : "memory");
}

} // namespace
} // namespace pf3d
