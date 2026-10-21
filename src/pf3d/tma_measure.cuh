#pragma once

// Private implementation for kernels.cu, included after pf3d/kernels.cuh
// supplies the measurement arguments and shared-halo layout.
#include <cudaTypedefs.h>
#include <vector>

namespace pf3d {
namespace {

// Tensor copies require 16-byte rows and a 128-byte shared base. Loading from
// x-4 puts the left ghost at raw[3] and the first interior value at raw[4].
constexpr int kTmaRowStride = 40;
constexpr int kTmaBufferWords =
    ((kHaloLeadingPadding + kTmaRowStride * kHaloY * kHaloZ + 31) / 32) * 32;
constexpr unsigned kTmaHaloBytes = kTmaRowStride * kHaloY * kHaloZ * sizeof(float);
constexpr unsigned kTmaPipelineBytes = 2 * kTmaBufferWords * sizeof(float);
static_assert(kTmaBufferWords % 32 == 0);

bool measurement_tensor_map(const MeasureArgs3D& args, CUtensorMap* result) {
    struct Entry {
        const float* field;
        int edge, planes, count;
        CUtensorMap map;
    };
    // Ping-pong allocations keep their descriptors; moving cell origins are
    // supplied as copy coordinates rather than requiring descriptor rebuilds.
    static thread_local std::vector<Entry> entries;
    const int planes = args.storage.planes(args.B);
    for (const auto& entry : entries) {
        if (entry.field == args.phi && entry.edge == args.B
            && entry.planes == planes && entry.count == args.N) {
            *result = entry.map;
            return true;
        }
    }
    void* address = nullptr;
    cudaDriverEntryPointQueryResult query{};
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &address,
            12000, cudaEnableDefault, &query) != cudaSuccess
        || query != cudaDriverEntryPointSuccess || address == nullptr) {
        std::fprintf(stderr, "[3d] CUDA tensor-map encoder is unavailable\n");
        return false;
    }
    const auto encode = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(address);
    const cuuint64_t edge = args.B;
    const cuuint64_t sizes[4] = {edge, edge, static_cast<cuuint64_t>(planes),
                                static_cast<cuuint64_t>(args.N)};
    const cuuint64_t strides[3] = {edge * sizeof(float), edge * edge * sizeof(float),
                                  edge * edge * planes * sizeof(float)};
    const cuuint32_t box[4] = {kTmaRowStride, kHaloY, kHaloZ, 1};
    const cuuint32_t steps[4] = {1, 1, 1, 1};
    Entry entry{args.phi, args.B, planes, args.N, {}};
    const CUresult status = encode(&entry.map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4,
        const_cast<float*>(args.phi), sizes, strides, box, steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        std::fprintf(stderr, "[3d] tensor-map encoding failed: %d\n", int(status));
        return false;
    }
    entries.push_back(entry);
    *result = entry.map;
    return true;
}

__device__ __forceinline__ void initialize_tma_barrier(unsigned long long* barrier) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(barrier));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
                 "fence.proxy.async.shared::cta;" :: "r"(address) : "memory");
}

__device__ __forceinline__ void load_tma_halo(const CUtensorMap* map,
    int cell, int x, int y, int z, float* destination, unsigned long long* barrier) {
    const unsigned target = static_cast<unsigned>(__cvta_generic_to_shared(destination));
    const unsigned ready = static_cast<unsigned>(__cvta_generic_to_shared(barrier));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(ready), "r"(kTmaHaloBytes) : "memory");
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4, %5}], [%6];"
                 :: "r"(target), "l"(map), "r"(x), "r"(y), "r"(z), "r"(cell),
                    "r"(ready) : "memory");
}

__device__ __forceinline__ void wait_tma_halo(unsigned long long* barrier, unsigned phase) {
    const unsigned ready = static_cast<unsigned>(__cvta_generic_to_shared(barrier));
    unsigned done;
    do {
        asm volatile("{ .reg .pred complete; "
                     "mbarrier.try_wait.parity.shared::cta.b64 complete, [%1], %2; "
                     "selp.u32 %0, 1, 0, complete; }"
                     : "=r"(done) : "r"(ready), "r"(phase) : "memory");
    } while (!done);
}

} // namespace
} // namespace pf3d
