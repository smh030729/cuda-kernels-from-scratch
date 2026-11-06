#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>


#define CUDA_CHECK(call)                                              \
do {                                                                  \
    cudaError_t err = (call);                                         \
    if (err != cudaSuccess) {                                         \
        std::fprintf(                                                  \
            stderr,                                                    \
            "CUDA error at %s:%d: %s\n",                              \
            __FILE__,                                                  \
            __LINE__,                                                  \
            cudaGetErrorString(err));                                 \
        std::exit(EXIT_FAILURE);                                       \
    }                                                                 \
} while (0)


constexpr int WARP_SIZE = 32;


// ===========================================================
// generic address -> 32-bit shared memory address
//
// ldmatrix 的地址操作数需要 shared-memory address。
// ============================================================

__device__ __forceinline__
uint32_t cvta_to_shared_u32(const void* ptr)
{
    uint32_t smem_addr;

    asm volatile(
        "{\n"
        "  .reg .u64 smem_ptr;\n"
        "  cvta.to.shared.u64 smem_ptr, %1;\n"
        "  cvt.u32.u64 %0, smem_ptr;\n"
        "}\n"
        : "=r"(smem_addr)
        : "l"(ptr)
    );

    return smem_addr;
}


// ============================================================
// ldmatrix.x1
//
// 输入:
//      shared memory 中一个 8x8 FP16 matrix
//
// 输出:
//      每个 lane 一个 32-bit register
//
// 一个 register 中 pack 两个 FP16。
// ============================================================

__global__ void ldmatrix_x1_kernel(float* output)
{
    const int lane_id = threadIdx.x;


    // --------------------------------------------------------
    // 8x8 FP16 matrix
    //
    // 64 half = 128 bytes
    // --------------------------------------------------------

    __shared__ __half tile[8][8];


    // --------------------------------------------------------
    // 初始化：
    //
    // tile:
    //
    // 0  1  2 ... 7
    // 8  9 10 ...15
    // ...
    // --------------------------------------------------------

    for (int idx = lane_id;
         idx < 64;
         idx += WARP_SIZE)
    {
        const int row = idx / 8;
        const int col = idx % 8;

        tile[row][col] =
            __float2half(
                static_cast<float>(idx)
            );
    }


    __syncthreads();


    // --------------------------------------------------------
    // ldmatrix.x1 需要 8 条 row address。
    //
    // 对 x1 来说，lane 0~7 提供：
    //
    // lane0 -> row0 address
    // lane1 -> row1 address
    // ...
    // lane7 -> row7 address
    //
    // 这里让其他 lanes 重复这些地址。
    // --------------------------------------------------------

    const int addr_row =
        lane_id % 8;


    const uint32_t smem_addr =
        cvta_to_shared_u32(
            &tile[addr_row][0]
        );


    // --------------------------------------------------------
    // 每个 lane 最终得到一个 32-bit register。
    //
    // register 内包含两个 FP16。
    // --------------------------------------------------------

    uint32_t frag;


    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 "
        "{%0}, [%1];\n"
        : "=r"(frag)
        : "r"(smem_addr)
    );


    // --------------------------------------------------------
    // frag:
    //
    // 31                    16 15                     0
    // +-----------------------+-----------------------+
    // |       half #1         |       half #0         |
    // +-----------------------+-----------------------+
    //
    // 拆开两个 FP16。
    // --------------------------------------------------------

    const unsigned short bits0 =
        static_cast<unsigned short>(
            frag & 0xffff
        );

    const unsigned short bits1 =
        static_cast<unsigned short>(
            frag >> 16
        );


    const __half h0 =
        __ushort_as_half(bits0);

    const __half h1 =
        __ushort_as_half(bits1);


    // 转成 float，方便 host 直接打印

    output[lane_id * 2] =
        __half2float(h0);

    output[lane_id * 2 + 1] =
        __half2float(h1);
}


// ============================================================
// main
// ============================================================

int main()
{
    float* d_output = nullptr;

    constexpr int NUM_OUTPUTS =
        WARP_SIZE * 2;


    CUDA_CHECK(
        cudaMalloc(
            &d_output,
            NUM_OUTPUTS * sizeof(float)
        )
    );


    // 一个 block
    // 一个 warp
    ldmatrix_x1_kernel<<<1, WARP_SIZE>>>(
        d_output
    );


    CUDA_CHECK(
        cudaGetLastError()
    );

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    float h_output[NUM_OUTPUTS];


    CUDA_CHECK(
        cudaMemcpy(
            h_output,
            d_output,
            NUM_OUTPUTS * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    // --------------------------------------------------------
    // 打印每个 lane 拿到的两个 half
    // --------------------------------------------------------

    for (int lane = 0;
         lane < WARP_SIZE;
         ++lane)
    {
        std::printf(
            "lane %2d -> %5.1f  %5.1f\n",
            lane,
            h_output[lane * 2],
            h_output[lane * 2 + 1]
        );
    }


    CUDA_CHECK(
        cudaFree(d_output)
    );


    return 0;
}