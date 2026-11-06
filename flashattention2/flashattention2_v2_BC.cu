
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cstdint>
#include <stdexcept>
#include "flashattention2.cuh"

namespace fa2_v2 {

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;
constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr int Br = 64;
constexpr int Bc = 64;
constexpr int d = 64;
constexpr int SUB_BC = 32;
constexpr int SCORE_FRAGS = SUB_BC / MMA_N;
constexpr int OUT_FRAGS = d / MMA_N;
constexpr int K_STEPS = d / MMA_K;
constexpr float SCALE = 0.125f;

__device__ __forceinline__ int swizzle_col(int row, int col) {
    return col ^ ((row & 7) << 3);
}

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{\n"
        "  .reg .u64 smem_ptr;\n"
        "  cvta.to.shared.u64 smem_ptr, %1;\n"
        "  cvt.u32.u64 %0, smem_ptr;\n"
        "}\n"
        : "=r"(addr) : "l"(ptr)
    );
    return addr;
}

__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
    uint32_t addr = cvta_to_shared_u32(dst);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(addr), "l"(src) : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ void load_tile_async(
    half (*dst)[Bc], const half* src, int stride, int base_row, int base_col
) {
    for (int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
        const int row = idx / 8;
        const int col = (idx % 8) * 8;
        const int physical_col = swizzle_col(row, col);
        cp_async_16(&dst[row][physical_col],
                    &src[(base_row + row) * stride + base_col + col]);
    }
}

__device__ __forceinline__ float subgroup4_reduce_max(float x) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1));
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 2));
    return x;
}

__device__ __forceinline__ float subgroup4_reduce_sum(float x) {
    x += __shfl_xor_sync(0xffffffff, x, 1);
    x += __shfl_xor_sync(0xffffffff, x, 2);
    return x;
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t addr, uint32_t& x0, uint32_t& x1, uint32_t& x2, uint32_t& x3
) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
        : "r"(addr)
    );
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t addr, uint32_t& x0, uint32_t& x1
) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(x0), "=r"(x1)
        : "r"(addr)
    );
}

__device__ __forceinline__ void mma_m16n8k16(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float& d0, float& d1, float& d2, float& d3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

__device__ __forceinline__ uint32_t pack_half2(float x0, float x1) {
    const uint32_t a = __half_as_ushort(__float2half_rn(x0));
    const uint32_t b = __half_as_ushort(__float2half_rn(x1));
    return a | (b << 16);
}

__device__ __forceinline__ void compute_tile(
    const half (&Qs)[Br][d],
    const half (&Ks)[d][Bc],
    const half (&Vs)[Bc][d],
    int kv_base, int N, int warpid, int laneid,
    float (&u)[OUT_FRAGS][4],
    float& m0, float& m1, float& l0, float& l1
) {
    const int warp_q_base = warpid * MMA_M;
    const int lane_in_group = laneid % 4;
    const int ldm_mask = (laneid & 7) << 3;

    #pragma unroll 1
    for (int part = 0; part < Bc / SUB_BC; ++part) {
        float s[SCORE_FRAGS][4];

        #pragma unroll
        for (int n_iter = 0; n_iter < SCORE_FRAGS; ++n_iter) {
            s[n_iter][0] = 0.0f;
            s[n_iter][1] = 0.0f;
            s[n_iter][2] = 0.0f;
            s[n_iter][3] = 0.0f;
        }

        #pragma unroll
        for (int k_iter = 0; k_iter < K_STEPS; ++k_iter) {
            const int q_row = warp_q_base + (laneid & 15);
            const int q_col = k_iter * MMA_K + ((laneid & 16) >> 1);
            const uint32_t q_addr = cvta_to_shared_u32(&Qs[q_row][q_col ^ ldm_mask]);

            uint32_t q0, q1, q2, q3;
            ldmatrix_x4(q_addr, q0, q1, q2, q3);

            #pragma unroll
            for (int n_iter = 0; n_iter < SCORE_FRAGS; ++n_iter) {
                const int k_row = k_iter * MMA_K + (laneid & 15);
                const int k_col = part * SUB_BC + n_iter * MMA_N;
                const uint32_t k_addr = cvta_to_shared_u32(&Ks[k_row][k_col ^ ldm_mask]);

                uint32_t k0, k1;
                ldmatrix_x2_trans(k_addr, k0, k1);

                mma_m16n8k16(
                    q0, q1, q2, q3, k0, k1,
                    s[n_iter][0], s[n_iter][1],
                    s[n_iter][2], s[n_iter][3]
                );
            }
        }

        #pragma unroll
        for (int n_iter = 0; n_iter < SCORE_FRAGS; ++n_iter) {
            s[n_iter][0] *= SCALE;
            s[n_iter][1] *= SCALE;
            s[n_iter][2] *= SCALE;
            s[n_iter][3] *= SCALE;

            const int key0 = kv_base + part * SUB_BC
                           + n_iter * MMA_N + lane_in_group * 2;
            const int key1 = key0 + 1;

            if (key0 >= N) {
                s[n_iter][0] = -CUDART_INF_F;
                s[n_iter][2] = -CUDART_INF_F;
            }
            if (key1 >= N) {
                s[n_iter][1] = -CUDART_INF_F;
                s[n_iter][3] = -CUDART_INF_F;
            }
        }

        float local_max0 = -CUDART_INF_F;
        float local_max1 = -CUDART_INF_F;

        #pragma unroll
        for (int n_iter = 0; n_iter < SCORE_FRAGS; ++n_iter) {
            local_max0 = fmaxf(local_max0, s[n_iter][0]);
            local_max0 = fmaxf(local_max0, s[n_iter][1]);
            local_max1 = fmaxf(local_max1, s[n_iter][2]);
            local_max1 = fmaxf(local_max1, s[n_iter][3]);
        }

        const float tile_max0 = subgroup4_reduce_max(local_max0);
        const float tile_max1 = subgroup4_reduce_max(local_max1);
        const float m_new0 = fmaxf(m0, tile_max0);
        const float m_new1 = fmaxf(m1, tile_max1);
        const float alpha0 = expf(m0 - m_new0);
        const float alpha1 = expf(m1 - m_new1);

        float local_sum0 = 0.0f;
        float local_sum1 = 0.0f;

        #pragma unroll
        for (int n_iter = 0; n_iter < SCORE_FRAGS; ++n_iter) {
            s[n_iter][0] = expf(s[n_iter][0] - m_new0);
            s[n_iter][1] = expf(s[n_iter][1] - m_new0);
            s[n_iter][2] = expf(s[n_iter][2] - m_new1);
            s[n_iter][3] = expf(s[n_iter][3] - m_new1);

            local_sum0 += s[n_iter][0] + s[n_iter][1];
            local_sum1 += s[n_iter][2] + s[n_iter][3];
        }

        const float tile_sum0 = subgroup4_reduce_sum(local_sum0);
        const float tile_sum1 = subgroup4_reduce_sum(local_sum1);
        const float l_new0 = alpha0 * l0 + tile_sum0;
        const float l_new1 = alpha1 * l1 + tile_sum1;

        #pragma unroll
        for (int n_iter = 0; n_iter < OUT_FRAGS; ++n_iter) {
            u[n_iter][0] *= alpha0;
            u[n_iter][1] *= alpha0;
            u[n_iter][2] *= alpha1;
            u[n_iter][3] *= alpha1;
        }

        #pragma unroll
        for (int k_iter = 0; k_iter < SUB_BC / MMA_K; ++k_iter) {
            const int p_frag0 = 2 * k_iter;
            const int p_frag1 = 2 * k_iter + 1;
            const uint32_t p0 = pack_half2(s[p_frag0][0], s[p_frag0][1]);
            const uint32_t p1 = pack_half2(s[p_frag0][2], s[p_frag0][3]);
            const uint32_t p2 = pack_half2(s[p_frag1][0], s[p_frag1][1]);
            const uint32_t p3 = pack_half2(s[p_frag1][2], s[p_frag1][3]);

            #pragma unroll
            for (int n_iter = 0; n_iter < OUT_FRAGS; ++n_iter) {
                const int v_row = part * SUB_BC + k_iter * MMA_K + (laneid & 15);
                const int v_col = n_iter * MMA_N;
                const uint32_t v_addr = cvta_to_shared_u32(&Vs[v_row][v_col ^ ldm_mask]);

                uint32_t v0, v1;
                ldmatrix_x2_trans(v_addr, v0, v1);

                mma_m16n8k16(
                    p0, p1, p2, p3, v0, v1,
                    u[n_iter][0], u[n_iter][1],
                    u[n_iter][2], u[n_iter][3]
                );
            }
        }

        m0 = m_new0;
        m1 = m_new1;
        l0 = l_new0;
        l1 = l_new1;
    }
}

__device__ __forceinline__ void store_output(
    half* O, int N, int base_row, int warpid, int laneid,
    const float (&u)[OUT_FRAGS][4], float l0, float l1
) {
    const int groupid = laneid / 4;
    const int lane_in_group = laneid % 4;
    const int q_row0 = base_row + warpid * MMA_M + groupid;
    const int q_row1 = q_row0 + 8;
    const float inv_l0 = 1.0f / l0;
    const float inv_l1 = 1.0f / l1;

    #pragma unroll
    for (int n_iter = 0; n_iter < OUT_FRAGS; ++n_iter) {
        const int col0 = n_iter * MMA_N + lane_in_group * 2;
        const int col1 = col0 + 1;

        if (q_row0 < N) {
            O[q_row0 * d + col0] = __float2half_rn(u[n_iter][0] * inv_l0);
            O[q_row0 * d + col1] = __float2half_rn(u[n_iter][1] * inv_l0);
        }
        if (q_row1 < N) {
            O[q_row1 * d + col0] = __float2half_rn(u[n_iter][2] * inv_l1);
            O[q_row1 * d + col1] = __float2half_rn(u[n_iter][3] * inv_l1);
        }
    }
}

__global__ void flashattention2_v2b(
    const half* Q, const half* Kt, const half* V, half* O, int N
) {
    const int tid = threadIdx.x;
    const int warpid = tid / WARP_SIZE;
    const int laneid = tid % WARP_SIZE;
    const int base_row = blockIdx.x * Br;

    __shared__ __align__(16) half Qs[Br][d];
    __shared__ __align__(16) half Ks[d][Bc];
    __shared__ __align__(16) half Vs[Bc][d];

    load_tile_async(Qs, Q, d, base_row, 0);
    cp_async_commit();
    cp_async_wait();
    __syncthreads();

    float u[OUT_FRAGS][4] = {0.0f};
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0f, l1 = 0.0f;

    for (int kv_base = 0; kv_base < N; kv_base += Bc) {
        load_tile_async(Ks, Kt, N, 0, kv_base);
        load_tile_async(Vs, V, d, kv_base, 0);
        cp_async_commit();
        cp_async_wait();
        __syncthreads();

        compute_tile(Qs, Ks, Vs, kv_base, N, warpid, laneid,
                     u, m0, m1, l0, l1);
        __syncthreads();
    }

    store_output(O, N, base_row, warpid, laneid, u, l0, l1);
}

__global__ void flashattention2_v2c(
    const half* Q, const half* Kt, const half* V, half* O, int N
) {
    const int tid = threadIdx.x;
    const int warpid = tid / WARP_SIZE;
    const int laneid = tid % WARP_SIZE;
    const int base_row = blockIdx.x * Br;

    __shared__ __align__(16) half Qs[Br][d];
    __shared__ __align__(16) half Ks[2][d][Bc];
    __shared__ __align__(16) half Vs[2][Bc][d];

    load_tile_async(Qs, Q, d, base_row, 0);
    cp_async_commit();
    cp_async_wait();
    __syncthreads();

    float u[OUT_FRAGS][4] = {0.0f};
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0f, l1 = 0.0f;

    const int num_tiles = N / Bc;

    load_tile_async(Ks[0], Kt, N, 0, 0);
    load_tile_async(Vs[0], V, d, 0, 0);
    cp_async_commit();
    cp_async_wait();
    __syncthreads();

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int cur = tile & 1;
        const int next = cur ^ 1;
        const int kv_base = tile * Bc;

        if (tile + 1 < num_tiles) {
            const int next_base = (tile + 1) * Bc;
            load_tile_async(Ks[next], Kt, N, 0, next_base);
            load_tile_async(Vs[next], V, d, next_base, 0);
            cp_async_commit();
        }

        compute_tile(Qs, Ks[cur], Vs[cur], kv_base, N, warpid, laneid,
                     u, m0, m1, l0, l1);
        __syncthreads();

        if (tile + 1 < num_tiles) {
            cp_async_wait();
            __syncthreads();
        }
    }

    store_output(O, N, base_row, warpid, laneid, u, l0, l1);
}

__global__ void prepare_inputs_kernel(
    const float* Q, const float* K, const float* V,
    half* Qh, half* Kt, half* Vh, int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * d) return;

    const int row = idx / d;
    const int col = idx % d;

    Qh[idx] = __float2half_rn(Q[idx]);
    Kt[col * N + row] = __float2half_rn(K[idx]);
    Vh[idx] = __float2half_rn(V[idx]);
}

}

void launch_prepare_flashattention2_v2(
    const float* Q, const float* K, const float* V,
    half* Qh, half* Kt, half* Vh, int N, cudaStream_t stream
) {
    if (N <= 0) return;
    const int threads = 256;
    const int blocks = (N * fa2_v2::d + threads - 1) / threads;
    fa2_v2::prepare_inputs_kernel<<<blocks, threads, 0, stream>>>(
        Q, K, V, Qh, Kt, Vh, N
    );
}

void launch_flashattention2_v2b(
    const half* Q, const half* Kt, const half* V,
    half* O, int N, cudaStream_t stream
) {
    if (N <= 0) return;
    if (N % fa2_v2::Bc != 0) {
        throw std::invalid_argument("V2-B requires N to be a multiple of 64");
    }

    dim3 block(fa2_v2::NUM_THREADS);
    dim3 grid(N / fa2_v2::Br);
    fa2_v2::flashattention2_v2b<<<grid, block, 0, stream>>>(Q, Kt, V, O, N);
}

void launch_flashattention2_v2c(
    const half* Q, const half* Kt, const half* V,
    half* O, int N, cudaStream_t stream
) {
    if (N <= 0) return;
    if (N % fa2_v2::Bc != 0) {
        throw std::invalid_argument("V2-C requires N to be a multiple of 64");
    }

    dim3 block(fa2_v2::NUM_THREADS);
    dim3 grid(N / fa2_v2::Br);
    fa2_v2::flashattention2_v2c<<<grid, block, 0, stream>>>(Q, Kt, V, O, N);
}
