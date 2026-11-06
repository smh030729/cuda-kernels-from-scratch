
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdexcept>
#include "flashattention2.cuh"

namespace fa2_v0_br64_fp16 {

constexpr int D = 64;
constexpr int BR = 64;
constexpr int BC = 64;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;
constexpr int ROWS_PER_WARP = BR / NUM_WARPS;

__device__ __forceinline__ float warp_max(float val){
    for(int step = 16; step > 0; step >>= 1){
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, step));
    }
    return val;
}

__device__ __forceinline__ float warp_sum(float val){
    for(int step = 16; step > 0; step >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, step);
    }   
    return val;
}

__global__ void flashattention2_v0_br64_fp16(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    const int N
) {
    const int tid = threadIdx.x;
    const int warpid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    const int block_row = blockIdx.x * BR;
    const int warp_row = block_row + warpid * ROWS_PER_WARP;
    if(warp_row >= N) return;

    __shared__ half Qs[BR][D];
    __shared__ half Ks[D][BC + 1];
    __shared__ half Vs[BC][D];

    #pragma unroll
    for(int idx = tid; idx < BR * D; idx += blockDim.x){
        int row = idx / D;
        int col = idx % D;
        Qs[row][col] = Q[block_row * D + idx];
    }
    __syncthreads();

    float m[ROWS_PER_WARP], l[ROWS_PER_WARP];
    float u0[ROWS_PER_WARP], u1[ROWS_PER_WARP];
    const float scale = rsqrtf(D);

    #pragma unroll
    for(int i = 0; i < ROWS_PER_WARP; i++){
        m[i] = -CUDART_INF_F;
        l[i] = 0.0f;
        u0[i] = 0.0f;
        u1[i] = 0.0f;
    }

    #pragma unroll
    for(int kv_base = 0; kv_base < N; kv_base += BC){
        #pragma unroll
        for(int idx = tid; idx < BC * D; idx += blockDim.x){
            int row = idx / D;
            int col = idx % D;
            Ks[col][row] = K[kv_base * D + idx];
            Vs[row][col] = V[kv_base * D + idx];
        }
        __syncthreads();
        #pragma unroll
        for(int i = 0; i < ROWS_PER_WARP; i++){
            float p0 = 0.0f;
            float p1 = 0.0f;
            float pv0 = 0.0f;
            float pv1 = 0.0f;

            #pragma unroll
            for(int k = 0; k < WARP_SIZE; k++){
                p0 += __half2float(Qs[warpid * ROWS_PER_WARP + i][k]) * __half2float(Ks[k][lane]);
                p0 += __half2float(Qs[warpid * ROWS_PER_WARP + i][k + WARP_SIZE]) * __half2float(Ks[k + WARP_SIZE][lane]);
                p1 += __half2float(Qs[warpid * ROWS_PER_WARP + i][k]) * __half2float(Ks[k][lane + WARP_SIZE]);
                p1 += __half2float(Qs[warpid * ROWS_PER_WARP + i][k + WARP_SIZE]) * __half2float(Ks[k + WARP_SIZE][lane + WARP_SIZE]);
            }
            p0 *= scale;
            p1 *= scale;
            float tile_max = warp_max(fmaxf(p0, p1));
            float new_m = fmaxf(m[i], tile_max);
            float alpha = expf(m[i] - new_m);
            p0 = expf(p0 - new_m);
            p1 = expf(p1 - new_m);
            float tile_sum = warp_sum(p0 + p1);
            m[i] = new_m;
            l[i] = alpha * l[i] + tile_sum;
            #pragma unroll
            for(int k = 0; k < WARP_SIZE; k++){
                pv0 += __shfl_sync(0xffffffff, p0, k) * __half2float(Vs[k][lane]);
                pv0 += __shfl_sync(0xffffffff, p1, k) * __half2float(Vs[k + WARP_SIZE][lane]);
                pv1 += __shfl_sync(0xffffffff, p0, k) * __half2float(Vs[k][lane + WARP_SIZE]);
                pv1 += __shfl_sync(0xffffffff, p1, k) * __half2float(Vs[k + WARP_SIZE][lane + WARP_SIZE]);
            }
            u0[i] = u0[i] * alpha + pv0;
            u1[i] = u1[i] * alpha + pv1;
        }
        __syncthreads();
    }   
    #pragma unroll
        for(int i = 0; i < ROWS_PER_WARP; i++){
            O[(warp_row + i) * D + lane] = __float2half_rn(u0[i] / l[i]);
            O[(warp_row + i) * D + lane + WARP_SIZE] = __float2half_rn(u1[i] / l[i]);
        } 
}
} // namespace fa2_v0_br64_fp16

void launch_flashattention2_v0_br64_fp16(
    const half* Q, const half* K, const half* V,
    half* O, int N, cudaStream_t stream
) {
    if (N <= 0) return;
    if (N % fa2_v0_br64_fp16::BC != 0) {
        throw std::invalid_argument("V0 BR64 requires N to be a multiple of 64");
    }
    dim3 block(fa2_v0_br64_fp16::NUM_WARPS * fa2_v0_br64_fp16::WARP_SIZE);
    dim3 grid(N / fa2_v0_br64_fp16::BR);
    fa2_v0_br64_fp16::flashattention2_v0_br64_fp16<<<grid, block, 0, stream>>>(Q, K, V, O, N);
}
