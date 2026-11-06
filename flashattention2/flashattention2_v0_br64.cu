
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdexcept>

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

__global__ void flashattention2_v0_br64(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const int N
) {
    const int tid = threadIdx.x;
    const int warpid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    const int block_row = blockIdx.x * BR;
    const int warp_row = block_row + warpid * ROWS_PER_WARP;
    if(warp_row >= N) return;

    extern __shared__ float Qs[][D];
    __shared__ float Ks[D][BC + 1];
    __shared__ float Vs[BC][D];

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
                p0 += Qs[warpid * ROWS_PER_WARP + i][k] * Ks[k][lane];
                p0 += Qs[warpid * ROWS_PER_WARP + i][k + WARP_SIZE] * Ks[k + WARP_SIZE][lane];
                p1 += Qs[warpid * ROWS_PER_WARP + i][k] * Ks[k][lane + WARP_SIZE];
                p1 += Qs[warpid * ROWS_PER_WARP + i][k + WARP_SIZE] * Ks[k + WARP_SIZE][lane + WARP_SIZE];
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
                pv0 += __shfl_sync(0xffffffff, p0, k) * Vs[k][lane];
                pv0 += __shfl_sync(0xffffffff, p1, k) * Vs[k + WARP_SIZE][lane];
                pv1 += __shfl_sync(0xffffffff, p0, k) * Vs[k][lane + WARP_SIZE];
                pv1 += __shfl_sync(0xffffffff, p1, k) * Vs[k + WARP_SIZE][lane + WARP_SIZE];
            }
            u0[i] = u0[i] * alpha + pv0;
            u1[i] = u1[i] * alpha + pv1;
        }
        __syncthreads();
    }   
    #pragma unroll
        for(int i = 0; i < ROWS_PER_WARP; i++){
            O[(warp_row + i) * D + lane] = u0[i] / l[i];
            O[(warp_row + i) * D + lane + WARP_SIZE] = u1[i] / l[i];
        } 
}
void launch_flashattention2_v0_br64(const float* q, const float* k, const float* v,
                                    float* o, int n, cudaStream_t stream) {
    if (n <= 0) return;
    if (n % BR) throw std::invalid_argument("v0_br64 requires N divisible by 64");
    constexpr int bytes = BR * D * sizeof(float);
    const auto status = cudaFuncSetAttribute(flashattention2_v0_br64,
        cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    flashattention2_v0_br64<<<n / BR, NUM_WARPS * WARP_SIZE, bytes, stream>>>(q,k,v,o,n);
}
