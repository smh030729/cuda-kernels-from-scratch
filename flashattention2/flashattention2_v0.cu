#include <math_constants.h>

#include "flashattention2.cuh"

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;
constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;

constexpr int Br = NUM_WARPS;
constexpr int Bc = 64;
constexpr int d = 64;

constexpr int KEYS_PER_LANE = Bc / WARP_SIZE;
constexpr int OUTS_PER_LANE = d / WARP_SIZE;

constexpr float SCALE = 0.125f; // 1 / √d

__device__ float warp_reduce_max(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
  }
  return val;
}

__device__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

__global__ void flashattention2_v0(const float* Q, const float* K,
                                   const float* V, float* O, int N) {
  const int tid = threadIdx.x;
  const int warpid = tid / WARP_SIZE;
  const int laneid = tid % WARP_SIZE;
  const int bid = blockIdx.x;

  __shared__ float Qs[Br][d];
  __shared__ float Ks[d][Bc + 1];
  __shared__ float Vs[Bc][d];
  float s[KEYS_PER_LANE];
  float p[KEYS_PER_LANE];
  float u[OUTS_PER_LANE] = {0.0f};

  float m = -CUDART_INF_F;
  float l = 0.0f;

  const int num_qs = Br * d;
  const int base_row = bid * Br;
  for (int idx = tid; idx < num_qs; idx += blockDim.x) {
    const int row = idx / d;
    const int col = idx % d;
    const int global_row = base_row + row;
    if (global_row < N && row < Br)
      Qs[row][col] = Q[global_row * d + col];
    else
      Qs[row][col] = 0.0f;
  }
  __syncthreads();

  for (int kv_base = 0; kv_base < N; kv_base += Bc) {
    const int num_kvs = Bc * d;
    for (int idx = tid; idx < num_kvs; idx += blockDim.x) {
      const int row = idx / d;
      const int col = idx % d;
      const int global_row = kv_base + row;
      if (global_row < N) {
        Ks[col][row] = K[global_row * d + col];
        Vs[row][col] = V[global_row * d + col];
      } else {
        Ks[col][row] = 0.0f;
        Vs[row][col] = 0.0f;
      }
    }
    __syncthreads();
    float m_new;
    float l_new;
    float alpha;
    float local_max = -CUDART_INF_F;
    for (int bc_iter = 0; bc_iter < KEYS_PER_LANE; bc_iter++) {
      s[bc_iter] = 0.0f;
      int local_key = laneid + bc_iter * WARP_SIZE;
      if (kv_base + local_key < N) {
        for (int k = 0; k < d; k++) {
          s[bc_iter] += Qs[warpid][k] * Ks[k][laneid + bc_iter * WARP_SIZE];
        }
        s[bc_iter] *= SCALE;
      } else {
        s[bc_iter] = -CUDART_INF_F;
      }
      local_max = fmaxf(local_max, s[bc_iter]);
    }
    m_new = fmaxf(m, warp_reduce_max(local_max));
    alpha = expf(m - m_new);
    float local_sum = 0.0f;
    for (int bc_iter = 0; bc_iter < KEYS_PER_LANE; bc_iter++) {
      p[bc_iter] = expf(s[bc_iter] - m_new);
      local_sum += p[bc_iter];
    }
    l_new = alpha * l + warp_reduce_sum(local_sum);
    for (int d_iter = 0; d_iter < OUTS_PER_LANE; d_iter++) {
      u[d_iter] *= alpha;
      for (int bc_iter = 0; bc_iter < KEYS_PER_LANE; bc_iter++) {
        for (int laneidx = 0; laneidx < WARP_SIZE; laneidx++) {
          float pi = __shfl_sync(0xffffffff, p[bc_iter], laneidx);
          u[d_iter] +=
              pi *
              Vs[bc_iter * WARP_SIZE + laneidx][laneid + d_iter * WARP_SIZE];
        }
      }
    }
    m = m_new;
    l = l_new;
    __syncthreads();
  }
  float sm_scale = 1.0f / l;
  for (int d_iter = 0; d_iter < OUTS_PER_LANE; d_iter++) {
    u[d_iter] *= sm_scale;
    const int global_row = base_row + warpid;
    if (global_row < N)
      O[global_row * d + laneid + d_iter * WARP_SIZE] = u[d_iter];
  }
}

void launch_flashattention2_v0(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N,
    cudaStream_t stream
) {
    constexpr int BR = 4;
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_THREADS = BR * WARP_SIZE;

    dim3 block(NUM_THREADS);
    dim3 grid((N + BR - 1) / BR);

    flashattention2_v0<<<grid, block, 0, stream>>>(Q, K, V, O, N);
}
