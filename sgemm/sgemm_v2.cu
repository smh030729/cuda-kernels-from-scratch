#include <cuda_runtime.h>

#include <cassert>

#include "sgemm.cuh"

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;

constexpr int WM = 64;
constexpr int WN = 64;

constexpr int WNITER = 2;

constexpr int TM = 8;
constexpr int TN = 4;

constexpr int WARP_SIZE = 32;
constexpr int NUM_THREADS = 128;

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN>
__global__ void sgemm_v2(const float* A, const float* B, float* C, int M, int N,
                         int K, float alpha, float beta) {
  constexpr int WMITER = (WM * WN) / (WARP_SIZE * TM * TN * WNITER);
  constexpr int WSUBM = WM / WMITER;
  constexpr int WSUBN = WN / WNITER;

  __shared__ float As[BK][BM];
  __shared__ float Bs[BK][BN];

  const int tid = threadIdx.x;
  const int warp_idx = tid / WARP_SIZE;
  const int lane_idx = tid % WARP_SIZE;
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;

  A += block_row * K;
  B += block_col;
  C += block_row * N + block_col;

  constexpr int WARPS_COL = BN / WN;
  const int warp_row = warp_idx / WARPS_COL;
  const int warp_col = warp_idx % WARPS_COL;

  constexpr int THREADS_COL_IN_SUBTILE = WSUBN / TN;
  const int thread_row_in_warp = lane_idx / THREADS_COL_IN_SUBTILE;
  const int thread_col_in_warp = lane_idx % THREADS_COL_IN_SUBTILE;

  float reg_m[WMITER * TM];
  float reg_n[WNITER * TN];

  float thread_results[WMITER * TM][WNITER * TN] = {0.0f};

  for (int bk = 0; bk < K; bk += BK) {
#pragma unroll
    for (int idx = tid * 4; idx < BM * BK; idx += 4 * NUM_THREADS) {
      int row = idx / BK;
      int col = idx % BK;
      float4 tmp = *reinterpret_cast<const float4*>(&(A[bk + row * K + col]));
      As[col + 0][row] = tmp.x;
      As[col + 1][row] = tmp.y;
      As[col + 2][row] = tmp.z;
      As[col + 3][row] = tmp.w;
    }

#pragma unroll
    for (int idx = tid * 4; idx < BK * BN; idx += 4 * NUM_THREADS) {
      int row = idx / BN;
      int col = idx % BN;
      float4 tmp = *reinterpret_cast<const float4*>(&(B[(bk + row) * N + col]));
      *reinterpret_cast<float4*>(&(Bs[row][col])) = tmp;
    }
    __syncthreads();

#pragma unroll
    for (int idx_dot = 0; idx_dot < BK; idx_dot++) {
#pragma unroll
      for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; w_sub_row_idx++) {
#pragma unroll
        for (int i = 0; i < TM; i++) {
          reg_m[w_sub_row_idx * TM + i] =
              As[idx_dot][warp_row * WM + w_sub_row_idx * WSUBM +
                          thread_row_in_warp * TM + i];
        }
      }

#pragma unroll
      for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; w_sub_col_idx++) {
#pragma unroll
        for (int i = 0; i < TN; i++) {
          reg_n[w_sub_col_idx * TN + i] =
              Bs[idx_dot][warp_col * WN + w_sub_col_idx * WSUBN +
                          thread_col_in_warp * TN + i];
        }
      }

#pragma unroll
      for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; w_sub_row_idx++) {
#pragma unroll
        for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; w_sub_col_idx++) {
#pragma unroll
          for (int i = 0; i < TM; i++) {
#pragma unroll
            for (int j = 0; j < TN; j++) {
              thread_results[w_sub_row_idx * TM + i][w_sub_col_idx * TN + j] +=
                  reg_m[w_sub_row_idx * TM + i] * reg_n[w_sub_col_idx * TN + j];
            }
          }
        }
      }
    }
    __syncthreads();
  }
// 写回C
#pragma unroll
  for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; w_sub_row_idx++) {
#pragma unroll
    for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; w_sub_col_idx++) {
#pragma unroll
      for (int i = 0; i < TM; i++) {
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
          float4 out;
          const int result_row = w_sub_row_idx * TM + i;
          const int result_col = w_sub_col_idx * TN + j;
          out.x = thread_results[result_row][result_col + 0];
          out.y = thread_results[result_row][result_col + 1];
          out.z = thread_results[result_row][result_col + 2];
          out.w = thread_results[result_row][result_col + 3];
          if (beta != 0) {
            float4 old = *reinterpret_cast<const float4*>(
                &C[(warp_row * WM + w_sub_row_idx * WSUBM +
                    thread_row_in_warp * TM + i) *
                       N +
                   warp_col * WN + w_sub_col_idx * WSUBN +
                   thread_col_in_warp * TN + j]);
            out.x = out.x * alpha + beta * old.x;
            out.y = out.y * alpha + beta * old.y;
            out.z = out.z * alpha + beta * old.z;
            out.w = out.w * alpha + beta * old.w;
          } else {
            out.x = out.x * alpha;
            out.y = out.y * alpha;
            out.z = out.z * alpha;
            out.w = out.w * alpha;
          }
          *reinterpret_cast<float4*>(&C[(warp_row * WM + w_sub_row_idx * WSUBM +
                                         thread_row_in_warp * TM + i) *
                                            N +
                                        warp_col * WN + w_sub_col_idx * WSUBN +
                                        thread_col_in_warp * TN + j]) = out;
        }
      }
    }
  }
}

void launch_sgemm_v2(const float* A, const float* B, float* C, int M, int N,
                     int K, float alpha, float beta, cudaStream_t stream) {
  if (M == 0 || N == 0 || K == 0) {
    return;
  }

  // 当前 kernel 暂不处理边界
  assert(M % BM == 0);
  assert(N % BN == 0);
  assert(K % BK == 0);

  dim3 block(NUM_THREADS);

  dim3 grid(N / BN, M / BM);

  sgemm_v2<BM, BN, BK, WM, WN, WNITER, TM, TN>
      <<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}