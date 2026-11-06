#include "sgemm_v3.cuh"
#include <cstdint>
#include <stdexcept>

namespace {
constexpr int BM = 128, BN = 128, BK = 16;
constexpr int THREADS = 256, WM = 32, WN = 64, TM = 8, TN = 4;

__device__ __forceinline__ void copy16(float* dst, const float* src) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
               "r"(address), "l"(src) : "memory");
}

// A's XOR layout separates transposed stores and preserves float4 reads along M.
__device__ __forceinline__ int a_offset(int row, int k) {
  return k * BM + (row ^ (((k / 4) & 3) * 8));
}

__device__ __forceinline__ void store_a(float* shared, int row, int k, float4 value) {
  shared[a_offset(row, k)] = value.x;
  shared[a_offset(row, k + 1)] = value.y;
  shared[a_offset(row, k + 2)] = value.z;
  shared[a_offset(row, k + 3)] = value.w;
}

template<bool ASYNC>
__device__ __forceinline__ void load_b(const float* B, float* shared,
                                      int N, int col0, int k0) {
  #pragma unroll
  for (int index = threadIdx.x * 4; index < BK * BN; index += THREADS * 4) {
    const float* src = B + size_t(k0 + index / BN) * N + col0 + index % BN;
    if constexpr (ASYNC) copy16(shared + index, src);
    else *reinterpret_cast<float4*>(shared + index) = *reinterpret_cast<const float4*>(src);
  }
  if constexpr (ASYNC) asm volatile("cp.async.commit_group;" ::: "memory");
}

template<bool ASYNC>
__device__ __forceinline__ void wait_tile() {
  if constexpr (ASYNC) asm volatile("cp.async.wait_group 0;" ::: "memory");
  // wait_group is per-thread; the barrier also publishes A's ordinary shared stores.
  __syncthreads();
}

template<bool ASYNC_B, int STAGES>
__global__ __launch_bounds__(THREADS) void sgemm_v3_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int N, int K, float alpha, float beta) {
  __shared__ __align__(16) float As[STAGES][BM * BK];
  __shared__ __align__(16) float Bs[STAGES][BK * BN];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int warp_m = warp / 2, warp_n = warp % 2;
  const int lane_m = lane / 8, lane_n = lane % 8;
  const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
  float accum[TM][2 * TN] = {};
  constexpr int A_COPIES = BM * BK / (THREADS * 4);
  #pragma unroll
  for (int i = 0; i < A_COPIES; ++i) {
    const int index = (threadIdx.x + i * THREADS) * 4;
    const float4 value = *reinterpret_cast<const float4*>(
        A + size_t(row0 + index / BK) * K + index % BK);
    store_a(As[0], index / BK, index % BK, value);
  }
  load_b<ASYNC_B>(B, Bs[0], N, col0, 0);
  wait_tile<ASYNC_B>();

  int stage = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    const int next = k0 + BK;
    float4 prefetch_a[A_COPIES];
    if constexpr (STAGES == 2) {
      if (next < K) {
        #pragma unroll
        for (int i = 0; i < A_COPIES; ++i) {
          const int index = (threadIdx.x + i * THREADS) * 4;
          prefetch_a[i] = *reinterpret_cast<const float4*>(
              A + size_t(row0 + index / BK) * K + next + index % BK);
        }
        load_b<ASYNC_B>(B, Bs[stage ^ 1], N, col0, next);
      }
    }
    // Bound live ranges and instruction footprint while retaining some load overlap.
    #pragma unroll 2
    for (int kk = 0; kk < BK; ++kk) {
      float a[TM], b[2 * TN];
      #pragma unroll
      for (int i = 0; i < TM; i += 4) {
        const int row = warp_m * WM + lane_m * TM + i;
        const float4 value = *reinterpret_cast<const float4*>(&As[stage][a_offset(row, kk)]);
        a[i] = value.x; a[i + 1] = value.y;
        a[i + 2] = value.z; a[i + 3] = value.w;
      }
      #pragma unroll
      for (int jn = 0; jn < 2; ++jn) {
        const int col = warp_n * WN + jn * 32 + lane_n * TN;
        const float4 value = *reinterpret_cast<const float4*>(&Bs[stage][kk * BN + col]);
        b[jn * TN] = value.x; b[jn * TN + 1] = value.y;
        b[jn * TN + 2] = value.z; b[jn * TN + 3] = value.w;
      }
      #pragma unroll
      for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < 2 * TN; ++j)
          accum[i][j] = fmaf(a[i], b[j], accum[i][j]);
      }
    }
    if (next < K) {
      if constexpr (STAGES == 2) {
        #pragma unroll
        for (int i = 0; i < A_COPIES; ++i) {
          const int index = (threadIdx.x + i * THREADS) * 4;
          store_a(As[stage ^ 1], index / BK, index % BK, prefetch_a[i]);
        }
      } else {
        __syncthreads(); // Finish all readers before overwriting a single stage.
        #pragma unroll
        for (int i = 0; i < A_COPIES; ++i) {
          const int index = (threadIdx.x + i * THREADS) * 4;
          const float4 value = *reinterpret_cast<const float4*>(
              A + size_t(row0 + index / BK) * K + next + index % BK);
          store_a(As[0], index / BK, index % BK, value);
        }
        load_b<ASYNC_B>(B, Bs[0], N, col0, next);
      }
      wait_tile<ASYNC_B>();
      if constexpr (STAGES == 2) stage ^= 1;
    }
  }
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = row0 + warp_m * WM + lane_m * TM + i;
    #pragma unroll
    for (int jn = 0; jn < 2; ++jn) {
      const int col = col0 + warp_n * WN + jn * 32 + lane_n * TN;
      float4 out = {alpha * accum[i][jn * TN], alpha * accum[i][jn * TN + 1],
                    alpha * accum[i][jn * TN + 2], alpha * accum[i][jn * TN + 3]};
      if (beta != 0.f) {
        const float4 old = *reinterpret_cast<const float4*>(C + size_t(row) * N + col);
        out.x = fmaf(beta, old.x, out.x); out.y = fmaf(beta, old.y, out.y);
        out.z = fmaf(beta, old.z, out.z); out.w = fmaf(beta, old.w, out.w);
      }
      *reinterpret_cast<float4*>(C + size_t(row) * N + col) = out;
    }
  }
}

// Generic boundary / unaligned path, including K=0.
__global__ void sgemm_v3_boundary(const float* A, const float* B, float* C,
                                 int M, int N, int K, float alpha, float beta) {
  __shared__ float a[16][16], b[16][16];
  const int x = threadIdx.x, y = threadIdx.y;
  const int row = blockIdx.y * 16 + y, col = blockIdx.x * 16 + x;
  float sum = 0.f;
  for (int k0 = 0; k0 < K; k0 += 16) {
    a[y][x] = row < M && k0 + x < K ? A[size_t(row) * K + k0 + x] : 0.f;
    b[y][x] = col < N && k0 + y < K ? B[size_t(k0 + y) * N + col] : 0.f;
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 16; ++k) sum = fmaf(a[y][k], b[k][x], sum);
    __syncthreads();
  }
  if (row < M && col < N) {
    const size_t offset = size_t(row) * N + col;
    C[offset] = beta == 0.f ? alpha * sum : fmaf(beta, C[offset], alpha * sum);
  }
}

template<bool ASYNC_B, int STAGES>
void launch(const float* A, const float* B, float* C, int M, int N, int K,
            float alpha, float beta, cudaStream_t stream) {
  const bool aligned = ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B) |
                         reinterpret_cast<uintptr_t>(C)) & 15) == 0;
  if (M % BM || N % BN || K == 0 || K % BK || !aligned) {
    sgemm_v3_boundary<<<dim3((N + 15) / 16, (M + 15) / 16), dim3(16, 16), 0, stream>>>(
        A, B, C, M, N, K, alpha, beta);
  } else {
    sgemm_v3_kernel<ASYNC_B, STAGES><<<dim3(N / BN, M / BM), THREADS, 0, stream>>>(
        A, B, C, N, K, alpha, beta);
  }
}
} // namespace

const char* sgemm_v3_variant_name(int variant) {
  static const char* names[] = {"v3_sync_single", "v3_async_single", "v3_async_double"};
  if (variant < 0 || variant >= SGEMM_V3_VARIANTS) throw std::invalid_argument("invalid v3 variant");
  return names[variant];
}

void launch_sgemm_v3_variant(int variant, const float* A, const float* B, float* C,
                            int M, int N, int K, float alpha, float beta, cudaStream_t stream) {
  if (M < 0 || N < 0 || K < 0) throw std::invalid_argument("negative GEMM dimension");
  if (M == 0 || N == 0) return;
  switch (variant) {
    case 0: return launch<false,1>(A,B,C,M,N,K,alpha,beta,stream);
    case 1: return launch<true,1>(A,B,C,M,N,K,alpha,beta,stream);
    case 2: return launch<true,2>(A,B,C,M,N,K,alpha,beta,stream);
    default: throw std::invalid_argument("invalid v3 variant");
  }
}

void launch_sgemm_v3(const float* A, const float* B, float* C, int M, int N, int K,
                     float alpha, float beta, cudaStream_t stream) {
  launch_sgemm_v3_variant(2, A, B, C, M, N, K, alpha, beta, stream);
}
