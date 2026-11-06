#pragma once

#include <cuda_runtime.h>

#include <cmath>

constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val = fmaxf(__shfl_down_sync(0xffffffff, val, offset), val);
  }
  return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* warp_sum) {
  int tid = threadIdx.x;
  int warpid = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  int num_warps = blockDim.x / WARP_SIZE;

  val = warp_reduce_sum(val);
  if (lane == 0) {
    warp_sum[warpid] = val;
  }
  __syncthreads();
  float block_sum = 0.0f;
  if (warpid == 0) {
    block_sum = (lane < num_warps) ? warp_sum[lane] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
  }
  __syncthreads();
  return block_sum;
}

__device__ __forceinline__ float block_reduce_max(float val, float* warp_max) {
  int tid = threadIdx.x;
  int warpid = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  int num_warps = blockDim.x / WARP_SIZE;

  val = warp_reduce_max(val);
  if (lane == 0) {
    warp_max[warpid] = val;
  }
  __syncthreads();
  float block_max = -INFINITY;
  if (warpid == 0) {
    block_max = (lane < num_warps) ? warp_max[lane] : -INFINITY;
    block_max = warp_reduce_max(block_max);
  }
  __syncthreads();
  return block_max;
}