#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Contiguous row-major A/B: FP16; C and accumulation: FP32.
void launch_sgemm_v4(const half* A, const half* B, float* C, int M, int N, int K,
                     float alpha = 1.f, float beta = 0.f, cudaStream_t stream = 0);

constexpr int SGEMM_V4_VARIANTS = 6;
const char* sgemm_v4_variant_name(int variant);
void launch_sgemm_v4_variant(int variant, const half* A, const half* B, float* C,
    int M, int N, int K, float alpha = 1.f, float beta = 0.f, cudaStream_t stream = 0);
