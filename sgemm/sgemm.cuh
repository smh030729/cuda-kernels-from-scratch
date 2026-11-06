#pragma once

#include <cuda_runtime.h>

void launch_sgemm_v1(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = 0
);

void launch_sgemm_v2(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = 0
);

// Row-major FP32 C = alpha*A*B + beta*C. CUDA Cores; no Tensor Core instructions.
void launch_sgemm_v3(
    const float* A, const float* B, float* C, int M, int N, int K,
    float alpha = 1.0f, float beta = 0.0f, cudaStream_t stream = 0
);
