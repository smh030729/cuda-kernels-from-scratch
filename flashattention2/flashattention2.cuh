#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

void launch_flashattention2_v0(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N,
    cudaStream_t stream = 0
);

void launch_flashattention2_v0_br64(const float* q, const float* k, const float* v,
                                    float* o, int n, cudaStream_t stream = 0);

void launch_flashattention2_v1(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N,
    cudaStream_t stream = 0
);

void launch_flashattention2_v2a(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N,
    cudaStream_t stream = 0
);

// FP16 Q/K/V are row-major [N, 64]; accumulation is FP32 and O is FP16.
void launch_flashattention2_v0_fp16(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int N,
    cudaStream_t stream = 0
);

// BR64 SIMT baseline; requires N % 64 == 0 (N <= 0 is a no-op).
void launch_flashattention2_v0_br64_fp16(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int N,
    cudaStream_t stream = 0
);

void launch_flashattention2_v1_fp16(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int N,
    cudaStream_t stream = 0
);

// Original V1 variant retaining an FP32 output tensor.
void launch_flashattention2_v1_fp16(const half* q, const half* k, const half* v,
                                    float* o, int n, cudaStream_t stream);

void launch_flashattention2_v2a_fp16(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int N,
    cudaStream_t stream = 0
);

// Qh/Vh are [N, 64]; Kt is the transposed FP16 K, [64, N].
void launch_prepare_flashattention2_v2(
    const float* Q,
    const float* K,
    const float* V,
    half* Qh,
    half* Kt,
    half* Vh,
    int N,
    cudaStream_t stream = 0
);

// V2b/V2c require N to be a positive multiple of 64 (N <= 0 is a no-op).
void launch_flashattention2_v2b(
    const half* Q,
    const half* Kt,
    const half* V,
    half* O,
    int N,
    cudaStream_t stream = 0
);

void launch_flashattention2_v2c(
    const half* Q,
    const half* Kt,
    const half* V,
    half* O,
    int N,
    cudaStream_t stream = 0
);

// Br=64, Bc=32, Q fragments cached in registers; same Kt layout and N constraints as V2c.
void launch_flashattention2_v2c_bc32_qreg(
    const half* Q,
    const half* Kt,
    const half* V,
    half* O,
    int N,
    cudaStream_t stream = 0
);
