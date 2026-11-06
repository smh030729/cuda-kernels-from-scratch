#pragma once

#include<cuda_runtime.h>

constexpr int BM = 32;
constexpr int BN = 32;

void launch_transpose_v1(
    const float* input,
    float* output,
    int m,
    int n,
    cudaStream_t stream = 0
);

void launch_transpose_v2(
    const float* input,
    float* output,
    int m,
    int n,
    cudaStream_t stream = 0
);

