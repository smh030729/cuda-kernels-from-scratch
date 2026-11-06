#pragma once

#include <cuda_runtime.h>

void launch_conv2d_v2(
    const float* input,
    const float* weight,
    float* output,
    int N,
    int Cin,
    int H,
    int W,
    int Cout,
    int KH,
    int KW,
    int stride_h = 1,
    int stride_w = 1,
    int pad_h = 0,
    int pad_w = 0,
    cudaStream_t stream = 0
);

void launch_conv2d_v3(
    const float* input,
    const float* weight,
    float* output,
    int N,
    int Cin,
    int H,
    int W,
    int Cout,
    int KH,
    int KW,
    int stride_h = 1,
    int stride_w = 1,
    int pad_h = 0,
    int pad_w = 0,
    cudaStream_t stream = 0
);