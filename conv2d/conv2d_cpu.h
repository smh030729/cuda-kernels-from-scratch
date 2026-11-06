#pragma once

void conv2d_cpu(
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
    int pad_w = 0
);