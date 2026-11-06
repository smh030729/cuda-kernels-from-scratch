#pragma once

#include<cstddef>
#include<cmath>
#include<cuda_runtime.h>

void launch_rmsnorm_v1(
    const float* input,
    float* output,
    const float* weight,
    float eps,
    std::size_t rows,
    std::size_t cols,
    cudaStream_t stream = 0
);