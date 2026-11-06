#pragma once

#include<cuda_runtime.h>
#include<cstddef>


void launch_softmax_v1(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols,
    cudaStream_t stream = 0
);

void launch_softmax_reg_vec(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols,
    cudaStream_t stream = 0
);

void launch_softmax_shared_vec(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols,
    cudaStream_t stream = 0
);

void launch_softmax(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols,
    cudaStream_t stream = 0
);