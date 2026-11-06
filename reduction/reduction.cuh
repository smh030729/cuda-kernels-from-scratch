#pragma once

#include<cuda_runtime.h>
#include<cstddef>

std::size_t reduction_v1_workspace_size(std::size_t n);

__global__ void reduction_v1(
    const float* input,
    float* output,
    std::size_t n
);

void launch_reduction_v1(
    const float* input,
    float* output,
    float* workspace1,
    float* workspace2,
    std::size_t n,
    cudaStream_t stream = 0
);