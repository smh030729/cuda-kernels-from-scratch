#pragma once

#include<cstddef>
#include<cuda_runtime.h>

constexpr int THREADS = 256;
constexpr int ITEM_PER_THREAD = 4;

void launch_histogram_v1(
    const unsigned char* input,
    unsigned int* output,
    std::size_t n,
    cudaStream_t stream = 0
);

void launch_histogram_v2(
    const unsigned char* input,
    unsigned int* output,
    std::size_t n,
    cudaStream_t stream = 0
);
