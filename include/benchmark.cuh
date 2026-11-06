#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

#include "cuda_check.cuh"

template <typename Func>
float benchmark_cuda(
    Func func,
    int warmup = 10,
    int repeat = 100,
    cudaStream_t stream = 0
) {
    if (warmup < 0 || repeat <= 0) {
        throw std::invalid_argument(
            "warmup must >= 0 and repeat must > 0"
        );
    }

    for (int i = 0; i < warmup; ++i) {
        func();
    }

    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));

    for (int i = 0; i < repeat; ++i) {
        func();
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        ) 
    );

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repeat;
}