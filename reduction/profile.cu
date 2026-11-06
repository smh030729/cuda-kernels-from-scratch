#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "reduction.cuh"
#include "reduction_utils.cuh"


constexpr int THREADS = 256;


int main()
{
    // 固定一个规模
    // 2^24 = 16M floats
    const size_t N = 1 << 24;


    float* h_input = nullptr;

    cudaMallocHost(
        &h_input,
        N * sizeof(float)
    );


    for(size_t i = 0; i < N; i++)
    {
        h_input[i] = 1.0f;
    }


    float* d_input;
    float* d_output;


    cudaMalloc(
        &d_input,
        N * sizeof(float)
    );


    // reduction 第一层 block 数
    size_t num_blocks =
        (N + 2 * THREADS - 1)
        /
        (2 * THREADS);


    cudaMalloc(
        &d_output,
        num_blocks * sizeof(float)
    );


    cudaMemcpy(
        d_input,
        h_input,
        N * sizeof(float),
        cudaMemcpyHostToDevice
    );


    /*
        profile kernel
    */

    dim3 block(
        THREADS
    );

    dim3 grid(
        num_blocks
    );


    size_t smem_size =
        (THREADS / 32)
        *
        sizeof(float);



    printf(
        "N=%zu block=%d grid=%zu smem=%zu\n",
        N,
        THREADS,
        num_blocks,
        smem_size
    );


    reduction_v1<<<
        grid,
        block,
        smem_size
    >>>(
        d_input,
        d_output,
        N
    );


    cudaError_t err =
        cudaDeviceSynchronize();


    if(err != cudaSuccess)
    {
        printf(
            "CUDA error: %s\n",
            cudaGetErrorString(err)
        );
        return -1;
    }


    float result;

    cudaMemcpy(
        &result,
        d_output,
        sizeof(float),
        cudaMemcpyDeviceToHost
    );


    printf(
        "result=%f\n",
        result
    );


    cudaFree(d_input);
    cudaFree(d_output);

    cudaFreeHost(h_input);


    return 0;
}