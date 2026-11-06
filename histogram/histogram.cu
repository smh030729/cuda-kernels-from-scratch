#include"histogram.cuh"

__global__ void histogram_v1(
    const unsigned char* input,
    unsigned int* output,
    std::size_t n
) {
    std::size_t idx =blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        atomicAdd(
            &output[input[idx]],
            1
        );
    }
}

__global__ void histogram_v2(
    const unsigned char* input,
    unsigned int* output,
    std::size_t n
) {
    __shared__ unsigned int hint[256];
    for(int tid = threadIdx.x; tid < 256; tid += blockDim.x){
        hint[tid] = 0;
    }
    __syncthreads();

    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t stride = gridDim.x * blockDim.x;
    for(std::size_t i = idx; i < n; i += stride){
        atomicAdd(
            &hint[input[i]],
            1
        );
    }
    __syncthreads();

    for(int i = threadIdx.x; i < 256; i += blockDim.x){
        if(hint[i] > 0){
            atomicAdd(
                &output[i],
                hint[i]
            );
        }
    }
}

void launch_histogram_v1(
    const unsigned char* input,
    unsigned int* output,
    std::size_t n,
    cudaStream_t stream
) {
    if(n == 0) return;
    std::size_t num_blocks = (n + THREADS - 1) / THREADS;
    histogram_v1<<<
    num_blocks,
    THREADS,
    0,
    stream
    >>> (
        input,
        output,
        n
    );
}

void launch_histogram_v2(
    const unsigned char* input,
    unsigned int* output,
    std::size_t n,
    cudaStream_t stream
) {
    if(n == 0) return;
    std::size_t num_blocks = (n + THREADS * ITEM_PER_THREAD - 1) / (THREADS * ITEM_PER_THREAD);
    histogram_v2<<<
    num_blocks,
    THREADS,
    0,
    stream
    >>> (
        input,
        output,
        n
    );
}