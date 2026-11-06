#include"reduction_utils.cuh"
#include"softmax.cuh"

constexpr int THREADS = 256;
constexpr int NUM_WARPS = THREADS / WARP_SIZE;

__global__ void softmax_v1(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols
){   
    std::size_t row = static_cast<std::size_t>(blockIdx.x);
    input += row * cols;
    output += row * cols;
    extern __shared__ float smem[];
    __shared__ float block_max;
    __shared__ float block_sum;
    float local_max = -INFINITY;
    for(std::size_t i = threadIdx.x ; i < cols; i += blockDim.x){
        local_max = fmaxf(input[i], local_max);
    }
    local_max = block_reduce_max(local_max,smem);
    if(threadIdx.x == 0) block_max = local_max;
    __syncthreads();
    float local_sum = 0.0f;
    for(std::size_t i = threadIdx.x ; i < cols; i += blockDim.x){
        float val = expf(input[i] - block_max);
        output[i] = val;
        local_sum += val;
    }
    local_sum = block_reduce_sum(local_sum, smem);
    if(threadIdx.x == 0) block_sum = local_sum;
    __syncthreads();
    float inv_sum = 1.0 / block_sum;
    for(std::size_t i = threadIdx.x ; i < cols; i += blockDim.x){
        output[i] *= inv_sum;
    }
}

void launch_softmax_v1(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols,
    cudaStream_t stream
) {
    if(rows == 0 || cols == 0) return;
    softmax_v1<<<
    rows,
    THREADS,
    NUM_WARPS * sizeof(float),
    stream
>>>(
    input,
    output,
    rows,
    cols
);
}