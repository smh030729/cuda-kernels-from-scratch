#include"rmsnorm_v1.cuh"
#include"reduction_utils.cuh"

constexpr int THREADS = 256;
constexpr int NUM_WARPS = THREADS / WARP_SIZE;

__global__ void rmsnorm_v1(
    const float* input,
    float* output,
    const float* weight,
    float eps,
    std::size_t rows,
    std::size_t cols
) {
    std::size_t row = blockIdx.x;
    input += row * cols;
    output += row * cols;

    __shared__ float block_sum;
    __shared__ float warp_sum[NUM_WARPS];
    std::size_t vec_cols = cols / 4;
    float val = 0.0f;
    float4 in, wei;
    if(cols % 4 == 0){
        for(std::size_t tid = threadIdx.x; tid < vec_cols; tid += blockDim.x){
            in = reinterpret_cast<const float4*>(input)[tid];
            val += in.x * in.x;
            val += in.y * in.y;
            val += in.z * in.z;
            val += in.w * in.w;
        }
    }
    else {
        for(std::size_t tid = threadIdx.x; tid < cols; tid += blockDim.x){
            val += input[tid] * input[tid];
        }
    }
    
    val = block_reduce_sum(val, warp_sum);
    if(threadIdx.x == 0) {
        block_sum = val;
        block_sum = 1.0f / sqrtf(block_sum / cols + eps);
    }
    __syncthreads();
    val = block_sum;
    if(cols % 4 == 0){
        for(std::size_t tid = threadIdx.x; tid < vec_cols; tid += blockDim.x){
            in = reinterpret_cast<const float4*>(input)[tid];
            wei = reinterpret_cast<const float4*>(weight)[tid];
            in.x *= val * wei.x;
            in.y *= val * wei.y;
            in.z *= val * wei.z;
            in.w *= val * wei.w;
            reinterpret_cast<float4*>(output)[tid] = in;
        }
    }
    else {
        for(std::size_t tid = threadIdx.x; tid < cols; tid += blockDim.x){
            output[tid] = input[tid] * weight[tid] * val;
        }
    }
    
}

void launch_rmsnorm_v1(
    const float* input,
    float* output,
    const float* weight,
    float eps,
    std::size_t rows,
    std::size_t cols,
    cudaStream_t stream
) {
    rmsnorm_v1<<<
        rows,
        THREADS,
        0,
        stream
    >>> (
        input,
        output,
        weight,
        eps,
        rows,
        cols
    );
}