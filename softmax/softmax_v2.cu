#include"reduction_utils.cuh"
#include"softmax.cuh"
#include<cmath>
#include<stdexcept>

constexpr int THREADS = 256;
constexpr int NUM_WARPS = THREADS / WARP_SIZE;

__global__ void softmax_reg_vec(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols
){  
    int tid = threadIdx.x;
    std::size_t row = static_cast<std::size_t>(blockIdx.x);
    input += row * cols;
    output += row * cols;
    extern __shared__ float smem[];
    __shared__ float block_max;
    __shared__ float block_sum;
    float local_max = -INFINITY;
    std::size_t vec_cols = cols / 4;
    float4 val;
    bool isvalid = static_cast<std::size_t>(tid) < vec_cols;

    if(isvalid){
        val = reinterpret_cast<const float4 *>(input)[tid];
        local_max = fmaxf(val.x, local_max);
        local_max = fmaxf(val.y, local_max);
        local_max = fmaxf(val.z, local_max);
        local_max = fmaxf(val.w, local_max);
    }
    local_max = block_reduce_max(local_max,smem);
    if(threadIdx.x == 0) block_max = local_max;
    __syncthreads();

    float local_sum = 0.0f;
    if(isvalid){
        val.x = expf(val.x - block_max);
        val.y = expf(val.y - block_max);
        val.z = expf(val.z - block_max);
        val.w = expf(val.w - block_max);

        local_sum += val.x;
        local_sum += val.y;
        local_sum += val.z;
        local_sum += val.w;
    }
    local_sum = block_reduce_sum(local_sum, smem);
    if(threadIdx.x == 0) block_sum = local_sum;
    __syncthreads();

    float inv_sum = 1.0 / block_sum;
    if(isvalid){
       val.x *= inv_sum;
       val.y *= inv_sum;
       val.z *= inv_sum;
       val.w *= inv_sum;
       reinterpret_cast<float4*>(output)[tid] = val;
    }
}

__global__ void softmax_shared_vec(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols
){  
    int tid = threadIdx.x;
    std::size_t row = static_cast<std::size_t>(blockIdx.x);
    input += row * cols;
    output += row * cols;
    std::size_t vec_cols = cols / 4;
    __shared__ float smem[NUM_WARPS];
    extern __shared__ float4 val[];
    __shared__ float block_max;
    __shared__ float block_sum;
    float local_max = -INFINITY;

    for(std::size_t i = tid ; i < vec_cols; i += blockDim.x){
        val[i] = reinterpret_cast<const float4 *>(input)[i];
        local_max = fmaxf(val[i].x, local_max);
        local_max = fmaxf(val[i].y, local_max);
        local_max = fmaxf(val[i].z, local_max);
        local_max = fmaxf(val[i].w, local_max);
    }
    local_max = block_reduce_max(local_max,smem);
    if(threadIdx.x == 0) block_max = local_max;
    __syncthreads();

    float local_sum = 0.0f;
    for(std::size_t i = tid ; i < vec_cols; i += blockDim.x){
        val[i].x = expf(val[i].x - block_max);
        val[i].y = expf(val[i].y - block_max);
        val[i].z = expf(val[i].z - block_max);
        val[i].w = expf(val[i].w - block_max);

        local_sum += val[i].x;
        local_sum += val[i].y;
        local_sum += val[i].z;
        local_sum += val[i].w;
    }
    local_sum = block_reduce_sum(local_sum, smem);
    if(threadIdx.x == 0) block_sum = local_sum;
    __syncthreads();

    float inv_sum = 1.0 / block_sum;
    for(std::size_t i = tid ; i < vec_cols; i += blockDim.x){
        val[i].x *= inv_sum;
        val[i].y *= inv_sum;
        val[i].z *= inv_sum;
        val[i].w *= inv_sum;
        reinterpret_cast<float4*>(output)[i] = val[i];
    }
}

void launch_softmax_reg_vec(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols,
    cudaStream_t stream
) {
    softmax_reg_vec<<<
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

void launch_softmax_shared_vec(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols,
    cudaStream_t stream
) {
    softmax_shared_vec<<<
        rows,
        THREADS,
        cols * sizeof(float),
        stream
    >>>(
        input,
        output,
        rows,
        cols
    );
}

void launch_softmax(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols,
    cudaStream_t stream
) {
    if(rows == 0 || cols == 0) return;
    if(cols % 4 != 0) launch_softmax_v1(
            input,
            output,
            rows,
            cols,
            stream
        );
    else if(4 * THREADS >= cols) launch_softmax_reg_vec(
            input,
            output,
            rows,
            cols,
            stream
        );
    else if(8 * THREADS >= cols) launch_softmax_shared_vec(
            input,
            output,
            rows,
            cols,
            stream
        );
    else launch_softmax_v1(
            input,
            output,
            rows,
            cols,
            stream
        );
}