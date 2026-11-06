#include"reduction.cuh"
#include"reduction_utils.cuh"

constexpr int THREADS = 256;
constexpr int NUM_WARPS = THREADS / WARP_SIZE;

std::size_t reduction_v1_workspace_size(std::size_t n) {
    return (n + 2 * THREADS - 1) / (2 * THREADS);
}

__global__ void reduction_v1(
    const float* input,
    float* output,
    std::size_t n
) {
    extern __shared__ float warp_sums[];

    int tid = threadIdx.x;
    std::size_t idx = blockIdx.x * blockDim.x * 2 + tid;
    float val = 0.0f;
    if(idx < n) val += input[idx];
    if(idx + blockDim.x < n) val += input[idx + blockDim.x];

    float block_sum = block_reduce_sum(val, warp_sums);

    if(tid == 0) output[blockIdx.x] = block_sum;
}

void launch_reduction_v1(
    const float* input,
    float* output,
    float* workspace1,
    float* workspace2,
    std::size_t n,
    cudaStream_t stream
){  
    if(n == 0){
        cudaMemsetAsync(
        output,
        0,
        sizeof(float),
        stream
    );
    return;
    }
    const float* current_input = input;
    float* current_output = nullptr;
    std::size_t current_n = n;
    std::size_t num_blocks = (n + 2*THREADS -1) / (2 * THREADS);
    int num_warps = NUM_WARPS;
    bool use_ws1 = true;

    while(true){
        if(num_blocks == 1) current_output = output;
            else current_output = use_ws1 ? workspace1 : workspace2;
        reduction_v1<<<
            num_blocks,
            THREADS,
            num_warps * sizeof(float),
            stream
        >>>(
            current_input,
            current_output,
            current_n
        );

        if(num_blocks == 1) break;

        current_input = current_output;
        current_n = num_blocks;
        num_blocks = (current_n + 2*THREADS -1) / (2 * THREADS);
        use_ws1 = !use_ws1;
    }
}
