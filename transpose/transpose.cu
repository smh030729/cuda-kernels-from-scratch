#include"transpose.cuh"

template<const int BM, const int BN>
__global__ void transpose_v1(
    const float* input,
    float* output,
    int M,
    int N
) {
    __shared__ float val[BM][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    input += by * BM * N + bx * BN;
    output += bx * BN * M + by * BM;
    for(int tx = threadIdx.x; tx < BN; tx += blockDim.x)
    {
        for(int ty = threadIdx.y; ty < BM; ty += blockDim.y)
        {
            val[ty][tx] = input[ty * N + tx];
        }
    }
    __syncthreads();

    for(int tx = threadIdx.x; tx < BN; tx += blockDim.x)
    {
        for(int ty = threadIdx.y; ty < BM; ty += blockDim.y)
        {   
            int tid = ty * BN + tx;
            int new_tx = tid % BM;
            int new_ty = tid / BM;
            output[new_ty * M + new_tx] = val[new_tx][new_ty];
        }
    }
}

template<const int BM, const int BN>
__global__ void transpose_v2(
    const float* input,
    float* output,
    int M,
    int N
) {
    __shared__ float val[BM][BN + 1];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    input += by * BM * N + bx * BN;
    output += bx * BN * M + by * BM;
    for(int tx = threadIdx.x; tx < BN; tx += blockDim.x)
    {
        for(int ty = threadIdx.y; ty < BM; ty += blockDim.y)
        {
            val[ty][tx] = input[ty * N + tx];
        }
    }
    __syncthreads();

    for(int tx = threadIdx.x; tx < BN; tx += blockDim.x)
    {
        for(int ty = threadIdx.y; ty < BM; ty += blockDim.y)
        {   
            int tid = ty * BN + tx;
            int new_tx = tid % BM;
            int new_ty = tid / BM;
            output[new_ty * M + new_tx] = val[new_tx][new_ty];
        }
    }
}

void launch_transpose_v1(
    const float* input,
    float* output,
    int M,
    int N,
    cudaStream_t stream
) {
    dim3 block(32, 8);
    dim3 grid(N / BN, M / BM);

    transpose_v1<BM, BN><<<grid, block, 0, stream>>>(
        input,
        output,
        M,
        N
    );
}

void launch_transpose_v2(
    const float* input,
    float* output,
    int M,
    int N,
    cudaStream_t stream
) {
    dim3 block(32, 8);
    dim3 grid(N / BN, M / BM);

    transpose_v2<BM, BN><<<grid, block, 0, stream>>>(
        input,
        output,
        M,
        N
    );
}