#include<sgemm.cuh>
#include<cuda_runtime.h>

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;

constexpr int TM = 8;
constexpr int TN = 8;

constexpr int NUM_THREADS = 256;

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_v1(
    const float*  A,
    const float*  B,
    float*  C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
) {
    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    int tid = threadIdx.x;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int thread_row = tid / (BN / TN);
    int thread_col = tid % (BN / TN);

    float thread_results[TM][TN] = {0.0f};

    float reg_a[TM];
    float reg_b[TN];

    A += block_row * K;
    B += block_col;
    C += block_row * N + block_col;

    for(int idx = tid * 4; idx < BM * BK; idx += 4 * NUM_THREADS){
        int row = idx / BK;
        int col = idx % BK;
        float4 tmp = *reinterpret_cast<const float4*>(&(A[row * K + col]));

        As[0][col + 0][row] = tmp.x;
        As[0][col + 1][row] = tmp.y;
        As[0][col + 2][row] = tmp.z;
        As[0][col + 3][row] = tmp.w;
    }

    for(int idx = tid * 4; idx < BK * BN; idx += 4 * NUM_THREADS){
        int row = idx / BN;
        int col = idx % BN;
        float4 tmp = *reinterpret_cast<const float4*>(&(B[row * N + col]));
        *reinterpret_cast<float4*>(&(Bs[0][row][col])) = tmp;
    }
    __syncthreads();

    int read_buf = 0;
    int write_buf = 1;

    constexpr int ITERATION_A = (BM * BK + 4 * NUM_THREADS - 1) / (4 * NUM_THREADS);
    constexpr int ITERATION_B = (BK * BN + 4 * NUM_THREADS - 1) / (4 * NUM_THREADS);
    float4 input_a[ITERATION_A];
    float4 input_b[ITERATION_B];

    for(int bk = 0; bk < K; bk += BK){
        int next_bk = bk + BK;
        if(next_bk < K){
            #pragma unroll
            for(int it = 0; it < ITERATION_A; it++){
                int idx = (tid + it * NUM_THREADS) * 4;
                if(idx < BM * BK){
                    int row = idx / BK;
                    int col = idx % BK;
                    float4 tmp = *reinterpret_cast<const float4*>(&(A[next_bk + row * K + col]));
                    input_a[it] = tmp;
                }
            }
            
            #pragma unroll
            for(int it = 0; it < ITERATION_B; it++){
                int idx = (tid + it * NUM_THREADS) * 4;
                if(idx < BK * BN){
                    int row = idx / BN;
                    int col = idx % BN;
                    float4 tmp = *reinterpret_cast<const float4*>(&(B[(next_bk + row) * N + col]));
                    input_b[it] = tmp;
                }
            }

        }

        #pragma unroll 
        for(int k = 0; k < BK; k++){
            #pragma unroll
            for(int i = 0; i < TM; i++){
                reg_a[i] = As[read_buf][k][thread_row * TM + i];
            }

            #pragma unroll
            for(int j = 0; j < TN; j++){
                reg_b[j] = Bs[read_buf][k][thread_col * TN + j];
            }

            #pragma unroll
            for(int i = 0; i < TM; i++){
                #pragma unroll
                for(int j = 0; j < TN; j++){
                    thread_results[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        if(next_bk < K){
            #pragma unroll
            for(int it = 0; it < ITERATION_A; it++){
                int idx = (tid + it * NUM_THREADS) * 4;
                if(idx < BM * BK){
                    int row = idx / BK;
                    int col = idx % BK;
                    float4 tmp = input_a[it];
                    As[write_buf][col + 0][row] = tmp.x;
                    As[write_buf][col + 1][row] = tmp.y;
                    As[write_buf][col + 2][row] = tmp.z;
                    As[write_buf][col + 3][row] = tmp.w;
                }
            }

            #pragma unroll
            for(int it = 0; it < ITERATION_B; it++){
                int idx = (tid + it * NUM_THREADS) * 4;
                if(idx < BK * BN){
                    int row = idx / BN;
                    int col = idx % BN;
                    *reinterpret_cast<float4*>(&Bs[write_buf][row][col]) = input_b[it];
                }
            }
            __syncthreads();

            read_buf ^= 1;
            write_buf ^= 1;
        }
    }

    #pragma unroll
    for(int i = 0; i < TM; i++){
        int row = thread_row * TM + i;
        #pragma unroll
        for(int j = 0; j < TN; j += 4){
            int col = thread_col * TN + j;
            
            float4 out;

            out.x = thread_results[i][j + 0];
            out.y = thread_results[i][j + 1];
            out.z = thread_results[i][j + 2];
            out.w = thread_results[i][j + 3];

            if(beta != 0.0f){
                float4 old =*reinterpret_cast<const float4*>(&C[row * N + col]);

                out.x = alpha * out.x + beta * old.x;
                out.y = alpha * out.y + beta * old.y;
                out.z = alpha * out.z + beta * old.z;
                out.w = alpha * out.w + beta * old.w;
            }
            else{
                out.x *= alpha;
                out.y *= alpha;
                out.z *= alpha;
                out.w *= alpha;
            }

            *reinterpret_cast<float4*>(&C[row * N + col]) = out;
        }
    }
}

void launch_sgemm_v1(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    cudaStream_t stream
)
{
    dim3 block(NUM_THREADS);

    dim3 grid(N / BN,M / BM);

    sgemm_v1<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(
        A,
        B,
        C,
        M,
        N,
        K,
        alpha,
        beta
    );
}
    