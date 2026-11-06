#include "conv2d.cuh"

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;

constexpr int TM = 4;
constexpr int TN = 4;

constexpr int NUM_THREADS = 256;

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void conv2d_v3(
    const float* input,
    const float* weight,
    float* output,
    int N,
    int Cin,
    int H,
    int W,
    int Cout,
    int KH,
    int KW,
    int Hout,
    int Wout,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    const int P = N * Hout * Wout;
    const int K_gemm = Cin * KH * KW;
    const int tid = threadIdx.x;

    const int block_m = blockIdx.y * BM;
    const int block_p = blockIdx.x * BN;

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;
    const int lane_row = lane_id / 8;
    const int lane_col = lane_id % 8;
    const int thread_row = 4 * warp_row + lane_row;
    const int thread_col = 8 * warp_col + lane_col; //消除写reg_w的bank conflict

    __shared__ float Ws[BK][BM + 4]; //消除写入Ws的bank conflict
    __shared__ float Xs[BK][BN];

    float accum[TM][TN] = {0.0f};

    float reg_w[TM];
    float reg_x[TN];

    for(int bk = 0; bk < K_gemm; bk += BK){
        for(int idx = tid; idx < BM * BK; idx += blockDim.x){
            int local_m = idx / BK;
            int local_k = idx % BK;
            int global_m = block_m + local_m;
            int global_k = bk + local_k;
            if (global_m < Cout && global_k < K_gemm){
                Ws[local_k][local_m] = weight[global_m * K_gemm + global_k];
            } else{
                Ws[local_k][local_m] = 0.0f;
            }
        }

        for(int idx = tid; idx < BK * BN; idx += blockDim.x){
            int local_k = idx / BN;
            int local_p = idx % BN;
            int global_k = bk + local_k;
            int global_p =  block_p + local_p;

            float x = 0.0f;
            if(global_k < K_gemm && global_p < P){
                int tmp_k = global_k;
                int kw = tmp_k % KW;
                tmp_k /= KW;
                int kh = tmp_k % KH;
                tmp_k /= KH;
                int ci = tmp_k;

                int tmp_p = global_p;
                int wo = tmp_p % Wout;
                tmp_p /= Wout;
                int ho = tmp_p % Hout;
                tmp_p /= Hout;
                int n = tmp_p;

                const int hi = ho * stride_h - pad_h + kh;
                const int wi = wo * stride_w - pad_w + kw;
                if(hi >= 0 && hi < H && wi >= 0 && wi < W){
                    const int input_idx = ((n * Cin + ci) * H + hi) * W + wi;
                    x = input[input_idx];
                }
            }
            Xs[local_k][local_p] = x;
        }
        __syncthreads();

        #pragma unroll
        for(int k = 0; k < BK; k++){
            #pragma unroll
            for(int i = 0; i < TM; i++){
                reg_w[i] = Ws[k][thread_row * TM + i];
            } 
        
            #pragma unroll
            for(int i = 0; i < TN; i++){
                reg_x[i] = Xs[k][thread_col * TN + i];
            } 
            
            #pragma unroll
            for (int i = 0; i < TM; i++){
                #pragma unroll
                for (int j = 0; j < TN; j++){
                    accum[i][j] += reg_w[i] * reg_x[j];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int j = 0; j < TN; j++){
        int global_p = block_p + thread_col * TN + j;
        if(global_p >= P) continue;
        int tmp_p = global_p;
        int wo = tmp_p % Wout;
        tmp_p /= Wout;
        int ho = tmp_p % Hout;
        tmp_p /= Hout;
        int n = tmp_p; 

        #pragma unroll
        for(int i = 0; i < TM; i++){
            int co = block_m + thread_row * TM + i;
            if(co >= Cout) continue;
            const int output_idx = ((n * Cout + co) * Hout + ho) * Wout + wo;
            output[output_idx] = accum[i][j];
        }
    }
}

void launch_conv2d_v3(
    const float* input,
    const float* weight,
    float* output,
    int N,
    int Cin,
    int H,
    int W,
    int Cout,
    int KH,
    int KW,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    cudaStream_t stream
) {
    const int Hout = (H + 2 * pad_h - KH) / stride_h + 1;
    const int Wout = (W + 2 * pad_w - KW) / stride_w + 1;
    if (Hout <= 0 || Wout <= 0) {
        return;
    }
    const int P = N * Hout * Wout;

    dim3 grid ((P + BN - 1) / BN, (Cout + BM - 1) / BM);

    conv2d_v3<BM, BN, BK, TM, TN><<<grid, NUM_THREADS, 0, stream>>>(
        input,
        weight,
        output,
        N,
        Cin,
        H,
        W,
        Cout,
        KH,
        KW,
        Hout,
        Wout,
        stride_h,
        stride_w,
        pad_h,
        pad_w
    );
}