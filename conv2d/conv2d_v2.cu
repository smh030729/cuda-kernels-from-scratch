#include"conv2d.cuh"

__global__ void conv2d_v2(
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
    const int total = Cout * P;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= total) return;

    const int m = idx / P;
    const int p = idx % P;
    const int co = m;
    
    int tmp_p = p;
    const int wo = tmp_p % Wout;
    tmp_p /= Wout;
    const int ho = tmp_p % Hout;
    tmp_p /= Hout;
    const int n = tmp_p;

    const int K_gemm = Cin * KH * KW;
    float sum = 0.0f;

    for(int k = 0; k < K_gemm; k++){
        int tmp_k = k;
        const int kw = tmp_k % KW;
        tmp_k /= KW;
        const int kh = tmp_k % KH;
        tmp_k /= KH;
        const int ci = tmp_k;

        const int hi = ho * stride_h - pad_h + kh;
        const int wi = wo * stride_w - pad_w + kw;

        float x = 0.0f;
        if(hi >= 0 && hi < H && wi >= 0 && wi < W){
            const int input_idx = ((n * Cin + ci) * H + hi) * W + wi;
            x = input[input_idx];
        }

        float w = 0.0f;
        const int weight_idx = ((co * Cin + ci) * KH + kh) * KW + kw;
        w = weight[weight_idx];
        sum += w * x;
    }

    const int output_idx = ((n * Cout + co) * Hout + ho) * Wout + wo;
    output[output_idx] = sum;
}

void launch_conv2d_v2(
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
    const int total = Cout * P;
    constexpr int BLOCK_SIZE = 256;
    const int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    conv2d_v2<<<grid, BLOCK_SIZE, 0, stream>>>(
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