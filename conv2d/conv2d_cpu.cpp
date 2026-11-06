#include "conv2d_cpu.h"

void conv2d_cpu(
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
    int pad_w
) {
    const int Hout = (H + 2 * pad_h - KH) / stride_h + 1;
    const int Wout = (W + 2 * pad_w - KW) / stride_w + 1;

    for (int n = 0; n < N; ++n) {
        for (int co = 0; co < Cout; ++co) {
            for (int ho = 0; ho < Hout; ++ho) {
                for (int wo = 0; wo < Wout; ++wo) {

                    float sum = 0.0f;

                    for (int ci = 0; ci < Cin; ++ci) {
                        for (int kh = 0; kh < KH; ++kh) {
                            for (int kw = 0; kw < KW; ++kw) {

                                const int hi = ho * stride_h - pad_h + kh;
                                const int wi = wo * stride_w - pad_w + kw;

                                if (hi < 0 || hi >= H || wi < 0 || wi >= W){
                                    continue;
                                }

                                const int input_idx = ((n * Cin + ci) * H + hi) * W + wi;
                                const int weight_idx = ((co * Cin + ci) * KH + kh) * KW + kw;

                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                    const int output_idx = ((n * Cout + co) * Hout + ho) * Wout + wo;
                    output[output_idx] = sum;
                }
            }
        }
    }
}