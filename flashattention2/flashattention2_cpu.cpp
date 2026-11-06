#include"flashattention2_cpu.h"
#include<cmath>

void flashattention2_cpu(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N,
    int d
) {
    float* S = new float[N * N];
    float* P = new float[N * N];
    float scale = 1.0 / std::sqrt(d);
    for(int i = 0; i < N; i++){
        float max = -INFINITY;
        float row_sum = 0.0f;
        for(int j = 0; j < N; j++){
            float sum = 0.0f;
            for(int k = 0; k < d; k++){
                sum += Q[i * d + k] * K[j * d + k];
            }
            float res = sum * scale;
            S[i * N + j] = res;
            if(res > max) max = res;
        }
        for(int j = 0; j < N; j++){
            P[i * N + j] = std::exp(S[i * N + j] - max);
            row_sum += P[i * N + j];
        }
        for(int j = 0; j < N; j++){
            P[i * N + j] /= row_sum;
        }

    }
    for(int i = 0; i < N; i++){
        for(int j = 0; j < d; j++){
            float sum = 0.0f;
            for(int k = 0; k < N; k++){
                sum += P[i * N + k] * V[k * d + j];
            }
            O[i * d + j] = sum;
        }
    }
    delete[] S;
    delete[] P;
}