#include"sgemm_cpu.h"

void sgemm_cpu(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
) {
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++){
            float sum = 0.0f;

            for(int k = 0; k < K; k++){
                sum += A[i * K + k] * B[k * N + j];
            }
            
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}