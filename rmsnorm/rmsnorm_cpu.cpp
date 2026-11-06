#include"rmsnorm_cpu.h"

void rmsnorm_cpu(
    const float* input,
    float* output,
    const float* weight,
    float eps,
    std::size_t rows,
    std::size_t cols
) {
    for(std::size_t r = 0; r < rows; r++){
        float sum = 0.0f;
        for(std::size_t c = 0; c <cols; c++){
            float x = input[c];
            sum += x * x;
        }
        
        sum = 1.0f / sqrtf(sum / cols + eps);
        for(std::size_t c = 0; c < cols; c++){
            output[c] = input[c] * sum * weight[c];
        }
    input += cols;
    output += cols;
    }
}