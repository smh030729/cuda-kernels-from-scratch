#include"softmax_cpu.h"

void softmax_cpu(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols
) {
    for(std::size_t r = 0; r < rows; r++){
        float max = -INFINITY;
        double sum = 0.0f;
        for(std::size_t c = 0; c < cols ; c++)
        {   
            if(input[c] > max) max = input[c];
        }

        for(std::size_t c = 0; c < cols ; c++){
            output[c] = std::exp(input[c] - max);
            sum += output[c];
        }

        for(std::size_t c = 0; c < cols ; c++){
            output[c] /= static_cast<float>(sum);
        }
        input += cols;
        output += cols;
    }
}