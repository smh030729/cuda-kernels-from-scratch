#include"reduction_cpu.h"

float reduction_cpu(
    const float* input,
    std::size_t n
){
    float sum = 0.0f;
    for(int i = 0; i < n; i++){
        sum += input[i];
    }
    return sum;
}