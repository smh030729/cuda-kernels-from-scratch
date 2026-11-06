#pragma once 

#include<cmath>
#include<cstddef>

void rmsnorm_cpu(
    const float* input,
    float* output,
    const float* weight,
    float eps,
    std::size_t rows,
    std::size_t cols
);