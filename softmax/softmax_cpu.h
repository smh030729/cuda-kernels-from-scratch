#pragma once

#include<cmath>
#include<cstddef>

void softmax_cpu(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols
);
