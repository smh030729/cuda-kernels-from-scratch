#pragma once

#include<cstddef>

void histogram_cpu(
    const unsigned char* input,
    unsigned int* output,
    std::size_t n
);