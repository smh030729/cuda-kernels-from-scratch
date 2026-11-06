#pragma once

#include<random>
#include<cstddef>

inline void fill_random(
    float* data,
    std::size_t n,
    float high = 1.0f,
    float low = -1.0f,
    unsigned int seed = 42 
) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(low, high);

    for (std::size_t i = 0; i < n; ++i) {
        data[i] = dist(gen);
    }
}