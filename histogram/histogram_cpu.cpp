#include"histogram_cpu.h"

void histogram_cpu(
    const unsigned char* input,
    unsigned int* output,
    std::size_t n
) {
    for(int i = 0; i < 256; i++){
        output[i] = 0;
    }
    for(std::size_t i = 0; i < n; i++){
        output[input[i]]++;
    }
}