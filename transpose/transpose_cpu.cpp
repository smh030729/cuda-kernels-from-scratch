#include"transpose_cpu.h"

void transpose_cpu(
    const float* input, //[m,n]
    float* output, //[n,m]
    int m,
    int n
) {
    for(int r = 0; r < m; r++)
    {
        for(int c = 0; c < n; c++){
            output[c * m + r] = input[r * n + c];
        }
    }
}