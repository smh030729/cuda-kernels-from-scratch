#pragma once

void flashattention2_cpu(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N,
    int d
);