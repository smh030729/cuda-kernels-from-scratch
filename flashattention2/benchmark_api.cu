#include "flashattention2.cuh"

// Minimal C ABI for the Python benchmark; all tensors are device FP16 buffers.
extern "C" int fa2_launch(
    int version, const void* Q, const void* K, const void* V,
    void* O, int N, void* stream
) {
    if (N <= 0 || N % 64 != 0) return cudaErrorInvalidValue;
    const half* q = static_cast<const half*>(Q);
    const half* k = static_cast<const half*>(K);
    const half* v = static_cast<const half*>(V);
    half* o = static_cast<half*>(O);
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    switch (version) {
        case 0: launch_flashattention2_v0_fp16(q, k, v, o, N, cuda_stream); break;
        case 1: launch_flashattention2_v1_fp16(q, k, v, o, N, cuda_stream); break;
        case 2: launch_flashattention2_v2a_fp16(q, k, v, o, N, cuda_stream); break;
        case 3: launch_flashattention2_v2b(q, k, v, o, N, cuda_stream); break;
        case 4: launch_flashattention2_v2c(q, k, v, o, N, cuda_stream); break;
        case 5: launch_flashattention2_v2c_bc32_qreg(q, k, v, o, N, cuda_stream); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
