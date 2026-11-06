#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void* ptr){
    uint32_t smem_addr;
    asm volatile(
        "{\n"
        "  .reg .u64 smem_ptr;\n"
        "  cvta.to.shared.u64 smem_ptr, %1;\n"
        "  cvt.u32.u64 %0, smem_ptr;\n"
        "}\n"
        : "=r"(smem_addr)
        : "l"(ptr)
    );
    return smem_addr;
}

__global__ void ldmatrix_x1(float* output){
    const int laneid = threadIdx.x;
    __shared__ __half tile[8][8];
    for(int idx = laneid; idx < 64; idx += blockDim.x){
        int row = idx / 8;
        int col = idx % 8;
        tile[row][col] = __float2half(static_cast<float>(idx));
    }
    __syncthreads();
    const int addr_row = laneid % 8;
    const uint32_t smem_addr = cvta_to_shared_u32(&tile[addr_row][0]);
    
    uint32_t frag;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 "
        "{%0}, [%1];\n"
        : "=r"(frag)
        : "r"(smem_addr)
    );

    __half h0, h1;
    unsigned short bits0 = static_cast<unsigned short>(frag & 0xffff);
    unsigned short bits1 = static_cast<unsigned short>(frag >> 16);

    h0 = __ushort_as_half(bits0);
    h1 = __ushort_as_half(bits1);

    output[2 * laneid + 0] = __half2float(h0);
    output[2 * laneid + 1] = __half2float(h1);

}

int main(){
    float* h_output = new float[64];
    float* d_output;
    cudaMalloc(&d_output, sizeof(float) * 64);
    ldmatrix_x1<<<1, 32>>>(d_output);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, sizeof(float) * 64, cudaMemcpyDeviceToHost);
    for(int lane = 0; lane < 32; lane++){
        std::printf("lane %2d : %5.1f  %5.1f\n", lane, h_output[2 * lane + 0], h_output[2 * lane + 1]);
    }
}