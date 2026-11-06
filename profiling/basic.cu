#include "conv2d/conv2d.cuh"
#include "histogram/histogram.cuh"
#include "rmsnorm/rmsnorm_v1.cuh"
#include "softmax/softmax.cuh"
#include "transpose/transpose.cuh"
#include "cuda_check.cuh"
#include <cuda_profiler_api.h>
#include <string>
#include <vector>
#include <stdexcept>

// Fixed shapes and warm-up keep the benchmark entry points separate from NCU capture.
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string op = argv[1];
    constexpr int side = 2048, count = side * side;
    std::vector<float> host(count);
    for (int i = 0; i < count; ++i) host[i] = ((i % 127) - 63) / 64.f;
    float *x, *y, *weight;
    CUDA_CHECK(cudaMalloc(&x, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&weight, 64 * 32 * 9 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(x, host.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weight, host.data(), 64 * 32 * 9 * sizeof(float), cudaMemcpyHostToDevice));
    if (op == "hist_v1" || op == "hist_v2") {
        std::vector<unsigned char> bytes(count * 4);
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = i % 256;
        CUDA_CHECK(cudaMemcpy(x, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
    }
    auto launch = [&] {
        if (op == "softmax_v1") launch_softmax_v1(x,y,2048,1024);
        else if (op == "softmax_reg") launch_softmax_reg_vec(x,y,2048,1024);
        else if (op == "softmax_shared") launch_softmax_shared_vec(x,y,2048,1024);
        else if (op == "rms_vec") launch_rmsnorm_v1(x,y,weight,1e-6f,2048,1024);
        else if (op == "rms_scalar") launch_rmsnorm_v1(x,y,weight,1e-6f,2048,1023);
        else if (op == "trans_v1") launch_transpose_v1(x,y,side,side);
        else if (op == "trans_v2") launch_transpose_v2(x,y,side,side);
        else if (op == "conv_v2") launch_conv2d_v2(x,weight,y,1,32,64,64,64,3,3,1,1,1,1);
        else if (op == "conv_v3") launch_conv2d_v3(x,weight,y,1,32,64,64,64,3,3,1,1,1,1);
        else if (op == "hist_v1") launch_histogram_v1(reinterpret_cast<unsigned char*>(x),reinterpret_cast<unsigned*>(y),count*4);
        else if (op == "hist_v2") launch_histogram_v2(reinterpret_cast<unsigned char*>(x),reinterpret_cast<unsigned*>(y),count*4);
        else throw std::invalid_argument("unknown profile selector");
        CUDA_CHECK(cudaGetLastError());
    };
    CUDA_CHECK(cudaMemset(y,0,count*sizeof(float)));
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(y,0,count*sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaProfilerStart());
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaProfilerStop());
    CUDA_CHECK(cudaFree(weight)); CUDA_CHECK(cudaFree(y)); CUDA_CHECK(cudaFree(x));
}
