#include <cstdio>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>

#include "transpose.cuh"
#include "transpose_cpu.h"

#include "benchmark.cuh"
#include "cuda_check.cuh"
#include "random.h"

constexpr int WARMUP = 5;
constexpr int REPEAT = 20;
constexpr float TOL = 1e-5f;

bool check_result(
    const float* ref,
    const float* output,
    size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        if (std::fabs(ref[i] - output[i]) > TOL)
        {
            printf(
                "Mismatch at %zu: ref = %f, output = %f\n",
                i,
                ref[i],
                output[i]
            );
            return false;
        }
    }

    return true;
}

void benchmark_transpose(int M, int N)
{
    size_t numel = static_cast<size_t>(M) * N;
    size_t bytes = numel * sizeof(float);

    // -------------------------
    // Host memory
    // -------------------------
    std::vector<float> h_input(numel);
    std::vector<float> h_ref(numel);
    std::vector<float> h_v1(numel);
    std::vector<float> h_v2(numel);

    fill_random(h_input.data(), numel);

    // CPU reference
    transpose_cpu(
        h_input.data(),
        h_ref.data(),
        M,
        N
    );

    // -------------------------
    // Device memory
    // -------------------------
    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ============================================================
    // V1 correctness
    // ============================================================
    launch_transpose_v1(
        d_input,
        d_output,
        M,
        N,
        stream
    );

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(
        h_v1.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    bool matched_v1 = check_result(
        h_ref.data(),
        h_v1.data(),
        numel
    );

    // ============================================================
    // V2 correctness
    // ============================================================
    launch_transpose_v2(
        d_input,
        d_output,
        M,
        N,
        stream
    );

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(
        h_v2.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    bool matched_v2 = check_result(
        h_ref.data(),
        h_v2.data(),
        numel
    );

    // ============================================================
    // Benchmark V1
    // ============================================================
    float time_v1 = benchmark_cuda(
        [&]()
        {
            launch_transpose_v1(
                d_input,
                d_output,
                M,
                N,
                stream
            );
        },
        WARMUP,
        REPEAT,
        stream
    );

    // ============================================================
    // Benchmark V2
    // ============================================================
    float time_v2 = benchmark_cuda(
        [&]()
        {
            launch_transpose_v2(
                d_input,
                d_output,
                M,
                N,
                stream
            );
        },
        WARMUP,
        REPEAT,
        stream
    );

    // benchmark_cuda 假设返回 ms
    double bw_v1 =
        static_cast<double>(2 * bytes) /
        (time_v1 * 1e-3) /
        1e9;

    double bw_v2 =
        static_cast<double>(2 * bytes) /
        (time_v2 * 1e-3) /
        1e9;

    double speedup = time_v1 / time_v2;

    printf(
        "%6d x %-6d | "
        "V1: %8.4f ms  %8.2f GB/s  %s | "
        "V2: %8.4f ms  %8.2f GB/s  %s | "
        "Speedup: %.2fx\n",
        M,
        N,
        time_v1,
        bw_v1,
        matched_v1 ? "PASS" : "FAIL",
        time_v2,
        bw_v2,
        matched_v2 ? "PASS" : "FAIL",
        speedup
    );

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

int main()
{
    printf(
        "Matrix        | "
        "V1 Time      V1 BW          | "
        "V2 Time      V2 BW          | "
        "Speedup\n"
    );

    printf(
        "------------------------------------------------------------------------------------------\n"
    );

    benchmark_transpose(256, 256);
    benchmark_transpose(512, 512);
    benchmark_transpose(1024, 1024);
    benchmark_transpose(2048, 2048);
    benchmark_transpose(4096, 4096);
    benchmark_transpose(8192, 8192);

    return 0;
}