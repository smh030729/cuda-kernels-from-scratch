#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "benchmark.cuh"
#include "cuda_check.cuh"
#include "random.h"

#include "reduction.cuh"
#include "reduction_cpu.h"


bool check_close(
    float gpu_result,
    double reference,
    double atol = 1e-3,
    double rtol = 1e-4
) {
    double diff =
        std::abs(
            static_cast<double>(gpu_result)
            - reference
        );

    double tolerance =
        atol
        + rtol * std::abs(reference);

    if (diff > tolerance) {
        std::cout
            << "FAILED"
            << " | GPU = " << gpu_result
            << " | CPU = " << reference
            << " | diff = " << diff
            << '\n';

        return false;
    }

    return true;
}


bool test_correctness(
    std::size_t n
) {
    // ---------------------------------
    // Host input
    // ---------------------------------

    std::vector<float> h_input(n);

    fill_random(
        h_input.data(),
        n,
        -1.0f,
        1.0f
    );


    // ---------------------------------
    // CPU reference
    // ---------------------------------

    double reference =
        reduction_cpu(
            h_input.data(),
            n
        );


    // ---------------------------------
    // Device memory
    // ---------------------------------

    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_workspace1 = nullptr;
    float* d_workspace2 = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            n * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_output,
            sizeof(float)
        )
    );


    std::size_t workspace_size =
        reduction_v1_workspace_size(n);

    if (workspace_size > 0) {
        CUDA_CHECK(
            cudaMalloc(
                &d_workspace1,
                workspace_size * sizeof(float)
            )
        );

        CUDA_CHECK(
            cudaMalloc(
                &d_workspace2,
                workspace_size * sizeof(float)
            )
        );
    }


    // ---------------------------------
    // H2D
    // ---------------------------------

    if (n > 0) {
        CUDA_CHECK(
            cudaMemcpy(
                d_input,
                h_input.data(),
                n * sizeof(float),
                cudaMemcpyHostToDevice
            )
        );
    }


    // ---------------------------------
    // Run GPU
    // ---------------------------------

    launch_reduction_v1(
        d_input,
        d_output,
        d_workspace1,
        d_workspace2,
        n
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // ---------------------------------
    // D2H
    // ---------------------------------

    float gpu_result = 0.0f;

    CUDA_CHECK(
        cudaMemcpy(
            &gpu_result,
            d_output,
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    // ---------------------------------
    // Check
    // ---------------------------------

    bool correct =
        check_close(
            gpu_result,
            reference
        );

    std::cout
        << "N = "
        << n
        << " : "
        << (correct ? "PASS" : "FAIL")
        << '\n';


    // ---------------------------------
    // Cleanup
    // ---------------------------------

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    if (d_workspace1 != nullptr) {
        CUDA_CHECK(cudaFree(d_workspace1));
    }

    if (d_workspace2 != nullptr) {
        CUDA_CHECK(cudaFree(d_workspace2));
    }

    return correct;
}


void run_benchmark(
    std::size_t n
) {
    // ---------------------------------
    // Host input
    // ---------------------------------

    std::vector<float> h_input(n);

    fill_random(
        h_input.data(),
        n,
        -1.0f,
        1.0f
    );


    // ---------------------------------
    // Device memory
    // ---------------------------------

    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_workspace1 = nullptr;
    float* d_workspace2 = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            n * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_output,
            sizeof(float)
        )
    );


    std::size_t workspace_size =
        reduction_v1_workspace_size(n);

    CUDA_CHECK(
        cudaMalloc(
            &d_workspace1,
            workspace_size * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_workspace2,
            workspace_size * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            n * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    // ---------------------------------
    // Benchmark
    // ---------------------------------

    float avg_ms =
        benchmark_cuda(
            [&]() {
                launch_reduction_v1(
                    d_input,
                    d_output,
                    d_workspace1,
                    d_workspace2,
                    n
                );
            },
            10,     // warmup
            100     // repeat
        );


    // ---------------------------------
    // Effective bandwidth
    // ---------------------------------

    double input_bytes =
        static_cast<double>(n)
        * sizeof(float);

    double bandwidth =
        input_bytes
        / (avg_ms * 1e-3)
        / 1e9;


    std::cout << '\n';

    std::cout
        << "===== Reduction v1 Benchmark =====\n";

    std::cout
        << "N          : "
        << n
        << '\n';

    std::cout
        << "Latency    : "
        << avg_ms
        << " ms\n";

    std::cout
        << "Bandwidth  : "
        << bandwidth
        << " GB/s\n";


    // ---------------------------------
    // Cleanup
    // ---------------------------------

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_workspace1));
    CUDA_CHECK(cudaFree(d_workspace2));
}


int main() {

    // =================================
    // Correctness
    // =================================

    std::cout
        << "===== Correctness Test =====\n";

    const std::vector<std::size_t> test_sizes = {
        1,
        31,
        32,
        33,
        255,
        256,
        257,
        511,
        512,
        513,
        1000,
        4097,
        1 << 20
    };

    bool all_passed = true;

    for (std::size_t n : test_sizes) {
        all_passed &=
            test_correctness(n);
    }

    if (!all_passed) {
        std::cerr
            << "\nCorrectness test failed.\n";

        return 1;
    }


    // =================================
    // Performance
    // =================================

    constexpr std::size_t N =
        1ULL << 24;

    run_benchmark(N);

    return 0;
}