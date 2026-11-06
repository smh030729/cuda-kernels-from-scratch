#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "benchmark.cuh"
#include "cuda_check.cuh"

#include "histogram_cpu.h"
#include "histogram.cuh"


constexpr int NUM_BINS = 256;

constexpr int WARMUP = 5;
constexpr int REPEAT = 20;


// ============================================================
// input generation
// ============================================================

enum class InputPattern {
    Uniform,
    Hotspot
};


void fill_input(
    unsigned char* data,
    std::size_t n,
    InputPattern pattern
) {
    if (pattern == InputPattern::Uniform) {
        std::mt19937 gen(1234);

        std::uniform_int_distribution<int> dist(
            0,
            255
        );

        for (std::size_t i = 0; i < n; ++i) {
            data[i] =
                static_cast<unsigned char>(
                    dist(gen)
                );
        }
    }
    else {
        std::fill(
            data,
            data + n,
            static_cast<unsigned char>(0)
        );
    }
}


const char* pattern_name(
    InputPattern pattern
) {
    switch (pattern) {
        case InputPattern::Uniform:
            return "Uniform";

        case InputPattern::Hotspot:
            return "Hotspot";

        default:
            return "Unknown";
    }
}


// ============================================================
// correctness
// ============================================================

bool check_result(
    const std::vector<unsigned int>& cpu,
    const std::vector<unsigned int>& gpu
) {
    for (int i = 0; i < NUM_BINS; ++i) {
        if (cpu[i] != gpu[i]) {
            std::cout
                << "bin "
                << i
                << " mismatch: CPU = "
                << cpu[i]
                << ", GPU = "
                << gpu[i]
                << '\n';

            return false;
        }
    }

    return true;
}


bool check_correctness(
    std::size_t n,
    InputPattern pattern
) {
    std::vector<unsigned char> h_input(n);

    std::vector<unsigned int> h_output_cpu(
        NUM_BINS
    );

    std::vector<unsigned int> h_output_v1(
        NUM_BINS
    );

    std::vector<unsigned int> h_output_v2(
        NUM_BINS
    );


    fill_input(
        h_input.data(),
        n,
        pattern
    );


    // --------------------------------------------------------
    // CPU
    // --------------------------------------------------------

    histogram_cpu(
        h_input.data(),
        h_output_cpu.data(),
        n
    );


    // --------------------------------------------------------
    // GPU allocation
    // --------------------------------------------------------

    unsigned char* d_input = nullptr;
    unsigned int* d_output = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            n * sizeof(unsigned char)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_output,
            NUM_BINS * sizeof(unsigned int)
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            n * sizeof(unsigned char),
            cudaMemcpyHostToDevice
        )
    );


    // --------------------------------------------------------
    // V1
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemset(
            d_output,
            0,
            NUM_BINS * sizeof(unsigned int)
        )
    );

    launch_histogram_v1(
        d_input,
        d_output,
        n,
        0
    );

    CUDA_CHECK(cudaDeviceSynchronize());


    CUDA_CHECK(
        cudaMemcpy(
            h_output_v1.data(),
            d_output,
            NUM_BINS * sizeof(unsigned int),
            cudaMemcpyDeviceToHost
        )
    );


    bool v1_passed =
        check_result(
            h_output_cpu,
            h_output_v1
        );


    // --------------------------------------------------------
    // V2
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemset(
            d_output,
            0,
            NUM_BINS * sizeof(unsigned int)
        )
    );

    launch_histogram_v2(
        d_input,
        d_output,
        n,
        0
    );

    CUDA_CHECK(cudaDeviceSynchronize());


    CUDA_CHECK(
        cudaMemcpy(
            h_output_v2.data(),
            d_output,
            NUM_BINS * sizeof(unsigned int),
            cudaMemcpyDeviceToHost
        )
    );


    bool v2_passed =
        check_result(
            h_output_cpu,
            h_output_v2
        );


    std::cout
        << "N = "
        << std::setw(9)
        << n
        << " | "
        << std::setw(8)
        << pattern_name(pattern)
        << " | V1: "
        << (v1_passed ? "PASS" : "FAIL")
        << " | V2: "
        << (v2_passed ? "PASS" : "FAIL")
        << '\n';


    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));


    return v1_passed && v2_passed;
}


// ============================================================
// performance
// ============================================================

void benchmark(
    std::size_t n,
    InputPattern pattern
) {
    std::vector<unsigned char> h_input(n);

    fill_input(
        h_input.data(),
        n,
        pattern
    );


    unsigned char* d_input = nullptr;
    unsigned int* d_output = nullptr;


    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            n * sizeof(unsigned char)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_output,
            NUM_BINS * sizeof(unsigned int)
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            n * sizeof(unsigned char),
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // V1
    // ========================================================

    CUDA_CHECK(
        cudaMemset(
            d_output,
            0,
            NUM_BINS * sizeof(unsigned int)
        )
    );


    auto func_v1 = [&]() {
        launch_histogram_v1(
            d_input,
            d_output,
            n,
            0
        );
    };


    float time_v1 =
        benchmark_cuda(
            func_v1,
            WARMUP,
            REPEAT,
            0
        );


    // ========================================================
    // V2
    // ========================================================

    CUDA_CHECK(
        cudaMemset(
            d_output,
            0,
            NUM_BINS * sizeof(unsigned int)
        )
    );


    auto func_v2 = [&]() {
        launch_histogram_v2(
            d_input,
            d_output,
            n,
            0
        );
    };


    float time_v2 =
        benchmark_cuda(
            func_v2,
            WARMUP,
            REPEAT,
            0
        );


    // ========================================================
    // throughput
    // ========================================================

    double throughput_v1 =
        static_cast<double>(n) /
        (time_v1 * 1e-3) /
        1e9;

    double throughput_v2 =
        static_cast<double>(n) /
        (time_v2 * 1e-3) /
        1e9;

    double speedup =
        static_cast<double>(time_v1) /
        static_cast<double>(time_v2);


    std::cout
        << std::setw(10)
        << n
        << " | "
        << std::setw(8)
        << pattern_name(pattern)
        << " | "
        << std::fixed
        << std::setprecision(4)
        << std::setw(9)
        << time_v1
        << " | "
        << std::setw(9)
        << time_v2
        << " | "
        << std::setprecision(2)
        << std::setw(7)
        << speedup
        << "x"
        << " | "
        << std::setw(8)
        << throughput_v1
        << " | "
        << std::setw(8)
        << throughput_v2
        << '\n';


    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}


// ============================================================
// main
// ============================================================

int main() {

    // ========================================================
    // correctness
    // ========================================================

    std::cout
        << "================ Histogram Correctness ================\n";


    const std::vector<std::size_t> correctness_sizes = {
        1,
        31,
        32,
        255,
        256,
        257,
        1024,
        4096,
        1 << 20
    };


    bool all_passed = true;


    for (std::size_t n : correctness_sizes) {

        all_passed &=
            check_correctness(
                n,
                InputPattern::Uniform
            );

        all_passed &=
            check_correctness(
                n,
                InputPattern::Hotspot
            );
    }


    std::cout
        << "\nOverall correctness: "
        << (all_passed ? "PASS" : "FAIL")
        << "\n\n";


    if (!all_passed) {
        return 1;
    }


    // ========================================================
    // benchmark
    // ========================================================

    std::cout
        << "================ Histogram Benchmark ==================\n";

    std::cout
        << "         N |  Pattern |    V1(ms) |    V2(ms)"
        << " | Speedup | V1 GE/s | V2 GE/s\n";

    std::cout
        << "------------------------------------------------------------"
        << "----------------\n";


    const std::vector<std::size_t> benchmark_sizes = {
        1 << 16,
        1 << 18,
        1 << 20,
        1 << 22,
        1 << 24,
        1 << 26
    };


    for (std::size_t n : benchmark_sizes) {

        benchmark(
            n,
            InputPattern::Uniform
        );

        benchmark(
            n,
            InputPattern::Hotspot
        );
    }


    return 0;
}