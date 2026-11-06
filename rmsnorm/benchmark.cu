#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "benchmark.cuh"
#include "cuda_check.cuh"
#include "random.h"

#include "rmsnorm_cpu.h"
#include "rmsnorm_v1.cuh"


constexpr float EPS = 1e-5f;

constexpr int WARMUP = 10;
constexpr int REPEAT = 100;

constexpr std::size_t CORRECTNESS_ROWS = 4;
constexpr std::size_t BENCH_ROWS = 4096;


// ============================================================
// correctness
// ============================================================

bool check_correctness(
    std::size_t rows,
    std::size_t cols
) {
    const std::size_t numel = rows * cols;
    const std::size_t data_bytes =
        numel * sizeof(float);

    const std::size_t weight_bytes =
        cols * sizeof(float);

    std::vector<float> h_input(numel);
    std::vector<float> h_weight(cols);

    std::vector<float> h_output_cpu(numel);
    std::vector<float> h_output_gpu(numel);

    fill_random(h_input.data(), numel);
    fill_random(h_weight.data(), cols);


    // CPU reference
    rmsnorm_cpu(
        h_input.data(),
        h_output_cpu.data(),
        h_weight.data(),
        EPS,
        rows,
        cols
    );


    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_weight = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            data_bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_output,
            data_bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_weight,
            weight_bytes
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            data_bytes,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_weight,
            h_weight.data(),
            weight_bytes,
            cudaMemcpyHostToDevice
        )
    );


    launch_rmsnorm_v1(
        d_input,
        d_output,
        d_weight,
        EPS,
        rows,
        cols,
        0
    );

    CUDA_CHECK(cudaDeviceSynchronize());


    CUDA_CHECK(
        cudaMemcpy(
            h_output_gpu.data(),
            d_output,
            data_bytes,
            cudaMemcpyDeviceToHost
        )
    );


    // error check
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;

    for (std::size_t i = 0; i < numel; ++i) {
        float cpu = h_output_cpu[i];
        float gpu = h_output_gpu[i];

        float abs_error =
            std::fabs(cpu - gpu);

        float rel_error =
            abs_error /
            std::max(
                std::fabs(cpu),
                1e-6f
            );

        max_abs_error =
            std::max(
                max_abs_error,
                abs_error
            );

        max_rel_error =
            std::max(
                max_rel_error,
                rel_error
            );
    }


    constexpr float ATOL = 1e-4f;
    constexpr float RTOL = 1e-4f;

    bool passed =
        max_abs_error < ATOL ||
        max_rel_error < RTOL;


    std::cout
        << "cols = "
        << std::setw(5)
        << cols
        << " | "
        << (cols % 4 == 0
            ? "vector"
            : "scalar")
        << " | "
        << (passed
            ? "PASS"
            : "FAIL")
        << " | max_abs = "
        << std::scientific
        << max_abs_error
        << " | max_rel = "
        << max_rel_error
        << std::defaultfloat
        << '\n';


    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_weight));

    return passed;
}


// ============================================================
// performance
// ============================================================

void benchmark(
    std::size_t rows,
    std::size_t cols
) {
    const std::size_t numel =
        rows * cols;

    const std::size_t data_bytes =
        numel * sizeof(float);

    const std::size_t weight_bytes =
        cols * sizeof(float);


    std::vector<float> h_input(numel);
    std::vector<float> h_weight(cols);

    fill_random(
        h_input.data(),
        numel
    );

    fill_random(
        h_weight.data(),
        cols
    );


    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_weight = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            data_bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_output,
            data_bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_weight,
            weight_bytes
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            data_bytes,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_weight,
            h_weight.data(),
            weight_bytes,
            cudaMemcpyHostToDevice
        )
    );


    auto func = [&]() {
        launch_rmsnorm_v1(
            d_input,
            d_output,
            d_weight,
            EPS,
            rows,
            cols,
            0
        );
    };


    float time_ms =
        benchmark_cuda(
            func,
            WARMUP,
            REPEAT,
            0
        );


    /*
        Logical memory traffic:

        pass 1:
            input read       = rows * cols

        pass 2:
            input read       = rows * cols
            weight read      = rows * cols
            output write     = rows * cols

        total:
            4 * rows * cols * sizeof(float)

        注意：
        weight 很可能被 cache，
        第二次 input 也可能命中 cache。

        所以这里只能叫 effective/logical bandwidth，
        不能认为是真实 DRAM bandwidth。
    */

    double logical_bytes =
        4.0 *
        static_cast<double>(numel) *
        sizeof(float);

    double bandwidth_gb_s =
        logical_bytes /
        (time_ms * 1e-3) /
        1e9;


    std::cout
        << std::setw(6)
        << cols
        << " | "
        << std::setw(7)
        << (cols % 4 == 0
            ? "VECTOR"
            : "SCALAR")
        << " | "
        << std::fixed
        << std::setprecision(4)
        << std::setw(9)
        << time_ms
        << " ms"
        << " | "
        << std::setprecision(2)
        << std::setw(9)
        << bandwidth_gb_s
        << " GB/s"
        << '\n';


    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_weight));
}


// ============================================================
// main
// ============================================================

int main() {

    // --------------------------------------------------------
    // correctness
    // --------------------------------------------------------

    std::cout
        << "================ RMSNorm Correctness ================\n";

    const std::vector<std::size_t> correctness_cols = {
        1,
        3,
        4,
        31,
        32,
        63,
        64,
        127,
        128,
        255,
        256,
        511,
        512,
        768,
        1024,
        1025,
        1028,
        1536,
        2048,
        4096,
        8192
    };


    bool all_passed = true;

    for (
        std::size_t cols
        : correctness_cols
    ) {
        bool passed =
            check_correctness(
                CORRECTNESS_ROWS,
                cols
            );

        all_passed &=
            passed;
    }


    std::cout
        << "\nOverall correctness: "
        << (all_passed
            ? "PASS"
            : "FAIL")
        << "\n\n";


    if (!all_passed) {
        return 1;
    }


    // --------------------------------------------------------
    // performance
    // --------------------------------------------------------

    std::cout
        << "================ RMSNorm Benchmark =================\n";

    std::cout
        << "Rows = "
        << BENCH_ROWS
        << '\n';

    std::cout
        << "  Cols |    Path |      Time | Effective BW\n";

    std::cout
        << "-----------------------------------------------------\n";


    const std::vector<std::size_t> benchmark_cols = {
        32,
        64,
        128,
        256,
        512,
        768,
        1024,
        1280,
        1536,
        2048,
        2560,
        3072,
        4096,
        6144,
        8192
    };


    for (
        std::size_t cols
        : benchmark_cols
    ) {
        benchmark(
            BENCH_ROWS,
            cols
        );
    }


    return 0;
}