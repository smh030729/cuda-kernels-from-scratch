#include "softmax.cuh"
#include "softmax_cpu.h"

#include "benchmark.cuh"
#include "cuda_check.cuh"
#include "random.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>


// ============================================================
// correctness check
// ============================================================

bool check_close(
    const std::vector<float>& gpu,
    const std::vector<float>& ref,
    float atol = 1e-5f,
    float rtol = 1e-4f
) {
    if (gpu.size() != ref.size()) {
        return false;
    }

    float max_abs_err = 0.0f;

    for (std::size_t i = 0; i < gpu.size(); ++i) {
        float abs_err =
            std::abs(gpu[i] - ref[i]);

        float tol =
            atol + rtol * std::abs(ref[i]);

        max_abs_err =
            std::max(max_abs_err, abs_err);

        if (abs_err > tol) {
            std::cerr
                << "Mismatch at index " << i
                << ", gpu = " << gpu[i]
                << ", ref = " << ref[i]
                << ", abs_err = " << abs_err
                << ", tol = " << tol
                << '\n';

            return false;
        }
    }

    return true;
}


// ============================================================
// 根据 cols 判断 launch_softmax 会走哪个分支
// 这里只用于打印 benchmark 信息
// ============================================================

std::string get_dispatch_path(std::size_t cols) {
    constexpr std::size_t THREADS = 256;

    if (cols % 4 != 0) {
        return "V1";
    }

    if (cols <= 4 * THREADS) {
        return "REG_VEC";
    }

    if (cols <= 8 * THREADS) {
        return "SHARED_VEC";
    }

    return "V1";
}


// ============================================================
// correctness
// ============================================================

bool test_correctness(
    std::size_t rows,
    std::size_t cols
) {
    std::size_t n =
        rows * cols;

    std::vector<float> h_input(n);
    std::vector<float> h_ref(n);
    std::vector<float> h_v1(n);
    std::vector<float> h_opt(n);

    fill_random(
        h_input.data(),
        n,
        -5.0f,
        5.0f,
        42
    );


    // CPU reference
    softmax_cpu(
        h_input.data(),
        h_ref.data(),
        rows,
        cols
    );


    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            n * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_output,
            n * sizeof(float)
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


    // --------------------------------------------------------
    // V1 correctness
    // --------------------------------------------------------

    launch_softmax_v1(
        d_input,
        d_output,
        rows,
        cols
    );

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    CUDA_CHECK(
        cudaMemcpy(
            h_v1.data(),
            d_output,
            n * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    bool v1_pass =
        check_close(
            h_v1,
            h_ref
        );


    // --------------------------------------------------------
    // dispatched optimized version correctness
    // --------------------------------------------------------

    launch_softmax(
        d_input,
        d_output,
        rows,
        cols
    );

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    CUDA_CHECK(
        cudaMemcpy(
            h_opt.data(),
            d_output,
            n * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    bool opt_pass =
        check_close(
            h_opt,
            h_ref
        );


    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));


    std::cout
        << "rows = " << std::setw(4) << rows
        << ", cols = " << std::setw(5) << cols
        << ", path = " << std::setw(10)
        << get_dispatch_path(cols)
        << ", V1 = "
        << (v1_pass ? "PASS" : "FAIL")
        << ", OPT = "
        << (opt_pass ? "PASS" : "FAIL")
        << '\n';


    return v1_pass && opt_pass;
}


// ============================================================
// performance benchmark
// ============================================================

void run_benchmark(
    std::size_t rows,
    std::size_t cols
) {
    std::size_t n =
        rows * cols;

    std::vector<float> h_input(n);

    fill_random(
        h_input.data(),
        n,
        -5.0f,
        5.0f,
        42
    );


    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            n * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_output,
            n * sizeof(float)
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


    constexpr int WARMUP = 20;
    constexpr int REPEAT = 100;


    // ========================================================
    // V1
    // ========================================================

    float v1_ms =
        benchmark_cuda(
            [&]() {
                launch_softmax_v1(
                    d_input,
                    d_output,
                    rows,
                    cols
                );
            },
            WARMUP,
            REPEAT
        );


    // ========================================================
    // REG_VEC
    //
    // 当前实现要求：
    // cols % 4 == 0
    // cols <= 4 * 256 = 1024
    // ========================================================

    bool reg_valid =
        cols % 4 == 0 &&
        cols <= 4 * 256;

    float reg_ms = 0.0f;

    if (reg_valid) {
        reg_ms =
            benchmark_cuda(
                [&]() {
                    launch_softmax_reg_vec(
                        d_input,
                        d_output,
                        rows,
                        cols
                    );
                },
                WARMUP,
                REPEAT
            );
    }


    // ========================================================
    // SHARED_VEC
    //
    // 当前实现要求 cols % 4 == 0。
    // 这里 benchmark 暂时限制到 8192，
    // dynamic shared memory 最大为 32 KB。
    // ========================================================

    bool shared_valid =
        cols % 4 == 0 &&
        cols <= 8192;

    float shared_ms = 0.0f;

    if (shared_valid) {
        shared_ms =
            benchmark_cuda(
                [&]() {
                    launch_softmax_shared_vec(
                        d_input,
                        d_output,
                        rows,
                        cols
                    );
                },
                WARMUP,
                REPEAT
            );
    }


    // ========================================================
    // dispatcher
    // ========================================================

    float dispatch_ms =
        benchmark_cuda(
            [&]() {
                launch_softmax(
                    d_input,
                    d_output,
                    rows,
                    cols
                );
            },
            WARMUP,
            REPEAT
        );


    // ========================================================
    // Speedup relative to V1
    // ========================================================

    float reg_speedup =
        reg_valid
        ? v1_ms / reg_ms
        : 0.0f;

    float shared_speedup =
        shared_valid
        ? v1_ms / shared_ms
        : 0.0f;


    // ========================================================
    // 找三种 kernel 中最快的
    // ========================================================

    std::string best = "V1";
    float best_ms = v1_ms;

    if (reg_valid && reg_ms < best_ms) {
        best_ms = reg_ms;
        best = "REG_VEC";
    }

    if (shared_valid && shared_ms < best_ms) {
        best_ms = shared_ms;
        best = "SHARED_VEC";
    }


    // ========================================================
    // output
    // ========================================================

    std::cout
        << std::setw(6) << cols
        << std::setw(12)
        << std::fixed
        << std::setprecision(4)
        << v1_ms;


    if (reg_valid) {
        std::cout
            << std::setw(12) << reg_ms
            << std::setw(10)
            << std::setprecision(2)
            << reg_speedup << "x";
    }
    else {
        std::cout
            << std::setw(12) << "-"
            << std::setw(11) << "-";
    }


    if (shared_valid) {
        std::cout
            << std::setw(14)
            << std::setprecision(4)
            << shared_ms
            << std::setw(10)
            << std::setprecision(2)
            << shared_speedup << "x";
    }
    else {
        std::cout
            << std::setw(14) << "-"
            << std::setw(11) << "-";
    }


    std::cout
        << std::setw(14) << best
        << std::setw(14)
        << std::setprecision(4)
        << dispatch_ms
        << '\n';


    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}


// ============================================================
// main
// ============================================================

int main() {

    // ========================================================
    // Correctness
    //
    // 特意覆盖：
    // - 非 4 倍数 -> V1
    // - register kernel
    // - 1024 边界
    // - shared kernel
    // - 2048 边界
    // - 大 cols -> V1
    // ========================================================

    std::cout
        << "===== Softmax Correctness Test =====\n";

    std::vector<std::size_t> test_cols = {
        1,
        3,
        4,
        31,
        32,
        255,
        256,
        511,
        512,
        768,
        1024,
        1028,
        1536,
        2048,
        2052,
        4096
    };

    bool all_passed = true;

    for (std::size_t cols : test_cols) {
        all_passed &=
            test_correctness(
                4,
                cols
            );
    }

    if (!all_passed) {
        std::cerr
            << "\nCorrectness test failed.\n";

        return 1;
    }


    // ========================================================
    // Performance
    // ========================================================

    std::cout
    << "\n===== Softmax Kernel Comparison =====\n";

std::cout
    << std::setw(6)  << "Cols"
    << std::setw(12) << "V1(ms)"
    << std::setw(12) << "REG(ms)"
    << std::setw(11) << "REG/SP"
    << std::setw(14) << "SHARED(ms)"
    << std::setw(11) << "SHR/SP"
    << std::setw(14) << "Best"
    << std::setw(14) << "Dispatch(ms)"
    << '\n';


constexpr std::size_t BENCH_ROWS =
    16384;


std::vector<std::size_t> bench_cols = {
    32,
    64,
    128,
    256,
    512,
    768,
    1024,

    1280,
    1536,
    1792,
    2048,

    2560,
    3072,
    4096,
    6144,
    8192
};


for (std::size_t cols : bench_cols) {
    run_benchmark(
        BENCH_ROWS,
        cols
    );
}
    return 0;
}