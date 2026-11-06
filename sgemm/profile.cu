#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "sgemm.cuh"


#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)             \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                      \
} while (0)


int main(int argc, char** argv) {
    if (argc != 2 && argc != 5) {
        std::cerr
            << "Usage:\n"
            << "  " << argv[0] << " v1\n"
            << "  " << argv[0] << " v2\n"
            << "  " << argv[0] << " v1 M N K\n"
            << "  " << argv[0] << " v2 M N K\n";

        return EXIT_FAILURE;
    }

    // ------------------------------------------------------------
    // Parse arguments
    // ------------------------------------------------------------
    const std::string version = argv[1];

    if (version != "v1" && version != "v2") {
        std::cerr << "Version must be v1 or v2.\n";
        return EXIT_FAILURE;
    }

    int M = 4096;
    int N = 4096;
    int K = 4096;

    if (argc == 5) {
        M = std::stoi(argv[2]);
        N = std::stoi(argv[3]);
        K = std::stoi(argv[4]);
    }

    // 两个 kernel 当前都不处理边界
    if (M <= 0 || N <= 0 || K <= 0) {
        std::cerr << "M, N, K must be positive.\n";
        return EXIT_FAILURE;
    }

    if (M % 128 != 0) {
        std::cerr << "M must be a multiple of 128.\n";
        return EXIT_FAILURE;
    }

    if (N % 128 != 0) {
        std::cerr << "N must be a multiple of 128.\n";
        return EXIT_FAILURE;
    }

    if (K % 8 != 0) {
        std::cerr << "K must be a multiple of 8.\n";
        return EXIT_FAILURE;
    }

    constexpr float alpha = 1.0f;
    constexpr float beta  = 0.0f;

    // ------------------------------------------------------------
    // Print configuration
    // ------------------------------------------------------------
    const int grid_x = N / 128;
    const int grid_y = M / 128;

    const int block_threads =
        (version == "v1")
        ? 256
        : 128;

    std::cout << "=====================================\n";
    std::cout << "SGEMM profiling\n";
    std::cout << "Version : " << version << '\n';
    std::cout << "M       : " << M << '\n';
    std::cout << "N       : " << N << '\n';
    std::cout << "K       : " << K << '\n';
    std::cout << "Grid    : ("
              << grid_x << ", "
              << grid_y << ")\n";
    std::cout << "Block   : "
              << block_threads
              << " threads\n";
    std::cout << "alpha   : " << alpha << '\n';
    std::cout << "beta    : " << beta << '\n';
    std::cout << "=====================================\n";

    // ------------------------------------------------------------
    // Allocate host memory
    //
    // 用 size_t，避免大矩阵时 int 乘法溢出
    // ------------------------------------------------------------
    const size_t num_A =
        static_cast<size_t>(M) * K;

    const size_t num_B =
        static_cast<size_t>(K) * N;

    const size_t num_C =
        static_cast<size_t>(M) * N;

    const size_t bytes_A =
        num_A * sizeof(float);

    const size_t bytes_B =
        num_B * sizeof(float);

    const size_t bytes_C =
        num_C * sizeof(float);

    std::vector<float> h_A(num_A);
    std::vector<float> h_B(num_B);

    // 简单确定性初始化即可。
    // profile 不做 correctness benchmark，所以没必要复杂随机数。
    for (size_t i = 0; i < num_A; ++i) {
        h_A[i] =
            static_cast<float>((i % 13) - 6) * 0.01f;
    }

    for (size_t i = 0; i < num_B; ++i) {
        h_B[i] =
            static_cast<float>((i % 17) - 8) * 0.01f;
    }

    // ------------------------------------------------------------
    // Allocate device memory
    // ------------------------------------------------------------
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    // ------------------------------------------------------------
    // H2D
    //
    // 全部发生在 profiling 区间之外
    // ------------------------------------------------------------
    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            h_A.data(),
            bytes_A,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            h_B.data(),
            bytes_B,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));

    // 使用 default stream 即可
    cudaStream_t stream = 0;

    // ------------------------------------------------------------
    // Warmup
    //
    // 用于：
    // - CUDA context 初始化
    // - module/kernel 首次加载
    // - 避免第一次 launch 的额外开销影响观察
    //
    // 不在正式 profiling range 内
    // ------------------------------------------------------------
    if (version == "v1") {
        launch_sgemm_v1(
            d_A,
            d_B,
            d_C,
            M,
            N,
            K,
            alpha,
            beta,
            stream
        );
    }
    else {
        launch_sgemm_v2(
            d_A,
            d_B,
            d_C,
            M,
            N,
            K,
            alpha,
            beta,
            stream
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------------------------------------------------
    // Profiling range
    //
    // 配合：
    //
    // ncu --profile-from-start off ...
    //
    // 这样 NCU 只采 cudaProfilerStart() ~ Stop() 之间的 kernel。
    // ------------------------------------------------------------
    CUDA_CHECK(cudaProfilerStart());

    if (version == "v1") {
        launch_sgemm_v1(
            d_A,
            d_B,
            d_C,
            M,
            N,
            K,
            alpha,
            beta,
            stream
        );
    }
    else {
        launch_sgemm_v2(
            d_A,
            d_B,
            d_C,
            M,
            N,
            K,
            alpha,
            beta,
            stream
        );
    }

    CUDA_CHECK(cudaGetLastError());

    // 必须确保唯一的目标 kernel 已经执行完成
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaProfilerStop());

    // ------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return EXIT_SUCCESS;
}