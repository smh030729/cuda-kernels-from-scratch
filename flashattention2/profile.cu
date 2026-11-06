
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "flashattention2.cuh"

#define CUDA_CHECK(call) do {                                             \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)            \
                  << " at " << __FILE__ << ":" << __LINE__ << '\n';      \
        std::exit(EXIT_FAILURE);                                         \
    }                                                                    \
} while (0)

constexpr int HEAD_DIM = 64;

void launch_selected(
    const std::string& version,
    const float* Q, const float* K, const float* V,
    float* O, int N, cudaStream_t stream
) {
    if (version == "v0") {
        launch_flashattention2_v0(Q, K, V, O, N, stream);
    } else if (version == "v0_br64") {
        launch_flashattention2_v0_br64(Q, K, V, O, N, stream);
    } else if (version == "v1") {
        launch_flashattention2_v1(Q, K, V, O, N, stream);
    } else if (version == "v2a") {
        launch_flashattention2_v2a(Q, K, V, O, N, stream);
    } else {
        std::cerr << "Unsupported version: " << version << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void launch_selected_fp16(
    const std::string& version,
    const half* Q, const half* K, const half* Kt, const half* V,
    half* O, int N, cudaStream_t stream
) {
    if (version == "v0_fp16") {
        launch_flashattention2_v0_fp16(Q, K, V, O, N, stream);
    } else if (version == "v0_br64_fp16") {
        launch_flashattention2_v0_br64_fp16(Q, K, V, O, N, stream);
    } else if (version == "v1_fp16") {
        launch_flashattention2_v1_fp16(Q, K, V, O, N, stream);
    } else if (version == "v2a_fp16") {
        launch_flashattention2_v2a_fp16(Q, K, V, O, N, stream);
    } else if (version == "v2b") {
        launch_flashattention2_v2b(Q, Kt, V, O, N, stream);
    } else if (version == "v2c") {
        launch_flashattention2_v2c(Q, Kt, V, O, N, stream);
    } else if (version == "v2c_bc32_qreg") {
        launch_flashattention2_v2c_bc32_qreg(Q, Kt, V, O, N, stream);
    }
}

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " v0|v0_br64 [N]\n"
                  << "  " << argv[0] << " v1 [N]\n"
                  << "  " << argv[0] << " v2a [N]\n"
                  << "  " << argv[0] << " v0_fp16|v1_fp16|v1_fp16_out32|v2a_fp16 [N]\n"
                  << "  " << argv[0] << " v0_br64_fp16 [N]\n"
                  << "  " << argv[0] << " v2b|v2c|v2c_bc32_qreg [N]\n";
        return EXIT_FAILURE;
    }

    const std::string version = argv[1];
    const bool out32 = version == "v1_fp16_out32";
    const bool fp16 = out32 || version == "v0_fp16" || version == "v0_br64_fp16" || version == "v1_fp16"
                   || version == "v2a_fp16" || version == "v2b" || version == "v2c" || version == "v2c_bc32_qreg";

    if (!fp16 && version != "v0" && version != "v0_br64" && version != "v1" && version != "v2a") {
        std::cerr << "Unsupported version: " << version << '\n';
        return EXIT_FAILURE;
    }

    int N = 4096;
    if (argc == 3) N = std::stoi(argv[2]);

    if (N <= 0) {
        std::cerr << "N must be positive.\n";
        return EXIT_FAILURE;
    }

    if (N % 64 != 0) {
        std::cerr << "N must be a multiple of 64.\n";
        return EXIT_FAILURE;
    }

    const size_t num_elements = static_cast<size_t>(N) * HEAD_DIM;
    const size_t bytes = num_elements * sizeof(float);

    std::cout << "=====================================\n"
              << "FlashAttention-2 Profiling\n"
              << "Version  : " << version << '\n'
              << "N        : " << N << '\n'
              << "Head Dim : " << HEAD_DIM << '\n'
              << "Input    : " << (fp16 ? "FP16" : "FP32")
              << ", Output: " << (fp16 && !out32 ? "FP16" : "FP32") << '\n'
              << "K layout : " << ((version == "v2b" || version == "v2c" || version == "v2c_bc32_qreg") ? "Kt [64, N]" : "K [N, 64]") << '\n'
              << "Q/K/V/O  : [" << N << ", " << HEAD_DIM << "]\n"
              << "=====================================\n";

    std::vector<float> h_Q(num_elements);
    std::vector<float> h_K(num_elements);
    std::vector<float> h_V(num_elements);

    for (size_t i = 0; i < num_elements; ++i) {
        h_Q[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.01f;
        h_K[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.01f;
        h_V[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.01f;
    }

    float* d_Q = nullptr;
    float* d_K = nullptr;
    float* d_V = nullptr;
    float* d_O = nullptr;

    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_O, 0, bytes));

    cudaStream_t stream = 0;

    half* d_Qh = nullptr;
    half* d_Kh = nullptr;
    half* d_Kt = nullptr;
    half* d_Vh = nullptr;
    half* d_Oh = nullptr;
    if (fp16) {
        const size_t half_bytes = num_elements * sizeof(half);
        CUDA_CHECK(cudaMalloc(&d_Qh, half_bytes));
        CUDA_CHECK(cudaMalloc(&d_Kh, half_bytes));
        CUDA_CHECK(cudaMalloc(&d_Kt, half_bytes));
        CUDA_CHECK(cudaMalloc(&d_Vh, half_bytes));
        CUDA_CHECK(cudaMalloc(&d_Oh, half_bytes));
        std::vector<half> h_Kh(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
            h_Kh[i] = __float2half_rn(h_K[i]);
        }
        CUDA_CHECK(cudaMemcpy(d_Kh, h_Kh.data(), half_bytes, cudaMemcpyHostToDevice));
        launch_prepare_flashattention2_v2(d_Q, d_K, d_V, d_Qh, d_Kt, d_Vh, N, stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    const auto launch = [&]() {
        if (out32) {
            launch_flashattention2_v1_fp16(d_Qh, d_Kh, d_Vh, d_O, N, stream);
        } else if (fp16) {
            launch_selected_fp16(version, d_Qh, d_Kh, d_Kt, d_Vh, d_Oh, N, stream);
        } else {
            launch_selected(version, d_Q, d_K, d_V, d_O, N, stream);
        }
    };

    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaProfilerStart());

    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaProfilerStop());

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_Qh));
    CUDA_CHECK(cudaFree(d_Kh));
    CUDA_CHECK(cudaFree(d_Kt));
    CUDA_CHECK(cudaFree(d_Vh));
    CUDA_CHECK(cudaFree(d_Oh));

    return EXIT_SUCCESS;
}
