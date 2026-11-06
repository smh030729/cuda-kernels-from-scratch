#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cuda_check.cuh"
#include "flashattention2.cuh"
#include "flashattention2_cpu.h"
#include "random.h"

constexpr int HEAD_DIM = 64;

bool check_output(const char* name, const float* output, const float* reference, size_t count) {
    float max_error = 0.0f;
    bool passed = true;
    for (size_t i = 0; i < count; ++i) {
        const float error = std::abs(output[i] - reference[i]);
        max_error = std::max(max_error, error);
        if (!std::isfinite(output[i]) || error > 5e-4f + 2e-3f * std::abs(reference[i])) {
            passed = false;
        }
    }
    std::cout << "  " << name << ": " << (passed ? "PASS" : "FAIL")
              << " max_abs_error=" << max_error << '\n';
    return passed;
}

bool check_size(int N, cudaStream_t stream) {
    const size_t count = static_cast<size_t>(N) * HEAD_DIM;
    const size_t float_bytes = count * sizeof(float);
    const size_t half_bytes = count * sizeof(half);
    std::vector<float> Q(count), K(count), V(count), reference(count), output(count);
    std::vector<half> Qh(count), Kh(count), Vh(count), prepared(count), half_output(count);
    fill_random(Q.data(), count, 1.0f, -1.0f, 42);
    fill_random(K.data(), count, 1.0f, -1.0f, 43);
    fill_random(V.data(), count, 1.0f, -1.0f, 44);
    for (size_t i = 0; i < count; ++i) {
        Qh[i] = __float2half_rn(Q[i]);
        Kh[i] = __float2half_rn(K[i]);
        Vh[i] = __float2half_rn(V[i]);
        // The reference must use exactly the same quantized input values.
        Q[i] = __half2float(Qh[i]);
        K[i] = __half2float(Kh[i]);
        V[i] = __half2float(Vh[i]);
    }
    flashattention2_cpu(Q.data(), K.data(), V.data(), reference.data(), N, HEAD_DIM);

    float *d_Q, *d_K, *d_V, *d_O;
    half *d_Qh, *d_Kh, *d_Kt, *d_Vh, *d_Oh;
    CUDA_CHECK(cudaMalloc(&d_Q, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_K, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_O, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_Qh, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_Kh, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_Kt, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_Vh, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_Oh, half_bytes));
    CUDA_CHECK(cudaMemcpy(d_Q, Q.data(), float_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, K.data(), float_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, V.data(), float_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Kh, Kh.data(), half_bytes, cudaMemcpyHostToDevice));
    launch_prepare_flashattention2_v2(d_Q, d_K, d_V, d_Qh, d_Kt, d_Vh, N, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    bool passed = true;
    for (int input = 0; input < 3; ++input) {
        const half* device = input == 0 ? d_Qh : (input == 1 ? d_Kt : d_Vh);
        CUDA_CHECK(cudaMemcpy(prepared.data(), device, half_bytes, cudaMemcpyDeviceToHost));
        for (int row = 0; row < N; ++row) {
            for (int col = 0; col < HEAD_DIM; ++col) {
                const half expected = input == 0 ? Qh[row * HEAD_DIM + col]
                                    : (input == 1 ? Kh[row * HEAD_DIM + col] : Vh[row * HEAD_DIM + col]);
                const size_t index = input == 1 ? static_cast<size_t>(col) * N + row
                                               : static_cast<size_t>(row) * HEAD_DIM + col;
                if (__half2float(prepared[index]) != __half2float(expected)) passed = false;
            }
        }
    }
    std::cout << "N=" << N << " prepare=" << (passed ? "PASS" : "FAIL") << '\n';

    using FloatLaunch = void (*)(const float*, const float*, const float*, float*, int, cudaStream_t);
    const FloatLaunch float_launches[] = {
        launch_flashattention2_v0, launch_flashattention2_v1, launch_flashattention2_v2a, launch_flashattention2_v0_br64
    };
    const char* float_names[] = {"v0", "v1", "v2a", "v0_br64"};
    for (int version = 0; version < 4; ++version) {
        if (version == 3 && N % 64) continue;
        CUDA_CHECK(cudaMemsetAsync(d_O, 0xff, float_bytes, stream));
        float_launches[version](d_Q, d_K, d_V, d_O, N, stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(output.data(), d_O, float_bytes, cudaMemcpyDeviceToHost));
        passed = check_output(float_names[version], output.data(), reference.data(), count) && passed;
    }

    CUDA_CHECK(cudaMemsetAsync(d_O, 0xff, float_bytes, stream));
    launch_flashattention2_v1_fp16(d_Qh, d_Kh, d_Vh, d_O, N, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(output.data(), d_O, float_bytes, cudaMemcpyDeviceToHost));
    passed = check_output("v1_fp16_out32", output.data(), reference.data(), count) && passed;

    using HalfLaunch = void (*)(const half*, const half*, const half*, half*, int, cudaStream_t);
    const HalfLaunch half_launches[] = {
        launch_flashattention2_v0_fp16, launch_flashattention2_v1_fp16,
        launch_flashattention2_v2a_fp16, launch_flashattention2_v2b, launch_flashattention2_v2c,
        launch_flashattention2_v0_br64_fp16, launch_flashattention2_v2c_bc32_qreg
    };
    const char* half_names[] = {"v0_fp16", "v1_fp16", "v2a_fp16", "v2b", "v2c", "v0_br64_fp16", "v2c_bc32_qreg"};
    for (int version = 0; version < 7; ++version) {
        if (version >= 3 && N % 64 != 0) {
            bool rejected = false;
            try {
                half_launches[version](d_Qh, d_Kt, d_Vh, d_Oh, N, stream);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            passed = rejected && passed;
            continue;
        }
        CUDA_CHECK(cudaMemsetAsync(d_Oh, 0xff, half_bytes, stream));
        half_launches[version](d_Qh, (version == 3 || version == 4 || version == 6) ? d_Kt : d_Kh, d_Vh, d_Oh, N, stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(half_output.data(), d_Oh, half_bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) output[i] = __half2float(half_output[i]);
        passed = check_output(half_names[version], output.data(), reference.data(), count) && passed;
    }
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_Qh));
    CUDA_CHECK(cudaFree(d_Kh));
    CUDA_CHECK(cudaFree(d_Kt));
    CUDA_CHECK(cudaFree(d_Vh));
    CUDA_CHECK(cudaFree(d_Oh));
    return passed;
}

int main() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    bool passed = true;
    for (int N : {1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256}) {
        passed = check_size(N, stream) && passed;
    }
    launch_flashattention2_v2c_bc32_qreg(nullptr, nullptr, nullptr, nullptr, 0, stream);
    launch_flashattention2_v0_fp16(nullptr, nullptr, nullptr, nullptr, 0, stream);
    launch_flashattention2_v0_br64_fp16(nullptr, nullptr, nullptr, nullptr, 0, stream);
    launch_flashattention2_v1_fp16(nullptr, nullptr, nullptr, static_cast<half*>(nullptr), 0, stream);
    launch_flashattention2_v2a_fp16(nullptr, nullptr, nullptr, nullptr, 0, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamDestroy(stream));
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
