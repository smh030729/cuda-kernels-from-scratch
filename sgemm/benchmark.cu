#include "sgemm.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>


#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        std::cerr << "CUDA error: "                                        \
                  << cudaGetErrorString(err)                               \
                  << " at " << __FILE__ << ":" << __LINE__                 \
                  << std::endl;                                            \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)


#define CUBLAS_CHECK(call)                                                 \
do {                                                                       \
    cublasStatus_t status = (call);                                        \
    if (status != CUBLAS_STATUS_SUCCESS) {                                 \
        std::cerr << "cuBLAS error: "                                      \
                  << static_cast<int>(status)                              \
                  << " at " << __FILE__ << ":" << __LINE__                 \
                  << std::endl;                                            \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)


template <typename Func>
float benchmark_cuda(
    Func func,
    int warmup,
    int repeat,
    cudaStream_t stream
) {
    // warmup
    for (int i = 0; i < warmup; ++i) {
        func();
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));

    for (int i = 0; i < repeat; ++i) {
        func();
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        )
    );

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repeat;
}


double calc_gflops(
    int M,
    int N,
    int K,
    float time_ms
) {
    const double flops =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);

    return flops / (time_ms * 1e6);
}


bool check_result(
    const std::vector<float>& C,
    float expected,
    float atol = 1e-2f,
    float rtol = 1e-4f
) {
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;

    for (size_t i = 0; i < C.size(); ++i) {

        const float abs_error =
            std::fabs(C[i] - expected);

        const float rel_error =
            abs_error /
            std::max(std::fabs(expected), 1e-6f);

        max_abs_error =
            std::max(max_abs_error, abs_error);

        max_rel_error =
            std::max(max_rel_error, rel_error);

        if (abs_error > atol &&
            rel_error > rtol) {

            std::cerr
                << "Mismatch at index "
                << i
                << ": got "
                << C[i]
                << ", expected "
                << expected
                << std::endl;

            return false;
        }
    }

    std::cout
        << "max_abs_error = "
        << max_abs_error
        << ", max_rel_error = "
        << max_rel_error
        << std::endl;

    return true;
}


int main()
{
    std::vector<int> sizes = {
        256,
        512,
        1024,
        2048,
        4096,
        8192
    };

    constexpr int WARMUP = 20;
    constexpr int REPEAT = 100;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaStream_t stream;
    CUDA_CHECK(
        cudaStreamCreate(&stream)
    );

    cublasHandle_t handle;
    CUBLAS_CHECK(
        cublasCreate(&handle)
    );

    CUBLAS_CHECK(
        cublasSetStream(
            handle,
            stream
        )
    );

    std::cout
        << "Size,"
        << "CUBLAS_ms,"
        << "CUBLAS_GFLOPS,"
        << "V1_ms,"
        << "V1_GFLOPS,"
        << "V1_Ratio,"
        << "V1_Correct,"
        << "V2_ms,"
        << "V2_GFLOPS,"
        << "V2_Ratio,"
        << "V2_Correct"
        << std::endl;


    for (int size : sizes)
    {
        const int M = size;
        const int N = size;
        const int K = size;

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


        // ========================================================
        // Host data
        // ========================================================

        std::vector<float> h_A(
            num_A,
            1.0f
        );

        std::vector<float> h_B(
            num_B,
            2.0f
        );

        std::vector<float> h_C(
            num_C,
            0.0f
        );


        // ========================================================
        // Device memory
        // ========================================================

        float* d_A = nullptr;
        float* d_B = nullptr;

        float* d_C_cublas = nullptr;
        float* d_C_v1 = nullptr;
        float* d_C_v2 = nullptr;


        CUDA_CHECK(
            cudaMalloc(
                &d_A,
                bytes_A
            )
        );

        CUDA_CHECK(
            cudaMalloc(
                &d_B,
                bytes_B
            )
        );

        CUDA_CHECK(
            cudaMalloc(
                &d_C_cublas,
                bytes_C
            )
        );

        CUDA_CHECK(
            cudaMalloc(
                &d_C_v1,
                bytes_C
            )
        );

        CUDA_CHECK(
            cudaMalloc(
                &d_C_v2,
                bytes_C
            )
        );


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


        CUDA_CHECK(
            cudaMemset(
                d_C_cublas,
                0,
                bytes_C
            )
        );

        CUDA_CHECK(
            cudaMemset(
                d_C_v1,
                0,
                bytes_C
            )
        );

        CUDA_CHECK(
            cudaMemset(
                d_C_v2,
                0,
                bytes_C
            )
        );


        // ========================================================
        // cuBLAS
        //
        // Our matrices are row-major:
        //
        // C = A * B
        //
        // cuBLAS interprets memory as column-major.
        //
        // Use:
        //
        // C^T = B^T * A^T
        //
        // Therefore swap A/B and M/N.
        // ========================================================

        auto cublas_func = [&]() {

            CUBLAS_CHECK(
                cublasSgemm(
                    handle,

                    CUBLAS_OP_N,
                    CUBLAS_OP_N,

                    N,
                    M,
                    K,

                    &alpha,

                    d_B,
                    N,

                    d_A,
                    K,

                    &beta,

                    d_C_cublas,
                    N
                )
            );
        };


        const float cublas_ms =
            benchmark_cuda(
                cublas_func,
                WARMUP,
                REPEAT,
                stream
            );


        const double cublas_gflops =
            calc_gflops(
                M,
                N,
                K,
                cublas_ms
            );


        // ========================================================
        // SGEMM V1
        // ========================================================

        auto v1_func = [&]() {

            launch_sgemm_v1(
                d_A,
                d_B,
                d_C_v1,

                M,
                N,
                K,

                alpha,
                beta,

                stream
            );
        };


        const float v1_ms =
            benchmark_cuda(
                v1_func,
                WARMUP,
                REPEAT,
                stream
            );


        const double v1_gflops =
            calc_gflops(
                M,
                N,
                K,
                v1_ms
            );


        // ========================================================
        // SGEMM V2
        // ========================================================

        auto v2_func = [&]() {

            launch_sgemm_v2(
                d_A,
                d_B,
                d_C_v2,

                M,
                N,
                K,

                alpha,
                beta,

                stream
            );
        };


        const float v2_ms =
            benchmark_cuda(
                v2_func,
                WARMUP,
                REPEAT,
                stream
            );


        const double v2_gflops =
            calc_gflops(
                M,
                N,
                K,
                v2_ms
            );


        // ========================================================
        // Correctness
        //
        // A = 1
        // B = 2
        //
        // C[i][j] =
        // sum_k 1 * 2
        // = 2K
        // ========================================================

        const float expected =
            2.0f *
            static_cast<float>(K);


        // V1
        CUDA_CHECK(
            cudaMemcpy(
                h_C.data(),
                d_C_v1,
                bytes_C,
                cudaMemcpyDeviceToHost
            )
        );

        const bool v1_correct =
            check_result(
                h_C,
                expected
            );


        // V2
        CUDA_CHECK(
            cudaMemcpy(
                h_C.data(),
                d_C_v2,
                bytes_C,
                cudaMemcpyDeviceToHost
            )
        );

        const bool v2_correct =
            check_result(
                h_C,
                expected
            );


        // ========================================================
        // Ratios
        // ========================================================

        const double v1_ratio =
            v1_gflops /
            cublas_gflops;

        const double v2_ratio =
            v2_gflops /
            cublas_gflops;


        // ========================================================
        // Output
        // ========================================================

        std::cout
            << size << ","

            << cublas_ms << ","
            << cublas_gflops << ","

            << v1_ms << ","
            << v1_gflops << ","
            << v1_ratio << ","
            << (v1_correct ? "PASS" : "FAIL") << ","

            << v2_ms << ","
            << v2_gflops << ","
            << v2_ratio << ","
            << (v2_correct ? "PASS" : "FAIL")

            << std::endl;


        // ========================================================
        // Cleanup
        // ========================================================

        CUDA_CHECK(
            cudaFree(d_A)
        );

        CUDA_CHECK(
            cudaFree(d_B)
        );

        CUDA_CHECK(
            cudaFree(d_C_cublas)
        );

        CUDA_CHECK(
            cudaFree(d_C_v1)
        );

        CUDA_CHECK(
            cudaFree(d_C_v2)
        );
    }


    CUBLAS_CHECK(
        cublasDestroy(handle)
    );

    CUDA_CHECK(
        cudaStreamDestroy(stream)
    );

    return 0;
}