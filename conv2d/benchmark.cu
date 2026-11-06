#include "conv2d.cuh"
#include "conv2d_cpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>


#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error: "                           \
                  << cudaGetErrorString(err)                   \
                  << " at " << __FILE__                       \
                  << ":" << __LINE__                          \
                  << std::endl;                               \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)


struct ConvConfig {
    int N;
    int Cin;
    int H;
    int W;

    int Cout;
    int KH;
    int KW;

    int stride_h;
    int stride_w;

    int pad_h;
    int pad_w;
};


static void fill_random(
    std::vector<float>& data,
    float low = -1.0f,
    float high = 1.0f
) {
    static std::mt19937 gen(1234);

    std::uniform_real_distribution<float> dist(
        low,
        high
    );

    for (float& x : data) {
        x = dist(gen);
    }
}


static bool check_result(
    const std::vector<float>& ref,
    const std::vector<float>& result,
    float atol = 1e-3f,
    float rtol = 1e-3f
) {
    if (ref.size() != result.size()) {
        return false;
    }

    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;

    size_t max_error_idx = 0;

    for (size_t i = 0; i < ref.size(); ++i) {

        const float abs_error =
            std::abs(ref[i] - result[i]);

        const float rel_error =
            abs_error /
            std::max(std::abs(ref[i]), 1e-6f);

        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
            max_error_idx = i;
        }

        max_rel_error =
            std::max(max_rel_error, rel_error);

        if (abs_error >
            atol + rtol * std::abs(ref[i])) {

            std::cout
                << "Mismatch at index "
                << i
                << ": ref = "
                << ref[i]
                << ", result = "
                << result[i]
                << ", abs_error = "
                << abs_error
                << '\n';

            return false;
        }
    }

    std::cout
        << "max_abs_error = "
        << max_abs_error
        << ", max_rel_error = "
        << max_rel_error
        << ", max_error_idx = "
        << max_error_idx
        << '\n';

    return true;
}


template <typename Func>
static float benchmark_cuda(
    Func func,
    int warmup = 10,
    int repeat = 100
) {
    for (int i = 0; i < warmup; ++i) {
        func();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        func();
    }

    CUDA_CHECK(cudaEventRecord(stop));
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


static double conv_gflops(
    const ConvConfig& cfg,
    int Hout,
    int Wout,
    float ms
) {
    const double flops =
        2.0 *
        cfg.N *
        cfg.Cout *
        Hout *
        Wout *
        cfg.Cin *
        cfg.KH *
        cfg.KW;

    return flops / (ms * 1e6);
}


static void run_case(
    const ConvConfig& cfg
) {
    const int Hout =
        (cfg.H + 2 * cfg.pad_h - cfg.KH)
        / cfg.stride_h + 1;

    const int Wout =
        (cfg.W + 2 * cfg.pad_w - cfg.KW)
        / cfg.stride_w + 1;


    const size_t input_size =
        static_cast<size_t>(cfg.N) *
        cfg.Cin *
        cfg.H *
        cfg.W;

    const size_t weight_size =
        static_cast<size_t>(cfg.Cout) *
        cfg.Cin *
        cfg.KH *
        cfg.KW;

    const size_t output_size =
        static_cast<size_t>(cfg.N) *
        cfg.Cout *
        Hout *
        Wout;


    std::vector<float> h_input(input_size);
    std::vector<float> h_weight(weight_size);

    std::vector<float> h_output_cpu(output_size);
    std::vector<float> h_output_v2(output_size);
    std::vector<float> h_output_v3(output_size);


    fill_random(h_input);
    fill_random(h_weight);


    // ============================================================
    // CPU reference
    // ============================================================

    conv2d_cpu(
        h_input.data(),
        h_weight.data(),
        h_output_cpu.data(),

        cfg.N,
        cfg.Cin,
        cfg.H,
        cfg.W,

        cfg.Cout,
        cfg.KH,
        cfg.KW,

        cfg.stride_h,
        cfg.stride_w,

        cfg.pad_h,
        cfg.pad_w
    );


    // ============================================================
    // GPU memory
    // ============================================================

    float* d_input = nullptr;
    float* d_weight = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            input_size * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_weight,
            weight_size * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_output,
            output_size * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            input_size * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_weight,
            h_weight.data(),
            weight_size * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));


    // ============================================================
    // V2 correctness
    // ============================================================

    launch_conv2d_v2(
        d_input,
        d_weight,
        d_output,

        cfg.N,
        cfg.Cin,
        cfg.H,
        cfg.W,

        cfg.Cout,
        cfg.KH,
        cfg.KW,

        cfg.stride_h,
        cfg.stride_w,

        cfg.pad_h,
        cfg.pad_w,

        stream
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(
        cudaMemcpy(
            h_output_v2.data(),
            d_output,
            output_size * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    const bool v2_correct =
        check_result(
            h_output_cpu,
            h_output_v2
        );


    // ============================================================
    // V3 correctness
    // ============================================================

    launch_conv2d_v3(
        d_input,
        d_weight,
        d_output,

        cfg.N,
        cfg.Cin,
        cfg.H,
        cfg.W,

        cfg.Cout,
        cfg.KH,
        cfg.KW,

        cfg.stride_h,
        cfg.stride_w,

        cfg.pad_h,
        cfg.pad_w,

        stream
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(
        cudaMemcpy(
            h_output_v3.data(),
            d_output,
            output_size * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    const bool v3_correct =
        check_result(
            h_output_cpu,
            h_output_v3
        );


    // ============================================================
    // Benchmark V2
    // ============================================================

    const float v2_ms =
        benchmark_cuda(
            [&]() {
                launch_conv2d_v2(
                    d_input,
                    d_weight,
                    d_output,

                    cfg.N,
                    cfg.Cin,
                    cfg.H,
                    cfg.W,

                    cfg.Cout,
                    cfg.KH,
                    cfg.KW,

                    cfg.stride_h,
                    cfg.stride_w,

                    cfg.pad_h,
                    cfg.pad_w,

                    stream
                );
            }
        );


    // ============================================================
    // Benchmark V3
    // ============================================================

    const float v3_ms =
        benchmark_cuda(
            [&]() {
                launch_conv2d_v3(
                    d_input,
                    d_weight,
                    d_output,

                    cfg.N,
                    cfg.Cin,
                    cfg.H,
                    cfg.W,

                    cfg.Cout,
                    cfg.KH,
                    cfg.KW,

                    cfg.stride_h,
                    cfg.stride_w,

                    cfg.pad_h,
                    cfg.pad_w,

                    stream
                );
            }
        );


    const double v2_gflops =
        conv_gflops(
            cfg,
            Hout,
            Wout,
            v2_ms
        );

    const double v3_gflops =
        conv_gflops(
            cfg,
            Hout,
            Wout,
            v3_ms
        );


    std::cout
        << cfg.N << ","
        << cfg.Cin << ","
        << cfg.H << ","
        << cfg.W << ","
        << cfg.Cout << ","
        << cfg.KH << ","
        << cfg.KW << ","
        << Hout << ","
        << Wout << ","

        << std::fixed
        << std::setprecision(6)

        << v2_ms << ","
        << v2_gflops << ","
        << (v2_correct ? "PASS" : "FAIL") << ","

        << v3_ms << ","
        << v3_gflops << ","
        << (v3_correct ? "PASS" : "FAIL") << ","

        << (v2_ms / v3_ms)

        << '\n';


    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_output));
}


int main()
{
    std::vector<ConvConfig> configs = {

        // N, Cin, H, W, Cout, KH, KW,
        // stride_h, stride_w, pad_h, pad_w

        {1, 32,  56, 56,  32, 3, 3, 1, 1, 1, 1},

        {1, 64,  56, 56,  64, 3, 3, 1, 1, 1, 1},

        {1, 64,  28, 28, 128, 3, 3, 1, 1, 1, 1},

        {1, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1},

        {1, 128, 14, 14, 256, 3, 3, 1, 1, 1, 1}
    };


    std::cout
        << "N,Cin,H,W,Cout,KH,KW,Hout,Wout,"
        << "V2_ms,V2_GFLOPS,V2_Correct,"
        << "V3_ms,V3_GFLOPS,V3_Correct,"
        << "V3_Speedup"
        << '\n';


    for (const auto& cfg : configs) {

        run_case(cfg);
    }


    return 0;
}