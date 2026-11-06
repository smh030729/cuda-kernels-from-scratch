#include "sgemm_v4.cuh"
#include "cuda_check.cuh"
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define BLAS_CHECK(call) do { auto s = (call); if (s != CUBLAS_STATUS_SUCCESS) \
  throw std::runtime_error("cuBLAS status " + std::to_string(int(s))); } while(0)

template<typename T> struct Buffer {
  T* ptr;
  explicit Buffer(size_t n) { CUDA_CHECK(cudaMalloc(&ptr, std::max(size_t(1), n) * sizeof(T))); }
  ~Buffer() { cudaFree(ptr); }
};

struct Shape { int m, n, k; };

void reference_cpu(const half* a, const half* b, const float* c, std::vector<float>& out,
                   Shape s, float alpha, float beta) {
  for (int i = 0; i < s.m; ++i) for (int j = 0; j < s.n; ++j) {
    double sum = 0.;
    for (int k = 0; k < s.k; ++k) sum += double(__half2float(a[size_t(i) * s.k + k])) * __half2float(b[size_t(k) * s.n + j]);
    out[size_t(i) * s.n + j] = float(double(alpha) * sum + (beta == 0.f ? 0. : double(beta) * c[size_t(i) * s.n + j]));
  }
}

void check_close(const std::vector<float>& value, const std::vector<float>& reference,
                 double atol, double rtol, const std::string& label) {
  double max_abs = 0., squared = 0., ref_squared = 0.;
  for (size_t i = 0; i < value.size(); ++i) {
    const double diff = std::abs(double(value[i]) - reference[i]);
    if (!std::isfinite(value[i]) || diff > atol + rtol * std::abs(reference[i]))
      throw std::runtime_error(label + " mismatch at " + std::to_string(i) +
          ": got " + std::to_string(value[i]) + " expected " + std::to_string(reference[i]));
    max_abs = std::max(max_abs, diff);
    squared += diff * diff;
    ref_squared += double(reference[i]) * reference[i];
  }
  std::cout << "{\"kind\":\"check\",\"label\":\"" << label << "\",\"max_abs\":" << max_abs
            << ",\"relative_l2\":" << std::sqrt(squared / std::max(ref_squared, 1e-30)) << "}\n";
}

void correctness() {
  std::mt19937 random(20261106);
  std::uniform_real_distribution<float> dist(-.5f, .5f);
  const std::vector<Shape> shapes = {{1,1,0}, {1,1,1}, {13,17,11}, {127,129,33},
      {128,128,16}, {256,256,64}, {384,256,48}, {256,128,24}, {63,97,257},
      {128,128,32}, {128,128,16}, {130,136,40}, {1153,136,40},
      {1024,1024,32}, {1000,1016,40}, {128,128,32}};
  int checks = 0;
  for (size_t si = 0; si < shapes.size(); ++si) {
    const auto s = shapes[si];
    std::vector<half> a(std::max(1, s.m * s.k)), b(std::max(1, s.k * s.n));
    std::vector<float> c(s.m * s.n);
    for (auto& v : a) v = dist(random);
    for (auto& v : b) v = dist(random);
    for (auto& v : c) v = dist(random);
    const float alpha = si == 10 ? 0.f : (si % 2 ? -.75f : 1.f);
    const float beta = si % 3 ? .25f : 0.f;
    const int offset = si == shapes.size() - 1 ? 1 : 0;
    std::vector<float> initial(c.size() + offset + 1, 123456.f), reference(c.size()), actual(c.size());
    if (beta == 0.f) std::fill(c.begin(), c.end(), NAN);
    std::copy(c.begin(), c.end(), initial.begin() + offset);
    reference_cpu(a.data(), b.data(), c.data(), reference, s, alpha, beta);
    Buffer<half> da(a.size() + offset), db(b.size() + offset);
    Buffer<float> dc(initial.size());
    CUDA_CHECK(cudaMemcpy(da.ptr + offset, a.data(), a.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db.ptr + offset, b.data(), b.size() * sizeof(half), cudaMemcpyHostToDevice));
    for (int variant = 0; variant < SGEMM_V4_VARIANTS; ++variant) {
      CUDA_CHECK(cudaMemcpy(dc.ptr, initial.data(), initial.size() * 4, cudaMemcpyHostToDevice));
      launch_sgemm_v4_variant(variant, da.ptr + offset, db.ptr + offset, dc.ptr + offset,
                              s.m, s.n, s.k, alpha, beta);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(actual.data(), dc.ptr + offset, actual.size() * 4, cudaMemcpyDeviceToHost));
      check_close(actual, reference, 1e-4, 2e-5,
          "cpu_shape" + std::to_string(si) + "_variant" + std::to_string(variant));
      float guard;
      CUDA_CHECK(cudaMemcpy(&guard, dc.ptr + offset + c.size(), 4, cudaMemcpyDeviceToHost));
      if (guard != 123456.f) throw std::runtime_error("output suffix guard overwritten");
      if (offset) {
        CUDA_CHECK(cudaMemcpy(&guard, dc.ptr, 4, cudaMemcpyDeviceToHost));
        if (guard != 123456.f) throw std::runtime_error("output prefix guard overwritten");
      }
      ++checks;
    }
  }
  launch_sgemm_v4(nullptr, nullptr, nullptr, 0, 13, 17);
  launch_sgemm_v4(nullptr, nullptr, nullptr, 13, 0, 17);
  CUDA_CHECK(cudaGetLastError());
  std::cout << "{\"kind\":\"correctness_complete\",\"checks\":" << checks << "}\n";
}

int main(int argc, char** argv) try {
  int M = 4096, N = 4096, K = 4096, rounds = 9, repeat = 10, variant = -1, impl = -1;
  bool check = false, profile = false;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--check") { check = true; continue; }
    if (key == "--profile") { profile = true; continue; }
    if (i + 1 == argc) throw std::invalid_argument("missing argument");
    const int value = std::stoi(argv[++i]);
    if (key == "--m") M = value;
    else if (key == "--n") N = value;
    else if (key == "--k") K = value;
    else if (key == "--size") M = N = K = value;
    else if (key == "--rounds") rounds = value;
    else if (key == "--repeat") repeat = value;
    else if (key == "--variant") variant = value;
    else if (key == "--impl") impl = value;
    else throw std::invalid_argument("unknown option: " + key);
  }
  std::cout << std::setprecision(10);
  if (check) { correctness(); return 0; }
  if (M <= 0 || N <= 0 || K <= 0 || rounds <= 0 || repeat <= 0)
    throw std::invalid_argument("positive dimensions, rounds and repeat required");
  cudaDeviceProp device;
  CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
  cublasHandle_t handle;
  BLAS_CHECK(cublasCreate(&handle));
  BLAS_CHECK(cublasSetMathMode(handle, static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
  cublasMath_t math;
  BLAS_CHECK(cublasGetMathMode(handle, &math));
  if (math != static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)) throw std::runtime_error("FP32 reduction mode not set");
  int blas_version;
  BLAS_CHECK(cublasGetVersion(handle, &blas_version));
  std::cout << "{\"kind\":\"environment\",\"gpu\":\"" << device.name
            << "\",\"cublas_version\":" << blas_version << ",\"math\":\"FP16_input_FP32_accum_output_no_reduced_reduction\",\"M\":"
            << M << ",\"N\":" << N << ",\"K\":" << K << "}\n";
  std::mt19937 rng(20261106);
  std::uniform_real_distribution<float> dist(-.5f, .5f);
  std::vector<half> a(size_t(M) * K), b(size_t(K) * N);
  std::vector<float> result(size_t(M) * N), reference(result.size());
  for (auto& v : a) v = dist(rng);
  for (auto& v : b) v = dist(rng);
  Buffer<half> da(a.size()), db(b.size());
  Buffer<float> dc(result.size()), dr(result.size());
  CUDA_CHECK(cudaMemcpy(da.ptr, a.data(), a.size() * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db.ptr, b.data(), b.size() * sizeof(half), cudaMemcpyHostToDevice));
  const float alpha = 1.f, beta = 0.f;
  auto blas = [&](float* output) {
    BLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                          &alpha, db.ptr, CUDA_R_16F, N, da.ptr, CUDA_R_16F, K,
                          &beta, output, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };
  blas(dr.ptr);
  CUDA_CHECK(cudaMemcpy(reference.data(), dr.ptr, reference.size() * 4, cudaMemcpyDeviceToHost));
  // Independently check sampled cuBLAS outputs against FP64 CPU dot products.
  double sampled_max_abs = 0.;
  for (int sample = 0; sample < 128; ++sample) {
    const int row = sample == 0 ? 0 : (sample == 1 ? M - 1 : int(rng() % M));
    const int col = sample == 0 ? 0 : (sample == 1 ? N - 1 : int(rng() % N));
    double value = 0.;
    for (int k = 0; k < K; ++k) value += double(__half2float(a[size_t(row) * K + k])) * __half2float(b[size_t(k) * N + col]);
    const double diff = std::abs(value - reference[size_t(row) * N + col]);
    sampled_max_abs = std::max(sampled_max_abs, diff);
    if (diff > .002 + .0002 * std::abs(value)) throw std::runtime_error("cuBLAS CPU sample mismatch");
  }
  std::cout << "{\"kind\":\"cpu_sample_check\",\"samples\":128,\"max_abs\":" << sampled_max_abs << "}\n";
  struct Arm { int impl, variant; std::string name; };
  std::vector<Arm> arms = {{0, -1, "cublas_fp16_fp32"}};
  for (int v = 0; v < SGEMM_V4_VARIANTS; ++v)
    if (variant == -1 || variant == v) arms.push_back({3,v,sgemm_v4_variant_name(v)});
  if (impl >= 0) arms.erase(std::remove_if(arms.begin(), arms.end(),
      [&](const Arm& arm) { return arm.impl != impl; }), arms.end());
  if (arms.empty()) throw std::invalid_argument("no valid implementation selected");
  auto launch = [&](const Arm& arm) {
    if (arm.impl == 0) blas(dc.ptr);
    else launch_sgemm_v4_variant(arm.variant, da.ptr, db.ptr, dc.ptr, M,N,K);
    CUDA_CHECK(cudaGetLastError());
  };
  for (const auto& arm : arms) {
    launch(arm);
    CUDA_CHECK(cudaMemcpy(result.data(), dc.ptr, result.size() * 4, cudaMemcpyDeviceToHost));
    check_close(result, reference, .002, .0002, arm.name);
    for (int i = 0; i < 10; ++i) launch(arm);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  if (profile) {
    if (arms.size() != 1) throw std::invalid_argument("profile requires --impl and optionally --variant");
    CUDA_CHECK(cudaProfilerStart());
    launch(arms.front());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaProfilerStop());
  } else {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<int> order(arms.size());
    for (int round = 0; round < rounds; ++round) {
      std::iota(order.begin(), order.end(), 0);
      std::rotate(order.begin(), order.begin() + round % order.size(), order.end());
      if (round % 2) std::reverse(order.begin(), order.end());
      for (int index : order) {
        for (int i = 0; i < 3; ++i) launch(arms[index]);
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeat; ++i) launch(arms[index]);
        CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= repeat;
        std::cout << "{\"kind\":\"sample\",\"round\":" << round << ",\"impl\":\""
                  << arms[index].name << "\",\"ms\":" << ms << ",\"tflops\":"
                  << 2. * M * N * K / (ms * 1e9) << "}" << std::endl;
      }
    }
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  }
  BLAS_CHECK(cublasDestroy(handle));
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
