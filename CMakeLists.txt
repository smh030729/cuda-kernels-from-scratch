cmake_minimum_required(VERSION 3.20)

project(
    cuda_kernels_from_scratch
    LANGUAGES CXX CUDA
)

# C++ / CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 4070 Ti SUPER -> Compute Capability 8.9
set(CMAKE_CUDA_ARCHITECTURES 89)

add_subdirectory(reduction)
add_subdirectory(softmax)
add_subdirectory(rmsnorm)
add_subdirectory(sgemm)
add_subdirectory(flashattention2)
add_subdirectory(transpose)
add_subdirectory(histogram)
add_subdirectory(conv2d)
add_executable(catalog_basic_profile profiling/basic.cu
    softmax/softmax_v1.cu softmax/softmax_v2.cu rmsnorm/rmsnorm_v1.cu
    histogram/histogram.cu transpose/transpose.cu conv2d/conv2d_v2.cu conv2d/conv2d_v3.cu)
target_include_directories(catalog_basic_profile PRIVATE . include)
target_compile_options(catalog_basic_profile PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)

foreach(demo ldmatrix_x1 x1)
    add_executable(${demo}_demo tensorcore/${demo}.cu)
endforeach()
