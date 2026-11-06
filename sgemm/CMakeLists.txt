find_package(CUDAToolkit REQUIRED)

add_executable(
    sgemm_bench
    benchmark.cu
    sgemm_cpu.cpp
    sgemm_v1.cu
    sgemm_v2.cu
)

target_include_directories(
    sgemm_bench
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

add_executable(
    sgemm_profile
    profile.cu
    sgemm_v1.cu
    sgemm_v2.cu
)

target_include_directories(
    sgemm_profile
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(
    sgemm_bench
    PRIVATE
    CUDA::cublas
)

target_compile_options(sgemm_profile PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
)

add_executable(sgemm_compare
    benchmark_compare.cu sgemm_v1.cu sgemm_v2.cu sgemm_v3.cu)
target_include_directories(sgemm_compare PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sgemm_compare PRIVATE CUDA::cublas)
target_compile_options(sgemm_compare PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>)

add_executable(sgemm_v4_compare benchmark_v4.cu sgemm_v4.cu)
target_include_directories(sgemm_v4_compare PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sgemm_v4_compare PRIVATE CUDA::cublas)
target_compile_options(sgemm_v4_compare PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>)
