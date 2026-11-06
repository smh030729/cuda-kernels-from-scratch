add_executable(
    conv2d_bench

    benchmark.cu
    conv2d_cpu.cpp
    conv2d_v2.cu
    conv2d_v3.cu
)

target_include_directories(
    conv2d_bench
    PRIVATE
    
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)