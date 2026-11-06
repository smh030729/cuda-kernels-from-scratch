add_executable(
    softmax_bench

    benchmark.cu
    softmax_cpu.cpp
    softmax_v1.cu
    softmax_v2.cu
)

target_include_directories(
    softmax_bench
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)