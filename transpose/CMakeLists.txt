add_executable(
    transpose_bench

    benchmark.cu
    transpose_cpu.cpp
    transpose.cu
)

target_include_directories(
    transpose_bench
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)