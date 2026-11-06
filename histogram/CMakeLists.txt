add_executable(
    histogram_bench

    benchmark.cu
    histogram_cpu.cpp
    histogram.cu
)

target_include_directories(
    histogram_bench
    PRIVATE
    
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)