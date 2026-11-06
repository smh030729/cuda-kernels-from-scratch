add_executable(
    rmsnorm_bench

    benchmark.cu
    rmsnorm_cpu.cpp
    rmsnorm_v1.cu
)

target_include_directories(
    rmsnorm_bench
    PRIVATE
    
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)