add_executable(
    reduction_bench

    benchmark.cu
    reduction_cpu.cpp
    reduction.cu
)

target_include_directories(
    reduction_bench
    PRIVATE
    
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

add_executable(
    reduction_profile

    profile.cu
    reduction.cu
)


target_include_directories(
    reduction_profile
    PRIVATE

    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)