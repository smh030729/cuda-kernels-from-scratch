set(FLASHATTENTION2_FP16_SOURCES
    flashattention2_v0_fp16.cu
    flashattention2_v0_br64_fp16.cu
    flashattention2_v1_fp16.cu
    flashattention2_v2a_fp16.cu
    flashattention2_v2_BC.cu
    flashattention2_v2c_bc32_qreg.cu
)

set(FLASHATTENTION2_SOURCES
    flashattention2_v0.cu
    flashattention2_v0_br64.cu
    flashattention2_v1.cu
    flashattention2_v2_A.cu
    ${FLASHATTENTION2_FP16_SOURCES}
)

add_executable(flashattention2_profile profile.cu ${FLASHATTENTION2_SOURCES})
add_executable(flashattention2_check
    check.cu
    flashattention2_cpu.cpp
    ${FLASHATTENTION2_SOURCES}
)

add_library(flashattention2_benchmark SHARED benchmark_api.cu ${FLASHATTENTION2_FP16_SOURCES})

foreach(target flashattention2_profile flashattention2_check flashattention2_benchmark)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
    )
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
    )
endforeach()
