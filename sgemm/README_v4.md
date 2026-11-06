# V4：FP16 Tensor Core GEMM

V4 独立于 V1–V3 的 FP32 CUDA Core 路径，计算连续 row-major 矩阵：

```cpp
// A[M,K], B[K,N]: half; C[M,N]: float
// C = alpha * A * B + beta * C，FP32 累加。
#include "sgemm_v4.cuh"
launch_sgemm_v4(A, B, C, M, N, K, alpha, beta, stream);
```

参考用户提供的 `mysgemm_tma_grouped` 的 tile、fragment、XOR 布局与写回映射。
RTX 4070 Ti SUPER 为 SM89，不支持该参考代码使用的 Hopper TMA 路径；V4 使用
16B `cp.async.cg.shared.global` 搬运 A/B，以 `commit_group`、`wait_group 0` 和 CTA barrier
替换 TMA / mbarrier。没有 TMA descriptor，也没有预先转置 A/B 的外部步骤。

## 计算与流水线

- BM=128、BN=128、BK=32；WM=32、WN=64；8 warps / 256 threads。
- 每 warp 负责 32×64 输出；每线程持有 64 个 FP32 累加器。
- A 使用 `ldmatrix.x4`，B 使用 `ldmatrix.x4.trans`，通过
  `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` 计算。
- 每线程搬运连续 8 个 half，即 16B；A/B 同时支持 `cp.async`。
- 默认双缓冲先预加载首块；循环中发起下一块复制，计算当前块，然后等待并同步 CTA，交换 buffer。
  `wait_group` 处理本线程复制完成，`__syncthreads()` 保证跨线程消费与复用 shared 的安全。
- 直接按 MMA accumulator 的 lane 映射，以 `float2` 写回；`beta=0` 时不读原 C。
- 默认编译结果：126 registers/thread，32 KiB shared/CTA，无 spill。
  `__launch_bounds__(256,2)` 给编译器提供每 SM 至少容纳两个 CTA 的资源目标；不保证实际始终驻留两个 CTA。

XOR 索引的单位均为 half：

```cpp
// A: 128×32；按连续 8 个 half 的 16B 向量重排。
a = row*32 + ((col/8) ^ ((row>>1)&3))*8 + col%8;

// B: 两个连续的 32×64 panel；每个 panel 内按行重排 16B 向量。
b = (col/64)*32*64 + row*64
    + (((col%64)/8) ^ (row&7))*8 + col%8;
```

写入和 `ldmatrix` 读取采用相同映射。每个 16B 向量内部顺序保持不变，起始地址仍满足对齐要求。

## 消融入口

通过 `launch_sgemm_v4_variant(variant, ...)` 或 benchmark 的 `--variant` 选择：

| ID | 名称 | 说明 |
|---:|---|---|
| 0 | `wmma_plain` | WMMA、普通布局、同步单缓冲 |
| 1 | `ptx_plain_sync` | 显式 ldmatrix/MMA、普通布局、同步单缓冲 |
| 2 | `ptx_xor_sync` | 显式 ldmatrix/MMA、XOR、同步单缓冲 |
| 3 | `ptx_xor_async_single` | XOR、cp.async、单缓冲 |
| 4 | `ptx_xor_async_double` | XOR、cp.async、双缓冲；默认 API |
| 5 | `ptx_xor_double_group8` | 在双缓冲基础上按 8 个 M tile 分组映射 CTA |

布局效果使用 1→2 的对照；0→1 同时改变 WMMA 的加载与 epilogue，不能归因于单一指令。
grouped CTA 旨在提高相邻 CTA 的 B 复用机会，不保证物理执行顺序或缓存命中率。
4096³ 实测没有优于线性 CTA，因此默认选择 4。group8 编译产生 8B spill stores/loads，保留用于分析。

快路径要求 K>0、K/N 为 8 的倍数，A/B 起始地址 16B 对齐，C 起始地址 8B 对齐。
M 不必为 128 的倍数；尾部 tile 使用 mask 和零填充。
WMMA 基线另要求 M/N 为 128 的倍数、K 为 32 的倍数。
其他非负尺寸、K=0 或不满足对齐条件时走 16×16 SIMT fallback；M/N=0 直接返回。
fallback 的职责是正确处理边界，不纳入 Tensor Core 性能结论。

## 构建、正确性与对照

在仓库根目录的 WSL shell 中执行：

```bash
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --target sgemm_v4_compare -j 3
build/sgemm/sgemm_v4_compare --check
NVIDIA_TF32_OVERRIDE=0 build/sgemm/sgemm_v4_compare --size 4096 --rounds 9 --repeat 10
```

`--check` 执行 16 种配置 × 6 种实现的 96 组 CPU FP64 reference 全输出校验，
覆盖量化后 FP16 输入、非整块、非零 beta、beta=0 时 C 含 NaN、未对齐地址和 K=0；另测空输出。
benchmark 对每种实现做完整输出 cuBLAS 对照，并独立用 CPU FP64 校验 128 个 cuBLAS 输出点。
memcheck、racecheck、synccheck 均已通过。

cuBLAS 使用 `cublasGemmEx`，A/B 为 `CUDA_R_16F`，C 为 `CUDA_R_32F`，
`CUBLAS_COMPUTE_32F`、`CUBLAS_GEMM_DEFAULT`，设置
`CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`。
这是默认算法的同精度 Tensor Core 对照；没有进行 cuBLASLt 算法搜索。
`NVIDIA_TF32_OVERRIDE=0` 不会关闭 FP16 Tensor Core。

先在 Windows 管理员 PowerShell 中锁频：

```powershell
nvidia-smi -lgc 2100,2100
```

再在 WSL 运行完整六种尺寸的对照（输出目录必须尚不存在）：

```bash
python3 sgemm/run_compare_v4.py --clock-mhz 2100 --output /new/output/path
```

脚本记录源码/二进制哈希、独立进程、原始 CUDA Event 采样与 GPU 时钟。
只计 GEMM，FP16 数据生成、H2D、校验和预热不计时；输入驻留，重复运行，不显式清空缓存。

2026-11-06，RTX 4070 Ti SUPER，Graphics/SM 2100 MHz，4096³：

| 实现 | ms | TFLOPS | cuBLAS 吞吐比例 |
|---|---:|---:|---:|
| cuBLAS FP16→FP32 | 2.088 | 65.83 | 100% |
| WMMA 基线 | 3.583 | 38.36 | 58.26% |
| PTX 普通布局 | 3.577 | 38.42 | 58.36% |
| XOR，同步单缓冲 | 2.423 | 56.72 | 86.15% |
| XOR，异步单缓冲 | 2.386 | 57.60 | 87.49% |
| **XOR，异步双缓冲** | **2.242** | **61.29** | **93.10%** |
| grouped CTA 双缓冲 | 2.273 | 60.46 | 91.83% |

默认 V4 相对 WMMA 基线加速 1.60×。8192³ 达到 63.10 TFLOPS、cuBLAS 的 97.16%。
2048³ 中同步 XOR 比异步双缓冲更快，完整尺寸结果见 [实验报告](results/20261106_v4/report.md)。
V3 是 FP32 输入，不能直接用 V4/V3 的耗时比例声称同精度优化收益。

[性能分析面试材料](results/20261106_v4/interview_analysis.md) 包含 bank conflict 地址推导、
源码级 wavefront、资源限制、SchedulerStats 与 cuBLAS 动态指令对照。

## NCU

```bash
/opt/nvidia/nsight-compute/2025.1.1/ncu \
  --profile-from-start off --clock-control none \
  --section SpeedOfLight --section Occupancy --section WarpStateStats \
  --section MemoryWorkloadAnalysis_Tables -o /path/to/v4_report \
  build/sgemm/sgemm_v4_compare --profile --impl 3 --variant 4 --size 4096
```

`--impl 0` 为 cuBLAS，`--impl 3` 为 V4；profile 范围内只运行一次指定实现。
完整七实现的采集脚本及报告在 `results/20261106_v4/`，NCU replay duration 不用于计算上表吞吐比例。
测试结束可在管理员 PowerShell 执行 `nvidia-smi -rgc` 恢复默认频率。

参考：[PTX ISA](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html)、
[cuBLAS 12.8](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html)。
