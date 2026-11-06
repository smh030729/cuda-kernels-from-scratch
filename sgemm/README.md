# FP32 CUDA Core SGEMM

`sgemm_v1.cu` / `sgemm_v2.cu` 保留原实现；新增 `sgemm_v3.cu`。
接口均为连续 row-major 矩阵，计算 `C = alpha*A*B + beta*C`，输入、累加和输出均为 FP32。

```cpp
launch_sgemm_v3(A, B, C, M, N, K, alpha, beta, stream);
```

## V3 的实现

- CTA tile：128×128×16；256 threads，8 个 warp；每 warp 32×64，每线程输出 8×8。
- A：`float4` global→register 预取，转置写入 XOR shared 布局，保留 shared→register 的 `float4` 读取。
- B：16B `cp.async.cg.shared.global` 直接搬入下一阶段 shared。
- A/B 双缓冲：先发起下一 tile 搬运，计算当前 tile，再提交 A 的 shared stores、等待 B 复制并同步 CTA，交换阶段。
- 内层 K 循环展开 2 次，FP32 `fmaf` 累加。无 Tensor Core、FP16 或 TF32 路径。
- 快路径要求 M/N 为 128 的倍数、K 为 16 的正倍数，A/B/C 起始地址 16B 对齐。
  其他非负尺寸（包括 K=0）及非 16B 对齐地址使用 16×16 tiled fallback；M/N=0 返回。

A 的 shared 索引为 `k*128 + (row ^ (((k/4)&3)*8))`。
它给转置写入的 K 向量组错开 bank，同时保留沿 M 连续四个元素的对齐读取。
这不是额外的 global 转置；完整 GEMM kernel 的读写和计算全部计时。

## 构建与校验

```bash
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sgemm_compare -j 3
NVIDIA_TF32_OVERRIDE=0 build/sgemm/sgemm_compare --check
```

`sgemm_compare` 是新增对照程序。原 `sgemm_bench` / `sgemm_profile` 继续覆盖原 v1/v2。
36 组完整矩阵 CPU FP64 reference 对照覆盖三种 V3 消融、非整块、非对齐、alpha/beta、K=0，另测 M/N=0。
memcheck、racecheck、synccheck 均为零错误。

## 锁频对照

先在 Windows 管理员 PowerShell 中执行：

```powershell
nvidia-smi -lgc 2100,2100
```

再在仓库根目录执行：

```bash
NVIDIA_TF32_OVERRIDE=0 build/sgemm/sgemm_compare --size 4096 --rounds 9 --repeat 10
/home/smh/ai_infra/nano-vllm/.venv/bin/python sgemm/run_compare.py \
  --clock-mhz 2100 --output /new/output/path
```

比较 cublasSgemm（`CUBLAS_PEDANTIC_MATH`）、v1、v2，以及布局和线程划分相同的三种消融：
`v3_sync_single`、`v3_async_single`、`v3_async_double`。默认 V3 API 对应最后一种。
程序使用随机输入、逐实现正确性检查、充分预热、轮转/反序运行与 CUDA Event 计时。
Python 脚本记录源码/二进制哈希、原始采样及 GPU 时钟；观测频率偏离目标时明确报错。

2026-11-06 / RTX 4070 Ti SUPER / Graphics 与 SM 2100 MHz / 4096³：

| 实现 | ms | TFLOP/s | 严格 FP32 cuBLAS 吞吐比例 |
|---|---:|---:|---:|
| cuBLAS | 5.590 | 24.585 | 100% |
| 原 v1 | 6.853 | 20.056 | 81.58% |
| 原 v2 | 10.887 | 12.624 | 51.35% |
| 新 v3 | 6.517 | 21.089 | **85.78%** |

相对较快旧版 v1，V3 吞吐提升 **5.16%**；相对 v2 提升 **67.06%**。
最终结论以 2100 MHz 结果为准；最初请求锁 2400 MHz 的实验出现降频，保留记录但不混入最终结论。

[完整报告、NCU 分析和原始数据](results/20261106/report.md)

## 单 kernel NCU

```bash
/opt/nvidia/nsight-compute/2025.1.1/ncu \
  --profile-from-start off --clock-control none \
  --section SpeedOfLight --section Occupancy --section WarpStateStats \
  -o /path/to/v3_report \
  build/sgemm/sgemm_compare --profile --impl 3 --variant 2 --size 4096
```

`--impl`：0=cuBLAS，1=v1，2=v2，3=v3；`--variant`：0=同步单缓冲，1=异步单缓冲，2=异步双缓冲。
profile 模式仍先预热、校验，`cudaProfilerStart/Stop` 之间只执行一次指定实现。

测试后在管理员 PowerShell 执行 `nvidia-smi -rgc` 恢复默认频率。
