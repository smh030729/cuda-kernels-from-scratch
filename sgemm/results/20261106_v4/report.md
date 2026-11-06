# Tensor Core V4：2100 MHz cuBLAS 对照

2026-11-06，RTX 4070 Ti SUPER / SM89，CUDA 12.8.93，cuBLAS 12.8.5（API version 120805）。
A/B 为 FP16，FP32 累加及输出，连续 row-major，alpha=1、beta=0。

**4096³：V4 为 2.242 ms / 61.29 TFLOPS，达到同精度默认 cuBLAS 的 93.10%，相对 WMMA 基线加速 1.60×。**
**8192³：V4 为 17.424 ms / 63.10 TFLOPS，达到 cuBLAS 的 97.16%。**

## 实现与对照范围

参考用户提供的 `mysgemm_tma_grouped`，保留显式 ldmatrix/MMA、XOR 布局、float2 epilogue、
分块和 grouped CTA 思路。SM89 使用 A/B 的 16B cp.async + wait_group + CTA barrier；没有 TMA。
默认为 128×128×32 CTA、32×64 warp tile、256 threads、双缓冲、线性 CTA 映射。
源码和复现入口见 [V4 说明](../../README_v4.md)。

cuBLAS 为 cublasGemmEx，FP16 输入、FP32 compute/output，CUBLAS_GEMM_DEFAULT，
禁用 reduced-precision reduction；并非 V3 的纯 FP32 cublasSgemm，也未做 cuBLASLt 搜索。
所有实现使用相同量化后的 FP16 输入；无需预先转置或转换设备矩阵。

## 4096³ 消融

| 实现 | 中位数 ms | TFLOPS | cuBLAS 吞吐比例 |
|---|---:|---:|---:|
| cuBLAS | 2.087629 | 65.835 | 100.00% |
| WMMA 普通布局 | 3.583078 | 38.358 | 58.26% |
| PTX 普通布局，同步单缓冲 | 3.577241 | 38.420 | 58.36% |
| PTX XOR，同步单缓冲 | 2.423309 | 56.715 | 86.15% |
| PTX XOR，异步单缓冲 | 2.386125 | 57.599 | 87.49% |
| **PTX XOR，异步双缓冲（默认）** | **2.242355** | **61.292** | **93.10%** |
| PTX XOR，双缓冲 + group8 | 2.273382 | 60.456 | 91.83% |

相同 PTX 框架下，普通布局→XOR 同步版加速约 1.48×；异步单缓冲→双缓冲约 1.064×。
WMMA→PTX 还涉及 shared epilogue 与加载方式变化，不能将二者差异全部归因于 bank conflict。
group8 在此尺寸未取得更高吞吐，默认保持线性映射。

## 其他尺寸（始终使用默认 variant 4）

| M×N×K | cuBLAS ms | V4 ms | V4 TFLOPS | cuBLAS 吞吐比例 |
|---|---:|---:|---:|---:|
| 1024×1024×1024 | 0.037929 | 0.052756 | 40.706 | 71.89% |
| 2048×2048×2048 | 0.279296 | 0.298501 | 57.554 | 93.57% |
| 4096×4096×4096 | 2.087629 | 2.242355 | 61.292 | 93.10% |
| 8192×8192×8192 | 16.929279 | 17.424383 | 63.102 | 97.16% |
| 1024×4096×2048 | 0.279194 | 0.298650 | 57.525 | 93.49% |
| 4096×1024×2048 | 0.279347 | 0.298342 | 57.584 | 93.63% |

没有按尺寸挑最快 variant 后再拼接成绩。2048³ 的同步 XOR 版本实测 0.287846 ms，优于默认双缓冲；
1024³ 的 group8 比默认版更快。可见 cp.async / 双缓冲 / grouped CTA 并非对每个规模都有收益。
所有版本、尺寸与原始范围见 [summary.json](locked2100/summary.json)。

## NCU：4096³

单 kernel，已预热，`--clock-control none` 使用已有手动 2100 MHz 锁频。
NCU 默认 replay/cache 行为不同于事件 benchmark，以下计数用于诊断，吞吐比例采用上面的 CUDA Event 结果。

| 实现 | shared load bank conflicts | Short Scoreboard / issued inst | Long Scoreboard / issued inst | regs/thread | shared/CTA | achieved occupancy |
|---|---:|---:|---:|---:|---:|---:|
| WMMA | 285,212,672 | 8.361 | 11.192 | 124 | 24 KiB | 32.29% |
| PTX 普通布局 | 285,212,672 | 5.681 | 10.813 | 127 | 16 KiB | 32.08% |
| PTX XOR 同步 | 0 | 0.185 | 7.084 | 126 | 16 KiB | 34.37% |
| PTX XOR 异步单缓冲 | 0 | 0.122 | 1.608 | 127 | 16 KiB | 32.37% |
| **V4 异步双缓冲** | **80,115** | **0.162** | **0.139** | **126** | **32 KiB** | **32.32%** |
| group8 双缓冲 | 50,229 | 0.258 | 0.265 | 128 | 32 KiB | 32.37% |
| cuBLAS | 44,826 | 0.052 | 0.068 | 234 | 32 KiB | 16.28% |

Scoreboard 数值来自 `smsp__average_warps_issue_stalled_*_per_issue_active.ratio`，
表示按 issued instruction 归一化的平均等待指标，不是 stall 百分比，也不是单条内存指令的硬件延迟。
NCU raw CSV 的该 ratio 单位行显示 `inst`，表中保留数值并明确归一化口径。

- 布局消融 1→2：shared load conflicts 从 2.85 亿降至 0，Short Scoreboard 从 5.681 降至 0.185。
- 默认双缓冲相对普通布局，实测 shared load conflicts 降低 99.972%；异步版有少量剩余硬件计数，未声称所有情况下绝对零冲突。
- XOR 同步→异步双缓冲，Long Scoreboard 从 7.084 降至 0.139。cp.async 改变依赖链，双缓冲提供复制与计算重叠；不能据此声称 global memory 物理延迟被消除。
- 默认 V4 编译为 126 registers/thread、32 KiB shared，无 spill。group8 编译为 128 registers/thread，产生 8B spill stores 和 8B spill loads；它的性能差异还涉及 CTA 映射，未单独归因于 spill。
- cuBLAS 的 achieved occupancy 更低但吞吐更高，说明 occupancy 本身不是优化目标。
  其采集到的 kernel 为 `ampere_s1688gemm_fp16_128x128_ldg8_stages_32x1_nn`，使用 Tensor Core。

[NCU 指标及单位](locked2100/profiles/selected_metrics.json)，每个实现的 `.ncu-rep`、原始 CSV 同目录。
[SASS](sass.txt) 中确认 `HMMA.16816.F32` 和 `LDGSTS.E.BYPASS.128`；[资源用量](resources.txt) 与编译日志相符。

## 正确性与测量协议

- 96 组 CPU FP64 reference 全输出对照，基于 FP16 量化后的输入计算 reference。
  包含非整块、group8 不完整分组、alpha/beta、K=0、未对齐指针、输出 guard 等场景；另测 M/N=0。
- 六种 benchmark 尺寸中，每实现全输出与同精度 cuBLAS 对照；每进程独立抽查 128 个输出点与 CPU FP64 点积。
- memcheck、racecheck、synccheck 均为 0 errors/hazards。
- 4096³：5 个独立进程，每进程 9 轮，每轮每实现 10 次 kernel 均值，共 45 个轮次样本，取中位数。
  1024³/2048³/两种矩形：3 进程 × 9 轮；8192³：3 进程 × 7 轮。详细 repeat 见 environment.json。
- 先做正确性检查和预热，每轮轮转及反序各实现顺序；CUDA Event 只计重复 GEMM。
  不计数据生成、FP16 转换、H2D、验证；输入驻留，不清缓存。小尺寸包含 launch/提交间隙影响。
- Graphics/SM 时钟每 0.5 秒采样，全部 **92 个观测样本均为 2100 MHz**；这是离散观测结果，不能证明采样间隔内从未波动。
  source/binary SHA256、每进程起止时间、原始采样和完整时钟记录均已保存。
- 4096³ 默认 V4 的 45 个轮次均值范围为 2.217574–2.357760 ms；cuBLAS 为 2.064384–2.183987 ms。
  报告不使用最小值或单次最好结果。

原始证据：[environment.json](locked2100/environment.json)、[telemetry.csv](locked2100/telemetry.csv)、
[samples.csv](locked2100/samples.csv)、[正确性](check_final.jsonl)、
[memcheck](memcheck.log)、[racecheck](racecheck.log)、[synccheck](synccheck.log)。

试测的其他 tile / warp / group 配置及源码快照保留在本目录；正式结果只对应 environment.json 中的最终源码哈希。
朋友简历中的 103.15% 是另一张 RTX 5080 的测量，不能直接迁移到本机；本机可复现结论是上表的 93.10% / 97.16%。

后续增加了相同二进制在 2100 MHz 下的源码级、调度器与指令分析，见
[性能分析面试材料：地址推导、逐指令证据与 12 个问题](interview_analysis.md)。
新增数据单独保存，不覆盖上述事件计时与原 NCU 记录。
