# CUDA Core FP32 SGEMM 优化与锁频对照

2026-11-06，RTX 4070 Ti SUPER，Graphics/SM 2100 MHz，row-major FP32 A/B/C，FP32 FMA 累加。

4096³：新版 **6.516941 ms / 21.089 TFLOP/s**，达到同批严格 FP32 cuBLAS 吞吐的 **85.78%**。
相对仓库较快的旧版 v1，吞吐提升 **5.16%**；相对旧版 v2 提升 **67.06%**。

朋友简历 CUDA Core 的 4096³ 结果为 RTX 5080 上 26.23 TFLOP/s、同批严格 FP32 cuBLAS 的 84.0%。本次比例处于相近量级；不同 GPU、软件版本和测量协议不能据此比较优化水平高低。

## 实现与选择

- 保留 sgemm_v1.cu、sgemm_v2.cu；新增 sgemm_v3.cu 和 launch_sgemm_v3 接口。
- CTA tile 128×128×16；8 个 warp，每 warp 32×64，每线程 8×8=64 个 FP32 累加器。原 v2 为 4 warp、每线程 128 个累加器。
- 沿用已有分块、warp/register tiling 和向量化思路；A 用 float4 从 global 预取到寄存器，随后转置写入 shared，并使用 XOR 布局；shared→register 采用 float4。
- B 用 16B cp.async.cg.shared.global 直接写入下一阶段 shared；A/B 双缓冲，把下一 tile 的搬运与当前 tile 的 FFMA 重叠。A 没有全局预转置步骤，计时包含整个 GEMM kernel。
- A 保留寄存器中转，以保住 shared 读取效率；并非 A/B 全部使用 cp.async。每次换阶段使用 wait_group 0 和 CTA barrier，保证复制完成及跨线程可见。
- K 内循环展开 2 次，以控制活跃值和指令占用；展开因子的探索记录保留在本目录。探索阶段的 2400 MHz 请求存在降频风险，最终未对展开因子单独做严格定频消融，因此不把正式收益单独归因于 I-cache 或展开因子。
- 最终生产代码仅保留三种可复现消融：同步单缓冲、B 异步单缓冲、B 异步双缓冲。早期全异步 row-major A 的实现、较差配置和调参数据保存在本目录。
- 快路径要求 M/N 为 128 的倍数、K 为 16 的正倍数，以及 A/B/C 起始地址 16B 对齐；其余输入走带边界检查的 16×16 tiled fallback。fallback 保证覆盖范围，不宣称已做性能优化。

## 严格 FP32 与测量协议

- 对照为 cublasSgemm，设置并回读 CUBLAS_PEDANTIC_MATH；进程设置 NVIDIA_TF32_OVERRIDE=0。CUDA 编译未启用 --use_fast_math。
- row-major C=A×B，通过 cuBLAS 的列主序 Cᵀ=Bᵀ×Aᵀ 等价调用，不计入任何额外布局转换。alpha=1、beta=0。
- A/B 使用固定种子的 [-0.5,0.5] 随机 FP32 数据，不使用全 1 输入；所有实现使用同一批 A/B。
- 计时前，每个实现做全矩阵数值对照，并将 cuBLAS 的 128 个输出位置与 CPU FP64 dot 对照。H2D、D2H 和校验均不计时。
- 每个实现先预热 10 次；每轮额外预热 3 次。CUDA Event 记录默认执行流中多次 kernel 的平均耗时，轮内轮转并反转实现顺序。
- 4096³ 有 5 个独立进程，每进程每实现 9 条观测，每条平均 10 次执行，共 45 条观测/实现。其余尺寸各 3 个独立进程。
- 输入常驻显存，反复使用同一批矩阵，未主动冲刷缓存；这是 warm steady-state 对照。性能采样时不运行其他 GPU 检查或 profiler。

## 性能矩阵

| M×N×K | cuBLAS TFLOP/s | 原 v1 | 原 v2 | 新 v3 | v3/cuBLAS | v3/v1 |
|---|---:|---:|---:|---:|---:|---:|
| 1024x1024x1024 | 18.625 | 17.549 | 10.106 | 17.906 | 96.14% | 1.020× |
| 2048x2048x2048 | 23.443 | 19.636 | 14.105 | 20.746 | 88.49% | 1.057× |
| 4096x4096x4096 | 24.585 | 20.056 | 12.624 | 21.089 | 85.78% | 1.052× |
| 8192x8192x8192 | 25.448 | 20.549 | 12.611 | 21.576 | 84.79% | 1.050× |
| 1024x4096x2048 | 23.461 | 19.647 | 13.976 | 20.707 | 88.26% | 1.054× |
| 4096x1024x2048 | 23.460 | 19.645 | 12.951 | 20.718 | 88.31% | 1.055× |

4096³ 消融，布局、CTA/warp/thread tile 相同：

| 实现 | 中位数 ms | TFLOP/s | 观测范围 ms |
|---|---:|---:|---:|
| v3_sync_single | 7.194317 | 19.104 | 7.080346–7.385808 |
| v3_async_single | 7.088742 | 19.388 | 6.991456–7.274803 |
| v3_async_double | 6.516941 | 21.089 | 6.435430–6.654771 |

只引入异步搬运、紧接着等待的收益有限；流水线还需要独立计算来覆盖搬运时间。单缓冲/双缓冲消融同时改变 A 的寄存器预取和 B 的异步重叠，不能把全部收益归到 B 的一条 cp.async 指令。

## Nsight Compute 与 Roofline

4096³、每个实现单独采集；--clock-control none 由用户锁频，包含 Roofline/Occupancy/WarpStateStats/MemoryWorkloadAnalysis。NCU 默认缓存控制和 replay 会改变执行条件，以下分析计数器，不把 NCU 单次 duration 与上面的事件中位数混用。

| 指标 | v1 | v2 | v3 |
|---|---:|---:|---:|
| 寄存器/线程 | 128 | 206 | 127 |
| 静态 shared，KB（十进制） | 16.384 | 8.192 | 32.768 |
| 实测 occupancy % | 32.6754 | 16.404 | 32.659 |
| shared load bank conflicts | 2.68435e+08 | 0 | 74530 |
| shared store bank conflicts | 1.67772e+07 | 1.67772e+07 | 796415 |
| Long Scoreboard，NCU per-issue ratio | 0.005309 | 1.81343 | 0.035784 |
| Short Scoreboard，NCU per-issue ratio | 1.04154 | 0.204385 | 0.713924 |

- v2：高寄存器占用与同步供数路径对应较低驻留率、较高 Long Scoreboard；新版同时调整了线程分工和流水线，不能将该差异只归因于 cp.async。
- v1：已经有寄存器预取和双缓冲，Long Scoreboard 本身很低；新版主要改善其 shared 访问布局，load conflicts 从 268,435,456 降到 74,530（约 -99.97%），Short Scoreboard 也下降。
- v3 的实际 DRAM 算术强度按 2MNK/(DRAM read+write) 估算为 **121.17 FLOP/byte**；DRAM 流量约 1.134 GB，FMA pipe active 为 61.60%。这与只按最少 A/B/C 字节算的理想强度不同，L2 复用和重复 tile 读取必须区分。
- 原 v1/v2 与新版的 BM/BN 都是 128，忽略 C 写回时的 CTA global-load 理论强度均为 32 FLOP/byte。本次收益主要来自供数、shared 访问与指令/资源安排，不能归因为又提高了这一层的理论复用倍数。
- cuBLAS 实际 kernel：`void Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>(Params)`，Tensor pipe active 为 0%。自写二进制 SASS 检查 Tensor Core 指令数为 0，计算使用 FFMA。

## 正确性与边界

- 12 组形状/地址/alpha-beta 设置 × 3 个消融，共 36 次完整矩阵 FP64 CPU reference 对照；另检查 M=0、N=0。
- 覆盖奇数尺寸、K 尾部、K=0、非 16B 对齐、alpha=0、非零 beta；beta=0 时把旧 C 置 NaN，检查不会依赖旧值；输出前后 guard 保持不变。
- 小矩阵门槛 atol=1e-4、rtol=2e-5；大矩阵全输出对照严格 FP32 cuBLAS，atol=2e-3、rtol=2e-4，另做 128 点 FP64 独立校验。
- Compute Sanitizer memcheck：0 errors；racecheck：0 errors、0 warnings；synccheck：0 errors。
- GPU 时钟遥测 164 条，Graphics/SM 全部为 2100 MHz；温度 43.7–66.7 °C。显存未单独锁频。
- 测量前后源文件和可执行文件 SHA256 已复核相同。

## 文件与复现

- `locked2100/summary.json`、`samples.csv`、各进程 jsonl：原始结果及汇总。
- `locked2100/environment.json`：源码/二进制 SHA256、命令、精度和计时协议。
- `locked2100/profiles/*_final_n4096.ncu-rep`、`*_final_metrics.csv`、`ncu_summary.json`：四个实现的分析数据。
- 最初请求锁 2400 MHz 的 `locked2400/` 数据保留作审计；遥测实际 2175–2400 MHz，未满足固定频率要求，不作为本文最终性能结论。其原始采样脚本另存为 `run_compare_requested2400.py.txt`。
- `v3_initial.cu.txt`、`v3_hybrid_loop2.cu.txt`、`v3_hybrid_loop4.cu.txt`、`tune*.jsonl`：探索过程与较差配置，未覆盖删除。

```bash
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sgemm_compare -j 3
NVIDIA_TF32_OVERRIDE=0 build/sgemm/sgemm_compare --check
NVIDIA_TF32_OVERRIDE=0 build/sgemm/sgemm_compare --size 4096 --rounds 9 --repeat 10
/home/smh/ai_infra/nano-vllm/.venv/bin/python sgemm/run_compare.py --clock-mhz 2100 --output /new/output/path
```

正式采样前在 Windows 管理员 PowerShell 使用 `nvidia-smi -lgc 2100,2100`；结束后 `nvidia-smi -rgc` 恢复默认。

## 官方资料

- [cuBLAS 12.8：math mode 与精度控制](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html)。
- [PTX ISA 12.8：cp.async / wait_group](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html)。
