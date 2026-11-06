# CUDA Kernels from Scratch

基于 CUDA C++ 与 PTX 的算子实现。重点项目是 **GEMM**（FP32 CUDA Core 与混合精度 Tensor Core）和 **FlashAttention-2 Forward**。每个实验同时给出计算形状、精度、基线和硬件计数，便于从实现定位到性能变化。

## 构建与复测

本机为 RTX 4070 Ti SUPER（SM89）、CUDA Toolkit 12.8、Linux/WSL。CMake 以 SM89 为目标。cuBLAS 随 Toolkit 安装。官方 Attention 对照另外使用 PyTorch、Triton 与 flash-attn 2.8.3.post1。

```bash
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
build/sgemm/sgemm_compare --check
build/sgemm/sgemm_v4_compare --check
build/flashattention2/flashattention2_check
```

**NCU 口径**　Nsight Compute 2025.1.1，外部锁频 2100 MHz，--clock-control none，默认 replay/cache control。每个选择器独立进程，只采一个目标 kernel。采集前后读取 SM 频率；每个版本的 kernel 名称、输入参数、读数和原始指标单位均保存在 [profiling/results](profiling/results)。表内 duration 是 NCU 单次运行时间。**吞吐对照**另用未挂 profiler 的 CUDA Event 中位数，两者的缓存和执行条件不同。

```bash
# Windows 管理员 PowerShell 锁频：nvidia-smi -lgc 2100,2100
# 回到 WSL；NCU 也可设为 ncu 可执行文件的绝对路径
export NCU=$(command -v ncu)
for group in gemm fa2 basic; do
  python3 profiling/collect.py --group "$group" --output build/my-profiles
done
# 结束后可用管理员 PowerShell 执行 nvidia-smi -rgc
```

## GEMM

连续 row-major A、B、C，计时形状 M=N=K=4096，alpha=1、beta=0。FP32 组对照严格 FP32 cublasSgemm，FP16 组对照 cublasGemmEx 默认算法，FP16 输入、FP32 累加与输出。两个精度组分别比较。

**CUDA Core**　[V1](sgemm/sgemm_v1.cu) 使用 128×128×8 CTA、8×8 线程输出 tile、float4 读写与双缓冲。[V2](sgemm/sgemm_v2.cu) 显式划分 64×64 warp tile，128 个线程承担相同 CTA 输出，每线程持有更多累加器，采用单级 shared。[V3](sgemm/sgemm_v3.cu) 改为 128×128×16 CTA、8 个 warp；A 在寄存器预取后转置写入 XOR 布局，沿行向量读取，B 可用 16B cp.async。保留同步单缓冲、异步单缓冲与异步双缓冲三种消融，默认双缓冲。

**Tensor Core**　[V4](sgemm/sgemm_v4.cu) 固定 128×128×32 CTA 与 32×64 warp tile。WMMA 基线使用普通 shared 布局；显式 PTX 版使用 ldmatrix 加载 fragment，以 mma.sync.m16n8k16 累加并用 float2 写回。依次比较普通布局、A/B XOR、16B cp.async 单缓冲、交叠搬运与计算的双缓冲，以及沿 M 分组的 group8 CTA 映射。

NCU 表中 **冲突**为全 kernel 的 shared load bank conflict 计数，**Short / Long** 为 WarpStateStats 中按发射指令归一化的平均 Scoreboard 等待，**Occ** 为 achieved occupancy。

| 版本 / 选择器 | NCU ms | 寄存器/线程 | Occ % | 冲突 | Short | Long |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 6.949 | 128 | 32.7 | 268,435,456 | 1.04 | 0.01 |
| v2 | 10.810 | 206 | 16.4 | 0 | 0.21 | 1.83 |
| v3 sync | 7.277 | 125 | 32.5 | 0 | 0.79 | 1.03 |
| v3 async | 7.145 | 122 | 32.5 | 0 | 0.84 | 0.73 |
| v3 double | 6.598 | 127 | 32.7 | 67,625 | 0.71 | 0.04 |
| cublas fp32 | 5.729 | 202 | 16.7 | 0 | 0.04 | 0.05 |
| v4 wmma | 3.503 | 124 | 32.6 | 285,212,672 | 8.36 | 11.52 |
| v4 plain | 3.488 | 127 | 32.6 | 285,212,672 | 5.71 | 11.02 |
| v4 xor | 2.377 | 126 | 32.5 | 0 | 0.19 | 7.62 |
| v4 async | 2.294 | 127 | 32.3 | 0 | 0.12 | 1.62 |
| v4 double | 2.522 | 126 | 32.3 | 78,500 | 0.16 | 0.14 |
| v4 group8 | 2.182 | 128 | 32.4 | 47,416 | 0.26 | 0.27 |
| cublas half | 2.385 | 234 | 16.3 | 45,713 | 0.05 | 0.07 |

**瓶颈分析**　V1 的 shared load 冲突为 2.68 亿；V2 虽消除了这项冲突，但每线程使用 206 个寄存器，实际 occupancy 只有约 16%，运行更慢。保持 V3 的布局，异步单缓冲到双缓冲时 Long Scoreboard 下降，说明产生独立计算后等待更容易隐藏。V4 中仅切换 PTX 布局，冲突从 2.85 亿降至 0，Short Scoreboard 同时显著下降。双缓冲继续降低 Long Scoreboard。group8 的表现随规模变化，单次 NCU duration 与稳态排名可能不同；默认配置根据多轮事件计时选择。

**独立吞吐对照**　4096³ 下 V3 为 **21.09 TFLOPS**，对应 FP32 cuBLAS 的 **85.78%**。V4 线性双缓冲为 **61.29 TFLOPS**，对应同精度 cuBLAS 的 **93.10%**，8192³ 达 **97.16%**。实测计时、尾块约束、正确性与测试命令分别见 [FP32 报告](sgemm/results/20261106/report.md)、[Tensor Core 报告](sgemm/results/20261106_v4/report.md) 和 [逐指令分析](sgemm/results/20261106_v4/interview_analysis.md)。

## FlashAttention-2 Forward

单头、非 causal、无 dropout，B=H=1、D=64，缩放因子 1/8。分块计算 QK、online softmax 与 PV，避免将完整 N×N 注意力矩阵写回显存。下表 N=16384；V2b 及以后使用预转置的 Kt，NCU 只覆盖 attention 核心。

| 版本 | 输入 / 输出 | 相对前一设计的改动 |
|---|---|---|
| [V0](flashattention2/flashattention2_v0.cu)、[V0 FP16](flashattention2/flashattention2_v0_fp16.cu) | FP32 / FP32 或 FP16 / FP16 | SIMT 基线，Br=4、Bc=64，每 warp 处理一行 Q。FP16 版本将 tile 保存在 half shared 中 |
| [BR64 FP32](flashattention2/flashattention2_v0_br64.cu)、[BR64 FP16](flashattention2/flashattention2_v0_br64_fp16.cu) | 同输入精度 | 仍用 SIMT，提高到 64 行 Q，每 warp 维护 16 行输出与在线归一化状态；属于独立的 tile 大小实验 |
| [V1](flashattention2/flashattention2_v1.cu)、[V1 FP16](flashattention2/flashattention2_v1_fp16.cu) | FP32 / FP32、FP16 / FP16、FP16 / FP32 | 64×64 tile，引入 ldmatrix 和 FP16 MMA，FP32 累加。FP32 接口在载入时将操作数转成 half；out32 选择器保留 FP32 写回 |
| [V2a](flashattention2/flashattention2_v2_A.cu)、[V2a FP16](flashattention2/flashattention2_v2a_fp16.cu) | FP32 / FP32 或 FP16 / FP16 | 按 ldmatrix 地址模式重排 Q/K/V shared tile，以 XOR 减少 bank conflict，仍为同步供数 |
| [V2b / V2c](flashattention2/flashattention2_v2_BC.cu) | FP16 / FP16 | Kt 为 [64,N]。V2b 引入 16B cp.async 单缓冲；V2c 进一步使用 K/V 双缓冲，在计算当前 tile 时预取下一 tile |
| [V2c BC32/Qreg](flashattention2/flashattention2_v2c_bc32_qreg.cu) | FP16 / FP16 | Bc 由 64 缩至 32，Q fragment 跨 KV 循环驻留寄存器，并分时复用 Q/K shared 空间；静态 shared 从 40 KiB 降至 16 KiB |

| 版本 / 选择器 | NCU ms | 寄存器/线程 | Occ % | 冲突 | Short | Long |
|---|---:|---:|---:|---:|---:|---:|
| v0 | 66.711 | 39 | 16.7 | 4,486 | 0.54 | 4.87 |
| v0_br64 | 19.163 | 255 | 16.2 | 3,526,507 | 2.55 | 3.55 |
| v1 | 5.753 | 72 | 32.4 | 293,606,723 | 9.45 | 7.85 |
| v2a | 3.225 | 96 | 32.3 | 13,325 | 0.10 | 7.70 |
| v0_fp16 | 38.578 | 38 | 40.6 | 2,542 | 0.51 | 3.99 |
| v0_br64_fp16 | 47.491 | 255 | 16.4 | 0 | 2.62 | 11.45 |
| v1_fp16 | 5.242 | 72 | 32.4 | 293,603,778 | 9.72 | 8.18 |
| v1_fp16_out32 | 5.236 | 72 | 32.4 | 293,606,423 | 9.73 | 8.15 |
| v2a_fp16 | 3.202 | 96 | 32.3 | 8,609 | 0.08 | 7.67 |
| v2b | 1.323 | 96 | 26.6 | 25,483 | 0.12 | 0.41 |
| v2c | 1.756 | 96 | 15.4 | 5,361 | 0.11 | 0.10 |
| v2c_bc32_qreg | 1.299 | 96 | 27.0 | 5,935 | 0.68 | 0.09 |

**瓶颈分析**　BR64 的重用伴随每线程 255 个寄存器，FP16 版本还出现编译器 spill，增大 tile 在此形状下反而慢于 Br=4。V1 FP16 在 shared 加载上出现约 2.94 亿次冲突；V2a FP16 同时降低了冲突和 Short Scoreboard。V2a→V2b 同时引入 K 预转置和异步搬运，Long Scoreboard 大幅下降。V2c 的双缓冲降低等待，但占用更多 shared，实际 occupancy 降至约 15%；BC32/Qreg 回升至约 27%。仅从这组联合改动无法拆分 Bc 和 Q 常驻各自的收益。

**与官方 FA2 的独立对照**　当前源码重新构建，在相同 FP16 输入与输出、相同 CUDA Graph + Events 协议下，与 flash-attn 2.8.3.post1 的 flash_attn_func 比较。七轮交错执行取中位数，所有路径先对照分块 FP32 reference。含转置列从普通 K 开始，直接计时转置与 attention 两步。

| N | 路径 | 官方 ms | 核心 ms | 官方吞吐占比 | 含 K 转置 ms | 含转置占比 |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | v2b | 0.0865 | 0.1320 | 65.55% | 0.1404 | 61.62% |
| 4096 | v2c | 0.0865 | 0.1227 | 70.49% | 0.1306 | 66.26% |
| 4096 | v2c_bc32_qreg | 0.0865 | 0.1488 | 58.14% | 0.1558 | 55.54% |
| 16384 | v2b | 1.1963 | 1.3892 | 86.12% | 1.4201 | 84.24% |
| 16384 | v2c | 1.1963 | 1.4918 | 80.19% | 1.5150 | 78.97% |
| 16384 | v2c_bc32_qreg | 1.1963 | 1.3736 | 87.09% | 1.3882 | 86.17% |

N=4096 时原 V2c 的事件中位数更低，N=16384 时 BC32/Qreg 更快。见 [本轮全部 FP16 路径、误差和七轮样本](profiling/results/fa2_official.json)、[FA2 校验](profiling/results/fa2_check.txt) 与 [历史实验及计时细节](flashattention2/BENCHMARK.md)。

## 其他算子

| 算子与输入 | 版本特性及瓶颈观察 |
|---|---|
| [Reduction](reduction)，2²⁴ 个 float | V1 每线程合并两个值，warp shuffle 后合并各 warp 部分和，完整求和由双工作区迭代完成。所采为第一层 kernel，DRAM 吞吐接近峰值 |
| [Softmax](softmax)，2048×1024 FP32 | V1 将指数结果暂存 global；reg_vec 以 float4 保存在寄存器中，shared_vec 将行缓存放入 shared。三版同形状比较寄存器路径更快。实际分发对 4 对齐且列数 ≤1024 选寄存器，1025–2048 的对齐行选 shared，其余回退 V1 |
| [RMSNorm](rmsnorm)，2048 行 | V1 对每行做平方和归约与缩放；列数为 4 的倍数时走 float4。1024 列向量路径与 1023 列标量路径分别采集，DRAM 吞吐较高，尺寸也有差异 |
| [Histogram](histogram)，2²⁴ 个 uint8、256 bins | V1 逐元素 global atomicAdd；V2 在 CTA 内累计 shared 直方图，再按 bin 合并到 global。固定 0–255 循环输入下，减少全局原子更新后时间下降，输入分布会影响竞争 |
| [Transpose](transpose)，2048² FP32 | 两版采用 32×32 shared tile 和 32×8 线程。V2 增加一列 padding，本次 shared load 冲突从 406 万降至 0。入口要求整 tile |
| [Conv2d](conv2d)，N=1、Cin=32、Cout=64、64²、3×3、stride=1、pad=1 | V2 逐输出计算隐式 GEMM 索引，V3 采用 64×64×8 分块和 4×4 寄存器 tile，复用输入与权重。本例 V3 在较低 occupancy 下仍显著更快 |

DRAM / SM 百分比取 NCU 相对持续峰值的 elapsed 指标。

| 版本 / 选择器 | NCU μs | 寄存器/线程 | Occ % | 冲突 | DRAM % | SM % |
|---|---:|---:|---:|---:|---:|---:|
| softmax_v1 | 22.496 | 23 | 92.2 | 768 | 57.15 | 41.52 |
| softmax_reg | 15.872 | 25 | 89.7 | 436 | 81.14 | 30.62 |
| softmax_shared | 18.720 | 28 | 89.3 | 0 | 83.79 | 32.28 |
| rms_vec | 17.600 | 24 | 87.6 | 305 | 75.22 | 18.74 |
| rms_scalar | 20.384 | 24 | 90.8 | 402 | 62.98 | 28.94 |
| hist_v1 | 709.248 | 16 | 96.7 | 0 | 4.39 | 3.20 |
| hist_v2 | 181.760 | 16 | 96.4 | 1,477 | 14.09 | 12.49 |
| trans_v1 | 48.384 | 16 | 92.3 | 4,063,232 | 54.17 | 19.32 |
| trans_v2 | 32.224 | 16 | 93.8 | 0 | 81.48 | 29.01 |
| conv_v2 | 326.496 | 39 | 87.1 | 0 | 0.28 | 76.21 |
| conv_v3 | 63.936 | 48 | 16.6 | 0 | 1.68 | 26.63 |
| reduction_v1 | 123.264 | 16 | 89.9 | 6 | 94.20 | 30.31 |

完整采集命令的选择器与指标名见 [采集脚本](profiling/collect.py)。二进制 NCU 报告生成在本地输出目录，仓库中保留可直接审阅的 JSON。
