# GEMM 性能分析：面试中的证据与解释

本文基于已保存的 V3/V4 源码、定频事件 benchmark 和 NCU 报告。数值口径见
[V4 实验报告](report.md) 与 [V3 实验报告](../20261106/report.md)。
新增分析不修改 kernel，也不替换原始性能结果。已完成相同二进制、2100 MHz 下的
SchedulerStats / SourceCounters / ComputeWorkloadAnalysis 补采，30 个时钟观测样本全部为 2100 MHz。
补采数据见 [analysis_summary.json](interview_profiles/analysis_summary.json)。

## 1. 你如何定位瓶颈，而不是逐个尝试优化技巧？

可以这样回答：

> 我先固定输入精度、尺寸和时钟，使用独立于 profiler 的 CUDA Event benchmark 确认性能差距。
> 然后用 NCU 看 shared 访问、Scoreboard、资源限制和计算管线，提出具体假设。
> 在 Tensor Core 路径中，我保留同一套显式 ldmatrix/MMA 计算框架，只改变 shared 布局，
> 验证 bank conflicts、Short Scoreboard 和运行时间是否一起改善；再分别比较异步单缓冲和双缓冲。

4096³ 的证据：

| 对照 | CUDA Event 中位数 | 关键变化 |
|---|---:|---|
| PTX 普通布局→XOR 同步 | 3.577→2.423 ms | load bank conflicts 285,212,672→0；Short Scoreboard 5.681→0.185 |
| XOR 同步→异步单缓冲 | 2.423→2.386 ms | Long Scoreboard 7.084→1.608 |
| 异步单缓冲→双缓冲 | 2.386→2.242 ms | Long Scoreboard 1.608→0.139；搬运下一 tile 与当前 tile 计算重叠 |

这些对照支持 shared 布局和流水线有效。异步指令替换还改变指令数量、依赖与 cache 路径，
所以不能把全部收益解释为一种 stall 的下降。

## 2. 2.85 亿 bank conflicts 究竟来自哪里？能从地址算出来吗？

可以。源码的 A/B 索引和 lane 分工允许直接推导，并已用
[枚举脚本](check_ldmatrix_layout.py) 检查所有 warp、kk 和 fragment 位置。
结果保存在 [ldmatrix_layout_check.json](ldmatrix_layout_check.json)。

以每个 `ldmatrix.x4` 的第一个 8×8 half 子矩阵为例，每个供地址的 lane 对应一行连续 8 个 half，
即 16B、4 个 bank word。表中是各行第一个 32-bit word 的 bank 编号：

| 局部行 | A 普通布局 | A XOR | B 普通布局 | B XOR |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 16 | 16 | 0 | 4 |
| 2 | 0 | 4 | 0 | 8 |
| 3 | 16 | 20 | 0 | 12 |
| 4 | 0 | 8 | 0 | 16 |
| 5 | 16 | 24 | 0 | 20 |
| 6 | 0 | 12 | 0 | 24 |
| 7 | 16 | 28 | 0 | 28 |

普通 A 有四行争用同一组 bank，对应 4-way；普通 B 的八行争用同一组 bank，对应 8-way。
XOR 后，八行各占不同的四个 bank，覆盖全部 32 个 bank。
这是行向量地址的分析；`ldmatrix` 的供地址 lane 与最终接收 fragment 元素的 lane 不应混为一谈。

`ldmatrix.x4` 每 warp 读取 4×8×8×2=512B，即便没有冲突，也理想需要 4 个 128B wavefront。
因此按这一访问模型：

- A 普通布局：每条 x4 为 16 个 wavefront，较理想值多 12 个。
- B 普通布局：每条 x4.trans 为 32 个 wavefront，较理想值多 28 个。
- XOR：两者均为理想的 4 个 wavefront。

4096³ 时有 1024 个 CTA、128 个 K tile、每 CTA 8 个 warp；每 tile 两个 kk 步骤，
每步骤每 warp 两条 A x4、四条 B x4.trans。因此动态 warp 指令数为：

```text
A：1024 × 128 × 8 × 2 × 2 = 4,194,304
B：1024 × 128 × 8 × 2 × 4 = 8,388,608

预测额外 wavefront：
4,194,304 × (16−4) + 8,388,608 × (32−4)
= 285,212,672
```

该预测与普通 PTX 版本 NCU 的 aggregate load bank conflict 计数完全一致。
这比只背“冲突降低了多少”更有解释力。

补采的 SourceCounters 还验证了每个操作数的实际服务量：

| 操作数 | 动态 ldmatrix warp 指令 | 普通布局实际 wavefront | 理想 wavefront | XOR 后实际 wavefront |
|---|---:|---:|---:|---:|
| A | 4,194,304 | 67,108,864 | 16,777,216 | 16,777,216 |
| B | 8,388,608 | 268,435,456 | 33,554,432 | 33,554,432 |

这些 SASS 指令分别关联到 `sgemm_v4.cu` 第 109 行和第 119 行的 A/B ldmatrix。
XOR 同步与默认异步双缓冲两种实现都达到上述理想服务量。

## 3. 为什么 XOR 同步版是零冲突，双缓冲版却还有 80,115？

原报告中同步 XOR 的 aggregate 计数为 0；异步双缓冲为 80,115。补采中异步双缓冲该计数为 77,930，
但逐指令 LDSM 的实际和理想 wavefront 均为 50,331,648，extra wavefront 为 0。
因此已确认：剩余 aggregate 计数不能直接解释为这些 ldmatrix 的行布局造成的额外 wavefront。

NVIDIA 文档说明，Memory Tables 的 aggregate 计数还可能包含 shared/L1 不同客户端之间的 bank 争用；
Source 页的逐指令计数可能更低。因此，**cross-client contention 是有依据的解释候选，
本次剩余 aggregate 计数的具体来源仍未完全分解**，不能当成已证实的根因。
参见 [NVIDIA Compute Triage：shared bank conflicts](https://docs.nvidia.com/nsight-compute/ComputeTriage/#symptom-shared-memory-bank-conflicts-are-significant)。

## 4. Short Scoreboard 高，就说明 bank conflict 严重吗？

不能单独这么判断。Short Scoreboard 是相关短延迟执行路径上的依赖等待，shared load 是本项目中的重要来源，
它不直接等价于 bank conflict。

本次判断有三类证据：地址推导确有 4-way/8-way 冲突；对应硬件计数很高；
只改 shared 布局后 conflicts、Short Scoreboard 和时间同时下降。因此能判断它是原实现的重要瓶颈。
同理，Long Scoreboard 不能直接解释为“全部在等待 HBM”，cache 路径中的 load 依赖也需要区分。

补采的 PC sampling 还提供了一个重要例子：普通布局中，Short Scoreboard 采样主要落在消费 fragment 的
`HMMA.16816.F32` 上，而不是生产它们的 `LDSM` 上。普通同步版的 Long Scoreboard 采样则几乎全在
`STS.128`，对应源码第 33 行的 shared store，它正在消费 global load 的结果。
**stall 采样位置常是等待操作数的消费者，不必是延迟来源指令本身。**
采样次数不是精确延迟；不将不同运行的原始 sample 数直接换算成加速比例。

## 5. cp.async 为什么有效？是否消除了 global load 延迟？

可以这样回答：

> 同步搬运中，shared store 消费 global load 的寄存器结果；调度器遇到这条依赖时需要等待。
> cp.async 把数据直接搬入 shared，复制完成由显式 wait 管理。我再通过双缓冲，先发起下一块复制，
> 用当前块的独立 MMA 计算覆盖等待时间。它改善的是依赖安排、指令开销及可重叠程度，global 访问本身仍有延迟。

XOR 同步、异步单缓冲、异步双缓冲的 Long Scoreboard 分别为 7.084、1.608、0.139。
这些是按 issued instruction 归一化的 NCU warp 状态指标；不能当作内存延迟或端到端收益的直接比例。
即使不流水化，异步搬运也可能减少寄存器中转与指令开销，所以不能说“没有双缓冲就必然零收益”。

## 6. 为什么最终 occupancy 只有约 32%？shared 太大了吗？

当前最终版 NCU 的限制是：

| 资源 | 允许的 resident CTA/SM |
|---|---:|
| registers | 2 |
| shared memory | 3 |
| warp 数 | 6 |
| CTA 数硬件上限 | 24 |

取最小值为 2 CTA。每 CTA 8 个 warp，SM89 上限为 48 个 warp，理论 occupancy 为
`2×8/48=33.33%`，实际为 32.32%。因此**这版主要受寄存器驻留限制，单纯减少 shared 不会解除当前 2 CTA 上限**。
126 registers/thread 还需要按硬件粒度分配，不能只按裸乘法精确预测驻留率；NCU 的资源限制值是直接证据。

`__launch_bounds__(256,2)` 给编译器提供寄存器资源目标，不会强制调度器永远驻留两个 CTA，
也不是任意情况下都能无代价降低寄存器用量。

原 XOR 同步报告出现实际 34.37% 略高于理论 33.33% 的不一致，不能据此声称超过硬件驻留上限。
该读数需要结合采集归一化、重放与重采样核对；资源上限仍以 launch/occupancy 分析为准。
本次补采该实现为 32.87%，落在 33.33% 理论上限内；尚未确认上次轻微不一致的唯一原因。

## 7. cuBLAS 的 occupancy 只有 16%，为什么反而更快？

cuBLAS 在该尺寸使用 128-thread CTA，234 registers/thread，同样为 2 CTA/SM，
但每 CTA 仅 4 个 warp，所以理论 occupancy 为 `2×4/48=16.67%`，实测 16.28%。
V4 则是 256-thread CTA，2 CTA 对应 16 个 warp，实测 32.32%。

两者的工作分配、每线程输出数、指令组合及流水线不同；少量 warp 也可能提供足够的指令级并行。
原报告的 Tensor pipe active 为 V4 45.04%、cuBLAS 47.98%，与 cuBLAS 更快的观察相符，
但这一个百分比不足以完整解释二者差距，更不能直接等同于峰值 FLOPS 利用率。

因此我优化的是单位时间完成的计算，而不是单独追求 occupancy 或 issue-active 百分比最大。
补采进一步显示 V4 的 eligible warps/scheduler 约 0.319、cuBLAS 约 0.143，
但 cuBLAS 执行的辅助指令也显著更少，不能将这些值按同一理想 issue 频率直接比较。
更详细的指令证据见第 12 问。

## 8. 为什么 grouped CTA 没有收益？

已确认：4096³ 下，线性版为 2.242 ms，group8 为 2.273 ms；后者产生 8B spill stores/loads。
原 NCU 中 L2 read sector hit rate 分别为 93.53% 和 91.34%，该次采集中没有看到更好的 L2 命中。

grouped CTA 只改变逻辑 tile 映射，目的是增加 B 的跨 CTA 复用机会；它不保证实际调度顺序。
映射指令、寄存器压力、spill、缓存行为都可能影响收益。
**当前不能把全部差异归因于 spill 或 L2 命中率**：尚未完成只改变其中一项的隔离实验，
且默认 replay/cache 条件也不同于反复使用驻留输入的 benchmark。

## 9. 为什么 bank conflicts 降低 99.97%，耗时没有降低 99.97%？

计数器不是总时间的线性分解。冲突涉及 shared 供数，而 kernel 还要执行 MMA、global 搬运、地址计算、
同步和写回；不同 warp 的等待也可以互相覆盖。减少某种等待后，限制可能转移到另一个资源。

V3 是现成例子：load conflicts 相对原 v1 减少约 99.97%，但整体吞吐只增加 5.16%。
原 v1 已有预取和双缓冲，而且与 V3 的线程分工等也不同；不能把全部差异归因于布局。
V4 的同框架布局消融更适合展示独立贡献。

## 10. 如何保证与 cuBLAS 的比较可信？

> 我区分 FP32 CUDA Core 和 FP16 Tensor Core 两条精度路径，分别选择同精度 cuBLAS。
> 使用固定输入、正确性检查、预热、多进程多轮事件计时，并轮换实现顺序。
> Graphics/SM 手动锁 2100 MHz，记录遥测和 source/binary hash。
> NCU 用于解释硬件行为，最终延迟用未挂 profiler 的 benchmark 测量。

4096³ 每实现 45 个轮次样本，取中位数；FP16 输入生成和 H2D 不计时，输入常驻并反复使用。
V4 对比的是默认 cublasGemmEx，不应宣称超过所有 cuBLASLt 搜索配置。
时钟遥测是离散观测，不能证明采样间隔内绝对没有波动。

## 11. 还有哪些不能说成已证实？

- `no_instruction` 或循环展开数量，单独不能证明 I-cache 已成瓶颈。
- Math Pipe Throttle 高，单独不能证明 Tensor Core 在整个执行期饱和。
- Long Scoreboard 降低，不等于所有等待消失；等待可能转移到 wait/barrier 等位置。
- 更高 L2 命中率、occupancy、issue active，均不能脱离总工作量与绝对耗时单独评价。
- 当前没有把 V4 相对 cuBLAS 的全部差距定位到某一条指令，因此不应给出单一、确定的根因。

## 12. 补采后，对剩余性能差距有什么更具体的认识？

先看同一 4096³ 工作量的调度器和指令证据：

| 实现 | active warps/scheduler | eligible warps/scheduler | 无 eligible 的周期比例 | 动态 SASS warp 指令总数 | 非 HMMA 指令 |
|---|---:|---:|---:|---:|---:|
| 普通 PTX | 3.928 | 0.135 | 89.74% | 1.945 亿 | 1.610 亿 |
| XOR 同步 | 3.849 | 0.231 | 82.63% | 2.264 亿 | 1.928 亿 |
| 默认 V4 | 3.880 | 0.319 | 78.83% | 2.444 亿 | 2.109 亿 |
| cuBLAS | 1.952 | 0.143 | 90.91% | 0.986 亿 | 0.315 亿 |

前三者表明，驻留 warp 数基本接近，而 eligible warp 数随着供数优化增加。
但表的最后一行也表明：**更高的 eligible、更多的发射并不自动等于更高 GEMM 吞吐，
还必须比较每次发射完成的工作，以及有多少是辅助指令。**

本次 SourceCounters 中，V4 的 IMAD、LOP3 分别执行约 5,926 万和 4,325 万条；
非 HMMA 指令总量约为 cuBLAS 的 6.70 倍。边界检查、索引与地址生成、布局变换等是值得进一步隔离的开销来源。
这是对“仍有辅助指令开销”的直接证据，但没有证明它独自解释了全部性能差距；各类指令的代价与重叠能力不同。

比较 HMMA 数量时必须区分指令形状：

- V4：`HMMA.16816.F32`，33,554,432 条。
- cuBLAS：`HMMA.1688.F32`，67,108,864 条。

二者 K 维计算量相差一倍，不能说 cuBLAS 做了两倍矩阵运算。
按相应矩阵形状核算，二者都对应 `2×4096³` FLOPs。
动态 warp 指令数统计的是硬件 SASS，不是源代码行数或 PTX 指令数。

另一个可讲的取舍：XOR 同步版比普通版执行更多整数/布局指令，却依靠减少 shared 服务量而更快。
优化目标是总执行时间，不是单独最小化指令数。

## 文件与复现

指标解释参考 [NVIDIA Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。
原始补采报告、raw/source CSV、指令统计和时钟记录保存在 `interview_profiles/`；
cuBLAS 无用户源码，使用 SASS 页导出。

```bash
# 保持手动 2100 MHz 锁频，在仓库根目录执行；输出目录必须尚不存在。
python3 sgemm/results/20261106_v4/profile_interview.py --output /new/output/path

# 汇总已保存于 interview_profiles 的 raw/source CSV，并校验 wavefront 结论。
python3 sgemm/results/20261106_v4/summarize_interview.py
python3 sgemm/results/20261106_v4/check_ldmatrix_layout.py
```

[补采环境与 SHA256](interview_profiles/environment.json)、
[原始时钟记录](interview_profiles/telemetry.csv)、
[可复核的结构化分析](interview_profiles/analysis_summary.json)。
