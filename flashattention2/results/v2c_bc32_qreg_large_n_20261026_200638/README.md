# V2c BC32/Q-register：增大 N 的 occupancy 测试

RTX 4070 Ti SUPER，66 SM；Br=64、Bc=32、D=64，128 threads/block。
同一 kernel 和可执行文件，三个尺寸各采集三轮，轮内随机顺序；以下报告中位数。
NCU 2026.3.1；--clock-control none --cache-control all，未进行外部锁频。
主要用于观察 occupancy 随工作规模变化；耗时为 NCU profiling 耗时，不与 CUDA Graph benchmark 混用。

| N | Grid blocks | Waves/SM | 理论 occupancy | 实际 occupancy 中位数 | 实际范围 |
|---:|---:|---:|---:|---:|---:|
| 16384 | 256 | 0.78 | 41.67% | 27.05% | 27.05–27.06% |
| 32768 | 512 | 1.55 | 41.67% | 34.44% | 34.43–36.20% |
| 65536 | 1024 | 3.10 | 41.67% | 38.28% | 38.10–38.54% |

各尺寸资源占用一致：96 registers/thread、16 KiB static shared/block；寄存器和 shared 各允许 5 blocks/SM。

N=65536 时实际 occupancy 达到理论值的约 91.9%，支持此前 grid/工作规模不足是差距重要来源的判断。
增大 N 同时增加 grid block 数和每 block 的 KV 循环长度，也会摊薄启动、收尾影响；本实验不能把这些因素单独拆分，也不能据此认定剩余差距的唯一原因。

manifest.json 保存命令、源码/可执行文件 SHA256 和 GPU 状态；summary.json 保存统计值；九份 .ncu-repz 为原始报告。
测试前后源码和可执行文件 SHA256 一致。
