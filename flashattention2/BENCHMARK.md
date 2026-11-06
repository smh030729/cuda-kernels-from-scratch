# Official FlashAttention-2 comparison

2026-11-06 更新：当前源码的 2100 MHz NCU 与官方吞吐对照见[根 README](../README.md#flashattention-2-forward)。下文保留早期实验口径。当前构建已补齐 FP32 BR64 的 launcher，并同时提供 V1 FP16 输入的 FP16 / FP32 输出接口。

The current V0 baseline is **Br=4**, selected by `v0_fp16`. The primary results below are the previously measured Br=4 comparison in `results/official_comparison.json`; selecting this baseline again does not constitute a new timing run. The BR64 implementation and its separately measured data remain available as an additional experiment.

## Configuration

Measured on 2026-10-25, NVIDIA RTX 4070 Ti SUPER 16GB, WSL Ubuntu 22.04.

- Official baseline: `flash-attn 2.8.3.post1`, public `flash_attn_func` API.
- PyTorch `2.11.0+cu128`, Triton `3.6.0`, CUDA compiler `12.8.93`.
- Batch=1, heads=1, head_dim=64, non-causal forward, dropout=0, scale=0.125.
- All implementations receive the same independently generated FP16 Q/K/V values and return FP16 O.
- The repository FP16 versions now round FP32 accumulator results to half at the final store. Original FP32-input versions retain FP32 output.
- V0 retains FP32 scalar arithmetic; the tensor-core versions retain their existing FP16 MMA operands and FP32 accumulators/softmax state.
- V0 uses four query rows per block and one query row per warp. V1 and later versions use 64 query rows per block. Both use four warps per block. V0-to-V1 changes include the tile organization and Tensor Core arithmetic; their measured speedup is not an isolated measurement of either change.

The FP16 launch functions' O argument is now `half*`, replacing their previous `float*` interface. Their header, profile program and correctness checker were updated together. Earlier `.ncu-rep` files were collected before this output change and are not used as the timing source for this comparison.

## Method

`bench_official.py` loads `libflashattention2_benchmark.so` through a minimal C ABI and passes PyTorch device buffers and the current CUDA stream directly to the existing launchers. This avoids a Torch C++ extension build and does not add a conversion kernel to the custom core path.

Both official and custom implementations are measured with Triton's `do_bench_cudagraph`: CUDA Graph capture and CUDA Events timing. Each outer round uses the median of ten graph-replay measurements; seven outer rounds run in seeded shuffled order, and the reported result is the median of those seven values. Compilation, graph construction, correctness checks, allocation and CPU dispatch are outside the measured GPU execution. The official public forward API's recorded GPU work remains included.

These are warm, repeatedly reused buffers. No cold-cache flush, fixed GPU clocks or exclusive desktop-GPU isolation is applied. Raw samples and before/after GPU state are retained. This is a single-head shape comparison, not a general multi-head or serving benchmark and not a claim against the fastest available attention implementation across all libraries.

### K transpose

V0/V1/V2a use ordinary contiguous K `[N,64]`; V2b/V2c require contiguous Kt `[64,N]`.

- Core timing: Kt is already available before execution.
- `with_transpose`: starts from the same row-major FP16 K, executes `Kt.copy_(K.T)` into a preallocated buffer, then launches attention in the same graph.
- `transpose_k`: measures that PyTorch transpose-copy alone. It is not an independently optimized custom transpose kernel.

Q/K/V already have FP16 dtype, so no FP32-to-FP16 input preparation is needed in either path. No output conversion is needed either.

### Correctness

Before timing, each implementation is compared with a chunked FP32 reference:

```text
reference = softmax(Q.float() @ K.float().T * 0.125) @ V.float()
```

Query rows are processed in groups of 128 to avoid allocating the full N-by-N score matrix. TF32 is disabled for reference matmuls. All outputs pass `atol=5e-4, rtol=2e-3` on the tested random inputs (seed=42, normal distribution). Absolute and RMS errors, and differences from official output, are saved in JSON. Passing these cases does not imply bitwise identity or correctness for every possible input distribution.

## Results

Relative performance is `official_time / custom_time * 100%` for the same attention shape. It is a throughput ratio, not a latency percentage or hardware utilization.

### N=16384

| Implementation | Median ms | Percentage of official performance |
|---|---:|---:|
| Official FA2 | 0.903415 | 100.00% |
| V0_fp16 (Br=4) | 28.034976 | 3.22% |
| V1_fp16 | 3.851981 | 23.45% |
| V2a_fp16 | 1.758604 | 51.37% |
| V2b core | 1.077618 | **83.83%** |
| V2c core | 1.137500 | 79.42% |
| V2b including K transpose | 1.096999 | **82.35%** |
| V2c including K transpose | 1.154440 | 78.26% |

Standalone K transpose: 0.021598 ms. A separately measured median need not add exactly to another median, so the combined path is measured directly.

For V2b core, the seven per-round samples span 1.075356–1.080775 ms; official samples span 0.900207–0.907675 ms. At N=16384, the custom versions' largest error against FP32 reference is approximately `2.51e-5`; their largest difference from official FP16 output is `3.05e-5`.

### Shape sensitivity

| N | Official ms | V2b core ms | V2b relative | V2c core ms | V2c relative |
|---:|---:|---:|---:|---:|---:|
| 1024 | 0.008795 | 0.025499 | 34.49% | 0.024017 | 36.62% |
| 4096 | 0.065332 | 0.099859 | 65.42% | 0.092325 | 70.76% |
| 16384 | 0.903415 | 1.077618 | 83.83% | 1.137500 | 79.42% |

Double buffering is faster than V2b in the two smaller tested shapes, but slower at N=16384. The timing experiment establishes this ordering; it does not alone establish the precise resource bottleneck causing it.

All primary V0 Br=4–V2c samples, accuracy statistics, source/library SHA256 hashes and GPU states are in [results/official_comparison.json](results/official_comparison.json).

### Separate BR64 experiment

The BR64 FP16 adaptation uses 64 query rows per block and 16 rows per warp. It took 0.614251, 2.480558 and 44.013056 ms at N=1024, 4096 and 16384 in a separate run. Those figures are not the primary V0 baseline and are not an as-is measurement of the original FP32 BR64 source.

Its compiler resource report confirms 255 registers/thread, a 3072-byte stack frame, 3060 bytes of spill stores and 3624 bytes of spill loads. Spill byte counts are static compiler statistics, not total runtime traffic; the precise runtime contribution was not quantified. The original FP32 BR64 source is preserved unchanged.

The separate results remain in [results/official_comparison_br64.json](results/official_comparison_br64.json).

## Reproduce

From the CUDA repository root, using the already installed sibling nano-vllm Python environment:

```bash
cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build --target flashattention2_benchmark flashattention2_check flashattention2_profile -j4

./build/flashattention2/flashattention2_check
../nano-vllm/.venv/bin/python flashattention2/bench_official.py \
  --lengths 1024 4096 16384 --repeats 7
```

The Python script requires PyTorch, Triton and flash-attn. `--library` and `--output` select alternate shared-library and result paths. The default result filename is `official_comparison.json`, and the default V0 implementation is Br=4.

For the primary V0 Br=4 profiling, select:

```bash
./build/flashattention2/flashattention2_profile v0_fp16 16384
```

The separate BR64 implementation is still available through the `v0_br64_fp16` profile selector. Its launcher requires N to be a multiple of 64; partial tiles are rejected before launch.

Additional verification: [CPU correctness log](results/br64_check.txt), [Compute Sanitizer memory-check log](results/br64_memcheck.txt), [BR64 compiler resource report](results/br64_ptxas.txt). Memory checking reports zero errors across the boundary-size suite, including legal BR64 sizes and rejection of unsupported partial tiles.

## V2c BC32 / register-resident Q

The separate `v2c_bc32_qreg` variant uses Br=64, Bc=32 and caches Q fragments in registers.
Its shared staging storage is 16 KiB; original V2c remains available as `v2c`.
The profile selector, correctness suite and official benchmark include the new variant.
See [implementation, validation, resources and preliminary results](results/v2c_bc32_qreg_20261026/final/README.md).
These new timing results are not clock-locked; they are a separate experiment from the tables above.
