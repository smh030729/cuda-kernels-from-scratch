# V2c: Bc=32 and register-resident Q

This is an independent variant named `v2c_bc32_qreg`; the original V2c is preserved.
Br=64, Bc=32, D=64, 4 warps/block; FP16 inputs/output and FP32 accumulation.
Q/V are row-major [N,64], Kt is contiguous [64,N]. N must be a multiple of 64;
N<=0 is a no-op, matching the existing V2c launcher contract.

## Implementation

- Cache all four Q MMA fragments per thread (16 32-bit registers of Q data) before the KV loop.
- Reuse Q's 8 KiB initialization staging area as the two 4 KiB K buffers, after a block barrier.
- Keep two 4 KiB V buffers. Total static shared memory is 16 KiB.
- For 32-column K shared rows, use col XOR (((row >> 1) & 3) << 3).
  For 64-column Q/V rows, retain col XOR ((row & 7) << 3).
  The 32-column layout accounts for the row stride already selecting one half of the banks.
- Keep 16B cp.async transfers and the original wait/barrier double-buffer protocol.
- Profile selector and Python C ABI version 5 select the new variant.

## Resources

NCU 2026.3.1, N=16384 on RTX 4070 Ti SUPER:

| Metric | Original V2c | BC32/Q-register |
|---|---:|---:|
| Static shared memory/block | 40 KiB | 16 KiB |
| Registers/thread | 96 | 96 |
| Shared-memory block limit/SM | 2 | 5 |
| Register block limit/SM | 5 | 5 |
| Theoretical occupancy | 16.67% | 41.67% |
| Achieved occupancy | 15.44% | 27.04% |

The new kernel's standalone compiler report has zero stack frame, spill loads and spill stores.
SASS has four Q LDSM.M88.4 instructions in initialization, before the KV loop.
Achieved occupancy is workload-specific; it need not reach the theoretical capacity.

## Preliminary performance

Core attention, same-run CUDA Graph timings, seven shuffled rounds, median.
**Clocks were not locked in this run. Do not combine these times with the previous fixed-clock run.**
The combined change is measured here; no individual attribution to Bc or Q caching is claimed.

| N | Original V2c ms | BC32/Q-register ms | Latency change |
|---:|---:|---:|---:|
| 1024 | 0.023818 | 0.029011 | +21.80% |
| 4096 | 0.092026 | 0.111006 | +20.62% |
| 16384 | 1.138365 | 1.075456 | -5.53% |

The variant improves the N=16384 latency in this run, but regresses the smaller shapes.
At N=16384 it is approximately tied with the original single-buffer V2b
(1.070637 ms versus this variant's 1.075456 ms); it is not established as a new overall winner.
Changing Bc doubles the number of transfer stages; the smaller-shape slowdown's precise cause
has not been isolated. No automatic shape dispatch was added.

## Validation

- Existing CPU reference suite, including legal lengths and rejection of unsupported lengths: PASS.
- FP32 PyTorch reference and official flash-attn comparison at N=1024/4096/16384: PASS
  with atol=5e-4, rtol=2e-3.
- Compute Sanitizer memcheck, racecheck, synccheck at N=256: zero errors/hazards.
- The original V2c source, new kernel source and shared-library SHA256 match the tested artifacts.
- V1_fp16 was independently edited after the library build, during the benchmark. Its current
  source hash differs from the recorded hash; that concurrent edit was preserved. V1 rows describe
  the previously built binary. The V2c comparison is unaffected. See source_hash_review.json.
- Raw NCU reports, imported metrics, compiler resources and SASS are alongside this document.

## Run

From the CUDA repository root:

```bash
cmake --build build --target flashattention2_profile flashattention2_check flashattention2_benchmark -j4
./build/flashattention2/flashattention2_check
./build/flashattention2/flashattention2_profile v2c_bc32_qreg 16384
../nano-vllm/.venv/bin/python flashattention2/bench_official.py --lengths 1024 4096 16384 --repeats 7 --output flashattention2/results/official_comparison_bc32_qreg.json
```

The benchmark measures both core attention and the path including K transpose.
