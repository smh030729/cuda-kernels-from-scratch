"""Compare repository FP16 attention forwards against official flash-attn on one GPU."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time

import flash_attn
from flash_attn import flash_attn_func
import torch
import triton
from triton.testing import do_bench_cudagraph


VERSIONS = ("v0_fp16", "v1_fp16", "v2a_fp16", "v2b", "v2c", "v2c_bc32_qreg")


def gpu_state():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.sm,utilization.gpu,memory.used",
        "--format=csv",
    ], text=True)


def reference(q, k, v):
    """FP32 reference with chunked query rows to avoid an N-by-N allocation."""
    q, k, v = q.float(), k.float(), v.float()
    output = torch.empty_like(q)
    for start in range(0, len(q), 128):
        scores = torch.matmul(q[start:start + 128], k.T) * 0.125
        output[start:start + 128] = torch.softmax(scores, -1) @ v
    return output


def errors(output, expected):
    actual = output.float()
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=2e-3)
    diff = actual - expected
    return dict(max_abs=diff.abs().max().item(), rms=diff.square().mean().sqrt().item())


def benchmark_size(api, n, repeats, seed):
    torch.manual_seed(seed)
    q, k, v = [torch.randn(n, 64, device="cuda", dtype=torch.float16) for _ in range(3)]
    kt = k.T.contiguous()
    outputs = {name: torch.empty_like(q) for name in VERSIONS}
    q4, k4, v4 = (x.view(1, n, 1, 64) for x in (q, k, v))
    state_before = gpu_state()

    def launch(version, key, output):
        status = api.fa2_launch(version, q.data_ptr(), key.data_ptr(), v.data_ptr(),
                                output.data_ptr(), n, torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f"Custom CUDA launch failed: status={status}, version={version}")

    def official():
        return flash_attn_func(q4, k4, v4, dropout_p=0.0, softmax_scale=0.125, causal=False)

    funcs = {"official_fa2": official}
    for version, name in enumerate(VERSIONS):
        key = kt if version >= 3 else k
        def core(version=version, key=key, output=outputs[name]):
            launch(version, key, output)
        funcs[name] = core
        if version >= 3:
            def with_transpose(version=version, output=outputs[name]):
                kt.copy_(k.T)
                launch(version, kt, output)
            funcs[name + "_with_transpose"] = with_transpose
    funcs["transpose_k"] = lambda: kt.copy_(k.T)

    expected = reference(q, k, v)
    official_output = official().view(n, 64)
    assert official_output.dtype == torch.float16
    correctness = {"official_fa2": errors(official_output, expected)}
    for name in VERSIONS:
        funcs[name]()
        torch.cuda.synchronize()
        correctness[name] = errors(outputs[name], expected)
        correctness[name]["max_abs_vs_official"] = (outputs[name].float() - official_output.float()).abs().max().item()
    for name in (version + "_with_transpose" for version in VERSIONS[3:]):
        funcs[name]()
        torch.cuda.synchronize()
        correctness[name] = errors(outputs[name.split("_with")[0]], expected)

    for fn in funcs.values():
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    samples = {name: [] for name in funcs}
    rng = random.Random(seed)
    for repeat in range(repeats):
        names = list(funcs)
        rng.shuffle(names)
        for name in names:
            # Both paths execute inside CUDA Graphs timed with CUDA Events.
            ms = do_bench_cudagraph(funcs[name], rep=20, return_mode="median")
            samples[name].append(ms)
        print(f"N={n} round={repeat + 1}/{repeats}", flush=True)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    baseline = medians["official_fa2"]
    rows = []
    for name, ms in medians.items():
        row = dict(name=name, median_ms=ms, min_ms=min(samples[name]), max_ms=max(samples[name]))
        if name != "transpose_k":
            row["percent_of_official"] = baseline / ms * 100
            row["effective_tflops"] = 4 * n * n * 64 / (ms * 1e9)
        rows.append(row)
        print(f"N={n:5d} {name:24s} {ms:.6f} ms", flush=True)
    return dict(n=n, seed=seed, results=rows, samples_ms=samples, correctness=correctness,
                gpu_before=state_before, gpu_after=gpu_state())


@torch.inference_mode()
def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 16384])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--library", default=str(root / "build/flashattention2/libflashattention2_benchmark.so"))
    parser.add_argument("--output", default=str(root / "flashattention2/results/official_comparison.json"))
    args = parser.parse_args()
    if args.repeats < 1 or any(n <= 0 or n % 64 for n in args.lengths):
        parser.error("Lengths must be positive multiples of 64; repeats must be positive")
    api = ctypes.CDLL(args.library)
    api.fa2_launch.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    api.fa2_launch.restype = ctypes.c_int
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = [Path(__file__), root / "flashattention2/benchmark_api.cu",
               root / "flashattention2/flashattention2.cuh", Path(args.library)]
    sources += [root / "flashattention2" / name for name in (
        "flashattention2_v0_fp16.cu", "flashattention2_v1_fp16.cu",
        "flashattention2_v2a_fp16.cu", "flashattention2_v2_BC.cu",
        "flashattention2_v2c_bc32_qreg.cu")]
    report = dict(
        args=vars(args), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        flash_attn=flash_attn.__version__, triton=triton.__version__,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        configuration=dict(batch=1, heads=1, head_dim=64, input_dtype="float16",
                           output_dtype="float16", causal=False, dropout=0, scale=0.125,
                           forward_only=True, timer="CUDA Graph + CUDA Events; warm repeated inputs",
                           transpose="torch.Tensor.copy_ from K.T into preallocated contiguous Kt"),
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sources}, results=[],
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    for n in args.lengths:
        report["results"].append(benchmark_size(api, n, args.repeats, args.seed))
        path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
