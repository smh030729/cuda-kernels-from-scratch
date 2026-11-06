from pathlib import Path
import csv
import hashlib
import io
import json
import random
import statistics
import subprocess

REPO = Path("/home/smh/ai_infra/cuda-kernels-from-scratch")
OUT = Path(__file__).resolve().parent
NCU = "/tmp/var/target/linux-desktop-glibc_2_11_3-x64/ncu"
BINARY = REPO / "build/flashattention2/flashattention2_profile"
VERSIONS = ("v2c", "v2c_bc32_qreg")
LENGTHS = (16384, 32768, 65536)
REPEATS = 3
METRICS = (
    "gpu__time_duration.sum",
    "launch__grid_size", "launch__waves_per_multiprocessor",
    "launch__registers_per_thread", "launch__shared_mem_per_block_static",
    "launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem",
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if (OUT / "manifest.json").exists():
        raise RuntimeError("This run directory already contains results.")
    files = [REPO / "flashattention2/flashattention2_v2_BC.cu",
             REPO / "flashattention2/flashattention2_v2c_bc32_qreg.cu", BINARY]
    manifest = dict(
        clock_control="none; no external clock lock", cache_control="all", repeats=REPEATS,
        lengths=LENGTHS, versions=VERSIONS, timer="NCU gpu__time_duration.sum",
        ncu_version=subprocess.check_output([NCU, "--version"], text=True),
        sha256={str(path.relative_to(REPO)): digest(path) for path in files},
        runs=[])
    rng = random.Random(42)
    with (OUT / "telemetry.csv").open("w") as telemetry, (OUT / "telemetry_error.log").open("w") as errors:
        monitor = subprocess.Popen([
            "nvidia-smi", "--query-gpu=timestamp,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits", "-lms", "200",
        ], stdout=telemetry, stderr=errors)
        try:
            for repeat in range(1, REPEATS + 1):
                lengths = list(LENGTHS)
                rng.shuffle(lengths)
                for length in lengths:
                    versions = list(VERSIONS)
                    rng.shuffle(versions)
                    for version in versions:
                        name = f"{version}_n{length}_r{repeat}"
                        command = [
                            NCU, "--clock-control", "none", "--cache-control", "all",
                            "--profile-from-start", "off", "--launch-count", "1",
                            "--section", "SpeedOfLight", "--section", "Occupancy",
                            "--section", "LaunchStats", "--export", str(OUT / name),
                            str(BINARY), version, str(length)]
                        process = subprocess.run(command, cwd=REPO, text=True,
                                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                        (OUT / (name + ".log")).write_text(process.stdout)
                        if process.returncode:
                            raise RuntimeError(process.stdout)
                        imported = subprocess.check_output([
                            NCU, "--import", str(OUT / (name + ".ncu-repz")),
                            "--page", "raw", "--csv"], text=True)
                        (OUT / (name + "_metrics.csv")).write_text(imported)
                        rows = list(csv.DictReader(io.StringIO(imported)))
                        units, values = rows[0], rows[-1]
                        metrics = {
                            key: dict(value=float(values[key].replace(",", "")), unit=units[key])
                            for key in METRICS}
                        assert metrics["gpu__time_duration.sum"]["unit"] == "ms"
                        manifest["runs"].append(dict(
                            n=length, version=version, repeat=repeat, command=command, metrics=metrics))
                        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                        print(f"N={length} {version} round={repeat}: "
                              f"{metrics['gpu__time_duration.sum']['value']:.6f} ms, occupancy "
                              f"{metrics['sm__warps_active.avg.pct_of_peak_sustained_active']['value']:.2f}%",
                              flush=True)
        finally:
            monitor.terminate()
            monitor.wait()
    for path in files:
        assert digest(path) == manifest["sha256"][str(path.relative_to(REPO))], path
    manifest["hashes_verified_after_collection"] = True
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = []
    for length in LENGTHS:
        versions = {}
        for version in VERSIONS:
            runs = [run for run in manifest["runs"] if run["n"] == length and run["version"] == version]
            metrics = {}
            for key in METRICS:
                samples = [run["metrics"][key]["value"] for run in runs]
                metrics[key] = dict(median=statistics.median(samples), min=min(samples),
                                    max=max(samples), unit=runs[0]["metrics"][key]["unit"])
            versions[version] = metrics
        old = versions["v2c"]["gpu__time_duration.sum"]["median"]
        new = versions["v2c_bc32_qreg"]["gpu__time_duration.sum"]["median"]
        summary.append(dict(n=length, versions=versions,
                            latency_change_pct=(new/old-1)*100,
                            throughput_change_pct=(old/new-1)*100))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("OUTPUT", OUT, flush=True)
    for row in summary:
        print(row["n"], "latency change %", row["latency_change_pct"],
              "throughput change %", row["throughput_change_pct"], flush=True)

if __name__ == "__main__":
    main()
