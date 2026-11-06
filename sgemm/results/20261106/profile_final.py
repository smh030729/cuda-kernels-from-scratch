from pathlib import Path
import argparse
import os
import subprocess

root = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
ncu = '/opt/nvidia/nsight-compute/2025.1.1/ncu'
env = dict(os.environ, NVIDIA_TF32_OVERRIDE='0')
for label, impl in [('v1',1),('v2',2),('v3',3),('cublas',0)]:
    command = [ncu,'--profile-from-start','off','--clock-control','none',
        '--section','SpeedOfLight','--section','SpeedOfLight_RooflineChart',
        '--section','Occupancy','--section','WarpStateStats',
        '--section','MemoryWorkloadAnalysis_Tables','--force-overwrite',
        '-o',str(out / f'{label}_final_n4096'),
        str(root/'build/sgemm/sgemm_compare'),'--profile','--impl',str(impl),
        '--variant','2','--size','4096']
    with (out/f'ncu_{label}.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    with (out/f'{label}_final_metrics.csv').open('w') as file:
        subprocess.run([ncu,'--import',str(out/f'{label}_final_n4096.ncu-rep'),
                        '--page','raw','--csv'],stdout=file,check=True)
    print(label,'complete',flush=True)
with (out/'synccheck.log').open('w') as file:
    subprocess.run(['/usr/local/cuda/bin/compute-sanitizer','--tool','synccheck',
        '--error-exitcode','12',str(root/'build/sgemm/sgemm_compare'),'--check'],
        stdout=file,stderr=subprocess.STDOUT,check=True,env=env)
