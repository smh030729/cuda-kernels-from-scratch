"""Supplementary source/scheduler profiling of the already benchmarked binary."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / 'sgemm'))
from run_compare_v4 import monitor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
original = json.loads((Path(__file__).resolve().parent / 'locked2100/environment.json').read_text())
binary = root / 'build/sgemm/sgemm_v4_compare'
assert hashlib.sha256(binary.read_bytes()).hexdigest() == original['binary_sha256']
for name, expected in original['source_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
ncu = '/opt/nvidia/nsight-compute/2025.1.1/ncu'
env = dict(os.environ, NVIDIA_TF32_OVERRIDE='0')
stop = threading.Event()
thread = threading.Thread(target=monitor, args=(stop, out / 'telemetry.csv'), daemon=True)
thread.start()
try:
    for name, impl, variant in [('plain',3,1), ('xor_sync',3,2), ('v4',3,4),
                                 ('group8',3,5), ('cublas',0,4)]:
        command = [ncu, '--profile-from-start','off', '--clock-control','none',
                   '--import-source','yes', '--force-overwrite', '-o',str(out / name)]
        for section in ['SchedulerStats','SourceCounters','ComputeWorkloadAnalysis',
                        'LaunchStats','Occupancy','WarpStateStats','MemoryWorkloadAnalysis_Tables']:
            command += ['--section',section]
        command += [str(binary),'--profile','--impl',str(impl),'--variant',str(variant),'--size','4096']
        with (out / (name+'.log')).open('w') as file:
            subprocess.run(command, stdout=file, stderr=subprocess.STDOUT, check=True, env=env)
        view = 'sass' if name == 'cublas' else 'cuda,sass'
        for page, options in [('raw',[]), ('source',['--print-source',view])]:
            with (out / (name+'_'+page+'.csv')).open('w') as file:
                subprocess.run([ncu,'--import',str(out / (name+'.ncu-rep')),
                                '--page',page,'--csv',*options], stdout=file, check=True)
        print(name, 'complete', flush=True)
finally:
    stop.set()
    thread.join()
with (out / 'telemetry.csv').open() as file:
    rows = list(csv.DictReader(file))
result = dict(binary_sha256=original['binary_sha256'], source_sha256=original['source_sha256'],
              clock_samples=len(rows), requested_graphics_MHz=2100,
              clock_verified=bool(rows) and all(float(r['graphics_MHz'])==2100 and
                                              float(r['sm_MHz'])==2100 for r in rows))
for name, expected in original['source_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
assert hashlib.sha256(binary.read_bytes()).hexdigest() == original['binary_sha256']
(out / 'environment.json').write_text(json.dumps(result,indent=2)+'\n')
if not result['clock_verified']:
    raise RuntimeError('Observed clock departed from 2100 MHz; inspect telemetry.csv')
print(json.dumps(result), flush=True)
