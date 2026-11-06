"""Profile the final binary; keep replay durations separate from event timings."""
from pathlib import Path
import csv
import json
import os
import subprocess

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent / 'locked2100' / 'profiles'
out.mkdir(parents=True, exist_ok=True)
ncu = '/opt/nvidia/nsight-compute/2025.1.1/ncu'
env = dict(os.environ, NVIDIA_TF32_OVERRIDE='0')
keys = [
    'gpu__time_duration.sum', 'launch__registers_per_thread',
    'launch__shared_mem_per_block_static',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
    'smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio',
    'smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio',
    'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'dram__bytes_read.sum', 'dram__bytes_write.sum',
]
result = {}
for label, impl, variant in [('wmma',3,0), ('plain',3,1), ('xor_sync',3,2),
                             ('xor_single',3,3), ('v4',3,4), ('group8',3,5), ('cublas',0,4)]:
    report = out / f'{label}_n4096'
    command = [ncu, '--profile-from-start', 'off', '--clock-control', 'none',
        '--section', 'SpeedOfLight', '--section', 'SpeedOfLight_RooflineChart',
        '--section', 'Occupancy', '--section', 'WarpStateStats',
        '--section', 'MemoryWorkloadAnalysis_Tables', '--force-overwrite',
        '-o', str(report), str(root / 'build/sgemm/sgemm_v4_compare'),
        '--profile', '--impl', str(impl), '--variant', str(variant), '--size', '4096']
    with (out / f'{label}.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=env)
    raw = out / f'{label}_raw.csv'
    with raw.open('w') as file:
        subprocess.run([ncu, '--import', str(report)+'.ncu-rep', '--page', 'raw', '--csv'],
                       stdout=file, check=True)
    with raw.open() as file:
        rows = list(csv.DictReader(file))
    units, data = rows[0], rows[-1]
    result[label] = {'kernel': data.get('Kernel Name'), 'metrics': {
        key: {'value': data[key], 'unit': units[key]} for key in keys if key in data}}
    (out / 'selected_metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(label, 'complete', flush=True)
