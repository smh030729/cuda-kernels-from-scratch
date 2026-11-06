"""Capture one launch per implementation. Run from any directory after a Release build."""
import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
args = argparse.ArgumentParser()
args.add_argument('--output', required=True, type=Path)
args.add_argument('--group', choices=['gemm','fa2','basic','demo'], required=True)
cfg = args.parse_args()
out = cfg.output.resolve()
out.mkdir(parents=True, exist_ok=True)
ncu = os.environ.get('NCU') or shutil.which('ncu')
if not ncu:
    raise SystemExit('Set NCU to the ncu executable')
metrics = ['gpu__time_duration.sum','launch__registers_per_thread',
           'sm__warps_active.avg.pct_of_peak_sustained_active',
           'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
           'smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio',
           'smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio',
           'dram__throughput.avg.pct_of_peak_sustained_elapsed',
           'sm__throughput.avg.pct_of_peak_sustained_elapsed']
cases = {}
if cfg.group == 'gemm':
    for impl, variant, name in [(1,0,'v1'),(2,0,'v2'),(3,0,'v3_sync'),(3,1,'v3_async'),(3,2,'v3_double'),(0,0,'cublas_fp32')]:
        cases[name] = ['sgemm/sgemm_compare','--profile','--impl',impl,'--variant',variant,'--size',4096]
    for variant,name in enumerate(['wmma','plain','xor','async','double','group8']):
        cases['v4_'+name] = ['sgemm/sgemm_v4_compare','--profile','--impl',3,'--variant',variant,'--size',4096]
    cases['cublas_half'] = ['sgemm/sgemm_v4_compare','--profile','--impl',0,'--size',4096]
elif cfg.group == 'fa2':
    for name in 'v0 v0_br64 v1 v2a v0_fp16 v0_br64_fp16 v1_fp16 v1_fp16_out32 v2a_fp16 v2b v2c v2c_bc32_qreg'.split():
        cases[name] = ['flashattention2/flashattention2_profile',name,16384]
elif cfg.group == 'demo':
    for name in ['ldmatrix_x1','x1']:
        cases[name] = [name + '_demo']
else:
    for name in 'softmax_v1 softmax_reg softmax_shared rms_vec rms_scalar hist_v1 hist_v2 trans_v1 trans_v2 conv_v2 conv_v3'.split():
        cases[name] = ['catalog_basic_profile',name]
    cases['reduction_v1'] = ['reduction/reduction_profile']
data = {}
for name,(exe,*params) in cases.items():
    path = out / (cfg.group+'_'+name)
    before = subprocess.check_output(['nvidia-smi','--query-gpu=clocks.current.sm','--format=csv,noheader,nounits'],text=True).strip()
    cmd = [ncu,'--clock-control','none','--profile-from-start','on' if name=='reduction_v1' or cfg.group=='demo' else 'off',
           '--launch-count','1','--metrics',','.join(metrics),'-o',str(path),str(root/'build'/exe),*map(str,params)]
    with path.with_suffix('.log').open('w') as f:
        subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,NVIDIA_TF32_OVERRIDE='0'))
    raw = subprocess.check_output([ncu,'-i',str(path)+'.ncu-rep','--page','raw','--csv','--print-units','base'],text=True)
    units,row = list(csv.DictReader(raw.splitlines()))
    after = subprocess.check_output(['nvidia-smi','--query-gpu=clocks.current.sm','--format=csv,noheader,nounits'],text=True).strip()
    data[name] = dict(kernel=row['Kernel Name'],args=params,sm_mhz_before_after=[before,after],
                      metrics={m:[float(row[m].replace(',','')),units[m]] for m in metrics})
    (out/(cfg.group+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    print(name, flush=True)
