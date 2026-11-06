"""Summarize raw and source CSVs without double-counting CUDA/SASS rows."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import json

out = Path(__file__).resolve().parent / 'interview_profiles'
metrics = [
    'smsp__warps_active.avg.per_cycle_active', 'smsp__warps_eligible.avg.per_cycle_active',
    'smsp__issue_inst0.avg.pct_of_peak_sustained_active',
    'smsp__issue_active.avg.per_cycle_active', 'smsp__inst_executed.sum',
    'sm__maximum_warps_per_active_cycle_pct',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
    'sm__pipe_tensor_cycles_active_v2.avg.pct_of_peak_sustained_active',
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
    'lts__t_sector_op_read_hit_rate.pct',
]
columns = ['Instructions Executed', 'L1 Wavefronts Shared Excessive',
           'L1 Wavefronts Shared', 'L1 Wavefronts Shared Ideal',
           'stall_long_sb', 'stall_short_sb', 'stall_barrier', 'stall_math', 'stall_wait']
summary = {}
for name in ['plain','xor_sync','v4','group8','cublas']:
    with (out / (name+'_raw.csv')).open() as file:
        raw = list(csv.DictReader(file))
    units, data = raw[0], raw[-1]
    selected = {key: {'value':data[key], 'unit':units[key]} for key in metrics if key in data}
    with (out / (name+'_source.csv')).open() as file:
        rows = list(csv.reader(file))
    header_index = next(i for i,row in enumerate(rows) if row and row[0] in ['Line No','Address'])
    header = rows[header_index]
    mixed = header[0] == 'Line No'
    addr_index, code_index = (2,3) if mixed else (0,1)
    by_opcode = defaultdict(Counter)
    ldsm_by_line = defaultdict(Counter)
    full_opcodes = Counter()
    seen = set()
    line, source = '', ''
    top = []
    for row in rows[header_index+1:]:
        if len(row) <= code_index:
            continue
        if mixed and row[0].isdigit():
            line, source = row[0], row[1]
        address = row[addr_index]
        if not address.startswith('0x') or address in seen:
            continue
        seen.add(address)
        tokens = row[code_index].strip().split()
        opcode = tokens[1] if tokens[0].startswith('@') else tokens[0]
        values = {col:float(row[header.index(col)].replace(',',''))
                  if row[header.index(col)] not in ['','-'] else 0. for col in columns}
        by_opcode[opcode.split('.')[0]].update(values)
        if opcode.startswith('LDSM') and mixed:
            ldsm_by_line[line].update(values)
        full_opcodes[opcode] += values['Instructions Executed']
        top.append(dict(line=line,source=source,sass=row[code_index].strip(),**values))
    total = sum(v['Instructions Executed'] for v in by_opcode.values())
    hmma = by_opcode['HMMA']['Instructions Executed']
    summary[name] = dict(metrics=selected, instructions=total, hmma_instructions=hmma,
        non_hmma_instructions=total-hmma, full_opcodes=full_opcodes,
        ldsm=by_opcode['LDSM'], ldsm_by_cuda_line=ldsm_by_line,
        top_long_scoreboard=sorted(top,key=lambda v:-v['stall_long_sb'])[:3],
        top_short_scoreboard=sorted(top,key=lambda v:-v['stall_short_sb'])[:3])
    (out / (name+'_instruction_summary.json')).write_text(json.dumps(by_opcode,indent=2)+'\n')
for name in ['xor_sync','v4','group8']:
    assert summary[name]['ldsm']['L1 Wavefronts Shared Excessive'] == 0
    assert summary[name]['ldsm']['L1 Wavefronts Shared'] == 50331648
assert summary['plain']['ldsm']['L1 Wavefronts Shared Excessive'] == 285212672
(out / 'analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for name, data in summary.items():
    print(name, 'instructions', int(data['instructions']), 'non-HMMA', int(data['non_hmma_instructions']),
          'LDSM excessive', int(data['ldsm']['L1 Wavefronts Shared Excessive']))
