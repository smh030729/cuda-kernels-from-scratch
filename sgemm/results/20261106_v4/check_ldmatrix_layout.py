"""Enumerate the actual V4 ldmatrix row addresses (32 banks, 4-byte words)."""
from collections import Counter
from pathlib import Path
import json

def a_index(row, col, swizzle):
    return row*32 + (((col//8)^((row>>1)&3)) if swizzle else col//8)*8 + col%8

def b_index(row, col, swizzle):
    return (col//64)*32*64 + row*64 + ((((col%64)//8)^(row&7)) if swizzle
                                     else (col%64)//8)*8 + col%8

def wavefronts(index, row, col):
    # x4 is four 8x8 half matrices. Each group of 8 lanes supplies eight 16B rows.
    total = 0
    for group in range(4):
        banks = Counter()
        for lane in range(group*8, group*8+8):
            rr = row + (lane&7) + ((lane>>3)&1)*8
            cc = col + (lane>>4)*8
            address = index(rr,cc)
            assert address%8 == 0
            assert [index(rr,cc+i) for i in range(8)] == list(range(address,address+8))
            for word in range(4):
                banks[(address//2+word)%32] += 1
        total += max(banks.values())
    return total

result = {}
for swizzle in [False,True]:
    a = lambda r,c: a_index(r,c,swizzle)
    b = lambda r,c: b_index(r,c,swizzle)
    assert sorted(a(r,c) for r in range(128) for c in range(32)) == list(range(128*32))
    assert sorted(b(r,c) for r in range(32) for c in range(128)) == list(range(32*128))
    wa,wb=set(),set()
    for warp in range(8):
        for kk in [0,16]:
            for m in range(2):
                wa.add(wavefronts(a,(warp//2)*32+m*16,kk))
            for n in range(0,8,2):
                wb.add(wavefronts(b,kk,(warp%2)*64+n*8))
    assert wa == ({4} if swizzle else {16})
    assert wb == ({4} if swizzle else {32})
    # 4096^3: 32x32 CTAs, 128 K tiles, 8 warps, 2 kk steps.
    a_instructions = 32*32*128*8*2*2
    b_instructions = 32*32*128*8*2*4
    result['xor' if swizzle else 'plain'] = {
        'A_wavefronts_per_ldmatrix_x4': next(iter(wa)),
        'B_wavefronts_per_ldmatrix_x4_trans': next(iter(wb)),
        'ideal_wavefronts_per_x4':4,
        'A_dynamic_ldmatrix_warp_instructions':a_instructions,
        'B_dynamic_ldmatrix_warp_instructions':b_instructions,
        'predicted_layout_excess_wavefronts':a_instructions*(next(iter(wa))-4)
                                               +b_instructions*(next(iter(wb))-4),
    }
result['plain']['measured_aggregate_load_bank_conflicts'] = 285212672
assert result['plain']['predicted_layout_excess_wavefronts'] == 285212672
out = Path(__file__).resolve().parent / 'ldmatrix_layout_check.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
