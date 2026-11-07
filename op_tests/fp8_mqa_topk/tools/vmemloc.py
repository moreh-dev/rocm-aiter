#!/usr/bin/env python3
"""vmemloc.py <file.s> <mangled-substring>: VMEM instructions inside the hand-vmcnt pipeline regions of a kernel
(between an inline-asm 'buffer_load_dwordx4 ... offen' and the next 's_waitcnt vmcnt(0)'), grouped by opcode.
Expected there: buffer_load_dwordx4 (K), buffer_load_dword (scales), buffer_store_dwordx2 (candidates / samples),
buffer_store_dword / dwordx4 (DENSE logits). Anything else (global_*, flat_*, scratch_*, other buffer ops) adds to
vmcnt behind the hand-placed counts' back."""
import re, sys, collections
path, key = sys.argv[1], sys.argv[2]
L = open(path).read().split("\n")
st = next(i for i, l in enumerate(L) if re.match(r"^(\S*" + re.escape(key) + r"\S*):", l))
en = next(i for i in range(st, len(L)) if L[i].startswith(".Lfunc_end"))
inside = False; cnt = collections.Counter(); ex = collections.defaultdict(list)
EXPECTED = ("buffer_load_dwordx4", "buffer_load_dword", "buffer_store_dwordx2", "buffer_store_dword", "buffer_store_dwordx4")
for i in range(st, en):
    t = L[i].split(";")[0].strip()
    if t.startswith("buffer_load_dwordx4") and "offen" in t: inside = True
    if t.startswith("s_waitcnt") and "vmcnt(0)" in t: inside = False
    if not inside: continue
    op = t.split()[0] if t else ""
    if op.startswith(("buffer_", "global_", "flat_", "scratch_")):
        cnt[op] += 1
        if op not in EXPECTED and len(ex[op]) < 3: ex[op].append((i - st, t[:70]))
print(f"{L[st].split(':')[0][:70]}: VMEM inside pipeline regions: {dict(cnt)}")
for op, e in ex.items():
    for x in e: print("   UNEXPECTED", x)
