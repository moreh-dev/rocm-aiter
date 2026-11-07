#!/usr/bin/env python3
"""scratchloc.py <file.s> <mangled-substring>: where are the scratch_* (spill / call-save) instructions of a kernel?
Classifies each one as inside a pipeline region (between an inline-asm 'buffer_load_dwordx4 ... offen' prologue and the
following s_waitcnt vmcnt(0)) - unsafe with hand-placed vmcnt - or outside. Also lists s_swappc (calls)."""
import re, sys
path, key = sys.argv[1], sys.argv[2]
L = open(path).read().split("\n")
st = next(i for i, l in enumerate(L) if re.match(r"^(\S*" + re.escape(key) + r"\S*):", l))
en = next(i for i in range(st, len(L)) if L[i].startswith(".Lfunc_end"))
inside = False; n_in = n_out = 0; calls = 0; ex = []
for i in range(st, en):
    t = L[i].strip()
    if t.startswith("buffer_load_dwordx4") and "offen" in t: inside = True
    if t.startswith("s_waitcnt") and "vmcnt(0)" in t: inside = False
    if t.startswith("s_swappc") or t.startswith("s_call"): calls += 1
    if t.startswith("scratch_"):
        if inside: n_in += 1; ex.append((i - st, t[:60]))
        else: n_out += 1
print(f"{L[st].split(':')[0][:60]}: scratch ops inside pipeline regions {n_in}, outside {n_out}, calls {calls}")
for e in ex[:5]: print("   inside:", e)
