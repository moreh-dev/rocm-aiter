#!/usr/bin/env python3
"""asmloadwaw.py <file.s> <mangled-substring>: inline-asm VMEM loads write their destination VGPRs asynchronously, but
the compiler treats the asm output as written at the asm statement: if the value is never used it may reuse the
registers at once, and the late load data then overwrites the new value. Reports every load (buffer_load ... offen)
whose destination is written by another instruction before the load is known complete (straight-line, in-order
vmcnt: a 's_waitcnt vmcnt(k)' completes it once >= k VMEM ops were issued after it). Follows fallthrough only."""
import re, sys
path, key = sys.argv[1], sys.argv[2]
L = open(path).read().split("\n")
st = next(i for i, l in enumerate(L) if re.match(r"^(\S*" + re.escape(key) + r"\S*):", l))
en = next(i for i in range(st, len(L)) if L[i].startswith(".Lfunc_end"))
def vregs(tok):   # VGPRs only (AGPRs a[n] are a separate register file)
    out = set()
    for a, b in re.findall(r"\bv\[(\d+):(\d+)\]", tok): out |= set(range(int(a), int(b) + 1))
    out |= {int(x) for x in re.findall(r"\bv(\d+)\b", tok)}
    return out
ins = []
for i in range(st, en):
    t = L[i].split(";")[0].strip()
    if not t or (t.startswith(".") and not t.endswith(":")): continue
    ins.append((i - st, t))
VMEM = ("buffer_", "global_", "flat_", "scratch_")
label_at = {}
for k, (rel, t) in enumerate(ins):
    if t.endswith(":"): label_at[t[:-1]] = k
out = []
for k, (rel, t) in enumerate(ins):
    if not (t.startswith("buffer_load_dword") and "offen" in t): continue
    dst = vregs(t.split()[1].rstrip(","))
    stack = [(k + 1, 0)]; seen = set(); hit = None
    while stack and hit is None:
        j, after = stack.pop()
        while j < len(ins):
            if (j, after) in seen: break
            seen.add((j, after))
            rel2, t2 = ins[j]
            if t2.endswith(":"): j += 1; continue
            op = t2.split()[0]
            if op in ("s_endpgm", "s_setpc_b64"): break
            m = re.match(r"s_waitcnt\s.*vmcnt\((\d+)\)", t2)
            if m and after >= int(m.group(1)): break
            if op == "s_branch":
                tgt = t2.split()[-1]
                if tgt in label_at: j = label_at[tgt]; continue
                break
            if op.startswith("s_cbranch"):
                tgt = t2.split()[-1]
                if tgt in label_at: stack.append((label_at[tgt], after))
                j += 1; continue
            if op.startswith(VMEM):
                after = min(after + 1, 64); j += 1; continue
            if op.startswith("s_"): j += 1; continue
            d = vregs(t2[len(op):].split(",")[0])
            if d & dst:
                hit = (rel, t[:58], rel2, t2[:70], after); break
            j += 1
    if hit: out.append(hit)
print(f"{L[st].split(':')[0][:60]}: {len(out)} asm-load destinations overwritten before the load is known complete (CFG)")
for o in out[:10]: print("  load @%d %s\n     write @%d %s   (VMEM issued after the load: %d)" % o)
