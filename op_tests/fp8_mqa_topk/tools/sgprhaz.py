#!/usr/bin/env python3
"""sgprhaz.py <file.s> [mangled-substring]: gfx9 'VALU writes SGPR -> VMEM reads that SGPR' hazard (needs 5 wait
states). The compiler's hazard recognizer does not see inline-asm VMEM instructions, so a v_readlane (SGPR spill
reload) or v_readfirstlane right before an asm buffer_load/store can feed it a half-written descriptor / soffset.
Reports, per function, every buffer_* whose SGPR operands were written by a VALU within the previous 5 wait states."""
import re, sys
path = sys.argv[1]; key = sys.argv[2] if len(sys.argv) > 2 else ""
L = open(path).read().split("\n")
def sregs(tok):
    out = set()
    for a, b in re.findall(r"\bs\[(\d+):(\d+)\]", tok): out |= set(range(int(a), int(b) + 1))
    out |= {int(x) for x in re.findall(r"\bs(\d+)\b", tok)}
    return out
fn = None; hist = []; viol = {}
for line in L:
    m = re.match(r"^(_Z\S+):", line)
    if m: fn = m.group(1); hist = []; continue
    if fn is None or (key and key not in fn): continue
    t = line.strip()
    if not t or t.startswith(";") or t.startswith(".") or t.endswith(":"):
        if re.match(r"^\.LBB", t): hist = []     # conservative: reset at block labels
        continue
    op = t.split()[0]
    ws = 1
    if op == "s_nop":
        try: ws = int(t.split()[1], 0) + 1
        except Exception: ws = 1
    if op.startswith("buffer_"):
        args = t[len(op):].split(",")
        rd = set()
        for a in args[2:]: rd |= sregs(a)          # rsrc and soffset
        dist = 0
        for (w_op, w_regs, w_ws) in reversed(hist):
            if dist >= 5: break
            if w_regs & rd:
                viol.setdefault(fn, []).append((dist, w_op, t[:70])); break
            dist += w_ws
    # record SGPR writes by VALU (v_readlane / v_readfirstlane / VOP3 with SGPR dst like v_cmp_*_e64)
    if op.startswith("v_"):
        dst = t[len(op):].split(",")[0]
        hist.append((op, sregs(dst) if re.search(r"\bs\[?\d", dst) and "v" not in dst.split("[")[0].replace("vcc","") else set(), ws))
    else:
        hist.append((op, set(), ws))
    hist = hist[-12:]
tot = 0
for f, v in viol.items():
    tot += len(v)
    print(f"{f[:70]}: {len(v)} violations, e.g. {v[0]}")
print(f"total violations: {tot}")
