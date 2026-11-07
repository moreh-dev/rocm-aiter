#!/usr/bin/env python3
"""sgprhaz2.py <file.s> [mangled-substring]: 'VALU writes SGPR -> VMEM reads that SGPR' (5 wait states) over the
control-flow graph: walks back from every buffer_/global_/scratch_ instruction through fallthrough AND branch
predecessors (sgprhaz.py reset its history at block labels and missed hazards across block boundaries)."""
import re, sys, collections
path = sys.argv[1]; key = sys.argv[2] if len(sys.argv) > 2 else ""
L = open(path).read().split("\n")
def sregs(tok):
    out = set()
    for a, b in re.findall(r"\bs\[(\d+):(\d+)\]", tok): out |= set(range(int(a), int(b) + 1))
    out |= {int(x) for x in re.findall(r"\bs(\d+)\b", tok)}
    return out
funcs = []
cur = None
for line in L:
    m = re.match(r"^(_Z\S+):", line)
    if m: cur = [m.group(1), []]; funcs.append(cur); continue
    if cur is None: continue
    if line.startswith(".Lfunc_end"): cur = None; continue
    t = line.split(";")[0].strip()
    if not t: continue
    if re.match(r"^\.LBB\d+_\d+:", t): cur[1].append(("LABEL", t[:-1])); continue
    if t.startswith("."): continue
    cur[1].append(("INS", t))
tot = 0
for fn, items in funcs:
    if key and key not in fn: continue
    ins = []          # (text, op, valu_sgpr_writes, ws, labels_here)
    pend_labels = []
    for kind, t in items:
        if kind == "LABEL": pend_labels.append(t); continue
        op = t.split()[0]
        ws = 1
        if op == "s_nop":
            try: ws = int(t.split()[1], 0) + 1
            except Exception: pass
        w = set()
        if op.startswith("v_"):
            dst = t[len(op):].split(",")[0].strip()
            if re.match(r"^s(\[|\d)", dst): w = sregs(dst)
        ins.append((t, op, w, ws, pend_labels)); pend_labels = []
    label_at = {}
    for i, x in enumerate(ins):
        for lb in x[4]: label_at[lb] = i
    preds = collections.defaultdict(list)   # instruction index -> branch source indices (for block starts)
    for i, x in enumerate(ins):
        if x[1].startswith("s_cbranch") or x[1] == "s_branch":
            tgt = x[0].split()[-1]
            if tgt in label_at: preds[label_at[tgt]].append(i)
    def falls(i):   # does instruction i-1 fall through into i?
        if i == 0: return False
        return ins[i - 1][1] not in ("s_branch", "s_endpgm", "s_setpc_b64")
    viol = []
    for i, x in enumerate(ins):
        op = x[1]
        if not (op.startswith("buffer_") or op.startswith("global_") or op.startswith("scratch_")): continue
        args = x[0][len(op):].split(",")
        rd = set()
        for a in args[1:]: rd |= sregs(a)
        if not rd: continue
        # DFS back over predecessors: (index of the instruction to examine, wait states between it and the VMEM)
        stack = []
        def push_preds(j, d):
            if falls(j): stack.append((j - 1, d))
            for b in preds.get(j, []): stack.append((b, d))
        push_preds(i, 0)
        seen = set(); hit = None
        while stack and hit is None:
            j, d = stack.pop()
            if d >= 5 or (j, d) in seen or j < 0: continue
            seen.add((j, d))
            if ins[j][2] & rd: hit = (d, ins[j][0][:50]); break
            push_preds(j, d + ins[j][3])
        if hit: viol.append((x[0][:60], hit))
    if viol:
        tot += len(viol)
        print(f"{fn[:70]}: {len(viol)} violations")
        for v in viol[:6]: print("   ", v)
print(f"total violations: {tot}")
