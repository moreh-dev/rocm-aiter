import re, sys
def regs(tok):
    out = []
    for a, b in re.findall(r"\bv\[(\d+):(\d+)\]", tok): out += list(range(int(a), int(b) + 1))
    out += [int(x) for x in re.findall(r"\bv(\d+)\b", tok)]
    return out
path, key = sys.argv[1], sys.argv[2]
L = open(path).read().split("\n")
start = next(i for i, l in enumerate(L) if re.match(r"^(\S*" + re.escape(key) + r"\S*):", l))
end = next(i for i in range(start, len(L)) if L[i].startswith(".Lfunc_end"))
ins = []
for l in L[start+1:end]:
    t = l.split(";")[0].strip()
    if not t or t.startswith(".") or t.endswith(":"):
        if re.match(r"^\.LBB", l): ins.append(("LABEL", l.split(":")[0]))
        continue
    ins.append(("I", t))
worst = {}
bad = []
for i, (k, t) in enumerate(ins):
    if k != "I" or not t.startswith("v_permlane32_swap"): continue
    ops = set(regs(t[len("v_permlane32_swap_b32"):]))
    ws, j = 0, i - 1
    while j >= 0 and ws < 3:
        kk, tt = ins[j]
        if kk == "LABEL": ws = 99; break   # block boundary: give up (conservative: report as unknown)
        op = tt.split()[0]
        if op.startswith("v_") and not op.startswith("v_cmp") and set(regs(tt.split(",")[0])) & ops:
            break
        if op.startswith("v_permlane32_swap") and set(regs(tt)) & ops: break
        ws += (int(tt.split()[1]) + 1) if op == "s_nop" else 1
        j -= 1
    worst[ws] = worst.get(ws, 0) + 1
    if ws < 2: bad.append((i, t, ins[j][1] if j >= 0 else "?"))
print(f"{path}: permlanes by wait states before (99 = block boundary within 3): {dict(sorted(worst.items()))}; violations: {len(bad)}")
for b in bad[:5]: print("   ", b)
