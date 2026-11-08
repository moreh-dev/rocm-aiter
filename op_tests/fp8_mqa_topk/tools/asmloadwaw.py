#!/usr/bin/env python3
"""asmloadwaw.py <file.s> <mangled-substring>: inline-asm VMEM loads (buffer_load* ... offen right after the 's_nop 4'
our asm strings start with) write their destination VGPRs ASYNCHRONOUSLY, but the compiler treats the asm output as
written at the asm statement. Reports every compiler instruction that WRITES (register reuse: late data overwrites it)
or READS (copy / spill store of not-yet-loaded data) such a destination before the load is known complete.
Walks the CFG from each load (fallthrough + branch targets), counting VMEM ops: 's_waitcnt vmcnt(k)' completes the load
once >= k VMEM ops were issued after it (in-order). Path-sensitive for the structurizer's flag branches
('s_mov_b64 s[x:y], 0|-1|exec' ... 's_and(n2)_b64 vcc, exec, s[x:y]' ... 's_cbranch_vcc(n)z'), so merged tails are
not reported through infeasible paths. Reads by the instruction's own later asm loads (voffset reuse) are ignored."""
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
label_at = {t[:-1]: k for k, (r, t) in enumerate(ins) if t.endswith(":")}
VMEM = ("buffer_", "global_", "flat_", "scratch_")
def is_asm_load(k):
    t = ins[k][1]
    if not (t.startswith("buffer_load_dword") and "offen" in t): return False
    j = k - 1
    while j >= 0 and ins[j][1].endswith(":"): j -= 1
    return j >= 0 and ins[j][1].startswith("s_nop 4")
out = []
for k in range(len(ins)):
    if not is_asm_load(k): continue
    rel, t = ins[k]
    dst = vregs(t.split()[1].rstrip(","))
    stack = [(k + 1, 0, ())]; seen = set(); hit = None
    while stack and hit is None:
        j, after, flags = stack.pop()
        fl = dict(flags)
        vcc = None
        while j < len(ins):
            st_key = (j, after, tuple(sorted(fl.items())))
            if st_key in seen: break
            seen.add(st_key)
            r2, t2 = ins[j]
            if t2.endswith(":"): j += 1; continue
            op = t2.split()[0]
            args = [a.strip() for a in t2[len(op):].split(",")]
            if op in ("s_endpgm", "s_setpc_b64"): break
            m = re.match(r"s_waitcnt\s.*vmcnt\((\d+)\)", t2)
            if m and after >= int(m.group(1)): break
            # flag tracking (structurizer)
            if op == "s_mov_b64" and len(args) == 2 and args[0].startswith("s["):
                fl[args[0]] = args[1] if args[1] in ("0", "-1", "exec") else None
            elif op.startswith("s_") and args and args[0].startswith("s["):
                fl[args[0]] = None
            if op in ("s_andn2_b64", "s_and_b64") and args[0] == "vcc" and args[1] == "exec":
                f = fl.get(args[2], None)
                if f is None: vcc = None
                elif op == "s_andn2_b64": vcc = "exec" if f == "0" else "0"
                else: vcc = "0" if f == "0" else "exec"
                j += 1; continue
            if op == "s_branch":
                tg = args[-1]
                if tg in label_at: j = label_at[tg]; continue
                break
            if op.startswith("s_cbranch"):
                tg = args[-1]
                take = fall = True
                if op in ("s_cbranch_vccnz", "s_cbranch_vccz") and vcc is not None:
                    nz = vcc == "exec"
                    take = nz if op == "s_cbranch_vccnz" else not nz
                    fall = not take
                if take and tg in label_at: stack.append((label_at[tg], after, tuple(fl.items())))
                if not fall: break
                j += 1; continue
            if op.startswith("v_") or op.startswith("ds_"):
                vcc = None if ("vcc" in t2.split(",")[0]) else vcc
            if op.startswith(VMEM):
                if not is_asm_load(j):
                    rd = set()
                    for a in args: rd |= vregs(a)
                    if rd & dst: hit = (rel, t[:58], r2, "READ " + t2[:66], after); break
                after = min(after + 1, 64); j += 1; continue
            if op.startswith("s_"): j += 1; continue
            d = vregs(args[0]) if args else set()
            u = set()
            for a in args[1:]: u |= vregs(a)
            if d & dst: hit = (rel, t[:58], r2, "WRITE " + t2[:65], after); break
            if u & dst: hit = (rel, t[:58], r2, "READ " + t2[:66], after); break
            j += 1
    if hit: out.append(hit)
print(f"{L[st].split(':')[0][:60]}: {len(out)} asm-load destinations touched before the load is known complete (CFG)")
for o in out[:10]: print("  load @%d %s\n     %s @%d   (VMEM issued after the load: %d)" % (o[0], o[1], o[3], o[2], o[4]))
