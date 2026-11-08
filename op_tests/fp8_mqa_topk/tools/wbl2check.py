#!/usr/bin/env python3
"""wbl2check.py <file.s> [mangled-substring]: every 'buffer_wbl2' (agent/system release writeback) must be COMPLETE -
an 's_waitcnt vmcnt(k)' with k <= VMEM ops issued after it (in-order) - before the next VMEM/FLAT store or atomic on
every path. The compiler drops that wait when SIInsertWaitcnts' scoreboard is empty (it does not count buffer_wbl2);
the release flag can then become visible before the L2 writeback. CFG walk, path-sensitive for the structurizer's
flag branches like asmloadwaw.py."""
import re, sys
path = sys.argv[1]; key = sys.argv[2] if len(sys.argv) > 2 else ""
L = open(path).read().split("\n")
funcs = []; cur = None
for i, l in enumerate(L):
    m = re.match(r"^(_Z\S+):", l)
    if m: cur = [m.group(1), []]; funcs.append(cur); continue
    if cur is None: continue
    if l.startswith(".Lfunc_end"): cur = None; continue
    t = l.split(";")[0].strip()
    if not t or (t.startswith(".") and not t.endswith(":")): continue
    cur[1].append((i, t))
STORE = re.compile(r"^(buffer_store|buffer_atomic|global_store|global_atomic|flat_store|flat_atomic|scratch_store)")
VMEM = ("buffer_", "global_", "flat_", "scratch_")
tot = 0
for fn, ins in funcs:
    if key and key not in fn: continue
    label_at = {t[:-1]: k for k, (r, t) in enumerate(ins) if t.endswith(":")}
    bad = []
    for k, (r0, t0) in enumerate(ins):
        if not t0.startswith("buffer_wbl2"): continue
        stack = [(k + 1, 0, ())]; seen = set(); hit = None
        while stack and hit is None:
            j, after, flags = stack.pop(); fl = dict(flags); vcc = None
            while j < len(ins):
                sk = (j, after, tuple(sorted(fl.items())))
                if sk in seen: break
                seen.add(sk)
                r, t = ins[j]
                if t.endswith(":"): j += 1; continue
                op = t.split()[0]; args = [a.strip() for a in t[len(op):].split(",")]
                if op in ("s_endpgm", "s_setpc_b64"): break
                m = re.match(r"s_waitcnt\s.*vmcnt\((\d+)\)", t)
                if m and after >= int(m.group(1)): break
                if op == "s_mov_b64" and len(args) == 2 and args[0].startswith("s["):
                    fl[args[0]] = args[1] if args[1] in ("0", "-1", "exec") else None
                elif op.startswith("s_") and args and args[0].startswith("s["):
                    fl[args[0]] = None
                if op in ("s_andn2_b64", "s_and_b64") and args[0] == "vcc" and args[1] == "exec":
                    f = fl.get(args[2], None)
                    vcc = None if f is None else (("exec" if f == "0" else "0") if op == "s_andn2_b64" else ("0" if f == "0" else "exec"))
                    j += 1; continue
                if op == "s_branch":
                    tg = args[-1]
                    if tg in label_at: j = label_at[tg]; continue
                    break
                if op.startswith("s_cbranch"):
                    tg = args[-1]; take = fall = True
                    if op in ("s_cbranch_vccnz", "s_cbranch_vccz") and vcc is not None:
                        nz = vcc == "exec"; take = nz if op == "s_cbranch_vccnz" else not nz; fall = not take
                    if take and tg in label_at: stack.append((label_at[tg], after, tuple(fl.items())))
                    if not fall: break
                    j += 1; continue
                if op.startswith("v_") and "vcc" in t.split(",")[0]: vcc = None
                if STORE.match(op): hit = (r0 - ins[0][0], t0, r - ins[0][0], t[:70]); break
                if op.startswith(VMEM) or op.startswith("buffer_wbl2") or op.startswith("buffer_inv"): after += 1
                j += 1
        if hit: bad.append(hit)
    if bad:
        tot += len(bad)
        print(f"{fn[:70]}: {len(bad)} buffer_wbl2 not complete before the next store/atomic")
        for b in bad[:4]: print("   wbl2 @%d -> store @%d %s" % (b[0], b[2], b[3]))
print(f"total wbl2 violations: {tot}")
