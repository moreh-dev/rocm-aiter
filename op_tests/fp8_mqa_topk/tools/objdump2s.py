#!/usr/bin/env python3
"""objdump2s.py <llvm-objdump -d output> > out.s: AMDGPU disassembly -> the .s layout the ISA checkers read
(function labels, '.LBB<addr>:' labels at branch targets, branches rewritten to those labels, '.Lfunc_end')."""
import re, sys
lines = open(sys.argv[1]).read().split("\n")
funcs = []; cur = None
for l in lines:
    m = re.match(r"^([0-9a-f]+) <(\S+)>:", l)
    if m:
        cur = [m.group(2), int(m.group(1), 16), []]; funcs.append(cur); continue
    if cur is None: continue
    m = re.match(r"^\s+(\S.*?)\s*//\s*([0-9A-Fa-f]+):(.*)$", l)
    if not m: continue
    text, addr, rest = m.group(1).strip(), int(m.group(2), 16), m.group(3)
    tgt = None
    t = re.search(r"<(\S+)\+0x([0-9a-f]+)>", rest)
    if t and text.split()[0] in ("s_branch",) or (t and text.split()[0].startswith("s_cbranch")):
        tgt = cur[1] + int(t.group(2), 16)
    cur[2].append((addr, text, tgt))
out = []
for name, base, ins in funcs:
    targets = {t for _, _, t in ins if t is not None}
    out.append(f"{name}:")
    for addr, text, tgt in ins:
        if addr in targets: out.append(f".LBB{addr:x}:")
        if tgt is not None:
            op = text.split()[0]
            out.append(f"\t{op} .LBB{tgt:x}")
        else:
            out.append(f"\t{text}")
    out.append(f".Lfunc_end_{name}:")
print("\n".join(out))
