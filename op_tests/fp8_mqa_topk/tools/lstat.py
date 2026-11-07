#!/usr/bin/env python3
"""lstat.py <file.s> <mangled-substring> [--dump HEADER] [--blocks]
Loop statistics from a -save-temps device .s using LLVM's block annotations ("in Loop: Header=BBx_y" /
"=>This Inner Loop Header"): every loop's blocks (wherever they are placed) are summed. Reports per loop:
instruction classes, MFMA count, per-MFMA ratios, waits, nops, stores, estimated encoded bytes."""
import re, sys, collections

path, key = sys.argv[1], sys.argv[2]
L = open(path).read().split("\n")
start = next(i for i, l in enumerate(L) if re.match(r"^(\S*" + re.escape(key) + r"\S*):", l))
fname = L[start].split(":")[0]
end = next(i for i in range(start, len(L)) if L[i].startswith(".Lfunc_end"))
B = L[start + 1:end]

def mnem(l):
    s = l.strip()
    if not s or s.startswith((";", ".", "//")) or re.match(r"^\S+:", s):
        return None
    return s.split()[0]

def cls(m):
    if "mfma" in m: return "mfma"
    if m.startswith(("buffer_store", "global_store", "scratch_store")): return "vmem_st"
    if m.startswith(("buffer_", "global_", "scratch_", "flat_")): return "vmem_ld"
    if m.startswith("ds_"): return "lds"
    if m.startswith(("s_load", "s_buffer_load")): return "smem"
    if m.startswith("s_waitcnt"): return "wait"
    if m.startswith("s_nop"): return "nop"
    if m.startswith(("s_cbranch", "s_branch")): return "branch"
    if m.startswith("s_barrier"): return "barrier"
    if m.startswith("s_setprio"): return "setprio"
    if m.startswith("s_"): return "salu"
    if m.startswith("v_permlane"): return "permlane"
    if m.startswith("v_readfirstlane") or m.startswith("v_readlane"): return "valu"
    if m.startswith("v_"): return "valu"
    return "other"

def size(l):
    """encoded size estimate: 64-bit encodings for VOP3/VOP3P/MAI/MUBUF/DS/SMEM, 32-bit for e32/SOP, +4 B literal"""
    m = mnem(l)
    if not m: return 0
    s = l.strip()
    if m.startswith(("v_mfma", "buffer_", "global_", "ds_", "s_load", "s_buffer", "scratch_", "flat_", "v_pk_", "v_permlane")):
        return 8
    if m.startswith("v_"):
        b = 4 if m.endswith("_e32") else 8
        if m.endswith("_e32") and re.search(r",\s*0x[0-9a-f]+|\s0x[0-9a-f]+,", s):
            v = int(re.search(r"0x([0-9a-f]+)", s).group(1), 16)
            if v > 64: b += 4
        return b
    b = 4
    mm = re.search(r"0x([0-9a-f]+)", s)
    if mm and m.startswith(("s_mov", "s_add", "s_sub", "s_and", "s_or", "s_cmp", "s_min", "s_max", "s_mul", "s_lshl", "s_cselect", "s_xor")):
        if int(mm.group(1), 16) > 64: b += 4
    return b

# split into blocks
blocks = []   # (label, header_of_loop or None, is_header, lines)
cur = None
for l in B:
    m = re.match(r"^(\.LBB\S+):|^; %bb\.(\d+):", l)
    if m:
        cur = {"label": m.group(1) or ("%bb." + m.group(2)), "loop": None, "hdr": False, "lines": [], "depth": 0}
        mm = re.search(r"in Loop: Header=(BB\S+) Depth=(\d+)", l)
        if mm: cur["loop"] = mm.group(1); cur["depth"] = int(mm.group(2))
        blocks.append(cur)
        continue
    if cur is None:
        cur = {"label": "entry", "loop": None, "hdr": False, "lines": [], "depth": 0}
        blocks.append(cur)
    mm = re.search(r"in Loop: Header=(BB\S+) Depth=(\d+)", l)
    if mm and not cur["lines"]:
        cur["loop"] = mm.group(1); cur["depth"] = int(mm.group(2))
    if "=>This Inner Loop Header" in l or "=>This Loop Header" in l:
        cur["hdr"] = True
        cur["loop"] = cur["label"].lstrip(".L")
    cur["lines"].append(l)

loops = collections.defaultdict(list)
for b in blocks:
    if b["loop"]:
        loops[b["loop"]].append(b)

def stats(lines):
    c = collections.Counter(); ops = collections.Counter(); by = 0
    for x in lines:
        m = mnem(x)
        if m:
            c[cls(m)] += 1; ops[m] += 1; by += size(x)
    return c, ops, by

allc, allops, allby = stats(B)
print(f"{fname}\n  function: {sum(allc.values())} instructions, est. {allby} B")
for l in L[end:end + 40]:
    if re.search(r"; (codeLenInByte|NumVgprs|NumAgprs|TotalNumVgprs|NumSGPRsForWavesPerEU|ScratchSize|Occupancy|LDSByteSize):", l) or "codeLenInByte" in l:
        print("  " + l.strip())
res = {}
for h, bl in loops.items():
    lines = [x for b in bl for x in b["lines"]]
    c, ops, by = stats(lines)
    n = c["mfma"]
    if n < 8: continue
    res[h] = (c, ops, by, lines, bl)
    tot = sum(c.values())
    print(f"\n  loop {h}: {len(bl)} blocks, {n} MFMA, {tot} instructions, est. {by} B")
    print("   ", dict(sorted(c.items(), key=lambda t: -t[1])))
    print(f"    per MFMA: valu {c['valu']/n:.3f} salu {c['salu']/n:.3f} permlane {c['permlane']/n:.3f} nop {c['nop']/n:.3f} "
          f"vmem_st {c['vmem_st']/n:.3f} all-nonMFMA {(tot-n)/n:.3f}")
    waits = [x.strip().split(";")[0].strip() for x in lines if (mnem(x) or "").startswith("s_waitcnt")]
    print("    waits:", dict(collections.Counter(waits)))
    print("    nops:", dict(collections.Counter(x.strip() for x in lines if (mnem(x) or "").startswith("s_nop"))))
    print("    ops:", " ".join(f"{k}:{v}" for k, v in ops.most_common(60)))
if "--dump" in sys.argv:
    h = sys.argv[sys.argv.index("--dump") + 1]
    c, ops, by, lines, bl = res[h]
    for b in bl:
        print(f"==== block {b['label']}")
        print("\n".join(b["lines"]))
