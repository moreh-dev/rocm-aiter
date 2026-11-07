#!/usr/bin/env python3
"""Summarize -Rpass-analysis=kernel-resource-usage remarks: one line per kernel."""
import re, sys, subprocess
raw = open(sys.argv[1]).read().splitlines()
rows, cur = [], None
for ln in raw:
    m = re.search(r"Function Name: (\S+)", ln)
    if m:
        cur = {"name": m.group(1)}; rows.append(cur); continue
    if cur is None: continue
    for key, pat in (("vgpr", r"\bVGPRs: (\d+)"), ("agpr", r"AGPRs: (\d+)"), ("sgpr", r"TotalSGPRs: (\d+)"),
                     ("scratch", r"ScratchSize \[bytes/lane\]: (\d+)"), ("occ", r"Occupancy \[waves/SIMD\]: (\d+)"),
                     ("lds", r"LDS Size \[bytes/block\]: (\d+)")):
        m = re.search(pat, ln)
        if m: cur[key] = m.group(1)
names = [r["name"] for r in rows]
try:
    dem = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
except Exception:
    dem = names
for r, d in zip(rows, dem):
    d = re.sub(r"\(.*", "", d).replace("fmt::", "")
    print(f"{d:60s} vgpr {r.get('vgpr','?'):>3} agpr {r.get('agpr','?'):>3} sgpr {r.get('sgpr','?'):>3} "
          f"scratch {r.get('scratch','?'):>4} occ {r.get('occ','?')} lds {r.get('lds','?')}")
