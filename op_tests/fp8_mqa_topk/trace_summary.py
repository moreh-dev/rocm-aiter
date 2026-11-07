#!/usr/bin/env python3
"""Summarize a rocprofv3 kernel_trace.csv: per pipeline call (separated by the 512 MB fill kernel), kernel durations
and the gaps between consecutive kernels (us)."""
import csv, sys, re, statistics
rows = list(csv.DictReader(open(sys.argv[1])))
rows.sort(key=lambda r: int(r["Start_Timestamp"]))
calls, cur = [], None
for r in rows:
    name = r["Kernel_Name"]
    if "fill" in name.lower() or "FillFunctor" in name:
        cur = []; calls.append(cur); continue
    if cur is not None:
        cur.append((re.sub(r"\(.*", "", name)[:70], int(r["Start_Timestamp"]), int(r["End_Timestamp"])))
for i, c in enumerate(calls):
    if not c: continue
    t0, t1 = c[0][1], c[-1][2]
    busy = sum(e - s for _, s, e in c)
    print(f"call {i}: span {(t1 - t0) / 1e3:8.1f} us  kernels {len(c)}  busy {busy / 1e3:8.1f} us  gaps {(t1 - t0 - busy) / 1e3:6.1f} us")
    if i in (len(calls) - 1,) or i == 2 or i == 8:
        prev = None
        for n, s, e in c:
            gap = (s - prev) / 1e3 if prev else 0.0
            print(f"     {n:70s} {(e - s) / 1e3:8.1f} us  (gap before {gap:5.1f})")
            prev = e
