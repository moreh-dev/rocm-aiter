# Fused FP8 MQA logits + top-k on gfx950: progress log

Design and method: [DESIGN.md](DESIGN.md). Everything runs on MI355X amd-smi GPU 7 (HIP 4) through `gpu7.sh`,
container `minh-pham-dev-kernel` (ROCm 7.2.3, torch 2.12, vLLM 0.30.0).

Files:
* `csrc/kernels/fp8_mqa_topk/` - device code (scan kernel family, tau / select / fallback) + `fp8_mqa_topk_capi.cu`
  (extern "C" launchers used by the dev harness).
* `op_tests/fp8_mqa_topk/` - `build.sh`, `gpu7.sh`, `fmt_harness.py` (data, exactness check, timing, reference
  paths), test and bench scripts, `results/` (JSON).

## 2026-11-07 - setup and reference numbers

* Branch `tmp/dev-fuse-logits-topk` = `d35d86a38` (v35wes).
* Nam's fuse48 sources extracted from `MV-5041-new-update.tar.gz` and rebuilt unchanged in
  `experiments/fuse_topk_ref/` (v35a, v35h, v48, fuse48, topk3 with `-O3 -fno-honor-nans`; v35w, v35hw with the
  v35wes module flags).
* Nam's own benches reproduced on this box (1 layer, top-512, 8 chunks of ISL 128,000):

| path | ISL 128K (ms) | Nam's number |
|---|---|---|
| base = v35a + vLLM top-k | 12.07 / 12.24 | 12.24 |
| baseu = v35w + vLLM top-k | 11.11 | - |
| hi16 = v35h + topk3 | 9.87 / 9.92 | 9.87 |
| hi16u = v35hw + topk3 | 9.01 | 9.02 |
| fuse48 (v35a based) | 10.19 | 10.25 |

  fuse48 parts at 16384 x 28672: sample + tau 0.207 ms, v48 scan 1.624 ms, select 0.147 ms (0 wrong rows, 0
  fallback rows, ~1.14K candidates per row).

## 2026-11-07 - step 0: fuse48 ported onto v35wes (7 kernels + 6 torch ops)

Code: `fmt_scan.cuh` (scan family = copy of mqa_v35wes::kernel with MODE_DENSE / MODE_EMIT), `fmt_select.cuh` (tau,
select, fallback), `fp8_mqa_topk_capi.cu`; driver `fmt_steps.Step0`.

* DENSE copy: 239 VGPR, 0 scratch, occupancy 2 - identical to the shipped kernel; bit-exact with the aiter op on every
  test shape (valid windows).
* tau now from fp32 samples (exact target-th largest sample, 32-bit keys) instead of 16-bit buckets; fallback
  recompute rewritten to be bit-identical with v35wes (plain MFMA + scalar even/odd fma chains). Forced-fallback tests
  (tau = +inf, every row recomputed) are exact.
* **Lesson: a register spill in the scan kernel is a correctness bug, not just a slowdown.** The first EMIT port
  (Nam's per-lane tau / window VGPRs) needed 256 VGPR + 20 B/lane scratch; scratch loads/stores are VMEM ops and
  count in vmcnt, so v35wes's hand-placed `s_waitcnt vmcnt(N)` (which assumes exactly 1 K load + 2*SUB stores per
  stage) let MFMAs read K registers before their loads returned: wrong top-k on the partial last row block
  (masked-only pipeline), e.g. 9 of 777 rows. Fix: wave-uniform per-row tau and window bounds stay in SGPRs, compared
  twice per condition and merged by halves in SALU -> 255 VGPR, 0 scratch, all tests exact. Every build must be
  checked for `scratch 0` (build.sh prints it).
* Exactness (`test_exact.py --step 0 --quick`): 14/14 cases, 0 wrong rows (random, multi-request, non-causal, N <=
  CAP, M not a multiple of 32, SplitN up to 14, stress sets offset / ties / periodic / increasing, forced fallback).
  The `periodic` stress set (keys n % 16 == 8 boosted, invisible to a stride-16 sample at offset 0) gives ~4.9K
  candidates per row instead of ~1.1K - the strided sample's known weakness; still exact.
* Timing, ISL 128K (8 chunks, ms; `results/bench_step0_128k.json`):

| baseu | hi16u | fuse48 (Nam) | step 0 |
|---|---|---|---|
| 10.902 | 8.984 (-17.6 %) | 10.191 (-6.5 %) | 9.698 (-11.0 %) |

* Parts at 16384 x 28672 (`parts.py`, ms): aiter dense 1.365, our dense 1.372, **EMIT scan 1.501 (+10 % over dense)**,
  torch prep + sample + tau 0.227, select (4 kernels) 0.142; total 1.857 (hi16u 1.767).
  -> The remaining gap to hi16u is the emission cost in the scan epilogue (~0.14 ms) plus the separate sample
  (~0.23 ms) and select (~0.14 ms) passes. Steps 1-4 remove the passes; the emission epilogue needs its own VALU work.

## 2026-11-07 - step 1: 4 launches (sample+tau kernel, EMIT, select, 1 fallback kernel)

* `MODE_SAMPLE` scan: strided DENSE pipeline over the sample keys j*S (per-lane voffset (t*S)*128 + piece*16,
  soffset token_base*S*128, window [ceil(ks/S), ceil(ke/S))), SplitN = 1, then tau of the wave's 4 rows in the kernel
  tail; resets the fallback counter. Gate: the stored samples are bit-identical to the dense logits at j*S.
* `fallback1_kernel`: the 3 fallback kernels merged; per-row completion counters (acq_rel, agent scope) let the wave
  that completes a row's last 32-token block run its select and reset the counter. A first version overran its
  scratch (256 waves x one-wave-per-row path) and faulted the GPU process; now only 64 waves take that path.
* New gates in `test_exact.py` (review findings): EMIT with tau = -inf emits every valid (m, n) exactly once with
  dense bits (SplitN up to 16), strided samples == dense, odd windows (ks < 0, ke > N, ke < ks), poisoned workspaces.
  Step 0 and step 1: all cases + gates pass.

**Why step 1 is not faster than step 0 (measured):**
* Launches are free here. A rocprofv3 kernel trace of the pipelines shows ~0 us between back-to-back kernels queued
  from ctypes (the GPU never waits for the host); only the 6 tiny torch ops of step 0 can leave gaps, and only when
  the GPU is idle when they are enqueued (in the benchmark and in serving they queue behind other work: ~23 us of GPU
  time per chunk).
* What moves into the sample kernel's tail gets SLOWER: tau runs at the scan kernel's occupancy (1 WG / CU, 2 waves /
  SIMD) instead of 7 waves / SIMD in its own kernel, and the per-row selection is latency-bound. The strided sample
  pass is also ~10 % slower than the dense kernel on a compacted K[::S] (88 vs 79 us at S = 32).
* Investigating the tail found three real problems, all fixed (they also sped up step 0's tau and select):
  1. conditional loads written as atomics (`c < b ? load : 0`) compiled to one exec branch + flat_load + vmcnt(0) per
     load -> made unconditional (clamped index, masked after);
  2. ballot + s_bcnt1 per register serialized every compare through VCC -> per-lane VALU count + DPP reduction;
  3. the 20/32-step bitwise search costs ~1000 VALU per row (PMC: tail = 16.2 M VALU) -> relative two-pass radix
     (8 bits below the common prefix of the row's min/max sample, then 8 more inside the boundary bin; resolution
     ~spread / 2^16 independent of the score offset): 8.7 M VALU, tail 78 -> 63 us at S = 32.
* S = 32 beats S = 16 for every fused variant (sample + tau cost ~1/S, candidates grow only ~target * S).

ISL 128K (ms, same run, `results/bench_step1b_128k.json`):

| baseu | hi16u | step 0 S16 | step 1 S16 | step 0 S32 | step 1 S32 |
|---|---|---|---|---|---|
| 10.918 | 8.998 | 9.171 | 9.293 | **8.952** | 9.032 |

Per 16384 x 28672 chunk at S = 32 (us): sample + tau 151 (step 0: 79 + 50 + torch 23), EMIT 1500, select ~130.

## 2026-11-07 - design review + emission epilogue (shared by every step)

Three independent reviewers (correctness / memory model, gfx950 perf + ISA prototypes, step-4 feasibility vs PR 60618)
read DESIGN.md and the sources; full reports in the session transcript, key findings:
* Memory model: on gfx950 an agent-scope release does NOT wait for inline-asm VMEM stores (the candidate stores); every
  hand-off needs an explicit `s_waitcnt vmcnt(0)` in every wave -> barrier -> one acq_rel agent atomic -> broadcast ->
  acquire fence (`buffer_inv sc1`) in every reading wave. sc0 = workgroup scope, sc1 = agent scope.
* Steps 3/4: empty row blocks and zero-work splits must still take part in the completion protocol.
* Expected value of persistence on gfx950: ~1-2 %. One 8-wave WG per CU (VGPR-bound) means nothing overlaps inside a
  CU, the PR does not overlap its select with the next scan either, launch gaps are ~0, and the static dispatch with the
  makespan model is already at the work/256 bound for full chunks.
* Emission codegen: Nam's literal `?:` port becomes exec branches that split the sched_barrier-pinned sub-tiles; my
  SGPR-tau fix ("EV5") avoided the spill but costs +12.5 VALU per pair (the kernel is VALU-issue-bound). Branch-free
  "EV4" (per-row SGPR counters, one full-wave mbcnt with a per-half correction, interleaved pair slots, overflow
  dropped by the buffer range check) costs +7.5.

Applied:
* EMIT = EV4: 246 VGPR at PF 4, 236 at PF 2, 0 scratch, no exec branches in the loops, 0 permlane hazard violations
  (`tools/permcheck.py`), vmcnt ladder unchanged. Clean loop 18.875 VALU/MFMA (dense 17.0, previous EMIT ~20).
  **EMIT scan overhead over dense: +10.3 % -> +3.6 %** (16384x28672: 1.412 vs 1.363 ms; 16384x16384 +3.8 %,
  13312x32000 +3.1 %).
* PF = 2 costs nothing (dense PF2 == PF4 within 0.2 %, bit-exact; EMIT +0.1-0.5 %) and frees 10 VGPRs: the fused
  kernels of steps 2-4 use PF 2.

ISL 128K (ms, `results/bench_ev4_128k.json`): baseu 10.936 | hi16u 8.994 | **step 0 (S32) 8.554 (-21.8 %)** |
step 1 (S32) 8.629 (-21.1 %).

## 2026-11-07 - steps 2-4: fused kernel, in-kernel select, persistent single launch

Code: `fmt_fused.cuh` (`fused_kernel<FUSE_SELECT, PERSIST, REVERSE_ROWS>`, one item body for all three steps),
launchers `fmt_step2/3/4` in `fp8_mqa_topk_capi.cu`, drivers `fmt_steps.Step2/3/4`. All use PF 2 and S = 32.

| step | launches | kernels |
|---|---|---|
| 2 | 3 | fused (sample + tau + EMIT) -> `select_kernel<32>` -> `fallback1_kernel` |
| 3 | 2 | fused + in-kernel select (last split of each row block) -> `fallback1_kernel` |
| 4 | 1 | persistent fused: claim loop -> global barrier -> in-kernel fallback drain -> last WG resets counters |

* **Row-wide tau inside the scan kernel.** Split 0 of each row block runs the sample pass through the SAME EV4
  emission pipeline (runtime key stride `kvm = S`, records `fp32 | idx << 32` into `smp`), computes tau per row
  (`wave_tau_ld`) and publishes it with a release store of the launch epoch into `tau_ready[bx]`; the other splits
  poll it and acquire. Split-0 work items are dispatched (grid) or claimed (persistent) before any split 1, so the
  wait cannot deadlock. The first version used per-split taus (each split targets the full K: SplitN x ~1.4K
  candidates per row; LDS survivor compaction made it worse) and was replaced.
* tau = -inf only when nvalid <= K (steps 0/1: nvalid <= CAP). Part of step 2's lead at 16384 x 4096 comes from this
  rule: steps 0/1 emit every key of a row with <= 4096 valid keys.
* Step 3: after its EMIT pipelines each WG does vmcnt(0) -> barrier -> acq_rel `block_done[bx]` (self-resetting);
  the last split selects the block's 32 rows (`fused_select_row`, a `__noinline__` call, 32 candidate registers per
  lane, radix path beyond). Failing rows go to the fallback list.
* Step 4: grid = min(256, items). Items are claimed with a relaxed atomic, split-major, heaviest row blocks first.
  The item body sits behind a `__noinline__` call: inlined, the persistent loop spilled inside the pipelines. A
  global barrier is followed by the in-kernel fallback drain (8 waves recompute a row into a per-WG scratch row, then
  `wave_select_dense`); the last WG out resets the counters and the fallback count. Per-WG sample slots.

Bugs found on the way (all fixed, all tests exact):
1. **GPU memory fault in step 2:** an inline-asm `buffer_load` directly after a `v_readlane` (SGPR spill reload)
   needs 5 wait states, and the hazard recognizer does not see inside inline asm. Found with rocgdb (`set amdgpu
   precise-memory on`). Fix: `s_nop 4` inside the asm string; `tools/sgprhaz.py` checks the ISA (0 violations).
2. Relative radix select returned wrong rows when a pass's bits overlapped the fixed prefix: it now tracks
   `top`/`sh` explicitly.
3. The `__noinline__` item body produced "invalid operand" in the asm loads: its scalar arguments arrive in VGPRs, so
   they are now made uniform with `readfirstlane`.
4. **Long context: tau over more than 2048 samples** took a radix path with one dependent global load per step.
   Inside the 1-WG/CU kernel this took 251 us per block (576 x 250000 chunk: 2.88 ms vs 0.64 for step 0). Now every
   lane keeps its 8 largest sample keys from batched loads, and the relative radix runs on the 512 kept keys. They
   are a subset of the samples, so tau stays a lower bound of the target-th largest sample and exactness is
   unchanged. A 2048-sample cap (adaptive stride) was tried first and rejected: tau got too coarse at N >= 196608
   (up to 6074 candidates/row > CAP 4096, 33 / 841 fallback rows, 100-367 ms per chunk).
5. **Spin-waits slowed the producer.** An ACQUIRE atomic load in the poll loop emits `buffer_inv sc1` on every poll.
   With 13 x 18 WGs waiting on 18 split-0 WGs (576 x 250000) the producers ran 2-4x slower. Polls are now relaxed,
   with one acquire fence after the WG barrier: sample 352 -> 175 us, tau 251 -> 66 us, kernel 1.18 -> 0.75 ms
   (`ts_probe.py`, a `-DFMT_TS` build that timestamps each WG phase).
6. 48 candidate registers in the in-kernel select (to keep 2-3K-candidate rows off the radix path) raised the call
   frame from 316 to 644 B and made step 3 30 % slower on 16384 x 4096. Reverted to 32; the standalone select keeps a
   48-register path (204 VGPR, 0 scratch).

Gates on every build: `scratch` of the fused kernels only outside the hand-vmcnt pipeline regions
(`tools/scratchloc.py`: inside 0), `tools/sgprhaz.py` 0 violations, `test_exact.py --quick` 16/16 for steps 0-4
(random, multi-request, non-causal, odd windows, M % 32 != 0, SplitN up to 14, offset / ties / periodic /
increasing, 1024 x 196608 increasing, forced fallback, poisoned workspaces), `repeat_check.py 2|3|4` (30 back-to-back
launches over 3 alternating shapes: self-resetting counters and epochs).

### Results (1 layer, top-512, ms; `results/bench_steps_128k_fixed.json`, 3 rounds, after the race fix below)

ISL 128,000 (8 chunks):

| baseu | hi16u | fuse48 | step 0 | step 1 | **step 2** | step 3 | step 4 |
|---|---|---|---|---|---|---|---|
| 10.924 | 9.007 (-17.5 %) | 10.187 (-6.7 %) | 8.578 (-21.5 %) | 8.651 (-20.8 %) | **8.337 (-23.7 %)** | 8.581 (-21.4 %) | 9.416 (-13.8 %) |

Per chunk, step 3 - step 2 = +14-44 us (+68 us on the 3-split 13312 x 32000 chunk); step 4 - step 3 = +79-141 us.
(The pre-fix run, `results/bench_steps_128k_final.json`, is the same within noise: step 2 8.333, step 3 8.563, step 4
9.450 ms.)
0 wrong rows and 0 fallback rows on every chunk. Steps 2-4: 1.16-1.44K candidates per row on average (max 2457);
steps 0/1 emit every key of a row with <= CAP = 4096 valid keys (tau = -inf on the 16384 x 4096 chunk).

Long context and small M (`results/bench_long_shapes_fixed.json`):

| chunk (SplitN) | baseu | hi16u | step 0 | step 2 | step 3 | step 4 |
|---|---|---|---|---|---|---|
| 16384 x 65536 (1) | 4.841 | 4.136 | 3.654 | **3.642** | 3.656 | 3.896 |
| 16384 x 262144 (1) | 19.081 | 19.094 | **14.107** | 14.328 | 14.363 | 15.154 |
| 2048 x 131072 (4) | 1.339 | 1.162 | **0.973** | 1.045 | 1.104 | 1.205 |
| 576 x 250000 (14) | 0.788 | 0.682 | **0.565** | 0.741 | 0.801 | 0.905 |

### Why step 3 and step 4 are not faster (measured)

* **Launches cost nothing here.** rocprofv3 shows ~0 us between back-to-back kernels of one pipeline, so fusing a
  launch only pays if the work it moves gets cheaper. On gfx950 it gets more expensive.
* **The in-kernel select is slower than its own kernel.** The fused kernel needs all 256 VGPRs (1 WG/CU), so the
  select cannot be inlined (that spilled inside the hand-vmcnt pipelines, which gives wrong results, see step 0). Behind
  a call it pays the call frame (316 B/lane scratch) and runs on the 4 rows of each wave in turn. Probe: 61-69 us per
  32-row block at every N, i.e. 122-138 us per CU per chunk. The standalone `select_kernel<32>` (204 VGPR, no
  frame) does the same rows in less time: step 3 - step 2 = +14-44 us per 16384-row chunk of the 128K run (equal
  within noise at 64K and 262K pools). At 16384 x 4096 the in-kernel select (61 us) costs more than the block's whole
  EMIT scan (55 us).
* **Persistence has nothing to win and costs per item.**
  * Static dispatch with the makespan model is already at the work/256 bound for full chunks.
  * With 1 WG per CU, nothing overlaps inside a CU (PR 60618 does not overlap its select with the next scan either).
  * Costs: the item body behind a call (callee-saved VGPR save/restore per item; 540 B scratch frame), the claims,
    and the end barrier.
  * PMC step 4 vs step 3: VALU 608 M -> 656 M, SALU 79 M -> 111 M, VMEM reads +4.8 %, wave cycles +24 %. The hot
    loops are the same (19.0 vs 18.875 VALU/MFMA).
* **Few row blocks, long rows (SplitN > 1):** split 0 samples the whole row (1/S of a full-row scan = SplitN/S of a
  split's work) and computes tau while the other splits wait. At 576 x 250000 (14 splits) that is 175 + 66 us before
  a 482 us EMIT, so step 2 is 0.741 vs 0.565 ms for step 0 (separate sample + tau kernels spread over all CUs). At
  SplitN = 4 (2048 x 131072): +7 %; at SplitN <= 3: within 0-1 %.

### Status vs the goal

* Step 4 (one launch, exact) is implemented, but it is **not** faster. 9.42 ms at 128K: -13.8 % vs baseu, behind
  hi16u, step 0 and steps 1-3. The reasons are above, and none of them is a missing optimization of the persistent
  loop.
* **Best path: step 2** (3 launches), 8.34 ms at 128K: -23.7 % vs baseu, -7.4 % vs hi16u. It also wins at 64K pools.
  Step 0 is 1.6 % faster at 262K pools and leads on small-M chunks (SplitN >= 4).
* A single launch at step-3 speed is possible without persistence: the selecting WG recomputes its rare failing rows
  itself. That only removes the free `fallback1` launch, so it was not built.
* Open, not done:
  1. Distributed sampling for SplitN > 1. Each split samples its own stage range into its own segment; chunks are
     claimed with epoch-valued flags (work stealing, so nothing assumes co-residency); the last finisher computes
     tau. Expected: 576 x 250000 0.74 -> ~0.52 ms.
  2. A faster in-kernel select (2 rows per wave in flight, no call frame) is the only lever left for step 3/4.

## 2026-11-07 - intermittent step-2 race: inline-asm prefetch registers reused before the load landed (fixed)

* **Found:** after the review hardening (acquire fence after the polls, explicit wait before the persistent barrier)
  one quick-test case failed once: step 2, (1000, 3000, rand, causal), SplitN 8, 2 wrong rows, 0 fallback rows.
  Repeating: step 2 failed 1 of 6 quick suites; `repro_flaky.py` (fresh workspace per launch, poisoned) 16 of 300
  launches (0.7-5 % across runs); steps 3 and 4 0 of 300. So the earlier "all tests exact" held for each run of the
  suite, but step 2 had a ~1 % intermittent failure on this shape.
* **Where:** always row block 31 (the only partial block: its whole range runs through the multi-stage masked
  pipeline), wave 0 (the only wave issuing the inline-asm scale loads), the last 32-token sub-tile of a pipeline run
  (split 7, once split 0's EMIT). Three symptoms: that sub-tile's candidate slots still poisoned while the count was
  right; records of another split's key range in the segment; the sub-tile's upper-row candidates not counted.
  `repro_fused.py` (fused kernel alone, finite tau): 4 of 600 launches; with tau = -inf (every key stored) 0 of 400.
* **Ruled out, each by experiment:**
  * The 8 B spill: built without it (`-DFMT_NO_TAU_STREAM`), still failing.
  * The poison fill: it runs on the same stream, and the failures persist with a sync after it and without poison.
  * A descriptor-SGPR WAR after the last store: waiting before the reloads did not fix it.
  * VALU-SGPR -> VMEM hazards: the new CFG-aware `tools/sgprhaz2.py` reports 0.
  * Permlane hazards: 0.
  * Unexpected VMEM inside the pipeline regions (`tools/vmemloc.py`): only out-of-line blocks whose loads are waited
    for immediately.
* **Cause:** the last stages issue clamped prefetches as inline-asm loads whose results are never used. The compiler
  treats an asm output as written when the asm statement executes, so it saw the destination VGPR as free and reused
  it. ISA: `buffer_load_dword v1 ... offen` (the scale prefetch), then `v_add_f32 v1, ...` in the deferred tail, with
  no covering vmcnt on that path. The hardware writes v1 when the load returns; when that came after the reuse, the
  scale value replaced a tail value of the scale-producer wave.
* **Fix:** `FLUSH_PEND` now starts with `asm volatile("s_waitcnt vmcnt(0)" :: "v"(pf_reg0), "v"(pf_reg1),
  "v"(pf_sc0), "v"(pf_sc1) : "memory")`, in both the fused and the scan kernels. The registers stay allocated to the
  in-flight loads until they have landed.
  * New gate `tools/asmloadwaw.py`: walks the CFG from every inline-asm load to the vmcnt that covers it and reports
    any write to its destination.
  * Before the fix: 1 hit in each step-2 fused kernel and in every `scan_kernel` mode (EMIT, DENSE, SAMPLE).
  * After the fix: 0 in every kernel. Register and scratch use are unchanged.
* **Verified:**
  * `repro_fused.py` 0 of 1200 and `repro_flaky.py` (full step 2) 0 of 1200, vs 4-16 failures per 300-600 launches
    before.
  * `repro_flaky.py` for every step: 0 of 600 on 1000 x 3000 (steps 0-4) and 0 of 400 on 777 x 9000 (steps 2-4).
  * `test_exact.py --quick` 16/16 for steps 0-4, the full suite 28/28 for step 2, `repeat_check.py 2|3|4` OK.
  * All ISA gates clean on every kernel: asmloadwaw 0, sgprhaz2 0, permcheck 0, scratch inside pipelines 0.
  * Timing unchanged (the tables above are post-fix).
* **The shipped v35wes kernel** (`module_fp8_mqa_logits.so`, from `csrc/kernels/fp8_mqa_logits.cu`) has the same
  pattern in its `CLEAN_LOGITS = false` instances (NUM_WARPS 8 and 4).
  * The scale prefetch's destination v1 is set to `OOB_VOFF`, merged into the voffset of the drain's final pair
    stores (`v_cndmask`), and read by `buffer_store_dword` 2-4 instructions later, before any covering wait (twice).
  * If the late load lands inside that window, the scale's bit pattern (~1e9 for normal scales) becomes the voffset
    of every lane of that store. That is beyond num_records (2N*4), so the last sub-tile's logits of one row pair
    are silently not written.
  * Not reproduced: sentinel-filled `out=`, 9 shapes, ~4,700 launches (`repro_shipped.py`), 0 in-window differences
    and 0 stray writes. The window is only a few instructions wide.
  * The same one-line drain wait fixes it. Not applied here: it is the shipped kernel, outside this task.

## 2026-11-08 - plan steps 1-3: fair baseline, 4-row tau, distributed sampling (branch `tmp/more-dev-fuse-logits-topk`)

### 1. Measurements first

* **Fair baseline.** `step0ks32` is step 0 with step 2's tau rule (tau = -inf only when nvalid <= K).
  * ISL 128K: step 0 8.614 -> 8.481 ms. Step 2 (8.343) led a fair step 0 by 1.6 %, not 3.1 %: the rule was worth
    0.13 ms, almost all on the 16384 x 4096 chunk (`results/m1_*`).
* **Step 2 phases** (`ts_probe.py`, us per row block, before this round):

| chunk (splits) | split-0 sample | tau | scan per split |
|---|---|---|---|
| 16384 x 28672 (1) | 36 | 29 | 747 |
| 16384 x 262144 (1) | 224 | 116 | 6599 |
| 2048 x 131072 (4) | 108 | 42 | 950 |
| 576 x 250000 (14) | 181 | 70 | 484 |

* **Step 0's separate passes** (`parts.py --S 32 --krule`):
  * 262K: sample 0.440 + tau 0.153 ms per chunk.
  * 2048 x 131072: sample + tau + torch prep 0.085 ms.
  * 576 x 250000: sample + tau + torch prep 0.072 ms.
  * At SplitN 1, step 2's in-kernel sampling (2 x 224 us per CU) costs the same as step 0's sample kernel; tau
    is the excess. With many splits, split 0 serializes sampling + tau for the whole row.

### 3. Faster in-kernel tau (commit 5c0b5ec1c)

* `wave_tau_multi`: the 4 rows of a wave in lockstep.
  * Each lane keeps its 8 largest keys per row, then the relative two-pass radix runs on the 4 rows together (4 x
    256-bin LDS histograms per wave). The kept keys are a subset of the samples, so tau stays a valid lower bound.
* `wave_find256`: a DPP scan with readlane broadcasts replaces 6 dependent `__shfl_up` round trips. It helps at
  the fused kernel's 2 waves / SIMD. It was first applied everywhere, which slowed step 0's high-occupancy tau
  kernel on long rows (step 0's sample + tau + prep 70.5 -> 97.0 us on 576 x 250000; the "fair step 0" runs m3-m7
  carry this). Now chosen per caller: the standalone tau and select kernels use the shuffle version again (see the
  final table).
* **Pitfall:** the first version used 64-bit per-lane addresses, and with ~150 VGPRs live the compiler serialized
  every load (`global_load; s_waitcnt vmcnt(0)`): 2x SLOWER. Fixed with a uniform row base + 32-bit lane offset +
  immediates, no clamping (8192 records of slack), and 8 loads per row per round.
* Tau per block after 5c0b5ec1c: 16384 x 28672 29.0 -> 16.4 us; 16384 x 262144 116 -> 81 us (1.4-1.8x, short of
  the planned 2-3x). ISL 128K: step 2 8.343 -> 8.254 ms (fair step 0 8.422).
* **Reaching 2-3x (2026-11-08, later).** Timestamps inside `wave_tau_multi` (`-DFMT_TS`) showed the long-row tau is
  spent streaming the samples (75.9 of 82.3 us at 262K):
  * neither fewer VALU nor more loads in flight helped: a med3 insertion halved the per-sample VALU, and 16 loads
    per row and round halved the rounds, but streaming stayed at 75.9 / 68.8 us (28K: 9.0 us either way).
  * The cost tracks bytes: 2 MB per row block at 262K (32 rows x 8192 u64 records), 229 KB at 28K, i.e. ~27 GB/s -
    a CU's share of memory bandwidth while every other CU streams K.
  * So the sample pass of the plain variant now stores only the 4-byte value (a compile-time store flavour of the
    same pipeline: one store per pair, vmcnt counts unchanged), pair-interleaved, so a 128-byte line carries 32
    useful values instead of 16. The insertion uses floats and one `v_med3_f32` per slot (b0 = max(a0, x),
    bt = med3(a(t-1), at, x)), no per-sample key conversion.
  * Tau per block: **16384 x 28672 29.0 -> 14.0 us (2.1x), 16384 x 65536 36.0 -> 16.2 us (2.2x), 16384 x 262144
    116.2 -> 32.7 us (3.6x)** (streaming at 262K 75.9 -> 26.7 us; final code, re-probed after the permlane fix).
  * The gate caught a hazard on the way: with the value-only flavour also in the distributed and persistent kernels,
    the register allocator copied (`v_mov`) or spilled (`scratch_store`) in-flight inline-asm prefetch registers
    (`asmloadwaw.py`: 2 hits each). Those variants keep the u64 records (their tau computes ~1 row per wave over
    several segments, not the 4-row lockstep); all gates are clean.
* **A real permlane hazard in the sample flavour (found by `tools/permcheck.py`, fixed).** In the value-only build
  the compiler copied an inline-asm operand right before the swap: `v_mov_b32 v38, v238` directly followed by
  `v_permlane32_swap_b32 v38, v34`, 0 wait states where gfx950 needs 2 (VALU write -> permlane read). The swap
  helper had no `s_nop` there because the tuned scan schedule never put a VALU write of the operand that close, and
  the hazard recognizer does not look inside inline asm. Every value-only kernel had this one site, in the deferred
  pair swap of `SUBTILE`.
  * **It corrupted samples on the hardware.** The `m9` runs (value-only flavour, hazard present) show step 2 with
    more candidates than the u64 path on every value-only launch, falling off as ~1/N: 16384 x 8192 1417 -> 1490
    (+5.2 %), 16384 x 16384 +1.9 %, 16384 x 28672 +1.0 %, 16384 x 262144 +0.1 %; plus 381 more tau = -inf rows at
    16384 x 4096. Exactness held (a tau that is too high is caught by the count check, one that is too low only
    adds candidates), so the tests could not see it.
  * Fix: in the sample flavour every permlane gets the full 2 wait states (`s_nop 1`, all three swap sites of
    `run_pipeline`); the scan flavour's code is unchanged. With the fix (`m10`) step 2's candidate counts equal
    the u64 path's exactly on every chunk.
  * The 16384 x 4096 slowdown that had limited the value-only flavour to N >= 16384 (0.281 -> 0.300 ms in `m9`) was
    this hazard: on the fixed build value-only samples are equal or faster at short N too (16384 x 4096 0.278 vs
    0.281, 16384 x 8192 0.514 vs 0.516, 16384 x 12288 0.733 vs 0.736 ms; `results/m10_valsall_check.json` vs
    `results/m10_short_vals16k.json`). So the plain variant always uses value-only samples: no threshold, 4 fewer
    kernel instantiations.

### 2. Distributed sampling for SplitN > 1

**Design (as planned).**
* Chunk c = split c's key range, sampled into its own segment (`[c][Mp][NSBc]`), with counts in `scnt[c][m]`.
* A WG claims its own chunk first, then any unclaimed one (atomic exchange of the launch epoch). It only ever waits
  for chunks a running WG has claimed, so nothing assumes co-residency and it cannot deadlock.
* Once all chunks of the block are done, every wave claims R rows (R = clamp(ceil(32 / (SplitN * 8)), 1, 4)),
  computes their tau over all segments (32 loads in flight), publishes `A.tau` plus a row flag, then waits for its
  own 4 rows.
* All flags are epoch-valued and never reset (the host re-zeroes them only on epoch wrap).
* Forced-split tests with grids far larger than the GPU make WGs steal chunks of WGs not yet dispatched (dbg hooks
  8 / 16).

**Problems found and fixed:**
1. **Steal decision computed per wave.** Each wave read the claim flags itself and chose a different next chunk.
   Waves skipped the chunk wave 0 had claimed, so 2324 of 4096 rows ended up with tau = -inf (unwritten counts),
   and the WG's barriers fell out of step, a deadlock at 16 splits. Found with `cmp_tau.py` / `dbg_tau.py` (per-row
   claimer, target, kept count, count table). Wave 0 now decides and broadcasts through LDS.
2. **Static review workflow** (4 lenses: memory model, progress/uniformity, exactness, host/API; every finding
   handed to an adversarial verifier; 4 confirmed, 0 refuted).
   * **The compiler drops the `s_waitcnt vmcnt(0)` after `buffer_wbl2 sc1` in agent releases.** SIInsertWaitcnts
     does not count wbl2, so with an empty scoreboard the flag store can overtake the L2 writeback.
     * Two sites in step 4's item body: the chunk-done release, and the long-standing `block_done` RMW (step 4
       with SplitN > 1 could read another split's stale candidates).
     * Fix: `FMT_RELEASE_AGENT()` (inline asm: `s_waitcnt vmcnt(0) lgkmcnt(0); buffer_wbl2 sc1; s_waitcnt
       vmcnt(0)`) before every flag store / RMW, at all 8 release sites: the fused item body (chunk-done, row-tau,
       `tau_ready`, `block_done`), the persistent kernel (`ctr[1]`, `ctr[3]`) and fallback1 (per-row done,
       `fb_exit`). No release / acq_rel atomics remain; the RMWs have acquire order only.
     * New gate `tools/wbl2check.py`: on the reviewed snapshot it finds exactly the 2 sites; on the final build, 0.
   * **Out-of-bounds read in the split-0 sampling path:** when M % 4 is 1 or 2, the 4-row tau loads of the row
     pair past M used an unguarded base. Since 5c0b5ec1c that path served every SplitN in steps 2-3 (most reachable
     on few-row, long-row chunks: rows of ~131K+ keys read past the 8192-record slack). That pair now gets a valid
     base in both variants. The new 1001 x 9000 / 1002 x 20000 cases only exercise the path; their over-read
     would stay inside the slack, so they could not have caught the bug.
3. **Register pressure.** In the shared kernel the distributed code raised the scan loop from 19.85 to 20.15
   VALU/MFMA and from 1 to 24 SGPR-spill readlane/writelane. Step 4 got ~10 % slower (`m4` / `m5`), and the seven
   single-split 128K chunks +1.0 %. ISL 128K went 8.254 -> 8.435 (`m5`), but 0.115 ms of that rise is the
   oversized 3-split chunk of item 4.
   * Moving the tau phase behind a `__noinline__` call was worse: the caller spilled 52 -> 484 B.
   * Fix: two compiled variants (`DIST` template flag). The launcher uses DIST only for SplitN >= 4 grids that fit
     on the GPU (rb * SplitN <= CUs). At SplitN = 2 the variant cost more than it saved (1.978 vs 1.920 ms).
4. **Oversized grids** (13312 x 32000, 3 splits, 1248 WGs). The split-0 WGs run first and steal every chunk, then
   compute 32 rows two at a time: 1.60 -> 1.90 ms. Fix: inside DIST, a grid that does not fit uses one whole-range
   chunk and 4 rows per wave. Stealing keeps either mode correct.
5. **Persistent sample buffer (intermediate work, never committed).** A driver edit during this round gave step 4
   the per-block layout while its kernel (plain variant) uses per-WG slots: a GPU memory fault in the tests. The
   committed driver keeps the per-WG slots (unchanged from 5c0b5ec1c).

**Gates and tests on the final build** (value-only samples in every plain launch, permlane fix):
* ISA gates on all 26 functions of the module (9 fused kernels, `fused_item_call`, scan / tau / select / fallback
  kernels): asm-load touches 0, VALU-SGPR -> VMEM 0, wbl2 0, permlane 0. Scratch inside the pipelines is 0, except
  step 4's known 11 compiler-waited reloads.
* `test_exact --quick`: 16/16 for steps 0-1; 22/22 for steps 2-4, including the forced-split stealing and
  M % 4 in {1, 2} cases.
* Repro loops (fresh poisoned workspaces, steps 2-4): 8193 x 8192, 1000 x 3000, 1001 x 9000, 1001 x 16384 (150
  launches each), 576 x 250000 (100): 0 wrong rows. `repeat_check` 2|3|4 pass.
* Benchmarks: 0 wrong rows and 0 fallback rows in all 64 (chunk, path) entries; step 2's candidate counts equal
  step 0's, or are slightly lower (1400.7 vs 1405.2 on 13312 x 32000).
* Tau per block on the final code (`ts_probe.py`, `-DFMT_TS` build): 16384 x 28672 14.0 us, 16384 x 65536 16.2 us,
  16384 x 262144 32.7 us (16384 x 8192: 10.4 us).

### Results (ms, 1 layer, top-512; final build, `results/m11_128k_final.json`, `results/m11_long_final.json`, 3 rounds)

"Fair step 0" is `step0ks32` at its best: its standalone tau and select kernels use the shuffle `find256`.
"Step 2 before" is the last run before this round on the 946ab423f build: `results/m1_*`, and
`results/bench_long_shapes_r2.json` for 4096 x 131072.

| chunk (splits, variant) | fair step 0 | step 2 before | **step 2 now** | step 3 | step 4 |
|---|---|---|---|---|---|
| ISL 128K (8 chunks) | 8.491 | 8.343 | **8.254 (-2.8 %)** | 8.456 | 9.404 |
| 16384 x 65536 (1) | 3.660 | 3.644 | **3.607 (-1.4 %)** | 3.608 | 3.861 |
| 13312 x 32000 (3, plain) | 1.595 | 1.600 | **1.577 (-1.1 %)** | 1.644 | 1.809 |
| 16384 x 262144 (1) | 14.180 | 14.388 | 14.207 (+0.2 %) | 14.192 | 15.203 |
| 4096 x 131072 (2, plain) | 1.879 | 1.916 | 1.899 (+1.1 %) | 1.931 | 2.102 |
| 2048 x 131072 (4, distributed) | 0.970 | 1.044 | 1.030 (+6.2 %) | 1.092 | 1.200 |
| 576 x 250000 (14, distributed) | 0.560 | 0.742 | 0.620 (+10.7 %) | 0.683 | 0.902 |

* ISL 128K per chunk: step 2 is faster than fair step 0 on all 8 chunks, from -7.5 % (16384 x 4096: 0.278 vs 0.301)
  to -0.8 % (13312 x 32000).
* 16384 x 262144: +0.03 %, +0.1 % and +0.2 % in the three runs `m9`-`m11` (14.20 vs 14.18-14.20 ms), i.e. on par;
  it was +1.1 % with the u64 samples (`m8`).
* The earlier runs of this round (`m3`-`m7`) measured step 0 with the DPP `find256` in its tau kernel and are
  therefore 2-5 % slow on the long-row few-row shapes (576 x 250000: 0.587-0.595 vs 0.560 now). `m9` is the
  value-only build with the permlane hazard (inflated step-2 candidates), `m10_*_vals16k` the fixed build with the
  value-only flavour limited to N >= 16384.

### Status vs the plan

* Step 1: done.
* Step 2 (distributed sampling): implemented as planned (own key range per split, rows spread over all splits' waves,
  work-stealing chunk claims) and verified, including forced stealing on grids far larger than the GPU.
* Step 3 (faster in-kernel tau): **2.1-3.6x per block** on the final code (14.0 / 16.2 / 32.7 us at 28K / 64K / 262K
  pools, from 29.0 / 36.0 / 116.2): the 4-row lockstep gave 1.4-1.8x; value-only samples, which halve the streamed
  bytes, gave the rest.
* **Against step 0 at its best, step 2 wins** at ISL 128K (-2.8 %, on every one of its 8 chunks), at 64K pools
  (-1.4 %) and on the 3-split chunk (-1.1 %), and is on par at 262K pools (+0.2 %).
* **It trails** on the 2-split chunk (+1.1 %) and on few-row long-context chunks: 2048 x 131072 +6.2 %,
  576 x 250000 +10.7 %.
* Same-build gaps before -> after this round on those two chunks: 576 x 250000 +30.6 % -> +10.7 % (step 2 0.742 ->
  0.620); 2048 x 131072 +7.1 % -> +6.2 % (step 2 1.044 -> 1.030).
* What remains, measured:
  * The critical path before the scan in the distributed variant is ~100 us at 576 x 250000: the slowest chunk's
    sampling (~34 us), then one row's tau per wave plus two flag hand-offs (~67 us). Step 0 runs its separate
    passes (sample + tau + prep 70.5 us) across all CUs with no cross-split wait.
  * The distributed variant's scan loop carries more register pressure (+1.5 % VALU, SGPR-spill reloads), and it
    keeps u64 sample records (the value-only flavour there made the allocator copy / spill in-flight prefetch
    registers).

## 2026-11-09 - comparison with moreh-dev/rocm-aiter PR #16 (`fp8_mqa_logits_topk`)

PR #16 (namle, 90fb19697 on main 269675dea) is the hi16 design as an aiter op: v35wes logits stored as their high 16
bits (`[M, N]` 2-byte keys, half of baseu's bytes), a two-pass top-K over them, keys tied at the K-th recomputed in
fp32. It was built untouched from a worktree (`workspace/rocm-aiter-pr16`, local branch `tmp/pr16-compare` = the PR
commit, its own aiter JIT, CK from this tree: same submodule pin) and loaded by path as bench path `pr16`.
Same harness, data, GPU and timing as the m11 runs. Every path has 0 wrong rows on every chunk (exact top-K of the
v35wes logits), PR #16 included.

| ms, 1 layer, top-512 | baseu | PR #16 | step0ks32 | step2 | step2 vs PR #16 |
|---|---|---|---|---|---|
| ISL 128K, 16384-token chunks | 10.937 | 8.926 | 8.471 | **8.232** | -7.8 % (step 0 -5.1 %) |
| ISL 128K, 8192-token chunks | 10.775 | 9.249 | 9.435 | **9.173** | -0.8 % (step 0 +2.0 %) |
| 16384 x 65536 | 4.854 | 4.100 | 3.655 | **3.611** | -11.9 % |
| 16384 x 262144 | 19.050 | 18.679 | **14.119** | 14.167 | -24.2 % |
| 13312 x 32000 | 1.987 | 1.594 | 1.590 | **1.573** | -1.3 % |
| 4096 x 131072 | 2.490 | 2.218 | **1.873** | 1.894 | -14.6 % |
| 2048 x 131072 | 1.293 | 1.139 | **0.969** | 1.027 | -9.9 % |
| 576 x 250000 | 0.756 | 0.659 | **0.564** | 0.622 | -5.7 % |

(`results/pr16_128k_c16k.json` with hi16u 8.986 = PR #16 within 0.7 %, `results/pr16_128k_c8k.json`,
`results/pr16_long.json`.)

* PR #16 still writes and re-reads an `[M, N]` matrix, so it falls behind as N grows (-2 % vs baseu at 262K pools);
  the fused paths cost N/32 + the scan. It is ahead on short rows: 8192 x 2048 0.104 ms vs 0.152 (step 2) / 0.163
  (step 0), and on the 8192-row chunks up to ~16K pools. Its one outlier: 8192 x 4096 0.273 ms (slower than its
  8192 x 6144, 0.267; reproduced).
* 8192-token chunks are where the grid model hurts steps 0/2: it gives every 8192-row chunk SplitN 2 (and the
  5120-row tail chunk 8), so step 2's other split waits for split 0's sample + tau. Forcing SplitN 1 on those chunks
  (`step2sp1` / `step0ks32sp1`, `results/pr16_128k_c8k_split1.json`): step 2 8.835 ms (-4.5 % vs PR #16, faster on
  every chunk from 6K pools up), step 0 9.243 (= PR #16); the 5120 x 32000 tail still prefers SplitN 8. The model
  prices only the scan, not the step-2 tau hand-off.
