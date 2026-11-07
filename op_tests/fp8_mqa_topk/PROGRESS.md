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

### Results (1 layer, top-512, ms; `results/bench_steps_128k_final.json`, 3 rounds)

ISL 128,000 (8 chunks):

| baseu | hi16u | fuse48 | step 0 | step 1 | **step 2** | step 3 | step 4 |
|---|---|---|---|---|---|---|---|
| 10.912 | 8.986 (-17.7 %) | 10.200 (-6.5 %) | 8.587 (-21.3 %) | 8.658 (-20.7 %) | **8.333 (-23.6 %)** | 8.563 (-21.5 %) | 9.450 (-13.4 %) |

Per chunk, step 3 - step 2 = +15-45 us (+64 us on the 3-split 13312 x 32000 chunk); step 4 - step 3 = +77-165 us.
0 wrong rows and 0 fallback rows on every chunk; ~1.4K candidates per row (max ~2.4K).

Long context and small M (`results/bench_long_shapes_final.json`):

| chunk (SplitN) | baseu | hi16u | step 0 | step 2 | step 3 | step 4 |
|---|---|---|---|---|---|---|
| 16384 x 65536 (1) | 4.856 | 4.132 | 3.656 | **3.631** | 3.647 | 3.921 |
| 16384 x 262144 (1) | 19.073 | 18.270 | **14.137** | 14.358 | 14.336 | 15.295 |
| 2048 x 131072 (4) | 1.339 | 1.156 | **0.974** | 1.041 | 1.096 | 1.209 |
| 576 x 250000 (14) | 0.763 | 0.671 | **0.563** | 0.738 | 0.797 | 0.906 |

### Why step 3 and step 4 are not faster (measured)

* **Launches cost nothing here.** rocprofv3 shows ~0 us between back-to-back kernels of one pipeline, so fusing a
  launch only pays if the work it moves gets cheaper. On gfx950 it gets more expensive.
* **The in-kernel select is slower than its own kernel.** The fused kernel needs all 256 VGPRs (1 WG/CU), so the
  select cannot be inlined (that spilled inside the hand-vmcnt pipelines, which gives wrong results, see step 0). Behind
  a call it pays the call frame (316 B/lane scratch) and runs on the 4 rows of each wave in turn. Probe: 61-69 us per
  32-row block at every N, i.e. 122-138 us per CU per chunk. The standalone `select_kernel<32>` (204 VGPR, no
  frame) does the same rows in less time: step 3 - step 2 = +15-45 us per 16384-row chunk. At 16384 x 4096 the in-kernel select (61 us) costs more than the block's whole
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
  a 482 us EMIT, so step 2 is 0.738 vs 0.563 ms for step 0 (separate sample + tau kernels spread over all CUs). At
  SplitN = 4 (2048 x 131072): +7 %; at SplitN <= 3: within 0-1 %.

### Status vs the goal

* Step 4 (one launch, exact) is implemented, but it is **not** faster. 9.45 ms at 128K: -13.4 % vs baseu, behind
  hi16u, step 0 and steps 1-3. The reasons are above, and none of them is a missing optimization of the persistent
  loop.
* **Best path: step 2** (3 launches), 8.33 ms at 128K: -23.6 % vs baseu, -7.3 % vs hi16u. It also wins at 64K pools.
  Step 0 is 1.6 % faster at 262K pools and leads on small-M chunks (SplitN >= 4).
* A single launch at step-3 speed is possible without persistence: the selecting WG recomputes its rare failing rows
  itself. That only removes the free `fallback1` launch, so it was not built.
* Open, not done:
  1. Distributed sampling for SplitN > 1. Each split samples its own stage range into its own segment; chunks are
     claimed with epoch-valued flags (work stealing, so nothing assumes co-residency); the last finisher computes
     tau. Expected: 576 x 250000 0.74 -> ~0.52 ms.
  2. A faster in-kernel select (2 rows per wave in flight, no call frame) is the only lever left for step 3/4.
