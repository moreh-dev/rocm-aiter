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
