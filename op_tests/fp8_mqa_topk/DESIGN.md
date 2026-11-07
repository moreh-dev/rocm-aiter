# Fused FP8 MQA logits + top-k on gfx950 (MI355X): design

Branch `tmp/dev-fuse-logits-topk` of rocm-aiter (on top of `namle/hip-fp8-mqa-logits-v35w`, kernel **v35wes** in
`csrc/kernels/fp8_mqa_logits.cu`). Progress log: [PROGRESS.md](PROGRESS.md).

## 0. Problem and goal

GLM-5.3-Flash kpool indexer prefill, per layer and per 16,384-token chunk:

    logits[m, n] = sum_h relu(Q[m,h,:] . K[n,:]) * w[m,h] * kv_scale[n]     n in [ks[m], ke[m]) (pools)
    out[m, :]    = indices (n - ks[m]) of the K = 512 largest logits of row m, -1 padded if fewer valid

Shapes: Q [M, 32, 128] fp8 e4m3, K [N, 128] fp8 e4m3, kv_scale [N] f32 (power of two), w [M, 32] f32, ks/ke [M]
int32. M <= 16384 rows per chunk, N = pools = context / 4 (up to ~262K at 1M tokens).

Today this is two ops: dense logits (v35wes, fp32 [M, N] written to HBM) + `top_k_per_row_prefill`.
Nam's MV-5041 work (README section 18, sources in `MV-5041-new-update.tar.gz` -> `roofline/data_1009_hi16/`) built
**fuse48** on the older v35a kernel: 7 kernels + 6 torch ops per chunk:

    torch K[::S], scale[::S], ks_s/ke_s  ->  v35h(sample, 16-bit)  ->  tau_kernel  ->  v48 (scan + candidate emission)
      ->  select_kernel  ->  fb_compute + fb_select + fallback_kernel (exact recompute of flagged rows; ~never)

**Goal (user, 2026-11-07):** cut the launches step by step and reach **step 4 = ONE persistent kernel** that is
faster than every multi-launch version and than the unfused path. If a step cannot be achieved, show the real reason.

| step | launches per chunk | what changes |
|---|---|---|
| 0 | 7 + 6 torch | fuse48 ported onto v35wes (reference point on the shipped kernel) |
| 1 | 4 | sample kernel reads K with stride S itself (no torch prep) and computes tau in its tail; 3 fallback kernels -> 1 |
| 2 | 3 | tau computed inside the scan kernel (prologue sample pass per split) |
| 3 | 2 | select at the end of the scan kernel (last split of a row block selects) |
| 4 | 1 | persistent scan kernel: claims row-block items heaviest-first, sample -> tau -> scan -> select -> in-kernel fallback |

## 1. Reference numbers (this box, GPU 7, 2026-11-07; 1 layer, top-512, sum over the 8 chunks of ISL 128,000)

Reproduced with Nam's own scripts (`experiments/fuse_topk_ref/`):

| path | ms | note |
|---|---|---|
| base = v35a logits + vLLM top_k_per_row_prefill | 12.07-12.24 | Nam: 12.24 |
| **baseu = v35w logits + vLLM top-k** | **11.11** | logits 6.76 ms + top-k ~4.4 ms; v35wes is 1-3 % faster than v35w |
| hi16 = v35h 16-bit logits + topk3 | 9.87-9.92 | |
| **hi16u = v35hw 16-bit logits + topk3** | **9.01** | strongest unfused reference |
| fuse48 (v35a based, 7 kernels) | 10.19 | parts per 16384x28672 chunk: pre (sample+tau) 0.207, v48 1.624, select 0.147 ms |

Targets: step 0 ~= hi16u; step 4 clearly below hi16u (rough budget at 128K: scan ~6.8-7.1 + in-kernel sample ~0.4 +
select ~0.3-1.2 ms -> ~7.5-8.7 ms).

## 2. Exactness contract

For every row the output index set must have the same multiset of fp32 logit values as the exact top-K of that row,
where "logit" means the value the dense v35wes kernel computes (bit-exact; the scan kernel must reproduce it
bit-for-bit). Ties at the K-th value may pick different indices. Rows with < K valid keys: all valid indices, then -1.
Indices are relative to ks (vLLM convention). Exactness must never depend on the threshold being right: any row the
gate cannot serve is detected and recomputed exactly (fallback).

Why candidate sets are exact: the scan emits every key with logit >= tau (tau per row, or per (row, split)). If no
segment overflowed and the emitted set of a row contains >= K keys that all lie above a common threshold, the K-th
largest logit of the row is >= that threshold, so the top-K is a subset of the candidates. The select then works on
the exact fp32 candidate values.

* Row-wide tau (all steps): C = sum over splits of counts; require no overflow and C >= min(K, nvalid).
* (Planned per-split tau_s for steps 2-4 - every split targets the full K inside its own range - was built and
  dropped: SplitN x ~1.4K candidates per row made the select slower than the serialized row-wide sample. Steps 2-4
  use row-wide tau: split 0 samples the whole row range and publishes tau, see PROGRESS.md.)

## 3. Threshold (tau) from a sample

Sample = keys n = j*S (j integer) inside the range being covered. For a row with nvalid valid keys and ns valid
samples: e = K * ns / nvalid (expected samples above the K-th value), target = ceil(e + 5*sqrt(e)) + 8,
tau = the target-th largest SAMPLE logit (exact fp32 value; Nam used the lower edge of a 16-bit bucket - we keep full
fp32 samples, so there is no bucket rounding and no "large common offset" precision problem).
tau = -inf when ns < target, or nvalid <= CAP (steps 0/1) / nvalid <= K (steps 2-4). Expected candidates ~ target * S
(~1.4K at S = 32, K = 512, e ~ 16, target ~ 44, independent of N). P(fewer than K keys >= tau) is a > 5 sigma event
per row.

S = 32 for every fused variant (as implemented; S = 16 costs more sample + tau than it saves in candidates). The sample
count is NOT capped: a 2048-sample cap made tau too coarse at N >= 196608 (up to 6K candidates/row > CAP). Rows with
more than 2048 samples use a per-lane top-8 filter before the radix (section 4.2).

## 4. Kernels

All new device code lives in `csrc/kernels/fp8_mqa_topk/` (headers) + `fp8_mqa_topk_capi.cu` (extern "C" launchers for
the ctypes dev harness). The shipped `csrc/kernels/fp8_mqa_logits.cu` stays untouched.

### 4.1 Scan kernel (template, copy of mqa_v35wes::kernel)

Same geometry as v35wes: 8 waves x 4 rows = 32 rows per WG, K stage = 2 sub-tiles x 32 tokens in a 2-stage LDS ring,
PF = 4 register prefetch slots, hand-placed vmcnt (W_STEADY counts K loads + 2*SUB stores per stage), clean/masked
stage split, scalar epilogue co-issued under MFMA, deferred row-pair tails, early barrier + s_setprio staircase.
Build flags: `-O3 --offload-arch=gfx950 -fno-honor-nans -fno-slp-vectorize -mllvm -disable-machine-sink`.

Modes (compile-time):
* `DENSE`: v35wes store (fp32 logits). Used (a) to prove the template is bit-exact with the aiter op,
  (b) for the sample pass: with `STRIDED` the K/scale loads use a runtime stride S (voffset = (t*S)*128 + piece*16,
  soffset = token_base*S*128; scales voffset = tid*S*4) and the output row stride is the sample count.
* `EMIT`: Nam's v48 pair_finish: `cand = res >= tau` (masked stages also check the window, row < M, n < N);
  `mask = ballot(cand)`; lanes 0-31 = row r, 32-63 = row r+1; position = per-row wave-uniform counter + mbcnt;
  one `buffer_store_dwordx2` of `(n << 32) | bits(res)` per pair, to the row's segment slot min(pos, CAP); lanes that
  are not candidates store to an out-of-range voffset (dropped), overflow goes to dump slot CAP.
  **Exactly one VMEM store per pair in every mode** (dummy OOB stores where needed) so the hand vmcnt counts hold.
  Counts per (split, row) written at the end (may exceed CAP = overflow).
* Candidate segments: `cand[(s * M + m) * (CAP + 1) + i]`, u64; CAP = 4096.

### 4.2 tau / select / fallback (port of fuse48.hip, 32-bit keys)

* tau: one wave per row; samples in registers (up to 32 x 64 = 2048) -> relative two-pass radix (8-bit LDS histogram
  just below the common prefix of the row's min/max key, then 8 more bits). More samples: each lane keeps its 8
  largest keys (batched loads, branch-free insertion) and the same radix runs on the 512 kept keys - a subset of the
  samples, so the result is still a lower bound of the target-th largest sample.
* select: one wave per row; read counts of all splits, flag the row (append to the fallback list) if any check of
  section 2 fails; else exact top-K of the fp32 candidates (<= 2048 in registers: binary search; else radix), emit
  keys > T then ties == T up to K, positions by ballot/mbcnt. Output n - ks, -1 padding.
* fallback: recompute the whole row exactly with a `recompute` that is bit-identical to v35wes (plain MFMA
  `__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(..., 0,0,0,0,0,0)`, same operand layout, scalar even/odd fma
  chains `ae = ye*w0; ae = fma(ye, w_v, ae)`, `p = ae + ao`, permlane32 swap, `(p0 + p1) * scale`) into a scratch row,
  then radix select. NOTE fuse48's recompute32 mirrors v35/v35a (scaled MFMA with 2^-25, v_pk_mul clamp) and is NOT
  bit-exact with v35wes.
* Memory model (as implemented): an agent-scope release does NOT wait for inline-asm stores, so every hand-off is
  `s_waitcnt vmcnt(0)` in every wave -> WG barrier -> one agent-scope atomic by one lane (acq_rel counter, or release
  store of the launch epoch) -> broadcast through LDS -> acquire fence in every reading wave. Spin-waits poll with
  RELAXED agent-scope loads and acquire once after the barrier (an acquire load per poll emits `buffer_inv sc1` per
  iteration and slowed the producers 2-4x). sc0 = workgroup scope, sc1 = agent scope.
* Counters (fallback count, split-completion counters, persistent claim/done counters) live in a small int32
  workspace zeroed ONCE at allocation; whoever consumes a counter resets it to 0 for the next launch (no memset
  launches).

### 4.3 Per step

* **Step 0** (reference): torch `K[::S]` + DENSE scan on the sample (fp32) -> tau kernel -> EMIT scan (v35wes split
  structure, row-wide tau) -> select -> fb_compute + fb_select + fallback_kernel.
* **Step 1**: sample kernel = DENSE + STRIDED scan with SplitN = 1 (each WG owns whole rows), output fp32 samples
  [M, Ns]; after its pipelines each wave computes tau for its 4 rows (glc reads of its own samples) and writes
  tau[m]. ks_s = ceil(ks/S), ke_s = ceil(ke/S) computed in-kernel. Fallback = ONE kernel: the first FBR flagged rows
  are recomputed by all waves in parallel, a per-row completion counter (acq_rel, agent scope) lets the wave that
  finishes the last slice run the select; further rows one wave per row. -> 4 launches.
* **Step 2** (as implemented, `fmt_fused.cuh`): split ranges are CONTIGUOUS stage ranges. Split 0 of a row block
  runs the sample pass through the same EV4 emission pipeline (key stride S, records into `smp`) over the whole block
  range, computes tau per row and publishes it (release store of the launch epoch in `tau_ready[bx]`); the other
  splits poll and acquire. Then the masked and clean EMIT pipelines. -> 3 launches (fused, select, fallback1).
* **Step 3**: FUSE_SELECT: after its EMIT pipelines a WG writes its counts; SplitN == 1 -> it selects its 32 rows
  directly (8 waves x 4 rows); SplitN > 1 -> release fence + atomic on block_done[b]; the last split acquires and
  selects (reads all splits' segments after the acquire). Flagged rows -> fallback list -> fallback kernel. -> 2
  launches. The select is a `__noinline__` call (inlined it spilled into the pipelines).
* **Step 4**: PERSISTENT: grid = min(items, #CUs) WGs (1 WG of 8 waves per CU at ~250 VGPR); items = (row block,
  split) ordered heaviest first (reverse block order for a causal chunk, splits of a block adjacent); a WG claims
  items from a global atomic counter (wave 0 lane 0, broadcast through LDS), runs load Q/W -> sample -> tau -> scan ->
  completion -> select of its block if last, and recomputes flagged rows itself (8 waves recompute a row into a
  per-WG scratch row, then select). The last WG to finish resets the claim/done counters. -> 1 launch.
  As implemented: claims are split-major (every split 0 before any split 1, so the tau wait cannot deadlock), the
  item body is a `__noinline__` call, and the fallback rows are drained after a global barrier.

## 5. Risks to watch (each must be measured, not assumed)

* VGPR: v35wes = 241 VGPR at 2 waves/SIMD (256 is the cliff for 8-wave WGs). EMIT adds per-pair tau, counters,
  64-bit data -> must stay <= 256 with 0 scratch. Check `-Rpass-analysis=kernel-resource-usage` and the ISA.
* Issue port: v35wes runs at ~86 cycles/MFMA with ~17 VALU/MFMA; each extra VALU per MFMA costs ~4 cycles. EMIT's
  compare/ballot/mbcnt/select adds VALU; the removed fp32 store saves energy (kernel is also at the 1400 W cap).
* Hazards: inline-asm permlane / buffer ops are invisible to the hazard recognizer; keep the >= 2-instruction rule
  before every permlane and run Nam's `check_asm_loads.py` style audit on the ISA (all loads hazard-free).
* In-kernel select runs at the scan's occupancy (2 waves/SIMD); measure it against the standalone select.
* Cross-XCD coherence for split hand-offs (section 4.2).
* Code size / I-cache (3 pipeline instantiations in one kernel).

## 6. Test and benchmark method

* Data: Nam's `gen` (random Q/K, weights (rand-0.2)*0.1, causal kpool layout: row t of the chunk sees pools
  [0, (pos+1)//4)), multi-request (reqs 2/4, prefix layout), non-causal, plus stress sets: large common score offset
  with a narrow top tail, heavy ties, scores increasing with position (forces overflow -> fallback), N <= CAP,
  M not a multiple of 32, forced SplitN 1..16, forced fallback (tau scaled up).
* Check: gathered fp32 values of the output (from the aiter v35wes dense logits) sorted == sorted top-K values per row,
  for every row; -1 padding exact.
* Timing: per chunk, 512 MB fill before every call (L2 + MALL flush), event timing, synchronize after every call
  (never queue unbounded GPU work), median over interleaved rounds; ISL totals = sum over the 16,384-token chunks.
  Compare in the same run: baseu (aiter fp8_mqa_logits + vLLM top_k_per_row_prefill), hi16u (v35hw + topk3 from
  `experiments/fuse_topk_ref`), fuse48 (Nam), step 0..4.
* GPU: only `op_tests/fp8_mqa_topk/gpu7.sh` (amd-smi GPU 7 = HIP 4, flock-serialized, refuses if someone else holds
  VRAM, kill inside the container).
