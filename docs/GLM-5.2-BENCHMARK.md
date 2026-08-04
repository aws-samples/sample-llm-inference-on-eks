# GLM-5.2-FP8 on EKS p5en.48xlarge — Benchmark Report

**Date**: 2026-07-07 → 07-08
**Model**: `zai-org/GLM-5.2-FP8` — 753B MoE (DSA sparse attention, 256 experts/top-8, 1 MTP layer), FP8 weights ~756 GB
**Hardware**: p5en.48xlarge (8× H200 141 GB, 16× EFA), Karpenter `reserved-capacity-pool` (us-west-2d capacity block)
**Load generator**: NVIDIA genai-perf 0.0.16.post1 (Triton 26.06 SDK pod, in-cluster); `sglang.bench_serving` for PD-router runs and all decode-heavy runs (genai-perf SSE parser fails against the router at concurrency 40). **Measurement volume in the original 07-07/07-08 campaigns was left at genai-perf's defaults, so none of those figures is a steady-state measurement — latency and throughput alike.** Rows marked with a prime (L2′, L3′, P1′) and the tables in Open item 0 are much deeper re-measurements from 2026-07-29→31 and are the ones to prefer — but they are still whole-run averages that include ramp-up, not steady-state values, and each is a single run with no variance estimate. See the warning below and § Traceability and known defects.

> [!IMPORTANT]
> **Reading guide.** This report's *qualitative* findings (crash mechanisms and config
> interactions — e.g. the `mem-fraction-static` OOM line, the `expandable_segments`
> incompatibility, the `max_prefill_tokens` cap) are the parts that hold up. Its
> shape-vs-shape comparisons do **not**: every one is n=1 without a variance estimate. Its *quantitative*
> tables were taken at 2–3 requests per concurrency slot and should not be cited as
> absolute numbers, or compared across concurrency levels. The
> § Traceability and known defects section lists what is unrecoverable and why; Open
> item 0 tracks the re-measurement. Reviewed against
> [BENCHMARK-REPORTING-PRINCIPLES.md](BENCHMARK-REPORTING-PRINCIPLES.md) on 2026-07-30.

The report covers three experiment campaigns, each with its own question:

| Campaign | Question | Section |
|---|---|---|
| Deployment shapes (07-07) | Which of 4 shapes serves a prefill-heavy 8K/1K workload best? | Executive Summary, Findings |
| KV offloading (07-08) | Does host-RAM KV offload pay, on which shape, and which engine does it better? | § KV-cache offloading |
| Decode-heavy rematch (07-08) | Does PD 1P+1D win on its theoretical home turf (1K/4K)? | § Decode-heavy workload |

## Workload

All runs use the same profile unless noted:

| Parameter | Value |
|---|---|
| Input length (ISL) | 8000 tokens, stddev 0 |
| Output length (OSL) | 1024 tokens, pinned via `max_tokens` + `ignore_eos` |
| Concurrency | 20 (and 40 for 2-node shapes) |
| Thinking mode | **off** — `{"chat_template_kwargs":{"enable_thinking":false}}` (nested form required; flat `enable_thinking` is ignored) |
| Endpoint | streaming `/v1/chat/completions`, in-cluster ClusterIP |

> [!WARNING]
> **The TTFT figures in this report are not steady-state measurements. Do not cite
> them as absolute latency, and do not compare them across concurrency levels.**
>
> Every genai-perf run below used the tool's defaults for measurement volume. Those
> defaults send `max(10, 2 × concurrency)` requests — 40 at concurrency 20, 80 at
> concurrency 40 — which is **two requests per concurrency slot**. `--num-prompts`
> does not change this; it only sizes the sampling pool. genai-perf also defaults
> `--stability-percentage` to `999`, so nothing checked whether the queue had
> settled, and nothing warned that it had not.
>
> The distributions are very wide — at concurrency 20 (row L2) TTFT reads p10 840 ms,
> p25 878 ms, p50 2,458 ms, p75 10,062 ms, p90 14,813 ms, max 17,624 ms, a 21× spread; at
> concurrency 40 (row L3) it is much tighter (p50 1,230 ms, p75 1,541 ms, p90 1,542 ms,
> max 2,440 ms). ⚠️ **Neither spread shows whether the queue had settled.** Percentiles are
> sorted, so they always rise, and a stationary heavy-tailed series can be just as wide; only
> per-window or time-ordered data can answer it, and none was retained for these runs. An
> earlier version of this warning read the 21× spread as proof "the queue was still growing";
> that inference is withdrawn. What does establish the defect is the depth itself — 2 requests
> per concurrency slot — and the re-measurements in Open item 0.
>
> **Throughput and ITL are also affected, and at concurrency 40 severely.** A
> deeper re-measurement of the TP8 shape (Open item 0, 2026-07-29) read
> throughput 698 tok/s at c40 where the shallow run read 1,070 (−35%), and ITL
> 54.5 ms where the shallow run read 27.4 ms. At concurrency 20 the same comparison
> moved throughput only −4%. **Treat the c40 rows below as unusable in full, not just
> their TTFT.**
> Correct methodology: [benchmark-commands.md](benchmark-commands.md#measuring-steady-state).

## Executive Summary

Six configurations were benchmarked across four deployment shapes. Key numbers (concurrency as noted):

**Prefer the primed rows (L2′, L3′, P1′), but do not treat them as steady state
either.** They were measured 10–20× deeper than the unprimed rows, which makes them much
closer to the truth — but genai-perf computes its statistics over the *whole* run,
ramp-up included, so they remain **contaminated by the transient — by an unknown amount and
in an unknown direction**. (On a smaller rig, the only one whose window data survives,
dropping the ramp-up window raised TTFT p50 by 5–19%; that direction does not carry over —
the depth bias reverses sign between rigs.) See
[BENCHMARK-METHODOLOGY.md](BENCHMARK-METHODOLOGY.md) Step 3. Every unprimed row was taken
at 2–3 requests per concurrency slot: the struck-through TTFT columns are unusable and so
are the rate columns (ITL, tok/s/user, total tok/s) — that defect was confirmed, not
assumed, in Open item 0. Nothing here is bolded as a winner, because **every row is a single
run with no variance estimate and includes ramp-up**, so no comparison in this table is
established. R4's deep counterpart is in Open item 0.

| # | Shape | Config | Conc. | ~~TTFT p50~~ | ~~TTFT p90/p99~~ | ~~ITL avg~~ | ~~tok/s/user~~ | ~~Total tok/s~~ | Ran to completion |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 1-node TP8 SGLang | chunk 2048 (default), mem 0.85, MTP 5-1-6 | 20 | 2,040 ms | 14,656 / 17,160 ms | 27.9 ms | 40.1 | 476 | ✅ |
| R2 | 1-node TP8 SGLang | chunk 32K (16K eff.), mem 0.85 | 20 | — | — | — | — | — | ❌ OOM |
| R4 | 1-node TP8 SGLang | chunk 32K (effective), mem **0.80**, MTP 1-1-2 | 20 | 3,202 ms | 17,145 / 17,790 ms | 27.5 ms | 39.3 | 456 | ✅ |
| V1 | 1-node TP8 vLLM 0.24 | defaults, mem 0.85, MTP 5 drafts | 20 | 1,953 ms | 18,314 / 25,420 ms | 28.7 ms | 38.8 | 454 | ✅ |
| L2 | 2-node TP16 LWS+EFA | mem **0.80**, MTP 1-1-2 | 20 | 2,458 ms | 14,813 / 17,365 ms | 32.4 ms | 32.9 | 396 | ✅ |
| L3 | 2-node TP16 LWS+EFA | (same) | **40** | 1,230 ms | 1,542 / 1,936 ms | 41.1 ms | 26.2 | 730 | ✅ |
| L2′ | **L2 re-measured deeply** (2026-07-30, 22.1 req/slot, whole-run avg) | (same shape) | 20 | 839 ms | 901 / 1,638 ms | 36.6 ms | 27.8 | 523.6 | ✅ |
| L3′ | **L3 re-measured deeply** (2026-07-30, 11.6 req/slot, whole-run avg) | (same shape) | 40 | 848 ms | 1,644 / 3,291 ms | 57.2 ms | 17.9 | 659.1 | ✅ |
| P1 | 2-node PD 1P+1D NIXL | both TP8, mem 0.85 | 20 | 11,560 ms | 35,258 / 38,412 ms | 25.0 ms | 40.5 | 356 | ✅ |
| P1′ | **P1 re-measured deeply** (2026-07-31, 25.4 req/slot, whole-run avg) | (same shape) | 20 | 5,228 ms | 7,695 / 19,561 ms | 27.2 ms | 36.7 | 602.6 | ✅ |
| P2 | 2-node PD 1P+1D NIXL | (same, **`bench_serving` — not comparable to the rows above**) | 40 | 17,246 ms (median) | — / 53,122 ms | 31.7 ms | ~31 | 712 | ✅ |

The last column was previously headed *Stable*, which implied a stability criterion
had been applied. None was — genai-perf's stability check was disabled by default
(`--stability-percentage 999`). It records only whether the run finished without
the server crashing.

**Row P2 was produced by a different load generator** (`sglang.bench_serving`)
than every other row. Per [benchmark-commands.md](benchmark-commands.md), the two
tools define concurrency and latency differently and their outputs are not
comparable; P2 must not be read against P1 or L3 as if it shared their basis.

### Recommendation matrix (for this 8K-in / 1K-out, prefill-heavy workload)

| Priority | Recommended shape | Basis |
|---|---|---|
| Max throughput per dollar | **Undetermined** | The former pick (2× independent TP8 replicas) was an extrapolation from a shallow single-node number (456 × 2 ≈ 912 tok/s). At the deeper depth TP8 c20 reads 552 tok/s, so the same arithmetic gives ~1,104 — and neither figure has been measured on two replicas. Mixing shallow, deeper and extrapolated bases in one comparison is not valid; see Open item 1. |
| Hard TTFT SLO at c20–c40 | **Undetermined** | Deeper runs read TTFT p50 level between the two (869 vs 839 ms at c20; 892 vs 848 ms at c40), c40 tails comparable (p90/p99 1,772 / 3,362 ms TP8 vs 1,644 / 3,291 TP16), and TP16's throughput 5.2% (c20) / 5.6% (c40) lower, relative to TP8. But each arm is **n=1 with no variance estimate** and includes ramp-up, so none of those differences is established — including the absence of one. This row previously read "no evidence favours TP16", which overstates what a single unreplicated pair can show. Cost remains a separate, non-measurement argument: TP16 uses twice the hardware. |
| Low concurrency / single-node budget | 1-node TP8 (SGLang and vLLM both viable) | The 456 / 454 / 476 tok/s readings are shallow and n=1 with no variance estimate, so they establish neither a "wall" nor a tie between engines. What survives: both engines served this model and workload without crashing at mem 0.80. |
| Strict ITL SLO (8K/1K) | **Undetermined** | The earlier verdict ("PD's only edge is ITL *variance*, at a 37% ITL-mean cost") was an artefact of 2-requests-per-slot sampling and is withdrawn. Deeper runs happen to read PD's ITL mean below one node's (27.2 vs 34.4 ms at c20) at the cost of TTFT (5,228 vs 869 ms) — but that is **one comparison, n=1, no variance estimate, ramp-up included, c20 only** (c40 is not measurable through the router). No cross-arm claim follows from it, including a directional one: "unlikely to be noise" was asserted here without a variance estimate to support it. |
| Decode-heavy workloads (1K/4K) | **Undetermined** | The § Decode-heavy workload comparison is internally consistent (same tool, same depth both arms) but both arms are shallow, and shallow measurement is now known to distort PD far more than single-node TP8 (41% vs 4% throughput error at c20 on 8K/1K, relative to the deeper value). Its −32%/−27% throughput gaps have not been re-measured at any greater depth. |

> [!IMPORTANT]
> **Finding 5 (TP8 vs TP16) is still unresolved. Three earlier attempts to settle it were
> each wrong** — first claiming the finding was *inverted* (on a shallow TP8 c40 reading of
> 1,070 tok/s), then claiming TP16 had never been measured deeply at all, then declaring
> TP16 the loser by 5–6% from the numbers below. **That third claim does not survive its
> own confounds** and is withdrawn.
>
> | 8K/1K | TP8 | TP16 | depth (req/slot) |
> |---|---|---|---|
> | c20 total tok/s | 552.2 | 523.6 | 23.3 vs 22.1 |
> | c20 TTFT p50 | 869 ms | 839 ms | (as above) |
> | c40 total tok/s | 698.1 | 659.1 | **14.7 vs 11.6 — 21% fewer** |
> | c40 TTFT p50 | 892 ms | 848 ms | (as above) |
>
> Two defects, either one fatal:
>
> 1. **n=1, with no variance estimate, and unequal sample sizes.** Each arm was run once.
>    At c40 the arms also completed different numbers of requests (14.7 vs 11.6 per slot —
>    TP16's run stopped after ~4 windows against TP8's ~4.8), which costs precision in the
>    smaller arm. A 6% gap with no repeat runs and no confidence interval is not a result.
>    *(An earlier version of this box called the differing depth itself the confound. That
>    was wrong and circular — under fixed concurrency and duration, a faster arm completes
>    more requests by definition. See [BENCHMARK-METHODOLOGY.md](BENCHMARK-METHODOLOGY.md)
>    Step 2.)*
> 2. **Neither number is a steady-state value.** Both were computed over the whole run
>    including ramp-up (see [BENCHMARK-METHODOLOGY.md](BENCHMARK-METHODOLOGY.md) Step 3),
>    and the contamination is not equal between arms — TP16 reached a shallower depth, so
>    proportionally more of its run is ramp-up.
>
> What can be said is only negative: the original "TP16 turns the corner at c40" rested on a
> 2-requests-per-slot reading of 730 tok/s and is **unsupported**. Whether TP16 is behind,
> level, or ahead of one node is **not established in any direction** — the deeper pair
> differs by 5–6%, which a single unreplicated run cannot resolve. Note `--request-count`
> cannot fix this: it
> collapses the run to one window, so ramp-up cannot be trimmed
> ([BENCHMARK-METHODOLOGY.md](BENCHMARK-METHODOLOGY.md) Step 2). Resolving it means repeat
> runs per arm with an uncertainty estimate, on trimmed stationary windows; see Open item 2.
>
> Also unexplained: the depth bias on TP16 changes sign with concurrency — relative to the
> deeper value, the shallow reading is **21% low** at c20 (414 vs 524) but **12% high** at
> c40 (739 vs 659). And no NCCL or EFA counters were ever collected,
> so the "cross-node allreduce tax" remains a hypothesis, not a measured cost.

## Findings

### 1. Chunked-prefill size did not change throughput here — but the comparison is confounded

**Observed:** across the TP8 runs, total throughput read 456–476 tok/s and TTFT tails
did not move; vLLM on defaults read 454 tok/s.

**Why this does not isolate chunk size:** the runs being compared differ in three
parameters at once, not one:

| | chunk | mem-fraction | MTP (speculative) |
|---|---|---|---|
| R1 | 2048 | 0.85 | 5-1-6 |
| R4 | 32K (effective) | **0.80** | **1-1-2** |

No single-variable chunked-prefill sweep was run, so the flat throughput cannot be
attributed to chunk size. The SGLang↔vLLM comparison spans engine, mem-fraction and
speculative-decoding config simultaneously.

**On "same number":** 454 / 456 / 476 tok/s spans 4.8% (476 ÷ 454). Each is a single shallow run
with no repeats and no variance estimate, so whether that spread is noise or signal is
undetermined — it is not evidence of a shared ceiling. The earlier claim that this
showed "two engines, three schedulers, same number" is withdrawn.

**Cause not established:** the earlier explanation — that prefill arrival rate
saturates 8× H200 compute so scheduling can only redistribute the queue — is a
plausible hypothesis, but **no GPU-utilisation, MFU, or profiling data was collected**,
so it remains unverified. What can be said: the SGLang cookbook's reported
chunked-prefill gains (+34–78%) did not reproduce on this workload/config.

### 2. `--mem-fraction-static` safe line is 0.80, single- and multi-node

0.85 OOM-crashed twice under load, with different triggers:

- **TP8 + 32K chunk** (R2): large-chunk prefill activations spiked ~3 GiB/GPU against ~2 GiB free → all 8 ranks OOM mid-benchmark.
- **TP16 + default 8K chunk** (L1, initial deploy): the static pool is a *percentage of total VRAM*, so TP16's per-GPU weight savings (~94 GiB → ~47 GiB) were silently absorbed by the KV pool, leaving only ~600 MiB activation headroom → OOM at concurrency 20.

Dropping to 0.80 frees ~7 GiB/GPU for activations; both shapes then survived every run with zero restarts. Multi-node does **not** automatically gain memory headroom — the fraction semantics guarantee it doesn't.

### 3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is incompatible with TP serving

Attempted as a fragmentation mitigation; crashed at startup — custom all-reduce cannot IPC-register VMM-backed allocations (`custom_all_reduce.cuh:37: CUDA error: invalid argument` during CUDA graph capture). Do not combine with multi-GPU TP.

### 4. `--chunked-prefill-size` is silently capped by `--max-prefill-tokens`

SGLang's default `max_prefill_tokens=16384` caps the effective chunk. Setting `--chunked-prefill-size=32768` alone gives you 16K chunks; both flags must be raised together. (Made no difference here — see Finding 1 — but matters on workloads that are genuinely chunk-bound.)

### 5. Whether TP16 justifies itself is still unresolved

**No valid TP8-vs-TP16 comparison exists in this report yet.** Deeper re-measurements at
c20/c40 put TP16 within 5–6% of a single TP8 node (523.6 vs 552.2 tok/s at c20, −5.2%;
659.1 vs 698.1 at c40, −5.6%) while using twice the hardware — but each arm is **a single run with no
variance estimate**, and neither figure excludes ramp-up, so a 5–6% gap cannot be read off
them — in either direction. See the box above, and Open item 2 for what a valid run
requires. What is safe to say is only about the *old* claim: "c40 is where TP16 turns the
corner" rested on a 2-requests-per-slot reading of 730 tok/s and is unsupported.

**Originally observed at c20 (shallow):** TP16 read lower total throughput than TP8
(396 vs 456 tok/s) and higher ITL (32.4 vs 27.5 ms), on 2× the hardware. Both numbers were
shallow readings, and the deeper re-measurement neither confirmed nor overturned the gap —
it is within the range a single unreplicated run cannot resolve.

**Mechanism not established.** The standing hypothesis — under-fed GPUs at low
concurrency plus two cross-node EFA allreduces per decode step — is consistent with the
ITL direction, but **no NCCL or EFA counters were collected at any point**, so the
allreduce cost has never been measured. Whether the second node contributes at all remains
unexplained.

**Withdrawn — the c20→c40 "1.84× gain".** It compared L2 (c20, 396) against L3 (c40,
730). The 2026-07-29 re-measurement showed shallow-run throughput error is itself
strongly concurrency-dependent on this workload (−4% at c20 vs −35% at c40 on TP8), so
these two numbers do not share a basis and their ratio is not interpretable. Comparing
across concurrency levels is exactly what the warning at the top of this report
prohibits.

> [!CAUTION]
> **Superseded (2026-07-31).** This box previously argued the finding might be
> *inverted*, on the strength of TP8 reading **1,070 tok/s at c40** against TP16's 730 —
> one node beating two by 47%. Both figures were 2-requests-per-slot readings. At the deeper
> depth the pair is 698 (TP8) vs 659 (TP16) — no 47% inversion, and n=1 on both sides so no
> direction is established either. Retained here because the retracted claim circulated; the
> current position is in the Finding 5 box above.

> [!CAUTION]
> **Retracted (2026-07-28):** this finding previously claimed TTFT p90 "collapsed
> from 14.8 s to 1.5 s (10×)" at concurrency 40, attributed to prefill bursts being
> absorbed whole. That comparison is withdrawn — both runs measured only two
> requests per concurrency slot, and the concurrency-20 distribution never reached a
> plateau (see the warning at the top of this report). Whether TP16 genuinely
> improves TTFT at higher concurrency is **unresolved** and needs re-measurement
> with a proper measurement window.
>
> ~~the throughput figures above are unaffected~~ — **this caveat was itself wrong,
> and is withdrawn (2026-07-30).** The 07-29 re-measurement found throughput to be the
> *most* depth-sensitive metric on this workload (−35% at c40), so the throughput
> figures carry the same defect as the TTFT figures.
>
> A follow-up experiment on cheaper hardware (1× L40S, Qwen3-8B, 2K/256) found that
> short measurement windows understated TTFT there, and did **not** reproduce a
> direction reversal. Note that the *direction* of the shallow-measurement bias has
> since proven rig-dependent (it inverted on B300/Kimi-K3), so that experiment does not
> transfer to this hardware — see
> [BENCHMARK-METHODOLOGY.md](BENCHMARK-METHODOLOGY.md). The mechanism
> behind this specific 10× figure remains unexplained.

### 6. PD 1P:1D showed no advantage on an 8:1 ISL:OSL workload — at shallow depth only

PD delivered its core promise — decode purity (ITL p99 27.5 ms, max 108 ms; per-user
floor 36 tok/s). At concurrency 40, TTFT read median 17 s / p99 53 s while total
throughput was 712 tok/s against a 1,013 tok/s peak output reading.

**Attribution not established:** the gap between peak and sustained output is
*consistent with* the prefill node bottlenecking while decode has spare capacity, but
**no per-node GPU-utilisation or queue-depth telemetry was collected**, so "the decode
node idled" is an inference from two throughput numbers, not an observation. Input
throughput matching TP16 (~5.6K tok/s) is consistent with equal total compute
statically partitioned; that reading is also shallow.

At the time we hypothesized PD might still win on decode-heavy traffic. That was tested on
07-08, and the test *appeared* to say no — but its conclusion has since been **withdrawn**
(§ Decode-heavy workload): both arms were measured at 2–3 requests/slot, and equal depth
turned out not to mean equal distortion. **The decode-heavy question is open.**

### 7. Engine latency profiles differed, on a single shallow run each

vLLM read a lower TTFT p50 (1.9 s) and lower time-to-second-token (640 ms); SGLang read
a lower TTFT max (17.8 s vs 25.7 s). Both are n=1 shallow runs with no variance
estimate, and the configurations differ in more than the engine (see Finding 1), so this
does not establish a general "vLLM favours the median, SGLang bounds the tail"
characterisation — it records what these two runs read. The earlier framing of a shared
"455 tok/s wall" is withdrawn for the reasons in Finding 1.

*TTST = time to second token (the first inter-token interval), i.e. the first decode
step after prefill.*

## Deployment shapes tested

| Shape | Manifest | Image |
|---|---|---|
| 1-node TP8 SGLang | `sglang/glm-5.2-fp8-p5en.yaml` (R4 config committed) | `lmsysorg/sglang:v0.5.13.post1` |
| 1-node TP8 vLLM | deployed ad-hoc (`glm-5-2-vllm`), not committed | `vllm/vllm-openai:v0.24.0` |
| 2-node TP16 LWS | `lws/lws-glm-5.2-tp16-p5en.yaml` (0.80 + MTP 1-1-2 committed) | ECR `sglang-efa-p5:v0.5.13.post1-nixl` (EFA 1.49 + aws-ofi-nccl + NIXL) — as recorded at the time; see note below |
| 2-node PD 1P+1D | deployed ad-hoc (`lws-glm-5-2-prefill`/`-decode` + `glm-5-2-router`), manifest: `lws/lws-glm-5.2-pd-p5en.yaml` | same ECR image |

> [!NOTE]
> **The `-nixl` tag recorded for TP16 above could not be located on 2026-07-31.** The
> source ECR repo holds only `v0.5.13.post1-efa` for that version, so the exact image
> used for the TP16 rows is no longer retrievable and those runs are not
> byte-reproducible. The committed manifest now points at `-efa`: TP16 needs EFA +
> aws-ofi-nccl for NCCL allreduce, while NIXL is only for PD KV transfer. Whether the
> original image differed from `-efa` in any way that affects the numbers is
> **unknown** — not investigated.

GLM-5.2 requires SGLang ≥ v0.5.13.post1 (`glm_moe_dsa` arch) and transformers ≥ 5.x for its tokenizer — old Triton SDK images (≤24.12) cannot tokenize it; use 26.06+.

## Traceability and known defects

Recorded against [BENCHMARK-REPORTING-PRINCIPLES.md](BENCHMARK-REPORTING-PRINCIPLES.md).
These are properties of the 07-07/07-08 campaigns that **cannot be fixed by
re-measurement of the pending runs** — they are gaps in what was captured at the time.
Listed so the numbers are not read as more solid than they are.

### The exact commands that produced these tables were not recorded

The § Reproduce block is corrected methodology, **not** what generated the numbers
above. Specifically unrecoverable for the 07-07 runs:

- the actual `--num-prompts` per run (so prompt reuse, and any mid-run prefix-cache
  warming, cannot be ruled out)
- per-run genai-perf JSON artifacts — not archived
- repeat count and run-to-run variance: **every figure is a single run (n=1)**, no
  confidence interval

Per principle 12, these are marked unrecoverable rather than reconstructed from memory.

### Raw data for the 07-29→31 re-measurements is also not published

The primed rows and the Open item 0 tables were produced by genai-perf runs whose
artifacts live **outside this repo**, in the operator's git-ignored `local/logs/`
(aggregate CSVs per run: `bench-glm-5.2-tp16-{probe,steady}-c{20,40}.csv` and the
single-node/PD equivalents). Consequences a reader should know:

- **The percentiles cannot be independently re-derived** from anything committed here.
- Only the percentiles genai-perf emits by default are available (avg, min, max, p99,
  p95, p90, p75, p50, p25, p10, p5, p1) — for PD only p10/p50/p90 were transcribed into
  this report, short of the full distribution the methodology asks for.
- The per-request `profile_export.json` files, which are what a last-three-window
  re-derivation would need, were **not retained** for the GLM-5.2 runs. That is why the
  ramp-up contamination described in
  [BENCHMARK-METHODOLOGY.md](BENCHMARK-METHODOLOGY.md) Step 3 was quantified on the
  L40S rig (whose raw exports survive) and not on these.

**Raw artifacts are deliberately not committed** (Open item 4, decided 2026-08-03): the
per-run exports are 300+ MB each. The consequence is accepted — these figures are not
independently verifiable from this repo — rather than left as a pending action. The
obligation it creates applies to future runs: retain `profile_export.json` long enough to
perform the last-3-window trim before discarding it.

### Prefix/hierarchical caching was enabled and is an undisclosed confounder

The committed TP8 config (`k8s-manifest/sglang/glm-5.2-fp8-p5en.yaml`) runs
`--enable-hierarchical-cache --hicache-ratio=2 --hicache-write-policy=write_through`,
and SGLang's radix cache is on by default. With synthetic prompts of identical length
and an unrecorded prompt-pool size, the cache hit rate during these runs is **unknown**
and was never reported. On this repo's Kimi-K3 runs the measured hit rate was 0.0%, but
that does not transfer — different engine, workload and pool.

### The 07-07 table does not close arithmetically

For a closed-loop harness, `total tok/s ≈ concurrency / E2E-latency × OSL`. Recomputing
each row from its own reported ITL and TTFT:

| Table | Tool | Depth | Closure error |
|---|---|---|---|
| Executive Summary (07-07) | genai-perf | 2 req/slot | **+30% … +55%** |
| § Decode-heavy (07-08) | `bench_serving` | 3 req/slot | +0% … +2% |
| Open item 0 deeper whole-run | genai-perf | 23.3 req/slot | +3% |

Per-row for 07-07: R1 +41%, R4 +43%, V1 +44%, L2 +45%, L3 +30%, P1 +55%. The other two
tables close. **Cause not established** — a plausible hypothesis is that the shallow
window included ramp-up/drain intervals during which in-flight requests were well below
nominal concurrency, but genai-perf's per-window timestamps were not retained, so this is
unverified. Reported here rather than omitted (principle 5: an irreconcilable table must
be flagged, not ignored).

Separately, `tok/s/user` does not equal `1/ITL` (R1: 40.1 reported vs 35.8 implied) and
the ratio drifts between 1.01 and 1.12 across rows, so it is not a fixed definitional
offset. Neither column's definition — statistical basis, whether TTFT is included — was
recorded.

### Unreported and discarded runs

- **R3 does not appear anywhere in this report.** The run labels jump R1 → R2 → R4. What
  R3 was, and why it is absent, was not recorded.
- **L1** appears only as an OOM post-mortem (Finding 2) and in no results table.
- **No discard criterion was defined** for the campaign. Whether any run was dropped for
  reasons other than crashing is not documented.

### Scope of every conclusion in this report

Results speak only for: GLM-5.2-FP8 on p5en.48xlarge (8× H200), SGLang v0.5.13.post1 /
vLLM 0.24.0, 8K-in/1K-out (and the 1K-in/4K-out rematch), concurrency 20 and 40,
thinking disabled, `mem-fraction-static` 0.80–0.85, on the specific cluster used. Do not
extrapolate to other models, GPU types, engine versions, concurrency levels, sequence
shapes, or to multi-replica fleets.

## Crash post-mortems

| Event | Trigger | Root cause | Fix |
|---|---|---|---|
| TP8 OOM under load | c20 benchmark, ~4 min in | 0.85 static pool + 32K-chunk activation spikes | mem 0.80 |
| TP8 startup crash | CUDA graph capture | `expandable_segments` vs custom all-reduce IPC | remove env var |
| TP16 OOM under load | c20 benchmark, ~4 min in | 0.85 static pool absorbs TP16 weight savings into KV | mem 0.80 |
| genai-perf c40 failures ×2 | router streaming at c40 | genai-perf 0.0.16 SSE parser (`splintered SSE`, `orjson` error); backends healthy | none — `sglang.bench_serving` was used at the time, which is why row P2 is unusable. Report as not measurable with genai-perf |

## KV-cache offloading — 2026-07-08 (summary; full doc: KV-CACHE-ARCHITECTURE.md)

A separate experiment campaign measured KV-cache offload and externalization on the same
nodes. Full methodology, architecture diagrams, and per-experiment data live in
**[KV-CACHE-ARCHITECTURE.md](KV-CACHE-ARCHITECTURE.md)**. Headlines:

- **Long-document reuse (30K-token docs, re-asked after GPU eviction)**: host-RAM reload
  beats recompute by **22×** (vLLM + LMCache, CUDA-IPC path), **8.1×** (SGLang HiCache),
  1.14× (vLLM native connector — superseded by lmcache).
- **SGLang + LMCache is incompatible with GLM-5.2** (DSA fused `kv_buffer` vs the
  adapter's `k_buffer` assumption; upstream-confirmed, unfixed — sglang#15739). SGLang's
  correct path is HiCache (the only DSA-indexer-aware offload).
- **Externalized L2 (Redis as its own Deployment)** verified: KV survives pod death
  (4.07s vs 7.11s cold) and **crosses instances** (A computes → B consumes, 3.90s vs
  6.47s cold). L2's network constant ≈ 3s/30K tokens → break-even at ~10K-token inputs.
- **On PD**: L1 offload does *not* help the current request (P→D transfer floor), but a
  shared L2 removes the multi-turn history-recompute cost on P. Current-request hand-off
  stays on NIXL/EFA; the store serves the cross-request timeline.
- Benefit is binary on capacity (working set ≤ pool or nothing); caches add no measurable
  cold-path overhead; ITL/TPOT are unaffected — throughput gains are indirect (freed
  prefill capacity) and not yet measured.

## Decode-heavy workload: PD 1P+1D rematch — 2026-07-08

**Question.** All 07-07 runs used a prefill-heavy profile (8K-in/1K-out), which structurally
disadvantages a 1P+1D split. Does PD win when the workload is inverted to its theoretical
home turf — short prompts, long generations, where decode purity should shine?

**Experiment.** Same two CB nodes, same fp8-KV configs. Workload flipped to
**1K-in / 4K-out** (`sglang.bench_serving`, random dataset, thinking off). Two shapes:

- *Baseline*: 1-node TP8 SGLang (the committed `glm-5.2-fp8-p5en.yaml` config)
- *Challenger*: PD 1P+1D over NIXL/EFA (`lws-glm-5.2-pd-p5en.yaml` + sglang-router)

each at concurrency 20 (60 prompts) and 40 (120 prompts).

> [!NOTE]
> These runs used `sglang.bench_serving`, whose `--num-prompts` *is* the real request
> count — so unlike the 07-07 genai-perf runs, the volume here was not silently
> capped. It was still only **3 requests per concurrency slot** (60/20, 120/40),
> which is short of steady state, so **no absolute value in this table is a
> steady-state figure** — throughput and ITL included. (An earlier version of this note
> claimed "throughput and ITL are unaffected"; the 07-29 re-measurement found
> throughput to be the most depth-sensitive metric on the 8K/1K workload, so that
> exemption is withdrawn.)
>
> **This section's conclusion is withdrawn (2026-08-03).** It rested on the argument that
> matched shallow depth preserves direction: both arms used the same tool, depth, prompt
> counts and nodes, differing only in shape. The 8K/1K re-measurement disproves the
> premise — at the *same* depth, going deeper moved PD's throughput by ~41% but TP8's by
> only ~4% at c20. **Equal depth does not mean equal distortion**, so a −32%/−27%
> shallow gap cannot be assumed to survive at depth; on the 8K/1K workload a comparable
> gap inverted entirely. Both arms also remain subject to the unrecorded cache state noted
> in § Traceability and known defects.

| Metric | TP8 c20 | PD c20 | TP8 c40 | PD c40 |
|---|---|---|---|---|
| Output throughput (tok/s) | **987** | 672 | **1,521** | 1,114 |
| — per node | **987** | 336 | **1,521** | 557 |
| TTFT p50 (ms) | **424** | 2,272 | **309** | 1,370 |
| ITL mean (ms) | **19.7** | 29.3 | **25.7** | 35.2 |
| ITL p99 (ms) | 36.4 | **31.0** | 36.4* | **36.3** |
| Max ITL (ms) | 400 | **220** | — | **141** |
| E2E median (s) | **78.7** | 118.6 | **102.6** | 141.6 |

\* single-node c40 p95/p99 ITL not captured separately; mean/TPOT p99 28.5 ms.

**Conclusion — withdrawn; the decode-heavy question is reopened.** This previously read
"PD 1P+1D has no applicable regime on this model/hardware", on the strength of the shallow
gaps below: −32%/−27% throughput (c20/c40) on 2× the nodes (≈37% per-node efficiency),
TTFT 4–5× worse, ITL mean 37–49% slower. Those gaps are **not reliable at this depth** —
see the box above — and on the 8K/1K workload the analogous shallow finding (PD −32%
throughput, ITL mean +37%) did not survive deeper measurement: PD's throughput moved 41%
while single-node TP8's moved 4%, which is enough to unmake the gap without establishing one
in the other direction. The 1K/4K numbers below have not been
re-measured, so nothing here should be cited for or against PD. The **relative**
numbers below (throughput gaps, ITL means, TTFT ratios) are all subject to the same
invalidation and none of them should be quoted.

The one observation that does not depend on cross-arm magnitudes: PD's ITL distribution is
internally near-flat (c40: p99 36.25 vs median 35.18 ms) with max-ITL spikes at 141 ms
against single-node's 400 ms. That is a within-arm shape, not a comparison of levels, and
it is consistent with decode never being interrupted by prefill — though not separately
instrumented. **Whether that smoothness is worth its cost is no longer answerable from
this data**, because the ITL-mean penalty it was weighed against (−37%) is one of the
withdrawn numbers; on 8K/1K the equivalent penalty inverted into an advantage once measured
deeply.

**Scope of the withdrawal:** it covers the quantitative comparisons in this section (1P+1D
vs 1-node TP8 at 1K/4K, c20 and c40). PD's untested structural options are unaffected and
remain open: asymmetric xP:yD scaling at fleet size, or a shared KV store that removes the
per-request P→D push (Open items 5/8).

Corollary, with a caveat: a single TP8 node read 1,521 tok/s at c40 on this profile,
which is *consistent with* the un-measured "2× TP8 + LB" idea being attractive. But that
reading is shallow (3 req/slot), and shallow readings inflated throughput by up to 35% on
the 8K/1K workload, so it cannot be used to size a fleet. (The "sub-second median TTFT"
that previously accompanied this sentence is dropped.)

## Open items

0. **Re-measure all TTFT figures with a real measurement window** (blocks any latency
   claim from this report). Every run here used genai-perf's default measurement
   volume — two requests per concurrency slot — so no TTFT number is a steady-state
   value. Requires: `--measurement-interval` sized from the observed request
   throughput (≥10 requests per slot **in each window** — at the ~0.54 req/s TP8 sustained
   at c20 that needs a ~310 s interval, not the 180 s these runs used, which gives only
   ~5.8 requests/slot per window; see the sizing formula in
   [benchmark-commands.md](benchmark-commands.md#measuring-steady-state)),
   `--stability-percentage 10`,
   `--num-prompts` above the total request count, and **≥4 concurrency points** so a
   single outlier can be told apart from a trend. Both TP8 and TP16 must be
   re-measured together; pairing old TP8 numbers with new TP16 numbers is not a
   valid comparison.

   **Done for TP8, TP16 and PD at c20/c40** (2026-07-29 → 07-31). Same engine version
   (`lmsysorg/sglang:v0.5.13.post1`) and the committed configs, on a different
   cluster/account, each A/B'd against 2 req/slot with the server drained to GPU-idle
   between runs. Depths reached: TP8 23.3 req/slot (c20) / 14.7 (c40); TP16 22.1 / 11.6;
   PD 25.4 (c20 only — c40 is not measurable through the router, see below).
   Still open: ≥4 concurrency points per shape (only 2 were captured), and the
   decode-heavy 1K/4K workload.

   TP16 deeper whole-run values, against its shallow counterpart on the same rig:

   | TP16 8K/1K | 2 req/slot | deeper whole-run |
   |---|---|---|
   | c20 total tok/s | 414.0 | **523.6** (22.1 req/slot) |
   | c20 TTFT p50 | 2,103 ms | **839 ms** |
   | c40 total tok/s | 738.7 | **659.1** (11.6 req/slot) |
   | c40 TTFT p50 | 1,347 ms | **848 ms** |

   Note the direction flips between concurrency levels — shallow measurement *understated*
   TP16's c20 throughput by 21% but *overstated* c40 by 12%. Cause not established. The
   TP8-vs-TP16 conclusion this enables is in the Finding 5 box above.

   TP8 deeper whole-run values:

   | c20 | R4 (2 req/slot) | deeper whole-run |
   |---|---|---|
   | TTFT p50 | 3,202 ms | **869 ms** |
   | TTFT p90 | 17,145 ms | **1,514 ms** |
   | Total tok/s | 456 | 552 |

   So R4 **overstated** TTFT by ~3.7× (p50) and ~11× (p90) — the shallow window did not
   flatter these numbers, it inflated them. The deeper run's percentiles are also much more
   tightly grouped (c20 p10 867 → p75 870 ms), but neither that nor R4's wide spread is
   evidence about settling: percentiles carry no time information, and per-window figures
   were not captured for either run.

   ⚠️ The re-measurement also **contradicted this report's assumption that "throughput
   and ITL are much less affected and remain usable"**: at c40, shallow measurement read
   throughput 1,070 tok/s vs 698 at the deeper depth, and ITL 27.4 ms vs 54.5. On this rig throughput
   was the *most* depth-sensitive metric measured; the mechanism was not investigated.
   The c40 rows of this report should be treated as unusable in full, not just their
   TTFT. See [BENCHMARK-METHODOLOGY.md](BENCHMARK-METHODOLOGY.md)
   §"Third rig".

   **PD 1P+1D re-measured 2026-07-31 (c20 only)** — same tool, same workload, drained
   to GPU-idle first, 25.4 requests/slot. Percentiles are tightly grouped (p10 2,887 →
   p50 5,228 → p90 7,695 ms) but that does not demonstrate settling, and only p10/p50/p90
   were transcribed — short of the full distribution the methodology asks for. This
   inverts row P1's conclusions:

   | c20 | P1 (2 req/slot) | deeper whole-run | vs 1-node TP8 (deeper) |
   |---|---|---|---|
   | TTFT p50 | 11,560 ms | **5,228 ms** | 869 ms |
   | Total tok/s | 356 | **602.6** | 552.2 |
   | ITL avg | 25.0 ms | **27.2 ms** | 34.4 ms |
   | tok/s/user | 40.5 | **36.7** | 29.5 |

   The third column is the TP8 figure at the same concurrency, for scale only — **not a
   validated comparison**: both arms are n=1 without a variance estimate and both include
   ramp-up, so no cross-arm claim follows in either direction.

   Shallow measurement read PD's throughput 41% below the deeper value (356 vs 602.6 —
   equivalently, the deeper value is 69% higher). The same comparison on single-node TP8
   moved only 4%. Percentages here are always stated as *error relative to the deeper
   value*, to keep them comparable. What this does establish is **negative**: the published
   claims "PD loses ~32% throughput on 2× the nodes" and "PD's only edge is ITL variance, at
   a 37% ITL-mean cost" both rest on shallow readings that moved by 41% on this shape, so
   neither is supported. It does **not** establish the reverse. Why depth affects PD so much
   more than TP8 was **not investigated** — no queue or KV telemetry was collected.

   ⚠️ **PD at c40 could not be measured.** genai-perf aborts during parsing against the
   sglang-router (`orjson.JSONDecodeError`): of 850,089 response chunks, 72 (0.0085%)
   arrive truncated, and 96% of those become valid JSON when joined with the following
   chunk — i.e. SSE frames split across reads without client-side reassembly. The load
   phase completes (830 requests) but no summary is ever written, so a completed
   20-minute run is discarded. Reproduced twice. The router logged zero errors, so
   which side splits the frames is **not established**. Per
   [benchmark-commands.md](benchmark-commands.md), the fix is *not* to substitute
   another load generator — c40-through-the-router is simply **not measurable with
   genai-perf** at this version.
1. **2× TP8 replicas at c40** — never measured. Any extrapolation must start from the
   deeper single-node figure (552 tok/s at c20 → ~1,104 for two replicas), not the
   shallow 456; the earlier "~912 tok/s" and "~3,000 tok/s on 1K/4K" estimates were built
   on shallow numbers and are withdrawn. Even the corrected figure assumes perfect linear
   scaling with no shared-resource contention, which is itself untested. Requires freeing
   the PD nodes and scaling `glm-5-2` to `replicas: 2`.
2. **TP8-vs-TP16 with uncertainty quantified — still open.** Both shapes were re-measured
   deeply on 2026-07-30/31 (same cluster, same tool, same workload and concurrency, drained
   between runs) and TP16 came out 5–6% behind one node — but each arm is n=1 with no
   variance estimate and includes ramp-up, so the gap is not established. A valid run needs:
   **repeat runs per arm with a confidence interval or at least the run-to-run spread**,
   window-boundary trimming to stationary windows (Step 3), **≥4 concurrency points**, and
   retained `profile_export.json`. It does *not* need matched request counts — under fixed
   concurrency and duration those differ because throughput differs, which is the thing
   being measured. The mechanism is separately unmeasured — no NCCL/EFA counters were ever
   collected.
3. **Re-derive the 07-29→31 numbers from stationary windows only.** Every "steady
   state" figure in this report is a whole-run average including ramp-up; on the L40S rig
   that bias is +5–19% on TTFT p50. The GLM-5.2 per-request exports were not retained, so
   this requires re-running rather than reprocessing.
4. **Raw artifacts stay out of the repo — decided 2026-08-03.** The per-run exports are
   300+ MB each and the aggregate CSVs live in a git-ignored `local/logs/`, so **the
   figures in this report are not independently verifiable from anything committed here**,
   and that is accepted rather than pending. What this obliges instead: future runs must
   **retain `profile_export.json`** long enough to do the last-3-window trim
   ([BENCHMARK-METHODOLOGY.md](BENCHMARK-METHODOLOGY.md) Step 3) — not retaining them is
   why the ramp-up bias had to be quantified on a different rig — and any figure published
   from a run whose raw data is already gone should be labelled as unverifiable, as the
   § Raw data section does for these.
5. PD at 2P:1D ratio (3 nodes) — the last untested PD configuration. Note 1P+1D is **no longer ruled out**: the 8K/1K and 1K/4K conclusions against it have both been withdrawn (§ Decode-heavy workload, Open item 0). Neither is it shown to be ahead — the deeper 8K/1K pair is n=1.
6. MTP acceptance-length telemetry was not collected; draft-token tuning was done on cookbook guidance, not measured accept rates.
7. **Record cache state in future runs** — report the prefix/hierarchical-cache hit rate
   (and the prompt-pool size) alongside each result, so cache warming can be excluded as a
   confounder. It could not be excluded for the 07-07/07-08 runs.
8. ~~LMCache (round B)~~ **done 07-08** (vLLM 22× ✅, sglang incompatible ❌ — see § KV-cache offloading). Still open from it: LMCache **L2 remote backend** (Redis/Mooncake) for cross-node sharing, and the PD variant — decode reading from a shared KV store instead of per-request P→D push.
9. HiCache `ratio=4–6` on p5en (2TB RAM) for larger working sets; and `hicache-storage-backend` (L3: file/mooncake/nixl) for cross-restart persistence.

## Reproduce

The command below is **corrected methodology**, not what produced the tables above.
The original runs omitted `--measurement-interval` / `--stability-percentage` and
therefore measured only two requests per concurrency slot — that is exactly the
defect recorded in the warning at the top. Numbers from this command will not match
the tables; that is the point.

```bash
# client pod: Triton 26.06 SDK (genai-perf + transformers 5.x)
kubectl exec deploy/triton-26-06 -- bash -lc "
genai-perf profile -m zai-org/GLM-5.2-FP8 \
  --url <service>.default.svc.cluster.local:80 \
  --endpoint-type chat --streaming \
  --concurrency 20 \
  --measurement-interval 310000 \
  --stability-percentage 10 \
  --warmup-request-count 20 \
  --num-prompts 2000 \
  --synthetic-input-tokens-mean 8000 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 1024 --output-tokens-stddev 0 \
  --tokenizer zai-org/GLM-5.2-FP8 \
  --extra-inputs max_tokens:1024 \
  --extra-inputs ignore_eos:true \
  --extra-inputs '{\"chat_template_kwargs\":{\"enable_thinking\":false}}'"
# 310 s interval: TP8 at c20 sustained ~0.54 req/s (552 tok/s / 1024 OSL). A window
# needs (10 x 20) / 0.54 = ~370 s to hold 10 requests/slot, and a window is 1.2x the
# interval, so 370 / 1.2 = ~310 s. NOTE the published runs used 180000, which gives only
# ~5.8 requests/slot per window - one reason those figures are provisional.
# Recompute from your own measured throughput: benchmark-commands.md#measuring-steady-state.

# When genai-perf's SSE parser chokes (PD router, c40+): do NOT switch load
# generator. Cross-tool numbers are not comparable and that substitution is what
# made row P2 permanently unusable. Report the point as "not measurable with
# genai-perf" — see benchmark-commands.md.
```

⚠️ **This command alone does not produce a steady-state result.** genai-perf's printed
summary aggregates every window, ramp-up included, so it is contaminated by the transient by
an unknown amount. Two
further steps are required and were **not** applied to any figure in this report:

1. Keep `profile_export.json` and trim using its `window_boundaries` array — drop the
   early windows and keep requests whose *last response* falls inside the ones retained.
   Do not slice on `3 × interval`: a window is always 1.2× the configured interval
   (`inference_profiler.cc:1229-1230`), so that
   cuts into the windows you meant to keep.
2. Publish per-window percentiles, not just whole-run ones — percentiles discard request
   order and cannot show whether the queue settled.

And do **not** use `--request-count` to pin depth across arms: it collapses the run to a
single window, leaving nothing to trim. Procedure and the source references for both points:
[BENCHMARK-METHODOLOGY.md](BENCHMARK-METHODOLOGY.md) Steps 2–3.
