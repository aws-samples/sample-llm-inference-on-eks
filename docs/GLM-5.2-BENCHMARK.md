# GLM-5.2-FP8 on EKS p5en.48xlarge — Benchmark Report

**Date**: 2026-07-07 → 07-08
**Model**: `zai-org/GLM-5.2-FP8` — 753B MoE (DSA sparse attention, 256 experts/top-8, 1 MTP layer), FP8 weights ~756 GB
**Hardware**: p5en.48xlarge (8× H200 141 GB, 16× EFA), Karpenter `reserved-capacity-pool` (us-west-2d capacity block)
**Load generator**: NVIDIA genai-perf 0.0.16.post1 (Triton 26.06 SDK pod, in-cluster); `sglang.bench_serving` for PD-router runs and all decode-heavy runs (genai-perf SSE parser fails against the router at concurrency 40). **Measurement volume was left at genai-perf's defaults — see the warning below; TTFT figures are not steady-state.**

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
> The per-request distributions confirm it had not. At concurrency 20 (row L2) TTFT
> climbs across the whole run — p10 840 ms, p25 878 ms, p50 2,458 ms, p75 10,062 ms,
> p90 14,813 ms, max 17,624 ms — a 21× spread with no plateau: the queue was still
> growing when the run ended. At concurrency 40 (row L3) the distribution flattens
> (p50 1,230 ms, p75 1,541 ms, p90 1,542 ms, max 2,440 ms).
>
> **Throughput and ITL are much less affected** and remain usable — they are
> rate measurements, not queue-depth measurements. Re-measurement of TTFT with
> `--measurement-interval` + `--stability-percentage 10` is tracked in Open Items.
> Correct methodology: [benchmark-commands.md](benchmark-commands.md#measuring-steady-state).

## Executive Summary

Six configurations were benchmarked across four deployment shapes. Key numbers (concurrency as noted):

TTFT columns are struck through: per the warning above they are not steady-state
values. They are left in place for traceability, not for citation.

| # | Shape | Config | Conc. | ~~TTFT p50~~ | ~~TTFT p90/p99~~ | ITL avg | tok/s/user | **Total tok/s** | Stable |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 1-node TP8 SGLang | chunk 2048 (default), mem 0.85, MTP 5-1-6 | 20 | 2,040 ms | 14,656 / 17,160 ms | 27.9 ms | 40.1 | 476 | ✅ |
| R2 | 1-node TP8 SGLang | chunk 32K (16K eff.), mem 0.85 | 20 | — | — | — | — | — | ❌ OOM |
| R4 | 1-node TP8 SGLang | chunk 32K (effective), mem **0.80**, MTP 1-1-2 | 20 | 3,202 ms | 17,145 / 17,790 ms | 27.5 ms | 39.3 | 456 | ✅ |
| V1 | 1-node TP8 vLLM 0.24 | defaults, mem 0.85, MTP 5 drafts | 20 | 1,953 ms | 18,314 / 25,420 ms | 28.7 ms | 38.8 | 454 | ✅ |
| L2 | 2-node TP16 LWS+EFA | mem **0.80**, MTP 1-1-2 | 20 | 2,458 ms | 14,813 / 17,365 ms | 32.4 ms | 32.9 | 396 | ✅ |
| L3 | 2-node TP16 LWS+EFA | (same) | **40** | **1,230 ms** | **1,542 / 1,936 ms** | 41.1 ms | 26.2 | **730** | ✅ |
| P1 | 2-node PD 1P+1D NIXL | both TP8, mem 0.85 | 20 | 11,560 ms | 35,258 / 38,412 ms | **25.0 ms** | 40.5 | 356 | ✅ |
| P2 | 2-node PD 1P+1D NIXL | (same, bench_serving) | 40 | 17,246 ms (median) | — / 53,122 ms | 31.7 ms | ~31 | 712 | ✅ |

### Recommendation matrix (for this 8K-in / 1K-out, prefill-heavy workload)

| Priority | Recommended shape | Why |
|---|---|---|
| Max throughput per dollar | **2× independent TP8 replicas** (extrapolated ~912 tok/s) | No cross-node allreduce tax; each node at full efficiency. *Not yet measured — see Open Items.* |
| Hard TTFT SLO at high concurrency | **Undetermined — needs re-measurement** | The previous recommendation rested on TP16's concurrency-40 TTFT p99, which is not a steady-state number (see warning above). TP16's *throughput* advantage at concurrency 40 (730 tok/s) stands. |
| Low concurrency / single-node budget | 1-node TP8, SGLang or vLLM (they tie) | ~455 tok/s wall; fine for ≤10 concurrent 8K requests |
| Strict ITL SLO or decode-heavy workloads | ~~PD disaggregation~~ **1-node TP8** | PD 1P+1D lost the decode-heavy rematch too (see § Decode-heavy workload); its only surviving edge is ITL *variance*, at a 37% ITL-mean cost |

## Findings

### 1. The single-node wall is prefill compute, not scheduling

Chunked-prefill size was swept 2048 → 16384 → 32768 on TP8: total throughput stayed at 456–476 tok/s and TTFT tails did not move. vLLM on all-default settings landed at 454 tok/s — two engines, three schedulers, same number. At 20 concurrent 8K prompts the prefill arrival rate saturates 8× H200 compute in a co-located setup; scheduling parameters only redistribute the queue, they cannot shrink it. The SGLang cookbook's chunked-prefill gains (+34–78 %) did not reproduce on this workload/config.

### 2. `--mem-fraction-static` safe line is 0.80, single- and multi-node

0.85 OOM-crashed twice under load, with different triggers:

- **TP8 + 32K chunk** (R2): large-chunk prefill activations spiked ~3 GiB/GPU against ~2 GiB free → all 8 ranks OOM mid-benchmark.
- **TP16 + default 8K chunk** (L1, initial deploy): the static pool is a *percentage of total VRAM*, so TP16's per-GPU weight savings (~94 GiB → ~47 GiB) were silently absorbed by the KV pool, leaving only ~600 MiB activation headroom → OOM at concurrency 20.

Dropping to 0.80 frees ~7 GiB/GPU for activations; both shapes then survived every run with zero restarts. Multi-node does **not** automatically gain memory headroom — the fraction semantics guarantee it doesn't.

### 3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is incompatible with TP serving

Attempted as a fragmentation mitigation; crashed at startup — custom all-reduce cannot IPC-register VMM-backed allocations (`custom_all_reduce.cuh:37: CUDA error: invalid argument` during CUDA graph capture). Do not combine with multi-GPU TP.

### 4. `--chunked-prefill-size` is silently capped by `--max-prefill-tokens`

SGLang's default `max_prefill_tokens=16384` caps the effective chunk. Setting `--chunked-prefill-size=32768` alone gives you 16K chunks; both flags must be raised together. (Made no difference here — see Finding 1 — but matters on workloads that are genuinely chunk-bound.)

### 5. TP16 needs concurrency ≥ ~40 to justify itself

At concurrency 20, TP16 was *worse* than TP8 (396 vs 456 tok/s): 16 GPUs under-fed, and every decode step pays two cross-node EFA allreduces (ITL 27.5 → 32.4 ms). At concurrency 40 throughput rose to 730 tok/s — an 1.84× gain over its own concurrency-20 result. The cost is per-user decode speed (26 vs 39 tok/s) — the allreduce tax again.

> [!CAUTION]
> **Retracted (2026-07-28):** this finding previously claimed TTFT p90 "collapsed
> from 14.8 s to 1.5 s (10×)" at concurrency 40, attributed to prefill bursts being
> absorbed whole. That comparison is withdrawn — both runs measured only two
> requests per concurrency slot, and the concurrency-20 distribution never reached a
> plateau (see the warning at the top of this report). Whether TP16 genuinely
> improves TTFT at higher concurrency is **unresolved** and needs re-measurement
> with a proper measurement window; the throughput figures above are unaffected.
>
> A follow-up experiment on cheaper hardware (1× L40S, Qwen3-8B, 2K/256) confirmed
> that short measurement windows understate TTFT badly, but did **not** reproduce a
> direction reversal — TTFT degraded monotonically with concurrency under every
> measurement length tested. So the mechanism behind this specific 10× figure
> remains unexplained; it is not simply an artifact of measuring too briefly.

### 6. PD 1P:1D is the wrong ratio for an 8:1 ISL:OSL workload

PD delivered its core promise — decode purity (ITL p99 27.5 ms, max 108 ms; per-user floor 36 tok/s) — but at concurrency 40 the single prefill node became the whole bottleneck: TTFT median 17 s, p99 53 s, while the decode node idled (peak output 1013 tok/s vs 712 sustained shows the spare capacity). Input throughput matched TP16 (~5.6K tok/s), confirming equal total compute — PD just partitions it statically, and this workload needs most of it on prefill. At the time we hypothesized PD might still win on decode-heavy traffic — tested on 07-08, see § Decode-heavy workload: it does not.

### 7. Engine scheduling styles differ where the wall doesn't

At the same 455 tok/s wall, vLLM favors the median user (TTFT p50 1.9 s, TTST 640 ms) while SGLang bounds the tail better (TTFT max 17.8 s vs 25.7 s). Pick by SLO shape; throughput won't change.

## Deployment shapes tested

| Shape | Manifest | Image |
|---|---|---|
| 1-node TP8 SGLang | `sglang/glm-5.2-fp8-p5en.yaml` (R4 config committed) | `lmsysorg/sglang:v0.5.13.post1` |
| 1-node TP8 vLLM | deployed ad-hoc (`glm-5-2-vllm`), not committed | `vllm/vllm-openai:v0.24.0` |
| 2-node TP16 LWS | `lws/lws-glm-5.2-tp16-p5en.yaml` (0.80 + MTP 1-1-2 committed) | ECR `sglang-efa-p5:v0.5.13.post1-nixl` (EFA 1.49 + aws-ofi-nccl + NIXL) |
| 2-node PD 1P+1D | deployed ad-hoc (`lws-glm-5-2-prefill`/`-decode` + `glm-5-2-router`), manifest: `lws/lws-glm-5.2-pd-p5en.yaml` | same ECR image |

GLM-5.2 requires SGLang ≥ v0.5.13.post1 (`glm_moe_dsa` arch) and transformers ≥ 5.x for its tokenizer — old Triton SDK images (≤24.12) cannot tokenize it; use 26.06+.

## Crash post-mortems

| Event | Trigger | Root cause | Fix |
|---|---|---|---|
| TP8 OOM under load | c20 benchmark, ~4 min in | 0.85 static pool + 32K-chunk activation spikes | mem 0.80 |
| TP8 startup crash | CUDA graph capture | `expandable_segments` vs custom all-reduce IPC | remove env var |
| TP16 OOM under load | c20 benchmark, ~4 min in | 0.85 static pool absorbs TP16 weight savings into KV | mem 0.80 |
| genai-perf c40 failures ×2 | router streaming at c40 | genai-perf 0.0.16 SSE parser (`splintered SSE`, `orjson` error); backends healthy | use `sglang.bench_serving` |

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
> which is short of steady state. Treat the TTFT rows as indicative of the
> TP8-vs-PD *gap* (a 5× difference is well outside what measurement length explains)
> rather than as absolute latency. Throughput and ITL are unaffected.

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

**Conclusion — PD 1P+1D has no applicable regime on this model/hardware.**
It lost on its home turf: −32%/−27% throughput (c20/c40) while consuming 2× the nodes
(≈37% per-node efficiency), TTFT 4–5× worse, ITL mean 37–49% slower. The one promise PD
kept is ITL *smoothness* — its ITL distribution is near-zero-variance (c40: p99 36.25 vs
median 35.18) and max-ITL spikes halve (141 vs 400 ms) because decode is never interrupted
by prefill. But that trades a permanent 37% ITL-mean regression for the removal of
occasional spikes — no realistic SLO prefers that. Combined with the 07-07 prefill-heavy
result, both workload extremes are now measured, and 1-node TP8 wins both. PD's remaining
untested chances are structural, not workload-shaped: asymmetric xP:yD scaling at fleet
size, or a shared KV store that removes the per-request P→D push (Open items 2/4).

Corollary: the decode-heavy numbers also strengthen the un-measured "2× TP8 + LB"
recommendation — a single TP8 node already sustains 1,521 tok/s at c40 on this
profile. (The "sub-second median TTFT" that previously accompanied this sentence is
dropped — see the note above on measurement volume.)

## Open items

0. **Re-measure all TTFT figures with a real measurement window** (blocks any latency
   claim from this report). Every run here used genai-perf's default measurement
   volume — two requests per concurrency slot — so no TTFT number is a steady-state
   value. Requires: `--measurement-interval` sized from the observed request
   throughput (≥10 requests per slot across the 3 windows perf_analyzer averages —
   at the 0.39 req/s measured for TP16 c20 that is ~180 s; see the sizing formula in
   [benchmark-commands.md](benchmark-commands.md#measuring-steady-state)),
   `--stability-percentage 10`,
   `--num-prompts` above the total request count, and **≥4 concurrency points** so a
   single outlier can be told apart from a trend. Both TP8 and TP16 must be
   re-measured together; pairing old TP8 numbers with new TP16 numbers is not a
   valid comparison. Needs 2× p5en.48xlarge, and the us-west-2d capacity block has
   expired.
1. **2× TP8 replicas at c40** — the throughput-per-dollar favorite is extrapolated (456 × 2 ≈ 912 tok/s on 8K/1K; ~3,000 tok/s on 1K/4K), not measured. Requires freeing the PD nodes and scaling `glm-5-2` to `replicas: 2`.
2. PD at 2P:1D ratio (3 nodes) — the last untested PD configuration, only relevant if fleet-level asymmetric scaling is on the table (1P+1D is ruled out for both workload shapes).
3. MTP acceptance-length telemetry was not collected; draft-token tuning was done on cookbook guidance, not measured accept rates.
4. ~~LMCache (round B)~~ **done 07-08** (vLLM 22× ✅, sglang incompatible ❌ — see § KV-cache offloading). Still open from it: LMCache **L2 remote backend** (Redis/Mooncake) for cross-node sharing, and the PD variant — decode reading from a shared KV store instead of per-request P→D push.
5. HiCache `ratio=4–6` on p5en (2TB RAM) for larger working sets; and `hicache-storage-backend` (L3: file/mooncake/nixl) for cross-restart persistence.

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
  --measurement-interval 180000 \
  --stability-percentage 10 \
  --warmup-request-count 20 \
  --num-prompts 2000 \
  --synthetic-input-tokens-mean 8000 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 1024 --output-tokens-stddev 0 \
  --tokenizer zai-org/GLM-5.2-FP8 \
  --extra-inputs max_tokens:1024 \
  --extra-inputs ignore_eos:true \
  --extra-inputs '{\"chat_template_kwargs\":{\"enable_thinking\":false}}'"
# 180 s window: at the ~0.39 req/s this workload sustained at c20, three windows
# give ~210 requests (>10 per slot). Scale it with measured request throughput —
# see benchmark-commands.md#measuring-steady-state.

# alternative when genai-perf's SSE parser chokes (PD router, high concurrency):
python3 -m sglang.bench_serving --backend sglang-oai-chat \
  --base-url http://<service>:80 --model zai-org/GLM-5.2-FP8 \
  --dataset-name random --random-input-len 8000 --random-output-len 1024 \
  --random-range-ratio 1.0 --num-prompts 150 --max-concurrency 40 \
  --extra-request-body '{"chat_template_kwargs":{"enable_thinking":false}}'
```
