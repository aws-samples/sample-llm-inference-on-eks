# Kimi-K3 Multi-Node Performance on 3× p5en (TP8×PP3)

This document states the measured performance of `moonshotai/Kimi-K3` deployed
across three p5en.48xlarge instances as tensor-parallel 8 × pipeline-parallel 3
(24× H200). All data was measured on an EKS cluster on 2026-08-09. It states
observed facts and marks every inference as such.

For *why* this topology rather than TP24 or PP2, see
[KIMI-K3-PP-TOPOLOGY.md](KIMI-K3-PP-TOPOLOGY.md). For the single-node B300
deployment see [KIMI-K3-B300-PERFORMANCE.md](KIMI-K3-B300-PERFORMANCE.md).

> [!WARNING]
> **Read these limits before quoting any number below.**
>
> 1. **A 1,048,576-token window is servable but not interactively usable.**
>    Measured 2026-08-12: a **1,037,947-token** request returned the correct answer
>    with 5 of 5 needles recalled, and its TTFT was **126.1 s** — 94% of end-to-end
>    latency. That cost is paid on *every* request, because this model cannot use
>    prefix caching (§ 3.1). Treat the full window as a **batch-only** capability.
>    The usable ceiling for a product is **128,000 tokens** (TTFT 8.44 s) and the
>    comfortable ceiling is **31,999 tokens** (2.24 s) — see § 4.1 for the curve.
> 2. **Output length fell short of the request at every point.**
>    `output_sequence_length` came out 630–650 against 1024 requested (and 246–248
>    against 256) despite `ignore_eos:true`. The decode phase is therefore ~37%
>    shorter than designed, so prefill occupies a larger share of each request
>    than intended and the throughput figures **understate pure decode capacity by
>    an unquantified amount**. Cause not established. The same shortfall appears on
>    the 2-node TP16 baseline (247.35 of 256), so it is not specific to PP3.
> 3. **The peak-throughput sweep (§ 5) used fixed request counts, not stability
>    detection** — single window, so ramp-up cannot be trimmed out of it. § 4 and
>    § 7 *are* stability-detected and trimmed. Do not put figures from the two
>    methods in one table.

## 1. Subject and environment

The model is Kimi-K3, a 2.8-trillion-parameter Mixture-of-Experts architecture
built on Kimi Delta Attention, with a context window of 1,048,576 tokens. Weights
are MXFP4, approximately 1.56 TB across 118 files.

The hardware is 3× p5en.48xlarge — 8× H200 each, 24 GPUs total, 143,771 MiB per
GPU. All three nodes must sit in **one availability zone**: EFA is an intra-AZ
fabric, and a cross-AZ group cannot form at all (§ 8). Capacity was obtained as
spot instances.

The engine is vLLM `0.1.dev19262+gb6bbf29dd.d20260727` (a pre-release nightly; K3
support is not in any stable release as of this writing), in a locally built image
adding EFA userspace plus one upstream patch — see § 3, which is not optional for
long-context use. Engine arguments follow the official recipe's
[h200 multi_node_tp_pp profile](https://recipes.vllm.ai/moonshotai/Kimi-K3?hardware=h200)
with `pipeline-parallel-size` and `nnodes` raised from 2 to 3:
TP8, PP3, `--gpu-memory-utilization 0.93`, `--max-model-len 1048576`,
`--max-num-batched-tokens 4096`, `--moe-backend marlin`,
`--attention-backend FLASHMLA`, `--disable-custom-all-reduce`.
`--max-num-seqs` is discussed in § 6 — the recipe's value of 5 is a bottleneck on
this shape. Manifest: `k8s-manifest/lws/lws-kimi-k3-tp8pp3-p5en.yaml`.

The load generator is NVIDIA genai-perf 0.0.16.post1 in an in-cluster Triton 26.06
SDK pod, over ClusterIP to the streaming `/v1/chat/completions` endpoint.
Thinking was disabled for every run
(`{"chat_template_kwargs":{"enable_thinking":false}}`); verified per deployment by
checking that a probe response carried `content` and `reasoning: null`.

## 2. Memory and KV capacity

Per-GPU, as reported by the engine at `--gpu-memory-utilization 0.93`:

| Item | Value |
|---|---|
| GPU memory total | 143,771 MiB |
| used at idle, model loaded | 129,193 MiB |
| free at idle | ~14.4 GiB |
| weights + non-torch | 68.89 GiB |
| peak activation (profiled) | 1.29–2.19 GiB |
| KV cache | ~63.6 GiB |

Engine-reported KV pool: **6,361,156 tokens**, block size 384. A full-window request
costs `cdiv(1048576, 384) + 1 = 2,732` blocks (the `+1` is the fixed KDA state block;
see [KIMI-K3-PP-TOPOLOGY.md](KIMI-K3-PP-TOPOLOGY.md) § 4). The pool holds roughly
16,574 blocks — *roughly*, because that token figure is itself derived rather than
physical and cannot be divided by the block size to recover a count; see the note in
[KIMI-K3-PP-TOPOLOGY.md](KIMI-K3-PP-TOPOLOGY.md) § 6. Hence the engine's own line:

```
Maximum concurrency for 1,048,576 tokens per request: 6.07x
```

That figure is **KV capacity arithmetic computed once at startup, not an
achievable concurrency**. It is derived, not physical: `kv_cache_utils.py:958`
divides `num_blocks` by the per-request block total, and the token figure on the
line above it is then back-computed as `max_concurrency × max_model_len`
(`:2227`) — so do not divide 6,361,156 by 384 to recover a block count, it does
not yield an integer. Measured behaviour at a full window differs from the
figure in both directions: the scheduler kept only 2–4 requests resident (KV peak
42.5–65.8%) rather than 6, and output quality degraded as concurrency rose past
2 — control tokens leaked into the text at 4 concurrent requests and the text
collapsed at 6. **Cause not established**; it is unrelated to the § 3 crash
(`IndexError` count stayed 0) and no preemption occurred. Anything above 2
concurrent full-window requests is unvalidated.

## 3. A chunk-prefill crash blocks long context on stock images

**No released vLLM can serve long context on this shape.** Any input larger than
`--max-num-batched-tokens` is chunk-prefilled, and before the fix every chunk
boundary after the first threw:

```
IndexError: index_fill_(): Expected dtype int64 for index
  at vllm/v1/worker/gpu/model_states/mamba_hybrid.py:314 postprocess_state
```

Requests still returned HTTP 200 and the engine stayed up, so the failure was
**silent** — the KDA state hand-off was simply skipped. It fired **1,280 times**
during one 128K run, which is why an earlier attempt at long-context measurement
was discarded rather than reported.

Root cause, read from the running image: `idx_mapping` is created int32
(`vllm/v1/worker/gpu/input_batch.py:114`) and `torch.index_fill_` accepts only
int64 indices. The scalar branch is reached because `model_runner.py:1445` passes a
literal `0` for `num_sampled` on non-final prefill chunks. The sibling tensor
branch already used a Triton kernel, which takes int32 natively — the scalar branch
was missed.

The fix is upstream
[PR #50327](https://github.com/vllm-project/vllm/pull/50327), merged 2026-08-03 as
commit `e578de31`, which replaces `index_fill_` with a new
`_fill_num_accepted_kernel` rather than casting. **It is in no release**: v0.26.0
shipped 2026-07-27, seven days earlier — verified by reading the file at that tag
on 2026-08-09, when issue
[#50947](https://github.com/vllm-project/vllm/issues/50947) was still open. This
repo backports it verbatim
(`k8s-manifest/lws/Dockerfile.efa-vllm-kimi-k3-p50327`); that layer is a stopgap
and should be deleted once a release carries the commit.

Post-fix verification on the deployed image:

| Probe | Result |
|---|---|
| 18,029-token input (4.4× the chunk threshold) | HTTP 200; `IndexError` **0** on all 3 pods |
| 91,700-token needle-in-haystack, needles at ~15K / ~46K / ~76K | **3/3 recalled exactly** |
| 128K benchmark, 124 requests × ~31 chunks each | `IndexError` **0** |

The needle test matters beyond the crash: pre-fix the exception meant the KDA state
hand-off was skipped while requests still looked successful. Post-fix it runs, and
long-range retrieval across ~22 chunk boundaries works. **Scope:** one
temperature-0 retrieval test. It shows long-context attention functions on the
chunk-prefill path; it is **not** a general output-quality evaluation and does not
rule out degradation on other axes.

### 3.1 Prefix caching is unavailable, so every request repays the full prefill

This is the single fact that decides whether a long window is practical, and it is
easy to miss because nothing in the manifest turns it off.

vLLM's own default is on (`vllm/config/cache.py:93`, `enable_prefix_caching: bool =
True`), yet this deployment reports `enable_prefix_caching=False` and
`Prefix cache hit rate: 0.0%` for the entire life of the process. The matching
force-disable path is:

```python
# vllm/v1/engine/core.py:268-272
if vllm_config.cache_config.enable_prefix_caching:
    logger.info("Disabling prefix caching: model has non-causal attention layers.")
    vllm_config.cache_config.enable_prefix_caching = False
```

K3's 69 KDA layers hold a recurrent state, so a decode token's KV is not a pure
causal function of the prefix and cannot be reused across requests.

⚠️ **Inference, not confirmation:** the `Disabling prefix caching` line was not
found in this deployment's startup log. What is confirmed is
`enable_prefix_caching=False` in the engine config dump and a 0.0% hit rate
throughout. Two other force-disable paths exist in this vLLM
(`models/config.py:166`, Unlimited-OCR-specific; `mla_attention.py:479`, for
TRITON_MLA/FLASHINFER under batch invariance) and **neither matches this
configuration** — this one runs FLASHMLA. Confirming the exact path needs the log
line or a debugger.

**Consequence for deployment.** "Load a large document once, then ask many
questions" — the main real use for a huge window — does not work here: each
follow-up question repays the full prefill, so a 1,037,947-token document costs
126.1 s *per turn*, not once.

**A flag that does not do what it looks like:** the single-node B300 manifest
(`k8s-manifest/vllm/kimi-k3-p6-b300-vllm.yaml`) passes
`--enable-prefix-caching`. The same code path overrides it, so that flag buys
nothing on this model. Do not copy it expecting caching.

## 4. Concurrency scaling at short context

ISL 1000 / OSL 256, `--max-num-seqs 32`, stability-detected with
`--stability-percentage 10`, **ramp-up window trimmed** — these are steady-state
figures, not genai-perf's printed summary (which aggregates all windows including
ramp-up).

| Concurrency | TTFT p50 (ms) | ITL p50 (ms) | latency p50 (ms) | req/s | output tok/s | per-user tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 237.3 | 16.87 | 4,588 | 0.222 | 57.0 | 59.3 |
| 4 | 515.1 | 20.14 | 5,717 | 0.694 | 172.2 | 49.7 |
| 8 | 743.1 | 23.04 | 6,610 | 1.238 | 312.6 | 43.4 |
| 16 | 1,147.7 | 25.97 | 7,827 | 2.000 | 505.5 | 31.6 |
| 32 | 1,440.0 | 34.38 | 10,289 | 3.110 | 784.3 | 24.5 |

Per-window depth 10.0–16.0 requests/slot, meeting the ≥10 bar in
[BENCHMARK-METHODOLOGY.md](BENCHMARK-METHODOLOGY.md); measurement intervals were
derived from measured throughput per point, not copied. The engine was drained to
`Reqs Running: 0, Waiting: 0` before every run, and engine-side sampling showed
`Waiting: 0` throughout each — the requested concurrency was real, not queueing.

**This is not evidence that PP3 beats TP16.** The 2-node TP16 baseline (TTFT p50
505 ms, ITL p50 33.4 ms, 109.5 tok/s at c4) was measured at a different time on
16 GPUs; PP3 has 24, i.e. 50% more compute. Attributing any difference to the
*topology* would need a same-node-count TP24 vs TP8×PP3 A/B, and TP24 cannot start
at all ([KIMI-K3-PP-TOPOLOGY.md](KIMI-K3-PP-TOPOLOGY.md) § 2). **Cause not
established.**

### 4.1 TTFT against context length — where the usable ceiling is

Measured 2026-08-12, concurrency 1, OSL 256 requested, `--request-count 3` per
point, engine drained to idle between points. Concurrency 1 is the right lens here:
§ 3.1 means every request repays its full prefill, so the single-request floor is
what a user actually experiences. The 1,030,004 row comes from the same sweep at
OSL 1024.

| ISL | TTFT p50 | ITL p50 | end-to-end p50 | per-user tok/s | prefill tok/s |
|---:|---:|---:|---:|---:|---:|
| 7,999 | **0.87 s** | 17.48 ms | 5.33 s | 57.20 | 9,174 |
| 31,999 | **2.24 s** | 17.90 ms | 6.80 s | 55.98 | 14,317 |
| 128,000 | **8.44 s** | 18.46 ms | 13.15 s | 54.17 | 15,165 |
| 262,001 | **19.19 s** | 19.35 ms | 24.13 s | 51.67 | 13,652 |
| 524,002 | **47.00 s** | 20.97 ms | 52.32 s | 47.64 | 11,150 |
| 1,030,004 | **126.1 s** | 24.07 ms | 135.1 s | 41.53 | 8,168 |

**TTFT grows faster than context.** Each doubling costs 2.3–2.7× the TTFT, and
prefill throughput peaks at **15,165 tok/s around 128,000 tokens** then falls to
8,168 at 1,030,004 — a 46% decline. Attention cost rising with sequence length is
the expected explanation but was **not isolated here**; no profiling was run.

**ITL is nearly independent of context length** — 17.48 ms at 7,999 tokens versus
24.07 ms at 1,030,004, only 38% worse across a 129× range. Decode speed is not the
problem at long context; the entire cost is in the first token. (Contrast the ITL
blow-up to 1,305.79 ms seen at a full window with 6 concurrent requests — that is a
concurrency effect, not a length effect.)

Reading the curve for deployment:

| ISL | TTFT p50 | Suitability |
|---|---|---|
| ≤ 31,999 | ≤ 2.24 s | interactive chat |
| 128,000 | 8.44 s | acceptable — practical ceiling for document Q&A |
| 262,001 | 19.19 s | needs a progress indicator |
| ≥ 524,002 | ≥ 47 s | batch only |

Same caveats as § 5: 3 requests/slot, single window, so these are capability
probes and not steady-state figures.

## 5. Peak aggregate throughput: ≥2,169 tok/s, not saturated

To isolate *decode* capacity from prefill interference: short input (ISL 1000) so
prefill is a small share of each request, long output (OSL 1024 requested), then
climb concurrency until aggregate throughput stops rising. `--max-num-seqs` was
raised to 128 and then 512 so it would not be the binding constraint.
`--request-count 4×C` instead of stability detection (see § 9 on cost). Engine-side
sampling confirmed `Running: C, Waiting: 0` in steady state at **every** point.

| Concurrency | output tok/s | req/s | TTFT p50 (ms) | ITL p50 (ms) | per-user tok/s | gain vs prev |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 600.6 | 0.927 | 1,445 | 32.4 | 30.7 | — |
| 64 | 831.1 | 1.316 | 1,488 | 45.9 | 21.7 | 1.384× |
| 96 | 986.9 | 1.566 | 1,605 | 58.0 | 17.1 | 1.187× |
| 128 | 1,133.8 | 1.774 | 1,622 | 68.4 | 14.7 | 1.149× |
| 192 | 1,363.8 | 2.128 | 1,689 | 85.5 | 11.6 | 1.203× |
| 256 | 1,584.2 | 2.437 | 1,621 | 100.1 | 10.0 | 1.162× |
| 384 | 1,880.2 | 2.970 | 1,733 | 124.1 | 8.1 | 1.187× |
| 512 | **2,168.9** | 3.336 | 1,753 | 146.9 | 6.8 | 1.154× |

**Throughput never plateaued.** Past c128 the gain settles at a steady 1.15–1.20×
and does not keep decaying — the dip at c96/c128 looked like the onset of
saturation but was not. Aggregate rose 3.6× from c32 to c512, so **2,169 tok/s is
a floor, not this server's ceiling.**

**What saturates is latency, not throughput.** TTFT barely moves (1,445 → 1,753 ms,
+21%) because requests enter the batch immediately; the whole cost lands on ITL:
32.4 → 146.9 ms, 4.5× worse, with per-user speed falling to 22% of its c32 value.
At c512 a user receives 6.8 tok/s — slower than comfortable reading. **A throughput
number at that concurrency describes a service nobody would use.**

Useful reading for capacity planning is the band, not the maximum:

| Per-user experience | Concurrency | Aggregate |
|---|---|---|
| comfortable (≥20 tok/s) | 32–64 | 600–831 tok/s |
| acceptable (≥10 tok/s) | 96–256 | 987–1,584 tok/s |
| throughput-figure only | 384+ | 1,880–2,169+ tok/s |

KV is not the constraint at short context: at ~2,024 tokens/request, c512 uses
1,036,288 of 6,319,795 tokens = **16.3%**.

## 6. `--max-num-seqs`: the recipe's 5 is a bottleneck here

The recipe sets 5, a value that came from the 2-node TP16 shape whose KV pool held
about one full-length request. On this shape a c4 run at ISL 1000 used **0.2%** of
the pool, so 5 caps the benchmark rather than the hardware. Raising it is what
made § 5 measurable at all.

Raising it does **not** cost activation memory, which is worth stating precisely
because the PP2 shape did once CUDA-OOM. Activation profiling is sized by
`max_num_batched_tokens`, not by this flag: `profile_run` calls
`_dummy_run(self.max_num_tokens, …)` and `max_num_tokens :=
scheduler_config.max_num_batched_tokens` (`vllm/v1/worker/gpu/model_runner.py:153`
and `:693`). Measured across three deployments, activation stayed at 1.29–2.19 GiB
at all of 32 / 128 / 512.

It does cost a little KV, and the relationship is linear rather than noise:

| `--max-num-seqs` | KV pool (tokens) | vs 32 | vs the line below |
|---:|---:|---:|---:|
| 5 | 6,363,454 | +2,298 | 0.0004% |
| 32 | 6,361,156 | — | fitted |
| 128 | 6,355,029 | −6,127 (−0.10%) | 0.034% |
| 512 | 6,319,795 | −41,361 (−0.65%) | fitted |

Fitting the two extremes 32 and 512 gives `KV ≈ 6,363,913 − 86.17 × max_num_seqs`
tokens. The other two points land on that line to within **0.034% and 0.0004%** —
and the `max_num_seqs=5` point came from a **separate deployment** on a different
day (a cold start that accidentally used a stale rendered manifest), so the
relationship reproduces across process restarts rather than being an artifact of one
boot. Too regular to be measurement noise.

**The mechanism is still not established.** 86 tokens/slot works out to roughly
900 KB of HBM per slot, three orders of magnitude more than the O(`max_num_seqs`)
bookkeeping tensors that are easy to point at (e.g. `num_accepted_tokens_gpu`, an
int32 array of `max_num_reqs`, `mamba_hybrid.py:75-77`). A per-request block table
would be the right order of magnitude at a 1M window, but that was **not
verified** — no allocation was traced.

Practical consequence: the cost is irrelevant at short context, and only marginal
even at a full window — at 512 it costs 41,361 tokens, about 4% of one 1,048,576-token
request's budget. There is no reason to raise it on a full-window deployment though,
since concurrency there is capped at 2 by output quality (§ 2) long before KV runs
out. The committed manifest ships 32 as a serving default — 512 is where the
≥2,169 tok/s figure came from, and is not the default because of the per-user latency
above.

## 7. Long context: 128K measured

ISL 128,000 / OSL 1024 requested, concurrency 4, patched image,
stability-detected and trimmed to windows 1–3 (N=124, 1,800 s). `IndexError` was 0
throughout, and the engine reported `Running: 4, Waiting: 0` on 54 of 60 samples.

| Metric | avg | p50 | p90 | p99 |
|---|---|---|---|---|
| TTFT (ms) | 11,336 | 11,696 | 12,293 | 12,299 |
| ITL (ms) | 320.5 | **49.2** | 293.3 | 1,309.8 |
| request latency (ms) | 58,184 | 59,849 | 59,896 | 60,313 |

0.0689 req/s, 47.1 output tok/s, 684 tokens/request. Window depths 9.2 / 10.0 /
10.2 / 10.8.

**ITL avg 320 against p50 49 — a 6.5× gap — is the headline, not the average.** It
is the latency projection of the prefill/decode alternation in § 7.1: most token
gaps are ~49 ms, but those landing in a prefill phase wait hundreds of ms, and p99
reaches 1.3 s. Quoting the average alone would misrepresent the experience in both
directions.

Set against § 5, the same server delivers **≥2,169 tok/s at ISL 1000 and 47.1 tok/s
at ISL 128K — a 46× spread.** Any single-number claim about "output capacity" must
name its input length.

### 7.1 Prefill and decode alternate in long phases

Engine logs at 128K, sampled every 10 s:

```
prefill=12801.9  gen=55.1      prefill=12801.7  gen=3.3
prefill=0.0      gen=200.6     prefill=12801.6  gen=3.2
prefill=0.0      gen=145.3     prefill=25604.2  gen=4.5
prefill=12801.6  gen=3.3       prefill=0.0      gen=200.4
```

Decode collapses to 3.2–3.4 tok/s while `prefill` reads non-zero, and `prefill`
reads exactly 0.0 while decode runs at ~200 tok/s.

> [!CAUTION]
> **`Avg prompt tput` is not a rate — do not read it as one.** It is the prompt
> tokens of prefills that *completed* inside the 10 s reporting window, divided by
> 10 s, so it quantises to multiples of ISL/10. Verified against three observed
> values, to within 0.002%:
>
> | observed | ISL × completions / 10 s |
> |---|---|
> | 12,801.9 | 128,019 × 1 / 10 |
> | 25,604.2 | 128,019 × 2 / 10 |
> | 103,790.2 | 1,037,925 × 1 / 10 |
>
> So `prefill=0.0` does **not** mean prefill stopped — a 1,030,004-token prefill
> spans a dozen windows and reports 0.0 in all but the last. And the "pinned"
> 12,801.x is just one 128K prompt finishing per window.

**A retracted conclusion.** An earlier version of this section argued that because
raising `--max-num-batched-tokens` 4096 → 16384 left the prefill figure unchanged at
~12,801, *something other than that flag must cap per-step tokens*. **That inference
is withdrawn** — the figure was never a rate, so it could not have responded to a
budget change. No evidence here supports an unknown per-step cap.

What the same A/B did establish, from independent measurements, is that 16384 is a
net loss on this shape, and it was reverted for these reasons:

| | 4096 | 16384 |
|---|---|---|
| peak activation | 1.29–2.19 GiB | 4.49 GiB |
| KV pool | 6,361,156 tokens | 5,485,683 (−13.8%) |
| decode during prefill | 3.2–3.4 tok/s | 0.7–0.9 tok/s |

The alternation itself remains **unexplained**: the scheduler does share one
per-step token budget across running and waiting requests
(`vllm/v1/core/sched/scheduler.py:110-114`, `:454`, `:480`, `:518`), which would
starve 1-token decodes behind a large prefill chunk, but that was not
demonstrated here. `--long-prefill-token-threshold` (`scheduler.py:516-517`,
default off) is the upstream knob aimed at this case and is **untested**.

## 8. Operational notes

**EFA is intra-AZ; a cross-AZ group cannot form.** With only an instance-type
nodeSelector, the scheduler placed leader and worker in different AZs and NCCL died
at connect time with `NET/OFI CM: Unable to insert remote address into address
vector` / `Failed call to fi_av_insert`. The manifest pins all pods to one zone via
self-selecting `podAffinity` on `topology.kubernetes.io/zone`. Do **not** substitute
the LeaderWorkerSet `exclusive-topology` annotation — the controller turns it into
`podAntiAffinity`, which does the opposite. With 3 nodes the odds of a split are
higher than with 2.

**`privileged: true` is not what grants EFA access.** The EFA device plugin injects
`/dev/infiniband` whenever a pod requests `vpc.amazonaws.com/efa`; NCCL still chose
`efa` with GDRDMA after removing `privileged`, the added capabilities, and the
`/dev/infiniband` hostPath.

**The group must restart as a unit.** `restartPolicy:
RecreateGroupOnPodRestart`, not `None`. With `None`, a crash-looping leader creates
a fresh TCPStore that the workers' already-joined ranks never rejoin, and rendezvous
stalls forever — vLLM's TCPStore rendezvous has no re-join path. Three pods make
this more acute than two.

**Weight loading.** ~1.56 TB per node from S3 to node-local NVMe: about 26 minutes
on cold nodes, 5–6 minutes when the NVMe cache is warm, so config changes that only
restart pods are cheap while node replacement is not.

**Drain between runs.** Back-to-back runs inherit the previous run's queue backlog;
on another rig in this repo that moved TTFT p50 by 34%, larger than the effect being
studied. Gate on `Reqs Running: 0, Waiting: 0`.

**`kubectl exec` disconnecting does not kill the load generator.** A severed exec
kept generating load and silently doubled the concurrency of the next run. Always
`pkill -f perf_analyzer; pkill -f genai-perf` and confirm the engine is idle before
measuring.

## 9. Cost note on methodology

The 128K point in § 7 took 43 minutes of stability detection (4 windows) on three
p5en instances. The `≥10 requests/slot/window` rule in
[BENCHMARK-METHODOLOGY.md](BENCHMARK-METHODOLOGY.md) was written for short requests;
at ~60 s per request it forces ≥10-minute windows, and one data point costs roughly
what forty short-context points cost. For long-context work prefer
`--request-count` — and then say so in the report, because a single window cannot be
trimmed.

## 10. Reproduction

| Item | Path |
|---|---|
| Manifest | `k8s-manifest/lws/lws-kimi-k3-tp8pp3-p5en.yaml` |
| Patched image build | `k8s-manifest/lws/Dockerfile.efa-vllm-kimi-k3-p50327` |
| The backported patch | `k8s-manifest/lws/vllm-50327-mamba-hybrid-int32.patch` |
| Base EFA image build | `k8s-manifest/lws/Dockerfile.efa-vllm-kimi-k3` |
| Topology reasoning | [KIMI-K3-PP-TOPOLOGY.md](KIMI-K3-PP-TOPOLOGY.md) |
| genai-perf invocations and gotchas | [benchmark-commands.md](benchmark-commands.md) |
| Steady-state / trimming procedure | [BENCHMARK-METHODOLOGY.md](BENCHMARK-METHODOLOGY.md) |
