# Kimi-K3 Multi-Node Performance on 3× p5en (TP8×PP3)

This document states the measured performance of `moonshotai/Kimi-K3` deployed
across three p5en.48xlarge instances as tensor-parallel 8 × pipeline-parallel 3
(24× H200). All data was measured on an EKS cluster on 2026-08-09. It states
observed facts and marks every inference as such.

For *why* this topology rather than TP24 or PP2, see
[KIMI-K3-PP-TOPOLOGY.md](KIMI-K3-PP-TOPOLOGY.md). For the single-node B300
deployment see [KIMI-K3-B300-PERFORMANCE.md](KIMI-K3-B300-PERFORMANCE.md).

> [!WARNING]
> **Read these three limits before quoting any number below.**
>
> 1. **Full 1M context was never exercised.** The engine starts and serves at
>    `--max-model-len 1048576`, and retrieval was verified at **91,700 tokens**.
>    The span from 91.7K to 1M is unverified. Concurrency at a full 1M window
>    (6, § 6) is the engine's own arithmetic, not a measurement.
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

Engine-reported KV pool: **6,361,156 tokens**, block size 384. That is
**16,565 blocks**, and a 1M request costs `cdiv(1048576, 384) + 1 = 2,732` blocks
(the `+1` is the fixed KDA state block; see
[KIMI-K3-PP-TOPOLOGY.md](KIMI-K3-PP-TOPOLOGY.md) § 4). Hence the engine's own line:

```
Maximum concurrency for 1,048,576 tokens per request: 6.07x
```

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

| `--max-num-seqs` | KV pool (tokens) | vs 32 |
|---:|---:|---:|
| 32 | 6,361,156 | — |
| 128 | 6,355,029 | −6,127 (−0.10%) |
| 512 | 6,319,795 | −41,361 (−0.65%) |

A least-squares fit gives `KV ≈ 6,363,913 − 86.2 × max_num_seqs` tokens, and the
middle point sits on that line to within 0.034% — too regular to be measurement
noise. **The mechanism is not established.** 86 tokens/slot is ~900 KB of HBM per
slot, three orders of magnitude more than the O(`max_num_seqs`) bookkeeping tensors
that are easy to point at (e.g. `num_accepted_tokens_gpu`, an int32 array of
`max_num_reqs`, `mamba_hybrid.py:75-77`). A per-request block table would be the
right order of magnitude at 1M context, but that was **not verified**.

Practical consequence: the cost is irrelevant at short context and **relevant at
1M**, where the whole pool only affords 6 concurrent requests. Do not leave
`--max-num-seqs` at a large value on a 1M deployment; 41,361 tokens is a
noticeable fraction of one request's budget. The committed manifest ships 32 as a
serving default — 512 is where the ≥2,169 tok/s figure came from, and is not the
default because of the per-user latency above.

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

### 7.1 Prefill and decode alternate in long phases — cause not established

Engine logs at 128K, sampled every 10 s:

```
prefill=12801.9  gen=55.1      prefill=12801.7  gen=3.3
prefill=0.0      gen=200.6     prefill=12801.6  gen=3.2
prefill=0.0      gen=145.3     prefill=25604.2  gen=4.5
prefill=12801.6  gen=3.3       prefill=0.0      gen=200.4
```

Two strictly alternating regimes on a **~50–60 s period**: during prefill, decode
collapses to 3.2–3.4 tok/s; during decode, prefill is exactly 0.0. The prefill rate
is pinned near 12,801 tok/s.

**A plausible explanation that measurement refuted.** The scheduler shares one
per-step token budget between running and waiting requests
(`vllm/v1/core/sched/scheduler.py:110-114`, `:454`, `:480`, `:518`), so on paper a
single 128K prefill chunk consuming all 4096 would starve every other request's
1-token decode. Raising `--max-num-batched-tokens` 4096 → 16384 as a controlled A/B
**disproved that**:

| | 4096 | 16384 |
|---|---|---|
| prefill throughput | ~12,801 tok/s | **~12,801 tok/s — unchanged** |
| alternation period | ~50–60 s | **~50–60 s — unchanged** |
| decode during prefill | 3.2–3.4 tok/s | **0.7–0.9 tok/s — worse** |
| peak activation | 1.29–2.19 GiB | 4.49 GiB |
| KV pool | 6,361,156 tokens | 5,485,683 (−13.8%) |

A 4× budget changed the prefill rate by nothing, and chunk count per request should
have dropped 31 → 8 yet the period did not move. **Something other than that flag
caps per-step tokens; what, is not established** — identifying it needs the PP
microbatch scheduling path or a profiler, neither of which was run. The setting was
reverted: 16384 is a pure loss here (less KV, slower decode, target problem
unchanged). The identical `12801.x` under both budgets is the strongest available
clue.

`--long-prefill-token-threshold` (`scheduler.py:516-517`, default off) is the other
upstream knob aimed at this. **Untested** — and since the budget was not the binding
constraint, there is no particular reason to expect it to help.

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
