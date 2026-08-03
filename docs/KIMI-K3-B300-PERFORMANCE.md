# Kimi-K3 Single-Node Performance on B300

This document states the measured performance of `moonshotai/Kimi-K3` deployed on a
single p6-b300.48xlarge with tensor parallelism TP8. All data was measured on an EKS
cluster on 2026-07-29. It states observed facts only.

A Chinese version of this document is available at
[KIMI-K3-B300-PERFORMANCE-zh.md](KIMI-K3-B300-PERFORMANCE-zh.md).

## 1. Subject and environment

The model is Kimi-K3, a 2.8-trillion-parameter Mixture-of-Experts architecture
activating 16 of 896 routed experts per token plus a shared expert, built on Kimi
Delta Attention and Attention Residuals, with native vision input and a context
window of 1,048,576 tokens. Weights are in quantization-aware-trained MXFP4 format,
occupying approximately 1.56 TB across 120 files.

The hardware is one p6-b300.48xlarge: 8× B300 (268 GB HBM3e each, 2,149 GiB total),
192 vCPUs, 4 TiB host memory, approximately 28 TiB local NVMe (8-disk RAID), and 16
EFA interfaces. In us-west-2 this instance type is offered only in the us-west-2a
availability zone; capacity for this test was obtained through a Capacity Block
(Karpenter capacity-type `reserved`).

The inference engine is vLLM, image `vllm/vllm-openai:kimi-k3`, which resolved to
version `0.1.dev19262+gb6bbf29dd.d20260727`. Startup arguments follow the official
recipe's B300 profile: TP8, `--gpu-memory-utilization 0.95`,
`--max-model-len 1048576`, `--kv-cache-dtype fp8`, MLA prefill backend
`TRTLLM_RAGGED` with `use_prefill_query_quantization` enabled,
`--enable-prefix-caching`, `--load-format fastsafetensors`, `--moe-backend auto`.

The load generator is NVIDIA genai-perf 0.0.16.post1, running in an in-cluster Triton
26.06 SDK pod, reaching the service's streaming `/v1/chat/completions` endpoint over
ClusterIP.

## 2. Memory footprint

Per-GPU allocation reported by the engine at `--gpu-memory-utilization 0.95`:

| Item | Usage |
|---|---|
| Weights and non-torch | 195.8 GiB |
| KV cache | 55.0 GiB |
| CUDA Graph | 4.7 GiB |
| Peak activation | 3.5 GiB |

The 2.8T MXFP4 weights place approximately 196 GiB on each GPU under TP8, so a single
8-GPU node holds the model in full; no multi-node deployment is required. Under this
configuration, KV-cache allocation for `--max-model-len=1048576` (the full 1M-token
context) succeeded. The engine additionally reports that setting KV cache explicitly
to 61.02 GiB (`--kv-cache-memory=65514832384`) would fully utilize the card, an
increase of approximately 11% over the current 55.0 GiB.

## 3. Startup time

Cold-start phase durations after the node became Ready:

| Phase | Duration |
|---|---|
| Container image pull | approx. 2 min |
| Download 1.5 TB of weights to local NVMe | approx. 10 min (approx. 2.3 GB/s) |
| Engine initialization (load, graph capture, KV allocation) | 499 s |
| **Total** | **approx. 22 min** |

CUDA Graph capture took 78–82 seconds on each of the 8 TP workers. The container
restart count was 0 throughout the test session. Weights reside on node-local NVMe,
so a pod restart on the same node does not re-download them.

## 4. Measurement methodology

Two workloads were tested: 1024 input / 1024 output tokens (1:1 balanced, concurrency
1–256), and 8000 input / 1024 output tokens (8:1, prefill-heavy, concurrency 8–64).
Both shapes were chosen for this test. Output length was pinned via
`max_tokens:1024` and `ignore_eos:true`; every request ended with
`finish_reason: length` and none with `stop`. Thinking mode was disabled via
`{"chat_template_kwargs":{"enable_thinking":false}}`, with the `reasoning` field
measured as null in responses. Measured input length landed at 1023.97–1024.00
(per-point averages), matching the setting.

> [!CAUTION]
> **The depth procedure described below does not yield steady-state values** — corrected
> 2026-08-03, after the method was re-examined against the perf_analyzer source. Two
> defects, both affecting every figure in this document:
>
> - **`--request-count` collapses the run to a single measurement window**
>   (`inference_profiler.cc`: `// If request-count is specified, then only measure one
>   window and exit`), so the points below cannot have ramp-up excluded — the depth is
>   pinned, but the window structure needed to trim ramp-up is gone.
> - Even the calibration run at c64 reports over **all** its windows, ramp-up included,
>   because `--stability-percentage` controls when perf_analyzer stops, not what it
>   reports.
>
> These are therefore **fixed-depth whole-run averages, not steady-state measurements**,
> and on another rig the same class of contamination biased TTFT p50 low by 5–19%.
>
> **Sharing one depth across the sweep does not make the sweep self-consistent.** The
> depth a run *needs* to settle falls as concurrency rises (measured elsewhere: 23.3
> requests/slot at c20 vs 14.7 at c40), so a single `12 × concurrency` reused at 1/8/16/32/64
> is not equally deep relative to each point's requirement — it is likely too shallow at the
> low-concurrency end and adequate at the high end. Conclusions that compare *across*
> concurrency points — the efficiency knee, the shape of the throughput curve — inherit that
> unevenness on top of the ramp-up contamination above, and should be treated as
> provisional. The per-point numbers are left as measured. Current procedure:
> [BENCHMARK-METHODOLOGY.md](BENCHMARK-METHODOLOGY.md).

Measurement depth was fixed at one common value across points: runtime stability detection was
first enabled on the deepest queue (concurrency 64) using
`--measurement-interval 120000 --stability-percentage 10`, yielding an observed
`Request Count` of 768, i.e. 12 requests per concurrency slot. All other points used
`--request-count = 12 × concurrency` to hold that same depth, giving
12/96/192/384/768. Each point was preceded by 10–20 warmup requests excluded from
measurement; the prompt pool was 2000 entries, larger than the total request count, so
that reuse would not raise the prefix-cache hit rate.

## 5. Performance data

### 5.1 Balanced workload, 1024/1024

| Concurrency | TTFT avg | TTFT p90 | TTFT p99 | ITL avg | Output per user (tok/s) | Request throughput (req/s) | Total generation throughput (tok/s) |
|---|---|---|---|---|---|---|---|
| 1   | 241 ms   | 243 ms   | 253 ms    | 9.45 ms  | 105.8 | 0.10 | 103 |
| 8   | 669 ms   | 816 ms   | 819 ms    | 14.57 ms | 68.8  | 0.52 | 516 |
| 16  | 667 ms   | 769 ms   | 1,019 ms  | 17.84 ms | 56.1  | 0.84 | 840 |
| 32  | 986 ms   | 1,165 ms | 1,804 ms  | 24.12 ms | 41.6  | 1.26 | 1,251 |
| 64  | 1,128 ms | 1,278 ms | 3,266 ms  | 32.98 ms | 30.6  | 1.85 | 2,064 |
| 128 | 1,472 ms | 1,730 ms | 6,587 ms  | 43.86 ms | 23.2  | 2.80 | 3,290 |
| 256 | 1,969 ms | 1,987 ms | 13,166 ms | 69.26 ms | 14.9  | 3.64 | 4,634 |

TTFT is time to first token; ITL is inter-token latency. Latency metrics are measured
client-side. Total generation throughput is the steady-state median reported by the
engine's own `log_stats` (18–64 samples per point, with `Reqs Running` matching the
target concurrency).

### 5.2 Prefill-heavy workload, 8000/1024

A second workload was run on the same deployment at the same measurement depth
(12 requests/slot), with 8000 input and 1024 output tokens (8:1). Measured input length
landed at 7,999.91–7,999.96. The two workloads side by side:

| Concurrency | **1K/1K** tok/s / TTFT / ITL | **8K/1K** tok/s / TTFT / ITL |
|---|---|---|
| 8  | 516 / 669 ms / 14.6 ms     | 478 / 1,390 ms / 18.1 ms |
| 16 | 840 / 667 ms / 17.8 ms     | 870 / 1,228 ms / 22.3 ms |
| 32 | 1,251 / 986 ms / 24.1 ms   | 1,267 / 1,932 ms / 34.2 ms |
| 64 | 2,064 / 1,128 ms / 33.0 ms | **82** / 5,106 ms / 114.5 ms |

At concurrency 64 the 8K workload's TTFT p99 was 18,972 ms, worst-case ITL
1,821.89 ms, and request throughput 0.68 req/s — below the 0.98 req/s measured at
concurrency 32 on the same workload.

## 6. Patterns in the data

**The efficiency knee for the 1K/1K workload is at concurrency 256.** Throughput gain
versus ITL gain per doubling: 16→32 is +49% / +35%, 32→64 is +65% / +37%, 64→128 is
+59% / +33% — each a throughput gain exceeding the latency cost. 128→256 inverts for
the first time, +41% throughput for +58% ITL. Absolute throughput is still rising
(3,290 → 4,634 tok/s), so this is not a hard ceiling. Concurrency 512 was not tested.

**Parallel efficiency declines as concurrency rises.** Relative to concurrency 1, the
ratio of throughput gain to concurrency gain is: 5.01× at concurrency 8 (62.6%
efficiency), 8.16× at 16 (51.0%), 12.15× at 32 (38.0%), 20.04× at 64 (31.3%), 31.9× at
128 (24.9%), and 45.0× at 256 (17.6%). Expressed per GPU, throughput rises from
12.9 tok/s at concurrency 1 to 579.2 tok/s at concurrency 256.

**Average latency grows more slowly than load; tail latency does not.** As concurrency
increased from 1 to 256 (256×), TTFT average grew 8.2× (241 → 1,969 ms) and ITL grew
7.3× (9.45 → 69.26 ms). However TTFT p99 rose from 1,804 ms at concurrency 32 to
13,166 ms at 256, and worst-case ITL from 48.94 ms to 455.76 ms; the p99-to-average
ratio widened from 1.8× to 6.7×.

**With 8K input, output throughput through concurrency 32 is essentially the same as
with 1K input.** The 8K workload reached 93%, 104%, and 101% of the 1K figures at
concurrency 8, 16, and 32. The cost appears in latency: TTFT roughly doubles and ITL
is 1.2–1.4×.

**With 8K input, throughput regresses at concurrency 64.** Total throughput fell to
82 tok/s (4% of the 1K workload at the same concurrency), request throughput dropped
from 0.98 req/s at concurrency 32 to 0.68 req/s, TTFT averaged 5,106 ms and ITL
114.51 ms. Engine throughput samples at this point are bimodal: median 82 tok/s while
p75 is 1,836 tok/s and max 2,112 tok/s (108 samples). Concurrency 48 was not tested,
so the degradation boundary is not located.

**Throughout that degradation, KV-cache usage peaked at 20.8% and the preemption count
was 0.** `Reqs Waiting` stayed at 0 during steady state on every 1K point; on the 8K
workload 9 samples showed queueing.

**Concurrency 8 and concurrency 16 have essentially identical latency.** Their TTFT
averages are 669 ms and 667 ms and their p90 values 816 ms and 769 ms, while
concurrency 16 delivers 63% higher total throughput (840 vs 516 tok/s).

**Single-stream decode rate is 105.8 tok/s.** At concurrency 1, ITL is 9.45 ms with a
very narrow distribution (9.45–9.46 ms from p1 through max).

**Prefix caching did not take effect for this workload.** Despite
`--enable-prefix-caching` being enabled, the measured prefix-cache hit rate was 0.0%
throughout, meaning the results above were not affected by cache reuse.

## 7. Functional observations

**Tool calling does not take effect in this configuration.** The B300 recipe profile
sets `VLLM_USE_RUST_FRONTEND=1`, and that frontend explicitly logs at startup that it
ignores the `enable_auto_tool_choice` and `structured_outputs_config` arguments;
`--tool-call-parser kimi_k3` and `--enable-auto-tool-choice` are therefore inert.
Setting `VLLM_USE_RUST_FRONTEND` to 0 is required to use tool calling. Also logged:
Model Runner V2 does not support the `thinking_token_budget` request parameter.

**Thinking mode is on by default and can be disabled.** By default, model output goes
to the `reasoning` field while `content` is null; it is disabled via the nested form
`chat_template_kwargs.enable_thinking=false`, whereas a flat `enable_thinking` has no
effect. Once disabled, `reasoning` is null and `content` returns normally.

**Vision is loaded; the video modality is not supported.** Frontend logs show the chat
backend loaded with the `kimi_k3` renderer, and the video modality disabled because
its placeholder tokens did not resolve.

**Functional verification.** An arithmetic question (84 × 3 ÷ 2) returned the correct
result 126 with `finish_reason: stop`, and the `kimi_k3` parser correctly separated
reasoning content from the final answer.

## 8. Scope and what was not covered

The data in this document applies to: single-node TP8, thinking mode disabled,
concurrency 1–256 on the 1024/1024 workload, and concurrency 8–64 on the 8000/1024
workload.

Not covered: concurrency 512 and above on the 1K workload; concurrency 48 (the
degradation boundary) and 128 and above on the 8K workload; longer inputs (the longest
tested is 8000 tokens against a 1M-token window); decode-heavy workloads such as 1K in
/ 4K out; end-to-end behavior with thinking mode enabled; speculative decoding (the
`Inferact/Kimi-K3-DSpark` draft-model configuration constrains `--max-num-seqs` to 32,
not deployed here); multi-node TEP16 and PD-disaggregated deployment shapes; and a
comparison run with KV cache raised to 61.02 GiB.

The deployment manifest used is
[`k8s-manifest/vllm/kimi-k3-p6-b300-vllm.yaml`](../k8s-manifest/vllm/kimi-k3-p6-b300-vllm.yaml).
