# Model Index

Directory layout is by **stack** (`sglang/`, `vllm/`, `lws/`); this index is the
**by-model** view. Paths are relative to `k8s-manifest/`.

Status legend: ✅ actively maintained (recently verified) · 📦 archived example
(older engine/image pins, apply-at-your-own-risk; kept as reference).

## GLM-5.2 (`zai-org/GLM-5.2-FP8`, 753B MoE)

Benchmarked head-to-head in [GLM-5.2-BENCHMARK.md](GLM-5.2-BENCHMARK.md).
KV-cache offload & externalization (HiCache / LMCache / Redis L2) measured in
[KV-CACHE-ARCHITECTURE.md](KV-CACHE-ARCHITECTURE.md).

| Shape | Manifest | Instances | Status |
|---|---|---|---|
| 1-node TP8, SGLang | `sglang/glm-5.2-fp8-p5en.yaml` | 1× p5en.48xlarge | ✅ |
| 1-node TP8, vLLM | `vllm/glm-5.2-fp8-p5en-vllm.yaml` | 1× p5en.48xlarge | ✅ |
| 2-node TP16, SGLang LWS+EFA | `lws/lws-glm-5.2-tp16-p5en.yaml` | 2× p5en.48xlarge | ✅ |
| 2-node PD-disagg (NIXL/EFA), SGLang | `lws/lws-glm-5.2-pd-p5en.yaml` | 2× p5en.48xlarge | ✅ |

## DeepSeek-V3.2 (`deepseek-ai/DeepSeek-V3.2`, FP8)

| Shape | Manifest | Instances | Status |
|---|---|---|---|
| 2-node TP16, SGLang LWS+EFA | `lws/lws-deepseek-v3.2-tp16-p5.yaml` | 2× p5.48xlarge | ✅ |
| 2-node PD-disagg, SGLang | `lws/lws-deepseek-v3.2-pd-p5en.yaml` | 2× p5en.48xlarge | ✅ |
| 2-node PD-disagg (older, p5) | `lws/lws-deepseek-v3.2-pd-p5.yaml` | 2× p5.48xlarge | 📦 |

## DeepSeek-R1 (`deepseek-ai/DeepSeek-R1`, 671B FP8)

| Shape | Manifest | Instances | Status |
|---|---|---|---|
| 1-node TP8, SGLang | `sglang/ds-r1-671b-p5en.yaml` | 1× p5en.48xlarge | 📦 |
| 2-node TP16, SGLang LWS | `lws/lws-deepseek-r1-tp16-p5en.yaml` | 2× p5en.48xlarge | 📦 |

## DeepSeek-R1 Distills

| Model | Manifest | Instances | Status |
|---|---|---|---|
| Distill-Llama-8B, SGLang | `sglang/ds-r1-distill-llama-8b-sglang.yaml` | GPU nodepool | 📦 |
| Distill-Llama-8B, vLLM | `vllm/ds-r1-distill-llama-8b-vllm.yaml` | GPU nodepool | 📦 |
| Distill-Llama-70B, SGLang | `sglang/ds-r1-distill-llama-70b-sglang.yaml` | g6e.12xlarge | 📦 |
| Distill-Llama-70B, SGLang DP2 | `sglang/ds-r1-distill-llama-70b-sglang-dp2-g6e.yaml` | g6e.48xlarge | 📦 |
| Distill-Llama-70B, SGLang TP8 | `sglang/ds-r1-distill-llama-70b-sglang-p4d-tp8.yaml` | p4d.24xlarge | 📦 |
| Distill-Llama-70B, vLLM | `vllm/ds-r1-distill-llama-70b-vllm.yaml` | g6e.12xlarge | 📦 |
| Distill-Qwen-14B, vLLM | `vllm/ds-r1-distill-qwen-14b-vllm.yaml` | GPU nodepool | 📦 |
| Distill-Qwen-32B, SGLang | `sglang/ds-r1-distill-qwen-32b-sglang.yaml` | g6e.12xlarge | 📦 |
| Distill-Qwen-32B, vLLM | `vllm/ds-r1-distill-qwen-32b-vllm.yaml` | g6e.12xlarge | 📦 |

## Kimi-K3 (`moonshotai/Kimi-K3`, 2.8T MXFP4 MoE)

**Start with the single-node B300 shape** — it is the one that has been benchmarked
([English](KIMI-K3-B300-PERFORMANCE.md) / [中文](KIMI-K3-B300-PERFORMANCE-zh.md),
2026-07-29) and it runs the recipe without the H200 profile's restrictions. The
2-node TP16 shape exists for sites without B300 capacity.

**If you need the full 1M context window**, that is the 3-node TP8×PP3 shape —
measured 2026-08-09 in [KIMI-K3-PP3-PERFORMANCE.md](KIMI-K3-PP3-PERFORMANCE.md),
with the topology and KV-block reasoning in
[KIMI-K3-PP-TOPOLOGY.md](KIMI-K3-PP-TOPOLOGY.md). Read the warning at the top of
the performance doc first: full 1M was never exercised (retrieval verified to
91.7K), and long context requires the patched image described below.

Args follow the official recipe's per-hardware profile —
[b300](https://recipes.vllm.ai/moonshotai/Kimi-K3?hardware=b300) for the single-node
manifest, [h200](https://recipes.vllm.ai/moonshotai/Kimi-K3?hardware=h200) for the
TP16 one. The two differ in more than TP size: the 2.8T MXFP4 checkpoint needs
~1680 GB, so on 2× 1128 GB the recipe drops context from 1M to 32K, `--max-num-seqs`
to 5, and switches the MoE kernel to `marlin` — and the manifest goes one step
further to 29,696 tokens, because 32K does not fit on this rig (the engine refuses to
start; see the arg comment). Pre-release: requires the `vllm/vllm-openai:kimi-k3`
image (vLLM ≥ 0.27.0 nightly), the manifests here not pinned to an immutable tag.
The TP16 shape additionally needs an EFA-enabled rebuild — the upstream image ships
no EFA userspace, so cross-node NCCL would fall back to TCP.

| Shape | Manifest | Instances | Status |
|---|---|---|---|
| 1-node TP8, vLLM — **default** | `vllm/kimi-k3-p6-b300-vllm.yaml` | 1× p6-b300.48xlarge | ✅ |
| 2-node TP16, vLLM LWS+EFA — fallback, 29.7K context / `max-num-seqs 5` | `lws/lws-kimi-k3-tp16-p5en.yaml` | 2× p5en.48xlarge | ✅ |
| 2-node TP8×PP2, vLLM LWS+EFA — ~2× the context of TP16 (~41K), **untested** | `lws/lws-kimi-k3-tp8pp2-p5en.yaml` | 2× p5en.48xlarge | ⚠️ |
| 3-node TP8×PP3, vLLM LWS+EFA — serves at 1M `max-model-len`; **retrieval measured to 91.7K, full 1M unverified**; needs the stopgap image below | `lws/lws-kimi-k3-tp8pp3-p5en.yaml` | 3× p5en.48xlarge | ✅ |

> [!IMPORTANT]
> **The PP3 manifest pins a locally patched image, and that is a stopgap — as of
> 2026-08-09.** Long context is unusable on a stock image: any input larger than
> `--max-num-batched-tokens` is chunk-prefilled, and before the fix every chunk
> boundary after the first threw `IndexError: index_fill_(): Expected dtype int64
> for index` from `mamba_hybrid.py:314`. Requests still returned HTTP 200 and the
> engine survived, so the failure was **silent** — the KDA state hand-off was just
> skipped. It fired 1,280 times during one 128K run.
>
> The fix is upstream **[PR #50327](https://github.com/vllm-project/vllm/pull/50327)**
> (merged 2026-08-03, commit `e578de31`), backported verbatim in
> `lws/Dockerfile.efa-vllm-kimi-k3-p50327` +
> `lws/vllm-50327-mamba-hybrid-int32.patch` → tag `kimi-k3-efa1.49-p50327`.
> **No released vLLM carried it**: v0.26.0 shipped 2026-07-27, seven days earlier
> (verified by reading the file at that tag; issue
> [#50947](https://github.com/vllm-project/vllm/issues/50947) was still open).
>
> **Retire it as soon as you can.** Check whether the vLLM your base image is built
> from contains `e578de31` — or simply grep the installed
> `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` for
> `_fill_num_accepted_kernel`. If present, delete both files and retag the manifest
> to a plain EFA image. The Dockerfile omits `patch --forward` on purpose, so a base
> image that already has the fix makes the **build fail** instead of silently
> reversing it — treat that failure as the signal to clean up, not as a bug.

**How context capacity works here, and why only PP moves it.** Capacity is counted
in *blocks*, not GiB: a request draws whole blocks from every kv-cache group and the
totals are summed, and K3 is hybrid — 24 full-attention (MLA) layers whose blocks
scale with context, plus 69 KDA layers costing one state block per request. Block
size is 384 tokens (raised so the attention page matches the KDA page), so a request
needs `cdiv(max_model_len, 384) + 1` blocks: **79** at 29,696 tokens, **2,732** at 1M.

Critically, **MLA KV is not sharded by TP** — the spec is built with `num_kv_heads=1`
("Only has one vector instead of K + V", `deepseek_v2.py:634`). So widening TP does
nothing for the KV pool. Only **PP** helps, by giving each GPU fewer layers to cache,
and only **more nodes** reduce the per-GPU weight footprint:

| Shape | GPUs | weights/GPU | layers/GPU | est. blocks | max context @ ~1× |
|---|---|---|---|---|---|
| TP16 | 16 | 129.75 GiB *(measured)* | 93 | ~80 *(measured)* | 30,336 *(measured)* |
| TP8×PP2 | 16 | 129.75 GiB (unchanged) | 47 | ~158 | ~60K |
| TP8×PP3 | 24 | ~86.5 GiB | 31 | ~12,900 | ~1M ✅ |

PP2 roughly doubles usable context but is **~17× short of 1M** — on 16 GPUs the
weights still take 129.75 GiB/GPU, leaving the same 0.82 GiB for KV. **1M genuinely
requires the third node.** Estimates above scale the two measured TP16 figures by
layer count; they assume the measured 8.89 GiB/GPU of non-weight non-KV overhead
(activations, CUDA graphs, NCCL buffers) stays constant under PP, which is untested.
Do **not** compute available KV as `gpu_memory_utilization × 143.77 − weights` — that
yields 9.71 GiB against a measured 0.82 GiB, because it ignores that overhead.

On 3 nodes the shape must be TP8×PP3, **not** TP24: `vocab_size` is 163,840 and
vLLM's vocab padding is a fixed 64 (it does not scale with `tp_size`), so
`divide(163840, 24)` trips a hard assert and the engine aborts at init. Every other
TP-sharded dimension does divide by 24 — only vocab fails. TP8×PP3 also splits the
93 layers evenly (31 each), which PP2 cannot (47/46). Only the TP16 shape has been
run; both PP shapes are untested.

## Qwen

| Model | Manifest | Instances | Status |
|---|---|---|---|
| Qwen3-32B, SGLang | `sglang/qwen3-32b-sglang-g6e.yaml` | g6e.12xlarge | 📦 |
| Qwen3-32B, SGLang | `sglang/qwen3-32b-p4de.yaml` | p4de.24xlarge | 📦 |
| Qwen3-32B, SGLang | `sglang/qwen3-32b-p5-4xl.yaml` | p5.4xlarge | 📦 |
| Qwen2.5-VL-7B, vLLM | `vllm/qwen2.5-vl-7b.yaml` | g6e.2xlarge | 📦 |
| Qwen3-8B, SGLang | `sglang/qwen3-8b-sglang-g6e-2xl.yaml` | g6e.2xlarge–16xlarge | ✅ |

Qwen3-8B on a single L40S is the cheap rig for validating **benchmark methodology**
(see [benchmark-commands.md](benchmark-commands.md#measuring-steady-state)) — it
saturates fast enough to exercise queueing without needing p5-class capacity. Not a
performance reference.

## Others

| Model | Manifest | Instances | Status |
|---|---|---|---|
| gpt-oss-120B, vLLM | `vllm/gpt-oss-120b-vllm-p5-4xl.yaml` | p5.4xlarge | 📦 |
| Llama-4-Scout-17B-16E, vLLM | `vllm/llama4-scout.yaml` | p5en.48xlarge | 📦 |

## Supporting resources

| Purpose | File |
|---|---|
| Karpenter GPU nodepool | `infra/nodepool.yaml` |
| Priority class | `infra/priority-class.yaml` |
| genai-perf client pod (Triton 26.06 SDK) | `genai-perf/genai-perf-triton-2606.yaml` |
| Chat UI | `addons/open-webui.yaml` |
| EFA image builds | `lws/Dockerfile.efa-*` (suffix = engine + version, e.g. `-sglang-0513`, `-vllm-kimi-k3`; `-nixl-` = adds NIXL for PD KV transfer) |
| Benchmark command reference | [benchmark-commands.md](benchmark-commands.md) |
