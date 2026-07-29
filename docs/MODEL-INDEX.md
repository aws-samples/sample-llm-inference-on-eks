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

Args follow the official recipe's B300 profile
([recipes.vllm.ai](https://recipes.vllm.ai/moonshotai/Kimi-K3?hardware=b300)).
Pre-release: requires the `vllm/vllm-openai:kimi-k3` image (vLLM ≥ 0.27.0
nightly), the one manifest here not pinned to an immutable tag.

Deployed and benchmarked 2026-07-29 — measured performance in
[English](KIMI-K3-B300-PERFORMANCE.md) / [中文](KIMI-K3-B300-PERFORMANCE-zh.md).

| Shape | Manifest | Instances | Status |
|---|---|---|---|
| 1-node TP8, vLLM | `vllm/kimi-k3-p6-b300-vllm.yaml` | 1× p6-b300.48xlarge | ✅ |

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
| EFA image builds (SGLang) | `lws/Dockerfile.efa-*` (suffix = sglang version; `-nixl-` = adds NIXL for PD KV transfer) |
| Benchmark command reference | [benchmark-commands.md](benchmark-commands.md) |
