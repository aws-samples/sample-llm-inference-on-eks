# Benchmark & Smoke-Test Command Reference

Copy-paste reference for load-testing models deployed in this repo (formerly `prompts.sh`).
Commands are grouped by **tool generation** — flags changed incompatibly between versions,
so match the command shape to the genai-perf version in your client pod
(`genai-perf --version`).

## Quick smoke test (any OpenAI-compatible endpoint)

```bash
curl -X POST "http://<service>/v1/chat/completions" -H "Content-Type: application/json" --data '{
  "model": "<served-model-name>",
  "messages": [{"role": "user", "content": "Hello, who are you?"}],
  "max_tokens": 32
}'
```

For thinking/reasoning models (GLM-5.2, DeepSeek-R1): add
`"chat_template_kwargs": {"enable_thinking": false}` (nested — a flat
`enable_thinking` key is silently ignored) to get plain `content` instead of
`reasoning_content`.

---

## genai-perf ≤ 0.0.x (Triton SDK 24.09 / 24.12 era) — LEGACY

Required `--service-kind openai`; tokenizer had to be manually pointed at a
compatible HF repo. These SDK images ship transformers 4.x and **cannot tokenize
modern models** (GLM-5.2, DeepSeek-V3+) — kept only for the record.

```bash
genai-perf profile -m deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --url <service> \
  --service-kind openai \
  --endpoint-type chat \
  --num-prompts 100 \
  --synthetic-input-tokens-mean 200 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 100 --output-tokens-stddev 0 \
  --concurrency 20 \
  --streaming \
  --tokenizer hf-internal-testing/llama-tokenizer
```

## genai-perf 0.0.16+ (Triton SDK 26.06, `genai-perf/genai-perf-triton-2606.yaml`) — CURRENT

Breaking changes vs legacy:

- `--service-kind` **removed** — endpoint selected by `--endpoint-type chat` alone
- `--extra-inputs` nested JSON must be **one whole `{...}` value**;
  `key:{json}` form fails (parser splits on first `:`)
- Pin OSL exactly with `max_tokens` + `ignore_eos` — `--output-tokens-mean` alone
  doesn't guarantee it on OpenAI chat endpoints

```bash
genai-perf profile -m zai-org/GLM-5.2-FP8 \
  --url glm-5-2.default.svc.cluster.local:80 \
  --endpoint-type chat --streaming \
  --num-prompts 100 \
  --synthetic-input-tokens-mean 8000 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 1024 --output-tokens-stddev 0 \
  --concurrency 20 \
  --tokenizer zai-org/GLM-5.2-FP8 \
  --extra-inputs max_tokens:1024 \
  --extra-inputs ignore_eos:true \
  --extra-inputs '{"chat_template_kwargs":{"enable_thinking":false}}'
```

Known issue: at concurrency ≥40 against the sglang-router (PD setups), the SSE
parser intermittently fails (`splintered SSE response` / `orjson.JSONDecodeError`)
even though the backend is healthy. Use `sglang.bench_serving` below instead.

## sglang.bench_serving (ships inside sglang images) — router-safe alternative

```bash
python3 -m sglang.bench_serving \
  --backend sglang-oai-chat \
  --base-url http://<service>:80 \
  --model zai-org/GLM-5.2-FP8 \
  --dataset-name random \
  --random-input-len 8000 --random-output-len 1024 --random-range-ratio 1.0 \
  --num-prompts 150 \
  --max-concurrency 40 \
  --extra-request-body '{"chat_template_kwargs":{"enable_thinking":false}}'
```

## vllm bench serve (ships inside vLLM images)

```bash
vllm bench serve \
  --model zai-org/GLM-5.2-FP8 \
  --dataset-name random \
  --random-input 8000 --random-output 1024 \
  --request-rate 10 \
  --num-prompts 32 \
  --ignore-eos
```

---

Full workflow (client pod setup, thinking-mode pitfalls, metric interpretation):
see [GLM-5.2-BENCHMARK.md](GLM-5.2-BENCHMARK.md) → Reproduce.
