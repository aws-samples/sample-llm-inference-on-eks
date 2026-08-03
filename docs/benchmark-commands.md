# Benchmark & Smoke-Test Command Reference

**Use genai-perf** — it is the tool of record for every benchmark in this repo.
Jump to [genai-perf 0.0.16+](#genai-perf-0016-triton-sdk-2606-genai-perfgenai-perf-triton-2606yaml--current)
and read [Measuring steady state](#measuring-steady-state) before your first run;
the defaults do not measure steady state. The engine-native harnesses at the
bottom (`sglang.bench_serving`, `vllm bench serve`) are listed for reference only —
they define concurrency and latency differently, so results are **not comparable**
with genai-perf's or with each other. Do not mix tools within one comparison.

genai-perf commands are grouped by **tool generation** — flags changed incompatibly
between versions, so match the command shape to the version in your client pod
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
- **Set the measurement window explicitly** — the defaults do not measure steady
  state (see [Measuring steady state](#measuring-steady-state) below)

```bash
genai-perf profile -m zai-org/GLM-5.2-FP8 \
  --url glm-5-2.default.svc.cluster.local:80 \
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
  --extra-inputs '{"chat_template_kwargs":{"enable_thinking":false}}'
```

Set `--warmup-request-count` to the concurrency value (one warm request per slot;
warmup records are discarded by perf_analyzer and never enter the report). The
`180000` above is derived for *this* workload — see
[Measuring steady state](#measuring-steady-state) for how to size it for yours.

Known issue: at concurrency ≥40 against the sglang-router (PD setups), the SSE
parser fails (`splintered SSE response` / `orjson.JSONDecodeError`) even though the
backend is healthy. Quantified 2026-07-31 on GLM-5.2 PD at c40: of 850,089 response
chunks, **72 (0.0085%) arrive truncated**, and 96% of those parse cleanly once joined
with the following chunk — SSE frames split across reads, with no client-side
reassembly. One bad frame aborts the whole run, so a completed 20-minute profile
produces **no `profile_export_genai_perf.json` at all**. Reproduced twice; the router
logged zero errors, so which side splits the frames is not established.

Do **not** substitute another load generator to get past this — cross-tool numbers are
not comparable and silently corrupt the comparison they enter (this is how row P2 of
[GLM-5.2-BENCHMARK.md](GLM-5.2-BENCHMARK.md) became permanently unusable). Report the
point as **"not measurable with genai-perf"**. Two things that do help: capture stderr
(never wrap the profile call in `>/dev/null 2>&1`, or the failure looks like a missing
file with no cause), and note that `profile_export.json` — the raw per-request records
— *is* still written, so latency percentiles can be recomputed by hand. ⚠️ Hand-derived
figures are not comparable to genai-perf's own: the tool re-encodes output text with
the tokenizer, whereas counting SSE chunks miscounts tokens whenever MTP/speculative
decoding emits several per chunk. Keep them in a separate, clearly labelled table.

## Measuring steady state

**The genai-perf defaults do not produce steady-state latency numbers.** Verified
against the 0.0.16.post1 source, the NVIDIA docs, and by experiment on this repo's
hardware. Three facts drive this:

1. **`--num-prompts` does not control how much load is sent.** It is an alias for
   `--num-dataset-entries` — "the number of unique payloads to sample from. These
   will be reused until benchmarking is complete." Raising it does not lengthen the
   run.
2. **Without an explicit measurement window, genai-perf sends
   `max(10, 2 × concurrency)` requests** — it computes this and passes it to
   perf_analyzer as `--request-count`
   (`genai_perf/config/generate/perf_analyzer_config.py::_calculate_request_count`).
   Two requests per concurrency slot is not enough for the queue to fill, so
   latency is sampled while the server is still ramping up.
3. **genai-perf defaults `--stability-percentage` to `999`** (perf_analyzer's own
   default is `10`), which makes the stability check pass unconditionally. Nothing
   warns you that the run never stabilised.

**Numbers read this way are invalid, but which metric is wrong varies by rig — do not
assume.** Measured A/Bs (2 requests/slot vs runtime-detected depth):

| Rig | TTFT error | Throughput error |
|---|---|---|
| L40S, Qwen3-8B dense, 2K/256 | understated 6–14× (p50) | ±5% (none) |
| B300, Kimi-K3 2.8T MoE, 1K/1K | **over**stated 1.9–2.8× (p90) | not isolated |
| H200, GLM-5.2-FP8 MoE, 8K/1K | understated only 1.1–1.2× (p50) | **overstated up to 53%** |

On the H200/MoE rig throughput was the *most* depth-sensitive metric (c40: 1,070 tok/s
shallow vs 698 steady), with ITL moving alongside it (27.4 → 54.5 ms). So the advice
that "throughput survives, only latency is unusable" holds on the L40S/dense rig and
fails here. Treat the whole shallow run as unusable rather than trying to identify which
metric escaped. Full data:
[BENCHMARK-METHODOLOGY.md](BENCHMARK-METHODOLOGY.md).

**Drain the server to idle between runs.** Back-to-back runs inherit the previous run's
queue backlog; on the H200 rig that changed an identical shallow c20 command by 34% on
TTFT p50 (516 ms undrained vs 785 ms drained) — larger than the depth effect under
study. Gate on `nvidia-smi` utilisation returning to ~0.

Use time-window mode with a real stability threshold instead of the default
request-count mode (the two are mutually exclusive):

```
--measurement-interval <ms>  --stability-percentage 10
```

perf_analyzer then repeats measurement windows until the max/min ratio across the
most recent 3 windows is within the threshold for both throughput and latency.

**Size the interval from measured request throughput — do not copy a constant.**
Aim for at least 10 requests per concurrency slot across the 3 windows:

```
interval_ms ≳ (10 × concurrency) / (3 × requests_per_sec) × 1000
```

Get `requests_per_sec` from a short throwaway run (it is in the report as *Request
Throughput*), then round up. Two worked examples, same tooling, three orders of
magnitude apart in workload:

| Workload | Request throughput | Interval |
|---|---|---|
| 8B model, 2K-in/256-out, c20 | ~1.8 /s | 60 s |
| GLM-5.2 MoE TP8, 8K-in/1K-out, c20 | ~0.45 /s | 180 s |
| GLM-5.2 MoE TP8, 8K-in/1K-out, c40 | ~0.80 /s | 180 s |

Also keep the interval well above a single request's end-to-end latency, or a window
can close with almost nothing completed inside it.

Keep `--num-prompts` larger than the total requests the run will issue. If the pool
is smaller, prompts get reused, prefix-cache hit rate climbs mid-run, and TTFT
drifts downward — a self-inflicted trend.

The stability check compares whole windows against each other, so fluctuation with
a period shorter than the window averages out inside it.

> [!CAUTION]
> **The numbers genai-perf prints at the end of a stability-detected run still include
> every ramp-up window.** `--stability-percentage` decides when perf_analyzer *stops*, not
> what gets reported: it collects all windows, and GenAI-Perf then aggregates **all**
> requests in the export. Measured on the L40S rig, trimming to the last three windows
> raises TTFT p50 by 10–15%, so the printed summary is **biased low on latency**.
>
> **Do not publish the tool's summary as a steady-state result.** Trim it first:
> [BENCHMARK-METHODOLOGY.md](BENCHMARK-METHODOLOGY.md) Step 3 gives the procedure (keep
> requests whose *last response* lands in the final `3 × interval`, from
> `profile_export.json`). And do **not** reach for `--request-count` to pin depth — it
> collapses the run to a single window, which cannot be trimmed at all (Step 2).

When you report, give the **full TTFT distribution** (p1…max) for the trimmed subset, plus
the interval, the observed `Request Count` ÷ concurrency, and **per-window percentiles** —
percentiles over the whole run cannot show whether the queue had settled, because they
discard request order. A p50 that improves as concurrency rises is a methodology red flag,
not a result.

---

# Engine-native harnesses — reference only, not recommended

Both tools below ship inside the engine images and are handy for a quick sanity
check when genai-perf is unavailable. **Do not use them for reported numbers**:
their concurrency models and latency definitions differ from genai-perf's, so
figures are not comparable across tools. Anything published from this repo should
come from genai-perf.

## sglang.bench_serving (ships inside sglang images)

Its `--num-prompts` *is* the real request count (unlike genai-perf's), and it
survives the sglang-router at high concurrency where genai-perf's SSE parser fails.

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

Before writing up results, read
[BENCHMARK-REPORTING-PRINCIPLES.md](BENCHMARK-REPORTING-PRINCIPLES.md) — the rules
this repo's reports are held to (evidence boundaries, metric definitions, scope
limits, honest reporting of discarded runs).
