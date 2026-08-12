# Benchmark Methodology — Measuring Steady State and Comparing Fairly

**This document is the standard for benchmarks in this repo.** Follow §"Procedure".

> [!CAUTION]
> **Every number published from this methodology so far was computed over the whole run,
> ramp-up included.** Step 3 explains why: `--stability-percentage` controls when
> perf_analyzer stops, not what it reports, and GenAI-Perf aggregates every request in the
> export. So the figures in §"Measured basis", §"Second rig" and §"Third rig", and every
> figure in [GLM-5.2-BENCHMARK.md](GLM-5.2-BENCHMARK.md), are **deeper than the
> 2-requests-per-slot defaults but still contaminated by the transient**.
>
> **The size and direction of that contamination is unknown for all but one rig.** On the
> L40S rig — the only one whose per-request exports survive — dropping the ramp-up window
> raised TTFT p50 by 5–19%, i.e. the whole-run value was biased *low*. **Do not carry that
> direction over.** This document's own §"Second rig" and §"Third rig" show the depth bias
> reversing sign between rigs and even between concurrency levels on the same rig, so for
> GLM-5.2 and Kimi-K3 — where no window data was retained — the honest statement is that the
> whole-run figures are contaminated **by an unknown amount, in an unknown direction**.
> Re-deriving them from stationary windows is tracked in §"Validation checklist".

> [!IMPORTANT]
> **The numbers are not transferable between rigs.** Every *quantitative*
> claim in the original single-rig write-up has since been falsified on at least one
> other rig — which metric a shallow run ruins, and in which direction, turned out to be
> rig-specific:
>
> | Rig | TTFT error at 2 req/slot | Throughput error |
> |---|---|---|
> | 1× L40S, Qwen3-8B dense, 2K/256 | understated **6–14×** | ±5% (none) |
> | 8× B300, Kimi-K3 2.8T MoE, 1K/1K | **over**stated 1.9–2.8× | not isolated |
> | 8× H200, GLM-5.2-FP8 MoE, 8K/1K | understated only 1.1–1.2× | **overstated up to 53%** |
>
> So: **run the procedure, do not reuse the numbers.** Treat every figure in
> §"Measured basis", §"Second rig" and §"Third rig" as an observation from that rig, and
> never assume a shallow run left some particular metric intact.
>
> What generalised across all three rigs: steady state is a validity precondition, and it can
> only be evidenced with **per-window or time-ordered data** — percentiles cannot show it in
> either direction (see Step 3). Note also that the depth a run *realises* under
> stability detection falls as concurrency rises (23.3 req/slot at c20 vs 14.7 at c40 on rig
> 3) — an observation about outcomes, not a target to match.
>
> Open items are tracked in §"Validation checklist". Two of the five are gaps in how the
> procedure has been *applied* so far — window trimming (Step 3) and repeat runs with an
> uncertainty estimate (Step 2) have never been done; the rest concern whether *numbers*
> transfer between rigs.
>
> Created 2026-07-28 as a draft; procedure documented 2026-08-03, corrected 2026-08-04. No
> published comparison in this repo has yet followed Steps 2–3 as now written.

## The question this answers

When benchmarking LLM latency, two goals appear to conflict:

- **Steady state** — measure long enough that the request queue has settled, or the
  latency number is simply wrong.
- **Controlled comparison** — hold every parameter identical across models, or the
  comparison is confounded.

They appear to conflict because reaching steady state takes *longer on a slower
model*, so equal measurement duration and equal measurement depth cannot both hold.

## Proposed resolution

**These are not competing conditions — the apparent conflict comes from mistaking an
outcome for a controlled variable.**

A latency figure taken before the queue settles is not "latency under a different
condition" — it is a mis-measurement. Comparing two mis-measured numbers is not a
comparison. So steady state is not something to trade off against control; it is
what makes the numbers mean anything.

The dissolution: under a closed-loop generator the **experimental conditions are workload,
concurrency and window duration**. How many requests each arm completes in that window is a
*result* — largely a restatement of its throughput. There is nothing to equalise, so there
is no tension with measuring long enough to settle. What has to be verified instead is that
the windows you report are **stationary**, and unequal sample sizes are handled the ordinary
way (repeat runs, confidence intervals).

| Approach | What it fixes | Excludes ramp-up? | Use for |
|---|---|---|---|
| `--measurement-interval` + `--stability-percentage 10` | duration per window | ✅ yes — multiple windows, drop the transient ones | **everything** |
| `--request-count = N × concurrency` | total requests | ❌ **no — single window only** | nothing; see Step 2 |

`--request-count` looks attractive because it equalises request counts across arms, but it
collapses the run to one window (`inference_profiler.cc`), so ramp-up cannot be trimmed —
and equalising counts was never the right goal anyway (Step 2).

**What to hold fixed: workload, concurrency, and window duration.** Under a closed-loop
load generator those three *are* the experimental conditions. Concurrency caps the number
of in-flight requests, so the queue depth each arm sees is pinned by `--concurrency`, not by
how many requests it happens to finish.

**Realised `N` (`completed requests ÷ concurrency`) is an outcome, not a condition** —
under fixed concurrency and fixed duration it is essentially a restatement of throughput
plus sample size. Our own data shows this directly: at c20, TP16/TP8 read 0.948 on
throughput and **0.948 on N** — the same number twice. (At c40 the two diverge, 0.944 vs
0.789, only because the TP16 run stopped after ~4 windows against TP8's ~4.8 — again a
consequence of the run, not a controlled input.)

> [!CAUTION]
> **Do not require matched `N` across arms, and do not treat unmatched `N` as
> automatically fatal.** An earlier version of this document did both. It is circular: a
> faster arm completes more requests in the same window *by definition*, so demanding equal
> `N` demands equal throughput — which is the thing under test. What differing `N` actually
> costs you is **statistical precision** (fewer samples in the smaller arm), not validity.
>
> The condition that does matter is **stationarity of the windows you report**: show the
> retained windows agree with each other (Step 3), and the comparison stands regardless of
> how many requests each arm needed to get there. Handle the unequal sample sizes the
> ordinary way — repeat runs, and publish a confidence interval or at least the run-to-run
> spread. Every figure in this repo is n=1 with no variance estimate, which is a real
> limitation and the honest reason its cross-arm gaps stay provisional.

> [!WARNING]
> An earlier version of this section claimed the bias runs *in favour of slow models*,
> because short sampling **understates** TTFT. **Neither the direction nor the affected
> metric generalises** — on the B300/Kimi-K3 rig short sampling *overstated* TTFT p90
> by up to 2.8×, and on the H200/GLM-5.2 rig TTFT barely moved while *throughput* was
> overstated by 53%. See §"Second rig" and §"Third rig". Treat under-depth measurements
> as invalid, not as conservative in a known direction, and do not assume any
> particular metric escaped.

## Procedure

**Step 1 — Calibrate (once per model family / workload shape).**
Run the **slowest** model in the comparison set with runtime stability detection:

```
--measurement-interval <ms>  --stability-percentage 10
```

Size the interval so that **each individual window** holds ≥10 requests per concurrency
slot — a window is the unit the stability check compares, so a window with too few requests
makes the check meaningless:

```
window_s   ≳ (10 × concurrency) / requests_per_sec
interval_ms ≳ window_s / 1.2 × 1000        # a window is 1.2× the interval
```

Worked: GLM-5.2 TP8 at c20, 0.54 req/s → window ≥ 370 s → interval ≥ 310 s.

⚠️ **The stability-detected runs published in this repo were sized with an earlier version of
this formula** that divided by 3 (targeting 10 req/slot summed across three windows, ~3.3 per
window) and ignored the 1.2× factor. Those runs sit at **4.3–6.5 requests/slot per window** —
below what a per-window stability comparison needs, and one of the reasons their figures are
reported as provisional. Two other groups sit outside that range: the original GLM-5.2
campaigns ran at the default **2 req/slot**, and the Kimi-K3 sweep pinned **12 req/slot in a
single window** via `--request-count`, which cannot be window-trimmed at all. The worked examples in §"Third rig" and in
[benchmark-commands.md](benchmark-commands.md) are the intervals actually used, kept for
traceability, not the intervals this rule now asks for.

Round up, and keep the interval well above one request's end-to-end latency or a window can
close with almost nothing finished inside it.

**Step 2 — Compare all arms at one pinned duration, not `--request-count`.**

```
--concurrency <C>  --measurement-interval <ms>  --stability-percentage 10
```

…with the **same `<ms>` for every arm**, sized in Step 1 from the *slowest* arm so that even
it clears ≥10 requests/slot in a single window.

> [!CAUTION]
> **Do not use `--request-count` for the final comparison.** It looks like the right tool
> for pinning depth, but in 0.0.16.post1 it disables the window loop outright —
> `inference_profiler.cc`: `// If request-count is specified, then only measure one window
> and exit` → `if (request_count != 0) { *is_stable = true; break; }`. That leaves exactly
> one window, so there is nothing to trim in Step 3 and the whole measurement is
> ramp-up-contaminated by construction. An earlier version of this document told you to do
> this; it was self-contradictory and is withdrawn.

**What to report instead of a matched depth.** Give each arm's **trimmed sample size ÷
concurrency** (computed after the Step 3 window trim — not GenAI-Perf's `Request Count`,
which spans the whole run and so describes a different population than the figures you
publish), and give the evidence that the retained windows are stationary. The sample sizes
will differ between arms; that is expected and affects precision, not validity.

> [!CAUTION]
> **Two heuristics this document previously endorsed are withdrawn.**
>
> - *"Treat any cross-arm gap smaller than the depth gap as unresolved."* There is no basis
>   for it. Depth-response is neither uniform across arms nor monotone — the same depth
>   change moved PD's throughput 41% and TP8's 4%, and moved TP16's in *opposite
>   directions* at c20 vs c40. So a depth gap bounds nothing in either direction, and a
>   *larger* metric gap is not thereby validated.
> - *"An unmatched-`N` comparison is confounded, full stop."* Too strong, and circular —
>   see the note above Step 1. Under fixed concurrency and fixed duration, differing `N`
>   *is* the throughput difference under test.
>
> What actually blocks a cross-arm claim: windows that are not stationary, ramp-up left in
> the numbers, or n=1 with no variance estimate. All three apply to the figures currently in
> this repo, which is why its cross-arm gaps are reported as provisional rather than as
> results.

Keep `--num-prompts` above the total request count, or prompt reuse raises
prefix-cache hit rate mid-run and drags TTFT down.

**Step 3 — Trim to the stationary windows yourself. The tool will not do it for you.**

> [!CAUTION]
> **`--stability-percentage` decides when perf_analyzer *stops*, not what gets
> reported.** perf_analyzer collects every window including ramp-up
> ([`profile_data_collector.cc`](https://github.com/triton-inference-server/perf_analyzer/blob/main/src/profile_data_collector.cc)),
> and GenAI-Perf then computes its statistics over **all** requests in the export —
> `_parse_profile_data` is documented as parsing "the entire profile data"
> ([`llm_profile_data_parser.py`](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/genai_perf/profile_data_parser/llm_profile_data_parser.py)),
> with no window filtering. **So the summary genai-perf prints after a
> stability-detected run is not a steady-state measurement** — it is an average over
> ramp-up plus steady state.
>
> Measured on the L40S rig, dropping window 1 and keeping the rest: TTFT p50
> **1,729 → 2,059 ms** at c20 (+19%) and **3,345 → 3,510 ms** at c40 (+5%). Per-window
> percentiles show why — c20 window 1 p50 is 1,639 ms against 2,059 / 2,074 for windows 2
> and 3; c40 is 2,583 against 3,695 / 3,510. **The bias understates latency.**

**Use the `window_boundaries` array in `profile_export.json` — do not reconstruct windows
from the interval.** Each experiment carries an explicit list of boundary timestamps (ns),
`n+1` of them for `n` windows. Filter requests whose **last response timestamp** falls
between the boundaries you want to keep, and drop the early windows rather than a time
slice.

⚠️ **`interval` is not the window width — the multiplier is 1.2, hard-coded.** `Measure()`
sleeps for `config.measurement_window * 1.2` (`inference_profiler.cc:1229-1230`), so a
window is always 1.2× the value you pass to `--measurement-interval`. Confirmed empirically
too: 60 s configured produced three 72.0 s windows on the L40S rig. Consequently
`last 3 × interval` spans only ≈2.5 real windows and silently cuts into the one you meant to
keep. An earlier version of this document told you to slice on `3 × interval`; that was
wrong, and the "+10–15%" it produced was an artefact of the bad cutoff, not a measurement of
ramp-up. The figures above are recomputed from the exported boundaries.

⚠️ **`--warmup-request-count` does not hand you a warm queue.** perf_analyzer runs the
warmup as a separate load phase and then calls `WaitForWarmupAndCleanup()`, which joins every
worker thread and clears `workers_`/`threads_` before the measured load starts
(`load_manager.cc`). So warmup can prime weights, caches and CUDA graphs, but the
closed-loop queue starts **empty** either way, and window 1 still contains the queue-filling
transient. An earlier version of this document claimed a large warmup makes all three windows
usable; that was wrong.

⚠️ **Window count runs between 3 and `--max-trials` (default 10).** `stability_window` is 3
(`inference_profiler.cc:522`) but `DetermineStability` uses it as a *rolling* window —
`idx = infer_per_sec.size() - stability_window`, sliding forward as windows accumulate — so 3
is the **minimum**, reached when the run stabilises immediately. The **maximum** is
`max_trials_`: the collection loop is `while ((!early_exit) && (completed_trials <
max_trials_))` (`inference_profiler.cc:811`), default `max_trials = 10`
(`command_line_parser.h:66`). Exhausting it is not silent — perf_analyzer prints
`Failed to obtain stable measurement within N measurement windows ... Please try to increase
the --measurement-interval` and marks the run as not meeting the threshold
(`inference_profiler.cc:573-585`). **Treat that message as a failed run, not a slow one.**
(An earlier version of this document said raising the interval could not add windows, and a
later one said windows accumulate "indefinitely" — both wrong.)

**Define the trim rule before you look at the data.** Otherwise dropping "the windows that
differ" is post-hoc window-picking. Fix all three of these in advance:

- **Metric**: the one the comparison turns on (e.g. TTFT p50), plus throughput.
- **Threshold**: reuse the tool's own criterion — `max/min ≤ 1 + stability_threshold` across
  the retained windows, i.e. 10% at `--stability-percentage 10`.
- **Contiguity**: retained windows must be the **trailing** run of windows; drop from the
  front only, never from the middle.
- **Minimum retained**: **at least 2 windows, and prefer 3.** This is a hard constraint, not a
  preference — with one window `max/min = 1` by construction, so a single window passes any
  threshold vacuously and "the trim succeeded" would be meaningless.

Procedure: compute the per-window metric, then drop leading windows one at a time **while at
least 2 remain**, stopping as soon as the trailing set passes the threshold. Report how many you
dropped and the per-window values, so the reader can see the rule was applied rather than the
result chosen.

**If no trailing set of ≥2 windows passes, the run failed** — say "did not reach steady state
at C=x" rather than publishing a percentile. Note that with the minimum 3 windows you can drop
at most one before you are down to two, so a run that needs more trimming than that has to be
re-run with a longer interval (a longer window is more likely to contain the whole transient)
rather than trimmed further.

These runs stabilised in the minimum three, with every request inside them (0 before the
first boundary, 0 after the last), so the trim above drops window 1 and reports the trailing
two.

⚠️ **Filter on request *end*, not request start.** perf_analyzer attributes a request to
the window it *finishes* in — `inference_profiler.cc`: `// Only counting requests that end
within the time interval`, gated on `request_end_ns`. Filtering by start selects a
different sample than the stability check evaluated and right-truncates the final window.

Publish, for the trimmed subset: the **full TTFT distribution** (p1…max), the
`--measurement-interval`, each arm's **trimmed sample size ÷ C** (not GenAI-Perf's whole-run
`Request Count`), and the number of
windows run.

⚠️ **Percentiles cannot show settling, in either direction.** They are a sorted summary, so
p10 ≤ p50 ≤ p90 always — a "monotone ramp" is arithmetic, not evidence, and a heavy-tailed but
perfectly stationary series produces one too (a lognormal sample with no time trend by
construction reads p10 0.24 → p90 3.93). Nor is a narrow spread proof of settling. **Only
time-ordered data can answer this**: publish **per-window percentiles** (window 1, 2, 3 …) or
a TTFT-vs-time scatter, and show the retained windows agree within the threshold. Earlier
versions of this document claimed first that distribution shape *proves* settling and then
that a monotone ramp *disproves* it; both are withdrawn.

⚠️ Recomputing by hand introduces its own comparability problem: genai-perf re-encodes
output text with the tokenizer to count tokens, so **hand-derived throughput and ITL are
not comparable to the tool's own**. TTFT percentiles are safe (first-byte latency is
computed the same way either method).

**Drain the server to idle between runs.** Back-to-back runs inherit the previous
run's backlog; on rig 3 that contamination exceeded the depth effect being measured
(§"Third rig"). Gate on GPU utilisation returning to ~0 before each run.

## When a run never converges

If the windows never satisfy the stability threshold — perf_analyzer exhausts
`--max-trials` and prints `Failed to obtain stable measurement within N measurement
windows` — report **"did not reach steady state at C=x"** rather than publishing a
percentile.

⚠️ **That is a statement about the measurement, not about the deployment.** Under a
closed-loop generator, concurrency caps in-flight requests, so a non-converging run does
**not** show "the model cannot serve that concurrency" — it shows the run did not
demonstrate a settled state within the windows collected. An earlier version of this
document drew the stronger conclusion; it does not follow. The usual causes are a window
too short to contain the transient, or genuine non-stationarity in the workload or server
(cache filling, background load, thermal drift) — none of which is a capacity verdict.

First response: **lengthen the interval** so each window contains more of the transient,
and re-run. A longer window costs machine time; publishing an unsettled percentile costs
correctness, so the asymmetry favours the longer window.

## The depth rule stops being economic at very long input

The `≥10 requests/slot/window` target above was written for short prompts. Its cost
scales with per-request latency, and at long context that gets out of hand: on
Kimi-K3 TP8×PP3 (3× p5en) a single 128,000-token point took **43 minutes** of
stability detection — four windows — because one request runs ~60 s, forcing
≥10-minute windows. One data point cost roughly what forty short-context points cost.
At a 1,030,004-token input a compliant window would be ≥28 minutes.

For long-input work, use `--request-count` instead and **say so in the writeup**: a
fixed count produces a single window, so ramp-up cannot be trimmed and the figures are
capability probes rather than steady state. Never mix them into a table with
stability-detected numbers.

## `Avg prompt tput` in vLLM logs is not a rate

A trap worth knowing before using engine logs as evidence. The vLLM stats line reports
the prompt tokens of prefills that **completed** inside the 10 s reporting window,
divided by 10 s — so the value quantises to multiples of ISL/10 and says nothing about
instantaneous prefill speed. Verified on Kimi-K3 TP8×PP3 to within 0.002%:

| observed | ISL × completions / 10 s |
|---|---|
| 12,801.9 | 128,019 × 1 / 10 |
| 25,604.2 | 128,019 × 2 / 10 |
| 103,790.2 | 1,037,925 × 1 / 10 |

Two consequences. `prefill=0.0` does **not** mean prefill stopped — a long prefill
spans many windows and reports 0.0 in all but the last. And a value that looks
suspiciously constant across a config change is expected, not evidence: reading an
unchanged 12,801.x as proof that a token-budget flag had no effect produced a wrong
conclusion in this repo, later retracted
([KIMI-K3-PP3-PERFORMANCE.md](KIMI-K3-PP3-PERFORMANCE.md) § 7.1).

## Measured basis (single rig — this is the weakness)

1× L40S (g6e), Qwen3-8B TP1, sglang v0.5.13.post1, genai-perf 0.0.16.post1,
ISL/OSL 2000/256 pinned, thinking off.

| Observation | Value | Held up elsewhere? |
|---|---|---|
| Default measurement depth | `max(10, 2 × concurrency)` — 2 requests/slot | ✅ source-verified |
| TTFT p50 understatement at 2 requests/slot | 6–14× (grows with concurrency) | ❌ rig 2 (overstated), rig 3 (1.1–1.2×) |
| TTFT p90 understatement at 2 requests/slot | ~1.4–1.5× | ❌ rig 2 overstated by 1.9–2.8× |
| Throughput sensitivity to depth | ±5% (essentially none) | ❌ rig 3: overstated up to 53% |
| Trend (c20→c40 TTFT p90) across all depths tested | +101% … +115% (stable) | ❌ rig 3: shallow exaggerates the throughput curve |
| Realised depth under stability detection | 18 req/slot at c20, 12 at c40 | ✅ direction holds (rig 3: 23.3 → 14.7) |

**Only the first and last rows generalise.** Realised depth falling as concurrency rises is
the one measured regularity reproduced on another rig. It is an *outcome* of the run, not a
condition to equalise (Step 2) — its practical use is as a warning that a concurrency sweep
does not sample each point equally deeply.

Source-verified (not rig-specific): the `max(10, 2 × concurrency)` formula
(`perf_analyzer_config.py::_calculate_request_count`); `--num-prompts` is an alias
for `--num-dataset-entries`, a reused sampling pool; genai-perf defaults
`--stability-percentage` to `999` versus perf_analyzer's own `10`; `--request-count`
mode measures exactly one window.

## Second rig — 2026-07-29 (Kimi-K3 / B300): direction of bias does NOT hold

Measured on 1× p6-b300.48xlarge (8× B300), Kimi-K3 2.8T MoE TP8, vLLM
`0.1.dev19262`, ISL/OSL 1024/1024 pinned, thinking off. Calibrated `N` = 12
requests/slot at c64 by runtime detection, then compared against the same points
measured at the default 2 requests/slot.

| | TTFT p90 @ 2 req/slot | TTFT p90 @ 12 req/slot | Change |
|---|---|---|---|
| c32 | 2,256 ms | 1,165 ms | **−48%** |
| c64 | 3,549 ms | 1,278 ms | **−64%** |

**Deepening the measurement *lowered* TTFT — the opposite of the L40S rig, where
short windows understated it.** Reproduced independently at both concurrency levels.
The two runs differ sharply in spread — at 12 req/slot the mid-to-upper percentiles sit close
together (c64: p50 1,175 → p90 1,278, 9%) while at 2 req/slot they are far apart (p50 1,281 →
p90 3,549, 177%). ⚠️ **That is not evidence about settling**: percentiles carry no time
information, and a stationary heavy-tailed series can be just as wide. Per-window data was not
retained for this rig, so neither run is shown to have settled.

Plausible mechanism: at 2 req/slot the window is dominated by the initial burst, when
all C requests arrive at once and contend for prefill. No steady pipeline exists yet,
so every request pays near-worst-case queueing, and the run never reaches the regime
where prefill and decode interleave. On a fast MoE with large batch capacity that
startup transient is proportionally more severe than on a small dense model.

**What survives and what does not:**

- **Survives** — steady state is a validity precondition. (This bullet previously also
  claimed "depth must be fixed in requests/slot", that distribution shape *proves* settling,
  and that a monotone percentile ramp *disproves* it; all three are withdrawn — see
  §"Proposed resolution" and Step 3.)
- **Does not survive** — the claim that short measurement biases *in favour of slow
  models* because it understates TTFT. The bias direction is rig-dependent.
  **Consequence: short-window results cannot be assumed conservative in either
  direction, so they cannot be salvaged by arguing "the real number is only better/
  worse than this".** They are simply invalid.

This also means the §"Measured basis" understatement figures (6–14× at p50, 1.4–1.5×
at p90) are L40S/Qwen3-8B-specific, not general properties of the tool.

Full data: `local/logs/KIMI-K3-B300-BENCHMARK-internal.md` (git-ignored).

## Third rig — 2026-07-29 (GLM-5.2 / H200): throughput, not TTFT, is what breaks

Measured on 1× p5en.48xlarge (8× H200), GLM-5.2-FP8 MoE **TP8**, sglang
v0.5.13.post1 (same engine version as the original GLM-5.2 report), genai-perf
0.0.16.post1, ISL/OSL 8000/1024 pinned, thinking off. Runtime detection settled at
23.3 requests/slot at c20 and 14.7 at c40. Each pair below is a clean A/B: identical
prompt pool (2000), identical `--warmup-request-count`, server **drained to GPU-idle
before each run**, only measurement depth differing.

| | 2 req/slot | deeper whole-run | Change |
|---|---|---|---|
| **c20** — depth | 40 | 466 (23.3/slot) | |
| TTFT p50 | 785 ms | 869 ms | +11% |
| TTFT p90 | 1,634 ms | 1,514 ms | −7% |
| Total throughput | 577 tok/s | 552 tok/s | −4% |
| ITL avg | 24.5 ms | 34.4 ms | +40% |
| **c40** — depth | 80 | 589 (14.7/slot) | |
| TTFT p50 | 723 ms | 892 ms | +23% |
| TTFT p90 | 959 ms | 1,772 ms | +85% |
| Total throughput | **1,070 tok/s** | **698 tok/s** | **−35%** |
| ITL avg | 27.4 ms | 54.5 ms | +99% |

**The TTFT error was small (1.1–1.2× at p50) but throughput was inflated by 53% at
c40** (1,070 shallow vs 698 deeper). This is the reverse of the L40S rig, where TTFT moved
6–14× and throughput held to ±5%. ITL moved with throughput (27.4 → 54.5 ms at c40).

Mechanism not established — no batch-occupancy or KV-utilisation telemetry was
collected, so why the rate metrics are the depth-sensitive ones here is unmeasured.

The spreads differ here too, more sharply than on either prior rig — but again this says
nothing about settling, for the reason in Step 3:

```
c40 deeper  (589 req)  p10 887  p25 889  p50 892  p75 1640
c40 shallow  (80 req)  p10 328  p25 348  p50 723  p75  944
```

**A second methodology error surfaced, unrelated to depth:** the first attempt ran the
A/B back-to-back with no drain, and the later run inherited the previous run's queue
backlog. Undrained shallow c20 read p50 516 ms / 692 tok/s; drained, the same command
read 785 ms / 577 tok/s. **That contamination was larger than the depth effect under
study.** Draining to GPU-idle between runs is now a required step, not hygiene.

**Consequence for §"Measured basis":** the row "Throughput sensitivity to depth: ±5%
(essentially none)" is falsified. It held on the L40S/dense rig; here throughput was
the *most* depth-sensitive metric measured. Do not cite shallow-run throughput.

**Re-validation of the original GLM-5.2 report:** the archived TP8 figures
(`docs/GLM-5.2-BENCHMARK.md`, row R4: TTFT p50 3,202 ms / p90 17,145 ms, 456 tok/s)
compare against the deeper whole-run values p50 869 ms / p90 1,514 ms / 552 tok/s — i.e. the
original **overstated** TTFT by ~3.7× (p50) and ~11× (p90) — the opposite direction from this
document's original prediction. (An earlier version added that R4's wide percentile spread
showed "the queue was growing throughout"; that inference is withdrawn — percentiles carry no
time information.) Throughput reproduced within ~21% across a different account/cluster.

**TP16 on the same rig (2026-07-30), 2× p5en, same tool and workload** — this is the
measurement that resolves the c20-vs-c40 reversal described in §"Known unexplained
observation":

| TP16 8K/1K | 2 req/slot | deeper whole-run |
|---|---|---|
| c20 TTFT p90 | 12,119 ms | **901 ms** (22.1 req/slot) |
| c40 TTFT p90 | 1,684 ms | **1,644 ms** (11.6 req/slot) |
| c20 throughput | 414.0 tok/s | **523.6** |
| c40 throughput | 738.7 tok/s | **659.1** |

At 2 requests/slot the reversal reproduces — c20's p90 reads **7× worse than c40's**,
the same qualitative anomaly as the original report's 14.8 s vs 1.5 s. At the deeper depth it
disappears: p90 rises with concurrency (901 → 1,644 ms), which is the expected direction.

**So under-depth sampling does explain that anomaly**, at least on this shape — the
earlier conclusion that it could not (based on the L40S and B300 rigs failing to
reproduce a reversal) was wrong to generalise from other hardware. Note also that the
throughput bias **changes sign with concurrency here** (21% low at c20, 12% high at c40,
both relative to the deeper value); cause not established.

## Validation checklist

- [x] Reproduce on a second model class (MoE, not dense) — **done: Kimi-K3 2.8T MoE.
      Procedure held; bias direction did not (see above)**
- [x] Reproduce on a second GPU class — **done: B300 (Blackwell) vs L40S**
- [x] Reproduce on a second workload shape (8K/1K vs 2K/256) — **done: rig 3 ran
      8K/1K with a clean 2-req/slot A/B at both c20 and c40**
- [x] Re-check the GLM-5.2 / H200 case now that the bias direction is known to vary —
      **done: §"Third rig". TTFT bias small; throughput bias large. TP8 only**
- [x] Re-measure the **TP16** GLM-5.2 shape at greater depth — **done 2026-07-30 on
      2× p5en; the c20-vs-c40 reversal reproduced at 2 req/slot and vanished in the deeper
      whole-run measurement, so under-depth sampling does explain it on this shape**
- [ ] **Re-derive every published figure from stationary windows only** — all current
      numbers are whole-run averages including ramp-up (Step 3). Requires retaining
      `profile_export.json`, which was not done for the GLM-5.2 runs
- [ ] **Execute Step 2 as now written, at least once** — no comparison in this repo yet
      combines one pinned interval, window-boundary trimming to stationary windows, repeat
      runs with a spread or confidence interval, and each arm's trimmed sample size reported
      alongside
- [ ] Confirm the *interval* calibrated on the slowest arm gives every other arm ≥10
      req/slot per window — the only cross-arm quantity Step 1 actually sets
- [x] A/B the two paths on the **same** hardware — **done on rig 3** (same node, same
      pool, same warmup, drained between runs). The earlier 22% gap attributed to node
      sizes (8 vs 16 vCPU) is better explained by **inter-run queue contamination**:
      undrained vs drained changed the same shallow c20 command by 34% on p50
      (516 → 785 ms), larger than the depth effect itself
- [ ] Confirm short-window bias does **not** invert relative ordering between models
      (no rig has shown a cross-concurrency inversion, but cross-model is untested)
- [ ] Establish whether longer ISL needs a longer interval to settle — rig 3 (8K) realised
      23.3 req/slot at c20 vs rig 1 (2K) at 18, weakly consistent with longer prefill taking
      longer to settle, but the rigs differ in too many other ways to attribute it

## Known unexplained observation

On the original GLM-5.2 runs, TP16 TTFT p90 read 14.8 s at c20 and 1.5 s at c40 — an
apparent 10× improvement under heavier load. Both runs measured 2 requests/slot, so
neither is a steady-state value and the comparison is void. Deliberately short
measurement on the L40S rig **did not reproduce a direction reversal** — it
understated TTFT at both concurrency levels roughly equally.

**Updated 2026-07-29:** the B300 rig shows under-depth measurement can *overstate*
TTFT p90 by up to 2.8×, and that the overstatement grows with concurrency (1.94× at
c32, 2.78× at c64). That is a mechanism by which under-depth sampling could produce an
apparent improvement at higher concurrency — but it is the wrong sign to explain the
GLM-5.2 observation, where the *higher* concurrency read *lower*. Growing
overstatement with concurrency would inflate c40 more than c20, not less. So depth
still does not explain that reversal, and its cause remains unknown.

**Updated 2026-07-30 (rig 3, same model on H200/TP8):** shallow measurement again
failed to reverse direction — at 2 req/slot, TTFT p90 read 1,634 ms at c20 and 959 ms
at c40, so it *did* read lower at higher concurrency, but only by 1.7×, nothing like
the 10× in the original TP16 data. At the deeper depth the ordering corrected to the
expected direction (1,514 ms at c20 → 1,772 ms at c40).

## Resolved 2026-07-30 — the reversal *was* under-depth sampling

Measured on the shape where it was originally seen: **GLM-5.2 TP16, 2× p5en**, same tool
and workload, drained between runs.

| TP16 TTFT p90 | 2 req/slot | deeper whole-run |
|---|---|---|
| c20 | 12,119 ms | **901 ms** (22.1 req/slot) |
| c40 | 1,684 ms | **1,644 ms** (11.6 req/slot) |

At 2 requests/slot c20 reads **7× worse than c40** — the anomaly reproduces. In the deeper
whole-run measurement it is gone, and p90 rises with concurrency as expected. The original
14.8 s vs 1.5 s was a sampling artefact. (Both of these are whole-run values including
ramp-up; no window data was retained for this rig.)

**This corrects the two updates above.** Both concluded depth "cannot account for" the
reversal, generalising from rigs (L40S, B300, and TP8 on H200) that simply do not exhibit
it. The lesson is narrower than either claim: whether under-depth sampling inverts the
concurrency ordering is **shape-specific**, and the only way to know is to measure the
shape in question. What is still not established is *why* TP16 is susceptible where TP8
on the same node type is not — no queue-depth or allreduce telemetry was collected.
