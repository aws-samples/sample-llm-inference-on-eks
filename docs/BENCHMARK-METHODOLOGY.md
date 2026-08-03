# Benchmark Methodology — Measuring Steady State and Comparing Fairly

**This document is the standard for benchmarks in this repo.** Follow §"Procedure".

> [!CAUTION]
> **Every number published from this methodology so far was computed over the whole run,
> ramp-up included.** Step 3 explains why: `--stability-percentage` controls when
> perf_analyzer stops, not what it reports, and GenAI-Perf aggregates every request in the
> export. Measured on the L40S rig, isolating the last three windows raises TTFT p50 by
> 10–15%. So the existing figures in §"Measured basis", §"Second rig" and §"Third rig",
> and every "steady state" number in
> [GLM-5.2-BENCHMARK.md](GLM-5.2-BENCHMARK.md), are **closer to steady state than the
> 2-requests-per-slot defaults but still contaminated, and biased low on latency.** They
> are not the final word; re-deriving them from the last three windows is tracked in
> §"Validation checklist".

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
> What generalised across all three rigs: steady state is a validity precondition; depth
> must be held in requests-per-slot, not seconds; **distribution shape** (plateau vs
> monotone ramp) is the diagnostic; and `N` falls as concurrency rises, so one fixed
> depth must not be reused across a sweep.
>
> Open items are tracked in §"Validation checklist". Two of the five are gaps in how the
> procedure has been *applied* so far — last-three-window extraction (Step 3) and pinned
> depth across arms (Step 2) have never been executed; the rest concern whether *numbers*
> transfer between rigs.
>
> Created 2026-07-28 as a draft; procedure documented 2026-08-03. The procedure itself is
> settled, but Steps 2 and 3 have not yet been executed on any published comparison.

## The question this answers

When benchmarking LLM latency, two goals appear to conflict:

- **Steady state** — measure long enough that the request queue has settled, or the
  latency number is simply wrong.
- **Controlled comparison** — hold every parameter identical across models, or the
  comparison is confounded.

They appear to conflict because reaching steady state takes *longer on a slower
model*, so equal measurement duration and equal measurement depth cannot both hold.

## Proposed resolution

**These are not competing conditions. Steady state is a validity precondition;
measurement depth is the controlled variable.**

A latency figure taken before the queue settles is not "latency under a different
condition" — it is a mis-measurement. Comparing two mis-measured numbers is not a
comparison. So steady state is not something to trade off against control; it is
what makes the numbers mean anything.

The real problem hidden inside the question is different, and it is legitimate: if
steady state is determined *at runtime* by the tool, then measurement depth varies
per model and becomes an uncontrolled variable.

The resolution is to **convert steady state from a runtime decision into a
pre-fixed parameter**:

| Approach | Who decides depth | Same across models? | Use for |
|---|---|---|---|
| `--measurement-interval` + `--stability-percentage 10` | tool, at runtime | ✗ varies | calibration |
| `--request-count = N × concurrency` | you, in advance | ✓ identical | comparison |

**Requests per concurrency slot (`N`) is the quantity to hold fixed** — not seconds.
Queue depth is counted in requests, a dimensionless quantity independent of model
speed. Seconds are not: the same duration corresponds to entirely different queue
states on a fast versus a slow model.

Fixing seconds instead of `N` is not neutral: a slow model completes fewer requests
per slot in the same window, so it is sampled further from steady state. The two
models are then measured under different conditions, which is enough to invalidate
the comparison regardless of which way the error runs.

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

Size the interval from measured request throughput; aim for ≥10 requests per slot
across the 3 windows perf_analyzer averages:

```
interval_ms ≳ (10 × concurrency) / (3 × requests_per_sec) × 1000
```

Read the resulting `Request Count` from the report and divide by concurrency to get
the observed `N`. Round up for headroom.

**Step 2 — Compare (all models, fixed depth).**

```
--concurrency <C>  --request-count <N × C>
```

Every model now receives an identical treatment: same concurrency, same requests per
slot, same ISL/OSL, same tool version. Note that `--request-count` mode **skips**
perf_analyzer's stability check (source: `inference_profiler.cc` — `if request_count
!= 0 { *is_stable = true; break; }`), which is why Step 1 is not optional.

Keep `--num-prompts` above the total request count, or prompt reuse raises
prefix-cache hit rate mid-run and drags TTFT down.

**Step 3 — Extract the stable windows yourself. The tool will not do it for you.**

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
> Measured on the L40S rig, restricting to the last three windows vs using the whole
> run: TTFT p50 **1,729 → 1,988 ms** at c20 (+15%) and **3,345 → 3,673 ms** at c40
> (+10%). The excluded early requests read p50 350 ms (c20) and 165 ms (c40) — an
> order of magnitude faster, i.e. an empty queue. **The bias understates latency.**

Take the per-request records from `profile_export.json` (timestamps are ns), keep only
requests whose start falls inside the last three measurement windows, and recompute the
percentiles from those. Publish the **full TTFT distribution** (p1…max) for that subset,
plus `--measurement-interval`, the observed `Request Count`, and the number of windows
run. Distribution shape is the evidence of settling — a wide spread with no plateau means
the queue was still growing — and it is the one diagnostic that has held on all three
rigs.

⚠️ Recomputing by hand introduces its own comparability problem: genai-perf re-encodes
output text with the tokenizer to count tokens, so **hand-derived throughput and ITL are
not comparable to the tool's own**. TTFT percentiles are safe (first-byte latency is
computed the same way either method).

**Drain the server to idle between runs.** Back-to-back runs inherit the previous
run's backlog; on rig 3 that contamination exceeded the depth effect being measured
(§"Third rig"). Gate on GPU utilisation returning to ~0 before each run.

## When a model never converges

If `N` is insufficient for some model — the queue keeps growing regardless — that is
**a result, not a measurement failure**: the model cannot serve that concurrency.
Report "did not reach steady state at C=x" rather than publishing a percentile.

Raising `N` costs machine time. Lowering it costs correctness. The asymmetry favours
raising it.

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
| `N` chosen by runtime detection | 18 at c20, 12 at c40 | ✅ direction holds (rig 3: 23.3 → 14.7) |

**Only the first and last rows generalise.** `N` decreasing as concurrency rises is
the one measured regularity reproduced on another rig — which is itself the reason a
single fixed depth must not be reused across a concurrency sweep.

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
The distribution shape identifies the deeper run as the valid one: at 12 req/slot the
mid-to-upper percentiles plateau (c64: p50 1,175 → p90 1,278, a 9% spread), while at
2 req/slot they climb monotonically with no plateau (p50 1,281 → p90 3,549, 177%).

Plausible mechanism: at 2 req/slot the window is dominated by the initial burst, when
all C requests arrive at once and contend for prefill. No steady pipeline exists yet,
so every request pays near-worst-case queueing, and the run never reaches the regime
where prefill and decode interleave. On a fast MoE with large batch capacity that
startup transient is proportionally more severe than on a small dense model.

**What survives and what does not:**

- **Survives** — steady state is a validity precondition; depth must be fixed in
  requests/slot, not seconds; distribution shape is the evidence of settling. All of
  §"Proposed resolution" and §"Procedure" stand, and were used successfully here.
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

| | 2 req/slot | steady state | Change |
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
c40** (1,070 read vs 698 real). This is the reverse of the L40S rig, where TTFT moved
6–14× and throughput held to ±5%. ITL moved with throughput (27.4 → 54.5 ms at c40).

Mechanism not established — no batch-occupancy or KV-utilisation telemetry was
collected, so why the rate metrics are the depth-sensitive ones here is unmeasured.

**Distribution shape again identified the valid run**, and more cleanly than on either
prior rig:

```
c40 steady  (589 req)  p10 887  p25 889  p50 892  p75 1640    <- plateau
c40 shallow  (80 req)  p10 328  p25 348  p50 723  p75  944    <- monotone ramp
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
compare against steady state as p50 869 ms / p90 1,514 ms / 552 tok/s — i.e. the
original **overstated** TTFT by ~3.7× (p50) and ~11× (p90). That is the opposite
direction from this document's original prediction, and consistent with the report's
own observation that its TTFT distribution ramped with no plateau: the queue was
growing throughout, which is a different failure from under-sampling. Throughput
reproduced within ~21% across a different account/cluster.

**TP16 on the same rig (2026-07-30), 2× p5en, same tool and workload** — this is the
measurement that resolves the c20-vs-c40 reversal described in §"Known unexplained
observation":

| TP16 8K/1K | 2 req/slot | steady state |
|---|---|---|
| c20 TTFT p90 | 12,119 ms | **901 ms** (22.1 req/slot) |
| c40 TTFT p90 | 1,684 ms | **1,644 ms** (11.6 req/slot) |
| c20 throughput | 414.0 tok/s | **523.6** |
| c40 throughput | 738.7 tok/s | **659.1** |

At 2 requests/slot the reversal reproduces — c20's p90 reads **7× worse than c40's**,
the same qualitative anomaly as the original report's 14.8 s vs 1.5 s. At steady state it
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
- [x] Re-measure the **TP16** GLM-5.2 shape at steady state — **done 2026-07-30 on
      2× p5en; the c20-vs-c40 reversal reproduced at 2 req/slot and vanished at steady
      state, so under-depth sampling does explain it on this shape**
- [ ] **Re-derive every published figure from the last three windows only** — all current
      numbers are whole-run averages including ramp-up (Step 3). Requires retaining
      `profile_export.json`, which was not done for the GLM-5.2 runs
- [ ] **Execute Step 2 at least once** — every cross-arm comparison so far used runtime
      stability detection, so the arms ran at different depths (e.g. 14.7 vs 11.6 req/slot).
      No comparison in this repo has yet pinned `--request-count = N × C` across arms
- [ ] Confirm `N` calibrated on one model transfers to others in the same comparison
- [x] A/B the two paths on the **same** hardware — **done on rig 3** (same node, same
      pool, same warmup, drained between runs). The earlier 22% gap attributed to node
      sizes (8 vs 16 vCPU) is better explained by **inter-run queue contamination**:
      undrained vs drained changed the same shallow c20 command by 34% on p50
      (516 → 785 ms), larger than the depth effect itself
- [ ] Confirm short-window bias does **not** invert relative ordering between models
      (no rig has shown a cross-concurrency inversion, but cross-model is untested)
- [ ] Establish whether `N` must scale with ISL — rig 3 (8K) settled at 23.3 req/slot
      at c20 vs rig 1 (2K) at 18, weakly consistent with deeper queues for longer
      prefill, but the rigs differ in too many other ways to attribute it

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
the 10× in the original TP16 data. At steady state the ordering corrected to the
expected direction (1,514 ms at c20 → 1,772 ms at c40).

## Resolved 2026-07-30 — the reversal *was* under-depth sampling

Measured on the shape where it was originally seen: **GLM-5.2 TP16, 2× p5en**, same tool
and workload, drained between runs.

| TP16 TTFT p90 | 2 req/slot | steady state |
|---|---|---|
| c20 | 12,119 ms | **901 ms** (22.1 req/slot) |
| c40 | 1,684 ms | **1,644 ms** (11.6 req/slot) |

At 2 requests/slot c20 reads **7× worse than c40** — the anomaly reproduces. At steady
state it is gone, and p90 rises with concurrency as expected. The original 14.8 s vs 1.5 s
was a sampling artefact.

**This corrects the two updates above.** Both concluded depth "cannot account for" the
reversal, generalising from rigs (L40S, B300, and TP8 on H200) that simply do not exhibit
it. The lesson is narrower than either claim: whether under-depth sampling inverts the
concurrency ordering is **shape-specific**, and the only way to know is to measure the
shape in question. What is still not established is *why* TP16 is susceptible where TP8
on the same node type is not — no queue-depth or allreduce telemetry was collected.
