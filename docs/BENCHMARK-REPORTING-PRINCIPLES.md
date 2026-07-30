# Benchmark Reporting Principles

Rules for every benchmark report published from this repo (`docs/*-BENCHMARK.md`,
`docs/*-PERFORMANCE*.md`, and any performance claim in the README or a manifest
comment). They govern how results are *written up*, not how they are measured —
for measurement procedure see
[benchmark-commands.md](benchmark-commands.md#measuring-steady-state).

1. State only conclusions the data directly supports. No overstatement, no
   embellishment, no selective presentation.

2. Keep three things strictly separate:
   - **Observed fact** — what the data or logs directly show.
   - **Analytical inference** — derived from the data, but not directly verified.
   - **Causal hypothesis** — a possible explanation; must be explicitly labelled
     *speculation* or *unverified*.

3. No causal attribution without a controlled experiment, profiling, or explicit
   log evidence. Write "may be related to" or "cause not yet established" — not
   "caused by".

4. Define every metric: data source, statistical method, time window, token
   accounting, and whether the figure is a mean, median, or percentile.

5. Do not mix statistical bases. Metrics in a table should be mutually
   recomputable; where they cannot be reconciled, say so — never paper over a
   contradiction.

6. Comparisons must hold everything constant except the variable under study.
   Differences in caching, warmup, versions, load, or runtime environment must be
   disclosed as confounders.

7. Bimodal, long-tailed, or anomalous distributions cannot be summarised by a
   single mean or median. Report the distribution, the sample count, and the
   anomaly.

8. Do not use "best", "knee/inflection point", "essentially the same", or
   "significant improvement" without a definition. If used, give the criterion.

9. Bound the scope of every conclusion explicitly. A result speaks only for the
   hardware, versions, configuration, concurrency, and workload actually tested —
   do not extrapolate to untested cases.

10. Report failures, anomalies, cache hits, queueing, retries, and discarded runs
    honestly, and state the discard criterion.

11. Provide the full commands, configuration, versions, and raw-data locations, so
    that key numbers are traceable, recomputable, and reproducible.

12. When uncertain, write "insufficient evidence, no conclusion yet". Do not fill
    the gap with a narrative or promote a plausible guess to a fact.

## Pre-publication check

Go through these before writing, item by item:

- [ ] Is every fact traceable to its source?
- [ ] Can every metric be recomputed?
- [ ] Is every comparison fair?
- [ ] Does any inference cross the boundary of its evidence?
