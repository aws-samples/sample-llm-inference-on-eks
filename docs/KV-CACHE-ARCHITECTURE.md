# KV-Cache Offloading & Externalization — Architecture and Measurements

**Dates**: 2026-07-08 (all experiments)
**Model**: `zai-org/GLM-5.2-FP8` (753B MoE, DSA sparse attention), fp8 KV throughout
**Hardware**: 2× p5en.48xlarge (8× H200 each), capacity-block nodes; Redis on a CPU spot node
**Companion docs**: [GLM-5.2-BENCHMARK.md](GLM-5.2-BENCHMARK.md) (deployment-shape benchmarks),
[PD_DISAGGREGATION.md](PD_DISAGGREGATION.md) (PD operations)

**The question this document answers**: when a request's KV cache already exists somewhere,
how much faster is it than recomputing prefill — and where should that "somewhere" be?

## TL;DR — the tiered price list

Measured on the same ~30K-token long-document reuse scenario (`max_tokens=1` isolates
prefill, so latency ≈ TTFT):

| Tier | Where KV lives | Hot TTFT | Speedup vs cold | Sharing scope |
|---|---|---|---|---|
| GPU (radix/prefix cache) | HBM, in-engine | ~0 added | — | single instance, evictable |
| **L1 host RAM** | same pod (HiCache / LMCache local) | **0.79–2.2s** | **8–22×** | single instance |
| **L2 external store** | separate Deployment (Redis) | **~4s** | ~1.7× | **any instance, any restart** |
| Cold (no cache) | — | 7–18s\* | 1× | — |

\* cold prefill for 30K tokens: ~7s single-request, ~17.5s at concurrency 8 (queueing).

Two invariants found across every experiment:

- **Lookup is free; movement is the price.** Cache retrieval metadata is 6–9 ms in every
  tier. The tier differences are entirely data movement: CUDA-IPC zero-copy (L1/LMCache)
  ≈ 0.8s, paged host↔device copy (L1/HiCache) ≈ 2s, network (L2/Redis) ≈ 3s per 30K tokens.
- **Benefit is binary on capacity.** Working set ≤ pool → full speedup. Overflow → LRU
  evicts the working set and benefit is exactly zero (measured both ways).

## Scenario & method

**Long-document reuse**: N users × ~30K-token documents; after first (cold) computation,
GPU KV pool is flooded with distinct traffic to force eviction, then the same documents are
re-asked (hot). `max_tokens=1` so measured latency ≈ TTFT. Log-verified hits every time
(`#cached-token`, `Retrieved N tokens`, Redis `DBSIZE`) — never inferred from timing alone.

**What this method deliberately does NOT measure**: throughput. Output length 1 eliminates
decode; concurrency is kept low to avoid queue pollution. Cache effects on throughput are
indirect (freed prefill GPU-seconds → higher sustainable concurrency) and need a separate
mixed-traffic soak test — see Open items.

## Experiment 1 — engine × backend A/B (single node, L1 offload)

8 users × 30K-token docs, concurrency 8, flood sized to evict GPU but **fit in host pool**.
Four combinations, same workload, same fp8 KV:

| Engine + backend | Offload config | Cold TTFT avg | Hot TTFT avg | Speedup | Hit evidence |
|---|---|---|---|---|---|
| vLLM v0.24 + native | `--kv-offloading-backend=native`, 400GiB | 17.6s | 14.8s | 1.14× | `prefix cache hit rate: 2.5%` |
| SGLang v0.5.13 + HiCache | `ratio=2` (~330GB host) | 17.9s | 2.2s | 8.1× (wall 10.4×) | `#cached-token: 245K` = 100% reload |
| **vLLM v0.24 + LMCache** | `--kv-offloading-backend=lmcache`, 400GiB L1 | 17.5s | **0.79s** | **22×** (wall 30×: 30.0s→1.0s) | `Retrieved 30720 tokens in 0.006s` ×8 |
| SGLang v0.5.13 + LMCache | `--enable-lmcache` | — | — | ❌ crash | `AttributeError: 'DSATokenToKVPool' has no attribute 'k_buffer'` |

Cold prefill performance is equal everywhere; the offload implementations are not.
**LMCache on vLLM is the outright winner (22×)** — its MP architecture maps the engine's
GPU KV tensors via CUDA IPC and writes retrieved cache straight back to device
(6 ms for a 30K-token retrieval, per its own logs), vs HiCache's paged host↔device
copy path (8.1×, still throttled by the fp8 JIT-kernel fallback). vLLM's native
OffloadingConnector is not competitive and can be considered superseded by
`--kv-offloading-backend=lmcache` (the image already ships lmcache 0.5.0).

### SGLang + LMCache is incompatible with GLM-5.2 (upstream, confirmed)

`--enable-lmcache` crash-loops at scheduler init: SGLang's LMCache adapter
(`lmc_radix_cache.py`) hard-codes separate `k_buffer`/`v_buffer` pools (standard MHA
layout), but GLM-5.2's DSA attention uses a fused `kv_buffer` (`DSATokenToKVPool`).
Research findings (2026-07-08):

- **Known and unfixed in both projects**: identical crash reported for DeepSeek-V3.2
  in [sglang#15739](https://github.com/sgl-project/sglang/issues/15739) and for MLA models
  in [LMCache#2636](https://github.com/LMCache/LMCache/issues/2636) — both auto-closed stale.
  Fix PRs [sglang#20694](https://github.com/sgl-project/sglang/pull/20694),
  [sglang#24549](https://github.com/sgl-project/sglang/pull/24549),
  [LMCache#2629](https://github.com/LMCache/LMCache/pull/2629) all open since Mar–May, unmerged;
  sglang main still carries the `k_buffer` assumption.
- **LMCache officially does not validate SGLang** for any model family; its GLM-5.2 recipe
  marks vLLM "Validated" and SGLang "Not validated". SGLang support sits unfinished on
  LMCache's Q3 roadmap.
- **Patching past the init crash would be silently incorrect**: DSA requires the indexer
  sidecar (`index_k_with_scale_buffer`) to be cached alongside KV; no patch transfers it,
  breaking sparse-attention top-k on cache hits (same failure mode as
  [sglang#28895](https://github.com/sgl-project/sglang/issues/28895)). SGLang's own HiCache
  is the only DSA-indexer-aware offload path (`DSAIndexerPoolHost`).

**Engine guidance**: SGLang + GLM-5.2 → HiCache (8.1×, correct). LMCache required
(e.g. shared cross-engine pool) → vLLM (22×, officially validated).

## Experiment 2 — externalization: L2 store as a separate Deployment

Architecture under test:

```
┌─ vLLM Pod (GPU node) ─────────────────────────┐
│  vllm serve (engine, TP8)                     │
│      │ ZMQ + CUDA IPC                         │
│  lmcache server (same container)              │
│      ├─ L1: 100GB pinned host RAM (hot tier)  │
│      └─ L2 adapter: resp ──────────┐          │
└────────────────────────────────────┼──────────┘
                                     │ RESP protocol (network)
                    ┌────────────────▼─────────────────┐
                    │  lmcache-redis (own Deployment)  │
                    │  redis:7-alpine, 48GB LRU        │
                    │  runs on a cheap CPU spot node   │
                    └──────────────────────────────────┘
```

Configured via `lmcache server --l2-adapter '{"type":"resp","host":"lmcache-redis...","port":6379,...}'`.
lmcache 0.5.0 ships L2 adapters for: resp (Redis), s3, mooncake_store, nixl_store, fs,
aerospike, and more.

**Write path verified**: one 30K-token document → Redis holds 119 keys / 1.63 GB
(≈54 KB/token, fp8 KV all layers). Keys are **content-addressed**:
`zai-org/GLM-5.2-FP8@<meta>@<chunk-hash>` — no instance identity, which is what makes
cross-instance sharing work.

### 2a. Cache survives pod death

Kill the vLLM pod (L1 dies with the process; only Redis survives), restart, re-ask:

| Step | Latency |
|---|---|
| Same prompt after restart | **4.07s** (log: `Retrieved 30464 tokens in 0.007s`) |
| Cold control (new prompt) | 7.11s |

The new pod's GPU cache and L1 were empty — the KV could only have come from Redis.
**Prefill work now outlives any single pod.** HiCache cannot do this (its host pool dies
with the pod).

### 2b. Cross-instance sharing: A computes, B consumes

Two vLLM instances on different GPU nodes, same Redis. Instance A prefills a document;
instance B — which has **never seen the prompt** — is asked the same document:

| Step | Latency |
|---|---|
| [1] A cold prefill (KV → Redis) | 3.83s |
| [2] **B, first time seeing it** | **3.90s** (log: `Retrieved 30720 tokens in 0.007s`) |
| [3] B cold control (new prompt) | 6.47s |

B consumed A's KV. This is the mechanism kernel of every shared-cache topology (replica
pools, P/D sharing): content-addressed keys mean *who computed it is irrelevant*.

### The L2 constant: ~3 seconds per 30K tokens

Across both 2a and 2b, the L2 hit cost the same ≈3s over the theoretical floor — that's
the network movement of ~1.6 GB KV. Consequences:

- **Break-even input length ≈ 10K tokens** (fp8): below it, cold prefill is cheaper than
  the network fetch; above it, L2 wins and the margin grows linearly with input length
  (100K-token doc: save ~22s of ~25s).
- The constant is the store's bandwidth, not lookup: an RDMA-class L2 (Mooncake) exists
  precisely to shrink this constant toward the L1 figure. Untested here.

## PD disaggregation and KV caches — two different timelines

Two experiments that look contradictory but are complementary:

**(i) L1 offload does NOT help the current request in PD.** HiCache on the prefill node,
concurrency 20: 100% host-cache hit, **zero end-to-end gain** (~4%). Reason: every request
still pays the P→D KV transfer + bootstrap handshake, which dominates once prefill compute
is removed. (Also ruled out fp8-JIT-kernel fallback as the cause — bf16 rerun showed the
same null result. bf16 cold rounds ran ~50% slower than fp8, confirming KV-transfer volume
is a first-order PD cost: **fp8 KV is doubly valuable in PD**.)

**(ii) A shared L2 DOES help across requests in PD.** Experiment 2b is exactly the
multi-turn PD scenario: turn 2 arrives at the prefill node, which has no KV for the
history (it lives on the decode node) — vanilla PD recomputes the whole history every
turn. With a shared store, P looks up the previous turns' KV (the [2] path above) and
prefills only the new tokens.

### The decode node's two KV supply paths

In PD, D's KV is always *moved to it* — the movement path is a real performance variable:

| | Path A: NIXL/EFA direct push | Path B: pull from shared store |
|---|---|---|
| Flow | P HBM → EFA RDMA → D HBM | P → store → D (network, 2 hops) |
| Bandwidth | RDMA-class (~400GB/s aggregate) | store-bound (Redis ≈ 0.5GB/s class) |
| Measured cost | part of PD's ITL floor (+9.5ms/token vs single node) and TTFT overhead | ~3s / 30K tokens |
| Coupling | P and D tightly coupled (handshake, both online) | fully decoupled |

**Conclusion**: these are not alternatives — they serve different timelines:

```
current request's fresh KV   → Path A (EFA push)      — irreplaceable for speed
cross-turn / cross-request   → Path B (shared store)  — P skips history recompute
restart / failover           → Path B as backstop     — cache outlives instances
```

"D pulls the *current* request's KV from a shared store" is not performance-viable with a
TCP store (10× slower than EFA); it becomes viable only if the store itself is RDMA-class —
which is Mooncake's thesis, and the most valuable untested hypothesis here.

## Impact summary: TTFT vs throughput

**TTFT**: hit → large win (up to 22×); miss → exactly zero change, and the cold path shows
no measurable overhead from having caches enabled (cold rounds identical with/without).
Expected improvement = `hit-rate × (cold prefill − hit path)`. Traffic with no repeated
prefixes gets nothing.

**Throughput**: no direct effect — ITL/TPOT are untouched by any of this (decode speed is
independent of where prompt KV came from; the PD ITL floor is a *transfer* cost, not a
cache cost). The indirect effect is capacity: every hit frees prefill GPU-seconds for
other requests, ≈ `1/(1 − hit-rate × prefill-share)` extra sustainable concurrency
(e.g. 30% hits × 60% prefill share ≈ +22%). **Not yet measured** — needs a mixed-traffic
soak test (see Open items).

**Deployment guidance by traffic profile**:

| Traffic | Recommendation |
|---|---|
| Unique prompts (no reuse) | Skip all of this; caches idle harmlessly |
| RAG / system prompts / multi-turn, single replica | L1 offload only (vLLM+LMCache 22× or sglang HiCache 8.1×) |
| Same, N replicas or PD | + shared L2 (Redis for >10K-token contexts; Mooncake if RDMA-class needed) |
| Long docs < 10K tokens | L2 doesn't pay; L1 only |

## Operational notes

- **lmcache MP server must share the engine container** — CUDA IPC fails across container
  boundaries (`CUDA error: mapping of buffer object failed` when run as a sidecar). Launch
  pattern: `lmcache server ... & exec vllm serve ...` (see `vllm/glm-5.2-fp8-p5en-vllm.yaml`).
- Externalization layers: engine ↔ MP server (same container; survives engine process
  restarts) → L2 remote (pure network; survives pod death, shareable cross-node).
- sglang-router circuit-breaker does **not** auto-recover after a prefill restart —
  `kubectl rollout restart deploy/<router>`.
- fp8 KV triggers `Unsupported element_size = 656 for JIT HiCache kernel` (generic-path
  fallback). Benefit was still 8× despite it; bf16 avoids the warning but halves both KV
  pools — keep fp8.
- **Capacity sizing**: benefit is binary on `working set ≤ pool`. Size
  `hicache-ratio` / `kv-offloading-size` / Redis `maxmemory` from
  *active users × context length × ~54KB/token (fp8)*. p5en has ~2TB RAM (ratio 4–6 viable).

## Open items

1. **Mixed-traffic soak test** — x% repeated prefixes + real output lengths at c20–40,
   measuring total tok/s and TTFT p99 with/without caches: the missing throughput number.
2. **Mooncake as L2** (RDMA-class store): shrink the 3s network constant; would also test
   whether "D pulls current-request KV from store" becomes viable, enabling fully
   decoupled PD.
3. **SGLang HiCache L3** (`--hicache-storage-backend mooncake|file`) — sglang's equivalent
   externalization path, DSA-indexer-aware; untested.
4. Cache-aware routing (sglang-router prefix affinity) for N-replica topologies.

## Reproduce

```bash
# Redis as external L2 store (any CPU node)
kubectl apply -f - <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata: { name: lmcache-redis, labels: { app: lmcache-redis } }
spec:
  replicas: 1
  selector: { matchLabels: { app: lmcache-redis } }
  template:
    metadata: { labels: { app: lmcache-redis } }
    spec:
      containers:
      - name: redis
        image: public.ecr.aws/docker/library/redis:7-alpine
        args: ["redis-server", "--maxmemory", "48gb", "--maxmemory-policy", "allkeys-lru", "--save", ""]
        ports: [ { containerPort: 6379 } ]
---
apiVersion: v1
kind: Service
metadata: { name: lmcache-redis }
spec:
  selector: { app: lmcache-redis }
  ports: [ { port: 6379, targetPort: 6379 } ]
EOF

# vLLM container command (in-container MP server + L2 adapter):
#   lmcache server --l1-size-gb 100 --eviction-policy LRU \
#     --l2-adapter '{"type":"resp","host":"lmcache-redis.default.svc.cluster.local","port":6379,"max_capacity_gb":40}' &
#   sleep 5
#   exec vllm serve zai-org/GLM-5.2-FP8 ... --kv-offloading-size=100 --kv-offloading-backend=lmcache

# Verify writes:  kubectl exec deploy/lmcache-redis -- redis-cli DBSIZE
# Verify hits:    engine logs "Retrieved N tokens in 0.00Xs"
```
