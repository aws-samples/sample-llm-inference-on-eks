# Kimi-K3 on 3 Nodes: why TP8×PP3, and why 1M context needs the third node

This document explains the *choice of parallelism* for `moonshotai/Kimi-K3` on
three p5en.48xlarge, and the KV-capacity arithmetic behind it. The reasoning was
worked out on 2026-08-06 from the model's `config.json`, vLLM source, and the
official recipe — **before** the shape ran. Measured performance is in
[KIMI-K3-PP3-PERFORMANCE.md](KIMI-K3-PP3-PERFORMANCE.md); § 6 here checks the
predictions against it.

Sources are cited inline so every claim can be re-checked, per this repo's
"never attribute without evidence" rule. Where a figure is derived rather than
measured, it says so.

## 1. 24 GPUs is not one number, it is two axes

TP24 fails on divisibility. But **24 = 8 × 3** splits into two orthogonal axes
whose divisibility requirements are completely different:

| Axis | Shards what | Crosses what | Divisibility requirement |
|---|---|---|---|
| **TP8** | the **width inside each layer** (heads, intermediate dims) | intra-node NVLink | every sharded dim ÷ **8** |
| **PP3** | the **number of layers** (93 split into 3 stages) | inter-node EFA | layer count splits into 3 |

TP24 needs *every* sharded dimension divisible by 24. Split into TP8×PP3, widths
only need ÷8 and layers only need to split 3 ways — both hold comfortably. It also
keeps all TP collectives on NVLink, leaving only pipeline traffic on EFA.

## 2. Why TP24 cannot start at all

Every TP-sharded dimension, checked against
[`config.json`](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)
(values under `text_config`):

| TP-sharded dimension | Value | ÷8 (TP8) | ÷24 (TP24) |
|---|---|---|---|
| attention heads | 96 | ✅ 12/GPU | ✅ 4/GPU |
| KDA heads | 96 | ✅ 12/GPU | ✅ |
| q_proj out — 96×(128+64) | 18,432 | ✅ | ✅ |
| kv_b_proj out — 96×(128+128) | 24,576 | ✅ | ✅ |
| o_proj in — 96×128 | 12,288 | ✅ | ✅ |
| MoE intermediate | 3,072 | ✅ | ✅ |
| dense MLP intermediate | 33,792 | ✅ | ✅ |
| **vocab / lm_head** | **163,840** | ✅ 20,480 | ❌ **remainder 16** |

**Only vocab fails — and one failure is enough.** Traced through vLLM source:

1. `vocab_parallel_embedding.py:266-270` pads vocab via
   `pad_vocab_size(163840, padding_size)`
2. `padding_size` defaults to `DEFAULT_VOCAB_PADDING_SIZE = 64` (`:32`) and **does
   not scale with `tp_size`** — verified across every call site (`:246`, `:546`)
3. 163,840 is already a multiple of 64, so padding leaves it unchanged
4. `:310` then calls `divide(num_embeddings_padded, tp_size)`
5. `divide()` is a hard assert (`vllm/distributed/utils.py:53-64`):
   `assert numerator % denominator == 0`
6. 163,840 % 24 = 16 → `AssertionError` during engine init

At TP8: 163,840 ÷ 8 = 20,480. Clean.

### Three dimensions that look fatal but are not

Worth remembering, to avoid mis-diagnosing this class of problem:

- **`kv_lora_rank = 512`** (512 % 24 = 8) — it is the *input* dim of `kv_b_proj`,
  and `ColumnParallelLinear` shards only the **output** dim. `kv_a_proj_with_mqa`
  is `ReplicatedLinear`, not sharded at all. (Read from `deepseek_v2.py:512-525`,
  same MLA family.)
- **`hidden_size = 7168`** (% 24 = 16) — the residual-stream width, **replicated**
  on every rank, never sharded.
- **`num_experts = 896`** (% 24 = 8) — expert *count* only needs divisibility under
  **Expert Parallel**. TP shards each expert **internally**, and
  `moe_intermediate 3072 ÷ 24 = 128` is fine.

### No upstream 3-node configuration exists

`strategy_min_gpus` from
[recipes.vllm.ai](https://recipes.vllm.ai/moonshotai/Kimi-K3.json) is entirely
multiples of 8 or 16 — `single_node_tp: 8`, `multi_node_tp: 8`, `multi_node_tep: 8`,
`multi_node_tp_pp: 16`, `multi_node_dep: 16`. The 3-node shape is a local
extrapolation of the `multi_node_tp_pp` profile, not a supported recipe.

## 3. 93 layers split better at PP3 than at the recipe's PP2

`num_hidden_layers = 93`:

| PP | 93 ÷ PP | Result |
|---|---|---|
| **PP3** | 31.00 | ✅ **exact**, 31 layers per stage |
| PP2 (the recipe's) | 46.50 | ⚠️ uneven, vLLM must split unevenly |

93 = 3 × 31 is a coincidence, but a favourable one — it is why deviating from the
recipe's 2 nodes was defensible.

### The hybrid layer mix also splits close to evenly

K3 is a Mamba/linear-attention hybrid. Of the 93 layers
(`text_config.linear_attn_config`):

- **24 full-attention (MLA)** layers — `full_attn_layers` = [4, 8, 12, … 88, 92, 93]
- **69 KDA** (Kimi Delta Attention, linear) layers — the rest

Splitting at 31 layers per stage:

| Stage | Layers | full-attn | KDA |
|---|---|---|---|
| 0 (leader) | 1–31 | 7 | 24 |
| 1 | 32–62 | 8 | 23 |
| 2 | 63–93 | **9** | 22 |

full-attn is 7/8/9, not perfectly even — **stage 2 is the binding constraint**
(largest KV demand). Any headroom estimate has to be discounted for it.

## 4. Capacity is counted in blocks, not GiB

1M is not "a bit more VRAM". In vLLM v1 a request draws whole blocks from **every**
kv-cache group and the totals are **summed**
(`kv_cache_utils.py::get_max_concurrency_for_kv_cache_config`, lines 928-935).
K3 being hybrid means **two groups**:

| Group | Layers | Blocks per request |
|---|---|---|
| full-attention (MLA) | 24 | scales with context |
| KDA | 69 | **fixed at 1** — recurrent state, `mamba_cache_mode="none"` (default) |

The KDA figure comes from `MambaSpec.max_memory_usage_bytes`
(`vllm/v1/kv_cache_interface.py:729-737`), which returns `page_size_bytes * 1` in
mode `none`, `* 2` in `align`, and `* cdiv(max_model_len, block_size)` in `all`.
**Only mode `all` grows with context** — worth knowing, because switching modes
(e.g. for LMCache, which requires `align`) changes this arithmetic entirely.

Block size is **384 tokens** here, raised from vLLM's default so the attention page
matches the KDA page. The engine logs the decision at startup:

```
Setting attention block size to 384 tokens to ensure that attention page size is >= mamba page size.
Padding mamba page size by 8.68% to ensure that mamba page size and attention page size are exactly equal.
```

So:

```
blocks/request = cdiv(max_model_len, 384) + 1
     29,696 tokens →    78 + 1 =    79 blocks
  1,048,576 tokens → 2,731 + 1 = 2,732 blocks     (35× more)
```

### 2 nodes: short by 34×, and no parameter closes it

Measured on the 2-node TP16 deployment (2026-08-03): `max_concurrency` was
**1.012×** at `max_model_len` 29,696. Back-solving
`max_concurrency = num_blocks / blocks_per_request`:

```
num_blocks ≈ 1.012 × 79 ≈ 80 blocks in the whole pool
1M needs 2,732 → short by ~34×
```

Weights took 129.75 GiB of each GPU's 143.77 GiB, leaving **0.82 GiB/GPU** for KV.
No parameter recovers 34×.

> **A displayed figure that misleads.** vLLM logs
> `GPU KV cache size: 30,062 tokens`, but that is *derived*, not the pool's physical
> size: `kv_cache_utils.py:2227` computes it as `max_concurrency × max_model_len`.
> Do not divide it by block size to infer the block count — that path gave 78.29,
> not even an integer, which is what exposed the mistake.

### 3 nodes: PP fixes both halves of the arithmetic

*Predicted 2026-08-06, before the shape ran. See § 6 for how it compared.*

| | TP16 (2 nodes, measured) | TP8×PP3 (3 nodes, predicted) |
|---|---|---|
| weights/GPU | 129.75 GiB | ~86.5 GiB *(same total ÷ 24 GPUs)* |
| available/GPU @ util 0.97 | 0.82 GiB | ~53 GiB |
| **layers held per GPU** | **93** (TP does not shard layers) | **31** (PP shards layers) |
| block cost per GPU | ~10.5 MiB | ~7 MiB (layers ×⅓, offset by TP8 giving 2× the heads → net ~0.67×) |
| **blocks in pool** | **~80** | ~7,750 |
| 1M (needs 2,732) | ❌ short 34× | ✅ ~2.8× |

**The decisive row is "layers held per GPU".** TP shards each layer's width, so all
93 layers live on every GPU. PP shards layers, so each GPU holds 31. PP therefore
does two things at once: it thins the weights *and* makes each block cheaper per
GPU. TP24 — even if it could start — would only do the first.

## 5. The hybrid architecture is what makes 1M possible at all

Worth stating separately: **only 24 of 93 layers hold token-proportional KV.** The
other 69 KDA layers keep a fixed-size recurrent state — one block per request,
whether the context is 30K or 1M.

Per-token latent KV for the full-attention layers alone:

```
(kv_lora_rank 512 + qk_rope_head_dim 64) = 576 elems per layer per token
× 24 full-attn layers × 2 B (bf16) = 27,648 B/token = 27 KiB/token
1M tokens → 27 GiB across the whole model
```

If all 93 layers were full-attention that would be ~104 GiB of pure token KV, and 3
nodes would not be enough either. **This is the architectural reason K3 can claim a
1M window** — and also why its block size is forced up to 384 tokens (the KDA page
is larger than one page of attention KV, and both layer types draw from one shared
pool, so the page sizes must be equal).

## 6. How the predictions held up

The shape ran on 2026-08-09. Comparing against
[KIMI-K3-PP3-PERFORMANCE.md](KIMI-K3-PP3-PERFORMANCE.md) § 2:

| Quantity | Predicted (2026-08-06) | Measured (2026-08-09) | Error |
|---|---|---|---|
| blocks in pool | ~7,750 | **16,565** (6,361,156 tokens ÷ 384) | **2.1× under** |
| 1M concurrency | ~2.8×, discounted to ~2.5× for stage 2 | **6.07×** | 2.2–2.4× under |
| weights/GPU | ~86.5 GiB | 68.89 GiB incl. non-torch | ~1.26× over |

**The qualitative conclusion was right and the quantitative estimate was
systematically conservative.** PP3 does clear 1M with room to spare — by a wider
margin than predicted. Two contributors are identifiable, but their relative weight
is **not established**:

- weights came in lighter than the "total ÷ 24 GPUs" assumption, leaving more for KV
- the prediction used `util 0.97` while the manifest ships `0.93`, so the direction
  of that term is *against* the observed surplus, meaning the weight term must
  account for more than all of it

The § 3 warning about stage 2 being the binding stage stands as reasoning, but the
engine's reported pool already reflects whatever the real per-stage split costs, so
the 8/9 discount should **not** be applied on top of a measured pool figure.

One prediction to retire: § 8 of the original analysis expected PP3 to hold
"~2 concurrent 1M requests". The engine reports 6.07×. Note this is still the
engine's static arithmetic — **6 concurrent 1M requests were never actually run**.

## 7. If the engine refuses to allocate KV on first start

vLLM prints an `estimated maximum model length` in the failure message — set
`--max-model-len` just below it. This is what happened on the TP16 shape:

```
ValueError: To serve at least one request with the model's max seq len (32768),
0.88 GiB KV cache is needed, which is larger than the available KV cache memory
(0.82 GiB). Based on the available memory, the estimated maximum model length is 30336.
```

That is why the TP16 manifest sits at 29,696 (1024-aligned, ~2% under the 30,336
ceiling — the profiling figure varies run to run, so do not sit flush against it).

**One more lever in reserve:** `--kv-cache-dtype=fp8` halves the MLA KV bytes per
token, doubling block headroom. The single-node B300 manifest
(`k8s-manifest/vllm/kimi-k3-p6-b300-vllm.yaml`) uses it. It is deliberately **not**
in the PP3 manifest, so that shape differs from the h200 recipe in as few ways as
possible — add it only if measurement shows it is needed. With 6.07× measured at 1M
(§ 6), it has not been needed.

## 8. The PP trade-off

Pipeline parallelism introduces bubbles, so **single-request latency is typically
worse than TP at the same GPU count**; throughput needs enough in-flight requests to
fill the pipeline. The much larger KV budget is what makes that possible here.

Whether that trade paid off cannot be answered from this repo's data. A same-GPU-count
comparison would need TP24, which cannot start (§ 2), so no controlled A/B exists.
The 2-node TP16 baseline differs in GPU count (16 vs 24), date, and `max-num-seqs`.
What *was* measured is that throughput keeps scaling with concurrency well past the
point where per-user latency becomes unacceptable — see
[KIMI-K3-PP3-PERFORMANCE.md](KIMI-K3-PP3-PERFORMANCE.md) § 5.
