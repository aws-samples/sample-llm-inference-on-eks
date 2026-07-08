# Multi-Node Serving: LeaderWorkerSet + EFA

SGLang across multiple P5/P5en nodes using
[LeaderWorkerSet](https://github.com/kubernetes-sigs/lws) (LWS), with GPU-to-GPU
communication over AWS [EFA](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html).
Two topologies:

- **Multi-node tensor parallelism** (`lws-*-tp16-*.yaml`) — one model sharded
  TP=16 across 2 nodes; NCCL collectives run over EFA RDMA
- **Prefill/decode disaggregation** (`lws-*-pd-*.yaml`) — separate prefill and
  decode clusters; KV cache streams prefill→decode via NIXL over EFA, fronted by
  `sglang-router`

When to pick which (with benchmark data): [GLM-5.2-BENCHMARK.md](../../docs/GLM-5.2-BENCHMARK.md).
PD architecture deep-dive: [PD_DISAGGREGATION.md](../../docs/PD_DISAGGREGATION.md).

## Prerequisites

- EKS cluster with p5/p5en capacity (manifests pin
  `node.kubernetes.io/instance-type` per file)
- LWS controller installed
- EFA device plugin (`aws-efa-k8s-device-plugin`) — pods request
  `vpc.amazonaws.com/efa: 16`
- An EFA-enabled SGLang image in your registry (below)

## Images (`Dockerfile.efa-*`)

The stock `lmsysorg/sglang` image has no EFA userspace — cross-node NCCL would
silently fall back to TCP over eth0. These Dockerfiles add libfabric/EFA and the
aws-ofi-nccl plugin on top of pinned SGLang versions. Naming:
`Dockerfile.efa[-nixl]-sglang-<version>`; `-nixl-` variants add NIXL for PD KV
transfer.

| Dockerfile | SGLang | Adds | For |
|---|---|---|---|
| `Dockerfile.efa-sglang-0513` | v0.5.13.post1 | EFA 1.49 + aws-ofi-nccl (base wheel already ships NIXL w/ LIBFABRIC) | **Current** — GLM-5.2 manifests (TP16 and PD) |
| `Dockerfile.efa-nixl-sglang-0510` | v0.5.10 | EFA + aws-ofi-nccl + NIXL from source | DeepSeek-V3.2 PD manifests |
| `Dockerfile.efa-nixl-sglang-057` | v0.5.7 | EFA + aws-ofi-nccl + NIXL | archived |
| `Dockerfile.efa-sglang-057` | v0.5.7 | EFA + aws-ofi-nccl | archived |

> Lesson encoded in `Dockerfile.efa-sglang-0513`: sglang ≥0.5.13 base images
> already bundle a NIXL wheel with the LIBFABRIC backend — building NIXL from
> source on top installs a second `libnixl` and breaks imports (ABI conflict).
> Only add EFA userspace + aws-ofi-nccl; don't rebuild NIXL.

Build & push:

```bash
cd k8s-manifest/lws
docker build -f Dockerfile.efa-sglang-0513 -t sglang-efa-p5:v0.5.13.post1-efa .
docker tag  sglang-efa-p5:v0.5.13.post1-efa <account>.dkr.ecr.<region>.amazonaws.com/sglang-efa-p5:v0.5.13.post1-efa
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker push <account>.dkr.ecr.<region>.amazonaws.com/sglang-efa-p5:v0.5.13.post1-efa
```

Then update the `image:` in the manifest you deploy.

## Manifests

| File | Model | Topology | Status |
|---|---|---|---|
| `lws-glm-5.2-tp16-p5en.yaml` | GLM-5.2-FP8 | TP16, 2× p5en | ✅ current |
| `lws-glm-5.2-pd-p5en.yaml` | GLM-5.2-FP8 | 1P+1D + router, 2× p5en | ✅ current |
| `lws-deepseek-v3.2-tp16-p5.yaml` | DeepSeek-V3.2 | TP16, 2× p5 | ✅ |
| `lws-deepseek-v3.2-pd-p5en.yaml` | DeepSeek-V3.2 | 1P+1D + router, 2× p5en | ✅ |
| `lws-deepseek-v3.2-pd-p5.yaml` | DeepSeek-V3.2 | 1P+1D, 2× p5 | 📦 older variant |
| `lws-deepseek-r1-tp16-p5en.yaml` | DeepSeek-R1 | TP16, 2× p5en | 📦 |

Deploy:

```bash
kubectl apply -f lws-glm-5.2-tp16-p5en.yaml
# pods: <name>-0 (leader), <name>-0-1 (worker); the Service targets the leader
kubectl get pods -l leaderworkerset.sigs.k8s.io/name=lws-glm-5-2-sglang -w
```

First start on fresh nodes downloads the full weights (~756 GB for GLM-5.2) to
node-local NVMe — expect 20–40 min. Restarts on the same nodes reuse the cache
and come up in minutes.

## Key Configuration

### EFA / NCCL (both topologies)

| Setting | Value | Why |
|---|---|---|
| `FI_PROVIDER=efa`, `FI_EFA_USE_DEVICE_RDMA=1` | env | libfabric uses EFA with RDMA |
| `NCCL_NET_PLUGIN=ofi`, `NCCL_TUNER_PLUGIN=ofi` | env | NCCL routes through aws-ofi-nccl |
| `vpc.amazonaws.com/efa: "16"` | resources | all 16 EFA devices per p5en node |
| `/dev/infiniband` hostPath + `privileged` + `IPC_LOCK` | volume / securityContext | EFA device access |
| `--dist-init-addr=$(LWS_LEADER_ADDRESS)` `--nnodes=$(LWS_GROUP_SIZE)` `--node-rank=$(LWS_WORKER_INDEX)` | args | LWS wires the TP group |
| `--disable-custom-all-reduce` | args | use NCCL collectives (required multi-node) |

### Memory (hard-won)

Use `--mem-fraction-static=0.80` on H200 under heavy 8K-input load. The static
pool is a **percentage of total VRAM** — going multi-node does not create
activation headroom (TP16's per-GPU weight savings get absorbed into the KV
pool). 0.85 OOM-crashed both TP8 and TP16 under benchmark load. Do **not** set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` with TP — custom all-reduce
cannot IPC-register VMM allocations and CUDA graph capture fails at startup.

### PD-specific

- Prefill and decode are separate LWS resources + Services; the `sglang-router`
  Deployment fronts them (`--pd-disaggregation --prefill=... --decode=...`)
- KV transfer: `--disaggregation-transfer-backend=nixl` (LIBFABRIC/EFA)
- Both clusters must land in the **same AZ** for EFA RDMA. Do NOT use the LWS
  `exclusive-topology` annotation to arrange this — the controller translates it
  into podAntiAffinity that actively rejects the other group's zone. Use
  explicit podAffinity on the decode spec (see manifest comments).
- Size the P:D ratio to your traffic: 1P:1D inverts on prefill-heavy workloads
  (benchmarked at 8K-in/1K-out: prefill saturates while decode idles — see the
  benchmark doc)

## Verify EFA Is Actually Used

```bash
# devices visible in the pod
kubectl exec <pod> -- ls /dev/infiniband/

# libfabric sees the EFA provider
kubectl exec <pod> -- fi_info -p efa | head

# NCCL chose EFA (look for Libfabric/GDRDMA, NOT "via NET/Socket")
kubectl logs <leader-pod> | grep -E 'via NET/(Libfabric|Socket)' | head
```

`NCCL WARN Failed to initialize GDRCopy` alongside working `Libfabric/GDRDMA`
channels is benign.

## Troubleshooting

| Symptom | Check |
|---|---|
| Pods Pending, `Insufficient vpc.amazonaws.com/efa` | EFA device plugin DaemonSet running (`kubectl get ds -n kube-system aws-efa-k8s-device-plugin`)? Node actually p5/p5en? |
| NCCL falls back to `NET/Socket` | Image built with aws-ofi-nccl? `NCCL_NET_PLUGIN=ofi` set? `/dev/infiniband` mounted? |
| Decode pods Pending with `zone DoesNotExist` (PD) | Normal while prefill is still binding — scheduler retries after prefill lands |
| OOM under load | `--mem-fraction-static` too high — see Memory above |
| NIXL / KV-transfer errors (PD) | `fi_info -p efa` inside the pod; `kubectl logs <pod> \| grep -i nixl` |
| Router returns 200s but client SSE parse errors at high concurrency | genai-perf 0.0.16 SSE bug, not the backend — use `sglang.bench_serving` ([benchmark-commands.md](../../docs/benchmark-commands.md)) |

## References

- [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws)
- [SGLang PD Disaggregation](https://docs.sglang.io/advanced_features/pd_disaggregation.html)
- [NIXL](https://github.com/ai-dynamo/nixl)
- [AWS EFA](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl)
