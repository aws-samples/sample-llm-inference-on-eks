# Terraform — EKS GPU cluster from scratch

Terraform stack that builds the cluster the [`k8s-manifest/`](../../k8s-manifest)
examples run on: a lean EKS cluster with self-managed Karpenter, the NVIDIA GPU
Operator, and the AWS EFA device plugin.

This is **optional** — it's here in case you're starting from zero. If you
already have a GPU-capable cluster, or want the shortest path (EKS Auto Mode),
see [Creating a Cluster](../../README.md#creating-a-cluster) in the root README
and skip this directory.

System components (Karpenter, CoreDNS) run on **Fargate**, so the cluster boots
with zero EC2 nodes; every worker node is provisioned on demand by Karpenter.

> [!IMPORTANT]
> This stack manages its own GPU NodePool, named `gpu` — see
> [NodePool ownership](#nodepool-ownership). Do **not** `kubectl apply -f k8s-manifest/infra/nodepool.yaml`
> on a cluster built here: that file is the EKS Auto Mode alternative and would
> add a second, redundant GPU NodePool pointing at a NodeClass this stack
> doesn't create.

## What this stack creates

| File | Contents |
|---|---|
| `main.tf` | Terraform/provider versions, S3 backend, AWS/kubernetes/kubectl/helm providers, shared `data` + `locals` |
| `variables.tf` | Inputs — region, VPC CIDR, cluster name/version, GPU / EFA / LWS toggles, ODCR flags |
| `vpc.tf` | VPC (`10.6.0.0/16`, 4 AZs, single NAT), Karpenter + CNI discovery tags on private subnets |
| `vpc-endpoint.tf` | S3 Gateway endpoint + ECR API/DKR Interface endpoints + endpoint SG |
| `eks.tf` | EKS module v21, Fargate profiles, core EKS add-ons, `gp3` default StorageClass |
| `karpenter.tf` | Karpenter module + Helm releases, EC2NodeClasses / NodePools / FlowSchemas |
| `addons.tf` | `eks-blueprints-addons` (EBS CSI, metrics-server, node monitoring agent), Pod Identity associations, NVIDIA GPU Operator, AWS EFA device plugin, LeaderWorkerSet controller |
| `litellm-langfuse.tf` | Opt-in LiteLLM gateway + Langfuse observability, generated secrets, LiteLLM's Bedrock Pod Identity |
| `kubernetes/` | Helm value overrides and raw manifests consumed by the above |
| `cleanup.sh` | Ordered `terraform destroy` (nodes → add-ons → EKS → everything else) + orphaned ELB SG cleanup |

### Cluster (`eks.tf`)

- EKS **1.35** via `terraform-aws-modules/eks/aws ~> 21.15`, public endpoint enabled, cluster-creator admin access
- Fargate profiles for the `karpenter` namespace and for CoreDNS (`kube-system`, `k8s-app=kube-dns`)
- `create_security_group = false` / `create_node_security_group = false` — Fargate uses the cluster primary SG
- Managed add-ons:
  - `vpc-cni` — prefix delegation (`ENABLE_PREFIX_DELEGATION=true`, `WARM_PREFIX_TARGET=1`), installed `before_compute`
  - `coredns` — pinned to Fargate, autoscaling 2–10 replicas, custom Corefile
  - `kube-proxy` — IPVS mode, round-robin scheduler
  - `eks-pod-identity-agent` — installed `before_compute`
- Default `gp2` StorageClass is un-defaulted; a `gp3` StorageClass (`WaitForFirstConsumer`) becomes the default

### Networking (`vpc.tf`, `vpc-endpoint.tf`)

- 4 AZs — `us-west-2d` is deliberately included because `p5en` / `p6-b200` on-demand capacity is often only available there
- Private subnets: 4 × `/20` (~4091 usable IPs each); public: 4 × `/24` offset to `10.6.64.0/24`
- Private subnets tagged `karpenter.sh/discovery`, `kubernetes.io/role/internal-elb`, and `kubernetes.io/role/cni` (VPC CNI enhanced subnet discovery)
- S3 Gateway + ECR Interface endpoints keep image pulls and S3 traffic off the NAT gateway

### Karpenter (`karpenter.tf`)

- Karpenter **v1.12.0** (`karpenter-crd` + `karpenter` charts from `public.ecr.aws`)
- Controller IAM uses **IRSA**, not Pod Identity — Karpenter runs on Fargate, which doesn't support Pod Identity. The v21 module dropped its IRSA toggle, so the trust policy is injected via `iam_role_source_assume_policy_documents`.
- A `time_sleep` guard waits 60s after the Fargate profiles report ready; installing the chart earlier yields pods stuck on `default-scheduler`
- `nodeRepair` feature gate is **off** — it misjudged long model loading (DeepGEMM JIT compilation) as an unhealthy node and killed p5en nodes ~13 min after launch
- Two `FlowSchema`s give the controller and its leader election dedicated API priority

#### NodePools

| NodePool | NodeClass | Instances | Capacity types | Notes |
|---|---|---|---|---|
| `default` | `default` | `c` / `m` / `r`, >4 vCPU, Nitro, gen >2 | on-demand, spot | 500Gi gp3, RAID0 instance store, consolidate `WhenEmptyOrUnderutilized` |
| `gpu` | `gpu` | g6, g6e, g7e, p4/p4d/p4de, p5, p5en, p6-b200, p6-b300 | on-demand, spot | `nvidia.com/gpu` taint, 500Gi gp3, `expireAfter: 720h`, cpu limit 5000 |
| `reserved-capacity-pool` | `reserved-capacity` | p4/p4d/p4de, p5, p5en, p6-b200, p6-b300 | reserved (priority), on-demand | Opt-in, see [ODCR](#capacity-reservation-odcr--opt-in). `weight: 10`, 800Gi gp3 (3000 IOPS / 150 MB/s) |

All NodeClasses use `amiSelectorTerms: alias: al2023@latest`. The alias expands to every AL2023 variant, each carrying arch and GPU-count requirements, so GPU instance types resolve to `amazon-eks-node-al2023-<arch>-nvidia` on their own — no separate GPU AMI selector needed. IMDSv2 is enforced (`httpTokens: required`) on the GPU and reserved-capacity classes.

Instance-store NVMe is RAID0'd (`instanceStorePolicy: RAID0`), which is what puts the model cache path the manifests use (`/mnt/k8s-disks/0/models/`) on local disk.

### Add-ons (`addons.tf`)

Installed by default:

- **EKS managed add-ons** via `eks-blueprints-addons` (pinned to `1.23.0`): `aws-ebs-csi-driver`, `metrics-server`, `eks-node-monitoring-agent`
- **EKS Pod Identity** associations for `ebs-csi-controller-sa` and the CloudWatch agent
- **NVIDIA GPU Operator** `v26.3.0` — gated by `enable_gpu_operator`. Driver and `nvidia-container-toolkit` are **disabled** because the AL2023 NVIDIA AMI ships them; the Operator only provides the device plugin + GPU Feature Discovery. It depends on `kubectl_manifest.karpenter_node_pool` because its pods need a real (non-Fargate) node.
- **AWS EFA device plugin** `v0.5.30` — gated by `enable_aws_efa_device_plugin`, required by the multi-node [`lws/`](../../k8s-manifest/lws) examples. No affinity override: the chart's own `supportedInstanceLabels` already enumerates every EFA-capable instance type.
- **LeaderWorkerSet (LWS)** `v0.9.0` — gated by `enable_lws` (**on** by default). Installs the `LeaderWorkerSet` CRD + controller into `lws-system` from `oci://registry.k8s.io/lws/charts`; this is the controller the multi-node [`lws/`](../../k8s-manifest/lws) examples need, so a cluster built here can apply them directly. Chart defaults are kept: internal cert management (no cert-manager in this stack), `enablePrometheus=false`, and no tolerations, so the controller lands on the `default` NodePool instead of a tainted GPU node. Like the GPU Operator it depends on `kubectl_manifest.karpenter_node_pool` — the controller requests 1 CPU / 1Gi and there is no Fargate profile for `lws-system`.

Opt-in via variable:

- **LiteLLM + Langfuse** — gated by `enable_litellm_langfuse` (**off** by default). See [below](#litellm--langfuse--opt-in).

Wired but **off** (flip the flag in `addons.tf` to enable):

- `enable_aws_load_balancer_controller = false` — chart `3.1.0`, `enableServiceMutatorWebhook=false` (the webhook deadlocks a from-scratch apply: its own pods are Pending with no nodes yet, so every Service creation, including CoreDNS', is rejected)
- `enable_kube_prometheus_stack = false` — value overrides live in `kubernetes/kube-prometheus-stack/values.override.yaml` (EKS-unreachable components disabled, Karpenter scrape config + dashboards, 50Gi EBS for Prometheus, Slack alerting via `var.slack_api_url`)

> The files under `kubernetes/*/values*override.yaml` are **overrides only** —
> Helm merges them onto the upstream chart defaults. Each links to the upstream
> `values.yaml` it layers on top of; consult that for anything not listed.

Not installed: the [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws)
controller, which the `lws/` manifests need. Install it separately per the
[upstream instructions](https://github.com/kubernetes-sigs/lws#installation).

## Prerequisites

- Terraform >= 1.10 (for `use_lockfile` S3 state locking; >= 1.3 is enough with local state)
- AWS CLI configured with a profile that can create VPC/EKS/IAM resources
- `kubectl`, `helm`
- An S3 bucket for remote state (see [Backend](#backend)) — or drop the backend block and use local state
- GPU service quotas in the target region — a fresh account is often `0`; see the
  [quota checks](../../README.md#creating-a-cluster) in the root README
- (Optional) An EC2 On-Demand Capacity Reservation or Capacity Block, if you enable ODCR

## Backend

`main.tf` uses a partial backend config so no bucket name is committed:

```bash
cp backend.hcl.example backend.hcl   # git-ignored; fill in your bucket
terraform init -backend-config=backend.hcl
```

The bucket must already exist — Terraform will not create it. For a throwaway
single-user cluster you can delete the `backend "s3" {}` block from `main.tf`
and use local state (`terraform.tfstate` is git-ignored).

## Configuration

```bash
cp dev.auto.tfvars.example dev.auto.tfvars   # git-ignored
```

| Variable | Default | Purpose |
|---|---|---|
| `aws_profile` | `default` | AWS CLI profile for the providers |
| `region` | `us-west-2` | Region for all resources |
| `vpc_cidr` | `10.6.0.0/16` | VPC CIDR |
| `cluster_name` | `llm-inference` | Cluster name; also the Karpenter node IAM role name and the `karpenter.sh/discovery` tag value |
| `cluster_version` | `1.35` | Kubernetes version |
| `enable_gpu_operator` | `false` | Install the NVIDIA GPU Operator — **set `true` for GPU serving** |
| `enable_aws_efa_device_plugin` | `false` | Install the AWS EFA device plugin — **set `true` for the multi-node `lws/` examples** |
| `enable_lws` | `true` | Install the LeaderWorkerSet controller (needed by the multi-node `lws/` examples) |
| `enable_litellm_langfuse` | `false` | Install the [LiteLLM gateway + Langfuse](#litellm--langfuse--opt-in) stack |
| `enable_capacity_reservation` | `false` | Provision the ODCR NodeClass + NodePool |
| `capacity_reservation_id` | `null` | Required when `enable_capacity_reservation = true` |
| `slack_api_url` | placeholder | Alertmanager Slack webhook (only used when kube-prometheus-stack is enabled) |

The S3 backend is not covered by `aws_profile`, so pass the profile as an env var:

```bash
AWS_PROFILE=<profile> terraform init -backend-config=backend.hcl
AWS_PROFILE=<profile> terraform plan -out=planfile
AWS_PROFILE=<profile> terraform apply planfile
```

Expect roughly 20 minutes for a from-scratch apply. Then wire up kubectl (the
exact command is emitted as the `configure_kubectl` output):

```bash
$(terraform output -raw configure_kubectl)
```

Verify before deploying a model:

```bash
kubectl get nodepools                          # default, gpu
kubectl get pods -n karpenter                  # Running, on Fargate
kubectl get pods -n gpu-operator               # device plugin + GFD
kubectl get ds -n kube-system aws-efa-k8s-device-plugin
kubectl get deploy -n lws-system lws-controller-manager   # multi-node lws/ examples
```

Note that `gpu-operator` pods stay Pending until the first GPU node exists —
that's expected on an idle cluster.

Now continue to the root [Quick Start](../../README.md#quick-start), skipping
`kubectl apply -f k8s-manifest/infra/nodepool.yaml` (this stack provides its own
GPU NodePool).

## NodePool ownership

This stack and `k8s-manifest/infra/nodepool.yaml` are two alternative ways to get
a GPU NodePool, one per cluster style:

| | this stack | `k8s-manifest/infra/nodepool.yaml` |
|---|---|---|
| Karpenter | self-managed (Helm, on Fargate) | bundled with EKS Auto Mode |
| NodePool name | `gpu` | `gpu-nodepool` |
| NodeClass ref | `karpenter.k8s.aws/EC2NodeClass` → `gpu` | `eks.amazonaws.com/NodeClass` → `default` |
| GPU taint | `nvidia.com/gpu=true:NoSchedule` | `nvidia.com/gpu=Exists:NoSchedule` |

The names deliberately differ, so applying the Auto Mode file here can't clobber
this stack's NodePool — but it still isn't useful: `eks.amazonaws.com/NodeClass`
is an Auto Mode API this cluster doesn't serve, so the extra NodePool would sit
permanently unready. Pick one path.

The model manifests themselves are portable across both: every GPU toleration in
`k8s-manifest/` uses `operator: Exists`, which ignores the taint value.

## Capacity Reservation (ODCR) — opt-in

`enable_capacity_reservation` gates two resources, both filtered out of their
`for_each` in `karpenter.tf` when it's `false`:

1. `EC2NodeClass/reserved-capacity` (with `capacityReservationSelectorTerms`)
2. `NodePool/reserved-capacity-pool`

`karpenter.tf` wraps the ID in `coalesce(var.capacity_reservation_id, "unused")`
so template rendering doesn't break when the flag is off.

```hcl
enable_capacity_reservation = true
capacity_reservation_id     = "cr-xxxxxxxxxxxxxxxxx"
```

The pool prefers `reserved` capacity and falls back to `on-demand`. Schedule onto
it with the GPU toleration, optionally pinning the capacity type:

```yaml
spec:
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  nodeSelector:
    karpenter.sh/capacity-type: reserved   # optional: force reserved capacity
```

> [!WARNING]
> Capacity Blocks bill for the entire window whether or not you use them, and AWS
> reclaims the instance at `EndDate` with no grace period. For a short-lived
> block, consider managing its NodeClass/NodePool with `kubectl` outside
> Terraform so you can tear it down without touching state.

## LiteLLM + Langfuse — opt-in

`enable_litellm_langfuse = true` installs a [LiteLLM](https://docs.litellm.ai/)
gateway in front of your models and [Langfuse](https://langfuse.com/) for
tracing, both in their own namespaces. Off by default — nothing in
`k8s-manifest/` requires them.

| | Chart | Version | Namespace |
|---|---|---|---|
| LiteLLM | `oci://ghcr.io/berriai/litellm-helm` | `1.93.0` | `litellm` |
| Langfuse | `https://langfuse.github.io/langfuse-k8s` | `1.5.39` | `langfuse` |

What it gives you:

- **One OpenAI-compatible endpoint** for both the self-hosted models in
  [`k8s-manifest/`](../../k8s-manifest) and Bedrock, with virtual keys, per-key
  spend tracking and rate limits.
- **Bedrock access without static credentials** — an EKS Pod Identity
  association binds the `litellm` ServiceAccount to a role allowing
  `bedrock:InvokeModel[WithResponseStream]`, scoped to `anthropic.*` foundation
  models and inference profiles. boto3's default chain picks it up.
- **Every call traced in Langfuse** — LiteLLM's `success_callback` /
  `failure_callback` point at the in-cluster Langfuse. The project API keys are
  generated by Terraform and seeded into Langfuse with `LANGFUSE_INIT_*`, so the
  wiring works on first boot with no UI steps.

### Access

Both are **ClusterIP only**, matching every Service in `k8s-manifest/`. This
stack ships with `enable_aws_load_balancer_controller = false` and no
external-dns, so an Ingress would never get an address. Use port-forward:

```bash
# LiteLLM gateway (OpenAI-compatible)
kubectl port-forward -n litellm svc/litellm 4000:4000
terraform output -raw litellm_master_key    # the Bearer token / api_key

curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $(terraform output -raw litellm_master_key)" \
  -H 'Content-Type: application/json' \
  -d '{"model":"claude-sonnet-5","messages":[{"role":"user","content":"hi"}]}'

# Langfuse UI — traces land under the seeded "Default Project"
kubectl port-forward -n langfuse svc/langfuse-web 3000:3000
terraform output -json langfuse_admin_login
```

To expose either publicly instead, enable the ALB controller, set
`ingress.enabled: true` in the relevant `kubernetes/*/values.yaml`, and for
Langfuse also set `langfuse.nextauth.url` to the external URL — NextAuth
callbacks break if it doesn't match the URL you actually browse to.

### Configuring models

`kubernetes/litellm/values.yaml` holds the `proxy_config.model_list`. The
self-hosted entry there is an **example** matching
`k8s-manifest/vllm/glm-5.2-fp8-p5en-vllm.yaml` (Service `glm-5-2-vllm`,
`--served-model-name=zai-org/GLM-5.2-FP8`, `default` namespace) — edit it for
whatever you actually deployed:

```yaml
- model_name: <name-clients-use>
  litellm_params:
    model: openai/<the --served-model-name value>
    api_base: http://<service>.<namespace>.svc.cluster.local/v1
    api_key: dummy   # this repo's manifests serve without auth
```

An entry for a model you haven't deployed only fails when that model is called,
not at startup. Bedrock model IDs must exist in your region — check with
`aws bedrock list-inference-profiles --region <region>`. If you add a non-Anthropic
provider, widen the IAM `resources` in `litellm-langfuse.tf` to match.

### Caveats

- **Secrets are generated by Terraform and stored in state in plaintext**
  (master key, Postgres/ClickHouse/Redis/MinIO passwords, Langfuse keys). The
  state already holds the cluster's IAM wiring, so this is consistent with the
  rest of the stack — but if your state isn't treated as sensitive, move them to
  Secrets Manager and use the charts' `existingSecret` / `secretKeyRef` options.
- **Not production-sized.** Both charts run their bundled datastores: LiteLLM a
  single Postgres, Langfuse a single-node ClickHouse (no keeper quorum),
  Postgres, Redis and MinIO. That's ~5 StatefulSets claiming PVCs on `gp3`.
  Use external managed datastores for anything real.
- Both use `wait = false` — the ClickHouse/Postgres StatefulSets and LiteLLM's
  Prisma migration Job take minutes and nothing in the apply sequences behind
  them. `terraform apply` returning does not mean the pods are Ready.
- Both depend on `kubectl_manifest.karpenter_node_pool` and the `gp3`
  StorageClass: there is no Fargate profile for these namespaces, and the PVCs
  specify no `storageClassName`.

## Operations

```bash
# Karpenter state
kubectl get nodepools
kubectl get ec2nodeclasses
kubectl logs -f -n karpenter -l app.kubernetes.io/name=karpenter

# GPU nodes and advertised GPUs
kubectl get nodes -L karpenter.sh/nodepool -L node.kubernetes.io/instance-type
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'

# GPU Operator / EFA
kubectl get pods -n gpu-operator
kubectl get ds -n kube-system aws-efa-k8s-device-plugin

# LeaderWorkerSet
kubectl get deploy -n lws-system lws-controller-manager
kubectl get leaderworkersets -A

# LiteLLM / Langfuse (when enable_litellm_langfuse = true)
kubectl get pods -n litellm
kubectl get pods -n langfuse
kubectl logs -n litellm -l app.kubernetes.io/name=litellm

# Check an ODCR before enabling it
aws ec2 describe-capacity-reservations --capacity-reservation-ids cr-xxxxxxxxxxxxxxxxx
```

## Teardown

```bash
AWS_PROFILE=<profile> ./cleanup.sh
```

Order matters, which is why this isn't a bare `terraform destroy`: the script
deletes cluster Ingresses, any LeaderWorkerSets, the LiteLLM/Langfuse releases
and their namespaces (their PVCs must be deleted while the EBS CSI controller
still has a node to run on, or the volumes leak), and the Karpenter NodePools (so
nodes drain first), destroys `module.eks_blueprints_addons` then `module.eks`,
deletes ELB-controller-created security groups left behind, then destroys the
remainder.
Delete your model workloads first so nothing blocks the drain.

## Gotchas

- `terraform show planfile` fails with "Variables not allowed" on this stack — use `terraform plan -no-color > /tmp/plan.txt` and grep instead.
- The `helm` provider is pinned to `< 3.0.0` (v3 turned the provider's `kubernetes` block into an attribute), which in turn pins `eks-blueprints-addons` to `1.23.0`.
- `eks-blueprints-addons` emits deprecation warnings about `data.aws_region.current.name` — upstream issue, ignore.
- `dev.auto.tfvars` values must be quoted (`aws_profile = "default"`, not `aws_profile = default`).
- Several add-ons are wired but disabled (`enable_* = false`) rather than deleted — their presence in `addons.tf` is not evidence they're running.
