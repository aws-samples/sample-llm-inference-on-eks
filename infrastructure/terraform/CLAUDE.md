# infrastructure/terraform — Working Conventions

Guidance for AI assistants (and humans) working on the Terraform stack. Scoped to
this directory; repo-wide conventions are in the root [CLAUDE.md](../../CLAUDE.md).

Terraform stack for a lean EKS cluster hosting the GPU serving examples in
`k8s-manifest/`. System components (Karpenter, CoreDNS) run on Fargate; every
worker node is provisioned by Karpenter.

> Never write concrete environment values (account ID, AWS profile, state bucket,
> capacity reservation IDs) into tracked files. `backend.hcl` and
> `dev.auto.tfvars` are git-ignored; the tracked `*.example` files are templates.

## How to run Terraform here

- **Workflow**: always `terraform plan -out=planfile` then `terraform apply planfile` — never bare `terraform apply`.
- **Invocation**: pass the profile as an env var (`AWS_PROFILE=<profile> terraform <cmd>`). `aws_profile` in `dev.auto.tfvars` only reaches the providers; the S3 backend needs the env var.
- **Init**: the backend is a partial config — `terraform init -backend-config=backend.hcl`.
- **Teardown** goes through `cleanup.sh`, not a bare `terraform destroy`. Ordering matters: NodePools and Ingresses first, then `module.eks_blueprints_addons`, then `module.eks`, then the rest.
- Run `terraform fmt -recursive` before committing; CI checks it.

## File map

- `main.tf` — provider/version constraints, partial S3 backend, shared `data` + `locals`
- `variables.tf` — all inputs and their defaults
- `vpc.tf` — VPC module; discovery tags on private subnets
- `vpc-endpoint.tf` — S3 Gateway + ECR API/DKR Interface endpoints + endpoint SG
- `eks.tf` — EKS module, Fargate profiles, core managed add-ons, `gp3` default StorageClass, outputs
- `karpenter.tf` — Karpenter module + Helm releases, manifest loaders for `kubernetes/karpenter/{node-classes,node-pools,flow-schemas}/`
- `addons.tf` — `eks-blueprints-addons`, Pod Identity associations, GPU Operator, EFA device plugin, LWS controller
- `litellm-langfuse.tf` — opt-in LiteLLM and Langfuse (`enable_litellm` / `enable_langfuse`, both default `false`, independent), their generated secrets, LiteLLM's Bedrock Pod Identity
- `kubernetes/` — Helm value **overrides** (merged onto upstream chart defaults, not full copies) and raw manifests consumed by the above
- `cleanup.sh` — ordered destroy + cleanup of ELB-controller-created security groups

## Architectural invariants — don't "fix" these

- **IRSA for Karpenter, Pod Identity for everything else.** Karpenter runs on Fargate, which does not support Pod Identity. The v21 Karpenter module dropped its IRSA toggle, so the trust policy is injected via `iam_role_source_assume_policy_documents`.
- **`create_security_group = false` / `create_node_security_group = false`** on the EKS module — Fargate uses the cluster primary SG, so the module-managed node SG is deliberately skipped. Karpenter discovers the primary SG by its `karpenter.sh/discovery` tag.
- **`time_sleep` before the Karpenter chart.** A Fargate profile reports created well before its mutating admission path works; installing earlier leaves pods on `default-scheduler`, Pending forever, recreatable only by hand. Depending on the profiles alone is not enough.
- **`nodeRepair` feature gate stays off.** It misread long model loading (JIT compilation) as an unhealthy node and terminated GPU nodes minutes after launch.
- **`enableServiceMutatorWebhook=false`** on the ALB controller. The webhook has `failurePolicy=Fail` and an empty namespaceSelector, so on a from-scratch apply — when its own pods are Pending with no nodes yet — it rejects every Service creation including CoreDNS', deadlocking the bootstrap.
- **`amiSelectorTerms: alias: al2023@latest`** in every EC2NodeClass. The alias expands to all AL2023 variants with their arch/GPU-count requirements, so GPU instance types resolve to the NVIDIA variant on their own. Don't hand-pin GPU AMIs.
- **GPU Operator runs with `driver.enabled=false` and `toolkit.enabled=false`** — the AL2023 NVIDIA AMI already ships both, and a containerized driver collides with the baked-in one. The Operator only provides the device plugin + feature discovery.
- **GPU Operator depends on `kubectl_manifest.karpenter_node_pool`.** Its pods need a real (non-Fargate) node, which only exists once Karpenter can provision one; installing earlier makes `wait = true` time out and marks the release failed.
- **LWS depends on `kubectl_manifest.karpenter_node_pool`, same as the GPU Operator.** Its controller-manager requests 1 CPU / 1Gi and there is no Fargate profile for `lws-system`, so `wait = true` times out without a Karpenter-provisioned node. It runs with chart defaults — internal cert management (no cert-manager in this stack), `enablePrometheus=false` (the ServiceMonitor CRD isn't installed), and no tolerations so it stays off tainted GPU nodes. `enable_lws` defaults to `true` because the `k8s-manifest/lws/` examples don't work without it.
- **EFA device plugin gets no affinity override.** The chart's own `supportedInstanceLabels` tracks every EFA-capable instance type; a hand-maintained subset silently omits new families and wrongly includes small GPU sizes that have no EFA.
- **VPC endpoints are intentional** (S3 Gateway + ECR Interface) so private-subnet pods don't pay NAT egress for image pulls and S3.
- **Extra AZ beyond the usual three** — the newest GPU on-demand capacity is often only available in the fourth AZ. Public subnet CIDRs are offset to stay clear of the wider private ranges; changing the AZ count means re-checking that offset.
- **`instanceStorePolicy: RAID0`** on the GPU NodeClasses is what makes the manifests' `hostPath: /mnt/k8s-disks/0/models/` model cache land on local NVMe. Removing it silently pushes ~750 GB model downloads onto the EBS root volume.
- **LiteLLM and Langfuse are ClusterIP with `ingress.enabled: false`.** Not an oversight: this stack has `enable_aws_load_balancer_controller = false` and no external-dns, so an Ingress would never get an address. The upstream reference config these were ported from used an ALB + ACM wildcard cert + `var.dns_domain`; none of that exists here. Don't re-add the Ingress without also enabling the ALB controller, and if you do, `langfuse.nextauth.url` must match the external URL or NextAuth callbacks break.
- **`enable_litellm` and `enable_langfuse` are deliberately separate flags, and the integration between them is conditional.** Each half is useful alone (gateway without tracing; tracing backend for your own SDK-instrumented app), so neither may assume the other exists. The LiteLLM `success_callback`/`failure_callback` + `LANGFUSE_*` env block and Langfuse's `LANGFUSE_INIT_PROJECT_*` key seeding are all wrapped in `%{ if }` template directives driven by `local.litellm_langfuse_wired` — don't unconditionally re-add them, or a LiteLLM-only deploy logs a callback error on every request. The shared `random_uuid` key pair is likewise only created when both are on.
- **`cleanup.sh` destroys the LiteLLM/Langfuse releases and namespaces before `kubectl delete nodepool`.** Helm leaves StatefulSet PVCs behind, and the EBS CSI controller that deletes the gp3 volumes runs on a Karpenter node — reorder this after the NodePool deletion and the volumes leak as orphaned EBS.

## Relationship to `k8s-manifest/`

`k8s-manifest/infra/nodepool.yaml` is the **EKS Auto Mode** alternative to this
stack: it declares NodePool `gpu-nodepool` against `eks.amazonaws.com/NodeClass`,
while this stack declares NodePool `gpu` against `karpenter.k8s.aws/EC2NodeClass`.
The names are kept distinct on purpose — don't rename either to match the other.
The two paths are mutually exclusive; see "NodePool ownership" in `README.md`.
Never suggest `kubectl apply -f k8s-manifest/infra/` to someone using this stack
(only `priority-class.yaml` from that directory applies).

Model manifests are portable across both paths because their GPU tolerations use
`operator: Exists`, which ignores the taint value (`true` here vs `Exists` in the
Auto Mode NodePool). Keep it that way when adding manifests.

## Version pins that cascade

- The `helm` provider is held at `< 3.0.0` (v3 turned the provider's `kubernetes` block into an attribute), which is why `eks-blueprints-addons` is pinned rather than tracking latest. Bumping either requires bumping both plus rewriting the provider block in `main.tf`.
- The Karpenter chart version is a `local` in `karpenter.tf`, shared by the `karpenter-crd` and `karpenter` releases — keep them in lockstep.

## Capacity Reservation (ODCR) — opt-in

`enable_capacity_reservation` (default `false`) gates the `reserved-capacity`
EC2NodeClass and the `reserved-capacity-pool` NodePool; both are filtered out of
their `for_each` in `karpenter.tf`. Enabling it requires `capacity_reservation_id`
as well. `karpenter.tf` wraps the ID in `coalesce(..., "unused")` so template
rendering doesn't break when the flag is off — keep that guard if you touch the
template vars.

Short-lived Capacity Blocks are better managed with `kubectl` outside Terraform,
so they can be torn down before the reservation window closes without touching
state.

## Common gotchas

- `terraform show planfile` fails with "Variables not allowed" on this stack — use `terraform plan -no-color > /tmp/plan.txt` and grep it when summarizing actions.
- `eks-blueprints-addons` emits deprecation warnings about `data.aws_region.current.name` — upstream issue, ignore.
- `dev.auto.tfvars` strings must be quoted (`aws_profile = "default"`, not `aws_profile = default`).
- Several add-ons are wired but disabled (`enable_* = false`) rather than deleted. Don't treat their presence in `addons.tf` as evidence they're running.
