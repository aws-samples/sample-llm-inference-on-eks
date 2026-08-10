################################################################################
# LiteLLM (LLM gateway) + Langfuse (LLM observability) — two independent opt-ins
#
# `enable_litellm` and `enable_langfuse` are separate because each is useful
# alone: LiteLLM as a plain gateway in front of the k8s-manifest/ models and
# Bedrock, Langfuse as a tracing backend for an app you instrument yourself.
# Enable both and the tracing integration is wired automatically — LiteLLM gets
# Langfuse callbacks plus pre-seeded project keys. Both are ClusterIP only; reach
# them with `kubectl port-forward` (see README). LiteLLM's Bedrock access uses
# EKS Pod Identity, so there are no static keys.
################################################################################

locals {
  litellm_namespace  = "litellm"
  langfuse_namespace = "langfuse"

  litellm  = var.enable_litellm ? 1 : 0
  langfuse = var.enable_langfuse ? 1 : 0

  # The tracing integration only exists when both halves are deployed
  litellm_langfuse_wired = var.enable_litellm && var.enable_langfuse
}

################################################################################
# Generated secrets
#
# These land in Terraform state in plaintext. Acceptable for a demo stack whose
# state bucket is already sensitive (it holds the cluster's IAM wiring); if that
# is not true for you, move them to Secrets Manager and reference them via the
# charts' `existingSecret` / `secretKeyRef` options instead.
################################################################################

resource "random_password" "litellm_master_key" {
  count = local.litellm
  # special = false: the key is interpolated into a YAML scalar and pasted into
  # curl commands, so keep it shell- and YAML-safe.
  length  = 32
  special = false
}

resource "random_password" "litellm_postgres" {
  count   = local.litellm
  length  = 24
  special = false
}

resource "random_password" "langfuse_salt" {
  count   = local.langfuse
  length  = 32
  special = false
}

resource "random_password" "langfuse_nextauth_secret" {
  count   = local.langfuse
  length  = 32
  special = false
}

# Langfuse requires a 256-bit hex encryption key (`openssl rand -hex 32`)
resource "random_id" "langfuse_encryption_key" {
  count       = local.langfuse
  byte_length = 32
}

resource "random_password" "langfuse_postgres" {
  count   = local.langfuse
  length  = 24
  special = false
}

resource "random_password" "langfuse_clickhouse" {
  count   = local.langfuse
  length  = 24
  special = false
}

resource "random_password" "langfuse_redis" {
  count   = local.langfuse
  length  = 24
  special = false
}

resource "random_password" "langfuse_s3" {
  count   = local.langfuse
  length  = 24
  special = false
}

resource "random_password" "langfuse_admin" {
  count   = local.langfuse
  length  = 16
  special = false
}

# Project keys for the LiteLLM -> Langfuse integration. Generated here (rather
# than read back from Langfuse) so LiteLLM's callback config is known before
# Langfuse has ever booted; LANGFUSE_INIT_* seeds the same values on first start.
# Only needed when both halves are enabled.
resource "random_uuid" "langfuse_public_key" {
  count = local.litellm_langfuse_wired ? 1 : 0
}

resource "random_uuid" "langfuse_secret_key" {
  count = local.litellm_langfuse_wired ? 1 : 0
}

locals {
  langfuse_admin_email = "admin@${local.langfuse_namespace}.local"
  langfuse_public_key  = local.litellm_langfuse_wired ? "pk-lf-${random_uuid.langfuse_public_key[0].result}" : ""
  langfuse_secret_key  = local.litellm_langfuse_wired ? "sk-lf-${random_uuid.langfuse_secret_key[0].result}" : ""
}

################################################################################
# Pod Identity: allow LiteLLM to invoke Bedrock models
################################################################################

module "litellm_bedrock_pod_identity" {
  count   = local.litellm
  source  = "terraform-aws-modules/eks-pod-identity/aws"
  version = "~> 2.7"

  name = "${local.name}-litellm-bedrock"

  attach_custom_policy = true
  policy_statements = [
    {
      sid = "BedrockInvoke"
      actions = [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
      ]
      # Cross-region inference profiles (us./global.) need both the profile ARN
      # and the underlying foundation-model ARNs in every region of the profile.
      # Scoped to anthropic.* to match the model_list in kubernetes/litellm/
      # values.yaml — widen it if you add other providers there.
      resources = [
        "arn:aws:bedrock:*::foundation-model/anthropic.*",
        "arn:aws:bedrock:*:${data.aws_caller_identity.current.account_id}:inference-profile/*.anthropic.*",
      ]
    }
  ]

  associations = {
    litellm = {
      cluster_name    = module.eks.cluster_name
      namespace       = local.litellm_namespace
      service_account = "litellm"
    }
  }

  tags = local.tags
}

################################################################################
# Langfuse
################################################################################

resource "helm_release" "langfuse" {
  count            = local.langfuse
  name             = "langfuse"
  repository       = "https://langfuse.github.io/langfuse-k8s"
  chart            = "langfuse"
  version          = "1.5.39"
  namespace        = local.langfuse_namespace
  create_namespace = true
  # wait = false: the bundled ClickHouse/Postgres/Redis/MinIO StatefulSets can
  # take several minutes and the apply has nothing left to sequence behind them.
  wait = false

  values = [templatefile("${path.module}/kubernetes/langfuse/values.yaml", {
    salt            = random_password.langfuse_salt[0].result
    nextauth_secret = random_password.langfuse_nextauth_secret[0].result
    encryption_key  = random_id.langfuse_encryption_key[0].hex
    admin_email     = local.langfuse_admin_email
    admin_password  = random_password.langfuse_admin[0].result
    # Without LiteLLM there is no pre-shared key pair to seed; the project is
    # still created and you generate keys in the UI for your own SDK clients.
    seed_project_keys   = local.litellm_langfuse_wired
    langfuse_public_key = local.langfuse_public_key
    langfuse_secret_key = local.langfuse_secret_key
    postgres_password   = random_password.langfuse_postgres[0].result
    clickhouse_password = random_password.langfuse_clickhouse[0].result
    redis_password      = random_password.langfuse_redis[0].result
    s3_root_password    = random_password.langfuse_s3[0].result
  })]

  # Same constraint as the GPU Operator and LWS: there is no Fargate profile for
  # this namespace, so the pods need a Karpenter-provisioned node. Its
  # StatefulSets also claim PVCs with no storageClassName, which means they need
  # gp3 to already be the default StorageClass.
  depends_on = [
    kubectl_manifest.karpenter_node_pool,
    kubernetes_storage_class_v1.default_gp3,
  ]
}

################################################################################
# LiteLLM proxy
################################################################################

resource "helm_release" "litellm" {
  count            = local.litellm
  name             = "litellm"
  repository       = "oci://ghcr.io/berriai"
  chart            = "litellm-helm"
  version          = "1.93.0"
  namespace        = local.litellm_namespace
  create_namespace = true
  # wait = false: the Prisma migration Job must finish before the proxy is
  # Ready, and nothing in this stack depends on LiteLLM being up.
  wait = false

  values = [templatefile("${path.module}/kubernetes/litellm/values.yaml", {
    region            = local.region
    master_key        = "sk-${random_password.litellm_master_key[0].result}"
    postgres_password = random_password.litellm_postgres[0].result
    # Callbacks and the LANGFUSE_* env block are omitted entirely when Langfuse
    # is not deployed — a `langfuse` callback with no reachable host makes every
    # request log callback errors.
    langfuse_enabled    = local.litellm_langfuse_wired
    langfuse_host       = "http://langfuse-web.${local.langfuse_namespace}:3000"
    langfuse_public_key = local.langfuse_public_key
    langfuse_secret_key = local.langfuse_secret_key
  })]

  depends_on = [
    module.litellm_bedrock_pod_identity,
    kubectl_manifest.karpenter_node_pool,
    kubernetes_storage_class_v1.default_gp3,
  ]
}

################################################################################
# Outputs
################################################################################

output "litellm_port_forward" {
  description = "Reach the LiteLLM gateway at http://localhost:4000 (OpenAI-compatible /v1)"
  value       = local.litellm > 0 ? "kubectl port-forward -n ${local.litellm_namespace} svc/litellm 4000:4000" : null
}

output "litellm_master_key" {
  description = "LiteLLM master key — use as the Bearer token / OpenAI api_key"
  value       = local.litellm > 0 ? "sk-${random_password.litellm_master_key[0].result}" : null
  sensitive   = true
}

output "langfuse_port_forward" {
  description = "Reach the Langfuse UI at http://localhost:3000"
  value       = local.langfuse > 0 ? "kubectl port-forward -n ${local.langfuse_namespace} svc/langfuse-web 3000:3000" : null
}

output "langfuse_admin_login" {
  description = "Seeded Langfuse admin credentials (LANGFUSE_INIT_USER_*)"
  value = local.langfuse > 0 ? {
    email    = local.langfuse_admin_email
    password = random_password.langfuse_admin[0].result
  } : null
  sensitive = true
}
