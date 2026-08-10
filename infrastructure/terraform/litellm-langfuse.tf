################################################################################
# LiteLLM (LLM gateway) + Langfuse (LLM observability) — opt-in
#
# Gated behind `enable_litellm_langfuse` (default false). LiteLLM fronts both the
# self-hosted models from k8s-manifest/ and Bedrock, and reports every call to
# Langfuse. Both are ClusterIP only — reach them with `kubectl port-forward`
# (see README). Bedrock access uses EKS Pod Identity, so there are no static keys.
################################################################################

locals {
  litellm_namespace  = "litellm"
  langfuse_namespace = "langfuse"
  litellm_langfuse   = var.enable_litellm_langfuse ? 1 : 0
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
  count = local.litellm_langfuse
  # special = false: the key is interpolated into a YAML scalar and pasted into
  # curl commands, so keep it shell- and YAML-safe.
  length  = 32
  special = false
}

resource "random_password" "litellm_postgres" {
  count   = local.litellm_langfuse
  length  = 24
  special = false
}

resource "random_password" "langfuse_salt" {
  count   = local.litellm_langfuse
  length  = 32
  special = false
}

resource "random_password" "langfuse_nextauth_secret" {
  count   = local.litellm_langfuse
  length  = 32
  special = false
}

# Langfuse requires a 256-bit hex encryption key (`openssl rand -hex 32`)
resource "random_id" "langfuse_encryption_key" {
  count       = local.litellm_langfuse
  byte_length = 32
}

resource "random_password" "langfuse_postgres" {
  count   = local.litellm_langfuse
  length  = 24
  special = false
}

resource "random_password" "langfuse_clickhouse" {
  count   = local.litellm_langfuse
  length  = 24
  special = false
}

resource "random_password" "langfuse_redis" {
  count   = local.litellm_langfuse
  length  = 24
  special = false
}

resource "random_password" "langfuse_s3" {
  count   = local.litellm_langfuse
  length  = 24
  special = false
}

resource "random_password" "langfuse_admin" {
  count   = local.litellm_langfuse
  length  = 16
  special = false
}

# Pre-generated project keys, so LiteLLM's callback config is known before
# Langfuse has ever booted (LANGFUSE_INIT_* seeds these on first start)
resource "random_uuid" "langfuse_public_key" {
  count = local.litellm_langfuse
}

resource "random_uuid" "langfuse_secret_key" {
  count = local.litellm_langfuse
}

locals {
  langfuse_admin_email = "admin@${local.litellm_namespace}.local"
  langfuse_public_key  = local.litellm_langfuse > 0 ? "pk-lf-${random_uuid.langfuse_public_key[0].result}" : ""
  langfuse_secret_key  = local.litellm_langfuse > 0 ? "sk-lf-${random_uuid.langfuse_secret_key[0].result}" : ""
}

################################################################################
# Pod Identity: allow LiteLLM to invoke Bedrock models
################################################################################

module "litellm_bedrock_pod_identity" {
  count   = local.litellm_langfuse
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
  count            = local.litellm_langfuse
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
    salt                = random_password.langfuse_salt[0].result
    nextauth_secret     = random_password.langfuse_nextauth_secret[0].result
    encryption_key      = random_id.langfuse_encryption_key[0].hex
    langfuse_public_key = local.langfuse_public_key
    langfuse_secret_key = local.langfuse_secret_key
    admin_email         = local.langfuse_admin_email
    admin_password      = random_password.langfuse_admin[0].result
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
  count            = local.litellm_langfuse
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
    region              = local.region
    master_key          = "sk-${random_password.litellm_master_key[0].result}"
    postgres_password   = random_password.litellm_postgres[0].result
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
  value       = local.litellm_langfuse > 0 ? "kubectl port-forward -n ${local.litellm_namespace} svc/litellm 4000:4000" : null
}

output "litellm_master_key" {
  description = "LiteLLM master key — use as the Bearer token / OpenAI api_key"
  value       = local.litellm_langfuse > 0 ? "sk-${random_password.litellm_master_key[0].result}" : null
  sensitive   = true
}

output "langfuse_port_forward" {
  description = "Reach the Langfuse UI at http://localhost:3000"
  value       = local.litellm_langfuse > 0 ? "kubectl port-forward -n ${local.langfuse_namespace} svc/langfuse-web 3000:3000" : null
}

output "langfuse_admin_login" {
  description = "Seeded Langfuse admin credentials (LANGFUSE_INIT_USER_*)"
  value = local.litellm_langfuse > 0 ? {
    email    = local.langfuse_admin_email
    password = random_password.langfuse_admin[0].result
  } : null
  sensitive = true
}
