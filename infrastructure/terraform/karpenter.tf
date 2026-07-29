locals {
  karpenter_namespace = "karpenter"
  # 1.12.0 fixes nil-pointer panic in CapacityReservationFromEC2 when selecting
  # Capacity Blocks via capacityReservationSelectorTerms (aws/karpenter#9080)
  karpenter_version = "1.12.0"
}

################################################################################
# Controller & Node IAM roles, SQS Queue, Eventbridge Rules
################################################################################

# IRSA trust policy for Karpenter controller running on Fargate
# (EKS Fargate does not support Pod Identity; v21 karpenter module removed IRSA support,
# so we inject the IRSA trust policy via iam_role_source_assume_policy_documents)
data "aws_iam_policy_document" "karpenter_irsa" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [module.eks.oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "${module.eks.oidc_provider}:sub"
      values   = ["system:serviceaccount:${local.karpenter_namespace}:karpenter"]
    }

    condition {
      test     = "StringEquals"
      variable = "${module.eks.oidc_provider}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

module "karpenter" {
  source  = "terraform-aws-modules/eks/aws//modules/karpenter"
  version = "~> 21.15"

  cluster_name = module.eks.cluster_name
  namespace    = local.karpenter_namespace

  # Name needs to match role name passed to the EC2NodeClass
  node_iam_role_use_name_prefix = false
  node_iam_role_name            = local.name

  # EKS Fargate does not support pod identity; inject IRSA trust policy instead
  create_pod_identity_association         = false
  iam_role_source_assume_policy_documents = [data.aws_iam_policy_document.karpenter_irsa.json]

  tags = local.tags
}

################################################################################
# Helm charts
################################################################################

# Karpenter runs on Fargate, and Fargate only rewrites a pod's schedulerName to
# `fargate-scheduler` at admission time. A profile reports created via the API
# well before that mutating path actually works, so installing the Karpenter
# chart too early yields pods stuck on `default-scheduler` — Pending forever,
# and recreating the pods is the only fix. Depending on the profiles is not
# enough on its own; this adds a grace period on top.
resource "time_sleep" "fargate_profiles_ready" {
  depends_on      = [module.eks.fargate_profiles]
  create_duration = "60s"
}

resource "helm_release" "karpenter_crd" {
  name             = "karpenter-crd"
  namespace        = local.karpenter_namespace
  create_namespace = true
  repository       = "oci://public.ecr.aws/karpenter"
  chart            = "karpenter-crd"
  version          = local.karpenter_version

  lifecycle {
    ignore_changes = [repository_password]
  }
}

resource "helm_release" "karpenter" {
  name             = "karpenter"
  namespace        = local.karpenter_namespace
  create_namespace = true
  repository       = "oci://public.ecr.aws/karpenter"
  # To avoid 403 error from public ECR
  # ref: https://github.com/aws/karpenter-provider-aws/issues/6357
  # repository_username = data.aws_ecrpublic_authorization_token.token.user_name
  # repository_password = data.aws_ecrpublic_authorization_token.token.password
  chart = "karpenter"
  # version = "1.6.1"
  version = local.karpenter_version
  wait    = true

  values = [
    <<-EOT
    dnsPolicy: Default
    settings:
      clusterName: ${module.eks.cluster_name}
      clusterEndpoint: ${module.eks.cluster_endpoint}
      interruptionQueue: ${module.karpenter.queue_name}
      featureGates:
        # nodeRepair kept off: it misjudged long model-loading (DeepGEMM JIT) as
        # unhealthy and killed P5en nodes ~13min after launch
        nodeRepair: false
    serviceAccount:
      annotations:
        eks.amazonaws.com/role-arn: ${module.karpenter.iam_role_arn}
    webhook:
      enabled: false
    controller:
      resources:
        limits:
          cpu: 1
          memory: 1Gi
    EOT
  ]

  depends_on = [
    helm_release.karpenter_crd,
    module.eks.fargate_profiles,
    time_sleep.fargate_profiles_ready,
  ]

  lifecycle {
    ignore_changes = [
      repository_password
    ]
  }
}

################################################################################
# Karpenter Node Class & Node Pool
################################################################################

locals {
  karpenter_template_vars = {
    cluster_name            = module.eks.cluster_name
    karpenter_namespace     = local.karpenter_namespace
    capacity_reservation_id = coalesce(var.capacity_reservation_id, "unused")
  }
}

data "kubectl_path_documents" "karpenter_node_classes" {
  pattern = "${path.module}/kubernetes/karpenter/node-classes/*.yaml"
  vars    = local.karpenter_template_vars
}

data "kubectl_path_documents" "karpenter_node_pools" {
  pattern = "${path.module}/kubernetes/karpenter/node-pools/*.yaml"
  vars    = local.karpenter_template_vars
}

data "kubectl_path_documents" "karpenter_flow_schemas" {
  pattern = "${path.module}/kubernetes/karpenter/flow-schemas/*.yaml"
  vars    = local.karpenter_template_vars
}

resource "kubectl_manifest" "karpenter_node_class" {
  for_each = {
    for k, v in data.kubectl_path_documents.karpenter_node_classes.manifests :
    k => v if var.enable_capacity_reservation || !strcontains(k, "reserved-capacity")
  }
  depends_on = [helm_release.karpenter]
  yaml_body  = each.value
  wait       = true
}

resource "kubectl_manifest" "karpenter_node_pool" {
  for_each = {
    for k, v in data.kubectl_path_documents.karpenter_node_pools.manifests :
    k => v if var.enable_capacity_reservation || !strcontains(k, "reserved-capacity")
  }
  depends_on = [helm_release.karpenter, kubectl_manifest.karpenter_node_class]
  yaml_body  = each.value
}

resource "kubectl_manifest" "karpenter_flow_schema" {
  for_each  = data.kubectl_path_documents.karpenter_flow_schemas.manifests
  yaml_body = each.value
}


