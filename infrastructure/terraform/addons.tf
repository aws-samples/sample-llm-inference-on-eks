module "eks_blueprints_addons" {
  source = "aws-ia/eks-blueprints-addons/aws"
  # Pinned: 1.24+ requires helm provider v3, whose `kubernetes` attribute syntax
  # is incompatible with the provider block in main.tf
  version = "1.23.0"

  cluster_name      = module.eks.cluster_name
  cluster_endpoint  = module.eks.cluster_endpoint
  cluster_version   = module.eks.cluster_version
  oidc_provider_arn = module.eks.oidc_provider_arn
  # disable the Telemetry from AWS using CloudFormation
  observability_tag = null

  eks_addons = {
    aws-ebs-csi-driver        = { most_recent = true }
    metrics-server            = { most_recent = true }
    eks-node-monitoring-agent = { most_recent = true }
  }

  enable_aws_load_balancer_controller = false
  aws_load_balancer_controller = {
    chart_version = "3.1.0"
    set = [
      {
        name  = "vpcId" # explicitly set the vpcId, otherwise it may not able to retrieve the vpcId from the node
        value = module.vpc.vpc_id
      },
      # ALBGatewayAPI and NLBGatewayAPI feature gates removed — Gateway API is GA in v3.x
      {
        # The service mutator webhook (mservice.elbv2.k8s.aws) intercepts CREATE on
        # every Service in the cluster with failurePolicy=Fail and an empty
        # namespaceSelector. On a from-scratch apply the controller's own pods are
        # still Pending (no nodes yet), so the webhook has no endpoints and every
        # Service creation is rejected — including coredns', which deadlocks the
        # whole bootstrap. Its only job is defaulting loadBalancerClass on
        # type: LoadBalancer Services, which this stack never creates.
        name  = "enableServiceMutatorWebhook"
        value = "false"
      },
    ]
  }
  # EKS Managed Addons included metrics-server

  enable_kube_prometheus_stack = false
  kube_prometheus_stack = {
    values = [
      templatefile("${path.module}/kubernetes/kube-prometheus-stack/values.override.yaml", {
        slack_api_url = var.slack_api_url
      })
    ]
  }

  helm_releases = {}

  tags = local.tags
}


module "ebs_csi_pod_identity" {
  source  = "terraform-aws-modules/eks-pod-identity/aws"
  version = "~> 2.7"

  name                      = "ebs-csi"
  attach_aws_ebs_csi_policy = true

  associations = {
    this = {
      cluster_name    = module.eks.cluster_name
      namespace       = "kube-system"
      service_account = "ebs-csi-controller-sa"
    }
  }

  tags = local.tags
}

module "aws_cloudwatch_observability_pod_identity" {
  source  = "terraform-aws-modules/eks-pod-identity/aws"
  version = "~> 2.7"

  name = "aws-cloudwatch-observability"

  attach_aws_cloudwatch_observability_policy = true
  associations = {
    this = {
      cluster_name    = module.eks.cluster_name
      namespace       = "amazon-cloudwatch"
      service_account = "cloudwatch-agent"
    }
  }

  tags = local.tags
}


resource "helm_release" "nvidia_gpu_operator" {
  count            = var.enable_gpu_operator ? 1 : 0
  name             = "gpu-operator"
  repository       = "https://helm.ngc.nvidia.com/nvidia"
  chart            = "gpu-operator"
  version          = "v26.3.0" # Latest stable version as of now
  namespace        = "gpu-operator"
  create_namespace = true
  wait             = true

  values = [templatefile("${path.module}/kubernetes/gpu-operator/values-override.yaml", {})]

  # Its pods need a regular (non-Fargate) node, which only exists once Karpenter
  # is running and has provisioned one. Installing earlier makes `wait = true`
  # time out and marks the release failed, even though the pods come up fine
  # later once capacity arrives.
  depends_on = [kubectl_manifest.karpenter_node_pool]
}

resource "helm_release" "aws_efa_device_plugin" {
  count = var.enable_aws_efa_device_plugin ? 1 : 0
  name  = "aws-efa-k8s-device-plugin"
  # https://github.com/aws/eks-charts/tree/master/stable/aws-efa-k8s-device-plugin
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-efa-k8s-device-plugin"
  version    = "v0.5.30"
  namespace  = "kube-system"
  wait       = false

  # No affinity override: the chart's own `supportedInstanceLabels` already
  # enumerates every EFA-capable instance type (304 as of v0.5.30) and tracks
  # https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html#efa-instance-types.
  # Hand-maintaining a subset only risks omissions — the previous list stopped at
  # p6-b200 and silently excluded p6-b300/p6e-gb300, and it also wrongly included
  # small g6/g6e sizes that have no EFA at all.
  values = [
    <<-EOT
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
    EOT
  ]
}
