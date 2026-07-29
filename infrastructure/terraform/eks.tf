################################################################################
# Cluster
################################################################################

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 21.15"

  name               = local.name
  kubernetes_version = var.cluster_version

  # Give the Terraform identity admin access to the cluster
  # which will allow it to deploy resources into the cluster
  enable_cluster_creator_admin_permissions = true
  endpoint_public_access                   = true

  addons = {
    # Enable after creation to run on Karpenter managed nodes
    vpc-cni = {
      enabled                     = true
      most_recent                 = true
      resolve_conflicts_on_update = "OVERWRITE"
      before_compute              = true
      configuration_values = jsonencode({
        # enableNetworkPolicy : "true"
        env = {
          # Reference docs https://docs.aws.amazon.com/eks/latest/userguide/cni-increase-ip-addresses.html
          ENABLE_PREFIX_DELEGATION = "true"
          WARM_PREFIX_TARGET       = "1"
        }
      })
    }
    coredns = {
      enabled                     = true
      most_recent                 = true
      resolve_conflicts_on_update = "OVERWRITE"
      configuration_values = jsonencode({
        nodeSelector = {
          "eks.amazonaws.com/compute-type" = "fargate"
        }
        autoScaling = {
          enabled     = true
          minReplicas = 2
          maxReplicas = 10
        }
        corefile = <<-EOF
          .:53 {
              errors
              health {
                  lameduck 10s
              }
              ready
              kubernetes cluster.local in-addr.arpa ip6.arpa {
                  pods insecure
                  fallthrough in-addr.arpa ip6.arpa
                  ttl 30
              }
              prometheus :9153
              forward . /etc/resolv.conf
              cache 30
              loop
              reload
              loadbalance
          }
        EOF
      })
    }
    kube-proxy = {
      most_recent                 = true
      resolve_conflicts_on_update = "OVERWRITE"
      configuration_values = jsonencode({
        mode = "ipvs"
        ipvs = {
          scheduler = "rr"
        }
      })
    }
    eks-pod-identity-agent = {
      before_compute              = true
      most_recent                 = true
      resolve_conflicts_on_create = "OVERWRITE"
    }
  }

  # For migration to EKS Auto Mode only
  # bootstrap_self_managed_addons = true
  # cluster_compute_config = {
  #   enabled    = true
  #   node_pools = ["general-purpose", "system"]
  # }

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Fargate profiles use the cluster primary security group
  # Therefore these are not used and can be skipped
  create_security_group      = false
  create_node_security_group = false

  fargate_profiles = {
    karpenter = {
      selectors = [
        { namespace = "karpenter" }
      ]
    }
    coredns = {
      selectors = [
        { namespace = "kube-system"
          labels = {
            k8s-app = "kube-dns"
          }
        }
      ]
    }
  }
  tags = merge(local.tags, {
    # NOTE - if creating multiple security groups with this module, only tag the
    # security group that Karpenter should utilize with the following tag
    # (i.e. - at most, only one security group should have this tag in your account)
    "karpenter.sh/discovery" = local.name
  })
}

#---------------------------------------------------------------
# Disable default GP2 Storage Class
#---------------------------------------------------------------
resource "kubernetes_annotations" "disable_gp2" {
  annotations = {
    "storageclass.kubernetes.io/is-default-class" : "false"
  }
  api_version = "storage.k8s.io/v1"
  kind        = "StorageClass"
  metadata {
    name = "gp2"
  }
  force = true

  depends_on = [module.eks.cluster_id]
}
#---------------------------------------------------------------
# GP3 Storage Class - Set as default
#---------------------------------------------------------------
resource "kubernetes_storage_class_v1" "default_gp3" {
  metadata {
    name = "gp3"
    annotations = {
      "storageclass.kubernetes.io/is-default-class" : "true"
    }
  }

  storage_provisioner    = "ebs.csi.aws.com"
  reclaim_policy         = "Delete"
  allow_volume_expansion = true
  volume_binding_mode    = "WaitForFirstConsumer"
  parameters = {
    type = "gp3"
  }

  depends_on = [
    module.eks.cluster_id,
    kubernetes_annotations.disable_gp2
  ]
}

output "configure_kubectl" {
  description = "Configure kubectl: make sure you're logged in with the correct AWS profile and run the following command to update your kubeconfig"
  value       = "aws eks --region ${local.region} update-kubeconfig --name ${module.eks.cluster_name} --alias ${local.name}"
}

output "cluster_name" {
  description = "EKS cluster name (used by cleanup.sh to find leftover ELB security groups)"
  value       = module.eks.cluster_name
}
