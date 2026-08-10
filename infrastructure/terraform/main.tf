terraform {
  required_version = ">= 1.3"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 6.40"
    }
    helm = {
      # v3 changed the provider's `kubernetes` block into an attribute; stay on 2.x
      source  = "hashicorp/helm"
      version = ">= 2.9, < 3.0.0"
    }
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = ">= 1.14"
    }
    time = {
      source  = "hashicorp/time"
      version = ">= 0.9"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.5"
    }
  }

  # Partial backend config — supply bucket/key/region at init time:
  #   terraform init -backend-config=backend.hcl
  # See backend.hcl.example. For a throwaway single-user cluster you can drop
  # the block entirely and use local state.
  backend "s3" {}
}

provider "aws" {
  region  = local.region
  profile = var.aws_profile
}

# This provider is required for ECR to authenticate with public repos. Please note ECR authentication requires us-east-1 as region hence its hardcoded below.
# If your region is same as us-east-1 then you can just use one aws provider
provider "aws" {
  alias   = "ecr"
  region  = "us-east-1"
  profile = var.aws_profile
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name, "--region", local.region]
  }
}

provider "kubectl" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  load_config_file       = false
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name, "--region", local.region]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      # This requires the awscli to be installed locally where Terraform is executed
      args = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

################################################################################
# Common data/locals
################################################################################


data "aws_ecrpublic_authorization_token" "token" {
  provider = aws.ecr
}

data "aws_availability_zones" "available" {
  # Do not include local zones
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

data "aws_caller_identity" "current" {}

locals {
  name   = var.cluster_name
  region = var.region

  vpc_cidr = var.vpc_cidr
  # 4 AZs: us-west-2d is included because p5en / p6-b200 on-demand capacity is
  # often only available there (2a/2b/2c alone miss it).
  azs = slice(data.aws_availability_zones.available.names, 0, 4)

  tags = {
    Blueprint   = local.name
    GithubRepo  = "github.com/aws-samples/sample-llm-inference-on-eks"
    Environment = "dev"
  }
}
