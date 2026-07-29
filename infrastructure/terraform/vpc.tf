module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 6.6"

  name = local.name
  cidr = local.vpc_cidr

  azs = local.azs

  # Private: 4 x /20 filling 10.6.0.0/20 .. 10.6.48.0/20 (~4091 usable IPs each).
  # Public:  4 x /24 offset to 10.6.64.0/24 .. 10.6.67.0/24. The offset must stay
  # clear of the private range — at 3 AZs the public subnets started at
  # 10.6.48.0/24, which the 4th private /20 (10.6.48.0/20) now covers.
  private_subnets = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 4, k)]
  public_subnets  = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 8, k + 64)]

  enable_nat_gateway = true
  single_nat_gateway = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
    # Tags subnets for Karpenter auto-discovery
    "karpenter.sh/discovery" = local.name
    # Opt in to VPC CNI enhanced subnet discovery. A node's own subnet is always
    # usable without this tag, but untagged subnets are never picked for *new*
    # secondary ENIs.
    "kubernetes.io/role/cni" = 1
  }

  tags = local.tags
}
