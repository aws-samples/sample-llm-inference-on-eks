#!/usr/bin/env bash
# Ordered teardown. A bare `terraform destroy` deadlocks: the EKS module goes
# away while Karpenter nodes and ELB-controller-created security groups are
# still around. Order: drain nodes -> addons -> cluster -> everything else.
set -uo pipefail

CLUSTER_NAME=$(terraform output -raw cluster_name 2>/dev/null || true)
if [[ -z "$CLUSTER_NAME" ]]; then
  echo "No terraform outputs found (nothing applied?) — skipping kubectl cleanup."
else
  echo "Configuring kubectl for ${CLUSTER_NAME}..."
  eval "$(terraform output -raw configure_kubectl)"

  # Ingresses first: the ALB controller must still be running to delete the
  # load balancers it created. Best-effort — off by default in addons.tf.
  kubectl delete ingress --all --all-namespaces --ignore-not-found || true

  # LeaderWorkerSets before the NodePools: the controller recreates their pods,
  # so leaving them up just means Pending pods once the nodes are gone.
  kubectl delete leaderworkersets --all --all-namespaces --ignore-not-found || true

  # LiteLLM / Langfuse also before the NodePools. Helm leaves StatefulSet PVCs
  # behind by design, and the EBS CSI controller that has to delete the gp3
  # volumes runs on a Karpenter node — so this has to happen while nodes still
  # exist, or the volumes leak as orphaned EBS. No-op when the releases were
  # never installed (enable_litellm_langfuse = false leaves them at count 0).
  for release in helm_release.litellm helm_release.langfuse; do
    terraform destroy -target="$release" -auto-approve || true
  done
  kubectl delete namespace litellm langfuse --ignore-not-found || true

  # Deleting the NodePools drains and terminates every Karpenter node. Without
  # this the EKS destroy blocks on nodes it doesn't own.
  kubectl delete nodepool --all --ignore-not-found || true
fi

for target in module.eks_blueprints_addons module.eks; do
  echo "Destroying ${target}..."
  if ! terraform destroy -target="$target" -auto-approve; then
    echo "FAILED: terraform destroy of ${target}"
    exit 1
  fi
done

# The ALB controller tags the SGs it creates but does not always remove them,
# which then blocks the VPC destroy.
if [[ -n "$CLUSTER_NAME" ]]; then
  echo "Deleting leftover ELB-controller security groups..."
  for sg in $(aws ec2 describe-security-groups \
    --filters "Name=tag:elbv2.k8s.aws/cluster,Values=${CLUSTER_NAME}" \
    --query 'SecurityGroups[].GroupId' --output text); do
    aws ec2 delete-security-group --group-id "$sg" || true
  done
fi

echo "Destroying everything else..."
terraform destroy -auto-approve
