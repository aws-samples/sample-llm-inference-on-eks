
variable "aws_profile" {
  type        = string
  default     = "default"
  description = "The AWS CLI profile to use"
}

variable "vpc_cidr" {
  type        = string
  default     = "10.6.0.0/16"
  description = "The CIDR block for the VPC"
}

variable "region" {
  type        = string
  default     = "us-west-2"
  description = "The AWS region to deploy the resources"
}

variable "cluster_name" {
  type        = string
  default     = "llm-inference"
  description = "The name of the EKS cluster"
}

variable "cluster_version" {
  type        = string
  default     = "1.35"
  description = "The Kubernetes version for the EKS cluster"
}

variable "slack_api_url" {
  type        = string
  default     = "https://hooks.slack.com/services/xxxxxxx"
  description = "The Slack API URL used for the Prometheus Alertmanager"
}

variable "enable_gpu_operator" {
  type        = bool
  default     = false
  description = "Whether to install the NVIDIA GPU Operator (device plugin + GPU feature discovery; driver/toolkit come from the NVIDIA AMI)"
}

variable "enable_aws_efa_device_plugin" {
  type        = bool
  default     = false
  description = "Whether to enable the AWS EFA Device Plugin"
}

variable "enable_lws" {
  type        = bool
  default     = true
  description = "Whether to install the LeaderWorkerSet (LWS) controller, required by the multi-node k8s-manifest/lws examples"
}

variable "enable_litellm_langfuse" {
  type        = bool
  default     = false
  description = "Whether to install the LiteLLM gateway + Langfuse observability stack (ClusterIP only; LiteLLM gets Bedrock access via Pod Identity)"
}

variable "enable_capacity_reservation" {
  type        = bool
  default     = false
  description = "Whether to provision resources tied to an EC2 On-Demand Capacity Reservation (ODCR)"
}

variable "capacity_reservation_id" {
  type        = string
  default     = null
  description = "The ID of the capacity reservation. Required when enable_capacity_reservation is true."
}

