
variable "pipeline_json" {
  type        = string
  description = "The Kubeflow Pipeline for recommendation of coupon"
  default = "gs://mlops-cakir-kubeflow-v1/mlops-recommendation/mlops-recommendation-pipeline-deploy-v1.json"
}
variable "version" {
  type        = string
  default = "v0.0.1"
}
variable "pipeline_name" {
  type        = string
  default = "kubeflow-mlpipeline-coupon"
}

