terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.0.0"
    }
  }
}

resource "kubeflowpipelines_pipeline" "mlpipeline_coupon" {
    name    = var.pipeline_name
    url     = var.pipeline_json
    version = var.version
}


