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


# Create a Kubeflow Pipelines job to trigger the Kubeflow pipeline
resource "kubeflowpipelines_job" "pipeline_trigger" {
    name            = "pipeline-trigger"
    description     = "Trigger for coupon Kubeflow pipeline"
    service_account = "pipeline-runner"
    enabled         = true
    max_concurrency = 2
    no_catchup      = true

    pipeline_spec {
        pipeline_version_id = kubeflowpipelines_pipeline.mlpipeline_coupon.version_id
    }

    trigger {
        cron_schedule {
            start_time = "2023-06-23T00:00:00Z"
            end_time   = "2024-06-23T00:00:00Z"
            cron       = "0 0 1 * * ?"
        }
    }
}

