from google.cloud import aiplatform

PROJECT_ID = "cakir-kubeflow"
REGION = "us-central1"
aiplatform.init(project=PROJECT_ID,location=REGION)

job = aiplatform.PipelineJob(
    display_name='trigger-coupon-model-pipeline',
    template_path="gs://mlops-cakir-kubeflow-v1/mlops-recommendation/mlops-recommendation-pipeline-deploy-v1.json",
    pipeline_root="gs://mlops-cakir-kubeflow-v1/coupon-pipeline-v1",
    enable_caching=False
)
job.submit()
