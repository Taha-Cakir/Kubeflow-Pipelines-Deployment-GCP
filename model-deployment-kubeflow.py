from kfp.v2 import dsl
from kfp.v2.dsl import (Input,Output,Metrics,component,Model)
from google.cloud.aiplatform import pipeline_jobs
from typing import NamedTuple
from kfp.v2 import compiler


@component(
packages_to_install=["gcsfs","pandas","google-cloud-storage"]
)
def validate_input_ds(filename:str)-> NamedTuple("output", [("input_validation", str)]):

    import logging
    from google.cloud import storage
    import pandas as pd

    logging.basicConfig(level=logging.INFO)

    logging.info(f"Reading file: {filename}")
    df = pd.read_csv(filename)
    expected_num_cols = 26
    num_cols = len(df.columns)

    logging.info(f"Number of columns: {num_cols}")
    
    input_validation="true"
    
    if num_cols != expected_num_cols:
        input_validation="false"
        
    expected_col_names = ['destination', 'passanger', 'weather', 'temperature', 'time', 'coupon',
                               'expiration', 'gender', 'age', 'maritalStatus', 'has_children',
                               'education', 'occupation', 'income', 'car', 'Bar', 'CoffeeHouse',
                               'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50',
                               'toCoupon_GEQ5min', 'toCoupon_GEQ15min', 'toCoupon_GEQ25min',
                               'direction_same', 'direction_opp', 'Y']

    if set(df.columns) != set(expected_col_names):
        input_validation="false"

    return (input_validation,)


@component(
packages_to_install=["google-cloud-aiplatform","gcsfs","xgboost","category_encoders","imblearn","pandas","google-cloud-storage"]
)
def custom_training_job_component(
    max_depth:int,
    learning_rate:float,
    n_estimators:int,
    metrics: Output[Metrics]
)->NamedTuple("output", [("model_validation", str)]):
    import pandas as pd
    from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
    from sklearn.model_selection import train_test_split
    from category_encoders import HashingEncoder
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket("mlops-cakir-kubeflow-v1")

    def load_data(file_path):
        df = pd.read_csv(file_path)
        return df

    def preprocess_data(df):

        df = df.drop(columns=['car', 'toCoupon_GEQ5min', 'direction_opp'])
        df = df.fillna(df.mode().iloc[0])
        df = df.drop_duplicates()

        df_dummy = df.copy()
        age_list = []
        for i in df['age']:
            if i == 'below21':
                age = '<21'
            elif i in ['21', '26']:
                age = '21-30'
            elif i in ['31', '36']:
                age = '31-40'
            elif i in ['41', '46']:
                age = '41-50'
            else:
                age = '>50'
            age_list.append(age)
        df_dummy['age'] = age_list

        df_dummy['passanger_destination'] = df_dummy['passanger'].astype(str) + '-' + df_dummy['destination'].astype(str)
        df_dummy['marital_hasChildren'] = df_dummy['maritalStatus'].astype(str) + '-' + df_dummy['has_children'].astype(str)
        df_dummy['temperature_weather'] = df_dummy['temperature'].astype(str) + '-' + df_dummy['weather'].astype(str)
        df_dummy = df_dummy.drop(columns=['passanger', 'destination', 'maritalStatus', 'has_children', 'temperature','weather', 'Y'])

        df_dummy = pd.concat([df_dummy, df['Y']], axis = 1)
        df_dummy = df_dummy.drop(columns=['gender', 'RestaurantLessThan20'])
        df_le = df_dummy.replace({
            'expiration':{'2h': 0, '1d' : 1},
            'age':{'<21': 0, '21-30': 1, '31-40': 2, '41-50': 3, '>50': 4},
            'education':{'Some High School': 0, 'High School Graduate': 1, 'Some college - no degree': 2,
                         'Associates degree': 3, 'Bachelors degree': 4, 'Graduate degree (Masters or Doctorate)': 5},
            'Bar':{'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4},
            'CoffeeHouse':{'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4}, 
            'CarryAway':{'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4}, 
            'Restaurant20To50':{'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4},
            'income':{'Less than $12500':0, '$12500 - $24999':1, '$25000 - $37499':2, '$37500 - $49999':3,
                      '$50000 - $62499':4, '$62500 - $74999':5, '$75000 - $87499':6, '$87500 - $99999':7,
                      '$100000 or More':8},
            'time':{'7AM':0, '10AM':1, '2PM':2, '6PM':3, '10PM':4}
        })

        x = df_le.drop('Y', axis=1)
        y = df_le.Y

        return x, y

    def train_model(x_train, y_train,max_depth,learning_rate,n_estimators):
        
        model = XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=42,
            use_label_encoder=False
        )
        model.fit(x_train, y_train)
        return model

    def evaluate_model(model, x_test, y_test, x_sm_train_hashing, y_sm_train):
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)
        y_pred_train = model.predict(x_sm_train_hashing)
        y_pred_train_proba = model.predict_proba(x_sm_train_hashing)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # roc_auc_train_proba = roc_auc_score(y_sm_train, y_pred_train_proba[:, 1])
        # roc_auc_test_proba = roc_auc_score(y_test, y_pred_proba[:, 1])

        return accuracy,precision,recall

    def encode_features(x, n_components=27):
        hashing_ros_enc = HashingEncoder(cols=['passanger_destination', 'marital_hasChildren', 'occupation', 'coupon',
                                               'temperature_weather'], n_components=n_components).fit(x)
        x_test_hashing = hashing_ros_enc.transform(x.reset_index(drop=True))
        return x_test_hashing

    def oversample_data(x_train_hashing, y_train):
        sm = SMOTE(random_state=42)
        x_sm_train_hashing, y_sm_train = sm.fit_resample(x_train_hashing, y_train)
        return x_sm_train_hashing, y_sm_train

    def save_model_artifact(pipeline):
        artifact_name = 'model.bst'
        pipeline.save_model(artifact_name)
        model_artifact = bucket.blob('mlops-recommendation/artifacts/'+artifact_name)
        model_artifact.upload_from_filename(artifact_name)

    input_file = "gs://mlops-cakir-kubeflow-v1-kubeflow-v1/mlops-recommendation/in-vehicle-coupon-recommendation.csv"
    df = load_data(input_file)
    x, y = preprocess_data(df)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    x_train.fillna(x_train.mode().iloc[0], inplace=True)
    x_test.fillna(x_train.mode().iloc[0], inplace=True)
    
    model_name = 'xgboost'
    print("Training and evaluating", model_name, "model:")
    x_train_hashing = encode_features(x_train)
    x_test_hashing = encode_features(x_test)
    x_sm_train_hashing, y_sm_train = oversample_data(x_train_hashing,y_train)

    pipeline = train_model(x_sm_train_hashing,y_sm_train,max_depth,learning_rate,n_estimators)

    accuracy,precision,recall = evaluate_model(pipeline,x_test_hashing,y_test,x_sm_train_hashing,y_sm_train)
    metrics.log_metric("accurancy", accuracy)
    metrics.log_metric("precision", precision)
    metrics.log_metric("recall", recall)
    
    model_validation = "true"
    if accuracy>0.5 and precision>0.5 :
        save_model_artifact(pipeline)
        model_validation="true"
    else :
        model_validation="false"

    return (model_validation,)



@component(
    packages_to_install=["google-cloud-aiplatform"]
)
def model_deployment()-> NamedTuple("endpoint", [("endpoint", str)]):
    
    from google.cloud import aiplatform
    
    aiplatform.init(project="cakir-kubeflow", location="us-central1", staging_bucket="gs://mlops-cakir-kubeflow-v1")
    
    model = aiplatform.Model.upload(
        display_name="mlops-recommendation-model",
        artifact_uri="gs://mlops-cakir-kubeflow-v1/mlops-recommendation/artifacts/",
        serving_container_image_uri = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest",
        sync=False
    )
    
    DEPLOYED_NAME = "coupon-model-endpoint"
    TRAFFIC_SPLIT = {"0": 100}
    MIN_NODES = 1
    MAX_NODES = 1

    endpoint = model.deploy(
        deployed_model_display_name=DEPLOYED_NAME,
        traffic_split=TRAFFIC_SPLIT,
        machine_type="n1-standard-4",
        min_replica_count=MIN_NODES,
        max_replica_count=MAX_NODES
    )


@dsl.pipeline(
    pipeline_root="gs://mlops-cakir-kubeflow-v1/coupon-pipeline-v1",
    name="coupon-model-training-pipeline",
)
def pipeline(
    project: str = "cakir-kubeflow",
    region: str = "us-central1"
    ):
    
    max_depth=5
    learning_rate=0.2
    n_estimators=40
    
    file_name = "gs://mlops-cakir-kubeflow-v1/mlops-recommendation/in-vehicle-coupon-recommendation.csv"
    input_validation_task = validate_input_ds(file_name)
    
    with dsl.Condition(input_validation_task.outputs["input_validation"] == "true"):
        model_training = custom_training_job_component(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
        ).after(input_validation_task)
        
        with dsl.Condition(model_training.outputs["model_validation"] == "true"):
            task_deploy_model = model_deployment().after(model_training)


if __name__ == "__main__":
     # Create an argument parser
    parser = argparse.ArgumentParser(description='Data Drift Script')
    parser.add_argument('--display_name', type=str, help='pipeline display name')
    parser.add_argument('--location', type=str, help='region of pipeline')

    # Parse the command-line arguments
    args = parser.parse_args()
    location = args.location
    display_name = args.display_name

    compiler.Compiler().compile(pipeline_func=pipeline,package_path='coupon-pipeline-deploy-v1.json')

    start_pipeline = pipeline_jobs.PipelineJob(
        display_name=display_name,
        template_path="mlops-recommendation-pipeline-deploy-v1.json",
        enable_caching=False,
        location=args.test,
    )

    start_pipeline.run()