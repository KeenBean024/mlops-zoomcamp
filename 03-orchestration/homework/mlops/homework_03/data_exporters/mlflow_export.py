import mlflow
import mlflow.sklearn
if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

import pickle
# mlflow.set_tracking_uri("sqlite:////home/mlflow/mlflow.db")
# mlflow.set_experiment("hw3-orchestration")

@data_exporter
def export_data(artifacts):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    vectorizer, model = artifacts

    with open("dv.pkl", "wb") as f_out:
        pickle.dump(vectorizer, f_out)

    with mlflow.start_run():
        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",
        registered_model_name="sklearn-linear-reg-model",
        )

        mlflow.log_artifact("dv.pkl", artifact_path="artifacts")
    print("Saved")
