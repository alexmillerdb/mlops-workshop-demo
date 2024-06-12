import numpy as np
import mlflow
from mlflow import MlflowClient
from mlflow.models import make_metric, MetricThreshold

client = MlflowClient()

# Custom metrics to be included. Return empty list if custom metrics are not needed.
# Please refer to custom_metrics parameter in mlflow.evaluate documentation https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
# TODO(optional) : custom_metrics
def custom_metrics():

    # TODO(optional) : define custom metric function to be included in custom_metrics.
    def squared_diff_plus_one(eval_df, _builtin_metrics):
        """
        This example custom metric function creates a metric based on the ``prediction`` and
        ``target`` columns in ``eval_df`.
        """
        return np.sum(np.abs(eval_df["prediction"] - eval_df["target"] + 1) ** 2)

    return [make_metric(eval_fn=squared_diff_plus_one, greater_is_better=False)]


# Define model validation rules. Return empty dict if validation rules are not needed.
# Please refer to validation_thresholds parameter in mlflow.evaluate documentation https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
# TODO(optional) : validation_thresholds
def validation_thresholds():
    return {
        "f1_score": MetricThreshold(
            threshold=0.70,  # mean_squared_error should be <= 20
            # min_absolute_change=0.01,  # mean_squared_error should be at least 0.01 greater than baseline model accuracy
            # min_relative_change=0.01,  # mean_squared_error should be at least 1 percent greater than baseline model accuracy
            higher_is_better=True,
        ),
    }


# Define evaluator config. Return empty dict if validation rules are not needed.
# Please refer to evaluator_config parameter in mlflow.evaluate documentation https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
# TODO(optional) : evaluator_config
def evaluator_config():
    return {}

def get_training_run(model_name, model_version):
    version = client.get_model_version(model_name, model_version)
    return mlflow.get_run(run_id=version.run_id)


def generate_run_name(training_run):
    return None if not training_run else training_run.info.run_name + "-validation"


def generate_description(training_run):
    return (
        None
        if not training_run
        else "Model Training Details: {0}\n".format(get_run_link(training_run.info))
    )


def log_to_model_description(run, success):
    run_link = get_run_link(run.info)
    description = client.get_model_version(model_name, model_version).description
    status = "SUCCESS" if success else "FAILURE"
    if description != "":
        description += "\n\n---\n\n"
    description += "Model Validation Status: {0}\nValidation Details: {1}".format(
        status, run_link
    )
    client.update_model_version(
        name=model_name, version=model_version, description=description
    )
