# Databricks notebook source
import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "dev")
dbutils.widgets.text("reset_all_data", "false")

catalog = dbutils.widgets.get("catalog")
reset_all_data = dbutils.widgets.get("reset_all_data")
if reset_all_data.lower() == "true":
  reset_all_data = True
else:
  reset_all_data = False

print(f"Catalog: {catalog}")

# COMMAND ----------

from mlops_project_demo.utils.setup_init import DBDemos

dbdemos = DBDemos()
current_user = dbdemos.get_username()
schema = db = f'mlops_project_demo_{current_user}'
experiment_path = f"/Shared/mlops-workshop/experiments/hyperopt-feature-store-{current_user}"
dbdemos.setup_schema(catalog, db, reset_all_data=reset_all_data)

# COMMAND ----------

if catalog.lower() == "dev":
  model_alias_to_evaluate = "Dev"
  model_alias_updated = "Staging"
if catalog.lower() == "staging":
  model_alias_to_evaluate = "Staging"
  model_alias_updated = "Challenger"
if catalog.lower() == "prod":
  model_alias_to_evaluate = "Challenger"
  model_alias_updated = "Champion"

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri('databricks-uc')
model_name = f"hyperopt_feature_store"
model_full_name = f"{catalog}.{db}.{model_name}"

def get_latest_model_version(model_name):
    mlflow_client = MlflowClient()
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version
  
client = MlflowClient()
model_version = client.get_model_version_by_alias(model_full_name, model_alias_to_evaluate)
model_uri = f"models:/{model_full_name}@{model_alias_to_evaluate}"
print(f"Model URI to evaluate: {model_uri}")

# COMMAND ----------

# DBTITLE 1,Run validation steps and update tags
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F

fe = FeatureEngineeringClient()

# Select the feature table cols by model input schema
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)
input_column_names = loaded_model.metadata.get_input_schema().input_names()

# load validation set which will be used to score the model
validation_set = spark.table("test_set")

str_model_version_to_evaluate = str(model_version.version)

# Predict on a Spark DataFrame
try:
    output_df = fe.score_batch(model_uri=model_uri, df=validation_set) \
      .withColumn("target", F.when(F.col("purchased"), F.lit(1)).otherwise(F.lit(0)))
    display(output_df)
    client.set_model_version_tag(
        name=model_full_name,
        version=str_model_version_to_evaluate,
        key="predicts", 
        value=1  # convert boolean to string
    )
except Exception: 
    print("Unable to predict on features.")
    client.set_model_version_tag(
        name=model_full_name,
        version=str_model_version_to_evaluate,
        key="predicts", 
        value=0  # convert boolean to string
    )
    pass

if not loaded_model.metadata.signature:
    print("This model version is missing a signature.  Please push a new version with a signature!  See https://mlflow.org/docs/latest/models.html#model-metadata for more details.")
    # Update UC tag to note missing signature
    client.set_model_version_tag(
        name=model_full_name,
        version=str_model_version_to_evaluate,
        key="has_signature",
        value=0
    )
else:
    # Update UC tag to note existence of signature
    client.set_model_version_tag(
        name=model_full_name,
        version=str_model_version_to_evaluate,
        key="has_signature",
        value=1
    )

if not model_version.description:
    # Update UC tag to note lack of description
    client.set_model_version_tag(
        name=model_full_name,
        version=str_model_version_to_evaluate,
        key="has_description",
        value=0
    )
    print("Did you forget to add a description?")
elif not len(model_version.description) > 2:
    # Update UC tag to note description is too basic
    client.set_model_version_tag(
        name=model_full_name,
        version=str_model_version_to_evaluate,
        key="has_description",
        value=0
    )
    print("Your description is too basic, sorry.  Please resubmit with more detail (40 char min).")
else:
    # Update UC tag to note presence and sufficiency of description
    client.set_model_version_tag(
        name=model_full_name,
        version=str_model_version_to_evaluate,
        key="has_description",
        value=1
    )

# COMMAND ----------

# DBTITLE 1,Run validation steps on artifacts
import os

# Create local directory 
local_dir = "/tmp/model_artifacts"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)

# Download artifacts from tracking server - no need to specify DBFS path here
local_path = client.download_artifacts(model_version.run_id, "", local_dir)

# Tag model version as possessing artifacts or not
if not os.listdir(local_path):
  client.set_model_version_tag(
        name=model_full_name,
        version=str_model_version_to_evaluate,
        key="has_artifacts",
        value=0
    )
  print("There are no artifacts associated with this model.  Please include some data visualization or data profiling.  MLflow supports HTML, .png, and more.")
else:
  client.set_model_version_tag(
        name=model_full_name,
        version=str_model_version_to_evaluate,
        key="has_artifacts",
        value=1
    )
  print("Artifacts downloaded in: {}".format(local_path))
  print("Artifacts: {}".format(os.listdir(local_path)))

# COMMAND ----------

# DBTITLE 1,Programatically pull the updated model version tags
results = client.get_model_version(model_full_name, str_model_version_to_evaluate)
targets = loaded_model.metadata.get_output_schema().input_names()
model_type = results.tags['model_type']
baseline_model_uri = "models:/" + model_name + "@Champion"
evaluators = "default"
baseline_model_uri = "models:/" + model_full_name + "@Champion"

# COMMAND ----------

from mlflow.models import MetricThreshold
import matplotlib.pyplot as plt
import numpy as np

def prediction_target_scatter(eval_df, _builtin_metrics, artifacts_dir):
    """
    This example custom artifact generates and saves a scatter plot to ``artifacts_dir`` that
    visualizes the relationship between the predictions and targets for the given model to a
    file as an image artifact.
    """
    plt.scatter(eval_df["prediction"], eval_df["target"])
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.title("Targets vs. Predictions")
    plot_path = os.path.join(artifacts_dir, "example_scatter_plot.png")
    plt.savefig(plot_path)
    return {"example_scatter_plot_artifact": plot_path}
  
# helper methods
def get_run_link(run_info):
    return "[Run](#mlflow/experiments/{0}/runs/{1})".format(
        run_info.experiment_id, run_info.run_id
    )

def validation_thresholds():
    return {
        "f1_score": MetricThreshold(
            threshold=0.70,  # mean_squared_error should be <= 20
            # min_absolute_change=0.01,  # mean_squared_error should be at least 0.01 greater than baseline model accuracy
            # min_relative_change=0.01,  # mean_squared_error should be at least 1 percent greater than baseline model accuracy
            higher_is_better=True,
        ),
    }


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

# COMMAND ----------

import tempfile
import traceback

training_run = get_training_run(model_full_name, str_model_version_to_evaluate)
validation_thresholds_fn = validation_thresholds()
eval_data = output_df.toPandas()

# run evaluate
with mlflow.start_run(
    run_name=generate_run_name(training_run),
    description=generate_description(training_run),
) as run, tempfile.TemporaryDirectory() as tmp_dir:
    validation_thresholds_file = os.path.join(tmp_dir, "validation_thresholds.txt")
    with open(validation_thresholds_file, "w") as f:
        if validation_thresholds_fn:
            for metric_name in validation_thresholds_fn:
                f.write(
                    "{0:30}  {1}\n".format(
                        metric_name, str(validation_thresholds_fn[metric_name])
                    )
                )
    mlflow.log_artifact(validation_thresholds_file)

    try:
        eval_result = mlflow.evaluate(
            data=eval_data,
            targets=targets[0],
            predictions="prediction",
            model_type=model_type,
            evaluators=evaluators,
            validation_thresholds=validation_thresholds_fn
        )
        
        mlflow.log_metrics(eval_result.metrics)
        # Update alias to indicate model version has passed validation checks
        assert '0' not in results or 'fail' not in results, "The condition '0' in results or 'fail' in results should be True"
        print(f"Validation checks passed. Assigning {model_alias_updated} alias to model version {str_model_version_to_evaluate}.")
        client.set_registered_model_alias(model_name, model_alias_updated, str_model_version_to_evaluate)
        client.delete_registered_model_alias(model_name, model_alias_to_evaluate)
        
    except Exception as err:
        log_to_model_description(run, False)
        error_file = os.path.join(tmp_dir, "error.txt")
        with open(error_file, "w") as f:
            f.write("Validation failed : " + str(err) + "\n")
            f.write(traceback.format_exc())
        mlflow.log_artifact(error_file)

# COMMAND ----------

mlflow.log_metrics(eval_result.metrics)
