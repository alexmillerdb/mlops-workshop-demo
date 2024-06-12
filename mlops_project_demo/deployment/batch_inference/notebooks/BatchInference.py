# Databricks notebook source
import os

notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install -r ../../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "dev")
dbutils.widgets.text("reset_all_data", "false")
dbutils.widgets.text("input_table_name", "travel_purchase", label="Input Table Name")
dbutils.widgets.text("output_table_name", "travel_purchase_prediction", label="Output Table Name")

catalog = dbutils.widgets.get("catalog")
reset_all_data = dbutils.widgets.get("reset_all_data")
input_table_name = dbutils.widgets.get("input_table_name")
output_table_name = dbutils.widgets.get("output_table_name")

if reset_all_data.lower() == "true":
  reset_all_data = True
else:
  reset_all_data = False

print(f"Catalog: {catalog}")

# COMMAND ----------

import sys
import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
current_directory = os.getcwd()
root_directory = os.path.normpath(os.path.join(current_directory, '..', '..', '..'))
%cd $notebook_path
%cd ..
sys.path.append("../..")
sys.path.append(root_directory)

# COMMAND ----------

from utils.setup_init import DBDemos

dbdemos = DBDemos()
current_user = dbdemos.get_username()
schema = db = f'mlops_project_demo_{current_user}'
input_table_name = f"{catalog}.{db}.{input_table_name}"
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
model_uri = f"models:/{model_full_name}@{model_alias_updated}"

# COMMAND ----------

from mlflow import MlflowClient

# Get model version from alias
client = MlflowClient(registry_uri="databricks-uc")
model_version = client.get_model_version_by_alias(model_full_name, model_alias_updated).version

# COMMAND ----------

# Get datetime
from datetime import datetime

ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# COMMAND ----------


from predict import predict_batch

predict_batch(spark, model_uri, input_table_name, output_table_name, model_version, ts)
dbutils.notebook.exit(output_table_name)
