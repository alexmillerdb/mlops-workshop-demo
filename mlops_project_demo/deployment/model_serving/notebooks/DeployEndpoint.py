# Databricks notebook source
import os

notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering==0.2.0 databricks-sdk==0.20.0

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
  model_alias_to_evaluate = "prod"
  model_alias_updated = "Champion"

# COMMAND ----------

import uuid
import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri('databricks-uc')
model_name = f"hyperopt_feature_store"
endpoint_name = f"{catalog}_travel_purchase_predictions_{current_user}"
model_full_name = f"{catalog}.{db}.{model_name}"
model_uri = f"models:/{model_full_name}@{model_alias_updated}"

print(f"Endpoint Name: {endpoint_name}")

# COMMAND ----------

# MAGIC %md ## Publish feature tables as Databricks-managed online tables
# MAGIC
# MAGIC By publishing our tables to a Databricks-managed online table, Databricks will automatically synchronize the data written to your feature store to the realtime backend.
# MAGIC
# MAGIC Apart from Databricks-managed online tables, Databricks also supports different third-party backends. You can find more information about integrating Databricks feature tables with third-party online stores in the links below.
# MAGIC
# MAGIC * AWS dynamoDB ([doc](https://docs.databricks.com/machine-learning/feature-store/online-feature-stores.html))
# MAGIC * Azure cosmosDB [doc](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/online-feature-stores)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Publish the feature store with online table specs

# COMMAND ----------

# DBTITLE 1,Create online feature tables
from deployment.model_serving.deploy_utils import (
    create_online_table,
    wait_for_online_tables,
    online_table_exists,
)
from databricks.sdk import WorkspaceClient

create_online_table(
    spark=spark,
    table_name=f"{catalog}.{db}.user_features",
    pks=["user_id"],
    timeseries_key="ts",
    sync="triggered",
)
create_online_table(
    spark=spark,
    table_name=f"{catalog}.{db}.destination_features",
    pks=["destination_id"],
    timeseries_key="ts",
    sync="triggered",
)
create_online_table(
    spark=spark,
    table_name=f"{catalog}.{db}.destination_location_features",
    pks=["destination_id"],
    timeseries_key=None,
    sync="triggered",
)
create_online_table(
    spark=spark,
    table_name=f"{catalog}.{db}.availability_features",
    pks=["destination_id", "booking_date"],
    timeseries_key="ts",
    sync="triggered",
)

# wait for all the tables to be online
wait_for_online_tables(
    catalog=catalog,
    schema=schema,
    tables=[
        "user_features_online",
        "destination_features_online",
        "destination_location_features_online",
        "availability_features_online",
    ],
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Deploy Serverless Model serving Endpoint
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/feature_store/feature-store-expert-model-serving.png?raw=true" style="float: right" width="500px">
# MAGIC
# MAGIC Once our Model, Function and Online feature store are in Unity Catalog, we can deploy the model as using Databricks Model Serving.
# MAGIC
# MAGIC This will provide a REST API to serve our model in realtime.
# MAGIC
# MAGIC ### Enable model inference via the UI
# MAGIC
# MAGIC After calling `log_model`, a new version of the model is saved. To provision a serving endpoint, follow the steps below.
# MAGIC
# MAGIC 1. Within the Machine Learning menu, click [Serving menu](ml/endpoints) in the left sidebar. 
# MAGIC 2. Create a new endpoint, select the most recent model version from Unity Catalog and start the serverless model serving
# MAGIC
# MAGIC You can use the UI, in this demo We will use the API to programatically start the endpoint:

# COMMAND ----------

from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadSize,
    ServingEndpointDetailed,
    AutoCaptureConfigInput
)
from datetime import timedelta

# Get model version by alias
client = MlflowClient()
model = client.get_model_version_by_alias(
    name=model_full_name, alias=model_alias_updated
)
model_version = model.version
# Create model serving configurations using Databricks SDK
wc = WorkspaceClient()

# for more information see https://databricks-sdk-py.readthedocs.io/en/latest/dbdataclasses/serving.html#databricks.sdk.service.serving
served_models = [
    ServedModelInput(
        model_name=model_full_name,
        model_version=model_version,
        workload_size=ServedModelInputWorkloadSize.SMALL,
        scale_to_zero_enabled=True,
    )
]
auto_capture_config = AutoCaptureConfigInput(
        catalog_name=catalog,
        enabled=True,  # Enable inference tables
        schema_name=schema,
    )

endpoint_config = EndpointCoreConfigInput(served_models=served_models, auto_capture_config=auto_capture_config)

try:
    print(f"Creating endpoint {endpoint_name} with latest version...")
    wc.serving_endpoints.create_and_wait(
        endpoint_name, config=endpoint_config
    )
except Exception as e:
    if "already exists" in str(e):
        print(f"Endpoint exists, updating with latest model version...")
        wc.serving_endpoints.update_config_and_wait(
            endpoint_name, served_models=served_models, auto_capture_config=auto_capture_config
        )
    else:
        raise e

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/feature_store/feature-store-expert-model-serving-inference.png?raw=true" style="float: right" width="700"/>
# MAGIC
# MAGIC Once our model deployed, you can easily test your model using the Model Serving endpoint UI.
# MAGIC
# MAGIC Let's call it using the REST API directly.
# MAGIC
# MAGIC The endpoint will answer in millisec, what will happen under the hood is the following:
# MAGIC
# MAGIC * The endpoint receive the REST api call
# MAGIC * It calls our 4 online table to get the features
# MAGIC * Call the `distance_udf` function to compute the distance
# MAGIC * Call the ML model
# MAGIC * Returns the final answer

# COMMAND ----------

import requests
import json

test_df = spark.table("test_set")
lookup_keys = (
    test_df.drop("purchased")
    .toPandas()
    .sample(n=5)
    .astype({"ts": "str", "booking_date": "str"})
    .fillna(0)
    .to_dict(orient="records")
)

# Get the API endpoint and token for the current notebook context
API_ROOT = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
)
API_TOKEN = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)

data = {"dataframe_records": lookup_keys}
headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

response = requests.post(
    url=f"{API_ROOT}/serving-endpoints/{endpoint_name}/invocations",
    json=data,
    headers=headers,
)

print(json.dumps(response.json()))
