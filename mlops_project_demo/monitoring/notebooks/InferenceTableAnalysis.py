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

import sys
import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
current_directory = os.getcwd()
root_directory = os.path.normpath(os.path.join(current_directory, '..', '..'))
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
endpoint_name = f"{catalog}_travel_purchase_predictions_{current_user}"

# COMMAND ----------

import requests
from typing import Dict
from pyspark.sql import functions as F

def get_endpoint_status(endpoint_name: str) -> Dict:
    # Fetch the PAT token to send in the API request
    workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{workspace_url}/api/2.0/serving-endpoints/{endpoint_name}", json={"name": endpoint_name}, headers=headers).json()

    # Verify that Inference Tables is enabled.
    if "auto_capture_config" not in response.get("config", {}) or not response["config"]["auto_capture_config"]["enabled"]:
        raise Exception(f"Inference Tables is not enabled for endpoint {endpoint_name}. \n"
                        f"Received response: {response} from endpoint.\n"
                        "Please create an endpoint with Inference Tables enabled before running this notebook.")

    return response

response = get_endpoint_status(endpoint_name=endpoint_name)

# COMMAND ----------

auto_capture_config = response["config"]["auto_capture_config"]
catalog = auto_capture_config["catalog_name"]
schema = auto_capture_config["schema_name"]
# These values should not be changed - if they are, the monitor will not be accessible from the endpoint page.
payload_table_name = auto_capture_config["state"]["payload_table"]["name"]
payload_table_name = f"`{catalog}`.`{schema}`.`{payload_table_name}`"
print(f"Endpoint {endpoint_name} configured to log payload in table {payload_table_name}")

processed_table_name = f"{auto_capture_config['table_name_prefix']}_processed"
processed_table_name = f"`{catalog}`.`{schema}`.`{processed_table_name}`"
print(f"Processed requests with evaluation metrics will be saved to: {processed_table_name}")

# COMMAND ----------

# MAGIC %md Define the Json Path to extract the input and output values

# COMMAND ----------

from pyspark.sql.functions import col, from_json, explode, posexplode, monotonically_increasing_id, arrays_zip, concat, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType

# Create DataFrame
df = spark.table(payload_table_name) \
  .where('status_code == 200')

# Define the schema for the request JSON column
request_schema = StructType([
    StructField("dataframe_records", ArrayType(StructType([
        StructField("ts", StringType(), True),
        StructField("destination_id", IntegerType(), True),
        StructField("user_id", IntegerType(), True),
        StructField("user_latitude", DoubleType(), True),
        StructField("user_longitude", DoubleType(), True),
        StructField("booking_date", StringType(), True)
    ])))
])

# Define the schema for the response JSON column
response_schema = StructType([
    StructField("predictions", ArrayType(IntegerType()), True)
])

# Parse the JSON columns
df_parsed_request = (df
    .withColumn("parsed_request", from_json(col("request"), request_schema)) 
    .withColumn("parsed_response", from_json(col("response"), response_schema)) 
    .withColumn("__db_request_response", explode(arrays_zip(col("parsed_request.dataframe_records").alias("input"), col("parsed_response.predictions").alias("prediction"))))
    .select(
        "databricks_request_id",
        "date",
        (col("timestamp_ms") / 1000).alias("timestamp"),
        "execution_time_ms",
        "__db_request_response.input.*",
        "__db_request_response.prediction",
        concat(
            col("request_metadata").getItem("model_name"),
            lit("_"),
            col("request_metadata").getItem("model_version")).alias("model_id")
    )
)

display(df_parsed_request)

# COMMAND ----------

# DBTITLE 1,Write Table and Enable Change Data Feed
# Write delta table, in real-life scenario most likely change to streaming
df_parsed_request.write.format("delta").mode("overwrite").saveAsTable("inference_table_analysis")
spark.sql(f"ALTER TABLE inference_table_analysis SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")

# COMMAND ----------

# MAGIC %md ## Monitor inference table:
# MAGIC - Create monitor on inference table using `create_monitor` API. If monitor already exists, pass same parameters to `update_monitor`.
# MAGIC - More information: https://docs.databricks.com/en/lakehouse-monitoring/create-monitor-api.html

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType, MonitorInfoStatus, MonitorRefreshInfoState, MonitorMetric

w = WorkspaceClient()

# COMMAND ----------

TABLE_NAME = f"{catalog}.{schema}.inference_table_analysis"
BASELINE_TABLE = f"{catalog}.{schema}.training_baseline_predictions"
BASELINE_TABLE = None
MODEL_NAME = "hyperopt_feature_store"
TIMESTAMP_COL = "timestamp"
MODEL_ID_COL = "model_id" # Name of column to use as model identifier (here we'll use the model_name+version)
PREDICTION_COL = "prediction"  # What to name predictions in the generated tables
LABEL_COL = None # Name of ground-truth labels column
# ID_COL = "ID" # [OPTIONAL] only used for joining labels

# COMMAND ----------

help(w.quality_monitors.create)

# COMMAND ----------

# ML problem type, either "classification" or "regression"
PROBLEM_TYPE = MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION

# Window sizes to analyze data over
GRANULARITIES = ["1 day"]   

# Directory to store generated dashboard
username = spark.sql("SELECT current_user()").first()["current_user()"]
ASSETS_DIR = f"/Workspace/Users/{username}/databricks_lakehouse_monitoring/{TABLE_NAME}"

# Optional parameters
SLICING_EXPRS = None                   # Expressions to slice data with
CUSTOM_METRICS = None                  # A list of custom metrics to compute

# COMMAND ----------

print(f"Creating monitor for {TABLE_NAME}")

info = w.quality_monitors.create(
  table_name=TABLE_NAME,
  inference_log=MonitorInferenceLog(
    timestamp_col=TIMESTAMP_COL,
    granularities=GRANULARITIES,
    model_id_col=MODEL_ID_COL, # Model version number 
    prediction_col=PREDICTION_COL,
    problem_type=PROBLEM_TYPE,
    label_col=LABEL_COL # Optional
  ),
  baseline_table_name=BASELINE_TABLE,
  slicing_exprs=SLICING_EXPRS,
  output_schema_name=f"{catalog}.{schema}",
  assets_dir=ASSETS_DIR
)

# COMMAND ----------

import time

# Wait for monitor to be created
while info.status ==  MonitorInfoStatus.MONITOR_STATUS_PENDING:
  info = w.quality_monitors.get(table_name=TABLE_NAME)
  time.sleep(10)

assert info.status == MonitorInfoStatus.MONITOR_STATUS_ACTIVE, "Error creating monitor"
