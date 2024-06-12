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

# DBTITLE 1,Setup initialization
from utils.setup_init import DBDemos

dbdemos = DBDemos()
current_user = dbdemos.get_username()
schema = db = f'mlops_project_demo_{current_user}'
experiment_path = f"/Shared/mlops-workshop/experiments/hyperopt-feature-store-{current_user}"
dbdemos.setup_schema(catalog, db, reset_all_data=reset_all_data)

# COMMAND ----------

# DBTITLE 1,Setup feature datasets
from pyspark.sql import functions as F

if not spark.catalog.tableExists("destination_location"):
  print(f"Creating destination location table")
  destination_location_df = spark.read.option("inferSchema", "true").load("/databricks-datasets/travel_recommendations_realtime/raw_travel_data/fs-demo_destination-locations/",  format="csv", header="true")
  destination_location_df.write.mode('overwrite').saveAsTable('destination_location')

if not spark.catalog.tableExists("travel_purchase"):
  print(f"Creating travel purchase table")
  travel_purchase_df = spark.read.option("inferSchema", "true").load("/databricks-datasets/travel_recommendations_realtime/raw_travel_data/fs-demo_vacation-purchase_logs/", format="csv", header="true")
  travel_purchase_df = travel_purchase_df.withColumn("id", F.monotonically_increasing_id())
  travel_purchase_df.withColumn("booking_date", F.col("booking_date").cast('date')).write.mode('overwrite').saveAsTable('travel_purchase')

if not spark.catalog.tableExists("destination_location"):
  print(f"Creating destination location table")
  destination_location_df = spark.read.option("inferSchema", "true").load("/databricks-datasets/travel_recommendations_realtime/raw_travel_data/fs-demo_destination-locations/",  format="csv", header="true")
  destination_location_df.write.mode('overwrite').saveAsTable('destination_location')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 1: Create the feature tables
# MAGIC
# MAGIC The first step is to create our feature store tables. We'add a new datasource that we'll consume in streaming, making sure our Feature Table is refreshed in near realtime.
# MAGIC
# MAGIC In addition, we'll compute the "on-demande" feature (distance between the user and a destination, booking time) using the pandas API during training, this will allow us to use the same code for realtime inferences.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/feature_store/feature-store-expert-flow-training.png?raw=true" width="1200px"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute streaming features
# MAGIC
# MAGIC Availability of the destination can hugely affect the prices. Availability can change frequently especially around the holidays or long weekends during busy season. This data has a freshness requirement of every few minutes, so we use Spark structured streaming to ensure data is fresh when doing model prediction. 

# COMMAND ----------

# MAGIC %md <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/realtime/streaming.png"/>

# COMMAND ----------

# DBTITLE 1,Create Volume to store schema location
# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS feature_store_volume

# COMMAND ----------

destination_availability_stream = (
  spark.readStream
  .format("cloudFiles")
  .option("cloudFiles.format", "json") #Could be "kafka" to consume from a message queue
  .option("cloudFiles.inferSchema", "true")
  .option("cloudFiles.inferColumnTypes", "true")
  .option("cloudFiles.schemaEvolutionMode", "rescue")
  .option("cloudFiles.schemaHints", "event_ts timestamp, booking_date date, destination_id int")
  .option("cloudFiles.schemaLocation", f"/Volumes/{catalog}/{db}/feature_store_volume/stream/availability_schema")
  .option("cloudFiles.maxFilesPerTrigger", 1000) #Simulate streaming
  .load("/databricks-datasets/travel_recommendations_realtime/raw_travel_data/fs-demo_destination-availability_logs/json")
  .drop("_rescued_data")
  .withColumnRenamed("event_ts", "ts")
)

dbdemos.wait_for_all_stream()
# dbdemos.stop_all_streams_asynch(sleep_time=30)
# display(destination_availability_stream)

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

stream_table_name = f"{catalog}.{db}.availability_features"
if not spark.catalog.tableExists(stream_table_name):
  fe.create_table(
      name=stream_table_name, 
      primary_keys=["destination_id", "booking_date", "ts"],
      timestamp_keys=["ts"],
      schema=destination_availability_stream.schema,
      description="Destination Availability Features"
  )

# Now write the data to the feature table in "merge" mode using a stream
fe.write_table(
    name=stream_table_name, 
    df=destination_availability_stream,
    mode="merge",
    checkpoint_location= f"/Volumes/{catalog}/{db}/feature_store_volume/stream/availability_checkpoint",
    trigger={'once': True} #Refresh the feature store table once, or {'processingTime': '1 minute'} for every minute-
)
