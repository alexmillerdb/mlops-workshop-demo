# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering==0.2.0 databricks-sdk==0.20.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "dev")
dbutils.widgets.text("schema", "mlops_project_demo")
dbutils.widgets.text("table", "dbdemos_fs_travel")

catalog = dbutils.widgets.get("catalog")
schema = db = dbutils.widgets.get("schema")
table = dbutils.widgets.get("table")

# COMMAND ----------

# DBTITLE 1,Set the notebook path directory and call utils folder
import os
path_list = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")
dir_path = "/Workspace/" + "/".join(path_list[:-3])
util_path = dir_path + "/utils"
init_path = util_path + "/00-init-expert"

%cd $dir_path
%cd $util_path

# COMMAND ----------

# DBTITLE 1,Run Init Notebook - TO DO create Python Package instead
dbutils.notebook.run(path=init_path, timeout_seconds=3600, arguments={"catalog": catalog, "schema": schema, "table": table})

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

# MAGIC %md ## Compute batch features
# MAGIC
# MAGIC Calculate the aggregated features from the vacation purchase logs for destination and users. The destination features include popularity features such as impressions, clicks, and pricing features like price at the time of booking. The user features capture the user profile information such as past purchased price. Because the booking data does not change very often, it can be computed once per day in batch.

# COMMAND ----------

# MAGIC %sql SELECT * FROM travel_purchase 

# COMMAND ----------

# MAGIC %sql SELECT * FROM destination_location

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

def write_feature_table(name, df, primary_keys, description, timestamp_keys=None, mode="merge"):
  if not spark.catalog.tableExists(name):
    print(f"Feature table {name} does not exist. Creating feature table")
    fe.create_table(name=name,
                    primary_keys=primary_keys, 
                    timestamp_keys=timestamp_keys, 
                    df=df, 
                    description=description)
  else:
    print(f"Feature table {name} exists, writing updated results with mode {mode}")
    fe.write_table(
      df=df,
      name=name,
      mode=mode
    )


# COMMAND ----------

#Delete potential existing tables to reset all the demo
# delete_fss(catalog, db, ["user_features", "destination_features", "destination_location_features", "availability_features"])

from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

# For more details these functions are available under ./utils/00-init-expert
user_features_df = create_user_features(spark.table('travel_purchase'))
write_feature_table(
  name=f"{catalog}.{db}.user_features",
  primary_keys=["user_id", "ts"],
  timestamp_keys="ts",
  df=user_features_df,
  description="User Features")

destination_features_df = destination_features_fn(spark.table('travel_purchase'))
write_feature_table(
  name=f"{catalog}.{db}.destination_features",
  primary_keys=["destination_id", "ts"], 
  timestamp_keys="ts", 
  df=destination_features_df, 
  description="Destination Popularity Features")


#Add the destination location dataset
destination_location = spark.table("destination_location")
write_feature_table(
  name=f"{catalog}.{db}.destination_location_features", 
  primary_keys="destination_id", 
  df=destination_location, 
  description="Destination location features.")

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
  .option("cloudFiles.maxFilesPerTrigger", 100) #Simulate streaming
  .load("/databricks-datasets/travel_recommendations_realtime/raw_travel_data/fs-demo_destination-availability_logs/json")
  .drop("_rescued_data")
  .withColumnRenamed("event_ts", "ts")
)

DBDemos.stop_all_streams_asynch(sleep_time=30)
display(destination_availability_stream)

# COMMAND ----------

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
