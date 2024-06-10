# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering==0.2.0 databricks-sdk==0.20.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "dev")
dbutils.widgets.text("schema", "mlops_project_demo")
dbutils.widgets.text("table", "dbdemos_fs_travel")
dbutils.widgets.text("reset_all_data", "false")

catalog = dbutils.widgets.get("catalog")
schema = db = dbutils.widgets.get("schema")
table = dbutils.widgets.get("table")
reset_all_data = dbutils.widgets.get("reset_all_data")
if reset_all_data.lower() == "true":
  reset_all_data = True
else:
  reset_all_data = False

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")

# COMMAND ----------

# DBTITLE 1,Setup initialization
from mlops_project_demo.utils.setup_init import DBDemos

dbdemos = DBDemos()
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

# MAGIC %md ## Compute batch features
# MAGIC
# MAGIC Calculate the aggregated features from the vacation purchase logs for destination and users. The destination features include popularity features such as impressions, clicks, and pricing features like price at the time of booking. The user features capture the user profile information such as past purchased price. Because the booking data does not change very often, it can be computed once per day in batch.

# COMMAND ----------

# MAGIC %sql SELECT * FROM travel_purchase 

# COMMAND ----------

# MAGIC %sql SELECT * FROM destination_location

# COMMAND ----------

from mlops_project_demo.feature_engineering.features.helper_functions import (
  write_feature_table, 
  create_user_features, 
  destination_features_fn,
  delete_fss
)
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

if reset_all_data:
  delete_fss(catalog, db, ["user_features", "destination_features", "destination_location_features", "availability_features"])

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

# dbdemos.stop_all_streams_asynch(sleep_time=30)
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
