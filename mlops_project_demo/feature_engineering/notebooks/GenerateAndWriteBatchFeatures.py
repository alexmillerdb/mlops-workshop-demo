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

# MAGIC %md ## Compute batch features
# MAGIC
# MAGIC Calculate the aggregated features from the vacation purchase logs for destination and users. The destination features include popularity features such as impressions, clicks, and pricing features like price at the time of booking. The user features capture the user profile information such as past purchased price. Because the booking data does not change very often, it can be computed once per day in batch.

# COMMAND ----------

# MAGIC %sql SELECT * FROM travel_purchase 

# COMMAND ----------

# MAGIC %sql SELECT * FROM destination_location

# COMMAND ----------

from feature_engineering.features.helper_functions import (
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
