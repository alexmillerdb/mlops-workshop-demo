# Databricks notebook source
# MAGIC %md 
# MAGIC Do not edit the notebook, it contains import and helpers for the demo

# COMMAND ----------

# %run ./00-init-basic $catalog=catalog $schema=schema $table=table

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering==0.2.0 databricks-sdk==0.20.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")
reset_all_data = dbutils.widgets.get("reset_all_data") == "true"
dbutils.widgets.text("catalog", "dev")
dbutils.widgets.text("schema", "mlops_project_demo")
dbutils.widgets.text("table", "dbdemos_fs_travel")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
table = dbutils.widgets.get("table")

# COMMAND ----------

main_naming = table
dbName = db = schema

# COMMAND ----------

from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadSize,
    ServingEndpointDetailed,
)

# COMMAND ----------

# *****
# Loading Modules
# *****
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pyspark.sql.window as w
import timeit
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

import uuid
import os
import requests

from pyspark.sql.functions import lit, expr, rand, col, count, mean, unix_timestamp, window, when
from pyspark.sql.types import StringType, DoubleType, IntegerType, LongType
import numpy as np

#Add features from the time variable 
def add_time_features_spark(df):
    return add_time_features(df.pandas_api()).to_spark()

def add_time_features(df):
    # Extract day of the week, day of the month, and hour from the ts column
    df['day_of_week'] = df['ts'].dt.dayofweek
    df['day_of_month'] = df['ts'].dt.day
    df['hour'] = df['ts'].dt.hour
    
    # Calculate sin and cos values for the day of the week, day of the month, and hour
    df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
    df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
    df['day_of_month_sin'] = np.sin(df['day_of_month'] * (2 * np.pi / 30))
    df['day_of_month_cos'] = np.cos(df['day_of_month'] * (2 * np.pi / 30))
    df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
    df = df.drop(['day_of_week', 'day_of_month', 'hour'], axis=1)
    return df

# COMMAND ----------

def online_table_exists(table_name):
    w = WorkspaceClient()
    try:
        w.online_tables.get(name=table_name)
        return True
    except Exception as e:
        print(str(e))
        return 'already exists' in str(e)
    return False
  
def wait_for_online_tables(catalog, schema, tables, waiting_time = 300):
    sleep_time = 10
    import time
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    for table in tables:
        for i in range(int(waiting_time/sleep_time)):
            state = w.online_tables.get(name=f"{catalog}.{db}.{table}").status.detailed_state.value
            if state.startswith('ONLINE'):
                print(f'Table {table} online: {state}')
                break
            time.sleep(sleep_time)
            
def delete_fs(fs_table_name):
  print("Deleting Feature Table", fs_table_name)
  try:
    fs = feature_store.FeatureStoreClient()
    fs.drop_table(name=fs_table_name)
    spark.sql(f"DROP TABLE IF EXISTS {fs_table_name}")
  except Exception as e:
    print("Can't delete table, likely not existing: "+str(e))  

def delete_fss(catalog, db, tables):
  for table in tables:
    delete_fs(f"{catalog}.{db}.{table}")

def get_last_model_version(model_full_name):
    mlflow_client = MlflowClient(registry_uri="databricks-uc")
    # Use the MlflowClient to get a list of all versions for the registered model in Unity Catalog
    all_versions = mlflow_client.search_model_versions(f"name='{model_full_name}'")
    # Sort the list of versions by version number and get the latest version
    latest_version = max([int(v.version) for v in all_versions])
    # Use the MlflowClient to get the latest version of the registered model in Unity Catalog
    return mlflow_client.get_model_version(model_full_name, str(latest_version))


# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.model_selection import train_test_split
import mlflow
from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
import databricks.automl_runtime


import lightgbm
from lightgbm import LGBMClassifier
import pandas as pd

params = {
  "colsample_bytree": 0.40,
  "lambda_l1": 0.15,
  "lambda_l2": 2.1,
  "learning_rate": 4.1,
  "max_bin": 14,
  "max_depth": 12,
  "min_child_samples": 182,
  "n_estimators": 100,
  "num_leaves": 790,
  "path_smooth": 68.0,
  "subsample": 0.52,
  "random_state": 607,
}

# COMMAND ----------

import requests
import collections
import os


class DBDemos():
  @staticmethod
  def setup_schema(catalog, db, reset_all_data, volume_name = None):

    if reset_all_data:
      print(f'clearing up volume named `{catalog}`.`{db}`.`{volume_name}`')
      spark.sql(f"DROP VOLUME IF EXISTS `{catalog}`.`{db}`.`{volume_name}`")
      spark.sql(f"DROP SCHEMA IF EXISTS `{catalog}`.`{db}` CASCADE")

    def use_and_create_db(catalog, dbName, cloud_storage_path = None):
      print(f"USE CATALOG `{catalog}`")
      spark.sql(f"USE CATALOG `{catalog}`")
      spark.sql(f"""create database if not exists `{dbName}` """)

    assert catalog not in ['hive_metastore', 'spark_catalog'], "This demo only support Unity. Please change your catalog name."
    #If the catalog is defined, we force it to the given value and throw exception if not.
    current_catalog = spark.sql("select current_catalog()").collect()[0]['current_catalog()']
    if current_catalog != catalog:
      catalogs = [r['catalog'] for r in spark.sql("SHOW CATALOGS").collect()]
      if catalog not in catalogs:
        spark.sql(f"CREATE CATALOG IF NOT EXISTS `{catalog}`")
        if catalog == 'dbdemos':
          spark.sql(f"ALTER CATALOG `{catalog}` OWNER TO `account users`")
    use_and_create_db(catalog, db)

    if catalog == 'dbdemos':
      try:
        spark.sql(f"GRANT CREATE, USAGE on DATABASE `{catalog}`.`{db}` TO `account users`")
        spark.sql(f"ALTER SCHEMA `{catalog}`.`{db}` OWNER TO `account users`")
        for t in spark.sql(f'SHOW TABLES in {catalog}.{db}').collect():
          try:
            spark.sql(f'GRANT ALL PRIVILEGES ON TABLE {catalog}.{db}.{t["tableName"]} TO `account users`')
            spark.sql(f'ALTER TABLE {catalog}.{db}.{t["tableName"]} OWNER TO `account users`')
          except Exception as e:
            if "NOT_IMPLEMENTED.TRANSFER_MATERIALIZED_VIEW_OWNERSHIP" not in str(e) and "STREAMING_TABLE_OPERATION_NOT_ALLOWED.UNSUPPORTED_OPERATION" not in str(e) :
              print(f'WARN: Couldn t set table {catalog}.{db}.{t["tableName"]} owner to account users, error: {e}')
      except Exception as e:
        print("Couldn't grant access to the schema to all users:"+str(e))    

    print(f"using catalog.database `{catalog}`.`{db}`")
    spark.sql(f"""USE `{catalog}`.`{db}`""")    

    if volume_name:
      spark.sql(f'CREATE VOLUME IF NOT EXISTS {volume_name};')

                     
  #Return true if the folder is empty or does not exists
  @staticmethod
  def is_folder_empty(folder):
    try:
      return len(dbutils.fs.ls(folder)) == 0
    except:
      return True
    
  @staticmethod
  def is_any_folder_empty(folders):
    return any([DBDemos.is_folder_empty(f) for f in folders])
    
  @staticmethod
  def download_file_from_git(dest, owner, repo, path):
    def download_file(url, destination):
      local_filename = url.split('/')[-1]
      # NOTE the stream=True parameter below
      with requests.get(url, stream=True) as r:
        r.raise_for_status()
        print('saving '+destination+'/'+local_filename)
        with open(destination+'/'+local_filename, 'wb') as f:
          for chunk in r.iter_content(chunk_size=8192): 
            # If you have chunk encoded response uncomment if
            # and set chunk_size parameter to None.
            #if chunk: 
            f.write(chunk)
      return local_filename

    if not os.path.exists(dest):
      os.makedirs(dest)
    from concurrent.futures import ThreadPoolExecutor
    files = requests.get(f'https://api.github.com/repos/{owner}/{repo}/contents{path}').json()
    files = [f['download_url'] for f in files if 'NOTICE' not in f['name']]
    def download_to_dest(url):
      download_file(url, dest)
    with ThreadPoolExecutor(max_workers=10) as executor:
      collections.deque(executor.map(download_to_dest, files))

          

  #force the experiment to the field demos one. Required to launch as a batch
  @staticmethod
  def init_experiment_for_batch(demo_name, experiment_name):
    import mlflow
    #You can programatically get a PAT token with the following
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    xp_root_path = f"/Shared/dbdemos/experiments/{demo_name}"
    try:
      r = w.workspace.mkdirs(path=xp_root_path)
    except Exception as e:
      print(f"ERROR: couldn't create a folder for the experiment under {xp_root_path} - please create the folder manually or  skip this init (used for job only: {e})")
      raise e
    xp = f"{xp_root_path}/{experiment_name}"
    print(f"Using common experiment under {xp}")
    mlflow.set_experiment(xp)
    DBDemos.set_experiment_permission(xp)
    return mlflow.get_experiment_by_name(xp)

  @staticmethod
  def set_experiment_permission(experiment_path):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service import iam
    w = WorkspaceClient()
    try:
      status = w.workspace.get_status(experiment_path)
      w.permissions.set("experiments", request_object_id=status.object_id,  access_control_list=[
                            iam.AccessControlRequest(group_name="users", permission_level=iam.PermissionLevel.CAN_MANAGE)])    
    except Exception as e:
      print(f"error setting up shared experiment {experiment_path} permission: {e}")

    print(f"Experiment on {experiment_path} was set public")


  @staticmethod
  def get_active_streams(start_with = ""):
    return [s for s in spark.streams.active if len(start_with) == 0 or (s.name is not None and s.name.startswith(start_with))]

  @staticmethod
  def stop_all_streams_asynch(start_with = "", sleep_time=0):
    import threading
    def stop_streams():
        DBDemos.stop_all_streams(start_with=start_with, sleep_time=sleep_time)

    thread = threading.Thread(target=stop_streams)
    thread.start()

  @staticmethod
  def stop_all_streams(start_with = "", sleep_time=0):
    import time
    time.sleep(sleep_time)
    streams = DBDemos.get_active_streams(start_with)
    if len(streams) > 0:
      print(f"Stopping {len(streams)} streams")
      for s in streams:
          try:
              s.stop()
          except:
              pass
      print(f"All stream stopped {'' if len(start_with) == 0 else f'(starting with: {start_with}.)'}")

  @staticmethod
  def wait_for_all_stream(start = ""):
    import time
    actives = DBDemos.get_active_streams(start)
    if len(actives) > 0:
      print(f"{len(actives)} streams still active, waiting... ({[s.name for s in actives]})")
    while len(actives) > 0:
      spark.streams.awaitAnyTermination()
      time.sleep(1)
      actives = DBDemos.get_active_streams(start)
    print("All streams completed.")

  @staticmethod
  def get_last_experiment(demo_name, experiment_path = "/Shared/dbdemos/experiments/"):
    import requests
    import re
    from datetime import datetime
    #TODO: waiting for https://github.com/databricks/databricks-sdk-py/issues/509 to use the python sdk instead
    base_url =dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    r = requests.get(base_url+"/api/2.0/workspace/list", params={'path': f"{experiment_path}/{demo_name}"}, headers=headers).json()
    if 'objects' not in r:
      raise Exception(f"No experiment available for this demo. Please re-run the previous notebook with the AutoML run. - {r}")
    xps = [f for f in r['objects'] if f['object_type'] == 'MLFLOW_EXPERIMENT' and 'automl' in f['path']]
    xps = [x for x in xps if re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', x['path'])]
    sorted_xp = sorted(xps, key=lambda f: f['path'], reverse = True)
    if len(sorted_xp) == 0:
      raise Exception(f"No experiment available for this demo. Please re-run the previous notebook with the AutoML run. - {r}")

    last_xp = sorted_xp[0]

    # Search for the date pattern in the input string
    match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', last_xp['path'])

    if match:
        date_str = match.group(1)  # Extract the matched date string
        date = datetime.strptime(date_str, '%Y-%m-%d_%H:%M:%S')  # Convert to a datetime object
        # Calculate the difference in days from the current date
        days_difference = (datetime.now() - date).days
        if days_difference > 30:
            raise Exception(f"It looks like the last experiment {last_xp} is too old ({days} days). Please re-run the previous notebook to make sure you have the latest version.")
    else:
        raise Exception(f"Invalid experiment format or no experiment available. Please re-run the previous notebook. {last_xp['path']}")
    return last_xp

# COMMAND ----------

# DBTITLE 1,Untitled
def create_user_features(travel_purchase_df):
    """
    Computes the user_features feature group.
    """
    travel_purchase_df = travel_purchase_df.withColumn('ts_l', F.col("ts").cast("long"))
    travel_purchase_df = (
        # Sum total purchased for 7 days
        travel_purchase_df.withColumn("lookedup_price_7d_rolling_sum",
            F.sum("price").over(w.Window.partitionBy("user_id").orderBy(F.col("ts_l")).rangeBetween(start=-(7 * 86400), end=0))
        )
        # counting number of purchases per week
        .withColumn("lookups_7d_rolling_sum", 
            F.count("*").over(w.Window.partitionBy("user_id").orderBy(F.col("ts_l")).rangeBetween(start=-(7 * 86400), end=0))
        )
        # total price 7d / total purchases for 7 d 
        .withColumn("mean_price_7d",  F.col("lookedup_price_7d_rolling_sum") / F.col("lookups_7d_rolling_sum"))
         # converting True / False into 1/0
        .withColumn("tickets_purchased", F.col("purchased").cast('int'))
        # how many purchases for the past 6m
        .withColumn("last_6m_purchases", 
            F.sum("tickets_purchased").over(w.Window.partitionBy("user_id").orderBy(F.col("ts_l")).rangeBetween(start=-(6 * 30 * 86400), end=0))
        )
        .select("user_id", "ts", "mean_price_7d", "last_6m_purchases", "user_longitude", "user_latitude")
    )
    return travel_purchase_df



def destination_features_fn(travel_purchase_df):
    """
    Computes the destination_features feature group.
    """
    return (
        travel_purchase_df
          .withColumn("clicked", F.col("clicked").cast("int"))
          .withColumn("sum_clicks_7d", 
            F.sum("clicked").over(w.Window.partitionBy("destination_id").orderBy(F.col("ts").cast("long")).rangeBetween(start=-(7 * 86400), end=0))
          )
          .withColumn("sum_impressions_7d", 
            F.count("*").over(w.Window.partitionBy("destination_id").orderBy(F.col("ts").cast("long")).rangeBetween(start=-(7 * 86400), end=0))
          )
          .select("destination_id", "ts", "sum_clicks_7d", "sum_impressions_7d")
    )  

#Required for pandas_on_spark assign to work properly
import pyspark.pandas as ps
import timeit
ps.set_option('compute.ops_on_diff_frames', True)

#Compute distance between 2 points. Could use geopy instead
def compute_hearth_distance(lat1, lon1, lat2, lon2):
  dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
  a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
  return 2 * 6371 * np.arcsin(np.sqrt(a))


# COMMAND ----------

# Below are cleanup related functions   
import boto3
   
def delete_online_store_table(table_name):
  assert table_name.startswith('feature_store_dbdemos')
  if get_cloud_name() == "aws":
    delete_dynamodb_table(online_table_name)
  elif get_cloud_name() == "azure":
    delete_cosmosdb_container(online_table_name)
  else:
    raise Exception(f"cloud not supported : {get_cloud_name()}")
    
def delete_dynamodb_table(table_name):
  #TODO: add your key 
  #AWS_DYNAMO_DB_KEY_ID = aws_access_key_id=dbutils.secrets.get(scope="field-all-users-feature-store-example-write", key="field-eng-access-key-id")
  #AWS_DYNAMO_DB_KEY = aws_secret_access_key=dbutils.secrets.get(scope="field-all-users-feature-store-example-write", key="field-eng-secret-access-key")
  client = boto3.client('dynamodb', aws_access_key_id=AWS_DYNAMO_DB_KEY_ID, aws_secret_access_key=AWS_DYNAMO_DB_KEY, region_name="us-west-2")
  client.delete_table(TableName=table_name)
  waiter = client.get_waiter('table_not_exists')
  waiter.wait(TableName=table_name)
  print(f"table: '{table_name}' was deleted")

from azure.cosmos import cosmos_client
import azure.cosmos.exceptions as exceptions

def delete_cosmosdb_container(container_name, account_uri_field_demo, database = "field_demos"):
  #TODO: add your key
  #COSMO_DB_KEY = dbutils.secrets.get(scope="feature-store-example-write", key="field-eng-authorization-key")
  client = cosmos_client.CosmosClient(account_uri_field_demo, credential=COSMO_DB_KEY)
  database = client.get_database_client(database)
  container = database.get_container_client(container_name)
  try:
      database.delete_container(container)
      print('Container with id \'{0}\' was deleted'.format(container))
  except exceptions.CosmosResourceNotFoundError:
      print('A container with id \'{0}\' does not exist'.format(container))


# COMMAND ----------

def func_stop_streaming_query(query):
  import time
  while len(query.recentProgress) == 0 or query.status["isDataAvailable"]:
    print("waiting for stream to process all data")
    print(query.status)
    time.sleep(10)
  query.stop() 
  print("Just stopped one of the streaming queries.")
  

from mlflow.tracking.client import MlflowClient
def get_latest_model_version(model_name: str):
  client = MlflowClient()
  models = client.get_latest_versions(model_name, stages=["None"])
  for m in models:
    new_model_version = m.version
  return new_model_version


def cleanup(query, query2):
  func_stop_streaming_query(query)
  func_stop_streaming_query(query2)
  
  func_delete_model_serving_endpoint(model_serving_endpoint_name)

  fs_table_names = [
    fs_table_destination_popularity_features,
    fs_table_destination_location_features,
    fs_table_destination_availability_features,
    fs_table_user_features
  ]
  fs_online_table_names = [
    fs_online_destination_popularity_features,
    fs_online_destination_location_features,
    fs_online_destination_availability_features,
    fs_online_user_features
  ]
  db_name = database_name
  delta_checkpoint = fs_destination_availability_features_delta_checkpoint
  online_checkpoint = fs_destination_availability_features_online_checkpoint
  model = model_name

  fs = feature_store.FeatureStoreClient()

  for table_name in fs_table_names:
    try:
      fs.drop_table(name=table_name)
      print(table_name, "is dropped!")
    except Exception as ex:
      print(ex)

  try:
    drop_database(db_name)
  except Exception as ex:
    print(ex)


  for container_name in fs_online_table_names:
    try:
      print("currentlly working on this online table/container dropping: ", container_name)
      if cloud_name == "azure":
        delete_cosmosdb_container(container_name, account_uri_field_demo)
        print("\n")
      elif cloud_name == "aws":
        delete_dynamodb_table(table_name=container_name)
        print("\n")
    except Exception as ex:
      print(ex)

  try:    
    dbutils.fs.rm(
      delta_checkpoint, True
    )
  except Exception as ex:
      print(ex) 

  try:    
    dbutils.fs.rm(
      online_checkpoint, True
    )
  except Exception as ex:
    print(ex)

  try:
    delete_model(model_name)
  except Exception as ex:
    print(ex)  

# COMMAND ----------

destination_location_df = spark.read.option("inferSchema", "true").load("/databricks-datasets/travel_recommendations_realtime/raw_travel_data/fs-demo_destination-locations/",  format="csv", header="true")
destination_location_df.write.mode('overwrite').saveAsTable('destination_location')

# COMMAND ----------

def wait_for_feature_endpoint_to_start(fe, endpoint_name: str):
    for i in range (100):
        ep = fe.get_feature_serving_endpoint(name=endpoint_name)
        if ep.state == 'IN_PROGRESS':
            if i % 10 == 0:
                print(f"deployment in progress, please wait for your feature serving endpoint to be deployed {ep}")
            time.sleep(5)
        else:
            if ep.state != 'READY':
                raise Exception(f"Endpoint is in abnormal state: {ep}")
            print(f"Endpoint {endpoint_name} ready - {ep}")
            return ep
        
def get_headers():
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    return {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

# COMMAND ----------

import time

class EndpointApiClient:
    def __init__(self):
        self.base_url =dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
        self.token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def create_inference_endpoint(self, endpoint_name, served_models):
        data = {"name": endpoint_name, "config": {"served_models": served_models}}
        return self._post("api/2.0/serving-endpoints", data)

    def get_inference_endpoint(self, endpoint_name):
        return self._get(f"api/2.0/serving-endpoints/{endpoint_name}", allow_error=True)
      
      
    def inference_endpoint_exists(self, endpoint_name):
      ep = self.get_inference_endpoint(endpoint_name)
      if 'error_code' in ep and ep['error_code'] == 'RESOURCE_DOES_NOT_EXIST':
          return False
      if 'error_code' in ep and ep['error_code'] != 'RESOURCE_DOES_NOT_EXIST':
          raise Exception(f"endpoint exists ? {ep}")
      return True

    def create_endpoint_if_not_exists(self, endpoint_name, model_name, model_version, workload_size, scale_to_zero_enabled=True, wait_start=True):
      models = [{
            "model_name": model_name,
            "model_version": model_version,
            "workload_size": workload_size,
            "scale_to_zero_enabled": scale_to_zero_enabled,
      }]
      if not self.inference_endpoint_exists(endpoint_name):
        r = self.create_inference_endpoint(endpoint_name, models)
      #Make sure we have the proper version deployed
      else:
        ep = self.get_inference_endpoint(endpoint_name)
        if 'pending_config' in ep:
            self.wait_endpoint_start(endpoint_name)
            ep = self.get_inference_endpoint(endpoint_name)
        if 'pending_config' in ep:
            model_deployed = ep['pending_config']['served_models'][0]
            print(f"Error with the model deployed: {model_deployed} - state {ep['state']}")
        else:
            model_deployed = ep['config']['served_models'][0]
        if model_deployed['model_version'] != model_version:
          print(f"Current model is version {model_deployed['model_version']}. Updating to {model_version}...")
          u = self.update_model_endpoint(endpoint_name, {"served_models": models})
      if wait_start:
        self.wait_endpoint_start(endpoint_name)
      
      
    def list_inference_endpoints(self):
        return self._get("api/2.0/serving-endpoints")

    def update_model_endpoint(self, endpoint_name, conf):
        return self._put(f"api/2.0/serving-endpoints/{endpoint_name}/config", conf)

    def delete_inference_endpoint(self, endpoint_name):
        return self._delete(f"api/2.0/serving-endpoints/{endpoint_name}")

    def wait_endpoint_start(self, endpoint_name):
      i = 0
      while self.get_inference_endpoint(endpoint_name)['state']['config_update'] == "IN_PROGRESS" and i < 500:
        print("waiting for endpoint to build model image and start")
        time.sleep(30)
        i += 1
      
    # Making predictions

    def query_inference_endpoint(self, endpoint_name, data):
        return self._post(f"realtime-inference/{endpoint_name}/invocations", data)

    # Debugging

    def get_served_model_build_logs(self, endpoint_name, served_model_name):
        return self._get(
            f"api/2.0/serving-endpoints/{endpoint_name}/served-models/{served_model_name}/build-logs"
        )

    def get_served_model_server_logs(self, endpoint_name, served_model_name):
        return self._get(
            f"api/2.0/serving-endpoints/{endpoint_name}/served-models/{served_model_name}/logs"
        )

    def get_inference_endpoint_events(self, endpoint_name):
        return self._get(f"api/2.0/serving-endpoints/{endpoint_name}/events")

    def _get(self, uri, data = {}, allow_error = False):
        r = requests.get(f"{self.base_url}/{uri}", params=data, headers=self.headers)
        return self._process(r, allow_error)

    def _post(self, uri, data = {}, allow_error = False):
        return self._process(requests.post(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _put(self, uri, data = {}, allow_error = False):
        return self._process(requests.put(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _delete(self, uri, data = {}, allow_error = False):
        return self._process(requests.delete(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _process(self, r, allow_error = False):
      if r.status_code == 500 or r.status_code == 403 or not allow_error:
        print(r.text)
        r.raise_for_status()
      return r.json()

# COMMAND ----------

def get_active_streams(start_with = ""):
    return [s for s in spark.streams.active if len(start_with) == 0 or (s.name is not None and s.name.startswith(start_with))]
  
  
# Function to stop all streaming queries 
def stop_all_streams(start_with = "", sleep_time=0):
  import time
  time.sleep(sleep_time)
  streams = get_active_streams(start_with)
  if len(streams) > 0:
    print(f"Stopping {len(streams)} streams")
    for s in streams:
        try:
            s.stop()
        except:
            pass
    print(f"All stream stopped {'' if len(start_with) == 0 else f'(starting with: {start_with}.)'}")


def retrain_expert_model():
  return dbutils.widgets.get('retrain_model') == 'true' or not model_exists(model_name_expert)

def model_exists(model_name):
  try:
    client = mlflow.tracking.MlflowClient()             
    latest_model = client.get_latest_versions(model_name)
    return True
  except:
    return False

# COMMAND ----------

print(f"Catalog: {catalog}")
print(f"DB: {db}")

# COMMAND ----------

DBDemos.setup_schema(catalog, db, reset_all_data)

# COMMAND ----------

travel_purchase_df = spark.read.option("inferSchema", "true").load("/databricks-datasets/travel_recommendations_realtime/raw_travel_data/fs-demo_vacation-purchase_logs/", format="csv", header="true")
travel_purchase_df = travel_purchase_df.withColumn("id", F.monotonically_increasing_id())
travel_purchase_df.withColumn("booking_date", F.col("booking_date").cast('date')).write.mode('overwrite').saveAsTable('travel_purchase')

# COMMAND ----------

destination_location_df = spark.read.option("inferSchema", "true").load("/databricks-datasets/travel_recommendations_realtime/raw_travel_data/fs-demo_destination-locations/",  format="csv", header="true")
destination_location_df.write.mode('overwrite').saveAsTable('destination_location')
