# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering==0.2.0 databricks-sdk==0.20.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "dev")
dbutils.widgets.text("schema", "mlops_project_demo")
dbutils.widgets.text("table", "dbdemos_fs_travel")
dbutils.widgets.text("reset_all_data", "false")
dbutils.widgets.text("suffix", "alex_miller")

suffix = dbutils.widgets.get("suffix")
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

# # List of input args needed to run this notebook as a job.
# # Provide them via DB widgets or notebook arguments.

# # Notebook Environment
# dbutils.widgets.dropdown("env", "staging", ["staging", "prod"], "Environment Name")
# env = dbutils.widgets.get("env")

# # Path to the Hive-registered Delta table containing the training data.
# dbutils.widgets.text(
#     "training_data_path",
#     "/databricks-datasets/nyctaxi-with-zipcodes/subsampled",
#     label="Path to the training data",
# )

# # MLflow experiment name.
# dbutils.widgets.text(
#     "experiment_name",
#     f"/dev-mlops_project_demo-experiment",
#     label="MLflow experiment name",
# )
# # Unity Catalog registered model name to use for the trained mode.
# dbutils.widgets.text(
#     "model_name", "dev.mlops_project_demo.mlops_project_demo-model", label="Full (Three-Level) Model Name"
# )

# # Pickup features table name
# dbutils.widgets.text(
#     "pickup_features_table",
#     "dev.mlops_project_demo.trip_pickup_features",
#     label="Pickup Features Table",
# )

# # Dropoff features table name
# dbutils.widgets.text(
#     "dropoff_features_table",
#     "dev.mlops_project_demo.trip_dropoff_features",
#     label="Dropoff Features Table",
# )

# COMMAND ----------

# DBTITLE 1,Setup initialization
from mlops_project_demo.utils.setup_init import DBDemos

dbdemos = DBDemos()
dbdemos.setup_schema(catalog, db, reset_all_data=reset_all_data)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Compute on-demand live features
# MAGIC
# MAGIC User location is a context feature that is captured at the time of the query. This data is not known in advance. 
# MAGIC
# MAGIC Derivated features can be computed from this location. For example, user distance from destination can only be computed in realtime at the prediction time.
# MAGIC
# MAGIC This introduce a new challenge, we now have to link some function to transform the data and make sure the same is being used for training and inference (batch or realtime). 
# MAGIC
# MAGIC To solve this, Databricks introduced Feature Spec. With Feature Spec, you can create custom function (SQL/PYTHON) to transform your data into new features, and link them to your model and feature store.
# MAGIC
# MAGIC Because it's shipped as part of your FeatureLookup definition, the same code will be used at inference time, offering a garantee that we compute the feature the same way, and adding flexibility while increasing model version.
# MAGIC
# MAGIC Note that this function will be available as `catalog.schema.distance_udf` in the browser.

# COMMAND ----------

# DBTITLE 1,Define function to compute distance
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION distance_udf(lat1 DOUBLE, lon1 DOUBLE, lat2 DOUBLE, lon2 DOUBLE)
# MAGIC RETURNS DOUBLE
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT 'Calculate hearth distance from latitude and longitude'
# MAGIC AS $$
# MAGIC   import numpy as np
# MAGIC   dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
# MAGIC   a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
# MAGIC   return 2 * 6371 * np.arcsin(np.sqrt(a))
# MAGIC $$

# COMMAND ----------

# DBTITLE 1,Test the function to compute the distance between user and destination
# MAGIC %sql
# MAGIC SELECT distance_udf(user_latitude, user_longitude, latitude, longitude) AS hearth_distance, *
# MAGIC     FROM destination_location_features
# MAGIC         JOIN destination_features USING (destination_id)
# MAGIC         JOIN user_features USING (ts)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Train a custom model with batch, on-demand and streaming features
# MAGIC
# MAGIC That's all we have to do. We're now ready to train our model with this new feature.
# MAGIC
# MAGIC *Note: In a typical deployment, you would add more functions such as timestamp features (cos/sin for the hour/day of the week) etc.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get ground-truth training labels and key + timestamp

# COMMAND ----------

# Split to define a training and inference set
training_keys = spark.table('travel_purchase').select('ts', 'purchased', 'destination_id', 'user_id', 'user_latitude', 'user_longitude', 'booking_date')
training_df = training_keys.where("ts < '2022-11-23'")
test_df = training_keys.where("ts >= '2022-11-23'").cache()

display(training_df.limit(5))

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Create the training set
# MAGIC
# MAGIC Note the use of `FeatureFunction`, pointing to the new distance_udf function that we saved in Unity Catalog.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_engineering.entities.feature_function import FeatureFunction
from databricks.feature_engineering.entities.feature_lookup import FeatureLookup

fe = FeatureEngineeringClient()

feature_lookups = [ # Grab all useful features from different feature store tables
  FeatureLookup(
      table_name="user_features", 
      lookup_key="user_id",
      timestamp_lookup_key="ts",
      feature_names=["mean_price_7d"]
  ),
  FeatureLookup(
      table_name="destination_features", 
      lookup_key="destination_id",
      timestamp_lookup_key="ts"
  ),
  FeatureLookup(
      table_name="destination_location_features",  
      lookup_key="destination_id",
      feature_names=["latitude", "longitude"]
  ),
  FeatureLookup(
      table_name="availability_features", 
      lookup_key=["destination_id", "booking_date"],
      timestamp_lookup_key="ts",
      feature_names=["availability"]
  ),
  # Add our function to compute the distance between the user and the destination 
  FeatureFunction(
      udf_name="distance_udf",
      input_bindings={"lat1": "user_latitude", "lon1": "user_longitude", "lat2": "latitude", "lon2": "longitude"},
      output_name="distance"
  )]

#Create the training set
training_set = fe.create_training_set(
    df=training_df,
    feature_lookups=feature_lookups,
    exclude_columns=['user_id', 'destination_id', 'booking_date', 'clicked', 'price'],
    label='purchased'
)

# COMMAND ----------


training_set_df = training_set.load_df()
#Let's cache the training dataset for automl (to avoid recomputing it everytime)
training_features_df = training_set_df.cache()

display(training_features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use automl to build an ML model out of the box

# COMMAND ----------

from datetime import datetime
from databricks import automl

xp_path = f"/Shared/mlops-workshop/experiments/automl-feature-store-{suffix}"
xp_name = f"automl_purchase_expert_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
summary_cl = automl.classify(
    experiment_name = xp_name,
    experiment_dir = xp_path,
    dataset = training_features_df,
    target_col = "purchased",
    primary_metric="log_loss",
    timeout_minutes = 15
)
#Make sure all users can access dbdemos shared experiment
dbdemos.set_experiment_permission(f"{xp_path}/{xp_name}")

print(f"Best run id: {summary_cl.best_trial.mlflow_run_id}")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Save our best model to MLflow registry
# MAGIC
# MAGIC Next, we'll get Automl best model and add it to our registry. Because we the feature store to keep track of our model & features, we'll log the best model as a new run using the `FeatureStoreClient.log_model()` function.
# MAGIC
# MAGIC Because our model need live features, we'll wrap our best model with `OnDemandComputationModelWrapper`
# MAGIC
# MAGIC Because we re-define the `.predict()` function, the wrapper will automatically add our live feature depending on the input.

# COMMAND ----------

import mlflow
import os

mlflow.set_registry_uri('databricks-uc')
model_name = f"automl_feature_store"
model_full_name = f"{catalog}.{db}.{model_name}"

# creating sample input to be logged (do not include the live features in the schema as they'll be computed within the model)
df_sample = training_set_df.limit(10).toPandas()
x_sample = df_sample.drop(columns=["purchased"])

# getting the model created by AutoML 
best_model = summary_cl.best_trial.load_model()

#Get the conda env from automl run
artifacts_path = mlflow.artifacts.download_artifacts(run_id=summary_cl.best_trial.mlflow_run_id)
env = mlflow.pyfunc.get_default_conda_env()
with open(artifacts_path+"model/requirements.txt", 'r') as f:
    env['dependencies'][-1]['pip'] = f.read().split('\n')

#Create a new run in the same experiment as our automl run.
with mlflow.start_run(run_name="best_fs_model_expert", experiment_id=summary_cl.experiment.experiment_id) as run:
  #Use the feature store client to log our best model
  fe.log_model(
              model=best_model, # object of your model
              artifact_path="model", #name of the Artifact under MlFlow
              flavor=mlflow.sklearn, # flavour of the model (our LightGBM model has a SkLearn Flavour)
              training_set=training_set, # training set you used to train your model with AutoML
              input_example=x_sample, # Dataset example (Pandas dataframe)
              registered_model_name=model_full_name, # register your best model
              conda_env=env)

  #Copy automl images & params to our FS run
  for item in os.listdir(artifacts_path):
    if item.endswith(".png"):
      mlflow.log_artifact(artifacts_path+item)
  mlflow.log_metrics(summary_cl.best_trial.metrics)
  mlflow.log_params(summary_cl.best_trial.params)
  mlflow.log_param("automl_run_id", summary_cl.best_trial.mlflow_run_id)
  mlflow.set_tag(key='feature_store', value='expert_demo')

# COMMAND ----------


from mlflow.tracking import MlflowClient
from mlops_project_demo.training.utils.helper_functions import get_last_model_version

latest_model = get_last_model_version(model_full_name)
#Move it in Production
production_alias = "production"
if len(latest_model.aliases) == 0 or latest_model.aliases[0] != production_alias:
  print(f"updating model {latest_model.version} to Production")
  mlflow_client = MlflowClient(registry_uri="databricks-uc")
  mlflow_client.set_registered_model_alias(model_full_name, production_alias, version=latest_model.version)

# COMMAND ----------

# MAGIC %md
# MAGIC Our model is ready! you can open the Unity Catalog Explorer to review it.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Batch score test set
# MAGIC
# MAGIC Let's make sure our model is working as expected and try to score our test dataset

# COMMAND ----------

scored_df = fe.score_batch(model_uri=f"models:/{model_full_name}@{production_alias}", df=test_df, result_type="boolean")
display(scored_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Accuracy

# COMMAND ----------

from sklearn.metrics import accuracy_score

# simply convert the original probability predictions to true or false
pd_scoring = scored_df.select("purchased", "prediction").toPandas()
print("Accuracy: ", accuracy_score(pd_scoring["purchased"], pd_scoring["prediction"]))
