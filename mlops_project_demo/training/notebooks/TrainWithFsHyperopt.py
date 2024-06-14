# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %md ## Update widgets to avoid overwriting assets.
# MAGIC - catalog: for interactive development use "dev"
# MAGIC - reset_all_data: set to false unless you want to delete assets and start demo over

# COMMAND ----------

# DBTITLE 1,Move to training path and install requirements.txt
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

# DBTITLE 1,Setup system paths to import python modules
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

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION missing_nulls_udf(input_value FLOAT)
# MAGIC RETURNS FLOAT
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT 'Fill missing value with 0'
# MAGIC AS $$
# MAGIC   import numpy as np
# MAGIC   return 0.0 if input_value is None or np.isnan(input_value) else float(input_value)
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

# Save train/test set to delta table for future reuse and trackability
training_df.write.format("delta").mode("overwrite").saveAsTable("training_set")
test_df.write.format("delta").mode("overwrite").saveAsTable("test_set")

# pull feature store version at the time of the training and log to MLflow later
fs_version = spark.sql("DESCRIBE HISTORY training_set").orderBy("version", ascending=False).select("version").first()[0]

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
  # Add our function to fill null values 
  FeatureFunction(
      udf_name="missing_nulls_udf",
      input_bindings={"input_value": "availability"},
      output_name="availability_filled"
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
    exclude_columns=['user_id', 'destination_id', 'booking_date', 'clicked', 'price', 'availability'],
    label='purchased'
)

# COMMAND ----------

training_set_df = training_set.load_df()
#Let's cache the training dataset for automl (to avoid recomputing it everytime)
training_features_df = training_set_df.cache()
training_features_df.write.format("delta").option("overwriteSchema", "true").mode("overwrite").saveAsTable("training_data_baseline")

display(training_features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train model using hyperopt to find the optimal parameters for minimizing the loss function

# COMMAND ----------

import numpy as np
import pandas as pd
import cloudpickle
from sklearn.model_selection import train_test_split
import sklearn
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from hyperopt import (
    fmin, 
    hp, 
    tpe, 
    rand, 
    SparkTrials, 
    Trials, 
    STATUS_OK
)
from hyperopt.pyll.base import scope

# COMMAND ----------

# DBTITLE 1,Setting the search space for XGBoost model
# Setting search space for xgboost model
search_space = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': scope.int(hp.quniform('max_depth', 4, 15, 1)),
    'subsample': hp.uniform('subsample', .5, 1.0),
    'learning_rate': hp.loguniform('learning_rate', -7, 0),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 7),
    'reg_alpha': hp.loguniform('reg_alpha', -10, 10),
    'reg_lambda': hp.loguniform('reg_lambda', -10, 10),
    'gamma': hp.loguniform('gamma', -10, 10),
    'use_label_encoder': False,
    'verbosity': 0,
    'random_state': 24
}

# Set mlflow experiment path to track hyperopt runs
experiment = mlflow.set_experiment(experiment_name=experiment_path)

# COMMAND ----------

RANDOM_SEED = 123 

data_df = training_features_df.toPandas()

# Splitting the dataset into training/validation and holdout sets
train_val, test = train_test_split(
    data_df, 
    test_size=0.1,
    shuffle=True, 
    random_state=RANDOM_SEED
)

# Creating X, y for training/validation set
X_train_val = train_val.drop(columns=['purchased', 'ts'])
y_train_val = train_val['purchased']

# Creating X, y for test set
X_test = test.drop(columns=['purchased', 'ts'])
y_test = test['purchased']

# Splitting training/testing set to create training set and validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, 
    y_train_val,
    stratify=y_train_val,
    shuffle=True, 
    random_state=RANDOM_SEED
)

# COMMAND ----------

# DBTITLE 1,Create the hyperopt objective function
def train_model(params):
    """
    Creates a hyperopt training model funciton that sweeps through params in a nested run
    Args:
        params: hyperparameters selected from the search space
    Returns:
        hyperopt status and the loss metric value
    """
    # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
    # This sometimes doesn't log everything you may want so I usually log my own metrics and params just in case
    mlflow.xgboost.autolog()

    with mlflow.start_run(experiment_id=experiment.experiment_id):
      # Training xgboost classifier
      model = xgb.XGBClassifier(**params)
      model = model.fit(X_train, y_train)

      # Predicting values for training and validation data, and getting prediction probabilities
      y_train_pred = model.predict(X_train)
      y_train_pred_proba = model.predict_proba(X_train)[:, 1]
      y_val_pred = model.predict(X_val)
      y_val_pred_proba = model.predict_proba(X_val)[:, 1]

      # Evaluating model metrics for training set predictions and validation set predictions
      # Creating training and validation metrics dictionaries to make logging in mlflow easier
      metric_names = ['accuracy', 'precision', 'recall', 'f1', 'aucroc']
      # Training evaluation metrics
      train_accuracy = accuracy_score(y_train, y_train_pred).round(3)
      train_precision = precision_score(y_train, y_train_pred).round(3)
      train_recall = recall_score(y_train, y_train_pred).round(3)
      train_f1 = f1_score(y_train, y_train_pred).round(3)
      train_aucroc = roc_auc_score(y_train, y_train_pred_proba).round(3)
      training_metrics = {
          'Accuracy': train_accuracy, 
          'Precision': train_precision, 
          'Recall': train_recall, 
          'F1': train_f1, 
          'AUCROC': train_aucroc
      }
      training_metrics_values = list(training_metrics.values())

      # Validation evaluation metrics
      val_accuracy = accuracy_score(y_val, y_val_pred).round(3)
      val_precision = precision_score(y_val, y_val_pred).round(3)
      val_recall = recall_score(y_val, y_val_pred).round(3)
      val_f1 = f1_score(y_val, y_val_pred).round(3)
      val_aucroc = roc_auc_score(y_val, y_val_pred_proba).round(3)
      validation_metrics = {
          'Accuracy': val_accuracy, 
          'Precision': val_precision, 
          'Recall': val_recall, 
          'F1': val_f1, 
          'AUCROC': val_aucroc
      }
      validation_metrics_values = list(validation_metrics.values())

      conda_env =  _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), 
                              "scikit-learn=={}".format(sklearn.__version__), 
                              "xgboost=={}".format(xgb.__version__)],
        additional_conda_channels=None,
    )
      
      # Logging model signature, class, and name
      signature = infer_signature(X_train, y_val_pred)
      mlflow.xgboost.log_model(model, 'model', signature=signature, input_example=X_train.iloc[0:1], conda_env=conda_env)
      mlflow.set_tag('estimator_name', model.__class__.__name__)
      mlflow.set_tag('estimator_class', model.__class__)

      # Logging each metric
      for name, metric in list(zip(metric_names, training_metrics_values)):
          mlflow.log_metric(f'training_{name}', metric)
      for name, metric in list(zip(metric_names, validation_metrics_values)):
          mlflow.log_metric(f'validation_{name}', metric)

    # Set the loss to -1*validation auc roc so fmin maximizes the it
    return {'status': STATUS_OK, 'loss': -1*validation_metrics['AUCROC']}

# COMMAND ----------

# DBTITLE 1,Parallelize hyperopt across Spark
# Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep.
# A reasonable value for parallelism is the square root of max_evals.
spark_trials = SparkTrials(parallelism=2)

# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
# run called "xgboost_models" .
with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
    xgboost_best_params = fmin(
        fn=train_model, 
        space=search_space, 
        algo=tpe.suggest,
        trials=spark_trials,
        max_evals=10
    )

mlflow.end_run()

# COMMAND ----------

# DBTITLE 1,Query MLflow API sorting on best training run
# Querying mlflow api instead of using web UI. Sorting by validation aucroc and then getting top run for best run.
runs_df = mlflow.search_runs(experiment_ids=experiment.experiment_id, order_by=['metrics.validation_aucroc DESC'])
best_run = runs_df.iloc[0]

# extracting parameters, metrics, and metadata
metric_cols = [col for col in runs_df.columns if col.startswith('metrics.')]
param_cols = [col for col in runs_df.columns if col.startswith('params.')]
best_run_metrics_dict = best_run[metric_cols].to_dict()
best_run_params_dict = best_run[param_cols].to_dict()
best_run_id = best_run['run_id']
best_artifact_uri = best_run['artifact_uri']

# Loading model from best run
best_model = mlflow.xgboost.load_model('runs:/' + best_run_id + '/model')

# Predicting and evaluating best model on holdout set
y_test_pred = best_model.predict(X_test)
y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred).round(3)
test_precision = precision_score(y_test, y_test_pred).round(3)
test_recall = recall_score(y_test, y_test_pred).round(3)
test_f1 = f1_score(y_test, y_test_pred).round(3)
test_aucroc = roc_auc_score(y_test, y_test_pred_proba).round(3)

print(f'Testing Accuracy: {test_accuracy}')
print(f'Testing Precision: {test_precision}')
print(f'Testing Recall: {test_recall}')
print(f'Testing F1: {test_f1}')
print(f'Testing AUCROC: {test_aucroc}')

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

# DBTITLE 1,Log using Sklearn flavor
import mlflow
from mlflow.models import infer_signature
import os
import datetime

mlflow.set_registry_uri('databricks-uc')
model_name = f"hyperopt_feature_store"
model_full_name = f"{catalog}.{db}.{model_name}"

# creating sample input to be logged (do not include the live features in the schema as they'll be computed within the model)
x_sample = X_train.head(10)

#Get the conda env from hyperopt run
artifacts_path = mlflow.artifacts.download_artifacts(run_id=best_run_id)
env = mlflow.pyfunc.get_default_conda_env()
with open(artifacts_path+"model/requirements.txt", 'r') as f:
    env['dependencies'][-1]['pip'] = f.read().split('\n')

#Create a new run in the same experiment as our hyperopt run.
with mlflow.start_run(run_name="best_hyperopt_fs", experiment_id=experiment.experiment_id) as run:

  # define signature with the model
  signature = infer_signature(x_sample, params={"predict_method": "predict_proba"})

  #Use the feature store client to log our best model
  # log as sklearn
  fe.log_model(
    model=best_model, # object of your model
    artifact_path="model", #name of the Artifact under MlFlow
    flavor=mlflow.sklearn, # flavour of the model (our LightGBM model has a SkLearn Flavour)
    training_set=training_set, # training set you used to train your model with AutoML
    input_example=x_sample, # Dataset example (Pandas dataframe)
    signature=signature,
    registered_model_name=model_full_name, # register your best model
    conda_env=env,
    pyfunc_predict_fn="predict" # can also set to "predict_proba"
    )

  #Copy best run images & params to our FS run
  for item in os.listdir(artifacts_path):
    if item.endswith(".png") or item.endswith(".json"):
      mlflow.log_artifact(artifacts_path+item)
  mlflow.log_metrics(best_run_metrics_dict)
  mlflow.log_params(best_run_params_dict)
  mlflow.log_param("run_id", best_run_id)

mlflow.end_run()

# COMMAND ----------

# DBTITLE 1,Customize predict method using PyFunc
from mlflow.pyfunc import PythonModel

# for more details visit: https://mlflow.org/docs/latest/traditional-ml/creating-custom-pyfunc/notebooks/override-predict.html
class ModelWrapper(PythonModel):
    def __init__(self):
        self.model = None
    
    def load_context(self, context):
        """
        Load the model from specified artifacts directory
        """
        model_file_path = context.artifacts["model_file"]
        self.model = mlflow.xgboost.load_model(model_file_path)

    def predict(self, context, model_input, params={"predict_method": "predict_proba"}):
        params = params or {"predict_method": "predict"}
        predict_method = params.get("predict_method")

        if predict_method == "predict":
            return self.model.predict(model_input)
        elif predict_method == "predict_proba":
            return self.model.predict_proba(model_input)[:, 1]
        else:
            raise ValueError(
                f"The prediction method '{predict_method}' is not supported."
            )

# COMMAND ----------

# DBTITLE 1,Log using PyFunc flavor
import mlflow
from mlflow.models import infer_signature
import os
import datetime

mlflow.set_registry_uri('databricks-uc')
model_name = f"hyperopt_feature_store"
model_full_name = f"{catalog}.{db}.{model_name}"

# creating sample input to be logged (do not include the live features in the schema as they'll be computed within the model)
x_sample = X_train.head(10)

#Get the conda env from hyperopt run
artifacts_path = mlflow.artifacts.download_artifacts(run_id=best_run_id)
env = mlflow.pyfunc.get_default_conda_env()
with open(artifacts_path+"model/requirements.txt", 'r') as f:
    env['dependencies'][-1]['pip'] = f.read().split('\n')

#Create a new run in the same experiment as our hyperopt run.
with mlflow.start_run(run_name="best_hyperopt_fs", experiment_id=experiment.experiment_id) as run:

  # define signature with the model
  signature = infer_signature(x_sample, params={"predict_method": "predict_proba"})
  
  artifacts = {"model_file": f"runs:/{best_run_id}/model"}

  #Use the feature store client to log our best model
  # log as pyfunc
  fe.log_model(
    model=ModelWrapper(), # object of your model,
    artifact_path="model",
    artifacts=artifacts, #name of the Artifact under MlFlow
    flavor=mlflow.pyfunc, # flavour of the model (our LightGBM model has a SkLearn Flavour)
    training_set=training_set, # training set you used to train your model with AutoML
    signature=signature,
    input_example=x_sample, # Dataset example (Pandas dataframe)
    registered_model_name=model_full_name + "_pyfunc", # register your best model
    conda_env=env
    )

  #Copy best run images & params to our FS run
  for item in os.listdir(artifacts_path):
    if item.endswith(".png") or item.endswith(".json"):
      mlflow.log_artifact(artifacts_path+item)
  mlflow.log_metrics(best_run_metrics_dict)
  mlflow.log_params(best_run_params_dict)
  mlflow.log_param("run_id", best_run_id)

mlflow.end_run()

# COMMAND ----------

# DBTITLE 1,Batch scoring on the test set using PyFunc
from training.utils.helper_functions import get_last_model_version

#Create the training set
test_set = fe.create_training_set(
    df=test_df,
    feature_lookups=feature_lookups,
    exclude_columns=['user_id', 'destination_id', 'booking_date', 'clicked', 'price', "avalability"],
    label='purchased'
)

test_set_df = test_set.load_df().toPandas()
X_test = test_set_df.drop(columns=['purchased', 'ts'])
y_test = test_set_df['purchased']

model_version = get_last_model_version(model_full_name + "_pyfunc")
score_batch = fe.score_batch(
  model_uri=f'models:/{catalog}.{schema}.hyperopt_feature_store_pyfunc/{model_version.version}', 
  df=test_df).toPandas()

# Predicting and evaluating best model on holdout set
y_test_pred_proba = score_batch['prediction']
y_test_pred = (y_test_pred_proba >= 0.5).astype(int)

test_accuracy = accuracy_score(y_test, y_test_pred).round(3)
test_precision = precision_score(y_test, y_test_pred).round(3)
test_recall = recall_score(y_test, y_test_pred).round(3)
test_f1 = f1_score(y_test, y_test_pred).round(3)
test_aucroc = roc_auc_score(y_test, y_test_pred_proba).round(3)

print(f'Testing Accuracy: {test_accuracy}')
print(f'Testing Precision: {test_precision}')
print(f'Testing Recall: {test_recall}')
print(f'Testing F1: {test_f1}')
print(f'Testing AUCROC: {test_aucroc}')

# COMMAND ----------

# DBTITLE 1,Save baseline dataframe used in Lakehouse Monitoring
model_version = get_last_model_version(model_full_name)
score_batch = fe.score_batch(
  model_uri=f'models:/{catalog}.{schema}.hyperopt_feature_store/{model_version.version}', 
  df=training_df)

score_batch.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("training_baseline_predictions")

display(score_batch)

# COMMAND ----------

# DBTITLE 1,Set tags for registered model
from mlflow.tracking import MlflowClient

client = MlflowClient()
latest_model = get_last_model_version(model_full_name)
client.set_model_version_tag(name=model_full_name, version=latest_model.version, key="timestamp", value=datetime.datetime.fromtimestamp(run.info.start_time/1000.0))
client.set_model_version_tag(name=model_full_name, version=latest_model.version, key="train_date", value=datetime.date.today())
client.set_model_version_tag(name=model_full_name, version=latest_model.version, key="fs_table_version", value=fs_version)
client.set_model_version_tag(name=model_full_name, version=latest_model.version, key="model_type", value="classifier")
client.update_model_version(name=model_full_name, version=latest_model.version, description="Simple model showing how to use MLflow with UC")
client.set_registered_model_alias(name=model_full_name, alias=catalog, version=latest_model.version)
