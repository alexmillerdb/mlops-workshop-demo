from mlflow.tracking import MlflowClient

def get_last_model_version(model_full_name):
    mlflow_client = MlflowClient(registry_uri="databricks-uc")
    # Use the MlflowClient to get a list of all versions for the registered model in Unity Catalog
    all_versions = mlflow_client.search_model_versions(f"name='{model_full_name}'")
    # Sort the list of versions by version number and get the latest version
    latest_version = max([int(v.version) for v in all_versions])
    # Use the MlflowClient to get the latest version of the registered model in Unity Catalog
    return mlflow_client.get_model_version(model_full_name, str(latest_version))