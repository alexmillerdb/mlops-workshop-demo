import pytest
from unittest.mock import patch, MagicMock
from mlflow.tracking import MlflowClient
from mlops_project_demo.training.utils.helper_functions import get_last_model_version

def test_get_last_model_version():
    model_full_name = "test_model"
    
    # Mocking the MlflowClient
    with patch("mlflow.tracking.MlflowClient") as MockMlflowClient:
        mock_mlflow_client = MockMlflowClient.return_value

        # Mock the search_model_versions method
        mock_mlflow_client.search_model_versions.return_value = [
            MagicMock(version="1"),
            MagicMock(version="2"),
            MagicMock(version="3"),
        ]
        
        # Mock the get_model_version method
        expected_model_version = MagicMock()
        mock_mlflow_client.get_model_version.return_value = expected_model_version
        
        # Call the function
        result = get_last_model_version(model_full_name)
        
        # Assertions
        mock_mlflow_client.search_model_versions.assert_called_once_with(f"name='{model_full_name}'")
        mock_mlflow_client.get_model_version.assert_called_once_with(model_full_name, "3")
        assert result == expected_model_version