# import pyspark.sql
# from pyspark.sql import functions as F
# from pyspark.sql import Window
# import pytest
# import pandas as pd
# from datetime import datetime
# from pyspark.sql import SparkSession

# from mlops_project_demo.feature_engineering.features.helper_functions import (
#   create_user_features, 
#   destination_features_fn
# )


# @pytest.fixture(scope="session")
# def spark(request):
#     """fixture for creating a spark session
#     Args:
#         request: pytest.FixtureRequest object
#     """
#     spark = (
#         SparkSession.builder.master("local[1]")
#         .appName("pytest-pyspark-local-testing")
#         .getOrCreate()
#     )
#     request.addfinalizer(lambda: spark.stop())

#     return spark

# @pytest.mark.usefixtures("spark")
# def test_create_user_features(spark):
#     # Create a sample input DataFrame
#     data = [
#         ("user1", datetime.datetime(2023, 1, 1), 100, True, 10.0, 20.0),
#         ("user1", datetime.datetime(2023, 1, 2), 150, False, 10.0, 20.0),
#         ("user1", datetime.datetime(2023, 1, 8), 200, True, 10.0, 20.0),
#         ("user2", datetime.datetime(2023, 1, 1), 300, False, 30.0, 40.0),
#         ("user2", datetime.datetime(2023, 1, 3), 350, True, 30.0, 40.0),
#     ]
#     schema = ["user_id", "ts", "price", "purchased", "user_longitude", "user_latitude"]
#     travel_purchase_df = spark.createDataFrame(data, schema)

#     # Define the expected DataFrame
#     expected_data = [
#         ("user1", datetime.datetime(2023, 1, 1), 100.0, 1, 10.0, 20.0),
#         ("user1", datetime.datetime(2023, 1, 2), 125.0, 1, 10.0, 20.0),
#         ("user1", datetime.datetime(2023, 1, 8), 150.0, 2, 10.0, 20.0),
#         ("user2", datetime.datetime(2023, 1, 1), 300.0, 0, 30.0, 40.0),
#         ("user2", datetime.datetime(2023, 1, 3), 325.0, 1, 30.0, 40.0),
#     ]
#     expected_schema = ["user_id", "ts", "mean_price_7d", "last_6m_purchases", "user_longitude", "user_latitude"]
#     expected_df = spark.createDataFrame(expected_data, expected_schema)

#     # Run the function to get the actual output
#     result_df = create_user_features(travel_purchase_df)

#     # Collect the results for comparison
#     result = result_df.collect()
#     expected = expected_df.collect()

#     # Compare the actual output with the expected output
#     assert result == expected, "The create_user_features function did not return the expected result."

# @pytest.mark.usefixtures("spark")
# def test_destination_features_fn(spark):
#     # Create a sample input DataFrame
#     data = [
#         ("dest1", datetime.datetime(2023, 1, 1), True),
#         ("dest1", datetime.datetime(2023, 1, 2), False),
#         ("dest1", datetime.datetime(2023, 1, 8), True),
#         ("dest2", datetime.datetime(2023, 1, 1), False),
#         ("dest2", datetime.datetime(2023, 1, 3), True),
#     ]
#     schema = ["destination_id", "ts", "clicked"]
#     travel_purchase_df = spark.createDataFrame(data, schema)

#     # Define the expected DataFrame
#     expected_data = [
#         ("dest1", datetime.datetime(2023, 1, 1), 1, 1),
#         ("dest1", datetime.datetime(2023, 1, 2), 1, 2),
#         ("dest1", datetime.datetime(2023, 1, 8), 2, 3),
#         ("dest2", datetime.datetime(2023, 1, 1), 0, 1),
#         ("dest2", datetime.datetime(2023, 1, 3), 1, 2),
#     ]
#     expected_schema = ["destination_id", "ts", "sum_clicks_7d", "sum_impressions_7d"]
#     expected_df = spark.createDataFrame(expected_data, expected_schema)

#     # Run the function to get the actual output
#     result_df = destination_features_fn(travel_purchase_df)

#     # Collect the results for comparison
#     result = result_df.collect()
#     expected = expected_df.collect()

#     # Compare the actual output with the expected output
#     assert result == expected, "The destination_features_fn function did not return the expected result."
