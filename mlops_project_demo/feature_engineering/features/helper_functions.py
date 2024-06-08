from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.window as w

fe = FeatureEngineeringClient()

spark = SparkSession.builder \
  .appName("DBDemos") \
  .getOrCreate()

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

def delete_fs(fs_table_name):
  print("Deleting Feature Table", fs_table_name)
  try:
    fe.drop_table(name=fs_table_name)
    spark.sql(f"DROP TABLE IF EXISTS {fs_table_name}")
  except Exception as e:
    print("Can't delete table, likely not existing: "+str(e))  

def delete_fss(catalog, db, tables):
  for table in tables:
    delete_fs(f"{catalog}.{db}.{table}")