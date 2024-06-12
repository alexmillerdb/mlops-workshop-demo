# Databricks notebook source
dbutils.widgets.text("catalogs", "dev,staging,prod")
catalogs = dbutils.widgets.get("catalogs").split(",")

# COMMAND ----------

from utils.setup_init import DBDemos

dbdemos = DBDemos()
current_user = dbdemos.get_username()
schema = db = f"mlops_project_demo_{current_user}"

# COMMAND ----------

for catalog in catalogs:
  print(f"Creating catalog {catalog}")
  spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
  print(f"Creating schema {catalog}.{schema}")
  spark.sql(f"CREATE DATABASE IF NOT EXISTS {catalog}.{schema}")
