import time
from databricks.sdk import WorkspaceClient

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
            state = w.online_tables.get(name=f"{catalog}.{schema}.{table}").status.detailed_state.value
            if state.startswith('ONLINE'):
                print(f'Table {table} online: {state}')
                break
            time.sleep(sleep_time)

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
        
def create_online_table(spark, table_name, pks, timeseries_key=None, sync="triggered"):
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    online_table_name = table_name+"_online"
    if not online_table_exists(online_table_name):
        from databricks.sdk.service import catalog as c
        print(f"Creating online table for {online_table_name}...")
        spark.sql(f'ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)')
        if sync == "triggered":
            spec = c.OnlineTableSpec(
                source_table_full_name=table_name, 
                primary_key_columns=pks, 
                run_triggered={'triggered': 'true'}, 
                timeseries_key=timeseries_key)
        else:
            spec = c.OnlineTableSpec(
                source_table_full_name=table_name, 
                primary_key_columns=pks, 
                run_continuously={'continuous': 'true'}, 
                timeseries_key=timeseries_key)

        w.online_tables.create(name=online_table_name, spec=spec)