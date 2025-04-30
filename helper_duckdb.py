import json
import duckdb
from google.cloud import storage

def duckdb_init(credentials_path: str ="../../../../g-credentials.json", hmac_path: str="../../../../g-credentials-hmac.json"):
    client = storage.Client().from_service_account_json(credentials_path)

    # init duckdb
    duckdb.sql("INSTALL httpfs")
    duckdb.sql("LOAD httpfs")
    duckdb.sql("SET s3_endpoint='storage.googleapis.com'")

    hmac = json.load(open(hmac_path))

    # You will obtain the key_id from the previous step of
    # configuring settings in the Google Console.
    duckdb.sql(f"SET s3_access_key_id='{hmac['s3_access_key_id']}'")

    # You will obtain the secret_access_key from the previous step of
    # configuring settings in the Google Console.
    duckdb.sql(f"SET s3_secret_access_key='{hmac['s3_secret_access_key']}'")