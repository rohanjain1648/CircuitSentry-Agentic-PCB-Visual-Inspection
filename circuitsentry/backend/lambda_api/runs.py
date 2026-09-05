import json
import os
import boto3
from boto3.dynamodb.conditions import Key

CORS_HEADERS = {"Access-Control-Allow-Origin": "*", "Content-Type": "application/json"}


def _table():
    return boto3.resource("dynamodb").Table(os.environ["RUNS_TABLE_NAME"])


def list_handler(event, context):
    """One row per run (not per (run_id, seq)), summarised by its latest seq.

    A run with a recapture loop has several seq items; the dashboard's live
    run list wants a single row per board, showing where the run got to.
    """
    items = _table().scan()["Items"]
    latest_by_run = {}
    for item in items:
        run_id = item["run_id"]
        seq = int(item["seq"])
        current = latest_by_run.get(run_id)
        if current is None or seq > current["seq"]:
            latest_by_run[run_id] = {
                "run_id": run_id,
                "seq": seq,
                "next_action": item.get("next_action", "unknown"),
            }
    runs = sorted(latest_by_run.values(), key=lambda r: r["run_id"])
    return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps({"runs": runs}, default=str)}


def detail_handler(event, context):
    run_id = event["pathParameters"]["id"]
    items = _table().query(KeyConditionExpression=Key("run_id").eq(run_id))["Items"]
    return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps({"items": items}, default=str)}
