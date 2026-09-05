import json
import os
import boto3
from boto3.dynamodb.conditions import Key

CORS_HEADERS = {"Access-Control-Allow-Origin": "*", "Content-Type": "application/json"}


def _table():
    return boto3.resource("dynamodb").Table(os.environ["RUNS_TABLE_NAME"])


def list_handler(event, context):
    items = _table().scan()["Items"]
    runs = [{"run_id": i["run_id"], "seq": int(i["seq"]), "next_action": i["next_action"]} for i in items]
    return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps({"runs": runs}, default=str)}


def detail_handler(event, context):
    run_id = event["pathParameters"]["id"]
    items = _table().query(KeyConditionExpression=Key("run_id").eq(run_id))["Items"]
    return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps({"items": items}, default=str)}
