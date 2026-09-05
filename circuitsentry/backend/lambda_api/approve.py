import json
import os
import boto3
from boto3.dynamodb.conditions import Attr

CORS_HEADERS = {"Access-Control-Allow-Origin": "*", "Content-Type": "application/json"}


def lambda_handler(event, context):
    run_id = event["pathParameters"]["id"]
    body = json.loads(event.get("body") or "{}")
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(os.environ["RUNS_TABLE_NAME"])
    try:
        table.update_item(
            Key={"run_id": run_id, "seq": int(body["seq"])},
            UpdateExpression="SET approval_status = :s",
            ExpressionAttributeValues={":s": body["decision"]},
            # Without this, approving an unknown (run_id, seq) silently CREATES
            # a partial item with no regions/next_action, which then corrupts
            # the run list. Fail loudly with a 404 instead.
            ConditionExpression=Attr("run_id").exists(),
        )
    except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
        return {
            "statusCode": 404,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": "run not found", "run_id": run_id, "seq": body.get("seq")}),
        }
    return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps({"ok": True})}
