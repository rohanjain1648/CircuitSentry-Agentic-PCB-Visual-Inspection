import json
import os
import boto3

CORS_HEADERS = {"Access-Control-Allow-Origin": "*", "Content-Type": "application/json"}


def lambda_handler(event, context):
    run_id = event["pathParameters"]["id"]
    body = json.loads(event.get("body") or "{}")
    table = boto3.resource("dynamodb").Table(os.environ["RUNS_TABLE_NAME"])
    table.update_item(
        Key={"run_id": run_id, "seq": int(body["seq"])},
        UpdateExpression="SET approval_status = :s",
        ExpressionAttributeValues={":s": body["decision"]},
    )
    return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps({"ok": True})}
