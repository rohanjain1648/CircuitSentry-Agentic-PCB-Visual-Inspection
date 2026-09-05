import json
import os
import uuid
import boto3

CORS_HEADERS = {"Access-Control-Allow-Origin": "*", "Content-Type": "application/json"}


def lambda_handler(event, context):
    body = json.loads(event.get("body") or "{}")
    run_id = str(uuid.uuid4())

    sfn = boto3.client("stepfunctions")
    execution = sfn.start_execution(
        stateMachineArn=os.environ["STATE_MACHINE_ARN"],
        input=json.dumps({
            "run_id": run_id,
            "seq": 0,
            "bucket": os.environ["IMAGES_BUCKET_NAME"],
            "frame_key": f"{run_id}/0/frame.png",
            "golden_key": body["golden_key"],
            "attempt_counts": {},
        }),
    )
    return {
        "statusCode": 200,
        "headers": CORS_HEADERS,
        "body": json.dumps({"run_id": run_id, "execution_arn": execution["executionArn"]}),
    }
