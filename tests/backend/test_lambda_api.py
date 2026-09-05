import json
import os
import boto3
from moto import mock_aws
from circuitsentry.backend.lambda_api import capture, runs, approve


def _make_table():
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    table = dynamodb.create_table(
        TableName="test-runs",
        KeySchema=[
            {"AttributeName": "run_id", "KeyType": "HASH"},
            {"AttributeName": "seq", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "run_id", "AttributeType": "S"},
            {"AttributeName": "seq", "AttributeType": "N"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    table.wait_until_exists()
    return table


@mock_aws
def test_capture_starts_execution():
    sfn = boto3.client("stepfunctions", region_name="us-east-1")
    state_machine = sfn.create_state_machine(
        name="test-sm",
        definition=json.dumps({"StartAt": "Pass", "States": {"Pass": {"Type": "Pass", "End": True}}}),
        roleArn="arn:aws:iam::123456789012:role/test",
    )
    os.environ["STATE_MACHINE_ARN"] = state_machine["stateMachineArn"]
    os.environ["IMAGES_BUCKET_NAME"] = "test-bucket"

    event = {"body": json.dumps({"golden_key": "golden/board-rev1.png"})}
    response = capture.lambda_handler(event, None)

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert "run_id" in body
    assert "execution_arn" in body


@mock_aws
def test_runs_list_and_detail():
    table = _make_table()
    os.environ["RUNS_TABLE_NAME"] = "test-runs"
    table.put_item(Item={"run_id": "r1", "seq": 0, "next_action": "pass", "regions": []})
    table.put_item(Item={"run_id": "r1", "seq": 1, "next_action": "flag_for_approval", "regions": []})

    list_response = runs.list_handler({}, None)
    body = json.loads(list_response["body"])
    assert len(body["runs"]) == 2

    detail_response = runs.detail_handler({"pathParameters": {"id": "r1"}}, None)
    body = json.loads(detail_response["body"])
    assert len(body["items"]) == 2


@mock_aws
def test_approve_sets_status():
    table = _make_table()
    os.environ["RUNS_TABLE_NAME"] = "test-runs"
    table.put_item(Item={"run_id": "r1", "seq": 0, "next_action": "flag_for_approval", "regions": []})

    event = {
        "pathParameters": {"id": "r1"},
        "body": json.dumps({"seq": 0, "decision": "approved"}),
    }
    response = approve.lambda_handler(event, None)
    assert response["statusCode"] == 200

    item = table.get_item(Key={"run_id": "r1", "seq": 0})["Item"]
    assert item["approval_status"] == "approved"
