import io
import boto3
import cv2
import numpy as np
from moto import mock_aws
from circuitsentry.backend.lambda_perception.handler import lambda_handler


def _encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _board_with_defect():
    golden = np.full((300, 300, 3), 100, dtype=np.uint8)
    cv2.rectangle(golden, (40, 40), (100, 100), (200, 200, 200), -1)
    cv2.circle(golden, (150, 220), 30, (50, 50, 50), -1)
    frame = golden.copy()
    cv2.circle(frame, (220, 80), 25, (255, 255, 255), -1)
    return frame, golden


@mock_aws
def test_handler_writes_dynamodb_record_and_returns_next_action():
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")
    frame, golden = _board_with_defect()
    s3.put_object(Bucket="test-bucket", Key="run1/0/frame.png", Body=_encode(frame))
    s3.put_object(Bucket="test-bucket", Key="golden/board-rev1.png", Body=_encode(golden))

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

    import os
    os.environ["RUNS_TABLE_NAME"] = "test-runs"

    event = {
        "run_id": "run1",
        "seq": 0,
        "bucket": "test-bucket",
        "frame_key": "run1/0/frame.png",
        "golden_key": "golden/board-rev1.png",
        "attempt_counts": {},
    }
    result = lambda_handler(event, None)

    assert result["run_id"] == "run1"
    assert result["next_action"] in {"pass", "recapture", "flag_for_approval"}
    assert len(result["regions"]) >= 1

    item = table.get_item(Key={"run_id": "run1", "seq": 0})["Item"]
    assert item["next_action"] == result["next_action"]
