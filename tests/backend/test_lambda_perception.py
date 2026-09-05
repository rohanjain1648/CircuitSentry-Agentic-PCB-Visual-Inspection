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
    assert all(r["region_id"].startswith("run1:") for r in result["regions"])

    # The handler's return value is the ENTIRE next state input, so the S3
    # coordinates must be echoed back for the loop-back to work.
    assert result["bucket"] == "test-bucket"
    assert result["golden_key"] == "golden/board-rev1.png"
    assert "frame_key" in result

    item = table.get_item(Key={"run_id": "run1", "seq": 0})["Item"]
    assert item["next_action"] == result["next_action"]


def _ambiguous_board():
    """A defect small enough to land between LOW and HIGH confidence."""
    golden = np.full((300, 300, 3), 100, dtype=np.uint8)
    cv2.rectangle(golden, (40, 40), (100, 100), (200, 200, 200), -1)
    cv2.circle(golden, (150, 220), 30, (50, 50, 50), -1)
    frame = golden.copy()
    cv2.circle(frame, (220, 80), 18, (255, 255, 255), -1)
    return frame, golden


@mock_aws
def test_recapture_advances_seq_and_frame_key_and_echoes_s3_coordinates():
    import os

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")
    frame, golden = _ambiguous_board()
    s3.put_object(Bucket="test-bucket", Key="run2/0/frame.png", Body=_encode(frame))
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
    os.environ["RUNS_TABLE_NAME"] = "test-runs"

    event = {
        "run_id": "run2",
        "seq": 0,
        "bucket": "test-bucket",
        "frame_key": "run2/0/frame.png",
        "golden_key": "golden/board-rev1.png",
        "attempt_counts": {},
    }
    result = lambda_handler(event, None)
    assert result["next_action"] == "recapture", result

    # seq advanced, frame_key points at the new upload location, attempt
    # counts keyed by region_id.
    assert result["seq"] == 1
    assert result["frame_key"] == "run2/1/frame.png"
    assert result["bucket"] == "test-bucket"
    assert result["golden_key"] == "golden/board-rev1.png"
    assert result["attempt_counts"] == {"run2:0": 1}

    # The DynamoDB write used the ORIGINAL seq, so the loop-back writes a new item.
    assert table.get_item(Key={"run_id": "run2", "seq": 0})["Item"]["next_action"] == "recapture"

    # Feeding the echoed event straight back in (as Step Functions does) works.
    s3.put_object(Bucket="test-bucket", Key="run2/1/frame.png", Body=_encode(frame))
    second = lambda_handler(
        {k: result[k] for k in ("run_id", "seq", "bucket", "frame_key", "golden_key", "attempt_counts")},
        None,
    )
    assert second["attempt_counts"] == {"run2:0": 2}
    assert table.get_item(Key={"run_id": "run2", "seq": 1})["Item"]["seq"] == 1

    # A third pass hits MAX_RECAPTURE_ATTEMPTS and escalates instead of looping.
    s3.put_object(Bucket="test-bucket", Key=second["frame_key"], Body=_encode(frame))
    third = lambda_handler(
        {k: second[k] for k in ("run_id", "seq", "bucket", "frame_key", "golden_key", "attempt_counts")},
        None,
    )
    assert third["next_action"] == "flag_for_approval"
    assert third["seq"] == second["seq"]  # terminal: seq unchanged
