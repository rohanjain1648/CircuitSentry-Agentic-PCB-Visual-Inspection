import boto3
import numpy as np
from moto import mock_aws

from circuitsentry.capture_client.main import fulfil_recapture, recapture_bbox


def test_recapture_bbox_picks_the_region_that_asked_for_it():
    status = {
        "seq": 0,
        "next_action": "recapture",
        "regions": [
            {"region_id": "r:0", "bbox": [1, 1, 2, 2], "action": "pass"},
            {"region_id": "r:1", "bbox": [10, 20, 30, 40], "action": "recapture"},
        ],
    }
    assert recapture_bbox(status) == (10, 20, 30, 40)


def test_recapture_bbox_is_none_when_no_region_requested_one():
    assert recapture_bbox({"seq": 0, "regions": [{"bbox": [0, 0, 1, 1], "action": "pass"}]}) is None
    assert recapture_bbox({"seq": 0}) is None


@mock_aws
def test_fulfil_recapture_uploads_cropped_roi_at_the_next_seq():
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[50:100, 50:100] = 255

    status = {
        "seq": 0,
        "next_action": "recapture",
        "regions": [{"region_id": "run1:0", "bbox": [50, 50, 50, 50], "action": "recapture"}],
    }
    crop, new_seq = fulfil_recapture(status, frame, s3, "test-bucket", "run1")

    assert new_seq == 1
    assert crop.shape[0] > 50  # crop_roi upscales
    obj = s3.get_object(Bucket="test-bucket", Key="run1/1/frame.png")
    assert obj["Body"].read()


@mock_aws
def test_fulfil_recapture_returns_none_when_nothing_to_recapture():
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    status = {"seq": 0, "next_action": "recapture", "regions": []}
    assert fulfil_recapture(status, frame, s3, "test-bucket", "run1") is None
