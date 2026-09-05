import boto3
import numpy as np
from moto import mock_aws
from circuitsentry.capture_client.uploader import upload_frame


@mock_aws
def test_upload_frame_puts_object_and_returns_key():
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")
    frame = np.zeros((50, 50, 3), dtype=np.uint8)

    key = upload_frame(s3, "test-bucket", run_id="run1", seq=0, frame=frame)

    assert key == "run1/0/frame.png"
    obj = s3.get_object(Bucket="test-bucket", Key=key)
    assert obj["Body"].read()  # non-empty PNG bytes
