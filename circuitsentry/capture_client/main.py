"""Entry point tying camera + uploader + poller into the capture loop.

Usage: python -m circuitsentry.capture_client.main <api_base_url> <bucket> <golden_key>
"""
import sys
import time
import boto3
import cv2
import requests

from circuitsentry.capture_client.camera import capture_frame, crop_roi
from circuitsentry.capture_client.uploader import upload_frame
from circuitsentry.capture_client.poller import poll_run_status

POLL_INTERVAL_SECONDS = 2
MAX_POLL_ATTEMPTS = 30


def run(api_base_url: str, bucket: str, golden_key: str) -> None:
    s3_client = boto3.client("s3")
    video_capture = cv2.VideoCapture(0)
    try:
        frame = capture_frame(video_capture)
        start_response = requests.post(
            f"{api_base_url}/capture",
            json={"golden_key": golden_key},
            timeout=10,
        )
        start_response.raise_for_status()
        run_id = start_response.json()["run_id"]
        upload_frame(s3_client, bucket, run_id=run_id, seq=0, frame=frame)

        for _ in range(MAX_POLL_ATTEMPTS):
            time.sleep(POLL_INTERVAL_SECONDS)
            status = poll_run_status(api_base_url, run_id)
            if status["next_action"] == "recapture":
                # NOTE: bbox for the ROI to recapture is read from the run
                # detail response in a fuller implementation; kept simple here.
                continue
            if status["next_action"] in {"pass", "flag_for_approval"}:
                print(f"Run {run_id} finished: {status['next_action']}")
                return
        print(f"Run {run_id} timed out waiting for a terminal action")
    finally:
        video_capture.release()


if __name__ == "__main__":
    run(api_base_url=sys.argv[1], bucket=sys.argv[2], golden_key=sys.argv[3])
