"""Entry point tying camera + uploader + poller into the capture loop.

Usage: python -m circuitsentry.capture_client.main <api_base_url> <bucket> <golden_key>
"""
import sys
import time
import boto3
import cv2
import numpy as np
import requests

from circuitsentry.capture_client.camera import capture_frame, crop_roi
from circuitsentry.capture_client.uploader import upload_frame
from circuitsentry.capture_client.poller import poll_run_status

POLL_INTERVAL_SECONDS = 2
MAX_POLL_ATTEMPTS = 30
TERMINAL_ACTIONS = {"pass", "flag_for_approval"}


def recapture_bbox(status: dict) -> tuple[int, int, int, int] | None:
    """The bbox of the region that asked for a recapture, if any.

    The regions come straight from the run detail item written by the
    perception Lambda, so the bbox is in the coordinate frame of the frame
    that was inspected at ``status["seq"]`` — i.e. the last frame this client
    uploaded, which is what we crop from.
    """
    for region in status.get("regions") or []:
        if region.get("action") == "recapture":
            return tuple(int(v) for v in region["bbox"])
    return None


def fulfil_recapture(
    status: dict,
    last_frame: np.ndarray,
    s3_client,
    bucket: str,
    run_id: str,
) -> tuple[np.ndarray, int] | None:
    """Crop the requested ROI out of ``last_frame`` and upload it as the next seq.

    Returns ``(crop, new_seq)`` on success, or ``None`` if the run detail did
    not actually name a region to recapture (nothing to do yet).

    The perception handler wrote its record at ``seq`` and echoed
    ``seq + 1`` / ``{run_id}/{seq+1}/frame.png`` into the Step Functions loop,
    so that is exactly where the new crop has to land.
    """
    bbox = recapture_bbox(status)
    if bbox is None:
        return None
    new_seq = int(status["seq"]) + 1
    crop = crop_roi(last_frame, bbox)
    upload_frame(s3_client, bucket, run_id=run_id, seq=new_seq, frame=crop)
    return crop, new_seq


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

        # The frame the next recapture crops from, and the highest seq we have
        # already uploaded (so we never re-upload while waiting for the loop
        # to move on).
        last_frame = frame
        uploaded_seq = 0

        for _ in range(MAX_POLL_ATTEMPTS):
            time.sleep(POLL_INTERVAL_SECONDS)
            status = poll_run_status(api_base_url, run_id)
            if status["next_action"] in TERMINAL_ACTIONS:
                print(f"Run {run_id} finished: {status['next_action']}")
                return
            if status["next_action"] == "recapture":
                if int(status["seq"]) < uploaded_seq:
                    continue  # already fulfilled this request; awaiting the next result
                fulfilled = fulfil_recapture(status, last_frame, s3_client, bucket, run_id)
                if fulfilled is None:
                    continue
                last_frame, uploaded_seq = fulfilled
                print(f"Run {run_id}: uploaded recapture crop at seq {uploaded_seq}")
        print(f"Run {run_id} timed out waiting for a terminal action")
    finally:
        video_capture.release()


if __name__ == "__main__":
    run(api_base_url=sys.argv[1], bucket=sys.argv[2], golden_key=sys.argv[3])
