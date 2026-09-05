import os
import cv2
import numpy as np
import boto3
from decimal import Decimal
from circuitsentry.vision.pipeline import inspect
from circuitsentry.decision.engine import decide

_ACTION_PRIORITY = {"flag_for_approval": 2, "recapture": 1, "pass": 0}


def _load_image(s3_client, bucket: str, key: str) -> np.ndarray:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = np.frombuffer(obj["Body"].read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _region_id(run_id: str, index: int) -> str:
    """Stable-by-detection-order identity for a region within a run.

    Attempt counts are keyed by this id rather than by bbox: a recapture
    uploads a cropped+zoomed frame, so bboxes are expressed in a different
    coordinate frame each iteration and are useless as an identity key
    (the count would silently reset, defeating the <=2-attempt bound).

    Simplification (documented, not hidden): identity here is detection
    *order*, which holds for the single/few-region boards this project
    targets but is not a true correspondence for multi-region frames whose
    composition changes a lot between captures. A full solution would remap
    coordinates through the recapture crop/homography.
    """
    return f"{run_id}:{index}"


def lambda_handler(event: dict, context) -> dict:
    s3_client = boto3.client("s3")
    table_name = os.environ["RUNS_TABLE_NAME"]
    table = boto3.resource("dynamodb").Table(table_name)

    run_id = event["run_id"]
    seq = int(event["seq"])
    bucket = event["bucket"]
    golden_key = event["golden_key"]
    frame_key = event["frame_key"]

    frame = _load_image(s3_client, bucket, frame_key)
    golden = _load_image(s3_client, bucket, golden_key)

    regions = inspect(frame, golden)
    region_ids = [_region_id(run_id, i) for i in range(len(regions))]

    attempt_counts = {k: int(v) for k, v in (event.get("attempt_counts") or {}).items()}
    actions = decide(regions, attempt_counts, region_ids=region_ids)

    updated_attempt_counts = dict(attempt_counts)
    region_results = []
    next_action = "pass"
    for region_id, ra in zip(region_ids, actions):
        if ra.action == "recapture":
            updated_attempt_counts[region_id] = attempt_counts.get(region_id, 0) + 1
        region_results.append({
            "region_id": region_id,
            "bbox": list(ra.region.bbox),
            "confidence": ra.region.confidence,
            "defect_type_guess": ra.region.defect_type_guess,
            "action": ra.action,
        })
        if _ACTION_PRIORITY[ra.action] > _ACTION_PRIORITY[next_action]:
            next_action = ra.action

    table.put_item(Item={
        "run_id": run_id,
        "seq": seq,
        "regions": [
            {**r, "confidence": Decimal(str(r["confidence"]))} for r in region_results
        ],
        "next_action": next_action,
    })

    # The Step Functions task uses OutputPath "$.Payload", so this return value
    # becomes the ENTIRE input to the next state. bucket/frame_key/golden_key
    # must be echoed back, or the recapture loop-back re-enters Perception with
    # an event missing those keys and dies on KeyError.
    if next_action == "recapture":
        # Advance the sequence so the loop-back writes a NEW DynamoDB item
        # instead of overwriting this one, and point frame_key at where the
        # capture client is expected to upload the recaptured crop.
        next_seq = seq + 1
        next_frame_key = f"{run_id}/{next_seq}/frame.png"
    else:
        # "pass" / "flag_for_approval" are terminal: nothing more is uploaded.
        next_seq = seq
        next_frame_key = frame_key

    return {
        "run_id": run_id,
        "seq": next_seq,
        "bucket": bucket,
        "frame_key": next_frame_key,
        "golden_key": golden_key,
        "regions": region_results,
        "attempt_counts": updated_attempt_counts,
        "next_action": next_action,
    }
