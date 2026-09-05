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


def _bbox_key(bbox: tuple) -> str:
    return ",".join(str(v) for v in bbox)


def lambda_handler(event: dict, context) -> dict:
    s3_client = boto3.client("s3")
    table_name = os.environ["RUNS_TABLE_NAME"]
    table = boto3.resource("dynamodb").Table(table_name)

    frame = _load_image(s3_client, event["bucket"], event["frame_key"])
    golden = _load_image(s3_client, event["bucket"], event["golden_key"])

    regions = inspect(frame, golden)

    attempt_counts = {
        tuple(int(v) for v in k.split(",")): v_count
        for k, v_count in event.get("attempt_counts", {}).items()
    }
    actions = decide(regions, attempt_counts)

    updated_attempt_counts = dict(event.get("attempt_counts", {}))
    region_results = []
    next_action = "pass"
    for ra in actions:
        if ra.action == "recapture":
            key = _bbox_key(ra.region.bbox)
            updated_attempt_counts[key] = attempt_counts.get(ra.region.bbox, 0) + 1
        region_results.append({
            "bbox": list(ra.region.bbox),
            "confidence": ra.region.confidence,
            "defect_type_guess": ra.region.defect_type_guess,
            "action": ra.action,
        })
        if _ACTION_PRIORITY[ra.action] > _ACTION_PRIORITY[next_action]:
            next_action = ra.action

    table.put_item(Item={
        "run_id": event["run_id"],
        "seq": event["seq"],
        "regions": [
            {**r, "confidence": Decimal(str(r["confidence"]))} for r in region_results
        ],
        "next_action": next_action,
    })

    return {
        "run_id": event["run_id"],
        "seq": event["seq"],
        "regions": region_results,
        "attempt_counts": updated_attempt_counts,
        "next_action": next_action,
    }
