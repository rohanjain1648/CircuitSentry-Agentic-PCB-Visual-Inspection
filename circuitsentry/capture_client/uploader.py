import cv2
import numpy as np


def upload_frame(s3_client, bucket: str, run_id: str, seq: int, frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame as PNG")
    key = f"{run_id}/{seq}/frame.png"
    s3_client.put_object(Bucket=bucket, Key=key, Body=buf.tobytes(), ContentType="image/png")
    return key
