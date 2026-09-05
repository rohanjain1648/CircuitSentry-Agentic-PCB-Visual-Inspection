import cv2
import numpy as np


def capture_frame(video_capture) -> np.ndarray:
    ok, frame = video_capture.read()
    if not ok:
        raise RuntimeError("Failed to read frame from camera")
    return frame


def crop_roi(frame: np.ndarray, bbox: tuple[int, int, int, int], zoom: float = 1.5) -> np.ndarray:
    x, y, w, h = bbox
    crop = frame[y:y + h, x:x + w]
    new_size = (int(w * zoom), int(h * zoom))
    return cv2.resize(crop, new_size, interpolation=cv2.INTER_CUBIC)
