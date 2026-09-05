import cv2
import numpy as np


def candidate_mask(aligned_frame: np.ndarray, golden_ref: np.ndarray, thresh: int = 30) -> np.ndarray:
    """Absolute-diff two aligned same-shape BGR images into a cleaned binary candidate mask."""
    gray_frame = cv2.cvtColor(aligned_frame, cv2.COLOR_BGR2GRAY)
    gray_golden = cv2.cvtColor(golden_ref, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_frame, gray_golden)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
