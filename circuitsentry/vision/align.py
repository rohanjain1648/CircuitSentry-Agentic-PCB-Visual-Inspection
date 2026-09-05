import cv2
import numpy as np

MIN_MATCH_COUNT = 10


class AlignmentError(Exception):
    """Raised when a frame cannot be reliably aligned to the golden reference."""


def align(frame: np.ndarray, golden_ref: np.ndarray) -> np.ndarray:
    """Warp `frame` into `golden_ref`'s coordinate frame using ORB features + homography."""
    orb = cv2.ORB_create(nfeatures=2000)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_golden = cv2.cvtColor(golden_ref, cv2.COLOR_BGR2GRAY)

    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)
    kp_golden, des_golden = orb.detectAndCompute(gray_golden, None)

    if des_frame is None or des_golden is None or len(kp_frame) < MIN_MATCH_COUNT:
        raise AlignmentError("Not enough features detected to align frame")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_frame, des_golden)
    matches = sorted(matches, key=lambda m: m.distance)

    if len(matches) < MIN_MATCH_COUNT:
        raise AlignmentError(f"Only {len(matches)} matches found, need {MIN_MATCH_COUNT}")

    src_pts = np.float32([kp_frame[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_golden[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if homography is None:
        raise AlignmentError("Homography estimation failed")

    h, w = golden_ref.shape[:2]
    return cv2.warpPerspective(frame, homography, (w, h))
