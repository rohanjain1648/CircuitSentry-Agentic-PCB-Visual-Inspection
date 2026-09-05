import numpy as np
import cv2
from circuitsentry.vision.diff import candidate_mask


def test_identical_images_produce_empty_mask():
    golden = np.full((200, 200, 3), 100, dtype=np.uint8)
    mask = candidate_mask(golden, golden)
    assert mask.shape == (200, 200)
    assert mask.max() == 0


def test_added_blob_is_detected():
    golden = np.full((200, 200, 3), 100, dtype=np.uint8)
    frame = golden.copy()
    cv2.circle(frame, (100, 100), 20, (255, 255, 255), -1)  # simulate a missing/extra component
    mask = candidate_mask(frame, golden)
    assert mask[100, 100] == 255
    assert mask[5, 5] == 0
