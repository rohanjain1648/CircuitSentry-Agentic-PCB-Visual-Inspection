import numpy as np
import cv2
from circuitsentry.vision.score import score_regions, Region


def test_no_contours_yields_no_regions():
    mask = np.zeros((200, 200), dtype=np.uint8)
    assert score_regions(mask) == []


def test_large_blob_yields_high_confidence_region():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (50, 50), (100, 100), 255, -1)  # 50x50 = 2500px blob
    regions = score_regions(mask)
    assert len(regions) == 1
    region = regions[0]
    assert isinstance(region, Region)
    assert region.confidence > 0.5
    x, y, w, h = region.bbox
    assert (x, y) == (50, 50)


def test_tiny_speck_yields_low_confidence_region():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (50, 50), (52, 52), 255, -1)  # 2x2 = 4px speck
    regions = score_regions(mask)
    assert len(regions) == 1
    assert regions[0].confidence < 0.2
