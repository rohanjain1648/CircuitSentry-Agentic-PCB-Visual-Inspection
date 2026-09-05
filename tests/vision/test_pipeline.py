import numpy as np
import cv2
import pytest
from circuitsentry.vision.pipeline import inspect
from circuitsentry.vision.align import AlignmentError


def _board_with_defect():
    golden = np.full((300, 300, 3), 100, dtype=np.uint8)
    cv2.rectangle(golden, (40, 40), (100, 100), (200, 200, 200), -1)
    cv2.circle(golden, (150, 220), 30, (50, 50, 50), -1)
    frame = golden.copy()
    cv2.circle(frame, (220, 80), 25, (255, 255, 255), -1)  # extra blob = defect
    return frame, golden


def test_inspect_returns_regions_for_injected_defect():
    frame, golden = _board_with_defect()
    regions = inspect(frame, golden)
    assert len(regions) >= 1
    assert any(r.confidence > 0 for r in regions)


def test_inspect_propagates_alignment_error():
    blank = np.full((300, 300, 3), 127, dtype=np.uint8)
    golden = np.full((300, 300, 3), 100, dtype=np.uint8)
    cv2.rectangle(golden, (40, 40), (100, 100), (200, 200, 200), -1)
    with pytest.raises(AlignmentError):
        inspect(blank, golden)
