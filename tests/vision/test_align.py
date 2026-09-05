import numpy as np
import cv2
import pytest
from circuitsentry.vision.align import align, AlignmentError


def _synthetic_board(shift=(0, 0), size=(300, 300)):
    """A textured synthetic 'board' with a shifted copy for alignment testing."""
    rng = np.random.default_rng(42)
    base = rng.integers(0, 255, size=(size[1], size[0]), dtype=np.uint8)
    # add strong corner/blob features so ORB has something to match
    cv2.rectangle(base, (40, 40), (100, 100), 255, -1)
    cv2.rectangle(base, (180, 60), (260, 140), 0, -1)
    cv2.circle(base, (150, 220), 30, 200, -1)
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    shifted = cv2.warpAffine(base, M, size, borderValue=127)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR), cv2.cvtColor(shifted, cv2.COLOR_GRAY2BGR)


def test_align_recovers_shifted_frame_to_golden_shape():
    golden, shifted = _synthetic_board(shift=(15, -10))
    result = align(shifted, golden)
    assert result.shape == golden.shape


def test_align_raises_on_blank_frame():
    golden, _ = _synthetic_board()
    blank = np.full_like(golden, 127)
    with pytest.raises(AlignmentError):
        align(blank, golden)
