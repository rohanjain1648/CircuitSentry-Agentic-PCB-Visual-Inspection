import numpy as np
from circuitsentry.vision.align import align
from circuitsentry.vision.diff import candidate_mask
from circuitsentry.vision.score import score_regions, Region


def inspect(frame: np.ndarray, golden_ref: np.ndarray) -> list[Region]:
    """Full perception pipeline: align -> diff -> score. Raises AlignmentError on bad input."""
    aligned = align(frame, golden_ref)
    mask = candidate_mask(aligned, golden_ref)
    return score_regions(mask)
