"""Evaluation harness: run the perception pipeline over a labeled dataset.

CLI usage:

    python -m eval.run_eval <dataset_dir>

Dataset directory convention (deliberately minimal):

    <dataset_dir>/
        manifest.json        {"<name>": true, "<other>": false}   # has_defect
        <name>_frame.png
        <name>_golden.png

Every key in ``manifest.json`` must have a matching ``_frame`` and ``_golden``
image (any extension OpenCV can read; ``.png`` is looked up first).
"""
import json
import sys
from pathlib import Path

import cv2

from circuitsentry.vision.align import AlignmentError
from circuitsentry.vision.pipeline import inspect
from circuitsentry.decision.engine import HIGH_CONFIDENCE, LOW_CONFIDENCE

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def evaluate(dataset: list[dict]) -> dict:
    """Run inspect() over a labeled dataset and compute detection metrics.

    Each dataset entry: {"frame": np.ndarray, "golden_ref": np.ndarray, "has_defect": bool}

    Misalignment is a documented failure case (spec §6), not a crash: an entry
    whose frame cannot be aligned to its golden reference is scored as
    "nothing detected" (a false negative if it was defective) and counted in
    ``alignment_failures``, so a single bad pair cannot abort the whole run.
    """
    true_positives = false_positives = false_negatives = 0
    escalated_count = 0
    alignment_failures = 0

    for entry in dataset:
        try:
            regions = inspect(entry["frame"], entry["golden_ref"])
        except AlignmentError:
            alignment_failures += 1
            regions = []

        detected = any(r.confidence >= HIGH_CONFIDENCE for r in regions)
        escalated = any(LOW_CONFIDENCE <= r.confidence < HIGH_CONFIDENCE for r in regions)

        if escalated:
            escalated_count += 1
        if detected and entry["has_defect"]:
            true_positives += 1
        elif detected and not entry["has_defect"]:
            false_positives += 1
        elif not detected and entry["has_defect"]:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    escalation_rate = escalated_count / len(dataset) if dataset else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "escalation_rate": round(escalation_rate, 3),
        "alignment_failures": alignment_failures,
    }


def _find_image(dataset_dir: Path, stem: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = dataset_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for '{stem}' in {dataset_dir}")


def load_dataset(dataset_dir: str | Path) -> list[dict]:
    """Load a directory of <name>_frame / <name>_golden pairs plus manifest.json."""
    dataset_dir = Path(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text())

    dataset = []
    for name, has_defect in manifest.items():
        frame = cv2.imread(str(_find_image(dataset_dir, f"{name}_frame")), cv2.IMREAD_COLOR)
        golden = cv2.imread(str(_find_image(dataset_dir, f"{name}_golden")), cv2.IMREAD_COLOR)
        if frame is None or golden is None:
            raise ValueError(f"Could not decode the image pair for '{name}'")
        dataset.append({"frame": frame, "golden_ref": golden, "has_defect": bool(has_defect)})
    return dataset


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m eval.run_eval <dataset_dir>", file=sys.stderr)
        return 2
    metrics = evaluate(load_dataset(argv[1]))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
