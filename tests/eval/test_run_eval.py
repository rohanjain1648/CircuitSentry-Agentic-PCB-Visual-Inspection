import json

import numpy as np
import cv2

from eval.run_eval import evaluate, load_dataset, main


def _clean_pair():
    golden = np.full((200, 200, 3), 100, dtype=np.uint8)
    cv2.rectangle(golden, (20, 20), (60, 60), (200, 200, 200), -1)
    cv2.circle(golden, (150, 150), 30, (50, 50, 50), -1)
    cv2.rectangle(golden, (20, 120), (70, 170), (0, 0, 0), -1)
    return golden.copy(), golden


def _defective_pair():
    golden = np.full((200, 200, 3), 100, dtype=np.uint8)
    cv2.rectangle(golden, (20, 20), (60, 60), (200, 200, 200), -1)
    cv2.circle(golden, (150, 150), 30, (50, 50, 50), -1)
    cv2.rectangle(golden, (20, 120), (70, 170), (0, 0, 0), -1)
    frame = golden.copy()
    cv2.rectangle(frame, (120, 20), (170, 70), (255, 255, 255), -1)  # big, unambiguous defect
    return frame, golden


def test_evaluate_returns_metrics_dict():
    clean_frame, clean_golden = _clean_pair()
    defect_frame, defect_golden = _defective_pair()
    dataset = [
        {"frame": clean_frame, "golden_ref": clean_golden, "has_defect": False},
        {"frame": defect_frame, "golden_ref": defect_golden, "has_defect": True},
    ]
    result = evaluate(dataset)
    assert set(result.keys()) == {
        "precision", "recall", "f1", "escalation_rate", "alignment_failures",
    }
    assert result["recall"] == 1.0  # the injected defect must be detected
    assert result["alignment_failures"] == 0


def _unalignable_pair():
    """Featureless flat images: ORB finds nothing, so align() raises."""
    return (
        np.full((60, 60, 3), 10, dtype=np.uint8),
        np.full((60, 60, 3), 200, dtype=np.uint8),
    )


def test_unalignable_entry_does_not_crash_and_counts_as_a_miss():
    bad_frame, bad_golden = _unalignable_pair()
    defect_frame, defect_golden = _defective_pair()
    dataset = [
        {"frame": bad_frame, "golden_ref": bad_golden, "has_defect": True},
        {"frame": defect_frame, "golden_ref": defect_golden, "has_defect": True},
    ]

    result = evaluate(dataset)  # must not raise AlignmentError

    assert result["alignment_failures"] == 1
    # One of two defective boards was detected: the misaligned one is a miss.
    assert result["recall"] == 0.5


def test_load_dataset_and_cli(tmp_path, capsys):
    clean_frame, clean_golden = _clean_pair()
    defect_frame, defect_golden = _defective_pair()
    cv2.imwrite(str(tmp_path / "clean_frame.png"), clean_frame)
    cv2.imwrite(str(tmp_path / "clean_golden.png"), clean_golden)
    cv2.imwrite(str(tmp_path / "defect_frame.png"), defect_frame)
    cv2.imwrite(str(tmp_path / "defect_golden.png"), defect_golden)
    (tmp_path / "manifest.json").write_text(json.dumps({"clean": False, "defect": True}))

    dataset = load_dataset(tmp_path)
    assert len(dataset) == 2
    assert {e["has_defect"] for e in dataset} == {True, False}

    assert main(["run_eval", str(tmp_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["recall"] == 1.0


def test_cli_rejects_wrong_argument_count():
    assert main(["run_eval"]) == 2
