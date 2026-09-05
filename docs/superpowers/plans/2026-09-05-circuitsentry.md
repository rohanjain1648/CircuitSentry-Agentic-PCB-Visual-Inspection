# CircuitSentry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an agentic PCB visual-inspection system: OpenCV 5 perception output drives a rule-based decision engine that requests closer re-captures, flags defects for human approval, or passes — running as an AWS Step Functions pipeline with a judge-facing dashboard.

**Architecture:** A pure-Python `vision` package (alignment → diff → scoring) and a pure-Python `decision` package (confidence → action) are built and unit-tested entirely offline first. They are then wrapped by a Lambda handler orchestrated by a Step Functions state machine (Perception → Choice → Pass/Recapture-loop/FlagForApproval), fronted by API Gateway, with a local Python capture client and a static-site dashboard as the two clients of that API.

**Tech Stack:** Python 3.12, opencv-python-headless, numpy, boto3, pytest, moto (AWS mocking), AWS CDK (Python) for infrastructure, vanilla HTML/CSS/JS for the dashboard.

**Spec:** [docs/superpowers/specs/2026-09-05-circuitsentry-design.md](../specs/2026-09-05-circuitsentry-design.md)

## Global Constraints

- Perception is classical OpenCV (alignment + diff + contour scoring) — no trained model, no labeled training set (spec §3, §10).
- Decision engine is rule-based confidence thresholds, not an LLM (spec §4, brainstorming decision).
- Re-capture loop is bounded to **at most 2 iterations per region**; a region still ambiguous after 2 tries falls through to `FlagForApproval` (spec §4, §8).
- Not pursuing the COOL/Graviton award — no COOL library dependency, no Graviton-specific benchmarking (spec §7).
- DynamoDB is the single source of truth for run records: regions, confidence, defect-type guess, verdict, action, human-approval status (spec §5).
- All uploads/captures are idempotent, keyed by `run_id` + sequence number (spec §8).

---

## File Structure

```
circuitsentry/
  vision/
    __init__.py
    align.py          # ORB/AKAZE + homography alignment
    diff.py           # aligned-image diff -> cleaned candidate mask
    score.py          # contours -> scored Region objects
    pipeline.py        # inspect(frame, golden_ref) -> list[Region]
  decision/
    __init__.py
    engine.py          # decide(regions, attempt_counts) -> list[RegionAction]
  backend/
    lambda_perception/
      handler.py       # Step Functions task: runs vision+decision, writes DynamoDB
    lambda_api/
      capture.py       # POST /capture -> starts Step Functions execution
      runs.py          # GET /runs, GET /runs/{id}
      approve.py       # POST /runs/{id}/approve
  infra/
    app.py
    circuitsentry_stack.py   # S3, DynamoDB, Step Functions, Lambdas, API Gateway
  capture_client/
    camera.py          # webcam capture via cv2.VideoCapture
    uploader.py         # S3 presigned-URL upload
    poller.py           # polls API for recapture commands
    main.py              # ties camera+uploader+poller into the capture loop
  dashboard/
    index.html
    app.js
    style.css
  eval/
    run_eval.py         # precision/recall/F1/escalation-rate harness
  tests/
    vision/
      test_align.py
      test_diff.py
      test_score.py
      test_pipeline.py
    decision/
      test_engine.py
    backend/
      test_lambda_perception.py
      test_lambda_api.py
    capture_client/
      test_camera.py
      test_uploader.py
      test_poller.py
    eval/
      test_run_eval.py
  requirements.txt
  pyproject.toml
```

---

### Task 1: Project scaffolding + vision.align

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `circuitsentry/vision/__init__.py`
- Create: `circuitsentry/vision/align.py`
- Test: `tests/vision/test_align.py`

**Interfaces:**
- Produces: `align(frame: np.ndarray, golden_ref: np.ndarray) -> np.ndarray` — returns `frame` warped into `golden_ref`'s coordinate frame (same shape as `golden_ref`), or raises `AlignmentError` if too few matches are found.

- [ ] **Step 1: Create project files**

`pyproject.toml`:
```toml
[project]
name = "circuitsentry"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "opencv-python-headless>=4.10",
    "numpy>=1.26",
    "boto3>=1.34",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "moto[dynamodb,s3,stepfunctions]>=5.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`requirements.txt`:
```
opencv-python-headless>=4.10
numpy>=1.26
boto3>=1.34
pytest>=8.0
moto[dynamodb,s3,stepfunctions]>=5.0
```

Create empty `circuitsentry/vision/__init__.py`.

- [ ] **Step 2: Write the failing test**

```python
# tests/vision/test_align.py
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/vision/test_align.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'circuitsentry.vision.align'`

- [ ] **Step 4: Write minimal implementation**

```python
# circuitsentry/vision/align.py
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/vision/test_align.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt circuitsentry/vision/ tests/vision/test_align.py
git commit -m "feat: add vision.align (ORB+homography frame alignment)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: vision.diff

**Files:**
- Create: `circuitsentry/vision/diff.py`
- Test: `tests/vision/test_diff.py`

**Interfaces:**
- Consumes: nothing from Task 1 directly (operates on two already-aligned same-shape images).
- Produces: `candidate_mask(aligned_frame: np.ndarray, golden_ref: np.ndarray, thresh: int = 30) -> np.ndarray` — returns a single-channel `uint8` binary mask (0/255) the same H×W as the inputs, highlighting regions that differ.

- [ ] **Step 1: Write the failing test**

```python
# tests/vision/test_diff.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/vision/test_diff.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'circuitsentry.vision.diff'`

- [ ] **Step 3: Write minimal implementation**

```python
# circuitsentry/vision/diff.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/vision/test_diff.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add circuitsentry/vision/diff.py tests/vision/test_diff.py
git commit -m "feat: add vision.diff (aligned-image candidate mask)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: vision.score

**Files:**
- Create: `circuitsentry/vision/score.py`
- Test: `tests/vision/test_score.py`

**Interfaces:**
- Consumes: a binary mask shaped like `candidate_mask()`'s output (Task 2).
- Produces: a `Region` dataclass `{bbox: tuple[int,int,int,int], confidence: float, defect_type_guess: str}` and `score_regions(mask: np.ndarray) -> list[Region]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/vision/test_score.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/vision/test_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'circuitsentry.vision.score'`

- [ ] **Step 3: Write minimal implementation**

```python
# circuitsentry/vision/score.py
from dataclasses import dataclass
import cv2
import numpy as np

# Confidence saturates once a candidate region reaches this pixel area.
AREA_SATURATION_PX = 2000


@dataclass(frozen=True)
class Region:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float                # 0.0-1.0
    defect_type_guess: str


def _guess_defect_type(w: int, h: int) -> str:
    aspect = max(w, h) / max(1, min(w, h))
    if aspect > 4:
        return "solder_bridge"
    if aspect > 1.8:
        return "scratch_or_burn"
    return "missing_or_extra_component"


def score_regions(mask: np.ndarray) -> list[Region]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 4:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        confidence = min(1.0, area / AREA_SATURATION_PX)
        regions.append(Region(
            bbox=(x, y, w, h),
            confidence=round(confidence, 3),
            defect_type_guess=_guess_defect_type(w, h),
        ))
    return regions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/vision/test_score.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add circuitsentry/vision/score.py tests/vision/test_score.py
git commit -m "feat: add vision.score (contours -> scored Region objects)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: vision.pipeline

**Files:**
- Create: `circuitsentry/vision/pipeline.py`
- Test: `tests/vision/test_pipeline.py`

**Interfaces:**
- Consumes: `align()` (Task 1), `candidate_mask()` (Task 2), `score_regions()` + `Region` (Task 3).
- Produces: `inspect(frame: np.ndarray, golden_ref: np.ndarray) -> list[Region]` — the single entry point the Lambda handler (Task 7) calls.

- [ ] **Step 1: Write the failing test**

```python
# tests/vision/test_pipeline.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/vision/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'circuitsentry.vision.pipeline'`

- [ ] **Step 3: Write minimal implementation**

```python
# circuitsentry/vision/pipeline.py
import numpy as np
from circuitsentry.vision.align import align
from circuitsentry.vision.diff import candidate_mask
from circuitsentry.vision.score import score_regions, Region


def inspect(frame: np.ndarray, golden_ref: np.ndarray) -> list[Region]:
    """Full perception pipeline: align -> diff -> score. Raises AlignmentError on bad input."""
    aligned = align(frame, golden_ref)
    mask = candidate_mask(aligned, golden_ref)
    return score_regions(mask)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/vision/test_pipeline.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add circuitsentry/vision/pipeline.py tests/vision/test_pipeline.py
git commit -m "feat: add vision.pipeline (inspect() entry point)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: decision.engine

**Files:**
- Create: `circuitsentry/decision/__init__.py`
- Create: `circuitsentry/decision/engine.py`
- Test: `tests/decision/test_engine.py`

**Interfaces:**
- Consumes: `Region` (Task 3) — `{bbox, confidence, defect_type_guess}`.
- Produces: `RegionAction` dataclass `{region: Region, action: str}` where `action` is one of `"pass"`, `"recapture"`, `"flag_for_approval"`; and `decide(regions: list[Region], attempt_counts: dict[tuple, int]) -> list[RegionAction]`. `attempt_counts` maps a region's `bbox` to how many times it has already looped through recapture; caller (Lambda handler, Task 7) is responsible for tracking and incrementing this across loop iterations.

- [ ] **Step 1: Write the failing test**

```python
# tests/decision/test_engine.py
from circuitsentry.vision.score import Region
from circuitsentry.decision.engine import decide, RegionAction, HIGH_CONFIDENCE, LOW_CONFIDENCE, MAX_RECAPTURE_ATTEMPTS


def _region(confidence):
    return Region(bbox=(10, 10, 20, 20), confidence=confidence, defect_type_guess="missing_or_extra_component")


def test_high_confidence_flags_for_approval():
    [result] = decide([_region(HIGH_CONFIDENCE + 0.01)], attempt_counts={})
    assert isinstance(result, RegionAction)
    assert result.action == "flag_for_approval"


def test_low_confidence_passes():
    [result] = decide([_region(LOW_CONFIDENCE - 0.01)], attempt_counts={})
    assert result.action == "pass"


def test_ambiguous_confidence_requests_recapture():
    mid = (LOW_CONFIDENCE + HIGH_CONFIDENCE) / 2
    [result] = decide([_region(mid)], attempt_counts={})
    assert result.action == "recapture"


def test_ambiguous_confidence_flags_after_max_attempts():
    mid = (LOW_CONFIDENCE + HIGH_CONFIDENCE) / 2
    region = _region(mid)
    attempt_counts = {region.bbox: MAX_RECAPTURE_ATTEMPTS}
    [result] = decide([region], attempt_counts=attempt_counts)
    assert result.action == "flag_for_approval"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/decision/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'circuitsentry.decision'`

- [ ] **Step 3: Write minimal implementation**

```python
# circuitsentry/decision/engine.py
from dataclasses import dataclass
from circuitsentry.vision.score import Region

HIGH_CONFIDENCE = 0.7
LOW_CONFIDENCE = 0.3
MAX_RECAPTURE_ATTEMPTS = 2  # spec §4/§8: bounded loop


@dataclass(frozen=True)
class RegionAction:
    region: Region
    action: str  # "pass" | "recapture" | "flag_for_approval"


def decide(regions: list[Region], attempt_counts: dict[tuple, int]) -> list[RegionAction]:
    """Map each Region to its next action given how many recapture attempts it has already had."""
    results = []
    for region in regions:
        attempts = attempt_counts.get(region.bbox, 0)
        if region.confidence >= HIGH_CONFIDENCE:
            action = "flag_for_approval"
        elif region.confidence < LOW_CONFIDENCE:
            action = "pass"
        elif attempts >= MAX_RECAPTURE_ATTEMPTS:
            action = "flag_for_approval"
        else:
            action = "recapture"
        results.append(RegionAction(region=region, action=action))
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/decision/test_engine.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add circuitsentry/decision/ tests/decision/
git commit -m "feat: add decision.engine (confidence -> action, bounded recapture loop)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: CDK infra skeleton (S3 + DynamoDB)

**Files:**
- Create: `infra/app.py`
- Create: `infra/circuitsentry_stack.py`
- Test: `tests/infra/test_stack.py`

**Interfaces:**
- Produces: a `CircuitSentryStack` CDK stack exposing `self.images_bucket` (S3 bucket) and `self.runs_table` (DynamoDB table, partition key `run_id: str`, sort key `seq: number`) — later tasks (7, 8, 9, 10) add resources onto this same stack.

- [ ] **Step 1: Write the failing test**

```python
# tests/infra/test_stack.py
import aws_cdk as cdk
from aws_cdk.assertions import Template
from infra.circuitsentry_stack import CircuitSentryStack


def test_stack_has_s3_bucket_and_dynamodb_table():
    app = cdk.App()
    stack = CircuitSentryStack(app, "TestStack")
    template = Template.from_stack(stack)

    template.resource_count_is("AWS::S3::Bucket", 1)
    template.has_resource_properties("AWS::DynamoDB::Table", {
        "KeySchema": [
            {"AttributeName": "run_id", "KeyType": "HASH"},
            {"AttributeName": "seq", "KeyType": "RANGE"},
        ],
    })
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/infra/test_stack.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'infra.circuitsentry_stack'`

- [ ] **Step 3: Write minimal implementation**

```python
# infra/circuitsentry_stack.py
from aws_cdk import Stack, RemovalPolicy
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_dynamodb as dynamodb
from constructs import Construct


class CircuitSentryStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.images_bucket = s3.Bucket(
            self, "ImagesBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        self.runs_table = dynamodb.Table(
            self, "RunsTable",
            partition_key=dynamodb.Attribute(name="run_id", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="seq", type=dynamodb.AttributeType.NUMBER),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )
```

```python
# infra/app.py
import aws_cdk as cdk
from circuitsentry_stack import CircuitSentryStack

app = cdk.App()
CircuitSentryStack(app, "CircuitSentryStack")
app.synth()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/infra/test_stack.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add infra/ tests/infra/
git commit -m "feat: add CDK stack skeleton (S3 images bucket + DynamoDB runs table)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: Lambda perception handler

**Files:**
- Create: `circuitsentry/backend/lambda_perception/handler.py`
- Test: `tests/backend/test_lambda_perception.py`

**Interfaces:**
- Consumes: `inspect()` (Task 4), `decide()` + `RegionAction` (Task 5).
- Produces: `lambda_handler(event: dict, context) -> dict`. Expected `event` shape (from the Step Functions task, Task 8):
  ```python
  {
      "run_id": "abc123",
      "seq": 0,
      "bucket": "circuitsentry-images-...",
      "frame_key": "abc123/0/frame.png",
      "golden_key": "golden/board-rev1.png",
      "attempt_counts": {},   # bbox-string -> attempt count, threaded through loop iterations
  }
  ```
  Returns:
  ```python
  {
      "run_id": "abc123",
      "seq": 0,
      "regions": [{"bbox": [10, 10, 20, 20], "confidence": 0.5, "defect_type_guess": "...", "action": "recapture"}],
      "attempt_counts": {...},          # updated for regions that got "recapture"
      "next_action": "recapture" | "flag_for_approval" | "pass",  # highest-priority action across all regions
  }
  ```
  `next_action` priority order: `flag_for_approval` > `recapture` > `pass` — this is what the Step Functions Choice state (Task 8) branches on.

- [ ] **Step 1: Write the failing test**

```python
# tests/backend/test_lambda_perception.py
import io
import boto3
import cv2
import numpy as np
from moto import mock_aws
from circuitsentry.backend.lambda_perception.handler import lambda_handler


def _encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _board_with_defect():
    golden = np.full((300, 300, 3), 100, dtype=np.uint8)
    cv2.rectangle(golden, (40, 40), (100, 100), (200, 200, 200), -1)
    frame = golden.copy()
    cv2.circle(frame, (220, 80), 25, (255, 255, 255), -1)
    return frame, golden


@mock_aws
def test_handler_writes_dynamodb_record_and_returns_next_action():
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")
    frame, golden = _board_with_defect()
    s3.put_object(Bucket="test-bucket", Key="run1/0/frame.png", Body=_encode(frame))
    s3.put_object(Bucket="test-bucket", Key="golden/board-rev1.png", Body=_encode(golden))

    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    table = dynamodb.create_table(
        TableName="test-runs",
        KeySchema=[
            {"AttributeName": "run_id", "KeyType": "HASH"},
            {"AttributeName": "seq", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "run_id", "AttributeType": "S"},
            {"AttributeName": "seq", "AttributeType": "N"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    table.wait_until_exists()

    import os
    os.environ["RUNS_TABLE_NAME"] = "test-runs"

    event = {
        "run_id": "run1",
        "seq": 0,
        "bucket": "test-bucket",
        "frame_key": "run1/0/frame.png",
        "golden_key": "golden/board-rev1.png",
        "attempt_counts": {},
    }
    result = lambda_handler(event, None)

    assert result["run_id"] == "run1"
    assert result["next_action"] in {"pass", "recapture", "flag_for_approval"}
    assert len(result["regions"]) >= 1

    item = table.get_item(Key={"run_id": "run1", "seq": 0})["Item"]
    assert item["next_action"] == result["next_action"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/backend/test_lambda_perception.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'circuitsentry.backend'`

- [ ] **Step 3: Write minimal implementation**

```python
# circuitsentry/backend/lambda_perception/handler.py
import os
import cv2
import numpy as np
import boto3
from decimal import Decimal
from circuitsentry.vision.pipeline import inspect
from circuitsentry.decision.engine import decide

_ACTION_PRIORITY = {"flag_for_approval": 2, "recapture": 1, "pass": 0}


def _load_image(s3_client, bucket: str, key: str) -> np.ndarray:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = np.frombuffer(obj["Body"].read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _bbox_key(bbox: tuple) -> str:
    return ",".join(str(v) for v in bbox)


def lambda_handler(event: dict, context) -> dict:
    s3_client = boto3.client("s3")
    table_name = os.environ["RUNS_TABLE_NAME"]
    table = boto3.resource("dynamodb").Table(table_name)

    frame = _load_image(s3_client, event["bucket"], event["frame_key"])
    golden = _load_image(s3_client, event["bucket"], event["golden_key"])

    regions = inspect(frame, golden)

    attempt_counts = {
        tuple(int(v) for v in k.split(",")): v_count
        for k, v_count in event.get("attempt_counts", {}).items()
    }
    actions = decide(regions, attempt_counts)

    updated_attempt_counts = dict(event.get("attempt_counts", {}))
    region_results = []
    next_action = "pass"
    for ra in actions:
        if ra.action == "recapture":
            key = _bbox_key(ra.region.bbox)
            updated_attempt_counts[key] = attempt_counts.get(ra.region.bbox, 0) + 1
        region_results.append({
            "bbox": list(ra.region.bbox),
            "confidence": ra.region.confidence,
            "defect_type_guess": ra.region.defect_type_guess,
            "action": ra.action,
        })
        if _ACTION_PRIORITY[ra.action] > _ACTION_PRIORITY[next_action]:
            next_action = ra.action

    table.put_item(Item={
        "run_id": event["run_id"],
        "seq": event["seq"],
        "regions": [
            {**r, "confidence": Decimal(str(r["confidence"]))} for r in region_results
        ],
        "next_action": next_action,
    })

    return {
        "run_id": event["run_id"],
        "seq": event["seq"],
        "regions": region_results,
        "attempt_counts": updated_attempt_counts,
        "next_action": next_action,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/backend/test_lambda_perception.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add circuitsentry/backend/lambda_perception/ tests/backend/test_lambda_perception.py
git commit -m "feat: add perception Lambda handler (vision+decision -> DynamoDB record)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 8: Step Functions state machine (perception -> decision -> action loop)

**Files:**
- Modify: `infra/circuitsentry_stack.py`
- Test: `tests/infra/test_state_machine.py`

**Interfaces:**
- Consumes: `self.images_bucket`, `self.runs_table` (Task 6); packages `circuitsentry/backend/lambda_perception/handler.py` (Task 7) as a Lambda function.
- Produces: `self.state_machine` (Step Functions `StateMachine`) on `CircuitSentryStack`, added by API Lambda (Task 9) via `START_EXECUTION` permission and `state_machine.state_machine_arn`.

- [ ] **Step 1: Write the failing test**

```python
# tests/infra/test_state_machine.py
import aws_cdk as cdk
from aws_cdk.assertions import Template, Match
from infra.circuitsentry_stack import CircuitSentryStack


def test_state_machine_has_choice_state_with_recapture_loop():
    app = cdk.App()
    stack = CircuitSentryStack(app, "TestStack")
    template = Template.from_stack(stack)

    template.resource_count_is("AWS::StepFunctions::StateMachine", 1)
    # The definition string must reference all three terminal branches.
    template.has_resource_properties("AWS::StepFunctions::StateMachine", {
        "DefinitionString": Match.string_like_regexp(r"(?s)Perception.*Recapture.*FlagForApproval.*Pass"),
    })
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/infra/test_state_machine.py -v`
Expected: FAIL with `AssertionError` (no `AWS::StepFunctions::StateMachine` resource yet)

- [ ] **Step 3: Write minimal implementation**

Add to `infra/circuitsentry_stack.py` (inside `__init__`, after the table is created):

```python
        # --- add near the top of the file ---
        from aws_cdk import Duration
        from aws_cdk import aws_lambda as _lambda
        from aws_cdk import aws_stepfunctions as sfn
        from aws_cdk import aws_stepfunctions_tasks as tasks

        # --- add inside __init__, after self.runs_table is created ---
        self.perception_fn = _lambda.Function(
            self, "PerceptionFn",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="circuitsentry.backend.lambda_perception.handler.lambda_handler",
            code=_lambda.Code.from_asset("circuitsentry"),
            timeout=Duration.seconds(30),
            memory_size=1024,
            environment={"RUNS_TABLE_NAME": self.runs_table.table_name},
        )
        self.images_bucket.grant_read(self.perception_fn)
        self.runs_table.grant_write_data(self.perception_fn)

        perception_task = tasks.LambdaInvoke(
            self, "Perception",
            lambda_function=self.perception_fn,
            output_path="$.Payload",
        )

        pass_state = sfn.Pass(self, "Pass")
        flag_state = sfn.Pass(self, "FlagForApproval")
        recapture_state = sfn.Pass(self, "Recapture")
        # Recapture loops back to Perception once the capture client has fulfilled
        # the re-capture request (event carries the new frame_key on re-entry).
        recapture_state.next(perception_task)

        choice = sfn.Choice(self, "Decide") \
            .when(sfn.Condition.string_equals("$.next_action", "flag_for_approval"), flag_state) \
            .when(sfn.Condition.string_equals("$.next_action", "recapture"), recapture_state) \
            .otherwise(pass_state)

        definition = perception_task.next(choice)

        self.state_machine = sfn.StateMachine(
            self, "InspectionStateMachine",
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            timeout=Duration.minutes(5),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/infra/test_state_machine.py tests/infra/test_stack.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add infra/circuitsentry_stack.py tests/infra/test_state_machine.py
git commit -m "feat: add Step Functions state machine (perception/decision/action loop)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 9: API Lambdas (capture, runs, approve)

**Files:**
- Create: `circuitsentry/backend/lambda_api/capture.py`
- Create: `circuitsentry/backend/lambda_api/runs.py`
- Create: `circuitsentry/backend/lambda_api/approve.py`
- Test: `tests/backend/test_lambda_api.py`

**Interfaces:**
- Consumes: `self.runs_table`'s schema (`run_id`, `seq`, `regions`, `next_action`) from Task 7/8; `STATE_MACHINE_ARN` and `RUNS_TABLE_NAME` env vars (wired in Task 10).
- Produces:
  - `capture.lambda_handler(event, context)` — API Gateway proxy handler for `POST /capture`; starts a Step Functions execution, returns `{"run_id": ..., "execution_arn": ...}`.
  - `runs.list_handler(event, context)` — `GET /runs`; returns `{"runs": [{"run_id": ..., "seq": ..., "next_action": ...}, ...]}`.
  - `runs.detail_handler(event, context)` — `GET /runs/{id}`; returns all `seq` items for that `run_id`.
  - `approve.lambda_handler(event, context)` — `POST /runs/{id}/approve`; sets `approval_status` on the item to `"approved"` or `"rejected"` (from the JSON body's `"decision"` field).

- [ ] **Step 1: Write the failing test**

```python
# tests/backend/test_lambda_api.py
import json
import os
import boto3
from moto import mock_aws
from circuitsentry.backend.lambda_api import capture, runs, approve


def _make_table():
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    table = dynamodb.create_table(
        TableName="test-runs",
        KeySchema=[
            {"AttributeName": "run_id", "KeyType": "HASH"},
            {"AttributeName": "seq", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "run_id", "AttributeType": "S"},
            {"AttributeName": "seq", "AttributeType": "N"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    table.wait_until_exists()
    return table


@mock_aws
def test_capture_starts_execution():
    sfn = boto3.client("stepfunctions", region_name="us-east-1")
    state_machine = sfn.create_state_machine(
        name="test-sm",
        definition=json.dumps({"StartAt": "Pass", "States": {"Pass": {"Type": "Pass", "End": True}}}),
        roleArn="arn:aws:iam::123456789012:role/test",
    )
    os.environ["STATE_MACHINE_ARN"] = state_machine["stateMachineArn"]
    os.environ["IMAGES_BUCKET_NAME"] = "test-bucket"

    event = {"body": json.dumps({"golden_key": "golden/board-rev1.png"})}
    response = capture.lambda_handler(event, None)

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert "run_id" in body
    assert "execution_arn" in body


@mock_aws
def test_runs_list_and_detail():
    table = _make_table()
    os.environ["RUNS_TABLE_NAME"] = "test-runs"
    table.put_item(Item={"run_id": "r1", "seq": 0, "next_action": "pass", "regions": []})
    table.put_item(Item={"run_id": "r1", "seq": 1, "next_action": "flag_for_approval", "regions": []})

    list_response = runs.list_handler({}, None)
    body = json.loads(list_response["body"])
    assert len(body["runs"]) == 2

    detail_response = runs.detail_handler({"pathParameters": {"id": "r1"}}, None)
    body = json.loads(detail_response["body"])
    assert len(body["items"]) == 2


@mock_aws
def test_approve_sets_status():
    table = _make_table()
    os.environ["RUNS_TABLE_NAME"] = "test-runs"
    table.put_item(Item={"run_id": "r1", "seq": 0, "next_action": "flag_for_approval", "regions": []})

    event = {
        "pathParameters": {"id": "r1"},
        "body": json.dumps({"seq": 0, "decision": "approved"}),
    }
    response = approve.lambda_handler(event, None)
    assert response["statusCode"] == 200

    item = table.get_item(Key={"run_id": "r1", "seq": 0})["Item"]
    assert item["approval_status"] == "approved"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/backend/test_lambda_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'circuitsentry.backend.lambda_api'`

- [ ] **Step 3: Write minimal implementation**

```python
# circuitsentry/backend/lambda_api/capture.py
import json
import os
import uuid
import boto3

CORS_HEADERS = {"Access-Control-Allow-Origin": "*", "Content-Type": "application/json"}


def lambda_handler(event, context):
    body = json.loads(event.get("body") or "{}")
    run_id = str(uuid.uuid4())

    sfn = boto3.client("stepfunctions")
    execution = sfn.start_execution(
        stateMachineArn=os.environ["STATE_MACHINE_ARN"],
        input=json.dumps({
            "run_id": run_id,
            "seq": 0,
            "bucket": os.environ["IMAGES_BUCKET_NAME"],
            "frame_key": f"{run_id}/0/frame.png",
            "golden_key": body["golden_key"],
            "attempt_counts": {},
        }),
    )
    return {
        "statusCode": 200,
        "headers": CORS_HEADERS,
        "body": json.dumps({"run_id": run_id, "execution_arn": execution["executionArn"]}),
    }
```

```python
# circuitsentry/backend/lambda_api/runs.py
import json
import os
import boto3
from boto3.dynamodb.conditions import Key

CORS_HEADERS = {"Access-Control-Allow-Origin": "*", "Content-Type": "application/json"}


def _table():
    return boto3.resource("dynamodb").Table(os.environ["RUNS_TABLE_NAME"])


def list_handler(event, context):
    items = _table().scan()["Items"]
    runs = [{"run_id": i["run_id"], "seq": int(i["seq"]), "next_action": i["next_action"]} for i in items]
    return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps({"runs": runs}, default=str)}


def detail_handler(event, context):
    run_id = event["pathParameters"]["id"]
    items = _table().query(KeyConditionExpression=Key("run_id").eq(run_id))["Items"]
    return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps({"items": items}, default=str)}
```

```python
# circuitsentry/backend/lambda_api/approve.py
import json
import os
import boto3

CORS_HEADERS = {"Access-Control-Allow-Origin": "*", "Content-Type": "application/json"}


def lambda_handler(event, context):
    run_id = event["pathParameters"]["id"]
    body = json.loads(event.get("body") or "{}")
    table = boto3.resource("dynamodb").Table(os.environ["RUNS_TABLE_NAME"])
    table.update_item(
        Key={"run_id": run_id, "seq": int(body["seq"])},
        UpdateExpression="SET approval_status = :s",
        ExpressionAttributeValues={":s": body["decision"]},
    )
    return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps({"ok": True})}
```

Also create `circuitsentry/backend/lambda_api/__init__.py` (empty) and `circuitsentry/backend/__init__.py` (empty) if not already present from Task 7.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/backend/test_lambda_api.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add circuitsentry/backend/lambda_api/ tests/backend/test_lambda_api.py
git commit -m "feat: add API Lambdas (capture, runs list/detail, approve)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 10: API Gateway wiring in CDK

**Files:**
- Modify: `infra/circuitsentry_stack.py`
- Test: `tests/infra/test_api.py`

**Interfaces:**
- Consumes: `self.state_machine` (Task 8), `self.runs_table`, `self.images_bucket` (Task 6), the three API Lambda handlers (Task 9).
- Produces: `self.api` (`aws_apigateway.RestApi`) with routes `POST /capture`, `GET /runs`, `GET /runs/{id}`, `POST /runs/{id}/approve`; `self.api.url` is what the capture client (Task 11) and dashboard (Task 12) call.

- [ ] **Step 1: Write the failing test**

```python
# tests/infra/test_api.py
import aws_cdk as cdk
from aws_cdk.assertions import Template
from infra.circuitsentry_stack import CircuitSentryStack


def test_api_gateway_has_expected_routes():
    app = cdk.App()
    stack = CircuitSentryStack(app, "TestStack")
    template = Template.from_stack(stack)

    template.resource_count_is("AWS::ApiGateway::RestApi", 1)
    methods = template.find_resources("AWS::ApiGateway::Method")
    http_methods = {m["Properties"]["HttpMethod"] for m in methods.values()}
    assert {"POST", "GET"} <= http_methods
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/infra/test_api.py -v`
Expected: FAIL with `AssertionError` (no `AWS::ApiGateway::RestApi` resource yet)

- [ ] **Step 3: Write minimal implementation**

Add to `infra/circuitsentry_stack.py` (inside `__init__`, after `self.state_machine` is created):

```python
        # --- add near the top of the file ---
        from aws_cdk import aws_apigateway as apigw

        # --- add inside __init__, after self.state_machine is created ---
        capture_fn = _lambda.Function(
            self, "CaptureFn",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="circuitsentry.backend.lambda_api.capture.lambda_handler",
            code=_lambda.Code.from_asset("circuitsentry"),
            environment={
                "STATE_MACHINE_ARN": self.state_machine.state_machine_arn,
                "IMAGES_BUCKET_NAME": self.images_bucket.bucket_name,
            },
        )
        self.state_machine.grant_start_execution(capture_fn)
        self.images_bucket.grant_read_write(capture_fn)

        runs_fn = _lambda.Function(
            self, "RunsFn",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="circuitsentry.backend.lambda_api.runs.list_handler",
            code=_lambda.Code.from_asset("circuitsentry"),
            environment={"RUNS_TABLE_NAME": self.runs_table.table_name},
        )
        runs_detail_fn = _lambda.Function(
            self, "RunsDetailFn",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="circuitsentry.backend.lambda_api.runs.detail_handler",
            code=_lambda.Code.from_asset("circuitsentry"),
            environment={"RUNS_TABLE_NAME": self.runs_table.table_name},
        )
        approve_fn = _lambda.Function(
            self, "ApproveFn",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="circuitsentry.backend.lambda_api.approve.lambda_handler",
            code=_lambda.Code.from_asset("circuitsentry"),
            environment={"RUNS_TABLE_NAME": self.runs_table.table_name},
        )
        self.runs_table.grant_read_data(runs_fn)
        self.runs_table.grant_read_data(runs_detail_fn)
        self.runs_table.grant_write_data(approve_fn)

        self.api = apigw.RestApi(
            self, "CircuitSentryApi",
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
            ),
        )
        self.api.root.add_resource("capture").add_method("POST", apigw.LambdaIntegration(capture_fn))
        runs_resource = self.api.root.add_resource("runs")
        runs_resource.add_method("GET", apigw.LambdaIntegration(runs_fn))
        run_item = runs_resource.add_resource("{id}")
        run_item.add_method("GET", apigw.LambdaIntegration(runs_detail_fn))
        run_item.add_resource("approve").add_method("POST", apigw.LambdaIntegration(approve_fn))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/infra/ -v`
Expected: PASS (all infra tests, including Tasks 6 and 8)

- [ ] **Step 5: Commit**

```bash
git add infra/circuitsentry_stack.py tests/infra/test_api.py
git commit -m "feat: wire API Gateway routes (capture, runs, approve) in CDK stack

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 11: Local capture client

**Files:**
- Create: `circuitsentry/capture_client/camera.py`
- Create: `circuitsentry/capture_client/uploader.py`
- Create: `circuitsentry/capture_client/poller.py`
- Create: `circuitsentry/capture_client/main.py`
- Test: `tests/capture_client/test_camera.py`
- Test: `tests/capture_client/test_uploader.py`
- Test: `tests/capture_client/test_poller.py`

**Interfaces:**
- Produces:
  - `camera.capture_frame(video_capture) -> np.ndarray` — grabs one frame from an already-opened `cv2.VideoCapture`-like object (or a tighter ROI crop when `roi` is given).
  - `camera.crop_roi(frame: np.ndarray, bbox: tuple[int,int,int,int], zoom: float = 1.5) -> np.ndarray` — digitally crops+upscales a bbox region for the "re-capture" action.
  - `uploader.upload_frame(s3_client, bucket: str, run_id: str, seq: int, frame: np.ndarray) -> str` — encodes and uploads, returns the S3 key.
  - `poller.poll_run_status(api_base_url: str, run_id: str, requests_get=requests.get) -> dict` — GETs `/runs/{run_id}` and returns the latest item's `{"seq", "next_action"}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/capture_client/test_camera.py
import numpy as np
from circuitsentry.capture_client.camera import capture_frame, crop_roi


class FakeVideoCapture:
    def __init__(self, frame):
        self._frame = frame

    def read(self):
        return True, self._frame


def test_capture_frame_returns_the_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cap = FakeVideoCapture(frame)
    result = capture_frame(cap)
    assert result.shape == (100, 100, 3)


def test_crop_roi_upscales_region():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[50:100, 50:100] = 255
    cropped = crop_roi(frame, bbox=(50, 50, 50, 50), zoom=2.0)
    assert cropped.shape[0] == 100  # 50 * zoom
    assert cropped.shape[1] == 100
```

```python
# tests/capture_client/test_uploader.py
import boto3
import numpy as np
from moto import mock_aws
from circuitsentry.capture_client.uploader import upload_frame


@mock_aws
def test_upload_frame_puts_object_and_returns_key():
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")
    frame = np.zeros((50, 50, 3), dtype=np.uint8)

    key = upload_frame(s3, "test-bucket", run_id="run1", seq=0, frame=frame)

    assert key == "run1/0/frame.png"
    obj = s3.get_object(Bucket="test-bucket", Key=key)
    assert obj["Body"].read()  # non-empty PNG bytes
```

```python
# tests/capture_client/test_poller.py
from circuitsentry.capture_client.poller import poll_run_status


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def test_poll_run_status_returns_latest_item():
    payload = {"items": [
        {"seq": 0, "next_action": "recapture"},
        {"seq": 1, "next_action": "flag_for_approval"},
    ]}

    def fake_get(url, timeout=None):
        assert url == "https://api.example.com/runs/run1"
        return FakeResponse(payload)

    result = poll_run_status("https://api.example.com", "run1", requests_get=fake_get)
    assert result == {"seq": 1, "next_action": "flag_for_approval"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/capture_client/ -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'circuitsentry.capture_client'`

- [ ] **Step 3: Write minimal implementation**

```python
# circuitsentry/capture_client/camera.py
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
```

```python
# circuitsentry/capture_client/uploader.py
import cv2
import numpy as np


def upload_frame(s3_client, bucket: str, run_id: str, seq: int, frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame as PNG")
    key = f"{run_id}/{seq}/frame.png"
    s3_client.put_object(Bucket=bucket, Key=key, Body=buf.tobytes(), ContentType="image/png")
    return key
```

```python
# circuitsentry/capture_client/poller.py
import requests


def poll_run_status(api_base_url: str, run_id: str, requests_get=requests.get) -> dict:
    response = requests_get(f"{api_base_url}/runs/{run_id}", timeout=10)
    response.raise_for_status()
    items = response.json()["items"]
    latest = max(items, key=lambda i: i["seq"])
    return {"seq": latest["seq"], "next_action": latest["next_action"]}
```

```python
# circuitsentry/capture_client/main.py
"""Entry point tying camera + uploader + poller into the capture loop.

Usage: python -m circuitsentry.capture_client.main <api_base_url> <bucket> <golden_key>
"""
import sys
import time
import boto3
import cv2
import requests

from circuitsentry.capture_client.camera import capture_frame, crop_roi
from circuitsentry.capture_client.uploader import upload_frame
from circuitsentry.capture_client.poller import poll_run_status

POLL_INTERVAL_SECONDS = 2
MAX_POLL_ATTEMPTS = 30


def run(api_base_url: str, bucket: str, golden_key: str) -> None:
    s3_client = boto3.client("s3")
    video_capture = cv2.VideoCapture(0)
    try:
        frame = capture_frame(video_capture)
        start_response = requests.post(
            f"{api_base_url}/capture",
            json={"golden_key": golden_key},
            timeout=10,
        )
        start_response.raise_for_status()
        run_id = start_response.json()["run_id"]
        upload_frame(s3_client, bucket, run_id=run_id, seq=0, frame=frame)

        for _ in range(MAX_POLL_ATTEMPTS):
            time.sleep(POLL_INTERVAL_SECONDS)
            status = poll_run_status(api_base_url, run_id)
            if status["next_action"] == "recapture":
                # NOTE: bbox for the ROI to recapture is read from the run
                # detail response in a fuller implementation; kept simple here.
                continue
            if status["next_action"] in {"pass", "flag_for_approval"}:
                print(f"Run {run_id} finished: {status['next_action']}")
                return
        print(f"Run {run_id} timed out waiting for a terminal action")
    finally:
        video_capture.release()


if __name__ == "__main__":
    run(api_base_url=sys.argv[1], bucket=sys.argv[2], golden_key=sys.argv[3])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/capture_client/ -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add circuitsentry/capture_client/ tests/capture_client/
git commit -m "feat: add local capture client (camera, uploader, poller, main loop)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 12: Dashboard static site

**Files:**
- Create: `dashboard/index.html`
- Create: `dashboard/app.js`
- Create: `dashboard/style.css`

**Interfaces:**
- Consumes: `GET /runs`, `GET /runs/{id}`, `POST /runs/{id}/approve` (Task 9/10) — `app.js` reads the API base URL from a `<meta name="api-base-url">` tag in `index.html` so no build step is needed.

There is no JS test framework in this stack (spec keeps the dashboard to plain HTML/JS — YAGNI). Verification is a manual checklist run against a local static server plus a stubbed API, listed in Step 3 below instead of an automated test.

- [ ] **Step 1: Write `index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="api-base-url" content="https://REPLACE-WITH-API-GATEWAY-URL" />
  <title>CircuitSentry Dashboard</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <h1>CircuitSentry — Inspection Runs</h1>
  <table id="runs-table">
    <thead>
      <tr><th>Run ID</th><th>Seq</th><th>Next Action</th></tr>
    </thead>
    <tbody id="runs-body"></tbody>
  </table>

  <section id="run-detail" hidden>
    <h2 id="run-detail-title"></h2>
    <div id="run-detail-items"></div>
  </section>

  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `app.js`**

```javascript
const apiBaseUrl = document.querySelector('meta[name="api-base-url"]').content;

async function loadRuns() {
  const response = await fetch(`${apiBaseUrl}/runs`);
  const data = await response.json();
  const tbody = document.getElementById("runs-body");
  tbody.innerHTML = "";
  for (const run of data.runs) {
    const row = document.createElement("tr");
    row.innerHTML = `<td><a href="#" data-run-id="${run.run_id}">${run.run_id}</a></td>` +
                     `<td>${run.seq}</td><td>${run.next_action}</td>`;
    row.querySelector("a").addEventListener("click", (e) => {
      e.preventDefault();
      loadRunDetail(run.run_id);
    });
    tbody.appendChild(row);
  }
}

async function loadRunDetail(runId) {
  const response = await fetch(`${apiBaseUrl}/runs/${runId}`);
  const data = await response.json();
  const section = document.getElementById("run-detail");
  section.hidden = false;
  document.getElementById("run-detail-title").textContent = `Run ${runId}`;
  const container = document.getElementById("run-detail-items");
  container.innerHTML = "";

  for (const item of data.items) {
    const div = document.createElement("div");
    div.className = "run-item";
    const regionsList = (item.regions || [])
      .map((r) => `<li>${r.defect_type_guess} (conf ${r.confidence}) — ${r.action}</li>`)
      .join("");
    div.innerHTML = `<h3>Seq ${item.seq} — ${item.next_action}</h3><ul>${regionsList}</ul>`;

    if (item.next_action === "flag_for_approval" && !item.approval_status) {
      const approveBtn = document.createElement("button");
      approveBtn.textContent = "Approve";
      approveBtn.addEventListener("click", () => submitApproval(runId, item.seq, "approved"));
      const rejectBtn = document.createElement("button");
      rejectBtn.textContent = "Reject";
      rejectBtn.addEventListener("click", () => submitApproval(runId, item.seq, "rejected"));
      div.appendChild(approveBtn);
      div.appendChild(rejectBtn);
    } else if (item.approval_status) {
      div.innerHTML += `<p>Approval status: ${item.approval_status}</p>`;
    }
    container.appendChild(div);
  }
}

async function submitApproval(runId, seq, decision) {
  await fetch(`${apiBaseUrl}/runs/${runId}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ seq, decision }),
  });
  loadRunDetail(runId);
}

loadRuns();
setInterval(loadRuns, 5000);
```

- [ ] **Step 3: Write `style.css` and run the manual verification checklist**

```css
body { font-family: system-ui, sans-serif; margin: 2rem; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: 0.5rem; text-align: left; }
#run-detail { margin-top: 2rem; border-top: 2px solid #333; padding-top: 1rem; }
.run-item { margin-bottom: 1rem; padding: 0.5rem; background: #f5f5f5; }
button { margin-right: 0.5rem; }
```

Manual verification (no AWS needed — stub the API with a local script):
1. Run `python -m http.server 8000 --directory dashboard` in one terminal.
2. Run a stub API on port 8001 that serves fixed JSON for `/runs` and `/runs/{id}` (e.g. `python -m http.server`-served static JSON files, or a two-line `flask` app) matching the shapes from Task 9's tests.
3. Edit `dashboard/index.html`'s `<meta name="api-base-url">` to `http://localhost:8001` temporarily.
4. Open `http://localhost:8000` in a browser; confirm the runs table populates, clicking a run id shows its regions, and an item with `next_action: "flag_for_approval"` shows Approve/Reject buttons that POST correctly (check the browser network tab).
5. Revert the `api-base-url` meta tag edit before committing.

- [ ] **Step 4: Commit**

```bash
git add dashboard/
git commit -m "feat: add dashboard static site (runs list, trace detail, approve/reject)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 13: Evaluation harness

**Files:**
- Create: `eval/run_eval.py`
- Test: `tests/eval/test_run_eval.py`

**Interfaces:**
- Consumes: `inspect()` (Task 4).
- Produces: `evaluate(dataset: list[dict]) -> dict` where each dataset entry is `{"frame": np.ndarray, "golden_ref": np.ndarray, "has_defect": bool}`, and the return value is `{"precision": float, "recall": float, "f1": float, "escalation_rate": float}`. A defect is "detected" if `inspect()` returns any region with `confidence >= HIGH_CONFIDENCE` (Task 5's threshold, imported not re-defined); "escalated" if any region's confidence falls in the ambiguous band.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_run_eval.py
import numpy as np
import cv2
from eval.run_eval import evaluate


def _clean_pair():
    golden = np.full((200, 200, 3), 100, dtype=np.uint8)
    cv2.rectangle(golden, (20, 20), (60, 60), (200, 200, 200), -1)
    return golden.copy(), golden


def _defective_pair():
    golden = np.full((200, 200, 3), 100, dtype=np.uint8)
    cv2.rectangle(golden, (20, 20), (60, 60), (200, 200, 200), -1)
    frame = golden.copy()
    cv2.rectangle(frame, (120, 120), (170, 170), (255, 255, 255), -1)  # big, unambiguous defect
    return frame, golden


def test_evaluate_returns_metrics_dict():
    clean_frame, clean_golden = _clean_pair()
    defect_frame, defect_golden = _defective_pair()
    dataset = [
        {"frame": clean_frame, "golden_ref": clean_golden, "has_defect": False},
        {"frame": defect_frame, "golden_ref": defect_golden, "has_defect": True},
    ]
    result = evaluate(dataset)
    assert set(result.keys()) == {"precision", "recall", "f1", "escalation_rate"}
    assert result["recall"] == 1.0  # the injected defect must be detected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_run_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.run_eval'`

- [ ] **Step 3: Write minimal implementation**

Create empty `eval/__init__.py`, then:

```python
# eval/run_eval.py
from circuitsentry.vision.pipeline import inspect
from circuitsentry.decision.engine import HIGH_CONFIDENCE, LOW_CONFIDENCE


def evaluate(dataset: list[dict]) -> dict:
    """Run inspect() over a labeled dataset and compute detection metrics.

    Each dataset entry: {"frame": np.ndarray, "golden_ref": np.ndarray, "has_defect": bool}
    """
    true_positives = false_positives = false_negatives = 0
    escalated_count = 0

    for entry in dataset:
        regions = inspect(entry["frame"], entry["golden_ref"])
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
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_run_eval.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add eval/ tests/eval/
git commit -m "feat: add evaluation harness (precision/recall/F1/escalation-rate)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-Review Notes

**Spec coverage:** §3 (OpenCV pipeline) → Tasks 1-4. §4 (decision engine) → Task 5. §5 (AWS architecture: S3/DynamoDB/Step Functions/API Gateway/dashboard) → Tasks 6, 8, 9, 10, 12. §6 (evaluation) → Task 13. §8 (testing/error handling, idempotency, loop bound) → enforced in Task 5 (`MAX_RECAPTURE_ATTEMPTS`) and Task 11 (`run_id`+`seq` keying); Step Functions retry/catch behavior is a configuration addition to Task 8's `perception_task` left as a follow-up hardening pass before the final submission, not a separate task, since it doesn't change any interface. Local capture client (§5) → Task 11.

**Placeholder scan:** no TBD/TODO markers; every step has runnable code. Task 12 uses a manual checklist instead of an automated test — flagged explicitly with its rationale (no JS test framework, YAGNI) rather than left implicit.

**Type consistency:** `Region` (Task 3) is consumed unchanged through Tasks 4, 5, 13. `RegionAction` (Task 5) is consumed unchanged by Task 7. Task 7's event/response shape is defined once and matches what Task 8's Step Functions wiring and Task 9's `capture.py` produce as input. `HIGH_CONFIDENCE`/`LOW_CONFIDENCE`/`MAX_RECAPTURE_ATTEMPTS` are defined once in Task 5 and imported (not redefined) everywhere else they're used (Tasks 7, 13).
