# CircuitSentry — Agentic PCB Visual Inspection

**Competition**: OpenCV × AWS "OpenCVComp26" — Agentic Vision featured path
**Team**: solo
**Date**: 2026-09-05

## 1. Problem statement & impact

Manual visual inspection of PCBs (missing components, solder bridges,
tombstoning, scratches/burns) is slow, inconsistent, and doesn't scale for
small manufacturers or hobbyist/prototype runs who can't afford dedicated AOI
(automated optical inspection) hardware. CircuitSentry is a low-cost, laptop
webcam + AWS cloud system that performs the same function: an agent that
looks at a board, decides — based on what it actually sees — whether to look
closer, pass, or escalate to a human, rather than a one-shot classifier.

**Target users**: hobbyist electronics makers, small-batch PCB assemblers,
electronics education labs — anyone without access to commercial AOI who
wants an automated first-pass inspection with a human-in-the-loop safety net.

## 2. Why this is "Agentic Vision" (not just a vision demo)

The OpenCV output does not just get displayed — it changes what the system
does next:
- Ambiguous-confidence regions trigger a **new capture request** (tighter
  ROI/zoom) rather than a final verdict — the vision result changes the next
  tool call.
- High-confidence defects trigger a **human-approval request**, not an
  automatic verdict — the vision result changes what a human is asked to do.
- Only low-confidence-clean regions terminate the loop with "pass."

This perception → decision → action loop, including its re-capture branch,
is modeled explicitly as an AWS Step Functions state machine, which doubles
as the required agent workflow diagram.

## 3. Planned OpenCV 5 image/video analysis

Per-frame pipeline (classical CV, fully explainable, no training data
needed):
1. **Alignment**: ORB/AKAZE feature matching + homography of the incoming
   frame against a golden-reference image of the same board design.
2. **Diff**: absolute difference of aligned grayscale images, adaptive
   threshold, morphological open/close to suppress noise.
3. **Candidate localization**: contour extraction on the cleaned diff mask.
4. **Scoring**: per-contour confidence from area, diff intensity, and shape
   heuristics, with a coarse defect-type guess (missing component / solder
   bridge / tombstoning / scratch-burn).
5. Output: `[{bbox, defect_type_guess, confidence}, ...]` per frame.

## 4. Decision engine (rule-based)

- `confidence ≥ HIGH` → **flag**: write finding to DynamoDB, dashboard marks
  it "pending human approval."
- `LOW ≤ confidence < HIGH` → **request re-capture** of that bbox's region at
  tighter FOV/adjusted exposure; loop back to step 1 on the new crop. Bounded
  to at most 2 loop iterations per region to cap latency/cost — a region
  still ambiguous after 2 tries falls through to "flag for human" rather than
  looping forever.
- `confidence < LOW` → **pass**.

## 5. AWS architecture

- **Local capture client** (Python + OpenCV `VideoCapture` on the laptop):
  captures webcam frames, uploads via S3 presigned URL, and polls for/
  fulfills "re-capture this ROI" commands from the backend.
- **API Gateway → Lambda** (`POST /capture`): starts a Step Functions
  execution per inspection request.
- **Step Functions state machine**: `Perception` (Lambda, OpenCV) →
  `Decision` (Choice state) → `Pass` | `RequestRecapture` (loops back to
  `Perception` on the new image) | `FlagForApproval`.
- **DynamoDB**: one item per inspection run — regions, confidence, defect
  type guess, verdict, action taken, human-approval status/timestamp. This
  is both the audit trail and the evaluation dataset.
- **S3**: raw + cropped inspection images, keyed by `run_id`/sequence.
- **Dashboard** (S3 + CloudFront static site, backed by the same API
  Gateway): live run list; per-run trace view (each frame with boxes drawn,
  the decision made, and — if looped — the follow-up crop and its result);
  approve/reject buttons for pending flags.

Architecture diagram: to be produced separately (draw.io / AWS icons) once
implementation starts; this doc describes the components and data flow it
must show.

## 6. Evaluation & judge demonstration

- **Test set**: ~30–50 images from a public PCB-defect dataset (DeepPCB-style:
  missing component, solder bridge, tombstoning, scratch/burn) plus a
  handful of own photographed boards with deliberately induced defects, used
  for the live demo.
- **Metrics**: precision / recall / F1 on defect detection; **escalation
  rate** (% of inspections triggering the re-capture loop) as a direct,
  reportable measure of agentic behavior; human-approval agreement rate
  (does the human agree with the flagged verdict); latency and estimated
  cost per inspection.
- **Judge demonstration**: live webcam inspection via screen-share, or
  upload-and-replay through the dashboard. Either way the judge watches the
  full trace: perception (boxes drawn) → decision → optional re-capture loop
  → final verdict → human-approval click.
- **Documented failure cases**: false positives from lighting/shadow
  changes; false negatives on subtle defects; misalignment when board pose
  diverges too far from the golden reference. Each becomes a labeled example
  in the technical report, with the corresponding image pair.

## 7. Focus path

**Agentic Vision** only. Not pursuing the Best Use of COOL award — no
Graviton/COOL benchmarking commitment. (Lambda functions may incidentally
run on arm64/Graviton2 at no extra dev cost, but this is not a claimed COOL
submission.)

## 8. Testing & error handling

- Unit tests (pytest) for the alignment/diff/scoring pipeline against saved
  sample image pairs — iterable without any AWS dependency.
- Step Functions: retries on transient Lambda failures; a `Catch` on
  repeated failure routes to a "manual review" terminal state so a board is
  never silently dropped from the pipeline.
- Idempotent uploads keyed by `run_id` + sequence number, so a retried
  capture can't double-process.
- Loop bound (≤2 re-capture iterations per region) prevents infinite
  escalation; falls through to human flag instead.

## 9. Open items for the proposal (due Aug 13, 2026)

- Team name and team bio (solo — needs a short bio of relevant experience /
  past hackathons).
- Final architecture diagram (visual, from §5).
- Whether to request the $150 cloud compute grant (recommended: yes — this
  design fits comfortably within it).

## 10. Explicitly out of scope (YAGNI)

- Training a custom defect-detection model — classical CV is sufficient,
  explainable, and needs no labeled training set.
- COOL/Graviton benchmarking — not pursuing that award.
- Physical camera automation (motorized zoom/gimbal) — "re-capture" is
  fulfilled by the human/laptop repositioning or a fresh higher-res crop,
  not robotic actuation.
