# Live-Footage Test Code — Building Stage 1 Against a Real Camera

## Purpose

Before any invocation-gating strategy could be evaluated, the tracker and
classifier had to work together against a live webcam feed at all:
correctly finding an object, holding a stable identity for it, and
returning a label that wasn't noise. This folder is that development
process, kept in the order it happened. Every file is a real,
independently-run iteration, not a cleaned-up demo — the filenames were
the engineer's own notes on what was broken at the time.

The final evaluation reported in the paper does **not** run against a live
feed at all; it runs against pre-recorded video, in the sibling folder
`program-files-and-results/`. The reasoning for that switch (reproducibility
— identical frames for every strategy under comparison, not a fresh camera
draw each time) is discussed in the paper's Section 6. This folder is the
live-feed work that preceded and motivated that decision, and it is where
the multi-object contour tracker later reused in the offline pipelines was
actually built and debugged.

## Iteration history

| Script | What changed | Result |
|---|---|---|
| `testingalgo-1(doesnt identify and put bounding box)).py` | Baseline: Canny edges → dilate/erode → contour → single classifier call when an object is fully inside frame | Detection was unreliable enough that a bounding box frequently failed to appear at all; establishes the problem the next several iterations attack |
| `testingalgo-2(not accurate identifies everything as pen).py` | Adds CLAHE contrast boost on the L-channel and fuses Canny with Otsu thresholding (`bitwise_or`) to catch lighter-colored, non-black pens | Detection improved, but the classifier output was effectively constant — everything was labeled "pen" regardless of what was in view |
| `testingalgo-3(classifies everything as pen part 2).py` | Restructures the classification result into a shared dict (label + color) instead of a single string, cleans up the display logic | Same underlying classification bug persisted; ruled out that the *display* code was the problem, narrowing the search to the model's output itself |
| `testingalgo-4(threshold change multiple obj).py` | Swaps the detection stage to CLAHE-on-grayscale + plain Canny (drops the Otsu fusion), and adds an explicit debug print of the raw predicted index and confidence to the terminal | Existing to *diagnose* the label bug directly rather than guess at it — the print statement is the actual point of this file |
| `testingalgo-5(works with single obj inaccurate with mult obj).py` | Based on what the debug prints in `testingalgo-4` revealed, flips the decision rule: confidence **below** 0.55 is read as PEN rather than confidence above a threshold reading as PEN | Fixed the label-flip bug for a single tracked object; still could not handle more than one object in frame at once, which is what `FINALTRACKING.py` was built to solve |
| `FINALTRACKING.py` | Full rewrite: adds Hungarian-algorithm centroid matching to give each detected object a persistent ID across frames, a `is_closed_boundary()` solidity + border check so partial/broken contours aren't tracked, per-object boundary memory so a briefly-occluded object keeps its ID, and one classifier thread per object (guarded by a per-object lock) instead of one global "is processing" flag | The first version that tracked and classified **multiple simultaneous objects** correctly; this is architecturally the direct ancestor of Stage 1 in `program-files-and-results/src/pipelines/` |

## What the label-flip bug actually was

Iterations 2–4 all reported the same symptom: every object in frame,
regardless of appearance, was labeled the same way. The root cause,
diagnosed via the debug prints added in `testingalgo-4`, was a mismatch
between the order of the `LABELS` list in the test scripts and the actual
output convention of the trained model (a single sigmoid unit, where a
*low* value indicates "pen" — see `FINALTRACKING.py`'s inverted comparison,
`if conf < 0.5`). `testingalgo-5` patched this with a hardcoded threshold
(0.55) as a quick fix; `FINALTRACKING.py` and every pipeline in
`program-files-and-results/` use the corrected `conf < 0.5` convention
directly instead of a tuned workaround.

## Relationship to the reported pipeline

`FINALTRACKING.py` is Stage 1 only — detection, tracking, and an
always-invoke classifier call for every fully-visible object. It does
**not** contain the drift-triggered suspicion score, the fixed-interval
baseline, or any of the invocation-rate logging used to produce the
paper's results. Those were built afterward in
`program-files-and-results/src/pipelines/`, reusing this file's tracking
logic (`calculate_centroid`, `is_closed_boundary`,
`match_objects_hungarian`) largely unchanged.

## What's missing / not tracked here

- `MODEL_PATH` in every script is a hardcoded local Windows path to
  `pen_detector_model.h5`. The trained model file itself is not checked
  into this repository.
- These scripts open `cv2.VideoCapture(0)` and require a real webcam and a
  local display (`cv2.imshow`) to run; they cannot be executed headless or
  in CI as-is.
- There's no recorded log or video of these test runs — the "results" of
  this folder are the filenames' own annotations plus the working final
  script, not a reproducible dataset.

## Dependencies

`opencv-python`, `numpy`, `scipy` (for `linear_sum_assignment` in
`FINALTRACKING.py` only), `tensorflow` (Keras model loading and
`mobilenet.preprocess_input`).
