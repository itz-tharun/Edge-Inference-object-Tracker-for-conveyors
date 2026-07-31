# Drift-Triggered Classification Gating for Edge Video Pipelines

Code and results for a CVIP 2026 submission on reducing CNN classifier
invocations in a controlled single-object-class edge vision pipeline.

## What this is

A gating architecture that sits between a classical-CV tracker and a CNN
classifier, and decides **when** the classifier actually needs to run.
Classical geometric contour tracking (Canny edges, Hungarian assignment,
boundary-closure hysteresis) does the cheap per-frame work; a lightweight,
entirely classical appearance/motion "suspicion score" (color-histogram
drift + constant-velocity motion-prediction residual — no learned model
anywhere in the gate) decides when the CNN classifier must be invoked.

Compared against two baselines — **always-classify** (run the CNN every
frame) and **fixed-interval** (run it every N frames) — across four
controlled test scenarios (static, moving, crossing/occlusion,
multi-object) plus real hardware validation on a Raspberry Pi 5.

This is **not** a new classifier or tracking algorithm. The classifier is
an off-the-shelf MobileNetV2. The contribution is the gating decision layer.

## Repo structure

```
src/
  pipelines/           the three invocation strategies (current, reported version)
    always_classify_pipeline.py
    fixed_interval_pipeline.py
    drift_classify_pipeline.py
  model_conversion/
    convert_h5_to_float_tflite.py   .h5 -> float32 .tflite (see note below)
  benchmarking/
    pi_benchmark.py      two-stage latency benchmark, run on Raspberry Pi 5
  utils/
    laptop_preview_check.py         visual sanity check, .h5 model, on laptop
    laptop_preview_check_tflite.py  same, but for the .tflite file
  test_runner/
    run_all_tests.py     runs all scenario x pipeline combinations end-to-end

data/
  ground_truth/         hand-marked event frames per scenario x strategy
  suspicion_logs/        per-frame suspicion score logs, one folder per strategy
  synchronized_results/  frame-aligned tri-system comparison (cropped vs uncropped)
  pi_benchmark/          raw per-frame / per-call latency samples from the Pi 5
  summary/                aggregated accuracy/invocation-rate tables

results/
  logs/       raw console-output tables backing the reported numbers
  figures/    generated charts/tables (see note on cropped_vs_uncropped figure below)

archive/      superseded scripts, kept only for provenance (see below) — not
              used to produce any reported result
```

## Results summary

See `data/summary/FINAL_MASTER_SUMMARY.csv` and `ALL_RESULTS_SUMMARY.csv`
for the aggregated numbers. Per-scenario invocation counts:

| Scenario | Always-classify | Fixed-interval | Drift-triggered |
|---|---|---|---|
| Static | 286/1325 (21.6%) | 44/1375 (3.2%) | 2/1309 (0.15%) |
| Moving | 1057/1057 (100.0%) | 30/880 (3.41%) | 1/1154 (0.09%) |
| Crossing/occlusion | 1316/1425 (92.35%) | 65/1679 (3.87%) | 15/1718 (0.87%) |
| Multi-object | 2715/1361 frames (199.48%) | 85/1357 (6.26%) | 7/1391 (0.50%) |

Classifier validation (Test E, confusion matrix): Accuracy 71.4%, Precision
0.667, Recall 1.000, F1 0.800 — perfect recall, but any elongated non-pen
object (compass, glue applicator) is misclassified as PEN 100% of the time.
This is a classifier shape-bias finding, orthogonal to the gating
architecture (see paper §Findings, F8).

Raspberry Pi 5 (CPU-only, float32 `.tflite`, no throttling — confirmed via
`vcgencmd get_throttled` = `0x0`): derived end-to-end throughput
**~113 FPS** drift-triggered vs. **~33 FPS** always-classify vs. **~115
FPS** tracking-only ceiling. Derivation formula and raw latencies in
`data/pi_benchmark/`.

## Reproducing results

1. Install dependencies: `pip install -r requirements.txt`
2. Place your recorded video clips and `pen_detector_model.h5` /
   `pen_detector_model_float32.tflite` alongside the scripts (paths are
   configured at the top of each script — edit `BASE_FOLDER` etc. to match
   your layout).
3. Run any single strategy directly, e.g. `python src/pipelines/drift_classify_pipeline.py`,
   or run all scenario x strategy combinations via `src/test_runner/run_all_tests.py`.
4. Convert a trained `.h5` model to float32 TFLite with
   `src/model_conversion/convert_h5_to_float_tflite.py` before deploying to
   the Pi (see note on quantization below).
5. Benchmark on real Raspberry Pi hardware with `src/benchmarking/pi_benchmark.py`.

## Known, resolved issues (kept documented, not hidden)

- **Quantization bug (resolved):** an earlier int8-quantized `.tflite`
  conversion had a calibration/domain mismatch (`pixel/255` scaling implied
  by the quantization params vs. `mobilenet.preprocess_input`'s -1..1
  scaling actually used in training), causing systematic misclassification.
  Fixed by reconverting to plain float32 (no quantization), verified to
  exactly match the `.h5` model's output. All reported results use the
  float32 file.
- **Aspect-ratio bug (resolved):** elongation was originally measured with
  axis-aligned `cv2.boundingRect`, which under-measures a diagonally-rotated
  pen and can fail the `aspect_ratio > 2.0` filter entirely. Fixed with
  rotation-aware `cv2.minAreaRect`. `src/test_runner/run_all_tests.py` is
  the fixed version; the pre-fix version is kept in `archive/` for
  provenance only and was **not** used to produce any reported number.

## Archive folder

`archive/` contains scripts that are **not** part of the reported pipeline:
a pre-bugfix version of the test runner (kept so the fix is auditable) and
a one-off camera/device test utility used during development. Nothing in
`archive/` was used to generate any number in the paper.

## Flag for you to check before making this repo public

`results/figures/cropped_vs_uncropped_full_table.png` exists in the source
files, but the cropped-vs-uncropped training ablation (Test 2) is marked as
**not yet run** in the current project notes. Confirm whether this figure
is a stale/placeholder file from earlier planning or an actual completed
result before citing it anywhere — don't let it imply Test 2 is done if
it isn't yet.
