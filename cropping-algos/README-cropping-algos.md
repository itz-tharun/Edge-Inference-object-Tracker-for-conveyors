# Cropping Algorithms — Training Data Preparation for the Pen Classifier

## Purpose

SENTRY's Stage 2 classifier (MobileNetV2) needs a labeled set of pen /
non-pen crops to train on. A crop is only as useful as the segmentation
that produced it: a bounding box that includes half the desk behind the
pen, or one that clips the cap off, teaches the network the wrong thing.
This folder is the record of getting that segmentation right — eight
scripts, run roughly in the order listed below, each one responding to a
specific failure of the one before it.

None of these scripts are part of the runtime pipeline described in the
paper. They are offline, one-time data-preparation tools: point them at a
folder of raw photos, get back cropped, resized (224×224) training images.

## Why this problem is harder than "find the pen"

A bounding-box crop only has to be roughly right. A *training* crop has to
be right in a way that doesn't leak information the deployed system won't
have. Two separate difficulties showed up here:

1. **Segmenting the pen from an arbitrary background** without a trained
   model, matching the classical, learning-free philosophy of the rest of
   the project. Backgrounds varied across the source photos: off-white
   paper, wood desks, cluttered surfaces — none of which share a single
   threshold or color range.
2. **Deciding whether to keep the background at all.** Stripping it
   produces a clean silhouette, which looked like the "correct" thing to
   do. It later turned out to be the wrong call — see
   [Downstream consequence](#downstream-consequence-this-decision-had) below.

## Iteration history

| Script | Approach | Outcome |
|---|---|---|
| `canny-edge-it-1.py` | HSV color mask (tuned for blue pens) fused with Canny edges via `bitwise_or`, dilated and closed, background zeroed out per-contour | Worked only for blue pens on a plain background; the color prior doesn't generalize to other pen colors, and background pixels are set to black inside the crop |
| `canny-edge-it-2.py` | Drops the color assumption. Adaptive Gaussian thresholding (block size 31, C=4) instead of a fixed threshold, plus a dedicated "bridge" dilation step to reconnect a pen cap that Canny had split from the body | More background-agnostic, but the script has a stray typo (`if _name_ == "_main_":` — missing the double underscores `__name__`/`__main__`) that silently prevents the guarded call from running; kept as-is for provenance rather than corrected |
| `final-cropping-algo.py` | Same adaptive-threshold + bridging detection as `it-2`, but instead of zeroing the background it writes a 4-channel BGRA PNG with true alpha transparency, so the classifier's dataloader — not this script — decides how to fill the background | The transparency approach carried forward into the two training conditions discussed below |
| `off-whitebackground-it-1.py` | Otsu's automatic thresholding on a Gaussian-blurred grayscale image, targeted at the specific case of an off-white/paper background | Simpler and more robust for that one background type; picks the single largest contour rather than filtering by an area threshold |
| `off-whitebackgroundcrop-it-2.py` | Identical detection pipeline to `it-1`, but explicit about *why* it takes only the largest contour: earlier attempts at this background type occasionally left a "ghost" outline of a second, smaller object visible in the supposedly-transparent background | Same result, documented reasoning — the max-area constraint is a fix, not a stylistic choice |
| `otsu-threshold-crop.py` | Effectively the same Otsu pipeline as `off-whitebackground-it-1.py`, pointed at a different local input/output path | Functionally a duplicate; kept because it was the version actually run against a specific dataset folder during collection |
| `u2net-it-1.py` | Replaces the classical pipeline entirely with `rembg`'s `u2netp` background-removal model, downscaling to 640px before inference, trimming the result to its non-transparent bounding box | First attempt at a learned segmentation stage; correct on more background types than any classical script, at the cost of introducing a (small) neural model into what is otherwise a data-prep step, not the runtime pipeline |
| `u2net-it-2(faster).py` | Same `u2netp` model, but images are downscaled to 320×320 (the model's native input size, so nothing is wasted) and processed concurrently with a `ThreadPoolExecutor`; reports average per-image latency and derived FPS | Meant to make a batch of several hundred photos practical to process in one run rather than to run in real time |

## Downstream consequence this decision had

The paper's Section 5.10 classifier-training ablation traces directly back
to this folder. Training crops produced with the background stripped out
(`final-cropping-algo.py` and the `u2net` scripts, mode) leave silhouette
as close to the only signal the classifier can learn from. A second model,
trained on uncropped images with the background retained, reached
77.46% accuracy versus 59.78% for the background-removed version, and
correctly rejected 84.4% of crops of a drawing compass — a background-removed
model misclassified all of them as a pen, because outline alone cannot
distinguish a compass from a pen. The cropping choice made in this folder,
in other words, is not a cosmetic preprocessing detail; it measurably set
the ceiling on what the classifier downstream could learn.

## What's missing / not tracked here

- The raw input photo sets (`inputpens`, `pen1`, `newshi`, etc.) are
  referenced by absolute Windows paths in every script and are not part of
  this repository. Anyone re-running these scripts needs to supply their
  own source images and edit `input_dir` / `output_dir` at the top of each
  file.
- There's no single "run everything" entry point — each script was run
  manually against whatever folder was being processed at the time. If
  this pipeline needs to be reproducible end-to-end, the next step is
  consolidating the classical (non-U2Net) path into one parameterized
  script with the final adaptive-threshold + alpha-transparency logic from
  `final-cropping-algo.py`, since that combination is the one whose output
  (background-retained training set) is the one supported by the paper's
  results.

## Dependencies

`opencv-python`, `numpy`. The two `u2net-it-*` scripts additionally need
`rembg` and `Pillow` (`u2net-it-1.py` only) — these are not listed in the
project's top-level `requirements.txt`, since they were only ever used for
this offline data-prep step.
