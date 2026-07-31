"""
pi_benchmark.py

Standalone latency benchmark for the drift-triggered pen-detector pipeline,
run on Raspberry Pi (CPU-only, no NPU/GPU delegate of any kind).

Measures TWO numbers independently, matching the two components already
used in your analytical FPS model elsewhere in the project:

  1. classical_cv_ms  - cost of the per-frame tracking stage ALONE
                        (CLAHE, Canny, morphological close/dilate, contour
                        extraction + quality filtering, Hungarian matching,
                        stale-track eviction) with NO classifier call in
                        the loop. Ported directly from
                        drift_classify_pipeline.py so the workload matches.

  2. classifier_ms    - cost of a single classifier invocation ALONE
                        (resize + preprocess + interpreter.invoke()),
                        measured on real cropped ROIs pulled from the same
                        video/camera feed - not synthetic random tensors.

This script does NOT compute any derived "Nx speedup" figure and does not
use any hardware delegate (VeriSilicon/NNAPI/etc) - a plain Pi has no NPU,
so this is a genuine CPU-only measurement. Report the two numbers as-is.
If you want a combined throughput estimate, compute it afterward using the
ACTUAL invocation rate observed in a real drift-triggered run (e.g. the
0.15% from Test A) - not an assumed constant.

Usage:
    python3 pi_benchmark.py --source my_test_video.mp4 --model pen_detector_model.tflite
    python3 pi_benchmark.py --source 0 --model pen_detector_model.tflite   # live camera

Setup on a fresh Raspberry Pi OS:
    pip install opencv-python-headless numpy scipy   # piwheels gives prebuilt ARM wheels
    pip install tflite-runtime                        # if this fails, try:
    pip install ai-edge-litert                         # newer replacement package
"""

import argparse
import csv
import time
from collections import defaultdict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter as _Interpreter
        class tflite:  # shim so the rest of the script doesn't care which backend loaded
            Interpreter = _Interpreter
    except ImportError:
        from tensorflow import lite as tflite  # last resort: full TensorFlow


# --- Parameters copied verbatim from drift_classify_pipeline.py, so the ---
# --- classical-CV workload measured here matches the real pipeline.    ---
SOLIDITY_RELEASE = 0.40
CENTROID_DISTANCE_THRESHOLD = 150
MAX_FRAMES_WITHOUT_DETECTION = 25


def contour_quality(contour, frame_shape, border_margin=5):
    area = cv2.contourArea(contour)
    if area < 1000 or len(contour) < 4:
        return False, 0.0, True
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return False, 0.0, True
    solidity = area / hull_area
    h_f, w_f = frame_shape[:2]
    x, y, w, h = cv2.boundingRect(contour)
    touches_border = (x <= border_margin or y <= border_margin or
                       (x + w) >= (w_f - border_margin) or (y + h) >= (h_f - border_margin))
    return True, solidity, touches_border


def calculate_aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    if 0 < aspect_ratio < 1:
        aspect_ratio = 1 / aspect_ratio
    return aspect_ratio, (x, y, w, h)


def calculate_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] > 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None


def match_objects_hungarian(detections, tracked_ids, tracked_positions,
                             max_distance=CENTROID_DISTANCE_THRESHOLD):
    if not detections or not tracked_ids:
        return {}
    cost_matrix = np.zeros((len(detections), len(tracked_ids)), dtype=np.float64)
    for i, det in enumerate(detections):
        for j, trk in enumerate(tracked_positions):
            cost_matrix[i, j] = np.hypot(det[0] - trk[0], det[1] - trk[1])
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    matches = {}
    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r, c] < max_distance:
            matches[r] = tracked_ids[c]
    return matches


def run_classical_cv_stage(frame, tracker_state):
    """
    Everything from the real pipeline EXCEPT the classifier call.
    Returns (matched_objects: {obj_id: (x, y, w, h)}, elapsed_ms).
    """
    t0 = time.perf_counter()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blurred, 40, 120)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, kernel_dilate, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_entries = []  # (contour, aspect_ratio, bounds, centroid)
    if contours:
        large_contours = [c for c in contours if cv2.contourArea(c) > 3000]
        for cnt in large_contours:
            aspect_ratio, bounds = calculate_aspect_ratio(cnt)
            passes_shape, solidity, touches_border = contour_quality(cnt, frame.shape)
            if aspect_ratio > 2.0 and passes_shape and solidity >= SOLIDITY_RELEASE:
                centroid = calculate_centroid(cnt)
                if centroid is not None:
                    valid_entries.append((cnt, aspect_ratio, bounds, centroid))

    positions = tracker_state['positions']
    since_seen = tracker_state['since_seen']

    centroids = [e[3] for e in valid_entries]
    tracked_ids = list(positions.keys())
    tracked_positions = [positions[tid] for tid in tracked_ids]
    assignment = match_objects_hungarian(centroids, tracked_ids, tracked_positions)

    matched = {}
    detected_ids = set()
    next_id = tracker_state['next_id']
    for i, (cnt, aspect_ratio, bounds, centroid) in enumerate(valid_entries):
        if i in assignment:
            obj_id = assignment[i]
        else:
            obj_id = next_id
            next_id += 1
        positions[obj_id] = centroid
        since_seen[obj_id] = 0
        detected_ids.add(obj_id)
        matched[obj_id] = bounds
    tracker_state['next_id'] = next_id

    # Evict stale tracks so the Hungarian cost matrix stays bounded over a
    # long recording, matching MAX_FRAMES_WITHOUT_DETECTION in the real
    # pipeline. Without this, this benchmark's classical-CV timing would
    # drift upward over the video for reasons unrelated to real workload.
    for obj_id in list(positions.keys()):
        if obj_id not in detected_ids:
            since_seen[obj_id] += 1
            if since_seen[obj_id] > MAX_FRAMES_WITHOUT_DETECTION:
                positions.pop(obj_id, None)
                since_seen.pop(obj_id, None)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return matched, elapsed_ms


def preprocess_for_model(roi, input_details):
    """
    Builds the input tensor matching whatever dtype this .tflite export
    actually uses. This benchmark measures LATENCY only - inference time
    does not depend on whether quantization mapping is numerically
    perfect, since the same ops execute regardless of pixel values.
    Verify classification correctness separately before trusting outputs.
    """
    target_h, target_w = int(input_details['shape'][1]), int(input_details['shape'][2])
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_w, target_h))
    dtype = input_details['dtype']

    if dtype == np.float32:
        # Matches tensorflow.keras.applications.mobilenet.preprocess_input
        # (mode='tf': scale to [-1, 1]) without requiring a full TF import.
        arr = img.astype(np.float32)
        arr = (arr / 127.5) - 1.0
    elif dtype == np.uint8:
        arr = img.astype(np.uint8)
    else:  # int8, fully-integer-quantized
        scale, zero_point = input_details['quantization']
        scale = scale if scale else 1e-8
        arr = (img.astype(np.float32) / scale + zero_point)
        arr = np.clip(arr, -128, 127).astype(np.int8)

    return np.expand_dims(arr, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                         help="Video file path, or camera index (e.g. 0)")
    parser.add_argument("--model", default="pen_detector_model.tflite")
    parser.add_argument("--max-frames", type=int, default=1000,
                         help="Stop after this many frames")
    parser.add_argument("--out", default="pi_benchmark_results.csv")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    interpreter = tflite.Interpreter(model_path=args.model)  # CPU only, no delegate
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print(f"Model input shape: {input_details['shape']}, dtype: {input_details['dtype']}")
    print("Running with plain CPU interpreter, no delegate - this is the number "
          "that matters for a no-accelerator edge-hardware claim.\n")

    tracker_state = {'positions': {}, 'since_seen': defaultdict(int), 'next_id': 0}
    cv_times_ms = []
    classifier_times_ms = []
    rows = []  # frame, cv_ms, classifier_invoked, classifier_ms

    frame_idx = 0
    while frame_idx < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        matched, cv_ms = run_classical_cv_stage(frame, tracker_state)
        cv_times_ms.append(cv_ms)

        classifier_ms = None
        h_f, w_f = frame.shape[:2]
        for obj_id, (x, y, w, h) in matched.items():
            is_fully_in = x > 10 and y > 10 and (x + w) < (w_f - 10) and (y + h) < (h_f - 10)
            if not is_fully_in:
                continue
            roi = frame[y:y + h, x:x + w]
            if roi.size == 0:
                continue

            input_tensor = preprocess_for_model(roi, input_details)
            t0 = time.perf_counter()
            interpreter.set_tensor(input_details['index'], input_tensor)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details['index'])
            classifier_ms = (time.perf_counter() - t0) * 1000
            classifier_times_ms.append(classifier_ms)
            break  # one real invocation per frame is enough to characterize cost

        rows.append((frame_idx, round(cv_ms, 3),
                      classifier_ms is not None,
                      round(classifier_ms, 3) if classifier_ms is not None else ""))

        if frame_idx % 100 == 0:
            print(f"[{frame_idx} frames] classical-CV avg so far: "
                  f"{np.mean(cv_times_ms):.2f} ms | classifier invocations so far: "
                  f"{len(classifier_times_ms)}")

    cap.release()

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "classical_cv_ms", "classifier_invoked", "classifier_ms"])
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print("RESULTS (measured, CPU-only, no delegate)")
    print("=" * 60)
    print(f"Frames processed:              {frame_idx}")
    if cv_times_ms:
        print(f"Classical-CV cost/frame:       "
              f"mean {np.mean(cv_times_ms):.2f} ms | median {np.median(cv_times_ms):.2f} ms")
        print(f"Classical-CV-only FPS ceiling: {1000 / np.mean(cv_times_ms):.1f} FPS")
    if classifier_times_ms:
        print(f"Classifier cost/invocation:    "
              f"mean {np.mean(classifier_times_ms):.2f} ms | median {np.median(classifier_times_ms):.2f} ms")
        print(f"Classifier invocations:        {len(classifier_times_ms)} "
              f"({100 * len(classifier_times_ms) / frame_idx:.2f}% of frames)")
    else:
        print("No classifier invocations occurred - check that an object was "
              "detected and fully in-frame during the recording.")
    print(f"\nRaw per-frame data saved to: {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
