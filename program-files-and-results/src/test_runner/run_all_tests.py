"""
Synchronized Tri-System Test — VS Code Click-to-Run Version
==============================================================
No terminal, no command-line args. Edit the CONFIG section below if your
filenames differ, then just click Run (or press F5) in VS Code.

This runs ALL 8 combinations (4 videos x 2 models) automatically in one
go, and saves every result CSV into your FINALTEST folder.
"""
import os
import csv
import cv2
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

# ============================================================
# CONFIG — edit these if your filenames/paths differ
# ============================================================
BASE_FOLDER = r"C:\Users\dhars\Desktop\FINALTEST"

VIDEOS = {
    # only re-running moving/crossing - static and multi already produced
    # correct results before the aspect-ratio bug was fixed, no need to redo them
    "moving":   os.path.join(BASE_FOLDER, "moving.mp4"),
    "crossing": os.path.join(BASE_FOLDER, "crossing.mp4"),
}

MODELS = {
    "uncropped": os.path.join(BASE_FOLDER, "pen_detector_model_uncropped_best.h5"),
    "cropped":   os.path.join(BASE_FOLDER, "pen_detector_model_cropped_best.h5"),
}

# occluded frame ranges per scenario (only "crossing" needs this) — computed
# from your timestamps (2-3s, 11-13s, 30-31s) at 59.97fps
OCCLUDED_FRAMES = {
    "moving": "",
    "crossing": "120-180,660-780,1799-1859",
}

OUTPUT_FOLDER = BASE_FOLDER  # CSVs get saved here

# ============================================================
# Constants — identical to your original pipelines, do not edit
# ============================================================
CENTROID_DISTANCE_THRESHOLD = 150
SOLIDITY_RELEASE = 0.40
W_APPEARANCE = 0.6
W_TRACKING = 0.4
DRIFT_THRESHOLD = 0.4
HYSTERESIS_FRAMES = 3
FIXED_INTERVAL = 30


def calculate_histogram(roi):
    hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def calculate_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] > 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None


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
    """FIXED: uses cv2.minAreaRect (rotation-aware) instead of cv2.boundingRect
    (axis-aligned) to measure elongation. An axis-aligned box on a diagonally-
    oriented pen (e.g. ~45 degrees) comes out nearly SQUARE even though the
    pen itself is clearly elongated - this caused moving.mp4/crossing.mp4 to
    produce zero valid detections for the entire video. minAreaRect finds the
    true minimum-area rotated rectangle around the object regardless of its
    angle in frame, so a diagonal pen still measures as elongated correctly.
    Bounding box for cropping/is_fully_in checks stays axis-aligned (that's
    fine/expected - we just need a rectangular crop, not a rotated one)."""
    rect = cv2.minAreaRect(contour)
    (rw, rh) = rect[1]
    ar = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 0
    bounds = cv2.boundingRect(contour)  # still axis-aligned, used for cropping only
    return ar, bounds


def match_objects_hungarian(detections, tracked_ids, tracked_positions, max_distance=CENTROID_DISTANCE_THRESHOLD):
    if not detections or not tracked_ids:
        return {}
    cost = np.zeros((len(detections), len(tracked_ids)))
    for i, det in enumerate(detections):
        for j, trk in enumerate(tracked_positions):
            cost[i, j] = np.hypot(det[0] - trk[0], det[1] - trk[1])
    r, c = linear_sum_assignment(cost)
    return {ri: tracked_ids[ci] for ri, ci in zip(r, c) if cost[ri, ci] < max_distance}


def parse_occluded_frames(spec):
    occluded = set()
    if not spec:
        return occluded
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            occluded.update(range(int(a), int(b) + 1))
        elif part:
            occluded.add(int(part))
    return occluded


def load_model_robust(path):
    from tensorflow.keras.models import load_model as _load_model
    try:
        return _load_model(path)
    except Exception as e:
        print(f"[load_model failed: {e}]\nFalling back to rebuild+load_weights...")
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras import layers, models as kmodels
        base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        model = kmodels.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        model.build((None, 224, 224, 3))
        model.load_weights(path)
        return model


def classify(model, roi, preprocess_input):
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    arr = np.expand_dims(img, axis=0).astype("float32")
    arr = preprocess_input(arr)
    conf = model.predict(arr, verbose=0)[0][0]
    return "PEN" if conf < 0.5 else "NOT A PEN"


def run_single_test(video_path, model, preprocess_input, scenario_name, occluded_spec, out_path):
    occluded = parse_occluded_frames(occluded_spec)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[SKIP] Could not open {video_path}")
        return None
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    last_known_position = {}
    last_velocity = {}
    object_id_counter = 0

    displayed_label = {"always": {}, "fixed": {}, "drift": {}}
    ref_hist_drift = {}
    consec_high = defaultdict(int)
    never_classified_drift = set()

    rows = []
    frame_idx = 0
    total_invocations = {"always": 0, "fixed": 0, "drift": 0}

    print(f"[SYSTEM] {scenario_name}: processing {total_video_frames} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % 200 == 0:
            print(f"  [{scenario_name}] frame {frame_idx}/{total_video_frames} "
                  f"({frame_idx/total_video_frames*100:.0f}%) — "
                  f"invocations: always={total_invocations['always']} "
                  f"fixed={total_invocations['fixed']} drift={total_invocations['drift']}")

        h_f, w_f = frame.shape[:2]

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

        valid_entries = []
        for cnt in [c for c in contours if cv2.contourArea(c) > 3000]:
            ar, bounds = calculate_aspect_ratio(cnt)
            passes_shape, solidity, touches_border = contour_quality(cnt, frame.shape)
            if ar > 2.0 and passes_shape and solidity >= SOLIDITY_RELEASE:
                centroid = calculate_centroid(cnt)
                if centroid is not None:
                    valid_entries.append((cnt, ar, bounds, centroid))

        tracked_ids = list(last_known_position.keys())
        tracked_positions = [last_known_position[t] for t in tracked_ids]
        assignment = match_objects_hungarian([e[3] for e in valid_entries], tracked_ids, tracked_positions)

        matched = {}
        for i, (cnt, ar, bounds, centroid) in enumerate(valid_entries):
            if i in assignment:
                oid = assignment[i]
            else:
                oid = object_id_counter
                object_id_counter += 1
            if oid in last_known_position:
                prev = last_known_position[oid]
                last_velocity[oid] = (centroid[0] - prev[0], centroid[1] - prev[1])
            last_known_position[oid] = centroid
            matched[oid] = (cnt, bounds, centroid)

        ground_truth = None if frame_idx in occluded else "PEN"

        for oid, (cnt, (x, y, w, h), centroid) in matched.items():
            is_fully_in = x > 10 and y > 10 and (x + w) < (w_f - 10) and (y + h) < (h_f - 10)
            if not is_fully_in:
                continue
            roi = frame[y:y + h, x:x + w]
            if roi.size == 0:
                continue

            always_lbl = classify(model, roi, preprocess_input)
            total_invocations["always"] += 1
            displayed_label["always"][oid] = always_lbl
            always_invoked = True

            fixed_invoked = (oid not in displayed_label["fixed"]) or (frame_idx % FIXED_INTERVAL == 0)
            if fixed_invoked:
                displayed_label["fixed"][oid] = classify(model, roi, preprocess_input)
                total_invocations["fixed"] += 1

            if oid in ref_hist_drift:
                cur_hist = calculate_histogram(roi)
                sim = max(-1.0, min(1.0, cv2.compareHist(ref_hist_drift[oid], cur_hist, cv2.HISTCMP_CORREL)))
                appearance_drift = 1.0 - sim
            else:
                appearance_drift = 1.0
            if oid in last_velocity and oid in last_known_position:
                pred = (last_known_position[oid][0] + last_velocity[oid][0],
                        last_known_position[oid][1] + last_velocity[oid][1])
                pos_err = np.hypot(centroid[0] - pred[0], centroid[1] - pred[1])
                tracking_drift = min(pos_err / CENTROID_DISTANCE_THRESHOLD, 1.0)
            else:
                tracking_drift = 0.0
            suspicion = W_APPEARANCE * appearance_drift + W_TRACKING * tracking_drift

            if suspicion > DRIFT_THRESHOLD:
                consec_high[oid] += 1
            else:
                consec_high[oid] = 0

            never_classified = oid not in never_classified_drift
            drift_invoked = never_classified or (consec_high[oid] >= HYSTERESIS_FRAMES)
            if drift_invoked:
                ref_hist_drift[oid] = calculate_histogram(roi)
                never_classified_drift.add(oid)
                consec_high[oid] = 0
                displayed_label["drift"][oid] = classify(model, roi, preprocess_input)
                total_invocations["drift"] += 1

            rows.append({
                "frame": frame_idx,
                "object_id": oid,
                "ground_truth": ground_truth if ground_truth else "N/A",
                "always_label": always_lbl,
                "always_correct": (ground_truth == always_lbl) if ground_truth else None,
                "always_invoked": always_invoked,
                "fixed_label": displayed_label["fixed"].get(oid),
                "fixed_correct": (ground_truth == displayed_label["fixed"].get(oid)) if ground_truth else None,
                "fixed_invoked": fixed_invoked,
                "drift_label": displayed_label["drift"].get(oid),
                "drift_correct": (ground_truth == displayed_label["drift"].get(oid)) if ground_truth else None,
                "drift_invoked": drift_invoked,
                "drift_agrees_with_always": displayed_label["drift"].get(oid) == always_lbl,
            })

    cap.release()

    if not rows:
        print(f"[WARNING] {scenario_name}: no scored frames.")
        return None

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {"scenario": scenario_name, "total_frames": frame_idx, "scored_rows": len(rows)}
    for strat in ["always", "fixed", "drift"]:
        correct = sum(1 for r in rows if r[f"{strat}_correct"] is True)
        scored = sum(1 for r in rows if r[f"{strat}_correct"] is not None)
        acc = correct / scored * 100 if scored else float("nan")
        rate = total_invocations[strat] / len(rows) * 100
        summary[f"{strat}_accuracy"] = round(acc, 2)
        summary[f"{strat}_invocations"] = total_invocations[strat]
        summary[f"{strat}_rate_pct"] = round(rate, 2)
    agree = sum(1 for r in rows if r["drift_agrees_with_always"]) / len(rows) * 100
    summary["drift_agreement_with_always_pct"] = round(agree, 2)

    print(f"\n{'='*70}\nRESULTS — {scenario_name}\n{'='*70}")
    for strat in ["always", "fixed", "drift"]:
        print(f"{strat:>8}: accuracy={summary[f'{strat}_accuracy']:6.2f}%  "
              f"invocations={summary[f'{strat}_invocations']:4d}  "
              f"rate={summary[f'{strat}_rate_pct']:6.2f}%")
    print(f"Drift agreement with always: {summary['drift_agreement_with_always_pct']:.2f}%")
    print(f"Saved: {out_path}\n")

    return summary


def main():
    all_summaries = []
    for model_name, model_path in MODELS.items():
        print(f"\n{'#'*70}\nLOADING MODEL: {model_name} ({model_path})\n{'#'*70}")
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        model = load_model_robust(model_path)
        model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)

        for scenario, video_path in VIDEOS.items():
            full_scenario_name = f"{scenario}_{model_name}"
            out_csv = os.path.join(OUTPUT_FOLDER, f"synchronized_results_{full_scenario_name}.csv")
            summary = run_single_test(
                video_path, model, preprocess_input,
                full_scenario_name, OCCLUDED_FRAMES.get(scenario, ""), out_csv
            )
            if summary:
                all_summaries.append(summary)

    # write a master summary CSV across all 8 runs
    if all_summaries:
        master_path = os.path.join(OUTPUT_FOLDER, "ALL_RESULTS_SUMMARY.csv")
        with open(master_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            w.writeheader()
            w.writerows(all_summaries)
        print(f"\n{'#'*70}\nALL RUNS COMPLETE. Master summary saved to:\n{master_path}\n{'#'*70}")


if __name__ == "__main__":
    main()
