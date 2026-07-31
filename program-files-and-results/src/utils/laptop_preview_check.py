"""
laptop_preview_check.py
------------------------
Local sanity-check tool - run this on your laptop BEFORE sending a video
off to your friend's Raspberry Pi. It plays the recording back with the
SAME contour/tracking detection logic the Pi benchmark uses, draws the
detected box on screen, and runs the actual classifier on it live, so
you can visually confirm:

    (a) the classical-CV tracker actually detects the pen (passes the
        area / aspect-ratio / solidity filters), and
    (b) the classifier reads the crop as PEN, not NOT-A-PEN

This uses the same pen_detector_model.h5 Keras model used elsewhere in
this project, so what you see here is a real preview of what that test
will capture - not a separate, different check.

USAGE:
    python laptop_preview_check.py --video your_recording.mp4 --model pen_detector_model.h5

Controls while the window is open:
    q     = quit early
    space = pause / resume

If you just want a quick pass/fail without opening a window:
    python laptop_preview_check.py --video your_recording.mp4 --model pen_detector_model.h5 --headless

By default this ALSO saves an annotated copy of the video (boxes + labels
drawn in) to your Desktop, so you can review it afterward or send it to
your friend as proof of what the model sees. To turn that off:
    python laptop_preview_check.py --output ""

Dependencies (Windows/Mac/Linux laptop):
    pip install opencv-python numpy scipy tensorflow
    # (NOT opencv-python-headless here - you want the GUI build so
    #  cv2.imshow actually opens a window; headless is for the Pi.)
"""

import argparse
import sys
import time

import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    print("ERROR: scipy not found. pip install scipy")
    sys.exit(1)


CENTROID_DISTANCE_THRESHOLD = 150


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
    # cv2.boundingRect is axis-aligned, so a pen held diagonally reports a
    # near-square box (elongation lost) even though the object itself is
    # long and thin. cv2.minAreaRect finds the box at the object's own
    # rotation, so elongation is measured correctly at any angle - this is
    # what lets multiple pens at different orientations all pass the
    # aspect-ratio filter instead of only whichever one is roughly
    # horizontal/vertical in the frame.
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    long_side, short_side = max(rw, rh), min(rw, rh)
    ar = long_side / short_side if short_side > 0 else 0
    bounds = cv2.boundingRect(contour)
    return ar, bounds


def match_objects_hungarian(detections, tracked_ids, tracked_positions,
                             max_distance=CENTROID_DISTANCE_THRESHOLD):
    if not detections or not tracked_ids:
        return {}
    cost = np.zeros((len(detections), len(tracked_ids)), dtype=np.float64)
    for i, det in enumerate(detections):
        for j, trk in enumerate(tracked_positions):
            cost[i, j] = np.hypot(det[0] - trk[0], det[1] - trk[1])
    row_idx, col_idx = linear_sum_assignment(cost)
    matches = {}
    for r, c in zip(row_idx, col_idx):
        if cost[r, c] < max_distance:
            matches[r] = tracked_ids[c]
    return matches


def classify(model, roi):
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img, axis=0)
    img_preprocessed = preprocess_input(img_array.astype('float32'))
    conf = model.predict(img_preprocessed, verbose=0)[0][0]
    if conf < 0.5:
        return f"PEN ({int((1 - conf) * 100)}%)", (0, 255, 0)
    else:
        return f"NOT A PEN ({int(conf * 100)}%)", (0, 0, 255)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=r"C:\Users\dhars\Desktop\pen_test_video.mp4",
                         help="Path to the recorded video file")
    parser.add_argument("--model", default=r"C:\Users\dhars\Downloads\pen_detector_model.h5",
                         help="Path to the .h5 Keras model")
    parser.add_argument("--headless", action="store_true",
                         help="Skip the display window, just print a summary at the end")
    parser.add_argument("--speed", type=float, default=4.0,
                         help="Playback speed multiplier - higher plays faster. Default 4x, since real-time is usually too slow for a quick check.")
    parser.add_argument("--output", default=r"C:\Users\dhars\Desktop\pen_test_annotated_output.mp4",
                         help="Where to save the annotated output video (boxes + labels drawn in). "
                              "Pass --output \"\" to skip saving.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: could not open '{args.video}'")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    target_frame_ms = (1000.0 / src_fps) / args.speed

    print("[INFO] Loading Keras model...")
    model = load_model(args.model)
    model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)  # warm up
    print(f"[INFO] Video FPS reported: {src_fps:.1f}")

    # Annotations (boxes/labels) get drawn whenever they're needed for either
    # the live window or the saved file - drawing itself is cheap, so this
    # just controls whether we bother, not two separate code paths.
    save_output = bool(args.output)
    should_draw = (not args.headless) or save_output
    writer = None  # created lazily once we know the real frame size

    last_known_position = {}
    object_id_counter = 0

    total_frames = 0
    frames_with_detection = 0
    label_counts = {"PEN": 0, "NOT A PEN": 0}
    paused = False

    while True:
        if not paused:
            frame_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1
            h_f, w_f = frame.shape[:2]

            if save_output and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, src_fps, (w_f, h_f))
                if not writer.isOpened():
                    print(f"[WARN] Could not open '{args.output}' for writing - "
                          f"saving will be skipped. Check the folder exists.")
                    save_output = False
                    writer = None

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

            candidates = []
            for cnt in contours:
                if cv2.contourArea(cnt) <= 3000:
                    continue
                aspect_ratio, bounds = calculate_aspect_ratio(cnt)
                passes_shape, solidity, touches_border = contour_quality(cnt, frame.shape)
                if aspect_ratio > 2.0 and passes_shape and solidity >= 0.40:
                    centroid = calculate_centroid(cnt)
                    if centroid is not None:
                        candidates.append((bounds, solidity, aspect_ratio, centroid))

            tracked_ids = list(last_known_position.keys())
            tracked_positions = [last_known_position[t] for t in tracked_ids]
            centroids = [c[3] for c in candidates]
            assignment = match_objects_hungarian(centroids, tracked_ids, tracked_positions)

            new_positions = {}
            any_detection_this_frame = False

            for i, (bounds, solidity, aspect_ratio, centroid) in enumerate(candidates):
                object_id = assignment.get(i)
                if object_id is None:
                    object_id = object_id_counter
                    object_id_counter += 1
                new_positions[object_id] = centroid

                x, y, w, h = bounds
                is_fully_in = x > 10 and y > 10 and (x + w) < (w_f - 10) and (y + h) < (h_f - 10)

                if is_fully_in:
                    any_detection_this_frame = True
                    roi = frame[y:y + h, x:x + w]
                    if roi.size > 0:
                        label, color = classify(model, roi)
                        label_counts["PEN" if "NOT A PEN" not in label else "NOT A PEN"] += 1
                    else:
                        label, color = "empty ROI", (255, 255, 255)

                    if should_draw:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, f"[{object_id}] {label}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(frame, f"AR:{aspect_ratio:.2f} solidity:{solidity:.2f}",
                                    (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                elif should_draw:
                    x, y, w, h = bounds
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 150), 1)
                    cv2.putText(frame, "partially out of frame", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            last_known_position = new_positions
            if any_detection_this_frame:
                frames_with_detection += 1

            if should_draw:
                cv2.putText(frame, f"Frame {total_frames}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if save_output and writer is not None:
                writer.write(frame)

            if not args.headless:
                cv2.imshow("Laptop preview check - q to quit, space to pause", frame)

            elapsed_ms = (time.perf_counter() - frame_start) * 1000
            wait_ms = max(1, int(target_frame_ms - elapsed_ms))
        else:
            wait_ms = 30

        if not args.headless:
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
        # headless mode has no waitKey - just runs straight through

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
        print(f"\n[INFO] Annotated video saved to: {args.output}")

    pct = (frames_with_detection / total_frames * 100) if total_frames else 0
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total frames read           : {total_frames}")
    print(f"Frames with a valid detection: {frames_with_detection} ({pct:.1f}%)")
    print(f"Classification counts        : {label_counts}")
    if pct < 20:
        print("\n[FLAG] Detection rate is low - check lighting/background/pen")
        print("       visibility before sending this clip to the Pi test.")
    else:
        print("\nLooks usable - this clip is producing real detections.")


if __name__ == "__main__":
    main()