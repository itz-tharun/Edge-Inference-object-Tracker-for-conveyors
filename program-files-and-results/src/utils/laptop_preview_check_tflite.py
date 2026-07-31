"""
laptop_preview_check_tflite.py
--------------------------------
Same as laptop_preview_check.py, but runs the .tflite model instead of
the .h5 model - so you can visually confirm the float32 .tflite file
behaves the same way on your real footage before sending it to the Pi.

Auto-detects whether the .tflite file is int8 or float32 and
preprocesses accordingly, so this same script works with either file.

USAGE:
    python laptop_preview_check_tflite.py
    python laptop_preview_check_tflite.py --video other_clip.mp4 --model other_model.tflite

Controls: q = quit, space = pause/resume
Headless: add --headless
Disable saving the annotated output video: --output ""

Dependencies:
    pip install opencv-python numpy scipy
    pip install ai-edge-litert
"""

import argparse
import sys
import time

import cv2
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

from scipy.optimize import linear_sum_assignment

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


def preprocess_for_model(img_uint8_224, input_detail):
    """Auto-detects int8 vs float32 model input and preprocesses accordingly."""
    if input_detail["dtype"] == np.int8:
        scale, zero_point = input_detail["quantization"]
        if scale == 0:
            return img_uint8_224.astype(np.int8)
        real = img_uint8_224.astype(np.float32) / 255.0
        q = np.round(real / scale + zero_point)
        return np.clip(q, -128, 127).astype(np.int8)
    else:
        img = img_uint8_224.astype(np.float32)
        return img / 127.5 - 1.0


def dequantize_output(raw_out, output_detail):
    if output_detail["dtype"] == np.int8:
        out_scale, out_zero = output_detail["quantization"]
        return (float(raw_out) - out_zero) * out_scale if out_scale != 0 else float(raw_out)
    return float(raw_out)


def classify(interpreter, input_detail, output_detail, roi):
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    q = preprocess_for_model(img, input_detail)
    q = np.expand_dims(q, axis=0)
    interpreter.set_tensor(input_detail["index"], q)
    interpreter.invoke()
    raw_out = interpreter.get_tensor(output_detail["index"])[0][0]
    conf = dequantize_output(raw_out, output_detail)
    if conf < 0.5:
        return f"PEN ({int((1 - conf) * 100)}%)", (0, 255, 0)
    else:
        return f"NOT A PEN ({int(conf * 100)}%)", (0, 0, 255)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=r"C:\Users\dhars\Desktop\pen_test_video.mp4")
    parser.add_argument("--model", default=r"C:\Users\dhars\Desktop\pen_detector_model_float32.tflite")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--speed", type=float, default=4.0)
    parser.add_argument("--output", default=r"C:\Users\dhars\Desktop\pen_test_annotated_output_tflite.mp4")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: could not open '{args.video}'")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    target_frame_ms = (1000.0 / src_fps) / args.speed

    print("[INFO] Loading TFLite model...")
    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    print(f"[INFO] Model input dtype: {input_detail['dtype']} (int8 = quantized, float32 = fixed version)")
    print(f"[INFO] Video FPS reported: {src_fps:.1f}")

    save_output = bool(args.output)
    should_draw = (not args.headless) or save_output
    writer = None

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
                    print(f"[WARN] Could not open '{args.output}' for writing.")
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
                        label, color = classify(interpreter, input_detail, output_detail, roi)
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
                cv2.imshow("preview (tflite)", frame)

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


if __name__ == "__main__":
    main()