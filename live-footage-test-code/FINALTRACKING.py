import cv2
import time
import numpy as np
import threading
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

# --- 1. SETUP ---
MODEL_PATH = r'C:\Users\dhars\Downloads\pen_detector_model.h5'
model = load_model(MODEL_PATH)

# Frame counters
frame_count = 0          # frames counted within the current 1-second FPS window
total_frames = 0         # true running total across the whole session
fps_time = time.time()
fps = 0

# Multi-object tracking with boundary persistence
object_results = {}                       # object_id -> classification result
results_lock = threading.Lock()           # protects object_results across threads
processing_locks = defaultdict(threading.Lock)
boundary_memory = {}                      # object_id -> last known closed contour data
last_known_position = {}                  # object_id -> last known centroid
frame_since_last_seen = defaultdict(int)  # object_id -> frames since last detection

# Parameters for boundary persistence
MAX_FRAMES_WITHOUT_DETECTION = 10   # keep boundary memory for this many frames
CENTROID_DISTANCE_THRESHOLD = 100   # max pixels to consider it the same object


def calculate_centroid(contour):
    """Calculate centroid of a contour using image moments."""
    M = cv2.moments(contour)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    return None


def is_closed_boundary(contour, frame_shape, border_margin=5):
    """
    Real closed-boundary check.

    cv2.findContours always returns closed point loops, so 'closed' here
    really means: does the contour represent a solid, fully-formed shape
    rather than a broken/partial edge fragment?

    We check:
      1. Area is non-trivial.
      2. Solidity (area / convex-hull area) is high -> outline isn't full
         of gaps/holes, which happens when Canny edges don't fully connect.
      3. The contour doesn't touch the frame border -> it isn't a shape
         that's cut off / clipped by the edge of the image.
    """
    area = cv2.contourArea(contour)
    if area < 1000 or len(contour) < 4:
        return False

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return False
    solidity = area / hull_area
    if solidity < 0.7:
        return False

    h_f, w_f = frame_shape[:2]
    x, y, w, h = cv2.boundingRect(contour)
    if x <= border_margin or y <= border_margin or \
       (x + w) >= (w_f - border_margin) or (y + h) >= (h_f - border_margin):
        return False

    return True


def match_objects_hungarian(detections, tracked_ids, tracked_positions,
                             max_distance=CENTROID_DISTANCE_THRESHOLD):
    """
    Optimal assignment between new detections and existing tracked objects
    using the Hungarian algorithm, instead of greedy nearest-neighbor
    (which can misassign IDs when two objects are close together).

    detections: list of centroids for this frame's contours
    tracked_ids: list of existing object ids
    tracked_positions: list of last known centroids, same order as tracked_ids

    Returns: dict mapping detection_index -> object_id (only for matches
    within max_distance); unmatched detections are left out.
    """
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


def run_inference(roi_frame, object_id):
    """Run inference for a specific object in its own thread."""
    global object_results
    try:
        img = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        img_array = np.expand_dims(img, axis=0)
        img_preprocessed = preprocess_input(img_array.astype('float32'))

        preds = model.predict(img_preprocessed, verbose=0)
        conf = preds[0][0]

        print(f"[Object {object_id}] Confidence: {conf:.4f}")

        # INVERTED LOGIC - Low confidence = PEN
        if conf < 0.5:
            label = f"PEN ({int((1 - conf) * 100)}%)"
            color = (0, 255, 0)
        else:
            label = f"NOT A PEN ({int(conf * 100)}%)"
            color = (0, 0, 255)

        with results_lock:
            object_results[object_id] = {
                "label": label,
                "color": color,
                "confidence": conf
            }

        print(f"[Object {object_id}] Classification: {label}")

    except Exception as e:
        print(f"ML Error on Object {object_id}: {e}")
        with results_lock:
            object_results[object_id] = {
                "label": "ERROR",
                "color": (0, 0, 255),
                "confidence": 0
            }
    finally:
        processing_locks[object_id].release()


def calculate_aspect_ratio(contour):
    """Calculate aspect ratio to filter out non-pen objects."""
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    if aspect_ratio < 1 and aspect_ratio > 0:
        aspect_ratio = 1 / aspect_ratio
    return aspect_ratio, (x, y, w, h)


# --- 2. VIDEO LOOP ---
cap = cv2.VideoCapture(0)


print("\n" + "=" * 80)
print("MULTI-OBJECT PEN DETECTOR - WITH BOUNDARY PERSISTENCE (FIXED)")
print("=" * 80)
print("Features:")
print("  [+] Detects and classifies MULTIPLE objects simultaneously")
print("  [+] Real closed-boundary check (solidity + border test)")
print("  [+] Optimal ID matching via Hungarian algorithm")
print("  [+] Remembers closed boundaries even when edges break")
print("  [+] Shows frame counter and FPS in real-time")
print("  [+] Parallel inference for each object")
print("=" * 80 + "\n")

object_id_counter = 0  # global counter for new object IDs

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    total_frames += 1
    h_f, w_f = frame.shape[:2]

    # === FPS CALCULATION ===
    current_time = time.time()
    if current_time - fps_time >= 1.0:
        fps = frame_count
        frame_count = 0
        fps_time = current_time

    # --- EDGE DETECTION FOR CLOSED BOUNDARIES ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 40, 120)

    # === CLOSING OPERATION - makes boundaries tight and closed ===
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Dilation to fill gaps
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, kernel_dilate, iterations=1)

    # --- FIND ALL CONTOURS ---
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- FILTER CONTOURS BY SIZE, SHAPE, AND REAL CLOSURE ---
    newly_detected_contours = []

    if contours:
        large_contours = [c for c in contours if cv2.contourArea(c) > 3000]

        for cnt in large_contours:
            aspect_ratio, bounds = calculate_aspect_ratio(cnt)
            # Pens are elongated: aspect ratio > 2.0, and must be a real
            # solid, non-clipped shape (see is_closed_boundary docstring)
            if aspect_ratio > 2.0 and is_closed_boundary(cnt, frame.shape):
                newly_detected_contours.append((cnt, aspect_ratio, bounds))

    # === MATCH DETECTED CONTOURS TO TRACKED OBJECTS (Hungarian algorithm) ===
    detected_ids = set()
    matched_contours = {}  # object_id -> (contour, aspect_ratio, bounds)

    centroids = []
    valid_entries = []  # (contour, aspect_ratio, bounds, centroid), aligned with centroids
    for contour, aspect_ratio, bounds in newly_detected_contours:
        centroid = calculate_centroid(contour)
        if centroid is not None:
            centroids.append(centroid)
            valid_entries.append((contour, aspect_ratio, bounds, centroid))

    tracked_ids = list(last_known_position.keys())
    tracked_positions = [last_known_position[tid] for tid in tracked_ids]

    assignment = match_objects_hungarian(centroids, tracked_ids, tracked_positions)

    for i, (contour, aspect_ratio, bounds, centroid) in enumerate(valid_entries):
        if i in assignment:
            object_id = assignment[i]
        else:
            object_id = object_id_counter
            object_id_counter += 1
            print(f"[NEW OBJECT] Detected as Object {object_id}")

        detected_ids.add(object_id)
        frame_since_last_seen[object_id] = 0
        matched_contours[object_id] = (contour, aspect_ratio, bounds)
        last_known_position[object_id] = centroid

        # Store closed boundary in memory (already verified closed above)
        boundary_memory[object_id] = {
            "contour": contour.copy(),
            "aspect_ratio": aspect_ratio,
            "bounds": bounds,
            "centroid": centroid
        }

    # === PROCESS DETECTED CONTOURS ===
    for obj_idx, (contour, aspect_ratio, (x, y, w, h)) in matched_contours.items():
        is_fully_in = x > 10 and y > 10 and (x + w) < (w_f - 10) and (y + h) < (h_f - 10)

        with results_lock:
            if obj_idx not in object_results:
                object_results[obj_idx] = {
                    "label": "Scanning...",
                    "color": (255, 255, 255),
                    "confidence": 0
                }

        if is_fully_in:
            if processing_locks[obj_idx].acquire(blocking=False):
                roi = frame[y:y + h, x:x + w]
                if roi.size > 0:
                    threading.Thread(
                        target=run_inference,
                        args=(roi, obj_idx),
                        daemon=True
                    ).start()
                else:
                    processing_locks[obj_idx].release()

            with results_lock:
                clr = object_results[obj_idx]["color"]
                lbl = object_results[obj_idx]["label"]
        else:
            clr = (255, 255, 255)
            lbl = "Positioning..."

        # === DRAW BOUNDING BOX ===
        cv2.rectangle(frame, (x, y), (x + w, y + h), clr, 2)

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(frame, [approx], 0, clr, 2)

        label_with_id = f"[{obj_idx}] {lbl}"
        cv2.putText(frame, label_with_id, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2)

        cv2.putText(frame, f"AR:{aspect_ratio:.2f}", (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # === DRAW BOUNDARY MEMORY (objects not currently detected but in memory) ===
    memory_ids_to_process = list(boundary_memory.keys())

    for obj_id in memory_ids_to_process:
        if obj_id not in detected_ids:
            frame_since_last_seen[obj_id] += 1

            if frame_since_last_seen[obj_id] <= MAX_FRAMES_WITHOUT_DETECTION:
                if obj_id in boundary_memory:
                    mem_data = boundary_memory[obj_id]
                    contour = mem_data["contour"]
                    bounds = mem_data["bounds"]
                    x, y, w, h = bounds

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
                    cv2.drawContours(frame, [contour], 0, (100, 100, 100), 1)

                    label_with_id = f"[{obj_id}] MEMORY"
                    cv2.putText(frame, label_with_id, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            else:
                boundary_memory.pop(obj_id, None)
                last_known_position.pop(obj_id, None)
                frame_since_last_seen.pop(obj_id, None)
                with results_lock:
                    object_results.pop(obj_id, None)

    # === DISPLAY FRAME & FPS COUNTER ===
    cv2.putText(frame, f"Frame: {total_frames}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Detected: {len(detected_ids)} | Memory: {len(boundary_memory)}",
                (w_f - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # === DISPLAY ===
    stacked = np.hstack((frame, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))
    cv2.imshow("Pen Detector - Boundary Persistence", stacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\nTotal frames processed: {total_frames}")
print(f"Total unique objects tracked: {object_id_counter}")
print("=" * 80 + "\n")

cap.release()
cv2.destroyAllWindows()