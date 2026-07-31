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
MAX_FRAMES_WITHOUT_DETECTION = 25   # keep boundary memory for this many frames (was 10 - too aggressive)
CENTROID_DISTANCE_THRESHOLD = 150   # max pixels to consider it the same object (was 100 - too tight)

# --- DRIFT-TRIGGERED RE-VERIFICATION PARAMETERS ---
# NOTE: this is the ALWAYS-CLASSIFY BASELINE. The drift/suspicion score is
# still computed and logged (for comparison purposes) but does NOT control
# triggering here - every object gets reclassified every single frame.
DRIFT_THRESHOLD = 0.4        # kept for logging/comparison only in this baseline
HYSTERESIS_FRAMES = 3        # kept for logging/comparison only in this baseline
W_APPEARANCE = 0.6
W_TRACKING = 0.4

# --- BOUNDARY-CLOSURE HYSTERESIS PARAMETERS ---
# Two-tier thresholds so a boundary that's already confirmed "closed" doesn't
# flicker in and out just because solidity dips slightly for a frame or two.
SOLIDITY_CONFIRM = 0.70   # solidity needed to become "closed" in the first place
SOLIDITY_RELEASE = 0.40   # once closed, solidity must drop THIS low to even count as a bad frame
BOUNDARY_LOSS_BUFFER = 8  # consecutive bad frames tolerated before actually losing "closed" status

confirmed_closed = {}                       # object_id -> bool, has this object ever been confirmed closed
boundary_fail_streak = defaultdict(int)     # object_id -> consecutive frames below SOLIDITY_RELEASE

SMOOTHING_ALPHA = 0.3   # lower = smoother/steadier box, higher = more responsive to real movement
smoothed_bounds = {}    # object_id -> (x, y, w, h) exponentially smoothed, to stop visual jitter

# --- DIAGNOSTIC: WHY DID A CANDIDATE GET REJECTED THIS FRAME? ---
rejection_tally = defaultdict(int)   # reason -> count, reset every second alongside FPS
diagnostic_time = time.time()

reference_hist = {}                       # object_id -> color histogram at last classification
last_velocity = {}                        # object_id -> (vx, vy) estimated from recent centroids
consecutive_high_suspicion = defaultdict(int)   # object_id -> consecutive frames above threshold
invocation_count = defaultdict(int)       # object_id -> total number of times classifier was run
suspicion_log = []                        # rows of (frame, object_id, suspicion, triggered) for analysis
ground_truth_log = []                      # rows of (frame, note) - manually marked events for Test C analysis


def calculate_histogram(roi):
    """
    Cheap visual fingerprint of a crop, used to detect appearance drift.
    A color histogram is orders of magnitude cheaper than a CNN forward
    pass, so it's safe to compute every frame for every tracked object.
    """
    hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def compute_suspicion_score(object_id, roi, actual_centroid):
    """
    Combines two cheap signals into one "suspicion score":

      1. Appearance drift: how different does this crop look, compared
         to the crop we classified this object as, last time we classified it?
      2. Tracking drift: how far off was our motion prediction from where
         the object actually was detected this frame?

    Returns a suspicion score in [0, 1] (roughly) - higher means more
    reason to believe the tracked identity may no longer be reliable.
    """
    # --- Appearance drift ---
    if object_id in reference_hist and roi.size > 0:
        current_hist = calculate_histogram(roi)
        similarity = cv2.compareHist(reference_hist[object_id], current_hist, cv2.HISTCMP_CORREL)
        # similarity is roughly in [-1, 1]; clamp before converting to a drift score
        similarity = max(-1.0, min(1.0, similarity))
        appearance_drift = 1.0 - similarity
    else:
        # No reference yet (never classified) - treat as maximally uncertain
        appearance_drift = 1.0

    # --- Tracking drift ---
    if object_id in last_velocity and object_id in last_known_position:
        predicted = (
            last_known_position[object_id][0] + last_velocity[object_id][0],
            last_known_position[object_id][1] + last_velocity[object_id][1]
        )
        position_error = np.hypot(actual_centroid[0] - predicted[0],
                                   actual_centroid[1] - predicted[1])
        tracking_drift = min(position_error / CENTROID_DISTANCE_THRESHOLD, 1.0)
    else:
        tracking_drift = 0.0  # no velocity estimate yet, don't penalize on frame 1

    suspicion = W_APPEARANCE * appearance_drift + W_TRACKING * tracking_drift
    return suspicion


def calculate_centroid(contour):
    """Calculate centroid of a contour using image moments."""
    M = cv2.moments(contour)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    return None


def contour_quality(contour, frame_shape, border_margin=5):
    """
    Returns (passes_basic_shape_check, solidity, touches_border) instead of
    a single hard yes/no. This lets the caller apply hysteresis (loose
    threshold to accept, even looser to keep once already confirmed)
    instead of a single fixed cutoff that flickers on minor noise.
    """
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

    if current_time - diagnostic_time >= 1.0:
        if rejection_tally:
            print(f"[DIAGNOSTIC] Rejections in last ~1s: {dict(rejection_tally)}")
        rejection_tally.clear()
        diagnostic_time = current_time

    # --- EDGE DETECTION FOR CLOSED BOUNDARIES ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE (adaptive histogram equalization): normalizes LOCAL contrast so a
    # bright specular highlight (glare off a glossy pen) doesn't dominate the
    # gradient and split the outline into fragments. This is the main fix
    # for lighting-related solidity drops.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Slightly larger blur kernel than before - smooths out the sharp edge of
    # a specular hot-spot before Canny sees it as a false internal edge.
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blurred, 40, 120)

    # === CLOSING OPERATION - makes boundaries tight and closed ===
    # Slightly larger kernel + one more iteration than before, so a contour
    # that gets split by a glare gap has a better chance of being bridged
    # back into one solid closed shape.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=3)

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
            passes_shape, solidity, touches_border = contour_quality(cnt, frame.shape)

            # Diagnostic: tally exactly why each candidate got rejected, so we
            # can tell whether dropouts are from aspect ratio, low solidity,
            # or basic shape/area - border-touch alone no longer excludes a
            # candidate from the pool (see note below).
            if aspect_ratio <= 2.0:
                rejection_tally["aspect_ratio_too_low"] += 1
            elif not passes_shape:
                rejection_tally["failed_basic_shape"] += 1
            elif solidity < SOLIDITY_RELEASE:
                rejection_tally["solidity_below_release"] += 1
            if touches_border:
                rejection_tally["touches_border_flagged"] += 1  # informational, not a hard reject anymore

            # Loose acceptance into the candidate pool: use the LOW release
            # threshold for solidity, and DON'T hard-reject on border-touch
            # here anymore - border proximity is now handled with hysteresis
            # below, same as solidity, instead of an instant reject that was
            # causing objects near the frame edge to vanish into MEMORY status
            # every time they got close to (not even necessarily touching)
            # the border margin.
            if aspect_ratio > 2.0 and passes_shape and solidity >= SOLIDITY_RELEASE:
                newly_detected_contours.append((cnt, aspect_ratio, bounds, solidity, touches_border))

    # === MATCH DETECTED CONTOURS TO TRACKED OBJECTS (Hungarian algorithm) ===
    detected_ids = set()
    matched_contours = {}  # object_id -> (contour, aspect_ratio, bounds)

    centroids = []
    valid_entries = []  # (contour, aspect_ratio, bounds, solidity, touches_border, centroid)
    for contour, aspect_ratio, bounds, solidity, touches_border in newly_detected_contours:
        centroid = calculate_centroid(contour)
        if centroid is not None:
            centroids.append(centroid)
            valid_entries.append((contour, aspect_ratio, bounds, solidity, touches_border, centroid))

    tracked_ids = list(last_known_position.keys())
    tracked_positions = [last_known_position[tid] for tid in tracked_ids]

    assignment = match_objects_hungarian(centroids, tracked_ids, tracked_positions)

    for i, (contour, aspect_ratio, bounds, solidity, touches_border, centroid) in enumerate(valid_entries):
        if i in assignment:
            object_id = assignment[i]
        else:
            object_id = object_id_counter
            object_id_counter += 1
            print(f"[NEW OBJECT] Detected as Object {object_id}")

        # --- BOUNDARY-CLOSURE HYSTERESIS (now also covers border-touch) ---
        # To become confirmed in the first place, require BOTH good solidity
        # AND not touching the border (strict). Once confirmed, tolerate
        # either a solidity dip OR a border-touch frame as just a "bad frame"
        # that needs to repeat BOUNDARY_LOSS_BUFFER times before it's dropped,
        # instead of instantly losing status the moment it happens once.
        is_good_frame = (solidity >= SOLIDITY_CONFIRM) and not touches_border
        is_tolerable_frame = (solidity >= SOLIDITY_RELEASE)  # border-touch alone no longer disqualifies once confirmed

        if is_good_frame:
            confirmed_closed[object_id] = True
            boundary_fail_streak[object_id] = 0
        elif confirmed_closed.get(object_id, False) and is_tolerable_frame:
            boundary_fail_streak[object_id] = 0
        elif confirmed_closed.get(object_id, False):
            boundary_fail_streak[object_id] += 1
            if boundary_fail_streak[object_id] > BOUNDARY_LOSS_BUFFER:
                confirmed_closed[object_id] = False
        else:
            confirmed_closed[object_id] = False

        detected_ids.add(object_id)
        frame_since_last_seen[object_id] = 0

        # --- BOX POSITION SMOOTHING ---
        # Raw Canny/contour output wiggles a little every frame even for a
        # perfectly still object (sensor noise, tiny lighting variation).
        # Smooth (x, y, w, h) with an exponential moving average so the drawn
        # box stops visually "floating" once an object is confirmed closed.
        raw_x, raw_y, raw_w, raw_h = bounds
        if object_id in smoothed_bounds and confirmed_closed.get(object_id, False):
            sx, sy, sw, sh = smoothed_bounds[object_id]
            smoothed = (
                int(SMOOTHING_ALPHA * raw_x + (1 - SMOOTHING_ALPHA) * sx),
                int(SMOOTHING_ALPHA * raw_y + (1 - SMOOTHING_ALPHA) * sy),
                int(SMOOTHING_ALPHA * raw_w + (1 - SMOOTHING_ALPHA) * sw),
                int(SMOOTHING_ALPHA * raw_h + (1 - SMOOTHING_ALPHA) * sh),
            )
        else:
            smoothed = bounds  # first sighting or not yet confirmed - use raw position directly
        smoothed_bounds[object_id] = smoothed

        matched_contours[object_id] = (contour, aspect_ratio, smoothed)

        # Update velocity estimate BEFORE overwriting last_known_position,
        # so compute_suspicion_score() can use it to predict this frame's position.
        if object_id in last_known_position:
            prev_centroid = last_known_position[object_id]
            last_velocity[object_id] = (centroid[0] - prev_centroid[0],
                                         centroid[1] - prev_centroid[1])

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
            roi = frame[y:y + h, x:x + w]
            centroid_now = last_known_position.get(obj_idx, (x + w // 2, y + h // 2))

            never_classified = obj_idx not in reference_hist
            suspicion = compute_suspicion_score(obj_idx, roi, centroid_now)

            # Hysteresis: require several consecutive suspicious frames
            # before triggering, so single-frame noise doesn't cause
            # constant, wasteful reclassification.
            if suspicion > DRIFT_THRESHOLD:
                consecutive_high_suspicion[obj_idx] += 1
            else:
                consecutive_high_suspicion[obj_idx] = 0

            should_trigger = True  # always-classify baseline: no drift-gating at all

            suspicion_log.append((total_frames, obj_idx, round(float(suspicion), 4), should_trigger))

            if should_trigger and processing_locks[obj_idx].acquire(blocking=False):
                if roi.size > 0:
                    # Reset the reference fingerprint now, using the crop that
                    # triggered reclassification, so future drift is measured
                    # from this fresh checkpoint rather than the original one.
                    reference_hist[obj_idx] = calculate_histogram(roi)
                    consecutive_high_suspicion[obj_idx] = 0
                    invocation_count[obj_idx] += 1

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
        if not confirmed_closed.get(obj_idx, False):
            label_with_id += " (stabilizing)"
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
                print(f"[EVICTED] Object {obj_id} removed after {frame_since_last_seen[obj_id]} "
                      f"frames without detection (frame {total_frames})")
                boundary_memory.pop(obj_id, None)
                last_known_position.pop(obj_id, None)
                frame_since_last_seen.pop(obj_id, None)
                reference_hist.pop(obj_id, None)
                last_velocity.pop(obj_id, None)
                consecutive_high_suspicion.pop(obj_id, None)
                confirmed_closed.pop(obj_id, None)
                boundary_fail_streak.pop(obj_id, None)
                smoothed_bounds.pop(obj_id, None)
                with results_lock:
                    object_results.pop(obj_id, None)

    # === DISPLAY FRAME & FPS COUNTER ===
    cv2.putText(frame, f"Frame: {total_frames}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Detected: {len(detected_ids)} | Memory: {len(boundary_memory)}",
                (w_f - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    total_invocations = sum(invocation_count.values())
    cv2.putText(frame, f"Classifier calls (total): {total_invocations}",
                (w_f - 350, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

    # === DISPLAY ===
    cv2.putText(frame, "Press 'g' to mark event | 'q' to quit", (10, h_f - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    stacked = np.hstack((frame, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))
    cv2.imshow("Pen Detector - ALWAYS CLASSIFY BASELINE", stacked)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('g'):
        ground_truth_log.append((total_frames, "manual_event_marker"))
        print(f"[GROUND TRUTH] Event marked at frame {total_frames} "
              f"(e.g. crossing/occlusion happened around here)")

print(f"\nTotal frames processed: {total_frames}")
print(f"Total unique objects tracked: {object_id_counter}")
print(f"Total classifier invocations: {sum(invocation_count.values())}")
print("=" * 80 + "\n")

# --- EXPORT LOG FOR EXPERIMENT ANALYSIS ---
# This CSV is what you'll use to build the accuracy-vs-invocation-count
# plots and the threshold sweep comparisons for the paper.
import csv
with open("suspicion_log_always_classify.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "object_id", "suspicion_score", "triggered_reclassification"])
    writer.writerows(suspicion_log)
print("Suspicion log saved to suspicion_log_always_classify.csv")

with open("ground_truth_events_always_classify.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "note"])
    writer.writerows(ground_truth_log)
print(f"Ground truth events saved to ground_truth_events_always_classify.csv ({len(ground_truth_log)} events marked)")

cap.release()
cv2.destroyAllWindows()
