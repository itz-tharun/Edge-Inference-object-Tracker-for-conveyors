import cv2
import time
import numpy as np
import threading
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input

# --- CONFIG ---
MODEL_PATH = r'C:\Users\dhars\Downloads\pen_detector_model.h5'
LABELS = ["Pen", "Not a Pen"] # Ensure this matches your training order!

print("Loading Model...")
model = load_model(MODEL_PATH)

current_label = "Scanning..."
is_processing = False

def run_inference(roi_frame):
    global current_label, is_processing
    try:
        # 1. Resize and convert to RGB (OpenCV uses BGR, MobileNet wants RGB)
        img = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        # 2. Preprocess (This handles the scaling/normalization properly)
        img_array = np.expand_dims(img, axis=0)
        img_preprocessed = preprocess_input(img_array.astype('float32'))
        
        # 3. Predict
        preds = model.predict(img_preprocessed, verbose=0)
        idx = np.argmax(preds[0])
        conf = preds[0][idx]
        current_label = f"{LABELS[idx]} ({int(conf*100)}%)"
    except Exception as e:
        print(f"Inference Error: {e}")
    finally:
        is_processing = False

cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    h_f, w_f = frame.shape[:2]

    # --- IMAGE PROCESSING ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Dial in the Canny thresholds - lower values = more edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # MORPHOLOGY: This 'thickens' edges to help close the boundary
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found_obj = False
    for cnt in contours:
        if cv2.contourArea(cnt) > 2500: # Adjust based on pen size
            found_obj = True
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Closed Boundary Check (Safety margin of 20px from edge)
            is_fully_in = x > 20 and y > 20 and (x+w) < (w_f-20) and (y+h) < (h_f-20)

            if is_fully_in:
                color = (0, 255, 0)
                if not is_processing:
                    is_processing = True
                    roi = frame[y:y+h, x:x+w]
                    threading.Thread(target=run_inference, args=(roi,), daemon=True).start()
            else:
                color = (0, 0, 255)
                current_label = "Pen Entering..."

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, current_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if not found_obj: current_label = "Waiting..."

    # --- UI: SIDE BY SIDE ---
    # Convert edges to 3-channel so we can stack it with the color frame
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    stacked_view = np.hstack((frame, edges_bgr))

    # FPS Calculation
    fps = 1 / (time.time() - prev_time + 1e-5)
    prev_time = time.time()
    cv2.putText(stacked_view, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Detection (Left) | Canny Edges (Right)", stacked_view)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()