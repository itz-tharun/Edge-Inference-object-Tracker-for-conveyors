import cv2
import time
import numpy as np
import threading
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input

# --- 1. SETUP & MODEL ---
MODEL_PATH = r'C:\Users\dhars\Downloads\pen_detector_model.h5'
print("Loading Model...")
model = load_model(MODEL_PATH)

# Adjust these if your model's classes are in a different order (e.g., [Not Pen, Pen])
LABELS = ["Pen", "Not a Pen"] 

# Shared variables for threading
current_result = {"label": "Analyzing...", "color": (0, 255, 255)} # Default Yellow
is_processing = False

def run_inference(roi_frame):
    """Processes the frame through your MobileNet model."""
    global current_result, is_processing
    try:
        # Convert BGR (OpenCV) to RGB (MobileNet)
        img = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        # Preprocessing required for MobileNet .h5 models
        img_array = np.expand_dims(img, axis=0)
        img_preprocessed = preprocess_input(img_array.astype('float32'))
        
        # Get Prediction
        preds = model.predict(img_preprocessed, verbose=0)
        idx = np.argmax(preds[0])
        confidence = preds[0][idx]
        
        # Determine Label and Color based on model output
        label_text = LABELS[idx]
        if label_text == "Pen":
            box_color = (0, 255, 0)  # Green
        else:
            box_color = (0, 0, 255)  # Red
            
        current_result = {
            "label": f"{label_text} ({int(confidence*100)}%)",
            "color": box_color
        }
    except Exception as e:
        print(f"ML Error: {e}")
    finally:
        is_processing = False

# --- 2. VIDEO PROCESSING ---
cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    h_f, w_f = frame.shape[:2]

    # Pre-processing for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Advanced detection to catch colored pens (Canny + Otsu)
    edges = cv2.Canny(blurred, 30, 100)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    combined = cv2.bitwise_or(edges, thresh)
    
    # Clean up the edges
    kernel = np.ones((5,5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    object_in_view = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 2000 < area < 100000: # Filter noise and huge background objects
            object_in_view = True
            x, y, w, h = cv2.boundingRect(cnt)
            
            # CHECK CLOSED BOUNDARY (Is object fully inside frame?)
            is_fully_in = x > 20 and y > 20 and (x+w) < (w_f-20) and (y+h) < (h_f-20)

            if is_fully_in:
                # Trigger ML Inference if we aren't already processing
                if not is_processing:
                    is_processing = True
                    roi = frame[y:y+h, x:x+w]
                    threading.Thread(target=run_inference, args=(roi,), daemon=True).start()
                
                # Use the results from the ML model
                display_color = current_result["color"]
                display_label = current_result["label"]
            else:
                # Object is still entering or exiting
                display_color = (255, 255, 255) # White for "Wait"
                display_label = "Bringing object into frame..."

            # DRAW BOUNDING BOX AND LABEL
            cv2.rectangle(frame, (x, y), (x + w, y + h), display_color, 2)
            cv2.putText(frame, display_label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, display_color, 2)

    if not object_in_view:
        current_result = {"label": "Scanning...", "color": (0, 255, 255)}

    # UI Side-by-Side
    edge_visual = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack((frame, edge_visual))
    
    fps = 1 / (time.time() - prev_time + 1e-5)
    prev_time = time.time()
    cv2.putText(stacked, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Pen Detection Workflow", stacked)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
