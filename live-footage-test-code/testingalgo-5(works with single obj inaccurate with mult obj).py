import cv2
import time
import numpy as np
import threading
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input

# --- 1. SETUP ---
MODEL_PATH = r'C:\Users\dhars\Downloads\pen_detector_model.h5'
model = load_model(MODEL_PATH)

# Keeping standard labels - the logic below handles the 55% flip
LABELS = ["Pen", "Not a Pen"] 

current_result = {"label": "Scanning...", "color": (255, 255, 255)}
is_processing = False

def run_inference(roi_frame):
    global current_result, is_processing
    try:
        # Preprocessing for MobileNet
        img = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img_array = np.expand_dims(img, axis=0)
        img_preprocessed = preprocess_input(img_array.astype('float32'))
        
        preds = model.predict(img_preprocessed, verbose=0)
        idx = np.argmax(preds[0])
        confidence = preds[0][idx]
        
        # --- THE FLIPPED LOGIC ---
        # If the model's confidence is LOWER than 55%, we label it as a PEN
        if confidence < 0.55:
            current_result = {
                "label": f"PEN (Uncertain: {int(confidence*100)}%)",
                "color": (0, 255, 0) # Green
            }
        else:
            # High confidence results are labeled as NOT A PEN
            current_result = {
                "label": f"NOT A PEN ({int(confidence*100)}%)",
                "color": (0, 0, 255) # Red
            }
            
        print(f"Confidence: {confidence:.2f} | Resulting Label: {current_result['label']}")

    except Exception as e:
        print(f"ML Error: {e}")
    finally:
        is_processing = False

# --- 2. VIDEO LOOP ---
cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    h_f, w_f = frame.shape[:2]

    # Outline Detection (Canny)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120) 
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    obj_found = False
    for cnt in contours:
        if cv2.contourArea(cnt) > 3000:
            obj_found = True
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Boundary check
            is_fully_in = x > 20 and y > 20 and (x+w) < (w_f-20) and (y+h) < (h_f-20)

            if is_fully_in:
                if not is_processing:
                    is_processing = True
                    roi = frame[y:y+h, x:x+w]
                    threading.Thread(target=run_inference, args=(roi,), daemon=True).start()
                
                clr = current_result["color"]
                lbl = current_result["label"]
            else:
                clr = (255, 255, 255) # White
                lbl = "Positioning..."

            cv2.rectangle(frame, (x, y), (x + w, y + h), clr, 2)
            cv2.putText(frame, lbl, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2)

    if not obj_found:
        current_result = {"label": "Scanning...", "color": (255, 255, 255)}

    # UI View
    edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack((frame, edge_bgr))
    
    fps = 1 / (time.time() - prev_time + 1e-5)
    prev_time = time.time()
    cv2.putText(stacked, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Pen Detection (Confidence < 55%)", stacked)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
