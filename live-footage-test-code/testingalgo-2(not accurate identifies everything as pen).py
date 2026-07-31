import cv2
import time
import numpy as np
import threading
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input

# --- SETUP ---
MODEL_PATH = r'C:\Users\dhars\Downloads\pen_detector_model.h5'
LABELS = ["Pen", "Not a Pen"] 
model = load_model(MODEL_PATH)

current_label = "Scanning..."
is_processing = False

def run_inference(roi_frame):
    global current_label, is_processing
    try:
        img = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img_array = np.expand_dims(img, axis=0)
        img_preprocessed = preprocess_input(img_array.astype('float32'))
        
        preds = model.predict(img_preprocessed, verbose=0)
        idx = np.argmax(preds[0])
        current_label = f"{LABELS[idx]} ({int(preds[0][idx]*100)}%)"
    except Exception as e:
        print(f"ML Error: {e}")
    finally:
        is_processing = False

cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    h_f, w_f = frame.shape[:2]

    # 1. BOOST CONTRAST (Helps see non-black pens)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_img = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)

    # 2. BETTER EDGE DETECTION (Adaptive)
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Lower thresholds to catch lighter colors
    edges = cv2.Canny(blurred, 30, 100) 
    
    # Combine with Thresholding to catch solid colored shapes
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    combined_edges = cv2.bitwise_or(edges, thresh) # This catches black AND colors

    # 3. CLEAN UP
    kernel = np.ones((5,5), np.uint8)
    combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found_obj = False
    for cnt in contours:
        if 1500 < cv2.contourArea(cnt) < 50000: # Filter out background noise
            found_obj = True
            x, y, w, h = cv2.boundingRect(cnt)
            is_fully_in = x > 15 and y > 15 and (x+w) < (w_f-15) and (y+h) < (h_f-15)

            if is_fully_in:
                color = (0, 255, 0)
                if not is_processing:
                    is_processing = True
                    roi = frame[y:y+h, x:x+w]
                    threading.Thread(target=run_inference, args=(roi,), daemon=True).start()
            else:
                color = (0, 0, 255)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, current_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # UI SIDE BY SIDE
    combined_visual = cv2.cvtColor(combined_edges, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack((frame, combined_visual))
    
    fps = 1 / (time.time() - prev_time + 1e-5)
    prev_time = time.time()
    cv2.putText(stacked, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Original + ML (Left) | Advanced Edges (Right)", stacked)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
