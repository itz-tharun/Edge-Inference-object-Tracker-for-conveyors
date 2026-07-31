import cv2
import numpy as np
import os
import glob

def process_robust_batch():
    # 1. SETUP DIRECTORIES
    input_dir = r"C:\Users\dhars\Desktop\inputpens"
    output_dir = r"C:\Users\dhars\Desktop\outputpens"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. TUNING PARAMETERS
    # Lowered slightly to 2500 to ensure shadowed pens aren't discarded
    MIN_AREA_THRESHOLD = 2500 

    # 3. GET IMAGES
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_files:
        print("No images found in the input folder.")
        return

    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None: continue

        # 4. ROBUST DETECTION LOGIC
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Gentle blur to keep edge definition while removing grain
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # ADAPTIVE SETTINGS FOR SHADOWS:
        # Block Size increased to 31: Sees more 'context' around the pen to find edges in shadows.
        # Constant C increased to 4: Makes it more sensitive to subtle local changes.
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 31, 4)

        # 5. THE "BRIDGE" LOGIC (Closing the Gaps)
        # Use a larger kernel specifically to stitch the cap back to the body
        kernel_clean = np.ones((3,3), np.uint8)
        kernel_bridge = np.ones((7,7), np.uint8)
        
        # Remove tiny specks first
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)
        # Strong Dilate/Close to connect fragmented pen parts in shadows
        mask = cv2.dilate(mask, kernel_bridge, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_bridge)

        # 6. FIND AND ITERATE THROUGH EVERY PEN
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pen_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if area > MIN_AREA_THRESHOLD:
                pen_count += 1
                x, y, w, h = cv2.boundingRect(cnt)

                # Create a specific mask for ONLY this pen
                single_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(single_mask, [cnt], -1, 255, -1)

                # Extraction (Bitwise AND)
                clean_cutout = cv2.bitwise_and(img, img, mask=single_mask)
                pen_crop = clean_cutout[y:y+h, x:x+w]

                # 7. RESIZE & LETTERBOX (224x224)
                size = 224
                max_dim = max(h, w)
                if max_dim == 0: continue
                
                scale = size / max_dim
                nw, nh = int(w * scale), int(h * scale)
                resized = cv2.resize(pen_crop, (nw, nh), interpolation=cv2.INTER_AREA)
                
                final_canvas = np.zeros((size, size, 3), dtype=np.uint8)
                y_off, x_off = (size - nh) // 2, (size - nw) // 2
                final_canvas[y_off:y_off+nh, x_off:x_off+nw] = resized

                # 8. SAVE
                output_filename = f"{base_name}_pen_{pen_count}.jpg"
                cv2.imwrite(os.path.join(output_dir, output_filename), final_canvas)

        print(f"File: {os.path.basename(img_path)} | Detected: {pen_count} pens.")

    print(f"\nProcessing complete! Check your output folder.")

if _name_ == "_main_":
    process_robust_batch