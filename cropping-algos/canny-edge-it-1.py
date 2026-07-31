import cv2
import numpy as np
import os
import glob

def process_batch_pens():
    # 1. SETUP DIRECTORIES
    input_dir = r"C:\Users\dhars\Desktop\inputpens"
    output_dir = r"C:\Users\dhars\Desktop\outputpens"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. DEFINE PIXEL THRESHOLD
    MIN_AREA_THRESHOLD = 2000 

    # 3. GET ALL IMAGES
    image_extensions = ['*.jpg', ['*.jpeg'], ['*.png'], ['*.webp']]
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

        # 4. FUSION DETECTION LOGIC
        # --- A. Color Detection (Shadow Resistant) ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Hauser Blue range (Hue 90-130 is typically Blue)
        lower_blue = np.array([90, 50, 50]) 
        upper_blue = np.array([130, 255, 255])
        color_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # --- B. Edge Detection (Structure) ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 20, 80) # Lowered thresholds to catch faint edges

        # --- C. Merge Both (Fusion) ---
        # This combines the 'skeleton' from Canny with the 'color' of the pen
        combined_mask = cv2.bitwise_or(edged, color_mask)

        # 5. SOLIDIFY THE MASK
        kernel = np.ones((7,7), np.uint8)
        mask = cv2.dilate(combined_mask, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 6. FIND ALL BLOBS
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pen_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if area > MIN_AREA_THRESHOLD:
                pen_count += 1
                x, y, w, h = cv2.boundingRect(cnt)

                # Create specific solid mask for this pen
                single_pen_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(single_pen_mask, [cnt], -1, 255, -1)

                # Use the mask to grab ONLY the pen pixels from the original color image
                clean_cutout = cv2.bitwise_and(img, img, mask=single_pen_mask)
                pen_crop = clean_cutout[y:y+h, x:x+w]

                # 7. RESIZE & LETTERBOX (224x224)
                size = 224
                # Added a small check to prevent division by zero
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
                save_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(save_path, final_canvas)

        print(f"Processed {os.path.basename(img_path)}: Found {pen_count} pens (Shadows bypassed).")

    print(f"\nTask Complete! Check output folder.")

if __name__ == "__main__":
    process_batch_pens()


