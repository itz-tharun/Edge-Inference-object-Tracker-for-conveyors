import cv2
import numpy as np
import os
import glob

def process_robust_transparent():
    # 1. SETUP DIRECTORIES
    input_dir = r"C:\Users\dhars\Desktop\newshi"
    output_dir = r"C:\Users\dhars\Desktop\op"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. TUNING PARAMETERS
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

        # 4. ROBUST DETECTION LOGIC (Same as your working code)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 31, 4)

        # 5. THE "BRIDGE" LOGIC
        kernel_clean = np.ones((3,3), np.uint8)
        kernel_bridge = np.ones((7,7), np.uint8)
        
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)
        mask = cv2.dilate(mask, kernel_bridge, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_bridge)

        # 6. FIND AND ITERATE
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pen_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_AREA_THRESHOLD:
                pen_count += 1
                x, y, w, h = cv2.boundingRect(cnt)

                # --- NEW TRANSPARENCY LOGIC ---
                # Create the alpha mask specifically for this pen
                alpha_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.drawContours(alpha_mask, [cnt], -1, 255, -1)

                # Crop both the image and the alpha mask
                pen_crop = img[y:y+h, x:x+w]
                alpha_crop = alpha_mask[y:y+h, x:x+w]

                # Combine BGR channels with the Alpha channel
                b, g, r = cv2.split(pen_crop)
                bgra = cv2.merge([b, g, r, alpha_crop])

                # 7. RESIZE & LETTERBOX (224x224)
                size = 224
                max_dim = max(h, w)
                if max_dim == 0: continue
                
                scale = size / max_dim
                nw, nh = int(w * scale), int(h * scale)
                
                # Resize the 4-channel image
                resized_bgra = cv2.resize(bgra, (nw, nh), interpolation=cv2.INTER_AREA)
                
                # Create a 4-channel transparent canvas (0,0,0,0)
                # The 4th zero means 100% transparent background
                final_canvas = np.zeros((size, size, 4), dtype=np.uint8)
                
                y_off, x_off = (size - nh) // 2, (size - nw) // 2
                final_canvas[y_off:y_off+nh, x_off:x_off+nw] = resized_bgra

                # 8. SAVE AS PNG (Mandatory for transparency)
                output_filename = f"{base_name}_pen_{pen_count}.png"
                cv2.imwrite(os.path.join(output_dir, output_filename), final_canvas)

        print(f"File: {os.path.basename(img_path)} | Created {pen_count} transparent PNGs.")

    print(f"\nProcessing complete! PNGs saved to: {output_dir}")

if __name__ == "__main__":
    process_robust_transparent()