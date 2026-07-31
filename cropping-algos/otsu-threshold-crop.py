import cv2
import numpy as np
import os

input_folder = r"C:\Users\dhars\Desktop\pen"
output_folder = r"C:\Users\dhars\Desktop\pencrop"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(input_folder, filename)
        frame = cv2.imread(img_path)
        if frame is None: continue

        # 1. Blur slightly to remove 'salt and pepper' noise from off-white textures
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # 2. Convert to Grayscale
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # 3. OTSU'S THRESHOLDING (Automatically finds the 'off-white' cut-off)
        # THRESH_BINARY_INV + THRESH_OTSU will make the pen white and background black
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 4. Clean up the mask (fill holes in the pen, remove tiny dots in background)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2) 

        # 5. Find the Pen
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Sort by area and pick the largest (the pen)
            cnt = max(contours, key=cv2.contourArea)
            
            # Use a tiny area check—pens are thin, so 100 pixels is usually safe
            if cv2.contourArea(cnt) > 100: 
                x, y, w, h = cv2.boundingRect(cnt)
                
                # 6. Create BGRA (Transparency)
                b, g, r = cv2.split(frame)
                rgba = cv2.merge([b, g, r, mask])
                
                # 7. Crop to the box
                crop = rgba[y:y+h, x:x+w]
                
                output_path = os.path.join(output_folder, f"cropped_{filename.split('.')[0]}.png")
                cv2.imwrite(output_path, crop)
                print(f"SUCCESS: {filename} cropped to {w}x{h}")
            else:
                print(f"SKIP: {filename} - object too small (check area threshold).")
        else:
            print(f"FAIL: {filename} - No object found. Your background might be too dark.")

print("\nAll done. Check the 'cropped_objects' folder now.")
