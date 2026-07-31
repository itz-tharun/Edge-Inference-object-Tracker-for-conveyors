import cv2
import numpy as np
import os

input_folder = 'input_objects'
output_folder = 'cropped_objects'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(input_folder, filename)
        frame = cv2.imread(img_path)
        if frame is None: continue

        # 1. Pre-processing: Blur to handle off-white background noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # 2. Otsu's Thresholding: Auto-calculates the best cut-off for off-white
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3. Clean the mask (fills internal holes in the pen/object)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2) 

        # 4. Find all contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # CONSTRAINT: Find the object with the HIGHEST PIXEL COUNT (Max Area)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Additional check: ensure even the largest isn't just a tiny speck
            if cv2.contourArea(largest_contour) > 100: 
                # Get the bounding box of ONLY the largest object
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # 5. Create Transparency (BGRA)
                # We create a specific mask for ONLY the largest object to avoid 
                # 'ghosts' of other pens appearing in the transparent areas.
                final_mask = np.zeros_like(mask)
                cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                
                b, g, r = cv2.split(frame)
                rgba = cv2.merge([b, g, r, final_mask])
                
                # 6. Perform the Crop
                crop = rgba[y:y+h, x:x+w]
                
                output_path = os.path.join(output_folder, f"cropped_{filename.split('.')[0]}.png")
                cv2.imwrite(output_path, crop)
                print(f"DONE: Cropped the largest object in {filename} ({w}x{h} px)")
            else:
                print(f"SKIP: No significant object found in {filename}")
        else:
            print(f"FAIL: Could not find any contours in {filename}")

print("\nTask complete. Only the largest objects have been saved.")
