import os
import time
import cv2
import numpy as np
from rembg import remove, new_session
from concurrent.futures import ThreadPoolExecutor

# --- CONFIG ---
input_folder = r"C:\Users\dhars\Desktop\pen1"
output_folder = r"C:\Users\dhars\Desktop\croppenresult"
if not os.path.exists(output_folder): os.makedirs(output_folder)

# Load the ABSOLUTE fastest model available
# 'isnet-general-use' or 'u2netp' with high threads
session = new_session("u2netp") 

def process_image(filename):
    t0 = time.time()
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename.rsplit('.', 1)[0] + '.png')
    
    # 1. FAST LOAD with OpenCV
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Extreme Downscale (320x320 is the AI's native size)
    # This removes 99% of the CPU work
    img_small = cv2.resize(img, (320, 320))
    
    # 3. AI Inference
    res = remove(img_small, session=session)
    
    # 4. Save (Async)
    cv2.imwrite(output_path, cv2.cvtColor(res, cv2.COLOR_RGBA2BGRA))
    
    duration = (time.time() - t0) * 1000
    return duration

# --- MULTI-THREADED EXECUTION ---
files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))]
print(f"Blasting through {len(files)} images...")

with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_image, files))

avg = sum(results) / len(results)
print(f"\n🚀 Average Speed: {round(avg, 2)}ms")
print(f"Approximate FPS: {round(1000/avg, 2)}")