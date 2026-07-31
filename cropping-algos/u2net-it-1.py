import os
import time
from rembg import remove, new_session
from PIL import Image, ImageChops

# --- CONFIG ---
input_folder = r"C:\Users\dhars\Desktop\pen1"
output_folder = r"C:\Users\dhars\Desktop\croppenresult"
if not os.path.exists(output_folder): os.makedirs(output_folder)

# Load LITE model and explicitly set threads for speed
print("Warming up engine...")
session = new_session("u2netp") 

def trim(im):
    bg = Image.new(im.mode, im.size, (0,0,0,0))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    return im.crop(bbox) if bbox else im

# --- PROCESS ---
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        t0 = time.time()
        
        # 1. Open and SCALE DOWN immediately
        # Higher than 640 is wasted for u2netp; lower is even faster.
        img = Image.open(os.path.join(input_folder, filename))
        img.thumbnail((640, 640), Image.Resampling.LANCZOS) 
        
        # 2. AI Processing (Alpha matting disabled by default)
        no_bg = remove(img, session=session)
        
        # 3. Trim and Save
        final = trim(no_bg)
        final.save(os.path.join(output_folder, filename.rsplit('.', 1)[0] + '.png'))
        
        print(f"⚡ {filename}: {int((time.time() - t0) * 1000)}ms")
