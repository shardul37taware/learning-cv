import cv2
import numpy as np
import os
import time

# ====== CONFIGURATION ======
INPUT_FOLDER = r"D:\ODM\odm_data_langley-master\odm_data_langley-master\images"
OUTPUT_ORTHO = r".D:\ODM\odm_data_langley-master\odm_data_langley-master\orthophoto_result.jpg"
TEMP_FOLDER = "./ortho_temp"
LOG_FILE = "./processing_log.txt"

# Performance/Stability Settings
BATCH_SIZE = 5
MAX_IMAGE_DIM = 800
MAX_OUTPUT_DIM = 30000
SAVE_INTERVAL = 2
USE_GPU = False
MIN_MATCHES = 25

# ====== INITIALIZATION ======
os.makedirs(TEMP_FOLDER, exist_ok=True)
open(LOG_FILE, 'w').close()

def log_message(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

def safe_resize(img):
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = min(MAX_IMAGE_DIM/max(h,w), 1.0)
    return cv2.resize(img, (int(w*scale), int(h*scale)))

def enforce_max_dim(image):
    if image is None:
        return None
    h, w = image.shape[:2]
    if w > MAX_OUTPUT_DIM or h > MAX_OUTPUT_DIM:
        scale = min(MAX_OUTPUT_DIM/w, MAX_OUTPUT_DIM/h)
        return cv2.resize(image, (int(w*scale), int(h*scale)))
    return image

def safe_process():
    try:
        log_message("Starting processing")
        image_paths = [os.path.join(INPUT_FOLDER, f) 
                      for f in sorted(os.listdir(INPUT_FOLDER))
                      if f.lower().endswith(('.jpg','.png','.jpeg'))]
        
        if not image_paths:
            raise ValueError("No valid images found")

        result = None
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        for batch_num, i in enumerate(range(0, len(image_paths), BATCH_SIZE)):
            batch_paths = image_paths[i:i+BATCH_SIZE]
            log_message(f"Processing batch {batch_num+1} (images {i+1}-{i+len(batch_paths)})")

            batch = []
            for path in batch_paths:
                img = safe_resize(cv2.imread(path))
                if img is not None:
                    batch.append(img)
            
            if len(batch) < 2:
                log_message("Warning: Insufficient images in batch - skipping")
                continue

            try:
                status, panorama = stitcher.stitch(batch)
                if status != cv2.Stitcher_OK:
                    log_message(f"Batch stitching failed (status {status}), trying pairwise...")
                    panorama = batch[0]
                    for img in batch[1:]:
                        status, pano = stitcher.stitch([panorama, img])
                        if status == cv2.Stitcher_OK:
                            panorama = enforce_max_dim(pano)
                        else:
                            log_message("Pairwise stitching failed - skipping image")
                            break
                panorama = enforce_max_dim(panorama)
                if status == cv2.Stitcher_OK:
                    if result is None:
                        result = panorama
                    else:
                        if (result.shape[1] + panorama.shape[1]) > MAX_OUTPUT_DIM or \
                           (result.shape[0] + panorama.shape[0]) > MAX_OUTPUT_DIM:
                            log_message("Merging would exceed max dimensions - saving separately")
                            temp_path = os.path.join(TEMP_FOLDER, f"segment_{batch_num}.jpg")
                            cv2.imwrite(temp_path, result)
                            result = panorama
                        else:
                            status, result = stitcher.stitch([result, panorama])
                            if status != cv2.Stitcher_OK:
                                log_message("Merge failed - saving as new segment")
                                temp_path = os.path.join(TEMP_FOLDER, f"segment_{batch_num}.jpg")
                                cv2.imwrite(temp_path, result)
                                result = panorama

            except cv2.error as e:
                log_message(f"OpenCV error: {str(e)}")
                continue

            if batch_num % SAVE_INTERVAL == 0 and result is not None:
                temp_path = os.path.join(TEMP_FOLDER, f"partial_{batch_num}.jpg")
                cv2.imwrite(temp_path, result)
                log_message(f"Saved temporary result to {temp_path}")

        if result is not None:
            result = enforce_max_dim(result)
            cv2.imwrite(OUTPUT_ORTHO, result)
            log_message(f"Successfully saved final result to {OUTPUT_ORTHO}")
            return True
        else:
            raise RuntimeError("Processing completed but no result was generated")

    except Exception as e:
        log_message(f"CRITICAL ERROR: {str(e)}")
        log_message("Attempting to save partial results...")
        try:
            if 'result' in locals() and result is not None:
                emergency_path = "./partial_result_emergency.jpg"
                cv2.imwrite(emergency_path, result)
                log_message(f"Saved partial result to {emergency_path}")
        except:
            log_message("Failed to save partial result")
        return False

if safe_process():
    if os.path.exists(OUTPUT_ORTHO):
        log_message("\nFinal Verification:")
        log_message(f"Size: {os.path.getsize(OUTPUT_ORTHO)/1024/1024:.2f} MB")
        log_message(f"Dimensions: {cv2.imread(OUTPUT_ORTHO).shape}")
        log_message(f"Ortho image saved at: {OUTPUT_ORTHO}")
    else:
        log_message("Error: Final output file not found")
else:
    log_message("Processing failed - partial results available in ./ortho_temp")

print("\n=== PROCESSING LOG ===")
with open(LOG_FILE, 'r') as f:
    print(f.read())

print("\n=== PARTIAL RESULTS ===")
for filename in os.listdir(TEMP_FOLDER):
    filepath = os.path.join(TEMP_FOLDER, filename)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"{filename} - {size_kb:.1f} KB")
