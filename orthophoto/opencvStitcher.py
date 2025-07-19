import cv2
import numpy as np
import os
from tqdm import tqdm

# ====== SETTINGS ======
INPUT_FOLDER = r"D:\ODM\odm_data_langley-master\odm_data_langley-master\images"
OUTPUT_ORTHO = r"D:\ODM\odm_data_langley-master\odm_data_langley-master\orthophoto_result.jpg"
BATCH_SIZE = 10  # Smaller batches for stability
RESIZE_FACTOR = 0.5  # Balanced resolution/RAM
FEATURE_METHOD = "SIFT"  # Better for aerial images

# ====== FUNCTIONS ======
def load_images(folder):
    """Load images with consistent dimensions."""
    filenames = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    if not filenames:
        raise ValueError("No images found in folder!")
    
    # Get base size from first image
    base_img = cv2.imread(os.path.join(folder, filenames[0]))
    if base_img is None:
        raise ValueError(f"Couldn't read {filenames[0]}")
    
    h, w = base_img.shape[:2]
    new_h, new_w = int(h * RESIZE_FACTOR), int(w * RESIZE_FACTOR)
    
    images = []
    for filename in tqdm(filenames, desc="Loading images"):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (new_w, new_h))
            images.append(img)
    return images

def align_and_blend(batch):
    """Align and blend images with dimension checks."""
    if len(batch) < 2:
        return None
    
    # Initialize with first image's dimensions
    ref_img = batch[0]
    output_h, output_w = ref_img.shape[0]*2, ref_img.shape[1]*2
    blended = np.zeros((output_h, output_w, 3), dtype=np.float32)
    mask = np.zeros((output_h, output_w), dtype=np.float32)
    
    # Process each image
    for img in batch:
        # Skip if dimensions mismatch (safety check)
        if img.shape != ref_img.shape:
            img = cv2.resize(img, (ref_img.shape[1], ref_img.shape[0]))
        
        # Simple averaging (replace with proper homography for real use)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_mask = cv2.threshold(img_gray, 1, 1, cv2.THRESH_BINARY)
        
        # Resize to output dimensions if needed
        if img.shape[0] != output_h or img.shape[1] != output_w:
            img = cv2.resize(img, (output_w, output_h))
            img_mask = cv2.resize(img_mask, (output_w, output_h))
        
        blended += img.astype(np.float32) * img_mask[:, :, np.newaxis]
        mask += img_mask
    
    # Normalize
    mask[mask == 0] = 1  # Avoid division by zero
    result = (blended / mask[:, :, np.newaxis]).astype(np.uint8)
    return result

# ====== MAIN PIPELINE ======
if __name__ == "__main__":
    print("Loading images...")
    try:
        images = load_images(INPUT_FOLDER)
        if not images:
            raise ValueError("No valid images loaded!")
        
        print(f"Processing {len(images)} images...")
        final_ortho = None
        
        for i in tqdm(range(0, len(images), BATCH_SIZE), desc="Processing batches"):
            batch = images[i:i+BATCH_SIZE]
            blended_batch = align_and_blend(batch)
            
            if blended_batch is None:
                print(f"Skipping batch {i//BATCH_SIZE} (alignment failed)")
                continue
                
            if final_ortho is None:
                final_ortho = blended_batch
            else:
                # Simple concatenation (replace with proper stitching)
                final_ortho = np.concatenate((final_ortho, blended_batch), axis=1)
        
        if final_ortho is not None:
            cv2.imwrite(OUTPUT_ORTHO, final_ortho)
            print(f"\nOrthophoto saved to {OUTPUT_ORTHO}")
        else:
            print("Failed to create orthophoto")
            
    except Exception as e:
        print(f"Error: {str(e)}")