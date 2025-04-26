import os
import shutil
import random

# Define dataset path and class labels
dataset_path = "D:/sst/disaster"
classes = ['fire', 'flood', 'damage']  # Order defines class IDs

# Output folders
output_img_path = "D:/sst/disaster/output/images"
output_lbl_path = "D:/sst/disaster/output/labels"
os.makedirs(output_img_path, exist_ok=True)
os.makedirs(output_lbl_path, exist_ok=True)

# Collect all image paths with their class IDs
all_images = []

for class_id, class_name in enumerate(classes):
    class_folder = os.path.join(dataset_path, class_name)
    for img_file in os.listdir(class_folder):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(class_folder, img_file)
            all_images.append((img_path, class_id))

# Shuffle the list
random.shuffle(all_images)

# Rename, copy, and create label files
for idx, (img_path, class_id) in enumerate(all_images, start=1):
    new_img_name = f"{idx}.png"
    img_name_without_ext = str(idx)

    # Copy and rename image
    dst_img_path = os.path.join(output_img_path, new_img_name)
    shutil.copy(img_path, dst_img_path)

    # Write corresponding label file
    label_path = os.path.join(output_lbl_path, f"{img_name_without_ext}.txt")
    with open(label_path, 'w') as f:
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")