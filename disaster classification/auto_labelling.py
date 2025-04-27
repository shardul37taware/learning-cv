import os
import shutil

# Define your dataset root and class names
dataset_path = "D:/sst/disaster"
classes = ['fire', 'flood', 'damage']  # Order defines class IDs

# Output folders
output_img_path = "D:/sst/disaster/output/images"
output_lbl_path = "D:/sst/disaster/output/labels"
os.makedirs(output_img_path, exist_ok=True)
os.makedirs(output_lbl_path, exist_ok=True)

for class_id, class_name in enumerate(classes):
    class_folder = os.path.join(dataset_path, class_name)
    for img_file in os.listdir(class_folder):
        if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_name = os.path.splitext(img_file)[0]
            
            # Copy image
            src = os.path.join(class_folder, img_file)
            dst = os.path.join(output_img_path, img_file)
            shutil.copy(src, dst)
            
            # Create YOLO label with bounding box covering the whole image
            label_file = os.path.join(output_lbl_path, f"{img_name}.txt")
            with open(label_file, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
