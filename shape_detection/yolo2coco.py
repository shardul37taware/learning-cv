import os
import json
import yaml
from glob import glob
from PIL import Image

def convert_yolo_to_coco_from_yaml(yaml_path, split='train'):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    class_names = data['names']
    num_classes = data['nc']
    
    image_dir = os.path.abspath(os.path.join(os.path.dirname(yaml_path), data[split]))
    label_dir = image_dir.replace('/images', '/labels')

    images = []
    annotations = []
    categories = []
    ann_id = 0
    img_id = 0

    for idx, class_name in enumerate(class_names):
        categories.append({"id": idx, "name": class_name})

    for img_path in sorted(glob(f"{image_dir}/*.jpg")):
        img_id += 1
        img = Image.open(img_path)
        width, height = img.size
        file_name = os.path.basename(img_path)

        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        label_path = os.path.join(label_dir, file_name.replace('.jpg', '.txt'))
        if not os.path.exists(label_path):
            continue

        with open(label_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, w, h = map(float, line.strip().split())
                x = (x_center - w / 2) * width
                y = (y_center - h / 2) * height
                w *= width
                h *= height

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(class_id),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    output_path = os.path.join(os.path.dirname(yaml_path), f"instances_{split}.json")
    with open(output_path, 'w') as f:
        json.dump(coco_dict, f, indent=4)
    
    print(f"Saved COCO annotations: {output_path}")

# 🔁 Convert your train/val sets
convert_yolo_to_coco_from_yaml("path/to/data.yaml", split="train")
convert_yolo_to_coco_from_yaml("path/to/data.yaml", split="val")
