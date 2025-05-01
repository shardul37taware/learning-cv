import os
import shutil
import random

def split_dataset(
    source_dir,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    random.seed(seed)
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for cls in classes:
        class_dir = os.path.join(source_dir, cls)
        files = os.listdir(class_dir)
        random.shuffle(files)

        train_end = int(len(files) * train_ratio)
        val_end = train_end + int(len(files) * val_ratio)

        splits = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:]
        }

        for split_name, split_files in splits.items():
            split_class_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(split_class_dir, exist_ok=True)

            for file in split_files:
                src = os.path.join(class_dir, file)
                dst = os.path.join(split_class_dir, file)
                shutil.copy2(src, dst)

    print("Dataset split completed.")

# Usage
split_dataset(
    source_dir="D:/sst/disaster/dataset raw",
    output_dir="D:/sst/disaster/dataset/result",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
