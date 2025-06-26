import os
import random
import shutil
from tqdm import tqdm  # for progress bar (install with pip install tqdm)

def process_folder(input_folder, output_root):
    # Get all image files in the input folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    images = [f for f in os.listdir(input_folder) 
              if os.path.isfile(os.path.join(input_folder, f)) 
              and os.path.splitext(f)[1].lower() in image_extensions]
    
    # Shuffle the images
    random.shuffle(images)
    
    # Calculate split sizes
    total = len(images)
    train_count = int(0.7 * total)
    val_count = int(0.15 * total)
    test_count = total - train_count - val_count  # remainder goes to test
    
    # Create splits
    splits = {
        'train': images[:train_count],
        'val': images[train_count:train_count+val_count],
        'test': images[train_count+val_count:]
    }
    
    # Get folder name for naming
    folder_name = os.path.basename(os.path.normpath(input_folder))
    
    # Process each split
    for split_name, split_images in splits.items():
        # Create output directory structure
        output_dir = os.path.join(output_root, split_name, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy and rename files
        for i, image in enumerate(tqdm(split_images, desc=f"{folder_name} {split_name}"), 1):
            ext = os.path.splitext(image)[1]
            new_name = f"{folder_name}_{i}{ext}"
            src_path = os.path.join(input_folder, image)
            dst_path = os.path.join(output_dir, new_name)
            
            shutil.copy2(src_path, dst_path)  # copy2 preserves metadata

def main():
    # Configuration
    input_folders = [
        r"C:\Users\shard\Downloads\shapes\Triangle",
        r"C:\Users\shard\Downloads\shapes\Square",
        r"C:\Users\shard\Downloads\shapes\Circle"
    ]
    output_root = r"C:\Users\shard\Downloads\Shapes"  # Where to create train/val/test folders
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_root, split), exist_ok=True)
    
    # Process each input folder
    for folder in input_folders:
        if os.path.isdir(folder):
            print(f"\nProcessing: {folder}")
            process_folder(folder, output_root)
        else:
            print(f"Folder not found: {folder}")
    
    print("\nAll processing complete!")

if __name__ == "__main__":
    main()