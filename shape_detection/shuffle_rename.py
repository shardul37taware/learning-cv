import os
import random
from PIL import Image

def shuffle_and_rename_images(folder_path):
    # Get all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Filter for image files (you can add more extensions if needed)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    images = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
    
    # Shuffle the images
    random.shuffle(images)
    
    # Get the folder name
    folder_name = os.path.basename(os.path.normpath(folder_path))
    
    # Rename each image
    for i, image in enumerate(images, start=1):
        old_path = os.path.join(folder_path, image)
        ext = os.path.splitext(image)[1]
        new_name = f"{folder_name}_{i}{ext}"
        new_path = os.path.join(folder_path, new_name)
        
        # Handle potential name conflicts
        while os.path.exists(new_path):
            i += 1
            new_name = f"{folder_name}_{i}{ext}"
            new_path = os.path.join(folder_path, new_name)
        
        os.rename(old_path, new_path)
        print(f"Renamed {image} to {new_name}")

# List of your folders (replace with your actual folder paths)
folders = [
    "path/to/folder1",
    "path/to/folder2",
    "path/to/folder3"
]

for folder in folders:
    if os.path.isdir(folder):
        print(f"\nProcessing folder: {folder}")
        shuffle_and_rename_images(folder)
    else:
        print(f"Folder not found: {folder}")

print("\nAll folders processed!")