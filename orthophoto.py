import cv2
import os
import glob

def stitch_images_from_folder(folder_path, output_path='stitched_output.jpg', max_dim=800, show_preview=True):
    # Get all image paths (supports jpg/jpeg/png)
    image_paths = sorted(glob.glob(os.path.join(folder_path, '*.jpg')) +
                         glob.glob(os.path.join(folder_path, '*.jpeg')) +
                         glob.glob(os.path.join(folder_path, '*.png')))

    if len(image_paths) < 2:
        print("Need at least two images to stitch.")
        return

    print(f"[INFO] Found {len(image_paths)} images. Resizing and stitching...")

    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
        images.append(img_resized)

    print(f"[INFO] Resized images. Starting stitching...")

    stitcher = cv2.Stitcher_create() if int(cv2.__version__.split(".")[0]) >= 4 else cv2.createStitcher()
    status, stitched = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        print(f"[SUCCESS] Stitching completed. Saving to {output_path}")
        cv2.imwrite(output_path, stitched)

        if show_preview:
            cv2.imshow("Stitched Image", stitched)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print(f"[ERROR] Stitching failed with status code {status}.")

# Example usage
if __name__ == "__main__":
    folder_path = "D:\ODM\odm_boruszyn_kap-master\odm_boruszyn_kap-master\ images"  # Replace this
    stitch_images_from_folder(folder_path, output_path="stitched_output.jpg", max_dim=800)
