import os
import cv2

path = 'C:\\Users\\shard\\Downloads'
image_path = os.path.join(path, "city.jpg")

img = cv2.imread(image_path)

cv2.imshow("image", img)

cv2.waitKey()