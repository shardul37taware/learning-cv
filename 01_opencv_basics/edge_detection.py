import os
import cv2
import numpy as np

path = 'C:\\Users\\shard\\Downloads'
image_path = os.path.join(path, "city.jpg")

img = cv2.imread(image_path)

img_canny = cv2.Canny(img, 100, 300)
img_dilate = cv2.dilate(img_canny, np.ones((5,5), dtype=np.int8))
img_erode = cv2.erode(img_canny, np.ones((5,5), dtype=np.int8))


cv2.imshow("image", img)

cv2.imshow("canny edge detection", img_canny)
cv2.imshow("dialated", img_dilate)
cv2.imshow("eroded", img_erode)

cv2.waitKey()