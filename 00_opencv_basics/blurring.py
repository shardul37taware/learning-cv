import os
import cv2

path = 'C:\\Users\\shard\\Downloads'
image_path = os.path.join(path, "city.jpg")

img = cv2.imread(r"C:\Users\shardul\Pictures\Untitled design (1).png")

img_blur = cv2.blur(img, (3, 3))
img_gaussian = cv2.GaussianBlur(img, (3,3), 3)
img_median = cv2.medianBlur(img, 3)

# for i in range(50, 0, -1):
#     print(i)
#     cv2.imshow("video", cv2.blur(img, (i,i)))
#     cv2.waitKey(40)

cv2.imshow("image", img)

cv2.imshow("blur image", img_blur)
cv2.imshow("gaussian blur image", img_gaussian)
cv2.imshow("median blur image", img_median)

cv2.waitKey()