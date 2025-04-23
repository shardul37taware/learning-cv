import os
import cv2

path = 'C:\\Users\\shard\\Downloads'
image_path = os.path.join(path, "city.jpg")

img = cv2.imread(image_path)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img_thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)

img_blur = cv2.blur(img_gray, (3,3))
ret, img_thresh2 = cv2.threshold(img_blur, 50, 255, cv2.THRESH_BINARY)

thresh_adapt = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 30)

cv2.imshow("image", img)

cv2.imshow("grayscale image", img_gray)
cv2.imshow("thresholded image", img_thresh)
cv2.imshow("thresholded image 2", img_thresh2)
cv2.imshow("adaptive threshold", thresh_adapt)

cv2.waitKey()