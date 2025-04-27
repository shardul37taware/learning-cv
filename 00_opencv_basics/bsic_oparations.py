import os
import cv2

path = 'C:\\Users\\shard\\Downloads'
image_path = os.path.join(path, "bird.jpeg")

img = cv2.imread(image_path)

# res_img = cv2.resize(img, (336, 550))

print(img.shape)
# print(res_img.shape)

cv2.imshow("img", img)
# cv2.imshow('resixed img', res_img)

cropped_img = img[75:184 ,20:175]
cv2.imshow("cropped image", cropped_img)
cv2.waitKey()  