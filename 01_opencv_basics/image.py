import os
import cv2


path = 'C:\\Users\\shard\\Downloads'

img_path = os.path.join(path, 'bird.jpeg')
img = cv2.imread(img_path)

cv2.imshow('bird', img)

cv2.waitKey()