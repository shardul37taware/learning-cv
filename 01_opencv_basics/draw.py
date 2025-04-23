import os
import cv2

path = 'C:\\Users\\shard\\Downloads'
image_path = os.path.join(path, "city.jpg")

img = cv2.imread(image_path)

print(img.shape)

cv2.line(img, (100, 100), (200, 50), (0, 0, 255), 3)
cv2.rectangle(img, (250, 200), (300, 350), (255, 255, 255), 5)
cv2.circle(img, (350, 200), 100, (255, 0, 0), 7)
cv2.putText(img, "Hi There!", (350, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 10)

cv2.imshow("image", img)

cv2.waitKey()