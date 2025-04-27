import os
import cv2

path = 'C:\\Users\\shard\\Downloads'
image_path = os.path.join(path, "birds.jpg")

print(cv2.COLOR_BGR2GRAY)

img = cv2.imread(image_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

i = 0
for cnt in contours:
    print(cv2.contourArea(cnt))

    if cv2.contourArea(cnt) >= 100:

        cv2.drawContours(img, cnt, -1, (0, 255, 0  ), 2)
        
        x1, y1, w, h = cv2.boundingRect(cnt)

        cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)

        i+=1
print("\n",i)

cv2.imshow("image", img)

cv2.imshow("thershold", thresh)


cv2.waitKey()