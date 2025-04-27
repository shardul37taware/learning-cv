import os
import cv2


path = 'C:\\Users\\shard\\Downloads'
video_path = os.path.join(path, 'monkey.mp4')

vid = cv2.VideoCapture(video_path)

ret = True
while ret:
    ret, frame = vid.read()

    if ret:
        cv2.imshow("monkey", frame)
        cv2.waitKey(40)

vid.release()
cv2.destroyAllWindows