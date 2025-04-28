import numpy as np
import cv2
from PIL import Image

def getLimits(colour):

    c = np.uint8([[colour]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]

    if hue >= 165:
        lower = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upper = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:
        lower = np.array([0, 100, 100], dtype=np.uint8)
        upper = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lower = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upper = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lower, upper

colour = [0, 255, 0]      #yellow in bgr coloursapce

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit, upperLimit = getLimits(colour)

    mask = cv2.inRange(hsv, lowerLimit, upperLimit)
    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imshow("Colour Tracker", frame )

    if cv2.waitKey(1) & 0xff == ord(' '):
        break


cap.release()
cv2.destroyAllWindows