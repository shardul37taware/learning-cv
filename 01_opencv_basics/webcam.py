import cv2

webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()

    cv2.imshow('webcam', frame)
    if cv2.waitKey(40) & 0xFF = ord(' '):
        break

webcam.release()
cv2.destroyAllWindows()