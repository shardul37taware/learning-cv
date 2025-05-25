import cv2

webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()

    cv2.imshow('webcam', frame)
webcam.release()
cv2.destroyAllWindows()