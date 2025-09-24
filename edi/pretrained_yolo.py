from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')



vid = cv2.VideoCapture(r"D:\git\learning-cv\edi\stock-footage-cars-parking-and-leaving-cctv-feed.webm")

ret = True
while ret:
    ret, frame = vid.read()

    if ret:
        results = model(frame, classes=[1, 2, 3, 5, 7])

        for result in results:
            boxes = result.boxes.xyxy
            confs = result.boxes.conf
            classIds = result.boxes.cls

            for box, conf, classId in zip(boxes, confs, classIds):
                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow("video", frame)
        cv2.waitKey(40)

vid.release()
cv2.destroyAllWindows
