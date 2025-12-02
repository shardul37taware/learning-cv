import cv2
from PIL import Image
import numpy as np
from util import get_limits

# Select target color in BGR
target_color = [0, 255, 255]  # Yellow

cap = cv2.VideoCapture(0)

# ROI setup
ret, frame = cap.read()
H, W, _ = frame.shape
roi_width = W // 3
roi_x1 = (W - roi_width) // 2
roi_x2 = roi_x1 + roi_width
roi_center = ((roi_x1 + roi_x2) // 2, H // 2)

# Tracker setup
tracker = None
tracking = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerLimit, upperLimit = get_limits(color=target_color)
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    # Remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    mask_ = Image.fromarray(mask)
    bbox = mask_.getbbox()

    if not tracking:
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1

            # ignore very small blobs
            if w > 25 and h > 25:
                obj_center = (x1 + w // 2, y1 + h // 2)

                # start tracking only if outside ROI
                if not (roi_x1 <= obj_center[0] <= roi_x2):
                    try:
                        tracker = cv2.legacy.TrackerCSRT_create()
                        tracker.init(frame, (x1, y1, w, h))
                        tracking = True
                        print("Target locked!")
                    except:
                        print("Failed to initialize tracker")

    else:
        success, box = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            obj_center = (x + w // 2, y + h // 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.line(frame, roi_center, obj_center, (255, 255, 255), 2)

            error_x = obj_center[0] - roi_center[0]
            error_y = obj_center[1] - roi_center[1]
            print("Error:", error_x)

        else:
            tracking = False
            tracker = None
            print("Target lost")

    # Draw ROI box
    cv2.rectangle(frame, (roi_x1, 0), (roi_x2, H), (255, 0, 0), 2)

    cv2.imshow("Tracking", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):  # unlock and reacquire
        tracking = False
        tracker = None
        print("Manual unlock")

cap.release()
cv2.destroyAllWindows()
