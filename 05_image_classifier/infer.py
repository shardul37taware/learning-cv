from img2vec_pytorch import Img2Vec
from PIL import Image
import pickle
import cv2

with open('D:/git/learning-cv/05_image_classifier/model.p', 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()

# img_path = 'D:/sst/disaster/dataset/result/test/damage/damage (256).png'

# img = Image.open(img_path)

# features = img2vec.get_vec(img)

# pred = model.predict([features])

# print(pred)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    features = img2vec.get_vec(frame)
    pred = model.predict([features])
    


    # Display
    cv2.putText(frame, f"{pred})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Disaster Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()