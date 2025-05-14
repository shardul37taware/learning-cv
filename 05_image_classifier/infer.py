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

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR (OpenCV) to RGB (PIL)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    pil_img = pil_img.resize((224, 224))  # Resize for model

    try:
        # Extract features (shape: [1, 512] or similar)
        features = img2vec.get_vec(pil_img)  

        # Flatten to 2D (shape: [1, n_features])
        features_2d = features.reshape(1, -1)  

        # Predict
        pred = model.predict(features_2d)  
        pred_label = pred[0]  # Get class label

    except Exception as e:
        print(f"Prediction error: {e}")
        pred_label = "Error"

    # Display prediction
    cv2.putText(frame, f"Class: {pred_label}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Disaster Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break