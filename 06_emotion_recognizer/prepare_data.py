import os
import cv2
from utils import get_face_landmarks

data_dir = r'D:\git\learning-cv\06_emotion_recognizer\Emotions Dataset\test'

for emotion in os.listdir(data_dir):
    for img_path_ in os.listdir(os.path.join(data_dir, emotion)):
        img_path = os.path.join(data_dir, emotion, img_path_)

        img = cv2.imread(img_path)

        face_landmarks = get_face_landmarks(img)

        print(len(face_landmarks))