import os
import numpy as np

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import pickle

from PIL import Image

print('0')
# prepare data
input_dir = 'D:/sst/disaster/dataset raw'
categories = ['damage','fire','flood','normal']

data = []
labels = [] 
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

for category_idx, category in enumerate(categories):
    category_path = os.path.join(input_dir, category)
    for file in os.listdir(category_path):
        if not any(file.lower().endswith(ext) for ext in valid_extensions):
            continue
        img_path = os.path.join(category_path, file)
        print(f"Processing: {img_path}")
        try:
            with Image.open(img_path).convert('RGB') as img:
                img = img.resize((15, 15)) 
                data.append(np.asarray(img, dtype=np.float32).flatten())
                labels.append(category_idx)
        except Exception as e:
            print(f"Error with file {img_path}: {e}") 

data = np.asarray(data)
labels = np.asarray(labels)

print('1')
# test train split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

print('2')
# train classifier
classifier = SVC()
parameters = [{'gamma':[0.01, 0.001, 0.0001], 'C':[1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(x_train, y_train)

print('3')
# test performance
best_estimator = grid_search.best_estimator_
y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score*100)))

pickle.dump(best_estimator, open('./model.p', 'wb'))