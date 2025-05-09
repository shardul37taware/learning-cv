import os
from img2vec_pytorch import Img2Vec
from PIL import Image


# prepare data
data_dir = 'D:/sst/disaster/dataset/result'

train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

data = {}

for j, dir_ in enumerate([train_dir, val_dir]):
    features = []
    labels = []
    for category in os.listdir(dir_):
        for img_path in os.listdir(os.path.join(dir_, category)):
            img_path_ = os.path.join(dir_, category, img_path)
            img = Image.open(img_path_)

            img_features = Img2Vec.get_vec(img)

            features.append(img_features)
            labels.append(category)

    data[['training_data', 'validation_data'][j]] = features
    data[['training_labels','validation_labels'][j]] = labels

print(data.keys)