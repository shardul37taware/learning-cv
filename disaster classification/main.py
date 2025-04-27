from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data="D:\\git\\learning-cv\\01_opencv_basics\\disaster classification\\config.yaml", epochs=1)  # train the model
