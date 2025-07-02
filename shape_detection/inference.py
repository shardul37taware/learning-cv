import torch
import cv2
import numpy as np
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet

# --- Configuration ---
MODEL_PATH = r"C:\Users\shard\Downloads\epoch_10.pth"  # <-- Update this
MODEL_NAME = 'tf_efficientdet_d0'
NUM_CLASSES = 9
IMAGE_SIZE = 512
CLASS_NAMES = ['circle', 'circle_demi', 'circle_quarter', 'cross', 'pentagon', 
               'rectangle', 'square', 'star', 'triangle']  # <-- Match with your model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
COLORS = np.random.uniform(0, 255, size=(NUM_CLASSES, 3))

# --- Load model ---
def load_model():
    config = get_efficientdet_config(MODEL_NAME)
    config.num_classes = NUM_CLASSES
    config.image_size = (IMAGE_SIZE, IMAGE_SIZE)

    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=NUM_CLASSES)

    model = DetBenchPredict(net)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model

# --- Preprocessing ---
def preprocess(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    scale = IMAGE_SIZE / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image_rgb, (new_w, new_h))

    pad_h = IMAGE_SIZE - new_h
    pad_w = IMAGE_SIZE - new_w
    padded = np.pad(resized, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    tensor = torch.from_numpy(padded).float().permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0).to(DEVICE)
    return tensor, (h, w), scale

# --- Postprocessing ---
def postprocess(outputs, original_size, scale):
    dets = outputs[0]
    boxes = dets[:, :4].cpu().numpy()
    scores = dets[:, 4].cpu().numpy()
    labels = dets[:, 5].cpu().numpy().astype(int)

    keep = scores > 0.3
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    h, w = original_size
    boxes[:, [0, 2]] *= w / scale
    boxes[:, [1, 3]] *= h / scale
    return boxes, scores, labels

# --- Draw boxes ---
def draw(image, boxes, scores, labels):
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        color = COLORS[label]
        label_text = f"{CLASS_NAMES[label]}: {score:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(image, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return image

# --- Main webcam loop ---
def run_webcam():
    model = load_model()
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return

    print("Webcam running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor, original_size, scale = preprocess(frame)

        with torch.no_grad():
            output = model(input_tensor)

        boxes, scores, labels = postprocess(output, original_size, scale)
        frame = draw(frame, boxes, scores, labels)

        cv2.imshow("EfficientDet Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Run ---
if __name__ == '__main__':
    run_webcam()
