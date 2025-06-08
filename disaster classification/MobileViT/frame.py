import timm
import torch
import torchvision.transforms as transforms
import cv2
from collections import Counter
import numpy as np

# Load model
model = timm.create_model("mobilevit_s", pretrained=False, num_classes=4)
model.load_state_dict(torch.load("D:/git/learning-cv/disaster classification/MobileViT/mobilevit_s_disaster_II.pth", map_location=torch.device('cpu')))
model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Class labels
class_names = ["Damage", "Fire", "Flood", "Normal"]

# Load the image
image_path = r"D:\sst\disaster\dataset\result\test\flood\flood (266).jpg" # change this to your input image path
image = cv2.imread(image_path)

# Ensure image was loaded
if image is None:
    raise FileNotFoundError(f"Image at path '{image_path}' could not be loaded.")

# Preprocess image
input_tensor = transform(image).unsqueeze(0)  # [1, 3, 256, 256]

# Inference
with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_idx].item()

    label = class_names[pred_idx] if confidence >= 0.5 else "Normal"

# Annotate the image
cv2.putText(image, f"{label} ({confidence:.2f})", (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 6)

# Save the annotated image
output_path = r"D:\sst\disaster\output3.jpg"  # change this as needed
cv2.imwrite(output_path, image)
print(f"Output saved to {output_path}")
