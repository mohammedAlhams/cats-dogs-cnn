import torch
from torchvision import transforms
from PIL import Image
import sys

from models import BaselineCNN, ImprovedCNN
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Choose which model to use
MODEL_TYPE = "improved"   # "baseline" or "improved"

# Load model
if MODEL_TYPE == "baseline":
    model = BaselineCNN()
    model.load_state_dict(torch.load("baseline_cats_dogs.pth", map_location="cpu"))
else:
    model = ImprovedCNN()
    model.load_state_dict(torch.load("improved_cats_dogs.pth", map_location="cpu"))

model.eval()

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# Get image path from command line
if len(sys.argv) < 2:
    print("Usage: python3 predict.py <path_to_image>")
    sys.exit()

image_path = sys.argv[1]

# Load the image
img = Image.open(image_path).convert("RGB")
img = transform(img)
img = img.unsqueeze(0)  # add batch dimension

# Predict
with torch.no_grad():
    output = model(img)
    prob = torch.sigmoid(output).item()

# Interpret result
if prob >= 0.5:
    print(f"Prediction: DOG 🐶  (confidence: {prob:.2f})")
else:
    print(f"Prediction: CAT 😺  (confidence: {1 - prob:.2f})")
