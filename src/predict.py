import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

from model import ChestXRayCNN

# Device
device = torch.device("cpu")

# Project folder
project_dir = Path(__file__).resolve().parent.parent

# Load trained model
model = ChestXRayCNN().to(device)

model_path = project_dir / "models" / "chest_xray_model.pth"

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model.eval()

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Ask user for image path
image_path = input("Enter X-ray image path: ").strip().strip('"')

if not Path(image_path).is_file():
    print("Error: Image file not found.")
    raise SystemExit

# Open and preprocess image
image = Image.open(image_path).convert("L")
image = transform(image).unsqueeze(0)

# Prediction
with torch.no_grad():
    outputs = model(image.to(device))
    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probabilities, 1)

# Class names
classes = ["NORMAL", "PNEUMONIA"]

print("\nPrediction Result")
print("-------------------------")
print("Predicted Class:", classes[predicted.item()])
print(f"Model Confidence: {confidence.item() * 100:.2f}%")
print("-------------------------")
print("Educational prototype only - not for clinical diagnosis.")