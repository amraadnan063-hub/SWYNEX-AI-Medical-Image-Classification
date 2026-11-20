
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from pathlib import Path

from model import ChestXRayCNN

# Device
device = torch.device("cpu")

# Dataset path
DATA_DIR = Path(r"D:\SWYNEX-AI-Projects\dataset\chest_xray")

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load test dataset
test_dataset = datasets.ImageFolder(
    root=DATA_DIR / "test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

# Load trained model
model = ChestXRayCNN().to(device)

model_path = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "chest_xray_model.pth"
)

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model.eval()

# Predictions
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.tolist())

# Evaluation results
accuracy = accuracy_score(all_labels, all_predictions)

print("\nTEST RESULTS")
print("-" * 40)

print(f"Test Images: {len(test_dataset)}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_predictions,
    target_names=test_dataset.classes,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_predictions))

print("\nEvaluation completed successfully!")
