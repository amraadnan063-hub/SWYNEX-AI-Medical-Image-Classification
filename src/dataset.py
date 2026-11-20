
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Dataset location
DATA_DIR = r"D:\SWYNEX-AI-Projects\dataset\chest_xray"

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load training dataset
full_dataset = datasets.ImageFolder(
    root=DATA_DIR + r"\train",
    transform=transform
)

# 80% training and 20% validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Smaller batch size for CPU and memory
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

if __name__ == "__main__":
    print("Classes:", full_dataset.classes)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Batch size: 8")
    print("Data preparation successful!")

