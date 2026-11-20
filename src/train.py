import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from model import ChestXRayCNN
from dataset import train_loader, val_loader, train_dataset

# CPU configuration
device = torch.device("cpu")

# Reproducibility
torch.manual_seed(42)
torch.set_num_threads(2)

# Create model
model = ChestXRayCNN().to(device)

# Calculate class weights
labels = [
    train_dataset.dataset.targets[i]
    for i in train_dataset.indices
]

class_counts = torch.bincount(
    torch.tensor(labels),
    minlength=2
).float()

class_weights = len(labels) / (2.0 * class_counts)
class_weights = class_weights.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training settings
EPOCHS = 3

print("Starting Chest X-ray CNN Training...")
print("Device: CPU")
print("Training images:", len(train_dataset))
print("Validation images:", len(val_loader.dataset))

# Training loop
for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (images, batch_labels) in enumerate(train_loader):

        images = images.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, batch_labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

        if (batch_index + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch_index + 1}/{len(train_loader)}"
            )

    train_accuracy = 100 * correct / total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():

        for images, batch_labels in val_loader:

            images = images.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, batch_labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += batch_labels.size(0)
            val_correct += (
                predicted == batch_labels
            ).sum().item()

    val_accuracy = 100 * val_correct / val_total

    print("\n" + "=" * 40)
    print(f"Epoch: {epoch + 1}/{EPOCHS}")
    print(f"Training Loss: {running_loss / len(train_loader):.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print("=" * 40)

# Save trained model
project_dir = Path(__file__).resolve().parent.parent
save_dir = project_dir / "models"
save_dir.mkdir(parents=True, exist_ok=True)

model_path = save_dir / "chest_xray_model.pth"

torch.save(model.state_dict(), model_path)

print("\nTraining completed successfully!")
print("Model saved at:", model_path)
