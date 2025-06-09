import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image
# Select device (GPU if available, else CPU)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your PyTorch installation.")

device = torch.device("cuda")
print(f"Using device: {device}")

data_dir = "plant_identification_dataset"
if not os.path.exists(data_dir):
    raise RuntimeError("Manual dataset not found. Please download it manually from Kaggle.")


transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),  # Ensure RGB
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjusted to 3 values for RGB
])


# Load dataset
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "Train_Set_Folder"), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "Test_Set_Folder"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define CNN model (LeNet-style)
class PlantNet(nn.Module):
    def __init__(self, num_classes):
        super(PlantNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Get number of classes
num_classes = len(train_dataset.classes)
print(f"Number of Plant Classes: {num_classes}")

# Initialize model and move it to the correct device
model = PlantNet(num_classes).to(device)

# Loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to the same device as the model
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

print("Training complete!")

# Save the model
torch.save(model.state_dict(), "plant_leaves_identify.pth")
print("Model saved as 'plant_leaves_identify.pth'.")