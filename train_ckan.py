import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ✅ FIXED AMP (new API)
from torch.cuda.amp import autocast, GradScaler

# ======================
# DEVICE
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# IMPORT CKAN LAYER
# ======================
import sys
sys.path.append('./Convolutional-KANs')
sys.path.append('./Convolutional-KANs/kan_convolutional')

from kan_convolutional.KANConv import KAN_Convolutional_Layer


# ======================
# MODEL
# ======================
class CKAN_CIFAR10(nn.Module):
    def __init__(self, grid_size=10):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            in_channels=3,
            out_channels=8,
            kernel_size=(3,3),
            grid_size=grid_size
        )
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = KAN_Convolutional_Layer(
            in_channels=8,
            out_channels=8,
            kernel_size=(3,3),
            grid_size=grid_size
        )
        self.bn2 = nn.BatchNorm2d(8)

        self.pool1 = nn.MaxPool2d(2,2)

        self.conv3 = KAN_Convolutional_Layer(
            in_channels=8,
            out_channels=8,
            kernel_size=(3,3),
            grid_size=grid_size
        )
        self.bn3 = nn.BatchNorm2d(8)

        self.pool2 = nn.MaxPool2d(2,2)
        self.flat=nn.Flatten()

        self.linear1 = nn.Linear(288, 256)
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.pool1(x)

        x = self.bn3(self.conv3(x))
        x = self.pool2(x)

        x = self.flat(x)

        x = self.linear1(x)
        x = self.linear2(x)

        return x


# ======================
# DATA
# ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4,pin_memory=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4,pin_memory=True)


# ======================
# MODEL SETUP
# ======================
model = CKAN_CIFAR10(grid_size=5).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

scaler = GradScaler()


# ======================
# CHECKPOINT SETUP
# ======================
os.makedirs("checkpoints", exist_ok=True)

start_epoch = 0
print("🚀 Starting fresh training")

# ======================
# TRAIN FUNCTION
# ======================
def train():
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    total_batches = len(trainloader)

    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # ✅ PRINT PROGRESS
        if batch_idx % 50 == 0:
            print(f"Batch [{batch_idx}/{total_batches}] | Loss: {loss.item():.4f}")

    accuracy = 100 * correct / total
    return running_loss / total_batches, accuracy


# ======================
# TEST FUNCTION
# ======================
def test():
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return running_loss / len(testloader), accuracy


# ======================
# TRAIN LOOP
# ======================
epochs = 50

for epoch in range(start_epoch, epochs):

    print(f"\n🚀 Epoch [{epoch+1}/{epochs}]")

    train_loss, train_acc = train()
    test_loss, test_acc = test()

    scheduler.step()

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    print("-" * 50)

    # ✅ SAVE CHECKPOINT
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "checkpoints/latest.pth")
    # ALSO save per epoch (safe backup)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"checkpoints/epoch_{epoch+1}.pth")
print("✅ Training complete!")
