import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 - no dropout for clearer optimizer comparison."""

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 3 channels for RGB
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # Fully connected layers (no dropout!)
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv block 1: 32x32 -> 16x16
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Conv block 2: 16x16 -> 8x8
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Conv block 3: 8x8 -> 4x4
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten and FC layers (NO DROPOUT)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x