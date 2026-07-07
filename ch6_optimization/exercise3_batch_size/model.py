import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10."""

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Conv block 1
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # 32x32 -> 16x16

        # Conv block 2
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # 16x16 -> 8x8

        # Conv block 3
        x = self.relu(self.conv3(x))
        x = self.pool(x)  # 8x8 -> 4x4

        # Flatten and FC
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x