import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelAWithoutLinear(nn.Module):
    # no pooling (more strictly translation-equivariant)
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 15, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(15, 5, kernel_size=3, padding=1)
        self.head  = nn.Linear(5, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class ModelBWithoutLinear(nn.Module):
    # with pooling (less strictly equivariant, more robust/invariant in practice)
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 15, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(15, 5,  kernel_size=3, padding=1 )
        self.pool  = nn.MaxPool2d(2, 2)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.head  = nn.Linear(5, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x

class ModelA(nn.Module):
    # no pooling (more strictly translation-equivariant)
    def __init__(self):
        super().__init__()
        self.model_a = ModelAWithoutLinear()
        self.gap   = nn.AdaptiveAvgPool2d(1)  # (N,5,1,1)
        self.head  = nn.Linear(5, 10)

    def forward(self, x):
        x = self.model_a(x)
        x = self.gap(x).flatten(1)            # (N,5)
        return self.head(x)

class ModelB(nn.Module):
    # with pooling (less strictly equivariant, more robust/invariant in practice)
    def __init__(self):
        super().__init__()
        self.model_b = ModelBWithoutLinear()
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.head  = nn.Linear(5, 10)

    def forward(self, x):
        x = self.model_b(x)
        x = self.gap(x).flatten(1)
        return self.head(x)


# Test ModelA
model_a = ModelAWithoutLinear()
x = torch.randn(1, 3, 32, 32)
out = model_a(x)
print(f"ModelA output shape: {out.shape}")  # Should be [1, 5, 32, 32]

# Test ModelB
model_b = ModelBWithoutLinear()
out = model_b(x)
print(f"ModelB output shape: {out.shape}")  # Should be [1, 5, 8, 8]


# Tests Translation Equivariance and Invariance
# **Translation Equivariance**:
# This means if you shift the input by some amount, the output shifts by the same amount.
# Mathematically: `Conv(shift(x)) = shift(Conv(x))`.
# This is a fundamental property of convolution operations.
# **Translation Invariance**:
# This means the output doesn't change (much) when you shift the input.
# This is what we often want for classification - a cat should still be recognized as a cat even if it's moved a few pixels

