import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    Simple 2-layer MLP for MNIST classification.
    Input: 784 (28x28 flattened)
    Hidden: 256 with ReLU
    Output: 10 (logits - NO activation here!)
    """

    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(SimpleMLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        pass

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Quick self-test
if __name__ == "__main__":
    model = SimpleMLP()
    dummy_input = torch.randn(32, 1, 28, 28)  # Batch of 32 MNIST images
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Should be [32, 10]
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")  # Should be unbounded (logits)