import torch
import torch.nn as nn


class TinyModel(nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.fc1 = nn.Linear(2,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,2)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    model = TinyModel()
    dummy_input = torch.randn(10,2)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    probs = torch.softmax(output, dim=1)
    print(f"Probabilities: {probs}")
    print(f"Total Prob: {probs.sum(dim=1)}")