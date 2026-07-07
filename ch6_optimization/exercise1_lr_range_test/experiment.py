import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import os


class TinyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TinyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x


def get_dataloaders(batch_size=128):
    """Load MNIST train and validation sets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(val_loader), 100.0 * correct / total


def train_with_constant_lr(lr=0.02, epochs=20, device='cpu'):
    """Train with constant learning rate."""
    print(f"\n{'=' * 70}")
    print(f"TRAINING WITH CONSTANT LR = {lr}")
    print(f"{'=' * 70}\n")

    # Setup
    train_loader, val_loader = get_dataloaders()
    model = TinyModel(input_dim=784, output_dim=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    # Train
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch:2d}/{epochs} | LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    return history


def train_with_cosine_annealing(max_lr=0.1, min_lr=0.0, epochs=20, device='cpu'):
    """Train with cosine annealing learning rate schedule."""
    print(f"\n{'=' * 70}")
    print(f"TRAINING WITH COSINE ANNEALING LR (max={max_lr}, min={min_lr})")
    print(f"{'=' * 70}\n")

    # Setup
    train_loader, val_loader = get_dataloaders()
    model = TinyModel(input_dim=784, output_dim=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    # Train
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch:2d}/{epochs} | LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Step the scheduler after each epoch
        scheduler.step()

    return history


def run_comparison_experiment(device='cpu', epochs=20):
    """Run both experiments and save results."""
    os.makedirs('results', exist_ok=True)

    # Experiment 1: Constant LR
    history_constant = train_with_constant_lr(lr=0.02, epochs=epochs, device=device)
    with open('results/constant_lr.pkl', 'wb') as f:
        pickle.dump(history_constant, f)
    print(f"\n✅ Constant LR results saved!")

    # Experiment 2: Cosine Annealing
    history_cosine = train_with_cosine_annealing(max_lr=0.1, min_lr=0.0, epochs=epochs, device=device)
    with open('results/cosine_annealing.pkl', 'wb') as f:
        pickle.dump(history_cosine, f)
    print(f"\n✅ Cosine Annealing results saved!")

    # Print final comparison
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Method':<25} {'Final Train Acc':<20} {'Final Val Acc':<20}")
    print(f"{'-' * 70}")
    print(
        f"{'Constant LR (0.02)':<25} {history_constant['train_acc'][-1]:>18.2f}% {history_constant['val_acc'][-1]:>18.2f}%")
    print(
        f"{'Cosine Annealing (0.1→0)':<25} {history_cosine['train_acc'][-1]:>18.2f}% {history_cosine['val_acc'][-1]:>18.2f}%")
    print(f"{'=' * 70}")

    return history_constant, history_cosine


if __name__ == "__main__":
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Run comparison
    run_comparison_experiment(device=device, epochs=20)