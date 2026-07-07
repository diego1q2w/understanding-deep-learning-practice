import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import os
import time

from model import SimpleCNN


def get_dataloaders(batch_size=128):
    """Load CIFAR-10 train and validation sets with augmentation."""

    # Training transforms (with augmentation)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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


def train_with_sgd(epochs=100, lr=0.1, momentum=0.9, weight_decay=5e-4, device='cpu'):
    """Train with SGD + Momentum on CIFAR-10."""
    print(f"\n{'=' * 70}")
    print(f"TRAINING WITH SGD + MOMENTUM (CIFAR-10, No Dropout)")
    print(f"LR: {lr} | Momentum: {momentum} | Weight Decay: {weight_decay}")
    print(f"{'=' * 70}\n")

    # Setup
    train_loader, val_loader = get_dataloaders()
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay, nesterov=True)
    criterion = nn.CrossEntropyLoss()

    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
        'epoch_times': []
    }

    # Train
    start_time = time.time()
    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        history['epoch_times'].append(epoch_time)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | LR: {current_lr:.6f} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                  f"Best Val: {best_val_acc:.2f}% | Time: {epoch_time:.2f}s")

        scheduler.step()

    total_time = time.time() - start_time
    history['total_time'] = total_time
    history['best_val_acc'] = best_val_acc

    print(f"\n✅ SGD Training Complete!")
    print(f"Total Time: {total_time / 60:.2f} minutes")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Final Val Accuracy: {history['val_acc'][-1]:.2f}%")

    return history


def train_with_adamw(epochs=100, lr=0.001, weight_decay=5e-4, device='cpu'):
    """Train with AdamW on CIFAR-10."""
    print(f"\n{'=' * 70}")
    print(f"TRAINING WITH ADAMW (CIFAR-10, No Dropout)")
    print(f"LR: {lr} | Weight Decay: {weight_decay}")
    print(f"{'=' * 70}\n")

    # Setup
    train_loader, val_loader = get_dataloaders()
    model = SimpleCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                            betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.CrossEntropyLoss()

    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
        'epoch_times': []
    }

    # Train
    start_time = time.time()
    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        history['epoch_times'].append(epoch_time)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | LR: {current_lr:.6f} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                  f"Best Val: {best_val_acc:.2f}% | Time: {epoch_time:.2f}s")

        scheduler.step()

    total_time = time.time() - start_time
    history['total_time'] = total_time
    history['best_val_acc'] = best_val_acc

    print(f"\n✅ AdamW Training Complete!")
    print(f"Total Time: {total_time / 60:.2f} minutes")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Final Val Accuracy: {history['val_acc'][-1]:.2f}%")

    return history

def train_with_sgd_to_adamw_switch(switch_epoch=30, total_epochs=100, device='cpu'):
    """Train with SGD, then switch to AdamW (testing if this works)."""
    print(f"\n{'=' * 70}")
    print(f"TRAINING WITH SGD → AdamW SWITCH (CIFAR-10)")
    print(f"SGD for {switch_epoch} epochs, then AdamW for {total_epochs - switch_epoch} epochs")
    print(f"{'=' * 70}\n")

    # Setup
    train_loader, val_loader = get_dataloaders()
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: SGD
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=switch_epoch, eta_min=0.01)

    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'lr': [], 'epoch_times': [], 'optimizer': []
    }

    start_time = time.time()
    best_val_acc = 0

    # Phase 1: Train with SGD
    print(f"Phase 1: SGD (Epochs 1-{switch_epoch})")
    for epoch in range(1, switch_epoch + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['epoch_times'].append(time.time() - epoch_start)
        history['optimizer'].append('SGD')

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{total_epochs} [SGD] | LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Best: {best_val_acc:.2f}%")

        scheduler.step()

    # Switch to AdamW
    print(f"\n🔄 SWITCHING TO ADAMW at epoch {switch_epoch}")
    print(f"Phase 2: AdamW (Epochs {switch_epoch + 1}-{total_epochs})")

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - switch_epoch, eta_min=0)

    # Phase 2: Train with AdamW
    for epoch in range(switch_epoch + 1, total_epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['epoch_times'].append(time.time() - epoch_start)
        history['optimizer'].append('AdamW')

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{total_epochs} [AdamW] | LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Best: {best_val_acc:.2f}%")

        scheduler.step()

    history['total_time'] = time.time() - start_time
    history['best_val_acc'] = best_val_acc
    history['switch_epoch'] = switch_epoch

    print(f"\n✅ SGD→AdamW Training Complete!")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Final Val Accuracy: {history['val_acc'][-1]:.2f}%")

    return history


def train_with_adamw_to_sgd_switch(switch_epoch=30, total_epochs=100, device='cpu'):
    """Train with AdamW, then switch to SGD (the recommended approach)."""
    print(f"\n{'=' * 70}")
    print(f"TRAINING WITH AdamW → SGD SWITCH (CIFAR-10)")
    print(f"AdamW for {switch_epoch} epochs, then SGD for {total_epochs - switch_epoch} epochs")
    print(f"{'=' * 70}\n")

    # Setup
    train_loader, val_loader = get_dataloaders()
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: AdamW
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=switch_epoch, eta_min=0.0001)

    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'lr': [], 'epoch_times': [], 'optimizer': []
    }

    start_time = time.time()
    best_val_acc = 0

    # Phase 1: Train with AdamW
    print(f"Phase 1: AdamW (Epochs 1-{switch_epoch})")
    for epoch in range(1, switch_epoch + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['epoch_times'].append(time.time() - epoch_start)
        history['optimizer'].append('AdamW')

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{total_epochs} [AdamW] | LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Best: {best_val_acc:.2f}%")

        scheduler.step()

    # Switch to SGD
    print(f"\n🔄 SWITCHING TO SGD at epoch {switch_epoch}")
    print(f"Phase 2: SGD (Epochs {switch_epoch + 1}-{total_epochs})")

    # Use lower LR for SGD since we're already in a good region
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - switch_epoch, eta_min=0)

    # Phase 2: Train with SGD
    for epoch in range(switch_epoch + 1, total_epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['epoch_times'].append(time.time() - epoch_start)
        history['optimizer'].append('SGD')

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{total_epochs} [SGD] | LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Best: {best_val_acc:.2f}%")

        scheduler.step()

    history['total_time'] = time.time() - start_time
    history['best_val_acc'] = best_val_acc
    history['switch_epoch'] = switch_epoch

    print(f"\n✅ AdamW→SGD Training Complete!")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Final Val Accuracy: {history['val_acc'][-1]:.2f}%")

    return history


def run_comparison_experiment(epochs=100, switch_epoch=30, device='cpu'):
    """Run all four experiments and save results."""
    os.makedirs('results', exist_ok=True)

    # Experiment 1: Pure SGD
    print("\n" + "🟢 " * 35)
    history_sgd = train_with_sgd(epochs=epochs, lr=0.1, momentum=0.9,
                                 weight_decay=5e-4, device=device)
    with open('results/sgd_pure.pkl', 'wb') as f:
        pickle.dump(history_sgd, f)

    # Experiment 2: Pure AdamW
    print("\n" + "🔴 " * 35)
    history_adamw = train_with_adamw(epochs=epochs, lr=0.001, weight_decay=5e-4, device=device)
    with open('results/adamw_pure.pkl', 'wb') as f:
        pickle.dump(history_adamw, f)

    # Experiment 3: SGD → AdamW
    print("\n" + "🟢🔴 " * 35)
    history_sgd_adamw = train_with_sgd_to_adamw_switch(switch_epoch=switch_epoch,
                                                       total_epochs=epochs, device=device)
    with open('results/sgd_to_adamw.pkl', 'wb') as f:
        pickle.dump(history_sgd_adamw, f)

    # Experiment 4: AdamW → SGD
    print("\n" + "🔴🟢 " * 35)
    history_adamw_sgd = train_with_adamw_to_sgd_switch(switch_epoch=switch_epoch,
                                                       total_epochs=epochs, device=device)
    with open('results/adamw_to_sgd.pkl', 'wb') as f:
        pickle.dump(history_adamw_sgd, f)

    # Print final comparison
    print(f"\n{'=' * 80}")
    print("FINAL RESULTS COMPARISON - ALL 4 METHODS")
    print(f"{'=' * 80}")
    print(f"{'Method':<30} {'Best Val Acc':<15} {'Final Val Acc':<15} {'Train-Val Gap':<15}")
    print(f"{'-' * 80}")

    methods = [
        ('Pure SGD', history_sgd),
        ('Pure AdamW', history_adamw),
        (f'SGD→AdamW (switch@{switch_epoch})', history_sgd_adamw),
        (f'AdamW→SGD (switch@{switch_epoch})', history_adamw_sgd)
    ]

    for name, hist in methods:
        best = hist['best_val_acc']
        final_val = hist['val_acc'][-1]
        gap = hist['train_acc'][-1] - hist['val_acc'][-1]
        print(f"{name:<30} {best:>13.2f}% {final_val:>13.2f}% {gap:>13.2f}%")

    print(f"{'=' * 80}")

    return history_sgd, history_adamw, history_sgd_adamw, history_adamw_sgd


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Run all 4 experiments with switch at epoch 30
    run_comparison_experiment(epochs=100, switch_epoch=30, device=device)