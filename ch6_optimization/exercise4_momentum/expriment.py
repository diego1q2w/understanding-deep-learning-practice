import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import os
from model import SimpleCNN


def get_dataloaders(batch_size=128):
    """Create CIFAR-10 dataloaders."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def compute_velocity_magnitude(optimizer):
    """Compute L2 norm of velocity (momentum buffer) across all parameters."""
    velocity_norm = 0.0
    count = 0

    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue

            # Access momentum buffer
            state = optimizer.state[p]
            if 'momentum_buffer' in state:
                velocity = state['momentum_buffer']
                velocity_norm += velocity.norm().item() ** 2
                count += 1

    if count == 0:
        return 0.0

    return np.sqrt(velocity_norm)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train one epoch and track metrics."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    batch_losses = []
    velocity_magnitudes = []

    for data, target in tqdm(train_loader, desc="Training", leave=False):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        batch_losses.append(loss.item())
        velocity_magnitudes.append(compute_velocity_magnitude(optimizer))

        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    return {
        'loss': total_loss / len(train_loader),
        'accuracy': 100. * correct / total,
        'batch_losses': batch_losses,
        'velocity_magnitudes': velocity_magnitudes
    }


def validate(model, test_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Validating", leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return {
        'loss': total_loss / len(test_loader),
        'accuracy': 100. * correct / total
    }


def train_with_momentum(beta, epochs=50, base_lr=0.1, device='cpu'):
    """Train with specific momentum value."""

    # Adaptive LR scaling for high momentum
    if beta >= 0.99:
        lr = base_lr * 0.1  # 10x smaller for β=0.99
    else:
        lr = base_lr

    print(f"\n{'=' * 70}")
    print(f"TRAINING WITH MOMENTUM β={beta} (LR={lr})")
    print(f"{'=' * 70}\n")

    # Setup
    train_loader, test_loader = get_dataloaders(batch_size=128)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Create optimizer with specific momentum
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=beta, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # History tracking
    history = {
        'beta': beta,
        'lr': lr,  # Store actual LR used
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': [],
        'batch_losses': [],
        'velocity_magnitudes': [],
        'epoch_velocity_means': [],
    }

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, test_loader, criterion, device)
        scheduler.step()

        # Store metrics
        history['train_losses'].append(train_metrics['loss'])
        history['train_accs'].append(train_metrics['accuracy'])
        history['val_losses'].append(val_metrics['loss'])
        history['val_accs'].append(val_metrics['accuracy'])
        history['batch_losses'].extend(train_metrics['batch_losses'])
        history['velocity_magnitudes'].extend(train_metrics['velocity_magnitudes'])
        history['epoch_velocity_means'].append(np.mean(train_metrics['velocity_magnitudes']))

        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
        print(f"Avg Velocity: {history['epoch_velocity_means'][-1]:.6f}")

    print(f"\n{'=' * 70}")
    print(f"FINAL RESULTS (β={beta}, LR={lr}):")
    print(f"  Best Val Acc: {max(history['val_accs']):.2f}%")
    print(f"  Final Val Acc: {history['val_accs'][-1]:.2f}%")
    print(f"{'=' * 70}\n")

    return history


def run_all_experiments(device='cpu'):
    """Run experiments with different momentum values."""
    betas = [0.0, 0.5, 0.9, 0.99]
    results = {}

    for beta in betas:
        history = train_with_momentum(beta=beta, epochs=50, base_lr=0.1, device=device)
        results[f'beta_{beta}'] = history

        # Save individual result
        os.makedirs('results', exist_ok=True)
        with open(f'results/momentum_beta_{beta}.pkl', 'wb') as f:
            pickle.dump(history, f)

    # Save combined results
    with open('results/all_momentum_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\n" + "=" * 70)
    print("SUMMARY: ALL MOMENTUM VALUES")
    print("=" * 70)
    print(f"{'Beta':<8} {'Best Val Acc':<15} {'Final Val Acc':<15}")
    print("-" * 70)

    for beta in betas:
        key = f'beta_{beta}'
        best_acc = max(results[key]['val_accs'])
        final_acc = results[key]['val_accs'][-1]
        print(f"{beta:<8} {best_acc:<15.2f} {final_acc:<15.2f}")

    print("=" * 70)

    return results


if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    results = run_all_experiments(device=device)
    print("\n✅ All experiments completed! Results saved to results/")

