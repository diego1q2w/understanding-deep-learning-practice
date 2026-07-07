import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import os
import gc
from model import SimpleCNN


def get_dataloaders(batch_size, num_workers=0):
    """Get CIFAR-10 train/val loaders with augmentation."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def val_epoch(model, val_loader, criterion, device):
    """Validate one epoch."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def save_checkpoint(epoch, model, optimizer, scheduler, history, batch_size, filepath):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'batch_size': batch_size
    }
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved at epoch {epoch + 1}")


def load_checkpoint(filepath, model, optimizer, scheduler, device):
    """Load training checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['history'], checkpoint['batch_size']


def clear_memory(device):
    """Force garbage collection and clear GPU cache."""
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()


def train_with_batch_size(batch_size, target_steps, base_lr=0.1, device='cpu',
                          checkpoint_dir='checkpoints', resume=False):
    """
    Train with specific batch size for a TARGET NUMBER OF STEPS.

    Args:
        batch_size: Batch size to use
        target_steps: Total optimizer steps to take (for fair comparison)
        base_lr: Base learning rate (will be scaled linearly with batch size)
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        resume: Whether to resume from checkpoint
    """
    print(f"\n{'=' * 70}")
    print(f"TRAINING WITH BATCH SIZE = {batch_size}")
    print(f"{'=' * 70}")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f'{checkpoint_dir}/batch_{batch_size}_checkpoint.pth'

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(batch_size)

    # Calculate how many epochs needed to reach target_steps
    steps_per_epoch = len(train_loader)
    epochs = max(1, target_steps // steps_per_epoch)
    actual_steps = epochs * steps_per_epoch

    # Linear LR scaling
    lr_scale = min(batch_size / 128, 4.0)
    lr = base_lr * lr_scale

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Target steps: {target_steps}")
    print(f"Training for {epochs} epochs (actual steps: {actual_steps})")
    print(f"Learning rate: {lr:.4f} (scaled by {lr_scale:.2f}x)")

    # Model setup
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Initialize history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
        'batch_size': batch_size,
        'epochs': epochs,
        'steps_per_epoch': steps_per_epoch,
        'total_steps': actual_steps
    }

    start_epoch = 0
    best_val_acc = 0

    # Resume from checkpoint if requested
    if resume and os.path.exists(checkpoint_path):
        print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
        start_epoch, history, loaded_batch_size = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, device
        )
        start_epoch += 1  # Start from next epoch
        best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
        print(f"✅ Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")

    print(f"{'=' * 70}\n")

    # Training loop
    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Print progress every 10% of training
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1:4d}/{epochs} | "
                  f"Train: {train_acc:6.2f}% | Val: {val_acc:6.2f}% | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"LR: {current_lr:.6f}")

        # Save checkpoint every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            save_checkpoint(epoch, model, optimizer, scheduler, history,
                            batch_size, checkpoint_path)

    print(f"\nBest Val Acc: {best_val_acc:.2f}%")
    print(f"Final Train-Val Gap: {history['train_acc'][-1] - history['val_acc'][-1]:.2f}%\n")

    # Clean up to free memory
    del model, optimizer, scheduler, train_loader, val_loader
    clear_memory(device)

    return history


def run_batch_size_experiment(batch_sizes=[32, 64, 128, 256, 512], device='cpu',
                              resume=False, skip_completed=True):
    """
    Run experiment comparing different batch sizes.

    Args:
        batch_sizes: List of batch sizes to test
        device: Device to train on
        resume: Whether to resume from checkpoints
        skip_completed: Skip batch sizes that have completed results
    """
    # Calculate target steps
    base_batch = 128
    base_epochs = 100
    steps_per_epoch_base = 50000 // base_batch
    target_steps = base_epochs * steps_per_epoch_base

    print(f"\n{'=' * 70}")
    print(f"BATCH SIZE COMPARISON EXPERIMENT (CIFAR-10)")
    print(f"{'=' * 70}")
    print(f"Target optimizer steps: {target_steps}")
    print(f"Batch sizes to test: {batch_sizes}")
    print(f"Resume mode: {resume}")
    print(f"Skip completed: {skip_completed}")
    print(f"{'=' * 70}\n")

    results = {}
    os.makedirs('results', exist_ok=True)

    for batch_size in batch_sizes:
        result_path = f'results/batch_{batch_size}.pkl'

        # Skip if already completed and skip_completed is True
        if skip_completed and os.path.exists(result_path) and resume:
            print(f"⏭️  Skipping batch_size={batch_size} (already completed)")
            with open(result_path, 'rb') as f:
                results[batch_size] = pickle.load(f)
            continue

        # Train with this batch size
        history = train_with_batch_size(
            batch_size=batch_size,
            target_steps=target_steps,
            base_lr=0.1,
            device=device,
            resume=resume
        )
        results[batch_size] = history

        # Save results immediately
        with open(result_path, 'wb') as f:
            pickle.dump(history, f)

        print(f"✅ Saved results for batch_size={batch_size}\n")

    # Save all results
    with open('results/all_batch_sizes.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE!")
    print(f"{'=' * 70}\n")

    return results


if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # To resume training: set resume=True
    # To skip completed batch sizes: skip_completed=True (default)
    results = run_batch_size_experiment(
        batch_sizes=[32, 64, 128, 256, 512],
        device=device,
        resume=True,  # ✅ Set to True to resume from checkpoints
        skip_completed=True  # ✅ Skip batch sizes that are done
    )