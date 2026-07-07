import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pickle
from pathlib import Path
from tqdm import tqdm
from model import DeepMLPWithNorm, NormType, count_parameters


def get_dataloaders(batch_size=128):
    """Load CIFAR-10 train and validation sets."""
    # CIFAR-10 normalization (RGB: 3 channels)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),  # CIFAR-10 mean
            std=(0.2470, 0.2435, 0.2616)    # CIFAR-10 std
        )
    ])

    train_dataset = datasets.CIFAR10(
        root='data', train=True, download=True, transform=transform
    )
    val_dataset = datasets.CIFAR10(
        root='data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )

    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in loader:
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

    return total_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def train_model(norm_type, num_epochs=30, lr=0.01, device='cpu'):  # More epochs!
    """
    Train a model with specified normalization type.

    Returns:
        Dictionary with training history
    """
    print(f"\n{'=' * 70}")
    print(f"Training with {norm_type.value.upper()} Normalization")
    print(f"{'=' * 70}")

    # Create model for CIFAR-10
    model = DeepMLPWithNorm(
        input_dim=3072,        # 32×32×3 = 3072 (changed from 784)
        hidden_dim=512,        # Larger hidden dim for harder task
        output_dim=10,         # Same 10 classes
        num_layers=20,         # Deep network to stress-test norm
        activation='relu',
        norm_type=norm_type,
        init_method='xavier'   # Still use "bad" init to show norm saves it
    ).to(device)

    print(f"Parameters: {count_parameters(model):,}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    train_loader, val_loader = get_dataloaders(batch_size=128)

    # Training history
    history = {
        'norm_type': norm_type.value,
        'lr': lr,
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'activation_stats': [],
        'gradient_stats': []
    }

    # Get initial stats
    sample_data, sample_target = next(iter(train_loader))
    sample_data, sample_target = sample_data.to(device), sample_target.to(device)

    model.eval()
    with torch.no_grad():
        _, act_stats = model(sample_data, return_stats=True)
        history['activation_stats'].append(act_stats)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Get activation stats
        model.eval()
        with torch.no_grad():
            _, act_stats = model(sample_data, return_stats=True)
            history['activation_stats'].append(act_stats)

        # Get gradient stats (take one backward pass)
        model.train()
        optimizer.zero_grad()
        output = model(sample_data)
        loss = criterion(output, sample_target)
        loss.backward()
        grad_stats = model.get_gradient_stats()
        history['gradient_stats'].append(grad_stats)

        print(f"Epoch {epoch:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.2f}%")

    print(f"\n✅ Final: Val Acc = {val_acc:.2f}%")

    return history


if __name__ == '__main__':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Train all three variants
    lr_map = {
        NormType.NONE: 0.001,  # Much lower for stability
        NormType.BATCH: 0.05,  # Can handle higher
        NormType.LAYER: 0.03  # Middle ground
    }

    norm_types = [NormType.NONE, NormType.BATCH, NormType.LAYER]
    all_results = []

    for norm_type in norm_types:
        result = train_model(
            norm_type=norm_type,
            num_epochs=30,      # More epochs for CIFAR-10
            lr=lr_map[norm_type],            # Might need to adjust per variant!
            device=str(device)
        )
        all_results.append(result)

    # Save results
    output_file = results_dir / 'training_comparison_cifar10.pkl'  # New filename
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\n{'=' * 70}")
    print(f"✅ Results saved to {output_file}")
    print(f"{'=' * 70}")

    # Summary table (adjust threshold)
    print("\n📊 FINAL SUMMARY:")
    print(f"{'Norm Type':<15} {'Final Val Acc':<15} {'Status'}")
    print("-" * 50)
    for result in all_results:
        status = "✅ Good" if result['val_acc'][-1] > 45 else "❌ Poor"  # Lower bar!
        print(f"{result['norm_type']:<15} {result['val_acc'][-1]:>6.2f}%         {status}")

