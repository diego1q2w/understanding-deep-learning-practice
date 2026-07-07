import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle

from model import SimpleMLP
from train import train_with_mse, train_with_ce, validate


def run_experiment(loss_type='ce', epochs=10, lr=0.001, seed=42):
    """
    Run full experiment with specified loss type.
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # TODO: Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # TODO: Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # TODO: Create model and optimizer
    model = SimpleMLP().to(device)
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    # TODO: Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Train
        if loss_type == 'mse':
            train_metrics = train_with_mse(model, train_loader, optimizer, device)
        else:
            train_metrics = train_with_ce(model, train_loader, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, device, loss_type)

        # Log
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])

        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")

    # Final validation with predictions
    final_val = validate(model, val_loader, device, loss_type)
    history['final_predictions'] = final_val['predictions']
    history['final_targets'] = final_val['targets']

    # Save results
    with open(f'results/results_{loss_type}.pkl', 'wb') as f:
        pickle.dump(history, f)

    print(f"\n✅ {loss_type.upper()} training complete!")
    print(f"Final Val Accuracy: {history['val_acc'][-1] * 100:.2f}%")

    return history


if __name__ == "__main__":
    import os

    os.makedirs('results', exist_ok=True)

    # Run both experiments
    print("=" * 70)
    print("EXPERIMENT 1: Cross-Entropy Loss")
    print("=" * 70)
    results_ce = run_experiment(loss_type='ce', epochs=10)

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: MSE Loss")
    print("=" * 70)
    results_mse = run_experiment(loss_type='mse', epochs=10)