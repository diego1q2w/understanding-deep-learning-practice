import torch
import numpy as np
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from model import TinyModel
from training import train_model
from visualize import (plot_decision_boundary, plot_uncertainty_map,
                       plot_confidence_along_line)


def run_experiment():
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train CE model
    print("\n" + "=" * 70)
    print("Training with Cross-Entropy")
    print("=" * 70)
    model_ce = TinyModel().to(device)
    history_ce = train_model(model_ce, loader, epochs=200, criterion_type='ce', device=device)

    # Train MSE model
    print("\n" + "=" * 70)
    print("Training with MSE")
    print("=" * 70)
    model_mse = TinyModel().to(device)
    history_mse = train_model(model_mse, loader, epochs=200, criterion_type='mse', device=device)

    # Visualizations
    create_all_plots(model_ce, model_mse, X, y, device)


def create_all_plots(model_ce, model_mse, X, y, device):
    """Create all comparison plots."""

    # Plot 1: Decision boundaries comparison (2x2)
    fig1, axes = plt.subplots(2, 2, figsize=(16, 14))

    plot_decision_boundary(model_ce, X, y, "CE - Decision Boundary",
                           ax=axes[0, 0], device=device)
    plot_decision_boundary(model_mse, X, y, "MSE - Decision Boundary",
                           ax=axes[0, 1], device=device)
    plot_uncertainty_map(model_ce, X, y, "CE - Uncertainty Map",
                         ax=axes[1, 0], device=device)
    plot_uncertainty_map(model_mse, X, y, "MSE - Uncertainty Map",
                         ax=axes[1, 1], device=device)

    plt.tight_layout()
    plt.savefig('results/decision_boundaries.png', dpi=150)
    print("✅ Saved: results/decision_boundaries.png")

    # Plot 2: Confidence along line
    fig2 = plot_confidence_along_line(model_ce, model_mse, X, device)
    fig2.savefig('results/confidence_line.png', dpi=150)
    print("✅ Saved: results/confidence_line.png")

    # Plot 3: Temperature sweep (CE only)
    fig3, axes = plt.subplots(2, 3, figsize=(18, 12))
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    for idx, temp in enumerate(temperatures):
        ax = axes[idx // 3, idx % 3]
        plot_decision_boundary(model_ce, X, y,
                               f"CE - Temperature={temp}",
                               temperature=temp, ax=ax, device=device)

    plt.tight_layout()
    plt.savefig('results/temperature_sweep.png', dpi=150)
    print("✅ Saved: results/temperature_sweep.png")

    plt.show()


if __name__ == "__main__":
    import os

    os.makedirs('results', exist_ok=True)
    run_experiment()