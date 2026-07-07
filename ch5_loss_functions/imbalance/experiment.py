import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

from model import SimpleMLP
from imbalance import create_imbalanced_mnist, get_class_distribution
from losses import FocalLoss
from train import train_epoch_standard, train_epoch_weighted, train_epoch_focal, evaluate
from metrics import (compute_confusion_matrix, compute_per_class_metrics,
                     print_metrics_summary, plot_confusion_matrix, compute_macro_metrics)


def compute_class_weights(train_dataset, num_classes=10):
    """
    Compute balanced class weights using formula: N / (K × n_i)
    """
    distribution = get_class_distribution(train_dataset)

    # Total samples
    total = sum(distribution.values())

    # Compute weights
    weights = torch.zeros(num_classes)
    for cls in range(num_classes):
        n_samples = distribution.get(cls, 1)  # Avoid division by zero
        weights[cls] = total / (num_classes * n_samples)

    return weights


def train_model(model, train_loader, val_loader, epochs, loss_type,
                class_weights=None, focal_loss_fn=None, device='cpu'):
    """Train a model with specified loss type."""
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        if loss_type == 'standard':
            train_metrics = train_epoch_standard(model, train_loader, optimizer, device)
        elif loss_type == 'weighted':
            train_metrics = train_epoch_weighted(model, train_loader, optimizer,
                                                 class_weights, device)
        elif loss_type == 'focal':
            train_metrics = train_epoch_focal(model, train_loader, optimizer,
                                              focal_loss_fn, device)

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)

        history.append({
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_acc': val_metrics['accuracy']
        })

        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}")

    # Final evaluation with predictions
    final_results = evaluate(model, val_loader, device)

    return history, final_results


def run_experiment():
    """Main experiment runner."""
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Create imbalanced dataset
    print("=" * 70)
    print("CREATING IMBALANCED MNIST")
    print("=" * 70)
    minority_class = 5
    train_dataset, val_dataset = create_imbalanced_mnist(
        imbalance_ratio=0.1,
        minority_class=minority_class
    )

    # Show distribution
    train_dist = get_class_distribution(train_dataset)
    print("\nTraining set distribution:")
    for cls, count in sorted(train_dist.items()):
        marker = " ⚠️ MINORITY" if cls == minority_class else ""
        print(f"  Class {cls}: {count:5d} samples{marker}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Compute class weights
    class_weights = compute_class_weights(train_dataset).to(device)
    print("\nClass weights:")
    for cls in range(10):
        marker = " ⚠️" if cls == minority_class else ""
        print(f"  Class {cls}: {class_weights[cls]:.3f}{marker}")

    # Storage for results
    all_results = {}

    # ========================================================================
    # Experiment 1: Standard CE
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: STANDARD CROSS-ENTROPY")
    print("=" * 70)

    model_std = SimpleMLP().to(device)
    history_std, results_std = train_model(
        model_std, train_loader, val_loader,
        epochs=10, loss_type='standard', device=device
    )
    all_results['standard'] = results_std

    # ========================================================================
    # Experiment 2: Weighted CE
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: WEIGHTED CROSS-ENTROPY")
    print("=" * 70)

    model_weighted = SimpleMLP().to(device)
    history_weighted, results_weighted = train_model(
        model_weighted, train_loader, val_loader,
        epochs=10, loss_type='weighted',
        class_weights=class_weights, device=device
    )
    all_results['weighted'] = results_weighted

    # ========================================================================
    # Experiment 3: Focal Loss
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: FOCAL LOSS")
    print("=" * 70)

    focal_loss = FocalLoss(alpha=class_weights, gamma=2.0).to(device)
    model_focal = SimpleMLP().to(device)
    history_focal, results_focal = train_model(
        model_focal, train_loader, val_loader,
        epochs=10, loss_type='focal',
        focal_loss_fn=focal_loss, device=device
    )
    all_results['focal'] = results_focal

    # ========================================================================
    # Analysis & Visualization
    # ========================================================================
    analyze_results(all_results, minority_class)


def analyze_results(all_results, minority_class=5):
    """Analyze and visualize results from all experiments."""
    print("\n" + "=" * 70)
    print("FINAL RESULTS COMPARISON")
    print("=" * 70)

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    experiment_names = {
        'standard': 'Standard CE',
        'weighted': 'Weighted CE',
        'focal': 'Focal Loss'
    }

    f1_comparison = {}

    for idx, (exp_name, results) in enumerate(all_results.items()):
        y_true = results['targets']
        y_pred = results['predictions']

        # Compute confusion matrix
        cm = compute_confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        ax = axes[0, idx]
        plot_confusion_matrix(cm, title=experiment_names[exp_name], save_path=None)
        plt.sca(ax)

        # Compute metrics
        metrics = compute_per_class_metrics(y_true, y_pred)
        print_metrics_summary(metrics, minority_class)

        # Extract F1 scores for comparison
        f1_scores = [metrics[cls]['f1'] for cls in range(10)]
        f1_comparison[exp_name] = f1_scores

        # Plot per-class F1
        ax = axes[1, idx]
        colors = ['red' if cls == minority_class else 'blue' for cls in range(10)]
        bars = ax.bar(range(10), f1_scores, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{experiment_names[exp_name]}\nPer-Class F1 Scores',
                     fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(range(10))

        # Add text for minority class
        minority_f1 = f1_scores[minority_class]
        ax.text(minority_class, minority_f1 + 0.05, f'{minority_f1:.3f}',
                ha='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved: results/comparison.png")

    # Summary table
    print("\n" + "=" * 70)
    print("MINORITY CLASS (Class 5) PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'F1 Score':<12} {'Improvement':<15}")
    print("-" * 70)

    baseline_f1 = f1_comparison['standard'][minority_class]
    for exp_name in ['standard', 'weighted', 'focal']:
        f1 = f1_comparison[exp_name][minority_class]
        improvement = ((f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
        print(f"{experiment_names[exp_name]:<20} {f1:>10.3f}  {improvement:>+12.1f}%")

    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    run_experiment()