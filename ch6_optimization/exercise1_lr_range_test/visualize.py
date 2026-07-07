import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves():
    """Plot training and validation curves for both methods."""
    # Load results
    with open('results/constant_lr.pkl', 'rb') as f:
        constant = pickle.load(f)
    with open('results/cosine_annealing.pkl', 'rb') as f:
        cosine = pickle.load(f)

    epochs = range(1, len(constant['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Training Loss
    axes[0, 0].plot(epochs, constant['train_loss'], 'o-', label='Constant LR (0.02)',
                    linewidth=2, markersize=5, color='#ff7f0e')
    axes[0, 0].plot(epochs, cosine['train_loss'], 's-', label='Cosine Annealing (0.1→0)',
                    linewidth=2, markersize=5, color='#1f77b4')
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    axes[0, 1].plot(epochs, constant['val_loss'], 'o-', label='Constant LR',
                    linewidth=2, markersize=5, color='#ff7f0e')
    axes[0, 1].plot(epochs, cosine['val_loss'], 's-', label='Cosine Annealing',
                    linewidth=2, markersize=5, color='#1f77b4')
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Training Accuracy
    axes[1, 0].plot(epochs, constant['train_acc'], 'o-', label='Constant LR',
                    linewidth=2, markersize=5, color='#ff7f0e')
    axes[1, 0].plot(epochs, cosine['train_acc'], 's-', label='Cosine Annealing',
                    linewidth=2, markersize=5, color='#1f77b4')
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Training Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Validation Accuracy
    axes[1, 1].plot(epochs, constant['val_acc'], 'o-', label='Constant LR',
                    linewidth=2, markersize=5, color='#ff7f0e')
    axes[1, 1].plot(epochs, cosine['val_acc'], 's-', label='Cosine Annealing',
                    linewidth=2, markersize=5, color='#1f77b4')
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/learning_curves_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/learning_curves_comparison.png")


def plot_lr_schedule():
    """Plot the learning rate schedule over epochs."""
    with open('results/constant_lr.pkl', 'rb') as f:
        constant = pickle.load(f)
    with open('results/cosine_annealing.pkl', 'rb') as f:
        cosine = pickle.load(f)

    epochs = range(1, len(constant['lr']) + 1)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.plot(epochs, constant['lr'], 'o-', label='Constant LR',
            linewidth=2.5, markersize=6, color='#ff7f0e')
    ax.plot(epochs, cosine['lr'], 's-', label='Cosine Annealing',
            linewidth=2.5, markersize=6, color='#1f77b4')
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=13, fontweight='bold')
    ax.set_title('Learning Rate Schedule Comparison', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale to see cosine decay clearly

    plt.tight_layout()
    plt.savefig('results/lr_schedule.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/lr_schedule.png")


def plot_early_progress():
    """Zoom in on first 5 epochs to see early convergence speed."""
    with open('results/constant_lr.pkl', 'rb') as f:
        constant = pickle.load(f)
    with open('results/cosine_annealing.pkl', 'rb') as f:
        cosine = pickle.load(f)

    epochs = range(1, 6)  # First 5 epochs

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Validation Loss (first 5 epochs)
    axes[0].plot(epochs, constant['val_loss'][:5], 'o-', label='Constant LR',
                 linewidth=2.5, markersize=8, color='#ff7f0e')
    axes[0].plot(epochs, cosine['val_loss'][:5], 's-', label='Cosine Annealing',
                 linewidth=2.5, markersize=8, color='#1f77b4')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Early Convergence Speed: Validation Loss', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Validation Accuracy (first 5 epochs)
    axes[1].plot(epochs, constant['val_acc'][:5], 'o-', label='Constant LR',
                 linewidth=2.5, markersize=8, color='#ff7f0e')
    axes[1].plot(epochs, cosine['val_acc'][:5], 's-', label='Cosine Annealing',
                 linewidth=2.5, markersize=8, color='#1f77b4')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Early Convergence Speed: Validation Accuracy', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/early_progress.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/early_progress.png")


if __name__ == "__main__":
    print("Generating comparison visualizations...\n")
    print("1. Learning Curves (all epochs)")
    plot_learning_curves()
    print("\n2. LR Schedule")
    plot_lr_schedule()
    print("\n3. Early Progress (first 5 epochs)")
    plot_early_progress()
    print("\n✅ All visualizations complete!")
    plt.show()