import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_comparison():
    """Create comprehensive comparison plots."""

    # Load results
    with open('results/all_batch_sizes.pkl', 'rb') as f:
        results = pickle.load(f)

    batch_sizes = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(batch_sizes)))

    fig = plt.figure(figsize=(20, 12))

    # ============================================================
    # Plot 1: Training Loss vs Steps
    # ============================================================
    ax1 = plt.subplot(2, 3, 1)
    for batch_size, color in zip(batch_sizes, colors):
        history = results[batch_size]
        steps_per_epoch = history['steps_per_epoch']
        steps = np.arange(len(history['train_loss'])) * steps_per_epoch
        ax1.plot(steps, history['train_loss'], label=f'BS={batch_size}', color=color, linewidth=2)

    ax1.set_xlabel('Optimizer Steps', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss vs Optimizer Steps', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ============================================================
    # Plot 2: Validation Loss vs Steps
    # ============================================================
    ax2 = plt.subplot(2, 3, 2)
    for batch_size, color in zip(batch_sizes, colors):
        history = results[batch_size]
        steps_per_epoch = history['steps_per_epoch']
        steps = np.arange(len(history['val_loss'])) * steps_per_epoch
        ax2.plot(steps, history['val_loss'], label=f'BS={batch_size}', color=color, linewidth=2)

    ax2.set_xlabel('Optimizer Steps', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss vs Optimizer Steps', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ============================================================
    # Plot 3: Training Accuracy vs Steps
    # ============================================================
    ax3 = plt.subplot(2, 3, 4)
    for batch_size, color in zip(batch_sizes, colors):
        history = results[batch_size]
        steps_per_epoch = history['steps_per_epoch']
        steps = np.arange(len(history['train_acc'])) * steps_per_epoch
        ax3.plot(steps, history['train_acc'], label=f'BS={batch_size}', color=color, linewidth=2)

    ax3.set_xlabel('Optimizer Steps', fontsize=12)
    ax3.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax3.set_title('Training Accuracy vs Optimizer Steps', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ============================================================
    # Plot 4: Validation Accuracy vs Steps
    # ============================================================
    ax4 = plt.subplot(2, 3, 5)
    for batch_size, color in zip(batch_sizes, colors):
        history = results[batch_size]
        steps_per_epoch = history['steps_per_epoch']
        steps = np.arange(len(history['val_acc'])) * steps_per_epoch
        ax4.plot(steps, history['val_acc'], label=f'BS={batch_size}', color=color, linewidth=2)

    ax4.set_xlabel('Optimizer Steps', fontsize=12)
    ax4.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax4.set_title('Validation Accuracy vs Optimizer Steps', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ============================================================
    # Plot 5: Generalization Gap (Train - Val Acc)
    # ============================================================
    ax5 = plt.subplot(2, 3, 3)
    for batch_size, color in zip(batch_sizes, colors):
        history = results[batch_size]
        steps_per_epoch = history['steps_per_epoch']
        steps = np.arange(len(history['train_acc'])) * steps_per_epoch
        gap = np.array(history['train_acc']) - np.array(history['val_acc'])
        ax5.plot(steps, gap, label=f'BS={batch_size}', color=color, linewidth=2)

    ax5.set_xlabel('Optimizer Steps', fontsize=12)
    ax5.set_ylabel('Train-Val Gap (%)', fontsize=12)
    ax5.set_title('Generalization Gap (Train - Val)', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # ============================================================
    # Plot 6: Final Metrics Bar Chart
    # ============================================================
    ax6 = plt.subplot(2, 3, 6)

    final_train_acc = [results[bs]['train_acc'][-1] for bs in batch_sizes]
    final_val_acc = [results[bs]['val_acc'][-1] for bs in batch_sizes]
    final_gap = [results[bs]['train_acc'][-1] - results[bs]['val_acc'][-1] for bs in batch_sizes]

    x = np.arange(len(batch_sizes))
    width = 0.25

    ax6.bar(x - width, final_train_acc, width, label='Train Acc', color='steelblue', alpha=0.8)
    ax6.bar(x, final_val_acc, width, label='Val Acc', color='coral', alpha=0.8)
    ax6.bar(x + width, final_gap, width, label='Gap', color='lightgreen', alpha=0.8)

    ax6.set_xlabel('Batch Size', fontsize=12)
    ax6.set_ylabel('Accuracy / Gap (%)', fontsize=12)
    ax6.set_title('Final Metrics Comparison', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(batch_sizes)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('results/batch_size_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/batch_size_comparison.png")
    plt.show()


def plot_noise_analysis():
    """Analyze gradient noise by looking at loss curve smoothness."""

    with open('results/all_batch_sizes.pkl', 'rb') as f:
        results = pickle.load(f)

    batch_sizes = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(batch_sizes)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (batch_size, color) in enumerate(zip(batch_sizes, colors)):
        history = results[batch_size]

        # Plot last 20% of training to see noise clearly
        start_idx = int(len(history['val_loss']) * 0.8)
        val_loss = history['val_loss'][start_idx:]
        epochs = np.arange(len(val_loss))

        axes[idx].plot(epochs, val_loss, color=color, linewidth=2, alpha=0.7)
        axes[idx].set_title(f'Batch Size = {batch_size}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Epoch (last 20%)')
        axes[idx].set_ylabel('Val Loss')
        axes[idx].grid(True, alpha=0.3)

        # Calculate "smoothness" (inverse of std)
        smoothness = 1.0 / (np.std(val_loss) + 1e-8)
        axes[idx].text(0.05, 0.95, f'Smoothness: {smoothness:.1f}',
                       transform=axes[idx].transAxes,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       verticalalignment='top')

    # Remove extra subplot if odd number
    if len(batch_sizes) < 6:
        fig.delaxes(axes[-1])

    plt.suptitle('Validation Loss Smoothness (Last 20% of Training)',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('results/noise_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/noise_analysis.png")
    plt.show()


def print_summary():
    """Print summary statistics."""

    with open('results/all_batch_sizes.pkl', 'rb') as f:
        results = pickle.load(f)

    print(f"\n{'=' * 70}")
    print(f"{'BATCH SIZE EXPERIMENT SUMMARY':^70}")
    print(f"{'=' * 70}\n")

    print(f"{'BS':<6} {'Epochs':<8} {'Steps':<8} {'LR Scale':<10} {'Train Acc':<12} {'Val Acc':<12} {'Gap':<8}")
    print(f"{'-' * 70}")

    for batch_size in sorted(results.keys()):
        history = results[batch_size]
        epochs = history['epochs']
        total_steps = history['total_steps']
        lr_scale = min(batch_size / 128, 4.0)
        train_acc = history['train_acc'][-1]
        val_acc = history['val_acc'][-1]
        gap = train_acc - val_acc

        print(f"{batch_size:<6} {epochs:<8} {total_steps:<8} {lr_scale:<10.2f} "
              f"{train_acc:<12.2f} {val_acc:<12.2f} {gap:<8.2f}")

    print(f"\n{'=' * 70}\n")


if __name__ == '__main__':
    print_summary()
    plot_comparison()
    plot_noise_analysis()