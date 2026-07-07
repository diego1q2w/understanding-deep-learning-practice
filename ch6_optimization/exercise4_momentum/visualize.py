import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_all_comparisons():
    """Create comprehensive comparison plots."""
    # Load results
    with open('results/all_momentum_results.pkl', 'rb') as f:
        results = pickle.load(f)

    betas = [0.0, 0.5, 0.9, 0.99]
    colors = {0.0: '#e74c3c', 0.5: '#f39c12', 0.9: '#2ecc71', 0.99: '#3498db'}

    # Create figure with 6 subplots
    fig = plt.figure(figsize=(20, 12))

    # ============================================================
    # 1. Convergence Speed (First 15 epochs)
    # ============================================================
    ax1 = plt.subplot(2, 3, 1)
    for beta in betas:
        key = f'beta_{beta}'
        val_accs = results[key]['val_accs'][:15]
        ax1.plot(range(1, len(val_accs) + 1), val_accs,
                 marker='o', linewidth=2, label=f'β={beta}', color=colors[beta])

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax1.set_title('Convergence Speed (First 15 Epochs)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ============================================================
    # 2. Full Training Curves
    # ============================================================
    ax2 = plt.subplot(2, 3, 2)
    for beta in betas:
        key = f'beta_{beta}'
        val_accs = results[key]['val_accs']
        ax2.plot(range(1, len(val_accs) + 1), val_accs,
                 linewidth=2, label=f'β={beta}', color=colors[beta])

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax2.set_title('Full Training (50 Epochs)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ============================================================
    # 3. Velocity Magnitude Over Time
    # ============================================================
    ax3 = plt.subplot(2, 3, 3)
    for beta in betas:
        if beta == 0.0:
            continue  # No momentum, no velocity
        key = f'beta_{beta}'
        velocity = results[key]['epoch_velocity_means']
        ax3.plot(range(1, len(velocity) + 1), velocity,
                 linewidth=2, label=f'β={beta}', color=colors[beta])

    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Mean Velocity Magnitude', fontsize=12)
    ax3.set_title('Velocity Build-up and Decay', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ============================================================
    # 4. Loss Smoothness (First 1000 batches)
    # ============================================================
    ax4 = plt.subplot(2, 3, 4)
    window = 1000
    for beta in betas:
        key = f'beta_{beta}'
        batch_losses = results[key]['batch_losses'][:window]
        ax4.plot(range(len(batch_losses)), batch_losses,
                 alpha=0.7, linewidth=1, label=f'β={beta}', color=colors[beta])

    ax4.set_xlabel('Batch', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.set_title('Loss Smoothness (First 1000 Batches)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # ============================================================
    # 5. Loss Variance Analysis
    # ============================================================
    ax5 = plt.subplot(2, 3, 5)

    variances = []
    labels_list = []
    for beta in betas:
        key = f'beta_{beta}'
        # Compute variance of batch losses in first epoch (~391 batches for CIFAR-10)
        first_epoch_losses = results[key]['batch_losses'][:400]
        variance = np.var(first_epoch_losses)
        variances.append(variance)
        labels_list.append(f'β={beta}')

    bars = ax5.bar(labels_list, variances, color=[colors[b] for b in betas])
    ax5.set_ylabel('Loss Variance', fontsize=12)
    ax5.set_title('Loss Variance (First Epoch)', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    # ============================================================
    # 6. Final Performance Comparison
    # ============================================================
    ax6 = plt.subplot(2, 3, 6)

    best_accs = [max(results[f'beta_{beta}']['val_accs']) for beta in betas]
    final_accs = [results[f'beta_{beta}']['val_accs'][-1] for beta in betas]

    x = np.arange(len(betas))
    width = 0.35

    bars1 = ax6.bar(x - width / 2, best_accs, width, label='Best Val Acc', color='#2ecc71', alpha=0.8)
    bars2 = ax6.bar(x + width / 2, final_accs, width, label='Final Val Acc', color='#3498db', alpha=0.8)

    ax6.set_ylabel('Accuracy (%)', fontsize=12)
    ax6.set_title('Final Performance Summary', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels([f'β={b}' for b in betas])
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('results/momentum_full_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/momentum_full_analysis.png")
    plt.close()


def plot_zigzag_analysis():
    """Visualize the 'zigzag smoothing' effect of momentum."""
    with open('results/all_momentum_results.pkl', 'rb') as f:
        results = pickle.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    betas = [0.0, 0.5, 0.9, 0.99]
    colors = {0.0: '#e74c3c', 0.5: '#f39c12', 0.9: '#2ecc71', 0.99: '#3498db'}

    for idx, beta in enumerate(betas):
        ax = axes[idx // 2, idx % 2]
        key = f'beta_{beta}'

        # Plot batch losses for first 500 batches
        batch_losses = results[key]['batch_losses'][:500]
        ax.plot(batch_losses, alpha=0.6, linewidth=0.8, color=colors[beta], label='Raw loss')

        # Plot moving average
        window = 20
        moving_avg = np.convolve(batch_losses, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(batch_losses)), moving_avg,
                linewidth=2, color='black', label='Moving avg (20)')

        ax.set_xlabel('Batch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title(f'β={beta} - {"No Momentum" if beta == 0 else f"Momentum {beta}"}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Loss Smoothness: Effect of Momentum on Batch-to-Batch Variation',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('results/momentum_zigzag_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/momentum_zigzag_analysis.png")
    plt.close()


if __name__ == '__main__':
    plot_all_comparisons()
    plot_zigzag_analysis()
    print("\n🎉 All visualizations complete!")
