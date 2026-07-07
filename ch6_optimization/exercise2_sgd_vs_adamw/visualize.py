import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_all_methods_comparison():
    """Plot all 4 training methods together."""
    # Load all results
    with open('results/sgd_pure.pkl', 'rb') as f:
        sgd = pickle.load(f)
    with open('results/adamw_pure.pkl', 'rb') as f:
        adamw = pickle.load(f)
    with open('results/sgd_to_adamw.pkl', 'rb') as f:
        sgd_adamw = pickle.load(f)
    with open('results/adamw_to_sgd.pkl', 'rb') as f:
        adamw_sgd = pickle.load(f)

    epochs = range(1, len(sgd['train_loss']) + 1)
    switch_epoch = sgd_adamw['switch_epoch']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    colors = {
        'sgd': '#2ca02c',
        'adamw': '#d62728',
        'sgd_adamw': '#ff7f0e',
        'adamw_sgd': '#1f77b4'
    }

    # Plot 1: Training Loss
    axes[0, 0].plot(epochs, sgd['train_loss'], '-', label='Pure SGD',
                    linewidth=2, color=colors['sgd'], alpha=0.8)
    axes[0, 0].plot(epochs, adamw['train_loss'], '-', label='Pure AdamW',
                    linewidth=2, color=colors['adamw'], alpha=0.8)
    axes[0, 0].plot(epochs, sgd_adamw['train_loss'], '-', label=f'SGD→AdamW (sw@{switch_epoch})',
                    linewidth=2, color=colors['sgd_adamw'], alpha=0.8)
    axes[0, 0].plot(epochs, adamw_sgd['train_loss'], '-', label=f'AdamW→SGD (sw@{switch_epoch})',
                    linewidth=2, color=colors['adamw_sgd'], alpha=0.8)
    axes[0, 0].axvline(switch_epoch, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Training Loss - All Methods', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    axes[0, 1].plot(epochs, sgd['val_loss'], '-', label='Pure SGD',
                    linewidth=2, color=colors['sgd'], alpha=0.8)
    axes[0, 1].plot(epochs, adamw['val_loss'], '-', label='Pure AdamW',
                    linewidth=2, color=colors['adamw'], alpha=0.8)
    axes[0, 1].plot(epochs, sgd_adamw['val_loss'], '-', label=f'SGD→AdamW',
                    linewidth=2, color=colors['sgd_adamw'], alpha=0.8)
    axes[0, 1].plot(epochs, adamw_sgd['val_loss'], '-', label=f'AdamW→SGD',
                    linewidth=2, color=colors['adamw_sgd'], alpha=0.8)
    axes[0, 1].axvline(switch_epoch, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Validation Loss - All Methods', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Training Accuracy
    axes[1, 0].plot(epochs, sgd['train_acc'], '-', label='Pure SGD',
                    linewidth=2, color=colors['sgd'], alpha=0.8)
    axes[1, 0].plot(epochs, adamw['train_acc'], '-', label='Pure AdamW',
                    linewidth=2, color=colors['adamw'], alpha=0.8)
    axes[1, 0].plot(epochs, sgd_adamw['train_acc'], '-', label=f'SGD→AdamW',
                    linewidth=2, color=colors['sgd_adamw'], alpha=0.8)
    axes[1, 0].plot(epochs, adamw_sgd['train_acc'], '-', label=f'AdamW→SGD',
                    linewidth=2, color=colors['adamw_sgd'], alpha=0.8)
    axes[1, 0].axvline(switch_epoch, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Training Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Training Accuracy - All Methods', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Validation Accuracy (THE KEY PLOT!)
    axes[1, 1].plot(epochs, sgd['val_acc'], '-', label='Pure SGD',
                    linewidth=2.5, color=colors['sgd'], alpha=0.8)
    axes[1, 1].plot(epochs, adamw['val_acc'], '-', label='Pure AdamW',
                    linewidth=2.5, color=colors['adamw'], alpha=0.8)
    axes[1, 1].plot(epochs, sgd_adamw['val_acc'], '-', label=f'SGD→AdamW',
                    linewidth=2.5, color=colors['sgd_adamw'], alpha=0.8)
    axes[1, 1].plot(epochs, adamw_sgd['val_acc'], '-', label=f'AdamW→SGD',
                    linewidth=2.5, color=colors['adamw_sgd'], alpha=0.8)
    axes[1, 1].axvline(switch_epoch, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    axes[1, 1].text(switch_epoch, 50, 'Switch Point', rotation=90,
                    verticalalignment='bottom', fontsize=10, color='gray')
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Validation Accuracy - All Methods', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/all_methods_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/all_methods_comparison.png")


def plot_train_val_gap_all():
    """Plot train-val gap for all 4 methods."""
    with open('results/sgd_pure.pkl', 'rb') as f:
        sgd = pickle.load(f)
    with open('results/adamw_pure.pkl', 'rb') as f:
        adamw = pickle.load(f)
    with open('results/sgd_to_adamw.pkl', 'rb') as f:
        sgd_adamw = pickle.load(f)
    with open('results/adamw_to_sgd.pkl', 'rb') as f:
        adamw_sgd = pickle.load(f)

    epochs = range(1, len(sgd['train_acc']) + 1)
    switch_epoch = sgd_adamw['switch_epoch']

    sgd_gap = np.array(sgd['train_acc']) - np.array(sgd['val_acc'])
    adamw_gap = np.array(adamw['train_acc']) - np.array(adamw['val_acc'])
    sgd_adamw_gap = np.array(sgd_adamw['train_acc']) - np.array(sgd_adamw['val_acc'])
    adamw_sgd_gap = np.array(adamw_sgd['train_acc']) - np.array(adamw_sgd['val_acc'])

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    ax.plot(epochs, sgd_gap, '-', label='Pure SGD',
            linewidth=2.5, color='#2ca02c', alpha=0.8)
    ax.plot(epochs, adamw_gap, '-', label='Pure AdamW',
            linewidth=2.5, color='#d62728', alpha=0.8)
    ax.plot(epochs, sgd_adamw_gap, '-', label=f'SGD→AdamW (sw@{switch_epoch})',
            linewidth=2.5, color='#ff7f0e', alpha=0.8)
    ax.plot(epochs, adamw_sgd_gap, '-', label=f'AdamW→SGD (sw@{switch_epoch})',
            linewidth=2.5, color='#1f77b4', alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(switch_epoch, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Train-Val Accuracy Gap (%)', fontsize=13, fontweight='bold')
    ax.set_title('Overfitting Comparison: All Methods', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Annotate final gaps
    final_gaps = [
        (sgd_gap[-1], 'SGD', '#2ca02c'),
        (adamw_gap[-1], 'AdamW', '#d62728'),
        (sgd_adamw_gap[-1], 'SGD→AdamW', '#ff7f0e'),
        (adamw_sgd_gap[-1], 'AdamW→SGD', '#1f77b4')
    ]

    for i, (gap, name, color) in enumerate(final_gaps):
        ax.text(len(epochs) - 3, gap + 0.5 * i, f'{name}: {gap:.2f}%',
                fontsize=10, color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/train_val_gap_all_methods.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/train_val_gap_all_methods.png")


def plot_switch_point_detail():
    """Zoom in on the switch point to see what happens."""
    with open('results/sgd_to_adamw.pkl', 'rb') as f:
        sgd_adamw = pickle.load(f)
    with open('results/adamw_to_sgd.pkl', 'rb') as f:
        adamw_sgd = pickle.load(f)

    switch_epoch = sgd_adamw['switch_epoch']
    start = max(1, switch_epoch - 10)
    end = min(len(sgd_adamw['val_acc']), switch_epoch + 20)
    epochs = range(start, end + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # SGD → AdamW
    axes[0].plot(epochs, sgd_adamw['val_acc'][start - 1:end], 'o-',
                 linewidth=2.5, markersize=6, color='#ff7f0e')
    axes[0].axvline(switch_epoch, color='red', linestyle='--', linewidth=2,
                    label=f'Switch: SGD→AdamW')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('SGD → AdamW: What Happens at Switch?', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].text(switch_epoch - 5, axes[0].get_ylim()[0] + 2, 'SGD Phase',
                 fontsize=11, fontweight='bold', color='#2ca02c')
    axes[0].text(switch_epoch + 2, axes[0].get_ylim()[0] + 2, 'AdamW Phase',
                 fontsize=11, fontweight='bold', color='#d62728')

    # AdamW → SGD
    axes[1].plot(epochs, adamw_sgd['val_acc'][start - 1:end], 's-',
                 linewidth=2.5, markersize=6, color='#1f77b4')
    axes[1].axvline(switch_epoch, color='green', linestyle='--', linewidth=2,
                    label=f'Switch: AdamW→SGD')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('AdamW → SGD: What Happens at Switch?', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].text(switch_epoch - 5, axes[1].get_ylim()[0] + 2, 'AdamW Phase',
                 fontsize=11, fontweight='bold', color='#d62728')
    axes[1].text(switch_epoch + 2, axes[1].get_ylim()[0] + 2, 'SGD Phase',
                 fontsize=11, fontweight='bold', color='#2ca02c')

    plt.tight_layout()
    plt.savefig('results/switch_point_detail.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/switch_point_detail.png")


if __name__ == "__main__":
    print("Generating optimizer switching comparison visualizations...\n")
    print("1. All Methods Comparison")
    plot_all_methods_comparison()
    print("\n2. Train-Val Gap (All Methods)")
    plot_train_val_gap_all()
    print("\n3. Switch Point Detail")
    plot_switch_point_detail()
    print("\n✅ All visualizations complete!")
    plt.show()