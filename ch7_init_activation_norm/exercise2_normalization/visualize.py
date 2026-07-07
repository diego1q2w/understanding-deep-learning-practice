import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_curves(results, output_dir='results'):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Normalization Comparison (ReLU + Xavier Init)',
                 fontsize=16, fontweight='bold')

    colors = {'none': 'red', 'batch': 'green', 'layer': 'blue'}

    for result in results:
        norm = result['norm_type']
        color = colors[norm]
        epochs = range(1, len(result['train_acc']) + 1)

        # Train accuracy
        axes[0, 0].plot(epochs, result['train_acc'],
                        label=norm.upper(), color=color, linewidth=2)

        # Val accuracy
        axes[0, 1].plot(epochs, result['val_acc'],
                        label=norm.upper(), color=color, linewidth=2)

        # Train loss
        axes[1, 0].plot(epochs, result['train_loss'],
                        label=norm.upper(), color=color, linewidth=2)

        # Val loss
        axes[1, 1].plot(epochs, result['val_loss'],
                        label=norm.upper(), color=color, linewidth=2)

    axes[0, 0].set_title('Training Accuracy', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Validation Accuracy', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('Training Loss', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('Validation Loss', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_activation_variance_evolution(results, output_dir='results'):
    """Plot how activation variance evolves during training."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Activation Variance Across Layers Over Training',
                 fontsize=16, fontweight='bold')

    norm_names = {'none': 'No Norm', 'batch': 'BatchNorm', 'layer': 'LayerNorm'}

    for idx, result in enumerate(results):
        ax = axes[idx]
        norm = result['norm_type']

        # Plot variance at epochs 0, 5, 10, 20
        epochs_to_plot = [0, 5, 10, 19]  # Indices
        colors = plt.cm.viridis(np.linspace(0, 1, len(epochs_to_plot)))

        for i, epoch_idx in enumerate(epochs_to_plot):
            if epoch_idx < len(result['activation_stats']):
                stats = result['activation_stats'][epoch_idx]
                layers = [s['layer'] for s in stats]
                variances = [s['var'] for s in stats]

                ax.plot(layers, variances, 'o-', color=colors[i],
                        label=f'Epoch {epoch_idx}', linewidth=2, markersize=4)

        ax.set_title(norm_names[norm], fontsize=13, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Variance', fontsize=12)
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_path = Path(output_dir) / 'activation_variance_evolution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_gradient_flow(results, output_dir='results'):
    """Plot gradient norms at final epoch."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Gradient Flow at Final Epoch',
                 fontsize=16, fontweight='bold')

    norm_names = {'none': 'No Norm', 'batch': 'BatchNorm', 'layer': 'LayerNorm'}

    for idx, result in enumerate(results):
        ax = axes[idx]
        norm = result['norm_type']

        # Get final gradient stats
        final_grad_stats = result['gradient_stats'][-1]
        layers = [s['layer'] for s in final_grad_stats]
        grad_norms = [s['grad_norm'] for s in final_grad_stats]

        ax.plot(layers, grad_norms, 's-', color='purple',
                linewidth=2, markersize=6)
        ax.set_title(norm_names[norm], fontsize=13, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Gradient Norm', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'gradient_flow.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    # Load results
    results_file = Path('results/training_comparison.pkl')
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    print("Generating visualizations...")
    plot_training_curves(results)
    plot_activation_variance_evolution(results)
    plot_gradient_flow(results)
    print("\n✅ All visualizations generated!")