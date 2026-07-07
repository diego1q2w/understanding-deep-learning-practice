import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_activation_variance(results, output_dir='results'):
    """Plot activation variance across layers for all combinations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Activation Variance Across Layers (After 10 Steps)', fontsize=16, fontweight='bold')

    activations = ['relu', 'tanh', 'gelu']
    init_methods = ['xavier', 'he']

    for i, activation in enumerate(activations):
        for j, init_method in enumerate(init_methods):
            ax = axes[j, i]

            # Find matching result
            result = next(r for r in results if r['activation'] == activation and r['init_method'] == init_method)

            layers = [stat['layer'] for stat in result['act_stats_trained']]
            variances = [stat['var'] for stat in result['act_stats_trained']]

            ax.plot(layers, variances, 'o-', linewidth=2, markersize=6)
            ax.set_xlabel('Layer', fontsize=12)
            ax.set_ylabel('Variance', fontsize=12)
            ax.set_title(f'{activation.upper()} + {init_method.upper()}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            # Add horizontal reference line at 1.0
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Target (1.0)')

            # Color code by stability
            if result['var_ratio'] < 0.01:
                ax.set_facecolor('#ffcccc')  # Red for vanishing
                ax.text(0.5, 0.95, '⚠️ VANISHING', transform=ax.transAxes,
                        ha='center', va='top', fontsize=11, color='red', fontweight='bold')
            elif result['var_ratio'] > 100:
                ax.set_facecolor('#ffffcc')  # Yellow for exploding
                ax.text(0.5, 0.95, '⚠️ EXPLODING', transform=ax.transAxes,
                        ha='center', va='top', fontsize=11, color='orange', fontweight='bold')
            else:
                ax.set_facecolor('#ccffcc')  # Green for stable
                ax.text(0.5, 0.95, '✅ STABLE', transform=ax.transAxes,
                        ha='center', va='top', fontsize=11, color='green', fontweight='bold')

    plt.tight_layout()
    output_path = Path(output_dir) / 'activation_variance.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_gradient_norms(results, output_dir='results'):
    """Plot gradient norms across layers for all combinations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Gradient Norms Across Layers (After 10 Steps)', fontsize=16, fontweight='bold')

    activations = ['relu', 'tanh', 'gelu']
    init_methods = ['xavier', 'he']

    for i, activation in enumerate(activations):
        for j, init_method in enumerate(init_methods):
            ax = axes[j, i]

            # Find matching result
            result = next(r for r in results if r['activation'] == activation and r['init_method'] == init_method)

            layers = [stat['layer'] for stat in result['grad_stats_trained']]
            grad_norms = [stat['grad_norm'] for stat in result['grad_stats_trained']]

            ax.plot(layers, grad_norms, 's-', linewidth=2, markersize=6, color='purple')
            ax.set_xlabel('Layer', fontsize=12)
            ax.set_ylabel('Gradient Norm', fontsize=12)
            ax.set_title(f'{activation.upper()} + {init_method.upper()}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            # Color code by stability
            if result['grad_ratio'] > 100:
                ax.set_facecolor('#ffcccc')  # Red for vanishing
                ax.text(0.5, 0.95, '⚠️ VANISHING', transform=ax.transAxes,
                        ha='center', va='top', fontsize=11, color='red', fontweight='bold')
            elif result['grad_ratio'] < 0.01:
                ax.set_facecolor('#ffffcc')  # Yellow for exploding
                ax.text(0.5, 0.95, '⚠️ EXPLODING', transform=ax.transAxes,
                        ha='center', va='top', fontsize=11, color='orange', fontweight='bold')
            else:
                ax.set_facecolor('#ccffcc')  # Green for stable
                ax.text(0.5, 0.95, '✅ STABLE', transform=ax.transAxes,
                        ha='center', va='top', fontsize=11, color='green', fontweight='bold')

    plt.tight_layout()
    output_path = Path(output_dir) / 'gradient_norms.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_summary_comparison(results, output_dir='results'):
    """Create a bar chart comparing all combinations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Stability Comparison: All Combinations', fontsize=16, fontweight='bold')

    labels = [f"{r['activation']}\n{r['init_method']}" for r in results]
    var_ratios = [r['var_ratio'] for r in results]
    grad_ratios = [r['grad_ratio'] for r in results]

    x = np.arange(len(labels))
    width = 0.6

    # Variance ratio
    colors_var = ['green' if 0.1 < vr < 10 else 'red' for vr in var_ratios]
    ax1.bar(x, var_ratios, width, color=colors_var, alpha=0.7, edgecolor='black')
    ax1.axhline(y=1.0, color='blue', linestyle='--', linewidth=2, label='Ideal (1.0)')
    ax1.axhspan(0.1, 10, alpha=0.1, color='green', label='Stable Range')
    ax1.set_ylabel('Variance Ratio (Layer 20 / Layer 1)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Activation + Init', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0, ha='center')
    ax1.set_yscale('log')
    ax1.set_title('Activation Variance Stability', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Gradient ratio
    colors_grad = ['green' if 0.1 < gr < 10 else 'red' for gr in grad_ratios]
    ax2.bar(x, grad_ratios, width, color=colors_grad, alpha=0.7, edgecolor='black')
    ax2.axhline(y=1.0, color='blue', linestyle='--', linewidth=2, label='Ideal (1.0)')
    ax2.axhspan(0.1, 10, alpha=0.1, color='green', label='Stable Range')
    ax2.set_ylabel('Gradient Ratio (Layer 1 / Layer 20)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Activation + Init', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=0, ha='center')
    ax2.set_yscale('log')
    ax2.set_title('Gradient Flow Stability', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = Path(output_dir) / 'stability_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    # Load results
    results_file = Path('results/stability_analysis.pkl')
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    print("Generating visualizations...")
    plot_activation_variance(results)
    plot_gradient_norms(results)
    plot_summary_comparison(results)
    print("\n✅ All visualizations generated!")