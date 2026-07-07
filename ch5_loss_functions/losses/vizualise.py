import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_learning_curves():
    """Plot 1: Learning curves comparison"""
    # Load results
    with open('results/results_ce.pkl', 'rb') as f:
        ce = pickle.load(f)
    with open('results/results_mse.pkl', 'rb') as f:  # Fix the double /results/
        mse = pickle.load(f)

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs_ce = range(1, len(ce['train_loss']) + 1)
    epochs_mse = range(1, len(mse['train_loss']) + 1)

    # Plot 1: Training Loss
    axes[0, 0].plot(epochs_ce, ce['train_loss'], 'o-', label='Cross-Entropy',
                    linewidth=2, markersize=5, color='#1f77b4')
    axes[0, 0].plot(epochs_mse, mse['train_loss'], 's-', label='MSE',
                    linewidth=2, markersize=5, color='#ff7f0e')
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    axes[0, 1].plot(epochs_ce, ce['val_loss'], 'o-', label='Cross-Entropy',
                    linewidth=2, markersize=5, color='#1f77b4')
    axes[0, 1].plot(epochs_mse, mse['val_loss'], 's-', label='MSE',
                    linewidth=2, markersize=5, color='#ff7f0e')
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Training Accuracy
    axes[1, 0].plot(epochs_ce, np.array(ce['train_acc']) * 100, 'o-',
                    label='Cross-Entropy', linewidth=2, markersize=5, color='#1f77b4')
    axes[1, 0].plot(epochs_mse, np.array(mse['train_acc']) * 100, 's-',
                    label='MSE', linewidth=2, markersize=5, color='#ff7f0e')
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Training Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Validation Accuracy
    axes[1, 1].plot(epochs_ce, np.array(ce['val_acc']) * 100, 'o-',
                    label='Cross-Entropy', linewidth=2, markersize=5, color='#1f77b4')
    axes[1, 1].plot(epochs_mse, np.array(mse['val_acc']) * 100, 's-',
                    label='MSE', linewidth=2, markersize=5, color='#ff7f0e')
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/learning_curves.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/learning_curves.png")


def plot_confidence_distribution():
    """Plot 2: Confidence histograms"""
    # Load results
    with open('results/results_ce.pkl', 'rb') as f:
        ce = pickle.load(f)
    with open('results/results_mse.pkl', 'rb') as f:
        mse = pickle.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (results, name, color) in enumerate([
        (ce, 'Cross-Entropy', '#1f77b4'),
        (mse, 'MSE', '#ff7f0e')
    ]):
        # Get predictions and targets
        probs = results['final_predictions']  # Shape: [N, 10]
        targets = results['final_targets']  # Shape: [N]

        # Get max probability for each sample
        max_probs, pred_classes = probs.max(dim=1)

        # Separate correct vs incorrect
        correct_mask = (pred_classes == targets)
        correct_conf = max_probs[correct_mask].numpy()
        incorrect_conf = max_probs[~correct_mask].numpy()

        # Plot correct predictions (left column)
        axes[idx, 0].hist(correct_conf, bins=50, alpha=0.7, color=color,
                          edgecolor='black', linewidth=0.5)
        axes[idx, 0].set_xlabel('Max Predicted Probability', fontsize=11, fontweight='bold')
        axes[idx, 0].set_ylabel('Count', fontsize=11, fontweight='bold')
        axes[idx, 0].set_title(f'{name}\nCorrect Predictions (n={len(correct_conf)})',
                               fontsize=12, fontweight='bold')
        axes[idx, 0].axvline(correct_conf.mean(), color='red', linestyle='--',
                             linewidth=2, label=f'Mean: {correct_conf.mean():.3f}')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3, axis='y')

        # Plot incorrect predictions (right column)
        if len(incorrect_conf) > 0:
            axes[idx, 1].hist(incorrect_conf, bins=50, alpha=0.7, color=color,
                              edgecolor='black', linewidth=0.5)
            axes[idx, 1].set_xlabel('Max Predicted Probability', fontsize=11, fontweight='bold')
            axes[idx, 1].set_ylabel('Count', fontsize=11, fontweight='bold')
            axes[idx, 1].set_title(f'{name}\nIncorrect Predictions (n={len(incorrect_conf)})',
                                   fontsize=12, fontweight='bold')
            axes[idx, 1].axvline(incorrect_conf.mean(), color='red', linestyle='--',
                                 linewidth=2, label=f'Mean: {incorrect_conf.mean():.3f}')
            axes[idx, 1].legend()
            axes[idx, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('results/confidence_distribution.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/confidence_distribution.png")


def plot_prediction_scatter():
    """Plot 3: Prediction quality scatter"""
    # Load results
    with open('results/results_ce.pkl', 'rb') as f:
        ce = pickle.load(f)
    with open('results/results_mse.pkl', 'rb') as f:
        mse = pickle.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (results, name, color_correct, color_wrong) in enumerate([
        (ce, 'Cross-Entropy', '#2ca02c', '#d62728'),
        (mse, 'MSE', '#2ca02c', '#d62728')
    ]):
        probs = results['final_predictions']  # Shape: [N, 10]
        targets = results['final_targets']  # Shape: [N]

        # Get predicted class and check correctness
        pred_classes = probs.argmax(dim=1)
        correct_mask = (pred_classes == targets)

        # For each sample, get the probability of the TRUE class
        true_class_probs = probs[torch.arange(len(targets)), targets]

        # Separate correct and incorrect
        correct_indices = torch.where(correct_mask)[0]
        incorrect_indices = torch.where(~correct_mask)[0]

        # Add jitter to x-axis for visibility
        jitter = 0.15

        # Plot incorrect predictions first (so they're behind)
        if len(incorrect_indices) > 0:
            x_wrong = targets[incorrect_indices].numpy() + np.random.uniform(-jitter, jitter, len(incorrect_indices))
            y_wrong = true_class_probs[incorrect_indices].numpy()
            axes[idx].scatter(x_wrong, y_wrong, c=color_wrong, alpha=0.6,
                              s=20, label=f'Incorrect ({len(incorrect_indices)})',
                              edgecolors='black', linewidth=0.3)

        # Plot correct predictions on top
        if len(correct_indices) > 0:
            x_correct = targets[correct_indices].numpy() + np.random.uniform(-jitter, jitter, len(correct_indices))
            y_correct = true_class_probs[correct_indices].numpy()
            axes[idx].scatter(x_correct, y_correct, c=color_correct, alpha=0.4,
                              s=20, label=f'Correct ({len(correct_indices)})',
                              edgecolors='black', linewidth=0.3)

        axes[idx].set_xlabel('True Class', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Predicted Probability of True Class', fontsize=12, fontweight='bold')
        axes[idx].set_title(f'{name}\nPrediction Quality by Class',
                            fontsize=14, fontweight='bold')
        axes[idx].set_xticks(range(10))
        axes[idx].set_ylim([-0.05, 1.05])
        axes[idx].axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3)

        # Add accuracy text
        accuracy = correct_mask.float().mean() * 100
        axes[idx].text(0.02, 0.98, f'Accuracy: {accuracy:.2f}%',
                       transform=axes[idx].transAxes, fontsize=11,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('results/prediction_scatter.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/prediction_scatter.png")


if __name__ == "__main__":
    print("Generating visualizations...")
    print("\n1. Learning Curves")
    plot_learning_curves()
    print("\n2. Confidence Distribution")
    plot_confidence_distribution()
    print("\n3. Prediction Scatter")
    plot_prediction_scatter()
    print("\n✅ All visualizations complete!")
    plt.show()