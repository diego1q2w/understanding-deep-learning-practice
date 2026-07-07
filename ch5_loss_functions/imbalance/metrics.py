import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=range(num_classes))


def compute_per_class_metrics(y_true, y_pred, num_classes=10):
    """
    Compute precision, recall, F1 for each class.

    Returns:
        Dictionary with per-class metrics
    """
    metrics = {}

    for cls in range(num_classes):
        # TODO: For each class, compute TP, FP, FN, TN
        # tp = number of samples of class `cls` correctly predicted as `cls`
        # fp = number of samples NOT of class `cls` but predicted as `cls`
        # fn = number of samples of class `cls` but predicted as something else

        # Hint: Use boolean masks
        tp = ((y_true == cls) & (y_pred == cls)).sum()
        fp = ((y_true != cls) & (y_pred == cls)).sum()
        fn = ((y_true == cls) & (y_pred != cls)).sum()

        # TODO: Compute precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': (y_true == cls).sum()  # Number of actual samples
        }

    return metrics


def compute_macro_metrics(per_class_metrics):
    """Compute macro-averaged metrics (treats all classes equally)."""
    precisions = [m['precision'] for m in per_class_metrics.values()]
    recalls = [m['recall'] for m in per_class_metrics.values()]
    f1s = [m['f1'] for m in per_class_metrics.values()]

    return {
        'macro_precision': np.mean(precisions),
        'macro_recall': np.mean(recalls),
        'macro_f1': np.mean(f1s)
    }


def plot_confusion_matrix(cm, title="Confusion Matrix", save_path=None):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
    plt.ylabel('True Class', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return plt.gcf()


def print_metrics_summary(metrics, minority_class=5):
    """Print a nice summary of metrics."""
    print("\n" + "=" * 70)
    print("PER-CLASS METRICS")
    print("=" * 70)
    print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 70)

    for cls, m in sorted(metrics.items()):
        marker = " ⚠️ MINORITY" if cls == minority_class else ""
        print(f"{cls:<8} {m['precision']:>10.3f}  {m['recall']:>10.3f}  "
              f"{m['f1']:>10.3f}  {m['support']:>8}{marker}")

    # Macro average
    macro = compute_macro_metrics(metrics)
    print("-" * 70)
    print(f"{'MACRO':<8} {macro['macro_precision']:>10.3f}  "
          f"{macro['macro_recall']:>10.3f}  {macro['macro_f1']:>10.3f}")
    print("=" * 70)