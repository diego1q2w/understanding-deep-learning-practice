import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset
import numpy as np


def create_imbalanced_mnist(root='./data', imbalance_ratio=0.1, minority_class=5):
    """
    Create imbalanced MNIST by subsampling one class.

    Args:
        imbalance_ratio: Keep this fraction of minority class (0.1 = 10%)
        minority_class: Which digit to make rare (default: 5)

    Returns:
        train_dataset, val_dataset (both imbalanced)
    """
    # Load original MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_full = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    val_full = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    # Create imbalanced train set
    train_imbalanced = _create_imbalanced_subset(train_full, imbalance_ratio, minority_class)
    val_imbalanced = _create_imbalanced_subset(val_full, imbalance_ratio, minority_class)

    return train_imbalanced, val_imbalanced


def _create_imbalanced_subset(dataset, imbalance_ratio, minority_class):
    """Helper to create imbalanced subset."""
    # Get all indices
    targets = np.array(dataset.targets)

    # TODO: Find indices of minority class
    minority_indices = np.where(targets == minority_class)[0]
    majority_indices = np.where(targets != minority_class)[0]

    # TODO: Randomly sample only imbalance_ratio of minority class
    np.random.seed(42)
    n_minority_keep = int(len(minority_indices) * imbalance_ratio)
    minority_sampled = np.random.choice(minority_indices, n_minority_keep, replace=False)

    # TODO: Combine indices
    all_indices = np.concatenate([majority_indices, minority_sampled])
    np.random.shuffle(all_indices)

    # Create subset
    return Subset(dataset, all_indices)


def get_class_distribution(dataset):
    """Get class distribution for analysis."""
    if isinstance(dataset, Subset):
        targets = np.array([dataset.dataset.targets[i] for i in dataset.indices])
    else:
        targets = np.array(dataset.targets)

    unique, counts = np.unique(targets, return_counts=True)
    distribution = dict(zip(unique, counts))

    return distribution


# Quick test
if __name__ == "__main__":
    train_imb, val_imb = create_imbalanced_mnist(imbalance_ratio=0.1, minority_class=5)

    print("Training set distribution:")
    train_dist = get_class_distribution(train_imb)
    for cls, count in sorted(train_dist.items()):
        print(f"  Class {cls}: {count:5d} samples")

    print(f"\nClass 5 was reduced to ~{train_dist[5]} samples "
          f"(from ~{train_dist[0]} for other classes)")