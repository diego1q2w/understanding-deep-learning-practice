import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from resnet20 import RestNet20
from config import ModelConfig, OptimizerConfig, DataConfig, SchedulerConfig


def create_model(model_config: ModelConfig, device: str = 'cpu') -> nn.Module:
    """
    Factory function to create model from config.

    Args:
        model_config: Model configuration
        device: Device to put model on

    Returns:
        Initialized model
    """
    if model_config.name == "resnet20":
        model = RestNet20(
            net_type=model_config.net_type,
            num_classes=model_config.num_classes
        )
    else:
        raise ValueError(f"Unknown model: {model_config.name}")

    model = model.to(device)
    return model


def create_optimizer(optimizer_config: OptimizerConfig,
                     model_parameters) -> optim.Optimizer:
    """
    Factory function to create optimizer from config.

    Args:
        optimizer_config: Optimizer configuration
        model_parameters: Model parameters (from model.parameters())

    Returns:
        Initialized optimizer
    """
    if optimizer_config.name == "sgd":
        optimizer = optim.SGD(
            model_parameters,
            lr=optimizer_config.lr,
            momentum=optimizer_config.momentum,
            weight_decay=optimizer_config.weight_decay
        )
    elif optimizer_config.name == "adam":
        optimizer = optim.Adam(
            model_parameters,
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay
        )
    elif optimizer_config.name == "adamw":
        optimizer = optim.AdamW(
            model_parameters,
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config.name}")

    return optimizer


def create_dataloaders(data_config: DataConfig,
                       batch_size: int) -> tuple[DataLoader, DataLoader]:
    """
    Factory function to create train/val dataloaders from config.

    Args:
        data_config: Data configuration
        batch_size: Batch size for dataloaders

    Returns:
        (train_loader, val_loader)
    """
    if data_config.name == "cifar10":
        # Define transforms
        if data_config.augmentation:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

        # Load datasets
        train_dataset = datasets.CIFAR10(
            root=data_config.root,
            train=True,
            download=True,
            transform=train_transform
        )

        val_dataset = datasets.CIFAR10(
            root=data_config.root,
            train=False,
            download=True,
            transform=val_transform
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False if torch.backends.mps.is_available() else True
           # num_workers=data_config.num_workers,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False if torch.backends.mps.is_available() else True
            #num_workers=data_config.num_workers,
        )

        return train_loader, val_loader
    else:
        raise ValueError(f"Unknown dataset: {data_config.name}")


def create_scheduler(scheduler_config: SchedulerConfig,
                     optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler:
    """
    Factory function to create learning rate scheduler from config.

    Args:
        scheduler_config: Scheduler configuration
        optimizer: Optimizer to schedule

    Returns:
        Learning rate scheduler or None
    """
    if scheduler_config.name is None:
        return None

    if scheduler_config.name == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.step_size,
            gamma=scheduler_config.gamma
        )
    elif scheduler_config.name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.step_size
        )
    elif scheduler_config.name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_config.milestones or [30, 60, 90],
            gamma=scheduler_config.gamma
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_config.name}")

    return scheduler


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False