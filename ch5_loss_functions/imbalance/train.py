import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def train_epoch_standard(model, train_loader, optimizer, device):
    """Train one epoch with standard CE."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in tqdm(train_loader, desc="Training (Standard CE)"):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    return {
        'loss': total_loss / len(train_loader),
        'accuracy': correct / total
    }


def train_epoch_weighted(model, train_loader, optimizer, class_weights, device):
    """Train one epoch with weighted CE."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in tqdm(train_loader, desc="Training (Weighted CE)"):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss = F.cross_entropy(logits, target, weight=class_weights)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    return {
        'loss': total_loss / len(train_loader),
        'accuracy': correct / total
    }


def train_epoch_focal(model, train_loader, optimizer, focal_loss_fn, device):
    """Train one epoch with focal loss."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in tqdm(train_loader, desc="Training (Focal Loss)"):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss = focal_loss_fn(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    return {
        'loss': total_loss / len(train_loader),
        'accuracy': correct / total
    }


def evaluate(model, val_loader, device):
    """Evaluate model and return predictions."""
    model.eval()
    all_preds = []
    all_targets = []
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            pred = logits.argmax(dim=1)

            correct += (pred == target).sum().item()
            total += target.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    return {
        'accuracy': correct / total,
        'predictions': np.array(all_preds),
        'targets': np.array(all_targets)
    }