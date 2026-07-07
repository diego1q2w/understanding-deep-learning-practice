import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_with_mse(model, train_loader, optimizer, device):
    """
    Train one epoch with MSE loss.
    Key: Apply softmax BEFORE computing MSE!
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in tqdm(train_loader, desc="Training (MSE)"):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(data)  # Raw logits

        # TODO:
        # 1. Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)

        # 2. Convert target to one-hot
        target_onehot = F.one_hot(target, num_classes=10).float()

        # 3. Compute MSE loss
        loss = F.mse_loss(probs, target_onehot)

        # 4. Backward and update
        loss.backward()
        optimizer.step()

        # TODO: Track metrics
        total_loss += loss.item()
        pred = probs.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
        pass

    return {
        'loss': total_loss / len(train_loader),
        'accuracy': correct / total
    }


def train_with_ce(model, train_loader, optimizer, device):
    """
    Train one epoch with Cross-Entropy loss.
    Key: CE expects raw logits, does softmax internally!
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in tqdm(train_loader, desc="Training (CE)"):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(data)  # Raw logits

        # TODO:
        # 1. Compute CE loss (no softmax needed!)
        loss = F.cross_entropy(logits, target)

        # 2. Backward and update
        loss.backward()
        optimizer.step()

        # TODO: Track metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
        pass

    return {
        'loss': total_loss / len(train_loader),
        'accuracy': correct / total
    }


def validate(model, val_loader, device, loss_type='ce'):
    """
    Validate model and return predictions for analysis.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            logits = model(data)
            probs = F.softmax(logits, dim=1)

            # Compute loss based on type
            if loss_type == 'mse':
                target_onehot = F.one_hot(target, num_classes=10).float()
                loss = F.mse_loss(probs, target_onehot)
            else:  # ce
                loss = F.cross_entropy(logits, target)

            total_loss += loss.item()
            pred = probs.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

            all_probs.append(probs.cpu())
            all_targets.append(target.cpu())

    return {
        'loss': total_loss / len(val_loader),
        'accuracy': correct / total,
        'predictions': torch.cat(all_probs),
        'targets': torch.cat(all_targets)
    }