from lib2to3.fixer_util import does_tree_import

import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_epoch(model, data_loader,optimizer, criterion_type='ce', device='cpu'):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x_batch, y_batch in tqdm(data_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(x_batch)

        if criterion_type == 'ce':
            loss = F.cross_entropy(output, y_batch)
        else:
            probs = torch.softmax(output, dim=1)
            y_onehot = F.one_hot(y_batch, num_classes=probs.shape[1]).float()
            loss = F.mse_loss(output, y_onehot)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += y_batch.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == y_batch).sum().item()

    return {
        'loss': total_loss / len(data_loader),
        'accuracy': correct / total
    }


def train_model(model, train_loader, epochs=100, lr=0.01, criterion_type='ce', device='cpu'):
    """Full training loop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(epochs):
        metrics = train_epoch(model, train_loader, optimizer, criterion_type, device)
        history.append(metrics)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {metrics['loss']:.4f}, "
                  f"Acc: {metrics['accuracy']:.4f}")

    return history