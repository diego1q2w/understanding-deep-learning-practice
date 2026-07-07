import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm


class TinyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TinyModel, self).__init__()

        self.fc1 = nn.Linear(in_features=input_dim, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=output_dim)

        self.act = nn.ReLU()

    def forward(self, x):
        # Flatten MNIST images from (batch, 1, 28, 28) → (batch, 784)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x


def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader


def train_with_increasing_lr(dataloader, start_lr, end_lr, device='cpu'):
    """
    Train model for one epoch while exponentially increasing learning rate.

    Args:
        dataloader: Training data loader
        start_lr: Starting learning rate (e.g., 1e-6)
        end_lr: Ending learning rate (e.g., 10.0)
        device: Device to train on

    Returns:
        losses: List of loss values
        lrs: List of learning rates
    """
    model = TinyModel(input_dim=784, output_dim=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)
    criterion = nn.CrossEntropyLoss()

    # Calculate the multiplication factor for exponential growth
    num_iterations = len(dataloader)
    lr_mult = (end_lr / start_lr) ** (1.0 / num_iterations)

    losses = []
    lrs = []

    print(f"Starting LR Range Test: {start_lr:.2e} → {end_lr:.2e}")
    print(f"Number of iterations: {num_iterations}")
    print(f"LR multiplication factor per step: {lr_mult:.6f}\n")

    model.train()
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="LR Range Test")):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Record loss and current LR
        current_lr = optimizer.param_groups[0]['lr']
        losses.append(loss.item())
        lrs.append(current_lr)

        # Stop if loss explodes (becomes NaN or too large)
        if loss.item() > 10.0 or torch.isnan(loss):
            print(f"\n⚠️  Loss exploded at LR={current_lr:.2e}, stopping early")
            break

        # Exponentially increase learning rate for next iteration
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult

        # Stop if we've exceeded end_lr
        if optimizer.param_groups[0]['lr'] > end_lr:
            print(f"\n✅ Reached maximum LR: {end_lr:.2e}")
            break

    return losses, lrs


def plot_lr_vs_loss(losses, lrs, save_path='lr_range_test.png'):
    """
    Plot learning rate vs loss on a log scale.

    Args:
        losses: List of loss values
        lrs: List of learning rates
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Plot loss vs LR
    ax.plot(lrs, losses, linewidth=2, color='#1f77b4')
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate (log scale)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('LR Range Test: Finding Optimal Learning Rate', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # Find and mark the minimum loss
    min_loss_idx = losses.index(min(losses))
    min_lr = lrs[min_loss_idx]
    min_loss = losses[min_loss_idx]

    ax.axvline(min_lr, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Min Loss LR: {min_lr:.2e}')
    ax.scatter([min_lr], [min_loss], color='red', s=100, zorder=5, marker='o')

    # Suggest optimal LR (typically 1/10th of the min loss LR or in the steepest descent region)
    suggested_lr = min_lr / 10.0
    ax.axvline(suggested_lr, color='green', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Suggested LR: {suggested_lr:.2e}')

    ax.legend(fontsize=11, loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {save_path}")
    plt.show()

    # Print analysis
    print("\n" + "=" * 70)
    print("LR RANGE TEST RESULTS")
    print("=" * 70)
    print(f"Minimum Loss: {min_loss:.4f} at LR = {min_lr:.2e}")
    print(f"Suggested LR: {suggested_lr:.2e} (1/10 of min loss LR)")
    print(f"Safe Range: {suggested_lr / 3:.2e} to {suggested_lr * 3:.2e}")
    print("=" * 70)


if __name__ == "__main__":
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    dataloader = load_mnist()

    # Run LR range test
    losses, lrs = train_with_increasing_lr(
        dataloader=dataloader,
        start_lr=1e-6,
        end_lr=10.0,
        device=device
    )

    # Plot results
    plot_lr_vs_loss(losses, lrs)