import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


def plot_decision_boundary(model, X, y, title="Decision Boundary",
                           temperature=1.0, ax=None, device='cpu'):
    """
    Visualize decision boundary and confidence contours.

    Args:
        model: Trained model
        X: Data points (numpy array, shape [N, 2])
        y: Labels (numpy array, shape [N])
        title: Plot title
        temperature: Softmax temperature for calibration
        ax: Matplotlib axis (if None, creates new figure)
        device: Device for computation
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Create meshgrid covering the data region (with some padding)
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Flatten the meshgrid to create input points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid_points).to(device)

    # Get predictions
    model.eval()
    with torch.no_grad():
        # 1. Get logits from model
        logits = model(grid_tensor)

        # 2. Apply temperature scaling
        logits_scaled = logits / temperature

        # 3. Get probabilities with softmax
        probs = F.softmax(logits_scaled, dim=1)

        # 4. Get probability of class 1 (for binary)
        Z = probs[:, 1].cpu().numpy()

        # 5. Reshape back to grid
        Z = Z.reshape(xx.shape)

    # Plot filled contours (confidence heatmap)
    contourf = ax.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.8)
    plt.colorbar(contourf, ax=ax, label='P(Class 1)')

    # Plot decision boundary (where P=0.5)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3)

    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu_r',
                         edgecolors='black', s=50, linewidth=1.5)

    ax.set_xlabel('X1', fontsize=12, fontweight='bold')
    ax.set_ylabel('X2', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    return ax


def plot_uncertainty_map(model, X, y, title="Uncertainty Map", ax=None, device='cpu'):
    """
    Plot entropy (uncertainty) across input space.
    High entropy = model is uncertain
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Same meshgrid setup
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid_points).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(grid_tensor)
        probs = F.softmax(logits, dim=1)

        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        Z = entropy.cpu().numpy().reshape(xx.shape)

    # Plot entropy heatmap
    contourf = ax.contourf(xx, yy, Z, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(contourf, ax=ax, label='Entropy (Uncertainty)')

    # Plot data points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu_r',
               edgecolors='black', s=50, linewidth=1.5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    return ax


def plot_confidence_along_line(model_ce, model_mse, X, device='cpu'):
    """
    Plot how confidence changes along a horizontal line through the data.
    """
    # Create a horizontal line through the middle
    y_middle = X[:, 1].mean()
    x_range = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
    line_points = np.column_stack([x_range, np.full_like(x_range, y_middle)])
    line_tensor = torch.FloatTensor(line_points).to(device)

    # Get predictions from both models
    fig, ax = plt.subplots(figsize=(12, 6))

    for model, name, color in [(model_ce, 'Cross-Entropy', 'blue'),
                               (model_mse, 'MSE', 'orange')]:
        model.eval()
        with torch.no_grad():
            logits = model(line_tensor)
            probs = F.softmax(logits, dim=1)[:, 1]  # P(class 1)
            ax.plot(x_range, probs.cpu().numpy(), label=name,
                    linewidth=3, color=color)

    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2,
               label='Decision Boundary')
    ax.set_xlabel('Position along line', fontsize=12, fontweight='bold')
    ax.set_ylabel('P(Class 1)', fontsize=12, fontweight='bold')
    ax.set_title('Confidence Along Horizontal Line', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    return fig