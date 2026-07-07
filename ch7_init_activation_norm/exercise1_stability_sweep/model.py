import torch
import torch.nn as nn
import math


class DeepMLP(nn.Module):
    """
    Deep MLP for studying activation/gradient propagation.

    Args:
        input_dim: Input feature dimension (784 for MNIST)
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (10 for MNIST)
        num_layers: Number of hidden layers (default: 20)
        activation: 'relu', 'tanh', or 'gelu'
        init_method: 'xavier' or 'he'
    """

    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10,
                 num_layers=20, activation='relu', init_method='he'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation_name = activation
        self.init_method = init_method

        # Build layers
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply specified initialization to all linear layers."""
        for layer in self.layers:
            if self.init_method == 'xavier':
                nn.init.xavier_normal_(layer.weight)
            elif self.init_method == 'he':
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            else:
                raise ValueError(f"Unknown init method: {self.init_method}")

            # Initialize biases to zero
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # Output layer always uses Xavier
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x, return_activations=False):
        """
        Forward pass.

        Args:
            x: Input tensor [batch, features]
            return_activations: If True, return list of activations per layer
        """
        x = x.view(x.size(0), -1)  # Flatten

        activations = []

        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation(x)

            if return_activations:
                activations.append(x.clone())

        x = self.output_layer(x)

        if return_activations:
            return x, activations
        return x


def compute_activation_stats(model, x, device='cpu'):
    """
    Compute per-layer activation statistics.

    Returns:
        List of dicts with 'mean', 'var', 'min', 'max' for each layer
    """
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        _, activations = model(x, return_activations=True)

    stats = []
    for i, act in enumerate(activations):
        stats.append({
            'layer': i + 1,
            'mean': act.mean().item(),
            'var': act.var().item(),
            'std': act.std().item(),
            'min': act.min().item(),
            'max': act.max().item(),
            'zero_fraction': (act == 0).float().mean().item()  # For ReLU dead units
        })

    return stats


def compute_gradient_stats(model, x, y, device='cpu'):
    """
    Compute per-layer gradient statistics.

    Returns:
        List of dicts with gradient norms for each layer
    """
    model.train()
    x, y = x.to(device), y.to(device)

    # Forward + backward
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()

    stats = []
    for i, layer in enumerate(model.layers):
        if layer.weight.grad is not None:
            grad_norm = layer.weight.grad.norm().item()
            grad_mean = layer.weight.grad.abs().mean().item()
            grad_max = layer.weight.grad.abs().max().item()

            stats.append({
                'layer': i + 1,
                'grad_norm': grad_norm,
                'grad_mean': grad_mean,
                'grad_max': grad_max
            })

    # Clear gradients
    model.zero_grad()

    return stats