import torch
import torch.nn as nn
from enum import Enum


class NormType(Enum):
    NONE = "none"
    BATCH = "batch"
    LAYER = "layer"


class DeepMLPWithNorm(nn.Module):
    """
    Deep MLP with optional BatchNorm or LayerNorm.

    Architecture (per layer):
        Post-activation order: Linear → Norm → Activation

    Args:
        input_dim: Input features (784 for MNIST)
        hidden_dim: Hidden layer size
        output_dim: Number of classes
        num_layers: Number of hidden layers
        activation: 'relu', 'tanh', or 'gelu'
        norm_type: NormType.NONE, NormType.BATCH, or NormType.LAYER
        init_method: 'xavier' or 'he'
    """

    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10,
                 num_layers=20, activation='relu',
                 norm_type=NormType.NONE, init_method='he'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation_name = activation
        self.norm_type = norm_type
        self.init_method = init_method

        # Build layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.norms.append(self._create_norm_layer(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(self._create_norm_layer(hidden_dim))

        # Output layer (no norm before output)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Activation
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

    def _create_norm_layer(self, dim):
        """Create normalization layer based on norm_type."""
        if self.norm_type == NormType.BATCH:
            return nn.BatchNorm1d(dim)
        elif self.norm_type == NormType.LAYER:
            return nn.LayerNorm(dim)
        else:  # NormType.NONE
            return nn.Identity()  # No-op layer

    def _initialize_weights(self):
        """Initialize weights using specified method."""
        for layer in self.layers:
            if self.init_method == 'xavier':
                nn.init.xavier_normal_(layer.weight)
            elif self.init_method == 'he':
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # Output layer
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x, return_stats=False):
        """
        Forward pass.

        Args:
            x: Input [batch, features]
            return_stats: If True, return per-layer activation stats
        """
        x = x.view(x.size(0), -1)

        activation_stats = [] if return_stats else None

        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            x = norm(x)
            x = self.activation(x)

            if return_stats:
                with torch.no_grad():
                    activation_stats.append({
                        'layer': i + 1,
                        'mean': x.mean().item(),
                        'std': x.std().item(),
                        'var': x.var().item(),
                        'min': x.min().item(),
                        'max': x.max().item()
                    })

        x = self.output_layer(x)

        if return_stats:
            return x, activation_stats
        return x

    def get_gradient_stats(self):
        """Get gradient statistics for each layer."""
        stats = []
        for i, layer in enumerate(self.layers):
            if layer.weight.grad is not None:
                stats.append({
                    'layer': i + 1,
                    'grad_norm': layer.weight.grad.norm().item(),
                    'grad_mean': layer.weight.grad.abs().mean().item(),
                    'grad_max': layer.weight.grad.abs().max().item()
                })
        return stats


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
