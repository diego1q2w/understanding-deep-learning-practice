import torch
import torch.nn as nn


def get_conv_grad_magnitudes(model):
    """
    Recursively extract gradient magnitudes from all Conv2d layers.

    Args:
        model: PyTorch model

    Returns:
        torch.Tensor: Normalized gradient magnitudes (one per Conv2d layer)
    """
    grads = []

    for layer in model.children():
        # Case 1: Layer has children (Sequential, Block, etc.) - recurse
        if list(layer.children()):
            child_grads = get_conv_grad_magnitudes(layer)
            grads.extend(child_grads)

        # Case 2: Layer is a Conv2d - extract gradient
        elif isinstance(layer, nn.Conv2d):
            # Defensive: check if gradient exists
            if layer.weight.grad is not None:
                grad_norm = layer.weight.grad.norm()
                num_elements = layer.weight.grad.numel()
                # Detach to avoid memory issues, normalize by number of elements
                normalized_grad = (grad_norm / num_elements).detach().cpu().item()
                grads.append(normalized_grad)
            else:
                # Gradient not computed yet
                grads.append(0.0)

    return torch.tensor(grads)