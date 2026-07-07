import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def im2col(images, kw, kh, stride = 1):
    N, C, H, W = images.shape
    # Calculate output dimensions
    H_out = (H - kh) // stride + 1
    W_out = (W - kw) // stride + 1

    # Total number of patches
    num_patches = H_out * W_out

    # Size of each flattened patch
    patch_size = C * kh * kw
    col_matrix = torch.zeros((N, num_patches, patch_size), device=images.device, dtype=images.dtype)

    patch_index = 0
    for i in range(0, H - kh + 1, stride):  # Height direction
        for j in range(0, W - kw + 1, stride):  # Width direction
            patch = images[:, :, i:i + kh, j:j + kw]
            col_matrix[:, patch_index, :] = patch.reshape(N, C*kh*kw)
            patch_index +=1

    return col_matrix, H_out, W_out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvLayer, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.rand(out_channels, in_channels * kernel_size * kernel_size))
        self.bias = nn.Parameter(torch.rand(out_channels))

    def forward(self, x):
        N, C, H, W = x.shape
        # Use our custom im2col
        img_col_res, H_out, W_out = im2col(x, self.kernel_size, self.kernel_size, self.stride)

        # Matrix multiplication with learnable weights
        conv = img_col_res @ self.weight.T + self.bias

        # Apply ReLU
        conv = F.relu(conv)

        # Reshape back to image format
        conv = conv.reshape(N, H_out, W_out, self.out_channels)
        conv = conv.permute(0, 3, 1, 2)  # (N, Co, H, W)

        return conv

class CustomCNN(nn.Module):
    def __init__(self, in_channels=3, num_layers=3, out_channels_per_layer=5, kernel_size=3 ):
        super(CustomCNN, self).__init__()

        self.layers = nn.ModuleList()

        # First layer: in_channels -> out_channels_per_layer
        self.layers.append(ConvLayer(in_channels, out_channels_per_layer, kernel_size))

        # Subsequent layers: out_channels_per_layer -> out_channels_per_layer
        for _ in range(num_layers - 1):
            self.layers.append(ConvLayer(out_channels_per_layer, out_channels_per_layer, kernel_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Test the trainable model
model = CustomCNN(in_channels=3, num_layers=3, out_channels_per_layer=5)
test_images = torch.randn(2, 3, 32, 32)  # Use torch.randn instead of np.random.rand

# Forward pass
result = model(test_images)
print(f"Final output shape: {result.shape}")
