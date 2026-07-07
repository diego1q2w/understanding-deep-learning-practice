import numpy as np
import torch
import torch.nn as nn

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
    col_matrix = np.zeros((N, num_patches, patch_size))

    patch_index = 0
    for i in range(0, H - kh + 1, stride):  # Height direction
        for j in range(0, W - kw + 1, stride):  # Width direction
            patch = images[:, :, i:i + kh, j:j + kw]
            col_matrix[:, patch_index, :] = patch.reshape(N, C*kh*kw)
            patch_index +=1

    return col_matrix, H_out, W_out


def conv_im2col(images, kernel=3, stride=1, out_channels=1):
    N, C, H, W = images.shape

    # Now we need weights for EACH output channel
    weights = np.random.rand(out_channels, C * kernel * kernel)  # Shape: (Co, Cin*kh*kw)
    bias = np.random.rand(out_channels)  # One bias per output channel

    img_col_res, H_out, W_out = im2col(images, kernel, kernel, stride=stride)  # (N, num_patches, C*kh*kw)

    # Matrix multiply: (N, num_patches, C*kh*kw) @ (C*kh*kw, Co) = (N, num_patches, Co)
    conv = img_col_res @ weights.T + bias  # .T transposes weights to (C*kh*kw, Co)

    conv = relu(conv) ## for now apply RELU to all convs

    return conv, H_out, W_out  # Shape: (N, num_patches, Co)


def conv_multi_layers(images, kernel=3, stride=1, layers=3):
    conv, H_out, W_out = conv_im2col(images, kernel=kernel, stride=stride, out_channels=5)
    conv = reshape_conv(conv, H_out, W_out)
    for layer in range(1, layers):
        conv, H_out, W_out = conv_im2col(conv, kernel=kernel, stride=stride, out_channels=5)
        conv = reshape_conv(conv, H_out, W_out)

    return conv


def reshape_conv(conv, h_out, w_out):
    N, num_patches, Co = conv.shape
    conv = conv.reshape(N, h_out, w_out, Co)
    return conv.transpose((0, 3, 1, 2)) # (N, Co, H, W)

def relu(array):
    return np.where(array>0, array, 0)


# Create some dummy input
test_images = np.random.rand(2, 3, 32, 32)  # Batch of 2 RGB images

# Run your CNN
result = conv_multi_layers(test_images, kernel=3, stride=1, layers=3)
print(f"Final output shape: {result.shape}")  # Should be (2, 5, 26, 26)

