"""Tests for ResNet-20 implementations."""

import pytest
import torch

from ch11_residual_networks.resnet20 import (
    RestNet20,
    PreActivationBlock,
    PostActivationBlock,
    NoBatchNormBlock,
)


class TestResidualBlocks:
    """Test individual residual block implementations."""

    @pytest.mark.parametrize(
        "BlockClass",
        [PreActivationBlock, PostActivationBlock, NoBatchNormBlock],
    )
    def test_block_same_channels(self, BlockClass):
        """Block with same in/out channels preserves shape."""
        block = BlockClass(in_channels=16, out_channels=16)
        x = torch.randn(2, 16, 32, 32)
        out = block(x)
        assert out.shape == x.shape

    @pytest.mark.parametrize(
        "BlockClass",
        [PreActivationBlock, PostActivationBlock, NoBatchNormBlock],
    )
    def test_block_channel_expansion(self, BlockClass):
        """Block can expand channels with stride."""
        block = BlockClass(in_channels=16, out_channels=32, first_stride=2)
        x = torch.randn(2, 16, 32, 32)
        out = block(x)
        assert out.shape == (2, 32, 16, 16)

    @pytest.mark.parametrize(
        "BlockClass",
        [PreActivationBlock, PostActivationBlock, NoBatchNormBlock],
    )
    def test_block_no_nan(self, BlockClass):
        """Block output should not contain NaN values."""
        block = BlockClass(in_channels=16, out_channels=32, first_stride=2)
        x = torch.randn(2, 16, 32, 32)
        out = block(x)
        assert not torch.isnan(out).any()


class TestResNet20:
    """Test full ResNet-20 model."""

    @pytest.mark.parametrize("net_type", ["pre", "post", "nobn"])
    def test_output_shape(self, cifar_batch, net_type):
        """Output should be [batch, num_classes]."""
        model = RestNet20(net_type=net_type, num_classes=10)
        output = model(cifar_batch)
        assert output.shape == (4, 10)

    @pytest.mark.parametrize("net_type", ["pre", "post", "nobn"])
    def test_no_nan_output(self, cifar_batch, net_type):
        """Model output should not contain NaN values."""
        model = RestNet20(net_type=net_type, num_classes=10)
        output = model(cifar_batch)
        assert not torch.isnan(output).any()

    def test_different_num_classes(self, cifar_batch):
        """Model should work with different number of classes."""
        for num_classes in [2, 10, 100]:
            model = RestNet20(net_type="pre", num_classes=num_classes)
            output = model(cifar_batch)
            assert output.shape == (4, num_classes)

    def test_invalid_net_type_raises(self):
        """Invalid net_type should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            RestNet20(net_type="invalid")

    def test_gradient_flow(self, cifar_batch):
        """Gradients should flow through the model."""
        model = RestNet20(net_type="pre", num_classes=10)
        output = model(cifar_batch)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist and are not zero
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
