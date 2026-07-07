"""Tests for Focal Loss implementation."""

import pytest
import torch
import torch.nn.functional as F

from ch5_loss_functions.imbalance.losses import FocalLoss


class TestFocalLoss:
    """Test Focal Loss implementation."""

    def test_gamma_zero_approximates_cross_entropy(self, classification_batch):
        """Focal loss with gamma=0 should approximate cross entropy."""
        logits, targets = classification_batch

        focal = FocalLoss(gamma=0.0)
        ce = F.cross_entropy(logits, targets)
        focal_loss = focal(logits, targets)

        # They won't be exactly equal due to implementation differences,
        # but should be close
        assert torch.isclose(focal_loss, ce, atol=1e-4)

    def test_higher_gamma_focuses_hard_examples(self):
        """Higher gamma should weight hard examples more heavily."""
        # Create easy predictions (high confidence correct)
        logits_easy = torch.zeros(10, 10)
        logits_easy[range(10), range(10)] = 10.0  # High confidence

        # Create hard predictions (low confidence)
        logits_hard = torch.randn(10, 10) * 0.1

        targets = torch.arange(10)

        fl_low_gamma = FocalLoss(gamma=0.5)
        fl_high_gamma = FocalLoss(gamma=3.0)

        # Ratio of hard to easy loss
        ratio_low = (
            fl_low_gamma(logits_hard, targets) / fl_low_gamma(logits_easy, targets)
        )
        ratio_high = (
            fl_high_gamma(logits_hard, targets) / fl_high_gamma(logits_easy, targets)
        )

        # Higher gamma should amplify the difference between hard and easy
        assert ratio_high > ratio_low

    def test_output_is_scalar(self, classification_batch):
        """Focal loss should return a scalar."""
        logits, targets = classification_batch
        fl = FocalLoss(gamma=2.0)
        loss = fl(logits, targets)
        assert loss.dim() == 0

    def test_output_is_positive(self, classification_batch):
        """Focal loss should be positive."""
        logits, targets = classification_batch
        fl = FocalLoss(gamma=2.0)
        loss = fl(logits, targets)
        assert loss > 0

    def test_with_class_weights(self, classification_batch):
        """Focal loss should work with class weights."""
        logits, targets = classification_batch
        num_classes = logits.shape[1]

        alpha = torch.ones(num_classes)
        alpha[0] = 10.0  # Weight first class heavily

        fl_weighted = FocalLoss(alpha=alpha, gamma=2.0)
        loss = fl_weighted(logits, targets)

        assert not torch.isnan(loss)
        assert loss > 0

    def test_no_nan_with_confident_predictions(self):
        """Focal loss should not produce NaN even with very confident predictions."""
        # Very confident predictions (could cause numerical issues)
        logits = torch.zeros(10, 10)
        logits[range(10), range(10)] = 100.0  # Very high confidence
        targets = torch.arange(10)

        fl = FocalLoss(gamma=2.0)
        loss = fl(logits, targets)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flow(self, classification_batch):
        """Gradients should flow through focal loss."""
        logits, targets = classification_batch
        logits.requires_grad = True

        fl = FocalLoss(gamma=2.0)
        loss = fl(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
