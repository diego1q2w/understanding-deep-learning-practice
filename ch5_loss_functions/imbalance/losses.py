import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: FL = -α(1-pt)^γ log(pt)

    Focuses on hard examples by down-weighting easy ones.

    Args:
        alpha: Class balance weights (tensor of shape [num_classes])
        gamma: Focusing parameter (default 2.0)
               Higher gamma = more focus on hard examples
    """

    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Args:
            logits: Model outputs (raw scores), shape [batch, num_classes]
            targets: True labels, shape [batch]

        Returns:
            Focal loss value (scalar)
        """
        # TODO: Get probabilities
        probs = F.softmax(logits, dim=1)

        # TODO: Get probability of true class for each sample
        # We need to gather p[target] for each sample
        pt = probs.gather(1, targets.view(-1, 1)).squeeze(1)

        # TODO: Compute focal term: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # TODO: Compute log probability
        log_pt = torch.log(pt + 1e-8)  # Add epsilon for stability

        # TODO: Compute focal loss: -focal_weight × log(pt)
        loss = -focal_weight * log_pt

        # TODO: Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        # Return mean loss
        return loss.mean()


# Test focal loss
if __name__ == "__main__":
    # Create dummy data
    logits = torch.randn(10, 10)  # 10 samples, 10 classes
    targets = torch.randint(0, 10, (10,))

    # Test without alpha
    fl = FocalLoss(gamma=2.0)
    loss = fl(logits, targets)
    print(f"Focal loss (no alpha): {loss.item():.4f}")

    # Test with alpha
    alpha = torch.ones(10)
    alpha[5] = 5.0  # Weight class 5 more
    fl_weighted = FocalLoss(alpha=alpha, gamma=2.0)
    loss_weighted = fl_weighted(logits, targets)
    print(f"Focal loss (with alpha): {loss_weighted.item():.4f}")