"""
Loss functions for heatmap regression.

AdaptiveWingLoss: Wang et al. ICCV 2019
"Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss – best performing loss for heatmap landmark regression.
    Applies larger gradients near the peak (foreground) and smaller near background.

    Default hyperparameters from the paper:
        omega=14, theta=0.5, epsilon=1, alpha=2.1
    """

    def __init__(self, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        """
        pred:   (B, N, H, W) float – predicted heatmaps (raw or sigmoid)
        target: (B, N, H, W) float – Gaussian heatmap targets in [0, 1]
        """
        # Ensure float32 – caller may pass float16 from AMP forward pass
        pred = pred.float()
        target = target.float()
        diff = (target - pred).abs()
        alpha = self.alpha
        omega = self.omega
        epsilon = self.epsilon
        theta = self.theta

        # For each element compute adaptive-wing loss
        # Case 1: diff < theta
        A = omega * (1 / (1 + (theta / epsilon) ** (alpha - target))) * \
            (alpha - target) * ((theta / epsilon) ** (alpha - target - 1)) / epsilon
        C = theta * A - omega * torch.log(1 + (theta / epsilon) ** (alpha - target))

        loss = torch.where(
            diff < theta,
            omega * torch.log(1 + (diff / epsilon) ** (alpha - target)),
            A * diff - C,
        )
        return loss.mean()


class MSEHeatmapLoss(nn.Module):
    """Standard MSE loss on heatmaps."""

    def forward(self, pred, target):
        return F.mse_loss(pred, target)


def build_loss(name='awing', **kwargs):
    if name == 'awing':
        return AdaptiveWingLoss(
            omega=kwargs.get('omega', 14.0),
            theta=kwargs.get('theta', 0.5),
            epsilon=kwargs.get('epsilon', 1.0),
            alpha=kwargs.get('alpha', 2.1),
        )
    elif name == 'mse':
        return MSEHeatmapLoss()
    else:
        raise ValueError(f"Unknown loss: {name}. Choose from ['awing', 'mse']")
