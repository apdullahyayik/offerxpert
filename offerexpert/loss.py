"""Module for loss functions."""
import torch
from torch import nn


class ContrastiveLoss(nn.Module):
    """Constructive loss."""

    def __init__(self, margin=1.0):
        """Construct."""
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """Forward pass."""
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


import torch
import torch.nn as nn


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, output, target):
        # Apply a sigmoid function to the output to obtain a probability value in [0, 1]
        output = torch.sigmoid(output)

        # Compute the binary cross-entropy loss
        loss = -target * torch.log(output + 1e-15) - (1 - target) * torch.log(
            1 - output + 1e-15
        )

        # Take the mean over the batch
        loss = torch.mean(loss)

        return loss
