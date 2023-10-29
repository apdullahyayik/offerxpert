"""Module for `modelling`."""
import torch
from torch import nn


class OfferExpertModel(nn.Module):
    """Offer expert model."""

    def __init__(
        self,
        vector_size: int = 384,
        dropout_prob: float = 0.2,
        l2_reg_weight: float = 1e-5,
    ):
        """Construct."""
        super().__init__()
        self.vector_size = vector_size
        self.dropout_prob = dropout_prob
        self.l2_reg_weight = l2_reg_weight

        self.shared_tower = nn.Sequential(
            nn.Linear(self.vector_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(256, 512),
        )

    def forward(self, x1, x2):
        """Forward."""
        out1 = self.forward_shared_tower(x1)
        out2 = self.forward_shared_tower(x2)
        similarity_score = self._compute_similarity(out1, out2)
        return similarity_score

    def forward_shared_tower(self, x):
        """Forward product and offer to shared tower."""
        out = self.shared_tower(x)

        # l2_reg = torch.tensor(0.0)
        # for param in self.shared_tower.parameters():
        #     l2_reg += torch.norm(param, p=2)
        # out -= 0.5
        return out

    @staticmethod
    def _compute_similarity(output1, output2):
        normalized_output1 = torch.nn.functional.normalize(output1, p=2, dim=1)
        normalized_output2 = torch.nn.functional.normalize(output2, p=2, dim=1)
        similarity = torch.sum(
            normalized_output1 * normalized_output2, dim=1, keepdim=True
        )
        return similarity
