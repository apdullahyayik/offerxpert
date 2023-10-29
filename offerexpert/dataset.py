"""Module for dataset operations."""
# pylint: disable=invalid-name
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

_TEST_SET_RATIO = 0.2
_RANDOM_STATE = 42


def split_into_train_test(
    df: pd.DataFrame,
) -> tuple:
    """Split into train and test."""
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=_TEST_SET_RATIO, random_state=_RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


class OfferExpertDataset(Dataset):
    """Offer Expert Dataset."""

    def __init__(
        self,
        product_vectors: list[torch.Tensor],
        offer_vectors: list[torch.Tensor],
        target_vector: torch.Tensor,
    ):
        """Construct."""
        self.product_vectors = product_vectors
        self.offer_vectors = offer_vectors
        self.target_vector = target_vector
        self.vector_size = self.product_vectors[0].shape[0]

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.target_vector)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item."""
        product_vector = self.product_vectors[idx]
        offer_vector = self.offer_vectors[idx]
        target = self.target_vector[idx]
        return product_vector, offer_vector, target
