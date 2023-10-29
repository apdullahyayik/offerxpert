"""Module for inference."""
import json
from pathlib import Path

import faiss
import numpy as np
import torch

from offerexpert.embedding.sentence_transformer import (
    create_sentence_transformer_vectors,
)
from offerexpert.modeling import OfferExpertModel


class Inference:
    """Class for inference."""

    def __init__(
        self,
        model: OfferExpertModel,
        indexer,
        artefact_folder: Path,
    ):
        self._model = model
        self._indexer = indexer
        self._artefact_folder = artefact_folder

    @staticmethod
    def load_model(artefact_folder: Path) -> "Inference":
        """
        Load model and processor from disk.

        Args:
            artefact_folder (Path): The model file.

        Returns:
            AldrinModel: The initialized model.
        """
        model = OfferExpertModel(vector_size=300)
        model.load_state_dict(torch.load(artefact_folder / "model.pt"))
        model.eval()

        indexer = faiss.read_index(str(artefact_folder / "faiss-indexer.bin"))
        return Inference(model, indexer, artefact_folder)

    def __call__(self, text: str, top_n: int = 10) -> list[str]:
        """Get matched top N product ids for given offer as a text."""
        offer_vector = create_sentence_transformer_vectors([text], 1, False)
        out_model: torch.Tensor = self._model.forward_shared_tower(offer_vector)

        _, matched_prod_index = self._indexer.search(out_model.detach().numpy(), top_n)
        with open(self._artefact_folder / "prod_ids_test_positive", "r") as json_file:
            prod_ids = json.load(json_file)

        matched_prod_ids = np.array(prod_ids)[matched_prod_index[0]]
        return matched_prod_ids
