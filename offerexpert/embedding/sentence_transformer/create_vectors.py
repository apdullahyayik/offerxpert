"""Module for creating embedding vectors."""
import torch
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "average_word_embeddings_glove.840B.300d"
sentence_transformer = SentenceTransformer(_MODEL_NAME)


def create_vectors(
    list_: list[str], batch_size_process: int = 32, show_progress_bar: bool = True
) -> list[torch.Tensor]:
    """Create vectors."""
    return sentence_transformer.encode(
        list_,
        show_progress_bar=show_progress_bar,
        batch_size=batch_size_process,
        normalize_embeddings=False,
        convert_to_tensor=True,
    )
