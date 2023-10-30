"""Module for early stopping."""
import logging
import sys

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    stream=sys.stdout,
    level=logging.INFO,
)


class EarlyStop:
    """Early stop."""

    def __init__(self, patience: int):
        """Construct."""
        super().__init__()

        self._patience: int = patience
        self._patience_reset: int = patience
        self._best_val_loss: float = 1_000.0
        self._is_improvement: bool = False

    def __repr__(self) -> str:
        """Return a string representation of the class."""
        return f"EarlyStop(patience={self._patience})"

    def is_continue(self, metric_value: float) -> bool:
        """Return True if training should be continued."""
        return self._lowest_val_continue(metric_value)

    @property
    def is_improvement(self) -> bool:
        """Return True if last loss was an improvement."""
        return self._is_improvement

    def _lowest_val_continue(self, val_loss: float) -> bool:
        if val_loss < self._best_val_loss:
            self._is_improvement = True
            self._best_val_loss = val_loss
            self._reset()
            return True

        self._is_improvement = False
        self._patience -= 1
        logging.info("patience status: %s/%s", self._patience, self._patience_reset)
        return self._patience != 0

    def _reset(self):
        self._patience = self._patience_reset
