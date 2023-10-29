"""Module utilities."""
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


def plot_train_report(history: dict[str, np.array], onset_epoch: int = 0) -> Any:
    """Plot train and validation loss over epochs."""
    train_loss = history["loss"]
    val_loss = history["val_loss"]

    figure(figsize=(10, 8))
    plt.plot(train_loss[onset_epoch:], label="train loss", linewidth=3, marker="o")
    plt.plot(val_loss[onset_epoch:], label="val loss", linewidth=3, marker="o")
    xticks = list(
        range(
            0,
            len(train_loss[onset_epoch:]),
            10 if len(train_loss[onset_epoch:]) > 10 else 1,
        )
    )
    plt.xticks(xticks, labels=[str(i + 1) for i in xticks])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=20)
    plt.title(
        f"Train Report\nbest val loss: {val_loss.min():.2f}-epoch {val_loss.argmin()}",
        fontsize=25,
    )
    plt.xlabel("Epochs", fontsize=21)
    plt.legend(loc="upper right", fontsize=18)
    plt.axis("on")
    plt.grid(True)
    return plt
