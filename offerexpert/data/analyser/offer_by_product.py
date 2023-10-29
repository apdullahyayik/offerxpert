"""Module for analyzing number of offers by each product."""
import logging
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    stream=sys.stdout,
    level=logging.INFO,
)


def analyse_num_offer_by_product(df: pd.DataFrame, experiment_folder: Path):
    """Analyse number of verified offer by product."""
    df["num_offers"] = df["positivelyVerifiedOfferNames"].apply(
        _get_number_of_positive_offers
    )
    ax = df["num_offers"].plot(
        kind="barh", figsize=(8, 6), color="skyblue", stacked=True, width=4
    )
    ax.set_ylabel("Product", fontsize=12)
    ax.set_xlabel("Number of Offers", fontsize=12)
    ax.set_title("Number of Offers for Product", fontsize=14)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_yticklabels(df.index)  # Set y-axis labels
    ax.set_yticks([])  # Remove x-axis ticks and labels
    path = experiment_folder / "analyse_offer_by_product.png"
    plt.savefig(path)
    logging.info("Analyse for number of verified offer by product is saved at %s", path)
    print(
        f"\n{'-' * 40}\nStatistics for number of offers by product\n{'-' * 40}\n{df['num_offers'].describe()}"
    )


def _get_number_of_positive_offers(data: list | None) -> int:
    if data is None:
        return 0
    return len(data)
