"""Module for combining offer and product features."""
import logging
import os
from pathlib import Path

import pandas as pd

from offerexpert.data.provider.load_data import PATH_DATA


def process_combine_offer_and_product_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Process combine offer and product features."""
    df["offer"] = df.apply(
        lambda x: f"{x['offer-name']}, "
        # f"{x['offer-description']}, "
        # f"{x['offer-gtin14']}, "
        # f"{x['offer-priceAmount']}"
        .lower(),
        axis=1,
    )
    df.drop(
        columns=[
            "offer-name",
            "offer-description",
            "offer-gtin14",
            # "offer-priceAmount",
        ],
        inplace=True,
    )
    df["product"] = df.apply(
        lambda x:
        # f"{x['product-combinedNames']},"
        f"{x['product-positively_verified_offer_name']}, "
        # f"{x['product-productNames']}, "
        # f"{x['product-attributes']}, "
        # f"{x['product-gtin14']}, "
        # f"{x['product-amount_mean']}, "
        # f"{x['product-amount_median']}, "
        # f"{x['product-amount_standardDeviation']}"
        .lower(),
        axis=1,
    )
    df.drop(
        columns=[
            "product-combinedNames",
            "product-positively_verified_offer_name",
            "product-productNames",
            "product-attributes",
            "product-gtin14",
            # "product-amount_mean",
            "product-amount_median",
            # "product-amount_standardDeviation",
        ],
        inplace=True,
    )

    path_dataset = Path(PATH_DATA) / "data.csv"
    df.to_csv(
        path_dataset,
        index=False,
    )
    logging.info("Dataset is saved to %s.", path_dataset)
    return df
