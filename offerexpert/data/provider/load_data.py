"""Module for loading data."""
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from offerexpert.exceptions import OfferExpertEnvironmentError

PATH_DATA = os.environ.get("OFFER_EXPERT_PATH_DATA")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    stream=sys.stdout,
    level=logging.INFO,
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load data."""
    if PATH_DATA is None:
        raise OfferExpertEnvironmentError(
            "Set OFFER_EXPERT_PATH_DATA environment variable!"
        )

    with open(Path(PATH_DATA) / "offers.json", "r", encoding="utf-8") as fp:
        offers = json.load(fp)
    with open(Path(PATH_DATA) / "products.json", "r", encoding="utf-8") as fp:
        products = json.load(fp)
    logging.info("Read %s products and %s offers", len(products), len(offers))

    df_offers = pd.DataFrame(offers)
    df_prods = pd.DataFrame(products)
    return df_prods, df_offers
