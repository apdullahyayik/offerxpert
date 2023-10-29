"""Module for dropping fields."""
import pandas as pd


def processing_dropping_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fields."""
    remove_list = [
        "offer-lastUpdated",
        "offer-isbn10",
        "product-categoryId",
        "product-isbn10",
        "product-positivelyVerifiedOfferNames",
    ]
    df.drop(
        columns=remove_list,
        inplace=True,
    )
    return df
