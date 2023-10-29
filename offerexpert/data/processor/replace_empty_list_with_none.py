"""Module for replacing empty list with None."""
import pandas as pd


def process_replace_emtpy_list_with_none(df: pd.DataFrame) -> pd.DataFrame:
    """Process replace empty list with none."""
    df["positivelyVerifiedOfferNames"] = df["positivelyVerifiedOfferNames"].apply(
        _replace_empty_list_with_none
    )
    return df


def _replace_empty_list_with_none(list_: list) -> None | list:
    if len(list_) == 0:
        return None
    return list_
