"""Module for processing global trade item number."""
import pandas as pd


def process_global_trade_item_number(df: pd.DataFrame) -> pd.DataFrame:
    """Process global trade item number."""
    for kind in ["offer", "product"]:
        _add_explanation(df, kind)
    return df


def _add_explanation(df: pd.DataFrame, kind: str):
    df[f"{kind}-gtin14"] = df.apply(
        lambda x: f"Global Trade Item Number: {x[f'{kind}-gtin14']}"
        if x[f"{kind}-gtin14"] is not None
        else "",
        axis=1,
    )
