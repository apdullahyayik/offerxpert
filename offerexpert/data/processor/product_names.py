"""Module for processing product names."""
import pandas as pd


def process_product_names(df: pd.DataFrame) -> pd.DataFrame:
    """Process product names."""
    df["product-productNames"] = df.apply(
        lambda x: f"Product names: {','.join(x['product-productNames'])}"
        if x["product-productNames"] is not None
        else "",
        axis=1,
    )
    return df
