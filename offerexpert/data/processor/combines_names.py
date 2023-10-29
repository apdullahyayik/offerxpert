"""Module for processing combined names."""
import pandas as pd


def process_combined_names(df: pd.DataFrame) -> pd.DataFrame:
    """Process combined names."""
    df["product-combinedNames"] = df.apply(
        lambda x: f"Product combined: {','.join(x['product-combinedNames'])}"
        if x["product-combinedNames"] is not None
        else "",
        axis=1,
    )
    return df
