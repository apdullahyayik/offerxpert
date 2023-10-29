"""Module for processing attributes."""
import pandas as pd


def process_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Process attributes."""
    df["product-attributes"] = df.apply(
        lambda x: ",".join(
            [
                f"{','.join(e['values'])}-{','.join(e['units'])}"
                for e in x["product-attributes"]
            ]
        ),
        axis=1,
    )
    return df
