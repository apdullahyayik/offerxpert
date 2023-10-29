"""Module for processing price calculations."""
import pandas as pd


def process_price_calculations(df: pd.DataFrame) -> pd.DataFrame:
    """Process price calculations."""
    df_price_calculations = pd.json_normalize(df["priceCalculations"])  # type: ignore
    df_price_calculations = df_price_calculations.add_prefix("amount_")
    df = pd.concat([df, df_price_calculations], axis=1)
    df.drop(columns=["priceCalculations"], inplace=True)
    return df
