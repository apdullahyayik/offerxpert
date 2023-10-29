"""Module for filling missing values."""
import pandas as pd


def process_fill_missing_values(
    df: pd.DataFrame, col_names=("description", "name")
) -> pd.DataFrame:
    """Process fill missing values."""
    for name in col_names:
        df[name] = df[name].fillna("")
    return df
