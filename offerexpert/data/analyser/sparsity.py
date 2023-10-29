"""Module for analyzing sparsity."""
import pandas as pd


def analyze_sparsity(df: pd.DataFrame):
    """Analyze sparsity."""
    for column_name in df:
        num_missing_values = df[column_name].isna().sum()
        sparsity = (num_missing_values / df.shape[0]) * 100
        print(f"{column_name}': {sparsity:.2f}%")
