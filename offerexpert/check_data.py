"""Module for checking data."""

from pathlib import Path

from offerexpert.data.analyser import analyse_num_offer_by_product, analyze_sparsity
from offerexpert.data.provider import load_data


def check_data(experiment_folder: Path):
    """Check data."""
    df_prods, df_offers = load_data()

    # Analyse number of offers by product
    analyse_num_offer_by_product(df_prods, experiment_folder)

    # Analyse sparsity
    print(f"\n{'-'*30}\nSparsity of values for offer\n{'-'*30}")
    analyze_sparsity(df_offers)

    print(f"{'-'*30}\nSparsity of values for product\n{'-'*30}\n")
    analyze_sparsity(df_prods)
