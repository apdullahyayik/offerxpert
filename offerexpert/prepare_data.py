"""Module for data preparation."""
import pandas as pd

from offerexpert.data.processor.attributes import process_attributes
from offerexpert.data.processor.combine_offer_and_product_features import (
    process_combine_offer_and_product_features,
)
from offerexpert.data.processor.combines_names import process_combined_names
from offerexpert.data.processor.drop_fields import processing_dropping_fields
from offerexpert.data.processor.fill_missing_values import process_fill_missing_values
from offerexpert.data.processor.global_trade_item_number import (
    process_global_trade_item_number,
)
from offerexpert.data.processor.positively_verified_offer_name import (
    process_positively_verified_offer_name,
)
from offerexpert.data.processor.price_calculations import process_price_calculations
from offerexpert.data.processor.product_names import process_product_names
from offerexpert.data.processor.replace_empty_list_with_none import (
    process_replace_emtpy_list_with_none,
)
from offerexpert.data.provider import load_data
from offerexpert.data.sampler.generate_samples import generate_samples


def prepare_data(n_neg_sample: int) -> pd.DataFrame:
    """Prepare data."""
    df_prods, df_offers = load_data()
    df_prods = process_price_calculations(df_prods)
    df_prods = process_replace_emtpy_list_with_none(df_prods)
    df_offers = process_fill_missing_values(
        df_offers, col_names=("description", "name")
    )
    df_dataset = generate_samples(df_prods, df_offers, n_neg_sample)
    df_dataset = process_global_trade_item_number(df_dataset)
    df_dataset = process_combined_names(df_dataset)
    df_dataset = process_product_names(df_dataset)
    df_dataset = process_attributes(df_dataset)
    df_dataset = process_positively_verified_offer_name(df_dataset)
    df_dataset = processing_dropping_fields(df_dataset)
    df_dataset = process_combine_offer_and_product_features(df_dataset)
    return df_dataset
