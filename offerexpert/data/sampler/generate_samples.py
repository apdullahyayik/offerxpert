"""Module for generating positive and negative samples."""
# pylint: disable=too-many-locals
import random

import pandas as pd
from tqdm import tqdm


def generate_samples(
    df_prods: pd.DataFrame, df_offers: pd.DataFrame, n_neg_sample: int
) -> pd.DataFrame:
    """Generate positive and negative samples."""
    mapping_prod_id_to_offer_ids = _map_product_with_offers(df_prods)
    offer_ids_entire = _get_all_offer_ids(mapping_prod_id_to_offer_ids)

    dataset: list = []
    for prod_id, pos_offer_ids in tqdm(
        mapping_prod_id_to_offer_ids.items(), "Generating positive/negative samples"
    ):
        # Get product instance
        production_sample = df_prods[df_prods["id"] == prod_id].add_prefix("product-")
        production_sample.reset_index(drop=True, inplace=True)

        for pos_offer_id in pos_offer_ids:
            # Add positive instance
            positive_sample = df_offers[
                df_offers["offerId"] == pos_offer_id
            ].add_prefix("offer-")
            positive_sample.reset_index(drop=True, inplace=True)

            if positive_sample.empty:
                continue

            positive_instance = pd.concat([positive_sample, production_sample], axis=1)
            positive_instance["target"] = 1
            dataset.append(positive_instance)

            # Add negative instance/s
            negative_samples = _generate_negative_instances(
                df_offers, offer_ids_entire, pos_offer_id, n_neg_sample
            )
            for negative_sample in negative_samples:
                negative_instance = pd.concat(
                    [negative_sample, production_sample], axis=1
                )
                negative_instance["target"] = 0
                dataset.append(negative_instance)
    return pd.concat(dataset, ignore_index=True)


def _map_product_with_offers(df_prods: pd.DataFrame) -> dict[str, list[str]]:
    mapping_prod_id_to_offer_ids = {}
    for _, row in df_prods.iterrows():
        if row["positivelyVerifiedOfferNames"] is None:
            continue
        pos_offer_ids = [
            e["offerId"]
            for e in row["positivelyVerifiedOfferNames"]
            if isinstance(e, dict)
        ]
        mapping_prod_id_to_offer_ids[row["id"]] = pos_offer_ids
    return mapping_prod_id_to_offer_ids


def _get_all_offer_ids(mapping_prod_id_to_offer_ids: dict[str, list[str]]) -> list[str]:
    res = set()
    for offer_ids in mapping_prod_id_to_offer_ids.values():
        res.update(offer_ids)
    return list(res)


def _generate_negative_instances(
    df_offers: pd.DataFrame,
    offer_ids_entire: list[str],
    pos_offer_id: str,
    n_neg_sample: int,
) -> list:
    negative_offer_id_candidates = [e for e in offer_ids_entire if e != pos_offer_id]
    negative_offer_ids = random.sample(negative_offer_id_candidates, n_neg_sample)

    df_neg_offer_instances = df_offers[
        df_offers["offerId"].isin(negative_offer_ids)
    ].add_prefix("offer-")
    df_neg_offer_instances.reset_index(drop=True, inplace=True)

    negative_offer_instances = []
    for _, df_neg_offer_instance in df_neg_offer_instances.iterrows():
        df_neg_offer_instance = df_neg_offer_instance.to_frame().T
        df_neg_offer_instance.reset_index(drop=True, inplace=True)
        negative_offer_instances.append(df_neg_offer_instance)
    return negative_offer_instances


def _generate_positive_instances(
    df_offers: pd.DataFrame, pos_offer_ids
) -> list[pd.DataFrame]:
    positive_offer_instance = []
    for pos_offer_id in pos_offer_ids:
        df_positive_offer_instance = df_offers[
            df_offers["offerId"] == pos_offer_id
        ].add_prefix("offer-")
        df_positive_offer_instance.reset_index(drop=True, inplace=True)
        positive_offer_instance.append(df_positive_offer_instance)
    return positive_offer_instance
