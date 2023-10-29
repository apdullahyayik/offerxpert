"""Module for processing positively verified offer names."""
import pandas as pd


def process_positively_verified_offer_name(df: pd.DataFrame) -> pd.DataFrame:
    """Process positively verified names."""
    df["product-positively_verified_offer_name"] = df.apply(
        lambda x: ",".join(
            [
                e["name"]
                for e in x["product-positivelyVerifiedOfferNames"]
                if isinstance(e, dict)
            ]
        ),
        axis=1,
    )
    return df
