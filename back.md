def is_amount_in_expected_range(row) -> str:
# TODO add unit test.
mean_ = row["product-amount_mean"]
standard_deviation_ = row["product-amount_standardDeviation"]
amount = row["offer-priceAmount"]

    if mean_ is None or standard_deviation_ is None or amount is None:
        return ""

    lower_bound = float(mean_) - 2 * float(standard_deviation_)
    upper_bound = float(mean_) + 2 * float(standard_deviation_)

    if lower_bound <= float(amount) <= upper_bound:
        return "Amount is in expected range."

    return "Amount is in expected range."

df_dataset["cross-is_amount_in_expected_range"] = df_dataset.apply(
lambda x: is_amount_in_expected_range(x), axis=1
)

[//]: # (train hisrior,)

[//]: # (model, and json file save folder)

# cach data to make a quick train
# add glove vectors
# expand evaluation, just compute f1
# TODO add unit test.
# update read me
