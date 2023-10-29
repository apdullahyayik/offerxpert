"""Module for evaluating model."""
import json
import logging
import sys
from pathlib import Path

import faiss
import numpy as np
import seaborn as sn
import torch
from faiss import write_index
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from offerexpert.modeling import OfferExpertModel

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    stream=sys.stdout,
    level=logging.INFO,
)


def evaluate(
    experiment_folder: Path,
    cache_folder: Path,
    batch_size_process: int,
    batch_size_train: int,
    n_neg_sample: int,
):
    """Evaluate."""
    from offerexpert.build_model import build_data  # noqa

    _, test_data_loader, _, _, ids = build_data(
        batch_size_process,
        batch_size_train,
        n_neg_sample,
        cache_folder,
    )
    model = OfferExpertModel(vector_size=test_data_loader.dataset.vector_size)  # noqa
    model.load_state_dict(torch.load(experiment_folder / "model.pt"))
    measure_performance(model, test_data_loader, ids, experiment_folder)


def measure_performance(
    model: OfferExpertModel,
    test_data_loader: DataLoader,
    meta_data: dict[str, str],
    experiment_folder: Path,
):
    """Measure performance."""
    model.eval()

    # Save IDs in the test set
    prod_ids_test: np.array = meta_data["prod_ids_test"]
    targets = test_data_loader.dataset.target_vector  # noqa
    prod_ids_test_positive: np.array = prod_ids_test[targets == 1]
    path_prod_ids_test_positive = experiment_folder / "prod_ids_test_positive"
    with open(path_prod_ids_test_positive, "w") as json_file:
        json.dump(prod_ids_test_positive.tolist(), json_file)
    logging.info("Saved prod ids test positive to %s", path_prod_ids_test_positive)

    offer_ids_test: np.array = meta_data["offer_ids_test"]
    offer_ids_test_positive: np.array = offer_ids_test[targets == 1]
    path_offer_ids_test_positive = experiment_folder / "offer_ids_test_positive"
    with open(path_offer_ids_test_positive, "w") as json_file:
        json.dump(offer_ids_test_positive.tolist(), json_file)
    logging.info("Saved offer ids test positive to %s", path_offer_ids_test_positive)

    # Amount
    offer_amounts = meta_data["offer-priceAmount"]
    product_amount_means = meta_data["product-amount_mean"]
    product_amount_stds = meta_data["product-amount_standardDeviation"]

    product_outputs_test = _get_outputs_products(model, test_data_loader)
    offer_outputs_test = _get_outputs_offers(model, test_data_loader)

    # Create index
    indexer = faiss.IndexFlatL2(product_outputs_test.shape[1])
    indexer.train(product_outputs_test)
    indexer.add(product_outputs_test)

    # Save index
    path_faiss_index = experiment_folder / "faiss-indexer.bin"
    write_index(indexer, str(path_faiss_index))
    logging.info("Saved index to %s", path_faiss_index)

    # Search index
    _, matched_prod_indexes = indexer.search(offer_outputs_test, 10)  # type: ignore
    matched_prod_ids = [prod_ids_test_positive[e] for e in matched_prod_indexes]

    predictions = []
    num_correct = 0
    num_wrong = 0
    for idx, (predicted_indexes, predicted_ids, actual_id) in enumerate(
        zip(matched_prod_indexes, matched_prod_ids, prod_ids_test_positive)
    ):
        # select by amount, i.e. not outlier
        offer_amount_predicted = offer_amounts[idx]

        for index_prediction, predicted_index in enumerate(predicted_indexes):
            product_amount_mean = product_amount_means[predicted_index]
            product_amount_std = product_amount_stds[predicted_index]
            if isin_expected_range(
                offer_amount_predicted, product_amount_mean, product_amount_std
            ):
                break
        else:
            index_prediction = 0

        predicted_id = predicted_ids[index_prediction]

        if predicted_id == actual_id:
            num_correct += 1
            predictions.append(1)
        else:
            num_wrong += 1
            predictions.append(0)

    actual = [1 for _ in predictions]
    report = classification_report(actual, predictions)
    print(report)

    path_cm = experiment_folder / "cm.png"
    _save_confusion_matrix(actual, predictions, path_cm)
    logging.info("Confusion matrix is saved at %s", path_cm)

    logging.info(
        f"Acc: {(100 * num_correct / (num_wrong + num_correct)):.2f}% ({num_correct}/{num_correct + num_wrong})"
    )


def _get_outputs_products(
    model: OfferExpertModel, test_data_loader: DataLoader
) -> np.array:
    product_outputs = list()
    for batch in test_data_loader:
        product_vectors, _, target = batch
        product_vectors_positive = product_vectors[target == 1]
        output = model.forward_shared_tower(product_vectors_positive)
        product_outputs.append(output.detach().numpy())
    return np.concatenate(product_outputs, axis=0)


def _get_outputs_offers(
    model: OfferExpertModel, test_data_loader: DataLoader
) -> np.ndarray:
    offer_outputs = list()
    for batch in test_data_loader:
        _, offer_vectors, target = batch
        offer_vectors = offer_vectors[target == 1]
        output = model.forward_shared_tower(offer_vectors)
        offer_outputs.append(output.detach().numpy())
    return np.concatenate(offer_outputs, axis=0)


def isin_expected_range(
    value: float | str | None,
    mean: float | str | None,
    standard_deviation: float | str | None,
) -> bool:
    """Return whether given value is in expected range."""
    if value is None or mean is None or standard_deviation is None:
        return True

    if value == "" or mean == "" or standard_deviation == "":
        return True

    lower_bound = float(mean) - 3 * float(standard_deviation)
    upper_bound = float(mean) + 3 * float(standard_deviation)

    if lower_bound <= float(value) <= upper_bound:
        return True
    return False


def _save_confusion_matrix(
    y_true: list | tuple | np.ndarray | torch.Tensor,
    y_pred: list | tuple | np.ndarray | torch.Tensor,
    path_file: Path,
):
    labels = tuple(set(np.unique(y_pred)).union(set(np.unique(y_true))))
    cm_plot = confusion_matrix(y_true, y_pred, labels=labels)
    font_scale = 2
    plt.figure(figsize=(20, 14))
    sn.set(font_scale=font_scale)
    sn.heatmap(
        cm_plot,
        annot=True,
        cmap="Blues",
        xticklabels=labels,  # type: ignore
        yticklabels=labels,  # type: ignore
        fmt="g",
    )
    plt.savefig(path_file)
    plt.clf()
    sn.reset_defaults()
