"""Module for building model."""
# pylint: disable=too-many-arguments,too-many-locals,invalid-name
import logging
import pickle
import sys
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader

from offerexpert.dataset import OfferExpertDataset, split_into_train_test
from offerexpert.embedding.sentence_transformer import (
    create_sentence_transformer_vectors,
)
from offerexpert.evaluate import measure_performance
from offerexpert.loss import BinaryCrossEntropyLoss
from offerexpert.modeling import OfferExpertModel
from offerexpert.prepare_data import prepare_data
from offerexpert.trainer import trainer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    stream=sys.stdout,
    level=logging.INFO,
)


def build_model(
    n_epochs: int,
    lr: float,
    batch_size_train: int,
    batch_size_process: int,
    print_every_n_batch: int,
    patience: int,
    experiment_folder: Path,
    cache_folder: Path,
    n_neg_sample: int,
):
    """Build data, train model and evaluate."""
    experiment_folder.mkdir(exist_ok=True, parents=True)
    cache_folder.mkdir(exist_ok=True, parents=True)

    train_data_loader, test_data_loader, _, _, meta_data = build_data(
        batch_size_process,
        batch_size_train,
        n_neg_sample,
        cache_folder,
    )
    _train(
        train_data_loader,
        test_data_loader,
        lr,
        n_epochs,
        patience,
        print_every_n_batch,
        experiment_folder,
    )

    model = OfferExpertModel(vector_size=test_data_loader.dataset.vector_size)  # noqa
    model.load_state_dict(torch.load(experiment_folder / "model.pt"))
    measure_performance(model, test_data_loader, meta_data, experiment_folder)


def build_data(
    batch_size_process: int,
    batch_size_train: int,
    n_neg_sample: int,
    cache_folder: Path,
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor, dict]:
    """Build data."""
    path_cached_data = cache_folder / "built-data.pkl"

    if path_cached_data.exists():
        logging.info("detected cached data at %s.", path_cached_data)
        with open(path_cached_data, "rb") as fp:
            (
                train_data_loader,
                test_data_loader,
                y_test,
                y_train,
                meta_data,
            ) = pickle.load(fp)
            return train_data_loader, test_data_loader, y_test, y_train, meta_data

    df_dataset = prepare_data(n_neg_sample)
    logging.info("Prepared data")
    X_train, X_test, y_train, y_test = split_into_train_test(df_dataset)

    product_train_vector = create_sentence_transformer_vectors(
        X_train["product"].to_list(), batch_size_process
    )
    product_test_vector = create_sentence_transformer_vectors(
        X_test["product"].to_list(), batch_size_process
    )
    offer_train_vector = create_sentence_transformer_vectors(
        X_train["offer"].to_list(), batch_size_process
    )
    offer_test_vector = create_sentence_transformer_vectors(
        X_test["offer"].to_list(), batch_size_process
    )

    meta_data = {
        "offer_ids_train": X_train["offer-offerId"].values,
        "offer_ids_test": X_test["offer-offerId"].values,
        "prod_ids_train": X_train["product-id"].values,
        "prod_ids_test": X_test["product-id"].values,
        "product-amount_mean": X_test["product-amount_mean"].values,
        "product-amount_standardDeviation": X_test[
            "product-amount_standardDeviation"
        ].values,
        "offer-priceAmount": X_test["offer-priceAmount"].values,
    }

    y_train = torch.from_numpy(y_train.values)
    y_test = torch.from_numpy(y_test.values)
    train_dataset = OfferExpertDataset(
        product_train_vector, offer_train_vector, y_train
    )
    test_dataset = OfferExpertDataset(product_test_vector, offer_test_vector, y_test)

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True
    )
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size_train)

    with open(path_cached_data, "wb") as fp:
        pickle.dump((train_data_loader, test_data_loader, y_test, y_train, meta_data), fp)  # type: ignore
        logging.info("cached built data at %s, for further usage.", path_cached_data)
    return train_data_loader, test_data_loader, y_test, y_train, meta_data


def _train(
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    lr: float,
    n_epochs: int,
    patience: int,
    print_every_n_batch: int,
    experiment_folder: Path,
):
    model = OfferExpertModel(vector_size=train_data_loader.dataset.vector_size)  # type: ignore

    loss_func = BinaryCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    trainer(
        model,
        train_data_loader,
        test_data_loader,
        optimizer,
        loss_func,
        n_epochs,
        patience,
        print_every_n_batch,
        experiment_folder,
    )
