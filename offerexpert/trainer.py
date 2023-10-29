"""Module for training."""
# # pylint: disable=too-many-arguments,too-many-locals
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from offerexpert.early_stop import EarlyStop
from offerexpert.modeling import OfferExpertModel
from offerexpert.util import plot_train_report

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    stream=sys.stdout,
    level=logging.INFO,
)


def trainer(
    model: OfferExpertModel,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_func: torch.nn.Module,
    n_epochs: int,
    patience: int,
    print_every_n_batch: int,
    experiment_folder: Path,
):
    """Train model."""
    train_loss_all = torch.zeros(n_epochs)
    test_loss_all = torch.zeros(n_epochs)
    early_stop = EarlyStop(patience)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        num_batch = 0
        batch_done = time.time()
        data_count = 0

        for batch_idx, batch in enumerate(train_data_loader):
            num_batch += 1
            batch_start = time.time()
            batch_data_elapsed = batch_start - batch_done

            product_vectors, offer_vectors, target = batch

            optimizer.zero_grad()

            output = model(product_vectors, offer_vectors)
            output = torch.squeeze(output, -1)
            loss = loss_func(output, target.float())

            # output1 = model.forward_shared_tower(product_vectors)
            # output2 = model.forward_shared_tower(offer_vectors)
            # loss = loss_func(output1, output2, target)

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            batch_done = time.time()
            batch_elapsed = batch_done - batch_start

            data_count += product_vectors.shape[0]

            if batch_idx % print_every_n_batch == 0:
                logging.info(
                    "Train Epoch: %s [%.0f/%.0f (%.0f)]\tLoss: %.6f Batch time  %.3f Data load:%.3f",  # noqa: E501
                    epoch + 1,
                    data_count,
                    len(train_data_loader.dataset),  # type:ignore
                    100.0 * data_count / len(train_data_loader.dataset),
                    loss.item(),
                    batch_elapsed,
                    batch_data_elapsed,
                )

        train_loss = epoch_loss / num_batch
        train_loss_all[epoch] = train_loss

        # Validation Part
        model.eval()
        batches = 0
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_data_loader:
                batches += 1

                product_vectors, offer_vectors, target = batch

                output = model(product_vectors, offer_vectors)
                output = torch.squeeze(output, -1)
                test_loss += loss_func(output, target.float()).item()

                # output1 = model.forward_shared_tower(product_vectors)
                # output2 = model.forward_shared_tower(offer_vectors)
                # test_loss += loss_func(output1, output2, target)

            test_loss /= batches
            test_loss_all[epoch] = test_loss

        logging.info("Test set: Average loss: %.6f", test_loss)

        history = {
            "loss": train_loss_all[: epoch + 1].detach().cpu().numpy(),
            "val_loss": test_loss_all[: epoch + 1].detach().cpu().numpy(),
        }

        pd.DataFrame.from_dict(history).to_csv(
            experiment_folder / "train_history.csv",
            index=False,
        )

        # Epoch Done Here
        if early_stop.is_continue(test_loss):
            if early_stop.is_improvement:
                model_file = experiment_folder / "model.pt"
                logging.info("best model (at epoch %s) saved %s", epoch + 1, model_file)

                torch.save(model.state_dict(), model_file)
        else:
            logging.info(
                "training has been early-stopped at epoch %s, "
                "after %s number of unsuccessful consecutive epochs with test loss "  # noqa: E501
                "at: %.6f.",
                epoch + 1,
                patience,
                test_loss,
            )
            break

    plot_train_report(history, onset_epoch=1).savefig(
        experiment_folder / "train_report.png"
    )
