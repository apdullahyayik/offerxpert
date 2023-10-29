"""CLI entry point."""
# pylint:disable=import-outside-toplevel,relative-beyond-top-level
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from .cli import command, create_parser
from .version import VERSION

MAINPARSER_HELP = "print current version number"
SUBPARSERS_HELP = "%(prog)s must be called with a command:"
DESCRIPTION = """Description"""


def main():
    """Handle CLI arguments."""
    print("🌻💬 OfferXpert!")
    create_parser(
        DESCRIPTION,
        VERSION,
        [
            _build_model_parser,
            _evaluate_parser,
            _check_data_parser,
        ],
    )


def _build_model_parser(subparsers: Any) -> ArgumentParser:
    """Create parser for the 'build-model' command."""
    parser = subparsers.add_parser(
        "build-model",
        help="Build model.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=100,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size-train",
        type=int,
        default=256,
        help="Batch size used in training loop.",
    )
    parser.add_argument(
        "--batch-size-process",
        type=int,
        default=32,
        help="Batch size used in data processing.",
    )
    parser.add_argument(
        "--print-every-n-batch",
        type=int,
        default=10,
        help="Print every n number of batch/es in the training loop.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs allowed without improvement.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="experiment-default",
        help="Name of experiment.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="./models",
        help="Path to experiments folder.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Path to cache folder where built data is saved.",
    )
    parser.add_argument(
        "--n-neg-sample",
        type=int,
        default=1,
        help="Number of negative sampling.",
    )

    parser.set_defaults(func=_run_build_model)
    return parser


@command("Successfully ran build-model command", "build-model command failed")
def _run_build_model(_: ArgumentParser, args: Any):
    # pylint: disable=unused-argument
    from offerexpert.build_model import build_model

    experiment_folder = Path(args.experiment_dir) / args.experiment_name
    cache_folder = Path(args.cache_dir)
    build_model(
        args.n_epochs,
        args.lr,
        args.batch_size_train,
        args.batch_size_process,
        args.print_every_n_batch,
        args.patience,
        experiment_folder,
        cache_folder,
        args.n_neg_sample,
    )


def _evaluate_parser(subparsers: Any) -> ArgumentParser:
    """Create parser for the 'evaluate' command."""
    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model.",
    )
    parser.add_argument(
        "--batch-size-train",
        type=int,
        default=256,
        help="Batch size used in training loop.",
    )
    parser.add_argument(
        "--batch-size-process",
        type=int,
        default=32,
        help="Batch size used in data processing.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="experiment-default",
        help="Name of experiment.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="./models",
        help="Path to experiments folder.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Path to cache folder where built data is saved.",
    )
    parser.add_argument(
        "--n-neg-sample",
        type=int,
        default=1,
        help="Number of negative sampling.",
    )

    parser.set_defaults(func=_run_evaluate)
    return parser


@command("Successfully ran evaluate command", "evaluate command failed")
def _run_evaluate(_: ArgumentParser, args: Any):
    # pylint: disable=unused-argument
    from offerexpert.evaluate import evaluate

    experiment_folder = Path(args.experiment_dir) / args.experiment_name
    cache_folder = Path(args.cache_dir)
    evaluate(
        experiment_folder,
        cache_folder,
        args.batch_size_process,
        args.batch_size_train,
        args.n_neg_sample,
    )


def _check_data_parser(subparsers: Any) -> ArgumentParser:
    """Create parser for the 'check-data' command."""
    parser = subparsers.add_parser(
        "check-data",
        help="Evaluate model.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="experiment-default",
        help="Name of experiment.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="./models",
        help="Path to experiments folder.",
    )

    parser.set_defaults(func=_run_check_data)
    return parser


@command("Successfully ran check-data command", "check-data command failed")
def _run_check_data(_: ArgumentParser, args: Any):
    # pylint: disable=unused-argument
    from offerexpert.check_data import check_data

    experiment_folder = Path(args.experiment_dir) / args.experiment_name
    check_data(experiment_folder)


if __name__ == "__main__":
    main()
