"""CLI entry point."""
# pylint:disable=relative-beyond-top-level
import argparse
import sys
from functools import wraps
from typing import Any, Callable, Dict, List

from offerexpert.exceptions import OfferExpertError

MAINPARSER_HELP = "print current version number"
SUBPARSERS_HELP = "%(prog)s must be called with a command:"


def create_parser(
    description: str,
    version: str,
    subparsers_funcs: List[Callable[..., Any]],
):
    """Create parser with specified sub parsers."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {version}",
        help=MAINPARSER_HELP,
    )

    subparsers = parser.add_subparsers(help=SUBPARSERS_HELP, dest="command")
    subparsers.required = True

    for subparsers_func in subparsers_funcs:
        subparsers_func(subparsers)

    args = parser.parse_args()
    args.func(parser, args)


def command(success_msg: str, fail_msg: str) -> Callable:
    """Wrap a command for the CLI with a success and a fail message."""

    def decorator_command(func: Callable[..., Any]) -> Callable:
        @wraps(func)
        def wrapper_command(*args: List[Any], **kwargs: Dict[str, Any]):
            try:
                func(*args, **kwargs)
                print(success_msg + " ✅")
                sys.exit(0)
            except OfferExpertError as error:
                print("")
                print(fail_msg + " 💔 💔 💔")
                print("Error: " + str(error))
                print("")
                sys.exit((1))

        return wrapper_command

    return decorator_command
