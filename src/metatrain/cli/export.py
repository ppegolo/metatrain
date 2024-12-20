import argparse
import logging
from pathlib import Path
from typing import Any, Union

from metatensor.torch.atomistic import is_atomistic_model

from ..utils.architectures import find_all_architectures
from ..utils.io import check_file_extension, load_model
from .formatter import CustomHelpFormatter


logger = logging.getLogger(__name__)


def _add_export_model_parser(subparser: argparse._SubParsersAction) -> None:
    """Add `export_model` paramaters to an argparse (sub)-parser."""

    if export_model.__doc__ is not None:
        description = export_model.__doc__.split(r":param")[0]
    else:
        description = None

    # If you change the synopsis of these commands or add new ones adjust the completion
    # script at `src/metatrain/share/metatrain-completion.bash`.
    parser = subparser.add_parser(
        "export",
        description=description,
        formatter_class=CustomHelpFormatter,
    )
    parser.set_defaults(callable="export_model")

    parser.add_argument(
        "architecture_name",
        type=str,
        choices=find_all_architectures(),
        help="name of the model's architecture",
    )
    parser.add_argument(
        "path",
        type=str,
        help=(
            "Saved model which should be exported. Path can be either a URL or a "
            "local file."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        required=False,
        default="exported-model.pt",
        help="Filename of the exported model (default: %(default)s).",
    )


def _prepare_export_model_args(args: argparse.Namespace) -> None:
    """Prepare arguments for export_model."""
    args.model = load_model(
        path=args.__dict__.pop("path"),
        architecture_name=args.__dict__.pop("architecture_name"),
    )


def export_model(model: Any, output: Union[Path, str] = "exported-model.pt") -> None:
    """Export a trained model allowing it to make predictions.

    This includes predictions within molecular simulation engines. Exported models will
    be saved with a ``.pt`` file ending. If ``path`` does not end with this file
    extensions ``.pt`` will be added and a warning emitted.

    :param model: model to be exported
    :param output: path to save the model
    """
    path = str(
        Path(check_file_extension(filename=output, extension=".pt"))
        .absolute()
        .resolve()
    )
    extensions_path = str(Path("extensions/").absolute().resolve())

    if not is_atomistic_model(model):
        model = model.export()

    model.save(path, collect_extensions=extensions_path)
    logger.info(f"Model exported to '{path}' and extensions to '{extensions_path}'")
