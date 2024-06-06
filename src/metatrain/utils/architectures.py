import difflib
from importlib.util import find_spec
from pathlib import Path
from typing import Dict, List, Union

from omegaconf import OmegaConf

from .. import PACKAGE_ROOT


def check_architecture_name(name: str) -> None:
    """Check if the requested architecture is avalible.

    If the architecture is not found an :func:`ValueError` is raised. If an architecture
    with the same name as an experimental or deprecated architecture exist, this
    architecture is suggested. If no architecture exist the closest architecture is
    given to help debugging typos.

    :param name: name of the architecture
    :raises ValueError: if the architecture is not found
    """
    try:
        if find_spec(f"metatrain.{name}") is not None:
            return
        elif find_spec(f"metatrain.experimental.{name}") is not None:
            msg = (
                f"Architecture {name!r} is not a stable architecture. An "
                "experimental architecture with the same name was found. Set "
                f"`name: experimental.{name}` in your options file to use this "
                "experimental architecture."
            )
        elif find_spec(f"metatrain.deprecated.{name}") is not None:
            msg = (
                f"Architecture {name!r} is not a stable architecture. A "
                "deprecated architecture with the same name was found. Set "
                f"`name: deprecated.{name}` in your options file to use this "
                "deprecated architecture."
            )
    except ModuleNotFoundError:
        closest_match = difflib.get_close_matches(
            word=name,
            possibilities=find_all_architectures(),
            cutoff=0.3,
        )
        msg = (
            f"Architecture {name!r} is not a valid architecture. Do you mean "
            f"{', '.join(closest_match)}?"
        )

    raise ValueError(msg)


def get_architecture_name(path: Union[str, Path]) -> str:
    """Name of an architecture based on path to pointing inside an architecture.

    The function should be used to determine the ``ARCHITECTURE_NAME`` based on the name
    of the folder.

    :param absolute_architecture_path: absolute path of the architecture directory
    :returns: architecture name
    :raises ValueError: if ``absolute_architecture_path`` does not point to a valid
        architecture directory.

    .. seealso::
        :py:func:`get_architecture_path` to get the relative path within the metatrain
        project of an architecture name.
    """
    path = Path(path)

    if path.is_dir():
        directory = path
    elif path.is_file():
        directory = path.parent
    else:
        raise ValueError(f"`path` {str(path)!r} does not exist")

    architecture_path = directory.relative_to(PACKAGE_ROOT)
    name = str(architecture_path).replace("/", ".")

    try:
        check_architecture_name(name)
    except ValueError as err:
        raise ValueError(
            f"`path` {str(path)!r} does not point to a valid architecture folder"
        ) from err

    return name


def get_architecture_path(name: str) -> Path:
    """Return the relative path to the architeture directory.

    Path based on the ``name`` within the metatrain project directory.

    :param name: name of the architecture
    :returns: path to the architecture directory

    .. seealso::
        :py:func:`get_architecture_name` to get the name based on an absolute path of an
        architecture.
    """
    check_architecture_name(name)
    return PACKAGE_ROOT / Path(name.replace(".", "/"))


def find_all_architectures() -> List[str]:
    """Find all currentlty available architectures.

    To find the architectures the function searches for the mandatory
    ``default-hypers.yaml`` file in each architectire directory.

    :returns: List of architectures names
    """
    options_files_path = PACKAGE_ROOT.rglob("default-hypers.yaml")

    architecture_names = []
    for option_file_path in options_files_path:
        architecture_names.append(get_architecture_name(option_file_path))

    return architecture_names


def get_default_hypers(name: str) -> Dict:
    """Dictionary of the default architecture hyperparameters.

    :param: name of the architecture
    :returns: default hyper paremeters of the architectures
    """
    check_architecture_name(name)
    default_hypers = OmegaConf.load(get_architecture_path(name) / "default-hypers.yaml")
    return OmegaConf.to_container(default_hypers)