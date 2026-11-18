from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _run_in_tmp_path(monkeypatch: Any, tmp_path: Path) -> None:
    """Run every test in this directory from a fresh temporary directory.

    The shared training and checkpoint test mixins in
    :mod:`metatrain.utils.testing` call ``trainer.train`` with
    ``checkpoint_dir="."`` (and ``""``), so the trainer writes ``model_0.ckpt``
    into the *current working directory*.  Whether that pollutes the repository
    or a scratch directory then depends on where ``pytest`` was launched from,
    and a stale ``model_0.ckpt`` sitting next to the tests is how it shows up.

    Chdir-ing here rather than editing the shared mixins keeps the fix on this
    fork's side of the rebase: it touches no upstream file.  Every path the NEP
    tests read is absolute (``DATASET_PATH`` is built from ``__file__``) or
    comes from ``tmp_path``, so nothing else depends on the working directory.
    """
    monkeypatch.chdir(tmp_path)
