import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import metatensor.torch as mts
import metatomic.torch as mta
import numpy as np
import torch
from metatensor.torch import TensorMap
from metatomic.torch import ModelCapabilities, System

from .writers import Writer, _split_tensormaps


class DiskDatasetWriter(Writer):
    """
    Write systems and predictions to a zip file, each system in a separate folder inside
    the zip.

    :param path: Path to the output zip file.
    :param capabilities: Model capabilities.
    :param append: If True, open the zip file in append mode.
    """

    def __init__(
        self,
        path: Union[str, Path],
        capabilities: Optional[
            ModelCapabilities
        ] = None,  # unused, but matches base signature
        append: Optional[bool] = False,  # if True, open zip in append mode
    ):
        super().__init__(filename=path, capabilities=capabilities, append=append)
        self._append = bool(append)
        self._atom_counts_prefix: Optional[np.ndarray] = None

        num_existing = 0
        if self._append:
            # Continue indexing after the existing entries, instead of overwriting
            # them at "0/", "1/", ... Also pick up an existing atom-count file (if
            # any and not stale) so it can be extended in finish() rather than
            # dropped.
            try:
                with zipfile.ZipFile(path, "r") as existing:
                    namelist = existing.namelist()
                    # Continue after the highest existing entry number, not the
                    # entry count: they differ on zips with gaps or duplicated
                    # names, where count-based indexing would silently collide
                    # with existing entries.
                    entry_numbers = [
                        int(f.split("/")[0])
                        for f in namelist
                        if f.endswith("/system.mta")
                    ]
                    if entry_numbers:
                        num_existing = max(entry_numbers) + 1
                    if "_atom_counts.npy" in namelist:
                        with existing.open("_atom_counts.npy", "r") as f:
                            prefix = np.load(f)
                        if len(prefix) >= num_existing:
                            self._atom_counts_prefix = prefix[:num_existing]
            except (FileNotFoundError, zipfile.BadZipFile):
                pass  # append=True but the file doesn't exist (or is empty) yet
        self.index = num_existing

        mode: Literal["w", "a"] = "a" if append else "w"
        self.zip_file = zipfile.ZipFile(path, mode)
        self._atom_counts: List[int] = []

    def write(self, systems: List[System], predictions: Dict[str, TensorMap]) -> None:
        """
        Write a single (system, predictions) into the zip under
        a new folder "<index>/".

        :param systems: List of systems to write.
        :param predictions: Dictionary of TensorMaps with predictions for the systems.
        """

        if len(systems) == 1:
            # Avoid reindexing samples
            split_predictions = [predictions]
        else:
            split_predictions = _split_tensormaps(
                systems, predictions, istart_system=self.index
            )

        for system, preds in zip(systems, split_predictions, strict=True):
            # system
            with self.zip_file.open(f"{self.index}/system.mta", "w") as f:
                mta.save(f, system.to("cpu").to(torch.float64))
            self._atom_counts.append(len(system))

            # each target
            for target_name, tensor_map in preds.items():
                with self.zip_file.open(f"{self.index}/{target_name}.mts", "w") as f:
                    tensor_map = mts.make_contiguous(tensor_map)
                    buf = tensor_map.to("cpu").to(torch.float64)
                    # metatensor.torch.save_buffer returns a torch.Tensor buffer
                    buffer = buf.save_buffer()
                    np.save(f, buffer.numpy())

            self.index += 1

    def finish(self) -> None:
        """
        Write the per-structure atom-count file (``_atom_counts.npy``) and close
        the zip file.

        In append mode, this is skipped unless a previous session already had one
        (see ``__init__``), since this writer has no way to know the atom counts
        of entries it didn't write itself.
        """
        new_counts = np.array(self._atom_counts, dtype=np.int64)
        if not self._append:
            with self.zip_file.open("_atom_counts.npy", "w") as f:
                np.save(f, new_counts)
        elif self._atom_counts_prefix is not None:
            full_counts = np.concatenate([self._atom_counts_prefix, new_counts])
            # Zip files don't support in-place replacement of an entry: this adds a
            # second "_atom_counts.npy" rather than overwriting the first. It is
            # harmless, as reading a duplicated name returns the last one written,
            # so the old, superseded array is not used (it is dead weight in the
            # zip file, though)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Duplicate name")
                with self.zip_file.open("_atom_counts.npy", "w") as f:
                    np.save(f, full_counts)
        self.zip_file.close()
