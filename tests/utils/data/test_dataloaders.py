"""Tests for the shared dataloader-building utilities in
``metatrain.utils.data.dataloaders``, covering ``max_atoms_per_batch`` packing
end-to-end.

Architectures each used to carry their own ``test_train_max_atoms_per_batch``
regression test exercising this through a full model training. Since
``build_train_dataloaders``/``build_val_dataloaders`` are the single shared
implementation every architecture trainer calls into, one direct test here is
enough: it removes the duplication without losing coverage.
"""

from pathlib import Path

import torch
from metatensor.learn.data import Dataset
from metatomic.torch import NeighborListOptions
from omegaconf import OmegaConf

from metatrain.utils.data import build_train_dataloaders, build_val_dataloaders
from metatrain.utils.data.dataset import CollateFn, unpack_batch
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.data.samplers import MaxAtomDistributedBatchSampler
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


RESOURCES_PATH = Path(__file__).resolve().parents[2] / "resources"
DATASET_PATH = str(RESOURCES_PATH / "qm9_reduced_100.xyz")


def _build_dataset() -> Dataset:
    systems = read_systems(DATASET_PATH)
    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf))
    return Dataset.from_dict({"system": systems, "energy": targets["energy"]})


def test_max_atoms_per_batch_end_to_end():
    """``max_atoms_per_batch``/``min_atoms_per_batch`` pack batches by atom
    count instead of a fixed ``batch_size``, and the resulting ``DataLoader``
    iterates into correctly collated batches that respect those bounds."""
    dataset = _build_dataset()
    collate_fn = CollateFn(target_keys=["energy"])

    train_dataloaders, epoch_samplers = build_train_dataloaders(
        train_datasets=[dataset],
        train_distributed_samplers=[None],
        collate_fn_train=collate_fn,
        batch_size=5,
        max_atoms_per_batch=20,
        min_atoms_per_batch=1,
        num_workers=0,
    )
    assert len(train_dataloaders) == 1
    assert len(epoch_samplers) == 1

    batch_sampler = train_dataloaders[0].batch_sampler
    assert isinstance(batch_sampler, MaxAtomDistributedBatchSampler)
    assert batch_sampler is epoch_samplers[0]

    # Every packed batch respects the configured bounds, and small structures
    # get combined into the same batch (as opposed to silently falling back
    # to one structure per batch).
    packed_batches = batch_sampler.all_batches
    assert len(packed_batches) > 0
    for batch in packed_batches:
        atom_count = sum(len(dataset[i].system) for i in batch)
        assert 1 <= atom_count <= 20
    assert any(len(batch) > 1 for batch in packed_batches)

    # The DataLoader must actually be iterable end-to-end with real collation:
    # every emitted batch decodes back into systems within the atom bounds.
    n_batches = 0
    n_systems_seen = 0
    for raw_batch in train_dataloaders[0]:
        systems, targets, _ = unpack_batch(raw_batch)
        assert sum(len(s) for s in systems) <= 20
        assert "energy" in targets
        n_batches += 1
        n_systems_seen += len(systems)
    assert n_batches == len(packed_batches)
    assert n_systems_seen == len(dataset)

    # build_val_dataloaders mirrors the same packing logic.
    val_dataloaders = build_val_dataloaders(
        val_datasets=[dataset],
        val_distributed_samplers=[None],
        collate_fn_val=collate_fn,
        batch_size=5,
        max_atoms_per_batch=20,
        num_workers=0,
    )
    assert len(val_dataloaders) == 1
    val_batch_sampler = val_dataloaders[0].batch_sampler
    assert isinstance(val_batch_sampler, MaxAtomDistributedBatchSampler)
    for batch in val_batch_sampler.all_batches:
        atom_count = sum(len(dataset[i].system) for i in batch)
        assert atom_count <= 20


def test_worker_round_trip_is_lossless():
    """Batches collated in dataloader worker processes decode to the systems, neighbor
    lists and targets of the dataset."""
    systems = read_systems(DATASET_PATH)[:10]
    nl_options = NeighborListOptions(cutoff=3.0, full_list=True, strict=True)
    systems = [get_system_with_neighbor_lists(s, [nl_options]) for s in systems]
    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf))
    dataset = Dataset.from_dict({"system": systems, "energy": targets["energy"][:10]})
    collate_fn = CollateFn(target_keys=["energy"])

    loader = build_val_dataloaders(
        val_datasets=[dataset],
        val_distributed_samplers=[None],
        collate_fn_val=collate_fn,
        batch_size=4,
        max_atoms_per_batch=None,
        num_workers=2,
    )[0]

    decoded_systems, decoded_energies = [], []
    for batch in loader:
        batch_systems, batch_targets, _ = unpack_batch(batch)
        decoded_systems.extend(batch_systems)
        decoded_energies.append(batch_targets["energy"].block().values)

    assert len(decoded_systems) == len(systems)
    for system, decoded in zip(systems, decoded_systems, strict=True):
        assert torch.equal(system.positions, decoded.positions)
        assert torch.equal(system.types, decoded.types)
        assert torch.equal(system.cell, decoded.cell)
        assert torch.equal(system.pbc, decoded.pbc)
        assert decoded.known_neighbor_lists() == [nl_options]
        nl = system.get_neighbor_list(nl_options)
        decoded_nl = decoded.get_neighbor_list(nl_options)
        assert torch.equal(nl.samples.values, decoded_nl.samples.values)
        assert torch.equal(nl.values, decoded_nl.values)
    energies = torch.cat([t.block().values for t in targets["energy"][:10]])
    assert torch.equal(torch.cat(decoded_energies), energies)
