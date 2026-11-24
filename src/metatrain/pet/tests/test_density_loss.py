import copy
from pathlib import Path

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.pet import Trainer
from metatrain.utils.data import DatasetInfo, get_dataset
from metatrain.utils.data.atomic_basis_helpers import (
    get_prepare_atomic_basis_targets_transform,
)
from metatrain.utils.data.writers import DiskDatasetWriter
from metatrain.utils.density.basis import load_element_basis
from metatrain.utils.density.loss import DensityMSELoss, get_density_transforms
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DEFAULT_HYPERS, MODEL_HYPERS


AUX_BASIS = str(Path(__file__).parents[4] / "tests/resources/density/basis.json")
TARGET = "mtt::density"


def _irreps(atomic_number):
    offsets = load_element_basis(AUX_BASIS, atomic_number).offsets
    return [
        {"o3_lambda": angular, "o3_sigma": 1, "num": len(set(o.tolist()))}
        for angular, o in sorted(offsets.items())
    ]


def _water_dataset(path, n_systems):
    generator = torch.Generator().manual_seed(0)
    writer = DiskDatasetWriter(path)
    systems = []
    for _ in range(n_systems):
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.76, 0.59], [0.0, -0.76, 0.59]],
            dtype=torch.float64,
        )
        system = System(
            types=torch.tensor([8, 1, 1]),
            positions=positions
            + 0.1 * torch.randn(3, 3, generator=generator, dtype=torch.float64),
            cell=torch.zeros(3, 3, dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )
        keys, blocks = [], []
        for atomic_number, atoms in ((1, [1, 2]), (8, [0])):
            for irrep in _irreps(atomic_number):
                angular, n = irrep["o3_lambda"], irrep["num"]
                keys.append([angular, 1, atomic_number])
                blocks.append(
                    TensorBlock(
                        # a different spread for each type, so that the scales differ
                        values=atomic_number
                        * torch.randn(
                            len(atoms),
                            2 * angular + 1,
                            n,
                            generator=generator,
                            dtype=torch.float64,
                        ),
                        samples=Labels(
                            ["system", "atom"], torch.tensor([[0, a] for a in atoms])
                        ),
                        components=[
                            Labels(
                                ["o3_mu"], torch.arange(-angular, angular + 1)[:, None]
                            )
                        ],
                        properties=Labels(["n"], torch.arange(n)[:, None]),
                    )
                )
        target = TensorMap(
            Labels(["o3_lambda", "o3_sigma", "atom_type"], torch.tensor(keys)), blocks
        )
        writer.write([system], {TARGET: target})
        systems.append(system)
    writer.finish()
    return systems


class _NoScaler:
    def apply_scales(self, systems, targets, **kwargs):
        return targets


def test_validation_loss_is_the_physical_error(tmp_path, monkeypatch):
    """Check that the validation loss of a training with the density loss is the error
    of the predictions of the trained model, which include its scales and
    composition."""
    monkeypatch.chdir(tmp_path)
    systems = _water_dataset("dataset.zip", 4)
    irreps = {atomic_number: _irreps(atomic_number) for atomic_number in (1, 8)}
    dataset, target_info, extra_data_info = get_dataset(
        {
            "systems": {"read_from": "dataset.zip"},
            "targets": {
                TARGET: {
                    "quantity": "",
                    "unit": "",
                    "read_from": "dataset.zip",
                    "type": {"spherical": {"irreps": irreps}},
                    "sample_kind": "atom",
                    "num_subtargets": 1,
                }
            },
        }
    )
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 8],
        targets=target_info,
        extra_data=extra_data_info,
    )

    loss = init_with_defaults(LossSpecification)
    loss.update({"type": "density_mse", "metric": "coulomb", "aux_basis": AUX_BASIS})
    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["training"].update(
        num_epochs=1,
        batch_size=len(systems),
        num_workers=0,
        best_model_metric="loss",
        loss={TARGET: loss},
    )
    trainer = Trainer(hypers["training"])
    model = trainer.setup(MODEL_HYPERS, dataset_info)
    trainer.train(
        model=model,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )

    systems = [
        get_system_with_neighbor_lists(
            system.to(torch.float32), model.requested_neighbor_lists()
        )
        for system in systems
    ]
    model.eval()
    with torch.no_grad():
        prediction = model(systems, {TARGET: ModelOutput(sample_kind="atom")})
    target = mts.join(
        [dataset[i][TARGET] for i in range(len(systems))],
        axis="samples",
        different_keys="union",
    ).to(torch.float32)
    index = mts.join(
        [dataset[i]["mtt::aux::system_index"] for i in range(len(systems))],
        axis="samples",
    )
    prepare, _ = get_prepare_atomic_basis_targets_transform(
        dataset_info, model.internal_dataset_info()
    )
    extra = {"mtt::aux::system_index": index}
    target = prepare(systems, {TARGET: target}, extra)[1][TARGET]
    prediction = prepare(systems, prediction, extra)[1]
    (transform,) = get_density_transforms({TARGET: loss}, _NoScaler())
    extra = transform(systems, {TARGET: target}, {})[2]

    error = DensityMSELoss(
        TARGET, None, 1.0, "mean", aux_basis=AUX_BASIS, metric="coulomb"
    ).compute(prediction, {TARGET: target}, extra)
    assert abs(float(error) - trainer.best_metric) < 1e-4 * float(error)
