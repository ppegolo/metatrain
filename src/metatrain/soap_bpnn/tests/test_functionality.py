import copy
import subprocess
import sys

import pytest
import torch
from metatomic.torch import ModelOutput, System

from metatrain.soap_bpnn import SoapBpnn
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


def _make_hypers(legacy: bool = True, add_lambda_basis: bool = True) -> dict:
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["legacy"] = legacy
    hypers["add_lambda_basis"] = add_lambda_basis
    hypers["soap"]["max_angular"] = 2
    hypers["soap"]["max_radial"] = 2
    hypers["bpnn"]["num_neurons_per_layer"] = 4
    hypers["bpnn"]["num_hidden_layers"] = 1
    return hypers


def _make_system(model: SoapBpnn) -> System:
    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        cell=2.0 * torch.eye(3),
        pbc=torch.tensor([True, True, True]),
    )
    return get_system_with_neighbor_lists(system, model.requested_neighbor_lists())


@pytest.mark.parametrize("legacy", [True, False])
def test_scalar_output(legacy):
    """Tests scalar energy prediction for both legacy and modern paths."""
    hypers = _make_hypers(legacy)
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[6],
        targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
    )
    model = SoapBpnn(hypers, dataset_info)
    system = _make_system(model)
    output = model(
        [system],
        {"energy": ModelOutput(quantity="energy", unit="eV", sample_kind="system")},
    )
    values = output["energy"].block().values
    assert values.shape == (1, 1)


@pytest.mark.parametrize("legacy", [True, False])
def test_cartesian_rank1(legacy):
    """Tests rank-1 Cartesian (e.g. dipole) prediction."""
    hypers = _make_hypers(legacy)
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[6],
        targets={
            "dipole": get_generic_target_info(
                "dipole",
                {
                    "quantity": "dipole",
                    "unit": "",
                    "type": {"cartesian": {"rank": 1}},
                    "num_subtargets": 1,
                    "sample_kind": "system",
                },
            )
        },
    )
    model = SoapBpnn(hypers, dataset_info)
    system = _make_system(model)
    output = model([system], {"dipole": ModelOutput(sample_kind="system")})
    values = output["dipole"].block().values
    # shape: (n_structures, 3, n_properties)
    assert values.shape == (1, 3, 1)


@pytest.mark.parametrize("add_lambda_basis", [True, False])
@pytest.mark.parametrize("legacy", [True, False])
@pytest.mark.parametrize("sample_kind", ["atom", "system"])
def test_nc_stress(legacy, sample_kind, add_lambda_basis):
    """Tests that rank-2 Cartesian NC stress is symmetric."""
    hypers = _make_hypers(legacy, add_lambda_basis)
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[6],
        targets={
            "non_conservative_stress": get_generic_target_info(
                "non_conservative_stress",
                {
                    "quantity": "stress",
                    "unit": "",
                    "type": {"cartesian": {"rank": 2}},
                    "num_subtargets": 1,
                    "sample_kind": sample_kind,
                },
            )
        },
    )
    model = SoapBpnn(hypers, dataset_info)
    system = _make_system(model)
    outputs = {"non_conservative_stress": ModelOutput(sample_kind=sample_kind)}
    stress = model([system], outputs)["non_conservative_stress"].block().values
    assert torch.allclose(stress, stress.transpose(-3, -2))


@pytest.mark.parametrize("add_lambda_basis", [True, False])
@pytest.mark.parametrize("legacy", [True, False])
def test_spherical_output(legacy, add_lambda_basis):
    """Tests spherical tensor (l=1, sigma=1) prediction."""
    hypers = _make_hypers(legacy, add_lambda_basis)
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[6],
        targets={
            "spherical_target": get_generic_target_info(
                "spherical_target",
                {
                    "quantity": "",
                    "unit": "",
                    "type": {
                        "spherical": {
                            "irreps": [{"o3_lambda": 1, "o3_sigma": 1}],
                        }
                    },
                    "num_subtargets": 1,
                    "sample_kind": "system",
                },
            )
        },
    )
    model = SoapBpnn(hypers, dataset_info)
    system = _make_system(model)
    output = model([system], {"spherical_target": ModelOutput(sample_kind="system")})
    tmap = output["spherical_target"]
    assert len(tmap.keys) == 1
    assert tmap.keys.names == ["o3_lambda", "o3_sigma"]
    assert tmap.keys.values[0].tolist() == [1, 1]
    assert tmap.block(0).values.shape == (1, 3, 1)


@pytest.mark.parametrize("add_lambda_basis", [True, False])
def test_mlp_head(add_lambda_basis):
    """
    Tests scalar prediction with an MLP head (legacy only, MLPHeadMap is per-species).
    """
    hypers = _make_hypers(legacy=True, add_lambda_basis=add_lambda_basis)
    hypers["heads"] = {"energy": "mlp"}
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[6],
        targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
    )
    model = SoapBpnn(hypers, dataset_info)
    system = _make_system(model)
    output = model(
        [system],
        {"energy": ModelOutput(quantity="energy", unit="eV", sample_kind="system")},
    )
    values = output["energy"].block().values
    assert values.shape == (1, 1)


@pytest.mark.parametrize("add_lambda_basis", [True, False])
@pytest.mark.parametrize("legacy", [True, False])
def test_multiple_targets(legacy, add_lambda_basis):
    """Tests simultaneous prediction of energy + rank-2 Cartesian stress."""
    hypers = _make_hypers(legacy, add_lambda_basis)
    targets = {
        "energy": get_energy_target_info("energy", {"unit": "eV"}),
        "non_conservative_stress": get_generic_target_info(
            "non_conservative_stress",
            {
                "quantity": "stress",
                "unit": "",
                "type": {"cartesian": {"rank": 2}},
                "num_subtargets": 1,
                "sample_kind": "system",
            },
        ),
    }
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[6],
        targets=targets,
    )
    model = SoapBpnn(hypers, dataset_info)
    system = _make_system(model)
    outputs = {
        "energy": ModelOutput(quantity="energy", unit="eV", sample_kind="system"),
        "non_conservative_stress": ModelOutput(sample_kind="system"),
    }
    result = model([system], outputs)
    assert "energy" in result
    assert "non_conservative_stress" in result
    assert result["energy"].block().values.shape == (1, 1)
    stress = result["non_conservative_stress"].block().values
    assert stress.shape == (1, 3, 3, 1)
    assert torch.allclose(stress, stress.transpose(-3, -2))


# Worker executed in a subprocess that reproduces the DEPLOYMENT condition: the
# sphericart operator library is loaded from the exported model's extensions
# directory, but `sphericart.torch` (whose import attaches the operators'
# autograd formulas, in Python) is never imported. In such a process a model
# relying on those formulas computes correct energies but silently drops every
# dY/dR term from the forces, making them non-conservative.
#
# The check is the definition of a conservative force: the work integrated
# around a closed loop in configuration space must vanish. The pair list is
# held fixed for the whole loop, so the energy is a single-valued function of
# the positions by construction and any non-zero loop work is a
# forward/backward inconsistency.
_CLOSED_LOOP_WORKER = """
import sys

import torch

from metatensor.torch import Labels, TensorBlock
from metatomic.torch import (
    ModelEvaluationOptions,
    ModelOutput,
    NeighborListOptions,
    System,
    load_atomistic_model,
)

out = sys.argv[1]
payload = torch.load(out + "/payload.pt", weights_only=True)
model = load_atomistic_model(
    out + "/model.pt", extensions_directory=out + "/extensions"
)
# guard the validity of the test itself: nothing here may register the
# sphericart autograd formulas
assert "sphericart" not in sys.modules, "worker accidentally imported sphericart"

nl_options = NeighborListOptions(
    cutoff=payload["cutoff"], full_list=payload["full_list"], strict=payload["strict"]
)
evaluation_options = ModelEvaluationOptions(
    length_unit="Angstrom",
    outputs={"energy": ModelOutput(quantity="energy", unit="eV", sample_kind="system")},
)
samples = payload["nl_samples"]
cell = payload["cell"]
types = payload["types"]


def forces(positions):
    positions = positions.clone().requires_grad_(True)
    system = System(
        types=types,
        positions=positions,
        cell=cell,
        pbc=torch.tensor([True, True, True]),
    )
    i = samples[:, 0].long()
    j = samples[:, 1].long()
    shifts = samples[:, 2:].to(cell.dtype)
    vectors = positions.detach()[j] - positions.detach()[i] + shifts @ cell
    neighbors = TensorBlock(
        values=vectors.reshape(-1, 3, 1),
        samples=Labels(
            ["first_atom", "second_atom", "cell_shift_a", "cell_shift_b",
             "cell_shift_c"],
            samples,
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )
    system.add_neighbor_list(nl_options, neighbors)
    energy = model([system], evaluation_options, check_consistency=False)["energy"]
    energy = energy.block().values.sum()
    return -torch.autograd.grad(energy, positions)[0]


# closed rectangular loop spanned by two random displacement fields, walked
# with midpoint-rule work integration
generator = torch.Generator().manual_seed(7)
positions = payload["positions"]
amplitude = 0.02
u = torch.randn(positions.shape, generator=generator, dtype=positions.dtype)
v = torch.randn(positions.shape, generator=generator, dtype=positions.dtype)
u = u * amplitude
v = v * amplitude

n_per_leg = 8
work = 0.0
current = positions.clone()
for delta in (u, v, -u, -v):
    step = delta / n_per_leg
    for _ in range(n_per_leg):
        work += float((forces(current + 0.5 * step) * step).sum())
        current = current + step

print(f"LOOP_WORK_EV {work:.6e}")
"""


def test_exported_forces_are_energy_gradient(tmp_path):
    """Tests that an exported model's forces are conservative at deployment.

    The exported TorchScript model must be differentiable in a process that
    never imports ``sphericart.torch`` (i-PI, LAMMPS, plain metatomic/ASE
    deployments), where the sphericart operators have no Python-registered
    autograd formula. If the angular gradients are silently dropped there, the
    forces acquire a smooth non-gradient component, and the work around a
    closed loop in configuration space -- exactly zero for the gradient of any
    single-valued energy -- becomes finite (measured: ~5e-4 eV for this system
    with the unfixed model, against ~4e-7 eV of midpoint-integrator residual
    with the fixed one).
    """
    torch.manual_seed(0)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        dataset_info = DatasetInfo(
            length_unit="Angstrom",
            atomic_types=[8],
            targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
        )
        model = SoapBpnn(MODEL_HYPERS, dataset_info).to(torch.float64)
        model.eval()
        exported = model.export()
        exported.save(
            str(tmp_path / "model.pt"), collect_extensions=str(tmp_path / "extensions")
        )

        # base system and a fixed pair list, precomputed here and reused for the
        # whole loop in the worker
        n_atoms, box = 12, 8.0
        positions = torch.rand((n_atoms, 3), dtype=torch.float64) * box
        cell = box * torch.eye(3, dtype=torch.float64)
        types = torch.full((n_atoms,), 8, dtype=torch.int32)
        system = System(
            types=types,
            positions=positions,
            cell=cell,
            pbc=torch.tensor([True, True, True]),
        )
        nl_options = model.requested_neighbor_lists()[0]
        system = get_system_with_neighbor_lists(system, [nl_options])
        neighbors = system.get_neighbor_list(nl_options)
        torch.save(
            {
                "positions": positions,
                "cell": cell,
                "types": types,
                "nl_samples": neighbors.samples.values,
                "cutoff": nl_options.cutoff,
                "full_list": nl_options.full_list,
                "strict": nl_options.strict,
            },
            tmp_path / "payload.pt",
        )
    finally:
        torch.set_default_dtype(default_dtype)

    result = subprocess.run(
        [sys.executable, "-c", _CLOSED_LOOP_WORKER, str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    work = None
    for line in result.stdout.splitlines():
        if line.startswith("LOOP_WORK_EV"):
            work = float(line.split()[1])
    assert work is not None, result.stdout
    assert abs(work) < 1.0e-5, (
        f"non-zero closed-loop work ({work:.3e} eV): the exported model's "
        "forces are not the gradient of its energy at deployment"
    )
