"""Tests for the fixed coarse-grained priors ``SoftCore`` and ``HarmonicBonded``.

Both priors are *fixed* baselines: they never fit anything, so the property worth
testing is not an accuracy but the exactness of their analytic position gradient.
``remove_additive`` prefers that gradient over ``torch.autograd`` (so the additive
force subtraction can run inside fork-based DataLoader workers), which means a
wrong analytic gradient would silently corrupt every force-matching target rather
than raise.
"""

import pytest
import torch
from metatomic.torch import ModelOutput, System

from metatrain.utils.additive import HarmonicBonded, SoftCore
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)


@pytest.fixture
def dataset_info():
    return DatasetInfo(
        length_unit="angstrom",
        atomic_types=[8],
        targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
    )


def make_system(positions: torch.Tensor) -> System:
    return System(
        types=torch.full((positions.shape[0],), 8),
        positions=positions,
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )


def autograd_forces(model, positions: torch.Tensor) -> torch.Tensor:
    """Per-atom -dE/dr from autograd through the module's ``forward``."""
    positions = positions.clone().requires_grad_(True)
    system = make_system(positions)
    system = get_system_with_neighbor_lists(system, get_requested_neighbor_lists(model))
    energy = (
        model([system], {"energy": ModelOutput(unit="eV")})["energy"]
        .block()
        .values.sum()
    )
    (gradient,) = torch.autograd.grad(energy, positions)
    return gradient


def analytic_forces(model, positions: torch.Tensor) -> torch.Tensor:
    """Per-atom dE/dr from the module's own closed-form gradient."""
    system = make_system(positions)
    system = get_system_with_neighbor_lists(system, get_requested_neighbor_lists(model))
    contribution = model.analytic_contribution(
        [system], {"energy": ModelOutput(unit="eV")}
    )
    return contribution["energy"].block().gradient("positions").values.squeeze(-1)


def test_softcore_analytic_gradient_matches_autograd(dataset_info) -> None:
    model = SoftCore({}, dataset_info).to(torch.float64)
    # inside the WCA wall (r_min = 2^(1/6) * 3.159 A = 3.546 A), where the prior
    # is the only thing standing between the network and a collapsed bead pair
    torch.manual_seed(0)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [2.6, 0.0, 0.0], [1.3, 2.4, 0.0], [1.3, 0.8, 2.5]],
        dtype=torch.float64,
    )

    assert torch.allclose(
        analytic_forces(model, positions), autograd_forces(model, positions)
    )


def test_softcore_is_purely_repulsive(dataset_info) -> None:
    """The mean force already carries the PMF attraction: an attractive tail here
    would be double counted, and the prior would no longer be a pure wall."""
    model = SoftCore({}, dataset_info).to(torch.float64)

    for separation in [2.0, 3.0, 3.5, 3.6, 5.0]:
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [separation, 0.0, 0.0]], dtype=torch.float64
        )
        system = make_system(positions)
        system = get_system_with_neighbor_lists(
            system, get_requested_neighbor_lists(model)
        )
        energy = (
            model([system], {"energy": ModelOutput(unit="eV")})["energy"]
            .block()
            .values.sum()
        )
        assert energy >= 0.0
        if separation > 2.0 ** (1.0 / 6.0) * 3.16:  # beyond the WCA truncation
            assert energy == 0.0


def test_softcore_excludes_bonded_pairs(dataset_info) -> None:
    """Beads of the same CG molecule sit well inside the wall; leaving them
    unmasked would add ~1e3 eV per bonded pair to the baseline."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
    system = make_system(positions)

    masked = SoftCore({"bonds": [[0, 1]]}, dataset_info).to(torch.float64)
    unmasked = SoftCore({}, dataset_info).to(torch.float64)

    def energy_of(model):
        with_nl = get_system_with_neighbor_lists(
            system, get_requested_neighbor_lists(model)
        )
        return (
            model([with_nl], {"energy": ModelOutput(unit="eV")})["energy"]
            .block()
            .values.sum()
        )

    assert energy_of(masked) == 0.0
    assert energy_of(unmasked) > 0.0


def test_harmonic_bonded_analytic_gradient_matches_autograd(dataset_info) -> None:
    model = HarmonicBonded(
        {
            "bonds": [[0, 1], [1, 2]],
            "bond_k": [2.0, 3.0],
            "bond_r0": [1.4, 1.6],
            "angles": [[0, 1, 2]],
            "angle_k": [1.5],
            "angle_theta0": [1.9],
        },
        dataset_info,
    ).to(torch.float64)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.1, 1.3, 0.2]], dtype=torch.float64
    )

    assert torch.allclose(
        analytic_forces(model, positions), autograd_forces(model, positions)
    )


def test_harmonic_bonded_is_zero_at_equilibrium(dataset_info) -> None:
    """Parameters come from a Boltzmann inversion of the reference, so the prior
    must vanish (energy and force) at the values it was inverted from."""
    model = HarmonicBonded(
        {"bonds": [[0, 1]], "bond_k": [2.0], "bond_r0": [1.5]}, dataset_info
    ).to(torch.float64)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)

    system = get_system_with_neighbor_lists(
        make_system(positions), get_requested_neighbor_lists(model)
    )
    energy = (
        model([system], {"energy": ModelOutput(unit="eV")})["energy"]
        .block()
        .values.sum()
    )

    assert torch.allclose(energy, torch.zeros((), dtype=torch.float64))
    assert torch.allclose(
        analytic_forces(model, positions), torch.zeros((), dtype=torch.float64)
    )


def test_harmonic_bonded_rejects_degrees(dataset_info) -> None:
    """A theta0 in degrees is the likely mistake, and it silently produces a
    completely different (and much stiffer) prior, so it must be rejected."""
    with pytest.raises(ValueError, match="radian"):
        HarmonicBonded(
            {
                "bonds": [[0, 1]],
                "bond_k": [1.0],
                "bond_r0": [1.5],
                "angles": [[0, 1, 2]],
                "angle_k": [1.0],
                "angle_theta0": [109.5],
            },
            dataset_info,
        )
