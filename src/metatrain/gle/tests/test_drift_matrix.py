"""Tests for the GLE-specific part of the architecture: the drift matrix.

The GLE method only works if the matrix ``A`` that the model parametrizes is a
*stable* Ornstein-Uhlenbeck drift: its symmetric part must be positive
semi-definite, otherwise the propagator ``exp(-A t)`` grows and the dynamics
heat up instead of thermostatting. ``make_A`` is what guarantees that for any
network output, so the tests below check the guarantee rather than the formula.
"""

import copy

import pytest
import torch
from metatomic.torch import ModelOutput, System

from metatrain.gle.model import GLE
from metatrain.gle.trainer import make_A
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)


N_AUX = 2
N_GLE = 3 + N_AUX


@pytest.fixture
def model_hypers():
    hypers = copy.deepcopy(get_default_hypers("gle")["model"])
    hypers["d_pet"] = 2
    hypers["d_head"] = 2
    hypers["d_node"] = 2
    hypers["d_feedforward"] = 2
    hypers["num_heads"] = 1
    hypers["num_attention_layers"] = 1
    hypers["num_gnn_layers"] = 1
    hypers["num_auxiliary_variables"] = N_AUX
    return hypers


@pytest.fixture
def dataset_info():
    return DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 8],
        targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
    )


@pytest.fixture
def system():
    return System(
        types=torch.tensor([8, 1, 1, 8]),
        positions=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )


def _symmetric_eigenvalues(A: torch.Tensor) -> torch.Tensor:
    return torch.linalg.eigvalsh(0.5 * (A + A.transpose(-1, -2)))


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_make_A_is_always_stable(scale: float) -> None:
    """Whatever the network emits, the drift must be a stable OU drift.

    An unstable drift makes ``exp(-A t)`` blow up, so a model that trained past
    this point would heat the system rather than thermostat it. The tolerance is
    relative to the size of the matrix, since the eigensolver's round-off is.
    """
    torch.manual_seed(0)
    theta = scale * torch.randn(64, N_GLE**2, dtype=torch.float64)

    A = make_A(theta, N_GLE)

    assert A.shape == (64, N_GLE, N_GLE)
    tolerance = 1e-12 * A.abs().amax(dim=(-2, -1))
    assert torch.all(_symmetric_eigenvalues(A) > -tolerance[:, None])


def test_make_A_rejects_wrong_size() -> None:
    """A mismatch between the readout width and ``3 + n_aux`` must be loud: read
    silently, it would reinterpret the matrix entries and give a wrong drift."""
    with pytest.raises(ValueError, match="Expected theta.shape"):
        make_A(torch.zeros(4, N_GLE**2 + 1, dtype=torch.float64), N_GLE)


def test_model_emits_a_stable_drift(model_hypers, dataset_info, system) -> None:
    """The ``mtt::A`` output of a freshly initialized model must already map to a
    stable drift, since training starts from it."""
    model = GLE(model_hypers, dataset_info).to(torch.float64)
    system = get_system_with_neighbor_lists(system, get_requested_neighbor_lists(model))

    outputs = model([system], {"mtt::A": ModelOutput(sample_kind="atom")})

    theta = outputs["mtt::A"].block().values
    assert theta.shape == (len(system), N_GLE**2)
    assert torch.all(_symmetric_eigenvalues(make_A(theta, N_GLE)) > 0.0)


def test_mtt_A_is_only_returned_when_requested(
    model_hypers, dataset_info, system
) -> None:
    """``mtt::A`` is not a dataset target, so returning it unconditionally would
    make the generic evaluation loop trip over an output it has no target for."""
    model = GLE(model_hypers, dataset_info).to(torch.float64)
    system = get_system_with_neighbor_lists(system, get_requested_neighbor_lists(model))

    outputs = model([system], {"energy": ModelOutput(sample_kind="system")})

    assert "mtt::A" not in outputs


def test_theta_baseline_is_added_to_the_output(
    model_hypers, dataset_info, system
) -> None:
    """Delta-learning: the frozen per-type baseline must be added inside the model,
    so that the training loss and the exported model see the same ``theta_total``.
    Adding it only in the loss would silently deploy a different drift."""
    hypers = copy.deepcopy(model_hypers)
    hypers["zero_init_readout"] = True
    model = GLE(hypers, dataset_info).to(torch.float64)
    system = get_system_with_neighbor_lists(system, get_requested_neighbor_lists(model))

    # with a zero-initialized readout and a zero baseline, theta is exactly zero
    zero = model([system], {"mtt::A": ModelOutput(sample_kind="atom")})
    assert torch.allclose(
        zero["mtt::A"].block().values, torch.zeros(1, dtype=torch.float64)
    )

    baseline = torch.zeros_like(model.theta_baseline)
    baseline[8] = 1.0  # only the oxygen-tagged beads
    model.theta_baseline.copy_(baseline)

    shifted = model([system], {"mtt::A": ModelOutput(sample_kind="atom")})

    values = shifted["mtt::A"].block().values
    is_oxygen = system.types == 8
    assert torch.allclose(values[is_oxygen], torch.ones(1, dtype=torch.float64))
    assert torch.allclose(values[~is_oxygen], torch.zeros(1, dtype=torch.float64))
