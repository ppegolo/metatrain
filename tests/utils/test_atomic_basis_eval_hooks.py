import math
from types import SimpleNamespace

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.utils.atomic_basis import (
    DensityErrorEvalHooks,
    NullEvalHooks,
    get_eval_hooks,
)
from metatrain.utils.atomic_basis.pyscf import (
    RaggedMetricMatrices,
    overlap_matrix_name,
)


TARGET = "mtt::ri"


def _make_system(n_atoms: int) -> System:
    return System(
        positions=torch.zeros((n_atoms, 3), dtype=torch.float64),
        types=torch.ones(n_atoms, dtype=torch.int32),
        cell=torch.zeros((3, 3), dtype=torch.float64),
        pbc=torch.zeros(3, dtype=torch.bool),
    )


def _make_l0_map(per_atom_values: list[list[float]]) -> TensorMap:
    """One l=0 block; one row (system, atom) per entry of ``per_atom_values``."""
    samples = []
    system_index = 0
    for system_values in per_atom_values:
        for atom in range(len(system_values)):
            samples.append([system_index, atom])
        system_index += 1
    flat = [v for system_values in per_atom_values for v in system_values]
    values = torch.tensor(flat, dtype=torch.float64).reshape(-1, 1, 1)
    return TensorMap(
        keys=Labels(["o3_lambda", "o3_sigma"], torch.tensor([[0, 1]])),
        blocks=[
            TensorBlock(
                values=values,
                samples=Labels(
                    ["system", "atom"], torch.tensor(samples, dtype=torch.int32)
                ),
                components=[Labels(["o3_mu"], torch.tensor([[0]]))],
                properties=Labels(["n"], torch.tensor([[0]])),
            )
        ],
    )


def test_null_hooks_are_noops():
    hooks = NullEvalHooks()
    assert hooks.collate_transforms(torch.float64) == []
    hooks.update([], {}, {}, {})
    assert hooks.finalize() == {}


def _dense_layout() -> TensorMap:
    return TensorMap(
        keys=Labels(["o3_lambda", "o3_sigma"], torch.tensor([[0, 1]])),
        blocks=[
            TensorBlock(
                values=torch.empty(0, 1, 1, dtype=torch.float64),
                samples=Labels(
                    ["system", "atom"], torch.empty((0, 2), dtype=torch.int32)
                ),
                components=[Labels(["o3_mu"], torch.tensor([[0]]))],
                properties=Labels(["n"], torch.tensor([[0]])),
            )
        ],
    )


def test_get_eval_hooks_selection():
    infos = {
        "energy": SimpleNamespace(is_atomic_basis=False),
        TARGET: SimpleNamespace(is_atomic_basis=True, layout=_dense_layout()),
    }

    assert type(get_eval_hooks(None, infos)) is NullEvalHooks
    assert type(get_eval_hooks({}, infos)) is NullEvalHooks

    hooks = get_eval_hooks({"aux_basis": "def2-svp-jkfit"}, infos)
    assert type(hooks) is DensityErrorEvalHooks
    # only the atomic-basis target gets a loss; per-target dict works too
    assert set(hooks._target_to_aux_basis) == {TARGET}
    hooks = get_eval_hooks({"aux_basis": {TARGET: "def2-svp-jkfit"}}, infos)
    assert hooks._target_to_aux_basis == {TARGET: "def2-svp-jkfit"}


def test_get_eval_hooks_rejects_non_atomic_basis_runs():
    infos = {"energy": SimpleNamespace(is_atomic_basis=False)}
    with pytest.raises(ValueError, match="none of the evaluated targets"):
        get_eval_hooks({"aux_basis": "def2-svp-jkfit"}, infos)


def test_density_error_rmse_math():
    # The reported metric must be sqrt(sum_i dc_i^T S_i dc_i / sum_i N_atoms,i)
    # over ALL evaluated systems: a per-atom-normalized root integrated squared
    # density error, comparable across datasets with different molecule sizes.
    infos = {TARGET: SimpleNamespace(is_atomic_basis=True, layout=_dense_layout())}
    hooks = DensityErrorEvalHooks({TARGET: "unused"}, infos)
    assert len(hooks.collate_transforms(torch.float64)) == 1

    # batch 1: one 1-atom system, delta = [2], S = [[3]] -> error 12
    extra_1 = {
        overlap_matrix_name(TARGET): RaggedMetricMatrices.from_matrices(
            [torch.tensor([[3.0]], dtype=torch.float64)]
        )
    }
    hooks.update(
        [_make_system(1)],
        {TARGET: _make_l0_map([[3.0]])},
        {TARGET: _make_l0_map([[1.0]])},
        extra_1,
    )

    # batch 2: one 2-atom system, delta = [1, 1], S = 2*I -> error 4
    extra_2 = {
        overlap_matrix_name(TARGET): RaggedMetricMatrices.from_matrices(
            [2.0 * torch.eye(2, dtype=torch.float64)]
        )
    }
    hooks.update(
        [_make_system(2)],
        {TARGET: _make_l0_map([[1.0, 2.0]])},
        {TARGET: _make_l0_map([[0.0, 1.0]])},
        extra_2,
    )

    metrics = hooks.finalize()
    expected = math.sqrt((12.0 + 4.0) / 3)
    assert metrics == {
        f"{TARGET} density RMSE (overlap, per atom)": pytest.approx(expected)
    }


def test_density_error_skips_batches_without_target():
    infos = {TARGET: SimpleNamespace(is_atomic_basis=True, layout=_dense_layout())}
    hooks = DensityErrorEvalHooks({TARGET: "unused"}, infos)
    hooks.update([_make_system(1)], {"energy": None}, {"energy": None}, {})
    assert hooks.finalize() == {}


def test_get_eval_hooks_metric_selection():
    infos = {TARGET: SimpleNamespace(is_atomic_basis=True, layout=_dense_layout())}
    hooks = get_eval_hooks({"aux_basis": "def2-svp-jkfit", "metric": "coulomb"}, infos)
    assert hooks._metric == "coulomb"
    assert hooks._loss_fns[TARGET].metric == "coulomb"
    # default stays overlap
    hooks = get_eval_hooks({"aux_basis": "def2-svp-jkfit"}, infos)
    assert hooks._metric == "overlap"


def test_density_error_coulomb_metric_math():
    from metatrain.utils.atomic_basis.pyscf import coulomb_matrix_name

    infos = {TARGET: SimpleNamespace(is_atomic_basis=True, layout=_dense_layout())}
    hooks = DensityErrorEvalHooks({TARGET: "unused"}, infos, metric="coulomb")

    # one 1-atom system, delta = [2], J = [[5]] -> error 20
    extra = {
        coulomb_matrix_name(TARGET): RaggedMetricMatrices.from_matrices(
            [torch.tensor([[5.0]], dtype=torch.float64)]
        )
    }
    hooks.update(
        [_make_system(1)],
        {TARGET: _make_l0_map([[3.0]])},
        {TARGET: _make_l0_map([[1.0]])},
        extra,
    )
    metrics = hooks.finalize()
    expected = math.sqrt(20.0 / 1)
    assert metrics == {
        f"{TARGET} density RMSE (coulomb, per atom)": pytest.approx(expected)
    }
