"""The per-group electron-count penalty.

``charge_weight`` sees the total electron count and nothing else, so charge
moved from one part of a system to another is invisible to it: the two errors
cancel in the sum. That error is not harmless. Across a binding interface it
shifts each partner's potential by roughly ``dq/r``, and in a zwitterion it is
the difference between the neutral and the charge-separated form.

The tests build the factor by hand where they can, and drive the loss with
hand-made matrices so they run without ``pyscf``. The one test that needs a
real auxiliary basis is skipped when it is absent.
"""

import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.utils.loss import DensityMSELossViaC
from metatrain.utils.pyscf_loss import (
    charge_group_name,
    group_charge_factor_name,
    make_metric_spec,
    metric_matrix_name,
    pack_metric_matrices,
    parse_group_charge_weight,
    parse_metric_spec,
)


TARGET = "mtt::ri"


def test_the_spec_carries_the_weight_and_stays_backwards_compatible():
    # Absent, the spec is byte-identical to what it was before this term
    # existed: it is the extra_data key and the cache key, so a change would
    # silently invalidate configs and caches.
    assert make_metric_spec("coulomb") == "coulomb"
    assert make_metric_spec("coulomb", charge_weight=1.0) == (
        "coulomb|omega=0|eps=0.01|q=1"
    )
    assert parse_group_charge_weight("coulomb") == 0.0
    assert (
        parse_group_charge_weight(make_metric_spec("coulomb", charge_weight=1.0)) == 0
    )

    spec = make_metric_spec("coulomb", group_charge_weight=2.5)
    assert spec.endswith("|gq=2.5")
    assert parse_group_charge_weight(spec) == 2.5
    # The documented 8-tuple of parse_metric_spec is unchanged by the new token.
    assert parse_metric_spec(spec) == ("coulomb", 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 1.4)

    with pytest.raises(ValueError, match="group_charge_weight must be >= 0"):
        make_metric_spec("coulomb", group_charge_weight=-1.0)


def _target_pair(values, reference):
    """A one-system, one-block coefficient target and prediction."""
    maps = []
    for data in (values, reference):
        maps.append(
            TensorMap(
                Labels(
                    ["o3_lambda", "o3_sigma", "atom_type"],
                    torch.tensor([[0, 1, 1]], dtype=torch.int32),
                ),
                [
                    TensorBlock(
                        values=torch.tensor(data, dtype=torch.float64).reshape(
                            -1, 1, 1
                        ),
                        samples=Labels(
                            ["system", "atom"],
                            torch.tensor(
                                [[0, i] for i in range(len(data))], dtype=torch.int32
                            ),
                        ),
                        components=[
                            Labels("o3_mu", torch.zeros((1, 1), dtype=torch.int32))
                        ],
                        properties=Labels(
                            "properties", torch.zeros((1, 1), dtype=torch.int32)
                        ),
                    )
                ],
            )
        )
    return maps


def test_the_penalty_sees_charge_moved_between_groups():
    """Two atoms, one electron moved from the first to the second.

    The total is untouched, so a whole-system charge penalty reads zero while
    the per-group one reads the transfer twice -- once for each group.
    """
    predicted, reference = _target_pair([1.5, 0.5], [1.0, 1.0])
    # Unit charge functional on both basis functions, zero metric, so only the
    # penalty contributes and the expected value is exact.
    factor = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    spec = make_metric_spec("coulomb", group_charge_weight=1.0)
    extra = {
        metric_matrix_name(TARGET, spec): pack_metric_matrices(
            [torch.zeros((2, 2), dtype=torch.float64)]
        ),
        group_charge_factor_name(TARGET, spec): pack_metric_matrices([factor]),
    }
    loss = DensityMSELossViaC(
        TARGET,
        None,
        1.0,
        "sum",
        metric="coulomb",
        aux_basis="def2-universal-jfit",
        group_charge_weight=1.0,
    )
    # dc = (+0.5, -0.5): each group is off by half an electron.
    assert float(loss.compute({TARGET: predicted}, {TARGET: reference}, extra)) == (
        pytest.approx(0.5)
    )

    # The whole-system functional, one row summing both groups, reads zero:
    # this is exactly the blind spot the term exists to cover.
    whole = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    extra[group_charge_factor_name(TARGET, spec)] = pack_metric_matrices([whole])
    assert float(loss.compute({TARGET: predicted}, {TARGET: reference}, extra)) == (
        pytest.approx(0.0)
    )


def test_a_missing_factor_is_reported_clearly():
    predicted, reference = _target_pair([1.5, 0.5], [1.0, 1.0])
    spec = make_metric_spec("coulomb", group_charge_weight=1.0)
    loss = DensityMSELossViaC(
        TARGET,
        None,
        1.0,
        "sum",
        metric="coulomb",
        aux_basis="def2-universal-jfit",
        group_charge_weight=1.0,
    )
    extra = {
        metric_matrix_name(TARGET, spec): pack_metric_matrices(
            [torch.zeros((2, 2), dtype=torch.float64)]
        )
    }
    with pytest.raises(RuntimeError, match="group_charge_factor"):
        loss.compute({TARGET: predicted}, {TARGET: reference}, extra)


def test_the_hooks_ask_for_the_same_spec_the_loss_reads():
    """A divergence here would lose the factor without any error."""
    from metatrain.utils.density_hooks import _aux_bases_by_metric

    spec = {
        "type": "density_mse_via_c",
        "aux_basis": "def2-universal-jfit",
        "metric": "coulomb",
        "group_charge_weight": 3.0,
    }
    (built,) = _aux_bases_by_metric({TARGET: spec}).keys()
    loss = DensityMSELossViaC(
        TARGET, None, 1.0, "mean", **{k: v for k, v in spec.items() if k != "type"}
    )
    assert built == loss.metric
    assert parse_group_charge_weight(built) == 3.0


def test_the_factor_reproduces_the_masked_charge_vectors():
    """Rows of ``F_q`` must be the charge functional masked to each group."""
    pytest.importorskip("pyscf")
    from metatomic.torch import System

    from metatrain.utils.pyscf_loss import (
        compute_charge_vector,
        compute_group_charge_factor,
    )

    system = System(
        types=torch.tensor([8, 1, 1, 8], dtype=torch.int32),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )
    basis = "def2-universal-jfit"
    # Deliberately not contiguous: groups are atom labels, not atom ranges.
    groups = np.array([0, 1, 0, 1])
    factor = compute_group_charge_factor(system, basis, groups).numpy()
    charge = compute_charge_vector(system, basis).numpy()

    assert factor.shape == (2, len(charge))
    # Every basis function belongs to exactly one group, and nothing is lost.
    assert np.allclose(factor.sum(axis=0), charge)
    assert np.all(factor[0] * factor[1] == 0.0)

    # A vector of ones counts electrons; the two groups must sum to the total.
    ones = np.ones_like(charge)
    assert factor @ ones == pytest.approx(
        [charge @ (factor[0] != 0), charge @ (factor[1] != 0)]
    )
    assert (factor @ ones).sum() == pytest.approx(charge @ ones)


def test_one_label_per_atom_is_required():
    pytest.importorskip("pyscf")
    from metatomic.torch import System

    from metatrain.utils.pyscf_loss import compute_group_charge_factor

    system = System(
        types=torch.tensor([1, 1], dtype=torch.int32),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )
    with pytest.raises(ValueError, match="exactly one label per atom"):
        compute_group_charge_factor(system, "def2-universal-jfit", np.array([0]))


def test_the_groups_fall_back_to_the_ec_fragments():
    """A dataset prepared for the EC loss needs nothing added."""
    from metatrain.utils.pyscf_loss import _batch_charge_groups, ec_fragment_name

    def field(key, labels, systems):
        return {
            key: TensorMap(
                Labels.single(),
                [
                    TensorBlock(
                        values=torch.tensor(labels, dtype=torch.float64).reshape(-1, 1),
                        samples=Labels(
                            ["system", "atom"],
                            torch.tensor(
                                [
                                    [s, a]
                                    for s, count in enumerate(systems)
                                    for a in range(count)
                                ],
                                dtype=torch.int32,
                            ),
                        ),
                        components=[],
                        properties=Labels(
                            "label", torch.zeros((1, 1), dtype=torch.int32)
                        ),
                    )
                ],
            )
        }

    systems = [None, None]  # only their count is used
    sizes = [3, 2]
    fragments = field(ec_fragment_name(TARGET), [0, 0, 1, 0, 1], sizes)
    groups = _batch_charge_groups(TARGET, systems, fragments)
    assert [g.tolist() for g in groups] == [[0, 0, 1], [0, 1]]

    # The target's own field wins when both are present.
    both = dict(fragments)
    both.update(field(charge_group_name(TARGET), [0, 1, 2, 0, 0], sizes))
    groups = _batch_charge_groups(TARGET, systems, both)
    assert [g.tolist() for g in groups] == [[0, 1, 2], [0, 0]]

    with pytest.raises(RuntimeError, match="per-atom group labels"):
        _batch_charge_groups(TARGET, systems, {})
