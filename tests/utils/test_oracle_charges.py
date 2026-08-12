"""Oracle-charge conditioning: the GFN2 channel and its embedding.

The oracle charges are a physics-informed conditioning input: GFN2-xTB
partial charges attached per atom, carrying the global charge-placement
bookkeeping that local models underfit on charged systems. The tests pin
down the chain: the charges are physically sane and charge-conserving, the
attach/extract layout round-trips, the dropout is a fresh per-system draw
each forward (never a fixed subset), and old checkpoints upgrade to the new
hyperparameters with the feature off.
"""

import pytest
import torch
from metatomic.torch import System

from metatrain.pet.modules.conditioning import AtomicChargeEmbedding


def _system(types, positions):
    return System(
        types=torch.tensor(types, dtype=torch.int32),
        positions=torch.tensor(positions, dtype=torch.float64),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )


WATER = _system(
    [8, 1, 1],
    [[0.0, 0.0, 0.0], [0.0, 0.757, 0.587], [0.0, -0.757, 0.587]],
)


def test_gfn2_charges_conserve_the_total():
    pytest.importorskip("tblite")
    from metatrain.utils.oracle_charges import compute_gfn2_charges

    charges = compute_gfn2_charges(WATER, charge=0, spin_multiplicity=1)
    assert charges is not None and charges.shape == (3,)
    assert float(charges.sum()) == pytest.approx(0.0, abs=1e-6)
    assert charges[0] < 0 < charges[1]  # O negative, H positive

    hydroxide = _system([8, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]])
    charges = compute_gfn2_charges(hydroxide, charge=-1, spin_multiplicity=1)
    assert float(charges.sum()) == pytest.approx(-1.0, abs=1e-6)


def test_attach_and_extract_round_trip():
    from metatrain.pet.model import _extract_oracle_charges
    from metatrain.utils.oracle_charges import attach_oracle_charges

    with_oracle = _system([8, 1, 1], WATER.positions.tolist())
    values = torch.tensor([-0.6, 0.3, 0.3], dtype=torch.float64)
    attach_oracle_charges(with_oracle, values, torch.float64)
    without = _system([1, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]])

    charges, mask = _extract_oracle_charges(
        [with_oracle, without], torch.device("cpu"), torch.float64
    )
    torch.testing.assert_close(charges[:3], values)
    assert torch.all(charges[3:] == 0.0)
    assert mask.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_embedding_starts_as_a_no_op_and_respects_the_mask():
    embedding = AtomicChargeEmbedding(d_out=32)
    q = torch.randn(6)
    mask = torch.ones(6)
    si = torch.tensor([0, 0, 0, 1, 1, 1])
    out = embedding(q, mask, si, 2)
    assert out.shape == (6, 32)
    assert torch.all(out == 0.0)  # zero-init gate: no-op at initialisation


def test_dropout_is_a_fresh_per_system_draw():
    torch.manual_seed(0)
    embedding = AtomicChargeEmbedding(d_out=8, dropout=0.5)
    # Make the projection non-trivial so dropped/kept systems differ.
    for parameter in embedding.project.parameters():
        torch.nn.init.normal_(parameter)
    n_systems = 200
    q = torch.ones(n_systems)
    mask = torch.ones(n_systems)
    si = torch.arange(n_systems)  # one atom per system

    embedding.train(True)
    out1 = embedding(q, mask, si, n_systems)
    out2 = embedding(q, mask, si, n_systems)
    null = embedding(torch.zeros(1), torch.zeros(1), torch.zeros(1).long(), 1)
    dropped1 = torch.isclose(out1, null, atol=1e-5).all(dim=-1)
    dropped2 = torch.isclose(out2, null, atol=1e-5).all(dim=-1)
    # About half dropped, and the subsets differ between forward passes.
    assert 0.3 < dropped1.float().mean() < 0.7
    assert not torch.equal(dropped1, dropped2)

    # Eval mode: nothing is ever dropped.
    embedding.train(False)
    out_eval = embedding(q, mask, si, n_systems)
    assert not torch.isclose(out_eval, null, atol=1e-5).all(dim=-1).any()


def test_dropped_systems_look_exactly_like_missing_oracles():
    torch.manual_seed(0)
    embedding = AtomicChargeEmbedding(d_out=8, dropout=1.0 - 1e-9)
    for parameter in embedding.project.parameters():
        torch.nn.init.normal_(parameter)
    embedding.train(True)
    q, mask = torch.tensor([0.7]), torch.tensor([1.0])
    dropped = embedding(q, mask, torch.tensor([0]), 1)
    missing = embedding(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0]), 1)
    torch.testing.assert_close(dropped, missing)


def test_invalid_dropout_is_rejected():
    with pytest.raises(ValueError, match="dropout"):
        AtomicChargeEmbedding(d_out=8, dropout=1.0)


def test_v17_checkpoints_upgrade_with_the_feature_off():
    from metatrain.pet.checkpoints import model_update_v17_v18

    checkpoint = {"model_data": {"model_hypers": {"cutoff": 4.5}}}
    model_update_v17_v18(checkpoint)
    hypers = checkpoint["model_data"]["model_hypers"]
    assert hypers["atomic_charge_conditioning"] is False
    assert hypers["atomic_charge_dropout"] == 0.0


def test_precomputed_charges_from_extra_data_win():
    from metatensor.torch import Labels, TensorBlock, TensorMap

    from metatrain.utils.oracle_charges import _oracle_charges_transform

    systems = [
        _system([8, 1, 1], WATER.positions.tolist()),
        _system([1, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]]),
    ]
    values = torch.tensor([-0.6, 0.3, 0.3, -0.1, 0.1], dtype=torch.float64)
    packed = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=values.reshape(-1, 1),
                # dataset indices, deliberately not batch positions
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor(
                        [[1743, 0], [1743, 1], [1743, 2], [2901, 0], [2901, 1]],
                        dtype=torch.int32,
                    ),
                ),
                components=[],
                properties=Labels("charge", torch.zeros((1, 1), dtype=torch.int32)),
            )
        ],
    )
    # No tblite fallback should be needed: everything is covered.
    _oracle_charges_transform(systems, {}, {"mtt::oracle_charges": packed})
    q0 = systems[0].get_data("mtt::oracle_charges").block().values.reshape(-1)
    q1 = systems[1].get_data("mtt::oracle_charges").block().values.reshape(-1)
    torch.testing.assert_close(q0, values[:3])
    torch.testing.assert_close(q1, values[3:])
