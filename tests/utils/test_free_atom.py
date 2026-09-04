"""The free-atom (promolecule) baseline for atomic-basis density targets.

The baseline replaces the composition model's least-squares fit with RI-fitted
neutral-atom densities, so the network learns the deformation density instead
of fighting the nuclear/electronic cancellation. The tests pin down the three
load-bearing properties: the per-element coefficients are neutral and pure
``l=0``; the filled composition model predicts exactly those coefficients per
atom in the target's dense layout; and the ``atomic_baseline`` spec string
routes to this path through ``train_or_load_composition_model``.
"""

import pytest
import torch
from metatomic.torch import ModelOutput, System

from metatrain.utils.additive.free_atom import (
    FREE_ATOM_PREFIX,
    is_free_atom_spec,
    parse_free_atom_spec,
)


BASIS = "def2-svp"
AUX_BASIS = "def2-universal-jfit"
TARGET = "mtt::rho"

#: def2-universal-jfit radial counts per l, verified against PySCF.
IRREPS = {
    1: [{"num": 3, "o3_lambda": 0, "o3_sigma": 1}],
    6: [
        {"num": 6, "o3_lambda": 0, "o3_sigma": 1},
        {"num": 4, "o3_lambda": 1, "o3_sigma": 1},
        {"num": 3, "o3_lambda": 2, "o3_sigma": 1},
    ],
}


def test_the_spec_string_is_recognized_and_parsed():
    assert is_free_atom_spec("free_atom:def2-svp:def2-universal-jfit")
    assert not is_free_atom_spec("free_atom")  # no fields
    assert not is_free_atom_spec("/path/to/model.ckpt")
    assert not is_free_atom_spec({"energy": 1.0})

    # Without an option the ECP is looked up under the orbital basis name.
    assert parse_free_atom_spec(f"{FREE_ATOM_PREFIX}:def2-svp:{AUX_BASIS}") == (
        "def2-svp",
        AUX_BASIS,
        "def2-svp",
    )
    # The auxiliary basis may itself contain colons (even-tempered form).
    assert parse_free_atom_spec("free_atom:def2-svp:etb:def2-svp:2.0") == (
        "def2-svp",
        "etb:def2-svp:2.0",
        "def2-svp",
    )
    assert parse_free_atom_spec("free_atom:def2-svp:etb:def2-svp:2.0|ecp=none") == (
        "def2-svp",
        "etb:def2-svp:2.0",
        None,
    )
    assert parse_free_atom_spec(f"free_atom:cc-pvdz:{AUX_BASIS}|ecp=def2-svp") == (
        "cc-pvdz",
        AUX_BASIS,
        "def2-svp",
    )
    for bad in ("free_atom:", "free_atom:def2-svp", "free_atom::x", "other:a:b"):
        with pytest.raises(ValueError, match="free-atom baseline spec"):
            parse_free_atom_spec(bad)
    for bad in ("free_atom:a:b|ecp", "free_atom:a:b|ecp=", "free_atom:a:b|foo=1"):
        with pytest.raises(ValueError, match="free-atom baseline option"):
            parse_free_atom_spec(bad)


def test_the_atomic_coefficients_are_neutral_and_l0_only():
    pytest.importorskip("pyscf")
    import numpy as np

    from metatrain.utils.additive.free_atom import free_atom_coefficients
    from metatrain.utils.pyscf_loss import compute_charge_vector

    for atomic_number, n_l0 in ((1, 3), (6, 6), (8, 6)):
        coefficients = free_atom_coefficients(atomic_number, BASIS, AUX_BASIS)
        assert coefficients.shape == (n_l0,)

        # The electron count read off through the analytic charge functional
        # of a single-atom system must be exactly Z: the l=0 slots carry the
        # coefficients and every other slot carries zero moment anyway.
        system = System(
            types=torch.tensor([atomic_number], dtype=torch.int32),
            positions=torch.zeros((1, 3), dtype=torch.float64),
            cell=torch.zeros(3, 3, dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )
        s_vector = compute_charge_vector(system, AUX_BASIS).numpy()
        moments = s_vector[s_vector != 0.0]
        assert len(moments) == n_l0
        electrons = float(moments @ coefficients.numpy())
        assert electrons == pytest.approx(atomic_number, abs=1e-10)
        assert np.all(np.isfinite(coefficients.numpy()))


def test_an_ecp_element_is_neutral_against_its_effective_nucleus():
    """Iodine under the def2 ECP holds 25 electrons, not 53.

    A reference density computed with the ECP integrates to ``Z_eff``, so a
    baseline normalised to ``Z`` would carry 28 spurious electrons per iodine
    and leave the deformation target with a -28 e monopole. The atomic solver
    must also run under the ECP: def2-svp has no functions for the iodine
    core, and an all-electron atom in it is not a valid calculation at all.
    """
    pytest.importorskip("pyscf")

    from metatrain.utils.additive.free_atom import free_atom_coefficients
    from metatrain.utils.pyscf_loss import compute_charge_vector

    coefficients = free_atom_coefficients(53, BASIS, AUX_BASIS, ecp="def2-svp")
    system = System(
        types=torch.tensor([53], dtype=torch.int32),
        positions=torch.zeros((1, 3), dtype=torch.float64),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )
    s_vector = compute_charge_vector(system, AUX_BASIS).numpy()
    moments = s_vector[s_vector != 0.0]
    assert len(moments) == len(coefficients)
    assert float(moments @ coefficients.numpy()) == pytest.approx(25.0, abs=1e-10)

    # An element the ECP does not touch is unchanged by naming it.
    with_name = free_atom_coefficients(6, BASIS, AUX_BASIS, ecp="def2-svp")
    without = free_atom_coefficients(6, BASIS, AUX_BASIS)
    assert torch.allclose(with_name, without, atol=1e-12)


def _composition_model():
    from metatrain.composition import CompositionModel
    from metatrain.utils.data import DatasetInfo
    from metatrain.utils.data.target_info import get_generic_target_info

    target_info = get_generic_target_info(
        TARGET,
        {
            "quantity": "",
            "unit": "",
            "type": {"spherical": {"irreps": IRREPS}},
            "num_subtargets": 1,
            "sample_kind": "atom",
        },
    )
    return CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 6],
            targets={TARGET: target_info},
        ),
    )


def test_the_filled_model_predicts_the_free_atom_coefficients():
    pytest.importorskip("pyscf")
    from metatrain.utils.additive.free_atom import (
        free_atom_coefficients,
        set_free_atom_composition_weights,
    )

    model = _composition_model()
    set_free_atom_composition_weights(model, BASIS, AUX_BASIS)

    system = System(
        types=torch.tensor([6, 1], dtype=torch.int32),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.09]], dtype=torch.float64
        ),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )
    model.train(True)  # dense layout, the shape the trainer subtracts in
    output = model(
        [system],
        {TARGET: ModelOutput(quantity="", unit="", sample_kind="atom")},
    )[TARGET]

    block_l0 = output.block({"o3_lambda": 0, "o3_sigma": 1})
    carbon = free_atom_coefficients(6, BASIS, AUX_BASIS)
    hydrogen = free_atom_coefficients(1, BASIS, AUX_BASIS)
    # Atom 0 is carbon: its row carries all six s coefficients. Atom 1 is
    # hydrogen: three coefficients, the padded properties exactly zero.
    torch.testing.assert_close(block_l0.values[0, 0, :], carbon)
    torch.testing.assert_close(block_l0.values[1, 0, :3], hydrogen)
    assert torch.all(block_l0.values[1, 0, 3:] == 0.0)


def test_the_baseline_routes_through_train_or_load():
    pytest.importorskip("pyscf")
    from metatrain.composition import train_or_load_composition_model
    from metatrain.utils.additive.free_atom import free_atom_coefficients

    model = _composition_model()
    train_or_load_composition_model(
        composition_model=model,
        atomic_baseline=f"free_atom:{BASIS}:{AUX_BASIS}",
        train_datasets=[],  # never touched: nothing is fitted from data
        other_additive_models=[],
        batch_size=1,
        is_distributed=False,
        checkpoint_dir="",
    )
    weights = model.model.weights[TARGET]
    block_l0 = weights.block({"o3_lambda": 0, "o3_sigma": 1})
    hydrogen = free_atom_coefficients(1, BASIS, AUX_BASIS)
    row_h = list(model.model.atomic_types).index(1)
    torch.testing.assert_close(block_l0.values[row_h, 0, :3], hydrogen)
    # l>0 blocks (carbon has them) carry strictly zero weights.
    for key, block in weights.items():
        if int(key["o3_lambda"]) > 0:
            assert torch.all(block.values == 0.0)


def test_a_scalar_target_is_rejected():
    pytest.importorskip("pyscf")
    from metatrain.composition import CompositionModel
    from metatrain.utils.additive.free_atom import (
        set_free_atom_composition_weights,
    )
    from metatrain.utils.data import DatasetInfo
    from metatrain.utils.data.target_info import get_energy_target_info

    model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1],
            targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
        ),
    )
    with pytest.raises(ValueError, match="not a per-atom spherical target"):
        set_free_atom_composition_weights(model, BASIS, AUX_BASIS)
