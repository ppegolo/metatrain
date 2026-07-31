"""Tests that ``GLELoss.__init__`` actually finishes.

Why a test for a constructor
----------------------------
A refactor once inserted a method (``_make_A``) into the MIDDLE of ``__init__``, which
orphaned everything below the insertion point as unreachable code sitting after that
method's ``return``. Python raised nothing: the constructor simply stopped early. The
observable consequences were both silent in exactly the way this project keeps getting
burnt by:

* ``gamma0_reg_weight > 0`` became a NO-OP, because the pin target is built in the part
  of ``__init__`` that no longer ran and the regulariser is guarded by
  ``gamma0_target is not None``. A training run configured WITH the pin was identical to
  one without it, and the log said nothing.
* the two ``raise``\\ s that reject an inconsistent configuration (a pin with no
  baseline, a kernel weight with no kernel target) also stopped running, so the very
  configurations they exist to catch were accepted.

So these tests assert on the attributes the tail of ``__init__`` is responsible for,
rather than on any loss value: the failure mode is a constructor that returns early, and
a value-based test cannot distinguish "the regulariser is off" from "the regulariser is
on and small".
"""

import numpy as np
import pytest
import torch

from metatrain.gle.covariant import theta_size
from metatrain.gle.trainer import GLELoss


N_AUX = 2
MASS_BY_Z = {1: 12.0, 8: 16.0}


def _loss(**kwargs) -> GLELoss:
    defaults = dict(
        num_auxiliary_variables=N_AUX,
        bead_mass_by_z=MASS_BY_Z,
        temperature=300.0,
        jitter=1e-10,
    )
    defaults.update(kwargs)
    return GLELoss(**defaults)


def _baseline(covariant: bool, n_types: int = 9) -> torch.Tensor:
    """A per-type ``theta`` baseline, indexed by atomic-number tag."""
    width = theta_size(N_AUX) if covariant else (3 + N_AUX) ** 2
    generator = torch.Generator().manual_seed(0)
    return torch.randn(n_types, width, dtype=torch.float64, generator=generator)


def test_pin_frequency_grid_is_built():
    """``_band_of`` needs ``_pin_omegas``; nothing else sets it."""
    loss = _loss()
    assert hasattr(loss, "_pin_omegas")
    assert loss._pin_omegas.shape == (1,)
    assert loss._pin_omegas.item() == 0.0  # the default is the plain gamma0 point-pin


def test_pin_grid_follows_the_requested_frequencies():
    loss = _loss(gamma0_pin_freqs_thz=[0.0, 0.5, 1.0])
    assert loss._pin_omegas.shape == (3,)
    assert loss._pin_omegas[0].item() == 0.0
    assert loss._pin_omegas[2].item() > loss._pin_omegas[1].item() > 0.0


@pytest.mark.parametrize("covariant", [False, True])
def test_gamma0_pin_target_is_actually_built(covariant):
    """The pin is guarded by ``gamma0_target is not None``, so a target that is never
    built is a regulariser that silently does nothing."""
    baseline = _baseline(covariant)
    loss = _loss(
        gamma0_reg_weight=1.0,
        theta_baseline=baseline,
        gamma0_pin_freqs_thz=[0.0, 0.5],
        covariant=covariant,
    )
    assert loss.gamma0_target is not None
    assert loss.gamma0_target.shape == (baseline.shape[0], 2)
    assert torch.isfinite(loss.gamma0_target).all()


def test_pin_without_a_baseline_is_rejected():
    with pytest.raises(ValueError, match="requires a theta_baseline"):
        _loss(gamma0_reg_weight=1.0)


def test_kernel_weight_without_a_kernel_target_is_rejected():
    with pytest.raises(ValueError, match="requires kernel_target_file"):
        _loss(kernel_reg_weight=1.0)


def test_kernel_target_is_loaded():
    """``_kernel_loss`` reads ``_kt_ase`` / ``_kt_w`` / ``_kt_by_z``; if the tail of
    ``__init__`` does not run, the ``memory_kernel`` target dies with an
    ``AttributeError`` far from its cause."""
    kernel_target = {
        "t_ase": np.linspace(0.0, 1.0, 5),
        "w": np.full(5, 0.2),
        "by_z": {8: np.linspace(1.0, 0.0, 5)},
    }
    loss = _loss(kernel_target=kernel_target)
    assert loss._kt_ase.shape == (5,)
    assert loss._kt_w.shape == (5,)
    assert set(loss._kt_by_z) == {8}


def test_chi2_diagnostic_starts_undefined():
    """The trainer accumulates ``last_chi2_red`` every batch, so it must exist before
    the first call rather than only after one."""
    assert np.isnan(_loss().last_chi2_red)
