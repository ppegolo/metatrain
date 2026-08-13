"""The learning-rate schedule: warmup, optional stable phase, cosine decay.

The stable phase is what decouples the schedule from the training horizon, so
these tests pin down that a run stopped inside it is still at the peak rate,
that the decay reaches the configured floor, and that stepping past the end
holds the floor instead of letting the cosine turn back upwards.
"""

import pytest
import torch

from metatrain.pet.trainer import get_scheduler


def _rates(hypers, steps_per_epoch, n_steps):
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter], lr=1.0)
    scheduler = get_scheduler(optimizer, hypers, steps_per_epoch)
    rates = []
    for _ in range(n_steps):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return rates


def test_plain_cosine_is_unchanged_by_default():
    hypers = {"num_epochs": 10, "warmup_fraction": 0.0}
    rates = _rates(hypers, steps_per_epoch=10, n_steps=100)
    assert rates[0] == 1.0
    assert rates[50] == pytest.approx(0.5, abs=1e-6)  # halfway through the cosine
    assert rates[-1] < 0.01
    assert all(
        b <= a + 1e-12 for a, b in zip(rates, rates[1:], strict=False)
    )  # monotone


def test_warmup_then_decay():
    hypers = {"num_epochs": 10, "warmup_fraction": 0.2}
    rates = _rates(hypers, steps_per_epoch=10, n_steps=100)
    assert rates[0] == 0.0
    assert rates[10] == pytest.approx(0.5, abs=1e-6)  # mid-warmup
    assert rates[20] == pytest.approx(1.0, abs=1e-6)  # peak at the end of warmup
    assert rates[-1] < 0.01


def test_stable_phase_holds_the_peak_then_decays():
    hypers = {"num_epochs": 10, "warmup_fraction": 0.1, "stable_fraction": 0.6}
    rates = _rates(hypers, steps_per_epoch=10, n_steps=100)
    # peak from the end of warmup (step 10) to the start of the decay (step 70)
    assert all(r == pytest.approx(1.0, abs=1e-9) for r in rates[10:70])
    assert rates[70] == pytest.approx(1.0, abs=1e-6)
    assert rates[85] < 0.6
    assert rates[-1] < 0.01


def test_decay_floor_is_configurable():
    hypers = {"num_epochs": 10, "warmup_fraction": 0.0, "min_lr_ratio": 0.1}
    rates = _rates(hypers, steps_per_epoch=10, n_steps=100)
    assert rates[-1] == pytest.approx(0.1, abs=1e-3)
    assert min(rates) >= 0.1 - 1e-9


def test_stepping_past_the_end_holds_the_floor():
    # Restarting with a shorter schedule than the steps already taken must not
    # let the cosine come back up.
    hypers = {"num_epochs": 5, "warmup_fraction": 0.0}
    rates = _rates(hypers, steps_per_epoch=10, n_steps=200)
    assert max(rates[50:]) < 1e-6
