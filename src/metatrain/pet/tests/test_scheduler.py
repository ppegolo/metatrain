import math

import pytest
import torch

from metatrain.pet.trainer import get_scheduler


def _learning_rates(hold_fraction):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    hypers = {"num_epochs": 10, "warmup_fraction": 0.1, "hold_fraction": hold_fraction}
    scheduler = get_scheduler(optimizer, hypers, steps_per_epoch=10)
    rates = []
    for _ in range(100):
        rates.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    return rates


@pytest.mark.parametrize("hold_fraction", [0.0, 0.5])
def test_hold_fraction(hold_fraction):
    rates = _learning_rates(hold_fraction)
    decay_start = 10 + int(100 * hold_fraction)
    expected = (
        [step / 10 for step in range(10)]
        + [1.0] * (decay_start - 10)
        + [
            0.5 * (1 + math.cos(math.pi * (step - decay_start) / (100 - decay_start)))
            for step in range(decay_start, 100)
        ]
    )
    assert rates == pytest.approx(expected)
