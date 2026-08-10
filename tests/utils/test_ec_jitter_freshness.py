"""
The partner jitter must draw a fresh shift every step, not one per run.

The worker's torch *seed* is constant for the whole run whenever workers
persist across epochs (the trainers set ``persistent_workers``) and always in
the main process (``num_workers=0``), so a shift derived from the seed alone
would hand every system one fixed displaced placement for all of training —
the opposite of augmentation. These tests call the transform twice in a row,
as two training steps do, and require different shifts.

The cache checks here go through the real id key (``mtt::aux::system_index``),
the one :py:func:`metatrain.utils.data.byte_budget_cache.batch_system_ids`
reads; without it the cache path is never exercised at all.
"""

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

import metatrain.utils.pyscf_loss as pyscf_loss
from metatrain.utils.pyscf_loss import (
    compute_ec_machinery,
    ec_fragment_name,
    get_ec_machinery_transform,
    unpack_metric_matrices,
)


pyscf = pytest.importorskip("pyscf")

AUX_BASIS = "def2-universal-jfit"
TARGET = "mtt::density"
MOMENTS_KEY = f"{TARGET}_ec_machinery_moments"


def _hf_dimer() -> System:
    return System(
        types=torch.tensor([9, 1, 9, 1]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.92], [3.0, 0.0, 0.0], [3.0, 0.0, 0.92]],
            dtype=torch.float64,
        ),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )


def _extra(system_ids) -> dict:
    labels = [0, 0, 1, 1]
    samples = torch.zeros((len(labels), 2), dtype=torch.int32)
    samples[:, 1] = torch.arange(len(labels), dtype=torch.int32)
    return {
        ec_fragment_name(TARGET): TensorMap(
            Labels.single(),
            [
                TensorBlock(
                    values=torch.tensor([[float(v)] for v in labels]).to(torch.float64),
                    samples=Labels(["system", "atom"], samples),
                    components=[],
                    properties=Labels("fragment", torch.zeros((1, 1)).to(torch.int32)),
                )
            ],
        ),
        "mtt::aux::system_index": TensorMap(
            Labels.single(),
            [
                TensorBlock(
                    values=torch.tensor([[float(i)] for i in system_ids]),
                    samples=Labels(
                        "system",
                        torch.arange(len(system_ids), dtype=torch.int32).reshape(-1, 1),
                    ),
                    components=[],
                    properties=Labels(
                        "system_index", torch.zeros((1, 1)).to(torch.int32)
                    ),
                )
            ],
        ),
    }


def _moments(transform) -> torch.Tensor:
    _, _, out = transform([_hf_dimer()], {}, _extra([7]))
    return out[MOMENTS_KEY][0].values


def test_successive_steps_draw_different_shifts():
    """Two steps on the same system must not share one placement.

    This is exactly what a persistent worker (or ``num_workers=0``) does
    across epochs: same process, same torch seed, transform called again.
    """
    torch.manual_seed(42)
    transform = get_ec_machinery_transform({TARGET: AUX_BASIS}, 0.3)
    first = _moments(transform)
    second = _moments(transform)
    assert not torch.allclose(first, second, atol=1e-8)


def test_shifts_are_reproducible_from_the_global_seed():
    torch.manual_seed(4)
    transform = get_ec_machinery_transform({TARGET: AUX_BASIS}, 0.3)
    first = _moments(transform)
    torch.manual_seed(4)
    second = _moments(transform)
    assert torch.allclose(first, second, atol=0.0)


def test_jittered_batches_write_nothing_to_the_cache():
    """Through the real id key: a jittered batch must leave the cache empty,
    and the unaugmented path must cache and re-read the true machinery."""
    cache = pyscf_loss._metric_matrix_cache()
    cache._data.clear()
    cache._bytes = 0

    jittered = get_ec_machinery_transform({TARGET: AUX_BASIS}, 0.5)
    jittered([_hf_dimer()], {}, _extra([7]))
    assert len(cache._data) == 0

    plain = get_ec_machinery_transform({TARGET: AUX_BASIS}, 0.0)
    plain([_hf_dimer()], {}, _extra([7]))
    assert len(cache._data) == 3  # moments, vectors, constants

    jittered([_hf_dimer()], {}, _extra([7]))
    _, _, out = plain([_hf_dimer()], {}, _extra([7]))
    direct = compute_ec_machinery(_hf_dimer(), AUX_BASIS, [0, 0, 1, 1])
    (cached,) = unpack_metric_matrices(out[MOMENTS_KEY])
    assert torch.allclose(cached, direct[0], atol=0.0)
