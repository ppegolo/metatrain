import importlib

import pytest
import torch
from metatomic.torch import System

from metatrain.utils.atomic_basis.pyscf import (
    _load_auxiliary_basis,
    build_auxiliary_molecule,
    compute_coulomb_matrix,
    compute_metric_matrix,
    compute_overlap_matrix,
    compute_ragged_metric_matrices,
    metric_matrix_name,
    resolve_ri_aux_basis,
)


pytest.importorskip("pyscf")


def _make_systems() -> list[System]:
    cell = torch.zeros((3, 3), dtype=torch.float64)
    pbc = torch.zeros(3, dtype=torch.bool)
    return [
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            types=torch.tensor([1], dtype=torch.int32),
            cell=cell,
            pbc=pbc,
        ),
        System(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
                dtype=torch.float64,
            ),
            types=torch.tensor([8, 1, 1], dtype=torch.int32),
            cell=cell,
            pbc=pbc,
        ),
    ]


def test_build_auxiliary_molecule_and_overlap_matrix():
    system = _make_systems()[1]

    auxmol = build_auxiliary_molecule(system, "def2-svp-jkfit")
    S = compute_overlap_matrix(system, "def2-svp-jkfit")

    assert auxmol.nao == S.shape[0] == S.shape[1]
    torch.testing.assert_close(S, S.T)
    assert torch.linalg.eigvalsh(S).min().item() > 0.0


def test_etb_basis_builds_valid_overlap_matrix():
    """ETB spec 'etb:<ref>:<ratio>' should produce a valid PD overlap matrix."""
    system = _make_systems()[1]

    S = compute_overlap_matrix(system, "etb:def2-svp:2.0")

    assert S.ndim == 2
    assert S.shape[0] == S.shape[1]
    torch.testing.assert_close(S, S.T)
    assert torch.linalg.eigvalsh(S).min().item() > 0.0


def test_etb_basis_size_differs_from_reference():
    """ETB basis should produce a different number of AOs than the reference."""
    system = _make_systems()[1]

    S_jkfit = compute_overlap_matrix(system, "def2-svp-jkfit")
    S_etb = compute_overlap_matrix(system, "etb:def2-svp:2.0")

    # ETB and jkfit are different bases; sizes should differ.
    assert S_etb.shape[0] != S_jkfit.shape[0]


def test_build_auxiliary_molecule_and_coulomb_matrix():
    system = _make_systems()[1]

    auxmol = build_auxiliary_molecule(system, "def2-svp-jkfit")
    J = compute_coulomb_matrix(system, "def2-svp-jkfit")

    assert auxmol.nao == J.shape[0] == J.shape[1]
    torch.testing.assert_close(J, J.T)
    assert torch.linalg.eigvalsh(J).min().item() > 0.0


def test_compute_ragged_metric_matrices_matches_individual_results():
    systems = _make_systems()

    for metric, compute_single in (
        ("overlap", compute_overlap_matrix),
        ("coulomb", compute_coulomb_matrix),
    ):
        ragged = compute_ragged_metric_matrices(systems, "def2-svp-jkfit", metric)
        expected = [compute_single(system, "def2-svp-jkfit") for system in systems]
        assert ragged.sizes == [matrix.shape[0] for matrix in expected]
        for expected_matrix, actual_matrix in zip(
            expected, ragged.matrices(), strict=True
        ):
            torch.testing.assert_close(actual_matrix, expected_matrix)


def test_compute_metric_matrix_dtype_cast():
    system = _make_systems()[0]
    m64 = compute_metric_matrix(system, "def2-svp-jkfit", "overlap")
    ragged32 = compute_ragged_metric_matrices(
        [system], "def2-svp-jkfit", "overlap", dtype=torch.float32
    )
    torch.testing.assert_close(
        ragged32.matrices()[0], m64.to(torch.float32), rtol=0, atol=0
    )


def test_metric_matrix_name_dispatches_correctly():
    assert metric_matrix_name("tgt", "overlap") == "tgt_overlap_matrix"
    assert metric_matrix_name("tgt", "coulomb") == "tgt_coulomb_matrix"

    with pytest.raises(ValueError, match="Unknown RI metric"):
        metric_matrix_name("tgt", "euclidean")


def test_auxiliary_basis_cache_reuses_parsed_basis(monkeypatch):
    gto, _ = (
        importlib.import_module("pyscf.gto"),
        importlib.import_module("pyscf.data.elements"),
    )
    original_load = gto.basis.load
    calls: list[tuple[str, str]] = []

    def counted_load(aux_basis: str, symbol: str):
        calls.append((aux_basis, symbol))
        return original_load(aux_basis, symbol)

    _load_auxiliary_basis.cache_clear()
    monkeypatch.setattr(gto.basis, "load", counted_load)

    system = _make_systems()[1]
    compute_overlap_matrix(system, "def2-svp-jkfit")
    compute_overlap_matrix(system, "def2-svp-jkfit")

    assert sorted(calls) == [("def2-svp-jkfit", "H"), ("def2-svp-jkfit", "O")]


def test_resolve_ri_aux_basis():
    assert resolve_ri_aux_basis("target", "basis") == "basis"
    assert resolve_ri_aux_basis("target", {"target": "basis"}) == "basis"

    with pytest.raises(ValueError, match="No RI auxiliary basis configured"):
        resolve_ri_aux_basis("target", {"other": "basis"})


def test_missing_pyscf_dependency(monkeypatch):
    original_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name.startswith("pyscf"):
            raise ModuleNotFoundError(name)
        return original_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    # Earlier tests in this file warm the lru_caches, which would bypass the
    # patched importlib entirely and make this test vacuous when pyscf is
    # installed. Clear them (and restore afterwards so later tests are
    # unaffected by cache state).
    from metatrain.utils.atomic_basis.pyscf import _import_pyscf_modules

    _import_pyscf_modules.cache_clear()
    _load_auxiliary_basis.cache_clear()
    try:
        with pytest.raises(ImportError, match="require `pyscf`"):
            compute_overlap_matrix(_make_systems()[0], "def2-svp-jkfit")
    finally:
        _import_pyscf_modules.cache_clear()
        _load_auxiliary_basis.cache_clear()


def test_metric_cache_byte_budget_bounds_memory():
    # The per-worker metric-matrix cache must stay within its byte budget on
    # datasets too large to fit (the original unbounded dict OOM-killed
    # multi-node runs on >1M-system datasets), evicting least-recently-used
    # entries instead of growing without bound.
    from metatrain.utils.atomic_basis.pyscf import _ByteBudgetCache

    t = torch.zeros(100, dtype=torch.float32)  # 400 bytes each
    cache = _ByteBudgetCache(max_bytes=1000)  # fits 2 entries

    cache.put(("aux", 0), t)
    cache.put(("aux", 1), t)
    assert cache.get(("aux", 0)) is not None
    assert cache.get(("aux", 1)) is not None

    # 3rd entry exceeds the budget -> the least recently used entry (id 0,
    # touched before id 1 by the gets above) is evicted
    cache.put(("aux", 2), t)
    assert cache.get(("aux", 0)) is None  # evicted (least recently used)
    assert cache.get(("aux", 1)) is not None
    assert cache.get(("aux", 2)) is not None

    # re-putting an existing key must not double-count its bytes
    cache.put(("aux", 1), t)
    assert cache.get(("aux", 2)) is not None

    # an entry larger than the whole budget is never cached
    big = torch.zeros(1000, dtype=torch.float32)  # 4000 bytes
    cache.put(("aux", 3), big)
    assert cache.get(("aux", 3)) is None


def test_metric_cache_budget_env_override(monkeypatch):
    from metatrain.utils.atomic_basis.pyscf import (
        DEFAULT_METRIC_CACHE_MAX_BYTES,
        _metric_cache_max_bytes,
    )

    assert _metric_cache_max_bytes() == DEFAULT_METRIC_CACHE_MAX_BYTES
    monkeypatch.setenv("METATRAIN_METRIC_CACHE_MAX_BYTES", "12345")
    assert _metric_cache_max_bytes() == 12345
