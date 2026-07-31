"""Every architecture must reach every additive prior, and in the same order.

Two things are protected here, and only one of them is about features.

**Reachability.** ``SoftCore`` and ``HarmonicBonded`` were wired into PET alone.
Because the hypers schema forbids unknown keys, ``soft_core: true`` was not ignored
by the other architectures -- it was REJECTED, so they could not express the prior
at all. The only prior they could reach was ZBL, a nuclear-charge repulsion that is
meaningless on coarse-grained beads that are not nuclei. For a CG potential of mean
force that is not a missing convenience: the WCA wall is what keeps the short range
stable where the training set has no coverage, and without it the network
extrapolates into an attractive sink and the beads fuse.

**Order.** ``additive_models`` is a ``ModuleList``, so a module's POSITION is its
state-dict key. Inserting a prior anywhere but the end renumbers everything after
it, and an existing checkpoint then loads one prior's parameters into a different
prior. Nothing raises, because the shapes have nothing to do with the meaning. So
the order is asserted, not commented.

The prior is checked by its EFFECT -- a steep repulsive wall inside sigma and
nothing outside it -- rather than by ``isinstance``. A prior that is constructed but
never reaches the energy would pass a type check and change no physics at all.
"""

import copy

import pytest
import torch
from metatomic.torch import ModelOutput, System

from metatrain.utils.additive import (
    ZBL,
    HarmonicBonded,
    SoftCore,
    build_additive_priors,
)
from metatrain.utils.architectures import (
    get_default_hypers,
    get_hypers_classes,
    import_architecture,
)
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
from metatrain.utils.pydantic import validate_architecture_options


BEAD_TYPE = 8
SIGMA = 3.16
EPSILON = 0.008

# SPACE is registered under its experimental namespace; naming it "space" raises.
ARCHITECTURES = ["pet", "soap_bpnn", "experimental.space"]


@pytest.fixture
def dataset_info():
    return DatasetInfo(
        length_unit="angstrom",
        atomic_types=[BEAD_TYPE],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )


def _two_beads(separation: float) -> System:
    return System(
        types=torch.tensor([BEAD_TYPE, BEAD_TYPE]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [separation, 0.0, 0.0]], dtype=torch.float64
        ),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )


def _softcore_hypers(**overrides):
    hypers = {
        "soft_core": True,
        "soft_core_sigma_by_type": {BEAD_TYPE: SIGMA},
        "soft_core_epsilon_by_type": {BEAD_TYPE: EPSILON},
    }
    hypers.update(overrides)
    return hypers


def _bonded_hypers():
    return {
        "harmonic_bonded": True,
        "harmonic_bonded_bonds": [[0, 1]],
        "harmonic_bonded_bond_k": [1.0],
        "harmonic_bonded_bond_r0": [3.0],
    }


def test_no_priors_by_default(dataset_info):
    """Inheriting the shared schema must not switch anything on."""
    assert build_additive_priors({}, dataset_info, [BEAD_TYPE]) == []


def test_priors_are_built_in_the_canonical_order(dataset_info):
    """Composition lives at the call site; then ZBL, SoftCore, HarmonicBonded."""
    priors = build_additive_priors(
        _softcore_hypers(zbl=True, **_bonded_hypers()),
        dataset_info,
        [BEAD_TYPE],
    )
    assert [type(p) for p in priors] == [ZBL, SoftCore, HarmonicBonded]


def test_harmonic_bonded_stays_after_softcore(dataset_info):
    """The relative order must not depend on which priors happen to be enabled.

    A checkpoint written with both must find the same module at the same index.
    """
    both = build_additive_priors(
        _softcore_hypers(**_bonded_hypers()), dataset_info, [BEAD_TYPE]
    )
    assert [type(p) for p in both] == [SoftCore, HarmonicBonded]


def test_missing_hypers_do_not_raise(dataset_info):
    """A checkpoint from before a prior's hyper existed has no such key.

    Bracket access turns that into a KeyError at restart -- a failure this has had.
    """
    assert build_additive_priors({"soft_core": True}, dataset_info, [BEAD_TYPE]) != []


def _softcore_energy(prior, separation: float) -> float:
    system = _two_beads(separation)
    options = prior.requested_neighbor_lists()[0]
    system = get_system_with_neighbor_lists(system, [options])
    out = prior(
        [system],
        {"energy": ModelOutput(quantity="energy", unit="eV", per_atom=False)},
        None,
    )
    return float(out["energy"].block().values.item())


def test_softcore_actually_produces_a_wall(dataset_info):
    """A steep repulsion inside sigma, and exactly nothing outside it.

    Checked as an energy rather than by ``isinstance``: a prior that is constructed
    but never reaches the output would satisfy every structural check in this file
    and would change no physics at all.
    """
    prior = build_additive_priors(_softcore_hypers(), dataset_info, [BEAD_TYPE])[0]

    inside = _softcore_energy(prior, 0.85 * SIGMA)
    at_minimum = _softcore_energy(prior, 2.0 ** (1.0 / 6.0) * SIGMA)
    outside = _softcore_energy(prior, 2.0 * SIGMA)

    assert inside > 0.0, "no repulsion inside sigma"
    assert inside > 10 * EPSILON, f"the wall is too soft to stabilise: {inside}"
    # WCA is truncated at the LJ minimum, so it is exactly zero beyond it
    assert at_minimum == pytest.approx(0.0, abs=1e-9)
    assert outside == pytest.approx(0.0, abs=1e-12)
    # and it must be monotonically repulsive on the way in
    assert _softcore_energy(prior, 0.8 * SIGMA) > inside


# --- architecture level: the builder working proves nothing about its callers ---


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_architecture_schema_accepts_the_priors(architecture):
    """An architecture that has not opted in REJECTS ``soft_core``, not ignores it.

    This is the check that fails loudly when a new architecture is added without
    inheriting the shared hypers.
    """
    hypers_classes = get_hypers_classes(architecture)
    defaults = get_default_hypers(architecture)
    options = {
        "name": architecture,
        "model": {**defaults["model"], **_softcore_hypers()},
        "training": defaults["training"],
    }
    validated = validate_architecture_options(
        copy.deepcopy(options),
        hypers_classes["model"],
        hypers_classes["trainer"],
        architecture,
    )
    assert validated["model"]["soft_core"] is True


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_architecture_builds_the_prior(architecture, dataset_info):
    """Built with ``soft_core``, every architecture must actually hold a SoftCore.

    Checked on the model rather than on ``build_additive_priors`` because the failure
    guarded against is an architecture that never calls the builder at all -- which
    is exactly the state SOAP-BPNN and SPACE were in.
    """
    module = import_architecture(architecture)
    hypers = {**get_default_hypers(architecture)["model"], **_softcore_hypers()}
    model = module.__model__(hypers, dataset_info)

    kinds = [type(m) for m in model.additive_models]
    assert SoftCore in kinds, f"{architecture} built {kinds} with soft_core on"
    # composition stays first: it is the one every architecture always has
    assert kinds.index(SoftCore) > 0
