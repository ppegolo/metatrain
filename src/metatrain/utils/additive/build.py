"""Assemble an architecture's additive-prior stack from its hypers.

Why this is one function and not three copies
---------------------------------------------
Every architecture needs the same list -- composition, then whichever physical
priors the hypers switch on -- and that list is not merely repetitive, it is
ORDER-SENSITIVE in a way a copy can get wrong silently. ``additive_models`` is a
``ModuleList``, so a module's position IS its state-dict key: appending a prior
anywhere but the end renumbers every prior after it, and an existing checkpoint then
loads one prior's parameters into another. Nothing raises, because the shapes are
unrelated to the meaning.

So the order below is fixed and load-bearing, and it is asserted by
``test_additive_priors.py`` rather than left to a comment:

    0. composition    (always, built at the call site)
    1. ZBL            (if ``zbl``)
    2. SoftCore       (if ``soft_core``)
    3. HarmonicBonded (if ``harmonic_bonded``)

Hypers are read with ``.get``, never ``[...]``. A checkpoint written before a
prior's hyper existed simply has no such key, and bracket access turns that into a
``KeyError`` at restart -- a real failure this code has already had once.
"""

from typing import Any, Dict, List

import torch

from metatrain.utils.data import DatasetInfo

from .harmonic_bonded import HarmonicBonded
from .softcore import SoftCore
from .zbl import ZBL


def _targets_for(prior, dataset_info: DatasetInfo) -> Dict[str, Any]:
    """The subset of targets a given prior knows how to contribute to."""
    return {
        name: info
        for name, info in dataset_info.targets.items()
        if prior.is_valid_target(name, info)
    }


def _dataset_info_for(prior, dataset_info: DatasetInfo, atomic_types) -> DatasetInfo:
    return DatasetInfo(
        length_unit=dataset_info.length_unit,
        atomic_types=atomic_types,
        targets=_targets_for(prior, dataset_info),
    )


def build_additive_priors(
    hypers: Dict[str, Any],
    dataset_info: DatasetInfo,
    atomic_types: List[int],
) -> List[torch.nn.Module]:
    """The physical priors an architecture's hypers ask for, in the canonical order.

    The composition model is NOT included: it is built differently by each architecture
    (and is always present), so it stays at the call site and this list is appended to
    it.

    :param hypers: the architecture's model hypers. Only the keys declared by
        :class:`metatrain.utils.additive.hypers.AdditivePriorHypers` are read.
    :param dataset_info: the TRAINING dataset info, used to select which targets each
        prior applies to.
    :param atomic_types: the model's atomic types.
    """
    priors: List[torch.nn.Module] = []

    if hypers.get("zbl", False):
        priors.append(
            ZBL({}, dataset_info=_dataset_info_for(ZBL, dataset_info, atomic_types))
        )

    # SoftCore: a WCA excluded-volume wall the network learns a correction on top of.
    # For
    # a coarse-grained PMF this is what keeps the short range stable -- a raw
    # force-matched
    # PMF extrapolates into an attractive sink where the training set has no coverage,
    # and
    # the beads fuse.
    if hypers.get("soft_core", False):
        priors.append(
            SoftCore(
                {
                    "sigma_by_type": hypers.get("soft_core_sigma_by_type", {}),
                    "epsilon_by_type": hypers.get("soft_core_epsilon_by_type", {}),
                    "molecule_blocks": hypers.get("soft_core_molecule_blocks", []),
                    "bonds": hypers.get("soft_core_bonds", []) or None,
                    "exclusion_depth": hypers.get("soft_core_exclusion_depth", 2),
                },
                dataset_info=_dataset_info_for(SoftCore, dataset_info, atomic_types),
            )
        )

    # Appended AFTER SoftCore so the `additive_models.<i>` key of every existing
    # checkpoint
    # keeps pointing at the same module. See the module docstring.
    if hypers.get("harmonic_bonded", False):
        priors.append(
            HarmonicBonded(
                {
                    "bonds": hypers.get("harmonic_bonded_bonds", []),
                    "bond_k": hypers.get("harmonic_bonded_bond_k", []),
                    "bond_r0": hypers.get("harmonic_bonded_bond_r0", []),
                    "angles": hypers.get("harmonic_bonded_angles", []),
                    "angle_k": hypers.get("harmonic_bonded_angle_k", []),
                    "angle_theta0": hypers.get("harmonic_bonded_angle_theta0", []),
                },
                dataset_info=_dataset_info_for(HarmonicBonded, dataset_info,
                atomic_types),
            )
        )

    return priors
