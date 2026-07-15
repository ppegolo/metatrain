"""Atomic-basis ("DFT surrogate") target machinery.

Everything specific to atomic-basis targets — densify/pad/sparsify transforms,
PySCF metric matrices, RI density-loss data plumbing and the trainer hooks —
lives in this package, so architectures and users that only train MLIP-style
targets never need to look at it.
"""

from .eval_hooks import (  # noqa: F401
    DensityErrorEvalHooks,
    NullEvalHooks,
    get_eval_hooks,
)
from .helpers import (  # noqa: F401
    DensifyStatics,
    densify_atomic_basis_dataset_info,
    densify_atomic_basis_target,
    get_prepare_atomic_basis_targets_transform,
    pad_samples_atomic_basis_target,
    prepare_atomic_basis_targets,
    sparsify_atomic_basis_target,
)
from .trainer_hooks import (  # noqa: F401
    AtomicBasisTrainerHooks,
    NullTrainerHooks,
    get_trainer_hooks,
)
