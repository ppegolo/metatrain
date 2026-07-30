"""
GLE
===

GLE is a PET-based architecture for learning a position-dependent drift matrix
for generalized Langevin dynamics. It keeps PET's local environment encoder and
adds a per-atom ``mtt::A`` output. Training uses the GLE momentum transition
density, so the supervised target is the future bead momentum while the learned
quantity is the matrix parametrization.

{{SECTION_INSTALLATION}}

Additional outputs
------------------

In addition to the targets defined in the dataset, the GLE architecture can also output
the following additional quantities:

- ``mtt::A``: the per-atom unconstrained parametrization ``theta`` of the GLE drift
  matrix. :func:`metatrain.gle.trainer.make_A` maps it to the drift matrix itself.
- ``feature``: the internal PET features, before the different heads for each target.
- :ref:`mtt-aux-target-last-layer-features`: The features for a given target, taken
  before the last linear layer of the corresponding head.

{{SECTION_DEFAULT_HYPERS}}

Tuning hyperparameters
----------------------

The default hyperparameters above will work well in most cases, but they may not be
optimal for your specific dataset. There is good number of parameters to tune, both for
the :ref:`model <arch-{{architecture}}_model_hypers>` and the :ref:`trainer
<arch-{{architecture}}_trainer_hypers>`. Since seeing them for the first time might be
overwhelming, here we provide a **list of the parameters that are in general the most
important** (in decreasing order of importance):

.. container:: mtt-hypers-remove-classname

  .. autoattribute:: {{model_hypers_path}}.cutoff
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.num_neighbors_adaptive
      :no-index:

  .. autoattribute:: {{trainer_hypers_path}}.learning_rate
      :no-index:

  .. autoattribute:: {{trainer_hypers_path}}.batch_size
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.d_pet
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.d_node
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.num_gnn_layers
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.num_attention_layers
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.long_range
      :no-index:
"""

from typing import Dict, List, Literal, Optional

from typing_extensions import NotRequired, TypedDict

from metatrain.composition.documentation import FixedCompositionWeights
from metatrain.pet.modules.finetuning import FinetuneHypers, NoFinetuneHypers
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.long_range import LongRangeHypers
from metatrain.utils.scaler import FixedScalerWeights


class ModelHypers(TypedDict):
    """Hyperparameters of the COVARIANT GLE architecture.

    The backbone is a nested block naming any metatrain architecture plus that
    architecture's own hyperparameters, validated against ITS schema
    (:func:`metatrain.gle.wrapper.validate_backbone_hypers`). GLE no longer re-declares
    PET's hyperparameters, so adding a backbone does not mean editing this file -- which
    was the coupling the covariant rewrite removed.

    ``mtt::A`` is emitted as a per-atom SPHERICAL target (``l = 0`` and ``l = 2``), so the
    backbone must be able to predict equivariants. PET is invariant and can only reach the
    ``l = 2`` channel through rotational augmentation; training it without is a useful
    control rather than a misconfiguration.
    """

    backbone: dict = {"name": "soap_bpnn"}
    """The backbone architecture and its own hypers, e.g.
    ``{name: soap_bpnn, cutoff: 5.0, ...}``. Validated by that architecture's schema."""

    covariant: bool = True
    """Use the rotationally covariant drift construction.

    The Mori-Zwanzig kernel is a rank-2 Cartesian tensor, so rotational invariance of the
    underlying Hamiltonian forces ``K(RQ) = R K(Q) R^T``. With this set the auxiliary
    variables are 3-VECTORS and the state is ``3 (1 + num_auxiliary_variables)``; the
    Cartesian blocks are symmetric, since Onsager reciprocity makes the friction tensor
    symmetric and a pseudo-vector cannot be built from parity-even descriptors anyway.

    This is a different state space from the scalar construction, not a reparametrisation,
    so checkpoints are not interchangeable between the two."""

    num_auxiliary_variables: int = 13
    """Number of auxiliary momentum variables per bead.

    The learned matrix has dimension ``3 + num_auxiliary_variables``. The water
    scaffold keeps the hacked PET default of 13 auxiliary variables, giving a
    16 x 16 matrix.
    """
    theta_baseline_file: Optional[str] = None
    """Path to an ``.npz`` of per-type frozen theta baselines for Delta-learning.

    Keys are ``z<Z>`` (e.g. ``z8``) mapping each bead atomic-number tag to a length
    ``(3 + num_auxiliary_variables)**2`` unconstrained vector. The deployed drift is
    ``A = make_A(theta_net(Q) + theta_base[Z])``: ``theta_base`` is fit offline to the
    per-type Volterra memory kernel (the friction "self-energy") and frozen, and the
    PET network supplies the sign-free environment-dependent correction on top. Because
    ``make_A`` maps any theta to a stable drift, the correction can raise OR lower the
    friction relative to the baseline (unlike an additive-kernel embedding, which can
    only add). The baseline is added to the raw ``mtt::A`` output inside the model, so
    training and the exported deploy model see the same total. ``None`` (default) uses a
    zero baseline, i.e. the network learns the full drift as before."""
    zero_init_readout: bool = False
    """Zero-initialize the ``mtt::A`` readout (last linear layers, node and edge).

    With a frozen ``theta_baseline_file`` this makes the model an exact Delta-learner at
    initialization: ``theta_net(Q) = 0`` everywhere, so ``A = make_A(theta_base[Z])``.
    Corrections then grow only where the loss has gradient signal; directions the loss
    does not constrain (e.g. slow transport friction, invisible to a short transition
    lag) stay at the baseline instead of retaining random-init values. Default PyTorch
    init instead starts the "correction" at a magnitude comparable to the baseline
    itself. Gradients still flow (earlier layers keep their standard init)."""


class TrainerHypers(TypedDict):
    """Hyperparameters for training GLE models."""

    distributed: NotRequired[Optional[bool]] = None
    """Whether to use distributed training. When not set, distributed training
    is enabled automatically when running under more than one SLURM task.
    Setting this option explicitly is deprecated."""
    distributed_port: int = 39591
    """Port for distributed communication among processes"""
    batch_size: int = 16
    """The number of samples to use in each batch of training. This
    hyperparameter controls the tradeoff between training speed and memory usage. In
    general, larger batch sizes will lead to faster training, but might require more
    memory."""
    num_epochs: int = 1000
    """Number of epochs."""
    warmup_fraction: float = 0.01
    """Fraction of training steps used for learning rate warmup."""
    learning_rate: float = 1e-4
    """Learning rate."""
    weight_decay: Optional[float] = None
    temperature: float = 300.0
    """Temperature in kelvin used in the GLE transition density."""
    bead_mass_by_symbol: dict[str, float] = {"O": 18.01528}
    """Coarse-grained bead mass in dalton per bead type, keyed by the chemical
    symbol used as the bead's atomic-number tag in the CG structures."""
    transition_jitter: float = 1e-6
    """Diagonal jitter added before Cholesky factorization."""
    pairwise: bool = False
    """Use a pairwise, momentum-conserving GLE transition loss instead of the
    per-bead one.

    When ``False`` (default) each bead momentum is fit with an independent
    Ornstein-Uhlenbeck transition using its own drift matrix ``A_I``. This is a
    local thermostat that does not conserve total coarse momentum, so it cannot
    reproduce the exact projected (collective / hydrodynamic) dynamics of a
    translationally invariant system.

    When ``True`` the friction acts on the relative velocity of neighbor bead
    pairs (edges of the model neighbor list) with a symmetric edge drift
    ``A_e = 0.5 (A_I + A_J)``, and is pushed back to the beads with opposite sign,
    so total coarse momentum is conserved (a DPD-like memory kernel with
    ``sum_I K_IJ = 0``). Reuses the existing per-bead ``mtt::A`` output; no change
    to the model architecture. The edge set and cutoff are those of the model
    neighbor list. Runtime propagation must use a matching momentum-conserving
    (pairwise / Shardlow-style) OU step.
    """

    target_kind: str = "transition"
    """Training target. ``"transition"`` (default) fits the momentum transition
    density (per-bead or pairwise). ``"memory_kernel"`` fits A's analytic memory
    kernel ``-tr(A_ps e^{-A_ss t} A_sp)/3`` to a precomputed per-bead-type
    residual-force kernel (PMF-aware: the mean force and the slow model-error floor
    are removed offline, so A learns friction only and does not double-count the
    PMF cage at runtime)."""
    kernel_target_file: Optional[str] = None
    """Path to the ``.npz`` produced by ``build_kernel_target.py``: per-bead-type
    clean kernel ``K_z<Z>`` (keyed by atomic-number tag) on the ``t_match_ase``
    lag grid. Required when ``target_kind == "memory_kernel"``."""
    kernel_app_weight: float = 0.01
    """L2 penalty on the instantaneous 3x3 block A_pp in the memory-kernel loss
    (friction should be memory-carried, so A_pp is regularized toward zero)."""
    kernel_reg_weight: float = 0.0
    """Transport-aware regularizer for the ``transition`` target: weight of an
    auxiliary memory-kernel-matching term added to the transition NLL.

    When ``> 0`` the analytic kernel of the full drift ``A`` is matched to the
    precomputed per-type Volterra kernels in ``kernel_target_file`` (the same target
    as ``target_kind == "memory_kernel"``), on top of the transition loss. The
    transition NLL supplies environment dependence but does not pin the transport
    friction (the zero-frequency integral of the kernel); this term does, preventing
    the correction from eroding the per-type transport friction carried by the
    baseline. ``0.0`` (default) recovers the pure transition loss. Requires
    ``kernel_target_file`` to be set."""
    theta_reg_weight: float = 0.0
    """L2 penalty on the correction ``theta_net = mtt::A - theta_base[Z]`` (the
    Delta-learning residual around the frozen ``theta_baseline_file`` baseline).

    Keeps the correction small so the frozen per-type baseline stays the dominant
    prior (baseline-dominant Delta-learning). ``0.0`` (default) does not penalize the
    correction. Only meaningful together with ``theta_baseline_file``."""
    gamma0_reg_weight: float = 0.0
    """Weight of the zero-frequency-friction pin in the transition loss.

    Adds ``weight * mean(((gamma0(Q) - gamma0_base[Z]) / gamma0_base[Z])^2)`` where
    ``gamma0 = tr(A_pp - A_ps A_ss^-1 A_sp)/3`` is the transport friction (Schur
    complement) of each bead's drift, and the target is the frozen baseline's own
    per-type value. Rationale: at a single training lag the equilibrium transition
    marginal cannot distinguish dissipative friction from energy-conserving
    antisymmetric rotation into the auxiliaries, and the NLL actively drives
    ``gamma0`` to zero — deployed models then fail to thermostat and heat. This pin
    blocks that degenerate direction while leaving the finite-frequency response
    (vibrational spectrum shaping) free. Requires ``theta_baseline_file``."""
    gamma0_pin_freqs_thz: List[float] = [0.0]
    """Frequency grid (THz) for the friction pin controlled by ``gamma0_reg_weight``.

    The pinned quantity is the dissipative response
    ``Re tr(A_pp - A_ps (A_ss + i omega)^-1 A_sp)/3`` at each listed frequency,
    matched to the frozen baseline's values per bead type. ``[0.0]`` (default) is the
    plain zero-frequency gamma0 pin — which a trained model can evade by reshaping
    the kernel inside the cage-hopping band while keeping gamma0 fixed (observed:
    water D 6x with exact gamma0). A grid spanning the transport band, e.g.
    ``[0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]``, closes that escape while leaving the
    vibrational band (>2 THz) free for the NLL to shape."""

    log_interval: int = 1
    """Interval to log metrics."""
    checkpoint_interval: int = 100
    """Interval to save checkpoints."""
    atomic_baseline: FixedCompositionWeights | str = {}
    """The baselines for each target.

    By default, ``metatrain`` will fit a linear model (:class:`CompositionModel
    <metatrain.composition.CompositionModel>`) to compute the least squares
    baseline for each atomic species for each target.

    However, this hyperparameter allows you to provide your own baselines,
    either as a dictionary or as a path to a pre-trained composition model
    checkpoint. The value of the hyperparameter should either be:

    - a dictionary where the keys are the target names, and the values are
      either (1) a single baseline to be used for all atomic types, or
      (2) a dictionary mapping atomic types to their baselines.
    - a string path to a ``.ckpt`` file from a pre-trained composition model.

    For example:

    - ``atomic_baseline: {"energy": {1: -0.5, 6: -10.0}}`` will fix the energy
      baseline for hydrogen (Z=1) to -0.5 and for carbon (Z=6) to -10.0, while
      fitting the baselines for the energy of all other atomic types, as well
      as fitting the baselines for all other targets.
    - ``atomic_baseline: {"energy": -5.0}`` will fix the energy baseline for
      all atomic types to -5.0.
    - ``atomic_baseline: {"mtt:dos": 0.0}`` sets the baseline for the "mtt:dos"
      target to 0.0, effectively disabling the atomic baseline for that target.
    - ``atomic_baseline: "/path/to/model.ckpt"`` loads a pre-trained
      composition model checkpoint, overriding the default least-squares fit.

    This atomic baseline is substracted from the targets during training, which
    avoids the main model needing to learn atomic contributions, and likely makes
    training easier. When the model is used in evaluation mode, the atomic baseline
    is added on top of the model predictions automatically.

    .. note::

        This atomic baseline is a per-atom contribution. Therefore, if the property
        you are predicting is a sum over all atoms (e.g., total energy), the
        contribution of the atomic baseline to the total property will be the
        atomic baseline multiplied by the number of atoms of that type in the
        structure.
    """
    scale_targets: bool = True
    """
    Normalize targets to unit std during training.

    If true, a single scale is computed for each target, given by the uncentered
    standard deviation across all values in the dataset for that target.

    For targets with more than one property (i.e. > 1 block or >= 1 block with > 1
    property), per-property scales are also computed, and used to re-scale model
    predictions.

    See also :ref:`scale-targets`.
    """
    fixed_scaling_weights: FixedScalerWeights = {}
    """Weights for target scaling.

    This is passed to the ``fixed_weights`` argument of
    :meth:`Scaler.train_model <metatrain.utils.scaler.scaler.Scaler.train_model>`,
    see its documentation to understand exactly what to pass here.
    """
    per_structure_targets: list[str] = []
    """Targets to calculate per-structure losses and errors on."""
    num_workers: Optional[int] = None
    """Number of workers for data loading. If not provided, it is set
    automatically."""
    log_mae: bool = True
    """Log MAE alongside RMSE"""
    log_separate_blocks: bool = False
    """Log per-block error."""
    best_model_metric: Literal["rmse_prod", "mae_prod", "loss"] = "mae_prod"
    """Metric used to select best checkpoint (e.g., ``rmse_prod``)"""
    grad_clip_norm: float = 1.0
    """Maximum gradient norm value."""
    max_atoms_per_batch: Optional[int] = None
    """If set, use greedy atom-count packing instead of fixed ``batch_size``.
    Structures are accumulated into each batch until adding another would exceed this
    limit, producing variable numbers of structures per batch. Supported with any
    dataset type. When set, ``batch_size`` is ignored for constructing training
    and validation batches (it is still used internally for composition model and
    scaler fitting)."""
    min_atoms_per_batch: int = 0
    """Minimum total number of atoms required to keep a batch when
    ``max_atoms_per_batch`` is set. Batches whose total atom count falls below this
    threshold are discarded during packing. Defaults to ``0`` (no minimum)."""

    finetune: NoFinetuneHypers | FinetuneHypers = {
        "read_from": None,
        "method": "full",
        "config": {},
        "inherit_heads": {},
    }
    """Parameters for fine-tuning trained GLE models.

    See :ref:`label_fine_tuning_concept` for more details.
    """
