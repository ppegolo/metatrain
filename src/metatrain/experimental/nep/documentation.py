"""
NEP (Experimental)
==================

This is an interface to the NEP (neuroevolution potential) architecture
:footcite:p:`fan_neuroevolution_2021` as implemented in `torchnep
<https://github.com/brucefan1983/GPUMD>`_, a differentiable PyTorch
implementation of the GPUMD NEP descriptors and potential.

NEP combines Chebyshev radial functions and angular many-body descriptors with
a small per-element neural network.  It is designed to be fast and
data-efficient, and models trained here use the same functional form as GPUMD's
``nep.txt`` potentials (NEP3/NEP4/NEP5), optionally with a ZBL short-range
repulsion.

.. note::

   The element list required by NEP is derived automatically from the atomic
   numbers present in the dataset; it is *not* a user-facing hyperparameter.

{{SECTION_INSTALLATION}}

This architecture additionally requires the ``torchnep`` package, which is not
available on PyPI.  Install it from the GPUMD repository checkout with
``pip install <path-to-torchnep>``.

{{SECTION_DEFAULT_HYPERS}}

Tuning hyperparameters
----------------------

The most impactful hyperparameters (roughly in decreasing order of importance):

.. container:: mtt-hypers-remove-classname

  .. autoattribute:: {{model_hypers_path}}.cutoff_radial
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.n_max_radial
      :no-index:

  .. autoattribute:: {{trainer_hypers_path}}.learning_rate
      :no-index:

  .. autoattribute:: {{trainer_hypers_path}}.batch_size
      :no-index:

``cutoff_radial`` and ``cutoff_angular`` control the interaction range and
should be chosen based on the physical system.  Increasing ``n_max_radial``,
``n_max_angular`` and ``neurons`` improves accuracy at the cost of speed.
``l_max_3body`` controls the angular resolution of the three-body
descriptors; ``l_max_4body`` (0 or 2) and ``l_max_5body`` (0 or 1) enable
four- and five-body angular terms.

``mn_radial`` and ``mn_angular`` do not affect training: they only size the
neighbor lists that GPUMD allocates for the exported ``nep.txt``.  Raise them
if GPUMD reports an illegal memory access on a dense system; the defaults are
generous enough for typical condensed-phase structures.

Fine-tuning an existing GPUMD potential
---------------------------------------

Set ``nep_model`` to an existing ``nep.txt`` file to fine-tune it instead of
training from scratch:

.. code-block:: yaml

    model:
      nep_model: path/to/nep.txt

The architecture hyperparameters are read from the file and override the
values in the ``model`` section.  Since the loaded potential already predicts
total energies, the composition weights are fixed to zero, the target scale to
one, and the stored descriptor normalisation is kept.  Regular NEP3/NEP4/NEP5
and NEP-Charge potentials (optionally with universal ZBL) are supported;
flexible-ZBL files are not.

Fine-tuning a metatrain checkpoint
----------------------------------

To continue training from a metatrain NEP checkpoint on a new dataset, set
``finetune.read_from`` in the ``training`` section:

.. code-block:: yaml

    training:
      finetune:
        read_from: path/to/model.ckpt

The model architecture and weights are taken from the checkpoint, together with
its composition baselines, target scales and descriptor normalisation, which
are all kept fixed instead of being refitted on the new dataset.  The dataset
may not introduce new atomic types or new targets: a NEP potential has a single
output head, and it keeps predicting the target it was trained on.  Unlike the
``nep_model`` route, this preserves the model exactly, without going through
the limited precision and the composition fold of the ``nep.txt`` format.

Fine-tuning starts from the *best* model of the checkpoint (not its last
epoch), with a fresh optimizer.  The first logged metrics are therefore those
of a model that has already been trained for a full epoch at ``learning_rate``:
on an already-converged potential, the default ``0.001`` is large enough to
visibly degrade it before anything is logged.  Lower ``learning_rate`` (and, if
needed, ``scheduler_patience``) when fine-tuning.

Exporting to GPUMD
------------------

A trained model can be written as a GPUMD-compatible ``nep.txt`` file with
``model.export_nep("nep.txt")`` (from Python, after loading the checkpoint).
The target scale and the per-type composition baselines used by metatrain are
folded into the NEP network so that native NEP implementations (GPUMD,
NEP_CPU, calorine, ...) reproduce the metatrain predictions exactly (up to the
file format's 7-digit precision).  The fold is gated on the NEP ``version``:

- ``version: 5`` has a per-element bias, so the fold is always exact;
- ``version: 3`` and ``version: 4`` only have a global bias, so the fold is
  exact only when the folded per-type constants coincide (e.g. single-element
  models or uniform composition weights).  Otherwise, export the model as a
  NEP5 file with ``model.export_nep("nep.txt", version=5)``: NEP5 is the same
  potential with an extra per-type bias (set to zero on promotion), so the
  predictions are unchanged and the per-type composition fold becomes exact.
  The resulting file is a regular ``nep5`` potential usable by GPUMD;
- ZBL models export with any scale, since the ZBL term is an additive
  contribution excluded from the scaler (as in GPUMD);
- NEP-Charge (qNEP) models require ``scale_targets: false``, since the Ewald
  energy is quadratic in the predicted charges and cannot be folded.

{{SECTION_MODEL_HYPERS}}

"""

from typing import Literal, Optional

from typing_extensions import NotRequired, TypedDict

from metatrain.composition.documentation import FixedCompositionWeights
from metatrain.scaler.documentation import FixedScalerWeights
from metatrain.utils.loss import LossSpecification


###########################
#  MODEL HYPERPARAMETERS  #
###########################


class ModelHypers(TypedDict):
    """Hyperparameters for the NEP model.

    These mirror the keywords of a GPUMD ``nep.in`` file.
    """

    nep_model: Optional[str] = None
    """Path to an existing GPUMD ``nep.txt`` file to fine-tune.  The file's
    architecture hyperparameters override the values in this section."""
    version: int = 4
    """NEP version. ``3``: one shared neural network for all elements;
    ``4``: one neural network per element (recommended); ``5``: as ``4``
    with an extra per-element bias."""
    cutoff_radial: float = 8.0
    """Radial descriptor cutoff radius in length units."""
    cutoff_angular: float = 4.0
    """Angular descriptor cutoff radius in length units."""
    mn_radial: int = 200
    """Maximum number of radial neighbors per atom written to the exported
    ``nep.txt``.  This value does not affect training in any way: it is only
    the neighbor-list capacity that GPUMD allocates when running the exported
    potential.  It must be at least as large as the largest number of
    neighbors within ``cutoff_radial`` of any atom in the systems GPUMD will
    simulate, otherwise GPUMD writes out of bounds and crashes with an illegal
    memory access."""
    mn_angular: int = 100
    """Maximum number of angular neighbors per atom written to the exported
    ``nep.txt``.  As for ``mn_radial``, this only sizes GPUMD's neighbor
    lists (here within ``cutoff_angular``) and has no effect on training."""
    n_max_radial: int = 4
    """Number of radial descriptor components minus one."""
    n_max_angular: int = 4
    """Number of angular descriptor components (per angular order) minus
    one."""
    basis_size_radial: int = 8
    """Number of Chebyshev basis functions for the radial descriptors minus
    one."""
    basis_size_angular: int = 8
    """Number of Chebyshev basis functions for the angular descriptors minus
    one."""
    l_max_3body: int = 4
    """Maximum angular momentum for the three-body angular descriptors."""
    l_max_4body: int = 2
    """Maximum angular momentum for the four-body angular descriptors.
    ``0`` disables them; ``2`` enables them."""
    l_max_5body: int = 0
    """Maximum angular momentum for the five-body angular descriptors.
    ``0`` disables them; ``1`` enables them."""
    neurons: int = 30
    """Number of neurons in the hidden layer of the neural network."""
    charge_mode: int = 0
    """NEP-Charge (qNEP) mode.  ``0`` disables charges (regular NEP);
    ``1`` adds real- plus k-space Ewald electrostatics from predicted
    charges; ``2`` adds k-space-only electrostatics; ``3`` is mode 2 plus a
    dynamic-C6 van der Waals term.  Requires ``version: 4`` and periodic
    systems."""
    zbl_outer_cutoff: Optional[float] = None
    """If set, add GPUMD's universal ZBL short-range repulsion, smoothly
    switched off between half this value (inner cutoff) and this value (outer
    cutoff), in Å.  GPUMD recommends values between 1 and 2.5 Å.  The ZBL term
    is handled as an additive contribution that is excluded from target
    scaling, exactly as in GPUMD."""
    seed: int = 0
    """Random seed for weight initialisation."""


##############################
#  TRAINER HYPERPARAMETERS   #
##############################


class NoFinetuneHypers(TypedDict):
    """Hypers that indicate that no fine-tuning is to be applied."""

    read_from: None = None
    """No fine-tuning is indicated by setting this argument to None."""


class FinetuneHypers(TypedDict):
    """Hyperparameters to fine-tune a NEP model from a metatrain checkpoint.

    All model parameters are trainable; the architecture hyperparameters, the
    composition baselines and the target scales are taken from the checkpoint.
    """

    read_from: str
    """Path to the metatrain NEP checkpoint (``.ckpt``) to fine-tune."""


class TrainerHypers(TypedDict):
    """Hyperparameters for training NEP models."""

    distributed: NotRequired[Optional[bool]] = None
    """Whether to use distributed training. When not set, distributed training
    is enabled automatically when running under more than one SLURM task.
    Setting this option explicitly is deprecated."""
    distributed_port: int = 39591
    """Port for DDP communication."""
    batch_size: int = 8
    """The number of samples to use in each batch of training. This
    hyperparameter controls the tradeoff between training speed and memory usage. In
    general, larger batch sizes will lead to faster training, but might require more
    memory."""
    num_epochs: int = 100
    """Number of epochs."""
    num_workers: int = 0
    """Number of dataloader worker processes, also used when fitting the
    composition model.  The default of ``0`` loads data in the main process:
    NEP datasets are held in memory with precomputed neighbor lists, so with
    the ``fork`` start method each worker copies a large part of the parent
    process and mostly multiplies memory usage.  Increase only for disk
    datasets or when profiling shows data loading to be the bottleneck."""
    learning_rate: float = 0.001
    """Learning rate."""

    compute_q_scaler: bool = True
    """Compute the GPUMD-style descriptor normalisation (``1 / (max - min)``
    per descriptor dimension over the training set) before training starts.
    Only applied to freshly initialised models; restarted or fine-tuned models
    keep their stored normalisation."""

    finetune: NoFinetuneHypers | FinetuneHypers = {"read_from": None}
    """Parameters for fine-tuning a trained NEP model from a metatrain
    checkpoint.  Set ``read_from`` to the checkpoint path; the model
    architecture, weights, composition baselines, target scales and descriptor
    normalisation are taken from it.  To fine-tune a GPUMD ``nep.txt`` file
    instead, use the ``nep_model`` model hyperparameter."""

    scheduler_patience: int = 100
    """Number of epochs with no improvement before reducing the learning rate."""
    scheduler_factor: float = 0.8
    """Factor by which the learning rate is reduced on plateau."""

    log_interval: int = 1
    """Interval to log metrics."""
    checkpoint_interval: int = 100
    """Interval to save checkpoints."""
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
    fixed_scaling_weights: FixedScalerWeights | str = {}
    """Weights for target scaling.

    This is passed to the ``fixed_weights`` argument of
    :meth:`Scaler.train_model <metatrain.scaler.Scaler.train_model>`,
    see its documentation to understand exactly what to pass here.

    Apart from those options, one can pass a path to a model checkpoint. If that
    is the checkpoint of a Scaler model, the pre-trained scaler will be loaded.
    When passing a checkpoint for the scaler, ``atomic_baseline`` must also
    be a checkpoint for a composition model.  Fine-tuned ``nep.txt`` models fix
    their own scales, which cannot be combined with a scaler checkpoint.
    """
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

    Fine-tuned ``nep.txt`` models fix their own baselines to zero, which cannot
    be combined with a composition model checkpoint (the dict form is applied
    on top instead).
    """
    per_structure_targets: list[str] = []
    """Targets to calculate per-structure losses."""
    log_mae: bool = False
    """Log MAE alongside RMSE."""
    log_separate_blocks: bool = False
    """Log per-block error."""
    best_model_metric: Literal["rmse_prod", "mae_prod", "loss"] = "rmse_prod"
    """Metric used to select best checkpoint (e.g., ``rmse_prod``)."""

    loss: str | dict[str, LossSpecification] = "mse"
    """This section describes the loss function to be used. See the
    :ref:`loss-functions` for more details."""
