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
  models or uniform composition weights) — otherwise ``export_nep`` raises and
  suggests ``version: 5``;
- ZBL models export with any scale, since the ZBL term is an additive
  contribution excluded from the scaler (as in GPUMD);
- NEP-Charge (qNEP) models require ``scale_targets: false``, since the Ewald
  energy is quadratic in the predicted charges and cannot be folded.

{{SECTION_MODEL_HYPERS}}

"""

from typing import Literal, Optional

from typing_extensions import NotRequired, TypedDict

from metatrain.composition.documentation import FixedCompositionWeights
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
    learning_rate: float = 0.001
    """Learning rate."""

    compute_q_scaler: bool = True
    """Compute the GPUMD-style descriptor normalisation (``1 / (max - min)``
    per descriptor dimension over the training set) before training starts.
    Only applied to freshly initialised models; restarted or fine-tuned models
    keep their stored normalisation."""

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
    fixed_composition_weights: FixedCompositionWeights = {}
    """Weights for atomic contributions.

    This is passed to the ``fixed_weights`` argument of
    :meth:`CompositionModel.train_model
    <metatrain.composition.CompositionModel.train_model>`,
    see its documentation to understand exactly what to pass here.
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
