"""Backbone-agnostic GLE model: any metatrain architecture can predict ``mtt::A``.

The GLE architecture used to BE PET plus one extra output, binding it to
``PETBackend``'s ``preprocess`` / ``calculate_features`` / ``predict`` contract. That
coupling was never necessary: ``mtt::A`` is an ordinary per-atom spherical target, and every
metatrain architecture already knows how to grow a head for one (SOAP-BPNN builds a
``TensorBasis`` per ``(o3_lambda, o3_sigma)``; SPACE is equivariant by construction). So the
backbone becomes a CONFIG CHOICE and the GLE side never touches it.

What is genuinely GLE-specific, and therefore lives here:

* **the target declaration** -- ``mtt::A`` as ``l = 0 (+) 1 (+) 2`` (see
  :mod:`metatrain.gle.covariant`), injected into the ``DatasetInfo`` handed to the backbone
  so it builds the right head by itself;
* **the per-type isotropic baseline** ``theta_base[Z]``, added to the network's prediction so
  the model is an exact Delta-learner at initialisation. It has to live in the MODEL rather
  than the trainer because the deployment reads ``mtt::A`` off the exported artefact and
  expects the total;
* **zero-initialisation of the readout**, so that at step zero the prediction is exactly the
  baseline.

**The baseline is isotropic (l = 0 only), and that is the physics, not a simplification.**
A type-averaged kernel must be isotropic: averaging ``K(Q)`` over all orientations of a
bead's environment leaves only the scalar part. So type sets the MAGNITUDE -- R1 measured an
8x contrast across bead types -- while the environment sets the ORIENTATION, which is exactly
the ``l = 1, 2`` correction the network learns on top.

Nothing here constrains which architecture is used. PET is invariant and would need
rotational augmentation to learn the ``l > 0`` channels; training it WITHOUT augmentation is
a useful control that isolates what covariance buys, not a misconfiguration.
"""

from typing import Any, Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorMap
from metatomic.torch import (
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from metatrain.utils.abc import ModelInterface
from metatrain.utils.architectures import (
    get_default_hypers,
    get_hypers_classes,
    import_architecture,
)
from metatrain.utils.data import DatasetInfo
from metatrain.utils.pydantic import validate as pydantic_validate

from .covariant import gle_target_info_covariant, irrep_property_counts


GLE_OUTPUT = "mtt::A"


def validate_backbone_hypers(backbone: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a nested ``backbone`` block against THAT architecture's own schema.

    Delegating keeps the GLE schema from having to re-declare every backbone's
    hyperparameters -- which would reintroduce exactly the coupling this module removes, and
    would mean adding a backbone required editing the GLE architecture.
    """
    if "name" not in backbone:
        raise ValueError(
            "the `backbone` block needs a `name` (e.g. 'soap_bpnn', 'pet', 'space')"
        )
    name = backbone["name"]
    given = {key: value for key, value in backbone.items() if key != "name"}
    # Merge over the backbone's OWN defaults first: its `ModelHypers` marks every field
    # required, so validating a partial block would demand the user restate every
    # hyperparameter of an architecture they only wanted to name.
    hypers = {**get_default_hypers(name, base_precision=64)["model"], **given}
    model_hypers = get_hypers_classes(name)["model"]
    # `ModelHypers` is a TypedDict, NOT a pydantic BaseModel: calling it would build a plain
    # dict and validate nothing. metatrain wraps it in a `TypeAdapter` instead, which is what
    # enforces `extra="forbid"` -- so the same helper is used here, and unknown or ill-typed
    # hypers are rejected with that architecture's own error messages.
    validated = pydantic_validate(model_hypers, hypers)
    return {"name": name, **validated}


class GLEWrapper(ModelInterface):
    """Wrap any metatrain model so that it predicts the GLE drift parameters.

    :param hypers: GLE hypers, including a nested ``backbone`` block.
    :param dataset_info: dataset info for the underlying targets; ``mtt::A`` is added here,
        so the caller does not declare it.
    """

    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]
    __default_metadata__ = ModelMetadata(
        references={"architecture": ["covariant GLE (Mori-Zwanzig)"]}
    )

    def __init__(
        self,
        hypers: Dict[str, Any],
        dataset_info: DatasetInfo,
        metadata: Optional[ModelMetadata] = None,
    ):
        super().__init__(hypers, dataset_info, metadata or ModelMetadata())
        self._hypers = dict(hypers)
        self._dataset_info = dataset_info
        self.n_aux = int(hypers["num_auxiliary_variables"])
        backbone_hypers = validate_backbone_hypers(hypers["backbone"])
        self.backbone_name = backbone_hypers.pop("name")

        # Inject `mtt::A` into the targets so the backbone grows its own head for it.
        targets = dict(dataset_info.targets)
        targets[GLE_OUTPUT] = gle_target_info_covariant(self.n_aux)
        info = DatasetInfo(
            length_unit=dataset_info.length_unit,
            atomic_types=dataset_info.atomic_types,
            targets=targets,
        )

        module = import_architecture(self.backbone_name)
        self.backbone = module.__model__(backbone_hypers, info)

        # Per-type ISOTROPIC baseline: one value per property of the l = 0 block, per type.
        # Ordered to match the l = 0 block's properties, i.e. the M and N block grids.
        n_scalar = irrep_property_counts(self.n_aux)[0]
        max_type = max(dataset_info.atomic_types) + 1
        self.register_buffer(
            "theta_baseline", torch.zeros(max_type, n_scalar, dtype=torch.float64)
        )

        if hypers.get("zero_init_readout", False):
            self._zero_readout()

    def __getattr__(self, name: str):
        """Fall through to the backbone for anything the wrapper does not define.

        The trainer was written against a model that WAS the network (``additive_models``,
        ``scaler``, and friends live on it), so a wrapper has to forward those rather than
        enumerate them -- an explicit list would silently go stale the moment the trainer
        touched one more attribute.

        ``torch.nn.Module.__getattr__`` handles parameters, buffers and submodules, so it
        gets first refusal; only genuinely unknown names reach the backbone.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            backbone = self.__dict__.get("_modules", {}).get("backbone")
            if backbone is None or name == "backbone":
                raise
            return getattr(backbone, name)

    def _zero_readout(self) -> None:
        """Zero the last layer of the ``mtt::A`` head, wherever the backbone put it.

        Located by NAME rather than by a known attribute path, because that path differs
        between architectures -- which is the whole point of not depending on one.
        """
        zeroed = 0
        flat = GLE_OUTPUT.replace("mtt::", "")
        for name, parameter in self.backbone.named_parameters():
            if (GLE_OUTPUT in name or flat in name) and "last" in name:
                torch.nn.init.zeros_(parameter)
                zeroed += 1
        if zeroed == 0:
            raise RuntimeError(
                f"zero_init_readout found no last-layer parameters for {GLE_OUTPUT} in "
                f"the {self.backbone_name} backbone; the Delta-learning start would "
                "silently not be a Delta-learning start"
            )

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        predictions = self.backbone(systems, outputs, selected_atoms)
        if GLE_OUTPUT not in predictions:
            return predictions

        tensor = predictions[GLE_OUTPUT]
        keys = tensor.keys
        blocks = []
        species = torch.cat([system.types for system in systems]).to(torch.long)
        for key, block in tensor.items():
            values = block.values
            if int(key["o3_lambda"]) == 0:
                base = self.theta_baseline.to(values.dtype).index_select(0, species)
                values = values + base.unsqueeze(1)
            blocks.append(_rebuild(block, values))
        predictions[GLE_OUTPUT] = TensorMap(keys=keys, blocks=blocks)
        return predictions

    @property
    def finetune_config(self) -> Dict[str, Any]:
        """Fine-tuning configuration, empty unless the backbone declares one.

        The trainer records this at checkpoint time; backbones that have no notion of it
        (anything other than PET) would otherwise abort the run at the first save.
        """
        return getattr(self.backbone, "finetune_config", {}) or {}

    @property
    def covariant(self) -> bool:
        """This branch's GLE is covariant by construction."""
        return bool(self._hypers.get("covariant", True))

    @property
    def n_gle_variables(self) -> int:
        """Dimension of the extended state, which the trainer sizes its loss from.

        Covariant: the auxiliaries are 3-VECTORS, so the state is ``3 (1 + n_aux)`` rather
        than ``3 + n_aux``. Defined on the wrapper rather than delegated -- the backbone has
        no notion of a GLE state.
        """
        return 3 * (1 + self.n_aux) if self.covariant else 3 + self.n_aux

    def requested_inputs(self) -> Dict[str, ModelOutput]:
        """Extra per-system data the GLE loss needs, beyond positions and types.

        Owned by the wrapper rather than delegated: these are demanded by the GLE
        TRANSITION target (the momenta after the lag, and the lag itself), not by whatever
        backbone happens to compute the features. Backbones that have their own
        conditioning inputs contribute theirs on top.
        """
        inputs = {}
        backbone_inputs = getattr(self.backbone, "requested_inputs", None)
        if callable(backbone_inputs):
            inputs.update(backbone_inputs())
        return inputs

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.backbone.supported_outputs()

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return self.backbone.requested_neighbor_lists()


    # --- ModelInterface: delegate what the backbone owns, keep what is GLE's ------------
    #
    # The wrapper adds one output and one buffer; everything structural (heads, weights,
    # neighbour lists, export machinery) belongs to the backbone, so the checkpoint is the
    # backbone's plus the GLE-specific state. Re-implementing the backbone's serialisation
    # here would silently diverge from it the moment upstream changed.

    def restart(self, dataset_info: DatasetInfo) -> "GLEWrapper":
        targets = dict(dataset_info.targets)
        targets[GLE_OUTPUT] = gle_target_info_covariant(self.n_aux)
        self.backbone.restart(
            DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=dataset_info.atomic_types,
                targets=targets,
            )
        )
        return self

    def get_checkpoint(self) -> Dict[str, Any]:
        return {
            "architecture_name": "gle",
            "model_ckpt_version": self.__checkpoint_version__,
            "gle_hypers": self._hypers,
            "gle_state": {"theta_baseline": self.theta_baseline},
            "backbone_name": self.backbone_name,
            "backbone_checkpoint": self.backbone.get_checkpoint(),
        }

    @classmethod
    def load_checkpoint(
        cls, checkpoint: Dict[str, Any], context: str
    ) -> "GLEWrapper":
        raise NotImplementedError(
            "checkpoint loading for the covariant GLE wrapper is not implemented yet; "
            "this branch treats checkpoints as disposable while the construction is "
            "being validated"
        )

    def export(self, metadata=None):
        raise NotImplementedError(
            "export for the covariant GLE wrapper is not implemented yet: the deployment "
            "(run_gle.py) still assumes the scalar 3 + n_aux state and has to be moved to "
            "3 (1 + n_aux) first"
        )

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(
            "the covariant construction is a different state space from the scalar one, "
            "so scalar checkpoints are deliberately not upgradeable to it"
        )

def _rebuild(block, values):
    """A TensorBlock with new values and the original metadata."""
    from metatensor.torch import TensorBlock

    return TensorBlock(
        values=values,
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )
