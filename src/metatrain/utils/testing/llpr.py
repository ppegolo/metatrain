"""Shared behavioral tests of an architecture's last-layer (LLPR) interface.

Architectures that declare last-layer features (see
:mod:`metatrain.utils.last_layer`) opt in by subclassing
:class:`LLPRInterfaceTests` next to their own tests and implementing
``make_backbone``; the suite then runs in the architecture's own test
environment, with its own dependencies.
"""

from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.llpr.model import LLPRUncertaintyModel
from metatrain.utils.abc import ModelInterface
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.last_layer import LastLayerSlice, assemble_block_weights
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


SPHERICAL_TARGET = "mtt::spherical"
SPHERICAL_UNCERTAINTY = "mtt::aux::spherical_uncertainty"
SPHERICAL_ENSEMBLE = "mtt::aux::spherical_ensemble"
SPHERICAL_LLF = "mtt::aux::spherical_last_layer_features"
NUM_ENSEMBLE_MEMBERS = 32


def spherical_dataset_info(
    irreps: Optional[List[Dict[str, int]]] = None,
) -> DatasetInfo:
    """A single-element dataset info with one spherical target.

    :param irreps: the target's irreps; a lambda=0 and a lambda=2 block by
        default.
    :return: the dataset info.
    """
    if irreps is None:
        irreps = [
            {"o3_lambda": 0, "o3_sigma": 1},
            {"o3_lambda": 2, "o3_sigma": 1},
        ]
    target = get_generic_target_info(
        SPHERICAL_TARGET,
        {
            "quantity": "",
            "unit": "",
            "num_subtargets": 1,
            "sample_kind": "system",
            "type": {"spherical": {"irreps": irreps}},
        },
    )
    return DatasetInfo(
        length_unit="Angstrom", atomic_types=[6], targets={SPHERICAL_TARGET: target}
    )


def make_systems(model: ModelInterface, n_systems: int) -> List[System]:
    """Random four-atom carbon clusters with the model's neighbor lists.

    :param model: the model the systems are built for.
    :param n_systems: how many systems to build.
    :return: the systems.
    """
    torch.manual_seed(0)
    systems = []
    for _ in range(n_systems):
        system = System(
            types=torch.tensor([6, 6, 6, 6]),
            positions=torch.eye(4, 3, dtype=torch.float64)
            + 0.4 * torch.randn(4, 3, dtype=torch.float64),
            cell=torch.zeros((3, 3), dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )
        systems.append(
            get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
        )
    return systems


# a fixed non-trivial rotation (orthogonal, determinant +1)
ROTATION = torch.tensor(
    [
        [-0.37308922665846000, 0.78897943879082055, 0.48817606876690917],
        [-0.35050433284023819, -0.60703397465603248, 0.71320156076211638],
        [0.85904082651036540, 0.09497999146462899, 0.50301854797787238],
    ],
    dtype=torch.float64,
)


def rotate_system(model: ModelInterface, system: System) -> System:
    """The system under the fixed ``ROTATION``, with fresh neighbor lists.

    :param model: the model the system is built for.
    :param system: the system to rotate.
    :return: the rotated system.
    """
    rotated = System(
        types=system.types,
        positions=system.positions @ ROTATION.T,
        cell=system.cell,
        pbc=system.pbc,
    )
    return get_system_with_neighbor_lists(rotated, model.requested_neighbor_lists())


def wrap_backbone(
    backbone: ModelInterface,
    dataset_info: DatasetInfo,
    ensembles: bool,
    target: str = SPHERICAL_TARGET,
) -> LLPRUncertaintyModel:
    """The backbone wrapped in an LLPR model.

    :param backbone: the backbone to wrap.
    :param dataset_info: the dataset info the backbone was built with.
    :param ensembles: whether to request ensemble members for the target.
    :param target: the target ensembles are requested for.
    :return: the wrapped model.
    """
    num_ensemble_members = {target: NUM_ENSEMBLE_MEMBERS} if ensembles else {}
    model = LLPRUncertaintyModel(
        {"num_ensemble_members": num_ensemble_members}, dataset_info
    )
    model.set_wrapped_model(backbone)
    return model.to(torch.float64)


def random_target_like(
    layout: TensorMap, system_index: int, num_atoms: int = 1
) -> TensorMap:
    """A random target TensorMap with the layout's blocks for one system.

    :param layout: the target's layout.
    :param system_index: the system the samples point at.
    :param num_atoms: number of atoms, for per-atom layouts.
    :return: the random target.
    """
    blocks = []
    for block in layout.blocks():
        if "atom" in block.samples.names:
            samples = Labels(
                ["system", "atom"],
                torch.tensor([[system_index, a] for a in range(num_atoms)]),
            )
        else:
            samples = Labels(["system"], torch.tensor([[system_index]]))
        shape = [len(samples)] + list(block.values.shape[1:])
        blocks.append(
            TensorBlock(
                values=torch.randn(shape, dtype=torch.float64),
                samples=samples,
                components=block.components,
                properties=block.properties,
            )
        )
    return TensorMap(layout.keys, blocks)


def fit_llpr(
    model: LLPRUncertaintyModel,
    systems: List[System],
    target: str = SPHERICAL_TARGET,
) -> None:
    """Fit the model's covariance and Cholesky factor on random targets.

    :param model: the wrapped model.
    :param systems: the systems to fit on.
    :param target: the target to fit.
    """
    layout = model.dataset_info.targets[target].layout
    targets = [
        random_target_like(layout, i, num_atoms=len(systems[i].positions))
        for i in range(len(systems))
    ]
    dataset = Dataset.from_dict({"system": systems, target: targets})
    model.compute_covariance([dataset], batch_size=2, is_distributed=False)
    model.compute_cholesky_decomposition(regularizer=1e-3)


class LLPRInterfaceTests:
    """Contract of an architecture's LLPR interface on a spherical target:
    the declared weight slices reproduce the per-block readout exactly, and
    uncertainties and ensembles are O(3)-consistent."""

    block_keys = [
        "mtt::spherical_o3_lambda_0_o3_sigma_1",
        "mtt::spherical_o3_lambda_2_o3_sigma_1",
    ]

    def make_backbone(self, dataset_info: DatasetInfo) -> ModelInterface:
        """The architecture's model, small and in float64.

        :param dataset_info: the dataset info to build the model with.
        :return: the model.
        """
        raise NotImplementedError

    def test_slices_reproduce_readout(self) -> None:
        """Check the declared weight slices, applied to the exposed features,
        reproduce every block of the prediction."""
        dataset_info = spherical_dataset_info()
        backbone = self.make_backbone(dataset_info)
        state_dict = backbone.state_dict()
        system = make_systems(backbone, 1)[0]
        out = backbone(
            [system],
            {
                SPHERICAL_TARGET: ModelOutput(sample_kind="system"),
                SPHERICAL_LLF: ModelOutput(sample_kind="system"),
            },
        )
        llf = out[SPHERICAL_LLF]
        prediction = out[SPHERICAL_TARGET]
        assert llf.keys == prediction.keys
        for index, block_key in enumerate(self.block_keys):
            slices = [
                LastLayerSlice(*s)
                for s in backbone.last_layer_parameter_slices[SPHERICAL_TARGET][
                    block_key
                ]
            ]
            weights = assemble_block_weights(state_dict, slices)
            features = llf.block(index).values
            if features.dim() == 2:
                features = features.unsqueeze(1)
            computed = torch.einsum("smk,pk->smp", features, weights)
            reference = prediction.block(index).values
            if reference.dim() == 2:
                reference = reference.unsqueeze(1)
            assert torch.allclose(computed, reference, atol=1e-12)

    def test_uncertainty_component_resolved_and_rotation_invariant(self) -> None:
        """Check the lambda=2 uncertainty resolves the m-channels and its
        norm over m is rotation-invariant."""
        dataset_info = spherical_dataset_info()
        model = wrap_backbone(
            self.make_backbone(dataset_info), dataset_info, ensembles=False
        )
        systems = make_systems(model.model, 8)
        fit_llpr(model, systems)

        test_system = systems[0]
        rotated_system = rotate_system(model.model, test_system)
        outputs = {SPHERICAL_UNCERTAINTY: ModelOutput(sample_kind="system")}
        unc = model([test_system], outputs)[SPHERICAL_UNCERTAINTY]
        unc_rot = model([rotated_system], outputs)[SPHERICAL_UNCERTAINTY]

        lambda2 = unc.block(1).values  # (1, 5, 1)
        lambda2_rot = unc_rot.block(1).values
        # component-resolved: the five m-channels must not all coincide
        assert not torch.allclose(lambda2, lambda2.mean(dim=1, keepdim=True))
        # sum over m of the variances is the O(3)-invariant scalar
        assert torch.allclose(
            (lambda2**2).sum(dim=1), (lambda2_rot**2).sum(dim=1), rtol=1e-6
        )

    def test_ensemble_recentered_and_rotation_invariant(self) -> None:
        """Check ensemble members re-center on the prediction and their
        centered norm over m is rotation-invariant."""
        dataset_info = spherical_dataset_info()
        model = wrap_backbone(
            self.make_backbone(dataset_info), dataset_info, ensembles=True
        )
        systems = make_systems(model.model, 8)
        fit_llpr(model, systems)
        model.generate_ensemble()

        test_system = systems[0]
        rotated_system = rotate_system(model.model, test_system)
        outputs = {
            SPHERICAL_TARGET: ModelOutput(sample_kind="system"),
            SPHERICAL_ENSEMBLE: ModelOutput(sample_kind="system"),
        }
        out = model([test_system], outputs)
        out_rot = model([rotated_system], outputs)

        for index in range(2):
            members = out[SPHERICAL_ENSEMBLE].block(index).values
            prediction = out[SPHERICAL_TARGET].block(index).values
            if prediction.dim() == 2:
                prediction = prediction.unsqueeze(-1)
            if members.dim() == 2:
                members = members.unsqueeze(1)
            # re-centering: the ensemble mean is the model prediction
            assert torch.allclose(
                members.mean(dim=-1, keepdim=True), prediction, atol=1e-10
            )
            # each centered member's norm over m is rotation-invariant
            members_rot = out_rot[SPHERICAL_ENSEMBLE].block(index).values
            if members_rot.dim() == 2:
                members_rot = members_rot.unsqueeze(1)
            centered = members - members.mean(dim=-1, keepdim=True)
            centered_rot = members_rot - members_rot.mean(dim=-1, keepdim=True)
            assert torch.allclose(
                (centered**2).sum(dim=1), (centered_rot**2).sum(dim=1), rtol=1e-6
            )
