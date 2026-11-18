"""Train a per-atom GLE drift model (NEP descriptors + drift head) from Python.

The drift head is a standalone ``torch.nn.Module``, not a metatrain target:
its real losses are trajectory-level (Kalman marginal likelihood along paths,
kernel matching against binned ACFs) and are not expressible as metatrain's
per-structure target loss. So there is no ``mtt train`` for it — the training
loop lives here, and this script is the template.

Run it as-is for a self-contained demo on generated structures:

    pixi run python train_gle_drift.py

Then swap two things for a real run: :func:`load_dataset` (read your own
frames) and :func:`drift_loss` (plug in the trajectory-level objective).
"""

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ase
import ase.io
import torch
from metatomic.torch import System

from metatrain.experimental.nep.gle_model import GLEDriftModel
from metatrain.experimental.nep.modules.gle import GLEDriftConfig
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64  # the drift head follows the backbone dtype


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
# The head configuration is exactly the mapping that would sit under the
# target section of a metatrain yaml, so the same dict can be loaded from
# yaml with `yaml.safe_load` and passed through.
HEAD_OPTIONS = {
    "n_aux": 4,  # auxiliary momentum channels; L is (n_aux + 1)^2 blocks of 3x3
    "n_windows": 2,  # radial windows per neighbor class
    "r_cut": 5.0,  # basis cutoff; must not exceed the NEP radial cutoff
    "gamma_min": 0.01,  # slowest auxiliary rate at init, 1/ps
    "gamma_max": 10.0,  # fastest auxiliary rate at init, 1/ps
    "eps_floor": 1.0e-4,
    "markovian_block": True,  # False: pure-memory dissipation (physics switch)
    "n_classes": 2,  # bonded / non-bonded; use 1 without a topology
}

# Descriptor hyperparameters, shared with the NEP energy models.
NEP_HYPERS = {
    **get_default_hypers("experimental.nep")["model"],
    "cutoff_radial": 6.0,
    "cutoff_angular": 4.0,
    "n_max_radial": 4,
    "n_max_angular": 4,
    "neurons": 30,
}

BATCH_SIZE = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_dataset(
    path: Optional[Path] = None, n_frames: int = 24
) -> List[ase.Atoms]:
    """Load the training frames.

    Replace this with a reader for your own trajectory; the only requirements
    are that every frame carries the coarse-grained sites as atoms and, when
    ``n_classes > 1``, a bond list (here in ``atoms.info["bonds"]``).

    :param path: Optional extxyz file to read instead of generating frames.
    :param n_frames: Number of frames to generate when ``path`` is ``None``.
    :return: The frames.
    """
    if path is not None:
        return ase.io.read(path, ":")

    # a short chain of coarse-grained beads, rattled: enough to exercise both
    # neighbor classes (consecutive beads are "bonded")
    frames = []
    for seed in range(n_frames):
        generator = torch.Generator().manual_seed(seed)
        n_beads = 8
        positions = torch.zeros(n_beads, 3, dtype=torch.float64)
        positions[:, 0] = torch.arange(n_beads, dtype=torch.float64) * 3.8
        positions += 0.4 * torch.randn(n_beads, 3, generator=generator, dtype=torch.float64)
        atoms = ase.Atoms(
            numbers=[6] * n_beads,
            positions=positions.numpy(),
            cell=[40.0, 20.0, 20.0],
            pbc=True,
        )
        atoms.info["bonds"] = [[i, i + 1] for i in range(n_beads - 1)]
        frames.append(atoms)
    return frames


def to_system(atoms: ase.Atoms) -> System:
    """Convert an ASE frame to a metatomic ``System``."""
    return System(
        types=torch.tensor(atoms.numbers, dtype=torch.int32),
        positions=torch.tensor(atoms.get_positions(), dtype=DTYPE),
        cell=torch.tensor(atoms.get_cell().array, dtype=DTYPE),
        pbc=torch.tensor(atoms.pbc),
    )


def prepare(
    frames: List[ase.Atoms], model: GLEDriftModel
) -> List[Tuple[System, torch.Tensor]]:
    """Attach the neighbor lists once, up front, and collect the topology.

    :param frames: Input frames.
    :param model: The model whose neighbor lists are needed.
    :return: ``(system, bonds)`` pairs on the target device.
    """
    requested = get_requested_neighbor_lists(model)
    prepared = []
    for atoms in frames:
        system = get_system_with_neighbor_lists(to_system(atoms), requested)
        bonds = torch.tensor(
            atoms.info.get("bonds", []), dtype=torch.long
        ).reshape(-1, 2)
        prepared.append((system.to(device=DEVICE), bonds.to(DEVICE)))
    return prepared


# --------------------------------------------------------------------------
# loss
# --------------------------------------------------------------------------
def drift_loss(
    outputs: Dict[str, torch.Tensor], reference: torch.Tensor
) -> torch.Tensor:
    """Stand-in objective: supervised regression on the drift factor.

    THIS IS THE HOOK. The real training signal is trajectory-level — a Kalman
    marginal likelihood along paths, or matching the memory kernel against
    binned autocorrelation functions. Both consume ``outputs["L_blocks"]``
    (or ``outputs["A"]``) for the atoms of a batch and return a scalar, so
    they drop in here without touching anything else.

    :param outputs: What the model returned: ``L_blocks`` and ``A``.
    :param reference: Reference blocks with the shape of ``L_blocks``.
    :return: The scalar loss.
    """
    return ((outputs["L_blocks"] - reference) ** 2).mean()


def reference_blocks(dataset, model: GLEDriftModel, seed: int = 0) -> List[torch.Tensor]:
    """Fake regression targets, only so this script runs end to end."""
    generator = torch.Generator().manual_seed(seed)
    n_channels = model.config.n_channels
    references = []
    for system, _ in dataset:
        references.append(
            0.1
            * torch.randn(
                len(system), n_channels, n_channels, 3, 3,
                generator=generator, dtype=DTYPE,
            ).to(DEVICE)
        )
    return references


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def train() -> GLEDriftModel:
    """Build, train and checkpoint a drift model.

    :return: The trained model.
    """
    frames = load_dataset()
    atomic_types = sorted({int(z) for atoms in frames for z in atoms.numbers})

    config = GLEDriftConfig.from_options(HEAD_OPTIONS)
    model = GLEDriftModel(
        hypers=NEP_HYPERS, atomic_types=atomic_types, config=config
    ).to(device=DEVICE, dtype=DTYPE)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{len(frames)} frames, {n_parameters} trainable parameters")
    print(
        f"L is {config.n_channels}x{config.n_channels} blocks of 3x3: "
        f"{config.n_blocks} lower-triangular blocks x {config.n_basis} basis "
        "elements per atom"
    )

    dataset = prepare(frames, model)
    references = reference_blocks(dataset, model)
    split = int(0.8 * len(dataset))
    train_set = list(zip(dataset, references, strict=True))[:split]
    val_set = list(zip(dataset, references, strict=True))[split:]

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )

    best_validation = float("inf")
    for epoch in range(NUM_EPOCHS):
        model.train()
        training_loss = 0.0
        order = torch.randperm(len(train_set))
        for start in range(0, len(train_set), BATCH_SIZE):
            chunk = [train_set[int(i)] for i in order[start : start + BATCH_SIZE]]
            systems = [system for (system, _), _ in chunk]
            bonds = [bond for (_, bond), _ in chunk]
            # the model concatenates the atoms of the batch, so the reference
            # blocks are concatenated the same way
            reference = torch.cat([target for _, target in chunk])

            optimizer.zero_grad()
            loss = drift_loss(model(systems, bonds=bonds), reference)
            loss.backward()
            optimizer.step()
            training_loss += float(loss.detach()) * len(chunk)
        training_loss /= len(train_set)

        model.eval()
        validation_loss = 0.0
        # a trajectory-level loss needs position gradients of `L`, so drop the
        # `no_grad` here once the real objective goes in
        with torch.no_grad():
            for (system, bond), target in val_set:
                validation_loss += float(drift_loss(model([system], bonds=[bond]), target))
        validation_loss /= max(len(val_set), 1)
        scheduler.step(validation_loss)

        if validation_loss < best_validation:
            best_validation = validation_loss
            save_checkpoint(model, "gle_drift.pt")

        if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
            print(
                f"epoch {epoch:3d} | train {training_loss:.6f} | "
                f"val {validation_loss:.6f} | lr {optimizer.param_groups[0]['lr']:.2e}"
            )

    return model


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------
def save_checkpoint(model: GLEDriftModel, path: str) -> None:
    """Save weights together with everything needed to rebuild the model.

    :param model: The model to save.
    :param path: Destination path.
    """
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": dataclasses.asdict(model.config),
            "hypers": dict(model.hypers),
            "atomic_types": model.atomic_types,
        },
        path,
    )


def load_checkpoint(path: str) -> GLEDriftModel:
    """Rebuild a model saved by :func:`save_checkpoint`.

    :param path: Checkpoint path.
    :return: The model, on ``DEVICE`` and in ``DTYPE``.
    """
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    model = GLEDriftModel(
        hypers=checkpoint["hypers"],
        atomic_types=checkpoint["atomic_types"],
        config=GLEDriftConfig(**checkpoint["config"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device=DEVICE, dtype=DTYPE)


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------
def predict(model: GLEDriftModel, atoms: ase.Atoms) -> Dict[str, torch.Tensor]:
    """Drift factor and drift matrix of one frame.

    The integrator forms the noise map itself as ``B = sqrt(2 kB T) L``; it
    never needs a Cholesky decomposition, which is the whole point of
    predicting ``L`` rather than ``A``.

    :param model: A trained model.
    :param atoms: The frame to predict.
    :return: ``L_blocks`` and ``A`` for every atom of the frame.
    """
    system = get_system_with_neighbor_lists(
        to_system(atoms), get_requested_neighbor_lists(model)
    ).to(DEVICE)
    bonds = torch.tensor(atoms.info.get("bonds", []), dtype=torch.long).reshape(-1, 2)
    model.eval()
    with torch.no_grad():
        return model([system], bonds=[bonds.to(DEVICE)])


if __name__ == "__main__":
    trained = train()

    reloaded = load_checkpoint("gle_drift.pt")
    frame = load_dataset(n_frames=1)[0]
    outputs = predict(reloaded, frame)

    blocks, drift = outputs["L_blocks"], outputs["A"]
    print()
    print(f"L_blocks {tuple(blocks.shape)} | A {tuple(drift.shape)}")
    rates = torch.linalg.eigvalsh(drift.cpu().double())
    print(f"drift eigenvalues in [{float(rates.min()):.2e}, {float(rates.max()):.2e}] 1/ps")
    print(f"A symmetric to {float((drift - drift.transpose(-1, -2)).abs().max()):.1e}")
    print(f"A positive semidefinite: {bool(rates.min() >= -1e-10)}")
