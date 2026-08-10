"""Several loss terms on one target.

A density target is measured by more than one thing at once -- the quadratic
density error and the electrostatic complementarity, say -- so a target's
configuration may be a list of specifications instead of one. The terms are
summed with their own weights, the first keeps the target's own name so that
single-term configurations are untouched, and the machinery hooks must see
every term of the list, not just the first.

PySCF-free: the density and EC hooks are inspected through the configuration
helpers, which are pure dictionary handling.
"""

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.utils.data import TargetInfo
from metatrain.utils.density_hooks import (
    _aux_bases_by_metric,
    _ec_jitter,
    _ec_targets,
)
from metatrain.utils.loss import LossAggregator, build_reported_losses


def _scalar_map(values):
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor(values, dtype=torch.float64).reshape(-1, 1),
                samples=Labels(
                    ["system"],
                    torch.arange(len(values), dtype=torch.int32).reshape(-1, 1),
                ),
                components=[],
                properties=Labels.range("property", 1),
            )
        ],
    )


@pytest.fixture
def target_info():
    return TargetInfo(layout=_scalar_map([]), quantity="energy", unit="eV")


@pytest.fixture
def predictions_and_targets():
    return {"output": _scalar_map([1.0, 2.0])}, {"output": _scalar_map([0.0, 0.0])}


def test_terms_are_summed_with_their_own_weights(target_info, predictions_and_targets):
    predictions, targets = predictions_and_targets
    config = {
        "output": [
            {"type": "mse", "weight": 1.0, "reduction": "sum", "gradients": {}},
            {"type": "mae", "weight": 0.5, "reduction": "sum"},
        ]
    }
    loss = LossAggregator(targets={"output": target_info}, config=config)

    # 1.0 * (1^2 + 2^2) + 0.5 * (|1| + |2|)
    expected = torch.tensor(6.5, dtype=torch.float64)
    torch.testing.assert_close(loss(predictions, targets), expected)


def test_a_single_specification_is_unchanged(target_info, predictions_and_targets):
    """The list form must not alter how a one-term configuration behaves."""
    predictions, targets = predictions_and_targets
    single = {"type": "mse", "weight": 1.0, "reduction": "sum", "gradients": {}}
    one = LossAggregator(targets={"output": target_info}, config={"output": single})
    listed = LossAggregator(
        targets={"output": target_info}, config={"output": [single]}
    )

    assert list(one.losses) == list(listed.losses) == ["output"]
    torch.testing.assert_close(one(predictions, targets), listed(predictions, targets))


def test_extra_terms_are_named_and_reported_separately(target_info):
    config = {
        "output": [
            {"type": "mse", "weight": 1.0, "reduction": "sum", "gradients": {}},
            {"type": "mae", "weight": 0.5, "reduction": "mean"},
        ]
    }
    loss = LossAggregator(targets={"output": target_info}, config=config)

    assert list(loss.losses) == ["output", "output[1]"]
    # Every term keeps the target it measures, whatever key it is filed under.
    assert [term.target for term in loss.losses.values()] == ["output", "output"]
    assert loss.metadata["output"]["type"] == "mse"
    assert loss.metadata["output[1]"] == {
        "type": "mae",
        "weight": 0.5,
        "reduction": "mean",
        "gradients": {},
    }


def test_an_empty_list_is_rejected(target_info):
    with pytest.raises(ValueError, match="empty list of loss terms"):
        LossAggregator(targets={"output": target_info}, config={"output": []})


def test_a_reported_metric_may_not_be_a_list(target_info):
    """One aggregator reports one number under its target's name."""
    with pytest.raises(ValueError, match="must be a single one"):
        build_reported_losses(
            {"output": [{"type": "mse"}, {"type": "mae"}]},
            {"output": target_info},
        )


def test_hooks_see_every_term_of_a_list():
    """A density term and an EC term on one target both need their machinery."""
    specs = {
        "mtt::ri": [
            {
                "type": "density_mse_via_c",
                "aux_basis": "def2-universal-jfit",
                "metric": "coulomb",
            },
            {
                "type": "ec_mse",
                "aux_basis": "def2-universal-jfit",
                "partner_jitter": 0.4,
            },
        ]
    }
    assert _aux_bases_by_metric(specs) == {
        "coulomb": {"mtt::ri": "def2-universal-jfit"}
    }
    assert _ec_targets(specs) == {"mtt::ri": "def2-universal-jfit"}
    assert _ec_jitter(specs) == 0.4


def test_conflicting_jitters_inside_lists_are_caught():
    specs = {
        "a": [{"type": "ec_mse", "aux_basis": "x", "partner_jitter": 0.4}],
        "b": [{"type": "ec_mse", "aux_basis": "x", "partner_jitter": 0.2}],
    }
    with pytest.raises(ValueError, match="cannot differ between targets"):
        _ec_jitter(specs)


def test_the_options_validation_accepts_a_list():
    """The typed hypers gate the config before anything is built.

    A list has to survive ``check_architecture_options`` or the run dies at
    startup, whatever the aggregator supports.
    """
    import copy

    from metatrain.utils.architectures import (
        check_architecture_options,
        get_default_hypers,
    )

    base = get_default_hypers("pet")
    base["name"] = "pet"
    terms = [
        {"type": "density_mse_via_c", "aux_basis": "def2-universal-jfit"},
        {"type": "ec_mse", "aux_basis": "def2-universal-jfit", "weight": 1.0e4},
    ]

    for loss in ({"mtt::rho": terms}, {"mtt::rho": terms[0]}, "mse"):
        options = copy.deepcopy(base)
        options["training"]["loss"] = loss
        check_architecture_options("pet", options)

    # ... and a list must not become a hole in the validation
    options = copy.deepcopy(base)
    options["training"]["loss"] = {"mtt::rho": [{"weight": "not a number"}]}
    with pytest.raises(Exception, match="(?i)valid|type"):
        check_architecture_options("pet", options)


def test_the_config_expansion_handles_a_list():
    """``expand_loss_config`` runs before anything else reads the loss block.

    It fills in defaults per target, and it saw only mappings, so a list reached
    ``raw.items()`` and died with "ListConfig does not support attribute
    access". Each term must come out fully specified.
    """
    from omegaconf import OmegaConf

    from metatrain.utils.omegaconf import expand_loss_config

    conf = OmegaConf.create(
        {
            "training_set": {"targets": {"mtt::rho": {}}},
            "architecture": {
                "training": {
                    "loss": {
                        "mtt::rho": [
                            {"type": "density_mse_via_c", "aux_basis": "x"},
                            {"type": "ec_mse", "aux_basis": "x", "weight": 1.0e4},
                        ]
                    }
                }
            },
        }
    )
    terms = OmegaConf.to_container(
        expand_loss_config(conf)["architecture"]["training"]["loss"]["mtt::rho"],
        resolve=True,
    )

    assert isinstance(terms, list) and len(terms) == 2
    # Defaults filled on every term, not only the first.
    assert terms[0]["type"] == "density_mse_via_c"
    assert terms[0]["weight"] == 1.0
    assert terms[0]["reduction"] == "mean"
    assert terms[0]["aux_basis"] == "x"
    assert terms[1]["type"] == "ec_mse"
    assert terms[1]["weight"] == 1.0e4
    assert terms[1]["reduction"] == "mean"
    # Gradients belong to the first term alone; a second empty section would
    # build a duplicate of every gradient loss.
    assert "gradients" in terms[0]
    assert "gradients" not in terms[1]

    # A string term keeps its shorthand meaning inside a list.
    conf["architecture"]["training"]["loss"] = {"mtt::rho": ["mse", {"type": "mae"}]}
    terms = OmegaConf.to_container(
        expand_loss_config(conf)["architecture"]["training"]["loss"]["mtt::rho"],
        resolve=True,
    )
    assert [t["type"] for t in terms] == ["mse", "mae"]

    conf["architecture"]["training"]["loss"] = {"mtt::rho": []}
    with pytest.raises(ValueError, match="empty list of loss terms"):
        expand_loss_config(conf)


def test_the_expanded_list_builds_an_aggregator(target_info, predictions_and_targets):
    """End to end: what the expansion writes must be what the aggregator eats."""
    from omegaconf import OmegaConf

    from metatrain.utils.omegaconf import expand_loss_config

    conf = OmegaConf.create(
        {
            "training_set": {"targets": {"output": {}}},
            "architecture": {
                "training": {
                    "loss": {
                        "output": [
                            {"type": "mse", "weight": 1.0, "reduction": "sum"},
                            {"type": "mae", "weight": 0.5, "reduction": "sum"},
                        ]
                    }
                }
            },
        }
    )
    expanded = OmegaConf.to_container(
        expand_loss_config(conf)["architecture"]["training"]["loss"], resolve=True
    )
    loss = LossAggregator(targets={"output": target_info}, config=expanded)

    predictions, targets = predictions_and_targets
    assert list(loss.losses) == ["output", "output[1]"]
    torch.testing.assert_close(
        loss(predictions, targets), torch.tensor(6.5, dtype=torch.float64)
    )


def test_the_original_frame_is_claimed_from_any_term():
    """Missing this would not raise -- it would silently rotate the metric.

    The density and EC losses are evaluated against machinery built on the
    unaugmented geometry, so they must opt out of the rotational augmentation.
    A target that carries the density term second would otherwise be augmented.
    """
    from metatrain.utils.augmentation import original_frame_targets

    density = {"type": "density_mse_via_c", "aux_basis": "x"}
    assert original_frame_targets({"a": density}) == {"a"}
    assert original_frame_targets({"a": [density]}) == {"a"}
    assert original_frame_targets({"a": [{"type": "mse"}, density]}) == {"a"}
    assert original_frame_targets({"a": [{"type": "ec_mse", "aux_basis": "x"}]}) == {
        "a"
    }
    # ... and a list of ordinary losses still claims nothing
    assert original_frame_targets({"a": [{"type": "mse"}, {"type": "mae"}]}) == set()
    assert original_frame_targets("mse") == set()


def test_the_shorthand_string_form_still_passes_through():
    """``loss: mse`` reaches the helpers as a string and configures no hooks."""
    assert _aux_bases_by_metric({"output": "mse"}) == {}
    assert _ec_targets({"output": "mse"}) == {}
    assert _ec_jitter({"output": "mse"}) == 0.0
