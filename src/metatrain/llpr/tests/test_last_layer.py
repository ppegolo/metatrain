import torch

from metatrain.utils.last_layer import (
    LastLayerSlice,
    assemble_block_weights,
    resolve_last_layer_slices,
)


class NamesModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(4, 3, bias=False)
        self.b = torch.nn.Linear(2, 3, bias=False)
        self.last_layer_parameter_names = {
            "target": {"block": ["a.weight", "b.weight"]}
        }


def test_names_fallback_matches_concatenation():
    model = NamesModel()
    resolved = resolve_last_layer_slices(model, "target")
    assert list(resolved.keys()) == ["block"]
    weights = assemble_block_weights(model.state_dict(), resolved["block"])
    expected = torch.concatenate([model.a.weight, model.b.weight], dim=-1)
    assert torch.equal(weights, expected)
    assert resolve_last_layer_slices(model, "other") == {}


def test_slice_offset_transpose_and_scale():
    flat = torch.arange(20.0)
    state_dict = {"p": flat}
    # stored feature-major (n_features=4, n_properties=2) at offset 3, scale 0.5
    s = LastLayerSlice(
        parameter_name="p",
        offset=3,
        shape=(2, 4),
        scale=0.5,
        transpose_stored=True,
    )
    weights = assemble_block_weights(state_dict, [s])
    expected = 0.5 * flat[3 : 3 + 8].reshape(4, 2).T
    assert torch.equal(weights, expected)


def test_declared_slices_take_precedence():
    model = NamesModel()
    model.last_layer_parameter_slices = {
        "target": {"other_block": [("a.weight", 0, (3, 4), 1.0, False)]}
    }
    resolved = resolve_last_layer_slices(model, "target")
    assert list(resolved.keys()) == ["other_block"]
    weights = assemble_block_weights(model.state_dict(), resolved["other_block"])
    assert torch.equal(weights, model.a.weight)
