from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import models
from models.tsmixer import TSMixer, _FlattenBatchNorm1d, _MixerBlock, _build_norm
from pipelines.selection import resolve_requested_architectures


def _base_kwargs():
    return {
        "d_input_features": 5,
        "d_target_features": 2,
        "d_seq_in": 8,
        "d_seq_out": 4,
        "n_block": 2,
        "ff_dim": 16,
        "dropout": 0.1,
        "norm_type": "L",
        "activation": "relu",
    }


def test_tsmixer_forward_shape_multivariate_targets():
    model = TSMixer(**_base_kwargs())
    x = torch.randn(3, model.d_seq_in, model.d_input_features)
    y = torch.randn(3, model.d_seq_out, model.d_target_features)
    out = model._shared_step(x, y)

    assert out["pred"].shape == (3, model.d_seq_out, model.d_target_features)
    assert out["loss"] is not None


def test_tsmixer_forward_shape_target_subset_projection():
    kwargs = _base_kwargs()
    kwargs["target_indices"] = (1, 3)
    model = TSMixer(**kwargs)
    x = torch.randn(2, model.d_seq_in, model.d_input_features)
    y = torch.randn(2, model.d_seq_out, model.d_target_features)

    out = model._shared_step(x, y)
    assert out["pred"].shape == (2, model.d_seq_out, model.d_target_features)


def test_tsmixer_invalid_norm_type_raises():
    kwargs = _base_kwargs()
    kwargs["norm_type"] = "layer"

    with pytest.raises(ValueError, match="norm_type"):
        TSMixer(**kwargs)


def test_tsmixer_invalid_activation_raises():
    kwargs = _base_kwargs()
    kwargs["activation"] = "swish"

    with pytest.raises(ValueError, match="activation"):
        TSMixer(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_block", 0),
        ("ff_dim", 0),
        ("dropout", -0.1),
        ("dropout", 1.0),
    ],
)
def test_tsmixer_invalid_numeric_hparams_raise(field, value):
    kwargs = _base_kwargs()
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        TSMixer(**kwargs)


def test_tsmixer_is_exported_from_models_package():
    assert getattr(models, "TSMixer") is TSMixer


def test_benchmark_scope_resolves_tsmixer_architecture_case_insensitive(monkeypatch):
    monkeypatch.setattr(
        "pipelines.selection.load_hparams",
        lambda: {"GRU": {"lr": [0.001]}, "TSMixer": {"lr": [0.001]}},
    )
    args = SimpleNamespace(model=["tsmixer"], benchmark_architectures=["GRU"])
    assert resolve_requested_architectures(args) == ["TSMixer"]


def test_tsmixer_norm_configuration_matches_upstream_defaults():
    layer_norm = _build_norm(norm_type="L", d_seq_in=8, d_features=5)
    assert isinstance(layer_norm, nn.LayerNorm)
    assert layer_norm.eps == pytest.approx(1e-3)

    batch_norm = _build_norm(norm_type="B", d_seq_in=8, d_features=5)
    assert isinstance(batch_norm, _FlattenBatchNorm1d)
    assert batch_norm.norm.eps == pytest.approx(1e-3)
    assert batch_norm.norm.momentum == pytest.approx(0.01)


def test_tsmixer_block_temporal_mixing_applies_activation():
    block = _MixerBlock(
        d_seq_in=4,
        d_features=2,
        ff_dim=3,
        dropout=0.0,
        norm_type="L",
        activation="relu",
    )

    block.temporal_norm = nn.Identity()
    block.feature_norm = nn.Identity()
    block.temporal_dropout = nn.Identity()
    block.feature_dropout1 = nn.Identity()
    block.feature_dropout2 = nn.Identity()

    with torch.no_grad():
        block.temporal_mixing.weight.copy_(-torch.eye(4))
        block.temporal_mixing.bias.zero_()
        block.feature_fc1.weight.zero_()
        block.feature_fc1.bias.zero_()
        block.feature_fc2.weight.zero_()
        block.feature_fc2.bias.zero_()

    x = torch.ones(1, 4, 2)
    out = block(x)
    assert torch.allclose(out, x)
