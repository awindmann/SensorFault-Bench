from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch

import models
from config_loader import load_hparams
from models.moderntcn import ModernTCN
from pipelines.selection import resolve_requested_architectures


def _base_kwargs() -> dict:
    return {
        "d_input_features": 3,
        "d_target_features": 2,
        "d_seq_in": 16,
        "d_seq_out": 5,
        "target_indices": None,
        "d_model": 16,
        "num_blocks": 2,
        "large_size": 7,
        "small_size": 3,
        "ffn_ratio": 2,
        "patch_size": 8,
        "patch_stride": 4,
        "dropout": 0.1,
        "head_dropout": 0.0,
        "individual": False,
    }


def test_moderntcn_forward_shape_full_targets():
    model = ModernTCN(**_base_kwargs())
    x = torch.randn(2, model.d_seq_in, model.d_input_features)
    y_pred = model(x)
    assert y_pred.shape == (2, model.d_seq_out, model.d_target_features)


def test_moderntcn_forward_shape_target_indices():
    kwargs = _base_kwargs()
    kwargs["target_indices"] = (0, 2)
    model = ModernTCN(**kwargs)
    x = torch.randn(3, model.d_seq_in, model.d_input_features)
    y_pred = model(x)
    assert y_pred.shape == (3, model.d_seq_out, len(model.target_indices))


def test_moderntcn_encode_backbone_features_shape():
    model = ModernTCN(**_base_kwargs())
    x = torch.randn(4, model.d_seq_in, model.d_input_features)
    features = model.encode_backbone_features(x)
    assert features.shape == (
        4,
        model.d_input_features,
        model.d_model,
        model.patch_count,
    )


def test_moderntcn_patch_count_uses_floor_division():
    kwargs = _base_kwargs()
    kwargs["d_seq_in"] = 10
    kwargs["patch_size"] = 8
    kwargs["patch_stride"] = 4
    model = ModernTCN(**kwargs)
    x = torch.randn(2, model.d_seq_in, model.d_input_features)
    features = model.encode_backbone_features(x)
    assert model.patch_count == 2
    assert features.size(-1) == 2


def test_moderntcn_patch_count_mismatch_raises():
    model = ModernTCN(**_base_kwargs())
    model.patch_count += 1
    x = torch.randn(2, model.d_seq_in, model.d_input_features)
    with pytest.raises(ValueError, match="stem patch axis does not match"):
        model.encode_backbone_features(x)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"d_model": None}, "d_model must be provided"),
        ({"num_blocks": 0}, "num_blocks must be > 0"),
        ({"large_size": 4}, "large_size must be odd"),
        ({"small_size": 4}, "small_size must be odd"),
        ({"small_size": 9}, "small_size must be <= large_size"),
        ({"patch_stride": 9}, "patch_stride must be <= patch_size"),
        ({"patch_size": 17}, "patch_size must be <= d_seq_in"),
        ({"dropout": 1.0}, "0 <= dropout < 1"),
        ({"head_dropout": -0.1}, "0 <= head_dropout < 1"),
        ({"individual": "maybe"}, "invalid individual"),
    ],
)
def test_moderntcn_invalid_config_raises(override, message):
    kwargs = _base_kwargs()
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        ModernTCN(**kwargs)


@pytest.mark.parametrize(
    "unsupported_key",
    [
        "dims",
        "dw_dims",
        "small_kernel_merged",
        "kernel_size",
        "revin",
        "affine",
        "subtract_last",
        "freq",
    ],
)
def test_moderntcn_unsupported_upstream_kwargs_raise(unsupported_key):
    kwargs = _base_kwargs()
    kwargs[unsupported_key] = 1
    with pytest.raises(ValueError, match="unsupported upstream-specific argument"):
        ModernTCN(**kwargs)


def test_moderntcn_shared_step_contract():
    model = ModernTCN(**_base_kwargs())
    rng = torch.Generator().manual_seed(0)
    x = torch.randn(4, model.d_seq_in, model.d_input_features, generator=rng)
    y = torch.randn(4, model.d_seq_out, model.d_target_features, generator=rng)

    outputs = model._shared_step(x, y)
    assert set(outputs.keys()) == {"pred", "target", "loss"}
    assert outputs["pred"].shape == y.shape
    assert outputs["target"] is y
    assert outputs["loss"] is not None
    assert outputs["loss"].ndim == 0
    assert torch.isfinite(outputs["loss"])


def test_moderntcn_forward_shape_individual_head():
    kwargs = _base_kwargs()
    kwargs["individual"] = True
    model = ModernTCN(**kwargs)
    x = torch.randn(2, model.d_seq_in, model.d_input_features)
    y_pred = model(x)
    assert y_pred.shape == (2, model.d_seq_out, model.d_target_features)


def test_moderntcn_supports_repo_revin_path():
    kwargs = _base_kwargs()
    kwargs["target_indices"] = (0, 2)
    kwargs["use_revin"] = True
    model = ModernTCN(**kwargs)
    x = torch.randn(2, model.d_seq_in, model.d_input_features)
    y_pred = model(x)
    assert y_pred.shape == (2, model.d_seq_out, len(model.target_indices))


def test_moderntcn_load_from_checkpoint_roundtrip(tmp_path):
    model = ModernTCN(**_base_kwargs())
    checkpoint_path = tmp_path / "moderntcn.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": pl.__version__,
        },
        checkpoint_path,
    )

    loaded = ModernTCN.load_from_checkpoint(str(checkpoint_path))
    assert loaded.model_architecture == "ModernTCN"
    assert loaded.d_model == model.d_model
    assert loaded.patch_count == model.patch_count
    assert loaded.patch_size == model.patch_size
    assert loaded.patch_stride == model.patch_stride


def test_moderntcn_hparams_declared_in_baseline_grid():
    hparams = load_hparams()
    moderntcn = hparams.get("ModernTCN")
    assert moderntcn is not None
    required = {
        "d_model",
        "num_blocks",
        "large_size",
        "small_size",
        "ffn_ratio",
        "patch_size",
        "patch_stride",
        "dropout",
        "head_dropout",
        "individual",
        "lr",
    }
    assert required.issubset(set(moderntcn.keys()))
    assert moderntcn["dropout"]
    assert moderntcn["head_dropout"]
    assert moderntcn["individual"]
    assert all(0.0 <= value < 1.0 for value in moderntcn["dropout"])
    assert all(0.0 <= value < 1.0 for value in moderntcn["head_dropout"])
    assert all(isinstance(value, bool) for value in moderntcn["individual"])


def test_moderntcn_is_exported_from_models_package():
    assert getattr(models, "ModernTCN") is ModernTCN
    assert models.__all__.count("ModernTCN") == 1


def test_benchmark_scope_resolves_moderntcn_architecture_case_insensitive(monkeypatch):
    monkeypatch.setattr(
        "pipelines.selection.load_hparams",
        lambda: {"GRU": {"lr": [0.001]}, "ModernTCN": {"lr": [0.001]}},
    )
    args = SimpleNamespace(model=["moderntcn"], benchmark_architectures=["GRU"])
    assert resolve_requested_architectures(args) == ["ModernTCN"]


def test_benchmark_scope_includes_moderntcn(monkeypatch):
    monkeypatch.setattr(
        "pipelines.selection.load_hparams",
        lambda: {"GRU": {"lr": [0.001]}, "ModernTCN": {"lr": [0.001]}},
    )
    args = SimpleNamespace(model=None, benchmark_architectures=["GRU", "ModernTCN"])
    assert "ModernTCN" in resolve_requested_architectures(args)
