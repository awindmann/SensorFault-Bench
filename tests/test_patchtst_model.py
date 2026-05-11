import pytest
import torch

from config_loader import load_hparams
from models.patchtst import PatchTST


def _base_kwargs() -> dict:
    return {
        "d_input_features": 3,
        "d_target_features": 2,
        "d_seq_in": 16,
        "d_seq_out": 5,
        "target_indices": None,
        "d_model": 16,
        "d_ff": 32,
        "n_layers_enc": 2,
        "n_heads": 4,
        "patch_len": 8,
        "stride": 4,
        "dropout": 0.1,
        "factor": 1,
        "activation": "gelu",
    }


def test_patchtst_forward_shape_full_targets():
    model = PatchTST(**_base_kwargs())
    x = torch.randn(2, model.d_seq_in, model.d_input_features)
    y_pred = model(x)
    assert y_pred.shape == (2, model.d_seq_out, model.d_target_features)


def test_patchtst_forward_shape_target_indices():
    kwargs = _base_kwargs()
    kwargs["target_indices"] = (0, 2)
    model = PatchTST(**kwargs)
    x = torch.randn(3, model.d_seq_in, model.d_input_features)
    y_pred = model(x)
    assert y_pred.shape == (3, model.d_seq_out, len(model.target_indices))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"d_model": None}, "d_model must be provided"),
        ({"d_model": 0}, "d_model must be > 0"),
        ({"d_ff": 0}, "d_ff must be > 0"),
        ({"n_layers_enc": 0}, "n_layers_enc must be > 0"),
        ({"n_heads": 0}, "n_heads must be > 0"),
        ({"patch_len": 0}, "patch_len must be > 0"),
        ({"stride": 0}, "stride must be > 0"),
        ({"patch_len": 17}, "patch_len must be <= d_seq_in"),
        ({"dropout": -0.1}, "0 <= dropout < 1"),
        ({"dropout": 1.0}, "0 <= dropout < 1"),
        ({"activation": "swish"}, "Unsupported activation"),
        (
            {"d_model": 10, "n_heads": 4},
            "d_model must be divisible by n_heads",
        ),
    ],
)
def test_patchtst_invalid_config_raises(override, message):
    kwargs = _base_kwargs()
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        PatchTST(**kwargs)


def test_patchtst_shared_step_contract():
    model = PatchTST(**_base_kwargs())
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


def test_patchtst_hparams_declared_in_baseline_grid():
    hparams = load_hparams()
    patchtst = hparams.get("PatchTST")
    assert patchtst is not None
    required = {
        "d_model",
        "d_ff",
        "n_layers_enc",
        "n_heads",
        "patch_len",
        "stride",
        "dropout",
        "factor",
        "activation",
        "lr",
    }
    assert required.issubset(set(patchtst.keys()))
    assert patchtst["factor"]
    assert patchtst["activation"]
    assert all(isinstance(value, int) and value > 0 for value in patchtst["factor"])
    assert all(isinstance(value, str) and value for value in patchtst["activation"])


@pytest.mark.parametrize(
    ("d_seq_in", "patch_len", "stride", "expected_patch_count", "error_message"),
    [
        (16, 1, 16, 2, None),
        (16, 8, 4, 4, None),
        (16, 17, 4, None, "patch_len must be <= d_seq_in"),
    ],
)
def test_patchtst_patch_count_behavior(
    d_seq_in, patch_len, stride, expected_patch_count, error_message
):
    kwargs = _base_kwargs()
    kwargs["d_seq_in"] = d_seq_in
    kwargs["patch_len"] = patch_len
    kwargs["stride"] = stride

    if error_message is not None:
        with pytest.raises(ValueError, match=error_message):
            PatchTST(**kwargs)
        return

    model = PatchTST(**kwargs)
    assert model.patch_count == expected_patch_count


def test_patchtst_learned_pe_is_parameter():
    model = PatchTST(**_base_kwargs())
    w_pos = model.patch_embedding.W_pos
    assert isinstance(w_pos, torch.nn.Parameter)
    assert w_pos.shape == (model.patch_count, model.d_model)
