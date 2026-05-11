from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.base_module import BaseLitModule
from models.components.revin import RevIN
from pipelines.runner import (
    PipelineRunner,
    load_pipeline_spec_for_method,
    scope_policy_skip_reason_for_spec,
)
from pipelines.specs import PipelineSpec


class _DummyRevINModule(BaseLitModule):
    def __init__(self, *, target_indices=(1, 3), **kwargs):
        super().__init__(
            d_input_features=4,
            d_target_features=2,
            d_seq_in=6,
            d_seq_out=3,
            target_indices=target_indices,
            **kwargs,
        )
        self.model_architecture = "DummyRevIN"

    def _shared_step(self, x, y):
        x_norm = self._revin_norm_inputs(x)
        y_pred_raw = x_norm[:, -self.d_seq_out :, :]
        y_pred = self.project_targets(y_pred_raw)
        y_pred = self._revin_denorm_targets(y_pred)
        loss = None
        if y is not None:
            loss = torch.mean((y_pred - y) ** 2)
        return {"pred": y_pred, "target": y, "loss": loss}


def test_revin_roundtrip_restores_input():
    layer = RevIN(num_features=5, eps=1e-5, affine=True)
    x = torch.randn(3, 8, 5)
    restored = layer(layer(x, mode="norm"), mode="denorm")
    assert torch.allclose(restored, x, atol=1e-5, rtol=1e-4)


def test_revin_subset_denorm_with_target_indices():
    layer = RevIN(num_features=5, eps=1e-5, affine=True)
    x = torch.randn(2, 7, 5)
    target_indices = (0, 2, 4)
    idx = torch.as_tensor(target_indices, dtype=torch.long)
    x_norm = layer(x, mode="norm")
    subset_norm = torch.index_select(x_norm, dim=-1, index=idx)
    subset_denorm = layer(subset_norm, mode="denorm", target_indices=target_indices)
    expected_subset = torch.index_select(x, dim=-1, index=idx)
    assert torch.allclose(subset_denorm, expected_subset, atol=1e-5, rtol=1e-4)


def test_revin_full_feature_denorm_respects_target_index_order():
    layer = RevIN(num_features=4, eps=1e-5, affine=True)
    x = torch.randn(2, 9, 4)
    target_indices = (2, 0, 3, 1)
    idx = torch.as_tensor(target_indices, dtype=torch.long)
    x_norm = layer(x, mode="norm")
    reordered = torch.index_select(x_norm, dim=-1, index=idx)
    denorm = layer(reordered, mode="denorm", target_indices=target_indices)
    expected = torch.index_select(x, dim=-1, index=idx)
    assert torch.allclose(denorm, expected, atol=1e-5, rtol=1e-4)


def test_revin_denorm_before_norm_raises():
    layer = RevIN(num_features=3)
    with pytest.raises(RuntimeError, match="before norm"):
        layer(torch.randn(2, 4, 3), mode="denorm")


def test_revin_rejects_invalid_eps():
    with pytest.raises(ValueError, match="eps must be > 0"):
        RevIN(num_features=3, eps=0.0)


def test_revin_rejects_subset_feature_mismatch():
    layer = RevIN(num_features=4, eps=1e-5, affine=False)
    x = torch.randn(2, 7, 4)
    x_norm = layer(x, mode="norm")
    bad_subset = x_norm[:, :, :2]
    with pytest.raises(ValueError, match="feature mismatch"):
        layer(bad_subset, mode="denorm", target_indices=(0, 1, 2))


def test_base_module_requires_target_indices_when_revin_enabled():
    with pytest.raises(ValueError, match="requires target_indices"):
        _DummyRevINModule(target_indices=None, use_revin=True)


@pytest.mark.parametrize(
    "target_indices, expected_message",
    [
        ((), "non-empty"),
        ((1, 1), "unique"),
        ((0, 9), "range"),
    ],
)
def test_base_module_validates_target_indices(target_indices, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        _DummyRevINModule(target_indices=target_indices)


def test_base_module_rejects_non_scalar_tensor_target_index():
    bad_index = torch.tensor([1, 2], dtype=torch.int64)
    with pytest.raises(ValueError, match="scalar values"):
        _DummyRevINModule(target_indices=(bad_index,))


def test_base_module_revin_helpers_noop_when_disabled():
    module = _DummyRevINModule(use_revin=False)
    x = torch.randn(2, module.d_seq_in, module.d_input_features)
    y_pred = torch.randn(2, module.d_seq_out, module.d_target_features)
    assert module._revin_norm_inputs(x) is x
    assert module._revin_denorm_targets(y_pred) is y_pred


def test_base_module_revin_rejects_invalid_denorm_flag():
    with pytest.raises(ValueError, match="revin_denorm must be a bool"):
        _DummyRevINModule(use_revin=True, revin_denorm="false")


def test_base_module_revin_denorm_noop_when_disabled():
    module = _DummyRevINModule(use_revin=True, revin_denorm=False)
    x = torch.randn(2, module.d_seq_in, module.d_input_features)
    x_norm = module._revin_norm_inputs(x)
    idx = torch.as_tensor(module.target_indices, dtype=torch.long)
    y_pred = torch.index_select(x_norm[:, -module.d_seq_out :, :], dim=-1, index=idx)
    assert torch.equal(module._revin_denorm_targets(y_pred), y_pred)


def test_base_module_revin_shared_step_stays_in_normalized_space_when_denorm_disabled():
    module = _DummyRevINModule(use_revin=True, revin_affine=False, revin_denorm=False)
    x = torch.tensor(
        [
            [
                [10.0, 12.0, 14.0, 16.0],
                [11.0, 15.0, 19.0, 23.0],
                [12.0, 18.0, 24.0, 30.0],
                [13.0, 21.0, 29.0, 37.0],
                [14.0, 24.0, 34.0, 44.0],
                [15.0, 27.0, 39.0, 51.0],
            ]
        ],
        dtype=torch.float32,
    )
    y = torch.zeros(1, module.d_seq_out, module.d_target_features)

    outputs = module._shared_step(x, y)
    idx = torch.as_tensor(module.target_indices, dtype=torch.long)
    expected_norm = torch.index_select(
        module._revin_norm_inputs(x)[:, -module.d_seq_out :, :],
        dim=-1,
        index=idx,
    )
    expected_raw = torch.index_select(
        x[:, -module.d_seq_out :, :],
        dim=-1,
        index=idx,
    )

    assert torch.allclose(outputs["pred"], expected_norm, atol=1e-5, rtol=1e-4)
    assert not torch.allclose(outputs["pred"], expected_raw, atol=1e-5, rtol=1e-4)


def test_base_module_revin_denorm_restores_projected_targets():
    module = _DummyRevINModule(use_revin=True, revin_affine=True, revin_eps=1e-5)
    x = torch.randn(2, module.d_seq_in, module.d_input_features)
    y = torch.randn(2, module.d_seq_out, module.d_target_features)
    outputs = module._shared_step(x, y)
    assert module.revin_denorm is True
    idx = torch.as_tensor(module.target_indices, dtype=torch.long)
    expected = torch.index_select(x[:, -module.d_seq_out :, :], dim=-1, index=idx)
    assert outputs["pred"].shape == expected.shape
    assert torch.allclose(outputs["pred"], expected, atol=1e-5, rtol=1e-4)


def test_revin_recipe_renders_default_model_kwargs():
    spec = load_pipeline_spec_for_method("revin")
    defaults = {
        key: entry["default"] for key, entry in spec.recipe_params.items()
    }
    model_kwargs = spec.render_kwargs(spec.model_kwargs, defaults)
    module = _DummyRevINModule(**model_kwargs)

    assert model_kwargs["use_revin"] is True
    assert module.use_revin is True
    assert module.revin_denorm is True
    assert module.revin_affine is True
    assert module.revin_eps == pytest.approx(1e-5)


def test_revin_pipeline_has_empty_scope_for_unsupported_architecture():
    spec = PipelineSpec.from_yaml(Path("configs/pipelines/revin.yaml"))
    runner = PipelineRunner(
        spec,
        SimpleNamespace(seed=123, _recipe_param_overrides={}),
    )
    scope = runner.expected_tuning_scope(
        client=object(),
        experiment_id="exp_1",
        dataset_spec=SimpleNamespace(key="dummy"),
        architecture="SeasonalNaive",
        data_config_signature="sig",
    )
    assert scope.reference_budget == 0
    assert scope.target_budget == 0


@pytest.mark.parametrize(
    "recipe_name",
    ["revin.yaml"],
)
def test_revin_family_skip_policy_matches_benchmark_scope(recipe_name):
    spec = PipelineSpec.from_yaml(Path("configs/pipelines") / recipe_name)
    assert (
        scope_policy_skip_reason_for_spec(spec, "PatchTST")
        == "unsupported_benchmark_method_architecture"
    )
