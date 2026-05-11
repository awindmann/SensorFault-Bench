"""Tests for the adaptive robust loss integration."""

import math
from types import SimpleNamespace

import mlflow
import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, TensorDataset

from config_loader import load_defaults
from metrics.adaptive_robust_loss import (
    AdaptiveRobustLoss,
    _interpolate1d,
    affine_sigmoid,
    affine_softplus,
    inv_affine_sigmoid,
    barron_lossfun,
    log_base_partition_function,
    nll,
    validate_rloss_params,
)
from metrics.loss import build_loss, resolve_stateless_loss, RLOSS_PARAM_KEYS
from pipelines.training import optimizer_hparams_from_args


def _default_optimizer_args(**overrides):
    defaults = load_defaults()
    values = {
        "lr_scheduler": defaults["LR_SCHEDULER"],
        "optimizer": defaults["OPTIMIZER"],
        "optimizer_beta1": defaults["OPTIMIZER_BETA1"],
        "optimizer_beta2": defaults["OPTIMIZER_BETA2"],
        "optimizer_weight_decay": defaults["OPTIMIZER_WEIGHT_DECAY"],
        "optimizer_eps": defaults["OPTIMIZER_EPS"],
        "scheduler_type": defaults["SCHEDULER_TYPE"],
        "scheduler_factor": defaults["SCHEDULER_FACTOR"],
        "scheduler_patience": defaults["SCHEDULER_PATIENCE"],
        "scheduler_min_lr": defaults["SCHEDULER_MIN_LR"],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _optimizer_hparams(model_architecture: str) -> dict:
    return optimizer_hparams_from_args(
        _default_optimizer_args(),
        model_architecture=model_architecture,
    )


class _TinyAdaptiveDataModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        x = torch.randn(6, 10, 3)
        y = torch.randn(6, 5, 3)
        self._train = TensorDataset(x, y)
        self._val = TensorDataset(x[:3], y[:3])

    def train_dataloader(self):
        return DataLoader(self._train, batch_size=2)

    def val_dataloader(self):
        return DataLoader(self._val, batch_size=2)


class TestBarronLossfun:

    def test_alpha_0_cauchy(self):
        x = torch.randn(100)
        alpha = torch.zeros_like(x)
        scale = torch.ones_like(x)
        loss = barron_lossfun(x, alpha, scale)
        expected = torch.log1p(0.5 * (x / scale) ** 2)
        torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-4)

    def test_alpha_1_charbonnier(self):
        x = torch.randn(100)
        alpha = torch.tensor(1.0).expand_as(x)
        scale = torch.ones_like(x)
        loss = barron_lossfun(x, alpha, scale)
        expected = torch.sqrt((x / scale) ** 2 + 1.0) - 1.0
        torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-4)

    def test_alpha_2_l2(self):
        x = torch.randn(100)
        alpha = torch.full_like(x, 2.0)
        scale = torch.ones_like(x)
        loss = barron_lossfun(x, alpha, scale)
        expected = 0.5 * (x / scale) ** 2
        torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-4)

    def test_positive_loss(self):
        x = torch.randn(1000).clamp(min=0.01)
        alpha = torch.tensor(1.0).expand_as(x)
        scale = torch.ones_like(x)
        loss = barron_lossfun(x, alpha, scale)
        assert (loss >= 0).all()

    def test_scale_effect(self):
        x = torch.tensor([1.0, 1.0])
        alpha = torch.tensor([1.0, 1.0])
        scale_small = torch.tensor([0.1, 0.1])
        scale_large = torch.tensor([10.0, 10.0])
        loss_small = barron_lossfun(x, alpha, scale_small)
        loss_large = barron_lossfun(x, alpha, scale_large)
        assert (loss_small > loss_large).all()


class TestNLL:

    def test_finite_output(self):
        x = torch.randn(50)
        alpha = torch.tensor(1.0).expand_as(x)
        scale = torch.ones_like(x)
        result = nll(x, alpha, scale)
        assert torch.isfinite(result).all()

    def test_alpha_near_boundaries(self):
        x = torch.randn(10)
        for alpha_val in [0.001, 0.01, 0.5, 1.0, 1.5, 1.99, 1.999]:
            alpha = torch.tensor(alpha_val).expand_as(x)
            scale = torch.ones_like(x)
            result = nll(x, alpha, scale)
            assert torch.isfinite(result).all(), f"NaN/Inf at alpha={alpha_val}"


class TestLogBasePartitionFunction:

    def test_known_values(self):
        log_z_0 = log_base_partition_function(torch.tensor(0.0))
        log_z_2 = log_base_partition_function(torch.tensor(2.0))
        expected_0 = math.log(math.pi * math.sqrt(2))
        expected_2 = math.log(math.sqrt(2 * math.pi))
        assert abs(log_z_0.item() - expected_0) < 1e-4
        assert abs(log_z_2.item() - expected_2) < 1e-4

    def test_finite_range(self):
        alphas = torch.linspace(0.001, 3.0, 100)
        result = log_base_partition_function(alphas)
        assert torch.isfinite(result).all()


class TestCubicSplineHelper:

    @staticmethod
    def _reference_interpolate1d(
        x: torch.Tensor,
        values: torch.Tensor,
        tangents: torch.Tensor,
    ) -> torch.Tensor:
        last_knot = values.shape[0] - 1
        reference = []
        for x_value in x.reshape(-1).tolist():
            if x_value < 0.0:
                y_value = values[0].item() + tangents[0].item() * x_value
            elif x_value > last_knot:
                y_value = values[-1].item() + tangents[-1].item() * (
                    x_value - last_knot
                )
            else:
                left_idx = min(int(math.floor(x_value)), values.shape[0] - 2)
                tau = x_value - left_idx
                tau_sq = tau * tau
                tau_cu = tau_sq * tau
                h00 = 2.0 * tau_cu - 3.0 * tau_sq + 1.0
                h10 = tau_cu - 2.0 * tau_sq + tau
                h01 = -2.0 * tau_cu + 3.0 * tau_sq
                h11 = tau_cu - tau_sq
                y_value = (
                    values[left_idx].item() * h00
                    + tangents[left_idx].item() * h10
                    + values[left_idx + 1].item() * h01
                    + tangents[left_idx + 1].item() * h11
                )
            reference.append(y_value)
        return torch.tensor(reference, dtype=x.dtype).reshape_as(x)

    def test_interpolate1d_matches_piecewise_reference(self):
        values = torch.tensor([0.0, 1.0, 4.0, 9.0], dtype=torch.float32)
        tangents = torch.tensor([1.0, 2.0, 4.0, 6.0], dtype=torch.float32)
        x = torch.linspace(-1.5, 4.5, 61)

        expected = self._reference_interpolate1d(x, values, tangents)
        actual = _interpolate1d(x, values, tangents)

        torch.testing.assert_close(actual, expected)

    def test_interpolate1d_matches_knot_values_and_linear_extrapolation(self):
        values = torch.tensor([2.0, 5.0, 11.0], dtype=torch.float32)
        tangents = torch.tensor([3.0, -1.0, 4.0], dtype=torch.float32)
        x = torch.tensor([-2.0, 0.0, 1.0, 2.0, 3.5], dtype=torch.float32)

        actual = _interpolate1d(x, values, tangents)

        expected = torch.tensor(
            [
                values[0].item() + tangents[0].item() * -2.0,
                values[0].item(),
                values[1].item(),
                values[2].item(),
                values[2].item() + tangents[2].item() * 1.5,
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(actual, expected)


class TestBoundedTransforms:

    def test_affine_sigmoid_range(self):
        x = torch.randn(100)
        result = affine_sigmoid(x, lo=0.001, hi=1.999)
        assert (result > 0.001).all()
        assert (result < 1.999).all()

    def test_affine_sigmoid_midpoint(self):
        result = affine_sigmoid(torch.tensor(0.0), lo=0.0, hi=2.0)
        assert abs(result.item() - 1.0) < 1e-6

    def test_affine_sigmoid_roundtrip(self):
        val = 1.0
        latent = inv_affine_sigmoid(val, lo=0.001, hi=1.999)
        recovered = affine_sigmoid(latent, lo=0.001, hi=1.999)
        assert abs(recovered.item() - val) < 1e-5

    def test_affine_softplus_range(self):
        x = torch.randn(100)
        result = affine_softplus(x, lo=1e-5, ref=1.0)
        assert (result > 1e-5).all()

    def test_affine_softplus_zero_maps_to_ref(self):
        result = affine_softplus(torch.tensor(0.0), lo=1e-5, ref=1.0)
        assert abs(result.item() - 1.0) < 1e-5


class TestAdaptiveRobustLossModule:

    def _make_loss(self, num_dims=10, **kwargs):
        defaults = dict(
            alpha_lo=0.001, alpha_hi=1.999, alpha_init=1.0,
            scale_lo=1e-5, scale_init=1.0,
        )
        defaults.update(kwargs)
        return AdaptiveRobustLoss(num_dims=num_dims, **defaults)

    def test_forward_shape(self):
        loss_fn = self._make_loss(num_dims=15)
        pred = torch.randn(8, 5, 3)
        target = torch.randn(8, 5, 3)
        result = loss_fn(pred, target)
        assert result.shape == (8,)

    def test_forward_finite(self):
        loss_fn = self._make_loss(num_dims=10)
        pred = torch.randn(4, 5, 2)
        target = torch.randn(4, 5, 2)
        result = loss_fn(pred, target)
        assert torch.isfinite(result).all()

    def test_gradients_flow(self):
        loss_fn = self._make_loss(num_dims=6)
        pred = torch.randn(4, 3, 2, requires_grad=True)
        target = torch.randn(4, 3, 2)
        result = loss_fn(pred, target).mean()
        result.backward()
        assert pred.grad is not None
        assert loss_fn.latent_alpha.grad is not None
        assert loss_fn.latent_scale.grad is not None

    def test_initial_alpha(self):
        loss_fn = self._make_loss(num_dims=5, alpha_init=1.0)
        alpha = loss_fn.get_alpha()
        torch.testing.assert_close(
            alpha, torch.ones(1, 5), atol=1e-4, rtol=1e-4
        )

    def test_initial_scale(self):
        loss_fn = self._make_loss(num_dims=5, scale_init=1.0)
        scale = loss_fn.get_scale()
        torch.testing.assert_close(
            scale, torch.ones(1, 5), atol=1e-4, rtol=1e-4
        )

    def test_alpha_stays_bounded(self):
        loss_fn = self._make_loss(num_dims=5)
        with torch.no_grad():
            loss_fn.latent_alpha.fill_(100.0)
        alpha = loss_fn.get_alpha()
        assert (alpha > 0.0).all()
        assert (alpha < 2.0).all()

        with torch.no_grad():
            loss_fn.latent_alpha.fill_(-100.0)
        alpha = loss_fn.get_alpha()
        assert (alpha > 0.0).all()
        assert (alpha < 2.0).all()

    def test_scale_stays_positive(self):
        loss_fn = self._make_loss(num_dims=5)
        with torch.no_grad():
            loss_fn.latent_scale.fill_(-100.0)
        scale = loss_fn.get_scale()
        assert (scale > 0).all()
        assert (scale >= 1e-5).all()

    def test_num_dims_mismatch_raises(self):
        loss_fn = self._make_loss(num_dims=10)
        pred = torch.randn(4, 3, 2)
        target = torch.randn(4, 3, 2)
        with pytest.raises(ValueError, match="Expected 10 forecast dimensions"):
            loss_fn(pred, target)

    def test_is_nn_module(self):
        loss_fn = self._make_loss(num_dims=5)
        assert isinstance(loss_fn, nn.Module)
        params = list(loss_fn.parameters())
        assert len(params) == 2

    def test_mixed_precision_stays_fp32(self):
        loss_fn = self._make_loss(num_dims=6)
        pred = torch.randn(4, 3, 2, dtype=torch.float16)
        target = torch.randn(4, 3, 2, dtype=torch.float16)
        result = loss_fn(pred, target)
        assert torch.isfinite(result).all()
        assert result.dtype == torch.float16


class TestValidateRlossParams:

    def test_valid_defaults(self):
        validate_rloss_params(
            alpha_lo=0.001, alpha_hi=1.999, alpha_init=1.0,
            scale_lo=1e-5, scale_init=1.0, param_scope="per_horizon_feature",
        )

    def test_alpha_lo_zero_raises(self):
        with pytest.raises(ValueError, match="rloss_alpha_lo must be > 0"):
            validate_rloss_params(
                alpha_lo=0.0, alpha_hi=1.999, alpha_init=1.0,
                scale_lo=1e-5, scale_init=1.0, param_scope="per_horizon_feature",
            )

    def test_alpha_hi_2_raises(self):
        with pytest.raises(ValueError, match="rloss_alpha_hi must be < 2"):
            validate_rloss_params(
                alpha_lo=0.001, alpha_hi=2.0, alpha_init=1.0,
                scale_lo=1e-5, scale_init=1.0, param_scope="per_horizon_feature",
            )

    def test_alpha_init_out_of_range_raises(self):
        with pytest.raises(ValueError, match="rloss_alpha_init"):
            validate_rloss_params(
                alpha_lo=0.5, alpha_hi=1.5, alpha_init=1.6,
                scale_lo=1e-5, scale_init=1.0, param_scope="per_horizon_feature",
            )

    def test_invalid_scope_raises(self):
        with pytest.raises(ValueError, match="rloss_param_scope"):
            validate_rloss_params(
                alpha_lo=0.001, alpha_hi=1.999, alpha_init=1.0,
                scale_lo=1e-5, scale_init=1.0, param_scope="global",
            )

    def test_scale_lo_zero_raises(self):
        with pytest.raises(ValueError, match="rloss_scale_lo must be > 0"):
            validate_rloss_params(
                alpha_lo=0.001, alpha_hi=1.999, alpha_init=1.0,
                scale_lo=0.0, scale_init=1.0, param_scope="per_horizon_feature",
            )

    def test_scale_init_equal_scale_lo_raises(self):
        with pytest.raises(ValueError, match="rloss_scale_init.*must be > rloss_scale_lo"):
            validate_rloss_params(
                alpha_lo=0.001, alpha_hi=1.999, alpha_init=1.0,
                scale_lo=1e-5, scale_init=1e-5, param_scope="per_horizon_feature",
            )

    def test_alpha_lo_ge_alpha_hi_raises(self):
        with pytest.raises(ValueError, match="rloss_alpha_lo.*must be < rloss_alpha_hi"):
            validate_rloss_params(
                alpha_lo=1.5, alpha_hi=0.5, alpha_init=1.0,
                scale_lo=1e-5, scale_init=1.0, param_scope="per_horizon_feature",
            )


class TestBuildLoss:

    def test_stateless_loss_passthrough(self):
        loss_fn = build_loss("MSE")
        pred = torch.randn(4, 5, 2)
        target = torch.randn(4, 5, 2)
        result = loss_fn(pred, target)
        assert result.shape == (4,)

    def test_adaptive_loss_builds(self):
        loss_fn = build_loss(
            "AdaptiveRobustLoss",
            d_seq_out=5,
            d_target_features=2,
            rloss_alpha_lo=0.001,
            rloss_alpha_hi=1.999,
            rloss_alpha_init=1.0,
            rloss_scale_lo=1e-5,
            rloss_scale_init=1.0,
            rloss_param_scope="per_horizon_feature",
        )
        assert isinstance(loss_fn, AdaptiveRobustLoss)
        assert loss_fn.num_dims == 10

    def test_adaptive_loss_missing_params_raises(self):
        with pytest.raises(ValueError, match="Missing"):
            build_loss(
                "AdaptiveRobustLoss",
                d_seq_out=5,
                d_target_features=2,
                rloss_alpha_lo=0.001,
            )

    def test_rloss_params_with_non_adaptive_raises(self):
        with pytest.raises(ValueError, match="only valid with loss='AdaptiveRobustLoss'"):
            build_loss(
                "MSE",
                rloss_alpha_lo=0.001,
            )

    def test_adaptive_without_dimensions_raises(self):
        with pytest.raises(ValueError, match="d_seq_out"):
            build_loss(
                "AdaptiveRobustLoss",
                rloss_alpha_lo=0.001,
                rloss_alpha_hi=1.999,
                rloss_alpha_init=1.0,
                rloss_scale_lo=1e-5,
                rloss_scale_init=1.0,
                rloss_param_scope="per_horizon_feature",
            )

    def test_resolve_stateless_loss_rejects_adaptive(self):
        with pytest.raises(ValueError, match="build_loss"):
            resolve_stateless_loss("AdaptiveRobustLoss")


class TestCheckpointRoundtrip:

    def test_state_dict_roundtrip(self):
        loss_fn = AdaptiveRobustLoss(
            num_dims=10, alpha_lo=0.001, alpha_hi=1.999,
            alpha_init=1.0, scale_lo=1e-5, scale_init=1.0,
        )
        with torch.no_grad():
            loss_fn.latent_alpha.fill_(0.5)
            loss_fn.latent_scale.fill_(0.3)

        state = loss_fn.state_dict()
        loss_fn2 = AdaptiveRobustLoss(
            num_dims=10, alpha_lo=0.001, alpha_hi=1.999,
            alpha_init=1.0, scale_lo=1e-5, scale_init=1.0,
        )
        loss_fn2.load_state_dict(state)

        torch.testing.assert_close(loss_fn.get_alpha(), loss_fn2.get_alpha())
        torch.testing.assert_close(loss_fn.get_scale(), loss_fn2.get_scale())

    def test_loss_reproduces_after_restore(self):
        loss_fn = AdaptiveRobustLoss(
            num_dims=6, alpha_lo=0.001, alpha_hi=1.999,
            alpha_init=1.0, scale_lo=1e-5, scale_init=1.0,
        )
        with torch.no_grad():
            loss_fn.latent_alpha.normal_()
            loss_fn.latent_scale.normal_()

        pred = torch.randn(4, 3, 2)
        target = torch.randn(4, 3, 2)
        result1 = loss_fn(pred, target)

        state = loss_fn.state_dict()
        loss_fn2 = AdaptiveRobustLoss(
            num_dims=6, alpha_lo=0.001, alpha_hi=1.999,
            alpha_init=1.0, scale_lo=1e-5, scale_init=1.0,
        )
        loss_fn2.load_state_dict(state)
        result2 = loss_fn2(pred, target)

        torch.testing.assert_close(result1, result2)


class TestRecipeSpec:

    def test_recipe_loads(self):
        from pipelines.specs import PipelineSpec
        spec = PipelineSpec.from_yaml("configs/pipelines/adaptive_robust_loss.yaml")
        assert spec.pipeline_method == "adaptive_robust_loss"
        assert spec.pipeline_kind == "train"
        assert spec.model_hparams_mode == "inherit_baseline"
        assert spec.pipeline_id.startswith("arl_alo{rloss_alpha_lo}_ainit")

    def test_recipe_exposes_adaptive_loss_tuning_knobs(self):
        with open("configs/pipelines/adaptive_robust_loss.yaml", "r") as f:
            recipe = yaml.safe_load(f)
        assert recipe["pipeline_id"].startswith("arl_alo{rloss_alpha_lo}")
        for key in (
            "rloss_alpha_lo",
            "rloss_alpha_init",
            "rloss_scale_lo",
            "rloss_scale_init",
        ):
            entry = recipe["recipe_params"][key]
            assert "default" in entry
            assert isinstance(entry.get("grid"), list)
            assert entry["grid"]
            assert entry["default"] in entry["grid"]

    def test_recipe_has_all_rloss_params(self):
        from pipelines.specs import PipelineSpec
        spec = PipelineSpec.from_yaml("configs/pipelines/adaptive_robust_loss.yaml")
        for key in RLOSS_PARAM_KEYS:
            assert key in spec.recipe_params, f"Missing recipe param: {key}"

    def test_recipe_model_kwargs_includes_loss(self):
        from pipelines.specs import PipelineSpec
        spec = PipelineSpec.from_yaml("configs/pipelines/adaptive_robust_loss.yaml")
        assert spec.model_kwargs.get("loss") == "AdaptiveRobustLoss"

    def test_runner_registry_includes_method(self):
        from pipelines.runner import PIPELINE_RECIPE_PATHS_BY_METHOD
        assert "adaptive_robust_loss" in PIPELINE_RECIPE_PATHS_BY_METHOD

    def test_benchmark_recipe_order(self):
        from config_loader import load_defaults
        assert "adaptive_robust_loss" in load_defaults()["BENCHMARK_METHODS"]

    def test_selection_recipe_order(self):
        from config_loader import load_defaults
        from pipelines.selection import load_benchmark_recipe_specs_for_scope
        methods = [
            spec.pipeline_method
            for spec in load_benchmark_recipe_specs_for_scope(load_defaults())
        ]
        assert "adaptive_robust_loss" in methods

    @pytest.mark.parametrize("architecture", ["Chronos2", "SeasonalNaive"])
    def test_recipe_rejects_unsupported_benchmark_architectures(self, architecture):
        from pipelines.runner import scope_policy_skip_reason_for_spec
        from pipelines.specs import PipelineSpec

        spec = PipelineSpec.from_yaml("configs/pipelines/adaptive_robust_loss.yaml")
        assert (
            scope_policy_skip_reason_for_spec(spec, architecture)
            == "unsupported_benchmark_method_architecture"
        )


class TestModelBuildLossFn:

    def test_benchmark_model_with_adaptive_loss(self):
        from models.dlinear import DLinear
        model = DLinear(
            loss="AdaptiveRobustLoss",
            d_input_features=3,
            d_target_features=3,
            d_seq_in=10,
            d_seq_out=5,
            lr=0.001,
            rloss_alpha_lo=0.001,
            rloss_alpha_hi=1.999,
            rloss_alpha_init=1.0,
            rloss_scale_lo=1e-5,
            rloss_scale_init=1.0,
            rloss_param_scope="per_horizon_feature",
            **_optimizer_hparams("DLinear"),
        )
        assert isinstance(model.loss_fn, AdaptiveRobustLoss)
        assert model.loss_fn.num_dims == 15

        child_names = [name for name, _ in model.named_modules()]
        assert "loss_fn" in child_names

    def test_benchmark_model_with_mse_unchanged(self):
        from models.dlinear import DLinear
        model = DLinear(
            loss="MSE",
            d_input_features=3,
            d_target_features=3,
            d_seq_in=10,
            d_seq_out=5,
        )
        assert not isinstance(model.loss_fn, nn.Module)
        assert callable(model.loss_fn)

    def test_benchmark_model_rloss_params_with_mse_raises(self):
        from models.dlinear import DLinear
        with pytest.raises(ValueError, match="only valid with loss='AdaptiveRobustLoss'"):
            DLinear(
                loss="MSE",
                d_input_features=3,
                d_target_features=3,
                d_seq_in=10,
                d_seq_out=5,
                rloss_alpha_lo=0.001,
            )

    def test_training_step_with_adaptive_loss(self):
        from models.dlinear import DLinear
        model = DLinear(
            loss="AdaptiveRobustLoss",
            d_input_features=3,
            d_target_features=3,
            d_seq_in=10,
            d_seq_out=5,
            lr=0.001,
            rloss_alpha_lo=0.001,
            rloss_alpha_hi=1.999,
            rloss_alpha_init=1.0,
            rloss_scale_lo=1e-5,
            rloss_scale_init=1.0,
            rloss_param_scope="per_horizon_feature",
        )
        x = torch.randn(4, 10, 3)
        y = torch.randn(4, 5, 3)
        output = model._shared_step(x, y)
        assert torch.isfinite(output["loss"]).all()
        output["loss"].backward()
        assert model.loss_fn.latent_alpha.grad is not None

    def test_apply_loss_overrides_from_kwargs_uses_shared_builder(self):
        from models.dlinear import DLinear

        model = DLinear(
            loss="MSE",
            d_input_features=3,
            d_target_features=3,
            d_seq_in=10,
            d_seq_out=5,
        )

        model.apply_loss_overrides_from_kwargs(
            {
                "loss": "AdaptiveRobustLoss",
                "rloss_alpha_lo": 0.001,
                "rloss_alpha_hi": 1.999,
                "rloss_alpha_init": 1.0,
                "rloss_scale_lo": 1e-5,
                "rloss_scale_init": 1.0,
                "rloss_param_scope": "per_horizon_feature",
            }
        )

        assert isinstance(model.loss_fn, AdaptiveRobustLoss)
        assert model.hparams.loss == "AdaptiveRobustLoss"


class TestIntegrationLogging:

    def test_fit_logs_best_val_loss_and_adaptive_artifact(self, tmp_path):
        from models.dlinear import DLinear
        from pipelines.training import _LogAdaptiveLossArtifact

        tracking_dir = tmp_path / "mlruns"
        tracking_uri = f"file:{tracking_dir}"
        mlflow.set_tracking_uri(tracking_uri)
        logger = MLFlowLogger(
            tracking_uri=tracking_uri,
            experiment_name="adaptive-rloss-fit-test",
        )
        model = DLinear(
            loss="AdaptiveRobustLoss",
            d_input_features=3,
            d_target_features=3,
            d_seq_in=10,
            d_seq_out=5,
            lr=0.001,
            rloss_alpha_lo=0.001,
            rloss_alpha_hi=1.999,
            rloss_alpha_init=1.0,
            rloss_scale_lo=1e-5,
            rloss_scale_init=1.0,
            rloss_param_scope="per_horizon_feature",
            **_optimizer_hparams("DLinear"),
        )
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            logger=logger,
            max_epochs=1,
            callbacks=[_LogAdaptiveLossArtifact()],
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
        )

        trainer.fit(model, datamodule=_TinyAdaptiveDataModule())

        client = mlflow.MlflowClient()
        run = client.get_run(logger.run_id)
        artifacts = client.list_artifacts(logger.run_id, "diagnostics")

        assert "best_val_loss" in run.data.metrics
        assert "rloss_alpha_mean" in run.data.metrics
        assert "rloss_scale_mean" in run.data.metrics
        assert any(
            info.path.endswith("adaptive_loss_params.json")
            for info in artifacts
        )


class TestFinetuneLossOverrides:

    def test_finetune_single_run_uses_shared_loss_override_builder(self, monkeypatch):
        from pipelines import training as training_module

        recorded: dict[str, object] = {}

        class _DummyModel:
            def __init__(self):
                self.hparams = SimpleNamespace(lr=0.1)

            def set_model_seed(self, seed):
                self.seed_model = seed

            def apply_loss_overrides_from_kwargs(self, model_kwargs):
                recorded["loss_kwargs"] = dict(model_kwargs)

        class _DummyModelClass:
            @classmethod
            def load_from_checkpoint(cls, checkpoint_path, **model_kwargs):
                recorded["checkpoint_path"] = checkpoint_path
                recorded["load_kwargs"] = dict(model_kwargs)
                return _DummyModel()

        monkeypatch.setattr(training_module.models, "GRU", _DummyModelClass)
        monkeypatch.setattr(
            training_module,
            "derive_component_seeds",
            lambda **_kwargs: {
                "data_seed": 11,
                "model_seed": 22,
                "eval_seed": 33,
            },
        )
        monkeypatch.setattr(training_module, "set_seed", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(training_module.pl, "seed_everything", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(training_module, "get_tracking_uri", lambda _logdir: "file:/tmp/mlruns")
        monkeypatch.setattr(training_module.mlflow, "set_tracking_uri", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(training_module.mlflow, "MlflowClient", lambda: SimpleNamespace())
        monkeypatch.setattr(training_module, "_set_mlflow_storage_env", lambda _args: None)
        monkeypatch.setattr(
            training_module,
            "_build_datamodule",
            lambda **_kwargs: SimpleNamespace(
                n_inputs=3,
                n_outputs=3,
                target_column_indices=(0, 1, 2),
            ),
        )
        monkeypatch.setattr(
            training_module,
            "download_best_checkpoint",
            lambda _client, _run_id, dst_path=None: "/tmp/backbone.ckpt",
        )
        monkeypatch.setattr(
            training_module,
            "spec_to_tags",
            lambda _dataset_spec, *, n_inputs, n_outputs: {},
        )
        monkeypatch.setattr(training_module, "_current_git_commit", lambda: "deadbeef")

        def _fake_fit_and_finalize(**kwargs):
            recorded["fit_model"] = kwargs["model"]
            recorded["fit_hparams"] = dict(kwargs["hparams_to_log"])
            return "run_123"

        monkeypatch.setattr(training_module, "_fit_and_finalize", _fake_fit_and_finalize)

        model_kwargs = {
            "loss": "AdaptiveRobustLoss",
            "rloss_alpha_lo": 0.001,
            "rloss_alpha_hi": 1.999,
            "rloss_alpha_init": 1.0,
            "rloss_scale_lo": 1e-5,
            "rloss_scale_init": 1.0,
            "rloss_param_scope": "per_horizon_feature",
        }
        run_id = training_module.finetune_single_run(
            model_architecture="GRU",
            backbone_run_id="backbone_1",
            hparams={},
            dataset_spec=SimpleNamespace(key="dummy_ds"),
            args=SimpleNamespace(
                seed=123,
                data_split_seed=7,
                logdir="/tmp/mlruns",
                shuffle_batches_before_split=False,
                max_epochs=1,
                **vars(_default_optimizer_args()),
            ),
            data_config_signature="sig",
            model_kwargs=model_kwargs,
            finetune_epochs=2,
            finetune_lr_factor=0.5,
        )

        assert run_id == "run_123"
        assert recorded["checkpoint_path"] == "/tmp/backbone.ckpt"
        assert recorded["load_kwargs"]["map_location"] == "cpu"
        assert {
            key: value for key, value in recorded["load_kwargs"].items() if key != "map_location"
        } == model_kwargs
        assert recorded["loss_kwargs"] == model_kwargs
        assert recorded["fit_hparams"]["lr"] == 0.05


class TestNumericalStability:

    def test_fp32_stable(self):
        loss_fn = AdaptiveRobustLoss(
            num_dims=50, alpha_lo=0.001, alpha_hi=1.999,
            alpha_init=1.0, scale_lo=1e-5, scale_init=1.0,
        )
        pred = torch.randn(32, 10, 5)
        target = torch.randn(32, 10, 5)
        for _ in range(10):
            result = loss_fn(pred, target)
            assert torch.isfinite(result).all()
            result.mean().backward()
            assert torch.isfinite(loss_fn.latent_alpha.grad).all()
            assert torch.isfinite(loss_fn.latent_scale.grad).all()
            loss_fn.zero_grad()

    def test_large_residuals_stable(self):
        loss_fn = AdaptiveRobustLoss(
            num_dims=10, alpha_lo=0.001, alpha_hi=1.999,
            alpha_init=1.0, scale_lo=1e-5, scale_init=1.0,
        )
        pred = torch.randn(4, 5, 2) * 1000
        target = torch.zeros(4, 5, 2)
        result = loss_fn(pred, target)
        assert torch.isfinite(result).all()

    def test_zero_residuals_stable(self):
        loss_fn = AdaptiveRobustLoss(
            num_dims=10, alpha_lo=0.001, alpha_hi=1.999,
            alpha_init=1.0, scale_lo=1e-5, scale_init=1.0,
        )
        pred = torch.zeros(4, 5, 2)
        target = torch.zeros(4, 5, 2)
        result = loss_fn(pred, target)
        assert torch.isfinite(result).all()
