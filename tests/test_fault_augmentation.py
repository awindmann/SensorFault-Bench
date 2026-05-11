from pathlib import Path

import numpy as np
import pytest
import torch

from data.data_module import _instantiate_perturbations
from data.dataset import TrainPerturbedDataset
from data.perturbations import PERTURBATION_REGISTRY
from pipelines.specs import PipelineSpec
from pipelines.training import _extract_train_augmentation_identity_tags
from utils.parsing import (
    build_perturbation_scenarios_signature,
    parse_train_fault_profiles,
)


class DummyDataset:
    def __init__(
        self,
        x,
        y,
        input_columns,
        continuous_channels=None,
        discrete_channels=None,
    ):
        self._x = np.asarray(x, dtype=np.float32)
        self._y = np.asarray(y, dtype=np.float32)
        self.input_columns = tuple(input_columns)
        if continuous_channels is None:
            self.continuous_channels = ()
        else:
            self.continuous_channels = tuple(continuous_channels)
        if discrete_channels is None:
            self.discrete_channels = ()
        else:
            self.discrete_channels = tuple(discrete_channels)

    def __len__(self):
        return 1

    def __getitem__(self, _idx):
        return self._x.copy(), self._y.copy()


class _AddSeverityPerturbation:
    name = "add_severity"
    idx = 0
    channel_scope = "all"

    def __call__(
        self,
        x,
        y,
        severity,
        _rng,
        _cont_channels,
        _disc_channels,
        *,
        channel_count_mode="severity",
        channel_count_value=None,
    ):
        assert channel_count_mode == "severity"
        assert channel_count_value == severity
        return x + severity, y, list(range(x.shape[1]))


class _RandomSeveritySampler:
    def __init__(self):
        self.perturbation = _AddSeverityPerturbation()

    def __call__(self, rng: torch.Generator):
        severity = float(torch.rand((), generator=rng).item())
        return self.perturbation, severity


class _FixedSeveritySampler:
    def __init__(self, severity: float):
        self.perturbation = _AddSeverityPerturbation()
        self.severity = float(severity)

    def __call__(self, _rng: torch.Generator):
        return self.perturbation, self.severity


class _UnregisteredAllScopePerturbation:
    name = "rogue_fault"
    channel_scope = "all"

    def __call__(
        self,
        x,
        y,
        severity,
        _rng,
        _cont_channels,
        _disc_channels,
        *,
        channel_count_mode="severity",
        channel_count_value=None,
    ):
        assert channel_count_mode == "severity"
        assert channel_count_value == severity
        return x + severity, y, list(range(x.shape[1]))


def test_train_perturbed_dataset_returns_standard_tuple_and_preserves_y():
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[5.0]], dtype=np.float32)
    base = DummyDataset(
        x,
        y,
        input_columns=["a", "b"],
        continuous_channels=["a", "b"],
    )
    ds = TrainPerturbedDataset(
        base,
        perturbation_sampler=_FixedSeveritySampler(1.0),
        perturbation_probability=1.0,
        perturbation_generator=torch.Generator().manual_seed(123),
    )

    sample = ds[0]

    assert isinstance(sample, tuple)
    assert len(sample) == 2
    x_out, y_out = sample
    assert torch.allclose(y_out, torch.as_tensor(y))


def test_train_perturbed_dataset_probability_one_always_perturbs():
    x = np.zeros((2, 2), dtype=np.float32)
    y = np.zeros((1, 1), dtype=np.float32)
    base = DummyDataset(
        x,
        y,
        input_columns=["a", "b"],
        continuous_channels=["a", "b"],
    )
    ds = TrainPerturbedDataset(
        base,
        perturbation_sampler=_FixedSeveritySampler(1.0),
        perturbation_probability=1.0,
        perturbation_generator=torch.Generator().manual_seed(0),
    )

    x_out, _ = ds[0]

    assert not torch.allclose(x_out, torch.as_tensor(x))


def test_train_perturbed_dataset_rejects_non_finite_probability():
    base = DummyDataset(
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        input_columns=["a", "b"],
        continuous_channels=["a", "b"],
    )

    with pytest.raises(ValueError, match="must be finite"):
        TrainPerturbedDataset(
            base,
            perturbation_sampler=_FixedSeveritySampler(1.0),
            perturbation_probability=float("nan"),
            perturbation_generator=torch.Generator().manual_seed(0),
        )


def test_train_perturbed_dataset_uses_fresh_randomness_for_repeated_access():
    x = np.zeros((2, 2), dtype=np.float32)
    y = np.zeros((1, 1), dtype=np.float32)
    base = DummyDataset(
        x,
        y,
        input_columns=["a", "b"],
        continuous_channels=["a", "b"],
    )
    ds = TrainPerturbedDataset(
        base,
        perturbation_sampler=_RandomSeveritySampler(),
        perturbation_probability=1.0,
        perturbation_generator=torch.Generator().manual_seed(7),
    )

    x_first, _ = ds[0]
    x_second, _ = ds[0]

    assert not torch.allclose(x_first, x_second)


def test_instantiate_perturbations_rejects_unregistered_all_scope_class():
    with pytest.raises(ValueError, match="is not registered"):
        _instantiate_perturbations(
            [_UnregisteredAllScopePerturbation],
            registry=PERTURBATION_REGISTRY,
            channel_fraction_max=0.1,
        )


def test_extract_train_augmentation_identity_tags_ignores_runtime_fields():
    tags = _extract_train_augmentation_identity_tags(
        {
            "train_perturbation_profile": "holdout_simple",
            "train_perturbation_scenarios": [
                "packet_loss",
                "linear_drift",
                "scaling",
                "trimming_constant",
            ],
            "train_perturbation_scenarios_signature": '["packet_loss","linear_drift","scaling","trimming_constant"]',
            "train_perturbation_probability": 1.0,
            "train_perturbation_generator": torch.Generator().manual_seed(0),
        }
    )

    assert tags == {
        "train_perturbation_profile": "holdout_simple",
        "train_perturbation_scenarios_signature": '["packet_loss","linear_drift","scaling","trimming_constant"]',
        "train_perturbation_probability": 1.0,
    }


def test_fault_augmentation_recipe_renders_profile_inputs():
    spec = PipelineSpec.from_yaml(Path("configs/pipelines/fault_augmentation.yaml"))
    profile_name = spec.recipe_params["train_perturbation_profile"]["default"]
    assert profile_name in spec.train_fault_profiles
    scenarios = spec.train_fault_profiles[profile_name]["scenarios"]
    assert scenarios
    rendered = spec.render_kwargs(
        spec.datamodule_kwargs,
        {
            "train_perturbation_profile": profile_name,
            "train_perturbation_scenarios": scenarios,
            "train_perturbation_scenarios_signature": build_perturbation_scenarios_signature(
                scenarios
            ),
            "train_perturbation_probability": spec.recipe_params[
                "train_perturbation_probability"
            ]["default"],
            "train_perturbation_severity_max": spec.recipe_params[
                "train_perturbation_severity_max"
            ]["default"],
            "train_perturbation_channel_fraction_max": spec.recipe_params[
                "train_perturbation_channel_fraction_max"
            ]["default"],
        },
    )

    assert rendered["train_perturbation_scenarios"] == scenarios
    assert rendered["train_perturbation_probability"] == spec.recipe_params[
        "train_perturbation_probability"
    ]["default"]


def test_fault_augmentation_holdout_profiles_include_packet_loss():
    spec = PipelineSpec.from_yaml(Path("configs/pipelines/fault_augmentation.yaml"))

    for profile_name in ("holdout_simple", "holdout_varying", "holdout"):
        assert "packet_loss" in spec.train_fault_profiles[profile_name]["scenarios"]


def test_fault_augmentation_profiles_are_registry_backed():
    faug_spec = PipelineSpec.from_yaml(Path("configs/pipelines/fault_augmentation.yaml"))
    registry_names = tuple(PERTURBATION_REGISTRY.keys())
    faug_profiles = parse_train_fault_profiles(
        faug_spec.train_fault_profiles,
        registry_names=registry_names,
    )

    assert all(
        scenario in registry_names
        for scenarios in faug_profiles.values()
        for scenario in scenarios
    )
