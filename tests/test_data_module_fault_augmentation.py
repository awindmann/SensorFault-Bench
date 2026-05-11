import pandas as pd
import pytest
import torch

from data.data_module import TSDataModule
from data.dataset import PerturbedDataset, TSDataset, TrainPerturbedDataset
from data.datasets.base import ResolvedDatasetSpec
from utils.parsing import build_perturbation_scenarios_signature


def _write_dataset(tmp_path, name: str, df: pd.DataFrame) -> str:
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return str(path)


def _continuous_spec(tmp_path) -> ResolvedDatasetSpec:
    path = _write_dataset(
        tmp_path,
        "continuous",
        pd.DataFrame(
            {
                "x": list(range(160)),
                "y": list(range(160, 320)),
                "z": list(range(320, 480)),
            }
        ),
    )
    return ResolvedDatasetSpec(
        key="continuous",
        path=path,
        input_channels=("x", "y", "z"),
        target_channels=("x",),
        target_alias=None,
        split_mode="temporal",
        description=None,
        batch_column=None,
        continuous_channels=("x", "y", "z"),
        discrete_channels=(),
    )


def _discrete_only_spec(tmp_path) -> ResolvedDatasetSpec:
    path = _write_dataset(
        tmp_path,
        "discrete_only",
        pd.DataFrame(
            {
                "d1": [idx % 2 for idx in range(160)],
                "d2": [idx % 3 for idx in range(160)],
            }
        ),
    )
    return ResolvedDatasetSpec(
        key="discrete_only",
        path=path,
        input_channels=("d1", "d2"),
        target_channels=("d1",),
        target_alias=None,
        split_mode="temporal",
        description=None,
        batch_column=None,
        continuous_channels=(),
        discrete_channels=("d1", "d2"),
    )


def _make_fault_aug_datamodule(
    tmp_path,
    *,
    dataset_spec: ResolvedDatasetSpec | None = None,
    train_fault_profiles=None,
    train_perturbation_profile: str = "holdout_simple",
    train_perturbation_scenarios=(
        "linear_drift",
        "scaling",
        "trimming_constant",
        "packet_loss",
    ),
    train_perturbation_probability: float = 1.0,
    train_perturbation_severity_max: float = 1.0,
    train_perturbation_channel_fraction_max: float = 0.1,
    train_noise_std: float = 0.0,
) -> TSDataModule:
    resolved_spec = _continuous_spec(tmp_path) if dataset_spec is None else dataset_spec
    resolved_train_fault_profiles = (
        {
            "holdout_simple": {
                "scenarios": [
                    "linear_drift",
                    "scaling",
                    "trimming_constant",
                    "packet_loss",
                ],
            },
            "holdout_varying": {
                "scenarios": [
                    "nonlinear_drift",
                    "trimming_varying",
                    "time_varying_scaling",
                    "packet_loss",
                ],
            },
            "holdout": {
                "scenarios": [
                    "linear_drift",
                    "nonlinear_drift",
                    "scaling",
                    "trimming_constant",
                    "trimming_varying",
                    "time_varying_scaling",
                    "packet_loss",
                ],
            },
        }
        if train_fault_profiles is None
        else train_fault_profiles
    )
    scenarios_signature = build_perturbation_scenarios_signature(
        train_perturbation_scenarios
    )
    return TSDataModule(
        dataset_spec=resolved_spec,
        input_len=8,
        target_len=2,
        n_train_samples=16,
        n_val_samples=8,
        n_test_samples=4,
        perturbation_channel_fraction_max=0.5,
        perturbation_scenarios=("drift", "noise"),
        train_split=0.6,
        val_split=0.2,
        purged_fraction=0.0,
        shuffle_batches_before_split=False,
        batch_size=4,
        num_workers=0,
        seed=42,
        strict_iid=False,
        train_noise_std=train_noise_std,
        train_fault_profiles=resolved_train_fault_profiles,
        train_perturbation_profile=train_perturbation_profile,
        train_perturbation_scenarios=train_perturbation_scenarios,
        train_perturbation_scenarios_signature=scenarios_signature,
        train_perturbation_probability=train_perturbation_probability,
        train_perturbation_severity_max=train_perturbation_severity_max,
        train_perturbation_channel_fraction_max=train_perturbation_channel_fraction_max,
        train_perturbation_generator=torch.Generator().manual_seed(123),
    )


def test_datamodule_wraps_only_train_split_for_fault_augmentation(tmp_path):
    dm = _make_fault_aug_datamodule(tmp_path)

    dm.setup()

    assert isinstance(dm.ds_train, TrainPerturbedDataset)
    assert isinstance(dm.ds_train.base_ds, TSDataset)
    assert isinstance(dm.ds_val, TSDataset)
    assert isinstance(dm.ds_test, PerturbedDataset)


def test_datamodule_rejects_fault_augmentation_without_resolved_scenarios(tmp_path):
    with pytest.raises(ValueError, match="train_perturbation_scenarios is required"):
        TSDataModule(
            dataset_spec=_continuous_spec(tmp_path),
            input_len=8,
            target_len=2,
            n_train_samples=16,
            n_val_samples=8,
            n_test_samples=4,
            perturbation_channel_fraction_max=0.5,
            perturbation_scenarios=("drift",),
            train_split=0.6,
            val_split=0.2,
            purged_fraction=0.0,
            shuffle_batches_before_split=False,
            batch_size=4,
            num_workers=0,
            seed=42,
            strict_iid=False,
            train_fault_profiles={
                "holdout_simple": {
                    "scenarios": [
                        "linear_drift",
                        "scaling",
                        "trimming_constant",
                        "packet_loss",
                    ],
                }
            },
            train_perturbation_profile="holdout_simple",
            train_perturbation_probability=0.5,
            train_perturbation_severity_max=0.5,
            train_perturbation_channel_fraction_max=0.1,
            train_perturbation_generator=torch.Generator().manual_seed(1),
        )


def test_datamodule_rejects_unknown_fault_augmentation_profile(tmp_path):
    with pytest.raises(ValueError, match="Unknown train_perturbation_profile"):
        _make_fault_aug_datamodule(
            tmp_path,
            train_perturbation_profile="unknown_profile",
        )


def test_datamodule_rejects_profile_scenario_mismatch(tmp_path):
    with pytest.raises(ValueError, match="does not match the configured"):
        _make_fault_aug_datamodule(
            tmp_path,
            train_perturbation_profile="holdout_simple",
            train_perturbation_scenarios=(
                "nonlinear_drift",
                "time_varying_scaling",
                "trimming_varying",
            ),
        )


def test_datamodule_rejects_rt_plus_fault_augmentation(tmp_path):
    with pytest.raises(ValueError, match="cannot be enabled simultaneously"):
        _make_fault_aug_datamodule(
            tmp_path,
            train_noise_std=0.1,
        )


def test_datamodule_rejects_empty_train_channel_pool_for_selected_subset(
    tmp_path,
):
    dm = _make_fault_aug_datamodule(
        tmp_path,
        dataset_spec=_discrete_only_spec(tmp_path),
        train_fault_profiles={"holdout_simple": {"scenarios": ["linear_drift"]}},
        train_perturbation_profile="holdout_simple",
        train_perturbation_scenarios=("linear_drift",),
    )

    with pytest.raises(ValueError, match="does not support configured perturbations"):
        dm.setup()


def test_datamodule_train_sampler_samples_native_severity_up_to_max(tmp_path):
    dm = _make_fault_aug_datamodule(
        tmp_path,
        train_perturbation_severity_max=0.4,
    )
    rng = torch.Generator().manual_seed(99)

    seen_names = set()
    max_severity = 0.0
    for _ in range(64):
        perturbation, severity = dm.train_pert_sampler(rng)
        seen_names.add(perturbation.name)
        max_severity = max(max_severity, severity)

    assert seen_names <= {
        "linear_drift",
        "scaling",
        "trimming_constant",
        "packet_loss",
    }
    assert max_severity <= 0.4
