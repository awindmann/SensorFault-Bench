import numpy as np
import pandas as pd
import pytest
import torch

from pathlib import Path

from data.data_module import TSDataModule, validate_train_noise_config
from data.dataset import NoisyDataset, TSDataset
from data.datasets.base import ResolvedDatasetSpec
from pipelines.specs import PipelineSpec


class DummyDataset:
    def __init__(
        self,
        x,
        y,
        input_columns,
        target_columns=None,
        continuous_channels=None,
        discrete_channels=None,
    ):
        self._x = np.asarray(x, dtype=np.float32)
        self._y = np.asarray(y, dtype=np.float32)
        self.input_columns = tuple(input_columns)
        self.target_columns = tuple(target_columns or input_columns)
        self.continuous_channels = tuple(continuous_channels or [])
        self.discrete_channels = tuple(discrete_channels or [])

    def __len__(self):
        return 1

    def __getitem__(self, _idx):
        return self._x.copy(), self._y.copy()


def test_noisy_dataset_uses_raw_space_relative_noise_with_signed_x():
    input_means = torch.tensor([10.0, -4.0], dtype=torch.float32)
    input_stds = torch.tensor([2.0, 4.0], dtype=torch.float32)
    x_raw = np.array([[6.0, 0.0], [8.0, -6.0]], dtype=np.float32)
    x = ((x_raw - input_means.numpy()) / input_stds.numpy()).astype(np.float32)
    y = np.zeros((1, 1), dtype=np.float32)
    base = DummyDataset(
        x,
        y,
        input_columns=["a", "b"],
        target_columns=["a", "b"],
        continuous_channels=["a", "b"],
    )
    noise_std = 0.1
    ds = NoisyDataset(
        base,
        noise_std=noise_std,
        noise_channels="all",
        input_means=input_means,
        input_stds=input_stds,
        noise_generator=torch.Generator().manual_seed(123),
    )

    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    means = input_means.view(1, -1)
    stds = input_stds.view(1, -1)
    eps = torch.randn(
        x_tensor.shape,
        dtype=x_tensor.dtype,
        device=x_tensor.device,
        generator=torch.Generator().manual_seed(123),
    )
    x_raw_tensor = x_tensor * stds + means
    expected = (x_raw_tensor * (1 + noise_std * eps) - means) / stds
    x_noisy, _ = ds[0]

    assert torch.allclose(x_noisy, expected)


def test_noisy_dataset_channel_mask_applies_only_to_continuous():
    input_means = torch.tensor([10.0, 100.0], dtype=torch.float32)
    input_stds = torch.tensor([2.0, 10.0], dtype=torch.float32)
    x_raw = np.array([[8.0, 90.0], [12.0, 110.0]], dtype=np.float32)
    x = ((x_raw - input_means.numpy()) / input_stds.numpy()).astype(np.float32)
    y = np.zeros((1, 1), dtype=np.float32)
    base = DummyDataset(
        x,
        y,
        input_columns=["cont", "disc"],
        target_columns=["cont"],
        continuous_channels=["cont"],
        discrete_channels=["disc"],
    )
    noise_std = 0.5
    ds = NoisyDataset(
        base,
        noise_std=noise_std,
        noise_channels="continuous",
        input_means=input_means,
        input_stds=input_stds,
        noise_generator=torch.Generator().manual_seed(999),
    )

    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    means = input_means.view(1, -1)
    stds = input_stds.view(1, -1)
    eps = torch.randn(
        x_tensor.shape,
        dtype=x_tensor.dtype,
        device=x_tensor.device,
        generator=torch.Generator().manual_seed(999),
    )
    x_raw_tensor = x_tensor * stds + means
    noisy = (x_raw_tensor * (1 + noise_std * eps) - means) / stds
    mask = torch.tensor([1.0, 0.0], dtype=torch.float32)
    expected = x_tensor * (1 - mask) + noisy * mask
    x_noisy, _ = ds[0]

    assert torch.allclose(x_noisy, expected)


def test_noisy_dataset_channel_mask_applies_only_to_target_inputs():
    input_means = torch.tensor([1.0, 10.0], dtype=torch.float32)
    input_stds = torch.tensor([1.0, 2.0], dtype=torch.float32)
    x_raw = np.array([[2.0, 8.0], [3.0, 12.0]], dtype=np.float32)
    x = ((x_raw - input_means.numpy()) / input_stds.numpy()).astype(np.float32)
    y = np.zeros((1, 1), dtype=np.float32)
    base = DummyDataset(
        x,
        y,
        input_columns=["aux", "target"],
        target_columns=["target"],
        continuous_channels=["aux", "target"],
    )
    noise_std = 0.25
    ds = NoisyDataset(
        base,
        noise_std=noise_std,
        noise_channels="target_only",
        input_means=input_means,
        input_stds=input_stds,
        noise_generator=torch.Generator().manual_seed(7),
    )

    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    means = input_means.view(1, -1)
    stds = input_stds.view(1, -1)
    eps = torch.randn(
        x_tensor.shape,
        dtype=x_tensor.dtype,
        device=x_tensor.device,
        generator=torch.Generator().manual_seed(7),
    )
    x_raw_tensor = x_tensor * stds + means
    noisy = (x_raw_tensor * (1 + noise_std * eps) - means) / stds
    mask = torch.tensor([0.0, 1.0], dtype=torch.float32)
    expected = x_tensor * (1 - mask) + noisy * mask
    x_noisy, _ = ds[0]

    assert torch.allclose(x_noisy, expected)


def test_noisy_dataset_zero_input_stays_zero():
    x = np.zeros((2, 2), dtype=np.float32)
    y = np.zeros((1, 1), dtype=np.float32)
    base = DummyDataset(
        x,
        y,
        input_columns=["a", "b"],
        target_columns=["a", "b"],
        continuous_channels=["a", "b"],
    )
    ds = NoisyDataset(
        base,
        noise_std=0.5,
        noise_channels="all",
        input_means=torch.zeros(2, dtype=torch.float32),
        input_stds=torch.ones(2, dtype=torch.float32),
        noise_generator=torch.Generator().manual_seed(0),
    )
    x_noisy, _ = ds[0]

    assert torch.allclose(x_noisy, torch.zeros_like(x_noisy))


def test_noisy_dataset_distribution_matches_relative_noise():
    x_val = 2.0
    noise_std = 0.1
    x = np.array([[x_val]], dtype=np.float32)
    y = np.zeros((1, 1), dtype=np.float32)
    base = DummyDataset(
        x,
        y,
        input_columns=["x"],
        target_columns=["x"],
        continuous_channels=["x"],
    )
    ds = NoisyDataset(
        base,
        noise_std=noise_std,
        noise_channels="all",
        input_means=torch.zeros(1, dtype=torch.float32),
        input_stds=torch.ones(1, dtype=torch.float32),
        noise_generator=torch.Generator().manual_seed(321),
    )
    samples = []
    for _ in range(2000):
        x_noisy, _ = ds[0]
        samples.append(float(x_noisy.item()))
    samples = np.asarray(samples, dtype=np.float32)

    assert abs(samples.mean() - x_val) < 0.05
    assert abs(samples.std(ddof=0) - abs(x_val) * noise_std) < 0.05


def test_render_kwargs_preserves_float_for_train_noise_std():
    spec = PipelineSpec.from_yaml(Path("configs/pipelines/randomized_training.yaml"))
    param_values = {
        "smoothing_noise_std": 0.1,
        "smoothing_noise_channels": "continuous",
    }
    rendered = spec.render_kwargs(spec.datamodule_kwargs, param_values)

    assert isinstance(rendered["train_noise_std"], float)
    assert rendered["train_noise_std"] == 0.1
    assert rendered["train_noise_channels"] == "continuous"


def _noise_datamodule(tmp_path, train_noise_std) -> TSDataModule:
    path = tmp_path / "toy.csv"
    pd.DataFrame(
        {"a": np.arange(160, dtype=float), "b": np.arange(160, dtype=float) + 1.0}
    ).to_csv(path, index=False)
    spec = ResolvedDatasetSpec(
        key="toy",
        path=str(path),
        input_channels=("a", "b"),
        target_channels=("a",),
        target_alias=None,
        split_mode="temporal",
        description=None,
        batch_column=None,
        continuous_channels=("a", "b"),
        discrete_channels=(),
    )
    return TSDataModule(
        dataset_spec=spec,
        input_len=8,
        target_len=2,
        n_train_samples=16,
        n_val_samples=8,
        n_test_samples=4,
        perturbation_channel_fraction_max=0.5,
        perturbation_scenarios=("drift",),
        perturbation_severity_distribution="uniform",
        train_split=0.6,
        val_split=0.2,
        purged_fraction=0.0,
        shuffle_batches_before_split=False,
        batch_size=4,
        num_workers=0,
        seed=42,
        strict_iid=False,
        train_noise_std=train_noise_std,
        train_noise_channels="continuous",
        train_noise_generator=torch.Generator().manual_seed(1),
    )


@pytest.mark.parametrize("noise_std", [float("nan"), float("inf"), "nan"])
def test_train_noise_std_rejects_non_finite_values(tmp_path, noise_std):
    base = DummyDataset(
        np.zeros((2, 1), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        input_columns=["x"],
        continuous_channels=["x"],
    )

    with pytest.raises(ValueError, match="train_noise_std must be finite"):
        validate_train_noise_config(noise_std, "continuous")
    with pytest.raises(ValueError, match="train_noise_std must be finite"):
        _noise_datamodule(tmp_path, noise_std)
    with pytest.raises(ValueError, match="noise_std must be finite"):
        NoisyDataset(base, noise_std=noise_std, noise_channels="all")


def test_datamodule_wraps_train_split_only_for_positive_noise(tmp_path):
    noisy = _noise_datamodule(tmp_path, 0.2)
    noisy.setup()
    clean = _noise_datamodule(tmp_path, 0.0)
    clean.setup()

    assert isinstance(noisy.ds_train, NoisyDataset)
    assert noisy.ds_train.noise_std == 0.2
    assert isinstance(noisy.ds_val, TSDataset)
    assert isinstance(clean.ds_train, TSDataset)
