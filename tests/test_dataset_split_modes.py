from types import SimpleNamespace

import pandas as pd
import pytest

from data.data_module import TSDataModule
from data.datasets import DATASET_REGISTRY, DatasetSpec, ResolvedDatasetSpec, resolve_with_defaults, spec_to_tags
from pipelines.signatures import compute_data_config_signature


def _build_batched_dataframe(batch_ids: list[str], batch_length: int) -> pd.DataFrame:
    rows = []
    value = 0
    for batch_id in batch_ids:
        for _ in range(batch_length):
            rows.append(
                {
                    "x": value,
                    "y": value + 1000,
                    "d": value % 2,
                    "batch_id": batch_id,
                }
            )
            value += 1
    return pd.DataFrame(rows)


def _write_batched_dataset(tmp_path, *, batch_ids: list[str], batch_length: int) -> str:
    df = _build_batched_dataframe(batch_ids=batch_ids, batch_length=batch_length)
    path = tmp_path / "batched.csv"
    df.to_csv(path, index=False)
    return str(path)


def _make_batched_spec(path: str, *, split_mode: str) -> ResolvedDatasetSpec:
    return ResolvedDatasetSpec(
        key="toy_batched",
        path=path,
        input_channels=("x", "y", "d"),
        target_channels=("x",),
        target_alias=None,
        split_mode=split_mode,
        description=None,
        batch_column="batch_id",
    )


def _make_datamodule(
    path: str,
    *,
    split_mode: str,
    shuffle_batches_before_split: bool = False,
    purged_fraction: float = 0.0,
) -> TSDataModule:
    return TSDataModule(
        dataset_spec=_make_batched_spec(path, split_mode=split_mode),
        input_len=4,
        target_len=2,
        n_train_samples=None,
        n_val_samples=None,
        n_test_samples=None,
        perturbation_channel_fraction_max=0.5,
        perturbation_scenarios=("drift",),
        train_split=0.6,
        val_split=0.2,
        purged_fraction=purged_fraction,
        shuffle_batches_before_split=shuffle_batches_before_split,
        batch_size=4,
        num_workers=0,
        seed=42,
        strict_iid=False,
    )


def _unwrap_dataset(dataset):
    """Drill through PerturbedDataset/NoisyDataset wrappers to the underlying TSDataset."""
    while hasattr(dataset, "base_ds"):
        dataset = dataset.base_ds
    return dataset


def _assert_windows_respect_segments(dataset) -> None:
    ds = _unwrap_dataset(dataset)
    total_seq_len = ds.total_seq_len
    for start in ds.sample_idxs.tolist():
        assert any(
            seg.start <= start and start + total_seq_len <= seg.end
            for seg in ds.segments
        )


def _make_signature_args(*, shuffle_batches_before_split: bool, purged_fraction: float) -> SimpleNamespace:
    return SimpleNamespace(
        input_len=4,
        target_len=2,
        train_split=0.6,
        val_split=0.2,
        purged_fraction=purged_fraction,
        shuffle_batches_before_split=shuffle_batches_before_split,
        strict_iid=False,
        n_train_samples=None,
        n_val_samples=None,
        n_test_samples=None,
        seed=42,
        data_split_seed=None,
    )


def test_dataset_spec_requires_consistent_split_mode():
    with pytest.raises(ValueError, match="without batch_column"):
        DatasetSpec(key="demo", path="demo.csv", split_mode="within_batches")
    with pytest.raises(ValueError, match="with batch_column"):
        DatasetSpec(
            key="demo",
            path="demo.csv",
            split_mode="temporal",
            batch_column="batch_id",
        )


def test_resolved_dataset_spec_requires_consistent_split_mode():
    with pytest.raises(ValueError, match="without batch_column"):
        ResolvedDatasetSpec(
            key="demo",
            path="demo.csv",
            input_channels=("x",),
            target_channels=("x",),
            target_alias=None,
            split_mode="within_batches",
            description=None,
            batch_column=None,
        )
    with pytest.raises(ValueError, match="with batch_column"):
        ResolvedDatasetSpec(
            key="demo",
            path="demo.csv",
            input_channels=("x",),
            target_channels=("x",),
            target_alias=None,
            split_mode="temporal",
            description=None,
            batch_column="batch_id",
        )


def test_dataset_spec_rejects_incomplete_channel_partition():
    with pytest.raises(
        ValueError,
        match="must provide continuous_channels and discrete_channels together",
    ):
        DatasetSpec(
            key="demo",
            path="demo.csv",
            split_mode="temporal",
            input_channels=("x", "d"),
            continuous_channels=("x",),
        )


def test_resolved_dataset_spec_rejects_channel_partition_mismatch():
    with pytest.raises(ValueError, match="must partition input_channels exactly"):
        ResolvedDatasetSpec(
            key="demo",
            path="demo.csv",
            input_channels=("x", "d"),
            target_channels=("x",),
            target_alias=None,
            split_mode="temporal",
            description=None,
            batch_column=None,
            continuous_channels=("x",),
            discrete_channels=(),
        )


def test_registered_dataset_split_modes_are_explicit():
    assert DATASET_REGISTRY.get("BeijingAir_Tiantan").split_mode == "temporal"
    assert DATASET_REGISTRY.get("Penmanshiel_Hourly_WT08").split_mode == "temporal"
    assert (
        resolve_with_defaults("BeijingAir_Tiantan", data_root="data/processed")[0].split_mode
        == "temporal"
    )
    assert (
        resolve_with_defaults("Penmanshiel_Hourly_WT08", data_root="data/processed")[0].split_mode
        == "temporal"
    )


def test_internal_datasets_are_not_registered_in_benchmark_catalogue():
    for dataset_key in ("BeijingAir", "Penmanshiel_Hourly", "IndPenSim"):
        with pytest.raises(KeyError, match="not registered"):
            DATASET_REGISTRY.get(dataset_key)


def test_registered_dataset_channel_typing_is_explicit_for_hybrid_datasets():
    beijing = resolve_with_defaults("BeijingAir_Tiantan", data_root="data/processed")[0]
    assert "PM2.5" in beijing.input_channels
    assert beijing.target_channels == ("PM2.5",)
    assert "PM2.5" in beijing.continuous_channels
    assert beijing.discrete_channels == ("wd",)
    assert "wd" not in beijing.continuous_channels


def test_spec_to_tags_includes_split_mode():
    spec = ResolvedDatasetSpec(
        key="demo",
        path="demo.csv",
        input_channels=("x",),
        target_channels=("x",),
        target_alias=None,
        split_mode="temporal",
        description=None,
        batch_column=None,
    )
    tags = spec_to_tags(spec, n_inputs=1, n_outputs=1)
    assert tags["split_mode"] == "temporal"


def test_within_batches_split_keeps_every_batch_in_every_split(tmp_path):
    path = _write_batched_dataset(
        tmp_path,
        batch_ids=["batch_a", "batch_b", "batch_c"],
        batch_length=30,
    )
    dm = _make_datamodule(path, split_mode="within_batches")
    dm.setup()

    expected_ids = {"batch_a", "batch_b", "batch_c"}
    assert {seg.batch_id for seg in _unwrap_dataset(dm.ds_train).segments} == expected_ids
    assert {seg.batch_id for seg in _unwrap_dataset(dm.ds_val).segments} == expected_ids
    assert {seg.batch_id for seg in _unwrap_dataset(dm.ds_test).segments} == expected_ids

    _assert_windows_respect_segments(dm.ds_train)
    _assert_windows_respect_segments(dm.ds_val)
    _assert_windows_respect_segments(dm.ds_test)


def test_get_split_frames_returns_raw_prestandardized_splits(tmp_path):
    path = _write_batched_dataset(
        tmp_path,
        batch_ids=["batch_a", "batch_b", "batch_c"],
        batch_length=30,
    )
    dm = _make_datamodule(path, split_mode="within_batches")

    train_df, val_df, test_df, input_cols = dm.get_split_frames()

    assert tuple(input_cols) == ("x", "y", "d")
    assert not train_df.empty
    assert not val_df.empty
    assert not test_df.empty
    assert float(train_df["x"].mean()) != pytest.approx(0.0)

    dm.setup()
    train_dataset = _unwrap_dataset(dm.ds_train)
    assert len(train_df) == len(train_dataset.df)
    assert float(train_dataset.df["x"].mean()) == pytest.approx(0.0, abs=1e-6)


def test_within_batches_rejects_batch_shuffle(tmp_path):
    path = _write_batched_dataset(
        tmp_path,
        batch_ids=["batch_a", "batch_b", "batch_c"],
        batch_length=30,
    )
    dm = _make_datamodule(
        path,
        split_mode="within_batches",
        shuffle_batches_before_split=True,
    )
    with pytest.raises(ValueError, match="shuffle_batches_before_split must be false"):
        dm.setup()


def test_batched_datasets_reject_noncontiguous_repeated_batch_ids(tmp_path):
    path = _write_batched_dataset(
        tmp_path,
        batch_ids=["batch_a", "batch_b", "batch_a"],
        batch_length=24,
    )
    dm = _make_datamodule(path, split_mode="within_batches")
    with pytest.raises(ValueError, match="non-contiguous repeated batch ids"):
        dm.setup()


def test_within_batches_purges_each_batch_independently(tmp_path):
    path = _write_batched_dataset(
        tmp_path,
        batch_ids=["batch_a", "batch_b"],
        batch_length=50,
    )
    dm = _make_datamodule(path, split_mode="within_batches", purged_fraction=0.25)
    dm.setup()

    assert [seg.length for seg in _unwrap_dataset(dm.ds_train).segments] == [30, 30]
    assert [seg.length for seg in _unwrap_dataset(dm.ds_val).segments] == [8, 8]
    assert [seg.length for seg in _unwrap_dataset(dm.ds_test).segments] == [8, 8]


def test_within_batches_purge_raises_when_it_removes_usable_windows(tmp_path):
    path = _write_batched_dataset(
        tmp_path,
        batch_ids=["batch_a", "batch_b"],
        batch_length=50,
    )
    dm = _make_datamodule(path, split_mode="within_batches", purged_fraction=0.5)
    with pytest.raises(ValueError, match="Purging removed usable validation windows"):
        dm.setup()


def test_across_batches_preserves_whole_batch_assignment(tmp_path):
    path = _write_batched_dataset(
        tmp_path,
        batch_ids=["batch_a", "batch_b", "batch_c", "batch_d"],
        batch_length=24,
    )
    dm = _make_datamodule(path, split_mode="across_batches")
    dm.setup()

    assert [seg.batch_id for seg in _unwrap_dataset(dm.ds_train).segments] == ["batch_a", "batch_b"]
    assert [seg.batch_id for seg in _unwrap_dataset(dm.ds_val).segments] == ["batch_c"]
    assert [seg.batch_id for seg in _unwrap_dataset(dm.ds_test).segments] == ["batch_d"]


def test_data_config_signature_includes_split_mode_and_relevant_policy_only():
    temporal_spec = SimpleNamespace(
        key="demo",
        input_channels=None,
        target_channels=None,
        split_mode="temporal",
    )
    across_spec = SimpleNamespace(
        key="demo",
        input_channels=None,
        target_channels=None,
        split_mode="across_batches",
    )

    temporal_args_a = _make_signature_args(
        shuffle_batches_before_split=True,
        purged_fraction=0.1,
    )
    temporal_args_b = _make_signature_args(
        shuffle_batches_before_split=False,
        purged_fraction=0.1,
    )
    assert (
        compute_data_config_signature(dataset_spec=temporal_spec, args=temporal_args_a)
        == compute_data_config_signature(dataset_spec=temporal_spec, args=temporal_args_b)
    )

    across_args_a = _make_signature_args(
        shuffle_batches_before_split=True,
        purged_fraction=0.1,
    )
    across_args_b = _make_signature_args(
        shuffle_batches_before_split=True,
        purged_fraction=0.5,
    )
    assert (
        compute_data_config_signature(dataset_spec=across_spec, args=across_args_a)
        == compute_data_config_signature(dataset_spec=across_spec, args=across_args_b)
    )

    assert (
        compute_data_config_signature(dataset_spec=temporal_spec, args=temporal_args_a)
        != compute_data_config_signature(dataset_spec=across_spec, args=across_args_a)
    )


def test_data_config_signature_keeps_historical_channel_set_contract():
    args = _make_signature_args(
        shuffle_batches_before_split=False,
        purged_fraction=0.0,
    )
    canonical_spec = SimpleNamespace(
        key="demo",
        input_channels=("a", "b"),
        target_channels=("a", "b"),
        split_mode="temporal",
    )
    reordered_inputs = SimpleNamespace(
        key="demo",
        input_channels=("b", "a"),
        target_channels=("a", "b"),
        split_mode="temporal",
    )
    reordered_targets = SimpleNamespace(
        key="demo",
        input_channels=("a", "b"),
        target_channels=("b", "a"),
        split_mode="temporal",
    )

    canonical_signature = compute_data_config_signature(
        dataset_spec=canonical_spec,
        args=args,
    )

    assert canonical_signature == compute_data_config_signature(
        dataset_spec=reordered_inputs,
        args=args,
    )
    assert canonical_signature == compute_data_config_signature(
        dataset_spec=reordered_targets,
        args=args,
    )
