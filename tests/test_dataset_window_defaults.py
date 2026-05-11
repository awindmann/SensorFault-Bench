from types import SimpleNamespace

import pytest

from config_loader import load_dataset_windows
from utils.parsing import (
    parse_dataset_window_defaults,
    resolve_dataset_window_args,
)


def test_parse_dataset_window_defaults_rejects_missing_required_keys():
    with pytest.raises(ValueError, match="missing required key"):
        parse_dataset_window_defaults(
            {"ETTh1": {"input_len": 168}},
        )


def test_parse_dataset_window_defaults_rejects_non_positive_lengths():
    with pytest.raises(ValueError, match="ETTh1.input_len must be > 0"):
        parse_dataset_window_defaults(
            {"ETTh1": {"input_len": 0, "target_len": 24}},
        )


def test_load_dataset_windows_requires_benchmark_entries(tmp_path):
    path = tmp_path / "dataset_windows.yaml"
    path.write_text(
        "ETTh1:\n  input_len: 168\n  target_len: 24\n",
        encoding="utf-8",
    )
    defaults = {
        "DATA_FILES": ["ETTh1", "Penmanshiel_Hourly_WT08", "BeijingAir_Tiantan"],
        "DATA_ROOT": "data/processed",
    }

    with pytest.raises(ValueError, match="Penmanshiel_Hourly_WT08, BeijingAir_Tiantan"):
        load_dataset_windows(path, defaults=defaults)


def test_shipped_dataset_windows_define_positive_batch_sizes():
    defaults = {
        "DATA_FILES": ["ETTh1", "traffic", "Penmanshiel_Hourly_WT08", "BeijingAir_Tiantan"],
        "DATA_ROOT": "data/processed",
    }
    windows = load_dataset_windows(defaults=defaults)

    assert set(windows) == set(defaults["DATA_FILES"])
    for dataset, window in windows.items():
        assert window["input_len"] > 0, dataset
        assert window["target_len"] > 0, dataset
        assert window["batch_size"] > 0, dataset


def test_resolve_dataset_window_args_uses_dataset_defaults_when_cli_not_explicit():
    args = SimpleNamespace(input_len=90, target_len=30, batch_size=128, other="keep")
    dataset_spec = SimpleNamespace(key="ETTh1")
    resolved = resolve_dataset_window_args(
        args,
        dataset_spec=dataset_spec,
        dataset_window_defaults={
            "ETTh1": {"input_len": 168, "target_len": 24, "batch_size": 16}
        },
        explicit_arg_overrides={},
    )

    assert resolved is not args
    assert resolved.input_len == 168
    assert resolved.target_len == 24
    assert resolved.batch_size == 16
    assert resolved.other == "keep"
    assert args.input_len == 90
    assert args.target_len == 30
    assert args.batch_size == 128


def test_resolve_dataset_window_args_requires_explicit_override_metadata():
    args = SimpleNamespace(input_len=90, target_len=30)
    dataset_spec = SimpleNamespace(key="ETTh1")

    with pytest.raises(ValueError, match="explicit_arg_overrides is required"):
        resolve_dataset_window_args(
            args,
            dataset_spec=dataset_spec,
            dataset_window_defaults={"ETTh1": {"input_len": 168, "target_len": 24}},
        )


def test_resolve_dataset_window_args_explicit_pair_wins_over_dataset_defaults():
    args = SimpleNamespace(input_len=90, target_len=30)
    dataset_spec = SimpleNamespace(key="ETTh1")
    resolved = resolve_dataset_window_args(
        args,
        dataset_spec=dataset_spec,
        dataset_window_defaults={
            "ETTh1": {"input_len": 168, "target_len": 24, "batch_size": 16}
        },
        explicit_arg_overrides={"input_len": 48, "target_len": 12},
    )

    assert resolved.input_len == 48
    assert resolved.target_len == 12
    assert resolved.batch_size == 16


def test_resolve_dataset_window_args_explicit_batch_size_wins_over_dataset_default():
    args = SimpleNamespace(input_len=90, target_len=30, batch_size=128)
    dataset_spec = SimpleNamespace(key="ETTh1")
    resolved = resolve_dataset_window_args(
        args,
        dataset_spec=dataset_spec,
        dataset_window_defaults={
            "ETTh1": {"input_len": 168, "target_len": 24, "batch_size": 16}
        },
        explicit_arg_overrides={"batch_size": 4},
    )

    assert resolved.input_len == 168
    assert resolved.target_len == 24
    assert resolved.batch_size == 4


def test_resolve_dataset_window_args_rejects_partial_cli_override():
    args = SimpleNamespace(input_len=90, target_len=30)
    dataset_spec = SimpleNamespace(key="ETTh1")

    with pytest.raises(ValueError, match="provided together"):
        resolve_dataset_window_args(
            args,
            dataset_spec=dataset_spec,
            dataset_window_defaults={"ETTh1": {"input_len": 168, "target_len": 24}},
            explicit_arg_overrides={"input_len": 48},
        )


def test_resolve_dataset_window_args_allows_explicit_pair_for_unknown_dataset():
    args = SimpleNamespace(input_len=90, target_len=30)
    dataset_spec = SimpleNamespace(key="toy_dataset")
    resolved = resolve_dataset_window_args(
        args,
        dataset_spec=dataset_spec,
        dataset_window_defaults={"ETTh1": {"input_len": 168, "target_len": 24}},
        explicit_arg_overrides={"input_len": 48, "target_len": 12},
    )

    assert resolved.input_len == 48
    assert resolved.target_len == 12


def test_resolve_dataset_window_args_requires_yaml_for_unknown_dataset_without_override():
    args = SimpleNamespace(input_len=90, target_len=30)
    dataset_spec = SimpleNamespace(key="toy_dataset")

    with pytest.raises(ValueError, match="configs/dataset_windows.yaml"):
        resolve_dataset_window_args(
            args,
            dataset_spec=dataset_spec,
            dataset_window_defaults={"ETTh1": {"input_len": 168, "target_len": 24}},
            explicit_arg_overrides={},
        )
