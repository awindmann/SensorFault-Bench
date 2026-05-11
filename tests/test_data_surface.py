import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from config_loader import build_parser, load_dataset_windows, load_defaults
from data.data_module import TSDataModule
from data.datasets import DATASET_REGISTRY, resolve_with_defaults, spec_to_tags
from pipelines.signatures import build_signature, compute_data_config_signature
from scripts.generate_dataset_checksums import (
    DATASET_CHECKSUM_BUNDLES,
    build_checksum_lines,
    build_manifest_text,
    default_data_root,
    sha256_file_bytes,
)
from scripts import benchmark_dataset_contracts
from scripts import generate_dataset_checksums
from scripts.benchmark_dataset_contracts import (
    BENCHMARK_DATASET_CONTRACTS,
    BEIJING_AIR_TIANTAN,
    ETTH1,
    TRAFFIC,
    add_common_args,
    run_contract_cli,
    validate_processed_file,
)
from scripts.preprocess_beijing_air_tiantan import prepare_from_raw as prepare_beijing
from scripts.preprocess_penmanshiel_hourly_wt08 import prepare_from_raw as prepare_penmanshiel
from scripts.preprocess_traffic import prepare_from_raw as prepare_traffic
from utils.parsing import normalize_data_root


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BENCHMARK_DATASETS = {
    "ETTh1": "ETTh1.csv",
    "traffic": "traffic.parquet",
    "BeijingAir_Tiantan": "beijing_air_tiantan.parquet",
    "Penmanshiel_Hourly_WT08": "penmanshiel_hourly_wt08.parquet",
}
PROVENANCE_NOTEBOOKS = {
    "beijing_air_tiantan_derivation.ipynb",
    "penmanshiel_hourly_wt08_derivation.ipynb",
}


def _signature_args(**overrides):
    values = {
        "input_len": 96,
        "target_len": 96,
        "train_split": 0.6,
        "val_split": 0.2,
        "strict_iid": False,
        "shuffle_batches_before_split": False,
        "purged_fraction": 0.0,
        "n_train_samples": None,
        "n_val_samples": None,
        "n_test_samples": None,
        "data_split_seed": None,
        "seed": 42,
        "data_root": "data/processed",
        "logdir": "runs",
        "minio_endpoint": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_defaults_are_benchmark_safe_and_expose_data_root_cli():
    defaults = load_defaults()

    assert defaults["DATA_ROOT"] == "data/processed"
    assert defaults["LOGDIR"] == "runs"
    assert defaults["MINIO_ENDPOINT"] == ""

    parser = build_parser(defaults)
    parsed = parser.parse_args(
        [
            "--data-root",
            "s3://example-bucket/benchmark-data",
            "--logdir",
            "http://mlflow.example.invalid",
            "--minio-endpoint",
            "http://object-store.example.invalid",
        ]
    )
    assert parsed.data_root == "s3://example-bucket/benchmark-data"
    assert parsed.logdir == "http://mlflow.example.invalid"
    assert parsed.minio_endpoint == "http://object-store.example.invalid"

    parsed_standard_s3 = parser.parse_args(
        [
            "--data-root",
            "s3://example-bucket/benchmark-data",
        ]
    )
    assert parsed_standard_s3.data_root == "s3://example-bucket/benchmark-data"
    assert parsed_standard_s3.minio_endpoint == ""

    with pytest.raises(SystemExit):
        parser.parse_args(["--data-files"])


def test_registered_dataset_paths_resolve_under_explicit_data_root():
    local_specs = resolve_with_defaults(
        list(EXPECTED_BENCHMARK_DATASETS),
        [],
        data_root="data/processed",
    )
    s3_specs = resolve_with_defaults(
        list(EXPECTED_BENCHMARK_DATASETS),
        [],
        data_root="s3://example-bucket/benchmark-data/",
    )

    assert {spec.key: spec.path for spec in local_specs} == {
        key: f"data/processed/{filename}"
        for key, filename in EXPECTED_BENCHMARK_DATASETS.items()
    }
    assert {spec.key: spec.path for spec in s3_specs} == {
        key: f"s3://example-bucket/benchmark-data/{filename}"
        for key, filename in EXPECTED_BENCHMARK_DATASETS.items()
    }
    with pytest.raises(ValueError, match="explicit s3:// URI"):
        normalize_data_root("s3:example-bucket/benchmark-data")

    with pytest.raises(ValueError, match="DATA_FILES must contain"):
        resolve_with_defaults([], [], data_root="data/processed")


def test_registry_specs_own_relative_filenames_without_private_roots():
    for key, filename in EXPECTED_BENCHMARK_DATASETS.items():
        spec = DATASET_REGISTRY.get(key)
        assert spec.path == filename
        assert not Path(spec.path).is_absolute()
        assert "://" not in spec.path


def test_benchmark_dataset_registry_and_window_defaults_are_limited_to_benchmark_scope():
    expected_keys = set(EXPECTED_BENCHMARK_DATASETS)
    defaults = load_defaults()

    assert set(DATASET_REGISTRY.keys()) == expected_keys
    assert set(load_dataset_windows(defaults=defaults)) == expected_keys


def test_data_root_and_storage_knobs_do_not_change_signature_or_run_identity():
    local_spec = resolve_with_defaults(
        "BeijingAir_Tiantan",
        [],
        data_root="data/processed",
    )[0]
    remote_spec = resolve_with_defaults(
        "BeijingAir_Tiantan",
        [],
        data_root="s3://example-bucket/benchmark-data",
    )[0]
    local_args = _signature_args()
    remote_args = _signature_args(
        data_root="s3://example-bucket/benchmark-data",
        logdir="http://mlflow.example.invalid",
        minio_endpoint="http://object-store.example.invalid",
    )

    local_signature = compute_data_config_signature(
        dataset_spec=local_spec,
        args=local_args,
    )
    remote_signature = compute_data_config_signature(
        dataset_spec=remote_spec,
        args=remote_args,
    )
    assert local_signature == remote_signature

    local_run_identity = build_signature(
        "PatchTST",
        local_spec.key,
        {"lr": 0.001},
        pipeline_id="baseline",
        data_config_signature=local_signature,
    )
    remote_run_identity = build_signature(
        "PatchTST",
        remote_spec.key,
        {"lr": 0.001},
        pipeline_id="baseline",
        data_config_signature=remote_signature,
    )
    assert local_run_identity == remote_run_identity
    assert (
        spec_to_tags(local_spec, n_inputs=12, n_outputs=1)["dataset_path"]
        == "beijing_air_tiantan.parquet"
    )
    assert (
        spec_to_tags(remote_spec, n_inputs=12, n_outputs=1)["dataset_path"]
        == "beijing_air_tiantan.parquet"
    )


def test_missing_registered_dataset_file_names_expected_path_and_data_guide(tmp_path):
    dataset_spec = resolve_with_defaults(
        "ETTh1",
        [],
        data_root=tmp_path.as_posix(),
    )[0]
    with pytest.raises(FileNotFoundError, match="data/README.md") as exc_info:
        TSDataModule(dataset_spec=dataset_spec, num_workers=0)

    assert str(tmp_path / "ETTh1.csv") in str(exc_info.value)


def test_registered_csv_dataset_accepts_missing_datetime_column(tmp_path):
    csv_path = tmp_path / "ETTh1.csv"
    csv_path.write_text(
        "HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n"
        "1,2,3,4,5,6,7\n",
        encoding="utf-8",
    )
    dataset_spec = resolve_with_defaults(
        "ETTh1",
        [],
        data_root=tmp_path.as_posix(),
    )[0]

    data_module = TSDataModule(
        dataset_spec=dataset_spec,
        num_workers=0,
        perturbation_channel_fraction_max=0.5,
        perturbation_scenarios=("drift",),
    )

    assert data_module._read_dataframe(nrows=1).columns.tolist() == [
        "HUFL",
        "HULL",
        "MUFL",
        "MULL",
        "LUFL",
        "LULL",
        "OT",
    ]


def test_file_like_dataset_identifiers_raise_in_benchmark_resolution():
    for identifier in ("typo_dataset.csv", "ETTh1.csv", "traffic.parquet"):
        with pytest.raises(KeyError, match="not registered"):
            resolve_with_defaults(
                identifier,
                [],
                data_root="data/processed",
            )


def test_lowercase_dataset_alias_raises_in_benchmark_resolution():
    with pytest.raises(KeyError, match="not registered"):
        resolve_with_defaults(
            "beijingair_tiantan",
            [],
            data_root="data/processed",
        )


def test_dataset_target_overrides_raise_on_blank_and_extra_entries():
    with pytest.raises(ValueError, match="DATA_TARGETS entries must be non-empty"):
        resolve_with_defaults(
            ["ETTh1"],
            [""],
            data_root="data/processed",
        )

    with pytest.raises(ValueError, match="more entries than DATA_FILES"):
        resolve_with_defaults(
            ["ETTh1"],
            ["all", "ot"],
            data_root="data/processed",
        )


def test_data_guide_lists_benchmark_filenames_and_validation_surface():
    guide = (REPO_ROOT / "data/README.md").read_text(encoding="utf-8")

    for key, filename in EXPECTED_BENCHMARK_DATASETS.items():
        assert key in guide
        assert filename in guide
    for required_phrase in (
        "row-count and channel-count",
        "validation",
        "scripts/benchmark_dataset_contracts.py",
        "dataframe-content SHA256",
        "file-byte fixity",
        "scripts/generate_dataset_checksums.py",
        "https://www.kaggle.com/datasets/sensorfaultbench/beijing-air-tiantan-pm2-5-time-series-dataset",
        "https://www.kaggle.com/datasets/sensorfaultbench/penmanshiel-wt08-hourly-scada-time-series-dataset",
    ):
        assert required_phrase in guide


def test_data_guide_documents_traffic_runtime_and_csv_paths():
    guide = (REPO_ROOT / "data/README.md").read_text(encoding="utf-8")
    sources = guide[guide.index("## Sources And Terms") :]
    traffic_section = sources[
        sources.index("- `traffic`:")
        : sources.index("- `BeijingAir_Tiantan`:")
    ]

    assert "scripts/preprocess_traffic.py" in traffic_section
    assert "traffic.parquet" in traffic_section
    assert "DATA_ROOT/traffic.csv" in traffic_section


def test_dataset_scripts_take_data_root_defaults_from_yaml(monkeypatch):
    expected_root = Path("/tmp/benchmark-data")
    monkeypatch.setattr(
        benchmark_dataset_contracts,
        "load_defaults",
        lambda: {"DATA_ROOT": str(expected_root)},
    )
    parser = argparse.ArgumentParser()
    add_common_args(parser, contract=BEIJING_AIR_TIANTAN)

    parsed = parser.parse_args(["--validate-existing"])

    assert parsed.output == expected_root / BEIJING_AIR_TIANTAN.filename

    monkeypatch.setattr(
        generate_dataset_checksums,
        "load_defaults",
        lambda: {"DATA_ROOT": str(expected_root)},
    )

    assert default_data_root() == expected_root


def test_benchmark_contracts_top_level_help_lists_common_validation_flags(capsys):
    with pytest.raises(SystemExit) as exc_info:
        benchmark_dataset_contracts.main(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    for option in (
        "--dataset",
        "--output",
        "--validate-existing",
        "--require-checksums",
    ):
        assert option in help_text


def test_benchmark_contracts_match_registry_metadata_and_fail_clearly_without_raw(tmp_path):
    for contract in BENCHMARK_DATASET_CONTRACTS:
        spec = DATASET_REGISTRY.get(contract.key).resolve()
        input_channels = spec.input_channels
        continuous_channels = spec.continuous_channels
        discrete_channels = spec.discrete_channels
        if input_channels is None:
            raise AssertionError(f"{contract.key} must declare input channels.")
        if continuous_channels is None:
            raise AssertionError(f"{contract.key} must declare continuous channels.")
        if discrete_channels is None:
            raise AssertionError(f"{contract.key} must declare discrete channels.")
        assert spec.path == contract.filename
        assert spec.split_mode == contract.split_mode
        assert spec.target_alias == contract.target_alias
        assert len(input_channels) == contract.channel_count
        assert len(continuous_channels) == contract.continuous_channel_count
        assert len(discrete_channels) == contract.discrete_channel_count
        assert contract.content_sha256 is not None
        assert len(contract.content_sha256) == 64
        if contract.key == "traffic":
            assert (
                contract.source_url
                == "https://huggingface.co/datasets/thuml/Time-Series-Library/blob/main/traffic/traffic.csv"
            )
            assert contract.zenodo_url == "https://doi.org/10.5281/zenodo.4656132"
        if contract.key == "Penmanshiel_Hourly_WT08":
            assert contract.source_url == "https://zenodo.org/records/16807304"
            assert "2016-2022" in contract.source_label

        with pytest.raises(FileNotFoundError, match="Missing raw source"):
            run_contract_cli(
                contract=contract,
                argv=[
                    "--raw-source",
                    str(tmp_path / contract.source_label),
                    "--output",
                    str(tmp_path / contract.filename),
                    "--validate-existing",
                    "--require-checksums",
                ],
            )


def test_preprocessing_scripts_define_raw_conversion_callbacks():
    assert callable(prepare_beijing)
    assert callable(prepare_penmanshiel)
    assert callable(prepare_traffic)


def test_etth1_contract_validates_canonical_csv(tmp_path, capsys, monkeypatch):
    contract = replace(
        ETTH1,
        row_count=2,
        start_datetime="2016-07-01 00:00:00",
        end_datetime="2016-07-01 01:00:00",
    )
    output_path = tmp_path / contract.filename
    _benchmark_frame(
        "ETTh1",
        ["2016-07-01 00:00:00", "2016-07-01 01:00:00"],
        time_column=contract.time_column,
    ).to_csv(output_path, index=False)
    contract = replace(
        contract,
        content_sha256=benchmark_dataset_contracts.sha256_dataframe_content(
            pd.read_csv(output_path),
        ),
    )
    monkeypatch.setitem(benchmark_dataset_contracts.CONTRACTS_BY_KEY, "ETTh1", contract)

    assert benchmark_dataset_contracts.main(
        [
            "--dataset",
            "ETTh1",
            "--output",
            str(output_path),
            "--validate-existing",
            "--require-checksums",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset"] == "ETTh1"
    assert payload["rows"] == 2
    assert "benchmark_file_sha256" not in payload
    assert payload["content_sha256"] == contract.content_sha256


def test_checksum_validation_uses_content_not_parquet_container_bytes(tmp_path):
    contract = replace(
        TRAFFIC,
        row_count=2,
        start_datetime="2016-07-01 02:00:00",
        end_datetime="2016-07-01 03:00:00",
    )
    output_path = tmp_path / contract.filename
    frame = _benchmark_frame(
        "traffic",
        ["2016-07-01 02:00:00", "2016-07-01 03:00:00"],
    )
    frame["datetime"] = pd.to_datetime(frame["datetime"])

    frame_us = frame.copy()
    frame_us["datetime"] = frame_us["datetime"].astype("datetime64[us]")
    frame_us.to_parquet(output_path, index=False)
    expected_content_sha256 = benchmark_dataset_contracts.sha256_dataframe_content(
        pd.read_parquet(output_path),
    )
    first_payload = output_path.read_bytes()

    frame_ns = frame.copy()
    frame_ns["datetime"] = frame_ns["datetime"].astype("datetime64[ns]")
    frame_ns.to_parquet(output_path, index=False)
    assert output_path.read_bytes() != first_payload
    assert (
        benchmark_dataset_contracts.sha256_dataframe_content(
            pd.read_parquet(output_path),
        )
        == expected_content_sha256
    )

    validate_processed_file(
        replace(contract, content_sha256=expected_content_sha256),
        output_path,
        require_checksums=True,
    )


def _write_traffic_raw(
    raw_path: Path,
    timestamps: list[str],
    *,
    value_overrides: dict[tuple[int, str], object] | None = None,
) -> tuple[str, ...]:
    spec = DATASET_REGISTRY.get("traffic").resolve()
    input_channels = spec.input_channels
    if input_channels is None:
        raise AssertionError("traffic must declare input channels.")
    value_overrides = value_overrides or {}
    rows = []
    for row_idx, timestamp in enumerate(timestamps):
        row = {"date": timestamp}
        row.update(
            {
                channel: value_overrides.get(
                    (row_idx, channel),
                    float(row_idx) + channel_idx / 1000.0,
                )
                for channel_idx, channel in enumerate(input_channels)
            }
        )
        rows.append(row)
    pd.DataFrame(rows, columns=["date", *input_channels]).to_csv(
        raw_path,
        index=False,
    )
    return input_channels


def test_traffic_preprocessing_converts_raw_csv_to_benchmark_parquet(tmp_path):
    raw_path = tmp_path / "traffic.csv"
    output_path = tmp_path / "traffic.parquet"
    input_channels = _write_traffic_raw(
        raw_path,
        ["2016-07-01 02:00:00", "2016-07-01 03:00:00"],
    )

    prepare_traffic(raw_path, output_path)

    converted = pd.read_parquet(output_path)
    assert converted.columns.tolist() == ["datetime", *input_channels]
    assert converted["datetime"].tolist() == pd.to_datetime(
        ["2016-07-01 02:00:00", "2016-07-01 03:00:00"]
    ).tolist()
    assert converted[list(input_channels)].isna().sum().sum() == 0


def test_traffic_preprocessing_rejects_unexpected_source_filename(tmp_path):
    raw_path = tmp_path / "not_traffic.csv"
    output_path = tmp_path / "traffic.parquet"
    _write_traffic_raw(raw_path, ["2016-07-01 02:00:00"])

    with pytest.raises(ValueError, match="must be named traffic.csv"):
        prepare_traffic(raw_path, output_path)


def test_traffic_preprocessing_rejects_non_hourly_timestamps(tmp_path):
    raw_path = tmp_path / "traffic.csv"
    output_path = tmp_path / "traffic.parquet"
    _write_traffic_raw(
        raw_path,
        ["2016-07-01 02:00:00", "2016-07-01 04:00:00"],
    )

    with pytest.raises(ValueError, match="hourly frequency"):
        prepare_traffic(raw_path, output_path)


def test_traffic_preprocessing_rejects_non_finite_sensor_values(tmp_path):
    raw_path = tmp_path / "traffic.csv"
    output_path = tmp_path / "traffic.parquet"
    _write_traffic_raw(
        raw_path,
        ["2016-07-01 02:00:00"],
        value_overrides={(0, "0"): "inf"},
    )

    with pytest.raises(ValueError, match="non-finite sensor values"):
        prepare_traffic(raw_path, output_path)


def test_dataset_checksum_manifest_is_file_fixity_and_fails_on_missing_files(tmp_path):
    filenames = DATASET_CHECKSUM_BUNDLES["beijing_air_tiantan"]
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2014-06-03 10:00:00"]),
            "PM2.5": [1.0],
        }
    )
    for filename in filenames:
        path = tmp_path / filename
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)

    expected_lines = tuple(
        f"{sha256_file_bytes(tmp_path / filename)}  {filename}"
        for filename in filenames
    )
    assert build_checksum_lines(tmp_path, bundle="beijing_air_tiantan") == expected_lines
    assert (
        build_manifest_text(tmp_path, bundle="beijing_air_tiantan")
        == "\n".join(expected_lines) + "\n"
    )

    parquet_path = tmp_path / filenames[0]
    first_payload = parquet_path.read_bytes()
    frame_us = frame.copy()
    frame_us["datetime"] = frame_us["datetime"].astype("datetime64[us]")
    frame_us.to_parquet(parquet_path, index=False)
    assert parquet_path.read_bytes() != first_payload
    assert build_checksum_lines(tmp_path, bundle="beijing_air_tiantan") != expected_lines

    with pytest.raises(ValueError, match="Unknown dataset checksum bundle"):
        build_checksum_lines(tmp_path, bundle="ETTh1")

    (tmp_path / filenames[-1]).unlink()
    with pytest.raises(FileNotFoundError, match=filenames[-1]):
        build_checksum_lines(tmp_path, bundle="beijing_air_tiantan")


def test_processed_benchmark_file_rejects_extra_columns(tmp_path):
    spec = DATASET_REGISTRY.get("BeijingAir_Tiantan").resolve()
    input_channels = spec.input_channels
    if input_channels is None:
        raise AssertionError("BeijingAir_Tiantan must declare input channels.")
    contract = replace(BEIJING_AIR_TIANTAN, row_count=1)
    output_path = tmp_path / contract.filename
    frame = pd.DataFrame(
        [{column: 1 for column in ["datetime", *input_channels, "leakage_column"]}]
    )
    frame.to_parquet(output_path, index=False)

    with pytest.raises(ValueError, match="extra: \\['leakage_column'\\]"):
        validate_processed_file(
            contract,
            output_path,
            require_checksums=False,
        )


def _benchmark_frame(
    spec_key: str,
    timestamps: list[str],
    *,
    time_column: str = "datetime",
    value_overrides: dict[tuple[int, str], object] | None = None,
) -> pd.DataFrame:
    spec = DATASET_REGISTRY.get(spec_key).resolve()
    input_channels = spec.input_channels
    if input_channels is None:
        raise AssertionError(f"{spec_key} must declare input channels.")
    value_overrides = value_overrides or {}
    rows = []
    for row_idx, timestamp in enumerate(timestamps):
        row = {time_column: timestamp}
        row.update(
            {
                channel: value_overrides.get(
                    (row_idx, channel),
                    row_idx + channel_idx + 1,
                )
                for channel_idx, channel in enumerate(input_channels)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=[time_column, *input_channels])


def test_processed_benchmark_file_rejects_duplicate_or_non_hourly_timestamps(tmp_path):
    contract = replace(TRAFFIC, row_count=2)
    output_path = tmp_path / contract.filename

    _benchmark_frame(
        "traffic",
        ["2016-07-01 02:00:00", "2016-07-01 02:00:00"],
    ).to_parquet(output_path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        validate_processed_file(contract, output_path, require_checksums=False)

    _benchmark_frame(
        "traffic",
        ["2016-07-01 02:00:00", "2016-07-01 04:00:00"],
    ).to_parquet(output_path, index=False)
    with pytest.raises(ValueError, match="hourly frequency"):
        validate_processed_file(contract, output_path, require_checksums=False)


def test_processed_benchmark_file_rejects_wrong_time_interval(tmp_path):
    contract = replace(
        TRAFFIC,
        row_count=2,
        start_datetime="2016-07-01 02:00:00",
        end_datetime="2016-07-01 03:00:00",
    )
    output_path = tmp_path / contract.filename
    _benchmark_frame(
        "traffic",
        ["2000-01-01 00:00:00", "2000-01-01 01:00:00"],
    ).to_parquet(output_path, index=False)

    with pytest.raises(ValueError, match="time interval mismatch"):
        validate_processed_file(contract, output_path, require_checksums=False)


def test_processed_benchmark_file_rejects_non_finite_continuous_values(tmp_path):
    contract = replace(
        BEIJING_AIR_TIANTAN,
        row_count=1,
        start_datetime="2014-06-03 10:00:00",
        end_datetime="2014-06-03 10:00:00",
    )
    output_path = tmp_path / contract.filename
    _benchmark_frame(
        "BeijingAir_Tiantan",
        ["2014-06-03 10:00:00"],
        value_overrides={(0, "PM2.5"): float("inf")},
    ).to_parquet(output_path, index=False)

    with pytest.raises(ValueError, match="non-finite continuous values"):
        validate_processed_file(contract, output_path, require_checksums=False)


def test_processed_benchmark_file_rejects_invalid_discrete_values(tmp_path):
    contract = replace(
        BEIJING_AIR_TIANTAN,
        row_count=1,
        start_datetime="2014-06-03 10:00:00",
        end_datetime="2014-06-03 10:00:00",
    )
    output_path = tmp_path / contract.filename
    _benchmark_frame(
        "BeijingAir_Tiantan",
        ["2014-06-03 10:00:00"],
        value_overrides={(0, "wd"): "not-a-valid-wd"},
    ).to_parquet(output_path, index=False)

    with pytest.raises(ValueError, match="invalid discrete values"):
        validate_processed_file(contract, output_path, require_checksums=False)


def test_provenance_notebooks_are_markdown_only_and_document_derivation():
    provenance_dir = REPO_ROOT / "notebooks/provenance"
    notebooks = {path.name: path for path in provenance_dir.glob("*.ipynb")}
    assert set(notebooks) == PROVENANCE_NOTEBOOKS

    for notebook_path in notebooks.values():
        assert notebook_path.stat().st_size < 30000
        text = notebook_path.read_text(encoding="utf-8")
        assert "Original source:" in text
        assert "Benchmark derived dataset:" in text
        assert "## Slice Rationale" in text
        assert "## EDA Summary" in text
        assert "## Benchmark Validation" in text
        payload = json.loads(text)
        for cell in payload["cells"]:
            assert cell["cell_type"] == "markdown"
            assert "outputs" not in cell
            assert "execution_count" not in cell
