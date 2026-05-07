"""Benchmark dataset validation contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.datasets import DATASET_REGISTRY, ResolvedDatasetSpec
from config_loader import load_defaults


@dataclass(frozen=True)
class DerivedDatasetContract:
    key: str
    filename: str
    split_mode: str
    target_alias: str
    time_column: str
    start_datetime: str
    end_datetime: str
    row_count: int
    channel_count: int
    continuous_channel_count: int
    discrete_channel_count: int
    content_sha256: str | None
    source_label: str
    source_url: str
    zenodo_url: str | None
    discrete_value_domains: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)


ETTH1 = DerivedDatasetContract(
    key="ETTh1",
    filename="ETTh1.csv",
    split_mode="temporal",
    target_alias="all",
    time_column="date",
    start_datetime="2016-07-01 00:00:00",
    end_datetime="2018-06-26 19:00:00",
    row_count=17420,
    channel_count=7,
    continuous_channel_count=7,
    discrete_channel_count=0,
    content_sha256="f6183ff793de4f47893e0edc709f8b94766b391ef3b36ccc51e3458655f3863f",
    source_label="ETTh1.csv",
    source_url="https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small",
    zenodo_url=None,
)


BEIJING_AIR_TIANTAN = DerivedDatasetContract(
    key="BeijingAir_Tiantan",
    filename="beijing_air_tiantan.parquet",
    split_mode="temporal",
    target_alias="pm25",
    time_column="datetime",
    start_datetime="2014-06-03 10:00:00",
    end_datetime="2017-02-28 23:00:00",
    row_count=24038,
    channel_count=12,
    continuous_channel_count=11,
    discrete_channel_count=1,
    content_sha256="9ad18e546a4d43d868c71d0f1c2cfefc6f4de1bd6e35dfc54baa2aa4b2f86e65",
    source_label="PRSA_Data_Tiantan_20130301-20170228.csv",
    source_url="https://doi.org/10.24432/C5RK5G",
    zenodo_url=None,
    discrete_value_domains={"wd": tuple(range(16))},
)

PENMANSHIEL_HOURLY_WT08 = DerivedDatasetContract(
    key="Penmanshiel_Hourly_WT08",
    filename="penmanshiel_hourly_wt08.parquet",
    split_mode="temporal",
    target_alias="power",
    time_column="datetime",
    start_datetime="2016-08-18 15:00:00",
    end_datetime="2019-08-01 09:00:00",
    row_count=25867,
    channel_count=65,
    continuous_channel_count=65,
    discrete_channel_count=0,
    content_sha256="a6df990227602898d4b1d24250572c80d2429b28767d785c4850474e57fd88a9",
    source_label="Penmanshiel v3 SCADA 2016-2022 WT01-10 ZIP files",
    source_url="https://zenodo.org/records/16807304",
    zenodo_url=None,
)

TRAFFIC = DerivedDatasetContract(
    key="traffic",
    filename="traffic.parquet",
    split_mode="temporal",
    target_alias="all",
    time_column="datetime",
    start_datetime="2016-07-01 02:00:00",
    end_datetime="2018-07-02 01:00:00",
    row_count=17544,
    channel_count=862,
    continuous_channel_count=862,
    discrete_channel_count=0,
    content_sha256="8a501d87ed1253976aaed03a9e44bd89a87cef922abc45edae8417f9e7b604ed",
    source_label="traffic.csv",
    source_url="https://huggingface.co/datasets/thuml/Time-Series-Library/blob/main/traffic/traffic.csv",
    zenodo_url="https://doi.org/10.5281/zenodo.4656132",
)


BENCHMARK_DATASET_CONTRACTS = (
    PENMANSHIEL_HOURLY_WT08,
    ETTH1,
    BEIJING_AIR_TIANTAN,
    TRAFFIC,
)
CONTRACTS_BY_KEY = {contract.key: contract for contract in BENCHMARK_DATASET_CONTRACTS}


def _default_data_root() -> Path:
    defaults = load_defaults()
    if "DATA_ROOT" not in defaults:
        raise ValueError("configs/defaults.yaml is missing required key 'DATA_ROOT'.")
    return Path(defaults["DATA_ROOT"])


def sha256_dataframe_content(frame: pd.DataFrame) -> str:
    csv_payload = frame.to_csv(index=True, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(csv_payload).hexdigest()


def _require_recorded_checksum(value: str | None, *, name: str, dataset_key: str) -> str:
    if value is None:
        raise ValueError(
            f"{name} is not recorded for {dataset_key}. Fill the benchmark artifact "
            "contract from the frozen dataset content version before checksum-gated "
            "validation can pass."
        )
    return value


def _validate_hourly_datetime_column(
    frame: pd.DataFrame,
    *,
    dataset_key: str,
    time_column: str,
    expected_start: str,
    expected_end: str,
) -> None:
    try:
        datetimes = pd.to_datetime(frame[time_column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{dataset_key} {time_column} column contains unparsable values."
        ) from exc
    if datetimes.isna().any():
        raise ValueError(f"{dataset_key} {time_column} column contains missing values.")
    if datetimes.duplicated().any():
        raise ValueError(f"{dataset_key} {time_column} column contains duplicate values.")
    if not datetimes.is_monotonic_increasing:
        raise ValueError(f"{dataset_key} {time_column} column must be strictly increasing.")
    deltas = datetimes.diff().dropna()
    if not deltas.empty and not (deltas == pd.Timedelta(hours=1)).all():
        raise ValueError(f"{dataset_key} {time_column} column must have hourly frequency.")

    expected_start_ts = pd.Timestamp(expected_start)
    expected_end_ts = pd.Timestamp(expected_end)
    actual_start = pd.Timestamp(datetimes.iloc[0])
    actual_end = pd.Timestamp(datetimes.iloc[-1])
    if actual_start != expected_start_ts or actual_end != expected_end_ts:
        raise ValueError(
            f"{dataset_key} time interval mismatch: expected {expected_start_ts} to "
            f"{expected_end_ts}, got {actual_start} to {actual_end}."
        )


def _validate_channel_values(
    frame: pd.DataFrame,
    resolved: ResolvedDatasetSpec,
    *,
    contract: DerivedDatasetContract,
) -> None:
    dataset_key = contract.key
    continuous_channels = resolved.continuous_channels or ()
    if continuous_channels:
        continuous = frame.loc[:, list(continuous_channels)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        finite_mask = np.isfinite(continuous.to_numpy(dtype=float))
        if not finite_mask.all():
            raise ValueError(
                f"{dataset_key} contains missing or non-finite continuous values."
            )
    discrete_channels = resolved.discrete_channels or ()
    if discrete_channels and frame.loc[:, list(discrete_channels)].isna().any().any():
        raise ValueError(f"{dataset_key} contains missing discrete values.")
    if discrete_channels:
        domain_channels = set(contract.discrete_value_domains)
        discrete_channel_set = set(discrete_channels)
        missing_domains = sorted(discrete_channel_set - domain_channels)
        if missing_domains:
            raise ValueError(
                f"{dataset_key} is missing discrete value domain(s) for {missing_domains}."
            )
        unknown_domains = sorted(domain_channels - discrete_channel_set)
        if unknown_domains:
            raise ValueError(
                f"{dataset_key} declares discrete value domain(s) for non-discrete "
                f"channel(s): {unknown_domains}."
            )
        for channel in discrete_channels:
            allowed_values = set(contract.discrete_value_domains[channel])
            observed_values = frame[channel].dropna().unique().tolist()
            invalid_values = sorted(
                {value for value in observed_values if value not in allowed_values},
                key=str,
            )
            if invalid_values:
                raise ValueError(
                    f"{dataset_key} contains invalid discrete values in {channel}: "
                    f"{invalid_values}."
                )


def read_dataset_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(
        f"Benchmark dataset file format not supported for {path.name!r}. "
        "Expected .csv or .parquet."
    )


def validate_registry_contract(contract: DerivedDatasetContract) -> ResolvedDatasetSpec:
    spec = DATASET_REGISTRY.get(contract.key)
    resolved = spec.resolve()
    if resolved.path != contract.filename:
        raise ValueError(
            f"{contract.key} registry filename mismatch: expected "
            f"{contract.filename!r}, got {resolved.path!r}."
        )
    if resolved.split_mode != contract.split_mode:
        raise ValueError(
            f"{contract.key} split_mode mismatch: expected {contract.split_mode!r}, "
            f"got {resolved.split_mode!r}."
        )
    if resolved.target_alias != contract.target_alias:
        raise ValueError(
            f"{contract.key} target alias mismatch: expected {contract.target_alias!r}, "
            f"got {resolved.target_alias!r}."
        )
    input_channels = resolved.input_channels
    if input_channels is None:
        raise ValueError(f"{contract.key} must declare input channel metadata.")
    if len(input_channels) != contract.channel_count:
        raise ValueError(
            f"{contract.key} channel count mismatch: expected {contract.channel_count}, "
            f"got {len(input_channels)}."
        )
    continuous_channels = resolved.continuous_channels
    discrete_channels = resolved.discrete_channels
    if continuous_channels is None or discrete_channels is None:
        raise ValueError(f"{contract.key} must declare continuous/discrete channels.")
    if len(continuous_channels) != contract.continuous_channel_count:
        raise ValueError(
            f"{contract.key} continuous channel count mismatch: expected "
            f"{contract.continuous_channel_count}, got {len(continuous_channels)}."
        )
    if len(discrete_channels) != contract.discrete_channel_count:
        raise ValueError(
            f"{contract.key} discrete channel count mismatch: expected "
            f"{contract.discrete_channel_count}, got {len(discrete_channels)}."
        )
    return resolved


def validate_processed_file(
    contract: DerivedDatasetContract,
    output_path: Path,
    *,
    require_checksums: bool,
) -> dict[str, str | int]:
    resolved = validate_registry_contract(contract)
    if output_path.name != contract.filename:
        raise ValueError(
            f"{contract.key} must export to {contract.filename!r}, got "
            f"{output_path.name!r}."
        )
    if not output_path.exists():
        raise FileNotFoundError(
            f"Processed benchmark file for {contract.key} was not found at "
            f"{output_path}. Prepare the dataset under DATA_ROOT and see data/README.md."
        )
    frame = read_dataset_frame(output_path)
    if len(frame) != contract.row_count:
        raise ValueError(
            f"{contract.key} row count mismatch: expected {contract.row_count}, "
            f"got {len(frame)}."
        )
    input_channels = resolved.input_channels
    if input_channels is None:
        raise ValueError(f"{contract.key} must declare input channel metadata.")
    expected_columns = [contract.time_column, *input_channels]
    actual_columns = frame.columns.tolist()
    if actual_columns != expected_columns:
        missing_columns = sorted(set(expected_columns) - set(actual_columns))
        extra_columns = sorted(set(actual_columns) - set(expected_columns))
        raise ValueError(
            f"{contract.key} columns must exactly match the benchmark contract. "
            f"Missing: {missing_columns}; extra: {extra_columns}."
        )
    _validate_hourly_datetime_column(
        frame,
        dataset_key=contract.key,
        time_column=contract.time_column,
        expected_start=contract.start_datetime,
        expected_end=contract.end_datetime,
    )
    _validate_channel_values(frame, resolved, contract=contract)

    content_sha256 = sha256_dataframe_content(frame)
    if require_checksums:
        expected_content_sha256 = _require_recorded_checksum(
            contract.content_sha256,
            name="content_sha256",
            dataset_key=contract.key,
        )
        if content_sha256 != expected_content_sha256:
            raise ValueError(
                f"{contract.key} content checksum mismatch: expected "
                f"{expected_content_sha256}, got {content_sha256}."
            )
    return {
        "dataset": contract.key,
        "filename": contract.filename,
        "rows": len(frame),
        "channels": len(input_channels),
        "content_sha256": content_sha256,
    }


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    contract: DerivedDatasetContract | None,
) -> None:
    source_label = (
        contract.source_label
        if contract is not None
        else "the selected dataset source"
    )
    output_default = (
        _default_data_root() / contract.filename
        if contract is not None
        else None
    )
    parser.add_argument(
        "--raw-source",
        type=Path,
        default=None,
        help=f"Path to the original source file or directory ({source_label}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=output_default,
        help=(
            "Processed benchmark file path to validate. Defaults to the "
            "selected dataset filename under DATA_ROOT."
        ),
    )
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Validate an already exported processed file instead of preparing from raw data.",
    )
    parser.add_argument(
        "--require-checksums",
        action="store_true",
        help="Require recorded canonical content checksums to match.",
    )


def _run_contract_args(
    *,
    contract: DerivedDatasetContract,
    prepare_from_raw: Callable[[Path, Path], Path] | None = None,
    args: argparse.Namespace,
) -> int:
    output_path = args.output
    if output_path is None:
        output_path = _default_data_root() / contract.filename

    if args.raw_source is None and not args.validate_existing:
        raise ValueError(
            "--raw-source is required unless --validate-existing is used. "
            f"Expected source: {contract.source_label}. See data/README.md."
        )
    if args.raw_source is not None and not args.raw_source.exists():
        raise FileNotFoundError(
            f"Missing raw source for {contract.key}: {args.raw_source}. "
            f"Expected {contract.source_label}; see data/README.md for acquisition."
        )
    if not args.validate_existing:
        if prepare_from_raw is None:
            raise ValueError(
                f"{contract.key} does not define a raw-source preprocessing callback."
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prepared_path = prepare_from_raw(args.raw_source, output_path)
        if prepared_path != output_path:
            raise ValueError(
                f"{contract.key} preprocessing callback returned {prepared_path}, "
                f"expected {output_path}."
            )

    result = validate_processed_file(
        contract,
        output_path,
        require_checksums=args.require_checksums,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def run_contract_cli(
    *,
    contract: DerivedDatasetContract,
    prepare_from_raw: Callable[[Path, Path], Path] | None = None,
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=f"Validate {contract.key} benchmark data."
    )
    add_common_args(parser, contract=contract)
    args = parser.parse_args(argv)
    return _run_contract_args(
        contract=contract,
        prepare_from_raw=prepare_from_raw,
        args=args,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a staged benchmark dataset against its contract."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(CONTRACTS_BY_KEY),
        required=True,
        help="Registered benchmark dataset key to validate.",
    )
    add_common_args(parser, contract=None)
    args = parser.parse_args(argv)
    return _run_contract_args(
        contract=CONTRACTS_BY_KEY[args.dataset],
        args=args,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
