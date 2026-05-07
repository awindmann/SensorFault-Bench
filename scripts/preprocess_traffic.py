#!/usr/bin/env python3
"""Prepare and validate the Traffic benchmark dataset from the raw CSV."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.datasets import DATASET_REGISTRY
from scripts.benchmark_dataset_contracts import TRAFFIC, run_contract_cli


def _traffic_raw_csv(raw_source: Path) -> Path:
    if raw_source.is_file() and raw_source.suffix.lower() == ".csv":
        if raw_source.name != "traffic.csv":
            raise ValueError(
                "Traffic raw source must be named traffic.csv to match the benchmark source contract."
            )
        return raw_source
    if raw_source.is_dir():
        matches = sorted(raw_source.rglob("traffic.csv"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Expected exactly one traffic.csv under {raw_source}, found {matches}."
            )
    raise ValueError(
        "Traffic raw source must be traffic.csv or a directory containing exactly one "
        "traffic.csv file."
    )


def _expected_columns() -> list[str]:
    spec = DATASET_REGISTRY.get(TRAFFIC.key).resolve()
    input_channels = spec.input_channels
    if input_channels is None:
        raise ValueError(f"{TRAFFIC.key} must declare input channels.")
    return ["date", *input_channels]


def _validate_hourly_datetimes(datetimes: pd.Series) -> None:
    if datetimes.empty:
        raise ValueError("Traffic raw CSV contains no rows.")
    if datetimes.isna().any():
        raise ValueError("Traffic raw CSV contains unparsable datetime values.")
    if datetimes.duplicated().any():
        raise ValueError("Traffic raw CSV contains duplicate datetime values.")
    deltas = datetimes.diff().dropna()
    if not (deltas == pd.Timedelta(hours=1)).all():
        raise ValueError(
            "Traffic raw CSV timestamps must be strictly increasing at hourly frequency."
        )


def prepare_from_raw(raw_source: Path, output_path: Path) -> Path:
    raw_csv = _traffic_raw_csv(raw_source)
    frame = pd.read_csv(raw_csv)
    expected_columns = _expected_columns()
    actual_columns = frame.columns.tolist()
    if actual_columns != expected_columns:
        missing_columns = sorted(set(expected_columns) - set(actual_columns))
        extra_columns = sorted(set(actual_columns) - set(expected_columns))
        raise ValueError(
            "Traffic raw CSV columns must exactly match the benchmark contract. "
            f"Missing: {missing_columns}; extra: {extra_columns}."
        )
    frame = frame.rename(columns={"date": "datetime"})
    frame["datetime"] = pd.to_datetime(
        frame["datetime"],
        format="%Y-%m-%d %H:%M:%S",
        errors="raise",
    )
    _validate_hourly_datetimes(frame["datetime"])
    input_columns = [column for column in frame.columns if column != "datetime"]
    for column in input_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame[input_columns].isna().any().any():
        raise ValueError("Traffic raw CSV contains missing sensor values.")
    numeric_values = frame[input_columns].to_numpy(dtype="float64", copy=False)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Traffic raw CSV contains non-finite sensor values.")
    frame.to_parquet(output_path, index=False)
    return output_path


def main() -> int:
    return run_contract_cli(
        contract=TRAFFIC,
        prepare_from_raw=prepare_from_raw,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
