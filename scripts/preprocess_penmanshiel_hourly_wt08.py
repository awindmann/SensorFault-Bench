#!/usr/bin/env python3
"""Prepare and validate the Penmanshiel WT08 benchmark dataset."""

from __future__ import annotations

import csv
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.datasets import DATASET_REGISTRY
from scripts.benchmark_dataset_contracts import (
    PENMANSHIEL_HOURLY_WT08,
    run_contract_cli,
)


SELECTED_TURBINE = "08"
YEARS = (2016, 2017, 2018, 2019, 2020, 2021, 2022)
TARGET_COLUMN = "Power (kW)"
TARGET_GAP_MAX_DAYS = 1
FEATURE_LONG_GAP_THRESHOLD_DAYS = 21
LEAKAGE_RULES = {
    "cumulative energy counters": {
        "patterns": (
            "Energy Export",
            "Energy Import",
            "Energy Budget",
            "Energy Theoretical",
            "Virtual Production",
            "Lost Production",
            "Reactive Energy",
            "Equivalent Full Load Hours",
        ),
    },
    "availability and contractual indices": {
        "patterns": (
            "Capacity factor",
            "Data Availability",
            "Time-based",
            "Production-based",
            "Contractual Avail.",
            "System Avail",
            "Planning deviation",
            "Performance Index",
            "Production Factor",
            "Performance Ratio",
        ),
    },
    "power-curve and controller expectations": {
        "patterns": (
            "Potential power",
            "Available Capacity for Production",
            "Turbine Power setpoint",
            "Apparent power",
            "APE-",
        ),
        "exact": (
            "Power, Standard deviation (kW)",
            "Power, Minimum (kW)",
            "Power, Maximum (kW)",
        ),
    },
}


def _registered_channels() -> list[str]:
    channels = DATASET_REGISTRY.get(PENMANSHIEL_HOURLY_WT08.key).resolve().input_channels
    if channels is None:
        raise ValueError(f"{PENMANSHIEL_HOURLY_WT08.key} must declare input channels.")
    return list(channels)


def _scada_zip_dir(raw_source: Path) -> Path:
    if raw_source.is_file() and raw_source.suffix.lower() == ".zip":
        return raw_source.parent
    if raw_source.is_dir():
        return raw_source
    raise ValueError(
        "Penmanshiel WT08 raw source must be a directory containing the "
        "2016-2022 Penmanshiel_SCADA_*_WT01-10_*.zip files, or one such ZIP "
        "file beside the rest of that subset."
    )


def _find_scada_zips(raw_source: Path) -> list[dict[str, Path | int]]:
    raw_dir = _scada_zip_dir(raw_source)
    turbine_num = int(SELECTED_TURBINE)
    found = []
    for zip_path in sorted(raw_dir.glob("Penmanshiel_SCADA_*.zip")):
        match = re.search(r"(\d{4})_WT(?:_)?(\d+)-(\d+)", zip_path.name)
        if not match:
            continue
        year = int(match.group(1))
        wt_start = int(match.group(2))
        wt_end = int(match.group(3))
        if year in YEARS and wt_start <= turbine_num <= wt_end:
            found.append({"path": zip_path, "year": year})
    if not found:
        raise ValueError(
            f"No Penmanshiel v3 SCADA ZIPs for WT{SELECTED_TURBINE} and "
            f"years {YEARS} under {raw_dir}."
        )
    return sorted(found, key=lambda entry: int(entry["year"]))


def _extract_header_from_greenbyte_csv(
    archive: zipfile.ZipFile,
    csv_name: str,
) -> list[str]:
    with archive.open(csv_name) as handle:
        text_stream = io.TextIOWrapper(handle, encoding="utf-8", newline="")
        reader = csv.reader(text_stream)
        for row in reader:
            if not row:
                continue
            first_cell = row[0].strip()
            if first_cell.startswith("#") and "Date" in first_cell:
                return [row[0].lstrip("# ").strip()] + [
                    cell.strip() for cell in row[1:]
                ]
    raise ValueError(f"Missing Greenbyte header row in {csv_name}.")


def _load_turbine_data_from_zip(zip_path: Path) -> list[pd.DataFrame]:
    dataframes = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        turbine_csvs = [
            name
            for name in archive.namelist()
            if f"Turbine_Data_Penmanshiel_{SELECTED_TURBINE}_" in name
            and name.endswith(".csv")
        ]
        if not turbine_csvs:
            raise ValueError(
                f"No WT{SELECTED_TURBINE} turbine CSVs found in {zip_path.name}."
            )
        for csv_name in sorted(turbine_csvs):
            header = _extract_header_from_greenbyte_csv(archive, csv_name)
            with archive.open(csv_name) as handle:
                frame = pd.read_csv(
                    handle,
                    names=header,
                    header=None,
                    comment="#",
                    skip_blank_lines=True,
                    skipinitialspace=True,
                    parse_dates=[header[0]],
                    na_values=["NULL", "null", "NaN", "nan", ""],
                )
            if frame.empty:
                continue
            timestamp_col = frame.columns[0]
            non_numeric = [
                column
                for column in frame.select_dtypes(include=["object"]).columns
                if column != timestamp_col
            ]
            if non_numeric:
                frame = frame.drop(columns=non_numeric)
            for column in frame.columns[1:]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame = frame[frame[timestamp_col].notna()]
            dataframes.append(frame)
    if not dataframes:
        raise ValueError(f"No non-empty WT{SELECTED_TURBINE} frames in {zip_path}.")
    return dataframes


def _apply_leakage_pruning(frame: pd.DataFrame) -> pd.DataFrame:
    columns_info = [(column, column.lower()) for column in frame.columns[1:]]
    columns_lookup = {column.lower(): column for column in frame.columns[1:]}
    to_drop = set()
    for rule in LEAKAGE_RULES.values():
        matched = set()
        for column in rule.get("exact", ()):
            column_key = column.lower()
            if column_key in columns_lookup:
                matched.add(columns_lookup[column_key])
        pattern_lowers = [pattern.lower() for pattern in rule.get("patterns", ())]
        for original, lowered in columns_info:
            if any(pattern in lowered for pattern in pattern_lowers):
                matched.add(original)
        to_drop.update(matched)
    if to_drop:
        return frame.drop(columns=sorted(to_drop))
    return frame


def _summarize_missing_runs(
    series: pd.Series,
    timestamps: pd.Series,
    *,
    freq_minutes: int,
) -> pd.DataFrame:
    mask = series.isna()
    if not mask.any():
        return pd.DataFrame(columns=["start", "end", "rows", "days"])
    groups = (mask != mask.shift()).cumsum()
    missing_groups = groups[mask]
    rows = []
    for _, idx in missing_groups.groupby(missing_groups).groups.items():
        indices = list(idx)
        rows.append(
            {
                "start": timestamps.loc[indices[0]],
                "end": timestamps.loc[indices[-1]],
                "rows": len(indices),
                "days": len(indices) * freq_minutes / 60 / 24,
            }
        )
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


def _summarize_column_gaps(
    frame: pd.DataFrame,
    freq_delta: pd.Timedelta,
) -> pd.DataFrame:
    rows = []
    for column, series in frame.items():
        mask = series.isna()
        if not mask.any():
            rows.append(
                {
                    "column": column,
                    "missing_count": 0,
                    "max_gap_days": 0.0,
                }
            )
            continue
        groups = (mask != mask.shift()).cumsum()
        missing_groups = groups[mask]
        max_gap_rows = int(missing_groups.value_counts().max())
        rows.append(
            {
                "column": column,
                "missing_count": int(mask.sum()),
                "max_gap_days": float(
                    (max_gap_rows * freq_delta) / pd.Timedelta(days=1)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["max_gap_days", "missing_count", "column"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _select_best_target_segment(
    full_index: pd.DatetimeIndex,
    gap_runs: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    long_target_gaps = gap_runs.loc[gap_runs["days"] > TARGET_GAP_MAX_DAYS]
    segments = []
    start = full_index.min()
    for _, gap in long_target_gaps.sort_values("start").iterrows():
        end_before_gap = gap["start"] - pd.Timedelta(minutes=10)
        if end_before_gap >= start:
            segments.append((start, end_before_gap))
        start = gap["end"] + pd.Timedelta(minutes=10)
    if start <= full_index.max():
        segments.append((start, full_index.max()))
    if not segments:
        raise ValueError("No valid WT08 segment remained after target-gap splitting.")
    _, segment_start, segment_end = max(
        [((end - start) / pd.Timedelta(days=1), start, end) for start, end in segments],
        key=lambda item: (item[0], item[2], item[1]),
    )
    return segment_start, segment_end


def _build_target_segment(raw_source: Path) -> tuple[pd.DataFrame, str]:
    frames = []
    for zip_info in _find_scada_zips(raw_source):
        frames.extend(_load_turbine_data_from_zip(zip_info["path"]))
    timestamp_col = frames[0].columns[0]
    raw_frame = pd.concat(frames, ignore_index=True)
    raw_frame = (
        raw_frame.sort_values(timestamp_col)
        .drop_duplicates(subset=[timestamp_col], keep="first")
        .reset_index(drop=True)
    )
    raw_frame = raw_frame.dropna(subset=[timestamp_col])
    raw_frame = _apply_leakage_pruning(raw_frame)

    full_index = pd.date_range(
        raw_frame[timestamp_col].min().floor("10min"),
        raw_frame[timestamp_col].max().ceil("10min"),
        freq="10min",
    )
    raw_grid = (
        raw_frame.set_index(timestamp_col)
        .reindex(full_index)
        .reset_index()
        .rename(columns={"index": timestamp_col})
    )
    power_gap_runs = _summarize_missing_runs(
        raw_grid[TARGET_COLUMN],
        raw_grid[timestamp_col],
        freq_minutes=10,
    )
    segment_start, segment_end = _select_best_target_segment(
        full_index,
        power_gap_runs,
    )
    segment_grid = raw_frame[
        (raw_frame[timestamp_col] >= segment_start)
        & (raw_frame[timestamp_col] <= segment_end)
    ].copy()
    full_segment_index = pd.date_range(segment_start, segment_end, freq="10min")
    segment_grid = (
        segment_grid.set_index(timestamp_col)
        .reindex(full_segment_index)
        .reset_index()
        .rename(columns={"index": timestamp_col})
    )
    return segment_grid, timestamp_col


def _build_hourly_frame(segment_grid: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    numeric_columns = segment_grid.select_dtypes(include=[np.number]).columns.tolist()
    gap_stats = _summarize_column_gaps(
        segment_grid.set_index(timestamp_col)[numeric_columns],
        pd.Timedelta(minutes=10),
    )
    long_gap_columns = sorted(
        gap_stats.loc[
            gap_stats["max_gap_days"] > FEATURE_LONG_GAP_THRESHOLD_DAYS,
            "column",
        ].tolist()
    )
    retained_columns = [
        column for column in numeric_columns if column not in long_gap_columns
    ]
    frame_10min = segment_grid[[timestamp_col] + retained_columns].copy()
    frame_10min[retained_columns] = frame_10min[retained_columns].ffill()

    row_missing = frame_10min[retained_columns].isna().any(axis=1)
    if row_missing.any():
        first_complete_idx = int((~row_missing).idxmax())
        if row_missing.iloc[first_complete_idx]:
            raise ValueError("WT08 has no complete row after forward fill.")
        frame_10min = frame_10min.iloc[first_complete_idx:].reset_index(drop=True)
        frame_10min[retained_columns] = frame_10min[retained_columns].ffill()

    remaining_missing = int(frame_10min[retained_columns].isna().sum().sum())
    if remaining_missing != 0:
        raise ValueError(
            f"WT08 still has {remaining_missing} missing values after forward fill."
        )

    constant_columns = [
        column for column in retained_columns if frame_10min[column].nunique() <= 1
    ]
    if constant_columns:
        frame_10min = frame_10min.drop(columns=constant_columns)
        retained_columns = [
            column for column in retained_columns if column not in constant_columns
        ]
    if TARGET_COLUMN not in retained_columns:
        raise ValueError("WT08 lost the Power (kW) target during preprocessing.")

    series_10min = frame_10min.set_index(timestamp_col)[retained_columns]
    hourly = series_10min.resample("h", label="right", closed="right").mean()
    complete_target_hours = (
        series_10min[TARGET_COLUMN].resample("h", label="right", closed="right").count()
        == 6
    )
    hourly = (
        hourly.loc[complete_target_hours]
        .dropna(how="all")
        .reset_index()
        .rename(columns={timestamp_col: "datetime"})
    )
    if hourly.empty:
        raise ValueError("WT08 preprocessing produced an empty hourly frame.")
    remaining_hourly_missing = int(hourly[retained_columns].isna().sum().sum())
    if remaining_hourly_missing != 0:
        raise ValueError(
            f"WT08 hourly frame still has {remaining_hourly_missing} missing values."
        )

    registered_channels = _registered_channels()
    if retained_columns != registered_channels:
        raise ValueError(
            "WT08 derived channels do not match the registered benchmark contract."
        )
    return hourly[["datetime"] + registered_channels].sort_values("datetime").reset_index(drop=True)


def prepare_from_raw(raw_source: Path, output_path: Path) -> Path:
    segment_grid, timestamp_col = _build_target_segment(raw_source)
    output = _build_hourly_frame(segment_grid, timestamp_col)
    output.to_parquet(output_path, index=False)
    return output_path


def main() -> int:
    return run_contract_cli(
        contract=PENMANSHIEL_HOURLY_WT08,
        prepare_from_raw=prepare_from_raw,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
