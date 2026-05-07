#!/usr/bin/env python3
"""Prepare and validate the Beijing Air Tiantan benchmark dataset."""

from __future__ import annotations

import zipfile
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.benchmark_dataset_contracts import BEIJING_AIR_TIANTAN, run_contract_cli


SELECTED_STATION = "Tiantan"
TARGET_COLUMN = "PM2.5"
TARGET_GAP_SPLIT_HOURS = 24
EXPORT_FEATURE_GAP_HOURS = 72
NUMERIC_CHANNELS = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
]
WD_CODES = {
    "N": 0,
    "NNE": 1,
    "NE": 2,
    "ENE": 3,
    "E": 4,
    "ESE": 5,
    "SE": 6,
    "SSE": 7,
    "S": 8,
    "SSW": 9,
    "SW": 10,
    "WSW": 11,
    "W": 12,
    "WNW": 13,
    "NW": 14,
    "NNW": 15,
}


def _load_tiantan_raw(raw_source: Path) -> pd.DataFrame:
    if raw_source.is_file() and raw_source.suffix.lower() == ".csv":
        if "Tiantan" not in raw_source.name:
            raise ValueError(
                "Beijing Air Tiantan preprocessing requires the Tiantan station CSV."
            )
        return pd.read_csv(raw_source)
    if raw_source.is_file() and raw_source.suffix.lower() == ".zip":
        with zipfile.ZipFile(raw_source) as archive:
            matches = [
                name
                for name in archive.namelist()
                if "PRSA_Data_Tiantan_" in name and name.endswith(".csv")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one Tiantan CSV in {raw_source}, found {matches}."
                )
            with archive.open(matches[0]) as handle:
                return pd.read_csv(handle)
    if raw_source.is_dir():
        csv_matches = sorted(raw_source.rglob("PRSA_Data_Tiantan_*.csv"))
        zip_matches = sorted(raw_source.rglob("*.zip"))
        if len(csv_matches) == 1:
            return pd.read_csv(csv_matches[0])
        if len(csv_matches) > 1:
            raise ValueError(
                f"Expected one Tiantan CSV under {raw_source}, found {csv_matches}."
            )
        if len(zip_matches) == 1:
            return _load_tiantan_raw(zip_matches[0])
        if len(zip_matches) > 1:
            raise ValueError(
                f"Expected one Beijing Air ZIP under {raw_source}, found {zip_matches}."
            )
    raise ValueError(
        "Beijing Air Tiantan raw source must be the Tiantan CSV, the PRSA ZIP, "
        "or a directory containing exactly one of those files."
    )


def _build_station_base(raw_station: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "year",
        "month",
        "day",
        "hour",
        "wd",
        "station",
        *NUMERIC_CHANNELS,
    }
    missing_columns = sorted(required_columns - set(raw_station.columns))
    if missing_columns:
        raise ValueError(f"Tiantan raw source is missing columns: {missing_columns}.")
    stations = sorted(raw_station["station"].dropna().unique().tolist())
    if stations != [SELECTED_STATION]:
        raise ValueError(f"Expected only Tiantan station rows, found {stations}.")

    frame = raw_station.copy()
    frame["datetime"] = pd.to_datetime(frame[["year", "month", "day", "hour"]])
    frame = frame.sort_values("datetime").reset_index(drop=True)
    frame["wd_code"] = frame["wd"].map(WD_CODES)
    unknown_mask = frame["wd"].notna() & frame["wd_code"].isna()
    if unknown_mask.any():
        unknown = sorted(frame.loc[unknown_mask, "wd"].dropna().unique().tolist())
        raise ValueError(f"Unknown Tiantan wind-direction categories: {unknown}.")
    return frame


def _summarize_hourly_gaps(
    series: pd.Series,
    timestamps: pd.Series,
) -> pd.DataFrame:
    mask = series.isna()
    if not mask.any():
        return pd.DataFrame(columns=["start", "end", "rows", "hours", "days"])
    groups = (mask != mask.shift()).cumsum()
    missing_groups = groups[mask]
    rows = []
    for _, idx in missing_groups.groupby(missing_groups).groups.items():
        indices = list(idx)
        rows.append(
            {
                "start": timestamps.iloc[indices[0]],
                "end": timestamps.iloc[indices[-1]],
                "rows": len(indices),
                "hours": len(indices),
                "days": len(indices) / 24,
            }
        )
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


def _build_station_grid(base_frame: pd.DataFrame) -> pd.DataFrame:
    full_index = pd.date_range(
        base_frame["datetime"].min(),
        base_frame["datetime"].max(),
        freq="h",
    )
    return (
        base_frame.set_index("datetime")
        .reindex(full_index)
        .reset_index()
        .rename(columns={"index": "datetime"})
    )


def _build_gap_break_table(grid: pd.DataFrame) -> pd.DataFrame:
    breaks = []
    target_gaps = _summarize_hourly_gaps(grid[TARGET_COLUMN], grid["datetime"])
    for _, row in target_gaps.loc[target_gaps["hours"] > TARGET_GAP_SPLIT_HOURS].iterrows():
        breaks.append(
            {
                "column": TARGET_COLUMN,
                "start": row["start"],
                "end": row["end"],
            }
        )
    for column in [c for c in NUMERIC_CHANNELS + ["wd_code"] if c != TARGET_COLUMN]:
        feature_gaps = _summarize_hourly_gaps(grid[column], grid["datetime"])
        for _, row in feature_gaps.loc[feature_gaps["hours"] > EXPORT_FEATURE_GAP_HOURS].iterrows():
            breaks.append(
                {
                    "column": column,
                    "start": row["start"],
                    "end": row["end"],
                }
            )
    if not breaks:
        return pd.DataFrame(columns=["column", "start", "end"])
    return pd.DataFrame(breaks).sort_values(["start", "end", "column"]).reset_index(drop=True)


def _segments_from_breaks(
    grid: pd.DataFrame,
    breaks: pd.DataFrame,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    regions = (
        breaks.groupby(["start", "end"], dropna=False)["column"]
        .agg(lambda series: ", ".join(sorted(set(series))))
        .reset_index()
        .sort_values(["start", "end"])
    )
    segments = []
    start = grid["datetime"].min()
    for _, row in regions.iterrows():
        end_before_break = row["start"] - pd.Timedelta(hours=1)
        if end_before_break >= start:
            segments.append((start, end_before_break))
        start = max(start, row["end"] + pd.Timedelta(hours=1))
    if start <= grid["datetime"].max():
        segments.append((start, grid["datetime"].max()))
    if not segments:
        raise ValueError("No Tiantan segment remained after applying gap rules.")
    return segments


def _build_processed_frame(
    base_frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    full_index = pd.date_range(start, end, freq="h")
    frame = (
        base_frame.set_index("datetime")
        .reindex(full_index)
        .reset_index()
        .rename(columns={"index": "datetime"})
    )
    fill_columns = NUMERIC_CHANNELS + ["wd_code"]
    frame[fill_columns] = frame[fill_columns].ffill()
    row_missing = frame[fill_columns].isna().any(axis=1)
    if row_missing.any():
        first_complete_idx = int((~row_missing).idxmax())
        if row_missing.iloc[first_complete_idx]:
            raise ValueError("Tiantan has no complete row after forward fill.")
        frame = frame.iloc[first_complete_idx:].reset_index(drop=True)
        frame[fill_columns] = frame[fill_columns].ffill()
    remaining_missing = int(frame[fill_columns].isna().sum().sum())
    if remaining_missing != 0:
        raise ValueError(
            f"Tiantan still has {remaining_missing} missing values after forward fill."
        )
    return (
        frame[["datetime"] + NUMERIC_CHANNELS + ["wd_code"]]
        .rename(columns={"wd_code": "wd"})
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def prepare_from_raw(raw_source: Path, output_path: Path) -> Path:
    raw_station = _load_tiantan_raw(raw_source)
    base_frame = _build_station_base(raw_station)
    grid = _build_station_grid(base_frame)
    breaks = _build_gap_break_table(grid)
    segments = _segments_from_breaks(grid, breaks)
    _, start, end = max(
        [((segment_end - segment_start) / pd.Timedelta(days=1), segment_start, segment_end)
         for segment_start, segment_end in segments],
        key=lambda item: item[0],
    )
    output = _build_processed_frame(base_frame, start, end)
    output.to_parquet(output_path, index=False)
    return output_path


def main() -> int:
    return run_contract_cli(
        contract=BEIJING_AIR_TIANTAN,
        prepare_from_raw=prepare_from_raw,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
