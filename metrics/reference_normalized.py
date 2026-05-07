"""Hendrycks-style reference-normalized diagnostics.

Reference: Hendrycks & Dietterich, 2019
Paper: https://openreview.net/forum?id=HJz6tiCqYm
Repo: https://github.com/hendrycks/robustness/tree/master/ImageNet-C
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.parsing import require_numeric_series


_REQUIRED_GRID_COLUMNS = frozenset(
    {
        "scenario",
        "severity_bin_idx",
        "severity",
        "n_samples",
        "err_pert_mean",
    }
)


@dataclass(frozen=True)
class ReferenceNormalizedDiagnostics:
    scenario_df: pd.DataFrame
    mCE_snaive: float
    relative_mCE_snaive: float
    degref_snaive: float
    mCE_ws_snaive: float
    relative_mCE_ws_snaive: float
    degref_ws_snaive: float
    CE_cell_ws_meansev_snaive: float
    S_cell_ws_meansev_snaive: float
    ce_family_supported: bool
    relative_family_supported: bool
    degref_family_supported: bool
    cell_family_supported: bool


@dataclass(frozen=True)
class ReferenceNormalizedAnchorSummary:
    n_scenarios: int
    n_severity_levels: int
    ce_family_supported: bool
    relative_family_supported: bool
    degref_family_supported: bool
    cell_family_supported: bool


@dataclass(frozen=True)
class _ReferenceNormalizedFamilySupport:
    ce_family_supported: bool
    relative_family_supported: bool
    degref_family_supported: bool
    cell_family_supported: bool


def _require_grid_columns(
    grid_df: pd.DataFrame,
    *,
    context: str,
) -> None:
    missing_cols = sorted(_REQUIRED_GRID_COLUMNS - set(grid_df.columns))
    if missing_cols:
        raise ValueError(f"{context}: missing columns {missing_cols}.")


def _normalize_grid_df(
    grid_df: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    context = f"Reference-normalized diagnostics {label} grid"
    _require_grid_columns(grid_df, context=context)
    if grid_df.empty:
        raise ValueError(f"{context} is empty.")

    normalized = grid_df.loc[:, list(_REQUIRED_GRID_COLUMNS)].copy()
    normalized["scenario"] = normalized["scenario"].astype(str).str.strip()
    if (normalized["scenario"] == "").any():
        raise ValueError(f"{context} has empty scenario values.")

    for column in ("severity_bin_idx", "severity", "n_samples", "err_pert_mean"):
        normalized[column] = require_numeric_series(
            normalized[column],
            column_name=column,
            context=context,
            allow_nan=False,
            allow_infinite=False,
        )

    for int_col in ("severity_bin_idx", "n_samples"):
        if not np.allclose(normalized[int_col], np.floor(normalized[int_col])):
            raise ValueError(f"{context} has non-integer {int_col} values.")
        normalized[int_col] = normalized[int_col].astype(int)

    if (normalized["n_samples"] <= 0).any():
        raise ValueError(f"{context} has non-positive n_samples values.")

    duplicate_mask = normalized.duplicated(["scenario", "severity_bin_idx"], keep=False)
    if duplicate_mask.any():
        examples = (
            normalized.loc[duplicate_mask, ["scenario", "severity_bin_idx"]]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"{context} has duplicate (scenario, severity_bin_idx) rows. Examples: {examples}."
        )

    severity_counts = normalized.groupby("severity_bin_idx", dropna=False)["severity"].nunique()
    inconsistent_severity = severity_counts[severity_counts != 1]
    if not inconsistent_severity.empty:
        bad_indices = [int(idx) for idx in inconsistent_severity.index.tolist()[:5]]
        raise ValueError(
            f"{context} maps severity_bin_idx to multiple severity values. "
            f"Example severity_bin_idx values: {bad_indices}."
        )

    return normalized.sort_values(["scenario", "severity_bin_idx"]).reset_index(drop=True)


def _require_finite_clean_error(value: float, *, label: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(
            f"Reference-normalized diagnostics {label} clean error must be finite; "
            f"got {value!r}."
        )
    return numeric


def _severity_map(grid_df: pd.DataFrame) -> dict[int, float]:
    severity_values = (
        grid_df[["severity_bin_idx", "severity"]]
        .drop_duplicates()
        .sort_values("severity_bin_idx")
    )
    return {
        int(row.severity_bin_idx): float(row.severity)
        for row in severity_values.itertuples(index=False)
    }


def _require_complete_scenario_grid(
    grid_df: pd.DataFrame,
    *,
    label: str,
) -> int:
    n_severity_levels = int(grid_df["severity_bin_idx"].nunique())
    expected_severity_bin_idx = tuple(
        sorted(grid_df["severity_bin_idx"].astype(int).unique())
    )
    scenario_severity_sets = (
        grid_df.groupby("scenario", dropna=False)["severity_bin_idx"]
        .agg(lambda values: tuple(sorted({int(value) for value in values})))
        .reset_index(name="severity_bin_idx_set")
    )
    incomplete_scenarios = scenario_severity_sets[
        scenario_severity_sets["severity_bin_idx_set"] != expected_severity_bin_idx
    ]
    if not incomplete_scenarios.empty:
        examples = (
            incomplete_scenarios.assign(
                expected_severity_bin_idx=[expected_severity_bin_idx]
                * len(incomplete_scenarios)
            )
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Reference-normalized diagnostics require every scenario to contain "
            "the full severity_bin_idx grid. Examples: "
            f"{examples}."
        )
    if n_severity_levels <= 0:
        raise ValueError(
            "Reference-normalized diagnostics require at least one severity level."
        )
    if grid_df["scenario"].nunique() <= 0:
        raise ValueError("Reference-normalized diagnostics require at least one scenario.")
    return n_severity_levels


def _scenario_sums(
    grid_df: pd.DataFrame,
    *,
    value_column: str,
) -> pd.DataFrame:
    scenario_sums = (
        grid_df.groupby("scenario", dropna=False)
        .agg(err_pert_sum_reference=("err_pert_mean", "sum"))
        .reset_index()
        .sort_values("scenario")
        .reset_index(drop=True)
    )
    if scenario_sums.empty:
        raise ValueError(
            "Reference-normalized diagnostics require at least one scenario."
        )
    if value_column != "err_pert_sum_reference":
        scenario_sums = scenario_sums.rename(
            columns={"err_pert_sum_reference": value_column}
        )
    return scenario_sums


def _reference_family_support(
    *,
    reference_sums: pd.Series,
    reference_clean: float,
    n_severity_levels: int,
) -> _ReferenceNormalizedFamilySupport:
    ce_denominator = reference_sums.astype(float)
    relative_denominator = ce_denominator - float(reference_clean)
    reference_degradation = (
        ce_denominator - float(n_severity_levels) * float(reference_clean)
    )
    return _ReferenceNormalizedFamilySupport(
        ce_family_supported=bool((ce_denominator > 0.0).all()),
        relative_family_supported=bool((relative_denominator > 0.0).all()),
        degref_family_supported=bool((reference_degradation > 0.0).all()),
        cell_family_supported=True,
    )


def _require_positive_finite_eps(value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(
            "Reference-normalized diagnostics eps must be finite and strictly positive; "
            f"got {value!r}."
        )
    return numeric


def summarize_reference_normalized_anchor(
    *,
    reference_grid_df: pd.DataFrame,
    reference_clean_error: float,
    reference_label: str,
) -> ReferenceNormalizedAnchorSummary:
    reference_grid = _normalize_grid_df(
        reference_grid_df,
        label=f"reference '{reference_label}'",
    )
    reference_clean = _require_finite_clean_error(
        reference_clean_error,
        label=f"reference '{reference_label}'",
    )
    n_severity_levels = _require_complete_scenario_grid(
        reference_grid,
        label=f"reference '{reference_label}'",
    )
    scenario_sums = _scenario_sums(
        reference_grid,
        value_column="err_pert_sum_reference",
    )
    family_support = _reference_family_support(
        reference_sums=scenario_sums["err_pert_sum_reference"],
        reference_clean=reference_clean,
        n_severity_levels=n_severity_levels,
    )
    return ReferenceNormalizedAnchorSummary(
        n_scenarios=int(scenario_sums["scenario"].nunique()),
        n_severity_levels=n_severity_levels,
        ce_family_supported=family_support.ce_family_supported,
        relative_family_supported=family_support.relative_family_supported,
        degref_family_supported=family_support.degref_family_supported,
        cell_family_supported=family_support.cell_family_supported,
    )


def compute_reference_normalized_diagnostics(
    *,
    target_grid_df: pd.DataFrame,
    target_clean_error: float,
    reference_grid_df: pd.DataFrame,
    reference_clean_error: float,
    eps: float,
    target_label: str,
    reference_label: str,
) -> ReferenceNormalizedDiagnostics:
    """Compute reference-normalized diagnostics from severity-binned scenario summaries."""
    target_grid = _normalize_grid_df(target_grid_df, label=f"target '{target_label}'")
    reference_grid = _normalize_grid_df(
        reference_grid_df,
        label=f"reference '{reference_label}'",
    )
    target_clean = _require_finite_clean_error(
        target_clean_error,
        label=f"target '{target_label}'",
    )
    reference_clean = _require_finite_clean_error(
        reference_clean_error,
        label=f"reference '{reference_label}'",
    )
    eps = _require_positive_finite_eps(eps)

    target_scenarios = set(target_grid["scenario"].tolist())
    reference_scenarios = set(reference_grid["scenario"].tolist())
    if target_scenarios != reference_scenarios:
        raise ValueError(
            "Reference-normalized diagnostics require identical scenario sets. "
            f"Target-only={sorted(target_scenarios - reference_scenarios)}, "
            f"reference-only={sorted(reference_scenarios - target_scenarios)}."
        )

    target_severity_bin_idx = set(target_grid["severity_bin_idx"].tolist())
    reference_severity_bin_idx = set(reference_grid["severity_bin_idx"].tolist())
    if target_severity_bin_idx != reference_severity_bin_idx:
        raise ValueError(
            "Reference-normalized diagnostics require identical severity_bin_idx sets. "
            f"Target-only={sorted(target_severity_bin_idx - reference_severity_bin_idx)}, "
            f"reference-only={sorted(reference_severity_bin_idx - target_severity_bin_idx)}."
        )

    target_severity_map = _severity_map(target_grid)
    reference_severity_map = _severity_map(reference_grid)
    mismatched_severity = [
        severity_bin_idx
        for severity_bin_idx in sorted(target_severity_map)
        if not math.isclose(
            target_severity_map[severity_bin_idx],
            reference_severity_map[severity_bin_idx],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if mismatched_severity:
        raise ValueError(
            "Reference-normalized diagnostics require identical numeric severity values "
            "per severity_bin_idx. "
            f"Mismatched severity_bin_idx values: {mismatched_severity}."
        )

    merged = target_grid.merge(
        reference_grid,
        on=["scenario", "severity_bin_idx"],
        how="outer",
        suffixes=("_target", "_reference"),
        indicator=True,
    )
    if (merged["_merge"] != "both").any():
        examples = (
            merged.loc[
                merged["_merge"] != "both",
                ["scenario", "severity_bin_idx", "_merge"],
            ]
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Reference-normalized diagnostics require identical scenario/severity-bin "
            f"coverage. Examples: {examples}."
        )

    n_samples_mismatch = (
        merged["n_samples_target"].astype(int) != merged["n_samples_reference"].astype(int)
    )
    if n_samples_mismatch.any():
        examples = (
            merged.loc[
                n_samples_mismatch,
                [
                    "scenario",
                    "severity_bin_idx",
                    "n_samples_target",
                    "n_samples_reference",
                ],
            ]
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Reference-normalized diagnostics require matching n_samples per severity bin. "
            f"Examples: {examples}."
        )

    n_severity_levels = _require_complete_scenario_grid(
        target_grid,
        label=f"target '{target_label}'",
    )
    _require_complete_scenario_grid(
        reference_grid,
        label=f"reference '{reference_label}'",
    )
    scenario_sums = (
        merged.groupby("scenario", dropna=False)
        .agg(
            err_pert_sum_target=("err_pert_mean_target", "sum"),
            err_pert_sum_reference=("err_pert_mean_reference", "sum"),
        )
        .reset_index()
        .sort_values("scenario")
        .reset_index(drop=True)
    )
    family_support = _reference_family_support(
        reference_sums=scenario_sums["err_pert_sum_reference"],
        reference_clean=reference_clean,
        n_severity_levels=n_severity_levels,
    )
    # Hendrycks & Dietterich define relative CE on severity sums for one corruption,
    # subtracting the clean error once after aggregating severities.
    relative_denominator = scenario_sums["err_pert_sum_reference"] - reference_clean
    reference_degradation = (
        scenario_sums["err_pert_sum_reference"] - float(n_severity_levels) * reference_clean
    )

    target_degradation = (
        scenario_sums["err_pert_sum_target"] - float(n_severity_levels) * target_clean
    )
    degref = np.where(
        target_degradation <= 0.0,
        1.0,
        np.minimum(1.0, reference_degradation / target_degradation),
    )

    cell_metric_df = (
        merged.assign(
            CE_cell_snaive=(
                merged["err_pert_mean_target"].astype(float) + eps
            )
            / (merged["err_pert_mean_reference"].astype(float) + eps),
            S_cell_snaive=(
                merged["err_pert_mean_reference"].astype(float) + eps
            )
            / (merged["err_pert_mean_target"].astype(float) + eps),
        )
        .groupby("scenario", dropna=False)
        .agg(
            scenario_CE_cell_meansev_snaive=("CE_cell_snaive", "mean"),
            scenario_S_cell_meansev_snaive=("S_cell_snaive", "mean"),
        )
        .reset_index()
        .sort_values("scenario")
        .reset_index(drop=True)
    )
    scenario_df = pd.DataFrame({"scenario": scenario_sums["scenario"].astype(str)})
    scenario_df["scenario_mCE_snaive"] = pd.Series(np.nan, index=scenario_df.index, dtype=float)
    scenario_df["scenario_relative_mCE_snaive"] = pd.Series(
        np.nan,
        index=scenario_df.index,
        dtype=float,
    )
    scenario_df["scenario_degref_snaive"] = pd.Series(
        np.nan,
        index=scenario_df.index,
        dtype=float,
    )
    if family_support.ce_family_supported:
        scenario_df["scenario_mCE_snaive"] = (
            scenario_sums["err_pert_sum_target"] / scenario_sums["err_pert_sum_reference"]
        ).astype(float)
    if family_support.relative_family_supported:
        scenario_df["scenario_relative_mCE_snaive"] = (
            (scenario_sums["err_pert_sum_target"] - target_clean) / relative_denominator
        ).astype(float)
    if family_support.degref_family_supported:
        scenario_df["scenario_degref_snaive"] = pd.Series(degref, dtype=float)
    scenario_df = scenario_df.merge(
        cell_metric_df,
        on="scenario",
        how="left",
        validate="one_to_one",
    )

    def _mean_if_supported(metric_key: str, *, supported: bool) -> float:
        if not supported:
            return float("nan")
        return float(scenario_df[metric_key].mean())

    def _max_if_supported(metric_key: str, *, supported: bool) -> float:
        if not supported:
            return float("nan")
        return float(scenario_df[metric_key].max())

    def _min_if_supported(metric_key: str, *, supported: bool) -> float:
        if not supported:
            return float("nan")
        return float(scenario_df[metric_key].min())

    return ReferenceNormalizedDiagnostics(
        scenario_df=scenario_df,
        mCE_snaive=_mean_if_supported(
            "scenario_mCE_snaive",
            supported=family_support.ce_family_supported,
        ),
        relative_mCE_snaive=_mean_if_supported(
            "scenario_relative_mCE_snaive",
            supported=family_support.relative_family_supported,
        ),
        degref_snaive=_mean_if_supported(
            "scenario_degref_snaive",
            supported=family_support.degref_family_supported,
        ),
        mCE_ws_snaive=_max_if_supported(
            "scenario_mCE_snaive",
            supported=family_support.ce_family_supported,
        ),
        relative_mCE_ws_snaive=_max_if_supported(
            "scenario_relative_mCE_snaive",
            supported=family_support.relative_family_supported,
        ),
        degref_ws_snaive=_min_if_supported(
            "scenario_degref_snaive",
            supported=family_support.degref_family_supported,
        ),
        CE_cell_ws_meansev_snaive=_max_if_supported(
            "scenario_CE_cell_meansev_snaive",
            supported=family_support.cell_family_supported,
        ),
        S_cell_ws_meansev_snaive=_min_if_supported(
            "scenario_S_cell_meansev_snaive",
            supported=family_support.cell_family_supported,
        ),
        ce_family_supported=family_support.ce_family_supported,
        relative_family_supported=family_support.relative_family_supported,
        degref_family_supported=family_support.degref_family_supported,
        cell_family_supported=family_support.cell_family_supported,
    )
