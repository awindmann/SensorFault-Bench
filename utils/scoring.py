from __future__ import annotations

import tempfile
from typing import Any, Mapping, Sequence, TypeVar

import numpy as np
import pandas as pd
from utils.parsing import (
    build_seeded_eval_input_artifact_prefix,
    build_seeded_degradation_artifact_prefix,
    coerce_int,
    parse_bootstrap_ci_confidence_level,
    parse_bootstrap_ci_resamples,
    parse_optional_nonempty_string,
    parse_required_nonnegative_int,
    parse_required_positive_int,
    require_dataframe_columns,
    require_integer_series,
    require_tag_value_with_optional_param_match,
)

_T = TypeVar("_T")
_REQUIRED_DEGRADATION_COLUMNS = (
    "sample_id",
    "source_sample_idx",
    "err_clean",
)
_REQUIRED_DEGRADATION_SCENARIO_SAMPLE_COLUMNS = (
    "sample_id",
    "source_sample_idx",
    "pert_idx",
    "scenario",
    "severity",
    "err_pert",
)
_REQUIRED_DEGRADATION_SCENARIO_SUMMARY_COLUMNS = (
    "pert_idx",
    "scenario",
    "n_test_samples",
    "err_clean_global",
    "err_pert",
    "err_pert_CI_lo",
    "err_pert_CI_hi",
    "D",
    "D_CI_lo",
    "D_CI_hi",
)
_DEGRADATION_RUN_LEVEL_METRICS = (
    "D_w",
    "D_w_CI_lo",
    "D_w_CI_hi",
    "D_mean",
    "D_mean_CI_lo",
    "D_mean_CI_hi",
    "err_pert_ws",
    "err_pert_ws_CI_lo",
    "err_pert_ws_CI_hi",
    "err_pert_mean",
    "err_pert_mean_CI_lo",
    "err_pert_mean_CI_hi",
)
_DEGRADATION_SCENARIO_METRICS = (
    "D",
    "D_CI_lo",
    "D_CI_hi",
    "err_pert",
    "err_pert_CI_lo",
    "err_pert_CI_hi",
)
_DEGRADATION_OVERALL_METRIC_LABELS = (
    "D_w",
    "D_mean",
    "err_pert_ws",
    "err_pert_mean",
)
_DEGRADATION_SCENARIO_METRIC_LABELS = (
    "D",
    "err_pert",
)


def _require_zero_based_contiguous_ids(
    values: Sequence[int],
    *,
    key: str,
    context_name: str,
) -> list[int]:
    """Require a canonical zero-based contiguous integer ID surface."""
    observed_ids = sorted(int(value) for value in values)
    expected_ids = list(range(len(observed_ids)))
    if observed_ids != expected_ids:
        raise ValueError(
            f"{context_name} {key} values must be contiguous zero-based IDs; "
            f"got {observed_ids[:8]}."
        )
    return observed_ids


def build_degradation_metric_prefix(*, test_metric: Any) -> str:
    """Build the literal `degradation/...` metric prefix."""
    metric = parse_optional_nonempty_string(
        test_metric,
        key="test_metric",
        context="`degradation` metric prefix",
        disallow_none_token=True,
    )
    if metric is None:
        raise ValueError("`degradation` metric prefix requires test_metric.")
    return f"degradation/{metric}"


def build_degradation_metric_key(
    *,
    test_metric: Any,
    metric_name: Any,
) -> str:
    """Build one run-level key under the literal `degradation/...` family."""
    prefix = build_degradation_metric_prefix(test_metric=test_metric)
    metric = parse_optional_nonempty_string(
        metric_name,
        key="metric_name",
        context="`degradation` metric key",
        disallow_none_token=True,
    )
    if metric is None:
        raise ValueError("`degradation` metric key requires metric_name.")
    return f"{prefix}/{metric}"


def build_degradation_scenario_metric_key(
    *,
    test_metric: Any,
    scenario_idx: Any,
    metric_name: Any,
) -> str:
    """Build one per-scenario key under the literal `degradation/...` family."""
    prefix = build_degradation_metric_prefix(test_metric=test_metric)
    idx = parse_required_nonnegative_int(
        scenario_idx,
        key="scenario_idx",
        context="`degradation` scenario metric key",
    )
    metric = parse_optional_nonempty_string(
        metric_name,
        key="metric_name",
        context="`degradation` scenario metric key",
        disallow_none_token=True,
    )
    if metric is None:
        raise ValueError("`degradation` scenario metric key requires metric_name.")
    return f"{prefix}/scenario/{idx}/{metric}"


def required_degradation_metric_keys(
    *,
    test_metric: Any,
    scenario_indices: Sequence[int],
) -> tuple[str, ...]:
    """Return all required keys for one literal `degradation/...` metric bundle."""
    metric = parse_optional_nonempty_string(
        test_metric,
        key="test_metric",
        context="`degradation` metric key set",
        disallow_none_token=True,
    )
    if metric is None:
        raise ValueError("`degradation` metric key set requires test_metric.")
    required_keys = [f"{metric}_test"]
    required_keys.extend(
        build_degradation_metric_key(test_metric=metric, metric_name=metric_name)
        for metric_name in _DEGRADATION_RUN_LEVEL_METRICS
    )
    normalized_scenario_indices = {
        parse_required_nonnegative_int(
            raw_idx,
            key="scenario_idx",
            context="`degradation` metric key set",
        )
        for raw_idx in scenario_indices
    }
    for scenario_idx in sorted(normalized_scenario_indices):
        for metric_name in _DEGRADATION_SCENARIO_METRICS:
            required_keys.append(
                build_degradation_scenario_metric_key(
                    test_metric=metric,
                    scenario_idx=scenario_idx,
                    metric_name=metric_name,
                )
            )
    return tuple(required_keys)


def require_logged_degradation_metric_bundle(
    metrics: Mapping[str, object] | None,
    *,
    tags: Mapping[str, object] | None,
    params: Mapping[str, object] | None,
    run_id: str,
    test_metric: str,
    expected_idx_to_name: Mapping[int, str],
) -> str:
    """Require the logged literal `degradation/...` metric family plus worst-scenario label."""
    if metrics is None:
        raise ValueError(
            f"Run {run_id} is missing metrics required for the `degradation` bundle."
        )
    if not isinstance(metrics, Mapping):
        raise ValueError(
            f"Run {run_id} metrics must be a mapping; got {type(metrics).__name__}."
        )
    normalized_idx_to_name = _normalize_expected_idx_to_name(
        expected_idx_to_name,
        context_name=f"Run {run_id} `degradation` metric bundle",
    )
    required_metric_keys = required_degradation_metric_keys(
        test_metric=test_metric,
        scenario_indices=tuple(sorted(normalized_idx_to_name)),
    )
    for metric_key in required_metric_keys:
        require_float_metric(
            metrics,
            run_id=run_id,
            metric_key=metric_key,
        )
    metric_prefix = f"{build_degradation_metric_prefix(test_metric=test_metric)}/"
    unexpected_metric_keys = sorted(
        key_text
        for raw_metric_key in metrics
        for key_text in (str(raw_metric_key).strip(),)
        if key_text.startswith(metric_prefix) and key_text not in required_metric_keys
    )
    if unexpected_metric_keys:
        preview = ", ".join(unexpected_metric_keys[:8])
        if len(unexpected_metric_keys) > 8:
            preview = f"{preview}, ... ({len(unexpected_metric_keys) - 8} more)"
        raise ValueError(
            f"Run {run_id} has unexpected canonical `degradation` metric keys under "
            f"{metric_prefix!r}: {preview}."
        )
    worst_scenario_key = build_degradation_metric_key(
        test_metric=test_metric,
        metric_name="worst_scenario",
    )
    tag_values = {} if tags is None else dict(tags)
    param_values = {} if params is None else dict(params)
    worst_scenario = require_tag_value_with_optional_param_match(
        tag_values,
        param_values,
        key=worst_scenario_key,
        context=f"Run {run_id}",
        disallow_none_token=True,
    )
    allowed_names = set(normalized_idx_to_name.values())
    if worst_scenario not in allowed_names:
        allowed_rendered = ", ".join(sorted(allowed_names))
        raise ValueError(
            f"Run {run_id} has {worst_scenario_key}={worst_scenario!r}, but expected one of "
            f"[{allowed_rendered}]."
        )
    sorted_indices = sorted(normalized_idx_to_name)
    scenario_D_values = np.asarray(
        [
            require_float_metric(
                metrics,
                run_id=run_id,
                metric_key=build_degradation_scenario_metric_key(
                    test_metric=test_metric,
                    scenario_idx=scenario_idx,
                    metric_name="D",
                ),
                output_name=f"scenario/{scenario_idx}/D",
            )
            for scenario_idx in sorted_indices
        ],
        dtype=np.float64,
    )
    scenario_err_values = np.asarray(
        [
            require_float_metric(
                metrics,
                run_id=run_id,
                metric_key=build_degradation_scenario_metric_key(
                    test_metric=test_metric,
                    scenario_idx=scenario_idx,
                    metric_name="err_pert",
                ),
                output_name=f"scenario/{scenario_idx}/err_pert",
            )
            for scenario_idx in sorted_indices
        ],
        dtype=np.float64,
    )
    worst_position = int(np.argmax(scenario_D_values))
    expected_worst_idx = int(sorted_indices[worst_position])
    expected_worst_scenario = normalized_idx_to_name[expected_worst_idx]
    if worst_scenario != expected_worst_scenario:
        raise ValueError(
            f"Run {run_id} has {worst_scenario_key}={worst_scenario!r}, but scenario "
            f"metrics imply worst_scenario={expected_worst_scenario!r}."
        )
    run_level_expectations = {
        "D_w": float(scenario_D_values[worst_position]),
        "D_mean": float(scenario_D_values.mean(dtype=np.float64)),
        "err_pert_ws": float(scenario_err_values[worst_position]),
        "err_pert_mean": float(scenario_err_values.mean(dtype=np.float64)),
    }
    for metric_name, expected_value in run_level_expectations.items():
        actual_value = require_float_metric(
            metrics,
            run_id=run_id,
            metric_key=build_degradation_metric_key(
                test_metric=test_metric,
                metric_name=metric_name,
            ),
            output_name=metric_name,
        )
        if not np.isclose(
            actual_value,
            expected_value,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Run {run_id} has inconsistent {metric_name}={actual_value!r}; expected "
                f"{expected_value!r} from logged per-scenario metrics."
            )
    return worst_scenario


def _normalize_expected_idx_to_name(
    expected_idx_to_name: Mapping[int, str],
    *,
    context_name: str,
) -> dict[int, str]:
    if not expected_idx_to_name:
        raise ValueError(f"{context_name} expected_idx_to_name must be non-empty.")
    normalized: dict[int, str] = {}
    seen_names: set[str] = set()
    for raw_idx, raw_name in expected_idx_to_name.items():
        idx = parse_required_nonnegative_int(
            raw_idx,
            key="pert_idx",
            context=f"{context_name} expected_idx_to_name",
        )
        name = str(raw_name).strip()
        if not name:
            raise ValueError(
                f"{context_name} expected_idx_to_name contains blank scenario name for "
                f"pert_idx={idx}."
            )
        if name in seen_names:
            raise ValueError(
                f"{context_name} expected_idx_to_name contains duplicate scenario "
                f"name {name!r}."
            )
        seen_names.add(name)
        normalized[idx] = name
    expected_ids = list(range(len(normalized)))
    actual_ids = sorted(normalized)
    if actual_ids != expected_ids:
        raise ValueError(
            f"{context_name} expected_idx_to_name must use contiguous zero-based "
            f"indices {expected_ids}; got {actual_ids}."
        )
    return {idx: normalized[idx] for idx in expected_ids}


def _require_finite_float_series(
    df: pd.DataFrame,
    column: str,
    *,
    context_name: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any():
        raise ValueError(
            f"{context_name} column '{column}' contains non-numeric values."
        )
    floats = values.astype(float)
    if not np.isfinite(floats).all():
        raise ValueError(
            f"{context_name} column '{column}' contains non-finite values."
        )
    if min_value is not None and (floats < min_value).any():
        preview = ", ".join(f"{float(v):.6g}" for v in floats[floats < min_value][:8])
        raise ValueError(
            f"{context_name} column '{column}' contains values below {min_value}. "
            f"Examples: {preview}."
        )
    if max_value is not None and (floats > max_value).any():
        preview = ", ".join(f"{float(v):.6g}" for v in floats[floats > max_value][:8])
        raise ValueError(
            f"{context_name} column '{column}' contains values above {max_value}. "
            f"Examples: {preview}."
        )
    return floats


def validate_clean_test_samples(
    df_samples: pd.DataFrame,
    *,
    context_name: str,
) -> pd.DataFrame:
    """Validate canonical clean-test anchor samples and return a normalized frame."""
    require_dataframe_columns(
        df_samples,
        set(_REQUIRED_DEGRADATION_COLUMNS),
        context=context_name,
    )
    if df_samples.empty:
        raise ValueError(f"{context_name} is empty.")
    work = df_samples.loc[:, list(_REQUIRED_DEGRADATION_COLUMNS)].copy()
    work["sample_id"] = require_integer_series(
        work,
        "sample_id",
        context=context_name,
        min_value=0,
    )
    work["source_sample_idx"] = require_integer_series(
        work,
        "source_sample_idx",
        context=context_name,
        min_value=0,
    )
    work["err_clean"] = _require_finite_float_series(
        work,
        "err_clean",
        context_name=context_name,
        min_value=0.0,
    )
    duplicate_ids = work["sample_id"].duplicated()
    if duplicate_ids.any():
        examples = work.loc[duplicate_ids, "sample_id"].head(8).astype(int).tolist()
        raise ValueError(
            f"{context_name} contains duplicate sample_id values. Examples: {examples}."
        )
    _require_zero_based_contiguous_ids(
        work["sample_id"].tolist(),
        key="sample_id",
        context_name=context_name,
    )
    return work.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def validate_degradation_scenario_samples(
    df_samples: pd.DataFrame,
    *,
    expected_idx_to_name: Mapping[int, str],
    context_name: str,
) -> pd.DataFrame:
    """Validate canonical per-scenario degradation sample rows."""
    normalized_idx_to_name = _normalize_expected_idx_to_name(
        expected_idx_to_name,
        context_name=context_name,
    )
    require_dataframe_columns(
        df_samples,
        set(_REQUIRED_DEGRADATION_SCENARIO_SAMPLE_COLUMNS),
        context=context_name,
    )
    if df_samples.empty:
        raise ValueError(f"{context_name} is empty.")
    work = df_samples.loc[:, list(_REQUIRED_DEGRADATION_SCENARIO_SAMPLE_COLUMNS)].copy()
    work["sample_id"] = require_integer_series(
        work,
        "sample_id",
        context=context_name,
        min_value=0,
    )
    work["source_sample_idx"] = require_integer_series(
        work,
        "source_sample_idx",
        context=context_name,
        min_value=0,
    )
    work["pert_idx"] = require_integer_series(
        work,
        "pert_idx",
        context=context_name,
        min_value=0,
    )
    work["severity"] = _require_finite_float_series(
        work,
        "severity",
        context_name=context_name,
        min_value=0.0,
        max_value=1.0,
    )
    work["err_pert"] = _require_finite_float_series(
        work,
        "err_pert",
        context_name=context_name,
        min_value=0.0,
    )
    work["scenario"] = work["scenario"].astype(str).str.strip()
    if (work["scenario"] == "").any():
        raise ValueError(f"{context_name} column 'scenario' contains blank values.")

    expected_ids = list(normalized_idx_to_name)
    observed_ids = sorted(int(value) for value in work["pert_idx"].unique())
    if observed_ids != expected_ids:
        raise ValueError(
            f"{context_name} pert_idx values {observed_ids} do not match expected "
            f"{expected_ids}."
        )
    expected_scenarios = work["pert_idx"].map(normalized_idx_to_name)
    mismatch = expected_scenarios != work["scenario"]
    if mismatch.any():
        examples = (
            work.loc[mismatch, ["pert_idx", "scenario"]]
            .head(8)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"{context_name} scenario labels do not match expected_idx_to_name. "
            f"Examples: {examples}."
        )

    duplicate_pairs = work.duplicated(subset=["sample_id", "pert_idx"], keep=False)
    if duplicate_pairs.any():
        examples = (
            work.loc[duplicate_pairs, ["sample_id", "pert_idx"]]
            .head(8)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"{context_name} contains duplicate (sample_id, pert_idx) rows. "
            f"Examples: {examples}."
        )

    source_counts = (
        work.groupby("sample_id", sort=True)["source_sample_idx"]
        .nunique(dropna=False)
    )
    inconsistent_source = source_counts[source_counts != 1]
    if not inconsistent_source.empty:
        examples = [int(sample_id) for sample_id in inconsistent_source.index[:8]]
        raise ValueError(
            f"{context_name} has sample_id values with inconsistent source_sample_idx "
            f"across scenarios. Examples: {examples}."
        )

    sample_ids = _require_zero_based_contiguous_ids(
        work["sample_id"].unique(),
        key="sample_id",
        context_name=context_name,
    )
    expected_pairs = {
        (sample_id, pert_idx)
        for sample_id in sample_ids
        for pert_idx in expected_ids
    }
    observed_pairs = {
        (int(sample_id), int(pert_idx))
        for sample_id, pert_idx in work[["sample_id", "pert_idx"]].itertuples(index=False)
    }
    if observed_pairs != expected_pairs:
        missing_pairs = sorted(expected_pairs - observed_pairs)[:8]
        extra_pairs = sorted(observed_pairs - expected_pairs)[:8]
        raise ValueError(
            f"{context_name} does not contain exactly one row per (sample_id, pert_idx). "
            f"Missing examples: {missing_pairs}. Unexpected examples: {extra_pairs}."
        )

    return work.sort_values(
        ["sample_id", "pert_idx"],
        kind="mergesort",
    ).reset_index(drop=True)


def validate_degradation_scenario_summary(
    df_summary: pd.DataFrame,
    *,
    expected_idx_to_name: Mapping[int, str],
    expected_n_test_samples: int | None = None,
    context_name: str,
) -> pd.DataFrame:
    """Validate canonical per-scenario degradation summary rows."""
    normalized_idx_to_name = _normalize_expected_idx_to_name(
        expected_idx_to_name,
        context_name=context_name,
    )
    require_dataframe_columns(
        df_summary,
        set(_REQUIRED_DEGRADATION_SCENARIO_SUMMARY_COLUMNS),
        context=context_name,
    )
    if df_summary.empty:
        raise ValueError(f"{context_name} is empty.")
    work = df_summary.loc[:, list(_REQUIRED_DEGRADATION_SCENARIO_SUMMARY_COLUMNS)].copy()
    work["pert_idx"] = require_integer_series(
        work,
        "pert_idx",
        context=context_name,
        min_value=0,
    )
    work["scenario"] = work["scenario"].astype(str).str.strip()
    if (work["scenario"] == "").any():
        raise ValueError(f"{context_name} column 'scenario' contains blank values.")
    work["n_test_samples"] = require_integer_series(
        work,
        "n_test_samples",
        context=context_name,
        min_value=1,
    )
    for column in (
        "err_clean_global",
        "err_pert",
        "err_pert_CI_lo",
        "err_pert_CI_hi",
        "D",
        "D_CI_lo",
        "D_CI_hi",
    ):
        work[column] = _require_finite_float_series(
            work,
            column,
            context_name=context_name,
            min_value=0.0,
        )

    expected_ids = list(normalized_idx_to_name)
    observed_ids = work["pert_idx"].astype(int).tolist()
    if observed_ids != expected_ids:
        raise ValueError(
            f"{context_name} row order {observed_ids} does not match expected "
            f"benchmark order {expected_ids}."
        )
    duplicate_idx = work["pert_idx"].duplicated()
    if duplicate_idx.any():
        examples = work.loc[duplicate_idx, "pert_idx"].head(8).astype(int).tolist()
        raise ValueError(
            f"{context_name} contains duplicate pert_idx rows. Examples: {examples}."
        )
    expected_scenarios = work["pert_idx"].map(normalized_idx_to_name)
    mismatch = expected_scenarios != work["scenario"]
    if mismatch.any():
        examples = (
            work.loc[mismatch, ["pert_idx", "scenario"]]
            .head(8)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"{context_name} scenario labels do not match expected_idx_to_name. "
            f"Examples: {examples}."
        )
    unique_n_test_samples = sorted(int(value) for value in work["n_test_samples"].unique())
    if len(unique_n_test_samples) != 1:
        raise ValueError(
            f"{context_name} must use one canonical n_test_samples value across rows; "
            f"got {unique_n_test_samples}."
        )
    if expected_n_test_samples is not None:
        parsed_expected_n_test_samples = parse_required_positive_int(
            expected_n_test_samples,
            key="expected_n_test_samples",
        )
        mismatch_n = work["n_test_samples"] != parsed_expected_n_test_samples
        if mismatch_n.any():
            examples = (
                work.loc[mismatch_n, ["pert_idx", "n_test_samples"]]
                .head(8)
                .to_dict(orient="records")
            )
            raise ValueError(
                f"{context_name} n_test_samples does not match expected "
                f"{parsed_expected_n_test_samples}. Examples: {examples}."
            )
    err_clean_global = work["err_clean_global"].to_numpy(dtype=float)
    if not np.allclose(
        err_clean_global,
        err_clean_global[0],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{context_name} err_clean_global must be constant across rows."
        )
    for metric_name in ("err_pert", "D"):
        lo = work[f"{metric_name}_CI_lo"].to_numpy(dtype=float)
        value = work[metric_name].to_numpy(dtype=float)
        hi = work[f"{metric_name}_CI_hi"].to_numpy(dtype=float)
        if np.any((lo > value) | (value > hi)):
            raise ValueError(
                f"{context_name} has invalid {metric_name} confidence interval ordering."
            )
    return work.reset_index(drop=True)


def validate_degradation_artifact_bundle(
    clean_df: pd.DataFrame,
    scenario_samples_df: pd.DataFrame,
    scenario_summary_df: pd.DataFrame,
    *,
    expected_idx_to_name: Mapping[int, str],
    expected_n_test_samples: int | None = None,
    context_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate the canonical degradation artifact trio and their shared sample IDs."""
    clean = validate_clean_test_samples(
        clean_df,
        context_name=f"{context_name} clean_test_samples",
    )
    scenario_samples = validate_degradation_scenario_samples(
        scenario_samples_df,
        expected_idx_to_name=expected_idx_to_name,
        context_name=f"{context_name} scenario_samples",
    )
    resolved_n_test_samples = (
        parse_required_positive_int(expected_n_test_samples, key="expected_n_test_samples")
        if expected_n_test_samples is not None
        else int(len(clean))
    )
    scenario_summary = validate_degradation_scenario_summary(
        scenario_summary_df,
        expected_idx_to_name=expected_idx_to_name,
        expected_n_test_samples=resolved_n_test_samples,
        context_name=f"{context_name} scenario_summary",
    )
    if len(clean) != resolved_n_test_samples:
        raise ValueError(
            f"{context_name} clean_test_samples has {len(clean)} rows but expected "
            f"n_test_samples={resolved_n_test_samples}."
        )
    clean_sample_ids = clean["sample_id"].astype(int).tolist()
    scenario_sample_ids = sorted(
        int(sample_id) for sample_id in scenario_samples["sample_id"].unique()
    )
    if scenario_sample_ids != clean_sample_ids:
        raise ValueError(
            f"{context_name} scenario_samples sample IDs {scenario_sample_ids[:8]} do not "
            f"match clean_test_samples sample IDs {clean_sample_ids[:8]}."
        )
    if len(scenario_sample_ids) != resolved_n_test_samples:
        raise ValueError(
            f"{context_name} scenario_samples covers {len(scenario_sample_ids)} sample_id "
            f"values but expected n_test_samples={resolved_n_test_samples}."
        )
    clean_source_by_id = clean.set_index("sample_id")["source_sample_idx"]
    scenario_source_by_id = (
        scenario_samples[["sample_id", "source_sample_idx"]]
        .drop_duplicates()
        .set_index("sample_id")["source_sample_idx"]
    )
    if not clean_source_by_id.equals(scenario_source_by_id):
        mismatch_ids = [
            int(sample_id)
            for sample_id in clean_source_by_id.index
            if sample_id not in scenario_source_by_id.index
            or int(clean_source_by_id.loc[sample_id]) != int(scenario_source_by_id.loc[sample_id])
        ][:8]
        raise ValueError(
            f"{context_name} source_sample_idx values do not align between clean_test_samples "
            f"and scenario_samples. Examples: {mismatch_ids}."
        )
    expected_err_clean_global = float(clean["err_clean"].mean())
    actual_err_clean_global = float(scenario_summary["err_clean_global"].iloc[0])
    if not np.isclose(
        actual_err_clean_global,
        expected_err_clean_global,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{context_name} scenario_summary err_clean_global={actual_err_clean_global!r} "
            f"does not match clean_test_samples mean err_clean={expected_err_clean_global!r}."
        )
    return clean, scenario_samples, scenario_summary


def download_validated_degradation_artifact_bundle(
    client: Any,
    *,
    run_id: str,
    test_metric: str,
    eval_data_seed: int,
    expected_idx_to_name: Mapping[int, str],
    expected_n_test_samples: int | None = None,
    expected_clean_metric_value: float | None = None,
    context_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download and validate the canonical degradation artifact trio for one run."""
    eval_input_prefix = build_seeded_eval_input_artifact_prefix(
        test_metric=test_metric,
        eval_data_seed=eval_data_seed,
    )
    degradation_prefix = build_seeded_degradation_artifact_prefix(
        test_metric=test_metric,
        eval_data_seed=eval_data_seed,
    )
    artifact_paths = {
        "clean_test_samples.csv": f"{eval_input_prefix}/clean_test_samples.csv",
        "scenario_samples.csv": f"{degradation_prefix}/scenario_samples.csv",
        "scenario_summary.csv": f"{degradation_prefix}/scenario_summary.csv",
    }
    loaded_frames: dict[str, pd.DataFrame] = {}
    with tempfile.TemporaryDirectory(prefix="robust-") as tmpdir:
        for artifact_name, artifact_path in artifact_paths.items():
            try:
                local_path = client.download_artifacts(
                    run_id,
                    artifact_path,
                    dst_path=tmpdir,
                )
            except Exception as exc:
                raise FileNotFoundError(
                    f"{context_name} failed to download required artifact "
                    f"{artifact_name!r} from run {run_id}: {artifact_path}."
                ) from exc
            try:
                loaded_frames[artifact_name] = pd.read_csv(local_path)
            except Exception as exc:
                raise ValueError(
                    f"{context_name} could not read downloaded artifact "
                    f"{artifact_name!r} for run {run_id}."
                ) from exc
    clean, scenario_samples, scenario_summary = validate_degradation_artifact_bundle(
        loaded_frames["clean_test_samples.csv"],
        loaded_frames["scenario_samples.csv"],
        loaded_frames["scenario_summary.csv"],
        expected_idx_to_name=expected_idx_to_name,
        expected_n_test_samples=expected_n_test_samples,
        context_name=context_name,
    )
    if expected_clean_metric_value is not None:
        clean_metric_value = float(expected_clean_metric_value)
        bundle_clean_mean = float(clean["err_clean"].mean())
        if not np.isclose(
            clean_metric_value,
            bundle_clean_mean,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{context_name} clean metric mismatch: run metric={clean_metric_value!r} "
                f"but bundle mean err_clean={bundle_clean_mean!r}."
            )
    return clean, scenario_samples, scenario_summary


def _require_degradation_denominator(
    clean_mean: float,
    *,
    context_name: str,
) -> float:
    value = float(clean_mean)
    if not np.isfinite(value):
        raise ValueError(f"{context_name} clean-error denominator must be finite.")
    if value <= 0.0:
        raise ValueError(
            f"{context_name} clean-error denominator must be > 0; got {value!r}."
        )
    return value


def _degradation_ci_bounds(
    values: np.ndarray,
    *,
    confidence_level: float,
) -> tuple[float, float]:
    tail = (1.0 - float(confidence_level)) / 2.0
    quantiles = np.quantile(
        np.asarray(values, dtype=np.float64),
        [tail, 1.0 - tail],
        method="linear",
    )
    return float(quantiles[0]), float(quantiles[1])


def score_degradation_artifact_bundle(
    clean_df: pd.DataFrame,
    scenario_samples_df: pd.DataFrame,
    *,
    expected_idx_to_name: Mapping[int, str],
    bootstrap_resamples: int,
    bootstrap_confidence_level: float,
    bootstrap_seed: int,
    context_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float], str]:
    """Score one validated `degradation` artifact bundle with shared-anchor bootstrap."""
    normalized_idx_to_name = _normalize_expected_idx_to_name(
        expected_idx_to_name,
        context_name=context_name,
    )
    clean = validate_clean_test_samples(
        clean_df,
        context_name=f"{context_name} clean_test_samples",
    )
    scenario_samples = validate_degradation_scenario_samples(
        scenario_samples_df,
        expected_idx_to_name=normalized_idx_to_name,
        context_name=f"{context_name} scenario_samples",
    )
    n_test_samples = int(len(clean))
    n_scenarios = int(len(normalized_idx_to_name))
    if len(scenario_samples) != n_test_samples * n_scenarios:
        raise ValueError(
            f"{context_name} scenario_samples has {len(scenario_samples)} rows but expected "
            f"{n_test_samples * n_scenarios} rows for {n_test_samples} anchors and "
            f"{n_scenarios} scenarios."
        )

    clean_errors = clean["err_clean"].to_numpy(dtype=np.float64)
    clean_mean = _require_degradation_denominator(
        float(clean_errors.mean(dtype=np.float64)),
        context_name=context_name,
    )
    scenario_err_matrix = scenario_samples["err_pert"].to_numpy(dtype=np.float64).reshape(
        n_test_samples,
        n_scenarios,
    )
    scenario_err_means = scenario_err_matrix.mean(axis=0, dtype=np.float64)
    scenario_D_values = scenario_err_means / clean_mean
    worst_scenario_idx = int(np.argmax(scenario_D_values))
    worst_scenario_name = normalized_idx_to_name[worst_scenario_idx]
    err_pert_ws = float(scenario_err_means[worst_scenario_idx])
    err_pert_mean = float(scenario_err_means.mean(dtype=np.float64))
    D_mean = float(scenario_D_values.mean(dtype=np.float64))
    D_w = float(scenario_D_values[worst_scenario_idx])

    parsed_bootstrap_resamples = parse_bootstrap_ci_resamples(
        bootstrap_resamples,
        key="bootstrap_resamples",
    )
    if parsed_bootstrap_resamples is None:
        raise ValueError("bootstrap_resamples is required for `degradation` scoring.")
    confidence_level = parse_bootstrap_ci_confidence_level(
        bootstrap_confidence_level,
        key="bootstrap_confidence_level",
    )
    if confidence_level is None:
        raise ValueError(
            "bootstrap_confidence_level is required for `degradation` scoring."
        )
    parsed_bootstrap_seed = coerce_int(bootstrap_seed)
    if parsed_bootstrap_seed is None:
        raise ValueError("bootstrap_seed must be an integer for `degradation` scoring.")

    rng = np.random.default_rng(int(parsed_bootstrap_seed))
    n_resamples = int(parsed_bootstrap_resamples)
    bootstrap_run_draws = {
        "D_w": np.empty(n_resamples, dtype=np.float64),
        "D_mean": np.empty(n_resamples, dtype=np.float64),
        "err_pert_ws": np.empty(n_resamples, dtype=np.float64),
        "err_pert_mean": np.empty(n_resamples, dtype=np.float64),
    }
    bootstrap_scenario_D = np.empty((n_resamples, n_scenarios), dtype=np.float64)
    bootstrap_scenario_err = np.empty((n_resamples, n_scenarios), dtype=np.float64)
    for draw_idx in range(n_resamples):
        anchor_idx = rng.integers(0, n_test_samples, size=n_test_samples)
        boot_clean_mean = _require_degradation_denominator(
            float(clean_errors[anchor_idx].mean(dtype=np.float64)),
            context_name=f"{context_name} bootstrap draw {draw_idx}",
        )
        boot_scenario_err_means = scenario_err_matrix[anchor_idx].mean(
            axis=0,
            dtype=np.float64,
        )
        boot_scenario_D = boot_scenario_err_means / boot_clean_mean
        boot_worst_idx = int(np.argmax(boot_scenario_D))

        bootstrap_scenario_D[draw_idx, :] = boot_scenario_D
        bootstrap_scenario_err[draw_idx, :] = boot_scenario_err_means
        bootstrap_run_draws["D_w"][draw_idx] = float(boot_scenario_D[boot_worst_idx])
        bootstrap_run_draws["D_mean"][draw_idx] = float(
            boot_scenario_D.mean(dtype=np.float64)
        )
        bootstrap_run_draws["err_pert_ws"][draw_idx] = float(
            boot_scenario_err_means[boot_worst_idx]
        )
        bootstrap_run_draws["err_pert_mean"][draw_idx] = float(
            boot_scenario_err_means.mean(dtype=np.float64)
        )

    scenario_rows: list[dict[str, float | int | str]] = []
    metric_bundle: dict[str, float] = {
        "D_w": D_w,
        "D_mean": D_mean,
        "err_pert_ws": err_pert_ws,
        "err_pert_mean": err_pert_mean,
    }
    for metric_name in ("D_w", "D_mean", "err_pert_ws", "err_pert_mean"):
        ci_lo, ci_hi = _degradation_ci_bounds(
            bootstrap_run_draws[metric_name],
            confidence_level=float(confidence_level),
        )
        metric_bundle[f"{metric_name}_CI_lo"] = ci_lo
        metric_bundle[f"{metric_name}_CI_hi"] = ci_hi

    for pert_idx in range(n_scenarios):
        scenario_name = normalized_idx_to_name[pert_idx]
        err_pert_value = float(scenario_err_means[pert_idx])
        D_value = float(scenario_D_values[pert_idx])
        err_ci_lo, err_ci_hi = _degradation_ci_bounds(
            bootstrap_scenario_err[:, pert_idx],
            confidence_level=float(confidence_level),
        )
        D_ci_lo, D_ci_hi = _degradation_ci_bounds(
            bootstrap_scenario_D[:, pert_idx],
            confidence_level=float(confidence_level),
        )
        scenario_rows.append(
            {
                "pert_idx": int(pert_idx),
                "scenario": scenario_name,
                "n_test_samples": int(n_test_samples),
                "err_clean_global": float(clean_mean),
                "err_pert": err_pert_value,
                "err_pert_CI_lo": err_ci_lo,
                "err_pert_CI_hi": err_ci_hi,
                "D": D_value,
                "D_CI_lo": D_ci_lo,
                "D_CI_hi": D_ci_hi,
            }
        )
        metric_bundle[f"scenario/{pert_idx}/D"] = D_value
        metric_bundle[f"scenario/{pert_idx}/D_CI_lo"] = D_ci_lo
        metric_bundle[f"scenario/{pert_idx}/D_CI_hi"] = D_ci_hi
        metric_bundle[f"scenario/{pert_idx}/err_pert"] = err_pert_value
        metric_bundle[f"scenario/{pert_idx}/err_pert_CI_lo"] = err_ci_lo
        metric_bundle[f"scenario/{pert_idx}/err_pert_CI_hi"] = err_ci_hi

    scenario_summary = pd.DataFrame(
        scenario_rows,
        columns=list(_REQUIRED_DEGRADATION_SCENARIO_SUMMARY_COLUMNS),
    )
    clean, scenario_samples, scenario_summary = validate_degradation_artifact_bundle(
        clean,
        scenario_samples,
        scenario_summary,
        expected_idx_to_name=normalized_idx_to_name,
        expected_n_test_samples=n_test_samples,
        context_name=context_name,
    )
    return clean, scenario_samples, scenario_summary, metric_bundle, worst_scenario_name


def require_float_metric(
    metrics: Mapping[str, object],
    *,
    run_id: str,
    metric_key: str,
    output_name: str | None = None,
) -> float:
    """Require a numeric metric value from an MLflow metric mapping."""
    raw_value = metrics.get(metric_key)
    if raw_value is None:
        detail = (
            f" for output field '{output_name}'"
            if output_name is not None
            else ""
        )
        raise ValueError(
            f"Run {run_id} is missing required robustness metric '{metric_key}'{detail}."
        )
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Run {run_id} has non-numeric robustness metric '{metric_key}'={raw_value!r}."
        ) from exc
    if not np.isfinite(value):
        raise ValueError(
            f"Run {run_id} has non-finite robustness metric '{metric_key}'={value!r}."
    )
    return value


def extract_required_overall_degradation_metrics(
    metrics: Mapping[str, object],
    *,
    run_id: str,
    test_metric: str,
) -> dict[str, float]:
    """Extract the canonical overall degradation metric family from MLflow metrics."""
    bundle: dict[str, float] = {}
    for metric_name in _DEGRADATION_OVERALL_METRIC_LABELS:
        bundle[metric_name] = require_float_metric(
            metrics,
            run_id=run_id,
            metric_key=build_degradation_metric_key(
                test_metric=test_metric,
                metric_name=metric_name,
            ),
            output_name=metric_name,
        )
        bundle[f"{metric_name}_CI_lo"] = require_float_metric(
            metrics,
            run_id=run_id,
            metric_key=build_degradation_metric_key(
                test_metric=test_metric,
                metric_name=f"{metric_name}_CI_lo",
            ),
            output_name=f"{metric_name}_CI_lo",
        )
        bundle[f"{metric_name}_CI_hi"] = require_float_metric(
            metrics,
            run_id=run_id,
            metric_key=build_degradation_metric_key(
                test_metric=test_metric,
                metric_name=f"{metric_name}_CI_hi",
            ),
            output_name=f"{metric_name}_CI_hi",
        )
    return bundle


def extract_required_degradation_scenario_metrics(
    metrics: Mapping[str, object],
    *,
    run_id: str,
    test_metric: str,
    scenario_idx: int,
) -> dict[str, float]:
    """Extract the canonical per-scenario degradation metric family."""
    scenario_bundle: dict[str, float] = {}
    for metric_name in _DEGRADATION_SCENARIO_METRIC_LABELS:
        scenario_bundle[metric_name] = require_float_metric(
            metrics,
            run_id=run_id,
            metric_key=build_degradation_scenario_metric_key(
                test_metric=test_metric,
                scenario_idx=scenario_idx,
                metric_name=metric_name,
            ),
            output_name=metric_name,
        )
        scenario_bundle[f"{metric_name}_CI_lo"] = require_float_metric(
            metrics,
            run_id=run_id,
            metric_key=build_degradation_scenario_metric_key(
                test_metric=test_metric,
                scenario_idx=scenario_idx,
                metric_name=f"{metric_name}_CI_lo",
            ),
            output_name=f"{metric_name}_CI_lo",
        )
        scenario_bundle[f"{metric_name}_CI_hi"] = require_float_metric(
            metrics,
            run_id=run_id,
            metric_key=build_degradation_scenario_metric_key(
                test_metric=test_metric,
                scenario_idx=scenario_idx,
                metric_name=f"{metric_name}_CI_hi",
            ),
            output_name=f"{metric_name}_CI_hi",
        )
    return scenario_bundle


def extract_required_degradation_scenario_metric_with_ci(
    metrics: Mapping[str, object],
    *,
    run_id: str,
    test_metric: str,
    scenario_idx: int,
    metric_name: str,
) -> tuple[float, float, float]:
    """Extract one required per-scenario degradation metric and its CI siblings."""
    value = require_float_metric(
        metrics,
        run_id=run_id,
        metric_key=build_degradation_scenario_metric_key(
            test_metric=test_metric,
            scenario_idx=scenario_idx,
            metric_name=metric_name,
        ),
        output_name=metric_name,
    )
    ci_lo = require_float_metric(
        metrics,
        run_id=run_id,
        metric_key=build_degradation_scenario_metric_key(
            test_metric=test_metric,
            scenario_idx=scenario_idx,
            metric_name=f"{metric_name}_CI_lo",
        ),
        output_name=f"{metric_name}_CI_lo",
    )
    ci_hi = require_float_metric(
        metrics,
        run_id=run_id,
        metric_key=build_degradation_scenario_metric_key(
            test_metric=test_metric,
            scenario_idx=scenario_idx,
            metric_name=f"{metric_name}_CI_hi",
        ),
        output_name=f"{metric_name}_CI_hi",
    )
    return value, ci_lo, ci_hi


def partition_metric_bundle(
    metric_bundle: Mapping[str, _T],
) -> tuple[dict[str, _T], dict[int, dict[str, _T]]]:
    global_metrics: dict[str, _T] = {}
    scenario_metrics: dict[int, dict[str, _T]] = {}
    for metric_name, metric_value in metric_bundle.items():
        if not metric_name.startswith("scenario/"):
            global_metrics[str(metric_name)] = metric_value
            continue
        parts = str(metric_name).split("/", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid scenario metric key '{metric_name}'.")
        scope, pert_idx_token, scenario_metric_name = parts
        if scope != "scenario" or not scenario_metric_name:
            raise ValueError(f"Invalid scenario metric key '{metric_name}'.")
        try:
            pert_idx = int(pert_idx_token)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid scenario metric key '{metric_name}'.") from exc
        scenario_metrics.setdefault(pert_idx, {})[scenario_metric_name] = metric_value
    return global_metrics, {
        pert_idx: scenario_metrics[pert_idx] for pert_idx in sorted(scenario_metrics)
    }
