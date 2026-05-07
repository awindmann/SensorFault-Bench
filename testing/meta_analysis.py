from __future__ import annotations

import os
import tempfile
import warnings
from functools import lru_cache
from dataclasses import dataclass
import numpy as np
import pandas as pd
import yaml
from typing import Any, Callable, Mapping, Optional, Sequence
from scipy.stats import pearsonr
import mlflow

from config_loader import (
    load_dataset_windows,
    load_defaults,
    load_parsed_core_figure_registry,
    parse_explicit_cli_overrides,
)
from data.datasets import DATASET_REGISTRY, resolve_with_defaults
from data.perturbations import (
    PERTURBATION_REGISTRY,
    require_perturbation_channel_scope,
)
from metrics.reference_normalized import (
    compute_reference_normalized_diagnostics,
    summarize_reference_normalized_anchor,
)
from pipelines.ranking import (
    ALLOWED_PIPELINE_KINDS,
    perturbed_selection_metric_keys,
    resolve_selection_score,
    selection_metric_key_for_kind,
)
from pipelines.recipes import PIPELINE_RECIPE_PATHS_BY_METHOD
from pipelines.selection import (
    expected_perturbation_coupling_from_args,
    is_fully_tested,
    is_fixed_channel_fraction_complete,
    load_recipe_specs_for_scope,
    extract_recipe_defaults_for_scope,
    merge_recipe_defaults_for_scope,
    require_seed_tags,
    require_matching_perturbation_coupling_params,
    require_run_perturbation_idx_name_map,
    resolve_pipeline_tags,
)
from testing.evaluation import (
    _build_winner_selection_provenance_tag_payload_for_run,
    _build_testing_datamodule,
    _collect_degradation_forecast_samples,
    _prime_model_for_degradation_evaluation,
    _resolve_dataset_testing_coverage_scope,
    _resolve_model_loading_identity_for_run,
    _resolve_requested_runtime_device,
    _teardown_model_after_eval,
    load_model_with_loader,
)
from testing.shared import (
    _raise_non_finished_winner,
    _require_best_model_current_tags,
)
from utils.parsing import (
    DEGRADATION_SCORING_SEMANTICS,
    METHOD_DELTA_PAIR_BOOTSTRAP_CI_SEMANTICS,
    assert_no_duplicate_rows,
    build_method_delta_pair_bootstrap_ci_seed_key,
    build_metric_w_name,
    build_mlflow_tracking_uri,
    build_shared_anchor_bootstrap_ci_tag_payload,
    build_degradation_eval_context_tag_payload,
    coerce_int,
    normalize_yaml_value,
    parse_backbone_run_ids,
    parse_bootstrap_ci_confidence_level,
    parse_optional_unit_float,
    parse_optional_nonempty_string,
    parse_perturbation_channel_fraction_max,
    CoreFigureRegistry,
    parse_required_positive_int,
    parse_reference_normalization_anchor_model,
    parse_required_nonempty_string,
    require_eval_data_seed_tag,
    require_shared_anchor_bootstrap_ci_context_tags,
    require_degradation_eval_context_tags,
    require_dataframe_columns,
    require_improvement_selection_mode,
    require_integer_series,
    require_namespace_bool,
    require_namespace_nonempty_string,
    require_namespace_value,
    require_nonempty_tag_value,
    require_nonempty_string_series,
    require_numeric_series,
    require_stage_tag,
    require_tag_value_with_optional_param_match,
    require_tested_param,
    require_winner_selection_provenance_tags,
    resolve_meta_analysis_eval_data_seed_scope,
    sample_dataframe_records,
    resolve_dataset_window_args,
    tag_is_truthy,
    validate_scoped_raw_display_id_values,
    validate_raw_display_id_values,
)
from utils.scoring import (
    build_canonical_degradation_context_signature,
    build_degradation_metric_key,
    build_degradation_metric_prefix,
    build_fixed_channel_fraction_metric_key,
    build_fixed_channel_fraction_tag_key,
    download_validated_degradation_artifact_bundle,
    download_validated_fixed_channel_fraction_artifact_bundle,
    extract_required_degradation_scenario_metrics,
    extract_required_overall_degradation_metrics,
    require_float_metric,
    require_logged_degradation_metric_bundle,
    require_logged_fixed_channel_fraction_metric_bundle,
)
from utils.rng import derive_seed
from visualizations.semantics import (
    PlotSemanticsRecord,
    require_plot_semantics_mapping,
)
from visualizations.plots import (
    plot_error_distribution_overview,
    plot_forecast_extreme,
    plot_perturbation_curves,
    plot_heatmap,
    plot_pareto,
    plot_improvement_deltas_heatmap,
    plot_ranked_performance_robustness_pareto,
    plot_ranked_performance_robustness_pareto_panels,
    plot_pareto_dataset_panels,
    plot_perturbed_vs_clean_error,
    plot_perturbed_vs_clean_error_panels,
    plot_improvement_trajectory_subplots,
    plot_method_delta_pair_subplots,
    plot_improvement_comparison,
    plot_per_method_delta_scatter,
    plot_method_win_rate_heatmap,
    plot_selection_margin,
    plot_scenario_delta_heatmap,
    plot_method_scenario_delta_heatmap,
    robustness_metric_display_name,
    trajectory_output_label_for_method,
    plot_testing_coverage_heatmap,
)


@dataclass(frozen=True)
class _CoreRobustnessMetricSpec:
    metric_key: str
    higher_is_better: bool
    win_flag_col: str
    win_context_name: str
    win_rate_table_name: str
    win_rate_table_filename: str
    win_rate_figure_filename: str
    win_rate_figure_type: str


CORE_ROBUSTNESS_METRIC_SPECS: tuple[_CoreRobustnessMetricSpec, ...] = (
    _CoreRobustnessMetricSpec(
        metric_key="D_w",
        higher_is_better=False,
        win_flag_col="_wins_perf_dw",
        win_context_name="D_w",
        win_rate_table_name="method_win_rate",
        win_rate_table_filename="method_win_rate.csv",
        win_rate_figure_filename="method_win_rate_perf_and_dw.pdf",
        win_rate_figure_type="improvement_method_win_rate_dw",
    ),
    _CoreRobustnessMetricSpec(
        metric_key="D_mean",
        higher_is_better=False,
        win_flag_col="_wins_perf_dmean",
        win_context_name="D_mean",
        win_rate_table_name="method_win_rate_d_mean",
        win_rate_table_filename="method_win_rate_d_mean.csv",
        win_rate_figure_filename="method_win_rate_perf_and_d_mean.pdf",
        win_rate_figure_type="improvement_method_win_rate_d_mean",
    ),
    _CoreRobustnessMetricSpec(
        metric_key="err_pert_ws",
        higher_is_better=False,
        win_flag_col="_wins_perf_err_pert_ws",
        win_context_name="err_pert_ws",
        win_rate_table_name="method_win_rate_err_pert_ws",
        win_rate_table_filename="method_win_rate_err_pert_ws.csv",
        win_rate_figure_filename="method_win_rate_perf_and_err_pert_ws.pdf",
        win_rate_figure_type="improvement_method_win_rate_err_pert_ws",
    ),
)

OPTIONAL_DIAGNOSTIC_METRIC_SPECS: tuple[_CoreRobustnessMetricSpec, ...] = ()

REFERENCE_NORMALIZED_DIAGNOSTIC_METRIC_KEYS: tuple[str, ...] = (
    "mCE_snaive",
    "relative_mCE_snaive",
)

PIPELINE_METHOD_DELTA_RESULTS_COLUMNS: tuple[str, ...] = (
    "dataset",
    "robustness_method",
    "count",
    "delta_bootstrap_semantics",
    "delta_bootstrap_resamples",
    "delta_bootstrap_confidence_level",
    "delta_bootstrap_seed",
    "delta_err_clean_mean",
    "delta_err_clean_CI_lo",
    "delta_err_clean_CI_hi",
    "delta_D_w_mean",
    "delta_D_w_CI_lo",
    "delta_D_w_CI_hi",
    "delta_D_mean_mean",
    "delta_D_mean_CI_lo",
    "delta_D_mean_CI_hi",
    "delta_err_pert_ws_mean",
    "delta_err_pert_ws_CI_lo",
    "delta_err_pert_ws_CI_hi",
    "delta_err_pert_mean_mean",
    "delta_err_pert_mean_CI_lo",
    "delta_err_pert_mean_CI_hi",
)

METHOD_SCENARIO_FAMILY_DELTA_COLUMNS: tuple[str, ...] = (
    "dataset",
    "dataset_label",
    "robustness_method",
    "method_label",
    "scenario_family",
    "family_scenarios",
    "family_scenarios_display",
    "architecture_count",
    "scenario_count",
    "improved_scenario_count",
    "baseline_family_D_mean",
    "method_family_delta_D_mean",
    "baseline_family_D_rank_desc",
    "method_family_gain_rank_asc",
    "effect_direction",
)

METHOD_SCENARIO_FAMILY_SUMMARY_COLUMNS: tuple[str, ...] = (
    "robustness_method",
    "method_label",
    "scenario_family",
    "method_family_delta_rank",
    "family_scenarios",
    "family_scenarios_display",
    "dataset_count",
    "improved_dataset_count",
    "baseline_family_D_mean",
    "method_family_delta_D_mean",
    "method_family_delta_D_min",
    "method_family_delta_D_max",
    "baseline_impact_dataset_order",
    "largest_gain_dataset_order",
    "effect_direction",
    "practitioner_frame",
)

SCENARIO_REFERENCE_NORMALIZED_DIAGNOSTIC_METRIC_KEYS: tuple[str, ...] = ()

REFERENCE_NORMALIZATION_GROUP_COLS: tuple[str, ...] = (
    "dataset",
    "data_config_signature",
    "eval_data_seed",
    "test_metric",
)

REFERENCE_NORMALIZATION_FAMILY_SUPPORT_COLUMNS: tuple[str, ...] = (
    "ce_family_supported",
    "relative_family_supported",
)

_REFERENCE_NORMALIZATION_FAMILY_LABELS: dict[str, str] = {
    "ce_family_supported": "CE",
    "relative_family_supported": "relative-mCE",
}

_REFERENCE_NORMALIZED_RUN_METRIC_FAMILIES: dict[str, tuple[str, ...]] = {
    "ce_family_supported": ("mCE_snaive",),
    "relative_family_supported": ("relative_mCE_snaive",),
}

_REFERENCE_NORMALIZED_SCENARIO_METRIC_FAMILIES: dict[str, tuple[str, ...]] = {}


@dataclass(frozen=True)
class MetaAnalysisInputRows:
    result_df: pd.DataFrame
    scenario_summary_df: pd.DataFrame
    scenario_samples_df: pd.DataFrame
    severity_profile_df: pd.DataFrame


@dataclass(frozen=True)
class CanonicalAnalysisFrames:
    backbone_df: pd.DataFrame
    non_baseline_eval_df: pd.DataFrame
    method_selection_df: pd.DataFrame
    pipeline_method_candidates_df: pd.DataFrame
    variant_selection_summary_df: pd.DataFrame
    method_variant_breakdown_df: pd.DataFrame
    method_aggregates_df: pd.DataFrame
    pipeline_method_results_df: pd.DataFrame


@dataclass(frozen=True)
class FigureArtifactSpec:
    figure: Any
    rel_parts: tuple[str, ...]
    filename: str
    figure_type: str
    dataset: Optional[str] = None
    pipeline_method: Optional[str] = None
    pipeline_id: Optional[str] = None
    metric: Optional[str] = None
    optional: bool = True


@dataclass(frozen=True)
class AnalysisArtifacts:
    tables: dict[str, tuple[pd.DataFrame, str]]
    figure_specs: list[FigureArtifactSpec]
    headline_metrics: dict[str, float]


@dataclass(frozen=True)
class RhoEffAttachmentResult:
    result_df: pd.DataFrame
    fit_summary_df: pd.DataFrame


FIXED_CHANNEL_FRACTION_COLUMNS: tuple[str, ...] = (
    "dataset",
    "model_architecture",
    "robustness_method",
    "pipeline_id",
    "run_id",
    "data_config_signature",
    "fixed_channel_fraction",
    "eval_data_seed",
    "n_test_samples",
    "canonical_D_w",
    "fixed_fraction_D_w",
    "delta_D_w_fixed_fraction",
    "canonical_D_mean",
    "fixed_fraction_D_mean",
    "delta_D_mean_fixed_fraction",
    "canonical_worst_scenario",
    "fixed_fraction_worst_scenario",
    "canonical_channel_scoped_D_w",
    "fixed_fraction_channel_scoped_D_w",
    "delta_channel_scoped_D_w_fixed_fraction",
    "canonical_channel_scoped_D_mean",
    "fixed_fraction_channel_scoped_D_mean",
    "delta_channel_scoped_D_mean_fixed_fraction",
    "channel_scoped_scenario_count",
    "all_scope_scenario_count",
    "fixed_channel_count_min",
    "fixed_channel_count_median",
    "fixed_channel_count_max",
    "canonical_context_signature",
    "fixed_fraction_context_signature",
)

FIXED_CHANNEL_FRACTION_PAPER_SUMMARY_COLUMNS: tuple[str, ...] = (
    "dataset",
    "spearman_rho",
    "weakest_family_agreement",
    "core_method_sign_agreement",
)

_FIXED_CHANNEL_FRACTION_CORE_METHODS: tuple[str, ...] = (
    "adversarial_training",
    "ensemble",
    "revin",
)



CORE_REQUIRED_FIGURE_TYPES: frozenset[str] = frozenset(
    {
        "improvement_scenario_delta_heatmap_all_methods",
        "improvement_core_deltas_comparison_dataset",
        "improvement_core_delta_heatmap_dataset",
    }
)


class UnsupportedConfiguredBaselineParetoScope(ValueError):
    """The optional configured baseline rank-Pareto export is not defined for this scope."""


@lru_cache(maxsize=1)
def _core_figure_registry() -> CoreFigureRegistry:
    """Load the dedicated core-figure registry once per process."""
    return load_parsed_core_figure_registry()


@lru_cache(maxsize=1)
def _known_dataset_registry_keys() -> frozenset[str]:
    return frozenset(str(dataset_key).strip() for dataset_key in DATASET_REGISTRY.keys())


@lru_cache(maxsize=1)
def _known_robustness_method_keys() -> frozenset[str]:
    return frozenset(
        str(method_key).strip()
        for method_key in PIPELINE_RECIPE_PATHS_BY_METHOD.keys()
    )


@lru_cache(maxsize=1)
def _core_figure_supported_methods() -> frozenset[str]:
    return frozenset(
        str(method_key).strip() for method_key in _core_figure_registry().method_order
    )


def _require_nonempty_string_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    context: str,
    sample_cols: list[str],
) -> pd.DataFrame:
    """Validate that *columns* contain non-null, non-blank string values and
    return a copy with those columns stripped.  Raises on any violation."""
    working = df.copy()
    for column in columns:
        missing_mask = working[column].isna()
        if missing_mask.any():
            examples = _sample_records(working.loc[missing_mask, sample_cols], sample_cols)
            raise ValueError(
                f"{context}: rows are missing '{column}'. Examples: {examples}."
            )
        working[column] = working[column].astype(str).str.strip()
        empty_mask = working[column] == ""
        if empty_mask.any():
            examples = _sample_records(working.loc[empty_mask, sample_cols], sample_cols)
            raise ValueError(
                f"{context}: rows contain empty '{column}' values. Examples: {examples}."
            )
    return working


def _require_columns(df: pd.DataFrame, required_cols: set[str], *, context: str) -> None:
    require_dataframe_columns(df, required_cols, context=context)


def _require_architecture_families(
    df: pd.DataFrame,
    *,
    model_col: str,
    arch_map: Mapping[str, str],
    context: str,
) -> pd.Series:
    _require_columns(df, {model_col}, context=context)
    model_series = df[model_col].astype(str).str.strip()
    mapped = model_series.map(arch_map)
    missing_mask = mapped.isna()
    if missing_mask.any():
        examples = sorted(model_series.loc[missing_mask].drop_duplicates().tolist())[:5]
        raise ValueError(
            f"{context}: missing architecture family mapping for model(s) {examples}."
        )
    return mapped.astype(str)


def _assign_architecture_families(
    df: pd.DataFrame,
    *,
    arch_map: Mapping[str, str],
    context: str,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if "architecture_family" in df.columns:
        enriched = df.copy()
        family_series = enriched["architecture_family"]
        missing_mask = family_series.isna()
        if missing_mask.any():
            example_cols = ["architecture_family"]
            if "model_architecture" in enriched.columns:
                example_cols.insert(0, "model_architecture")
            elif "model" in enriched.columns:
                example_cols.insert(0, "model")
            examples = (
                enriched.loc[missing_mask, example_cols]
                .drop_duplicates()
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                f"{context}: existing 'architecture_family' has missing values. "
                f"Examples: {examples}."
            )
        cleaned_families = family_series.astype(str).str.strip()
        empty_mask = cleaned_families == ""
        if empty_mask.any():
            example_cols = ["architecture_family"]
            if "model_architecture" in enriched.columns:
                example_cols.insert(0, "model_architecture")
            elif "model" in enriched.columns:
                example_cols.insert(0, "model")
            examples = (
                enriched.loc[empty_mask, example_cols]
                .drop_duplicates()
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                f"{context}: existing 'architecture_family' has empty values. "
                f"Examples: {examples}."
            )
        enriched["architecture_family"] = cleaned_families
        return enriched

    if "model_architecture" in df.columns:
        source_col = "model_architecture"
    elif "model" in df.columns:
        source_col = "model"
    else:
        raise ValueError(
            f"{context}: missing columns ['model', 'model_architecture'] required "
            "for architecture family assignment."
        )

    enriched = df.copy()
    enriched["architecture_family"] = _require_architecture_families(
        enriched,
        model_col=source_col,
        arch_map=arch_map,
        context=f"{context}: architecture family assignment",
    )
    return enriched


def _sample_records(
    df: pd.DataFrame, columns: list[str], *, limit: int = 5
) -> list[dict[str, Any]]:
    return sample_dataframe_records(df, columns, limit=limit)


def _assert_no_duplicates(
    df: pd.DataFrame, key_cols: list[str], *, context: str
) -> None:
    assert_no_duplicate_rows(df, key_cols, context=context)


def _sorted_items(mapping: Mapping[Any, Any]) -> list[tuple[Any, Any]]:
    return sorted(mapping.items(), key=lambda item: str(item[0]))


def _sorted_records(
    records: list[dict[str, Any]], *, keys: list[str]
) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda entry: tuple(str(entry.get(key)) for key in keys),
    )


def _explicit_cli_args_for_recompute(args: Any) -> tuple[str, ...] | None:
    raw_args = getattr(args, "_explicit_cli_args", None)
    if raw_args is None:
        return None
    if not isinstance(raw_args, (list, tuple)):
        raise ValueError(
            "args._explicit_cli_args must be a list or tuple of CLI tokens when provided."
        )
    if not all(isinstance(token, str) for token in raw_args):
        raise ValueError("args._explicit_cli_args must contain only string CLI tokens.")
    return tuple(raw_args)


def _infer_namespace_overrides(
    args: Any,
    *,
    defaults: Mapping[str, Any],
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, default_value in defaults.items():
        if not hasattr(args, key):
            continue
        current_value = getattr(args, key)
        if normalize_yaml_value(current_value) != normalize_yaml_value(default_value):
            overrides[str(key)] = current_value
    return overrides


def _ensure_testing_override_metadata(
    args: Any,
    *,
    recipe_specs_for_scope: Sequence[Any],
    defaults: Mapping[str, Any],
) -> None:
    recipe_overrides = getattr(args, "_recipe_param_overrides", None)
    window_overrides = getattr(args, "_window_arg_overrides", None)
    missing_recipe = not hasattr(args, "_recipe_param_overrides")
    missing_window = not hasattr(args, "_window_arg_overrides")
    if not missing_recipe and not isinstance(recipe_overrides, Mapping):
        raise ValueError("args._recipe_param_overrides must be a mapping when present.")
    if not missing_window and not isinstance(window_overrides, Mapping):
        raise ValueError("args._window_arg_overrides must be a mapping when present.")
    if not missing_recipe and not missing_window:
        return

    extracted_defaults = extract_recipe_defaults_for_scope(recipe_specs_for_scope)
    merged_defaults = merge_recipe_defaults_for_scope(defaults, extracted_defaults)
    if (
        "INPUT_LEN" not in merged_defaults
        or "TARGET_LEN" not in merged_defaults
        or "BATCH_SIZE" not in merged_defaults
    ):
        raise ValueError(
            "Merged defaults are missing INPUT_LEN/TARGET_LEN/BATCH_SIZE required for "
            "testing coverage recomputation."
        )
    explicit_cli_args = _explicit_cli_args_for_recompute(args)

    if missing_recipe:
        recipe_param_defaults: dict[str, Any] = {}
        for defaults_dict in extracted_defaults:
            recipe_param_defaults.update(defaults_dict)
        if explicit_cli_args is not None:
            recipe_param_overrides = parse_explicit_cli_overrides(
                recipe_param_defaults,
                extra_args=explicit_cli_args,
            )
        else:
            recipe_param_overrides = _infer_namespace_overrides(
                args,
                defaults=recipe_param_defaults,
            )
        args._recipe_param_overrides = dict(recipe_param_overrides)
    if missing_window:
        window_defaults = {
            "input_len": merged_defaults["INPUT_LEN"],
            "target_len": merged_defaults["TARGET_LEN"],
            "batch_size": merged_defaults["BATCH_SIZE"],
        }
        if explicit_cli_args is not None:
            rebuilt_window_overrides = parse_explicit_cli_overrides(
                window_defaults,
                extra_args=explicit_cli_args,
            )
        else:
            rebuilt_window_overrides = _infer_namespace_overrides(
                args,
                defaults=window_defaults,
            )
        has_input = "input_len" in rebuilt_window_overrides
        has_target = "target_len" in rebuilt_window_overrides
        if has_input != has_target:
            raise ValueError(
                "Rebuilt testing window override metadata must include input_len and "
                "target_len together when overriding merged defaults."
            )
        args._window_arg_overrides = dict(rebuilt_window_overrides)


def _recompute_coverage_fractions_by_dataset(
    args: Any,
    *,
    resolved_specs: Sequence[Any],
) -> dict[str, dict[tuple[str, str], tuple[int, int]]]:
    recipe_specs_for_scope = load_recipe_specs_for_scope()
    defaults = load_defaults()
    _ensure_testing_override_metadata(
        args,
        recipe_specs_for_scope=recipe_specs_for_scope,
        defaults=defaults,
    )
    dataset_window_defaults = load_dataset_windows(defaults=defaults)
    coverage_fractions_by_dataset: dict[str, dict[tuple[str, str], tuple[int, int]]] = {}
    for dataset_spec in resolved_specs:
        dataset_args = resolve_dataset_window_args(
            args,
            dataset_spec=dataset_spec,
            dataset_window_defaults=dataset_window_defaults,
            explicit_arg_overrides=args._window_arg_overrides,
        )
        print(
            f"Recomputing testing coverage for dataset '{dataset_spec.key}' from MLflow metadata."
        )
        scope = _resolve_dataset_testing_coverage_scope(
            dataset_spec=dataset_spec,
            args=dataset_args,
            recipe_specs_for_scope=recipe_specs_for_scope,
        )
        fractions = dict(scope.dataset_coverage_fractions)
        if not fractions:
            raise ValueError(
                f"Coverage recomputation produced no backbone/method coverage rows for "
                f"dataset '{dataset_spec.key}'."
            )
        coverage_fractions_by_dataset[str(dataset_spec.key)] = fractions
    return coverage_fractions_by_dataset


def _resolve_configured_baseline_pareto_panels(
    args: Any,
    *,
    resolved_specs: Sequence[Any],
) -> tuple[tuple[str, ...], tuple[str, str], dict[str, str]]:
    defaults = load_defaults()
    recipe_specs_for_scope = load_recipe_specs_for_scope()
    _ensure_testing_override_metadata(
        args,
        recipe_specs_for_scope=recipe_specs_for_scope,
        defaults=defaults,
    )
    dataset_window_defaults = load_dataset_windows(defaults=defaults)
    if not resolved_specs:
        raise ValueError(
            "Cannot resolve configured baseline Pareto panels: resolved_specs is empty."
        )

    configured_dataset_keys: list[str] = []
    resolved_windows: dict[str, tuple[int, int]] = {}
    window_pairs_by_target_len: dict[int, set[tuple[int, int]]] = {}
    for dataset_spec in resolved_specs:
        dataset_key = str(dataset_spec.key)
        configured_dataset_keys.append(dataset_key)
        dataset_args = resolve_dataset_window_args(
            args,
            dataset_spec=dataset_spec,
            dataset_window_defaults=dataset_window_defaults,
            explicit_arg_overrides=args._window_arg_overrides,
        )
        input_len = coerce_int(
            require_namespace_value(dataset_args, key="input_len")
        )
        target_len = coerce_int(
            require_namespace_value(dataset_args, key="target_len")
        )
        if input_len is None or input_len <= 0:
            raise ValueError(
                "Cannot classify configured baseline Pareto dataset "
                f"'{dataset_key}': input_len must be a positive integer."
            )
        if target_len is None or target_len <= 0:
            raise ValueError(
                "Cannot classify configured baseline Pareto dataset "
                f"'{dataset_key}': target_len must be a positive integer."
            )
        resolved_windows[dataset_key] = (input_len, target_len)
        window_pairs_by_target_len.setdefault(target_len, set()).add(
            (input_len, target_len)
        )

    unique_target_lens = sorted(window_pairs_by_target_len)
    if len(unique_target_lens) != 2:
        raise UnsupportedConfiguredBaselineParetoScope(
            "Configured baseline rank-Pareto export requires exactly two target_len "
            f"groups but found {unique_target_lens}."
        )

    panel_titles_by_target_len: dict[int, str] = {}
    for rank_idx, target_len in enumerate(unique_target_lens):
        window_pairs = window_pairs_by_target_len[target_len]
        if len(window_pairs) != 1:
            raise UnsupportedConfiguredBaselineParetoScope(
                "Configured baseline rank-Pareto export requires a single "
                f"(input_len, target_len) pair per target_len group; found "
                f"{sorted(window_pairs)} for target_len={target_len}."
            )
        input_len, resolved_target_len = next(iter(window_pairs))
        label_prefix = "Short Horizon" if rank_idx == 0 else "Long Horizon"
        panel_titles_by_target_len[target_len] = (
            f"{label_prefix} ({input_len}/{resolved_target_len})"
        )

    panel_by_dataset = {
        dataset_key: panel_titles_by_target_len[target_len]
        for dataset_key, (_, target_len) in resolved_windows.items()
    }
    ordered_panel_titles = tuple(
        panel_titles_by_target_len[target_len] for target_len in unique_target_lens
    )
    return tuple(configured_dataset_keys), ordered_panel_titles, panel_by_dataset


def _build_configured_baseline_rank_pareto_figure_specs(
    backbone_df: pd.DataFrame,
    *,
    args: Any,
    arch_map: Mapping[str, str],
    resolved_specs: Sequence[Any],
) -> list[FigureArtifactSpec]:
    context = "Cannot build configured baseline rank-Pareto figures"
    perf_col = f"{args.test_metric}_test"
    try:
        configured_dataset_keys, ordered_panel_titles, panel_by_dataset = (
            _resolve_configured_baseline_pareto_panels(
                args,
                resolved_specs=resolved_specs,
            )
        )
    except UnsupportedConfiguredBaselineParetoScope as exc:
        warnings.warn(
            f"{context}: {exc} Skipping this optional figure export.",
            stacklevel=2,
        )
        return []
    registry = _core_figure_registry()
    baseline_rank_metric = registry.baseline_rank_pareto_metric
    baseline_rank_metric_title = robustness_metric_display_name(baseline_rank_metric)
    required_cols = {
        "dataset",
        "model",
        "model_architecture",
        perf_col,
        baseline_rank_metric,
    }
    _require_columns(backbone_df, required_cols, context=context)
    if backbone_df.empty:
        raise ValueError(f"{context}: backbone dataframe is empty.")

    plot_df = _assign_architecture_families(
        backbone_df,
        arch_map=arch_map,
        context=context,
    )
    plot_df["dataset"] = plot_df["dataset"].astype(str).str.strip()

    available_datasets = set(plot_df["dataset"].unique().tolist())
    missing_datasets = [
        dataset_key
        for dataset_key in configured_dataset_keys
        if dataset_key not in available_datasets
    ]
    if missing_datasets:
        raise ValueError(
            f"{context}: missing required configured datasets {missing_datasets}."
        )

    plot_df = plot_df.loc[
        plot_df["dataset"].isin(configured_dataset_keys)
    ].copy()
    if plot_df.empty:
        raise ValueError(
            f"{context}: filtered configured baseline dataframe is empty."
        )

    missing_metric_mask = plot_df[[perf_col, baseline_rank_metric]].isna().any(axis=1)
    if missing_metric_mask.any():
        examples = (
            plot_df.loc[
                missing_metric_mask,
                ["dataset", "model", perf_col, baseline_rank_metric],
            ]
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"{context}: missing required metrics in configured baseline rows. "
            f"Examples: {examples}."
        )

    _assert_no_duplicates(
        plot_df,
        ["dataset", "model"],
        context=(
            f"{context}: configured baseline rows must be unique per "
            "(dataset, model)"
        ),
    )

    plot_df["horizon_panel"] = plot_df["dataset"].map(panel_by_dataset)
    if plot_df["horizon_panel"].isna().any():
        examples = (
            plot_df.loc[plot_df["horizon_panel"].isna(), ["dataset"]]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"{context}: failed to classify dataset(s) into horizon panels. "
            f"Examples: {examples}."
        )

    overall_figure = plot_ranked_performance_robustness_pareto(
        plot_df,
        dataset_col="dataset",
        model_col="model",
        arch_col="architecture_family",
        perf_col=perf_col,
        robust_col=baseline_rank_metric,
        y_semantics=_require_plot_semantics_for_keys(
            test_metric=args.test_metric,
            required_keys=[baseline_rank_metric],
            context=context,
        )[baseline_rank_metric],
        x_title=f"Average {args.test_metric} Rank",
        y_title=baseline_rank_metric_title,
    )

    panel_frames: list[tuple[str, pd.DataFrame]] = []
    for panel_title in ordered_panel_titles:
        panel_df = plot_df.loc[plot_df["horizon_panel"] == panel_title].copy()
        if panel_df.empty:
            raise ValueError(
                f"{context}: panel '{panel_title}' has no baseline rows."
            )
        panel_frames.append((panel_title, panel_df))

    by_horizon_figure = plot_ranked_performance_robustness_pareto_panels(
        panel_frames,
        dataset_col="dataset",
        model_col="model",
        arch_col="architecture_family",
        perf_col=perf_col,
        robust_col=baseline_rank_metric,
        y_semantics=_require_plot_semantics_for_keys(
            test_metric=args.test_metric,
            required_keys=[baseline_rank_metric],
            context=context,
        )[baseline_rank_metric],
        x_title=f"Average {args.test_metric} Rank",
        y_title=baseline_rank_metric_title,
    )

    # 2x2 per-dataset raw-performance-vs-degradation Pareto.
    dataset_panel_frames: list[tuple[str, pd.DataFrame]] = [
        (dataset_key, plot_df.loc[plot_df["dataset"] == dataset_key].copy())
        for dataset_key in configured_dataset_keys
    ]
    empty_panels = [title for title, df in dataset_panel_frames if df.empty]
    if empty_panels:
        raise ValueError(
            f"{context}: per-dataset Pareto panels have no baseline rows for "
            f"datasets {empty_panels}."
        )

    specs: list[FigureArtifactSpec] = []

    if len(dataset_panel_frames) != 4:
        warnings.warn(
            f"{context}: skipping per-dataset baseline Pareto export because the "
            f"2x2 panel helper requires exactly 4 configured datasets, found "
            f"{len(dataset_panel_frames)}.",
            stacklevel=2,
        )
    else:
        by_dataset_figure = plot_pareto_dataset_panels(
            dataset_panel_frames,
            perf_col=perf_col,
            robust_col=baseline_rank_metric,
            x_semantics=_require_plot_semantics_for_keys(
                test_metric=args.test_metric,
                required_keys=[perf_col],
                context=context,
            )[perf_col],
            y_semantics=_require_plot_semantics_for_keys(
                test_metric=args.test_metric,
                required_keys=[baseline_rank_metric],
                context=context,
            )[baseline_rank_metric],
            model_col="model",
            arch_col="architecture_family",
            perf_lower_is_better=True,
            x_title=f"{args.test_metric} (Test)",
            y_title=baseline_rank_metric_title,
        )
        specs.append(
            FigureArtifactSpec(
                figure=by_dataset_figure,
                rel_parts=("2_baselines", "pareto", "dataset"),
                filename="architecture_pareto_by_dataset.pdf",
                figure_type="baseline_pareto_by_dataset_panels",
                metric=baseline_rank_metric,
                optional=True,
            ),
        )

    specs.extend(
        [
            FigureArtifactSpec(
                figure=by_horizon_figure,
                rel_parts=("2_baselines", "pareto", "rank"),
                filename="architecture_rank_pareto.pdf",
                figure_type="baseline_pareto_rank_by_horizon",
                metric=baseline_rank_metric,
                optional=True,
            ),
            FigureArtifactSpec(
                figure=overall_figure,
                rel_parts=("2_baselines", "pareto", "rank"),
                filename="architecture_rank_pareto_overall.pdf",
                figure_type="baseline_pareto_rank_overall",
                metric=baseline_rank_metric,
                optional=True,
            ),
        ]
    )
    return specs


def _build_core_baseline_figure_specs(
    backbone_df: pd.DataFrame,
    *,
    args: Any,
    full_coverage: bool,
) -> list[FigureArtifactSpec]:
    context = "Cannot build core baseline figures"
    registry = _core_figure_registry()
    perf_col = f"{args.test_metric}_test"
    baseline_rank_metric = registry.baseline_rank_pareto_metric
    required_cols = {
        "dataset",
        "model",
        "architecture_family",
        perf_col,
        baseline_rank_metric,
    }
    _require_columns(backbone_df, required_cols, context=context)
    if backbone_df.empty:
        raise ValueError(f"{context}: backbone dataframe is empty.")

    plot_df = _require_nonempty_string_columns(
        backbone_df,
        ["dataset", "model", "architecture_family"],
        context=context,
        sample_cols=["dataset", "model", "architecture_family", perf_col],
    )
    dataset_keys = [
        dataset_key for dataset_key, _ in registry.dataset_spec
    ]
    validate_raw_display_id_values(
        plot_df["dataset"].tolist(),
        raw_ids=dataset_keys,
        display_mapping=dict(registry.dataset_spec),
        context=context,
        id_label="dataset",
    )

    missing_datasets = [
        dataset_key
        for dataset_key in dataset_keys
        if dataset_key not in set(plot_df["dataset"].astype(str).str.strip())
    ]
    if missing_datasets and full_coverage:
        raise ValueError(
            f"{context}: missing required core-figure datasets {missing_datasets}."
        )

    _assert_no_duplicates(
        plot_df,
        ["dataset", "model"],
        context=f"{context}: baseline rows must be unique per (dataset, model)",
    )

    if missing_datasets and not full_coverage:
        print(
            f"{context}: full_coverage=false, rendering the available core-figure "
            f"dataset subset and omitting missing datasets {missing_datasets}."
        )
    dataset_panel_frames = []
    for dataset_key, dataset_label in registry.dataset_spec:
        dataset_frame = plot_df.loc[
            plot_df["dataset"].astype(str).str.strip() == dataset_key
        ].copy()
        if dataset_frame.empty and not full_coverage:
            continue
        dataset_panel_frames.append((dataset_label, dataset_frame))
    empty_panels = [label for label, frame in dataset_panel_frames if frame.empty]
    if empty_panels:
        raise ValueError(
            f"{context}: core baseline panels are missing rows for {empty_panels}."
        )
    if not dataset_panel_frames:
        raise ValueError(
            f"{context}: no renderable core-figure dataset panels remain."
        )
    semantics = _require_plot_semantics_for_keys(
        test_metric=args.test_metric,
        required_keys=[perf_col, baseline_rank_metric],
        context=context,
    )
    figure = plot_pareto_dataset_panels(
        dataset_panel_frames,
        perf_col=perf_col,
        robust_col=baseline_rank_metric,
        x_semantics=semantics[perf_col],
        y_semantics=semantics[baseline_rank_metric],
        model_col="model",
        arch_col="architecture_family",
        perf_lower_is_better=True,
        x_title=None,
        y_title=None,
    )
    return [
        FigureArtifactSpec(
            figure=figure,
            rel_parts=("2_baselines", "pareto", "dataset"),
            filename="architecture_pareto_by_dataset.pdf",
            figure_type="baseline_pareto_by_dataset_panels",
            dataset=None,
            pipeline_method="baseline",
            pipeline_id=None,
            metric=build_degradation_metric_key(
                test_metric=args.test_metric,
                metric_name=baseline_rank_metric,
            ),
            optional=False,
        )
    ]


def _build_plot_artifact_with_partial_coverage_tolerance(
    builder: Callable[[], Any],
    *,
    runtime_args: Any,
    context: str,
    empty_value: Any,
    suppress_markers: Sequence[str],
) -> Any:
    try:
        return builder()
    except ValueError as exc:
        if require_namespace_bool(runtime_args, key="full_coverage"):
            raise
        message = str(exc)
        if not any(marker in message for marker in suppress_markers):
            raise
        print(
            f"{context}: full_coverage=false, suppressing partial-coverage "
            f"plotting failure and continuing: {message}"
        )
        return empty_value


def _select_method_scenario_delta_heatmap_rows(
    scenario_delta_df: pd.DataFrame,
    *,
    context: str,
) -> pd.DataFrame:
    if scenario_delta_df.empty:
        return scenario_delta_df.copy()

    working_df = _require_nonempty_string_columns(
        scenario_delta_df,
        ["dataset", "robustness_method", "scenario"],
        context=context,
        sample_cols=["dataset", "robustness_method", "scenario"],
    )
    registry = _core_figure_registry()
    dataset_keys = tuple(
        str(dataset_key).strip() for dataset_key, _ in registry.dataset_spec
    )
    validate_scoped_raw_display_id_values(
        working_df["dataset"].tolist(),
        raw_ids=dataset_keys,
        display_mapping=dict(registry.dataset_spec),
        known_raw_ids=tuple(_known_dataset_registry_keys()),
        context=context,
        id_label="dataset",
    )
    method_keys = tuple(str(method_key).strip() for method_key in registry.method_order)
    validate_scoped_raw_display_id_values(
        working_df["robustness_method"].tolist(),
        raw_ids=method_keys,
        display_mapping=registry.method_display,
        known_raw_ids=tuple(_known_robustness_method_keys()),
        context=context,
        id_label="robustness_method",
    )
    scoped_df = working_df.loc[
        working_df["dataset"].isin(dataset_keys)
        & working_df["robustness_method"].isin(method_keys)
    ].copy()
    if scoped_df.empty:
        raise ValueError(
            f"{context}: has no rows in the core-figure dataset/method scope."
        )
    return scoped_df


def _build_method_scenario_delta_heatmap_figure_specs(
    scenario_delta_df: pd.DataFrame,
    *,
    args: Any,
    full_coverage: bool,
) -> list[FigureArtifactSpec]:
    context = "Cannot build method scenario-delta heatmap"
    registry = _core_figure_registry()
    delta_col = _core_delta_column("D")
    required_cols = {
        "dataset",
        "robustness_method",
        "scenario",
        delta_col,
    }
    _require_columns(scenario_delta_df, required_cols, context=context)
    semantics = _require_plot_semantics_for_keys(
        test_metric=args.test_metric,
        required_keys=[delta_col],
        context=context,
    )
    core_scope_df = _select_method_scenario_delta_heatmap_rows(
        scenario_delta_df,
        context=context,
    )
    present_dataset_keys = (
        set(core_scope_df["dataset"].astype(str).str.strip().tolist())
        if not core_scope_df.empty
        else set()
    )
    missing_datasets = [
        dataset_key
        for dataset_key, _ in registry.dataset_spec
        if dataset_key not in present_dataset_keys
    ]
    if missing_datasets and full_coverage:
        present_counts = (
            core_scope_df.groupby("dataset", dropna=False)
            .size()
            .sort_index()
            .to_dict()
        )
        raise ValueError(
            f"{context}: missing required core-figure datasets {missing_datasets}. "
            f"Present core dataset row counts: {present_counts}."
        )
    if full_coverage:
        expected_cells = {
            (dataset_key, method_key, scenario_key)
            for dataset_key, _ in registry.dataset_spec
            for method_key in registry.method_order
            for scenario_key in registry.scenario_display_order
        }
        actual_cells = {
            (str(dataset), str(method), str(scenario))
            for dataset, method, scenario in core_scope_df.loc[
                :, ["dataset", "robustness_method", "scenario"]
            ].itertuples(index=False, name=None)
        }
        missing_cells = sorted(expected_cells - actual_cells)
        if missing_cells:
            raise ValueError(
                f"{context}: missing required core dataset/method/scenario "
                f"cell(s): {missing_cells[:5]}."
            )
    dataset_spec = tuple(
        (dataset_key, dataset_label)
        for dataset_key, dataset_label in registry.dataset_spec
        if dataset_key in present_dataset_keys or full_coverage
    )
    if missing_datasets and not full_coverage:
        print(
            f"{context}: full_coverage=false, rendering the available core-figure "
            f"dataset subset and omitting missing datasets {missing_datasets}."
        )
    if not full_coverage and not core_scope_df.empty:
        ordered_scenarios = tuple(registry.scenario_display_order)
        ordered_scenario_set = set(ordered_scenarios)
        retained_groups: list[pd.DataFrame] = []
        omitted_groups: list[str] = []
        for (dataset_key, method_key), group_df in core_scope_df.groupby(
            ["dataset", "robustness_method"],
            dropna=False,
            sort=True,
        ):
            present_scenarios = set(
                group_df["scenario"].astype(str).str.strip().tolist()
            )
            extra_scenarios = sorted(present_scenarios - ordered_scenario_set)
            if extra_scenarios:
                retained_groups.append(group_df.copy())
                continue
            missing_scenarios = sorted(ordered_scenario_set - present_scenarios)
            if missing_scenarios:
                omitted_groups.append(
                    f"{dataset_key}/{method_key} missing {missing_scenarios}"
                )
                continue
            retained_groups.append(group_df.copy())
        if omitted_groups:
            print(
                f"{context}: full_coverage=false, omitting dataset/method cells with "
                "incomplete scenario coverage: "
                f"{omitted_groups}"
            )
        if retained_groups:
            core_scope_df = pd.concat(retained_groups, ignore_index=True)
        else:
            core_scope_df = core_scope_df.iloc[0:0].copy()
        retained_dataset_keys = set(
            core_scope_df["dataset"].astype(str).str.strip().tolist()
        )
        dataset_spec = tuple(
            (dataset_key, dataset_label)
            for dataset_key, dataset_label in registry.dataset_spec
            if dataset_key in retained_dataset_keys
        )
    if not dataset_spec:
        raise ValueError(f"{context}: no renderable core-figure dataset panels remain.")
    figure = plot_method_scenario_delta_heatmap(
        core_scope_df,
        dataset_spec=dataset_spec,
        method_display={
            method_key: registry.method_display[method_key]
            for method_key in registry.method_order
        },
        scenario_display_order=registry.scenario_display_order,
        scenario_display=registry.scenario_display,
        scenario_groups=registry.scenario_groups,
        value_col=delta_col,
        value_semantics=semantics[delta_col],
        colorbar_label="Delta Scenario Degradation",
    )
    return [
        FigureArtifactSpec(
            figure=figure,
            rel_parts=("3_improvements", "scenario_deltas"),
            filename="method_scenario_delta_heatmap_d.pdf",
            figure_type="improvement_scenario_delta_heatmap_all_methods",
            dataset=None,
            pipeline_method=None,
            pipeline_id=None,
            metric=f"degradation/{args.test_metric}/scenario/<pert_idx>/D",
            optional=False,
        )
    ]


def _build_testing_coverage_table(
    coverage_fractions_by_dataset: Mapping[
        str,
        Mapping[tuple[str, str], tuple[int, int]],
    ],
    *,
    coverage_source: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_name, fractions in sorted(coverage_fractions_by_dataset.items()):
        for (architecture, method), (seen, expected) in sorted(fractions.items()):
            ratio = float(seen) / float(expected) if int(expected) > 0 else np.nan
            rows.append(
                {
                    "dataset": str(dataset_name),
                    "backbone_architecture": str(architecture),
                    "robustness_method": str(method),
                    "seen": int(seen),
                    "expected": int(expected),
                    "coverage_ratio": ratio,
                    "coverage_source": str(coverage_source),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "backbone_architecture",
            "robustness_method",
            "seen",
            "expected",
            "coverage_ratio",
            "coverage_source",
        ],
    )


def _require_winner_pool_coverage_matches_testing_coverage(
    winner_pool_df: pd.DataFrame,
    coverage_fractions_by_dataset: Mapping[
        str,
        Mapping[tuple[str, str], tuple[int, int]],
    ],
    *,
    full_coverage: bool,
) -> None:
    """Require strict winner-pool rows for every fully tested coverage cell."""
    if not full_coverage:
        return
    context = "Meta-analysis winner-pool coverage validation"
    required_cols = {
        "dataset",
        "backbone_architecture",
        "robustness_method",
        "selection_pool",
    }
    require_dataframe_columns(winner_pool_df, required_cols, context=context)
    if not coverage_fractions_by_dataset:
        raise ValueError(f"{context}: testing coverage input is empty.")

    expected_records: list[dict[str, str]] = []
    incomplete_records: list[dict[str, Any]] = []
    for dataset_name, fractions in sorted(coverage_fractions_by_dataset.items()):
        if not fractions:
            raise ValueError(
                f"{context}: dataset '{dataset_name}' has no testing coverage cells."
            )
        for (architecture, method), (seen, expected) in sorted(fractions.items()):
            seen_count = coerce_int(seen)
            expected_count = coerce_int(expected)
            if seen_count is None or expected_count is None:
                raise ValueError(
                    f"{context}: coverage cell has non-integer counts. "
                    f"dataset={dataset_name}, architecture={architecture}, "
                    f"method={method}, seen={seen!r}, expected={expected!r}."
                )
            if expected_count <= 0:
                raise ValueError(
                    f"{context}: coverage cell has non-positive expected count. "
                    f"dataset={dataset_name}, architecture={architecture}, "
                    f"method={method}, expected={expected_count}."
                )
            record = {
                "dataset": str(dataset_name),
                "backbone_architecture": str(architecture),
                "robustness_method": str(method),
            }
            if seen_count != expected_count:
                incomplete = dict(record)
                incomplete["seen"] = seen_count
                incomplete["expected"] = expected_count
                incomplete_records.append(incomplete)
                continue
            expected_records.append(record)
    if incomplete_records:
        examples = incomplete_records[:5]
        raise ValueError(
            f"{context}: full_coverage=True but testing coverage is incomplete. "
            f"Examples: {examples}."
        )
    if not expected_records:
        raise ValueError(f"{context}: no fully tested coverage cells were found.")

    sample_cols = [
        "dataset",
        "backbone_architecture",
        "robustness_method",
        "selection_pool",
    ]
    working_df = winner_pool_df.loc[:, sample_cols].copy()
    for column in sample_cols:
        working_df[column] = require_nonempty_string_series(
            working_df,
            column,
            context=context,
            sample_cols=sample_cols,
        )
    bad_pool = working_df["selection_pool"] != "winner_pool"
    if bad_pool.any():
        examples = sample_dataframe_records(working_df.loc[bad_pool], sample_cols)
        raise ValueError(
            f"{context}: rows are not sourced from the winner_pool. "
            f"Examples: {examples}."
        )

    key_cols = ["dataset", "backbone_architecture", "robustness_method"]
    actual_counts = (
        working_df.groupby(key_cols, dropna=False)
        .size()
        .rename("winner_count")
        .reset_index()
    )
    duplicate_counts = actual_counts.loc[actual_counts["winner_count"] > 1]
    if not duplicate_counts.empty:
        examples = sample_dataframe_records(
            duplicate_counts,
            [*key_cols, "winner_count"],
        )
        raise ValueError(
            f"{context}: expected exactly one winner row per tested "
            f"(dataset, backbone_architecture, robustness_method), but found "
            f"duplicates. Examples: {examples}."
        )

    expected_df = pd.DataFrame(expected_records, columns=key_cols)
    missing_df = expected_df.merge(actual_counts, on=key_cols, how="left")
    missing_df = missing_df.loc[missing_df["winner_count"].isna()]
    if not missing_df.empty:
        examples = sample_dataframe_records(missing_df, key_cols)
        raise ValueError(
            f"{context}: tested coverage contains cells missing from the current "
            f"winner pool. Re-run testing/selection for the full reporting scope. "
            f"Examples: {examples}."
        )

    extra_df = actual_counts.merge(expected_df, on=key_cols, how="left", indicator=True)
    extra_df = extra_df.loc[extra_df["_merge"] == "left_only"]
    if not extra_df.empty:
        examples = sample_dataframe_records(extra_df, [*key_cols, "winner_count"])
        raise ValueError(
            f"{context}: current winner pool contains cells outside testing coverage. "
            f"Examples: {examples}."
        )


def _complete_testing_coverage_cells(
    coverage_fractions_by_dataset: Mapping[
        str,
        Mapping[tuple[str, str], tuple[int, int]],
    ],
    *,
    dataset_name: str,
    context: str,
) -> frozenset[tuple[str, str]] | None:
    dataset_coverage = coverage_fractions_by_dataset.get(str(dataset_name))
    if dataset_coverage is None:
        return None
    complete_cells: set[tuple[str, str]] = set()
    for (architecture, method), (seen, expected) in dataset_coverage.items():
        seen_count = coerce_int(seen)
        expected_count = coerce_int(expected)
        if seen_count is None or expected_count is None:
            raise ValueError(
                f"{context}: coverage cell has non-integer counts. "
                f"dataset={dataset_name}, architecture={architecture}, "
                f"method={method}, seen={seen!r}, expected={expected!r}."
            )
        if expected_count <= 0:
            raise ValueError(
                f"{context}: coverage cell has non-positive expected count. "
                f"dataset={dataset_name}, architecture={architecture}, "
                f"method={method}, expected={expected_count}."
            )
        if seen_count == expected_count:
            complete_cells.add((str(architecture), str(method)))
    return frozenset(complete_cells)


def _winner_pool_architecture_outside_testing_coverage(
    *,
    coverage_fractions_by_dataset: Mapping[
        str,
        Mapping[tuple[str, str], tuple[int, int]],
    ],
    dataset_name: str,
    backbone_architecture: str,
) -> bool:
    """Return whether a winner architecture is outside fully covered cells."""
    complete_cells = _complete_testing_coverage_cells(
        coverage_fractions_by_dataset,
        dataset_name=dataset_name,
        context="Meta-analysis winner-pool coverage filtering",
    )
    if complete_cells is None:
        return False
    complete_architectures = {architecture for architecture, _ in complete_cells}
    return str(backbone_architecture) not in complete_architectures


def _winner_pool_cell_outside_testing_coverage(
    *,
    coverage_fractions_by_dataset: Mapping[
        str,
        Mapping[tuple[str, str], tuple[int, int]],
    ],
    dataset_name: str,
    backbone_architecture: str,
    robustness_method: str,
) -> bool:
    """Return whether a winner row is outside fully covered testing cells."""
    complete_cells = _complete_testing_coverage_cells(
        coverage_fractions_by_dataset,
        dataset_name=dataset_name,
        context="Meta-analysis winner-pool coverage filtering",
    )
    if complete_cells is None:
        return False
    requested_cell = (str(backbone_architecture), str(robustness_method))
    return requested_cell not in complete_cells


def _chunk_values(values: list[str], *, chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, received {chunk_size}.")
    if not values:
        return []
    return [values[idx : idx + chunk_size] for idx in range(0, len(values), chunk_size)]


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    q = float(quantile)
    if q < 0.0 or q > 1.0:
        raise ValueError(f"Weighted quantile must be in [0, 1], received {q}.")
    if values.size == 0 or weights.size == 0:
        raise ValueError("Cannot compute weighted quantile on empty arrays.")
    if values.size != weights.size:
        raise ValueError(
            "Cannot compute weighted quantile with mismatched value/weight sizes."
        )
    if not np.isfinite(values).all():
        raise ValueError("Weighted quantile values must be finite.")
    if not np.isfinite(weights).all():
        raise ValueError("Weighted quantile weights must be finite.")
    if (weights <= 0).any():
        raise ValueError("Weighted quantile weights must be strictly positive.")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    total_weight = float(cumulative[-1])
    if total_weight <= 0.0:
        raise ValueError("Weighted quantile requires positive total weight.")
    target = q * total_weight
    return float(np.interp(target, cumulative, sorted_values))


def _build_binned_degradation_profile_df(
    scenario_samples_df: pd.DataFrame,
) -> pd.DataFrame:
    context = "Cannot build binned degradation severity-profile rows"
    if scenario_samples_df.empty:
        raise ValueError(f"{context} from empty scenario_samples_df.")
    required_cols = {
        "dataset",
        "data_config_signature",
        "stage",
        "robustness_method",
        "pipeline_method",
        "pipeline_kind",
        "pipeline_id",
        "run_id",
        "model_architecture",
        "backbone_architecture",
        "model",
        "sample_id",
        "pert_idx",
        "scenario",
        "severity",
        "err_pert",
        "err_clean_global",
    }
    _require_columns(
        scenario_samples_df,
        required_cols,
        context=context,
    )
    working_df = scenario_samples_df.copy()
    working_df = _require_nonempty_string_columns(
        working_df,
        [
            "dataset",
            "data_config_signature",
            "stage",
            "robustness_method",
            "pipeline_method",
            "pipeline_kind",
            "pipeline_id",
            "run_id",
            "model_architecture",
            "backbone_architecture",
            "model",
            "scenario",
        ],
        context=context,
        sample_cols=["run_id", "dataset", "scenario"],
    )
    for column in ("sample_id", "pert_idx"):
        numeric = pd.to_numeric(working_df[column], errors="raise")
        if not np.allclose(numeric, np.floor(numeric)):
            raise ValueError(
                f"{context}: column '{column}' must contain integer values."
            )
        working_df[column] = numeric.astype(int)
    for column in ("severity", "err_pert", "err_clean_global"):
        working_df[column] = require_numeric_series(
            working_df[column],
            column_name=column,
            context=context,
            allow_nan=False,
            allow_infinite=False,
        ).astype(float)
    if ((working_df["severity"] < 0.0) | (working_df["severity"] > 1.0)).any():
        raise ValueError(
            f"{context}: severity values must lie in [0, 1]."
        )
    if (working_df["err_clean_global"] <= 0.0).any():
        raise ValueError(
            f"{context}: err_clean_global must stay strictly positive."
        )
    working_df["severity_bin_idx"] = np.floor(
        np.minimum(working_df["severity"].to_numpy(dtype=float), 0.999999) / 0.1
    ).astype(int)
    working_df["severity"] = 0.05 + 0.1 * working_df["severity_bin_idx"].astype(float)
    grouped = (
        working_df.groupby(
            [
                "dataset",
                "data_config_signature",
                "stage",
                "robustness_method",
                "pipeline_method",
                "pipeline_kind",
                "pipeline_id",
                "run_id",
                "model_architecture",
                "backbone_architecture",
                "model",
                "pert_idx",
                "scenario",
                "severity_bin_idx",
                "severity",
            ],
            dropna=False,
            sort=True,
            as_index=False,
        )
        .agg(
            n_samples=("sample_id", "size"),
            err_clean_mean=("err_clean_global", "mean"),
            err_pert_mean=("err_pert", "mean"),
        )
    )
    return grouped


def _build_error_distribution_artifacts(
    scenario_samples_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if scenario_samples_df.empty:
        raise ValueError(
            "Cannot build error-distribution artifacts from empty scenario_samples_df."
        )

    required_cols = {
        "dataset",
        "data_config_signature",
        "stage",
        "robustness_method",
        "pipeline_id",
        "model_architecture",
        "run_id",
        "sample_id",
        "scenario",
        "err_clean",
        "err_pert",
    }
    _require_columns(
        scenario_samples_df,
        required_cols,
        context="Cannot build error-distribution artifacts",
    )

    working_df = scenario_samples_df.copy()
    text_cols = (
        "dataset",
        "data_config_signature",
        "stage",
        "robustness_method",
        "pipeline_id",
        "model_architecture",
        "run_id",
        "scenario",
    )
    for col in text_cols:
        if working_df[col].isna().any():
            examples = _sample_records(
                working_df.loc[working_df[col].isna(), ["dataset", "pipeline_id"]],
                ["dataset", "pipeline_id"],
            )
            raise ValueError(
                "Cannot build error-distribution artifacts because required "
                f"column '{col}' contains missing values. Examples: {examples}."
            )
        stripped = working_df[col].astype(str).str.strip()
        if (stripped == "").any():
            examples = _sample_records(
                working_df.loc[stripped == "", ["dataset", "pipeline_id"]],
                ["dataset", "pipeline_id"],
            )
            raise ValueError(
                "Cannot build error-distribution artifacts because required "
                f"column '{col}' contains empty values. Examples: {examples}."
            )
        working_df[col] = stripped

    numeric_cols = ("err_clean", "err_pert")
    for col in numeric_cols:
        working_df[col] = require_numeric_series(
            working_df[col],
            column_name=col,
            context="Cannot build error-distribution artifacts",
            allow_nan=False,
            allow_infinite=False,
        )
    working_df["weight"] = 1.0

    id_cols = [
        "dataset",
        "data_config_signature",
        "stage",
        "robustness_method",
        "pipeline_id",
        "model_architecture",
    ]
    clean_df = working_df[id_cols + ["weight", "err_clean"]].rename(
        columns={
            "err_clean": "error_value",
        }
    )
    clean_df["error_kind"] = "err_clean"
    pert_df = working_df[id_cols + ["weight", "err_pert"]].rename(
        columns={
            "err_pert": "error_value",
        }
    )
    pert_df["error_kind"] = "err_pert"
    long_df = pd.concat([clean_df, pert_df], ignore_index=True)
    stage_counts = (
        long_df.groupby(
            ["dataset", "robustness_method", "model_architecture", "pipeline_id"],
            dropna=False,
        )["stage"]
        .nunique()
    )
    bad_stage_counts = stage_counts[stage_counts != 1]
    if not bad_stage_counts.empty:
        examples = [
            {
                "dataset": dataset,
                "robustness_method": method,
                "model_architecture": architecture,
                "pipeline_id": pipeline_id,
                "stage_count": int(count),
            }
            for (dataset, method, architecture, pipeline_id), count in bad_stage_counts.head(5).items()
        ]
        raise ValueError(
            "Cannot build error-distribution artifacts because model variants map to multiple "
            f"stage values. Examples: {examples}."
        )
    long_df["model_variant"] = (
        long_df["model_architecture"].astype(str).str.strip()
        + " | "
        + long_df["robustness_method"].astype(str).str.strip()
    )

    summary_keys = [
        "dataset",
        "data_config_signature",
        "stage",
        "robustness_method",
        "pipeline_id",
        "model_architecture",
        "model_variant",
        "error_kind",
    ]
    summary_records: list[dict[str, Any]] = []
    for key_values, group in long_df.groupby(summary_keys, dropna=False, sort=True):
        values = group["error_value"].to_numpy(dtype=float)
        weights = group["weight"].to_numpy(dtype=float)
        total_weight = float(np.sum(weights))
        weighted_mean = float(np.average(values, weights=weights))
        weighted_var = float(np.average((values - weighted_mean) ** 2, weights=weights))
        weighted_std = float(np.sqrt(weighted_var))
        row = dict(zip(summary_keys, key_values))
        row.update(
            {
                "n_observations": int(len(group)),
                "total_weight": total_weight,
                "weighted_mean": weighted_mean,
                "weighted_std": weighted_std,
                "weighted_q05": _weighted_quantile(values, weights, 0.05),
                "weighted_q25": _weighted_quantile(values, weights, 0.25),
                "weighted_q50": _weighted_quantile(values, weights, 0.50),
                "weighted_q75": _weighted_quantile(values, weights, 0.75),
                "weighted_q95": _weighted_quantile(values, weights, 0.95),
                "min_value": float(np.min(values)),
                "max_value": float(np.max(values)),
            }
        )
        summary_records.append(row)

    if not summary_records:
        raise ValueError(
            "Cannot build error-distribution artifacts because no summary rows were produced."
        )
    summary_df = pd.DataFrame(summary_records)

    dataset_plot_frames: dict[str, Any] = {}
    for dataset_name, dataset_df in long_df.groupby("dataset", dropna=False, sort=True):
        if dataset_df.empty:
            raise ValueError(
                f"Cannot build error-distribution plot frame for dataset '{dataset_name}': no rows."
            )
        dataset_plot_frames[str(dataset_name)] = dataset_df.copy()
    if not dataset_plot_frames:
        raise ValueError(
            "Cannot build error-distribution artifacts because no dataset plot frames were produced."
        )
    return summary_df, dataset_plot_frames


def _build_error_distribution_figure_specs(
    scenario_samples_df: pd.DataFrame,
    *,
    canonical_method_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[FigureArtifactSpec]]:
    filtered_scenario_samples_df = _filter_rows_to_canonical_method_winners(
        scenario_samples_df,
        canonical_method_df=canonical_method_df,
        context="Cannot generate error-distribution plots",
        drop_out_of_scope_methods=True,
    )
    summary_df, error_distribution_plot_frames = _build_error_distribution_artifacts(
        filtered_scenario_samples_df
    )
    max_error_distribution_facets = 24
    specs: list[FigureArtifactSpec] = []
    for dataset_name, dataset_plot_df in _sorted_items(error_distribution_plot_frames):
        context = f"Cannot generate error-distribution plots for dataset '{dataset_name}'"
        plot_df = dataset_plot_df.copy()
        required_plot_cols = {
            "model_variant",
            "robustness_method",
            "pipeline_id",
            "model_architecture",
            "error_value",
            "error_kind",
            "weight",
        }
        _require_columns(plot_df, required_plot_cols, context=context)
        plot_df = plot_df.copy()
        plot_df["model_variant"] = (
            plot_df["model_architecture"].astype(str).str.strip()
            + " | "
            + plot_df["robustness_method"].astype(str).str.strip()
        )
        model_variants = sorted(plot_df["model_variant"].astype(str).str.strip().unique())
        if not model_variants:
            raise ValueError(f"{context}: no model variants found.")
        variant_chunks = _chunk_values(
            model_variants,
            chunk_size=max_error_distribution_facets,
        )
        total_parts = len(variant_chunks)
        dataset_slug = _slugify_figure_value(dataset_name, field="dataset")
        for part_idx, chunk_variants in enumerate(variant_chunks, start=1):
            chunk_df = plot_df[
                plot_df["model_variant"].astype(str).isin(chunk_variants)
            ].copy()
            if chunk_df.empty:
                raise ValueError(
                    f"{context}: plot chunk {part_idx} produced no rows."
                )
            methods = sorted(
                chunk_df["robustness_method"].astype(str).str.strip().unique()
            )
            if not methods:
                raise ValueError(f"{context}: missing robustness_method values.")
            pipeline_ids = sorted(
                chunk_df["pipeline_id"].astype(str).str.strip().unique()
            )
            if not pipeline_ids:
                raise ValueError(f"{context}: missing pipeline_id values.")
            part_suffix = "" if total_parts == 1 else f"_part{part_idx:02d}"
            specs.append(
                FigureArtifactSpec(
                    figure=plot_error_distribution_overview(
                        chunk_df,
                        dataset=str(dataset_name),
                        facet_col="model_variant",
                        value_col="error_value",
                        error_kind_col="error_kind",
                        max_facets=max_error_distribution_facets,
                    ),
                    rel_parts=("1_overview", "error_distributions", dataset_slug),
                    filename=f"err_clean_vs_err_pert_violin{part_suffix}.pdf",
                    figure_type="overview_error_distribution_dataset",
                    dataset=str(dataset_name),
                    pipeline_method=methods[0] if len(methods) == 1 else None,
                    pipeline_id=pipeline_ids[0] if len(pipeline_ids) == 1 else None,
                    metric="err_clean_vs_err_pert",
                    optional=True,
                )
            )
    return summary_df, specs


def _attach_clean_sample_errors_to_scenario_samples(
    scenario_samples_df: pd.DataFrame,
    *,
    clean_df: pd.DataFrame,
    context: str,
) -> pd.DataFrame:
    join_context = f"{context}: clean/scenario sample join"
    _require_columns(
        clean_df,
        {"sample_id", "source_sample_idx", "err_clean"},
        context=join_context,
    )
    _require_columns(
        scenario_samples_df,
        {"sample_id", "source_sample_idx", "scenario", "pert_idx"},
        context=join_context,
    )
    clean_working_df = clean_df.loc[:, ["sample_id", "source_sample_idx", "err_clean"]].copy()
    clean_sample_cols = ["sample_id", "source_sample_idx"]
    for column in clean_sample_cols:
        clean_working_df[column] = require_integer_series(
            clean_working_df,
            column,
            context=join_context,
            sample_cols=clean_sample_cols,
            min_value=0,
        )
    clean_working_df["err_clean"] = require_numeric_series(
        clean_working_df["err_clean"],
        column_name="err_clean",
        context=join_context,
        allow_nan=False,
        allow_infinite=False,
    ).astype(float)
    _assert_no_duplicates(
        clean_working_df,
        ["sample_id", "source_sample_idx"],
        context=f"{join_context}: clean rows must be unique per sample key",
    )

    scenario_working_df = scenario_samples_df.copy()
    scenario_sample_cols = ["sample_id", "source_sample_idx", "scenario", "pert_idx"]
    for column in ("sample_id", "source_sample_idx"):
        scenario_working_df[column] = require_integer_series(
            scenario_working_df,
            column,
            context=join_context,
            sample_cols=scenario_sample_cols,
            min_value=0,
        )

    merged_df = scenario_working_df.merge(
        clean_working_df,
        on=["sample_id", "source_sample_idx"],
        how="left",
        validate="many_to_one",
    )
    missing_clean_mask = merged_df["err_clean"].isna()
    if missing_clean_mask.any():
        examples = _sample_records(
            merged_df.loc[missing_clean_mask, scenario_sample_cols],
            scenario_sample_cols,
        )
        raise ValueError(
            f"{join_context}: scenario samples are missing clean anchors. Examples: {examples}."
        )
    return merged_df


def _resolve_dataset_profile_seed_map(
    result_df: pd.DataFrame,
    *,
    winner_runs_by_id: Mapping[str, Any],
) -> dict[str, int]:
    _require_columns(
        result_df,
        {"dataset", "run_id"},
        context="Cannot resolve dataset-profile seed map",
    )
    seed_map: dict[str, int] = {}
    for dataset_name, dataset_df in result_df.groupby("dataset", dropna=False, sort=True):
        run_ids = sorted(dataset_df["run_id"].astype(str).drop_duplicates().tolist())
        if not run_ids:
            raise ValueError(
                "Cannot resolve dataset-profile seed map because dataset "
                f"'{dataset_name}' has no winner-pool run IDs."
            )
        seed_values: set[int] = set()
        for run_id in run_ids:
            run_obj = winner_runs_by_id.get(run_id)
            if run_obj is None:
                raise ValueError(
                    "Cannot resolve dataset-profile seed map because run_id "
                    f"'{run_id}' is missing from winner_runs_by_id."
                )
            seed_values.add(int(require_seed_tags(run_obj)["seed_data"]))
        if len(seed_values) != 1:
            raise ValueError(
                "Cannot resolve dataset-profile seed map because dataset "
                f"'{dataset_name}' has inconsistent seed_data values: {sorted(seed_values)}."
            )
        seed_map[str(dataset_name)] = int(next(iter(seed_values)))
    return seed_map


def _build_improvement_delta_overview_figure_specs(
    method_delta_plot_df: pd.DataFrame,
    *,
    perf_col: str,
    test_metric: str,
) -> list[FigureArtifactSpec]:
    if method_delta_plot_df.empty:
        return []
    required_cols = {"dataset", "robustness_method", "backbone_architecture", "pipeline_id"}
    _require_columns(
        method_delta_plot_df,
        required_cols,
        context="Cannot generate improvement delta overview heatmaps",
    )
    metric_specs = _require_core_delta_metrics(method_delta_plot_df, perf_col=perf_col)
    delta_cols = [column for column, _ in metric_specs]
    semantics = _require_plot_semantics_for_keys(
        test_metric=test_metric,
        required_keys=delta_cols,
        context="Improvement delta overview heatmap",
    )
    _assert_no_duplicates(
        method_delta_plot_df,
        ["dataset", "robustness_method", "backbone_architecture"],
        context=(
            "Cannot generate improvement delta overview heatmaps because "
            "canonical method delta rows contain duplicate "
            "(dataset, robustness_method, backbone_architecture) entries"
        ),
    )
    specs: list[FigureArtifactSpec] = []
    for dataset_name, dataset_df in method_delta_plot_df.groupby(
        "dataset",
        dropna=False,
        sort=True,
    ):
        dataset_label = str(dataset_name)
        plot_df = dataset_df[
            ["robustness_method", "backbone_architecture", *delta_cols]
        ].copy()
        if plot_df.empty:
            continue
        dataset_slug = _slugify_figure_value(dataset_label, field="dataset")
        specs.append(
            FigureArtifactSpec(
                figure=plot_improvement_deltas_heatmap(
                    plot_df,
                    metric_cols=delta_cols,
                    metric_semantics=semantics,
                    row_id_cols=["robustness_method", "backbone_architecture"],
                    title=f"{dataset_label}: Improvement Delta Overview",
                ),
                rel_parts=("3_improvements", dataset_slug, "delta_overview"),
                filename="improvement_delta_heatmap.pdf",
                figure_type="improvement_delta_overview_heatmap_dataset",
                dataset=dataset_label,
                metric="core_deltas",
                optional=True,
            )
        )
    return specs


def _slugify_figure_value(value: Any, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Cannot build figure path slug from empty {field}.")
    token = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    token = token.strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    if not token:
        raise ValueError(
            f"Cannot build figure path slug for {field}='{text}'."
        )
    return token


def _render_figure_registry(
    *,
    figure_specs: list[FigureArtifactSpec],
    output_root: str,
    full_coverage: bool,
) -> list[dict[str, Any]]:
    allowed_roots = {"1_overview", "2_baselines", "3_improvements"}
    seen_paths: set[str] = set()
    manifest_rows: list[dict[str, Any]] = []

    def _spec_sort_key(spec: FigureArtifactSpec) -> tuple[str, ...]:
        base = [
            str(spec.dataset),
            str(spec.pipeline_method),
            str(spec.pipeline_id),
            str(spec.metric),
            str(spec.figure_type),
            str(spec.filename),
        ]
        base.extend(str(part) for part in spec.rel_parts)
        return tuple(base)
    for spec in sorted(figure_specs, key=_spec_sort_key):
        if spec.figure is None:
            if spec.optional:
                continue
            raise ValueError(
                f"Core figure '{spec.figure_type}' is missing figure object."
            )
        normalized_parts = [
            str(part).strip("/").strip() for part in spec.rel_parts if str(part).strip()
        ]
        if not normalized_parts:
            raise ValueError(
                f"Figure '{spec.filename}' has empty output directory parts."
            )
        if any(part == "by_dataset" for part in normalized_parts):
            raise ValueError(
                "Figure output path may not include unsupported segment "
                "'by_dataset'."
            )
        top_level = normalized_parts[0]
        if top_level not in allowed_roots:
            raise ValueError(
                "Figure output path must start with one of "
                f"{sorted(allowed_roots)}; received '{top_level}' for "
                f"figure '{spec.filename}'."
            )
        rel_path = "/".join(normalized_parts + [spec.filename])
        dedupe_key = rel_path.lower()
        if dedupe_key in seen_paths:
            raise ValueError(
                f"Duplicate figure artifact path generated: '{rel_path}'."
            )
        seen_paths.add(dedupe_key)
        out_dir = os.path.join(output_root, *normalized_parts)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, spec.filename)
        try:
            spec.figure.write_image(out_path)
        except Exception as exc:
            if full_coverage:
                raise
            print(
                "Skipping figure artifact because full_coverage=false and image "
                f"export failed for figures/{rel_path}: {exc}"
            )
            continue
        manifest_rows.append(
            {
                "artifact_path": f"figures/{rel_path}",
                "dataset": spec.dataset,
                "pipeline_method": spec.pipeline_method,
                "pipeline_id": spec.pipeline_id,
                "metric": spec.metric,
                "figure_type": spec.figure_type,
            }
        )
    return manifest_rows


def _core_robustness_metric_keys() -> list[str]:
    return [spec.metric_key for spec in CORE_ROBUSTNESS_METRIC_SPECS]


def _core_metric_delta_improved(
    delta: pd.Series,
    *,
    higher_is_better: bool,
) -> pd.Series:
    numeric_delta = pd.to_numeric(delta, errors="raise")
    if higher_is_better:
        return numeric_delta > 0.0
    return numeric_delta < 0.0


def _correlation_metric_titles() -> dict[str, str]:
    metric_keys = _core_robustness_metric_keys()
    return {
        metric_key: robustness_metric_display_name(metric_key)
        for metric_key in metric_keys
    }


def _plot_semantics_mapping(test_metric: str) -> dict[str, PlotSemanticsRecord]:
    clean_label = f"{test_metric} (Test)"
    return {
        clean_label: PlotSemanticsRecord(
            direction="minimize",
            axis_family="numeric",
            neutral_value=None,
            display_label=clean_label,
        ),
        f"{test_metric}_test": PlotSemanticsRecord(
            direction="minimize",
            axis_family="numeric",
            neutral_value=None,
            display_label=clean_label,
        ),
        "D_w": PlotSemanticsRecord(
            direction="minimize",
            axis_family="numeric",
            neutral_value=None,
            display_label=robustness_metric_display_name("D_w"),
        ),
        "D_mean": PlotSemanticsRecord(
            direction="minimize",
            axis_family="numeric",
            neutral_value=None,
            display_label=robustness_metric_display_name("D_mean"),
        ),
        "err_pert_ws": PlotSemanticsRecord(
            direction="minimize",
            axis_family="numeric",
            neutral_value=None,
            display_label=f"{test_metric}_w",
        ),
        f"{test_metric}_w": PlotSemanticsRecord(
            direction="minimize",
            axis_family="numeric",
            neutral_value=None,
            display_label=f"{test_metric}_w",
        ),
        "err_pert_mean": PlotSemanticsRecord(
            direction="minimize",
            axis_family="numeric",
            neutral_value=None,
            display_label="Mean Perturbed Error",
        ),
        "delta_err_clean": PlotSemanticsRecord(
            direction="minimize",
            axis_family="delta",
            neutral_value=0.0,
            display_label=f"Δ {test_metric} (Test)",
        ),
        f"delta_{test_metric}_test": PlotSemanticsRecord(
            direction="minimize",
            axis_family="delta",
            neutral_value=0.0,
            display_label=f"Δ {test_metric} (Test)",
        ),
        "delta_D_w": PlotSemanticsRecord(
            direction="minimize",
            axis_family="delta",
            neutral_value=0.0,
            display_label="Δ Worst-Scenario Degradation",
        ),
        "delta_D_mean": PlotSemanticsRecord(
            direction="minimize",
            axis_family="delta",
            neutral_value=0.0,
            display_label="Δ Mean Degradation",
        ),
        "delta_err_pert_ws": PlotSemanticsRecord(
            direction="minimize",
            axis_family="delta",
            neutral_value=0.0,
            display_label=f"Δ {test_metric}_w",
        ),
        f"delta_{test_metric}_w": PlotSemanticsRecord(
            direction="minimize",
            axis_family="delta",
            neutral_value=0.0,
            display_label=f"Δ {test_metric}_w",
        ),
        "delta_D": PlotSemanticsRecord(
            direction="minimize",
            axis_family="delta",
            neutral_value=0.0,
            display_label="Δ Scenario Degradation",
        ),
        "delta_err_pert": PlotSemanticsRecord(
            direction="minimize",
            axis_family="delta",
            neutral_value=0.0,
            display_label="Δ Scenario Perturbed Error",
        ),
        "D": PlotSemanticsRecord(
            direction="minimize",
            axis_family="numeric",
            neutral_value=None,
            display_label="Scenario Degradation",
        ),
        "err_pert": PlotSemanticsRecord(
            direction="minimize",
            axis_family="numeric",
            neutral_value=None,
            display_label="Scenario Perturbed Error",
        ),
    }


def _require_plot_semantics_for_keys(
    *,
    test_metric: str,
    required_keys: Sequence[str],
    context: str,
) -> dict[str, PlotSemanticsRecord]:
    return require_plot_semantics_mapping(
        _plot_semantics_mapping(test_metric),
        required_keys=required_keys,
        context=context,
    )


def _build_method_trajectory_plot_entries(
    backbone_df: pd.DataFrame,
    canonical_method_plot_df: pd.DataFrame,
    *,
    perf_col: str,
    test_metric: str,
    trajectory_metrics: Sequence[str],
    panel_spec: Sequence[tuple[str, str]] | None = None,
    full_coverage: bool = True,
) -> list[dict[str, Any]]:
    context = "Cannot generate method trajectory plots"
    registry = _core_figure_registry()
    if panel_spec is None:
        panel_spec = registry.dataset_spec
    baseline_required_cols = {
        "dataset",
        "data_config_signature",
        "model_architecture",
        perf_col,
        *trajectory_metrics,
    }
    improvement_required_cols = {
        "dataset",
        "data_config_signature",
        "robustness_method",
        "pipeline_id",
        "model_architecture",
        "backbone_architecture",
        perf_col,
        *trajectory_metrics,
    }
    _require_columns(
        backbone_df,
        baseline_required_cols,
        context=f"{context}: baseline source",
    )
    _require_columns(
        canonical_method_plot_df,
        improvement_required_cols,
        context=f"{context}: method source",
    )
    if canonical_method_plot_df.empty:
        return []

    panel_dataset_keys = {str(dataset_key).strip() for dataset_key, _ in panel_spec}
    method_values = require_nonempty_string_series(
        canonical_method_plot_df,
        "robustness_method",
        context=f"{context}: canonical method rows",
        sample_cols=["dataset", "robustness_method", "pipeline_id", "model_architecture"],
    )

    supported_methods = set(str(method_key).strip() for method_key in registry.method_order)
    unsupported_methods = sorted(
        method for method in method_values.unique() if method not in supported_methods
    )
    if unsupported_methods:
        print(
            "Cannot generate method trajectory plots: skipping unsupported "
            "robustness methods outside core-figure trajectory scope "
            f"{unsupported_methods}."
        )

    ordered_methods = [
        str(method_key).strip()
        for method_key in registry.method_order
        if str(method_key).strip() in set(method_values.unique())
    ]
    if not ordered_methods:
        raise ValueError(
            "Cannot generate method trajectory plots: all robustness methods are "
            "outside core-figure trajectory scope after filtering unsupported "
            f"methods {unsupported_methods}."
        )
    core_backbone_df = backbone_df.loc[
        backbone_df["dataset"].astype(str).str.strip().isin(panel_dataset_keys)
    ].copy()
    core_method_df = canonical_method_plot_df.loc[
        canonical_method_plot_df["dataset"].astype(str).str.strip().isin(panel_dataset_keys)
        & canonical_method_plot_df["robustness_method"].astype(str).str.strip().isin(
            ordered_methods
        )
    ].copy()

    trajectory_entries: list[dict[str, Any]] = []
    for robustness_method in ordered_methods:
        method_improvement_df = core_method_df.loc[
            core_method_df["robustness_method"].astype(str).str.strip()
            == robustness_method
        ].copy()
        if method_improvement_df.empty:
            if not full_coverage:
                print(
                    "Cannot generate method trajectory plots: full_coverage=false, "
                    f"skipping robustness method '{robustness_method}' because it has "
                    "no core-figure datasets after filtering."
                )
                continue
            raise ValueError(
                "Cannot generate method trajectory plots: robustness method "
                f"'{robustness_method}' has no core-figure datasets after filtering."
            )

        method_dataset_keys = set(
            method_improvement_df["dataset"].astype(str).str.strip().tolist()
        )
        method_backbone_df = core_backbone_df.loc[
            core_backbone_df["dataset"].astype(str).str.strip().isin(
                method_dataset_keys
            )
        ].copy()
        if method_backbone_df.empty:
            if not full_coverage:
                print(
                    "Cannot generate method trajectory plots: full_coverage=false, "
                    f"skipping robustness method '{robustness_method}' because it has "
                    "no core-figure baseline rows."
                )
                continue
            raise ValueError(
                "Cannot generate method trajectory plots: robustness method "
                f"'{robustness_method}' has no core-figure baseline rows."
            )

        for trajectory_metric in trajectory_metrics:
            metric_title = robustness_metric_display_name(trajectory_metric)
            figure_context = (
                "Cannot generate method trajectory plots for robustness method "
                f"'{robustness_method}' and metric '{trajectory_metric}'"
            )
            trajectory_panel_spec = panel_spec
            if not full_coverage:
                trajectory_panel_spec = []
                omitted_datasets: list[str] = []
                for dataset_key, dataset_title in panel_spec:
                    baseline_subset = method_backbone_df.loc[
                        method_backbone_df["dataset"].astype(str).str.strip()
                        == dataset_key
                    ].dropna(subset=[perf_col, trajectory_metric])
                    improvement_subset = method_improvement_df.loc[
                        method_improvement_df["dataset"].astype(str).str.strip()
                        == dataset_key
                    ].dropna(subset=[perf_col, trajectory_metric])
                    if baseline_subset.empty or improvement_subset.empty:
                        omitted_datasets.append(
                            f"{dataset_key} (missing baseline/improvement metric rows)"
                        )
                        continue
                    baseline_keys = {
                        (str(signature).strip(), str(backbone).strip())
                        for signature, backbone in baseline_subset[
                            ["data_config_signature", "model_architecture"]
                        ].itertuples(index=False, name=None)
                    }
                    improvement_keys = {
                        (str(signature).strip(), str(backbone).strip())
                        for signature, backbone in improvement_subset[
                            ["data_config_signature", "backbone_architecture"]
                        ].itertuples(index=False, name=None)
                    }
                    if not baseline_keys.intersection(improvement_keys):
                        omitted_datasets.append(
                            f"{dataset_key} (no paired baseline/improvement rows)"
                        )
                        continue
                    trajectory_panel_spec.append((dataset_key, dataset_title))
                if omitted_datasets:
                    print(
                        f"{figure_context}: full_coverage=false, omitting partial-coverage "
                        f"datasets {omitted_datasets}."
                    )
                if not trajectory_panel_spec:
                    print(
                        f"{figure_context}: full_coverage=false, no renderable datasets "
                        "remain after partial-coverage filtering; skipping."
                    )
                    continue
            figure = plot_improvement_trajectory_subplots(
                method_backbone_df,
                method_improvement_df,
                panel_spec=trajectory_panel_spec,
                perf_col=perf_col,
                robust_col=trajectory_metric,
                x_semantics=_require_plot_semantics_for_keys(
                    test_metric=test_metric,
                    required_keys=[perf_col],
                    context=figure_context,
                )[perf_col],
                y_semantics=_require_plot_semantics_for_keys(
                    test_metric=test_metric,
                    required_keys=[trajectory_metric],
                    context=figure_context,
                )[trajectory_metric],
                method_col="backbone_architecture",
                backbone_col="backbone_architecture",
                baseline_backbone_col="model_architecture",
                improvement_name_col="pipeline_id",
                robustness_output_label=trajectory_output_label_for_method(
                    robustness_method
                ),
                x_title=f"{test_metric} (Test)",
                y_title=metric_title,
                require_signature=True,
                signature_col="data_config_signature",
                improvement_identity_key_cols=[
                    "dataset",
                    "data_config_signature",
                    "robustness_method",
                    "pipeline_id",
                    "model_architecture",
                ],
                improvement_join_key_cols=[
                    "dataset",
                    "data_config_signature",
                    "robustness_method",
                    "pipeline_id",
                    "backbone_architecture",
                ],
                context=figure_context,
            )
            trajectory_entries.append(
                {
                    "dataset": None,
                    "robustness_method": robustness_method,
                    "pipeline_id": None,
                    "metric": trajectory_metric,
                    "figure": figure,
                }
            )

    return trajectory_entries


def _build_method_trajectory_figure_specs(
    trajectory_entries: Sequence[Mapping[str, Any]],
) -> list[FigureArtifactSpec]:
    specs: list[FigureArtifactSpec] = []
    registry = _core_figure_registry()
    sorted_entries = _sorted_records(
        list(trajectory_entries),
        keys=["robustness_method", "metric"],
    )
    for entry in sorted_entries:
        robustness_method = parse_required_nonempty_string(
            entry.get("robustness_method"),
            key="robustness_method",
            context="Method trajectory figure entry",
        )
        pipeline_id = parse_optional_nonempty_string(
            entry.get("pipeline_id"),
            key="pipeline_id",
            context=f"Method trajectory figure entry ({robustness_method})",
            disallow_none_token=True,
        )
        metric = str(entry["metric"])
        method_slug = _slugify_figure_value(
            robustness_method,
            field="robustness_method",
        )
        metric_slug = _slugify_figure_value(metric, field="metric")
        is_core_trajectory = (
            robustness_method == registry.core_improvement_trajectory_method
            and metric == registry.core_improvement_trajectory_metric
        )
        specs.append(
            FigureArtifactSpec(
                figure=entry["figure"],
                rel_parts=("3_improvements", "trajectories", method_slug),
                filename=f"trajectory_{metric_slug}.pdf",
                figure_type="improvement_trajectory",
                dataset=None,
                pipeline_method=robustness_method,
                pipeline_id=pipeline_id,
                metric=metric,
                optional=not is_core_trajectory,
            )
        )
    return specs


def _build_baseline_clean_vs_pert_plot_entries(
    analysis_df: pd.DataFrame,
    *,
    test_metric: str,
    arch_map: Mapping[str, str],
) -> list[dict[str, Any]]:
    if analysis_df.empty:
        return []

    clean_col = f"{test_metric}_test"
    mean_pert_metric = "err_pert_mean"
    at_worst_scenario_metric = build_metric_w_name(
        test_metric,
        key="test_metric",
        context="Baseline clean-vs-pert figures",
    )
    _require_columns(
        analysis_df,
        {"dataset", "model", clean_col, at_worst_scenario_metric},
        context="Baseline clean-vs-pert figures",
    )
    scatter_df = _assign_architecture_families(
        analysis_df,
        arch_map=arch_map,
        context="Baseline clean-vs-pert figures",
    )

    plot_entries: list[dict[str, Any]] = []
    for dataset_name, dataset_df in scatter_df.groupby("dataset"):
        plot_specs = [
            (
                mean_pert_metric,
                "Mean Perturbed Error vs Clean",
                f"{test_metric} (Mean Perturbed Error)",
                "circle",
            ),
            (
                at_worst_scenario_metric,
                "Worst-Scenario Error vs Clean",
                f"{test_metric}_w",
                "diamond",
            ),
        ]
        for metric_key, title_suffix, y_title, marker_symbol in plot_specs:
            if metric_key not in dataset_df.columns:
                continue
            plot_cols = ["model", "architecture_family", clean_col, metric_key]
            plot_subset = dataset_df[plot_cols].dropna(
                subset=[clean_col, metric_key]
            )
            if plot_subset.empty:
                continue
            title = f"{dataset_name}: {title_suffix} {test_metric}"
            plot_entries.append(
                {
                    "dataset": str(dataset_name),
                    "metric": metric_key,
                    "figure": plot_perturbed_vs_clean_error(
                        plot_subset,
                        model_col="model",
                        clean_col=clean_col,
                        pert_col=metric_key,
                        color_col="architecture_family",
                        title=title,
                        x_title=f"{test_metric} (Clean)",
                        y_title=y_title,
                        marker_symbol=marker_symbol,
                    ),
                }
            )
    return plot_entries


def _build_baseline_clean_vs_worst_panel_figure_specs(
    analysis_df: pd.DataFrame,
    *,
    args: Any,
    full_coverage: bool,
) -> list[FigureArtifactSpec]:
    context = "Baseline clean-vs-worst-error panels"
    clean_col = f"{args.test_metric}_test"
    worst_col = build_metric_w_name(
        args.test_metric,
        key="test_metric",
        context=context,
    )
    _require_columns(
        analysis_df,
        {"dataset", "model", clean_col, worst_col},
        context=context,
    )
    if analysis_df.empty:
        return []

    plot_df = _require_nonempty_string_columns(
        analysis_df,
        ["dataset", "model"],
        context=context,
        sample_cols=["dataset", "model", clean_col, worst_col],
    )
    registry = _core_figure_registry()
    dataset_display_map = dict(registry.dataset_spec)
    dataset_keys = [dataset_key for dataset_key, _ in registry.dataset_spec]
    validate_raw_display_id_values(
        plot_df["dataset"].tolist(),
        raw_ids=dataset_keys,
        display_mapping=dataset_display_map,
        context=context,
        id_label="dataset",
    )

    present_dataset_keys = set(plot_df["dataset"].astype(str).str.strip().tolist())
    missing_datasets = [
        dataset_key for dataset_key in dataset_keys if dataset_key not in present_dataset_keys
    ]
    if missing_datasets and full_coverage:
        raise ValueError(
            f"{context}: missing required core-figure datasets {missing_datasets}."
        )

    panel_frames: list[tuple[str, pd.DataFrame]] = []
    omitted_datasets: list[str] = []
    for dataset_key, dataset_label in registry.dataset_spec:
        dataset_frame = plot_df.loc[
            plot_df["dataset"].astype(str).str.strip() == dataset_key,
            ["model", clean_col, worst_col],
        ].dropna(subset=[clean_col, worst_col]).copy()
        if dataset_frame.empty:
            if full_coverage:
                raise ValueError(
                    f"{context}: dataset '{dataset_key}' has no clean/worst-error rows."
                )
            omitted_datasets.append(dataset_key)
            continue
        panel_frames.append((dataset_label, dataset_frame))

    if omitted_datasets:
        print(
            f"{context}: full_coverage=false, omitting dataset panels {omitted_datasets}."
        )
    if not panel_frames:
        raise ValueError(f"{context}: no renderable clean-vs-worst dataset panels remain.")

    semantics = _require_plot_semantics_for_keys(
        test_metric=args.test_metric,
        required_keys=[clean_col, worst_col],
        context=context,
    )
    figure = plot_perturbed_vs_clean_error_panels(
        panel_frames,
        model_col="model",
        clean_col=clean_col,
        pert_col=worst_col,
        x_semantics=semantics[clean_col],
        y_semantics=semantics[worst_col],
        x_title=f"{args.test_metric}_c",
        y_title=f"{args.test_metric}_w",
    )
    metric_slug = _slugify_figure_value(worst_col, field="metric")
    return [
        FigureArtifactSpec(
            figure=figure,
            rel_parts=("2_baselines", "clean_vs_pert"),
            filename=f"clean_vs_{metric_slug}_by_dataset.pdf",
            figure_type="baseline_clean_vs_worst_by_dataset_panels",
            dataset=None,
            metric=worst_col,
            optional=True,
        )
    ]


def _build_baseline_clean_vs_pert_figure_specs(
    plot_entries: Sequence[Mapping[str, Any]],
) -> list[FigureArtifactSpec]:
    specs: list[FigureArtifactSpec] = []
    sorted_entries = _sorted_records(
        list(plot_entries),
        keys=["dataset", "metric"],
    )
    for entry in sorted_entries:
        dataset = parse_required_nonempty_string(
            entry.get("dataset"),
            key="dataset",
            context="Baseline clean-vs-pert figure entry",
            disallow_none_token=True,
        )
        metric = parse_required_nonempty_string(
            entry.get("metric"),
            key="metric",
            context=f"Baseline clean-vs-pert figure entry ({dataset})",
            disallow_none_token=True,
        )
        dataset_slug = _slugify_figure_value(dataset, field="dataset")
        metric_slug = _slugify_figure_value(metric, field="metric")
        specs.append(
            FigureArtifactSpec(
                figure=entry["figure"],
                rel_parts=("2_baselines", "clean_vs_pert"),
                filename=f"{dataset_slug}_{metric_slug}.pdf",
                figure_type="baseline_clean_vs_pert",
                dataset=dataset,
                metric=metric,
                optional=True,
            )
        )
    return specs


def _build_correlation_summary(
    analysis_df: pd.DataFrame,
    *,
    group_col: str,
    metric_keys: list[str],
    compute_corr_fn: Callable[[pd.DataFrame, str], pd.Series],
) -> pd.DataFrame:
    if analysis_df.empty:
        return pd.DataFrame()
    required_cols = {group_col, *metric_keys}
    _require_columns(
        analysis_df,
        required_cols,
        context=f"{group_col} correlation source",
    )
    all_correlations: list[pd.DataFrame] = []
    for metric in metric_keys:
        corr = compute_corr_fn(analysis_df, metric)
        correlation_df = pd.DataFrame([{group_col: "all", **corr}])
        grouped_corr_rows: list[dict[str, Any]] = []
        for group_value, group_df in analysis_df.groupby(group_col, sort=True):
            grouped_corr_rows.append(
                {
                    group_col: group_value,
                    **compute_corr_fn(group_df, metric),
                }
            )
        grouped_corr_df = pd.DataFrame(grouped_corr_rows)
        correlation_df = pd.concat([correlation_df, grouped_corr_df]).astype(
            {"count": "int32"}
        )
        correlation_df.insert(0, "metric", metric)
        all_correlations.append(correlation_df)
    if not all_correlations:
        return pd.DataFrame()
    return pd.concat(all_correlations).set_index(["metric", group_col])


def _build_correlation_heatmaps_by_metric(
    correlation_summary_df: pd.DataFrame,
    *,
    index_col: str,
    entity_label: str,
    metric_titles: Mapping[str, str],
) -> dict[str, Any]:
    if correlation_summary_df.empty:
        return {}
    correlation_rows = correlation_summary_df.reset_index()
    _require_columns(
        correlation_rows,
        {index_col, "metric", "correlation", "p_value"},
        context=f"{entity_label} correlation summary",
    )
    expected_metrics = list(metric_titles.keys())
    present_metrics = set(correlation_rows["metric"].astype(str).tolist())
    missing_metrics = [metric for metric in expected_metrics if metric not in present_metrics]
    if missing_metrics:
        raise ValueError(
            f"{entity_label} correlation summary is missing required metrics: "
            f"{missing_metrics}."
        )
    heatmaps_by_metric: dict[str, Any] = {}
    for metric in expected_metrics:
        metric_rows = correlation_rows[correlation_rows["metric"] == metric]
        corr_pivot = metric_rows.pivot(
            index=index_col,
            columns="metric",
            values="correlation",
        )
        p_values = metric_rows.pivot(
            index=index_col,
            columns="metric",
            values="p_value",
        )
        metric_title = metric_titles[metric]
        heatmaps_by_metric[metric] = plot_heatmap(
            corr_pivot,
            (
                f"{entity_label} Correlation: {metric_title} "
                f"({metric}) vs. Performance Drop"
            ),
            xlabel="Robustness Metric",
            ylabel=entity_label,
            p_values_df=p_values,
            color_scale="RdYlGn_r",
        )
    return heatmaps_by_metric


def _append_overview_correlation_figure_specs(
    *,
    figure_specs: list[FigureArtifactSpec],
    model_corr_heatmaps_by_metric: Mapping[str, Any],
    dataset_corr_heatmaps_by_metric: Mapping[str, Any],
) -> None:
    for metric, fig_obj in _sorted_items(model_corr_heatmaps_by_metric):
        metric_slug = _slugify_figure_value(metric, field="metric")
        figure_specs.append(
            FigureArtifactSpec(
                figure=fig_obj,
                rel_parts=("1_overview", "correlations"),
                filename=f"model_correlation_heatmap_{metric_slug}.pdf",
                figure_type="overview_model_correlation_heatmap",
                metric=metric,
                optional=True,
            )
        )
    for metric, fig_obj in _sorted_items(dataset_corr_heatmaps_by_metric):
        metric_slug = _slugify_figure_value(metric, field="metric")
        figure_specs.append(
            FigureArtifactSpec(
                figure=fig_obj,
                rel_parts=("1_overview", "correlations"),
                filename=f"dataset_correlation_heatmap_{metric_slug}.pdf",
                figure_type="overview_dataset_correlation_heatmap",
                metric=metric,
                optional=True,
            )
        )


def _meta_analysis_args_payload(args: Any) -> dict[str, Any]:
    if not hasattr(args, "__dict__"):
        raise ValueError(
            "Meta-analysis args payload requires argparse-style namespace values."
        )
    payload = {}
    for key, value in sorted(vars(args).items(), key=lambda item: str(item[0])):
        payload[str(key)] = normalize_yaml_value(value)
    return payload


def _build_meta_analysis_eval_context_provenance_key(
    eval_context: Mapping[str, Any],
    *,
    run_id: str,
    include_eval_data_seed: bool,
) -> tuple[tuple[str, str], ...]:
    """Normalize winner-pool eval-context provenance for meta-analysis."""
    payload = build_degradation_eval_context_tag_payload(
        eval_context,
        context_name=f"Run {run_id}",
        include_eval_data_seed=include_eval_data_seed,
        validate_optional_eval_data_seed=include_eval_data_seed,
    )
    return tuple(
        sorted(
            payload.items()
        )
    )


def _build_meta_analysis_run_tag_payload(
    *,
    reference_normalization_anchor_model: str,
    unique_datasets: Sequence[str],
    unique_models: Sequence[str],
    coverage_source: str,
    eval_data_seed_mode: str,
    resolved_meta_eval_data_seed: int | None,
    meta_eval_context_tag_payload: Mapping[str, str],
    meta_n_test_samples: int,
    meta_bootstrap_ci_semantics: str,
    meta_bootstrap_ci_resamples: int,
    meta_bootstrap_ci_confidence_level: float,
    unique_pipeline_ids: Sequence[str],
    unique_pipeline_families: Sequence[str],
) -> dict[str, str]:
    """Build the persisted MLflow tag payload for a meta-analysis run."""
    tag_payload = {
        "analysis_scope": "meta_analysis",
        "reference_normalization_anchor_model": reference_normalization_anchor_model,
        "dataset_count": str(len(unique_datasets)),
        "model_count": str(len(unique_models)),
        "dataset_list": ",".join(unique_datasets),
        "eval_data_seed_mode": eval_data_seed_mode,
        "coverage_source": str(coverage_source),
    }
    tag_payload.update(meta_eval_context_tag_payload)
    tag_payload["n_test_samples"] = str(meta_n_test_samples)
    tag_payload.update(
        build_shared_anchor_bootstrap_ci_tag_payload(
            {
                "bootstrap_ci_semantics": meta_bootstrap_ci_semantics,
                "bootstrap_ci_resamples": meta_bootstrap_ci_resamples,
                "bootstrap_ci_confidence_level": meta_bootstrap_ci_confidence_level,
            },
            require_seed=False,
            context_name="meta_analysis_bootstrap_ci_context",
        )
    )
    if resolved_meta_eval_data_seed is not None:
        tag_payload["eval_data_seed"] = str(resolved_meta_eval_data_seed)
    if unique_pipeline_ids:
        tag_payload["pipeline_ids"] = ",".join(unique_pipeline_ids)
        tag_payload["pipeline_id_count"] = str(len(unique_pipeline_ids))
    if unique_pipeline_families:
        tag_payload["pipeline_families"] = ",".join(unique_pipeline_families)
        tag_payload["pipeline_method_count"] = str(len(unique_pipeline_families))
    return tag_payload


def _report_result_nan_columns(result_df: pd.DataFrame, *, args) -> None:
    nan_cols = result_df.columns[result_df.isnull().any()].tolist()
    if not nan_cols:
        return
    expected_optional_nan_cols: set[str] = set()
    if require_improvement_selection_mode(
        args,
        context="meta-analysis args",
    ) == "clean":
        expected_optional_nan_cols.update(
            perturbed_selection_metric_keys(
                test_metric=args.test_metric,
                run_id="meta-analysis",
            )
        )
    optional_nan_cols = [
        column for column in nan_cols if column in expected_optional_nan_cols
    ]
    unexpected_nan_cols = [
        column for column in nan_cols if column not in expected_optional_nan_cols
    ]
    if optional_nan_cols:
        print(
            "Note: Perturbed-validation selector columns are unset for at "
            f"least some clean-selected runs: {optional_nan_cols}. This is "
            "expected under improvement_selection_mode='clean' and does not "
            "block clean-selector main-result analysis."
        )
    if unexpected_nan_cols:
        print(
            "Note: Some runs have missing data in columns: "
            f"{unexpected_nan_cols[:5]}"
            f"{'...' if len(unexpected_nan_cols) > 5 else ''} "
            "(likely missing optional artifacts/metadata, analysis will continue)"
        )


def _meta_analysis_run_name(args: Any, *, n_test_samples: int) -> str:
    if n_test_samples <= 0:
        raise ValueError(
            f"Meta-analysis run name requires positive n_test_samples, got {n_test_samples}."
        )
    prefix = require_namespace_nonempty_string(
        args,
        key="mlflow_experiment_prefix",
    )
    test_metric = require_namespace_nonempty_string(
        args,
        key="test_metric",
    )
    _, eval_data_seed_label = resolve_meta_analysis_eval_data_seed_scope(
        require_namespace_value(args, key="eval_data_seed"),
        key="args.eval_data_seed",
    )
    return (
        f"{prefix}-{test_metric}-{DEGRADATION_SCORING_SEMANTICS}-"
        f"eval_data_seed{eval_data_seed_label}-"
        f"n_test_samples{n_test_samples}"
    )


def _require_meta_analysis_n_test_samples(
    winner_pool_df: pd.DataFrame,
    *,
    args: Any,
) -> int:
    """Require one winner-pool n_test_samples value and exact CLI parity."""
    if winner_pool_df.empty:
        raise ValueError(
            "Meta-analysis n_test_samples resolution requires non-empty winner-pool rows."
        )
    required_cols = {"run_id", "n_test_samples"}
    missing_cols = sorted(required_cols - set(winner_pool_df.columns))
    if missing_cols:
        raise ValueError(
            "Cannot resolve meta-analysis n_test_samples: winner pool is missing required "
            f"columns {missing_cols}."
        )
    numeric_samples = require_numeric_series(
        winner_pool_df["n_test_samples"],
        column_name="n_test_samples",
        context="meta-analysis winner pool",
        allow_nan=False,
        allow_infinite=False,
    )
    numeric_values = numeric_samples.to_numpy(dtype=float)
    if not np.allclose(numeric_values, np.floor(numeric_values)):
        raise ValueError(
            "Cannot resolve meta-analysis n_test_samples: winner pool contains non-integer "
            "n_test_samples values."
        )
    integer_samples = numeric_samples.astype(int)
    if (integer_samples <= 0).any():
        examples = (
            winner_pool_df.loc[
                integer_samples <= 0,
                ["run_id", "n_test_samples"],
            ]
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Cannot resolve meta-analysis n_test_samples: winner pool contains non-positive "
            f"n_test_samples values. Examples: {examples}."
        )
    cli_n_test_samples = coerce_int(require_namespace_value(args, key="n_test_samples"))
    if cli_n_test_samples is None:
        raise ValueError(
            "Meta-analysis requires explicit n_test_samples from CLI/config."
        )
    if cli_n_test_samples <= 0:
        raise ValueError(
            f"Meta-analysis requires positive n_test_samples, got {cli_n_test_samples}."
        )
    unique_counts = sorted(integer_samples.unique().tolist())
    if len(unique_counts) != 1:
        counts_by_run = winner_pool_df[["run_id"]].copy()
        counts_by_run["n_test_samples"] = integer_samples
        examples = (
            counts_by_run.drop_duplicates()
            .sort_values(["n_test_samples", "run_id"])
            .head(8)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Meta-analysis detected inconsistent n_test_samples across tested winner "
            f"runs. Found values: {unique_counts}. Examples: {examples}."
        )
    winner_pool_n_test_samples = int(unique_counts[0])
    if cli_n_test_samples != winner_pool_n_test_samples:
        raise ValueError(
            "Meta-analysis n_test_samples mismatch: args.n_test_samples="
            f"{cli_n_test_samples} but tested winner runs use "
            f"{winner_pool_n_test_samples}. Re-run with --n_test_samples "
            f"{winner_pool_n_test_samples} or retest winners with "
            f"--n_test_samples {cli_n_test_samples}."
        )
    return winner_pool_n_test_samples


def _assert_core_figure_specs_non_optional(
    figure_specs: list[FigureArtifactSpec],
) -> None:
    violations = sorted(
        {
            spec.figure_type
            for spec in figure_specs
            if spec.figure_type in CORE_REQUIRED_FIGURE_TYPES and spec.optional
        }
    )
    if violations:
        raise ValueError(
            "Core figure specs must be marked optional=False. "
            f"Violating types: {violations}."
        )


def _load_required_degradation_artifact_bundle(
    *,
    client: mlflow.MlflowClient,
    run_id: str,
    test_metric: str,
    eval_data_seed: int,
    expected_idx_to_name: Mapping[int, str],
    expected_n_test_samples: int,
    expected_clean_metric_value: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return download_validated_degradation_artifact_bundle(
        client,
        run_id=run_id,
        test_metric=test_metric,
        eval_data_seed=eval_data_seed,
        expected_idx_to_name=expected_idx_to_name,
        expected_n_test_samples=expected_n_test_samples,
        expected_clean_metric_value=expected_clean_metric_value,
        context_name=f"Run {run_id} degradation artifact bundle",
    )


def _registered_channel_scope_for_scenario(
    scenario_name: Any,
    *,
    context: str,
) -> str:
    scenario_key = parse_required_nonempty_string(
        scenario_name,
        key="scenario",
        context=context,
    )
    if scenario_key not in PERTURBATION_REGISTRY:
        raise ValueError(
            f"{context}: scenario {scenario_key!r} is not registered in "
            "PERTURBATION_REGISTRY."
        )
    return require_perturbation_channel_scope(
        PERTURBATION_REGISTRY[scenario_key],
        context=f"{context}: scenario {scenario_key!r}",
    )


def _fixed_channel_fraction_channel_scoped_values(
    scenario_summary_df: pd.DataFrame,
    *,
    context: str,
) -> dict[str, float | int]:
    _require_columns(
        scenario_summary_df,
        {"scenario", "D"},
        context=context,
    )
    if scenario_summary_df.empty:
        raise ValueError(f"{context}: scenario_summary_df is empty.")
    working = scenario_summary_df.copy()
    working["scenario"] = working["scenario"].astype(str).str.strip()
    working["channel_scope"] = [
        _registered_channel_scope_for_scenario(
            scenario_name,
            context=context,
        )
        for scenario_name in working["scenario"]
    ]
    channel_scoped = working["channel_scope"].isin(["continuous", "discrete"])
    channel_scoped_count = int(channel_scoped.sum())
    all_scope_count = int((working["channel_scope"] == "all").sum())
    if channel_scoped_count == 0:
        return {
            "D_w": float("nan"),
            "D_mean": float("nan"),
            "channel_scoped_scenario_count": 0,
            "all_scope_scenario_count": all_scope_count,
        }
    d_values = pd.to_numeric(working.loc[channel_scoped, "D"], errors="raise")
    return {
        "D_w": float(d_values.max()),
        "D_mean": float(d_values.mean()),
        "channel_scoped_scenario_count": channel_scoped_count,
        "all_scope_scenario_count": all_scope_count,
    }


def _fixed_channel_fraction_count_summary(
    scenario_samples_df: pd.DataFrame,
    *,
    context: str,
) -> dict[str, float]:
    _require_columns(
        scenario_samples_df,
        {
            "intensity_severity",
            "channel_scope",
            "derived_fixed_channel_count",
        },
        context=context,
    )
    scoped = scenario_samples_df.loc[
        scenario_samples_df["channel_scope"].astype(str).str.strip().isin(
            ["continuous", "discrete"]
        )
    ].copy()
    positive = scoped.loc[
        pd.to_numeric(scoped["intensity_severity"], errors="raise") > 0.0
    ]
    if positive.empty:
        return {
            "fixed_channel_count_min": float("nan"),
            "fixed_channel_count_median": float("nan"),
            "fixed_channel_count_max": float("nan"),
        }
    counts = pd.to_numeric(
        positive["derived_fixed_channel_count"],
        errors="raise",
    )
    return {
        "fixed_channel_count_min": float(counts.min()),
        "fixed_channel_count_median": float(counts.median()),
        "fixed_channel_count_max": float(counts.max()),
    }


def _parse_fixed_channel_fraction_args(args: Any) -> tuple[float | None, float]:
    max_fraction = parse_perturbation_channel_fraction_max(
        require_namespace_value(args, key="perturbation_channel_fraction_max"),
        key="perturbation_channel_fraction_max",
    )
    fixed_fraction = parse_optional_unit_float(
        getattr(args, "fixed_channel_fraction", None),
        key="fixed_channel_fraction",
        max_value=max_fraction,
    )
    return fixed_fraction, max_fraction


def _build_fixed_channel_fraction_table(
    *,
    result_df: pd.DataFrame,
    winner_runs_by_id: Mapping[str, Any],
    client: mlflow.MlflowClient,
    args: Any,
) -> pd.DataFrame:
    fixed_fraction, max_fraction = _parse_fixed_channel_fraction_args(args)
    if fixed_fraction is None:
        return pd.DataFrame(columns=FIXED_CHANNEL_FRACTION_COLUMNS)
    _require_columns(
        result_df,
        {
            "dataset",
            "model_architecture",
            "robustness_method",
            "pipeline_id",
            "run_id",
            "data_config_signature",
        },
        context="fixed-channel-fraction meta-analysis",
    )
    full_coverage = require_namespace_bool(args, key="full_coverage")
    rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []
    for result_row in result_df.itertuples(index=False):
        run_id = str(result_row.run_id)
        run = winner_runs_by_id.get(run_id)
        if run is None:
            raise ValueError(
                "fixed-channel-fraction meta-analysis cannot find canonical "
                f"winner run {run_id!r} in winner_runs_by_id."
            )
        try:
            complete = is_fixed_channel_fraction_complete(
                run,
                args=args,
                client=client,
                fixed_fraction=fixed_fraction,
            )
        except Exception as exc:
            if full_coverage:
                raise
            skipped_rows.append(
                {
                    "dataset": str(result_row.dataset),
                    "run_id": run_id,
                    "reason": str(exc),
                }
            )
            continue
        if not complete:
            reason = "missing_or_incomplete_fixed_channel_fraction"
            if full_coverage:
                raise ValueError(
                    f"Run {run_id} is missing complete fixed-channel-fraction "
                    f"outputs for fixed_channel_fraction={fixed_fraction}."
                )
            skipped_rows.append(
                {
                    "dataset": str(result_row.dataset),
                    "run_id": run_id,
                    "reason": reason,
                }
            )
            continue

        tags = run.data.tags
        if tags is None:
            raise ValueError(f"Run {run_id} is missing tags.")
        params = run.data.params
        if params is None:
            raise ValueError(f"Run {run_id} is missing params.")
        run_dataset = require_nonempty_tag_value(
            tags,
            key="dataset",
            run_id=run_id,
        )
        if run_dataset != str(result_row.dataset):
            raise ValueError(
                "fixed-channel-fraction meta-analysis result row has "
                f"dataset={str(result_row.dataset)!r} but run {run_id} is tagged "
                f"dataset={run_dataset!r}."
            )
        run_data_config_signature = require_nonempty_tag_value(
            tags,
            key="data_config_signature",
            run_id=run_id,
        )
        if run_data_config_signature != str(result_row.data_config_signature):
            raise ValueError(
                "fixed-channel-fraction meta-analysis result row has "
                f"data_config_signature={str(result_row.data_config_signature)!r} but "
                f"run {run_id} is tagged data_config_signature="
                f"{run_data_config_signature!r}."
            )
        eval_context = require_degradation_eval_context_tags(tags, run_id=run_id)
        bootstrap_ci_context = require_shared_anchor_bootstrap_ci_context_tags(
            tags,
            run_id=run_id,
            require_seed=True,
        )
        params_signature = require_nonempty_tag_value(
            tags,
            key="perturbation_scenario_params_signature",
            run_id=run_id,
        )
        canonical_context_signature = build_canonical_degradation_context_signature(
            degradation_eval_context=eval_context,
            bootstrap_ci_context=bootstrap_ci_context,
            perturbation_scenario_params_signature=params_signature,
        )
        test_metric = str(eval_context["test_metric"])
        idx_to_name = eval_context["perturbation_idx_name_map"]
        canonical_worst_scenario = require_logged_degradation_metric_bundle(
            run.data.metrics,
            tags=tags,
            params=params,
            run_id=run_id,
            test_metric=test_metric,
            expected_idx_to_name=idx_to_name,
        )
        fixed_fraction_worst_scenario = (
            require_logged_fixed_channel_fraction_metric_bundle(
                run.data.metrics,
                tags=tags,
                run_id=run_id,
                test_metric=test_metric,
                fixed_channel_fraction=fixed_fraction,
                perturbation_channel_fraction_max=max_fraction,
                expected_idx_to_name=idx_to_name,
            )
        )
        canonical_clean_df, _, canonical_summary_df = _load_required_degradation_artifact_bundle(
            client=client,
            run_id=run_id,
            test_metric=test_metric,
            eval_data_seed=int(eval_context["eval_data_seed"]),
            expected_idx_to_name=idx_to_name,
            expected_n_test_samples=int(eval_context["n_test_samples"]),
            expected_clean_metric_value=run.data.metrics[f"{test_metric}_test"],
        )
        context_signature_tag = build_fixed_channel_fraction_tag_key(
            fixed_channel_fraction=fixed_fraction,
            perturbation_channel_fraction_max=max_fraction,
            tag_name="context_signature",
        )
        fixed_fraction_context_signature = parse_required_nonempty_string(
            tags.get(context_signature_tag),
            key=context_signature_tag,
            context=f"Run {run_id}",
        )
        _, fixed_fraction_samples_df, fixed_fraction_summary_df, _ = (
            download_validated_fixed_channel_fraction_artifact_bundle(
                client,
                run_id=run_id,
                test_metric=test_metric,
                eval_data_seed=int(eval_context["eval_data_seed"]),
                fixed_channel_fraction=fixed_fraction,
                perturbation_channel_fraction_max=max_fraction,
                expected_idx_to_name=idx_to_name,
                expected_n_test_samples=int(eval_context["n_test_samples"]),
                expected_clean_df=canonical_clean_df,
                expected_context_signature=fixed_fraction_context_signature,
                expected_perturbation_scenarios_signature=str(
                    eval_context["perturbation_scenarios_signature"]
                ),
                expected_perturbation_scenario_params_signature=params_signature,
                expected_canonical_context_signature=canonical_context_signature,
                expected_bootstrap_ci_context=bootstrap_ci_context,
                context_name=(
                    f"Run {run_id} fixed-channel-fraction meta-analysis bundle"
                ),
            )
        )
        canonical_scoped = _fixed_channel_fraction_channel_scoped_values(
            canonical_summary_df,
            context=f"Run {run_id} canonical fixed-fraction comparison",
        )
        fixed_fraction_scoped = _fixed_channel_fraction_channel_scoped_values(
            fixed_fraction_summary_df,
            context=f"Run {run_id} fixed-fraction comparison",
        )
        count_summary = _fixed_channel_fraction_count_summary(
            fixed_fraction_samples_df,
            context=f"Run {run_id} fixed-fraction count summary",
        )
        canonical_d_w = require_float_metric(
            run.data.metrics,
            run_id=run_id,
            metric_key=build_degradation_metric_key(
                test_metric=test_metric,
                metric_name="D_w",
            ),
        )
        fixed_fraction_d_w = require_float_metric(
            run.data.metrics,
            run_id=run_id,
            metric_key=build_fixed_channel_fraction_metric_key(
                test_metric=test_metric,
                fixed_channel_fraction=fixed_fraction,
                perturbation_channel_fraction_max=max_fraction,
                metric_name="D_w",
            ),
        )
        canonical_d_mean = require_float_metric(
            run.data.metrics,
            run_id=run_id,
            metric_key=build_degradation_metric_key(
                test_metric=test_metric,
                metric_name="D_mean",
            ),
        )
        fixed_fraction_d_mean = require_float_metric(
            run.data.metrics,
            run_id=run_id,
            metric_key=build_fixed_channel_fraction_metric_key(
                test_metric=test_metric,
                fixed_channel_fraction=fixed_fraction,
                perturbation_channel_fraction_max=max_fraction,
                metric_name="D_mean",
            ),
        )
        row = {
            "dataset": str(result_row.dataset),
            "model_architecture": str(result_row.model_architecture),
            "robustness_method": str(result_row.robustness_method),
            "pipeline_id": str(result_row.pipeline_id),
            "run_id": run_id,
            "data_config_signature": str(result_row.data_config_signature),
            "fixed_channel_fraction": float(fixed_fraction),
            "eval_data_seed": int(eval_context["eval_data_seed"]),
            "n_test_samples": int(eval_context["n_test_samples"]),
            "canonical_D_w": float(canonical_d_w),
            "fixed_fraction_D_w": float(fixed_fraction_d_w),
            "delta_D_w_fixed_fraction": float(fixed_fraction_d_w - canonical_d_w),
            "canonical_D_mean": float(canonical_d_mean),
            "fixed_fraction_D_mean": float(fixed_fraction_d_mean),
            "delta_D_mean_fixed_fraction": float(
                fixed_fraction_d_mean - canonical_d_mean
            ),
            "canonical_worst_scenario": canonical_worst_scenario,
            "fixed_fraction_worst_scenario": fixed_fraction_worst_scenario,
            "canonical_channel_scoped_D_w": canonical_scoped["D_w"],
            "fixed_fraction_channel_scoped_D_w": fixed_fraction_scoped["D_w"],
            "delta_channel_scoped_D_w_fixed_fraction": (
                fixed_fraction_scoped["D_w"] - canonical_scoped["D_w"]
            ),
            "canonical_channel_scoped_D_mean": canonical_scoped["D_mean"],
            "fixed_fraction_channel_scoped_D_mean": fixed_fraction_scoped["D_mean"],
            "delta_channel_scoped_D_mean_fixed_fraction": (
                fixed_fraction_scoped["D_mean"] - canonical_scoped["D_mean"]
            ),
            "channel_scoped_scenario_count": int(
                fixed_fraction_scoped["channel_scoped_scenario_count"]
            ),
            "all_scope_scenario_count": int(
                fixed_fraction_scoped["all_scope_scenario_count"]
            ),
            **count_summary,
            "canonical_context_signature": canonical_context_signature,
            "fixed_fraction_context_signature": fixed_fraction_context_signature,
        }
        rows.append(row)

    if skipped_rows:
        warnings.warn(
            "Fixed-channel-fraction meta-analysis omitted incomplete fixed-fraction rows "
            f"in no-full-coverage mode. Skipped={len(skipped_rows)}. "
            f"Examples: {skipped_rows[:5]}.",
            stacklevel=2,
        )
    return pd.DataFrame(rows, columns=FIXED_CHANNEL_FRACTION_COLUMNS)


def _fixed_channel_fraction_baseline_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["robustness_method"].astype(str).str.strip().eq("baseline")
        & df["pipeline_id"].astype(str).str.strip().eq("baseline")
    )


def _agreement_token(matches: int, total: int) -> str:
    if matches < 0 or total < 0 or matches > total:
        raise ValueError(
            f"Agreement counts must satisfy 0 <= matches <= total; got {matches}/{total}."
        )
    return f"{matches}/{total}"


def _signed_delta_token(value: float) -> str:
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError(f"Delta sign is undefined for non-finite value {numeric_value}.")
    if numeric_value < 0.0:
        return "negative"
    if numeric_value > 0.0:
        return "positive"
    return "zero"


def _fixed_fraction_scenario_family_lookup() -> dict[str, str]:
    registry = _core_figure_registry()
    lookup: dict[str, str] = {}
    for family_name, scenario_names in registry.scenario_groups.items():
        family_key = str(family_name).strip()
        for scenario_name in scenario_names:
            scenario_key = str(scenario_name).strip()
            existing = lookup.get(scenario_key)
            if existing is not None:
                raise ValueError(
                    f"Scenario {scenario_key!r} is mapped to multiple families: "
                    f"{existing!r} and {family_key!r}."
                )
            lookup[scenario_key] = family_key
    return lookup


def _scenario_family_for_summary(
    scenario_name: Any,
    *,
    family_lookup: Mapping[str, str],
    context: str,
) -> str:
    scenario_key = parse_required_nonempty_string(
        scenario_name,
        key="scenario",
        context=context,
    )
    family = family_lookup.get(scenario_key)
    if family is None:
        raise ValueError(
            f"{context}: scenario {scenario_key!r} is not mapped by "
            "configs/reporting/core_figures.yaml CORE_SCENARIO_GROUPS."
        )
    return family


def _baseline_rank_spearman_rho(
    baseline_df: pd.DataFrame,
    *,
    dataset_name: str,
    full_coverage: bool,
) -> float:
    context = (
        "fixed-channel-fraction paper summary Spearman rho "
        f"for dataset={dataset_name}"
    )
    canonical_values = require_numeric_series(
        baseline_df["canonical_D_w"],
        column_name="canonical_D_w",
        context=context,
    )
    fixed_fraction_values = require_numeric_series(
        baseline_df["fixed_fraction_D_w"],
        column_name="fixed_fraction_D_w",
        context=context,
    )
    if len(baseline_df) < 2:
        if full_coverage:
            raise ValueError(
                f"{context} is undefined because at least two baseline rows are required."
            )
        return float("nan")
    if (
        canonical_values.nunique(dropna=False) < 2
        or fixed_fraction_values.nunique(dropna=False) < 2
    ):
        if full_coverage:
            raise ValueError(
                f"{context} is undefined because baseline D_w ranks are constant."
            )
        return float("nan")
    rho = canonical_values.corr(fixed_fraction_values, method="spearman")
    if not np.isfinite(float(rho)):
        if full_coverage:
            raise ValueError(f"{context} is undefined.")
        return float("nan")
    return float(rho)


def _weakest_family_agreement(
    baseline_df: pd.DataFrame,
    *,
    family_lookup: Mapping[str, str],
    dataset_name: str,
) -> str:
    matches = 0
    total = 0
    for row in baseline_df.itertuples(index=False):
        context = (
            "fixed-channel-fraction paper summary weakest-family agreement "
            f"for dataset={dataset_name}, model_architecture={row.model_architecture}"
        )
        canonical_family = _scenario_family_for_summary(
            row.canonical_worst_scenario,
            family_lookup=family_lookup,
            context=f"{context} canonical",
        )
        fixed_fraction_family = _scenario_family_for_summary(
            row.fixed_fraction_worst_scenario,
            family_lookup=family_lookup,
            context=f"{context} fixed fraction",
        )
        total += 1
        if canonical_family == fixed_fraction_family:
            matches += 1
    return _agreement_token(matches, total)


def _core_method_sign_agreement(
    dataset_df: pd.DataFrame,
    *,
    dataset_name: str,
    full_coverage: bool,
) -> str:
    baseline_df = dataset_df.loc[_fixed_channel_fraction_baseline_mask(dataset_df)].copy()
    baseline_join_keys = [
        "dataset",
        "model_architecture",
        "data_config_signature",
        "fixed_channel_fraction",
    ]
    _assert_no_duplicates(
        baseline_df,
        baseline_join_keys,
        context=(
            "fixed-channel-fraction paper summary baseline rows are not unique per "
            f"{baseline_join_keys}"
        ),
    )
    baseline_join_df = baseline_df[
        baseline_join_keys
        + [
            "canonical_D_w",
            "fixed_fraction_D_w",
        ]
    ].copy()

    method_df = dataset_df.loc[
        dataset_df["robustness_method"].isin(_FIXED_CHANNEL_FRACTION_CORE_METHODS)
    ].copy()
    if method_df.empty:
        if full_coverage:
            raise ValueError(
                "fixed-channel-fraction paper summary requires all core methods "
                f"for dataset={dataset_name}; missing "
                f"{list(_FIXED_CHANNEL_FRACTION_CORE_METHODS)}."
            )
        return _agreement_token(0, 0)
    present_methods = set(method_df["robustness_method"].unique())
    missing_methods = [
        method_name
        for method_name in _FIXED_CHANNEL_FRACTION_CORE_METHODS
        if method_name not in present_methods
    ]
    if missing_methods and full_coverage:
        raise ValueError(
            "fixed-channel-fraction paper summary requires all core methods "
            f"for dataset={dataset_name}; missing {missing_methods}."
        )
    method_identity_keys = baseline_join_keys + ["robustness_method"]
    _assert_no_duplicates(
        method_df,
        method_identity_keys,
        context=(
            "fixed-channel-fraction paper summary core-method rows are not unique per "
            f"{method_identity_keys}"
        ),
    )
    merged = method_df.merge(
        baseline_join_df,
        on=baseline_join_keys,
        how="left",
        suffixes=("_method", "_baseline"),
        indicator=True,
    )
    missing_baseline = merged["_merge"] != "both"
    if missing_baseline.any():
        examples = _sample_records(
            merged.loc[missing_baseline],
            method_identity_keys,
        )
        raise ValueError(
            "fixed-channel-fraction paper summary cannot compute core-method signs "
            f"for dataset={dataset_name}: missing paired baseline rows. "
            f"Examples: {examples}."
        )
    merged["canonical_delta_D_w"] = (
        pd.to_numeric(merged["canonical_D_w_method"], errors="raise")
        - pd.to_numeric(merged["canonical_D_w_baseline"], errors="raise")
    )
    merged["fixed_fraction_delta_D_w"] = (
        pd.to_numeric(merged["fixed_fraction_D_w_method"], errors="raise")
        - pd.to_numeric(merged["fixed_fraction_D_w_baseline"], errors="raise")
    )

    matches = 0
    total = 0
    for method_name in _FIXED_CHANNEL_FRACTION_CORE_METHODS:
        method_rows = merged.loc[merged["robustness_method"] == method_name]
        if method_rows.empty:
            continue
        canonical_mean_delta = float(method_rows["canonical_delta_D_w"].mean())
        fixed_fraction_mean_delta = float(method_rows["fixed_fraction_delta_D_w"].mean())
        total += 1
        if _signed_delta_token(canonical_mean_delta) == _signed_delta_token(
            fixed_fraction_mean_delta
        ):
            matches += 1
    return _agreement_token(matches, total)


def _build_fixed_channel_fraction_paper_summary_table(
    fixed_channel_fraction_df: pd.DataFrame,
    *,
    full_coverage: bool = True,
) -> pd.DataFrame:
    """Build the compact appendix table from the validated fixed-fraction table."""
    if fixed_channel_fraction_df.empty:
        return pd.DataFrame(columns=FIXED_CHANNEL_FRACTION_PAPER_SUMMARY_COLUMNS)
    required_cols = {
        "dataset",
        "model_architecture",
        "robustness_method",
        "pipeline_id",
        "data_config_signature",
        "fixed_channel_fraction",
        "canonical_D_w",
        "fixed_fraction_D_w",
        "canonical_worst_scenario",
        "fixed_fraction_worst_scenario",
    }
    _require_columns(
        fixed_channel_fraction_df,
        required_cols,
        context="fixed-channel-fraction paper summary",
    )
    work = fixed_channel_fraction_df.copy()
    identity_sample_cols = [
        "dataset",
        "model_architecture",
        "robustness_method",
        "pipeline_id",
        "data_config_signature",
    ]
    for column in (
        "dataset",
        "model_architecture",
        "robustness_method",
        "pipeline_id",
        "data_config_signature",
    ):
        work[column] = require_nonempty_string_series(
            work,
            column,
            context="fixed-channel-fraction paper summary",
            sample_cols=identity_sample_cols,
        )
        null_like_mask = work[column].str.lower().isin({"nan", "none"})
        if null_like_mask.any():
            examples = _sample_records(
                work.loc[null_like_mask],
                identity_sample_cols,
            )
            raise ValueError(
                "fixed-channel-fraction paper summary requires real "
                f"{column!r} values, not null-like string tokens. "
                f"Examples: {examples}."
            )
    work["fixed_channel_fraction"] = require_numeric_series(
        work["fixed_channel_fraction"],
        column_name="fixed_channel_fraction",
        context="fixed-channel-fraction paper summary",
    ).astype(float)
    invalid_fraction_mask = (
        (work["fixed_channel_fraction"] <= 0.0)
        | (work["fixed_channel_fraction"] > 1.0)
    )
    if invalid_fraction_mask.any():
        examples = _sample_records(
            work.loc[invalid_fraction_mask],
            identity_sample_cols + ["fixed_channel_fraction"],
        )
        raise ValueError(
            "fixed-channel-fraction paper summary requires "
            "0 < fixed_channel_fraction <= 1.0. "
            f"Examples: {examples}."
        )
    fixed_fractions = sorted(work["fixed_channel_fraction"].drop_duplicates().tolist())
    if len(fixed_fractions) != 1:
        raise ValueError(
            "fixed-channel-fraction paper summary requires a single "
            f"fixed_channel_fraction value; got {fixed_fractions}."
        )
    work["canonical_D_w"] = require_numeric_series(
        work["canonical_D_w"],
        column_name="canonical_D_w",
        context="fixed-channel-fraction paper summary",
    )
    work["fixed_fraction_D_w"] = require_numeric_series(
        work["fixed_fraction_D_w"],
        column_name="fixed_fraction_D_w",
        context="fixed-channel-fraction paper summary",
    )

    rows: list[dict[str, Any]] = []
    family_lookup = _fixed_fraction_scenario_family_lookup()
    for dataset_name, dataset_df in work.groupby("dataset", sort=True):
        baseline_df = dataset_df.loc[_fixed_channel_fraction_baseline_mask(dataset_df)]
        if baseline_df.empty:
            raise ValueError(
                "fixed-channel-fraction paper summary requires at least one baseline "
                f"row for dataset={dataset_name}."
            )
        rows.append(
            {
                "dataset": dataset_name,
                "spearman_rho": _baseline_rank_spearman_rho(
                    baseline_df,
                    dataset_name=str(dataset_name),
                    full_coverage=full_coverage,
                ),
                "weakest_family_agreement": _weakest_family_agreement(
                    baseline_df,
                    family_lookup=family_lookup,
                    dataset_name=str(dataset_name),
                ),
                "core_method_sign_agreement": _core_method_sign_agreement(
                    dataset_df,
                    dataset_name=str(dataset_name),
                    full_coverage=full_coverage,
                ),
            }
        )
    return pd.DataFrame(rows, columns=FIXED_CHANNEL_FRACTION_PAPER_SUMMARY_COLUMNS)


def _select_forecast_extreme_rows(
    scenario_samples_df: pd.DataFrame,
    *,
    canonical_method_df: pd.DataFrame,
    top_k: int,
    score_metric: str,
) -> pd.DataFrame:
    if top_k <= 0:
        raise ValueError(
            f"forecast_extreme_top_k must be positive, received {top_k}."
        )
    output_columns = [
        "dataset",
        "data_config_signature",
        "pipeline_method",
        "pipeline_kind",
        "robustness_method",
        "pipeline_id",
        "run_id",
        "model_architecture",
        "backbone_architecture",
        "sample_id",
        "source_sample_idx",
        "pert_idx",
        "scenario",
        "severity",
        "sample_score",
        "score_metric",
        "extreme_kind",
        "extreme_rank",
    ]
    if scenario_samples_df.empty:
        return pd.DataFrame(columns=output_columns)
    required_cols = {
        "dataset",
        "data_config_signature",
        "pipeline_method",
        "pipeline_kind",
        "robustness_method",
        "pipeline_id",
        "run_id",
        "model_architecture",
        "backbone_architecture",
        "sample_id",
        "source_sample_idx",
        "pert_idx",
        "scenario",
        "severity",
        "err_pert",
    }
    _require_columns(
        scenario_samples_df,
        required_cols,
        context="Cannot select forecast-extreme rows",
    )
    working_df = _filter_rows_to_canonical_method_winners(
        scenario_samples_df,
        canonical_method_df=canonical_method_df,
        context="Cannot select forecast-extreme rows",
        drop_out_of_scope_methods=True,
    ).copy()
    if working_df.empty:
        return pd.DataFrame(columns=output_columns)

    _assert_single_pipeline_per_method_backbone(
        working_df[
            [
                "dataset",
                "robustness_method",
                "backbone_architecture",
                "pipeline_id",
            ]
        ].drop_duplicates(),
        context="Cannot select forecast-extreme rows",
    )
    for column in (
        "dataset",
        "data_config_signature",
        "pipeline_method",
        "pipeline_kind",
        "robustness_method",
        "pipeline_id",
        "run_id",
        "model_architecture",
        "backbone_architecture",
        "scenario",
    ):
        if working_df[column].isna().any():
            examples = _sample_records(
                working_df.loc[working_df[column].isna()],
                [
                    "dataset",
                    "robustness_method",
                    "pipeline_id",
                    "run_id",
                ],
            )
            raise ValueError(
                f"Cannot select forecast-extreme rows because '{column}' has missing "
                f"values. Examples: {examples}."
            )
        working_df[column] = working_df[column].astype(str).str.strip()
        if (working_df[column] == "").any():
            examples = _sample_records(
                working_df.loc[working_df[column] == ""],
                [
                    "dataset",
                    "robustness_method",
                    "pipeline_id",
                    "run_id",
                ],
            )
            raise ValueError(
                f"Cannot select forecast-extreme rows because '{column}' has empty "
                f"values. Examples: {examples}."
            )
    for column in ("sample_id", "source_sample_idx", "pert_idx"):
        working_df[column] = require_integer_series(
            working_df,
            column,
            context="Cannot select forecast-extreme rows",
            sample_cols=(
                "dataset",
                "run_id",
                "sample_id",
                "source_sample_idx",
                "pert_idx",
            ),
            min_value=0,
        )
    working_df["severity"] = pd.to_numeric(
        working_df["severity"],
        errors="raise",
    ).astype(float)
    working_df["sample_score"] = pd.to_numeric(
        working_df["err_pert"],
        errors="raise",
    ).astype(float)
    working_df["score_metric"] = str(score_metric)

    group_cols = [
        "dataset",
        "data_config_signature",
        "pipeline_method",
        "pipeline_kind",
        "robustness_method",
        "pipeline_id",
        "run_id",
        "model_architecture",
        "backbone_architecture",
    ]
    selected_frames: list[pd.DataFrame] = []
    for _, group_df in working_df.groupby(group_cols, dropna=False, sort=True):
        worst_df = group_df.sort_values(
            ["sample_score", "severity", "scenario", "sample_id"],
            ascending=[False, True, True, True],
            kind="mergesort",
        ).head(top_k).copy()
        worst_df["extreme_kind"] = "worst"
        worst_df["extreme_rank"] = np.arange(1, len(worst_df) + 1, dtype=int)
        selected_frames.append(worst_df)

        best_df = group_df.sort_values(
            ["sample_score", "severity", "scenario", "sample_id"],
            ascending=[True, False, True, True],
            kind="mergesort",
        ).head(top_k).copy()
        best_df["extreme_kind"] = "best"
        best_df["extreme_rank"] = np.arange(1, len(best_df) + 1, dtype=int)
        selected_frames.append(best_df)

    selected_df = pd.concat(selected_frames, ignore_index=True)
    return selected_df.loc[:, output_columns].sort_values(
        ["dataset", "robustness_method", "backbone_architecture", "extreme_kind", "extreme_rank"],
        kind="mergesort",
    ).reset_index(drop=True)


def _is_optional_forecast_extreme_cuda_runtime_error(exc: Exception) -> bool:
    message = str(exc)
    if not message:
        return False
    normalized = message.lower()
    cuda_markers = ("cuda", "cudacachingallocator", "nvml", "nvidia")
    return any(marker in normalized for marker in cuda_markers)


def _build_forecast_extreme_plot_entries(
    forecast_extremes_df: pd.DataFrame,
    *,
    result_df: pd.DataFrame,
    winner_runs_by_id: Mapping[str, Any],
    client: mlflow.MlflowClient,
    args: Any,
    resolved_specs: Sequence[Any],
) -> list[dict[str, Any]]:
    if forecast_extremes_df.empty:
        return []
    _require_columns(
        forecast_extremes_df,
        {
            "dataset",
            "robustness_method",
            "pipeline_id",
            "run_id",
            "backbone_architecture",
            "sample_id",
            "source_sample_idx",
            "pert_idx",
            "scenario",
            "severity",
            "sample_score",
            "score_metric",
            "extreme_kind",
            "extreme_rank",
        },
        context="Cannot rerender forecast-extreme figures",
    )
    if not hasattr(args, "_window_arg_overrides"):
        raise ValueError(
            "Meta-analysis forecast-extreme rerendering requires "
            "args._window_arg_overrides."
        )

    extreme_rank = require_integer_series(
        forecast_extremes_df,
        "extreme_rank",
        context="Cannot rerender forecast-extreme figures",
        sample_cols=("dataset", "run_id", "extreme_kind", "extreme_rank"),
        min_value=1,
    )
    rank_one_df = forecast_extremes_df.loc[extreme_rank == 1].copy()
    if rank_one_df.empty:
        return []
    runtime_device = _resolve_requested_runtime_device(
        args,
        context_name="forecast-extreme rerendering",
    )
    run_context_df = result_df[
        ["dataset", "run_id", "eval_data_seed", "n_test_samples"]
    ].drop_duplicates()
    _assert_no_duplicates(
        run_context_df,
        ["dataset", "run_id"],
        context="Forecast-extreme rerender run context is duplicated",
    )
    rank_one_df = rank_one_df.merge(
        run_context_df,
        on=["dataset", "run_id"],
        how="left",
        indicator=True,
    )
    missing_context = rank_one_df["_merge"] != "both"
    if missing_context.any():
        examples = (
            rank_one_df.loc[
                missing_context,
                ["dataset", "run_id", "robustness_method", "pipeline_id"],
            ]
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Cannot rerender forecast-extreme figures because selected runs are missing "
            f"eval context rows. Examples: {examples}."
        )
    rank_one_df = rank_one_df.drop(columns="_merge")

    defaults = load_defaults()
    dataset_window_defaults = load_dataset_windows(defaults=defaults)
    dataset_specs_by_key = {str(spec.key): spec for spec in resolved_specs}
    datamodule_cache: dict[str, tuple[Any, Any, int, int]] = {}
    plot_entries: list[dict[str, Any]] = []

    for run_id, run_df in rank_one_df.groupby("run_id", dropna=False, sort=True):
        dataset_name = str(run_df["dataset"].iloc[0])
        run_obj = winner_runs_by_id.get(str(run_id))
        if run_obj is None:
            raise ValueError(
                f"Cannot rerender forecast-extreme figures because run_id '{run_id}' "
                "is missing from winner_runs_by_id."
            )
        seeds = require_seed_tags(run_obj)
        dataset_spec = dataset_specs_by_key.get(dataset_name)
        if dataset_spec is None:
            raise ValueError(
                f"Cannot rerender forecast-extreme figures because dataset "
                f"'{dataset_name}' is missing from resolved dataset specs."
            )
        eval_data_seed = coerce_int(run_df["eval_data_seed"].iloc[0])
        if eval_data_seed is None:
            raise ValueError(
                "Cannot rerender forecast-extreme figures because run "
                f"{run_id!r} is missing a valid eval_data_seed."
            )
        n_test_samples = coerce_int(run_df["n_test_samples"].iloc[0])
        if n_test_samples is None:
            raise ValueError(
                "Cannot rerender forecast-extreme figures because run "
                f"{run_id!r} is missing a valid n_test_samples value."
            )
        cached_entry = datamodule_cache.get(dataset_name)
        if cached_entry is None:
            dataset_args = resolve_dataset_window_args(
                args,
                dataset_spec=dataset_spec,
                dataset_window_defaults=dataset_window_defaults,
                explicit_arg_overrides=args._window_arg_overrides,
            )
            dataset_args.n_test_samples = int(n_test_samples)
            dm = _build_testing_datamodule(
                dataset_spec=dataset_spec,
                args=dataset_args,
                canonical_data_seed=seeds["seed_data"],
                eval_data_seed=eval_data_seed,
                val_seed=None,
            )
            datamodule_cache[dataset_name] = (
                dataset_args,
                dm,
                eval_data_seed,
                seeds["seed_data"],
            )
        else:
            dataset_args, dm, cached_eval_data_seed, cached_seed_data = cached_entry
            if int(cached_eval_data_seed) != int(eval_data_seed):
                raise ValueError(
                    "Cannot rerender forecast-extreme figures because dataset "
                    f"'{dataset_name}' uses inconsistent eval_data_seed values "
                    f"{cached_eval_data_seed} vs {eval_data_seed}."
                )
            if int(cached_seed_data) != int(seeds["seed_data"]):
                raise ValueError(
                    "Cannot rerender forecast-extreme figures because dataset "
                    f"'{dataset_name}' uses inconsistent seed_data values "
                    f"{cached_seed_data} vs {seeds['seed_data']}."
                )
            if int(dataset_args.n_test_samples) != int(n_test_samples):
                raise ValueError(
                    "Cannot rerender forecast-extreme figures because dataset "
                    f"'{dataset_name}' uses inconsistent n_test_samples values "
                    f"{dataset_args.n_test_samples} vs {n_test_samples}."
                )

        model = None
        try:
            model, _default_root_dir = load_model_with_loader(client, run_obj, dataset_args, dm)
            _prime_model_for_degradation_evaluation(
                model,
                dataset_args,
                dm,
                eval_seed=seeds["seed_eval"],
            )
            rendered_samples = _collect_degradation_forecast_samples(
                model=model,
                dm=dm,
                sample_rows=run_df,
                test_metric=args.test_metric,
                eval_data_seed=eval_data_seed,
                runtime_device=runtime_device,
                runtime_precision=require_namespace_value(args, key="precision"),
            )
        except Exception as exc:
            if _is_optional_forecast_extreme_cuda_runtime_error(exc):
                print(
                    "Warning: Skipping optional forecast-extreme rerender for "
                    f"run {run_id} on dataset {dataset_name} due to CUDA runtime error: {exc}"
                )
                continue
            raise
        finally:
            if model is not None:
                _teardown_model_after_eval(model)
                del model

        sample_payload_by_key = {
            (int(payload["sample_id"]), int(payload["pert_idx"])): payload
            for payload in rendered_samples
        }
        input_time_index = np.arange(1, int(dataset_args.input_len) + 1, dtype=float)
        output_time_index = np.arange(
            int(dataset_args.input_len) + 1,
            int(dataset_args.input_len) + int(dataset_args.target_len) + 1,
            dtype=float,
        )
        for row in run_df.sort_values(["extreme_kind"], kind="mergesort").itertuples(index=False):
            payload = sample_payload_by_key.get((int(row.sample_id), int(row.pert_idx)))
            if payload is None:
                raise ValueError(
                    "Forecast-extreme rerendering is missing selected sample payload for "
                    f"run_id={run_id}, sample_id={int(row.sample_id)}, "
                    f"pert_idx={int(row.pert_idx)}."
                )
            plot_entries.append(
                {
                    "dataset": dataset_name,
                    "robustness_method": str(row.robustness_method),
                    "pipeline_id": str(row.pipeline_id),
                    "run_id": str(run_id),
                    "backbone_architecture": str(row.backbone_architecture),
                    "extreme_kind": str(row.extreme_kind),
                    "figure": plot_forecast_extreme(
                        time_index=output_time_index,
                        target=payload["target"],
                        prediction_perturbed=payload["prediction_perturbed"],
                        prediction_clean=payload["prediction_clean"],
                        clean_input=payload["clean_input"],
                        perturbed_input=payload["perturbed_input"],
                        input_time_index=input_time_index,
                        input_feature_names=payload["input_feature_names"],
                        target_feature_names=payload["target_feature_names"],
                        affected_feature_names=payload["affected_feature_names"],
                        title=(
                            f"{dataset_name} · {str(row.backbone_architecture)} · "
                            f"{str(row.robustness_method)} · "
                            f"{str(row.extreme_kind).title()} Forecast Extreme"
                        ),
                        scenario=payload["scenario"],
                        severity=float(payload["severity"]),
                        sample_score=float(row.sample_score),
                        score_metric=str(row.score_metric),
                    ),
                }
            )
    return plot_entries


def _build_optional_forecast_extreme_plot_entries(
    *,
    args: Any,
    forecast_extremes_df: pd.DataFrame,
    result_df: pd.DataFrame,
    winner_runs_by_id: Mapping[str, Any],
    client: mlflow.MlflowClient,
    resolved_specs: Sequence[Any],
) -> list[dict[str, Any]]:
    if not require_namespace_bool(args, key="forecast_extremes"):
        print(
            "Skipping forecast-extreme figure rerender because "
            "forecast_extremes=false."
        )
        return []
    return _build_forecast_extreme_plot_entries(
        forecast_extremes_df,
        result_df=result_df,
        winner_runs_by_id=winner_runs_by_id,
        client=client,
        args=args,
        resolved_specs=resolved_specs,
    )


def _skip_reason_for_best_model_query_run(
    run: Any,
    *,
    args: Any,
    client: Any | None = None,
) -> str | None:
    run_params = run.data.params
    if run_params is None:
        raise ValueError(
            f"Run {run.info.run_id} from best_model=true query is missing params."
        )
    is_tested = require_tested_param(
        run_params,
        run_id=run.info.run_id,
    )
    if not is_tested:
        return "tested=false"
    if not is_fully_tested(run, args=args, client=client):
        return "stale_for_current_eval_context"
    return None


def _require_winner_selection_provenance_for_meta_analysis_run(
    run: Any,
    *,
    args: Any,
    test_metric: str,
) -> dict[str, Any]:
    """Require winner-selection provenance to match the current meta-analysis mode."""
    tags = run.data.tags
    if tags is None:
        raise ValueError(
            f"Run {run.info.run_id} is missing tags required for winner-selection provenance."
        )
    expected_context = _build_winner_selection_provenance_tag_payload_for_run(
        run,
        args=args,
        test_metric=test_metric,
    )
    try:
        return require_winner_selection_provenance_tags(
            tags,
            run_id=run.info.run_id,
            expected_context=expected_context,
        )
    except ValueError as exc:
        raise ValueError(
            f"Run {run.info.run_id} has invalid winner-selection provenance: {exc}"
        ) from exc


def _core_delta_column(metric_key: str) -> str:
    return f"delta_{metric_key}"


METHOD_ANALYSIS_IDENTITY_COLS: tuple[str, ...] = (
    "dataset",
    "data_config_signature",
    "robustness_method",
    "pipeline_id",
    "model_architecture",
)
RHO_EFF_GROUP_COLS: tuple[str, ...] = (
    "dataset",
    "data_config_signature",
    "eval_data_seed",
    "test_metric",
    "perturbation_scenarios_signature",
    "perturbation_channel_fraction_max",
)
RHO_EFF_FIT_SUMMARY_COLS: tuple[str, ...] = (
    *RHO_EFF_GROUP_COLS,
    "rho_eff_fit_status",
    "rho_eff_fit_slope",
    "rho_eff_fit_intercept",
    "rho_eff_fit_r2",
    "rho_eff_fit_rmse",
    "rho_eff_fit_n_rows_in_group",
    "rho_eff_fit_n_baselines_used",
    "rho_eff_fit_n_rows_scored",
    "rho_eff_fit_n_rows_non_positive_prediction",
    "rho_eff_fit_baseline_run_ids",
)


def _extract_run_robustness_metrics(
    run,
    *,
    expected_test_metric: str,
    expected_eval_context: Mapping[str, Any] | None = None,
    return_context: bool = False,
):
    tags = run.data.tags
    if tags is None:
        raise ValueError(
            f"Run {run.info.run_id} is missing tags required for robustness metric extraction."
        )
    eval_context = require_degradation_eval_context_tags(
        tags,
        run_id=run.info.run_id,
        expected_test_metric=expected_test_metric,
        expected_context=(
            dict(expected_eval_context)
            if expected_eval_context is not None
            else None
        ),
    )
    bootstrap_ci_context = require_shared_anchor_bootstrap_ci_context_tags(
        tags,
        run_id=run.info.run_id,
        require_seed=True,
    )
    run_test_metric = eval_context["test_metric"]
    base_key = build_degradation_metric_prefix(test_metric=run_test_metric)
    metrics = extract_required_overall_degradation_metrics(
        run.data.metrics,
        run_id=run.info.run_id,
        test_metric=run_test_metric,
    )
    at_worst_scenario_metric_name = build_metric_w_name(
        run_test_metric,
        key="test_metric",
        context=f"Run {run.info.run_id}",
    )
    metrics["at_worst_scenario"] = require_float_metric(
        run.data.metrics,
        run_id=run.info.run_id,
        metric_key=f"{base_key}/err_pert_ws",
        output_name=at_worst_scenario_metric_name,
    )
    if return_context:
        return metrics, {
            "eval_context": eval_context,
            "bootstrap_ci_context": bootstrap_ci_context,
            "test_metric": run_test_metric,
            "base_key": base_key,
            "at_worst_scenario_metric_name": at_worst_scenario_metric_name,
            "robustness_scoring_semantics": (
                DEGRADATION_SCORING_SEMANTICS
            ),
        }
    return metrics


def _require_logged_worst_scenario_param(
    run,
    *,
    base_key: str,
) -> str:
    params = run.data.params
    if params is None:
        raise ValueError(
            f"Run {run.info.run_id} is missing params required for worst_scenario extraction."
        )
    param_key = f"{base_key}/worst_scenario"
    tags = run.data.tags
    return require_tag_value_with_optional_param_match(
        tags,
        params,
        key=param_key,
        context=f"Run {run.info.run_id}",
        disallow_none_token=True,
    )


def _require_tested_parent_analysis_run(
    run: Any,
    *,
    context: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    run_tags = run.data.tags
    if run_tags is None:
        raise ValueError(
            f"Run {run.info.run_id} is missing tags in {context} analysis."
        )
    if run_tags.get("mlflow.parentRunId"):
        return None
    if run.info.status != "FINISHED":
        raise ValueError(
            f"Run {run.info.run_id} in {context} analysis is not FINISHED."
        )
    run_params = run.data.params
    if run_params is None:
        raise ValueError(
            f"Run {run.info.run_id} is missing params in {context} analysis."
        )
    if not require_tested_param(
        run_params,
        run_id=run.info.run_id,
    ):
        raise ValueError(
            f"Run {run.info.run_id} in {context} query has tested='false'."
        )
    return run_tags, run_params


def _extract_dataset_name_from_experiment(experiment_name: str, *, prefix: str) -> str:
    expected_prefix = f"{prefix}-"
    if not experiment_name.startswith(expected_prefix):
        raise ValueError(
            f"Experiment name '{experiment_name}' does not match expected prefix "
            f"'{expected_prefix}'."
        )
    dataset_name = experiment_name[len(expected_prefix):].strip()
    if not dataset_name:
        raise ValueError(
            f"Experiment name '{experiment_name}' is missing dataset suffix after prefix "
            f"'{expected_prefix}'."
        )
    return dataset_name


def _aggregate_worst_scenario_labels(
    source_df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    context: str,
) -> pd.Series:
    _require_columns(
        source_df,
        set(group_cols) | {"worst_scenario"},
        context=context,
    )

    def _summarize(group: pd.Series) -> str:
        parsed_values = sorted(
            {
                parse_required_nonempty_string(
                    raw_value,
                    key="worst_scenario",
                    context=context,
                    disallow_none_token=True,
                )
                for raw_value in group.tolist()
            }
        )
        if not parsed_values:
            raise ValueError(f"{context}: grouped worst_scenario values are empty.")
        return ",".join(parsed_values)

    return (
        source_df.groupby(list(group_cols), dropna=False)["worst_scenario"]
        .agg(_summarize)
        .rename("worst_scenario_labels")
    )


def _aggregate_numeric_summary_table(
    source_df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    numeric_cols: Sequence[str],
    context: str,
    at_worst_scenario_metric_key: str | None = None,
    reset_index: bool,
) -> pd.DataFrame:
    if source_df.empty:
        return pd.DataFrame()
    if not numeric_cols:
        return pd.DataFrame()

    group_cols_list = list(group_cols)
    _require_columns(
        source_df,
        set(group_cols_list) | set(numeric_cols),
        context=context,
    )

    grouped = source_df.groupby(group_cols_list, dropna=False)
    summary_df = grouped[list(numeric_cols)].agg(["mean", "std"])
    summary_df.columns = ["_".join(col).strip() for col in summary_df.columns]
    summary_df.insert(0, "count", grouped.size())

    if at_worst_scenario_metric_key is not None:
        if at_worst_scenario_metric_key not in numeric_cols:
            raise ValueError(
                f"{context}: at-worst-scenario metric "
                f"'{at_worst_scenario_metric_key}' is missing from numeric columns."
            )
        summary_df = summary_df.join(
            _aggregate_worst_scenario_labels(
                source_df,
                group_cols=group_cols_list,
                context=context,
            )
        )

    if reset_index:
        return summary_df.reset_index()
    return summary_df


def compute_improvement_deltas(
    backbone_df,
    improvement_df,
    metrics,
    *,
    pair_reference_df: pd.DataFrame | None = None,
):
    """Compute robustness improvement deltas by joining baselines with non-baseline variants.

    Computes deltas for each (dataset, architecture, pipeline_id) relative to its
    corresponding (dataset, architecture, baseline) run.

    Baseline-based improvements, including top-k baseline ensembles, remain paired
    to the current top-1 baseline winner for the same dataset, architecture, and
    data_config_signature. Non-baseline stacked methods must resolve an explicit
    single parent run from the current winner pool.

    Args:
        backbone_df: DataFrame with baseline (pipeline_id='baseline') results
        improvement_df: DataFrame with non-baseline variant results
        metrics: List of metric column names to compute deltas for
        pair_reference_df: Optional current winner-pool rows used to resolve the
            intervention base model for ``tau_mean``. When omitted, only baseline-
            paired rows may compute ``tau_mean``.

    Returns:
        DataFrame with delta columns added (delta_{metric} and pct_change_{metric})
    """
    if backbone_df.empty or improvement_df.empty:
        return pd.DataFrame()

    join_keys = ["dataset", "model_architecture"]
    has_signature_backbone = "data_config_signature" in backbone_df.columns
    has_signature_improvement = "data_config_signature" in improvement_df.columns
    if not has_signature_backbone and not has_signature_improvement:
        raise ValueError(
            "Cannot compute improvement deltas: data_config_signature is missing from both "
            "baseline and improvement dataframes. Signature-guarded joins are required."
        )
    if has_signature_backbone != has_signature_improvement:
        raise ValueError(
            "Cannot compute improvement deltas because data_config_signature is present in only "
            "one side of the merge."
        )
    if backbone_df["data_config_signature"].isna().any():
        raise ValueError(
            "Cannot compute improvement deltas: baseline rows contain missing "
            "data_config_signature values."
        )
    if improvement_df["data_config_signature"].isna().any():
        raise ValueError(
            "Cannot compute improvement deltas: improvement rows contain missing "
            "data_config_signature values."
        )
    join_keys.append("data_config_signature")

    missing_backbone_cols = [col for col in join_keys if col not in backbone_df.columns]
    if missing_backbone_cols:
        raise ValueError(
            "Cannot compute improvement deltas: baseline dataframe is missing join columns "
            f"{missing_backbone_cols}."
        )
    missing_improvement_cols = [col for col in join_keys if col not in improvement_df.columns]
    if missing_improvement_cols:
        raise ValueError(
            "Cannot compute improvement deltas: improvement dataframe is missing join columns "
            f"{missing_improvement_cols}."
        )

    baseline_dup_mask = backbone_df.duplicated(join_keys, keep=False)
    if baseline_dup_mask.any():
        examples = (
            backbone_df.loc[baseline_dup_mask, join_keys]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Cannot compute improvement deltas: baseline rows are not unique for join keys "
            f"{join_keys}. Examples: {examples}."
        )

    improvement_identity_keys = join_keys + ["robustness_method", "pipeline_id"]
    missing_identity_cols = [
        col for col in improvement_identity_keys if col not in improvement_df.columns
    ]
    if missing_identity_cols:
        raise ValueError(
            "Cannot compute improvement deltas: improvement dataframe is missing identity columns "
            f"{missing_identity_cols}."
        )
    improvement_dup_mask = improvement_df.duplicated(improvement_identity_keys, keep=False)
    if improvement_dup_mask.any():
        examples = (
            improvement_df.loc[improvement_dup_mask, improvement_identity_keys]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Cannot compute improvement deltas: improvement rows are not unique per variant key "
            f"{improvement_identity_keys}. Examples: {examples}."
        )

    missing_metric_cols_backbone = [metric for metric in metrics if metric not in backbone_df.columns]
    if missing_metric_cols_backbone:
        raise ValueError(
            "Cannot compute improvement deltas: baseline dataframe is missing requested "
            f"metric columns {missing_metric_cols_backbone}."
        )
    missing_metric_cols_improvement = [
        metric for metric in metrics if metric not in improvement_df.columns
    ]
    if missing_metric_cols_improvement:
        raise ValueError(
            "Cannot compute improvement deltas: improvement dataframe is missing requested "
            f"metric columns {missing_metric_cols_improvement}."
        )

    baseline_cols = join_keys + [m for m in metrics if m in backbone_df.columns]
    baseline_join_df = backbone_df[baseline_cols].copy()
    if "run_id" in backbone_df.columns:
        baseline_join_df["matched_baseline_run_id"] = backbone_df["run_id"]
    if "architecture_family" in backbone_df.columns:
        baseline_join_df["architecture_family"] = backbone_df["architecture_family"]
    has_worst_scenario_backbone = "worst_scenario" in backbone_df.columns
    has_worst_scenario_improvement = "worst_scenario" in improvement_df.columns
    if has_worst_scenario_backbone != has_worst_scenario_improvement:
        raise ValueError(
            "Cannot compute improvement deltas because worst_scenario is present in only "
            "one side of the merge."
        )
    requires_worst_scenario = any(
        str(metric).strip().endswith("_w") and str(metric).strip() != "D_w"
        for metric in metrics
    )
    if requires_worst_scenario and not has_worst_scenario_backbone:
        raise ValueError(
            "Cannot compute improvement deltas: companion `*_w` metrics require "
            "worst_scenario on both baseline and improvement rows."
        )
    if has_worst_scenario_backbone:
        for frame_name, frame in (
            ("baseline", backbone_df),
            ("improvement", improvement_df),
        ):
            raw_values = frame["worst_scenario"]
            stripped = raw_values.astype(str).str.strip()
            invalid_mask = raw_values.isna() | stripped.eq("") | stripped.str.lower().eq("none")
            if invalid_mask.any():
                raise ValueError(
                    "Cannot compute improvement deltas: "
                    f"{frame_name} rows contain missing worst_scenario values."
                )
        baseline_join_df["worst_scenario"] = backbone_df["worst_scenario"]

    # Join on dataset and architecture to compute deltas vs baseline
    merged = improvement_df.merge(
        baseline_join_df,
        on=join_keys,
        suffixes=("_improved", "_baseline"),
        how="inner"
    )

    if merged.empty:
        return pd.DataFrame()

    if {
        "architecture_family_improved",
        "architecture_family_baseline",
    }.issubset(merged.columns):
        arch_context = "Cannot compute improvement deltas"
        arch_compare_df = _require_nonempty_string_columns(
            merged[
                [
                    "dataset",
                    "robustness_method",
                    "pipeline_id",
                    "model_architecture",
                    "architecture_family_improved",
                    "architecture_family_baseline",
                ]
            ],
            ["architecture_family_improved", "architecture_family_baseline"],
            context=arch_context,
            sample_cols=[
                "dataset",
                "robustness_method",
                "pipeline_id",
                "model_architecture",
            ],
        )
        mismatch_mask = (
            arch_compare_df["architecture_family_improved"]
            != arch_compare_df["architecture_family_baseline"]
        )
        if mismatch_mask.any():
            example_cols = [
                "dataset",
                "robustness_method",
                "pipeline_id",
                "model_architecture",
                "architecture_family_improved",
                "architecture_family_baseline",
            ]
            examples = _sample_records(
                arch_compare_df.loc[mismatch_mask, example_cols],
                example_cols,
            )
            raise ValueError(
                "Cannot compute improvement deltas: baseline/improvement architecture_family "
                f"values disagree after the delta join. Examples: {examples}."
            )
        merged["architecture_family"] = arch_compare_df["architecture_family_improved"]
        merged = merged.drop(
            columns=["architecture_family_improved", "architecture_family_baseline"]
        )

    # Compute deltas for each metric
    for metric in metrics:
        baseline_col = f"{metric}_baseline"
        improved_col = f"{metric}_improved"

        if baseline_col in merged.columns and improved_col in merged.columns:
            merged[f"delta_{metric}"] = merged[improved_col] - merged[baseline_col]

            baseline_safe = pd.to_numeric(
                merged[baseline_col],
                errors="raise",
            ).astype(float)
            baseline_safe = baseline_safe.mask(baseline_safe.eq(0.0), np.nan)
            delta_numeric = pd.to_numeric(
                merged[f"delta_{metric}"],
                errors="raise",
            ).astype(float)
            pct_change = 100.0 * delta_numeric / baseline_safe
            pct_change = pct_change.mask(~np.isfinite(pct_change), np.nan)
            merged[f"pct_change_{metric}"] = pct_change

    clean_error_metric = next(
        (
            metric
            for metric in metrics
            if str(metric).endswith("_test")
        ),
        None,
    )
    if clean_error_metric is not None and f"delta_{clean_error_metric}" in merged.columns:
        merged["delta_err_clean"] = merged[f"delta_{clean_error_metric}"]
    if (
        "err_pert_mean_baseline" in merged.columns
        and "err_pert_mean_improved" in merged.columns
    ):
        merged["tau_mean"] = np.nan
        merged["tau_mean_status"] = "unsupported"
        merged["tau_mean_base_run_id"] = pd.Series(index=merged.index, dtype=object)
        if pair_reference_df is None:
            if "base_pipeline_method" in improvement_df.columns:
                for row in improvement_df.itertuples(index=False):
                    run_id = parse_optional_nonempty_string(
                        getattr(row, "run_id", None),
                        key="run_id",
                        context="Cannot compute tau_mean",
                        disallow_none_token=True,
                    ) or "<unknown>"
                    base_method = parse_optional_nonempty_string(
                        getattr(row, "base_pipeline_method", None),
                        key="base_pipeline_method",
                        context=f"Cannot compute tau_mean for run {run_id}",
                        disallow_none_token=True,
                    )
                    if base_method not in (None, "baseline"):
                        raise ValueError(
                            "Cannot compute tau_mean without pair_reference_df for "
                            f"run {run_id}: base_pipeline_method='{base_method}' "
                            "requires explicit base-run resolution."
                        )
            merged["tau_mean"] = (
                merged["err_pert_mean_baseline"] - merged["err_pert_mean_improved"]
            )
            merged["tau_mean_status"] = "computed:matched_baseline"
            if "matched_baseline_run_id" in merged.columns:
                merged["tau_mean_base_run_id"] = merged["matched_baseline_run_id"]
        else:
            context = "Cannot compute tau_mean"
            _require_columns(
                improvement_df,
                {"run_id", "base_pipeline_method"},
                context=context,
            )
            _require_columns(
                pair_reference_df,
                {
                    "run_id",
                    "dataset",
                    "data_config_signature",
                    "model_architecture",
                    "pipeline_method",
                    "err_pert_mean",
                },
                context=context,
            )
            pair_reference_working_df = _require_nonempty_string_columns(
                pair_reference_df,
                [
                    "run_id",
                    "dataset",
                    "data_config_signature",
                    "model_architecture",
                    "pipeline_method",
                ],
                context=context,
                sample_cols=["run_id", "dataset", "pipeline_method"],
            ).copy()
            pair_reference_working_df["err_pert_mean"] = require_numeric_series(
                pair_reference_working_df["err_pert_mean"],
                column_name="err_pert_mean",
                context=context,
                allow_nan=False,
                allow_infinite=False,
            )
            _assert_no_duplicates(
                pair_reference_working_df,
                ["run_id"],
                context=f"{context}: pair-reference rows are not unique per run_id",
            )
            pair_reference_by_run_id = pair_reference_working_df.set_index("run_id")

            tau_values: list[float] = []
            tau_statuses: list[str] = []
            tau_base_run_ids: list[str | None] = []
            for row in merged.itertuples(index=False):
                improved_run_id = parse_required_nonempty_string(
                    getattr(row, "run_id"),
                    key="run_id",
                    context=context,
                    disallow_none_token=True,
                )
                base_method = parse_required_nonempty_string(
                    getattr(row, "base_pipeline_method"),
                    key="base_pipeline_method",
                    context=f"{context} for run {improved_run_id}",
                    disallow_none_token=True,
                )
                backbone_run_id = parse_optional_nonempty_string(
                    getattr(row, "backbone_run_id", None),
                    key="backbone_run_id",
                    context=f"{context} for run {improved_run_id}",
                    disallow_none_token=True,
                )
                backbone_run_ids_raw = parse_optional_nonempty_string(
                    getattr(row, "backbone_run_ids", None),
                    key="backbone_run_ids",
                    context=f"{context} for run {improved_run_id}",
                    disallow_none_token=True,
                )
                backbone_run_ids = parse_backbone_run_ids(
                    backbone_run_ids_raw,
                    run_id=improved_run_id,
                )
                if backbone_run_id is not None and len(backbone_run_ids) == 1:
                    if backbone_run_id != backbone_run_ids[0]:
                        raise ValueError(
                            f"{context}: run {improved_run_id} has inconsistent "
                            "backbone_run_id and backbone_run_ids tags."
                        )
                if backbone_run_id is not None and len(backbone_run_ids) > 1:
                    raise ValueError(
                        f"{context}: run {improved_run_id} cannot define both "
                        "backbone_run_id and multi-run backbone_run_ids."
                    )
                if backbone_run_id is None and len(backbone_run_ids) == 1:
                    backbone_run_id = backbone_run_ids[0]

                matched_baseline_run_id = parse_optional_nonempty_string(
                    getattr(row, "matched_baseline_run_id", None),
                    key="matched_baseline_run_id",
                    context=f"{context} for run {improved_run_id}",
                    disallow_none_token=True,
                )
                if len(backbone_run_ids) > 1:
                    if base_method != "baseline":
                        tau_values.append(np.nan)
                        tau_statuses.append("unsupported:multi_backbone_nonbaseline")
                        tau_base_run_ids.append(None)
                        continue
                    if matched_baseline_run_id is None:
                        raise ValueError(
                            f"{context}: run {improved_run_id} is missing the matched "
                            "baseline run id required for baseline-paired tau_mean."
                        )
                    tau_base_run_id = matched_baseline_run_id
                    tau_status = "computed:matched_baseline"
                elif backbone_run_id is None:
                    if base_method != "baseline":
                        raise ValueError(
                            f"{context}: run {improved_run_id} has base_pipeline_method="
                            f"'{base_method}' but no backbone_run_id."
                        )
                    if matched_baseline_run_id is None:
                        raise ValueError(
                            f"{context}: run {improved_run_id} is missing the matched "
                            "baseline run id required for baseline-paired tau_mean."
                        )
                    tau_base_run_id = matched_baseline_run_id
                    tau_status = "computed:matched_baseline"
                else:
                    tau_base_run_id = backbone_run_id
                    tau_status = "computed:lineage"
                    if (
                        base_method == "baseline"
                        and matched_baseline_run_id is not None
                        and tau_base_run_id != matched_baseline_run_id
                    ):
                        raise ValueError(
                            f"{context}: run {improved_run_id} references baseline run "
                            f"{tau_base_run_id} but the canonical matched baseline is "
                            f"{matched_baseline_run_id}."
                        )

                if tau_base_run_id not in pair_reference_by_run_id.index:
                    raise ValueError(
                        f"{context}: run {improved_run_id} references base run "
                        f"{tau_base_run_id}, which is absent from the current winner pool."
                    )
                pair_row = pair_reference_by_run_id.loc[tau_base_run_id]
                pair_method = parse_required_nonempty_string(
                    pair_row["pipeline_method"],
                    key="pipeline_method",
                    context=f"{context} for base run {tau_base_run_id}",
                    disallow_none_token=True,
                )
                if pair_method != base_method:
                    raise ValueError(
                        f"{context}: run {improved_run_id} expects base_pipeline_method="
                        f"'{base_method}' but paired run {tau_base_run_id} has pipeline_method="
                        f"'{pair_method}'."
                    )
                pair_dataset = parse_required_nonempty_string(
                    pair_row["dataset"],
                    key="dataset",
                    context=f"{context} for base run {tau_base_run_id}",
                    disallow_none_token=True,
                )
                pair_signature = parse_required_nonempty_string(
                    pair_row["data_config_signature"],
                    key="data_config_signature",
                    context=f"{context} for base run {tau_base_run_id}",
                    disallow_none_token=True,
                )
                pair_architecture = parse_required_nonempty_string(
                    pair_row["model_architecture"],
                    key="model_architecture",
                    context=f"{context} for base run {tau_base_run_id}",
                    disallow_none_token=True,
                )
                if pair_dataset != getattr(row, "dataset"):
                    raise ValueError(
                        f"{context}: run {improved_run_id} pairs to base run {tau_base_run_id} "
                        "from a different dataset."
                    )
                if pair_signature != getattr(row, "data_config_signature"):
                    raise ValueError(
                        f"{context}: run {improved_run_id} pairs to base run {tau_base_run_id} "
                        "with a different data_config_signature."
                    )
                if pair_architecture != getattr(row, "model_architecture"):
                    raise ValueError(
                        f"{context}: run {improved_run_id} pairs to base run {tau_base_run_id} "
                        "with a different model_architecture."
                    )
                tau_values.append(
                    float(pair_row["err_pert_mean"]) - float(getattr(row, "err_pert_mean_improved"))
                )
                tau_statuses.append(tau_status)
                tau_base_run_ids.append(tau_base_run_id)

            merged["tau_mean"] = tau_values
            merged["tau_mean_status"] = tau_statuses
            merged["tau_mean_base_run_id"] = tau_base_run_ids
    return merged


def _attach_rho_eff(
    result_df: pd.DataFrame,
    *,
    test_metric: str,
) -> RhoEffAttachmentResult:
    """Attach a forecasting adaptation of Taori-style effective robustness.

    The maintained comparator fits a per-group log-log frontier on eligible
    baseline rows with positive clean and mean corrupted error, then scores the
    residual on the original error scale.

    Reference: Taori et al., 2020
    Paper: https://proceedings.neurips.cc/paper/2020/hash/d8330f857a17c53d217014ee776bfd50-Abstract.html
    Repo: https://github.com/modestyachts/imagenet-testbed
    """
    context = "Cannot attach rho_eff"
    clean_error_col = f"{test_metric}_test"
    required_cols = {
        *RHO_EFF_GROUP_COLS,
        "run_id",
        "pipeline_id",
        "pipeline_method",
        "pipeline_kind",
        "robustness_method",
        clean_error_col,
        "err_pert_mean",
    }
    _require_columns(result_df, required_cols, context=context)
    if result_df.empty:
        return RhoEffAttachmentResult(
            result_df=result_df.assign(
                mPC=pd.Series(dtype=float),
                rPC=pd.Series(dtype=float),
                rho_eff=pd.Series(dtype=float),
                rho_eff_status=pd.Series(dtype=object),
            ),
            fit_summary_df=pd.DataFrame(columns=RHO_EFF_FIT_SUMMARY_COLS),
        )

    working_df = _require_nonempty_string_columns(
        result_df,
        [
            *RHO_EFF_GROUP_COLS,
            "run_id",
            "pipeline_id",
            "pipeline_method",
            "pipeline_kind",
            "robustness_method",
        ],
        context=context,
        sample_cols=["run_id", "dataset", "pipeline_id"],
    ).copy()
    working_df[clean_error_col] = require_numeric_series(
        working_df[clean_error_col],
        column_name=clean_error_col,
        context=context,
        allow_nan=False,
        allow_infinite=False,
    )
    working_df["err_pert_mean"] = require_numeric_series(
        working_df["err_pert_mean"],
        column_name="err_pert_mean",
        context=context,
        allow_nan=False,
        allow_infinite=False,
    )
    working_df["mPC"] = working_df["err_pert_mean"].astype(float)
    positive_mpc = working_df["mPC"] > 0.0
    working_df["rPC"] = np.nan
    working_df.loc[positive_mpc, "rPC"] = (
        working_df.loc[positive_mpc, clean_error_col].astype(float)
        / working_df.loc[positive_mpc, "mPC"]
    )
    baseline_mask = (
        (working_df["pipeline_id"] == "baseline")
        & (working_df["pipeline_method"] == "baseline")
        & (working_df["robustness_method"] == "baseline")
        & (working_df["pipeline_kind"] == "train")
    )
    working_df["rho_eff"] = np.nan
    working_df["rho_eff_status"] = "unsupported"
    fit_summary_records: list[dict[str, Any]] = []

    for group_key, group_idx in working_df.groupby(list(RHO_EFF_GROUP_COLS), dropna=False).groups.items():
        index = working_df.index.intersection(group_idx)
        group_values = group_key if isinstance(group_key, tuple) else (group_key,)
        fit_summary: dict[str, Any] = {
            col: value for col, value in zip(RHO_EFF_GROUP_COLS, group_values)
        }
        baseline_index = baseline_mask.loc[index]
        baseline_index = baseline_index[baseline_index].index
        fit_df = working_df.loc[baseline_index]
        fit_df = fit_df.loc[
            (fit_df[clean_error_col].astype(float) > 0.0)
            & (fit_df["mPC"].astype(float) > 0.0)
        ]
        fit_summary.update(
            rho_eff_fit_status="unsupported",
            rho_eff_fit_slope=np.nan,
            rho_eff_fit_intercept=np.nan,
            rho_eff_fit_r2=np.nan,
            rho_eff_fit_rmse=np.nan,
            rho_eff_fit_n_rows_in_group=int(len(index)),
            rho_eff_fit_n_baselines_used=int(fit_df.shape[0]),
            rho_eff_fit_n_rows_scored=0,
            rho_eff_fit_n_rows_non_positive_prediction=0,
            rho_eff_fit_baseline_run_ids=",".join(
                fit_df["run_id"].astype(str).tolist()
            ),
        )
        if fit_df.shape[0] < 2:
            working_df.loc[
                index,
                "rho_eff_status",
            ] = "unsupported:insufficient_baselines"
            fit_summary["rho_eff_fit_status"] = "unsupported:insufficient_baselines"
            fit_summary_records.append(fit_summary)
            continue
        fit_clean = fit_df[clean_error_col].astype(float).to_numpy(dtype=float)
        fit_mpc = fit_df["mPC"].astype(float).to_numpy(dtype=float)
        if np.allclose(fit_clean, fit_clean[0], rtol=0.0, atol=1e-12):
            working_df.loc[
                index,
                "rho_eff_status",
            ] = "unsupported:zero_clean_variance"
            fit_summary["rho_eff_fit_status"] = "unsupported:zero_clean_variance"
            fit_summary_records.append(fit_summary)
            continue
        fit_log_clean = np.log(fit_clean)
        fit_log_mpc = np.log(fit_mpc)
        coeffs = np.polyfit(fit_log_clean, fit_log_mpc, deg=1)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])
        if not np.isfinite([slope, intercept]).all():
            working_df.loc[index, "rho_eff_status"] = "unsupported:nonfinite_fit"
            fit_summary["rho_eff_fit_status"] = "unsupported:nonfinite_fit"
            fit_summary_records.append(fit_summary)
            continue
        fitted_fit = intercept + slope * fit_log_clean
        if not np.isfinite(fitted_fit).all():
            working_df.loc[index, "rho_eff_status"] = "unsupported:nonfinite_fit"
            fit_summary["rho_eff_fit_status"] = "unsupported:nonfinite_fit"
            fit_summary_records.append(fit_summary)
            continue
        fit_residuals = fit_log_mpc - fitted_fit
        fit_summary["rho_eff_fit_slope"] = slope
        fit_summary["rho_eff_fit_intercept"] = intercept
        fit_summary["rho_eff_fit_rmse"] = float(
            np.sqrt(np.mean(np.square(fit_residuals)))
        )
        fit_ss_tot = float(np.sum(np.square(fit_log_mpc - float(fit_log_mpc.mean()))))
        if not np.isclose(fit_ss_tot, 0.0, atol=1e-12, rtol=0.0):
            fit_summary["rho_eff_fit_r2"] = float(
                1.0 - (np.sum(np.square(fit_residuals)) / fit_ss_tot)
            )
        target_clean = working_df.loc[index, clean_error_col].astype(float).to_numpy(dtype=float)
        target_index = pd.Index(index)
        positive_target_clean_mask = target_clean > 0.0
        if (~positive_target_clean_mask).any():
            working_df.loc[
                target_index[~positive_target_clean_mask],
                "rho_eff_status",
            ] = "unsupported:non_positive_clean_error"
        positive_target_index = target_index[positive_target_clean_mask]
        if len(positive_target_index) == 0:
            fit_summary["rho_eff_fit_status"] = "computed"
            fit_summary_records.append(fit_summary)
            continue
        fitted_log = intercept + slope * np.log(target_clean[positive_target_clean_mask])
        fitted = np.exp(fitted_log)
        if not np.isfinite(fitted).all():
            working_df.loc[
                positive_target_index,
                "rho_eff_status",
            ] = "unsupported:nonfinite_prediction"
            fit_summary["rho_eff_fit_status"] = "unsupported:nonfinite_prediction"
            fit_summary_records.append(fit_summary)
            continue
        working_df.loc[positive_target_index, "rho_eff_status"] = "computed"
        fit_summary["rho_eff_fit_status"] = "computed"
        non_positive_prediction_mask = fitted <= 0.0
        fit_summary["rho_eff_fit_n_rows_non_positive_prediction"] = int(
            non_positive_prediction_mask.sum()
        )
        if non_positive_prediction_mask.any():
            working_df.loc[
                positive_target_index[non_positive_prediction_mask],
                "rho_eff_status",
            ] = "unsupported:non_positive_prediction"
        valid_target_index = positive_target_index[~non_positive_prediction_mask]
        fit_summary["rho_eff_fit_n_rows_scored"] = int(len(valid_target_index))
        fit_summary_records.append(fit_summary)
        if len(valid_target_index) == 0:
            continue
        working_df.loc[valid_target_index, "rho_eff"] = (
            fitted[~non_positive_prediction_mask]
            - working_df.loc[valid_target_index, "mPC"].astype(float).to_numpy(dtype=float)
        )

    return RhoEffAttachmentResult(
        result_df=working_df,
        fit_summary_df=pd.DataFrame(
            fit_summary_records,
            columns=RHO_EFF_FIT_SUMMARY_COLS,
        ),
    )


def _resolve_reference_normalization_anchors(
    result_df: pd.DataFrame,
    *,
    reference_normalization_anchor_model: str,
    test_metric: str,
) -> pd.DataFrame:
    context = "Cannot resolve reference-normalization anchors"
    reference_normalization_anchor_model = parse_reference_normalization_anchor_model(
        reference_normalization_anchor_model,
        key="reference_normalization_anchor_model",
        context=context,
    )
    clean_error_col = f"{test_metric}_test"
    required_cols = {
        *REFERENCE_NORMALIZATION_GROUP_COLS,
        "run_id",
        "model_architecture",
        "pipeline_id",
        "pipeline_method",
        "pipeline_kind",
        "robustness_method",
        "selection_pool",
        clean_error_col,
    }
    _require_columns(result_df, required_cols, context=context)
    if result_df.empty:
        raise ValueError(f"{context}: result_df is empty.")

    working_df = _require_nonempty_string_columns(
        result_df,
        [
            *REFERENCE_NORMALIZATION_GROUP_COLS,
            "run_id",
            "model_architecture",
            "pipeline_id",
            "pipeline_method",
            "pipeline_kind",
            "robustness_method",
            "selection_pool",
        ],
        context=context,
        sample_cols=["run_id", "dataset", "pipeline_id"],
    )

    working_df[clean_error_col] = require_numeric_series(
        working_df[clean_error_col],
        column_name=clean_error_col,
        context=context,
        allow_nan=False,
        allow_infinite=False,
    )
    bad_selection_pool = working_df["selection_pool"] != "winner_pool"
    if bad_selection_pool.any():
        examples = _sample_records(
            working_df.loc[
                bad_selection_pool,
                ["run_id", "dataset", "pipeline_id", "selection_pool"],
            ],
            ["run_id", "dataset", "pipeline_id", "selection_pool"],
        )
        raise ValueError(
            f"{context}: result rows are not sourced from the winner_pool. Examples: {examples}."
        )

    anchor_mask = (
        (working_df["model_architecture"] == reference_normalization_anchor_model)
        & (working_df["pipeline_id"] == "baseline")
        & (working_df["pipeline_method"] == "baseline")
        & (working_df["robustness_method"] == "baseline")
        & (working_df["pipeline_kind"] == "train")
    )
    anchor_candidates = working_df.loc[
        anchor_mask,
        [
            *REFERENCE_NORMALIZATION_GROUP_COLS,
            "run_id",
            "model_architecture",
            clean_error_col,
        ],
    ].copy()
    _assert_no_duplicates(
        anchor_candidates,
        list(REFERENCE_NORMALIZATION_GROUP_COLS),
        context=(
            f"{context}: eligible anchor rows are not unique per "
            f"{list(REFERENCE_NORMALIZATION_GROUP_COLS)}"
        ),
    )

    anchor_df = anchor_candidates.rename(
        columns={
            "run_id": "anchor_run_id",
            "model_architecture": "anchor_model",
            clean_error_col: "anchor_clean_error",
        }
    )
    anchor_df["reference_normalization_anchor_model"] = (
        reference_normalization_anchor_model
    )
    return anchor_df.sort_values(
        list(REFERENCE_NORMALIZATION_GROUP_COLS)
    ).reset_index(drop=True)


def _attach_reference_normalized_diagnostics(
    result_df: pd.DataFrame,
    scenario_summary_df: pd.DataFrame,
    severity_profile_df: pd.DataFrame,
    *,
    reference_normalization_anchor_model: str,
    test_metric: str,
    eps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context = "Cannot attach reference-normalized diagnostics"
    clean_error_col = f"{test_metric}_test"
    _require_columns(
        result_df,
        {
            *REFERENCE_NORMALIZATION_GROUP_COLS,
            "run_id",
            clean_error_col,
        },
        context=context,
    )
    _require_columns(
        scenario_summary_df,
        {"run_id", "scenario"},
        context=context,
    )
    _require_columns(
        severity_profile_df,
        {"run_id", "scenario", "severity_bin_idx", "severity", "n_samples", "err_pert_mean"},
        context=context,
    )
    if result_df.empty:
        raise ValueError(f"{context}: result_df is empty.")
    if scenario_summary_df.empty:
        raise ValueError(f"{context}: scenario_summary_df is empty.")
    if severity_profile_df.empty:
        raise ValueError(f"{context}: severity_profile_df is empty.")

    result_working_df = _require_nonempty_string_columns(
        result_df,
        ["run_id", *REFERENCE_NORMALIZATION_GROUP_COLS],
        context=context,
        sample_cols=["run_id", "dataset", "pipeline_id"],
    )
    if (result_working_df["test_metric"] != str(test_metric)).any():
        examples = _sample_records(
            result_working_df.loc[
                result_working_df["test_metric"] != str(test_metric),
                ["run_id", "dataset", "test_metric"],
            ],
            ["run_id", "dataset", "test_metric"],
        )
        raise ValueError(
            f"{context}: result rows have unexpected test_metric values. Examples: {examples}."
        )
    _assert_no_duplicates(
        result_working_df,
        ["run_id"],
        context=f"{context}: result rows are not unique per run_id",
    )
    _assert_no_duplicates(
        scenario_summary_df,
        ["run_id", "scenario"],
        context=f"{context}: scenario rows are not unique per (run_id, scenario)",
    )

    grid_working_df = _require_nonempty_string_columns(
        severity_profile_df,
        ["run_id", "scenario"],
        context=context,
        sample_cols=["run_id", "scenario"],
    )

    reference_anchors_df = _resolve_reference_normalization_anchors(
        result_working_df,
        reference_normalization_anchor_model=reference_normalization_anchor_model,
        test_metric=test_metric,
    )
    anchor_lookup = {
        tuple(getattr(row, column) for column in REFERENCE_NORMALIZATION_GROUP_COLS): row
        for row in reference_anchors_df.itertuples(index=False)
    }

    grid_by_run_id = {
        str(run_id): run_df.copy()
        for run_id, run_df in grid_working_df.groupby("run_id", sort=False)
    }
    missing_grid_runs = sorted(
        set(result_working_df["run_id"].tolist()) - set(grid_by_run_id.keys())
    )
    if missing_grid_runs:
        raise ValueError(
            "Reference-normalized diagnostics require grid rows for every result run. "
            f"Missing run_ids: {missing_grid_runs[:5]}."
        )

    anchor_summaries_by_group: dict[tuple[str, ...], Any] = {}
    partial_family_records: list[dict[str, Any]] = []
    for anchor_row in reference_anchors_df.itertuples(index=False):
        group_key = tuple(
            getattr(anchor_row, column) for column in REFERENCE_NORMALIZATION_GROUP_COLS
        )
        anchor_run_id = str(anchor_row.anchor_run_id)
        anchor_grid_df = grid_by_run_id[anchor_run_id]
        try:
            anchor_summary = summarize_reference_normalized_anchor(
                reference_grid_df=anchor_grid_df,
                reference_clean_error=float(anchor_row.anchor_clean_error),
                reference_label=anchor_run_id,
            )
        except ValueError as exc:
            raise ValueError(
                f"{context}: resolved anchor '{anchor_run_id}' is invalid for analysis "
                f"group {dict(zip(REFERENCE_NORMALIZATION_GROUP_COLS, group_key))}. {exc}"
            ) from exc
        anchor_summaries_by_group[group_key] = anchor_summary
        unsupported_families = [
            _REFERENCE_NORMALIZATION_FAMILY_LABELS[column]
            for column in REFERENCE_NORMALIZATION_FAMILY_SUPPORT_COLUMNS
            if not getattr(anchor_summary, column)
        ]
        if unsupported_families:
            partial_family_records.append(
                {
                    **dict(zip(REFERENCE_NORMALIZATION_GROUP_COLS, group_key)),
                    "anchor_run_id": anchor_run_id,
                    "unsupported_families": ",".join(unsupported_families),
                }
            )

    run_records: list[dict[str, Any]] = []
    attached_run_groups: dict[str, tuple[str, ...]] = {}
    missing_anchor_groups: set[tuple[str, ...]] = set()
    for row in result_working_df.itertuples(index=False):
        group_key = tuple(
            getattr(row, column) for column in REFERENCE_NORMALIZATION_GROUP_COLS
        )
        anchor_row = anchor_lookup.get(group_key)
        if anchor_row is None:
            # These diagnostics must not block meta analysis
            missing_anchor_groups.add(group_key)
            continue
        target_run_id = str(row.run_id)
        anchor_run_id = str(anchor_row.anchor_run_id)
        attached_run_groups[target_run_id] = group_key
        diagnostics = compute_reference_normalized_diagnostics(
            target_grid_df=grid_by_run_id[target_run_id],
            target_clean_error=float(getattr(row, clean_error_col)),
            reference_grid_df=grid_by_run_id[anchor_run_id],
            reference_clean_error=float(anchor_row.anchor_clean_error),
            eps=eps,
            target_label=target_run_id,
            reference_label=anchor_run_id,
        )
        run_records.append(
            {
                "run_id": target_run_id,
                "mCE_snaive": diagnostics.mCE_snaive,
                "relative_mCE_snaive": diagnostics.relative_mCE_snaive,
            }
        )
    if missing_anchor_groups:
        missing_groups_df = pd.DataFrame(
            [
                dict(zip(REFERENCE_NORMALIZATION_GROUP_COLS, group_key))
                for group_key in sorted(missing_anchor_groups)
            ],
            columns=list(REFERENCE_NORMALIZATION_GROUP_COLS),
        )
        examples = _sample_records(
            missing_groups_df,
            list(REFERENCE_NORMALIZATION_GROUP_COLS),
        )
        warnings.warn(
            "Reference-normalized diagnostics are unavailable for analysis group(s) "
            "without an eligible anchor. Leaving diagnostic fields unset for those "
            f"groups. Examples: {examples}.",
            stacklevel=2,
        )
    if partial_family_records:
        invalid_groups_df = pd.DataFrame(
            _sorted_records(
                partial_family_records,
                keys=[
                    *REFERENCE_NORMALIZATION_GROUP_COLS,
                    "anchor_run_id",
                    "unsupported_families",
                ],
            ),
            columns=[
                *REFERENCE_NORMALIZATION_GROUP_COLS,
                "anchor_run_id",
                "unsupported_families",
            ],
        )
        examples = _sample_records(
            invalid_groups_df,
            [
                *REFERENCE_NORMALIZATION_GROUP_COLS,
                "anchor_run_id",
                "unsupported_families",
            ],
        )
        warnings.warn(
            "Reference-normalized diagnostics attached with unsupported metric "
            "families for some resolved anchor groups. Leaving only those family "
            f"columns unset. Examples: {examples}.",
            stacklevel=2,
        )

    run_metric_df = pd.DataFrame(
        run_records,
        columns=["run_id", *REFERENCE_NORMALIZED_DIAGNOSTIC_METRIC_KEYS],
    )
    _assert_no_duplicates(
        run_metric_df,
        ["run_id"],
        context=f"{context}: run-level diagnostics are not unique per run_id",
    )
    enriched_result_df = result_df.merge(
        run_metric_df,
        on="run_id",
        how="left",
        validate="one_to_one",
    )
    result_run_ids = enriched_result_df["run_id"].astype(str)
    for support_column, metric_keys in _REFERENCE_NORMALIZED_RUN_METRIC_FAMILIES.items():
        family_name = _REFERENCE_NORMALIZATION_FAMILY_LABELS[support_column]
        supported_run_ids = {
            run_id
            for run_id, group_key in attached_run_groups.items()
            if getattr(anchor_summaries_by_group[group_key], support_column)
        }
        unsupported_run_ids = {
            run_id
            for run_id, group_key in attached_run_groups.items()
            if not getattr(anchor_summaries_by_group[group_key], support_column)
        }
        if supported_run_ids:
            missing_family_metrics = result_run_ids.isin(supported_run_ids) & (
                enriched_result_df[list(metric_keys)].isna().any(axis=1)
            )
            if missing_family_metrics.any():
                examples = _sample_records(
                    enriched_result_df.loc[
                        missing_family_metrics,
                        ["run_id", "dataset", "pipeline_id"],
                    ],
                    ["run_id", "dataset", "pipeline_id"],
                )
                raise ValueError(
                    f"{context}: failed to attach run-level {family_name} diagnostics. "
                    f"Examples: {examples}."
                )
        if unsupported_run_ids:
            unexpected_family_metrics = result_run_ids.isin(unsupported_run_ids) & (
                enriched_result_df[list(metric_keys)].notna().any(axis=1)
            )
            if unexpected_family_metrics.any():
                examples = _sample_records(
                    enriched_result_df.loc[
                        unexpected_family_metrics,
                        ["run_id", "dataset", "pipeline_id"],
                    ],
                    ["run_id", "dataset", "pipeline_id"],
                )
                raise ValueError(
                    f"{context}: attached unsupported run-level {family_name} "
                    f"diagnostics. Examples: {examples}."
                )

    enriched_scenario_summary_df = scenario_summary_df.copy()

    enriched_anchor_df = reference_anchors_df.copy()
    if enriched_anchor_df.empty:
        enriched_anchor_df["n_scenarios"] = pd.Series(dtype=int)
        enriched_anchor_df["n_severity_levels"] = pd.Series(dtype=int)
        for support_column in REFERENCE_NORMALIZATION_FAMILY_SUPPORT_COLUMNS:
            enriched_anchor_df[support_column] = pd.Series(dtype=bool)
    else:
        enriched_anchor_df["n_scenarios"] = enriched_anchor_df.apply(
            lambda row: anchor_summaries_by_group[
                tuple(getattr(row, column) for column in REFERENCE_NORMALIZATION_GROUP_COLS)
            ].n_scenarios,
            axis=1,
        )
        enriched_anchor_df["n_severity_levels"] = enriched_anchor_df.apply(
            lambda row: anchor_summaries_by_group[
                tuple(getattr(row, column) for column in REFERENCE_NORMALIZATION_GROUP_COLS)
            ].n_severity_levels,
            axis=1,
        )
        for support_column in REFERENCE_NORMALIZATION_FAMILY_SUPPORT_COLUMNS:
            enriched_anchor_df[support_column] = enriched_anchor_df.apply(
                lambda row, column=support_column: getattr(
                    anchor_summaries_by_group[
                        tuple(
                            getattr(row, group_column)
                            for group_column in REFERENCE_NORMALIZATION_GROUP_COLS
                        )
                    ],
                    column,
                ),
                axis=1,
            )
    enriched_anchor_df = enriched_anchor_df.loc[
        :,
        [
            "dataset",
            "data_config_signature",
            "eval_data_seed",
            "test_metric",
            "anchor_model",
            "anchor_run_id",
            "anchor_clean_error",
            "n_scenarios",
            "n_severity_levels",
            "reference_normalization_anchor_model",
            *REFERENCE_NORMALIZATION_FAMILY_SUPPORT_COLUMNS,
        ],
    ]
    return (
        enriched_result_df,
        enriched_scenario_summary_df,
        enriched_anchor_df,
    )


def _sort_and_rank_method_aggregates(method_aggregates_df: pd.DataFrame) -> pd.DataFrame:
    """Sort method aggregates and rank methods within each selection metric family.

    Selection scores are only comparable when they come from the same
    selection metric (e.g., ``best_val_loss`` vs ``MSE_val`` are not
    cross-comparable). Ranking is therefore scoped to
    ``(dataset, selection_metric_name)``.
    """
    required_cols = {
        "dataset",
        "robustness_method",
        "selection_metric_name",
        "selection_score",
        "architectures_covered",
        "variant_count",
    }
    missing_cols = sorted(required_cols - set(method_aggregates_df.columns))
    if missing_cols:
        raise ValueError(
            "Cannot rank method aggregates: missing required columns "
            f"{missing_cols}."
        )
    ranked = method_aggregates_df.sort_values(
        [
            "dataset",
            "selection_metric_name",
            "selection_score",
            "architectures_covered",
            "variant_count",
            "robustness_method",
        ],
        ascending=[True, True, True, False, False, True],
    ).reset_index(drop=True)
    ranked["method_rank"] = (
        ranked.groupby(["dataset", "selection_metric_name"], dropna=False).cumcount() + 1
    )
    return ranked


def _bootstrap_percentile_ci_bounds(
    values: np.ndarray,
    *,
    confidence_level: float,
    context: str,
) -> tuple[float, float]:
    draws = np.asarray(values, dtype=np.float64)
    if draws.ndim != 1:
        raise ValueError(f"{context}: bootstrap draws must be one-dimensional.")
    if draws.size == 0:
        raise ValueError(f"{context}: bootstrap draws are empty.")
    if not np.isfinite(draws).all():
        raise ValueError(f"{context}: bootstrap draws must be finite.")
    tail = (1.0 - float(confidence_level)) / 2.0
    quantiles = np.quantile(draws, [tail, 1.0 - tail], method="linear")
    return float(quantiles[0]), float(quantiles[1])


def _build_pipeline_method_delta_results(
    method_delta_plot_df: pd.DataFrame,
    *,
    perf_col: str,
    test_metric: str,
    bootstrap_resamples: int,
    bootstrap_confidence_level: float,
) -> pd.DataFrame:
    context = "Cannot build pipeline-method delta results"
    if method_delta_plot_df.empty:
        return pd.DataFrame(columns=list(PIPELINE_METHOD_DELTA_RESULTS_COLUMNS))

    parsed_bootstrap_resamples = parse_required_positive_int(
        bootstrap_resamples,
        key="bootstrap_resamples",
    )
    confidence_level = parse_bootstrap_ci_confidence_level(
        bootstrap_confidence_level,
        key="bootstrap_confidence_level",
    )
    if confidence_level is None:
        raise ValueError(
            f"{context}: bootstrap_confidence_level is required."
        )

    perf_delta_col = (
        "delta_err_clean"
        if "delta_err_clean" in method_delta_plot_df.columns
        else f"delta_{perf_col}"
    )
    metric_source_cols = {
        "delta_err_clean": perf_delta_col,
        "delta_D_w": "delta_D_w",
        "delta_D_mean": "delta_D_mean",
        "delta_err_pert_ws": "delta_err_pert_ws",
        "delta_err_pert_mean": "delta_err_pert_mean",
    }
    _require_columns(
        method_delta_plot_df,
        {
            "dataset",
            "robustness_method",
            "backbone_architecture",
            "pipeline_id",
            "data_config_signature",
            "eval_data_seed",
            *metric_source_cols.values(),
        },
        context=context,
    )
    _assert_single_pipeline_per_method_backbone(
        method_delta_plot_df[
            [
                "dataset",
                "robustness_method",
                "backbone_architecture",
                "pipeline_id",
            ]
        ].drop_duplicates(),
        context=context,
    )

    working_df = _require_nonempty_string_columns(
        method_delta_plot_df,
        [
            "dataset",
            "robustness_method",
            "backbone_architecture",
            "pipeline_id",
            "data_config_signature",
        ],
        context=context,
        sample_cols=[
            "dataset",
            "robustness_method",
            "backbone_architecture",
            "pipeline_id",
        ],
    )
    working_df["eval_data_seed"] = require_integer_series(
        working_df,
        "eval_data_seed",
        context=context,
        sample_cols=[
            "dataset",
            "robustness_method",
            "backbone_architecture",
            "pipeline_id",
            "eval_data_seed",
        ],
        min_value=0,
    )
    for output_col, source_col in metric_source_cols.items():
        working_df[source_col] = require_numeric_series(
            working_df[source_col],
            column_name=source_col,
            context=context,
            allow_nan=False,
            allow_infinite=False,
        ).astype(float)

    rows: list[dict[str, Any]] = []
    for (dataset_name, method_name), group_df in working_df.groupby(
        ["dataset", "robustness_method"],
        dropna=False,
        sort=True,
    ):
        dataset_label = str(dataset_name)
        method_label = str(method_name)
        unique_eval_seeds = sorted(group_df["eval_data_seed"].drop_duplicates().tolist())
        if len(unique_eval_seeds) != 1:
            raise ValueError(
                f"{context}: ({dataset_label}, {method_label}) contains inconsistent "
                f"eval_data_seed values {unique_eval_seeds}."
            )
        unique_signatures = sorted(group_df["data_config_signature"].drop_duplicates().tolist())
        if len(unique_signatures) != 1:
            raise ValueError(
                f"{context}: ({dataset_label}, {method_label}) contains inconsistent "
                f"data_config_signature values {unique_signatures}."
            )
        data_config_signature = str(unique_signatures[0])
        group_seed = derive_seed(
            int(unique_eval_seeds[0]),
            build_method_delta_pair_bootstrap_ci_seed_key(
                test_metric,
                dataset=dataset_label,
                data_config_signature=data_config_signature,
                robustness_method=method_label,
            ),
        )
        row: dict[str, Any] = {
            "dataset": dataset_label,
            "robustness_method": method_label,
            "count": int(len(group_df)),
            "delta_bootstrap_semantics": METHOD_DELTA_PAIR_BOOTSTRAP_CI_SEMANTICS,
            "delta_bootstrap_resamples": int(parsed_bootstrap_resamples),
            "delta_bootstrap_confidence_level": float(confidence_level),
            "delta_bootstrap_seed": int(group_seed),
        }
        # Canonicalize pair order so bootstrap resampling is invariant to upstream row order.
        group_df = group_df.sort_values(
            ["backbone_architecture", "pipeline_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        n_pairs = int(len(group_df))
        for output_col, source_col in metric_source_cols.items():
            metric_values = group_df[source_col].to_numpy(dtype=np.float64)
            point_estimate = float(metric_values.mean(dtype=np.float64))
            metric_seed = derive_seed(int(group_seed), output_col)
            rng = np.random.default_rng(int(metric_seed))
            bootstrap_draws = np.empty(int(parsed_bootstrap_resamples), dtype=np.float64)
            for draw_idx in range(int(parsed_bootstrap_resamples)):
                sample_idx = rng.integers(0, n_pairs, size=n_pairs)
                bootstrap_draws[draw_idx] = float(
                    metric_values[sample_idx].mean(dtype=np.float64)
                )
            ci_lo, ci_hi = _bootstrap_percentile_ci_bounds(
                bootstrap_draws,
                confidence_level=float(confidence_level),
                context=f"{context} for ({dataset_label}, {method_label}, {output_col})",
            )
            row[f"{output_col}_mean"] = point_estimate
            row[f"{output_col}_CI_lo"] = ci_lo
            row[f"{output_col}_CI_hi"] = ci_hi
        rows.append(row)

    return pd.DataFrame(rows, columns=list(PIPELINE_METHOD_DELTA_RESULTS_COLUMNS))


def _scenario_family_lookup(registry: CoreFigureRegistry) -> dict[str, str]:
    scenario_to_family: dict[str, str] = {}
    for family_name, scenario_names in registry.scenario_groups.items():
        family_label = str(family_name).strip()
        if not family_label:
            raise ValueError("Core scenario group names must be non-empty.")
        for scenario_name in scenario_names:
            scenario_key = str(scenario_name).strip()
            if not scenario_key:
                raise ValueError(
                    f"Core scenario group '{family_label}' contains an empty scenario."
                )
            if scenario_key in scenario_to_family:
                raise ValueError(
                    f"Scenario '{scenario_key}' is assigned to multiple core groups."
                )
            scenario_to_family[scenario_key] = family_label
    expected_scenarios = set(registry.scenario_display_order)
    grouped_scenarios = set(scenario_to_family)
    if grouped_scenarios != expected_scenarios:
        raise ValueError(
            "Core scenario groups do not cover the scenario display order: "
            f"groups={sorted(grouped_scenarios)}, "
            f"order={sorted(expected_scenarios)}."
        )
    return scenario_to_family


def _effect_direction(delta_value: float) -> str:
    if delta_value < 0.0:
        return "improves"
    if delta_value > 0.0:
        return "worsens"
    return "unchanged"


def _ranked_dataset_value_text(
    rows_df: pd.DataFrame,
    *,
    value_col: str,
    ascending: bool,
) -> str:
    sorted_rows = rows_df.sort_values(
        [value_col, "dataset_label"],
        ascending=[ascending, True],
        kind="mergesort",
    )
    values: list[str] = []
    for row in sorted_rows.itertuples(index=False):
        values.append(
            f"{row.dataset_label}={float(getattr(row, value_col)):.6g}"
        )
    return "; ".join(values)


def _build_method_scenario_family_delta_tables(
    paired_scenario_delta_df: pd.DataFrame,
    *,
    registry: CoreFigureRegistry,
    delta_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context = "Cannot build method scenario-family delta framing"
    empty_delta_df = pd.DataFrame(columns=list(METHOD_SCENARIO_FAMILY_DELTA_COLUMNS))
    empty_summary_df = pd.DataFrame(
        columns=list(METHOD_SCENARIO_FAMILY_SUMMARY_COLUMNS)
    )
    if paired_scenario_delta_df.empty:
        return empty_delta_df, empty_summary_df

    required_cols = {
        "dataset",
        "robustness_method",
        "model_architecture",
        "scenario",
        "baseline_D",
        delta_col,
    }
    _require_columns(paired_scenario_delta_df, required_cols, context=context)
    working_df = _require_nonempty_string_columns(
        paired_scenario_delta_df,
        ["dataset", "robustness_method", "model_architecture", "scenario"],
        context=context,
        sample_cols=[
            "dataset",
            "robustness_method",
            "model_architecture",
            "scenario",
        ],
    )
    working_df["baseline_D"] = require_numeric_series(
        working_df["baseline_D"],
        column_name="baseline_D",
        context=context,
        allow_nan=False,
        allow_infinite=False,
    ).astype(float)
    working_df[delta_col] = require_numeric_series(
        working_df[delta_col],
        column_name=delta_col,
        context=context,
        allow_nan=False,
        allow_infinite=False,
    ).astype(float)

    registry_dataset_keys = tuple(dataset_key for dataset_key, _ in registry.dataset_spec)
    registry_dataset_display = dict(registry.dataset_spec)
    validate_scoped_raw_display_id_values(
        working_df["dataset"].tolist(),
        raw_ids=registry_dataset_keys,
        display_mapping=registry_dataset_display,
        known_raw_ids=tuple(_known_dataset_registry_keys()),
        context=context,
        id_label="dataset",
    )
    validate_scoped_raw_display_id_values(
        working_df["robustness_method"].tolist(),
        raw_ids=tuple(registry.method_order),
        display_mapping=registry.method_display,
        known_raw_ids=tuple(_known_robustness_method_keys()),
        context=context,
        id_label="robustness_method",
    )
    validate_scoped_raw_display_id_values(
        working_df["scenario"].tolist(),
        raw_ids=tuple(registry.scenario_display_order),
        display_mapping=registry.scenario_display,
        known_raw_ids=tuple(registry.scenario_display_order),
        context=context,
        id_label="scenario",
    )

    scenario_to_family = _scenario_family_lookup(registry)
    family_scenarios = {
        str(family_name): tuple(str(scenario) for scenario in scenarios)
        for family_name, scenarios in registry.scenario_groups.items()
    }
    family_scenarios_text = {
        family_name: ",".join(scenarios)
        for family_name, scenarios in family_scenarios.items()
    }
    family_scenarios_display_text = {
        family_name: ", ".join(registry.scenario_display[scenario] for scenario in scenarios)
        for family_name, scenarios in family_scenarios.items()
    }

    working_df["scenario_family"] = working_df["scenario"].map(scenario_to_family)
    missing_family = working_df["scenario_family"].isna()
    if missing_family.any():
        examples = _sample_records(
            working_df.loc[missing_family],
            ["dataset", "robustness_method", "scenario"],
        )
        raise ValueError(
            f"{context}: scenario rows are missing family assignments. "
            f"Examples: {examples}."
        )

    scenario_effect_df = (
        working_df.groupby(
            ["dataset", "robustness_method", "scenario_family", "scenario"],
            dropna=False,
            sort=True,
        )[delta_col]
        .mean()
        .rename("scenario_delta_D_mean")
        .reset_index()
    )
    improved_counts_df = (
        scenario_effect_df.assign(
            scenario_improved=scenario_effect_df["scenario_delta_D_mean"] < 0.0
        )
        .groupby(
            ["dataset", "robustness_method", "scenario_family"],
            dropna=False,
            sort=True,
        )["scenario_improved"]
        .sum()
        .rename("improved_scenario_count")
        .reset_index()
    )
    dataset_delta_df = (
        working_df.groupby(
            ["dataset", "robustness_method", "scenario_family"],
            dropna=False,
            sort=True,
        )
        .agg(
            architecture_count=("model_architecture", "nunique"),
            scenario_count=("scenario", "nunique"),
            baseline_family_D_mean=("baseline_D", "mean"),
            method_family_delta_D_mean=(delta_col, "mean"),
        )
        .reset_index()
    )
    dataset_delta_df = dataset_delta_df.merge(
        improved_counts_df,
        on=["dataset", "robustness_method", "scenario_family"],
        how="left",
        validate="one_to_one",
    )
    if dataset_delta_df["improved_scenario_count"].isna().any():
        raise ValueError(
            f"{context}: failed to attach improved scenario counts to dataset rows."
        )
    dataset_delta_df["improved_scenario_count"] = dataset_delta_df[
        "improved_scenario_count"
    ].astype(int)
    dataset_delta_df["dataset_label"] = dataset_delta_df["dataset"].map(
        registry_dataset_display
    )
    dataset_delta_df["method_label"] = dataset_delta_df["robustness_method"].map(
        registry.method_display
    )
    dataset_delta_df["family_scenarios"] = dataset_delta_df["scenario_family"].map(
        family_scenarios_text
    )
    dataset_delta_df["family_scenarios_display"] = dataset_delta_df[
        "scenario_family"
    ].map(family_scenarios_display_text)
    dataset_delta_df["effect_direction"] = dataset_delta_df[
        "method_family_delta_D_mean"
    ].map(_effect_direction)

    if dataset_delta_df[
        ["dataset_label", "method_label", "family_scenarios", "family_scenarios_display"]
    ].isna().any(axis=1).any():
        raise ValueError(
            f"{context}: failed to attach registry display metadata to dataset rows."
        )

    baseline_rank_idx = dataset_delta_df.sort_values(
        [
            "robustness_method",
            "scenario_family",
            "baseline_family_D_mean",
            "dataset",
        ],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).index
    dataset_delta_df.loc[baseline_rank_idx, "baseline_family_D_rank_desc"] = (
        dataset_delta_df.loc[baseline_rank_idx]
        .groupby(["robustness_method", "scenario_family"], dropna=False)
        .cumcount()
        + 1
    )
    gain_rank_idx = dataset_delta_df.sort_values(
        [
            "robustness_method",
            "scenario_family",
            "method_family_delta_D_mean",
            "dataset",
        ],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).index
    dataset_delta_df.loc[gain_rank_idx, "method_family_gain_rank_asc"] = (
        dataset_delta_df.loc[gain_rank_idx]
        .groupby(["robustness_method", "scenario_family"], dropna=False)
        .cumcount()
        + 1
    )
    dataset_delta_df["baseline_family_D_rank_desc"] = dataset_delta_df[
        "baseline_family_D_rank_desc"
    ].astype(int)
    dataset_delta_df["method_family_gain_rank_asc"] = dataset_delta_df[
        "method_family_gain_rank_asc"
    ].astype(int)
    dataset_delta_df = dataset_delta_df.sort_values(
        [
            "robustness_method",
            "scenario_family",
            "method_family_gain_rank_asc",
            "dataset",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    dataset_delta_df = dataset_delta_df.loc[
        :,
        list(METHOD_SCENARIO_FAMILY_DELTA_COLUMNS),
    ]

    summary_records: list[dict[str, Any]] = []
    for (method_name, family_name), group_df in dataset_delta_df.groupby(
        ["robustness_method", "scenario_family"],
        dropna=False,
        sort=True,
    ):
        delta_values = group_df["method_family_delta_D_mean"].to_numpy(dtype=float)
        baseline_values = group_df["baseline_family_D_mean"].to_numpy(dtype=float)
        mean_delta = float(delta_values.mean(dtype=np.float64))
        summary_records.append(
            {
                "robustness_method": str(method_name),
                "method_label": registry.method_display[str(method_name)],
                "scenario_family": str(family_name),
                "family_scenarios": family_scenarios_text[str(family_name)],
                "family_scenarios_display": family_scenarios_display_text[
                    str(family_name)
                ],
                "dataset_count": int(group_df["dataset"].nunique()),
                "improved_dataset_count": int(
                    (group_df["method_family_delta_D_mean"] < 0.0).sum()
                ),
                "baseline_family_D_mean": float(baseline_values.mean(dtype=np.float64)),
                "method_family_delta_D_mean": mean_delta,
                "method_family_delta_D_min": float(delta_values.min()),
                "method_family_delta_D_max": float(delta_values.max()),
                "baseline_impact_dataset_order": _ranked_dataset_value_text(
                    group_df,
                    value_col="baseline_family_D_mean",
                    ascending=False,
                ),
                "largest_gain_dataset_order": _ranked_dataset_value_text(
                    group_df,
                    value_col="method_family_delta_D_mean",
                    ascending=True,
                ),
                "effect_direction": _effect_direction(mean_delta),
            }
        )
    summary_df = pd.DataFrame(
        summary_records,
        columns=[
            column
            for column in METHOD_SCENARIO_FAMILY_SUMMARY_COLUMNS
            if column not in {"method_family_delta_rank", "practitioner_frame"}
        ],
    )
    if summary_df.empty:
        return dataset_delta_df, empty_summary_df

    rank_idx = summary_df.sort_values(
        [
            "robustness_method",
            "method_family_delta_D_mean",
            "scenario_family",
        ],
        ascending=[True, True, True],
        kind="mergesort",
    ).index
    summary_df.loc[rank_idx, "method_family_delta_rank"] = (
        summary_df.loc[rank_idx]
        .groupby("robustness_method", dropna=False)
        .cumcount()
        + 1
    )
    summary_df["method_family_delta_rank"] = summary_df[
        "method_family_delta_rank"
    ].astype(int)
    practitioner_frames: list[str] = []
    for row in summary_df.itertuples(index=False):
        practitioner_frames.append(
            f"{row.method_label}: {row.effect_direction} "
            f"{row.scenario_family} faults ({row.family_scenarios_display}); "
            f"baseline burden by dataset: {row.baseline_impact_dataset_order}; "
            f"method deltas by dataset: {row.largest_gain_dataset_order}."
        )
    summary_df["practitioner_frame"] = practitioner_frames
    summary_df = summary_df.sort_values(
        ["robustness_method", "method_family_delta_rank", "scenario_family"],
        kind="mergesort",
    ).reset_index(drop=True)
    summary_df = summary_df.loc[
        :,
        list(METHOD_SCENARIO_FAMILY_SUMMARY_COLUMNS),
    ]
    return dataset_delta_df, summary_df


def _require_core_delta_metrics(
    deltas_df: pd.DataFrame,
    *,
    perf_col: str,
) -> list[tuple[str, str]]:
    """Require core delta columns used in the primary comparison figure."""
    perf_delta_col = (
        "delta_err_clean"
        if "delta_err_clean" in deltas_df.columns
        else f"delta_{perf_col}"
    )
    metric_specs: list[tuple[str, str]] = [
        (perf_delta_col, "delta_err_clean")
    ] + [
        (_core_delta_column(spec.metric_key), spec.metric_key)
        for spec in CORE_ROBUSTNESS_METRIC_SPECS
    ]
    core_metric_names = ", ".join(spec.metric_key for spec in CORE_ROBUSTNESS_METRIC_SPECS)
    missing_cols = [column for column, _ in metric_specs if column not in deltas_df.columns]
    if missing_cols:
        raise ValueError(
            "Core metric deltas are missing required columns "
            f"{missing_cols}. Expected deltas for {perf_col} and [{core_metric_names}]."
        )
    return metric_specs


def _core_robustness_metric_spec(metric_key: str) -> _CoreRobustnessMetricSpec:
    for spec in CORE_ROBUSTNESS_METRIC_SPECS:
        if spec.metric_key == metric_key:
            return spec
    raise ValueError(f"Unknown core robustness metric '{metric_key}'.")


def _filter_core_metric_available_rows(
    deltas_df: pd.DataFrame,
    *,
    perf_col: str,
    metric_key: str,
    context: str,
) -> pd.DataFrame:
    delta_perf_col = (
        "delta_err_clean"
        if "delta_err_clean" in deltas_df.columns
        else f"delta_{perf_col}"
    )
    robust_delta_col = _core_delta_column(metric_key)
    required_cols = [delta_perf_col, robust_delta_col]
    missing_cols = [column for column in required_cols if column not in deltas_df.columns]
    if missing_cols:
        raise ValueError(
            f"{context}: missing required delta columns {missing_cols} for metric "
            f"'{metric_key}'."
        )

    perf_missing = deltas_df[delta_perf_col].isna()
    if perf_missing.any():
        examples = _sample_records(
            deltas_df.loc[
                perf_missing,
                ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
            ],
            ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
        )
        raise ValueError(
            f"{context}: performance deltas contain missing values for "
            f"'{delta_perf_col}'. Examples: {examples}."
        )

    robust_missing = deltas_df[robust_delta_col].isna()
    if not robust_missing.any():
        return deltas_df.copy()

    if metric_key not in REFERENCE_NORMALIZED_DIAGNOSTIC_METRIC_KEYS:
        examples = _sample_records(
            deltas_df.loc[
                robust_missing,
                ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
            ],
            ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
        )
        raise ValueError(
            f"{context}: non-reference core metric deltas contain missing values for "
            f"'{robust_delta_col}'. Examples: {examples}."
        )

    return deltas_df.loc[~robust_missing].copy()


def _build_variant_selection_summary(selection_df: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic per-variant diagnostics within each dataset/method group."""
    required_cols = {
        "dataset",
        "robustness_method",
        "pipeline_id",
        "selection_metric",
        "selection_value",
        "model_architecture",
    }
    missing_cols = sorted(required_cols - set(selection_df.columns))
    if missing_cols:
        raise ValueError(
            "Cannot build variant selection summary: missing required columns "
            f"{missing_cols}."
        )
    variant_selection_summary_df = (
        selection_df.groupby(
            ["dataset", "robustness_method", "pipeline_id", "selection_metric"],
            dropna=False,
        )
        .agg(
            selection_score=("selection_value", "mean"),
            architectures_covered=("model_architecture", "nunique"),
            run_count=("pipeline_id", "size"),
        )
        .reset_index()
    )
    variant_selection_summary_df = variant_selection_summary_df.sort_values(
        [
            "dataset",
            "robustness_method",
            "selection_score",
            "architectures_covered",
            "pipeline_id",
        ],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)
    variant_selection_summary_df["selection_rank"] = (
        variant_selection_summary_df.groupby(
            ["dataset", "robustness_method"], dropna=False
        ).cumcount()
        + 1
    )
    variant_selection_summary_df["is_representative"] = (
        variant_selection_summary_df["selection_rank"] == 1
    )
    variant_selection_summary_df["candidate_count"] = variant_selection_summary_df.groupby(
        ["dataset", "robustness_method"], dropna=False
    )["pipeline_id"].transform("nunique")
    return variant_selection_summary_df.rename(
        columns={"selection_metric": "selection_metric_name"}
    )


def _build_pipeline_method_candidates(
    variant_selection_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract one representative variant per dataset/method and validate uniqueness."""
    required_cols = {
        "dataset",
        "robustness_method",
        "pipeline_id",
        "is_representative",
    }
    missing_cols = sorted(required_cols - set(variant_selection_summary_df.columns))
    if missing_cols:
        raise ValueError(
            "Cannot build representative variant candidates: missing required columns "
            f"{missing_cols}."
        )
    pipeline_method_candidates_df = variant_selection_summary_df.loc[
        variant_selection_summary_df["is_representative"]
    ].copy()
    pipeline_method_candidates_df = pipeline_method_candidates_df.sort_values(
        ["dataset", "robustness_method", "pipeline_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    representative_counts = pipeline_method_candidates_df.groupby(
        ["dataset", "robustness_method"], dropna=False
    ).size()
    if (representative_counts != 1).any():
        bad_groups = representative_counts[representative_counts != 1]
        preview = ", ".join(
            f"{dataset}/{method}={int(count)}"
            for (dataset, method), count in bad_groups.sort_index().items()
        )
        raise ValueError(
            "Representative variant selection must output exactly one pipeline_id per "
            f"(dataset, robustness_method). Violations: {preview}."
        )
    if (pipeline_method_candidates_df["pipeline_id"].astype(str) == "baseline").any():
        raise ValueError(
            "Representative variant selection produced baseline pipeline_id values, "
            "which is not allowed."
        )
    return pipeline_method_candidates_df


def _build_method_selection_df(
    result_df: pd.DataFrame,
    *,
    test_metric: str,
    improvement_selection_mode: str,
) -> pd.DataFrame:
    """Build and validate non-baseline winner-pool selection candidates."""
    required_cols = {
        "pipeline_id",
        "robustness_method",
        "pipeline_kind",
        "dataset",
        "model_architecture",
        "selection_pool",
        "best_val_loss",
        f"{test_metric}_val",
    }
    missing_cols = sorted(required_cols - set(result_df.columns))
    if missing_cols:
        raise ValueError(
            "Cannot build method diagnostics: missing required columns "
            f"{missing_cols}. Ensure runs log pipeline tags and validation metrics."
        )

    selection_df = result_df.loc[
        result_df["pipeline_id"].astype(str) != "baseline"
    ].copy()
    if selection_df.empty:
        return selection_df

    if (selection_df["pipeline_id"].astype(str) == "baseline").any():
        raise ValueError(
            "method_selection_df contains baseline rows; expected non-baseline runs only."
        )
    if (selection_df["robustness_method"].astype(str) == "baseline").any():
        raise ValueError(
            "method_selection_df contains baseline robustness_method rows; "
            "selection candidates must be robustness improvements."
        )
    if (selection_df["selection_pool"].astype(str) != "winner_pool").any():
        raise ValueError(
            "method_selection_df contains non-winner-pool rows. "
            "Selection candidates must come from best_model=true parent runs."
        )

    missing_pid = selection_df["pipeline_id"].isna()
    if missing_pid.any():
        raise ValueError("Found non-baseline runs missing 'pipeline_id'.")

    missing_family = selection_df["robustness_method"].isna()
    if missing_family.any():
        missing_count = int(missing_family.sum())
        example_pids = (
            selection_df.loc[missing_family, "pipeline_id"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raise ValueError(
            f"Found {missing_count} non-baseline runs missing 'robustness_method'. "
            f"Example pipeline_ids: {example_pids}."
        )

    missing_kind = selection_df["pipeline_kind"].isna()
    if missing_kind.any():
        missing_count = int(missing_kind.sum())
        example_pids = (
            selection_df.loc[missing_kind, "pipeline_id"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raise ValueError(
            f"Found {missing_count} non-baseline runs missing 'pipeline_kind'. "
            f"Example pipeline_ids: {example_pids}."
        )

    kind_values = selection_df["pipeline_kind"].astype(str).str.strip()
    unknown_kind = ~kind_values.isin(ALLOWED_PIPELINE_KINDS)
    if unknown_kind.any():
        examples = (
            selection_df.loc[unknown_kind, ["pipeline_id", "pipeline_kind"]]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Cannot build method diagnostics because candidate rows contain unknown "
            f"pipeline_kind values. Allowed kinds: {list(ALLOWED_PIPELINE_KINDS)}. "
            f"Examples: {examples}."
        )
    selection_df["pipeline_kind"] = kind_values

    kind_mix = selection_df.groupby("robustness_method", dropna=False)[
        "pipeline_kind"
    ].nunique()
    mixed_kind_methods = kind_mix[kind_mix > 1]
    if not mixed_kind_methods.empty:
        preview = ", ".join(
            f"{method}={int(count)}"
            for method, count in mixed_kind_methods.sort_index().items()
        )
        raise ValueError(
            "Cannot build method diagnostics because robustness_method mixes pipeline_kind "
            f"values across runs (e.g. {preview})."
        )

    selection_df["selection_metric"] = selection_df.apply(
        lambda row: selection_metric_key_for_kind(
            pipeline_kind=str(row["pipeline_kind"]),
            robustness_method=str(row["robustness_method"]),
            test_metric=test_metric,
            improvement_selection_mode=improvement_selection_mode,
            run_id=(
                f"{row['dataset']}/{row['robustness_method']}/"
                f"{row['pipeline_id']}/{row['model_architecture']}"
            ),
        ),
        axis=1,
    )
    perturbed_metric_cols = perturbed_selection_metric_keys(
        test_metric=test_metric,
        run_id="method diagnostics",
    )
    perturbed_metric_set = set(perturbed_metric_cols)
    perturbed_mask = selection_df["selection_metric"].isin(perturbed_metric_set)
    required_selection_metric_cols = sorted(
        set(selection_df["selection_metric"].astype(str))
    )
    missing_selection_metric_cols = sorted(
        set(required_selection_metric_cols) - set(selection_df.columns)
    )
    if missing_selection_metric_cols:
        raise ValueError(
            "Cannot build method diagnostics because candidate rows are missing "
            f"required selection metric columns {missing_selection_metric_cols}."
        )
    if perturbed_mask.any():
        missing_perturbed_cols = sorted(perturbed_metric_set - set(selection_df.columns))
        if missing_perturbed_cols:
            raise ValueError(
                "Cannot build method diagnostics because candidate rows are missing "
                f"required perturbed selection metric columns {missing_perturbed_cols}."
            )
    numeric_metric_cols = set(required_selection_metric_cols)
    if perturbed_mask.any():
        numeric_metric_cols.update(perturbed_metric_cols)
    for metric_col in sorted(numeric_metric_cols):
        selection_df[metric_col] = pd.to_numeric(
            selection_df[metric_col],
            errors="raise",
        )
    selection_df["selection_value"] = selection_df.apply(
        lambda row: row[row["selection_metric"]],
        axis=1,
    )

    missing_metric = selection_df["selection_value"].isna()
    if missing_metric.any():
        missing_count = int(missing_metric.sum())
        examples = (
            selection_df.loc[
                missing_metric,
                [
                    "dataset",
                    "robustness_method",
                    "pipeline_id",
                    "pipeline_kind",
                    "selection_metric",
                ],
            ]
            .astype(str)
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"Found {missing_count} non-baseline runs missing required selection metric "
            f"({', '.join(required_selection_metric_cols)}). "
            f"Examples: {examples}."
        )
    if perturbed_mask.any():
        perturbed_rows = selection_df.loc[perturbed_mask].copy()
        missing_perturbed_values = perturbed_rows[list(perturbed_metric_cols)].isna().any(
            axis=1
        )
        if missing_perturbed_values.any():
            examples = (
                perturbed_rows.loc[
                    missing_perturbed_values,
                    [
                        "dataset",
                        "robustness_method",
                        "pipeline_id",
                        "pipeline_kind",
                    ],
                ]
                .astype(str)
                .drop_duplicates()
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                "Found non-baseline runs missing required perturbed selection metric(s) "
                f"{list(perturbed_metric_cols)}. Examples: {examples}."
            )

    metric_mix = selection_df.groupby(
        ["dataset", "robustness_method"], dropna=False
    )["selection_metric"].nunique()
    mixed_groups = metric_mix[metric_mix > 1]
    if not mixed_groups.empty:
        preview = ", ".join(
            f"{dataset}/{method}={int(count)}"
            for (dataset, method), count in mixed_groups.sort_index().items()
        )
        raise ValueError(
            "Cannot build method diagnostics because some dataset/method groups mix "
            f"selection metrics (e.g. {preview})."
        )

    selection_identity_cols = [
        "dataset",
        "robustness_method",
        "pipeline_id",
        "model_architecture",
    ]
    duplicate_selection_rows = selection_df.duplicated(
        selection_identity_cols, keep=False
    )
    if duplicate_selection_rows.any():
        examples = (
            selection_df.loc[duplicate_selection_rows, selection_identity_cols]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Cannot build method diagnostics: candidate rows are not unique per "
            f"{selection_identity_cols}. Examples: {examples}."
        )

    return selection_df


def _build_canonical_method_analysis_df(
    result_df: pd.DataFrame,
    *,
    test_metric: str,
    improvement_selection_mode: str,
    allowed_methods: set[str] | None = None,
) -> pd.DataFrame:
    """Build and validate the canonical non-baseline winner-pool method frame."""
    canonical_df = _build_method_selection_df(
        result_df,
        test_metric=test_metric,
        improvement_selection_mode=improvement_selection_mode,
    )
    if canonical_df.empty:
        return canonical_df

    required_cols = set(METHOD_ANALYSIS_IDENTITY_COLS) | {
        "selection_pool",
        "selection_metric",
        "selection_value",
    }
    missing_cols = sorted(required_cols - set(canonical_df.columns))
    if missing_cols:
        raise ValueError(
            "Canonical method-analysis dataframe is missing required columns "
            f"{missing_cols}."
        )

    if canonical_df["data_config_signature"].isna().any():
        raise ValueError(
            "Canonical method-analysis dataframe contains missing data_config_signature values."
        )

    if allowed_methods is not None:
        normalized_allowed_methods: set[str] = set()
        for raw_method in allowed_methods:
            method = parse_required_nonempty_string(
                raw_method,
                key="allowed_methods",
                context="Canonical method-analysis dataframe scope",
            )
            normalized_allowed_methods.add(method)
        out_of_scope_mask = ~canonical_df["robustness_method"].astype(str).str.strip().isin(
            normalized_allowed_methods
        )
        if out_of_scope_mask.any():
            dropped_methods = sorted(
                canonical_df.loc[out_of_scope_mask, "robustness_method"]
                .astype(str)
                .str.strip()
                .drop_duplicates()
                .tolist()
            )
            print(
                "Canonical method-analysis dataframe: dropping out-of-scope "
                f"robustness methods {dropped_methods}."
            )
            canonical_df = canonical_df.loc[~out_of_scope_mask].copy()
        if canonical_df.empty:
            return canonical_df

    identity_cols = list(METHOD_ANALYSIS_IDENTITY_COLS)
    duplicate_rows = canonical_df.duplicated(identity_cols, keep=False)
    if duplicate_rows.any():
        examples = (
            canonical_df.loc[duplicate_rows, identity_cols]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Canonical method-analysis dataframe contains duplicate rows per "
            f"{identity_cols}. Examples: {examples}."
        )

    if (canonical_df["selection_pool"].astype(str) != "winner_pool").any():
        raise ValueError(
            "Canonical method-analysis dataframe contains non-winner-pool rows."
        )
    return canonical_df


def _select_canonical_method_deltas(
    method_deltas_df: pd.DataFrame,
    *,
    canonical_method_df: pd.DataFrame,
) -> pd.DataFrame:
    """Require method delta rows to be a strict 1:1 projection of canonical method rows."""
    if canonical_method_df.empty:
        if method_deltas_df.empty:
            return method_deltas_df
        raise ValueError(
            "Cannot validate method deltas against canonical method source: "
            "canonical method dataframe is empty."
        )
    if method_deltas_df.empty:
        raise ValueError(
            "Method delta diagnostics are missing canonical method rows. "
            "No delta rows were produced."
        )

    identity_cols = list(METHOD_ANALYSIS_IDENTITY_COLS)
    missing_delta_cols = sorted(set(identity_cols) - set(method_deltas_df.columns))
    if missing_delta_cols:
        raise ValueError(
            "Cannot validate method deltas: missing identity columns "
            f"{missing_delta_cols}."
        )
    missing_canonical_cols = sorted(set(identity_cols) - set(canonical_method_df.columns))
    if missing_canonical_cols:
        raise ValueError(
            "Cannot validate method deltas against canonical method source: "
            f"canonical frame is missing identity columns {missing_canonical_cols}."
        )

    delta_dups = method_deltas_df.duplicated(identity_cols, keep=False)
    if delta_dups.any():
        examples = (
            method_deltas_df.loc[delta_dups, identity_cols]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Method deltas contain duplicate rows per canonical identity "
            f"{identity_cols}. Examples: {examples}."
        )

    canonical_keys = canonical_method_df[identity_cols].drop_duplicates().copy()
    delta_keys = method_deltas_df[identity_cols].drop_duplicates().copy()

    extra_rows = delta_keys.merge(
        canonical_keys,
        on=identity_cols,
        how="left",
        indicator=True,
    )
    extra_rows = extra_rows.loc[extra_rows["_merge"] == "left_only"]
    if not extra_rows.empty:
        examples = (
            extra_rows[identity_cols]
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Method delta diagnostics include rows outside canonical method source. "
            f"Examples: {examples}."
        )

    missing_rows = canonical_keys.merge(
        delta_keys,
        on=identity_cols,
        how="left",
        indicator=True,
    )
    missing_rows = missing_rows.loc[missing_rows["_merge"] == "left_only"]
    if not missing_rows.empty:
        examples = (
            missing_rows[identity_cols]
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Method delta diagnostics are missing canonical method rows. "
            f"Examples: {examples}."
        )
    return method_deltas_df.copy()


def _canonical_method_winner_plot_df(
    method_selection_df: pd.DataFrame,
    *,
    context: str,
    required_cols: set[str] | None = None,
) -> pd.DataFrame:
    """Return canonical non-baseline winner-pool rows for method-facing plots."""
    base_required_cols = {
        "dataset",
        "data_config_signature",
        "robustness_method",
        "pipeline_id",
        "model_architecture",
        "selection_pool",
    }
    if required_cols is not None:
        base_required_cols = set(base_required_cols) | set(required_cols)
    _require_columns(
        method_selection_df,
        base_required_cols,
        context=context,
    )
    if method_selection_df.empty:
        return method_selection_df.copy()

    canonical_df = method_selection_df.copy()
    for column in ("dataset", "robustness_method", "pipeline_id", "model_architecture"):
        missing_mask = canonical_df[column].isna()
        if missing_mask.any():
            examples = _sample_records(
                canonical_df.loc[
                    missing_mask,
                    ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
                ],
                ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
            )
            raise ValueError(
                f"{context}: canonical method winner rows contain missing '{column}' values. "
                f"Examples: {examples}."
            )
        canonical_df[column] = canonical_df[column].astype(str).str.strip()
        empty_mask = canonical_df[column] == ""
        if empty_mask.any():
            examples = _sample_records(
                canonical_df.loc[
                    empty_mask,
                    ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
                ],
                ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
            )
            raise ValueError(
                f"{context}: canonical method winner rows contain empty '{column}' values. "
                f"Examples: {examples}."
            )

    if (canonical_df["pipeline_id"] == "baseline").any():
        examples = _sample_records(
            canonical_df.loc[
                canonical_df["pipeline_id"] == "baseline",
                ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
            ],
            ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
        )
        raise ValueError(
            f"{context}: canonical method winner rows include baseline pipeline_id values. "
            f"Examples: {examples}."
        )
    if (canonical_df["robustness_method"] == "baseline").any():
        examples = _sample_records(
            canonical_df.loc[
                canonical_df["robustness_method"] == "baseline",
                ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
            ],
            ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
        )
        raise ValueError(
            f"{context}: canonical method winner rows include baseline robustness_method values. "
            f"Examples: {examples}."
        )
    if (canonical_df["selection_pool"].astype(str) != "winner_pool").any():
        examples = _sample_records(
            canonical_df.loc[
                canonical_df["selection_pool"].astype(str) != "winner_pool",
                ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
            ],
            ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
        )
        raise ValueError(
            f"{context}: canonical method winner rows include non-winner-pool rows. "
            f"Examples: {examples}."
        )

    _assert_no_duplicates(
        canonical_df,
        list(METHOD_ANALYSIS_IDENTITY_COLS),
        context=(
            f"{context}: canonical method winner rows are not unique per "
            f"{list(METHOD_ANALYSIS_IDENTITY_COLS)}"
        ),
    )
    return canonical_df


def _canonical_method_delta_plot_df(
    improvement_deltas_selected_df: pd.DataFrame,
    *,
    context: str,
) -> pd.DataFrame:
    """Return canonical method delta rows used by method-facing diagnostics."""
    if improvement_deltas_selected_df.empty:
        return improvement_deltas_selected_df.copy()

    required_cols = set(METHOD_ANALYSIS_IDENTITY_COLS)
    _require_columns(
        improvement_deltas_selected_df,
        required_cols,
        context=context,
    )

    canonical_df = improvement_deltas_selected_df.copy()
    for column in ("dataset", "robustness_method", "pipeline_id", "model_architecture"):
        missing_mask = canonical_df[column].isna()
        if missing_mask.any():
            examples = _sample_records(
                canonical_df.loc[
                    missing_mask,
                    ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
                ],
                ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
            )
            raise ValueError(
                f"{context}: canonical method delta rows contain missing '{column}' values. "
                f"Examples: {examples}."
            )
        canonical_df[column] = canonical_df[column].astype(str).str.strip()
        empty_mask = canonical_df[column] == ""
        if empty_mask.any():
            examples = _sample_records(
                canonical_df.loc[
                    empty_mask,
                    ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
                ],
                ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
            )
            raise ValueError(
                f"{context}: canonical method delta rows contain empty '{column}' values. "
                f"Examples: {examples}."
            )

    if (canonical_df["pipeline_id"] == "baseline").any():
        examples = _sample_records(
            canonical_df.loc[
                canonical_df["pipeline_id"] == "baseline",
                ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
            ],
            ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
        )
        raise ValueError(
            f"{context}: canonical method delta rows include baseline pipeline_id values. "
            f"Examples: {examples}."
        )
    if (canonical_df["robustness_method"] == "baseline").any():
        examples = _sample_records(
            canonical_df.loc[
                canonical_df["robustness_method"] == "baseline",
                ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
            ],
            ["dataset", "robustness_method", "pipeline_id", "model_architecture"],
        )
        raise ValueError(
            f"{context}: canonical method delta rows include baseline robustness_method values. "
            f"Examples: {examples}."
        )

    _assert_no_duplicates(
        canonical_df,
        list(METHOD_ANALYSIS_IDENTITY_COLS),
        context=(
            f"{context}: canonical method delta rows are not unique per "
            f"{list(METHOD_ANALYSIS_IDENTITY_COLS)}"
        ),
    )
    return canonical_df


def _filter_rows_to_canonical_method_winners(
    df: pd.DataFrame,
    *,
    canonical_method_df: pd.DataFrame,
    context: str,
    drop_out_of_scope_methods: bool = False,
) -> pd.DataFrame:
    """Filter non-baseline rows to canonical method winner identities while keeping baseline rows.

    Plot builders may enable ``drop_out_of_scope_methods`` so rows for methods
    outside the active core-figure registry are trimmed before canonical winner
    enforcement. Other callers keep the strict default behavior.
    """
    required_cols = {
        "dataset",
        "data_config_signature",
        "robustness_method",
        "pipeline_id",
        "model_architecture",
    }
    _require_columns(df, required_cols, context=context)

    working_df = df.copy()
    for column in (
        "dataset",
        "data_config_signature",
        "robustness_method",
        "pipeline_id",
        "model_architecture",
    ):
        missing_mask = working_df[column].isna()
        if missing_mask.any():
            examples = _sample_records(
                working_df.loc[
                    missing_mask,
                    [
                        "dataset",
                        "data_config_signature",
                        "robustness_method",
                        "pipeline_id",
                        "model_architecture",
                    ],
                ],
                [
                    "dataset",
                    "data_config_signature",
                    "robustness_method",
                    "pipeline_id",
                    "model_architecture",
                ],
            )
            raise ValueError(
                f"{context}: required key column '{column}' has missing values. "
                f"Examples: {examples}."
            )
        working_df[column] = working_df[column].astype(str).str.strip()
        empty_mask = working_df[column] == ""
        if empty_mask.any():
            examples = _sample_records(
                working_df.loc[
                    empty_mask,
                    [
                        "dataset",
                        "data_config_signature",
                        "robustness_method",
                        "pipeline_id",
                        "model_architecture",
                    ],
                ],
                [
                    "dataset",
                    "data_config_signature",
                    "robustness_method",
                    "pipeline_id",
                    "model_architecture",
                ],
            )
            raise ValueError(
                f"{context}: required key column '{column}' has empty values. "
                f"Examples: {examples}."
            )

    baseline_id_mask = working_df["pipeline_id"] == "baseline"
    baseline_method_mask = working_df["robustness_method"] == "baseline"
    incoherent_baseline_mask = baseline_id_mask ^ baseline_method_mask
    if incoherent_baseline_mask.any():
        examples = _sample_records(
            working_df.loc[
                incoherent_baseline_mask,
                [
                    "dataset",
                    "data_config_signature",
                    "robustness_method",
                    "pipeline_id",
                    "model_architecture",
                ],
            ],
            [
                "dataset",
                "data_config_signature",
                "robustness_method",
                "pipeline_id",
                "model_architecture",
            ],
        )
        raise ValueError(
            f"{context}: baseline rows must satisfy both "
            "pipeline_id='baseline' and robustness_method='baseline'. "
            f"Examples: {examples}."
        )

    if drop_out_of_scope_methods and not working_df.empty:
        unsupported_mask = (
            working_df["robustness_method"] != "baseline"
        ) & ~working_df["robustness_method"].isin(_core_figure_supported_methods())
        if unsupported_mask.any():
            dropped_methods = sorted(
                working_df.loc[unsupported_mask, "robustness_method"]
                .astype(str)
                .str.strip()
                .drop_duplicates()
                .tolist()
            )
            print(
                f"{context}: skipping unsupported robustness methods outside "
                f"core-figure plot scope {dropped_methods}."
            )
            working_df = working_df.loc[~unsupported_mask].copy()
            if working_df.empty:
                raise ValueError(
                    f"{context}: all rows are outside core-figure plot scope after "
                    f"filtering unsupported robustness methods {dropped_methods}."
                )

    if working_df.empty:
        return working_df

    non_baseline_mask = working_df["pipeline_id"] != "baseline"
    if not non_baseline_mask.any():
        return working_df

    canonical_df = _canonical_method_winner_plot_df(
        canonical_method_df,
        context=f"{context}: canonical method winner source",
    )
    if canonical_df.empty:
        raise ValueError(
            f"{context}: canonical method winner source is empty while non-baseline rows are present."
        )
    key_cols = [
        "dataset",
        "data_config_signature",
        "robustness_method",
        "pipeline_id",
        "model_architecture",
    ]
    canonical_keys = canonical_df[key_cols].drop_duplicates().reset_index(drop=True)

    non_baseline_keys = (
        working_df.loc[non_baseline_mask, key_cols]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    missing_keys = non_baseline_keys.merge(
        canonical_keys,
        on=key_cols,
        how="left",
        indicator=True,
    )
    missing_keys = missing_keys.loc[missing_keys["_merge"] == "left_only"]
    if not missing_keys.empty:
        examples = (
            missing_keys[key_cols]
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"{context}: non-baseline rows are outside canonical method winner pool. "
            f"Examples: {examples}."
        )

    filtered_non_baseline = working_df.loc[non_baseline_mask].merge(
        canonical_keys,
        on=key_cols,
        how="inner",
    )
    baseline_df = working_df.loc[~non_baseline_mask].copy()
    filtered = pd.concat([baseline_df, filtered_non_baseline], ignore_index=True)
    if filtered.empty:
        raise ValueError(f"{context}: filtering produced an empty dataframe.")
    return filtered


def _assert_single_pipeline_per_method_backbone(
    df: pd.DataFrame,
    *,
    context: str,
    dataset_col: str = "dataset",
    method_col: str = "robustness_method",
    backbone_col: str = "backbone_architecture",
    pipeline_col: str = "pipeline_id",
) -> None:
    required_cols = {dataset_col, method_col, backbone_col, pipeline_col}
    _require_columns(df, required_cols, context=context)
    if df.empty:
        return

    working_df = df[[dataset_col, method_col, backbone_col, pipeline_col]].copy()
    for column in (dataset_col, method_col, backbone_col, pipeline_col):
        missing_mask = working_df[column].isna()
        if missing_mask.any():
            examples = _sample_records(
                working_df.loc[missing_mask, [dataset_col, method_col, backbone_col, pipeline_col]],
                [dataset_col, method_col, backbone_col, pipeline_col],
            )
            raise ValueError(
                f"{context}: required mapping column '{column}' has missing values. "
                f"Examples: {examples}."
            )
        working_df[column] = working_df[column].astype(str).str.strip()
        empty_mask = working_df[column] == ""
        if empty_mask.any():
            examples = _sample_records(
                working_df.loc[empty_mask, [dataset_col, method_col, backbone_col, pipeline_col]],
                [dataset_col, method_col, backbone_col, pipeline_col],
            )
            raise ValueError(
                f"{context}: required mapping column '{column}' has empty values. "
                f"Examples: {examples}."
            )

    pipeline_counts = working_df.groupby(
        [dataset_col, method_col, backbone_col], dropna=False
    )[pipeline_col].nunique()
    ambiguous = pipeline_counts[pipeline_counts != 1]
    if ambiguous.empty:
        return

    examples: list[dict[str, Any]] = []
    for dataset_name, method_name, backbone_name in ambiguous.index.tolist()[:5]:
        matching = working_df[
            (working_df[dataset_col] == dataset_name)
            & (working_df[method_col] == method_name)
            & (working_df[backbone_col] == backbone_name)
        ]
        pipeline_ids = sorted(matching[pipeline_col].drop_duplicates().tolist())
        examples.append(
            {
                str(dataset_col): str(dataset_name),
                str(method_col): str(method_name),
                str(backbone_col): str(backbone_name),
                "pipeline_ids": pipeline_ids,
            }
        )
    raise ValueError(
        f"{context}: each (dataset, robustness_method, backbone_architecture) must map "
        f"to exactly one pipeline_id. Examples: {examples}."
    )


def meta_analysis(
    args,
    *,
    coverage_fractions_by_dataset: Mapping[str, Mapping[tuple[str, str], tuple[int, int]]] | None = None,
):
    print("Running meta analysis.")
    tracking_uri = build_mlflow_tracking_uri(args.logdir)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    experiments = client.search_experiments()
    expected_experiment_prefix = f"{args.mlflow_experiment_prefix}-"
    experiments = [
        exp for exp in experiments if exp.name.startswith(expected_experiment_prefix)
    ]
    resolved_specs = resolve_with_defaults(
        datasets=args.data_files,
        targets=args.data_targets,
        data_root=args.data_root,
    )
    datasets = [spec.key for spec in resolved_specs]
    resolved_meta_eval_data_seed, eval_data_seed_label = (
        resolve_meta_analysis_eval_data_seed_scope(
            require_namespace_value(args, key="eval_data_seed"),
            key="args.eval_data_seed",
        )
    )
    coverage_source = "in_memory"
    skipped_best_model_runs: list[dict[str, str]] = []
    winner_runs_by_id: dict[str, Any] = {}
    if coverage_fractions_by_dataset is None:
        coverage_fractions_by_dataset = _recompute_coverage_fractions_by_dataset(
            args,
            resolved_specs=resolved_specs,
        )
        coverage_source = "recomputed"
    else:
        coverage_fractions_by_dataset = {
            str(dataset_name): dict(fractions)
            for dataset_name, fractions in coverage_fractions_by_dataset.items()
        }

    data = list()
    scenario_summary_records: list[dict[str, Any]] = []
    scenario_samples_records: list[dict[str, Any]] = []
    bootstrap_ci_provenance_values: set[tuple[str, int, float]] = set()

    expected_coupling = expected_perturbation_coupling_from_args(args)
    expected_scenarios = expected_coupling["perturbation_scenarios"]
    canonical_scenario_idx_by_name = {
        scenario_name: idx for idx, scenario_name in enumerate(expected_scenarios)
    }
    winner_pool_eval_context_values: set[tuple[tuple[str, str], ...]] = set()

    def _prefer_backbone_current(runs, scope):
        if not runs:
            return runs
        run_tags_by_id: dict[str, dict[str, Any]] = {}
        for run in runs:
            tags = run.data.tags
            if tags is None:
                raise ValueError(
                    f"Run {run.info.run_id} in {scope} is missing tags."
                )
            run_tags_by_id[run.info.run_id] = tags
        tags_present = any(
            "backbone_current" in run_tags_by_id[run.info.run_id] for run in runs
        )
        if not tags_present:
            return runs
        current = [
            run
            for run in runs
            if tag_is_truthy(
                run_tags_by_id[run.info.run_id],
                key="backbone_current",
            )
        ]
        if not current:
            raise ValueError(
                f"{scope} has backbone_current tags but no runs marked as current."
            )
        if len(current) < len(runs):
            print(
                f"Filtering {scope} to backbone_current runs: {len(current)}/{len(runs)}."
            )
        return current
    for experiment in experiments:
        dataset_name = _extract_dataset_name_from_experiment(
            experiment.name,
            prefix=args.mlflow_experiment_prefix,
        )
        if dataset_name not in datasets:
            continue
        best_runs = client.search_runs(
            [experiment.experiment_id],
            "tags.best_model = 'true'",
        )
        for run in best_runs:
            run_tags = run.data.tags
            if run_tags is None:
                raise ValueError(
                    f"Run {run.info.run_id} from best_model=true query is missing tags."
                )
            if run_tags.get("mlflow.parentRunId"):
                continue
            if run.info.status != "FINISHED":
                _raise_non_finished_winner(run)
            scope_model_architecture = parse_required_nonempty_string(
                run_tags.get("model_architecture"),
                key="model_architecture",
                context=f"Run {run.info.run_id}",
            )
            if _winner_pool_architecture_outside_testing_coverage(
                coverage_fractions_by_dataset=coverage_fractions_by_dataset,
                dataset_name=str(dataset_name),
                backbone_architecture=scope_model_architecture,
            ):
                skipped_best_model_runs.append(
                    {
                        "run_id": str(run.info.run_id),
                        "dataset": str(dataset_name),
                        "reason": "outside_active_testing_coverage_scope",
                    }
                )
                continue
            _require_best_model_current_tags(
                run_tags,
                run_id=run.info.run_id,
            )
            resolved = resolve_pipeline_tags(run_tags, run_id=run.info.run_id)
            pipeline_id = resolved["pipeline_id"]
            pipeline_method = resolved["pipeline_method"]
            pipeline_kind = resolved["pipeline_kind"]
            robustness_method = resolved["robustness_method"]
            model_architecture, loader_kind = _resolve_model_loading_identity_for_run(
                run_tags,
                run_id=run.info.run_id,
            )
            backbone_architecture = model_architecture
            if _winner_pool_cell_outside_testing_coverage(
                coverage_fractions_by_dataset=coverage_fractions_by_dataset,
                dataset_name=str(dataset_name),
                backbone_architecture=str(backbone_architecture),
                robustness_method=str(robustness_method),
            ):
                skipped_best_model_runs.append(
                    {
                        "run_id": str(run.info.run_id),
                        "dataset": str(dataset_name),
                        "reason": "outside_active_testing_coverage_scope",
                    }
                )
                continue
            dataset_tag = run_tags.get("dataset")
            if dataset_tag is None or not str(dataset_tag).strip():
                raise ValueError(
                    f"Run {run.info.run_id} is missing required dataset tag."
                )
            if str(dataset_tag).strip() != str(dataset_name):
                raise ValueError(
                    f"Run {run.info.run_id} has dataset tag '{dataset_tag}' but was queried under "
                    f"experiment dataset '{dataset_name}'."
                )
            data_signature = run_tags.get("data_config_signature")
            if data_signature is None or not str(data_signature).strip():
                raise ValueError(
                    f"Run {run.info.run_id} is missing required data_config_signature tag."
                )
            skip_reason = _skip_reason_for_best_model_query_run(
                run,
                args=args,
                client=client,
            )
            if skip_reason is not None:
                skipped_best_model_runs.append(
                    {
                        "run_id": str(run.info.run_id),
                        "dataset": str(dataset_name),
                        "reason": str(skip_reason),
                    }
                )
                continue
            require_matching_perturbation_coupling_params(
                run,
                args=args,
                context="from best_model=true query",
            )
            run_idx_name_map = require_run_perturbation_idx_name_map(
                tags=run_tags,
                run_id=run.info.run_id,
                expected_scenarios=expected_scenarios,
            )
            stage = require_stage_tag(run_tags, run_id=run.info.run_id)

            robustness_metrics, robustness_context = _extract_run_robustness_metrics(
                run,
                expected_test_metric=args.test_metric,
                return_context=True,
            )
            eval_context = robustness_context["eval_context"]
            if (
                robustness_context["robustness_scoring_semantics"]
                != DEGRADATION_SCORING_SEMANTICS
            ):
                raise ValueError(
                    "Meta-analysis canonical winner pool only supports "
                    f"{DEGRADATION_SCORING_SEMANTICS!r} runs. "
                    f"Run {run.info.run_id} uses "
                    f"{robustness_context['robustness_scoring_semantics']!r}."
                )
            winner_pool_eval_context_values.add(
                _build_meta_analysis_eval_context_provenance_key(
                    eval_context,
                    run_id=run.info.run_id,
                    include_eval_data_seed=resolved_meta_eval_data_seed is not None,
                )
            )
            bootstrap_ci_context = robustness_context["bootstrap_ci_context"]
            bootstrap_ci_provenance_values.add(
                (
                    str(bootstrap_ci_context["bootstrap_ci_semantics"]),
                    int(bootstrap_ci_context["bootstrap_ci_resamples"]),
                    float(bootstrap_ci_context["bootstrap_ci_confidence_level"]),
                )
            )
            run_test_metric = robustness_context["test_metric"]
            _require_winner_selection_provenance_for_meta_analysis_run(
                run,
                args=args,
                test_metric=run_test_metric,
            )
            try:
                _ = resolve_selection_score(
                    metrics=run.data.metrics,
                    pipeline_kind=pipeline_kind,
                    robustness_method=robustness_method,
                    test_metric=run_test_metric,
                    improvement_selection_mode=require_improvement_selection_mode(
                        args,
                        context="meta-analysis args",
                    ),
                    run_id=run.info.run_id,
                )
            except ValueError as exc:
                raise ValueError(f"{exc} in winner pool.") from exc
            best_val_loss = run.data.metrics.get("best_val_loss")
            val_metric = run.data.metrics.get(f"{run_test_metric}_val")
            pert_ws_val_metric = run.data.metrics.get(f"{run_test_metric}_pert_ws_val")
            pert_mean_val_metric = run.data.metrics.get(
                f"{run_test_metric}_pert_mean_val"
            )
            test_metric = run.data.metrics.get(f"{run_test_metric}_test", None)
            at_worst_scenario_metric_key = robustness_context[
                "at_worst_scenario_metric_name"
            ]
            at_worst_scenario_value = robustness_metrics["at_worst_scenario"]
            base_key = robustness_context["base_key"]
            worst_scenario = _require_logged_worst_scenario_param(
                run,
                base_key=base_key,
            )
            model_display = loader_kind if pipeline_kind == "wrap" else model_architecture
            run_n_test_samples = int(eval_context["n_test_samples"])
            if run_n_test_samples <= 0:
                raise ValueError(
                    f"Run {run.info.run_id} has non-positive n_test_samples={run_n_test_samples}."
                )
            scenario_indices = sorted(int(idx) for idx in run_idx_name_map.keys())
            if not scenario_indices:
                raise ValueError(
                    f"Run {run.info.run_id} perturbation_idx_name_map has no scenario indices."
                )
            scenario_metrics_by_idx = {
                int(scenario_idx): extract_required_degradation_scenario_metrics(
                    run.data.metrics,
                    run_id=run.info.run_id,
                    test_metric=run_test_metric,
                    scenario_idx=int(scenario_idx),
                )
                for scenario_idx in scenario_indices
            }
            for scenario_idx in scenario_indices:
                metric_bundle = scenario_metrics_by_idx[scenario_idx]
                scenario_name = run_idx_name_map[int(scenario_idx)]
                canonical_scenario_idx = canonical_scenario_idx_by_name.get(scenario_name)
                if canonical_scenario_idx is None:
                    raise ValueError(
                        f"Run {run.info.run_id} scenario '{scenario_name}' is not in configured "
                        "perturbation_scenarios."
                    )
                scenario_record = {
                    "dataset": dataset_name,
                    "data_config_signature": data_signature,
                    "run_id": run.info.run_id,
                    "robustness_method": robustness_method,
                    "pipeline_method": pipeline_method,
                    "pipeline_id": pipeline_id,
                    "model_architecture": model_architecture,
                    "model": model_display,
                    "scenario_idx": int(canonical_scenario_idx),
                    "scenario": scenario_name,
                    "D": float(metric_bundle["D"]),
                    "D_CI_lo": float(metric_bundle["D_CI_lo"]),
                    "D_CI_hi": float(metric_bundle["D_CI_hi"]),
                    "err_pert": float(metric_bundle["err_pert"]),
                    "err_pert_CI_lo": float(metric_bundle["err_pert_CI_lo"]),
                    "err_pert_CI_hi": float(metric_bundle["err_pert_CI_hi"]),
                }
                scenario_summary_records.append(scenario_record)

            clean_error_value = require_float_metric(
                run.data.metrics,
                run_id=run.info.run_id,
                metric_key=f"{run_test_metric}_test",
                output_name=f"{run_test_metric}_test",
            )
            if clean_error_value <= 0.0:
                raise ValueError(
                    f"Run {run.info.run_id} has non-positive clean error "
                    f"{run_test_metric}_test={clean_error_value!r}, which is invalid for "
                    "canonical degradation analysis."
                )
            clean_df, scenario_samples_df, _ = _load_required_degradation_artifact_bundle(
                client=client,
                run_id=run.info.run_id,
                test_metric=run_test_metric,
                eval_data_seed=int(eval_context["eval_data_seed"]),
                expected_idx_to_name={
                    idx: scenario_name for idx, scenario_name in enumerate(expected_scenarios)
                },
                expected_n_test_samples=int(eval_context["n_test_samples"]),
                expected_clean_metric_value=clean_error_value,
            )
            scenario_samples_with_clean_df = _attach_clean_sample_errors_to_scenario_samples(
                scenario_samples_df,
                clean_df=clean_df,
                context=f"Run {run.info.run_id} meta-analysis scenario samples",
            )
            clean_error_value = float(clean_df["err_clean"].mean())
            for _, sample_row in scenario_samples_with_clean_df.iterrows():
                scenario_samples_records.append(
                    {
                        "dataset": dataset_name,
                        "data_config_signature": data_signature,
                        "stage": stage,
                        "robustness_method": robustness_method,
                        "pipeline_method": pipeline_method,
                        "pipeline_kind": pipeline_kind,
                        "pipeline_id": pipeline_id,
                        "run_id": run.info.run_id,
                        "model_architecture": model_architecture,
                        "backbone_architecture": backbone_architecture,
                        "model": model_display,
                        "sample_id": int(sample_row["sample_id"]),
                        "source_sample_idx": int(sample_row["source_sample_idx"]),
                        "pert_idx": int(sample_row["pert_idx"]),
                        "scenario": str(sample_row["scenario"]),
                        "severity": float(sample_row["severity"]),
                        "err_clean": float(sample_row["err_clean"]),
                        "err_pert": float(sample_row["err_pert"]),
                        "err_clean_global": float(clean_error_value),
                        "D": float(sample_row["err_pert"]) / float(clean_error_value),
                    }
                )

            data.append({
                "dataset": dataset_name,
                "data_config_signature": data_signature,
                "run_id": run.info.run_id,
                "model": model_display,
                "loader_kind": loader_kind,
                "model_architecture": model_architecture,
                "pipeline_id": pipeline_id,
                "pipeline_method": pipeline_method,
                "pipeline_kind": pipeline_kind,
                "stage": stage,
                "robustness_method": robustness_method,
                "base_pipeline_method": parse_optional_nonempty_string(
                    run_tags.get("base_pipeline_method"),
                    key="base_pipeline_method",
                    context=f"Meta-analysis run {run.info.run_id}",
                    disallow_none_token=True,
                ),
                "backbone_run_id": parse_optional_nonempty_string(
                    run_tags.get("backbone_run_id"),
                    key="backbone_run_id",
                    context=f"Meta-analysis run {run.info.run_id}",
                    disallow_none_token=True,
                ),
                "backbone_run_ids": parse_optional_nonempty_string(
                    run_tags.get("backbone_run_ids"),
                    key="backbone_run_ids",
                    context=f"Meta-analysis run {run.info.run_id}",
                    disallow_none_token=True,
                ),
                "backbone_architecture": backbone_architecture,
                "eval_data_seed": str(
                    require_eval_data_seed_tag(
                        run_tags,
                        run_id=run.info.run_id,
                    )
                ),
                "test_metric": run_test_metric,
                f"{run_test_metric}_val": val_metric,
                f"{run_test_metric}_pert_ws_val": pert_ws_val_metric,
                f"{run_test_metric}_pert_mean_val": pert_mean_val_metric,
                f"{run_test_metric}_test": test_metric,
                "best_val_loss": best_val_loss,
                "robustness_scoring_semantics": eval_context[
                    "robustness_scoring_semantics"
                ],
                "perturbation_channel_fraction_max": float(
                    eval_context["perturbation_channel_fraction_max"]
                ),
                "perturbation_scenarios_signature": str(
                    eval_context["perturbation_scenarios_signature"]
                ),
                "perturbation_scenarios_count": int(
                    eval_context["perturbation_scenarios_count"]
                ),
                **{
                    key: value
                    for key, value in robustness_metrics.items()
                    if key != "at_worst_scenario"
                },
                at_worst_scenario_metric_key: at_worst_scenario_value,
                "worst_scenario": worst_scenario,
                "n_test_samples": run_n_test_samples,
                "selection_pool": "winner_pool",
            })
            winner_runs_by_id[str(run.info.run_id)] = run

    if skipped_best_model_runs:
        warnings.warn(
            "Meta-analysis excluded best_model=true parent runs that are not current "
            "tested winners for the requested evaluation context. "
            f"Skipped={len(skipped_best_model_runs)}. "
            f"Examples: {skipped_best_model_runs[:5]}.",
            stacklevel=2,
        )

    if not data:
        _require_winner_pool_coverage_matches_testing_coverage(
            pd.DataFrame(
                columns=[
                    "dataset",
                    "backbone_architecture",
                    "robustness_method",
                    "selection_pool",
                ]
            ),
            coverage_fractions_by_dataset,
            full_coverage=require_namespace_bool(args, key="full_coverage"),
        )
        print("No data found for meta-analysis. Exiting.")
        return

    result_df = pd.DataFrame(data)
    if not bootstrap_ci_provenance_values:
        raise ValueError(
            "Meta-analysis winner pool is missing bootstrap-CI provenance."
        )
    if len(bootstrap_ci_provenance_values) != 1:
        raise ValueError(
            "Meta-analysis winner pool contains inconsistent bootstrap-CI provenance: "
            f"{sorted(bootstrap_ci_provenance_values)}."
        )
    (
        meta_bootstrap_ci_semantics,
        meta_bootstrap_ci_resamples,
        meta_bootstrap_ci_confidence_level,
    ) = next(iter(bootstrap_ci_provenance_values))
    if not winner_pool_eval_context_values:
        raise ValueError(
            "Meta-analysis winner pool is missing degradation evaluation context provenance."
        )
    if len(winner_pool_eval_context_values) != 1:
        raise ValueError(
            "Meta-analysis winner pool contains inconsistent degradation evaluation contexts: "
            f"{sorted(winner_pool_eval_context_values)}."
        )
    meta_eval_context_tag_payload = dict(next(iter(winner_pool_eval_context_values)))
    scenario_summary_columns = [
        "dataset",
        "data_config_signature",
        "run_id",
        "robustness_method",
        "pipeline_method",
        "pipeline_id",
        "model_architecture",
        "model",
        "scenario_idx",
        "scenario",
        "D",
        "D_CI_lo",
        "D_CI_hi",
        "err_pert",
        "err_pert_CI_lo",
        "err_pert_CI_hi",
    ]
    scenario_summary_df = pd.DataFrame(
        scenario_summary_records,
        columns=scenario_summary_columns,
    )
    scenario_samples_columns = [
        "dataset",
        "data_config_signature",
        "stage",
        "robustness_method",
        "pipeline_method",
        "pipeline_kind",
        "pipeline_id",
        "run_id",
        "model_architecture",
        "backbone_architecture",
        "model",
        "sample_id",
        "source_sample_idx",
        "pert_idx",
        "scenario",
        "severity",
        "err_clean",
        "err_pert",
        "err_clean_global",
        "D",
    ]
    scenario_samples_df = pd.DataFrame(
        scenario_samples_records,
        columns=scenario_samples_columns,
    )
    if scenario_summary_df.empty:
        raise ValueError(
            "Meta-analysis scenario_summary_df is empty. "
            "Expected scenario metrics for tested winner-pool runs."
        )
    if scenario_samples_df.empty:
        raise ValueError(
            "Meta-analysis scenario_samples_df is empty. "
            "Expected canonical scenario_samples artifacts for tested winner-pool runs."
        )
    fixed_channel_fraction_df = (
        _build_fixed_channel_fraction_table(
            result_df=result_df,
            winner_runs_by_id=winner_runs_by_id,
            client=client,
            args=args,
        )
    )
    fixed_channel_fraction_paper_summary_df = (
        _build_fixed_channel_fraction_paper_summary_table(
            fixed_channel_fraction_df,
            full_coverage=require_namespace_bool(args, key="full_coverage"),
        )
    )
    severity_profile_df = _build_binned_degradation_profile_df(scenario_samples_df)
    reference_normalization_anchor_model = parse_reference_normalization_anchor_model(
        require_namespace_value(args, key="reference_normalization_anchor_model"),
        key="reference_normalization_anchor_model",
        context="args",
    )
    (
        result_df,
        scenario_summary_df,
        reference_normalization_anchors_df,
    ) = _attach_reference_normalized_diagnostics(
        result_df,
        scenario_summary_df,
        severity_profile_df,
        reference_normalization_anchor_model=reference_normalization_anchor_model,
        test_metric=args.test_metric,
        eps=float(require_namespace_value(args, key="eps")),
    )
    rho_eff_attachment = _attach_rho_eff(result_df, test_metric=args.test_metric)
    result_df = rho_eff_attachment.result_df
    rho_eff_fits_df = rho_eff_attachment.fit_summary_df

    required_pipeline_cols = ["pipeline_id", "pipeline_method", "pipeline_kind"]
    missing_pipeline_cols = [col for col in required_pipeline_cols if col not in result_df.columns]
    if missing_pipeline_cols:
        raise ValueError(
            "Meta-analysis results missing required pipeline columns: "
            f"{missing_pipeline_cols}. Ensure runs log pipeline tags."
        )
    missing_pipeline_values = {
        col: int(result_df[col].isna().sum())
        for col in required_pipeline_cols
        if result_df[col].isna().any()
    }
    if missing_pipeline_values:
        raise ValueError(
            f"Meta-analysis results contain missing pipeline tags: {missing_pipeline_values}."
        )

    if "data_config_signature" not in result_df.columns:
        raise ValueError(
            "Meta-analysis results missing required data_config_signature column."
        )
    missing_signature = result_df["data_config_signature"].isna()
    if missing_signature.any():
        examples = (
            result_df.loc[missing_signature, ["dataset", "pipeline_id"]]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Meta-analysis results contain missing data_config_signature values. "
            f"Examples: {examples}."
        )
    signature_counts = result_df.groupby("dataset", dropna=False)[
        "data_config_signature"
    ].nunique()
    bad_signature_counts = signature_counts[signature_counts != 1]
    if not bad_signature_counts.empty:
        preview = ", ".join(
            f"{dataset}={int(count)}"
            for dataset, count in bad_signature_counts.sort_index().items()
        )
        raise ValueError(
            "Meta-analysis detected multiple data_config_signature values within dataset(s): "
            f"{preview}. This violates comparability constraints."
        )

    if "selection_pool" not in result_df.columns:
        raise ValueError(
            "Meta-analysis results missing selection_pool marker. "
            "Expected winner-pool provenance for all rows."
        )
    bad_selection_pool = result_df["selection_pool"].astype(str) != "winner_pool"
    if bad_selection_pool.any():
        bad_runs = (
            result_df.loc[bad_selection_pool, ["dataset", "pipeline_id"]]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Meta-analysis results contain rows not sourced from the tested winner pool. "
            f"Examples: {bad_runs}."
        )
    _require_winner_pool_coverage_matches_testing_coverage(
        result_df,
        coverage_fractions_by_dataset,
        full_coverage=require_namespace_bool(args, key="full_coverage"),
    )

    active_method_scope = {
        str(method_key).strip()
        for method_key in _core_figure_registry().method_order
    }
    method_selection_df = _build_canonical_method_analysis_df(
        result_df,
        test_metric=args.test_metric,
        improvement_selection_mode=require_improvement_selection_mode(
            args,
            context="meta-analysis args",
        ),
        allowed_methods=active_method_scope,
    )

    # Method-level and variant-level diagnostics from non-baseline winner-pool rows.
    pipeline_method_candidates_df = pd.DataFrame()
    variant_selection_summary_df = pd.DataFrame()
    method_variant_breakdown_df = pd.DataFrame()
    method_aggregates_df = pd.DataFrame()
    if not method_selection_df.empty:
        variant_selection_summary_df = _build_variant_selection_summary(method_selection_df)
        method_variant_breakdown_df = variant_selection_summary_df.copy()

        method_aggregates_df = (
            method_selection_df.groupby(
                ["dataset", "robustness_method", "selection_metric"],
                dropna=False,
            )
            .agg(
                selection_score=("selection_value", "mean"),
                architectures_covered=("model_architecture", "nunique"),
                variant_count=("pipeline_id", "nunique"),
                run_count=("pipeline_id", "size"),
            )
            .reset_index()
            .rename(columns={"selection_metric": "selection_metric_name"})
        )
        method_aggregates_df = _sort_and_rank_method_aggregates(method_aggregates_df)

        pipeline_method_candidates_df = _build_pipeline_method_candidates(
            variant_selection_summary_df
        )

    canonical_method_plot_df = _canonical_method_winner_plot_df(
        method_selection_df,
        context="Cannot build canonical method winner plot source",
    )

    critical_cols = [
        f"{args.test_metric}_test",
        build_metric_w_name(
            args.test_metric,
            key="test_metric",
            context="meta-analysis",
        ),
        *[
            column
            for column in ("D_w", "D_mean", "err_pert_ws", "err_pert_mean")
            if column in result_df.columns
        ],
    ]
    missing_critical = result_df[critical_cols].isna().any(axis=1)
    if missing_critical.any():
        missing_count = int(missing_critical.sum())
        examples = (
            result_df.loc[missing_critical, ["dataset", "robustness_method", "pipeline_id", "model_architecture"]]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"Meta-analysis found {missing_count} rows missing critical metrics {critical_cols}. "
            f"Examples: {examples}."
        )

    _report_result_nan_columns(result_df, args=args)

    # Split by pipeline_id: baseline vs robustness improvement methods
    pipeline_id_series = result_df["pipeline_id"].astype(str)
    baseline_mask = pipeline_id_series == "baseline"
    non_baseline_mask = ~baseline_mask

    backbone_df = result_df[baseline_mask].copy()
    non_baseline_eval_df = result_df[non_baseline_mask].copy()

    # Unique pipeline_ids for reporting
    unique_pipeline_ids = sorted(pipeline_id_series.unique())
    if "robustness_method" not in result_df.columns:
        raise ValueError("result_df is missing 'robustness_method' column — check upstream data processing")
    unique_pipeline_families = sorted(result_df["robustness_method"].dropna().unique())

    n_archs_meta = result_df["model_architecture"].dropna().nunique()
    n_methods_meta = len(unique_pipeline_families)
    print(
        f"\nMeta-analysis pool: {len(backbone_df)} baseline + "
        f"{len(non_baseline_eval_df)} non-baseline runs "
        f"({n_archs_meta} architectures, {n_methods_meta} methods)"
    )

    unique_models = [str(model) for model in sorted(backbone_df["model"].unique())] if not backbone_df.empty else []
    unique_datasets = [str(dataset) for dataset in sorted(backbone_df["dataset"].unique())] if not backbone_df.empty else []
    robustness_metric_columns = [
        metric_key
        for metric_key in ("D_w", "D_mean", "err_pert_ws", "err_pert_mean")
        if metric_key in result_df.columns
    ]
    at_worst_scenario_metric_key = build_metric_w_name(
        args.test_metric,
        key="test_metric",
        context="meta-analysis",
    )

    numeric_columns = result_df.select_dtypes(include=["number"]).columns.tolist()
    backbone_numeric_columns = backbone_df.select_dtypes(include=["number"]).columns.tolist()

    if not backbone_df.empty and backbone_numeric_columns:
        model_results_df = _aggregate_numeric_summary_table(
            backbone_df,
            group_cols=["model"],
            numeric_cols=backbone_numeric_columns,
            context="Model summary table",
            at_worst_scenario_metric_key=at_worst_scenario_metric_key,
            reset_index=False,
        )

        data_results_df = _aggregate_numeric_summary_table(
            backbone_df,
            group_cols=["dataset"],
            numeric_cols=backbone_numeric_columns,
            context="Dataset summary table",
            at_worst_scenario_metric_key=at_worst_scenario_metric_key,
            reset_index=False,
        )
    else:
        model_results_df = pd.DataFrame()
        data_results_df = pd.DataFrame()

    # Variant grouping by pipeline_id
    variant_group = ["model_architecture", "pipeline_id", "robustness_method"]

    # Variant results per dataset (canonical table for ranking/selection)
    variant_by_dataset_results_df = pd.DataFrame()
    if "dataset" in result_df.columns and numeric_columns:
        variant_by_dataset_group = ["dataset"] + variant_group
        variant_by_dataset_results_df = _aggregate_numeric_summary_table(
            result_df,
            group_cols=variant_by_dataset_group,
            numeric_cols=numeric_columns,
            context="Variant-by-dataset summary table",
            at_worst_scenario_metric_key=at_worst_scenario_metric_key,
            reset_index=True,
        )

    # Pipeline counts
    pipeline_counts_df = pipeline_id_series.value_counts().rename_axis("pipeline_id").reset_index(name="count")
    pipeline_counts_dict = dict(zip(pipeline_counts_df["pipeline_id"], pipeline_counts_df["count"]))

    # Pipeline family results (aggregated per dataset x robustness_method) from
    # canonical non-baseline winner-pool rows only.
    pipeline_method_results_df = pd.DataFrame()
    if not method_selection_df.empty:
        non_comparable_selection_cols = {
            "best_val_loss",
            f"{args.test_metric}_val",
            "selection_value",
        }
        method_numeric_cols = [
            column
            for column in method_selection_df.select_dtypes(include=["number"]).columns.tolist()
            if column not in non_comparable_selection_cols
        ]
        if method_numeric_cols and "robustness_method" in method_selection_df.columns:
            pipeline_method_group = ["dataset", "robustness_method"]
            pipeline_method_results_df = _aggregate_numeric_summary_table(
                method_selection_df,
                group_cols=pipeline_method_group,
                numeric_cols=method_numeric_cols,
                context="Pipeline-method summary table",
                at_worst_scenario_metric_key=at_worst_scenario_metric_key,
                reset_index=True,
            )

    canonical_frames = CanonicalAnalysisFrames(
        backbone_df=backbone_df,
        non_baseline_eval_df=non_baseline_eval_df,
        method_selection_df=method_selection_df,
        pipeline_method_candidates_df=pipeline_method_candidates_df,
        variant_selection_summary_df=variant_selection_summary_df,
        method_variant_breakdown_df=method_variant_breakdown_df,
        method_aggregates_df=method_aggregates_df,
        pipeline_method_results_df=pipeline_method_results_df,
    )

    # Compute correlation between val->test drop and robustness score for backbones
    analysis_df = backbone_df.copy()
    if not analysis_df.empty:
        analysis_df["val_to_test_drop_abs"] = analysis_df[f"{args.test_metric}_val"] - analysis_df[f"{args.test_metric}_test"]
        analysis_df["val_to_test_drop"] = analysis_df["val_to_test_drop_abs"] / analysis_df[f"{args.test_metric}_val"]
    
    def compute_corr(data, metric_name):
        val_to_test = require_numeric_series(
            data["val_to_test_drop"],
            column_name="val_to_test_drop",
            context="Correlation input",
            allow_nan=True,
            allow_infinite=False,
        )
        metric_series = require_numeric_series(
            data[metric_name],
            column_name=metric_name,
            context="Correlation input",
            allow_nan=True,
            allow_infinite=False,
        )
        valid = val_to_test.notna() & metric_series.notna()
        count = int(valid.sum())
        if count > 2:
            correlation, p_value = pearsonr(
                val_to_test[valid],
                metric_series[valid],
            )
        else:
            correlation, p_value = None, None
        return pd.Series({
            "count": count,
            "correlation": correlation,
            "p_value": p_value
        })

    # Correlation analysis for maintained canonical metrics
    core_robustness_metric_keys = _core_robustness_metric_keys()
    run_level_core_metric_keys = [
        metric_key
        for metric_key in core_robustness_metric_keys
        if metric_key in analysis_df.columns
    ]
    corr_metric_order = list(dict.fromkeys(run_level_core_metric_keys))
    corr_metrics = {
        metric: robustness_metric_display_name(metric)
        for metric in corr_metric_order
    }
    correlation_metric_titles = {
        metric_key: robustness_metric_display_name(metric_key)
        for metric_key in corr_metric_order
    }
    correlation_metric_order = list(correlation_metric_titles.keys())

    model_corr_summary = _build_correlation_summary(
        analysis_df,
        group_col="model",
        metric_keys=correlation_metric_order,
        compute_corr_fn=compute_corr,
    )
    dataset_corr_summary = _build_correlation_summary(
        analysis_df,
        group_col="dataset",
        metric_keys=correlation_metric_order,
        compute_corr_fn=compute_corr,
    )
    model_corr_heatmaps_by_metric = _build_correlation_heatmaps_by_metric(
        model_corr_summary,
        index_col="model",
        entity_label="Model",
        metric_titles=correlation_metric_titles,
    )
    dataset_corr_heatmaps_by_metric = _build_correlation_heatmaps_by_metric(
        dataset_corr_summary,
        index_col="dataset",
        entity_label="Dataset",
        metric_titles=correlation_metric_titles,
    )

    # --- 1. Correlation Analysis of Robustness Metrics ---
    if not analysis_df.empty and corr_metric_order:
        robustness_metrics_df = analysis_df[corr_metric_order]
        robustness_corr_matrix = robustness_metrics_df.corr()
        robustness_corr_heatmap = plot_heatmap(
            robustness_corr_matrix,
            "Robustness Metric Correlation",
            color_scale="RdYlGn",
        )
    else:
        robustness_corr_heatmap = None

    meta_input_rows = MetaAnalysisInputRows(
        result_df=result_df,
        scenario_summary_df=scenario_summary_df,
        scenario_samples_df=scenario_samples_df,
        severity_profile_df=severity_profile_df,
    )

    arch_map = {
        "DLinear": "Fully Connected",
        "ModernTCN": "Convolution",
        "GRU": "Recurrent",
        "PatchTST": "Attention",
        "TSMixer": "Fully Connected",
        "SeasonalNaive": "Statistical",
        "Chronos2": "Foundation",
    }

    # Add architecture_family to all dataframes that need it
    backbone_df = _assign_architecture_families(
        backbone_df,
        arch_map=arch_map,
        context="Meta-analysis baseline",
    )
    method_selection_df = _assign_architecture_families(
        method_selection_df,
        arch_map=arch_map,
        context="Meta-analysis method",
    )
    core_figures_require_full_coverage = require_namespace_bool(
        args,
        key="full_coverage",
    )
    if not core_figures_require_full_coverage:
        print(
            "Running core figure generation with full_coverage=false. "
            "Recognized partial-coverage plotting failures will be logged and suppressed."
        )
    baseline_core_figure_specs = _build_plot_artifact_with_partial_coverage_tolerance(
        lambda: _build_core_baseline_figure_specs(
            backbone_df,
            args=args,
            full_coverage=core_figures_require_full_coverage,
        ),
        runtime_args=args,
        context="Baseline architecture figure",
        empty_value=[],
        suppress_markers=(
            "no core-figure baseline rows remain after filtering",
            "no renderable core-figure dataset panels remain",
        ),
    )
    core_scenario_heatmap_specs: list[FigureArtifactSpec] = []
    baseline_clean_vs_worst_panel_specs: list[FigureArtifactSpec] = []

    dataset_pareto_plots = {}
    penmanshiel_no_seasonal_pareto_plots = {}
    perf_col = f"{args.test_metric}_test"
    if perf_col in analysis_df.columns:
        dataset_level_df = _assign_architecture_families(
            analysis_df,
            arch_map=arch_map,
            context="Dataset-level Pareto",
        )
        for metric, metric_title in corr_metrics.items():
            if metric not in dataset_level_df.columns:
                continue
            per_dataset = {}
            per_dataset_no_seasonal = {}
            for dataset_name, dataset_df in dataset_level_df.groupby("dataset"):
                if dataset_df[[perf_col, metric]].isnull().all().any():
                    continue
                full_plot = plot_pareto(
                    dataset_df,
                    perf_col=perf_col,
                    robust_col=metric,
                    x_semantics=_require_plot_semantics_for_keys(
                        test_metric=args.test_metric,
                        required_keys=[perf_col],
                        context="Dataset-level Pareto",
                    )[perf_col],
                    y_semantics=_require_plot_semantics_for_keys(
                        test_metric=args.test_metric,
                        required_keys=[metric],
                        context="Dataset-level Pareto",
                    )[metric],
                    model_col="model",
                    arch_col="architecture_family",
                    perf_lower_is_better=True,
                    flip_perf_axis=False,
                    x_include_zero=False,
                    title=f"{dataset_name}: {args.test_metric} vs {metric_title}",
                    x_title=f"{args.test_metric} (Test)",
                    y_title=f"{metric_title}",
                )
                per_dataset[dataset_name] = full_plot

                dataset_name_normalized = str(dataset_name).strip().casefold()
                if dataset_name_normalized in {"penmanshiel", "penmanshiel_wt"}:
                    if "model_architecture" not in dataset_df.columns:
                        print(
                            "Skipping SeasonalNaive-excluded Pareto plot for "
                            f"{dataset_name}: missing 'model_architecture' column."
                        )
                        continue
                    model_architecture = dataset_df["model_architecture"].astype(str).str.strip()
                    seasonal_naive_mask = model_architecture == "SeasonalNaive"
                    if not seasonal_naive_mask.any():
                        print(
                            "Skipping SeasonalNaive-excluded Pareto plot for "
                            f"{dataset_name}: no SeasonalNaive rows found."
                        )
                        continue
                    filtered_dataset_df = dataset_df.loc[~seasonal_naive_mask].copy()
                    if filtered_dataset_df.empty:
                        print(
                            "Skipping SeasonalNaive-excluded Pareto plot for "
                            f"{dataset_name}: all rows are SeasonalNaive."
                        )
                        continue
                    per_dataset_no_seasonal[dataset_name] = plot_pareto(
                        filtered_dataset_df,
                        perf_col=perf_col,
                        robust_col=metric,
                        x_semantics=_require_plot_semantics_for_keys(
                            test_metric=args.test_metric,
                            required_keys=[perf_col],
                            context="Dataset-level Pareto without SeasonalNaive",
                        )[perf_col],
                        y_semantics=_require_plot_semantics_for_keys(
                            test_metric=args.test_metric,
                            required_keys=[metric],
                            context="Dataset-level Pareto without SeasonalNaive",
                        )[metric],
                        model_col="model",
                        arch_col="architecture_family",
                        perf_lower_is_better=True,
                        flip_perf_axis=False,
                        x_include_zero=False,
                        title=(
                            f"{dataset_name}: {args.test_metric} vs {metric_title} "
                            "(without SeasonalNaive)"
                        ),
                        x_title=f"{args.test_metric} (Test)",
                        y_title=f"{metric_title}",
                    )
            if per_dataset:
                dataset_pareto_plots[metric] = per_dataset
            if per_dataset_no_seasonal:
                penmanshiel_no_seasonal_pareto_plots[metric] = per_dataset_no_seasonal

    dataset_clean_vs_pert_plot_entries: list[dict[str, Any]] = (
        _build_baseline_clean_vs_pert_plot_entries(
            analysis_df,
            test_metric=args.test_metric,
            arch_map=arch_map,
        )
    )
    baseline_clean_vs_worst_panel_specs = _build_plot_artifact_with_partial_coverage_tolerance(
        lambda: _build_baseline_clean_vs_worst_panel_figure_specs(
            analysis_df,
            args=args,
            full_coverage=core_figures_require_full_coverage,
        ),
        runtime_args=args,
        context="Baseline clean-vs-worst-error panels",
        empty_value=[],
        suppress_markers=(
            "missing required core-figure datasets",
            "no renderable clean-vs-worst dataset panels remain",
        ),
    )

    (
        error_distribution_summary_df,
        error_distribution_figure_specs,
    ) = _build_error_distribution_figure_specs(
        meta_input_rows.scenario_samples_df,
        canonical_method_df=canonical_method_plot_df,
    )

    # --- Improvement Analysis ---
    improvement_deltas_df = pd.DataFrame()
    improvement_deltas_long_df = pd.DataFrame()
    improvement_deltas_selected_df = pd.DataFrame()
    method_delta_plot_df = pd.DataFrame()
    pipeline_method_delta_results_df = pd.DataFrame(
        columns=list(PIPELINE_METHOD_DELTA_RESULTS_COLUMNS)
    )
    improvement_delta_overview_figure_specs: list[FigureArtifactSpec] = []
    core_metric_deltas_long_df = pd.DataFrame()
    method_scenario_family_delta_df = pd.DataFrame(
        columns=list(METHOD_SCENARIO_FAMILY_DELTA_COLUMNS)
    )
    method_scenario_family_summary_df = pd.DataFrame(
        columns=list(METHOD_SCENARIO_FAMILY_SUMMARY_COLUMNS)
    )
    method_win_rate_dfs: dict[str, pd.DataFrame] = {
        spec.metric_key: pd.DataFrame()
        for spec in CORE_ROBUSTNESS_METRIC_SPECS
    }
    selection_margin_df = pd.DataFrame()
    scenario_metric_specs = [
        {
            "metric_key": "D",
            "scenario_col": "D",
            "delta_col": _core_delta_column("D"),
            "title_suffix": "Scenario Degradation Delta vs Baseline",
        },
        {
            "metric_key": "err_pert",
            "scenario_col": "err_pert",
            "delta_col": _core_delta_column("err_pert"),
            "title_suffix": "Scenario Perturbed Error Delta vs Baseline",
        },
    ]
    scenario_metric_delta_dfs: dict[str, pd.DataFrame] = {
        spec["metric_key"]: pd.DataFrame()
        for spec in scenario_metric_specs
    }
    trajectory_plots: list[dict[str, Any]] = []
    deltas_heatmap_entries: list[dict[str, Any]] = []
    per_method_plot_entries: list[dict[str, Any]] = []
    comparison_plots_by_dataset: dict[str, Any] = {}
    method_win_rate_plots: dict[str, Any] = {}
    selection_margin_plot = None
    scenario_delta_heatmap_entries: list[dict[str, Any]] = []
    core_figure_registry = _core_figure_registry()
    if not backbone_df.empty and not method_selection_df.empty:
        print("\n--- Computing Improvement Deltas ---")
        perf_col = f"{args.test_metric}_test"
        if "err_pert_mean" not in robustness_metric_columns:
            raise ValueError(
                "Cannot compute Taori-style tau_mean because err_pert_mean is missing "
                "from the canonical winner-pool results."
            )
        all_metrics = list(
            dict.fromkeys(
                [
                    perf_col,
                    build_metric_w_name(
                        args.test_metric,
                        key="test_metric",
                        context="meta-analysis",
                    ),
                    *robustness_metric_columns,
                    *REFERENCE_NORMALIZED_DIAGNOSTIC_METRIC_KEYS,
                ]
            )
        )

        improvement_deltas_df = compute_improvement_deltas(
            backbone_df,
            method_selection_df,
            all_metrics,
            pair_reference_df=result_df,
        )
        improvement_deltas_selected_df = _select_canonical_method_deltas(
            improvement_deltas_df,
            canonical_method_df=method_selection_df,
        )

        if not improvement_deltas_selected_df.empty:
            print(
                "Computed deltas for "
                f"{len(improvement_deltas_selected_df)} canonical method rows (vs baseline)."
            )
            method_delta_plot_df = _canonical_method_delta_plot_df(
                improvement_deltas_selected_df,
                context="Cannot build canonical method delta plot source",
            )
            pipeline_method_delta_results_df = _build_pipeline_method_delta_results(
                method_delta_plot_df,
                perf_col=perf_col,
                test_metric=args.test_metric,
                bootstrap_resamples=meta_bootstrap_ci_resamples,
                bootstrap_confidence_level=meta_bootstrap_ci_confidence_level,
            )
            improvement_delta_overview_figure_specs = (
                _build_improvement_delta_overview_figure_specs(
                    method_delta_plot_df,
                    perf_col=perf_col,
                    test_metric=args.test_metric,
                )
            )

            delta_cols = [
                col
                for col in method_delta_plot_df.columns
                if col.startswith("delta_")
            ]
            long_metric_cols = delta_cols + [
                col for col in ("tau_mean",) if col in method_delta_plot_df.columns
            ]
            melted_records = []
            for _, row in method_delta_plot_df.iterrows():
                for metric_col in long_metric_cols:
                    if pd.notna(row[metric_col]):
                        metric_name = (
                            metric_col.replace("delta_", "")
                            if metric_col.startswith("delta_")
                            else metric_col
                        )
                        melted_records.append({
                            "dataset": row.get("dataset"),
                            "pipeline_id": row.get("pipeline_id"),
                            "robustness_method": row.get("robustness_method"),
                            "model_architecture": row.get("model_architecture"),
                            "metric_name": metric_name,
                            "delta_value": row[metric_col],
                        })

            if melted_records:
                improvement_deltas_long_df = pd.DataFrame(melted_records)

            metrics_to_plot = core_robustness_metric_keys
            if method_selection_df.empty:
                print("No method rows available; skipping improvement trajectories.")
            else:
                required_candidate_cols = {
                    "dataset",
                    "data_config_signature",
                    "robustness_method",
                    "pipeline_id",
                    "model_architecture",
                    "backbone_architecture",
                }
                _require_columns(
                    method_selection_df,
                    required_candidate_cols,
                    context="Cannot generate trajectory plots",
                )
                _require_columns(
                    backbone_df,
                    {
                        "dataset",
                        "data_config_signature",
                        "model_architecture",
                    },
                    context="Cannot generate trajectory plots",
                )
                method_selection_plot_df = _canonical_method_winner_plot_df(
                    canonical_method_plot_df,
                    context="Cannot generate trajectory plots",
                    required_cols=required_candidate_cols,
                )
                trajectory_plots = _build_plot_artifact_with_partial_coverage_tolerance(
                    lambda: _build_method_trajectory_plot_entries(
                        backbone_df,
                        method_selection_plot_df,
                        perf_col=perf_col,
                        test_metric=args.test_metric,
                        trajectory_metrics=run_level_core_metric_keys,
                        full_coverage=core_figures_require_full_coverage,
                    ),
                    runtime_args=args,
                    context="Trajectory plots",
                    empty_value=[],
                    suppress_markers=(
                        "has no core-figure datasets after filtering",
                        "has no core-figure baseline rows",
                    ),
                )
                if core_figures_require_full_coverage and not any(
                    entry["robustness_method"]
                    == core_figure_registry.core_improvement_trajectory_method
                    and entry["metric"]
                    == core_figure_registry.core_improvement_trajectory_metric
                    for entry in trajectory_plots
                ):
                    raise ValueError(
                        "Cannot generate trajectory plots: missing required core "
                        f"trajectory figure for robustness method "
                        f"'{core_figure_registry.core_improvement_trajectory_method}' "
                        "and metric "
                        f"'{core_figure_registry.core_improvement_trajectory_metric}'."
                    )

            # Generate dataset-local delta heatmaps
            heatmap_identity_cols = [
                "dataset",
                "robustness_method",
                "backbone_architecture",
            ]
            _require_columns(
                method_delta_plot_df,
                set(heatmap_identity_cols),
                context="Cannot generate method delta heatmaps",
            )
            _assert_no_duplicates(
                method_delta_plot_df,
                heatmap_identity_cols,
                context=(
                    "Cannot generate method delta heatmaps because canonical method delta rows "
                    "contain duplicate (dataset, robustness_method, backbone_architecture) entries"
                ),
            )
            agg_deltas = method_delta_plot_df.copy()
            if not agg_deltas.empty:
                missing_dataset = agg_deltas["dataset"].isna()
                if missing_dataset.any():
                    examples = (
                        agg_deltas.loc[
                            missing_dataset,
                            ["robustness_method", "backbone_architecture"],
                        ]
                        .drop_duplicates()
                        .head(5)
                        .to_dict(orient="records")
                    )
                    raise ValueError(
                        "Cannot generate dataset-local delta heatmaps because "
                        f"aggregated deltas contain missing dataset values. Examples: {examples}."
                    )
                delta_perf_col = (
                    "delta_err_clean"
                    if "delta_err_clean" in agg_deltas.columns
                    else f"delta_{perf_col}"
                )
                for dataset_name, dataset_agg in agg_deltas.groupby("dataset", dropna=False):
                    dataset_label = str(dataset_name)
                    for metric_to_plot in metrics_to_plot:
                        metric_spec = _core_robustness_metric_spec(metric_to_plot)
                        delta_robust_col = _core_delta_column(metric_to_plot)
                        metric_dataset_df = _filter_core_metric_available_rows(
                            dataset_agg,
                            perf_col=perf_col,
                            metric_key=metric_to_plot,
                            context=(
                                "Cannot generate method delta heatmap for "
                                f"'{dataset_label}/{metric_to_plot}'"
                            ),
                        )
                        if metric_dataset_df.empty:
                            continue
                        metric_title = robustness_metric_display_name(metric_to_plot)
                        heatmap = plot_method_delta_pair_subplots(
                            metric_dataset_df,
                            perf_delta_col=delta_perf_col,
                            robust_delta_col=delta_robust_col,
                            perf_semantics=_require_plot_semantics_for_keys(
                                test_metric=args.test_metric,
                                required_keys=[delta_perf_col],
                                context="Method delta heatmap",
                            )[delta_perf_col],
                            robust_semantics=_require_plot_semantics_for_keys(
                                test_metric=args.test_metric,
                                required_keys=[delta_robust_col],
                                context="Method delta heatmap",
                            )[delta_robust_col],
                            robust_higher_is_better=metric_spec.higher_is_better,
                            method_col="robustness_method",
                            baseline_col="backbone_architecture",
                            title=(
                                f"{dataset_label}: Baseline x Method Deltas "
                                f"({metric_title})"
                            ),
                        )
                        deltas_heatmap_entries.append(
                            {
                                "dataset": dataset_label,
                                "metric": metric_to_plot,
                                "figure": heatmap,
                            }
                        )

            # --- Dataset-level winner effectiveness plots ---
            per_method_plot_entries = []
            delta_perf_col = (
                "delta_err_clean"
                if "delta_err_clean" in method_delta_plot_df.columns
                else f"delta_{perf_col}"
            )
            scatter_source_df = method_delta_plot_df.copy()

            for dataset_name, dataset_df in scatter_source_df.groupby(
                "dataset",
                dropna=False,
            ):
                if pd.isna(dataset_name):
                    raise ValueError(
                        "Cannot generate dataset-level delta diagnostics because "
                        "improvement deltas contain missing dataset values."
                    )
                dataset_label = str(dataset_name)
                arch_col = (
                    "architecture_family"
                    if "architecture_family" in dataset_df.columns
                    else "model_architecture"
                )
                if delta_perf_col not in dataset_df.columns:
                    raise ValueError(
                        f"Cannot generate dataset-level delta diagnostics for dataset "
                        f"'{dataset_label}': missing '{delta_perf_col}'."
                    )
                for metric_to_plot in metrics_to_plot:
                    metric_spec = _core_robustness_metric_spec(metric_to_plot)
                    delta_robust_col = _core_delta_column(metric_to_plot)
                    metric_dataset_df = _filter_core_metric_available_rows(
                        dataset_df,
                        perf_col=perf_col,
                        metric_key=metric_to_plot,
                        context=(
                            "Cannot generate dataset-level delta scatter for "
                            f"'{dataset_label}/{metric_to_plot}'"
                        ),
                    )
                    if metric_dataset_df.empty:
                        continue
                    fig = plot_per_method_delta_scatter(
                        metric_dataset_df,
                        delta_perf_col=delta_perf_col,
                        delta_robust_col=delta_robust_col,
                        method_name=f"{dataset_label}: Winner Method Deltas",
                        perf_semantics=_require_plot_semantics_for_keys(
                            test_metric=args.test_metric,
                            required_keys=[delta_perf_col],
                            context="Dataset-level delta scatter",
                        )[delta_perf_col],
                        robust_semantics=_require_plot_semantics_for_keys(
                            test_metric=args.test_metric,
                            required_keys=[delta_robust_col],
                            context="Dataset-level delta scatter",
                        )[delta_robust_col],
                        robust_higher_is_better=metric_spec.higher_is_better,
                        arch_col=arch_col,
                        symbol_col="robustness_method",
                        perf_lower_is_better=True,
                        normalize_perf=False,
                        x_title=None,
                        y_title=None,
                    )
                    per_method_plot_entries.append(
                        {
                            "dataset": dataset_label,
                            "robustness_method": None,
                            "pipeline_id": None,
                            "metric": metric_to_plot,
                            "figure": fig,
                        }
                    )

            if not method_selection_df.empty:
                required_candidate_cols = {
                    "dataset",
                    "robustness_method",
                    "pipeline_id",
                    "model_architecture",
                }
                _require_columns(
                    method_delta_plot_df,
                    required_candidate_cols,
                    context="Cannot compute method-level improvement deltas",
                )
                if (method_delta_plot_df["pipeline_id"].astype(str) == "baseline").any():
                    raise ValueError(
                        "Method-level deltas contain baseline rows."
                    )
                selected_arch_key = [
                    "dataset",
                    "robustness_method",
                    "pipeline_id",
                    "model_architecture",
                ]
                _assert_no_duplicates(
                    method_delta_plot_df,
                    selected_arch_key,
                    context=(
                        "Method-level deltas contain duplicate architecture rows per "
                        "(dataset, robustness_method, pipeline_id)"
                    ),
                )
                if not method_delta_plot_df.empty:
                    metric_specs = _require_core_delta_metrics(
                        method_delta_plot_df,
                        perf_col=perf_col,
                    )
                    perf_delta_col, perf_metric_name = metric_specs[0]
                    perf_missing = method_delta_plot_df[perf_delta_col].isna()
                    if perf_missing.any():
                        examples = _sample_records(
                            method_delta_plot_df.loc[
                                perf_missing,
                                [
                                    "dataset",
                                    "robustness_method",
                                    "pipeline_id",
                                    "model_architecture",
                                ],
                            ],
                            [
                                "dataset",
                                "robustness_method",
                                "pipeline_id",
                                "model_architecture",
                            ],
                        )
                        raise ValueError(
                            "Cannot generate dataset-local core comparison plots because "
                            f"performance deltas contain missing values for '{perf_delta_col}'. "
                            f"Examples: {examples}."
                        )
                    core_records = []
                    for _, row in method_delta_plot_df.iterrows():
                        core_records.append(
                            {
                                "dataset": row.get("dataset"),
                                "robustness_method": row.get("robustness_method"),
                                "pipeline_id": row.get("pipeline_id"),
                                "model_architecture": row.get("model_architecture"),
                                "metric_name": perf_metric_name,
                                "delta_value": float(row.get(perf_delta_col)),
                            }
                        )
                    for spec in CORE_ROBUSTNESS_METRIC_SPECS:
                        metric_plot_df = _filter_core_metric_available_rows(
                            method_delta_plot_df,
                            perf_col=perf_col,
                            metric_key=spec.metric_key,
                            context="Cannot generate dataset-local core comparison plots",
                        )
                        for _, row in metric_plot_df.iterrows():
                            core_records.append(
                                {
                                    "dataset": row.get("dataset"),
                                    "robustness_method": row.get("robustness_method"),
                                    "pipeline_id": row.get("pipeline_id"),
                                    "model_architecture": row.get("model_architecture"),
                                    "metric_name": spec.metric_key,
                                    "delta_value": float(
                                        row.get(_core_delta_column(spec.metric_key))
                                    ),
                                }
                            )
                    if core_records:
                        core_metric_deltas_long_df = pd.DataFrame(core_records)
                        missing_dataset = core_metric_deltas_long_df["dataset"].isna()
                        if missing_dataset.any():
                            examples = (
                                core_metric_deltas_long_df.loc[
                                    missing_dataset,
                                    ["robustness_method", "pipeline_id", "model_architecture"],
                                ]
                                .drop_duplicates()
                                .head(5)
                                .to_dict(orient="records")
                            )
                            raise ValueError(
                                "Cannot generate dataset-local core comparison plots because "
                                f"core delta rows are missing dataset values. Examples: {examples}."
                            )
                        for dataset_name, dataset_core_df in core_metric_deltas_long_df.groupby(
                            "dataset", dropna=False
                        ):
                            dataset_label = str(dataset_name)
                            comparison_plots_by_dataset[dataset_label] = plot_improvement_comparison(
                                dataset_core_df.copy(),
                                method_col="robustness_method",
                                title=f"{dataset_label}: Core Improvement Deltas vs Baseline",
                            )
                    else:
                        core_metric_label = ", ".join(core_robustness_metric_keys)
                        print(
                            "No non-null method core deltas found for "
                            f"{args.test_metric}_test, {core_metric_label}."
                        )
        else:
            print("No improvement deltas could be computed.")
    else:
        print("No improvement data available for analysis.")

    diagnostic_perf_col = f"{args.test_metric}_test"
    diagnostic_delta_perf_col = f"delta_{diagnostic_perf_col}"
    if not method_delta_plot_df.empty:
        if "delta_err_clean" in method_delta_plot_df.columns:
            diagnostic_delta_perf_col = "delta_err_clean"
        required_win_cols = {
            "dataset",
            "robustness_method",
            "model_architecture",
            diagnostic_delta_perf_col,
        }
        required_win_cols.update(
            _core_delta_column(metric_key) for metric_key in core_robustness_metric_keys
        )
        _require_columns(
            method_delta_plot_df,
            required_win_cols,
            context="Cannot compute method win-rate heatmap",
        )
        win_base_df = method_delta_plot_df.copy()
        for spec in CORE_ROBUSTNESS_METRIC_SPECS:
            metric_key = spec.metric_key
            robust_delta_col = _core_delta_column(metric_key)
            metric_win_df = _filter_core_metric_available_rows(
                win_base_df,
                perf_col=diagnostic_perf_col,
                metric_key=metric_key,
                context=f"Cannot compute {spec.win_context_name} win-rate diagnostics",
            )
            if metric_win_df.empty:
                aggregated = pd.DataFrame(
                    columns=[
                        "dataset",
                        "robustness_method",
                        "architectures_compared",
                        "win_rate_pct",
                    ]
                )
                method_win_rate_dfs[metric_key] = aggregated
                method_win_rate_plots[metric_key] = plot_method_win_rate_heatmap(
                    aggregated,
                    title=(
                        f"% Architectures Beating Baseline on {args.test_metric} (Test) "
                        f"and {robustness_metric_display_name(metric_key)}"
                    ),
                )
                continue
            perf_delta_numeric = pd.to_numeric(
                metric_win_df[diagnostic_delta_perf_col], errors="raise"
            )
            robustness_improved = _core_metric_delta_improved(
                metric_win_df[robust_delta_col],
                higher_is_better=spec.higher_is_better,
            )
            metric_win_df[spec.win_flag_col] = (
                (perf_delta_numeric < 0.0)
                & robustness_improved
            )
            aggregated = (
                metric_win_df.groupby(["dataset", "robustness_method"], dropna=False)
                .agg(
                    architectures_compared=("model_architecture", "nunique"),
                    win_rate_pct=(spec.win_flag_col, "mean"),
                )
                .reset_index()
            )
            aggregated["win_rate_pct"] = aggregated["win_rate_pct"] * 100.0
            method_win_rate_dfs[metric_key] = aggregated
            method_win_rate_plots[metric_key] = plot_method_win_rate_heatmap(
                aggregated,
                title=(
                    f"% Architectures Beating Baseline on {args.test_metric} (Test) "
                    f"and {robustness_metric_display_name(metric_key)}"
                ),
            )
    else:
        print("Skipping win-rate diagnostics (no method deltas).")

    if not variant_selection_summary_df.empty:
        required_margin_cols = {
            "dataset",
            "robustness_method",
            "selection_metric_name",
            "selection_rank",
            "selection_score",
        }
        _require_columns(
            variant_selection_summary_df,
            required_margin_cols,
            context="Cannot compute selection margins",
        )
        winners = variant_selection_summary_df.loc[
            variant_selection_summary_df["selection_rank"] == 1,
            ["dataset", "robustness_method", "selection_metric_name", "selection_score"],
        ].rename(columns={"selection_score": "winner_score"})
        runners = variant_selection_summary_df.loc[
            variant_selection_summary_df["selection_rank"] == 2,
            ["dataset", "robustness_method", "selection_metric_name", "selection_score"],
        ].rename(columns={"selection_score": "runner_score"})
        if runners.empty:
            print("Skipping selection-margin plot (no runner-up candidates).")
        else:
            selection_margin_df = winners.merge(
                runners,
                on=["dataset", "robustness_method", "selection_metric_name"],
                how="inner",
            )
            selection_margin_df["selection_margin"] = (
                selection_margin_df["runner_score"] - selection_margin_df["winner_score"]
            )
            invalid_margin = selection_margin_df["selection_margin"] < -1e-10
            if invalid_margin.any():
                bad_rows = selection_margin_df.loc[
                    invalid_margin, ["dataset", "robustness_method", "winner_score", "runner_score"]
                ].head(5).to_dict(orient="records")
                raise ValueError(
                    "Selection margin produced negative values, indicating ranking inconsistency. "
                    f"Examples: {bad_rows}."
                )
            selection_margin_plot = plot_selection_margin(
                selection_margin_df,
                title=(
                    "Selection Margin (runner-up minus representative winner) "
                    "on validation selection score"
                ),
            )

    if not method_selection_df.empty:
        scenario_value_cols = [spec["scenario_col"] for spec in scenario_metric_specs]
        scenario_metrics_df = scenario_summary_df[
            [
                "dataset",
                "data_config_signature",
                "robustness_method",
                "pipeline_id",
                "model_architecture",
                "scenario",
                *scenario_value_cols,
            ]
        ].copy()
        if scenario_metrics_df.empty:
            raise ValueError(
                "Cannot compute scenario delta diagnostics because scenario-level records "
                "are missing."
            )
        baseline_scenario_df = scenario_metrics_df.loc[
            scenario_metrics_df["pipeline_id"].astype(str) == "baseline"
        ].copy()
        if baseline_scenario_df.empty:
            raise ValueError(
                "Cannot compute scenario delta diagnostics because baseline scenario records "
                "are missing."
            )
        baseline_key = ["dataset", "data_config_signature", "model_architecture", "scenario"]
        _assert_no_duplicates(
            baseline_scenario_df,
            baseline_key,
            context=(
                "Baseline scenario records contain duplicates per "
                "(dataset, model_architecture, scenario)"
            ),
        )
        selected_keys_df = method_delta_plot_df[
            [
                "dataset",
                "data_config_signature",
                "robustness_method",
                "pipeline_id",
                "model_architecture",
            ]
        ].drop_duplicates()
        selected_scenario_df = scenario_metrics_df.merge(
            selected_keys_df,
            on=[
                "dataset",
                "data_config_signature",
                "robustness_method",
                "pipeline_id",
                "model_architecture",
            ],
            how="inner",
        )
        selected_keys_present = selected_scenario_df[
            [
                "dataset",
                "data_config_signature",
                "robustness_method",
                "pipeline_id",
                "model_architecture",
            ]
        ].drop_duplicates()
        missing_scenario_keys = selected_keys_df.merge(
            selected_keys_present,
            on=[
                "dataset",
                "data_config_signature",
                "robustness_method",
                "pipeline_id",
                "model_architecture",
            ],
            how="left",
            indicator=True,
        )
        missing_scenario_keys = missing_scenario_keys[
            missing_scenario_keys["_merge"] == "left_only"
        ]
        if not missing_scenario_keys.empty:
            examples = (
                missing_scenario_keys[
                    [
                        "dataset",
                        "data_config_signature",
                        "robustness_method",
                        "pipeline_id",
                        "model_architecture",
                    ]
                ]
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                "Cannot compute scenario delta diagnostics because winner rows are missing "
                f"scenario records. Examples: {examples}."
            )
        if selected_scenario_df.empty:
            raise ValueError(
                "Cannot compute scenario delta diagnostics because winner rows "
                "have no scenario records."
            )
        selected_scenario_key = [
            "dataset",
            "data_config_signature",
            "robustness_method",
            "pipeline_id",
            "model_architecture",
            "scenario",
        ]
        _assert_no_duplicates(
            selected_scenario_df,
            selected_scenario_key,
            context=(
                "Winner scenario records contain duplicates per "
                "(dataset, data_config_signature, robustness_method, "
                "pipeline_id, model_architecture, scenario)"
            ),
        )
        baseline_metric_cols = [
            "dataset",
            "data_config_signature",
            "model_architecture",
            "scenario",
            *scenario_value_cols,
        ]
        merged_scenarios = selected_scenario_df.merge(
            baseline_scenario_df[baseline_metric_cols].rename(
                columns={
                    value_col: f"baseline_{value_col}"
                    for value_col in scenario_value_cols
                }
            ),
            on=["dataset", "data_config_signature", "model_architecture", "scenario"],
            how="left",
            indicator=True,
        )
        missing_baseline = merged_scenarios["_merge"] != "both"
        if missing_baseline.any():
            missing_count = int(missing_baseline.sum())
            examples = (
                merged_scenarios.loc[
                    missing_baseline,
                    [
                        "dataset",
                        "data_config_signature",
                        "robustness_method",
                        "pipeline_id",
                        "model_architecture",
                        "scenario",
                    ],
                ]
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                "Cannot compute scenario delta diagnostics because selected scenario rows "
                "have no matching baseline scenario rows. "
                f"Missing rows: {missing_count}. Examples: {examples}."
            )
        merged_scenarios = merged_scenarios.loc[
            merged_scenarios["_merge"] == "both"
        ].copy()
        if not merged_scenarios.empty:
            for spec in scenario_metric_specs:
                scenario_col = spec["scenario_col"]
                baseline_col = f"baseline_{scenario_col}"
                delta_col = spec["delta_col"]
                merged_scenarios[delta_col] = (
                    pd.to_numeric(merged_scenarios[scenario_col], errors="raise")
                    - pd.to_numeric(merged_scenarios[baseline_col], errors="raise")
                )
                scenario_delta_df = (
                    merged_scenarios.groupby(
                        ["dataset", "robustness_method", "scenario"], dropna=False
                    )[delta_col]
                    .mean()
                    .reset_index()
                )
                scenario_metric_delta_dfs[spec["metric_key"]] = scenario_delta_df
                if spec["metric_key"] == "D":
                    (
                        method_scenario_family_delta_df,
                        method_scenario_family_summary_df,
                    ) = _build_method_scenario_family_delta_tables(
                        merged_scenarios,
                        registry=core_figure_registry,
                        delta_col=delta_col,
                    )
                    core_scenario_heatmap_specs = (
                        _build_plot_artifact_with_partial_coverage_tolerance(
                            lambda: _build_method_scenario_delta_heatmap_figure_specs(
                                scenario_delta_df,
                                args=args,
                                full_coverage=core_figures_require_full_coverage,
                            ),
                            runtime_args=args,
                            context="Method scenario-delta heatmap",
                            empty_value=[],
                            suppress_markers=(
                                "received an empty dataframe",
                                "has no rows in the core-figure dataset/method scope",
                                "has no finite cells",
                                "no renderable core-figure dataset panels remain",
                            ),
                        )
                    )
                for dataset_name, dataset_scenario_delta_df in scenario_delta_df.groupby(
                    "dataset", dropna=False, sort=True
                ):
                    if dataset_scenario_delta_df.empty:
                        raise ValueError(
                            "Cannot generate scenario delta heatmap because grouped dataset "
                            f"frame is empty for metric '{spec['metric_key']}' and "
                            f"dataset '{dataset_name}'."
                        )
                    scenario_delta_heatmap_entries.append(
                        {
                            "dataset": str(dataset_name),
                            "metric_key": spec["metric_key"],
                            "delta_col": delta_col,
                            "figure": plot_scenario_delta_heatmap(
                                dataset_scenario_delta_df,
                                row_id_cols=("robustness_method",),
                                scenario_col="scenario",
                                value_col=delta_col,
                                value_semantics=_require_plot_semantics_for_keys(
                                    test_metric=args.test_metric,
                                    required_keys=[delta_col],
                                    context="Scenario delta heatmap",
                                )[delta_col],
                                title=(
                                    f"{dataset_name}: Selected Methods "
                                    f"{spec['title_suffix']}"
                                ),
                                color_label=None,
                            ),
                        }
                    )

    backbone_method_coverage_heatmaps: dict[str, Any] = {}
    if coverage_fractions_by_dataset:
        for dataset_name_cov, fractions in sorted(coverage_fractions_by_dataset.items()):
            if fractions:
                backbone_method_coverage_heatmaps[str(dataset_name_cov)] = (
                    plot_testing_coverage_heatmap(
                        fractions,
                        title=f"{dataset_name_cov}: Backbone x Method Coverage",
                    )
                )
    testing_coverage_df = _build_testing_coverage_table(
        coverage_fractions_by_dataset,
        coverage_source=coverage_source,
    )

    meta_n_test_samples = _require_meta_analysis_n_test_samples(
        result_df,
        args=args,
    )
    forecast_extreme_top_k = parse_required_positive_int(
        require_namespace_value(args, key="forecast_extreme_top_k"),
        key="forecast_extreme_top_k",
    )
    forecast_extremes_df = _select_forecast_extreme_rows(
        meta_input_rows.scenario_samples_df,
        canonical_method_df=canonical_method_plot_df,
        top_k=forecast_extreme_top_k,
        score_metric=args.test_metric,
    )
    forecast_extreme_plot_entries = _build_optional_forecast_extreme_plot_entries(
        args=args,
        forecast_extremes_df=forecast_extremes_df,
        result_df=result_df,
        winner_runs_by_id=winner_runs_by_id,
        client=client,
        resolved_specs=resolved_specs,
    )

    mlflow.set_experiment("Meta Analysis")
    run_name = _meta_analysis_run_name(
        args,
        n_test_samples=meta_n_test_samples,
    )
    with mlflow.start_run(run_name=run_name):
        tag_payload = _build_meta_analysis_run_tag_payload(
            reference_normalization_anchor_model=reference_normalization_anchor_model,
            unique_datasets=unique_datasets,
            unique_models=unique_models,
            coverage_source=str(coverage_source),
            eval_data_seed_mode=eval_data_seed_label,
            resolved_meta_eval_data_seed=resolved_meta_eval_data_seed,
            meta_eval_context_tag_payload=meta_eval_context_tag_payload,
            meta_n_test_samples=meta_n_test_samples,
            meta_bootstrap_ci_semantics=meta_bootstrap_ci_semantics,
            meta_bootstrap_ci_resamples=meta_bootstrap_ci_resamples,
            meta_bootstrap_ci_confidence_level=meta_bootstrap_ci_confidence_level,
            unique_pipeline_ids=unique_pipeline_ids,
            unique_pipeline_families=unique_pipeline_families,
        )
        mlflow.set_tags(tag_payload)
        meta_args_payload = _meta_analysis_args_payload(args)
        with tempfile.TemporaryDirectory(prefix="robust-") as tmpdir:
            meta_args_path = os.path.join(tmpdir, "meta_analysis_args.yaml")
            with open(meta_args_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    meta_args_payload,
                    handle,
                    sort_keys=True,
                    allow_unicode=False,
                    default_flow_style=False,
                )
            mlflow.log_artifact(meta_args_path, artifact_path="config")

        def _clean_numeric(value):
            if value is None or pd.isna(value):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        headline_metrics = {
            "models_evaluated": len(unique_models),
            "datasets_covered": len(unique_datasets),
            "runs_aggregated": len(meta_input_rows.result_df),
            "scenarios_tracked": int(
                meta_input_rows.scenario_summary_df["scenario"].nunique()
            )
            if not meta_input_rows.scenario_summary_df.empty
            else 0,
            "pipeline_ids_tracked": len(unique_pipeline_ids),
            "pipeline_families_tracked": len(unique_pipeline_families),
            "baseline_runs_analyzed": len(canonical_frames.backbone_df),
            "non_baseline_runs_analyzed": len(canonical_frames.non_baseline_eval_df),
        }
        val_key = f"{args.test_metric}_val"
        test_key = f"{args.test_metric}_test"
        if val_key in result_df.columns:
            headline_metrics[f"{val_key}_mean"] = result_df[val_key].mean()
        if test_key in result_df.columns:
            headline_metrics[f"{test_key}_mean"] = result_df[test_key].mean()
        at_worst_scenario_metric_key = build_metric_w_name(
            args.test_metric,
            key="test_metric",
            context="meta-analysis",
        )
        if at_worst_scenario_metric_key in result_df.columns:
            headline_metrics[
                f"{at_worst_scenario_metric_key}_mean"
            ] = result_df[at_worst_scenario_metric_key].mean()
        for metric in robustness_metric_columns:
            headline_metrics[f"avg_{metric}"] = result_df[metric].mean()

        for pipeline_id, count in pipeline_counts_dict.items():
            key = f"runs_pipeline_{str(pipeline_id).replace(' ', '_').replace('.', '_').lower()}"
            headline_metrics[key] = int(count)
        if not rho_eff_fits_df.empty:
            computed_fit_mask = rho_eff_fits_df["rho_eff_fit_status"] == "computed"
            headline_metrics["rho_eff_fit_groups_total"] = int(rho_eff_fits_df.shape[0])
            headline_metrics["rho_eff_fit_groups_computed"] = int(computed_fit_mask.sum())
            headline_metrics["rho_eff_fit_groups_unsupported"] = int((~computed_fit_mask).sum())
            headline_metrics["rho_eff_fit_rows_scored"] = int(
                rho_eff_fits_df["rho_eff_fit_n_rows_scored"].sum()
            )
            headline_metrics["rho_eff_fit_rows_non_positive_prediction"] = int(
                rho_eff_fits_df["rho_eff_fit_n_rows_non_positive_prediction"].sum()
            )
            computed_r2 = rho_eff_fits_df.loc[computed_fit_mask, "rho_eff_fit_r2"]
            finite_r2 = computed_r2[np.isfinite(computed_r2)]
            if not finite_r2.empty:
                headline_metrics["rho_eff_fit_r2_mean"] = float(finite_r2.mean())
                headline_metrics["rho_eff_fit_r2_min"] = float(finite_r2.min())
            computed_rmse = rho_eff_fits_df.loc[computed_fit_mask, "rho_eff_fit_rmse"]
            finite_rmse = computed_rmse[np.isfinite(computed_rmse)]
            if not finite_rmse.empty:
                headline_metrics["rho_eff_fit_rmse_mean"] = float(finite_rmse.mean())
                headline_metrics["rho_eff_fit_rmse_max"] = float(finite_rmse.max())

        numeric_metrics = {
            key: _clean_numeric(value) for key, value in headline_metrics.items()
        }
        numeric_metrics = {key: value for key, value in numeric_metrics.items() if value is not None}
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)

        tables = {
            "full_results": (result_df, "full_results.csv"),
            "analysis": (analysis_df, "analysis.csv"),
            "model_results": (model_results_df, "model_results.csv"),
            "dataset_results": (data_results_df, "dataset_results.csv"),
            "variant_results_by_dataset": (variant_by_dataset_results_df, "variant_results_by_dataset.csv"),
            "pipeline_method_results": (pipeline_method_results_df, "pipeline_method_results.csv"),
            "pipeline_method_delta_results": (
                pipeline_method_delta_results_df,
                "pipeline_method_delta_results.csv",
            ),
            "pipeline_counts": (pipeline_counts_df, "pipeline_counts.csv"),
            "method_aggregates": (method_aggregates_df, "method_aggregates.csv"),
            "method_variant_breakdown": (method_variant_breakdown_df, "method_variant_breakdown.csv"),
            "pipeline_method_candidates": (pipeline_method_candidates_df, "pipeline_method_candidates.csv"),
            "variant_selection_summary": (variant_selection_summary_df, "variant_selection_summary.csv"),
            "reference_normalization_anchors": (
                reference_normalization_anchors_df,
                "reference_normalization_anchors.csv",
            ),
            "rho_eff_fits": (rho_eff_fits_df, "rho_eff_fits.csv"),
            "scenario_summary": (
                meta_input_rows.scenario_summary_df,
                "scenario_summary.csv",
            ),
            "scenario_samples": (
                meta_input_rows.scenario_samples_df,
                "scenario_samples.csv",
            ),
            "testing_coverage": (
                testing_coverage_df,
                "testing_coverage.csv",
            ),
            "error_distribution_summary": (
                error_distribution_summary_df,
                "error_distribution_summary.csv",
            ),
            "model_correlations": (model_corr_summary, "model_correlations.csv"),
            "dataset_correlations": (dataset_corr_summary, "dataset_correlations.csv"),
            "improvement_deltas": (improvement_deltas_df, "improvement_deltas.csv"),
            "improvement_deltas_selected": (improvement_deltas_selected_df, "improvement_deltas_selected.csv"),
            "improvement_deltas_long": (improvement_deltas_long_df, "improvement_deltas_long.csv"),
            "core_metric_deltas_long": (core_metric_deltas_long_df, "core_metric_deltas_long.csv"),
            "method_scenario_family_delta": (
                method_scenario_family_delta_df,
                "method_scenario_family_delta.csv",
            ),
            "method_scenario_family_summary": (
                method_scenario_family_summary_df,
                "method_scenario_family_summary.csv",
            ),
            "selection_margin": (selection_margin_df, "selection_margin.csv"),
        }
        fixed_channel_fraction_for_tables, _ = _parse_fixed_channel_fraction_args(
            args
        )
        if fixed_channel_fraction_for_tables is not None:
            tables["fixed_channel_fraction"] = (
                fixed_channel_fraction_df,
                "fixed_channel_fraction.csv",
            )
            tables["fixed_channel_fraction_paper_summary"] = (
                fixed_channel_fraction_paper_summary_df,
                "fixed_channel_fraction_paper_summary.csv",
            )
        scenario_delta_table_specs = [
            ("D", "scenario_d_delta", "scenario_d_delta.csv"),
            (
                "err_pert",
                "scenario_err_pert_delta",
                "scenario_err_pert_delta.csv",
            ),
        ]
        for metric_key, table_name, filename in scenario_delta_table_specs:
            tables[table_name] = (
                scenario_metric_delta_dfs[metric_key],
                filename,
            )
        for spec in CORE_ROBUSTNESS_METRIC_SPECS:
            tables[spec.win_rate_table_name] = (
                method_win_rate_dfs[spec.metric_key],
                spec.win_rate_table_filename,
            )
        if not forecast_extremes_df.empty:
            tables["forecast_extremes"] = (
                forecast_extremes_df,
                "forecast_extremes.csv",
            )

        # Stage D table emission
        with tempfile.TemporaryDirectory(prefix="robust-") as tmpdir:
            for _, (df_obj, filename) in tables.items():
                path = os.path.join(tmpdir, filename)
                write_index = filename not in {
                    "fixed_channel_fraction.csv",
                    "fixed_channel_fraction_paper_summary.csv",
                }
                df_obj.to_csv(path, index=write_index)
            mlflow.log_artifacts(tmpdir, artifact_path="tables")

        figure_specs: list[FigureArtifactSpec] = []
        figure_specs.extend(baseline_core_figure_specs)
        figure_specs.extend(core_scenario_heatmap_specs)
        figure_specs.extend(baseline_clean_vs_worst_panel_specs)
        figure_specs.extend(error_distribution_figure_specs)
        figure_specs.extend(improvement_delta_overview_figure_specs)

        def _add_figure_spec(
            fig_obj,
            *,
            rel_parts: list[str],
            filename: str,
            figure_type: str,
            dataset: Optional[str] = None,
            pipeline_method: Optional[str] = None,
            pipeline_id: Optional[str] = None,
            metric: Optional[str] = None,
            optional: bool,
        ) -> None:
            if fig_obj is None:
                return
            figure_specs.append(
                FigureArtifactSpec(
                    figure=fig_obj,
                    rel_parts=tuple(str(part) for part in rel_parts),
                    filename=filename,
                    figure_type=figure_type,
                    dataset=dataset,
                    pipeline_method=pipeline_method,
                    pipeline_id=pipeline_id,
                    metric=metric,
                    optional=optional,
                )
            )

        def _add_core_figure_spec(
            fig_obj,
            *,
            rel_parts: list[str],
            filename: str,
            figure_type: str,
            dataset: Optional[str] = None,
            pipeline_method: Optional[str] = None,
            pipeline_id: Optional[str] = None,
            metric: Optional[str] = None,
        ) -> None:
            _add_figure_spec(
                fig_obj,
                rel_parts=rel_parts,
                filename=filename,
                figure_type=figure_type,
                dataset=dataset,
                pipeline_method=pipeline_method,
                pipeline_id=pipeline_id,
                metric=metric,
                optional=False,
            )

        def _add_diagnostic_figure_spec(
            fig_obj,
            *,
            rel_parts: list[str],
            filename: str,
            figure_type: str,
            dataset: Optional[str] = None,
            pipeline_method: Optional[str] = None,
            pipeline_id: Optional[str] = None,
            metric: Optional[str] = None,
        ) -> None:
            _add_figure_spec(
                fig_obj,
                rel_parts=rel_parts,
                filename=filename,
                figure_type=figure_type,
                dataset=dataset,
                pipeline_method=pipeline_method,
                pipeline_id=pipeline_id,
                metric=metric,
                optional=True,
            )

        summary_figures = [
            (
                robustness_corr_heatmap,
                ["1_overview", "correlations"],
                "robustness_metric_correlation.pdf",
                "overview_robustness_metric_correlation",
                None,
            ),
        ]
        for fig_obj, rel_parts, filename, figure_type, metric in summary_figures:
            _add_diagnostic_figure_spec(
                fig_obj,
                rel_parts=rel_parts,
                filename=filename,
                figure_type=figure_type,
                metric=metric,
            )
        _append_overview_correlation_figure_specs(
            figure_specs=figure_specs,
            model_corr_heatmaps_by_metric=model_corr_heatmaps_by_metric,
            dataset_corr_heatmaps_by_metric=dataset_corr_heatmaps_by_metric,
        )

        for metric, dataset_plots in _sorted_items(dataset_pareto_plots):
            metric_slug = _slugify_figure_value(metric, field="metric")
            for dataset_name, plot in _sorted_items(dataset_plots):
                dataset_slug = _slugify_figure_value(dataset_name, field="dataset")
                _add_diagnostic_figure_spec(
                    plot,
                    rel_parts=["2_baselines", "pareto", dataset_slug],
                    filename=f"pareto_frontier_{metric_slug}.pdf",
                    figure_type="baseline_pareto_by_dataset",
                    dataset=str(dataset_name),
                    metric=str(metric),
                )

        for metric, dataset_plots in _sorted_items(penmanshiel_no_seasonal_pareto_plots):
            metric_slug = _slugify_figure_value(metric, field="metric")
            for dataset_name, plot in _sorted_items(dataset_plots):
                dataset_slug = _slugify_figure_value(dataset_name, field="dataset")
                _add_diagnostic_figure_spec(
                    plot,
                    rel_parts=["2_baselines", "pareto", dataset_slug],
                    filename=f"pareto_frontier_{metric_slug}_no_seasonal_naive.pdf",
                    figure_type="baseline_pareto_by_dataset_no_seasonal_naive",
                    dataset=str(dataset_name),
                    metric=str(metric),
                )

        figure_specs.extend(
            _build_baseline_clean_vs_pert_figure_specs(
                dataset_clean_vs_pert_plot_entries
            )
        )

        sorted_forecast_extreme_entries = _sorted_records(
            forecast_extreme_plot_entries,
            keys=[
                "dataset",
                "robustness_method",
                "backbone_architecture",
                "extreme_kind",
                "pipeline_id",
                "run_id",
            ],
        )
        for entry in sorted_forecast_extreme_entries:
            dataset_name = str(entry["dataset"])
            dataset_slug = _slugify_figure_value(dataset_name, field="dataset")
            backbone_slug = _slugify_figure_value(
                str(entry["backbone_architecture"]),
                field="backbone_architecture",
            )
            method = str(entry["robustness_method"])
            method_slug = _slugify_figure_value(method, field="robustness_method")
            extreme_kind = str(entry["extreme_kind"])
            pipeline_id = str(entry["pipeline_id"])
            if method == "baseline":
                rel_parts = ["2_baselines", "forecast_extremes", dataset_slug]
                filename = f"{backbone_slug}__baseline__{extreme_kind}.pdf"
                figure_type = "baseline_forecast_extreme"
            else:
                rel_parts = [
                    "3_improvements",
                    dataset_slug,
                    "forecast_extremes",
                    method_slug,
                ]
                filename = f"{backbone_slug}__{method_slug}__{extreme_kind}.pdf"
                figure_type = "improvement_forecast_extreme"
            _add_diagnostic_figure_spec(
                entry["figure"],
                rel_parts=rel_parts,
                filename=filename,
                figure_type=figure_type,
                dataset=dataset_name,
                pipeline_method=method,
                pipeline_id=pipeline_id,
                metric=f"forecast_{extreme_kind}",
            )

        curve_source_df = meta_input_rows.scenario_samples_df
        curve_groups = []
        if not curve_source_df.empty:
            required_curve_cols = {
                "dataset",
                "robustness_method",
                "pipeline_id",
                "pipeline_kind",
                "backbone_architecture",
                "pert_idx",
                "scenario",
                "severity",
                "D",
            }
            missing_curve_cols = sorted(required_curve_cols - set(curve_source_df.columns))
            if missing_curve_cols:
                raise ValueError(
                    "scenario_samples_df is missing required columns for perturbation curves: "
                    f"{missing_curve_cols}."
                )
            curve_source_df = _filter_rows_to_canonical_method_winners(
                curve_source_df,
                canonical_method_df=canonical_method_plot_df,
                context="Cannot generate severity-curve figures",
                drop_out_of_scope_methods=True,
            )
            _assert_single_pipeline_per_method_backbone(
                curve_source_df[
                    [
                        "dataset",
                        "robustness_method",
                        "backbone_architecture",
                        "pipeline_id",
                    ]
                ].drop_duplicates(),
                context="Cannot generate severity-curve figures",
            )
            severity_numeric = pd.to_numeric(
                curve_source_df["severity"],
                errors="raise",
            ).astype(float)
            if ((severity_numeric < 0.0) | (severity_numeric > 1.0)).any():
                raise ValueError(
                    "Scenario-sample severity values for perturbation curves must lie in [0, 1]."
                )
            binned_curve_df = curve_source_df.copy()
            severity_bins = np.floor(np.minimum(severity_numeric, 0.999999) / 0.1).astype(int)
            binned_curve_df["severity"] = 0.05 + 0.1 * severity_bins
            binned_curve_df = (
                binned_curve_df.groupby(
                    [
                        "dataset",
                        "backbone_architecture",
                        "robustness_method",
                        "pipeline_id",
                        "pipeline_kind",
                        "pert_idx",
                        "scenario",
                        "severity",
                    ],
                    dropna=False,
                    sort=True,
                    as_index=False,
                )
                .agg(D=("D", "mean"))
            )
            curve_groups = binned_curve_df.groupby(
                [
                    "dataset",
                    "backbone_architecture",
                    "robustness_method",
                    "pipeline_id",
                    "pipeline_kind",
                ],
                dropna=False,
                sort=True,
            )
        for (
            dataset_name,
            backbone_architecture,
            robustness_method,
            pipeline_id,
            pipeline_kind,
        ), group_df in curve_groups:
            if pd.isna(dataset_name) or not str(dataset_name).strip():
                raise ValueError("Perturbation curve row is missing dataset.")
            if pd.isna(backbone_architecture) or not str(backbone_architecture).strip():
                raise ValueError(
                    "Perturbation curve row is missing backbone_architecture."
                )
            if pd.isna(robustness_method) or not str(robustness_method).strip():
                raise ValueError(
                    "Perturbation curve row is missing robustness_method."
                )
            if pd.isna(pipeline_id) or not str(pipeline_id).strip():
                raise ValueError(
                    "Perturbation curve row is missing pipeline_id."
                )
            if pd.isna(pipeline_kind) or not str(pipeline_kind).strip():
                raise ValueError(
                    "Perturbation curve row is missing pipeline_kind."
                )
            curve_idx_name_df = (
                group_df[["pert_idx", "scenario"]]
                .drop_duplicates()
                .copy()
            )
            curve_idx_name_df["scenario"] = (
                curve_idx_name_df["scenario"].astype(str).str.strip()
            )
            if (curve_idx_name_df["scenario"] == "").any():
                raise ValueError(
                    "Perturbation curve rows include empty scenario labels."
                )
            duplicate_curve_idx = curve_idx_name_df[
                curve_idx_name_df["pert_idx"].duplicated(keep=False)
            ]
            if not duplicate_curve_idx.empty:
                examples = duplicate_curve_idx.head(8).to_dict(orient="records")
                raise ValueError(
                    "Perturbation curve rows have inconsistent scenario labels for pert_idx. "
                    f"Examples: {examples}."
                )
            curve_idx_to_name = {
                int(idx): str(name)
                for idx, name in zip(
                    curve_idx_name_df["pert_idx"],
                    curve_idx_name_df["scenario"],
                )
            }
            df_curve = group_df[["pert_idx", "severity", "D"]].copy()
            curve_backbone = str(backbone_architecture).strip()
            curve_method = str(robustness_method).strip()
            curve_kind = str(pipeline_kind).strip()
            title = (
                f"{dataset_name} · {curve_backbone} · {curve_method} "
                f"({curve_kind}) Severity-Degradation Relationship"
            )
            fig_raw = plot_perturbation_curves(
                df_curve,
                model_name=f"{curve_backbone} | {curve_method}",
                value_col="D",
                y_semantics=_require_plot_semantics_for_keys(
                    test_metric=args.test_metric,
                    required_keys=["D"],
                    context="Severity-profile curves",
                )["D"],
                idx_to_name=curve_idx_to_name,
                title=title,
            )

            dataset_slug = _slugify_figure_value(dataset_name, field="dataset")
            method_slug = _slugify_figure_value(
                curve_method, field="robustness_method"
            )
            backbone_slug = _slugify_figure_value(
                curve_backbone, field="backbone_architecture"
            )
            if curve_method == "baseline":
                rel_parts = ["2_baselines", "severity_curves", dataset_slug]
                raw_filename = f"{backbone_slug}__baseline.pdf"
                figure_type = "baseline_perturbation_curve_raw"
            else:
                rel_parts = [
                    "3_improvements",
                    dataset_slug,
                    "severity_curves",
                    backbone_slug,
                ]
                raw_filename = f"{backbone_slug}__{method_slug}.pdf"
                figure_type = "improvement_perturbation_curve_raw"
            _add_diagnostic_figure_spec(
                fig_raw,
                rel_parts=rel_parts,
                filename=raw_filename,
                figure_type=figure_type,
                dataset=str(dataset_name),
                pipeline_method=curve_method,
                pipeline_id=str(pipeline_id),
                metric="D",
            )

        for dataset_name, dataset_plot in _sorted_items(comparison_plots_by_dataset):
            dataset_slug = _slugify_figure_value(dataset_name, field="dataset")
            _add_core_figure_spec(
                dataset_plot,
                rel_parts=["3_improvements", dataset_slug, "core_deltas"],
                filename="core_metric_deltas_comparison.pdf",
                figure_type="improvement_core_deltas_comparison_dataset",
                dataset=str(dataset_name),
            )

        sorted_heatmap_entries = _sorted_records(
            deltas_heatmap_entries,
            keys=["dataset", "metric"],
        )
        for entry in sorted_heatmap_entries:
            dataset_name = str(entry["dataset"])
            metric = str(entry["metric"])
            heatmap = entry["figure"]
            dataset_slug = _slugify_figure_value(dataset_name, field="dataset")
            metric_slug = _slugify_figure_value(metric, field="metric")
            _add_core_figure_spec(
                heatmap,
                rel_parts=["3_improvements", dataset_slug, "core_deltas"],
                filename=f"deltas_heatmap_{metric_slug}.pdf",
                figure_type="improvement_core_delta_heatmap_dataset",
                dataset=dataset_name,
                metric=metric,
            )

        for dataset_name, coverage_plot in _sorted_items(backbone_method_coverage_heatmaps):
            dataset_slug = _slugify_figure_value(dataset_name, field="dataset")
            _add_diagnostic_figure_spec(
                coverage_plot,
                rel_parts=["1_overview", "coverage"],
                filename=f"{dataset_slug}_backbone_method_coverage.pdf",
                figure_type="improvement_backbone_method_coverage",
                dataset=str(dataset_name),
                metric="coverage",
            )
        for spec in CORE_ROBUSTNESS_METRIC_SPECS:
            method_plot = method_win_rate_plots.get(spec.metric_key)
            if method_plot is None:
                continue
            _add_diagnostic_figure_spec(
                method_plot,
                rel_parts=["3_improvements", "diagnostics"],
                filename=spec.win_rate_figure_filename,
                figure_type=spec.win_rate_figure_type,
                metric=f"{args.test_metric}_test+{spec.metric_key}",
            )

        if selection_margin_plot is not None:
            _add_diagnostic_figure_spec(
                selection_margin_plot,
                rel_parts=["3_improvements", "diagnostics"],
                filename="selection_margin.pdf",
                figure_type="improvement_selection_margin",
            )

        sorted_scenario_delta_entries = _sorted_records(
            scenario_delta_heatmap_entries,
            keys=["dataset", "metric_key", "delta_col"],
        )
        for entry in sorted_scenario_delta_entries:
            dataset_name = str(entry["dataset"])
            metric_key = str(entry["metric_key"])
            delta_col = str(entry["delta_col"])
            dataset_slug = _slugify_figure_value(dataset_name, field="dataset")
            metric_slug = _slugify_figure_value(metric_key, field="metric")
            _add_diagnostic_figure_spec(
                entry["figure"],
                rel_parts=["3_improvements", dataset_slug, "scenario_deltas"],
                filename=f"scenario_delta_heatmap_{metric_slug}.pdf",
                figure_type="improvement_scenario_delta_heatmap",
                dataset=dataset_name,
                metric=delta_col,
            )

        figure_specs.extend(
            _build_method_trajectory_figure_specs(trajectory_plots)
        )

        sorted_per_method_entries = _sorted_records(
            per_method_plot_entries,
            keys=["dataset", "robustness_method", "pipeline_id", "metric"],
        )
        for entry in sorted_per_method_entries:
            dataset_name = str(entry["dataset"])
            robustness_method = parse_optional_nonempty_string(
                entry.get("robustness_method"),
                key="robustness_method",
                context=f"Delta-scatter figure entry ({dataset_name})",
                disallow_none_token=True,
            )
            pipeline_id = parse_optional_nonempty_string(
                entry.get("pipeline_id"),
                key="pipeline_id",
                context=f"Delta-scatter figure entry ({dataset_name})",
                disallow_none_token=True,
            )
            metric = str(entry["metric"])
            plot = entry["figure"]
            dataset_slug = _slugify_figure_value(dataset_name, field="dataset")
            metric_slug = _slugify_figure_value(metric, field="metric")
            rel_parts = ["3_improvements", dataset_slug, "delta_scatter"]
            filename = f"delta_scatter_{metric_slug}.pdf"
            if robustness_method is not None:
                method_slug = _slugify_figure_value(
                    robustness_method, field="robustness_method"
                )
                rel_parts.append(method_slug)
            if pipeline_id is not None:
                pipeline_slug = _slugify_figure_value(pipeline_id, field="pipeline_id")
                filename = f"{pipeline_slug}_delta_scatter_{metric_slug}.pdf"
            _add_diagnostic_figure_spec(
                plot,
                rel_parts=rel_parts,
                filename=filename,
                figure_type="improvement_delta_scatter_dataset",
                dataset=dataset_name,
                pipeline_method=robustness_method,
                pipeline_id=pipeline_id,
                metric=metric,
            )

        analysis_artifacts = AnalysisArtifacts(
            tables=tables,
            figure_specs=figure_specs,
            headline_metrics=numeric_metrics,
        )
        _assert_core_figure_specs_non_optional(analysis_artifacts.figure_specs)

        # Stage D figure emission + manifest creation
        with tempfile.TemporaryDirectory(prefix="robust-") as tmpdir:
            figure_manifest_records = _render_figure_registry(
                figure_specs=analysis_artifacts.figure_specs,
                output_root=tmpdir,
                full_coverage=require_namespace_bool(args, key="full_coverage"),
            )
            mlflow.log_artifacts(tmpdir, artifact_path="figures")

        figures_manifest_df = pd.DataFrame(
            figure_manifest_records,
            columns=[
                "artifact_path",
                "dataset",
                "pipeline_method",
                "pipeline_id",
                "metric",
                "figure_type",
            ],
        )
        if not figures_manifest_df.empty:
            figures_manifest_df = figures_manifest_df.sort_values(
                ["artifact_path", "figure_type", "dataset", "pipeline_method", "pipeline_id", "metric"],
                na_position="last",
            ).reset_index(drop=True)
        with tempfile.TemporaryDirectory(prefix="robust-") as tmpdir:
            manifest_path = os.path.join(tmpdir, "figures_manifest.csv")
            figures_manifest_df.to_csv(manifest_path, index=False)
            mlflow.log_artifact(manifest_path, artifact_path="tables")

        # --- 5. Testing Coverage Summary ---
        print("\n--- Testing Coverage Summary ---")
        if analysis_df.empty:
            print("No backbone evaluations available to summarize.")
        else:
            backbone_col = "backbone_architecture" if "backbone_architecture" in analysis_df.columns else "model"
            all_backbones = set(analysis_df[backbone_col].dropna().astype(str))

            tested_backbones_per_dataset = (
                analysis_df.groupby("dataset")[backbone_col]
                .nunique()
                .reset_index()
            )
            tested_backbones_per_dataset.rename(
                columns={backbone_col: "tested_backbone_count"},
                inplace=True,
            )
            tested_backbones_per_dataset["total_backbones"] = len(all_backbones)

            print("Number of unique backbones tested per dataset:")
            print(tested_backbones_per_dataset.to_string(index=False))

            any_missing = False
            for dataset in datasets:
                dataset_backbones = analysis_df[analysis_df["dataset"] == dataset][backbone_col]
                tested_backbones = set(dataset_backbones.dropna().astype(str))
                missing_backbones = all_backbones - tested_backbones
                if missing_backbones:
                    any_missing = True
                    missing_display = ', '.join(sorted(missing_backbones))
                    print(
                        f"  - Dataset '{dataset}': Missing {len(missing_backbones)} backbones"
                        f" -> {missing_display}"
                    )
            if not any_missing:
                print("All available backbones have been tested on all specified datasets.")

        if not non_baseline_eval_df.empty:
            print("\n--- Pipeline Variant Coverage ---")
            for pipeline_id, group in non_baseline_eval_df.groupby("pipeline_id"):
                archs = sorted(set(group["model_architecture"].dropna().astype(str)))
                datasets_covered = sorted(set(group["dataset"].dropna().astype(str)))
                pipeline_method = (
                    group["robustness_method"].iloc[0]
                    if "robustness_method" in group.columns
                    else "unknown"
                )
                print(
                    f"  - Pipeline '{pipeline_id}' (family: {pipeline_method}): "
                    f"{len(group)} runs across architectures {archs} "
                    f"covering datasets {datasets_covered}"
                )
        print("\nDone.")
