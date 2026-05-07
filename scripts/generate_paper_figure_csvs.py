from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "paper_artifact"
DEFAULT_CORE_FIGURES_CONFIG = REPO_ROOT / "configs" / "reporting" / "core_figures.yaml"
DEFAULT_BENCHMARK_SCOPE_CONFIG = REPO_ROOT / "configs" / "benchmark_scope.yaml"

PRIMARY_TABLES = (
    "analysis.csv",
    "scenario_summary.csv",
    "improvement_deltas_selected.csv",
)

BACKBONE_DISPLAY_ORDER = (
    "SeasonalNaive",
    "DLinear",
    "GRU",
    "ModernTCN",
    "TSMixer",
    "PatchTST",
    "Chronos2",
)
TRAJECTORY_METHOD = "adversarial_training"

MAIN_TABLE_FILENAME = "main_table_backbone.csv"
BACKBONE_SCENARIO_FILENAME = "backbone_scenario_heatmap_data.csv"
PGD_TRAJECTORY_FILENAME = "pgd_trajectory_data.csv"
PAPER_INPUTS_DIRNAME = "paper_inputs"
PAPER_INPUT_METADATA = {
    MAIN_TABLE_FILENAME: {
        "role": "derived paper input for the baseline architecture table",
        "source_tables": ("analysis.csv",),
    },
    BACKBONE_SCENARIO_FILENAME: {
        "role": "derived paper input for the baseline scenario heatmap",
        "source_tables": ("analysis.csv", "scenario_summary.csv"),
    },
    PGD_TRAJECTORY_FILENAME: {
        "role": "derived paper input for the PGD trajectory plot",
        "source_tables": ("analysis.csv", "improvement_deltas_selected.csv"),
    },
}

MAIN_TABLE_COLUMNS = (
    "dataset_order",
    "dataset",
    "dataset_label",
    "backbone_order",
    "backbone",
    "backbone_label",
    "run_id",
    "data_config_signature",
    "D_w",
    "D_w_CI_lo",
    "D_w_CI_hi",
    "MSE_c",
    "MSE_w",
    "D_mean",
    "MSE_mean_fault_time",
    "worst_scenario",
    "n_test_samples",
    "selection_pool",
)

BACKBONE_SCENARIO_COLUMNS = (
    "dataset_order",
    "dataset",
    "dataset_label",
    "backbone_order",
    "backbone",
    "backbone_label",
    "scenario_order",
    "scenario",
    "scenario_label",
    "scenario_group",
    "D",
    "D_CI_lo",
    "D_CI_hi",
    "err_pert",
    "err_pert_CI_lo",
    "err_pert_CI_hi",
    "MSE_c",
    "D_w",
    "MSE_w",
    "worst_scenario",
    "is_worst_scenario",
    "run_id",
    "data_config_signature",
)

PGD_TRAJECTORY_COLUMNS = (
    "dataset_order",
    "dataset",
    "dataset_label",
    "backbone_order",
    "backbone",
    "backbone_label",
    "architecture_family",
    "data_config_signature",
    "baseline_run_id",
    "baseline_pipeline_id",
    "baseline_MSE_c",
    "baseline_D_w",
    "baseline_MSE_w",
    "baseline_D_mean",
    "baseline_MSE_mean_fault_time",
    "baseline_worst_scenario",
    "improvement_method",
    "improvement_method_label",
    "improvement_run_id",
    "improvement_pipeline_id",
    "improvement_MSE_c",
    "improvement_D_w",
    "improvement_MSE_w",
    "improvement_D_mean",
    "improvement_MSE_mean_fault_time",
    "improvement_worst_scenario",
    "delta_MSE_c",
    "delta_D_w",
    "delta_MSE_w",
    "delta_D_mean",
    "delta_MSE_mean_fault_time",
    "selection_pool",
)


@dataclass(frozen=True)
class CoreFigureConfig:
    dataset_order: tuple[str, ...]
    dataset_labels: dict[str, str]
    method_labels: dict[str, str]
    scenario_order: tuple[str, ...]
    scenario_labels: dict[str, str]
    scenario_groups: dict[str, tuple[str, ...]]
    trajectory_method: str


@dataclass(frozen=True)
class BenchmarkScopeConfig:
    architecture_order: tuple[str, ...]
    method_comparison_architectures: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the three paper figure-input CSVs from the bundled "
            "paper_artifact manifest and primary result tables."
        )
    )
    parser.add_argument(
        "--paper-artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Directory containing MANIFEST.json and tables/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the generated CSVs will be written.",
    )
    parser.add_argument(
        "--core-figures-config",
        type=Path,
        default=DEFAULT_CORE_FIGURES_CONFIG,
        help="Core figure registry YAML used for dataset and scenario labels.",
    )
    parser.add_argument(
        "--benchmark-scope-config",
        type=Path,
        default=DEFAULT_BENCHMARK_SCOPE_CONFIG,
        help="Benchmark scope YAML used to validate architecture coverage.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite generated CSVs if they already exist.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return payload


def _require_mapping(value: Any, *, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _require_list(value: Any, *, key: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list.")
    return value


def _require_string(value: Any, *, key: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string.")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{key} must be non-empty.")
    return stripped


def _parse_named_display_pairs(value: Any, *, key: str) -> dict[str, str]:
    entries = _require_list(value, key=key)
    parsed: dict[str, str] = {}
    for index, entry in enumerate(entries):
        text = _require_string(entry, key=f"{key}[{index}]")
        if "=" not in text:
            raise ValueError(f"{key}[{index}] must use '<id>=<label>' syntax.")
        raw_id, label = text.split("=", 1)
        raw_id = raw_id.strip()
        label = label.strip()
        if not raw_id:
            raise ValueError(f"{key}[{index}] has an empty id.")
        if not label:
            raise ValueError(f"{key}[{index}] has an empty label.")
        if raw_id in parsed:
            raise ValueError(f"{key} contains duplicate id '{raw_id}'.")
        parsed[raw_id] = label
    return parsed


def _parse_string_sequence(value: Any, *, key: str) -> tuple[str, ...]:
    entries = _require_list(value, key=key)
    parsed: list[str] = []
    for index, entry in enumerate(entries):
        text = _require_string(entry, key=f"{key}[{index}]")
        if text in parsed:
            raise ValueError(f"{key} contains duplicate value '{text}'.")
        parsed.append(text)
    return tuple(parsed)


def _parse_scenario_groups(value: Any, *, key: str) -> dict[str, tuple[str, ...]]:
    entries = _require_list(value, key=key)
    groups: dict[str, tuple[str, ...]] = {}
    for index, entry in enumerate(entries):
        text = _require_string(entry, key=f"{key}[{index}]")
        if "=" not in text:
            raise ValueError(f"{key}[{index}] must use '<group>=<ids>' syntax.")
        group, members = text.split("=", 1)
        group = group.strip()
        if not group:
            raise ValueError(f"{key}[{index}] has an empty group.")
        member_values = tuple(
            member.strip() for member in members.split(",") if member.strip()
        )
        if not member_values:
            raise ValueError(f"{key}[{index}] has no scenarios.")
        if len(set(member_values)) != len(member_values):
            raise ValueError(f"{key}[{index}] contains duplicate scenarios.")
        if group in groups:
            raise ValueError(f"{key} contains duplicate group '{group}'.")
        groups[group] = member_values
    return groups


def load_core_figure_config(path: Path) -> CoreFigureConfig:
    payload = _load_yaml_mapping(path)
    dataset_labels = _parse_named_display_pairs(
        payload.get("CORE_FIGURE_DATASET_SPEC"),
        key="CORE_FIGURE_DATASET_SPEC",
    )
    method_labels = _parse_named_display_pairs(
        payload.get("CORE_METHOD_DISPLAY"),
        key="CORE_METHOD_DISPLAY",
    )
    scenario_order = _parse_string_sequence(
        payload.get("CORE_SCENARIO_DISPLAY_ORDER"),
        key="CORE_SCENARIO_DISPLAY_ORDER",
    )
    scenario_labels = _parse_named_display_pairs(
        payload.get("CORE_SCENARIO_DISPLAY"),
        key="CORE_SCENARIO_DISPLAY",
    )
    scenario_groups = _parse_scenario_groups(
        payload.get("CORE_SCENARIO_GROUPS"),
        key="CORE_SCENARIO_GROUPS",
    )
    trajectory_method = _require_string(
        payload.get("CORE_IMPROVEMENT_TRAJECTORY_METHOD"),
        key="CORE_IMPROVEMENT_TRAJECTORY_METHOD",
    )
    if trajectory_method != TRAJECTORY_METHOD:
        raise ValueError(
            "This script generates PGD trajectory CSVs and requires "
            f"CORE_IMPROVEMENT_TRAJECTORY_METHOD={TRAJECTORY_METHOD!r}, "
            f"got {trajectory_method!r}."
        )
    grouped_scenarios = tuple(
        scenario for scenarios in scenario_groups.values() for scenario in scenarios
    )
    if grouped_scenarios != scenario_order:
        raise ValueError(
            "CORE_SCENARIO_GROUPS must cover CORE_SCENARIO_DISPLAY_ORDER exactly."
        )
    if set(scenario_labels) != set(scenario_order):
        raise ValueError(
            "CORE_SCENARIO_DISPLAY must cover CORE_SCENARIO_DISPLAY_ORDER exactly."
        )
    if trajectory_method not in method_labels:
        raise ValueError(
            f"CORE_METHOD_DISPLAY is missing trajectory method '{trajectory_method}'."
        )
    return CoreFigureConfig(
        dataset_order=tuple(dataset_labels.keys()),
        dataset_labels=dataset_labels,
        method_labels=method_labels,
        scenario_order=scenario_order,
        scenario_labels=scenario_labels,
        scenario_groups=scenario_groups,
        trajectory_method=trajectory_method,
    )


def load_benchmark_scope_config(path: Path) -> BenchmarkScopeConfig:
    payload = _load_yaml_mapping(path)
    architectures = _require_mapping(
        payload.get("architectures"),
        key="architectures",
    )
    roles = _require_mapping(
        architectures.get("roles"),
        key="architectures.roles",
    )
    architecture_order = tuple(
        _require_string(value, key=f"architectures.display_order[{index}]")
        for index, value in enumerate(
            _require_list(
                architectures.get("display_order"),
                key="architectures.display_order",
            )
        )
    )
    method_comparison_architectures = tuple(
        _require_string(
            value,
            key=f"architectures.roles.method_comparison[{index}]",
        )
        for index, value in enumerate(
            _require_list(
                roles.get("method_comparison"),
                key="architectures.roles.method_comparison",
            )
        )
    )
    missing = [
        backbone
        for backbone in BACKBONE_DISPLAY_ORDER
        if backbone not in set(architecture_order)
    ]
    if missing:
        raise ValueError(
            "BACKBONE_DISPLAY_ORDER contains architecture(s) missing from "
            f"benchmark scope: {missing}."
        )
    extra_method_backbones = [
        backbone
        for backbone in method_comparison_architectures
        if backbone not in set(BACKBONE_DISPLAY_ORDER)
    ]
    if extra_method_backbones:
        raise ValueError(
            "Method-comparison architectures are missing from "
            f"BACKBONE_DISPLAY_ORDER: {extra_method_backbones}."
        )
    return BenchmarkScopeConfig(
        architecture_order=architecture_order,
        method_comparison_architectures=method_comparison_architectures,
    )


def _manifest_primary_source_id(manifest: dict[str, Any]) -> str:
    return _require_string(
        manifest.get("primary_source_id"),
        key="MANIFEST.json primary_source_id",
    )


def _manifest_primary_source(manifest: dict[str, Any]) -> dict[str, Any]:
    primary_source_id = _manifest_primary_source_id(manifest)
    source_exports = _require_list(
        manifest.get("source_exports"),
        key="MANIFEST.json source_exports",
    )
    matches = [
        _require_mapping(source, key="MANIFEST.json source_exports[]")
        for source in source_exports
        if isinstance(source, dict) and source.get("source_id") == primary_source_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "MANIFEST.json must contain exactly one source_exports entry for "
            f"primary_source_id={primary_source_id!r}."
        )
    return matches[0]


def _manifest_file_record(manifest: dict[str, Any], relative_path: str) -> dict[str, Any]:
    files = _require_list(manifest.get("files"), key="MANIFEST.json files")
    matches = [
        _require_mapping(record, key="MANIFEST.json files[]")
        for record in files
        if isinstance(record, dict) and record.get("path") == relative_path
    ]
    if len(matches) != 1:
        raise ValueError(
            "MANIFEST.json must contain exactly one file record for "
            f"{relative_path!r}."
        )
    return matches[0]


def _require_manifest_table(
    *,
    artifact_root: Path,
    manifest: dict[str, Any],
    primary_source: dict[str, Any],
    table_name: str,
) -> Path:
    primary_source_id = _manifest_primary_source_id(manifest)
    source_tables = {
        _require_string(table, key=f"{primary_source_id}.tables[]")
        for table in _require_list(primary_source.get("tables"), key="source tables")
    }
    if table_name not in source_tables:
        raise ValueError(
            f"Primary manifest source '{primary_source_id}' does not list "
            f"{table_name!r}."
        )
    relative_path = f"tables/{primary_source_id}/{table_name}"
    record = _manifest_file_record(manifest, relative_path)
    table_path = artifact_root / relative_path
    if not table_path.is_file():
        raise FileNotFoundError(f"Missing primary table: {table_path}")
    expected_size = record.get("size_bytes")
    if expected_size != table_path.stat().st_size:
        raise ValueError(
            f"{table_path} size does not match MANIFEST.json: "
            f"{table_path.stat().st_size} != {expected_size}."
        )
    expected_sha = _require_string(
        record.get("sha256"),
        key=f"MANIFEST.json files[{relative_path}].sha256",
    )
    actual_sha = _sha256(table_path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"{table_path} sha256 does not match MANIFEST.json: "
            f"{actual_sha} != {expected_sha}."
        )
    return table_path


def _load_primary_tables(artifact_root: Path) -> dict[str, pd.DataFrame]:
    manifest = _load_json_mapping(artifact_root / "MANIFEST.json")
    primary_source = _manifest_primary_source(manifest)
    tables: dict[str, pd.DataFrame] = {}
    for table_name in PRIMARY_TABLES:
        table_path = _require_manifest_table(
            artifact_root=artifact_root,
            manifest=manifest,
            primary_source=primary_source,
            table_name=table_name,
        )
        tables[table_name] = pd.read_csv(table_path)
    return tables


def _require_columns(df: pd.DataFrame, columns: set[str], *, context: str) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{context} is missing column(s): {', '.join(missing)}.")


def _nonempty_string_series(
    df: pd.DataFrame,
    column: str,
    *,
    context: str,
) -> pd.Series:
    series = df[column]
    if series.isna().any():
        raise ValueError(f"{context} contains missing values in '{column}'.")
    result = series.astype(str).str.strip()
    if (result == "").any():
        raise ValueError(f"{context} contains empty values in '{column}'.")
    return result


def _assert_no_duplicates(
    df: pd.DataFrame,
    columns: list[str],
    *,
    context: str,
) -> None:
    duplicates = df.duplicated(columns, keep=False)
    if duplicates.any():
        examples = df.loc[duplicates, columns].head(5).to_dict(orient="records")
        raise ValueError(f"{context} contains duplicate rows: {examples}.")


def _require_exact_key_coverage(
    df: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    expected_keys: set[tuple[str, ...]],
    context: str,
) -> None:
    actual_keys = {
        tuple(str(value) for value in values)
        for values in df.loc[:, list(columns)].itertuples(index=False, name=None)
    }
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing:
        raise ValueError(f"{context} is missing expected key(s): {missing[:5]}.")
    if unexpected:
        raise ValueError(f"{context} contains unexpected key(s): {unexpected[:5]}.")


def _require_winner_pool(df: pd.DataFrame, *, context: str) -> None:
    _require_columns(df, {"selection_pool"}, context=context)
    selection_pool = _nonempty_string_series(
        df,
        "selection_pool",
        context=context,
    )
    bad_rows = selection_pool != "winner_pool"
    if bad_rows.any():
        sample_cols = [
            column
            for column in ("dataset", "run_id", "pipeline_id", "selection_pool")
            if column in df.columns
        ]
        examples = df.loc[bad_rows, sample_cols].head(5).to_dict(orient="records")
        raise ValueError(
            f"{context} contains rows outside winner_pool: {examples}."
        )


def _require_one_signature_per_key(
    df: pd.DataFrame,
    key_columns: tuple[str, ...],
    *,
    context: str,
) -> None:
    _require_columns(df, set(key_columns) | {"data_config_signature"}, context=context)
    signature_counts = (
        df.loc[:, list(key_columns) + ["data_config_signature"]]
        .drop_duplicates()
        .groupby(list(key_columns), dropna=False)["data_config_signature"]
        .nunique()
    )
    mixed = signature_counts[signature_counts > 1]
    if not mixed.empty:
        examples = [
            dict(zip(key_columns, key if isinstance(key, tuple) else (key,)))
            for key in mixed.index[:5]
        ]
        raise ValueError(
            f"{context} contains multiple data_config_signature values for "
            f"{key_columns}: {examples}."
        )


def _scenario_group_lookup(config: CoreFigureConfig) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for group_name, scenarios in config.scenario_groups.items():
        for scenario in scenarios:
            lookup[scenario] = group_name
    return lookup


def _enrich_dataset_backbone_columns(
    df: pd.DataFrame,
    *,
    core_config: CoreFigureConfig,
    backbone_col: str,
    context: str,
) -> pd.DataFrame:
    result = df.copy()
    result["dataset"] = _nonempty_string_series(result, "dataset", context=context)
    result["backbone"] = _nonempty_string_series(
        result,
        backbone_col,
        context=context,
    )
    dataset_order = {dataset: index for index, dataset in enumerate(core_config.dataset_order, start=1)}
    backbone_order = {
        backbone: index for index, backbone in enumerate(BACKBONE_DISPLAY_ORDER, start=1)
    }
    result["dataset_order"] = result["dataset"].map(dataset_order)
    result["dataset_label"] = result["dataset"].map(core_config.dataset_labels)
    result["backbone_order"] = result["backbone"].map(backbone_order)
    result["backbone_label"] = result["backbone"]
    for column in ("dataset_order", "dataset_label", "backbone_order"):
        if result[column].isna().any():
            examples = result.loc[result[column].isna(), ["dataset", "backbone"]]
            raise ValueError(
                f"{context} contains rows outside configured display scope: "
                f"{examples.head(5).to_dict(orient='records')}."
            )
    result["dataset_order"] = result["dataset_order"].astype(int)
    result["backbone_order"] = result["backbone_order"].astype(int)
    return result


def build_main_table_backbone(
    analysis_df: pd.DataFrame,
    *,
    core_config: CoreFigureConfig,
) -> pd.DataFrame:
    context = "main_table_backbone.csv generation"
    required = {
        "dataset",
        "data_config_signature",
        "run_id",
        "model_architecture",
        "pipeline_id",
        "pipeline_method",
        "robustness_method",
        "D_w",
        "D_w_CI_lo",
        "D_w_CI_hi",
        "MSE_test",
        "MSE_w",
        "D_mean",
        "err_pert_mean",
        "worst_scenario",
        "n_test_samples",
        "selection_pool",
    }
    _require_columns(analysis_df, required, context=context)
    baseline = analysis_df.loc[
        (analysis_df["pipeline_method"].astype(str) == "baseline")
        & (analysis_df["robustness_method"].astype(str) == "baseline")
        & (analysis_df["pipeline_id"].astype(str) == "baseline")
    ].copy()
    if baseline.empty:
        raise ValueError(f"{context} found no baseline rows.")
    _require_winner_pool(baseline, context=context)

    baseline = _enrich_dataset_backbone_columns(
        baseline,
        core_config=core_config,
        backbone_col="model_architecture",
        context=context,
    )
    _assert_no_duplicates(
        baseline,
        ["dataset", "data_config_signature", "backbone"],
        context=context,
    )
    _assert_no_duplicates(
        baseline,
        ["dataset", "backbone"],
        context=context,
    )
    _require_one_signature_per_key(
        baseline,
        ("dataset", "backbone"),
        context=context,
    )
    _require_one_signature_per_key(
        baseline,
        ("dataset",),
        context=context,
    )
    expected = {
        (dataset, backbone)
        for dataset in core_config.dataset_order
        for backbone in BACKBONE_DISPLAY_ORDER
    }
    _require_exact_key_coverage(
        baseline,
        columns=("dataset", "backbone"),
        expected_keys=expected,
        context=context,
    )

    baseline["MSE_c"] = baseline["MSE_test"]
    baseline["MSE_mean_fault_time"] = baseline["err_pert_mean"]
    output = baseline.sort_values(
        ["dataset_order", "backbone_order"],
        kind="mergesort",
    )
    return output.loc[:, MAIN_TABLE_COLUMNS].reset_index(drop=True)


def build_backbone_scenario_heatmap_data(
    scenario_df: pd.DataFrame,
    main_table_df: pd.DataFrame,
    *,
    core_config: CoreFigureConfig,
) -> pd.DataFrame:
    context = "backbone_scenario_heatmap_data.csv generation"
    required = {
        "dataset",
        "data_config_signature",
        "run_id",
        "pipeline_method",
        "pipeline_id",
        "robustness_method",
        "model_architecture",
        "scenario",
        "D",
        "D_CI_lo",
        "D_CI_hi",
        "err_pert",
        "err_pert_CI_lo",
        "err_pert_CI_hi",
    }
    _require_columns(scenario_df, required, context=context)
    baseline_scenarios = scenario_df.loc[
        (scenario_df["pipeline_method"].astype(str) == "baseline")
        & (scenario_df["robustness_method"].astype(str) == "baseline")
        & (scenario_df["pipeline_id"].astype(str) == "baseline")
    ].copy()
    if baseline_scenarios.empty:
        raise ValueError(f"{context} found no baseline scenario rows.")

    baseline_scenarios = _enrich_dataset_backbone_columns(
        baseline_scenarios,
        core_config=core_config,
        backbone_col="model_architecture",
        context=context,
    )
    baseline_scenarios["scenario"] = _nonempty_string_series(
        baseline_scenarios,
        "scenario",
        context=context,
    )
    scenario_order = {
        scenario: index for index, scenario in enumerate(core_config.scenario_order)
    }
    scenario_groups = _scenario_group_lookup(core_config)
    baseline_scenarios["scenario_order"] = baseline_scenarios["scenario"].map(
        scenario_order
    )
    baseline_scenarios["scenario_label"] = baseline_scenarios["scenario"].map(
        core_config.scenario_labels
    )
    baseline_scenarios["scenario_group"] = baseline_scenarios["scenario"].map(
        scenario_groups
    )
    for column in ("scenario_order", "scenario_label", "scenario_group"):
        if baseline_scenarios[column].isna().any():
            examples = baseline_scenarios.loc[
                baseline_scenarios[column].isna(),
                ["dataset", "backbone", "scenario"],
            ]
            raise ValueError(
                f"{context} contains unknown scenarios: "
                f"{examples.head(5).to_dict(orient='records')}."
            )
    baseline_scenarios["scenario_order"] = baseline_scenarios["scenario_order"].astype(
        int
    )
    _assert_no_duplicates(
        baseline_scenarios,
        ["dataset", "data_config_signature", "backbone", "scenario"],
        context=context,
    )
    expected = {
        (dataset, backbone, scenario)
        for dataset in core_config.dataset_order
        for backbone in BACKBONE_DISPLAY_ORDER
        for scenario in core_config.scenario_order
    }
    _require_exact_key_coverage(
        baseline_scenarios,
        columns=("dataset", "backbone", "scenario"),
        expected_keys=expected,
        context=context,
    )

    main_lookup = main_table_df.loc[
        :,
        [
            "dataset",
            "data_config_signature",
            "run_id",
            "backbone",
            "MSE_c",
            "D_w",
            "MSE_w",
            "worst_scenario",
        ],
    ]
    heatmap = baseline_scenarios.merge(
        main_lookup,
        on=["dataset", "data_config_signature", "run_id", "backbone"],
        how="left",
        validate="many_to_one",
    )
    if heatmap["MSE_c"].isna().any():
        raise ValueError(
            f"{context} could not match every scenario row to a selected baseline run."
        )
    heatmap["is_worst_scenario"] = heatmap["scenario"].eq(
        heatmap["worst_scenario"]
    ).map({True: "true", False: "false"})
    output = heatmap.sort_values(
        ["dataset_order", "backbone_order", "scenario_order"],
        kind="mergesort",
    )
    return output.loc[:, BACKBONE_SCENARIO_COLUMNS].reset_index(drop=True)


def build_pgd_trajectory_data(
    improvement_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    *,
    core_config: CoreFigureConfig,
    benchmark_scope: BenchmarkScopeConfig,
) -> pd.DataFrame:
    context = "pgd_trajectory_data.csv generation"
    required = {
        "dataset",
        "data_config_signature",
        "run_id",
        "model_architecture",
        "pipeline_id",
        "pipeline_method",
        "robustness_method",
        "architecture_family",
        "MSE_test_improved",
        "D_w_improved",
        "MSE_w_improved",
        "D_mean_improved",
        "err_pert_mean_improved",
        "worst_scenario_improved",
        "MSE_test_baseline",
        "MSE_w_baseline",
        "D_w_baseline",
        "D_mean_baseline",
        "err_pert_mean_baseline",
        "matched_baseline_run_id",
        "worst_scenario_baseline",
        "delta_MSE_test",
        "delta_D_w",
        "delta_MSE_w",
        "delta_D_mean",
        "delta_err_pert_mean",
        "selection_pool",
    }
    _require_columns(improvement_df, required, context=context)
    method_rows = improvement_df.loc[
        (improvement_df["pipeline_method"].astype(str) == core_config.trajectory_method)
        & (
            improvement_df["robustness_method"].astype(str)
            == core_config.trajectory_method
        )
    ].copy()
    if method_rows.empty:
        raise ValueError(
            f"{context} found no rows for method {core_config.trajectory_method!r}."
        )
    _require_winner_pool(method_rows, context=context)

    method_rows = _enrich_dataset_backbone_columns(
        method_rows,
        core_config=core_config,
        backbone_col="model_architecture",
        context=context,
    )
    method_backbones = tuple(
        backbone
        for backbone in BACKBONE_DISPLAY_ORDER
        if backbone in set(benchmark_scope.method_comparison_architectures)
    )
    method_rows = method_rows.loc[method_rows["backbone"].isin(method_backbones)].copy()
    _assert_no_duplicates(
        method_rows,
        ["dataset", "data_config_signature", "backbone"],
        context=context,
    )
    _assert_no_duplicates(
        method_rows,
        ["dataset", "backbone"],
        context=context,
    )
    _require_one_signature_per_key(
        method_rows,
        ("dataset", "backbone"),
        context=context,
    )
    _require_one_signature_per_key(
        method_rows,
        ("dataset",),
        context=context,
    )
    expected = {
        (dataset, backbone)
        for dataset in core_config.dataset_order
        for backbone in method_backbones
    }
    _require_exact_key_coverage(
        method_rows,
        columns=("dataset", "backbone"),
        expected_keys=expected,
        context=context,
    )

    _require_columns(
        analysis_df,
        {
            "run_id",
            "pipeline_id",
            "pipeline_method",
            "robustness_method",
            "data_config_signature",
            "selection_pool",
        },
        context=f"{context} baseline lookup",
    )
    baseline_lookup = analysis_df.loc[
        (analysis_df["pipeline_method"].astype(str) == "baseline")
        & (analysis_df["robustness_method"].astype(str) == "baseline")
        & (analysis_df["pipeline_id"].astype(str) == "baseline")
    ].copy()
    _require_winner_pool(baseline_lookup, context=f"{context} baseline lookup")
    _assert_no_duplicates(
        baseline_lookup,
        ["run_id"],
        context=f"{context} baseline lookup",
    )
    baseline_lookup = baseline_lookup.loc[
        :, ["run_id", "pipeline_id", "data_config_signature"]
    ].rename(
        columns={
            "run_id": "matched_baseline_run_id",
            "pipeline_id": "baseline_pipeline_id",
            "data_config_signature": "baseline_data_config_signature",
        }
    )
    method_rows = method_rows.merge(
        baseline_lookup,
        on="matched_baseline_run_id",
        how="left",
        validate="many_to_one",
    )
    if method_rows["baseline_pipeline_id"].isna().any():
        raise ValueError(f"{context} could not match every row to a baseline run.")
    signature_mismatch = (
        method_rows["data_config_signature"].astype(str)
        != method_rows["baseline_data_config_signature"].astype(str)
    )
    if signature_mismatch.any():
        examples = method_rows.loc[
            signature_mismatch,
            [
                "dataset",
                "backbone",
                "run_id",
                "data_config_signature",
                "matched_baseline_run_id",
                "baseline_data_config_signature",
            ],
        ].head(5).to_dict(orient="records")
        raise ValueError(
            f"{context} matched baseline rows have incompatible "
            f"data_config_signature values: {examples}."
        )

    method_rows["baseline_run_id"] = method_rows["matched_baseline_run_id"]
    method_rows["baseline_MSE_c"] = method_rows["MSE_test_baseline"]
    method_rows["baseline_D_w"] = method_rows["D_w_baseline"]
    method_rows["baseline_MSE_w"] = method_rows["MSE_w_baseline"]
    method_rows["baseline_D_mean"] = method_rows["D_mean_baseline"]
    method_rows["baseline_MSE_mean_fault_time"] = method_rows[
        "err_pert_mean_baseline"
    ]
    method_rows["baseline_worst_scenario"] = method_rows["worst_scenario_baseline"]
    method_rows["improvement_method"] = method_rows["pipeline_method"]
    method_rows["improvement_method_label"] = core_config.method_labels[
        core_config.trajectory_method
    ]
    method_rows["improvement_run_id"] = method_rows["run_id"]
    method_rows["improvement_pipeline_id"] = method_rows["pipeline_id"]
    method_rows["improvement_MSE_c"] = method_rows["MSE_test_improved"]
    method_rows["improvement_D_w"] = method_rows["D_w_improved"]
    method_rows["improvement_MSE_w"] = method_rows["MSE_w_improved"]
    method_rows["improvement_D_mean"] = method_rows["D_mean_improved"]
    method_rows["improvement_MSE_mean_fault_time"] = method_rows[
        "err_pert_mean_improved"
    ]
    method_rows["improvement_worst_scenario"] = method_rows[
        "worst_scenario_improved"
    ]
    method_rows["delta_MSE_c"] = method_rows["delta_MSE_test"]
    method_rows["delta_MSE_mean_fault_time"] = method_rows["delta_err_pert_mean"]

    output = method_rows.sort_values(
        ["dataset_order", "backbone_order"],
        kind="mergesort",
    )
    return output.loc[:, PGD_TRAJECTORY_COLUMNS].reset_index(drop=True)


def build_figure_csv_frames_from_tables(
    tables: Mapping[str, pd.DataFrame],
    *,
    core_figures_config: Path,
    benchmark_scope_config: Path,
) -> dict[str, pd.DataFrame]:
    missing = sorted(
        table_name
        for table_name in PRIMARY_TABLES
        if table_name not in set(tables.keys())
    )
    if missing:
        raise ValueError(
            "Cannot build paper figure CSV frames, missing source table(s): "
            f"{', '.join(missing)}."
        )
    core_config = load_core_figure_config(core_figures_config)
    benchmark_scope = load_benchmark_scope_config(benchmark_scope_config)
    main_table = build_main_table_backbone(
        tables["analysis.csv"],
        core_config=core_config,
    )
    scenario_heatmap = build_backbone_scenario_heatmap_data(
        tables["scenario_summary.csv"],
        main_table,
        core_config=core_config,
    )
    trajectory = build_pgd_trajectory_data(
        tables["improvement_deltas_selected.csv"],
        tables["analysis.csv"],
        core_config=core_config,
        benchmark_scope=benchmark_scope,
    )
    return {
        MAIN_TABLE_FILENAME: main_table,
        BACKBONE_SCENARIO_FILENAME: scenario_heatmap,
        PGD_TRAJECTORY_FILENAME: trajectory,
    }


def build_figure_csv_frames(
    *,
    artifact_root: Path,
    core_figures_config: Path,
    benchmark_scope_config: Path,
) -> dict[str, pd.DataFrame]:
    tables = _load_primary_tables(artifact_root)
    return build_figure_csv_frames_from_tables(
        tables,
        core_figures_config=core_figures_config,
        benchmark_scope_config=benchmark_scope_config,
    )


def generate_paper_figure_csvs(
    *,
    artifact_root: Path,
    output_dir: Path,
    core_figures_config: Path,
    benchmark_scope_config: Path,
    force: bool = False,
) -> dict[str, Path]:
    artifact_root = artifact_root.resolve()
    output_dir = output_dir.resolve()
    core_figures_config = core_figures_config.resolve()
    benchmark_scope_config = benchmark_scope_config.resolve()
    frames = build_figure_csv_frames(
        artifact_root=artifact_root,
        core_figures_config=core_figures_config,
        benchmark_scope_config=benchmark_scope_config,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for filename, frame in frames.items():
        output_path = output_dir / filename
        if output_path.exists() and not force:
            raise FileExistsError(
                f"{output_path} already exists. Pass --force to overwrite it."
            )
        frame.to_csv(output_path, index=False)
        output_paths[filename] = output_path
    return output_paths


def main() -> None:
    args = parse_args()
    output_paths = generate_paper_figure_csvs(
        artifact_root=args.paper_artifact_root,
        output_dir=args.output_dir,
        core_figures_config=args.core_figures_config,
        benchmark_scope_config=args.benchmark_scope_config,
        force=args.force,
    )
    for filename, path in output_paths.items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()
