from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PUBLIC_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(PUBLIC_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(PUBLIC_REPO_ROOT))

from scripts.generate_paper_figure_csvs import (
    DEFAULT_BENCHMARK_SCOPE_CONFIG,
    DEFAULT_CORE_FIGURES_CONFIG,
    PAPER_INPUTS_DIRNAME,
    PAPER_INPUT_METADATA,
    PRIMARY_TABLES as PAPER_INPUT_SOURCE_TABLES,
    build_figure_csv_frames_from_tables,
)
from scripts.build_supplementary_tables import (
    SUPPLEMENTARY_TABLES_DIRNAME,
    SUPPLEMENTARY_TABLES_MANIFEST,
    build_supplementary_tables,
)


ARTIFACT_ID = "sensorfault-bench-neurips2026-paper-artifact-v2"
MAX_FILE_BYTES = 5 * 1024 * 1024
DEFAULT_CONFIGURATION_SOURCE_ID = "default_configuration"
COMPUTE_ACCOUNTING_DIRNAME = "compute_accounting"
# Subtrees that other tools write into the bundle. A forced rebuild keeps them.
CARRIED_OVER_SUBTREES = ("R", COMPUTE_ACCOUNTING_DIRNAME)

PRIVATE_VALUE_PATTERNS = {
    "local home path": re.compile(r"/Users/[A-Za-z0-9_.-]+"),
    "email address": re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    "GitHub SSH repository locator": re.compile(
        r"(?i)\bgit@github\.com:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?\b",
    ),
    "GitHub HTTPS repository locator": re.compile(
        r"(?i)\bhttps?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?\b",
    ),
    "cluster-local service name": re.compile(r"(?i)\b[A-Za-z0-9_.-]+\.svc\.cluster\.local\b"),
    "cluster-local service suffix": re.compile(r"(?i)\bsvc\.cluster\.local\b"),
    "MLflow private service host": re.compile(r"(?i)\bmlflow-server\.[A-Za-z0-9_.-]+"),
    "MinIO private endpoint": re.compile(r"(?i)\bhttps?://minio(?:[.:/][^\s`\"')]+)?"),
    "AWS access key assignment": re.compile(r"\bAWS_ACCESS_KEY_ID\s*="),
    "AWS secret key assignment": re.compile(r"\bAWS_SECRET_ACCESS_KEY\s*="),
    "MinIO root password assignment": re.compile(r"\bMINIO_ROOT_PASSWORD\s*="),
}
FORBIDDEN_OUTPUT_FILENAMES = {
    "meta_analysis_args.yaml",
    "scenario_samples.csv",
}

REQUIRED_CONTEXT_FIELDS = (
    "_explicit_cli_args",
    "benchmark_architectures",
    "data_files",
    "full_coverage",
    "improvement_selection_mode",
    "n_test_samples",
    "perturbation_scenarios",
    "test_metric",
)
PUBLIC_CONTEXT_FIELDS = (
    "_explicit_cli_args",
    "accelerator",
    "batch_size",
    "benchmark_architectures",
    "bootstrap_ci_confidence_level",
    "bootstrap_ci_resamples",
    "data_files",
    "devices",
    "eval_data_seed",
    "fixed_channel_fraction",
    "full_coverage",
    "improvement_method",
    "improvement_selection_mode",
    "input_len",
    "loss",
    "max_epochs",
    "max_hp_trials_per_model",
    "n_test_samples",
    "num_workers",
    "perturbation_channel_fraction_max",
    "perturbation_scenarios",
    "precision",
    "reference_normalization_anchor_model",
    "seed",
    "target_len",
    "test_metric",
    "train_split",
    "val_split",
)

PRIMARY_TABLES = (
    "analysis.csv",
    "full_results.csv",
    "testing_coverage.csv",
    "model_results.csv",
    "dataset_results.csv",
    "pipeline_method_results.csv",
    "pipeline_method_delta_results.csv",
    "improvement_deltas_selected.csv",
    "improvement_deltas_long.csv",
    "scenario_summary.csv",
    "scenario_d_delta.csv",
    "scenario_err_pert_delta.csv",
    "variant_selection_summary.csv",
    "pipeline_counts.csv",
    "figures_manifest.csv",
    "reference_normalization_anchors.csv",
    "rho_eff_fits.csv",
    "method_aggregates.csv",
    "core_metric_deltas_long.csv",
)
SELECTOR_TABLES = (
    "variant_selection_summary.csv",
    "improvement_deltas_selected.csv",
    "pipeline_method_delta_results.csv",
)
SEED_TABLES = (
    "analysis.csv",
    "pipeline_method_delta_results.csv",
    "full_results.csv",
    "improvement_deltas_selected.csv",
)
CHANNEL_FRACTION_TABLES = (
    "fixed_channel_fraction.csv",
    "fixed_channel_fraction_paper_summary.csv",
    "full_results.csv",
)

TABLE_METADATA = {
    "analysis.csv": {
        "role": "baseline architecture and selected winner diagnostics",
        "table_pool": "winner-only",
    },
    "full_results.csv": {
        "role": "selected evaluated result pool",
        "table_pool": "selected full result pool",
    },
    "testing_coverage.csv": {
        "role": "tested-run coverage accounting",
        "table_pool": "coverage summary",
    },
    "model_results.csv": {
        "role": "architecture-level aggregate table",
        "table_pool": "aggregate",
    },
    "dataset_results.csv": {
        "role": "dataset-level aggregate table",
        "table_pool": "aggregate",
    },
    "pipeline_method_results.csv": {
        "role": "method-level aggregate table",
        "table_pool": "aggregate",
    },
    "pipeline_method_delta_results.csv": {
        "role": "dataset-level method-baseline delta table",
        "table_pool": "winner-only method pairs",
    },
    "improvement_deltas_selected.csv": {
        "role": "selected method-baseline architecture pairs",
        "table_pool": "winner-only method pairs",
    },
    "improvement_deltas_long.csv": {
        "role": "long-form selected method delta table",
        "table_pool": "winner-only method pairs",
    },
    "scenario_summary.csv": {
        "role": "scenario-difficulty summary table",
        "table_pool": "baseline winner scenarios",
    },
    "scenario_d_delta.csv": {
        "role": "method scenario-degradation delta table",
        "table_pool": "winner-only method pairs",
    },
    "scenario_err_pert_delta.csv": {
        "role": "method scenario fault-time-error delta table",
        "table_pool": "winner-only method pairs",
    },
    "variant_selection_summary.csv": {
        "role": "selected variant identity and selector audit table",
        "table_pool": "winner-only variants",
    },
    "pipeline_counts.csv": {
        "role": "pipeline coverage count table",
        "table_pool": "coverage summary",
    },
    "figures_manifest.csv": {
        "role": "generated figure provenance index",
        "table_pool": "figure index",
    },
    "reference_normalization_anchors.csv": {
        "role": "SeasonalNaive anchor metadata for normalized scores",
        "table_pool": "aggregate",
    },
    "rho_eff_fits.csv": {
        "role": "effective-robustness fit diagnostics",
        "table_pool": "aggregate",
    },
    "method_aggregates.csv": {
        "role": "method coverage and aggregate support table",
        "table_pool": "aggregate",
    },
    "core_metric_deltas_long.csv": {
        "role": "long-form core metric delta table",
        "table_pool": "winner-only method pairs",
    },
    "fixed_channel_fraction.csv": {
        "role": "fixed selected-channel-fraction sensitivity rows",
        "table_pool": "winner-only sensitivity rows",
    },
    "fixed_channel_fraction_paper_summary.csv": {
        "role": "compact fixed selected-channel-fraction paper summary",
        "table_pool": "aggregate",
    },
}


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    export_date: str
    role: str
    relative_export_path: str
    expected_selection_mode: str
    table_files: tuple[str, ...]
    expected_fixed_channel_fraction: float | None = None


@dataclass(frozen=True)
class ForecastExampleSpec:
    dataset: str
    figure_group: str
    output_subdir: str
    batch_size: int
    eval_data_seed: int
    architecture: str
    scenario: str
    pert_idx: int
    sample_id: int
    source_idx: int
    severity: float


SOURCE_SPECS = (
    SourceSpec(
        source_id=DEFAULT_CONFIGURATION_SOURCE_ID,
        export_date="2026-04-26",
        role="results of the default configuration",
        relative_export_path="exports/2026-04-26_clean/artifacts_clean_final_10k",
        expected_selection_mode="clean",
        table_files=PRIMARY_TABLES,
    ),
    SourceSpec(
        source_id="selector_perturbed_validation",
        export_date="2026-04-29",
        role="selector-pressure sensitivity evidence",
        relative_export_path=(
            "exports/2026-04-29_pert_val/artifacts_pert_ws_val_final_10k"
        ),
        expected_selection_mode="perturbed_worst",
        table_files=SELECTOR_TABLES,
    ),
    SourceSpec(
        source_id="eval_data_seed_0",
        export_date="2026-04-30",
        role="evaluation-data seed sensitivity evidence",
        relative_export_path="exports/2026-04-30_seed01/artifacts_seed0",
        expected_selection_mode="clean",
        table_files=SEED_TABLES,
    ),
    SourceSpec(
        source_id="eval_data_seed_1",
        export_date="2026-04-30",
        role="evaluation-data seed sensitivity evidence",
        relative_export_path="exports/2026-04-30_seed01/artifacts_seed1",
        expected_selection_mode="clean",
        table_files=SEED_TABLES,
    ),
    SourceSpec(
        source_id="eval_data_seed_2",
        export_date="2026-05-01",
        role="evaluation-data seed sensitivity evidence",
        relative_export_path="exports/2026-05-01_seed23/artifacts_seed2",
        expected_selection_mode="clean",
        table_files=SEED_TABLES,
    ),
    SourceSpec(
        source_id="eval_data_seed_3",
        export_date="2026-05-01",
        role="evaluation-data seed sensitivity evidence",
        relative_export_path="exports/2026-05-01_seed23/artifacts_seed3",
        expected_selection_mode="clean",
        table_files=SEED_TABLES,
    ),
    SourceSpec(
        source_id="fixed_channel_fraction_0p5",
        export_date="2026-05-02",
        role="fixed selected-channel-fraction sensitivity evidence",
        relative_export_path="exports/2026-05-02_ch05/artifacts",
        expected_selection_mode="clean",
        expected_fixed_channel_fraction=0.5,
        table_files=CHANNEL_FRACTION_TABLES,
    ),
    SourceSpec(
        source_id="fixed_channel_fraction_0p25",
        export_date="2026-05-03",
        role="fixed selected-channel-fraction sensitivity evidence",
        relative_export_path="exports/2026-05-03_ch025/artifacts",
        expected_selection_mode="clean",
        expected_fixed_channel_fraction=0.25,
        table_files=CHANNEL_FRACTION_TABLES,
    ),
)

FORECAST_EXAMPLE_MANIFEST_PATH = "forecast_examples/manifest.json"
FORECAST_EXAMPLE_SOURCE_ID = "forecast_examples_2026-05-05"
FORECAST_EXAMPLE_SOURCE_EXPORT = "exports/2026-05-05_fcast/forecast_plots_final_traces"
FORECAST_EXAMPLE_N_TEST_SAMPLES = 10000
FORECAST_EXAMPLE_SPECS = (
    ForecastExampleSpec(
        dataset="BeijingAir_Tiantan",
        figure_group="short_term",
        output_subdir="forecast_plots_final_traces/beijing_air",
        batch_size=64,
        eval_data_seed=1806770612,
        architecture="PatchTST",
        scenario="drift",
        pert_idx=0,
        sample_id=2329,
        source_idx=572,
        severity=0.596497,
    ),
    ForecastExampleSpec(
        dataset="BeijingAir_Tiantan",
        figure_group="short_term",
        output_subdir="forecast_plots_final_traces/beijing_air",
        batch_size=64,
        eval_data_seed=1806770612,
        architecture="DLinear",
        scenario="spike",
        pert_idx=3,
        sample_id=8609,
        source_idx=3644,
        severity=0.492074,
    ),
    ForecastExampleSpec(
        dataset="Penmanshiel_Hourly_WT08",
        figure_group="short_term",
        output_subdir="forecast_plots_final_traces/penmanshiel",
        batch_size=64,
        eval_data_seed=1515776824,
        architecture="PatchTST",
        scenario="noise",
        pert_idx=2,
        sample_id=4638,
        source_idx=1136,
        severity=0.243317,
    ),
    ForecastExampleSpec(
        dataset="Penmanshiel_Hourly_WT08",
        figure_group="short_term",
        output_subdir="forecast_plots_final_traces/penmanshiel",
        batch_size=64,
        eval_data_seed=1515776824,
        architecture="ModernTCN",
        scenario="attenuation",
        pert_idx=1,
        sample_id=744,
        source_idx=1661,
        severity=0.884286,
    ),
    ForecastExampleSpec(
        dataset="traffic",
        figure_group="long_term",
        output_subdir="forecast_plots_final_traces/traffic",
        batch_size=16,
        eval_data_seed=74851880,
        architecture="PatchTST",
        scenario="time_stretch",
        pert_idx=4,
        sample_id=7218,
        source_idx=79,
        severity=0.494280,
    ),
    ForecastExampleSpec(
        dataset="traffic",
        figure_group="long_term",
        output_subdir="forecast_plots_final_traces/traffic",
        batch_size=16,
        eval_data_seed=74851880,
        architecture="PatchTST",
        scenario="missing_data",
        pert_idx=7,
        sample_id=664,
        source_idx=2366,
        severity=0.491388,
    ),
    ForecastExampleSpec(
        dataset="ETTh1",
        figure_group="long_term",
        output_subdir="forecast_plots_final_traces/etth1",
        batch_size=16,
        eval_data_seed=341970080,
        architecture="PatchTST",
        scenario="time_compress",
        pert_idx=5,
        sample_id=7968,
        source_idx=2167,
        severity=0.495240,
    ),
    ForecastExampleSpec(
        dataset="ETTh1",
        figure_group="long_term",
        output_subdir="forecast_plots_final_traces/etth1",
        batch_size=16,
        eval_data_seed=341970080,
        architecture="PatchTST",
        scenario="stuck_sensor",
        pert_idx=6,
        sample_id=8111,
        source_idx=427,
        severity=0.186316,
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen paper_artifact bundle for the NeurIPS paper."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Path to the experiment-output source tree used to build the artifact.",
    )
    parser.add_argument(
        "--supplementary-source-dir",
        type=Path,
        required=True,
        help=(
            "Directory holding the frozen inputs of the supplementary tables under "
            "their neutral file names, verified against pinned SHA-256 checksums."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory, normally paper_artifact.",
    )
    parser.add_argument(
        "--build-time-utc",
        help="Override build time for deterministic tests, in ISO-8601 UTC form.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    commit = _git(repo_root, "rev-parse", "HEAD")
    status = _git(repo_root, "status", "--short")
    return {
        "commit": commit,
        "worktree_dirty": bool(status.strip()),
    }


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return payload


def _generation_command(context: dict[str, Any]) -> str:
    raw_args = context.get("_explicit_cli_args")
    if not isinstance(raw_args, list) or not all(
        isinstance(item, str) for item in raw_args
    ):
        raise ValueError("_explicit_cli_args must be a list of strings.")
    return "uv run python run_analysis.py " + " ".join(raw_args)


def _sanitize_context(
    *,
    source_dir: Path,
    source_root: Path,
    spec: SourceSpec,
) -> dict[str, Any]:
    raw_config_path = source_dir / "config" / "meta_analysis_args.yaml"
    if not raw_config_path.is_file():
        raise FileNotFoundError(f"Missing source config: {raw_config_path}")
    raw_context = _load_yaml(raw_config_path)

    missing = [
        field for field in REQUIRED_CONTEXT_FIELDS if field not in raw_context
    ]
    if missing:
        raise ValueError(
            f"{raw_config_path} is missing required fields: {', '.join(missing)}"
        )

    actual_selection_mode = raw_context["improvement_selection_mode"]
    if actual_selection_mode != spec.expected_selection_mode:
        raise ValueError(
            f"{spec.source_id} expected improvement_selection_mode "
            f"{spec.expected_selection_mode!r}, got {actual_selection_mode!r}."
        )

    if spec.expected_fixed_channel_fraction is not None:
        actual_fraction = raw_context.get("fixed_channel_fraction")
        if actual_fraction != spec.expected_fixed_channel_fraction:
            raise ValueError(
                f"{spec.source_id} expected fixed_channel_fraction "
                f"{spec.expected_fixed_channel_fraction!r}, got {actual_fraction!r}."
            )

    public_context = {
        key: raw_context[key] for key in PUBLIC_CONTEXT_FIELDS if key in raw_context
    }
    public_context.update(
        {
            "source_id": spec.source_id,
            "source_role": spec.role,
            "source_export_relative_path": str(
                source_dir.relative_to(source_root)
            ),
            "source_config_sha256": _sha256(raw_config_path),
            "generation_command": _generation_command(raw_context),
        }
    )
    _assert_public_text(json.dumps(public_context, sort_keys=True), raw_config_path)
    return public_context


def _copy_table(
    *,
    source_root: Path,
    source_dir: Path,
    output_root: Path,
    spec: SourceSpec,
    table_name: str,
) -> dict[str, Any]:
    if table_name in FORBIDDEN_OUTPUT_FILENAMES:
        raise ValueError(f"{table_name} must not be shipped in paper_artifact.")
    metadata = TABLE_METADATA.get(table_name)
    if metadata is None:
        raise ValueError(f"{table_name} has no table metadata.")

    source_path = source_dir / "tables" / table_name
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source table: {source_path}")
    if source_path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"{source_path} exceeds {MAX_FILE_BYTES} bytes.")

    target_path = output_root / "tables" / spec.source_id / table_name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)
    _assert_public_file(target_path)

    return {
        "path": _artifact_relpath(output_root, target_path),
        "role": metadata["role"],
        "table_pool": metadata["table_pool"],
        "source_id": spec.source_id,
        "source_path": str(source_path.relative_to(source_root)),
        "size_bytes": target_path.stat().st_size,
        "sha256": _sha256(target_path),
    }


def _artifact_relpath(output_root: Path, path: Path) -> str:
    return path.relative_to(output_root).as_posix()


def _public_error_path(path: Path) -> str:
    if path.is_absolute():
        return path.name
    return path.as_posix()


def _assert_public_file(path: Path) -> None:
    if path.name in FORBIDDEN_OUTPUT_FILENAMES:
        raise ValueError(f"Forbidden output filename: {_public_error_path(path)}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"{_public_error_path(path)} exceeds {MAX_FILE_BYTES} bytes.")
    if path.suffix.lower() in {".csv", ".json", ".md", ".txt", ".yaml", ".yml"}:
        _assert_public_text(path.read_text(encoding="utf-8"), path)


def _assert_public_text(text: str, path: Path) -> None:
    offenders = [
        label
        for label, regex in PRIVATE_VALUE_PATTERNS.items()
        if regex.search(text)
    ]
    if offenders:
        raise ValueError(f"{_public_error_path(path)} contains private values: {offenders}")


def _collect_table_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "datasets": set(),
        "data_config_signatures": {},
        "eval_data_seeds": set(),
        "n_test_samples": set(),
        "robustness_scoring_semantics": set(),
        "selection_pools": set(),
        "test_metrics": set(),
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dataset = _nonempty(row.get("dataset"))
            signature = _nonempty(row.get("data_config_signature"))
            if dataset:
                metadata["datasets"].add(dataset)
            if dataset and signature:
                metadata["data_config_signatures"].setdefault(dataset, set()).add(
                    signature
                )
            for key, target in (
                ("eval_data_seed", "eval_data_seeds"),
                ("n_test_samples", "n_test_samples"),
                ("robustness_scoring_semantics", "robustness_scoring_semantics"),
                ("selection_pool", "selection_pools"),
                ("test_metric", "test_metrics"),
            ):
                value = _nonempty(row.get(key))
                if value is not None:
                    metadata[target].add(value)
    return metadata


def _nonempty(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _merge_csv_metadata(
    aggregate: dict[str, Any],
    table_metadata: dict[str, Any],
) -> None:
    for key in (
        "datasets",
        "eval_data_seeds",
        "n_test_samples",
        "robustness_scoring_semantics",
        "selection_pools",
        "test_metrics",
    ):
        aggregate.setdefault(key, set()).update(table_metadata[key])
    aggregate.setdefault("data_config_signatures", {})
    for dataset, signatures in table_metadata["data_config_signatures"].items():
        aggregate["data_config_signatures"].setdefault(dataset, set()).update(
            signatures
        )


def _freeze_sets(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {key: _freeze_sets(val) for key, val in sorted(value.items())}
    if isinstance(value, list):
        return [_freeze_sets(item) for item in value]
    return value


def _write_readme(output_root: Path) -> Path:
    readme_path = output_root / "README.md"
    readme_path.write_text(
        f"""# Paper Artifact

This directory is the entry point for the NeurIPS 2026 paper "Benchmarking
Sensor-Fault Robustness in Forecasting". The paper corresponds to release
`v1.0.0` of this repository. Later releases can extend the benchmark, but a
checkout of `v1.0.0` always restores the code, results, and evidence of the
paper.

## Paper to Repository Map

| Paper section | Bundled results | Commands |
| --- | --- | --- |
| "Results" and "Extended results" | `tables/{DEFAULT_CONFIGURATION_SOURCE_ID}/`, `{PAPER_INPUTS_DIRNAME}/` | [Reproducing the paper](../README.md#reproducing-the-paper) |
| "Sensitivity to the evaluation data seed", "Selector pressure", and "Sensitivity to the number of affected channels" | `tables/eval_data_seed_*/`, `tables/selector_perturbed_validation/`, `tables/fixed_channel_fraction_*/` | [Evaluation data seed, selector, and channel count](../docs/sensitivity_analyses.md#evaluation-data-seed-selector-and-channel-count) |
| "Sensitivity to severity weighting" | `{SUPPLEMENTARY_TABLES_DIRNAME}/severity_weighting_*.csv` | [Severity weighting](../docs/sensitivity_analyses.md#severity-weighting) |
| "Sensitivity to fault direction" | `{SUPPLEMENTARY_TABLES_DIRNAME}/fault_direction_architecture.csv` | [Fault direction](../docs/sensitivity_analyses.md#fault-direction) |
| "Dependence on TimeCompress" | `{SUPPLEMENTARY_TABLES_DIRNAME}/timing_*.csv` | [Timing](../docs/sensitivity_analyses.md#timing) |
| "ETTh1 target scope and horizon ablation" | `{SUPPLEMENTARY_TABLES_DIRNAME}/etth1_task_*.csv` | [ETTh1 target scope and horizon](../docs/sensitivity_analyses.md#etth1-target-scope-and-horizon) |
| "Training seed sensitivity" | `{SUPPLEMENTARY_TABLES_DIRNAME}/training_seed_*.csv`, `{SUPPLEMENTARY_TABLES_DIRNAME}/patchtst_clean_validation_near_tie*.csv` | [Training and search seeds](../docs/sensitivity_analyses.md#training-and-search-seeds) |
| "Full experimental details" | `{SUPPLEMENTARY_TABLES_DIRNAME}/selected_clean_validation_hyperparameters.csv`, `{COMPUTE_ACCOUNTING_DIRNAME}/` | [Compute accounting](#compute-accounting) |
| Figures | `R/figures/` | [Figure scripts](R/README.md) |

## Compute Accounting

[`{COMPUTE_ACCOUNTING_DIRNAME}/`]({COMPUTE_ACCOUNTING_DIRNAME}/) records the runs, evaluation
episodes, and GPU-hours of the paper's experiments: the default configuration,
the reruns with training and search seeds 1001 and 1002, the three additional
ETTh1 task settings, and the evaluations for the sensitivity checks. An overhead
audit covers the remaining recorded project compute.
[`scripts/compute_accounting_ledger.py`](../scripts/compute_accounting_ledger.py)
writes the folder's four CSV files in one read-only pass over the MLflow
experiments and analysis windows of
[`configs/reporting/compute_accounting.yaml`](../configs/reporting/compute_accounting.yaml).
Before it writes, the default configuration must reproduce the compute
accounting of version 0.1.0, which the script reads from that release. Running
the script needs access to the tracking server that holds the recorded runs.

## Bundle Contents

- `MANIFEST.json`: the machine-readable index of the bundle, with checksums,
  the dataset scope, source identifiers, generation commands, and the content
  that the bundle leaves out.
- `tables/`: the CSV analysis tables, one folder per evaluated setting.
  `{DEFAULT_CONFIGURATION_SOURCE_ID}/` holds the results of the default configuration. The
  other folders hold the sensitivity checks of evaluation data seeds, selector
  pressure, and fixed fractions of affected channels.
- `{PAPER_INPUTS_DIRNAME}/`: the CSV inputs of the paper's tables and figures, derived
  from the tables of the default configuration.
- `{SUPPLEMENTARY_TABLES_DIRNAME}/`: the hyperparameters selected on clean validation
  data, a diagnostic of near-ties in clean validation loss for PatchTST, and the
  architecture and method results for training and search seeds 42, 1001,
  and 1002. Further tables cover the checks of severity weighting, fault
  direction, timing, and the ETTh1 task. Its `README.md` describes each file.
- `{COMPUTE_ACCOUNTING_DIRNAME}/`: the compute accounting described above.
- `config/eval_context.json`: the evaluation context of the bundled analysis
  sources.
- `forecast_examples/manifest.json`: the sample identities of the curated
  forecast examples, with a record of which traces the bundle contains.
- `R/`: the scripts that reproduce the paper figures, together with their input
  CSVs, the curated forecast traces, and the rendered PDFs.

The artifact excludes checkpoints, MLflow run stores, raw and processed
datasets, complete outputs per sample, raw analysis configurations, forecast
traces beyond the curated R extracts, and complete export trees. The two
curated datasets are published as separate dataset records with their own
metadata (see `data/README.md` in the repository root).

## Paper Inputs

The files in `{PAPER_INPUTS_DIRNAME}/` are the small CSV inputs from which the
paper's tables and figures are built. The artifact builder derives them from the
tables of the default configuration and lists them in `MANIFEST.json` with row
counts, checksums, and source tables.

## Provenance

All paths needed to inspect the artifact are relative to this directory. Each
folder in `tables/` carries the name of its evaluated setting, which
`MANIFEST.json` uses as the source identifier and which stays fixed across
builds. For each source, the manifest also records the export date and the path
from which the build read the tables. These records serve to audit the build,
while inspecting the results needs only the directory tree described above.

## Verification

`MANIFEST.json` and `{SUPPLEMENTARY_TABLES_DIRNAME}/manifest.json` record the SHA-256
checksum and byte size of every bundled file.
""",
        encoding="utf-8",
    )
    return readme_path


def _write_eval_context(
    output_root: Path,
    contexts: dict[str, dict[str, Any]],
) -> Path:
    context_path = output_root / "config" / "eval_context.json"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(
        json.dumps(
            {
                "artifact_id": ARTIFACT_ID,
                "description": (
                    "Public-safe evaluation context derived from source "
                    "meta_analysis_args.yaml files. Private storage fields are "
                    "not shipped."
                ),
                "contexts": contexts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return context_path


def _compact_sample_spec(spec: ForecastExampleSpec) -> str:
    return (
        f"{spec.architecture}:{spec.scenario}:{spec.pert_idx}:"
        f"{spec.sample_id}:{spec.source_idx}:{spec.severity:.6f}"
    )


def _forecast_example_payload() -> dict[str, Any]:
    samples = []
    for spec in FORECAST_EXAMPLE_SPECS:
        samples.append(
            {
                "dataset": spec.dataset,
                "figure_group": spec.figure_group,
                "output_subdir": spec.output_subdir,
                "batch_size": spec.batch_size,
                "n_test_samples": FORECAST_EXAMPLE_N_TEST_SAMPLES,
                "eval_data_seed": spec.eval_data_seed,
                "architecture": spec.architecture,
                "robustness_method": "baseline",
                "pipeline_method": "baseline",
                "scenario": spec.scenario,
                "pert_idx": spec.pert_idx,
                "sample_id": spec.sample_id,
                "source_idx": spec.source_idx,
                "severity": spec.severity,
                "compact_sample": _compact_sample_spec(spec),
            }
        )

    return {
        "artifact_id": ARTIFACT_ID,
        "source_id": FORECAST_EXAMPLE_SOURCE_ID,
        "role": "curated qualitative forecast-example sample identities",
        "source_export_relative_path": FORECAST_EXAMPLE_SOURCE_EXPORT,
        "source_cli_relative_path": f"{FORECAST_EXAMPLE_SOURCE_EXPORT}/cli.md",
        "regeneration_script": "scripts/render_forecast_plots.py",
        "regeneration_mode": "from-runs",
        "paper_figures": [
            "figures/Figure_3_forecast_short_term.pdf",
            "figures/Figure_4_forecast_long_term.pdf",
        ],
        "curated_figure_trace_csvs_bundled": True,
        "full_trace_csvs_bundled": False,
        "replay_pdfs_bundled": False,
        "regeneration_note": (
            "Use these sample identities with render_forecast_plots.py from-runs "
            "against trained benchmark models, their MLflow runs, checkpoints, "
            "and processed datasets. The R subartifact bundles the curated trace "
            "extracts used by the plotted paper figures, while full forecast trace "
            "dumps are intentionally omitted."
        ),
        "samples": samples,
    }


def _write_forecast_example_manifest(output_root: Path) -> Path:
    manifest_path = output_root / FORECAST_EXAMPLE_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(_forecast_example_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _file_record(output_root: Path, path: Path, role: str) -> dict[str, Any]:
    _assert_public_file(path)
    return {
        "path": _artifact_relpath(output_root, path),
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _repo_relative_file_record(
    *,
    repo_root: Path,
    path: Path,
    role: str,
) -> dict[str, Any]:
    _assert_public_file(path)
    return {
        "path": path.relative_to(repo_root).as_posix(),
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _collect_optional_tree_records(
    *,
    repo_root: Path,
    directory: Path,
    role: str,
) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        records.append(
            _repo_relative_file_record(
                repo_root=repo_root,
                path=path,
                role=role,
            )
        )
    return records


def _compute_accounting_summary(
    *,
    output_root: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not records:
        return None
    compute_root = output_root / COMPUTE_ACCOUNTING_DIRNAME
    official_path = compute_root / "compute_ledger_official_runs.csv"
    episode_path = compute_root / "compute_evaluation_episodes.csv"
    overhead_path = compute_root / "compute_overhead_audit.csv"
    if not all(path.is_file() for path in (official_path, episode_path, overhead_path)):
        return None

    with official_path.open(newline="", encoding="utf-8") as handle:
        official_rows = list(csv.DictReader(handle))
    with episode_path.open(newline="", encoding="utf-8") as handle:
        episode_rows = list(csv.DictReader(handle))
    with overhead_path.open(newline="", encoding="utf-8") as handle:
        overhead_rows = list(csv.DictReader(handle))
    analysis_rows = [row for row in episode_rows if row["analysis"] != ""]

    def _hours(rows: list[dict[str, str]]) -> float:
        total = 0.0
        for row in rows:
            raw_value = str(row.get("evidence_duration_hours", "")).strip()
            if raw_value:
                total += float(raw_value)
        return round(total, 6)

    return {
        "root": COMPUTE_ACCOUNTING_DIRNAME,
        "ledger_script": "scripts/compute_accounting_ledger.py",
        "files": records,
        "official_ledger_rows": len(official_rows),
        "official_unique_run_ids": len({row["run_id"] for row in official_rows}),
        "overhead_rows": len(overhead_rows),
        "overhead_unique_run_ids": len({row["run_id"] for row in overhead_rows}),
        "official_duration_hours": _hours(official_rows),
        "overhead_duration_hours": _hours(overhead_rows),
        "evaluation_episode_rows": len(episode_rows),
        "analysis_evaluation_episodes": len(analysis_rows),
        "analysis_evaluation_hours": round(
            sum(float(row["span_seconds"]) for row in analysis_rows) / 3600.0,
            6,
        ),
        "duration_unit_note": (
            "Single-MIG-device hours under the one-device-per-job execution model."
        ),
    }


def _derived_paper_input_record(
    *,
    output_root: Path,
    path: Path,
    filename: str,
    source_id: str,
    row_count: int,
) -> dict[str, Any]:
    metadata = PAPER_INPUT_METADATA.get(filename)
    if metadata is None:
        raise ValueError(f"Missing derived paper-input metadata for {filename}.")
    _assert_public_file(path)
    source_tables = tuple(metadata["source_tables"])
    return {
        "path": _artifact_relpath(output_root, path),
        "role": metadata["role"],
        "source_id": source_id,
        "source_tables": list(source_tables),
        "source_paths": [
            f"tables/{source_id}/{table_name}" for table_name in source_tables
        ],
        "derivation_script": "scripts/generate_paper_figure_csvs.py",
        "row_count": row_count,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_derived_paper_inputs(
    *,
    output_root: Path,
    default_source_id: str,
) -> list[dict[str, Any]]:
    source_table_dir = output_root / "tables" / default_source_id
    table_frames: dict[str, pd.DataFrame] = {}
    for table_name in PAPER_INPUT_SOURCE_TABLES:
        table_path = source_table_dir / table_name
        if not table_path.is_file():
            raise FileNotFoundError(
                f"Missing primary source table for derived paper input: {table_path}"
            )
        table_frames[table_name] = pd.read_csv(table_path)

    derived_frames = build_figure_csv_frames_from_tables(
        table_frames,
        core_figures_config=DEFAULT_CORE_FIGURES_CONFIG,
        benchmark_scope_config=DEFAULT_BENCHMARK_SCOPE_CONFIG,
    )
    output_dir = output_root / PAPER_INPUTS_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for filename, frame in derived_frames.items():
        output_path = output_dir / filename
        frame.to_csv(output_path, index=False)
        records.append(
            _derived_paper_input_record(
                output_root=output_root,
                path=output_path,
                filename=filename,
                source_id=default_source_id,
                row_count=len(frame),
            )
        )
    return records


def _build_manifest(
    *,
    output_root: Path,
    source_root: Path,
    build_time_utc: str,
    contexts: dict[str, dict[str, Any]],
    table_records: list[dict[str, Any]],
    derived_paper_input_records: list[dict[str, Any]],
    csv_metadata_by_source: dict[str, dict[str, Any]],
    support_file_records: list[dict[str, Any]],
    supplementary_tables: dict[str, Any],
    supplementary_file_records: list[dict[str, Any]],
    r_file_records: list[dict[str, Any]],
    compute_file_records: list[dict[str, Any]],
) -> dict[str, Any]:
    default_context = contexts[DEFAULT_CONFIGURATION_SOURCE_ID]
    source_git = _git_metadata(PUBLIC_REPO_ROOT)

    source_exports = []
    for spec in SOURCE_SPECS:
        context = contexts[spec.source_id]
        source_exports.append(
            {
                "source_id": spec.source_id,
                "role": spec.role,
                "export_date": spec.export_date,
                "source_export_relative_path": spec.relative_export_path,
                "generation_command": context["generation_command"],
                "improvement_selection_mode": context[
                    "improvement_selection_mode"
                ],
                "fixed_channel_fraction": context.get("fixed_channel_fraction"),
                "tables": list(spec.table_files),
                "csv_metadata": _freeze_sets(
                    csv_metadata_by_source.get(spec.source_id, {})
                ),
            }
        )

    files = sorted(
        [
            *support_file_records,
            *table_records,
            *derived_paper_input_records,
            *supplementary_file_records,
        ],
        key=lambda record: record["path"],
    )
    manifest = {
        "artifact_id": ARTIFACT_ID,
        "artifact_version": 2,
        "built_at_utc": build_time_utc,
        "source_repository": {
            "name": "SensorFault-Bench",
            "git_commit": source_git["commit"],
        },
        "default_configuration_source_id": DEFAULT_CONFIGURATION_SOURCE_ID,
        "dataset_scope": default_context["data_files"],
        "test_metric": default_context["test_metric"],
        "n_test_samples": default_context["n_test_samples"],
        "improvement_selection_mode": default_context[
            "improvement_selection_mode"
        ],
        "perturbation_scenarios": default_context["perturbation_scenarios"],
        "robustness_parameters": {
            "phi": None,
            "cvar_alpha": None,
            "note": (
                "The submitted degradation and fault-time-error exports do not "
                "use phi or CVaR-alpha reporting parameters."
            ),
        },
        "data_config_signatures": _freeze_sets(
            csv_metadata_by_source[DEFAULT_CONFIGURATION_SOURCE_ID][
                "data_config_signatures"
            ]
        ),
        "source_exports": source_exports,
        "derived_paper_inputs": sorted(
            derived_paper_input_records,
            key=lambda record: record["path"],
        ),
        "supplementary_tables": {
            "root": SUPPLEMENTARY_TABLES_DIRNAME,
            "manifest_path": (
                f"{SUPPLEMENTARY_TABLES_DIRNAME}/{SUPPLEMENTARY_TABLES_MANIFEST}"
            ),
            "training_seeds": supplementary_tables["training_seeds"],
            "datasets": supplementary_tables["datasets"],
            "architectures": supplementary_tables["architectures"],
            "robustness_methods": supplementary_tables["robustness_methods"],
            "selection_metric": supplementary_tables["selection_metric"],
            "evaluation_windows_per_scenario": supplementary_tables[
                "evaluation_windows_per_scenario"
            ],
            "files": [
                {
                    **record,
                    "path": f"{SUPPLEMENTARY_TABLES_DIRNAME}/{record['path']}",
                }
                for record in supplementary_tables["files"]
            ],
        },
        "forecast_examples": {
            "manifest_path": FORECAST_EXAMPLE_MANIFEST_PATH,
            "source_id": FORECAST_EXAMPLE_SOURCE_ID,
            "source_export_relative_path": FORECAST_EXAMPLE_SOURCE_EXPORT,
            "sample_count": len(FORECAST_EXAMPLE_SPECS),
            "curated_figure_trace_csvs_bundled": True,
            "full_trace_csvs_bundled": False,
            "replay_pdfs_bundled": False,
        },
        "files": files,
        "intentional_omissions": [
            "MLflow run stores",
            "model checkpoints",
            "raw datasets",
            "processed runtime datasets",
            "tables/scenario_samples.csv sample-level dumps",
            "tables/forecast_extremes.csv replay-selection table",
            "raw config/meta_analysis_args.yaml files with private storage fields",
            "generic generated figure binaries outside paper_artifact/R/figures",
            "R package caches and temporary TeX figure build products",
            "full qualitative forecast trace CSV dumps outside the curated R figure extracts",
            "forecast replay PDFs",
            "complete historical export trees",
        ],
        "separate_dataset_metadata_route": (
            "Croissant JSON records for BeijingAir_Tiantan and "
            "Penmanshiel_Hourly_WT08 are uploaded through the dataset metadata "
            "route rather than bundled in this result manifest."
        ),
    }
    if r_file_records:
        manifest["r_figure_artifact"] = {
            "root": "R",
            "file_count": len(r_file_records),
            "files": r_file_records,
        }
    compute_summary = _compute_accounting_summary(
        output_root=output_root,
        records=compute_file_records,
    )
    if compute_summary is not None:
        manifest["compute_accounting"] = compute_summary
    return manifest


def build_paper_artifact(
    *,
    source_root: Path,
    supplementary_source_dir: Path,
    output_root: Path,
    build_time_utc: str,
    force: bool,
) -> None:
    """Build the bundle in a staging directory and replace ``output_root`` only on success.

    A failed build leaves an existing bundle untouched. A forced rebuild
    carries the existing ``R/`` and ``compute_accounting/`` subtrees over
    unchanged, because other tools write them.
    """
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")
    if output_root.exists() and not force:
        raise FileExistsError(
            f"{output_root} already exists. Pass --force to replace it."
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    # Staging beside the output keeps the final swap a same-filesystem rename.
    staging_parent = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-",
            dir=output_root.parent,
        )
    )
    try:
        staging_root = staging_parent / output_root.name
        staging_root.mkdir()
        for subtree in CARRIED_OVER_SUBTREES:
            existing_tree = output_root / subtree
            if existing_tree.exists():
                shutil.copytree(existing_tree, staging_root / subtree)
        _write_paper_artifact(
            source_root=source_root,
            supplementary_source_dir=supplementary_source_dir,
            output_root=staging_root,
            build_time_utc=build_time_utc,
        )
        _replace_output_root(staging_root=staging_root, output_root=output_root)
    finally:
        shutil.rmtree(staging_parent)


def _replace_output_root(*, staging_root: Path, output_root: Path) -> None:
    if not output_root.exists():
        staging_root.rename(output_root)
        return
    retired_parent = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.previous-",
            dir=output_root.parent,
        )
    )
    retired_root = retired_parent / output_root.name
    output_root.rename(retired_root)
    try:
        staging_root.rename(output_root)
    except BaseException:
        retired_root.rename(output_root)
        retired_parent.rmdir()
        raise
    shutil.rmtree(retired_parent)


def _write_paper_artifact(
    *,
    source_root: Path,
    supplementary_source_dir: Path,
    output_root: Path,
    build_time_utc: str,
) -> None:
    contexts: dict[str, dict[str, Any]] = {}
    table_records: list[dict[str, Any]] = []
    csv_metadata_by_source: dict[str, dict[str, Any]] = {}

    for spec in SOURCE_SPECS:
        source_dir = source_root / spec.relative_export_path
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing source export: {source_dir}")
        contexts[spec.source_id] = _sanitize_context(
            source_dir=source_dir,
            source_root=source_root,
            spec=spec,
        )
        csv_metadata_by_source[spec.source_id] = {}
        for table_name in spec.table_files:
            record = _copy_table(
                source_root=source_root,
                source_dir=source_dir,
                output_root=output_root,
                spec=spec,
                table_name=table_name,
            )
            table_records.append(record)
            _merge_csv_metadata(
                csv_metadata_by_source[spec.source_id],
                _collect_table_metadata(output_root / record["path"]),
            )

    readme_path = _write_readme(output_root)
    context_path = _write_eval_context(output_root, contexts)
    forecast_manifest_path = _write_forecast_example_manifest(output_root)
    derived_paper_input_records = _write_derived_paper_inputs(
        output_root=output_root,
        default_source_id=DEFAULT_CONFIGURATION_SOURCE_ID,
    )
    supplementary_root = output_root / SUPPLEMENTARY_TABLES_DIRNAME
    supplementary_tables = build_supplementary_tables(
        source_dir=supplementary_source_dir,
        repo_root=PUBLIC_REPO_ROOT,
        output_root=supplementary_root,
    )
    supplementary_file_records = [
        _file_record(
            output_root,
            path,
            "supplementary table file",
        )
        for path in sorted(supplementary_root.iterdir())
        if path.is_file()
    ]
    support_file_records = [
        _file_record(output_root, readme_path, "paper artifact documentation"),
        _file_record(
            output_root,
            context_path,
            "public-safe source evaluation context",
        ),
        _file_record(
            output_root,
            forecast_manifest_path,
            "curated qualitative forecast-example sample manifest",
        ),
    ]
    r_file_records = _collect_optional_tree_records(
        repo_root=output_root,
        directory=output_root / "R",
        role="R figure reproduction subartifact file",
    )
    compute_file_records = _collect_optional_tree_records(
        repo_root=output_root,
        directory=output_root / COMPUTE_ACCOUNTING_DIRNAME,
        role="compute-accounting evidence file",
    )

    manifest = _build_manifest(
        output_root=output_root,
        source_root=source_root,
        build_time_utc=build_time_utc,
        contexts=contexts,
        table_records=table_records,
        derived_paper_input_records=derived_paper_input_records,
        csv_metadata_by_source=csv_metadata_by_source,
        support_file_records=support_file_records,
        supplementary_tables=supplementary_tables,
        supplementary_file_records=supplementary_file_records,
        r_file_records=r_file_records,
        compute_file_records=compute_file_records,
    )
    manifest_path = output_root / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _assert_public_file(manifest_path)


def main() -> None:
    args = _parse_args()
    build_time_utc = args.build_time_utc
    if build_time_utc is None:
        build_time_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    build_paper_artifact(
        source_root=args.source_root,
        supplementary_source_dir=args.supplementary_source_dir,
        output_root=args.output,
        build_time_utc=build_time_utc,
        force=args.force,
    )


if __name__ == "__main__":
    main()
