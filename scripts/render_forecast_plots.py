#!/usr/bin/env python3
"""Regenerate forecast plots for selected benchmark samples from MLflow runs."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.parsing import (  # noqa: E402
    build_mlflow_tracking_uri,
    parse_required_bool,
    parse_required_nonempty_string,
    parse_required_nonnegative_int,
    parse_required_positive_int,
    parse_required_unit_interval_float,
    require_dataframe_columns,
    require_namespace_value,
    resolve_dataset_window_args,
    resolve_effective_eval_data_seed,
)
from visualizations.plots import plot_forecast_extreme  # noqa: E402


SAMPLE_SPEC_HELP = (
    "ARCH:SCENARIO:PERT_IDX:SAMPLE_ID:SOURCE_IDX:SEVERITY[:SAMPLE_SCORE]"
)
FORECAST_EXTREME_COLUMNS = {
    "dataset",
    "run_id",
    "model_architecture",
    "backbone_architecture",
    "robustness_method",
    "pipeline_method",
    "pipeline_kind",
    "pipeline_id",
    "sample_id",
    "source_sample_idx",
    "pert_idx",
    "scenario",
    "severity",
    "sample_score",
    "score_metric",
}
SCENARIO_SAMPLE_COLUMNS = {
    "dataset",
    "run_id",
    "model_architecture",
    "backbone_architecture",
    "robustness_method",
    "pipeline_method",
    "pipeline_id",
    "sample_id",
    "source_sample_idx",
    "pert_idx",
    "scenario",
    "severity",
    "err_pert",
}
OUTPUT_FORMATS = ("pdf", "html", "png")
TRACE_RENDER_MODES = ("full", "single-feature")
TRACE_SEGMENTS = {"input", "forecast"}
REQUIRED_TRACE_COLUMNS = {
    "dataset",
    "arch",
    "robustness_method",
    "scenario",
    "pert_idx",
    "sample_id",
    "source_sample_idx",
    "severity",
    "score_metric",
    "clean_sample_score",
    "perturbed_sample_score",
    "segment",
    "time_index",
    "time_step",
    "feature",
    "affected_feature",
    "clean_input",
    "perturbed_input",
    "ground_truth",
    "clean_prediction",
    "perturbed_prediction",
}


def _add_shared_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--format",
        choices=OUTPUT_FORMATS,
        default="pdf",
        help="Output figure format.",
    )
    parser.add_argument("--index", type=Path, default=None)


def _add_individual_render_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--individual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write individual sample figures.",
    )


def _add_from_runs_args(
    parser: argparse.ArgumentParser,
    defaults: Mapping[str, Any],
) -> None:
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", nargs="+", default=None)
    parser.add_argument("--method", nargs="+", default=None)
    parser.add_argument("--pipeline-id", nargs="+", default=None)
    parser.add_argument("--run-id", nargs="+", default=None)
    parser.add_argument(
        "--tracking-uri",
        default=defaults["LOGDIR"],
        help="MLflow tracking URI or local run directory.",
    )
    parser.add_argument("--minio-endpoint", default=defaults["MINIO_ENDPOINT"])
    parser.add_argument(
        "--experiment-prefix",
        default=defaults["MLFLOW_EXPERIMENT_PREFIX"],
    )
    parser.add_argument("--data-root", default=defaults["DATA_ROOT"])
    parser.add_argument("--test-metric", default=defaults["TEST_METRIC"])
    parser.add_argument("--precision", default=defaults["PRECISION"])
    parser.add_argument("--input-len", type=int, default=None)
    parser.add_argument("--target-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-test-samples", type=int, default=defaults["N_TEST_SAMPLES"])
    parser.add_argument("--eval-data-seed", type=int, default=defaults["EVAL_DATA_SEED"])
    parser.add_argument("--accelerator", default=defaults["ACCELERATOR"])
    parser.add_argument("--devices", default=defaults["DEVICES"])
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        metavar=SAMPLE_SPEC_HELP,
        help="Sample identity to render. May be repeated.",
    )
    parser.add_argument(
        "--samples-csv",
        type=Path,
        default=None,
        help=(
            "CSV with dataset, arch/model_architecture, scenario, pert_idx, "
            "sample_id, source_sample_idx/source_idx, severity, and optional "
            "run_id, sample_score, eval_data_seed, n_test_samples, and batch_size."
        ),
    )
    parser.add_argument(
        "--scenario-samples-csv",
        type=Path,
        default=None,
        help="Existing tables/scenario_samples.csv from run_analysis.py.",
    )
    parser.add_argument(
        "--forecast-extremes-csv",
        type=Path,
        default=None,
        help="Optional existing tables/forecast_extremes.csv subset from run_analysis.py.",
    )
    parser.add_argument(
        "--meta-analysis-run-id",
        default=None,
        help="MLflow meta-analysis run id containing tables/scenario_samples.csv.",
    )
    parser.add_argument("--scenario", nargs="+", default=None)
    parser.add_argument("--sample-id", nargs="+", type=int, default=None)
    parser.add_argument("--pert-idx", nargs="+", type=int, default=None)
    parser.add_argument("--extreme-kind", nargs="+", choices=("best", "worst"), default=None)
    parser.add_argument("--extreme-rank", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--export-traces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write per-sample forecast trace CSVs.",
    )


def _add_from_traces_args(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(format="html")
    parser.add_argument(
        "--trace-csv",
        action="append",
        type=Path,
        default=[],
        help="One forecast trace CSV to render. May be repeated.",
    )
    parser.add_argument(
        "--trace-root",
        action="append",
        type=Path,
        default=[],
        help="Directory searched recursively for forecast trace CSVs. May be repeated.",
    )
    parser.add_argument(
        "--mode",
        choices=TRACE_RENDER_MODES,
        default="full",
        help=(
            "Use full to recreate exported forecast plots from all trace channels, "
            "or single-feature for compact one-feature inspection."
        ),
    )
    parser.add_argument(
        "--preserve-layout",
        action="store_true",
        help=(
            "Preserve the result-bundle trace layout under output-dir, dropping "
            "the intermediate traces/ directory."
        ),
    )
    parser.add_argument(
        "--missing-robustness-method",
        default=None,
        help=(
            "Explicit method label for older trace CSVs that do not contain a "
            "robustness_method column."
        ),
    )


def _build_cli_parser(defaults: Mapping[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render benchmark forecast plots from MLflow runs or exported trace CSVs."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    from_runs = subparsers.add_parser(
        "from-runs",
        help="Regenerate plots from tested MLflow runs and checkpoints.",
    )
    _add_shared_render_args(from_runs)
    _add_individual_render_arg(from_runs)
    _add_from_runs_args(from_runs, defaults)

    from_traces = subparsers.add_parser(
        "from-traces",
        help="Replay exported forecast trace CSVs without MLflow or checkpoints.",
    )
    _add_shared_render_args(from_traces)
    _add_from_traces_args(from_traces)
    return parser


def parse_cli_args(
    argv: Sequence[str] | None = None,
    *,
    defaults: Mapping[str, Any] | None = None,
) -> Namespace:
    if defaults is None:
        from config_loader import load_defaults

        resolved_defaults = load_defaults()
    else:
        resolved_defaults = defaults
    parser = _build_cli_parser(resolved_defaults)
    cli_args = parser.parse_args(argv)
    if cli_args.command == "from-traces":
        if not cli_args.trace_csv and not cli_args.trace_root:
            parser.error("from-traces requires at least one --trace-csv or --trace-root")
        return cli_args

    if cli_args.sample and cli_args.dataset is None:
        parser.error("--dataset is required when using --sample")
    if (
        not cli_args.sample
        and cli_args.samples_csv is None
        and cli_args.scenario_samples_csv is None
        and cli_args.forecast_extremes_csv is None
        and cli_args.meta_analysis_run_id is None
    ):
        parser.error(
            "at least one --sample, --samples-csv, --forecast-extremes-csv, "
            "--scenario-samples-csv, or --meta-analysis-run-id is required"
        )
    if (cli_args.input_len is None) != (cli_args.target_len is None):
        parser.error("--input-len and --target-len must be provided together")
    if cli_args.limit is not None:
        parse_required_positive_int(cli_args.limit, key="limit")
    for key in ("sample_id", "pert_idx"):
        values = getattr(cli_args, key)
        if values is None:
            continue
        for value in values:
            parse_required_nonnegative_int(value, key=key)
    if cli_args.extreme_rank is not None:
        parse_required_positive_int(cli_args.extreme_rank, key="extreme_rank")
    if (
        (cli_args.extreme_kind is not None or cli_args.extreme_rank is not None)
        and cli_args.forecast_extremes_csv is None
    ):
        parser.error("--extreme-kind and --extreme-rank require --forecast-extremes-csv")
    if (
        not cli_args.individual
        and not cli_args.export_traces
    ):
        parser.error(
            "from-runs --no-individual requires --export-traces so at least one "
            "output is produced"
        )
    return cli_args


def _first_present(row: Mapping[str, Any], keys: Sequence[str], *, context: str) -> Any:
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if pd.isna(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    raise ValueError(f"{context} is missing required key(s): {list(keys)}.")


def _optional_present(row: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if pd.isna(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _optional_filter_values(values: Sequence[str] | None, *, key: str) -> set[str] | None:
    if values is None:
        return None
    parsed = {
        parse_required_nonempty_string(value, key=key)
        for value in values
    }
    if not parsed:
        raise ValueError(f"{key} must contain at least one value when provided.")
    return parsed


def _parse_sample_spec(value: str, *, index: int, dataset: str | None) -> dict[str, Any]:
    raw_value = str(value).strip()
    delimiter = "," if "," in raw_value else ":"
    parts = [part.strip() for part in raw_value.split(delimiter)]
    if len(parts) not in (6, 7):
        raise ValueError(
            f"--sample #{index + 1} must have 6 or 7 fields: {SAMPLE_SPEC_HELP}."
        )
    sample: dict[str, Any] = {
        "dataset": dataset,
        "arch": parts[0],
        "scenario": parts[1],
        "pert_idx": parts[2],
        "sample_id": parts[3],
        "source_sample_idx": parts[4],
        "severity": parts[5],
    }
    if len(parts) == 7:
        sample["sample_score"] = parts[6]
    return _normalize_sample(sample, index=index)


def _normalize_sample(
    sample: Mapping[str, Any],
    *,
    index: int,
    default_dataset: str | None = None,
) -> dict[str, Any]:
    context = f"sample[{index}]"
    dataset = _optional_present(sample, ("dataset",))
    if dataset is None:
        dataset = default_dataset
    if dataset is None:
        raise ValueError(f"{context} is missing required dataset.")
    source_idx = _first_present(
        sample,
        ("source_sample_idx", "source_idx"),
        context=context,
    )
    normalized: dict[str, Any] = {
        "dataset": parse_required_nonempty_string(dataset, key=f"{context}.dataset"),
        "arch": parse_required_nonempty_string(
            _first_present(
                sample,
                ("arch", "model_architecture", "backbone_architecture", "model"),
                context=context,
            ),
            key=f"{context}.arch",
        ),
        "scenario": parse_required_nonempty_string(
            _first_present(sample, ("scenario",), context=context),
            key=f"{context}.scenario",
        ),
        "pert_idx": parse_required_nonnegative_int(
            _first_present(sample, ("pert_idx",), context=context),
            key=f"{context}.pert_idx",
        ),
        "sample_id": parse_required_nonnegative_int(
            _first_present(sample, ("sample_id",), context=context),
            key=f"{context}.sample_id",
        ),
        "source_sample_idx": parse_required_nonnegative_int(
            source_idx,
            key=f"{context}.source_sample_idx",
        ),
        "severity": parse_required_unit_interval_float(
            _first_present(sample, ("severity",), context=context),
            key=f"{context}.severity",
        ),
        "order": index,
    }
    for key in ("sample_score",):
        value = _optional_present(sample, (key,))
        if value is not None:
            normalized[key] = float(value)
    for key in (
        "run_id",
        "pipeline_id",
        "pipeline_kind",
        "robustness_method",
        "pipeline_method",
    ):
        value = _optional_present(sample, (key,))
        if value is not None:
            normalized[key] = parse_required_nonempty_string(value, key=f"{context}.{key}")
    for key in ("eval_data_seed",):
        value = _optional_present(sample, (key,))
        if value is not None:
            normalized[key] = parse_required_nonnegative_int(value, key=f"{context}.{key}")
    for key in ("n_test_samples", "batch_size"):
        value = _optional_present(sample, (key,))
        if value is not None:
            normalized[key] = parse_required_positive_int(value, key=f"{context}.{key}")
    return normalized


def _load_samples_csv(path: Path, *, offset: int, default_dataset: str | None) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    return [
        _normalize_sample(row, index=offset + index, default_dataset=default_dataset)
        for index, row in enumerate(frame.to_dict(orient="records"))
    ]


def _optional_int_filter_values(
    values: Sequence[int] | None,
    *,
    key: str,
) -> set[int] | None:
    if values is None:
        return None
    parsed = {
        parse_required_nonnegative_int(value, key=key)
        for value in values
    }
    if not parsed:
        raise ValueError(f"{key} must contain at least one value when provided.")
    return parsed


def _passes_sample_filters(
    row: Mapping[str, Any],
    cli_args: Namespace,
    *,
    include_extreme_filters: bool,
) -> bool:
    dataset_filter = _optional_filter_values(
        [cli_args.dataset] if cli_args.dataset is not None else None,
        key="dataset",
    )
    model_filter = _optional_filter_values(cli_args.model, key="model")
    method_filter = _optional_filter_values(cli_args.method, key="method")
    pipeline_filter = _optional_filter_values(cli_args.pipeline_id, key="pipeline_id")
    run_filter = _optional_filter_values(cli_args.run_id, key="run_id")
    scenario_filter = _optional_filter_values(cli_args.scenario, key="scenario")
    sample_id_filter = _optional_int_filter_values(cli_args.sample_id, key="sample_id")
    pert_idx_filter = _optional_int_filter_values(cli_args.pert_idx, key="pert_idx")
    if dataset_filter is not None and str(row["dataset"]) not in dataset_filter:
        return False
    model_values = {
        str(row.get("model_architecture", "")).strip(),
        str(row.get("backbone_architecture", "")).strip(),
    }
    if model_filter is not None and model_values.isdisjoint(model_filter):
        return False
    method_values = {
        str(row.get("robustness_method", "")).strip(),
        str(row.get("pipeline_method", "")).strip(),
    }
    if method_filter is not None and method_values.isdisjoint(method_filter):
        return False
    if pipeline_filter is not None and str(row.get("pipeline_id", "")).strip() not in pipeline_filter:
        return False
    if run_filter is not None and str(row.get("run_id", "")).strip() not in run_filter:
        return False
    if scenario_filter is not None and str(row.get("scenario", "")).strip() not in scenario_filter:
        return False
    if sample_id_filter is not None:
        sample_id = parse_required_nonnegative_int(row.get("sample_id"), key="sample_id")
        if sample_id not in sample_id_filter:
            return False
    if pert_idx_filter is not None:
        pert_idx = parse_required_nonnegative_int(row.get("pert_idx"), key="pert_idx")
        if pert_idx not in pert_idx_filter:
            return False
    if include_extreme_filters and cli_args.extreme_kind is not None:
        allowed_kinds = set(cli_args.extreme_kind)
        if str(row.get("extreme_kind", "")).strip() not in allowed_kinds:
            return False
    if include_extreme_filters and cli_args.extreme_rank is not None:
        rank = parse_required_positive_int(row.get("extreme_rank"), key="extreme_rank")
        if rank != int(cli_args.extreme_rank):
            return False
    return True


def _load_forecast_extremes_samples(
    path: Path,
    *,
    cli_args: Namespace,
    offset: int,
) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    require_dataframe_columns(
        frame,
        FORECAST_EXTREME_COLUMNS,
        context=f"{path} forecast_extremes.csv",
    )
    selected_rows = [
        row
        for row in frame.to_dict(orient="records")
        if _passes_sample_filters(row, cli_args, include_extreme_filters=True)
    ]
    if cli_args.limit is not None:
        selected_rows = selected_rows[: int(cli_args.limit)]
    if not selected_rows:
        raise ValueError(f"{path} did not contain any forecast-extreme rows after filters.")
    metric_values = {
        parse_required_nonempty_string(row["score_metric"], key="score_metric")
        for row in selected_rows
    }
    expected_metric = parse_required_nonempty_string(cli_args.test_metric, key="test_metric")
    if metric_values != {expected_metric}:
        raise ValueError(
            "Selected forecast-extreme rows use score_metric values "
            f"{sorted(metric_values)}, but --test-metric is {expected_metric!r}."
        )
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows):
        sample = {
            **row,
            "arch": _first_present(
                row,
                ("backbone_architecture", "model_architecture"),
                context=f"forecast_extremes[{index}]",
            ),
        }
        samples.append(_normalize_sample(sample, index=offset + index))
    return samples


def _load_scenario_samples(
    path: Path,
    *,
    cli_args: Namespace,
    offset: int,
) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    require_dataframe_columns(
        frame,
        SCENARIO_SAMPLE_COLUMNS,
        context=f"{path} scenario_samples.csv",
    )
    selected_rows = [
        row
        for row in frame.to_dict(orient="records")
        if _passes_sample_filters(row, cli_args, include_extreme_filters=False)
    ]
    if cli_args.limit is not None:
        selected_rows = selected_rows[: int(cli_args.limit)]
    if not selected_rows:
        raise ValueError(f"{path} did not contain any scenario-sample rows after filters.")
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows):
        sample = {
            **row,
            "arch": _first_present(
                row,
                ("backbone_architecture", "model_architecture"),
                context=f"scenario_samples[{index}]",
            ),
            "sample_score": _first_present(
                row,
                ("err_pert",),
                context=f"scenario_samples[{index}]",
            ),
        }
        samples.append(_normalize_sample(sample, index=offset + index))
    return samples


def _download_meta_analysis_table(
    client: Any,
    *,
    run_id: str,
    artifact_path: str,
    tmpdir: str,
) -> Path:
    downloaded = client.download_artifacts(
        run_id,
        artifact_path,
        dst_path=tmpdir,
    )
    path = Path(downloaded)
    if not path.is_file():
        raise FileNotFoundError(
            f"Meta-analysis run {run_id!r} does not have {artifact_path}."
        )
    return path


def _load_selected_samples(
    cli_args: Namespace,
    *,
    client: Any,
) -> list[dict[str, Any]]:
    direct_samples = [
        _parse_sample_spec(value, index=index, dataset=cli_args.dataset)
        for index, value in enumerate(cli_args.sample)
    ]
    if direct_samples and cli_args.run_id is not None:
        if len(cli_args.run_id) != 1:
            raise ValueError("--sample with --run-id requires exactly one run id.")
        run_id = parse_required_nonempty_string(cli_args.run_id[0], key="run_id")
        for sample in direct_samples:
            sample["run_id"] = run_id
    samples = direct_samples
    if cli_args.samples_csv is not None:
        samples.extend(
            _load_samples_csv(
                cli_args.samples_csv,
                offset=len(samples),
                default_dataset=cli_args.dataset,
            )
        )
    if cli_args.scenario_samples_csv is not None:
        samples.extend(
            _load_scenario_samples(
                cli_args.scenario_samples_csv,
                cli_args=cli_args,
                offset=len(samples),
            )
        )
    if cli_args.forecast_extremes_csv is not None:
        samples.extend(
            _load_forecast_extremes_samples(
                cli_args.forecast_extremes_csv,
                cli_args=cli_args,
                offset=len(samples),
            )
        )
    if cli_args.meta_analysis_run_id is not None:
        with tempfile.TemporaryDirectory(prefix="scenario-samples-") as tmpdir:
            table_path = _download_meta_analysis_table(
                client,
                run_id=parse_required_nonempty_string(
                    cli_args.meta_analysis_run_id,
                    key="meta_analysis_run_id",
                ),
                artifact_path="tables/scenario_samples.csv",
                tmpdir=tmpdir,
            )
            samples.extend(
                _load_scenario_samples(
                    table_path,
                    cli_args=cli_args,
                    offset=len(samples),
                )
            )
    if not samples:
        raise ValueError("No samples were selected for forecast plot rendering.")
    return samples


def _build_runtime_args(defaults: Mapping[str, Any], cli_args: Namespace) -> Namespace:
    return Namespace(
        logdir=cli_args.tracking_uri,
        minio_endpoint=cli_args.minio_endpoint,
        mlflow_experiment_prefix=cli_args.experiment_prefix,
        data_root=cli_args.data_root,
        test_metric=parse_required_nonempty_string(cli_args.test_metric, key="test_metric"),
        precision=cli_args.precision,
        input_len=defaults["INPUT_LEN"],
        target_len=defaults["TARGET_LEN"],
        batch_size=defaults["BATCH_SIZE"],
        n_train_samples=defaults["N_TRAIN_SAMPLES"],
        n_val_samples=defaults["N_VAL_SAMPLES"],
        n_test_samples=parse_required_positive_int(
            cli_args.n_test_samples,
            key="n_test_samples",
        ),
        train_split=defaults["TRAIN_SPLIT"],
        val_split=defaults["VAL_SPLIT"],
        purged_fraction=defaults["PURGED_FRACTION"],
        shuffle_batches_before_split=defaults["SHUFFLE_BATCHES_BEFORE_SPLIT"],
        strict_iid=defaults["STRICT_IID"],
        perturbation_channel_fraction_max=defaults["PERTURBATION_CHANNEL_FRACTION_MAX"],
        perturbation_scenarios=defaults["PERTURBATION_SCENARIOS"],
        eval_data_seed=cli_args.eval_data_seed,
        full_coverage=defaults["FULL_COVERAGE"],
        bootstrap_ci_resamples=defaults["BOOTSTRAP_CI_RESAMPLES"],
        bootstrap_ci_confidence_level=defaults["BOOTSTRAP_CI_CONFIDENCE_LEVEL"],
        accelerator=cli_args.accelerator,
        devices=cli_args.devices,
        num_workers=0,
    )


def _select_dataset_spec(defaults: Mapping[str, Any], *, dataset: str, data_root: str):
    from data.datasets import resolve_with_defaults

    dataset_specs = resolve_with_defaults(
        defaults["DATA_FILES"],
        defaults["DATA_TARGETS"],
        data_root=data_root,
    )
    for spec in dataset_specs:
        if str(spec.key) == dataset:
            return spec
    raise ValueError(f"Dataset spec for {dataset!r} not found in configs/defaults.yaml.")


def _find_baseline_winner(
    client: Any,
    *,
    experiment_id: str,
    arch: str,
    args: Namespace,
):
    from mlflow.entities import ViewType
    from pipelines.selection import is_fully_tested
    from pipelines.training import search_runs_all

    filter_str = (
        "tags.pipeline_id = 'baseline' "
        "AND tags.pipeline_method = 'baseline' "
        "AND tags.robustness_method = 'baseline' "
        "AND tags.pipeline_kind = 'train' "
        f"AND tags.model_architecture = '{arch}' "
        "AND tags.best_model = 'true' "
        "AND params.tested = 'true'"
    )
    runs = search_runs_all(
        client,
        [experiment_id],
        filter_string=filter_str,
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    parent_runs = [
        run
        for run in runs
        if "mlflow.parentRunId" not in run.data.tags and run.info.status == "FINISHED"
    ]
    current_runs = [
        run
        for run in parent_runs
        if is_fully_tested(run, args=args, client=client)
    ]
    if len(current_runs) != 1:
        raise ValueError(
            f"Expected exactly one current tested baseline winner for {arch}; "
            f"found {len(current_runs)} among {len(parent_runs)} candidates."
        )
    return current_runs[0]


def _require_run_tag(run, key: str) -> str:
    return parse_required_nonempty_string(
        run.data.tags.get(key),
        key=key,
        context=f"Run {run.info.run_id}",
    )


def _require_matching_run_tag(
    run,
    *,
    tag_key: str,
    expected_value: Any,
    sample_key: str,
) -> None:
    actual_value = _require_run_tag(run, tag_key)
    expected = parse_required_nonempty_string(
        expected_value,
        key=sample_key,
        context=f"Sample for run {run.info.run_id}",
    )
    if actual_value != expected:
        raise ValueError(
            f"Run {run.info.run_id!r} {tag_key} tag {actual_value!r} does not "
            f"match sample {sample_key} {expected!r}."
        )


def _require_run_matches_sample(run, sample: Mapping[str, Any]) -> None:
    _require_matching_run_tag(
        run,
        tag_key="dataset",
        expected_value=sample["dataset"],
        sample_key="dataset",
    )
    _require_matching_run_tag(
        run,
        tag_key="model_architecture",
        expected_value=sample["arch"],
        sample_key="arch",
    )
    expected_method = sample.get("robustness_method")
    if expected_method is None:
        expected_method = sample.get("pipeline_method", "baseline")
    _require_matching_run_tag(
        run,
        tag_key="robustness_method",
        expected_value=expected_method,
        sample_key="robustness_method",
    )
    for sample_key, tag_key in (
        ("pipeline_method", "pipeline_method"),
        ("pipeline_kind", "pipeline_kind"),
        ("pipeline_id", "pipeline_id"),
    ):
        expected_value = sample.get(sample_key)
        if expected_value is None:
            continue
        _require_matching_run_tag(
            run,
            tag_key=tag_key,
            expected_value=expected_value,
            sample_key=sample_key,
        )


def _experiment_id_for_dataset(
    client: Any,
    *,
    experiment_prefix: str,
    dataset: str,
) -> str:
    experiment_name = f"{experiment_prefix}-{dataset}"
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_name!r} not found.")
    return experiment.experiment_id


def _resolve_run_for_sample(
    client: Any,
    *,
    sample: Mapping[str, Any],
    args: Namespace,
    experiment_id_cache: dict[str, str],
    run_cache: dict[tuple[str, str], Any],
) -> Any:
    from pipelines.selection import is_fully_tested

    run_id = sample.get("run_id")
    if run_id is not None:
        run = client.get_run(str(run_id))
        if run.info.status != "FINISHED":
            raise ValueError(f"Run {run_id!r} is not FINISHED.")
        if not is_fully_tested(run, args=args, client=client):
            raise ValueError(f"Run {run_id!r} is not current and fully tested.")
        _require_run_matches_sample(run, sample)
        return run

    method = _method_for_sample(sample)
    if method != "baseline":
        raise ValueError(
            "Samples without run_id can only resolve baseline winners. "
            f"Received robustness_method={method!r}."
        )
    dataset = str(sample["dataset"])
    arch = str(sample["arch"])
    cache_key = (dataset, arch)
    cached = run_cache.get(cache_key)
    if cached is not None:
        return cached
    experiment_id = experiment_id_cache.get(dataset)
    if experiment_id is None:
        experiment_id = _experiment_id_for_dataset(
            client,
            experiment_prefix=str(args.mlflow_experiment_prefix),
            dataset=dataset,
        )
        experiment_id_cache[dataset] = experiment_id
    run = _find_baseline_winner(
        client,
        experiment_id=experiment_id,
        arch=arch,
        args=args,
    )
    run_cache[cache_key] = run
    return run


def _group_samples_by_run(
    samples: Sequence[Mapping[str, Any]],
    *,
    client: Any,
    args: Namespace,
) -> dict[tuple[str, str], dict[str, Any]]:
    experiment_id_cache: dict[str, str] = {}
    run_cache: dict[tuple[str, str], Any] = {}
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for sample in samples:
        run = _resolve_run_for_sample(
            client,
            sample=sample,
            args=args,
            experiment_id_cache=experiment_id_cache,
            run_cache=run_cache,
        )
        key = (str(sample["dataset"]), str(run.info.run_id))
        if key not in grouped:
            grouped[key] = {
                "dataset": str(sample["dataset"]),
                "run": run,
                "samples": [],
            }
        grouped[key]["samples"].append(sample)
    return grouped


def _consistent_sample_override(
    samples: Sequence[Mapping[str, Any]],
    key: str,
    *,
    default: int | None,
    positive: bool = False,
) -> int | None:
    parser = parse_required_positive_int if positive else parse_required_nonnegative_int
    values = {
        parser(sample[key], key=key)
        for sample in samples
        if key in sample
    }
    if len(values) > 1:
        raise ValueError(f"Samples for one run have inconsistent {key} values: {sorted(values)}.")
    if values:
        return next(iter(values))
    return default


def _dataset_args_for_group(
    runtime_args: Namespace,
    cli_args: Namespace,
    samples: Sequence[Mapping[str, Any]],
) -> Namespace:
    args = Namespace(**vars(runtime_args))
    eval_data_seed = _consistent_sample_override(
        samples,
        "eval_data_seed",
        default=args.eval_data_seed,
    )
    n_test_samples = _consistent_sample_override(
        samples,
        "n_test_samples",
        default=args.n_test_samples,
        positive=True,
    )
    batch_size = _consistent_sample_override(
        samples,
        "batch_size",
        default=cli_args.batch_size,
        positive=True,
    )
    args.eval_data_seed = eval_data_seed
    args.n_test_samples = (
        None
        if n_test_samples is None
        else parse_required_positive_int(n_test_samples, key="n_test_samples")
    )
    args._window_arg_overrides = {
        "input_len": cli_args.input_len,
        "target_len": cli_args.target_len,
        "batch_size": batch_size,
    }
    return args


def _title_for_sample(
    *,
    dataset: str,
    sample: Mapping[str, Any],
    payload: Mapping[str, Any],
    score_metric: str,
) -> str:
    metric = parse_required_nonempty_string(score_metric, key="score_metric")
    method = _method_for_sample(sample)
    title = f"{dataset} - {sample['arch']} - {method}"
    return (
        f"{title}<br><sup>"
        f"Scenario: {payload['scenario']}, Severity: {float(payload['severity']):.3f}, "
        f"clean {metric}={float(payload['clean_sample_score']):.4f}, "
        f"pert {metric}={float(payload['perturbed_sample_score']):.4f}</sup>"
    )


def _method_for_sample(sample: Mapping[str, Any]) -> str:
    method = sample.get("robustness_method")
    if method is None:
        method = sample.get("pipeline_method", "baseline")
    return parse_required_nonempty_string(method, key="sample.robustness_method")


def _render_single_plot(
    *,
    dataset: str,
    sample: Mapping[str, Any],
    payload: Mapping[str, Any],
    score_metric: str,
    input_time_index: np.ndarray,
    output_time_index: np.ndarray,
):
    return plot_forecast_extreme(
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
        title=_title_for_sample(
            dataset=dataset,
            sample=sample,
            payload=payload,
            score_metric=score_metric,
        ),
        scenario=payload["scenario"],
        severity=float(payload["severity"]),
        sample_score=float(payload["sample_score"]),
        score_metric=score_metric,
    )


def _safe_name(value: Any) -> str:
    token = str(value).strip()
    if not token:
        raise ValueError("Cannot build output filename from an empty value.")
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in token)
    safe = safe.strip("_")
    if not safe:
        raise ValueError(f"Cannot build output filename from {value!r}.")
    return safe


def _output_name_parts(
    *,
    arch: Any,
    scenario: Any,
    sample_id: Any,
    severity: Any,
) -> tuple[str, str]:
    scenario_token = _safe_name(scenario)
    sample_id_value = parse_required_nonnegative_int(sample_id, key="sample_id")
    severity_value = parse_required_unit_interval_float(severity, key="severity")
    arch_token = _safe_name(arch)
    return (
        scenario_token,
        f"{arch_token}_{scenario_token}_sid{sample_id_value}_sev{severity_value:.3f}",
    )


def _plot_output_path(
    *,
    output_dir: Path,
    sample: Mapping[str, Any],
    payload: Mapping[str, Any],
    file_format: str,
) -> Path:
    scenario, stem = _output_name_parts(
        arch=sample["arch"],
        scenario=payload["scenario"],
        sample_id=sample["sample_id"],
        severity=payload["severity"],
    )
    scenario_dir = output_dir / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    return scenario_dir / f"{stem}.{file_format}"


def _planned_output_paths_for_sample(
    sample: Mapping[str, Any],
    *,
    output_dir: Path,
    file_format: str,
    individual: bool,
    export_traces: bool,
) -> tuple[tuple[str, Path], ...]:
    scenario, stem = _output_name_parts(
        arch=sample["arch"],
        scenario=sample["scenario"],
        sample_id=sample["sample_id"],
        severity=sample["severity"],
    )
    paths = []
    if individual:
        paths.append(("plot", output_dir / scenario / f"{stem}.{file_format}"))
    if export_traces:
        paths.append(("trace", output_dir / "traces" / scenario / f"{stem}.csv"))
    return tuple(paths)


def _describe_output_sample(sample: Mapping[str, Any]) -> str:
    parts = [
        f"order={sample.get('order')!r}",
        f"dataset={sample.get('dataset')!r}",
        f"arch={sample.get('arch')!r}",
        f"scenario={sample.get('scenario')!r}",
        f"sample_id={sample.get('sample_id')!r}",
        f"severity={sample.get('severity')!r}",
    ]
    for key in ("run_id", "robustness_method", "pipeline_method", "pipeline_id"):
        value = sample.get(key)
        if value is not None:
            parts.append(f"{key}={value!r}")
    return ", ".join(parts)


def _assert_no_output_path_collisions(
    samples: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    file_format: str,
    export_traces: bool,
    individual: bool = True,
) -> None:
    planned: dict[Path, str] = {}
    for sample in samples:
        for output_kind, output_path in _planned_output_paths_for_sample(
            sample,
            output_dir=output_dir,
            file_format=file_format,
            individual=individual,
            export_traces=export_traces,
        ):
            previous_sample = planned.get(output_path)
            current_sample = _describe_output_sample(sample)
            if previous_sample is not None:
                raise ValueError(
                    f"Forecast {output_kind} output collision at {output_path}: "
                    f"{previous_sample} and {current_sample} would overwrite each other. "
                    "Narrow the sample filters or render the colliding rows separately."
                )
            planned[output_path] = current_sample


def _write_figure(fig, path: Path, *, file_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "html":
        fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    else:
        fig.write_image(str(path))


def _as_2d_trace_array(payload: Mapping[str, Any], *, key: str) -> np.ndarray:
    if key not in payload:
        raise ValueError(f"Trace export payload is missing required key {key!r}.")
    array = np.asarray(payload[key], dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(
            f"Trace export payload {key!r} must be 1D or 2D. Got shape {array.shape}."
        )
    return array


def _trace_feature_names(
    payload: Mapping[str, Any],
    *,
    key: str,
    expected_count: int,
) -> list[str]:
    if key not in payload:
        raise ValueError(f"Trace export payload is missing required key {key!r}.")
    names = [
        parse_required_nonempty_string(name, key=f"{key}[{index}]")
        for index, name in enumerate(payload[key])
    ]
    if len(names) != expected_count:
        raise ValueError(
            f"Trace export payload {key!r} has {len(names)} names, "
            f"but the corresponding array has {expected_count} columns."
        )
    return names


def _finite_payload_score(payload: Mapping[str, Any], *, key: str) -> float:
    if key not in payload:
        raise ValueError(f"Trace export payload is missing required key {key!r}.")
    value = float(payload[key])
    if not np.isfinite(value):
        raise ValueError(f"Trace export payload {key!r} must be finite. Got {value}.")
    return value


def _export_sample_trace_csv(
    *,
    dataset: str,
    sample: Mapping[str, Any],
    payload: Mapping[str, Any],
    score_metric: str,
    input_time_index: np.ndarray,
    output_time_index: np.ndarray,
    output_dir: Path,
) -> Path:
    clean_input = _as_2d_trace_array(payload, key="clean_input")
    perturbed_input = _as_2d_trace_array(payload, key="perturbed_input")
    target = _as_2d_trace_array(payload, key="target")
    prediction_clean = _as_2d_trace_array(payload, key="prediction_clean")
    prediction_perturbed = _as_2d_trace_array(payload, key="prediction_perturbed")
    if clean_input.shape != perturbed_input.shape:
        raise ValueError(
            "Trace export requires clean_input and perturbed_input to have the same "
            f"shape. Got {clean_input.shape} vs {perturbed_input.shape}."
        )
    if target.shape != prediction_clean.shape or target.shape != prediction_perturbed.shape:
        raise ValueError(
            "Trace export requires target, prediction_clean, and prediction_perturbed "
            f"to have the same shape. Got {target.shape}, {prediction_clean.shape}, "
            f"{prediction_perturbed.shape}."
        )
    if len(input_time_index) != clean_input.shape[0]:
        raise ValueError("Trace export input_time_index length does not match clean_input.")
    if len(output_time_index) != target.shape[0]:
        raise ValueError("Trace export output_time_index length does not match target.")

    input_feature_names = _trace_feature_names(
        payload,
        key="input_feature_names",
        expected_count=clean_input.shape[1],
    )
    target_feature_names = _trace_feature_names(
        payload,
        key="target_feature_names",
        expected_count=target.shape[1],
    )
    affected_feature_names = {
        parse_required_nonempty_string(name, key=f"affected_feature_names[{index}]")
        for index, name in enumerate(payload["affected_feature_names"])
    }
    scenario = parse_required_nonempty_string(payload["scenario"], key="payload.scenario")
    severity = parse_required_unit_interval_float(payload["severity"], key="payload.severity")
    sample_id = parse_required_nonnegative_int(payload["sample_id"], key="payload.sample_id")
    source_sample_idx = parse_required_nonnegative_int(
        payload["source_sample_idx"],
        key="payload.source_sample_idx",
    )
    pert_idx = parse_required_nonnegative_int(payload["pert_idx"], key="payload.pert_idx")
    metadata = {
        "dataset": parse_required_nonempty_string(dataset, key="dataset"),
        "arch": parse_required_nonempty_string(sample["arch"], key="sample.arch"),
        "robustness_method": _method_for_sample(sample),
        "scenario": scenario,
        "pert_idx": pert_idx,
        "sample_id": sample_id,
        "source_sample_idx": source_sample_idx,
        "severity": severity,
        "score_metric": parse_required_nonempty_string(score_metric, key="score_metric"),
        "clean_sample_score": _finite_payload_score(payload, key="clean_sample_score"),
        "perturbed_sample_score": _finite_payload_score(payload, key="perturbed_sample_score"),
    }
    rows: list[dict[str, Any]] = []
    for time_pos, time_value in enumerate(input_time_index):
        for feature_idx, feature_name in enumerate(input_feature_names):
            rows.append(
                {
                    **metadata,
                    "segment": "input",
                    "time_index": float(time_value),
                    "time_step": int(time_pos),
                    "feature": feature_name,
                    "affected_feature": feature_name in affected_feature_names,
                    "clean_input": float(clean_input[time_pos, feature_idx]),
                    "perturbed_input": float(perturbed_input[time_pos, feature_idx]),
                    "ground_truth": None,
                    "clean_prediction": None,
                    "perturbed_prediction": None,
                }
            )
    for time_pos, time_value in enumerate(output_time_index):
        for feature_idx, feature_name in enumerate(target_feature_names):
            rows.append(
                {
                    **metadata,
                    "segment": "forecast",
                    "time_index": float(time_value),
                    "time_step": int(time_pos),
                    "feature": feature_name,
                    "affected_feature": feature_name in affected_feature_names,
                    "clean_input": None,
                    "perturbed_input": None,
                    "ground_truth": float(target[time_pos, feature_idx]),
                    "clean_prediction": float(prediction_clean[time_pos, feature_idx]),
                    "perturbed_prediction": float(prediction_perturbed[time_pos, feature_idx]),
                }
            )

    scenario_token, stem = _output_name_parts(
        arch=metadata["arch"],
        scenario=scenario,
        sample_id=sample_id,
        severity=severity,
    )
    scenario_dir = output_dir / "traces" / scenario_token
    scenario_dir.mkdir(parents=True, exist_ok=True)
    output_path = scenario_dir / f"{stem}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def _ordered_unique(values: pd.Series) -> list[Any]:
    return list(dict.fromkeys(values.tolist()))


def _single_value(frame: pd.DataFrame, column: str, *, path: Path) -> Any:
    if frame[column].isna().any():
        raise ValueError(f"{path} column {column!r} must not contain missing values.")
    values = frame[column].unique().tolist()
    if len(values) != 1:
        raise ValueError(f"{path} column {column!r} must have exactly one value.")
    return values[0]


def _read_trace_frame(
    path: Path,
    *,
    missing_robustness_method: str | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_TRACE_COLUMNS - set(frame.columns))
    if "robustness_method" in missing and missing_robustness_method is not None:
        frame["robustness_method"] = parse_required_nonempty_string(
            missing_robustness_method,
            key="missing_robustness_method",
        )
        missing = [column for column in missing if column != "robustness_method"]
    if missing:
        raise ValueError(f"{path} is missing required trace columns: {missing}.")
    if frame.empty:
        raise ValueError(f"{path} must not be empty.")
    frame["feature"] = frame["feature"].map(
        lambda feature: parse_required_nonempty_string(
            feature,
            key=f"{path.name}.feature",
        )
    )
    return frame


def _pivot_segment(
    frame: pd.DataFrame,
    *,
    segment: str,
    value_column: str,
    path: Path,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    rows = frame.loc[frame["segment"] == segment].copy()
    if rows.empty:
        raise ValueError(f"{path} has no rows for segment {segment!r}.")
    time_steps = _ordered_unique(rows["time_step"])
    rows["feature"] = rows["feature"].map(
        lambda feature: parse_required_nonempty_string(
            feature,
            key=f"{path.name}.feature",
        )
    )
    feature_names = _ordered_unique(rows["feature"])
    pivot = rows.pivot(index="time_step", columns="feature", values=value_column)
    pivot = pivot.reindex(index=time_steps, columns=feature_names)
    if pivot.isna().any().any():
        raise ValueError(
            f"{path} segment {segment!r} has missing values for {value_column!r}."
        )
    time_index_counts = rows.groupby("time_step")["time_index"].nunique(dropna=False)
    inconsistent_steps = time_index_counts.loc[time_index_counts != 1]
    if not inconsistent_steps.empty:
        raise ValueError(
            f"{path} segment {segment!r} has inconsistent time_index values "
            f"for time_step(s): {inconsistent_steps.index.tolist()}."
        )
    time_index = (
        rows.drop_duplicates("time_step")
        .set_index("time_step")
        .reindex(time_steps)["time_index"]
        .to_numpy(dtype=float)
    )
    if not np.isfinite(time_index).all():
        raise ValueError(f"{path} segment {segment!r} has non-finite time_index values.")
    return time_index, feature_names, pivot.to_numpy(dtype=float)


def _affected_feature_names(frame: pd.DataFrame) -> list[str]:
    affected_mask = frame["affected_feature"].map(
        lambda value: parse_required_bool(value, key="affected_feature")
    )
    affected = frame.loc[affected_mask, "feature"]
    return [
        parse_required_nonempty_string(feature, key="affected_feature")
        for feature in _ordered_unique(affected)
    ]


def _select_display_feature(frame: pd.DataFrame, *, path: Path) -> str:
    forecast_features = [
        parse_required_nonempty_string(feature, key=f"{path.name}.feature")
        for feature in _ordered_unique(frame.loc[frame["segment"] == "forecast", "feature"])
    ]
    input_features = {
        parse_required_nonempty_string(feature, key=f"{path.name}.feature")
        for feature in _ordered_unique(frame.loc[frame["segment"] == "input", "feature"])
    }
    common_features = [feature for feature in forecast_features if feature in input_features]
    if not common_features:
        raise ValueError(f"{path} has no feature present in both input and forecast rows.")
    affected_features = set(_affected_feature_names(frame))
    for feature in common_features:
        if feature in affected_features:
            return feature
    return common_features[0]


def _validate_trace_segments(frame: pd.DataFrame, *, path: Path) -> None:
    if frame["segment"].isna().any():
        raise ValueError(f"{path} segment must not contain missing values.")
    segments = {
        parse_required_nonempty_string(segment, key=f"{path.name}.segment")
        for segment in _ordered_unique(frame["segment"])
    }
    unsupported_segments = sorted(segments - TRACE_SEGMENTS)
    if unsupported_segments:
        raise ValueError(
            f"{path} has unsupported segment value(s): {unsupported_segments}. "
            f"Expected only {sorted(TRACE_SEGMENTS)}."
        )
    _, input_features, _ = _pivot_segment(
        frame,
        segment="input",
        value_column="clean_input",
        path=path,
    )
    _, perturbed_input_features, _ = _pivot_segment(
        frame,
        segment="input",
        value_column="perturbed_input",
        path=path,
    )
    if perturbed_input_features != input_features:
        raise ValueError(f"{path} clean and perturbed input feature order differs.")
    _, target_features, _ = _pivot_segment(
        frame,
        segment="forecast",
        value_column="ground_truth",
        path=path,
    )
    _, clean_prediction_features, _ = _pivot_segment(
        frame,
        segment="forecast",
        value_column="clean_prediction",
        path=path,
    )
    _, perturbed_prediction_features, _ = _pivot_segment(
        frame,
        segment="forecast",
        value_column="perturbed_prediction",
        path=path,
    )
    if clean_prediction_features != target_features:
        raise ValueError(f"{path} clean prediction feature order differs from target.")
    if perturbed_prediction_features != target_features:
        raise ValueError(f"{path} perturbed prediction feature order differs from target.")


def _trace_metadata(frame: pd.DataFrame, *, path: Path) -> dict[str, Any]:
    dataset = parse_required_nonempty_string(
        _single_value(frame, "dataset", path=path),
        key="dataset",
    )
    arch = parse_required_nonempty_string(
        _single_value(frame, "arch", path=path),
        key="arch",
    )
    robustness_method = parse_required_nonempty_string(
        _single_value(frame, "robustness_method", path=path),
        key="robustness_method",
    )
    scenario = parse_required_nonempty_string(
        _single_value(frame, "scenario", path=path),
        key="scenario",
    )
    score_metric = parse_required_nonempty_string(
        _single_value(frame, "score_metric", path=path),
        key="score_metric",
    )
    for column in ("pert_idx", "sample_id", "source_sample_idx"):
        parse_required_nonnegative_int(_single_value(frame, column, path=path), key=column)
    severity = float(_single_value(frame, "severity", path=path))
    clean_score = float(_single_value(frame, "clean_sample_score", path=path))
    perturbed_score = float(_single_value(frame, "perturbed_sample_score", path=path))
    if not np.isfinite(severity) or severity < 0.0 or severity > 1.0:
        raise ValueError(f"{path} severity must be finite and in [0, 1].")
    if not np.isfinite(clean_score):
        raise ValueError(f"{path} clean_sample_score must be finite.")
    if not np.isfinite(perturbed_score):
        raise ValueError(f"{path} perturbed_sample_score must be finite.")
    return {
        "dataset": dataset,
        "arch": arch,
        "robustness_method": robustness_method,
        "scenario": scenario,
        "score_metric": score_metric,
        "severity": severity,
        "clean_score": clean_score,
        "perturbed_score": perturbed_score,
    }


def _trace_title(
    metadata: Mapping[str, Any],
    *,
    mode: str,
    display_feature: str | None,
) -> str:
    if mode == "single-feature":
        if display_feature is None:
            raise ValueError("single-feature forecast trace rendering requires a feature.")
        return (
            f"{metadata['dataset']} - {metadata['arch']} - "
            f"{metadata['robustness_method']} - {metadata['scenario']} - "
            f"{display_feature}<br><sup>"
            f"Scenario: {metadata['scenario']}, Feature: {display_feature}, "
            f"Severity: {float(metadata['severity']):.3f}, "
            f"clean {metadata['score_metric']}={float(metadata['clean_score']):.4f}, "
            f"pert {metadata['score_metric']}={float(metadata['perturbed_score']):.4f}</sup>"
        )
    if mode == "full":
        return (
            f"{metadata['dataset']} - {metadata['arch']} - "
            f"{metadata['robustness_method']}<br><sup>"
            f"Scenario: {metadata['scenario']}, Severity: {float(metadata['severity']):.3f}, "
            f"clean {metadata['score_metric']}={float(metadata['clean_score']):.4f}, "
            f"pert {metadata['score_metric']}={float(metadata['perturbed_score']):.4f}</sup>"
        )
    raise ValueError(f"Unknown forecast trace render mode: {mode!r}.")


def load_forecast_trace_record(
    path: Path,
    *,
    mode: str = "full",
    missing_robustness_method: str | None = None,
) -> dict[str, Any]:
    frame = _read_trace_frame(
        path,
        missing_robustness_method=missing_robustness_method,
    )
    if mode not in TRACE_RENDER_MODES:
        raise ValueError(f"mode must be one of {TRACE_RENDER_MODES}. Got {mode!r}.")

    metadata = _trace_metadata(frame, path=path)
    _validate_trace_segments(frame, path=path)
    affected_feature_names = _affected_feature_names(frame)
    affected_features = set(affected_feature_names)
    display_feature = None
    if mode == "single-feature":
        display_feature = _select_display_feature(frame, path=path)
        plot_frame = frame.loc[frame["feature"] == display_feature].copy()
        plot_affected_features = (
            [display_feature] if display_feature in affected_features else []
        )
    else:
        plot_frame = frame.copy()
        plot_affected_features = affected_feature_names

    input_time, input_features, clean_input = _pivot_segment(
        plot_frame,
        segment="input",
        value_column="clean_input",
        path=path,
    )
    _, perturbed_input_features, perturbed_input = _pivot_segment(
        plot_frame,
        segment="input",
        value_column="perturbed_input",
        path=path,
    )
    if perturbed_input_features != input_features:
        raise ValueError(f"{path} clean and perturbed input feature order differs.")
    output_time, target_features, target = _pivot_segment(
        plot_frame,
        segment="forecast",
        value_column="ground_truth",
        path=path,
    )
    _, clean_prediction_features, clean_prediction = _pivot_segment(
        plot_frame,
        segment="forecast",
        value_column="clean_prediction",
        path=path,
    )
    _, perturbed_prediction_features, perturbed_prediction = _pivot_segment(
        plot_frame,
        segment="forecast",
        value_column="perturbed_prediction",
        path=path,
    )
    if clean_prediction_features != target_features:
        raise ValueError(f"{path} clean prediction feature order differs from target.")
    if perturbed_prediction_features != target_features:
        raise ValueError(f"{path} perturbed prediction feature order differs from target.")

    figure = plot_forecast_extreme(
        output_time,
        target,
        perturbed_prediction,
        prediction_clean=clean_prediction,
        clean_input=clean_input,
        perturbed_input=perturbed_input,
        input_time_index=input_time,
        input_feature_names=input_features,
        target_feature_names=target_features,
        affected_feature_names=plot_affected_features,
        title=_trace_title(metadata, mode=mode, display_feature=display_feature),
        scenario=str(metadata["scenario"]),
        severity=float(metadata["severity"]),
        sample_score=float(metadata["perturbed_score"]),
        score_metric=str(metadata["score_metric"]),
    )
    return {
        "figure": figure,
        "trace_csv": path.as_posix(),
    }


def load_forecast_trace_figure(
    path: Path,
    *,
    mode: str = "full",
    missing_robustness_method: str | None = None,
):
    return load_forecast_trace_record(
        path,
        mode=mode,
        missing_robustness_method=missing_robustness_method,
    )["figure"]


def _trace_paths(args: Namespace) -> list[Path]:
    paths: list[Path] = []
    for trace_csv in args.trace_csv:
        paths.append(trace_csv)
    for trace_root in args.trace_root:
        if not trace_root.is_dir():
            raise FileNotFoundError(f"Trace root does not exist: {trace_root}")
        paths.extend(sorted(trace_root.rglob("*.csv")))
    unique_paths = list(dict.fromkeys(paths))
    if not unique_paths:
        raise ValueError("No trace CSV files were provided.")
    return unique_paths


def _relative_trace_output_path(
    trace_path: Path,
    *,
    trace_roots: Sequence[Path],
    file_format: str,
) -> Path:
    relative_path: Path | None = None
    for trace_root in trace_roots:
        try:
            relative_path = trace_path.relative_to(trace_root)
            break
        except ValueError:
            continue
    if relative_path is None:
        return Path(f"{trace_path.stem}.{file_format}")

    parts = list(relative_path.with_suffix(f".{file_format}").parts)
    if "traces" in parts:
        parts.pop(parts.index("traces"))
    return Path(*parts)


def _trace_output_path(
    trace_path: Path,
    *,
    output_dir: Path,
    file_format: str,
    preserve_layout: bool,
    trace_roots: Sequence[Path],
) -> Path:
    if preserve_layout:
        relative_path = _relative_trace_output_path(
            trace_path,
            trace_roots=trace_roots,
            file_format=file_format,
        )
        return output_dir / relative_path
    return output_dir / f"{trace_path.stem}.{file_format}"


def render_traces(
    trace_paths: Sequence[Path],
    *,
    output_dir: Path,
    mode: str = "full",
    file_format: str = "html",
    preserve_layout: bool = False,
    trace_roots: Sequence[Path] = (),
    missing_robustness_method: str | None = None,
) -> list[dict[str, str]]:
    if mode not in TRACE_RENDER_MODES:
        raise ValueError(f"mode must be one of {TRACE_RENDER_MODES}. Got {mode!r}.")
    if file_format not in OUTPUT_FORMATS:
        raise ValueError(
            f"file_format must be one of {OUTPUT_FORMATS}. Got {file_format!r}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, str]] = []
    planned_outputs: dict[Path, Path] = {}
    for trace_path in trace_paths:
        output_path = _trace_output_path(
            trace_path,
            output_dir=output_dir,
            file_format=file_format,
            preserve_layout=preserve_layout,
            trace_roots=trace_roots,
        )
        previous_trace = planned_outputs.get(output_path)
        if previous_trace is not None and previous_trace != trace_path:
            raise ValueError(
                "Trace output filename collision: "
                f"{previous_trace} and {trace_path} both render to {output_path}."
            )
        planned_outputs[output_path] = trace_path
        trace_record = load_forecast_trace_record(
            trace_path,
            mode=mode,
            missing_robustness_method=missing_robustness_method,
        )
        figure = trace_record["figure"]
        _write_figure(figure, output_path, file_format=file_format)
        record = {
            "trace_csv": trace_path.as_posix(),
            "output_path": output_path.as_posix(),
        }
        if file_format == "html":
            record["output_html"] = output_path.as_posix()
        rendered.append(record)
    return rendered


def _render_group(
    *,
    client: Any,
    defaults: Mapping[str, Any],
    dataset_window_defaults: Mapping[str, Mapping[str, Any]],
    cli_args: Namespace,
    runtime_args: Namespace,
    dataset: str,
    run,
    samples: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    from pipelines.selection import require_seed_tags
    from testing.evaluation import (
        _build_testing_datamodule,
        _collect_degradation_forecast_samples,
        _prime_model_for_degradation_evaluation,
        _resolve_requested_runtime_device,
        _teardown_model_after_eval,
        load_model_with_loader,
    )

    seeds = require_seed_tags(run)
    group_args = _dataset_args_for_group(runtime_args, cli_args, samples)
    eval_data_seed = resolve_effective_eval_data_seed(
        group_args.eval_data_seed,
        canonical_seed_data=seeds["seed_data"],
    )
    dataset_spec = _select_dataset_spec(
        defaults,
        dataset=dataset,
        data_root=str(group_args.data_root),
    )
    dataset_args = resolve_dataset_window_args(
        group_args,
        dataset_spec=dataset_spec,
        dataset_window_defaults=dataset_window_defaults,
        explicit_arg_overrides=group_args._window_arg_overrides,
    )
    dm = _build_testing_datamodule(
        dataset_spec=dataset_spec,
        args=dataset_args,
        canonical_data_seed=seeds["seed_data"],
        eval_data_seed=eval_data_seed,
        val_seed=None,
    )
    model = None
    try:
        model, _default_root_dir = load_model_with_loader(client, run, dataset_args, dm)
        _prime_model_for_degradation_evaluation(
            model,
            dataset_args,
            dm,
            eval_seed=seeds["seed_eval"],
        )
        sample_rows = pd.DataFrame(samples)
        runtime_device = _resolve_requested_runtime_device(
            dataset_args,
            context_name="forecast plot rendering",
        )
        rendered_samples = _collect_degradation_forecast_samples(
            model=model,
            dm=dm,
            sample_rows=sample_rows,
            test_metric=str(dataset_args.test_metric),
            eval_data_seed=eval_data_seed,
            runtime_device=runtime_device,
            runtime_precision=require_namespace_value(dataset_args, key="precision"),
        )
    finally:
        if model is not None:
            _teardown_model_after_eval(model)
            del model

    payload_by_key = {
        (int(payload["sample_id"]), int(payload["pert_idx"])): payload
        for payload in rendered_samples
    }
    input_time_index = np.arange(1, int(dataset_args.input_len) + 1, dtype=float)
    output_time_index = np.arange(
        int(dataset_args.input_len) + 1,
        int(dataset_args.input_len) + int(dataset_args.target_len) + 1,
        dtype=float,
    )
    output_records: list[dict[str, str]] = []
    for sample in sorted(samples, key=lambda item: int(item["order"])):
        key = (int(sample["sample_id"]), int(sample["pert_idx"]))
        payload = payload_by_key.get(key)
        if payload is None:
            raise ValueError(
                f"Missing rendered payload for sample_id={key[0]}, pert_idx={key[1]}."
            )
        if cli_args.individual:
            fig = _render_single_plot(
                dataset=dataset,
                sample=sample,
                payload=payload,
                score_metric=str(dataset_args.test_metric),
                input_time_index=input_time_index,
                output_time_index=output_time_index,
            )
            plot_path = _plot_output_path(
                output_dir=cli_args.output_dir,
                sample=sample,
                payload=payload,
                file_format=cli_args.format,
            )
            _write_figure(fig, plot_path, file_format=cli_args.format)
        record = {
            "dataset": dataset,
            "run_id": str(run.info.run_id),
            "sample_id": str(int(sample["sample_id"])),
            "pert_idx": str(int(sample["pert_idx"])),
        }
        if cli_args.individual:
            record["plot_path"] = plot_path.as_posix()
        if cli_args.export_traces:
            trace_path = _export_sample_trace_csv(
                dataset=dataset,
                sample=sample,
                payload=payload,
                score_metric=str(dataset_args.test_metric),
                input_time_index=input_time_index,
                output_time_index=output_time_index,
                output_dir=cli_args.output_dir,
            )
            record["trace_csv"] = trace_path.as_posix()
        output_records.append(record)
    return output_records


def _main_from_runs(cli_args: Namespace, *, defaults: Mapping[str, Any]) -> list[dict[str, str]]:
    import mlflow
    from config_loader import load_dataset_windows
    from utils.env import set_mlflow_storage_env

    runtime_args = _build_runtime_args(defaults, cli_args)
    set_mlflow_storage_env(runtime_args)
    tracking_uri = build_mlflow_tracking_uri(runtime_args.logdir)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    selected_samples = _load_selected_samples(cli_args, client=client)
    _assert_no_output_path_collisions(
        selected_samples,
        output_dir=cli_args.output_dir,
        file_format=cli_args.format,
        individual=cli_args.individual,
        export_traces=cli_args.export_traces,
    )
    cli_args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_window_defaults = load_dataset_windows(defaults=defaults)
    grouped = _group_samples_by_run(
        selected_samples,
        client=client,
        args=runtime_args,
    )
    rendered: list[dict[str, str]] = []
    for key in sorted(grouped):
        group = grouped[key]
        group_records = _render_group(
            client=client,
            defaults=defaults,
            dataset_window_defaults=dataset_window_defaults,
            cli_args=cli_args,
            runtime_args=runtime_args,
            dataset=str(group["dataset"]),
            run=group["run"],
            samples=group["samples"],
        )
        rendered.extend(group_records)
    return rendered


def _main_from_traces(cli_args: Namespace) -> list[dict[str, str]]:
    return render_traces(
        _trace_paths(cli_args),
        output_dir=cli_args.output_dir,
        mode=cli_args.mode,
        file_format=cli_args.format,
        preserve_layout=cli_args.preserve_layout,
        trace_roots=cli_args.trace_root,
        missing_robustness_method=cli_args.missing_robustness_method,
    )


def main(argv: Sequence[str] | None = None) -> int:
    from config_loader import load_defaults

    defaults = load_defaults()
    cli_args = parse_cli_args(argv, defaults=defaults)
    if cli_args.command == "from-runs":
        rendered = _main_from_runs(cli_args, defaults=defaults)
    elif cli_args.command == "from-traces":
        rendered = _main_from_traces(cli_args)
    else:
        raise ValueError(f"Unknown forecast plot command: {cli_args.command!r}.")
    if cli_args.index is not None:
        cli_args.index.parent.mkdir(parents=True, exist_ok=True)
        cli_args.index.write_text(json.dumps(rendered, indent=2), encoding="utf-8")
    print(json.dumps({"rendered": len(rendered)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
