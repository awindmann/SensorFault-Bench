"""Compute-accounting ledger and full-project overhead audit.

Extends summarize_benchmark_compute.py to produce:
1. A per-run CSV ledger for official current-scope setup and testing spans.
2. Aggregate summary CSVs by dataset, architecture, and method.
3. A full-project overhead audit covering stale, out-of-scope, failed, killed,
   running, and soft-deleted runs across all MLflow experiments.

Read-only: does NOT call set_tag, log_metric, log_param, delete_run, or any
rerun/mutation command.

Run from the repo root, using the same MLflow prefix and tracking URI as the
benchmark run being audited:

    python scripts/compute_accounting_ledger.py \
        --mlflow-experiment-prefix test-1 \
        --logdir https://<mlflow-tracking-server>
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.entities import ViewType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import (
    build_parser,
    load_defaults,
    load_dataset_windows,
    parse_explicit_cli_overrides,
)
from data.datasets import resolve_with_defaults
from pipelines.ranking import ALLOWED_PIPELINE_KINDS
from pipelines.runner import (
    create_pipeline_runner,
    scope_policy_skip_reason_for_spec,
)
from pipelines.selection import (
    extract_recipe_defaults_for_scope,
    is_fully_tested,
    load_benchmark_recipe_specs_for_scope,
    merge_recipe_defaults_for_scope,
    resolve_pipeline_tags,
)
from pipelines.signatures import compute_data_config_signature
from pipelines.specs import PipelineSpec
from pipelines.training import get_tracking_uri, search_runs_all
from utils.parsing import (
    require_degradation_eval_context_tags,
    resolve_dataset_window_args,
)
from utils.scoring import required_degradation_metric_keys

# ---------------------------------------------------------------------------
# Duration metric keys — same as summarize_benchmark_compute.py
# ---------------------------------------------------------------------------
DURATION_METRIC_KEYS: tuple[str, ...] = ("ep_val_loss", "best_val_loss", "train_loss")

OUTPUT_DIR = PROJECT_ROOT / "evidence" / "compute_accounting"


def _ts_to_iso(ts_ms: int | None) -> str:
    if ts_ms is None:
        return ""
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()


def _require_run_tag(run, key: str) -> str:
    tags = run.data.tags or {}
    raw_value = tags.get(key)
    if raw_value is None or not str(raw_value).strip():
        raise ValueError(
            f"Run {run.info.run_id} is missing required MLflow tag '{key}'."
        )
    return str(raw_value).strip()


def _safe_tag(run, key: str) -> str:
    tags = run.data.tags or {}
    v = tags.get(key)
    return str(v).strip() if v is not None else ""


def _safe_param(run, key: str) -> str:
    params = run.data.params or {}
    v = params.get(key)
    return str(v).strip() if v is not None else ""


# ---------------------------------------------------------------------------
# Duration resolvers (identical logic to summarize_benchmark_compute.py)
# ---------------------------------------------------------------------------
def _resolve_run_duration_seconds(client, run) -> tuple[float, str]:
    start = run.info.start_time
    if start is None:
        raise ValueError(f"Run {run.info.run_id} is missing start_time.")
    for key in DURATION_METRIC_KEYS:
        try:
            history = client.get_metric_history(run.info.run_id, key)
        except Exception:
            continue
        if not history:
            continue
        last_ts = max(h.timestamp for h in history)
        if last_ts <= start:
            continue
        return (last_ts - start) / 1000.0, f"metric:{key}"
    return float("nan"), "no_metric_evidence"


def _resolve_run_duration_seconds_strict(client, run) -> tuple[float, str]:
    start = run.info.start_time
    if start is None:
        raise ValueError(f"Run {run.info.run_id} is missing start_time.")
    for key in DURATION_METRIC_KEYS:
        try:
            history = client.get_metric_history(run.info.run_id, key)
        except Exception:
            continue
        if not history:
            continue
        last_ts = max(h.timestamp for h in history)
        if last_ts <= start:
            raise ValueError(
                f"Run {run.info.run_id} metric '{key}' has no timestamp after "
                f"start_time ({last_ts} <= {start})."
            )
        return (last_ts - start) / 1000.0, f"metric:{key}"
    expected = ", ".join(DURATION_METRIC_KEYS)
    raise ValueError(
        f"Run {run.info.run_id} has no duration metric history. "
        f"Expected at least one of: {expected}."
    )


def _resolve_wrapper_duration_seconds_strict(
    client,
    run,
    *,
    test_metric: str,
) -> tuple[float, str]:
    start = run.info.start_time
    if start is None:
        raise ValueError(f"Run {run.info.run_id} is missing start_time.")
    metric_keys = (f"{test_metric}_val", *DURATION_METRIC_KEYS)
    for key in metric_keys:
        try:
            history = client.get_metric_history(run.info.run_id, key)
        except Exception:
            continue
        if not history:
            continue
        last_ts = max(h.timestamp for h in history)
        if last_ts <= start:
            raise ValueError(
                f"Run {run.info.run_id} metric '{key}' has no timestamp after "
                f"start_time ({last_ts} <= {start})."
            )
        return (last_ts - start) / 1000.0, f"metric:{key}"
    expected = ", ".join(metric_keys)
    raise ValueError(
        f"Wrapper run {run.info.run_id} has no setup-duration metric history. "
        f"Expected at least one of: {expected}."
    )


def _resolve_official_setup_duration_seconds_strict(
    client,
    run,
    *,
    test_metric: str,
) -> tuple[float, str]:
    kind = _require_run_tag(run, "pipeline_kind")
    if kind == "train":
        return _resolve_run_duration_seconds_strict(client, run)
    if kind == "wrap":
        return _resolve_wrapper_duration_seconds_strict(
            client,
            run,
            test_metric=test_metric,
        )
    raise ValueError(
        f"Run {run.info.run_id} has unsupported official setup pipeline_kind={kind!r}."
    )


def _metric_timestamps(client, run, key: str) -> list[int]:
    try:
        history = client.get_metric_history(run.info.run_id, key)
    except Exception:
        return []
    return [int(h.timestamp) for h in history]


def _resolve_testing_duration_seconds(client, run) -> tuple[float, str]:
    tags = run.data.tags
    if tags is None:
        return float("nan"), "no_tags"
    try:
        eval_context = require_degradation_eval_context_tags(
            tags, run_id=run.info.run_id,
        )
    except Exception:
        return float("nan"), "no_eval_context"
    test_metric = str(eval_context["test_metric"])
    scenario_indices = tuple(sorted(eval_context["perturbation_idx_name_map"]))
    terminal_keys = required_degradation_metric_keys(
        test_metric=test_metric,
        scenario_indices=scenario_indices,
    )
    terminal_timestamps_by_key: dict[str, int] = {}
    for key in terminal_keys:
        ts_list = _metric_timestamps(client, run, key)
        if not ts_list:
            return float("nan"), f"missing_metric:{key}"
        terminal_timestamps_by_key[key] = max(ts_list)
    end_key, end_ts = max(
        terminal_timestamps_by_key.items(),
        key=lambda item: (item[1], item[0]),
    )
    start_key = f"{test_metric}_val"
    start_candidates = [
        ts for ts in _metric_timestamps(client, run, start_key) if ts <= end_ts
    ]
    if not start_candidates:
        return float("nan"), f"no_{start_key}_anchor"
    start_ts = max(start_candidates)
    if end_ts <= start_ts:
        return float("nan"), "non_positive_span"
    return (end_ts - start_ts) / 1000.0, f"metric:{start_key}->{end_key}"


# ---------------------------------------------------------------------------
# Raw-span fallback for runs without metric evidence
# ---------------------------------------------------------------------------
def _raw_span_seconds(run) -> tuple[float, str]:
    start = run.info.start_time
    end = run.info.end_time
    if start is None:
        return float("nan"), "no_start_time"
    if end is None or end <= start:
        return float("nan"), "no_end_time"
    return (end - start) / 1000.0, "raw_span"


# ---------------------------------------------------------------------------
# Build args namespace (same as summarize_benchmark_compute.py)
# ---------------------------------------------------------------------------
def _recipe_param_defaults(extracted_defaults: list[dict[str, object]]) -> dict[str, object]:
    defaults: dict[str, object] = {}
    for defaults_dict in extracted_defaults:
        defaults.update(defaults_dict)
    return defaults


def _build_args_namespace(
    base_defaults: dict,
    recipe_specs: list[PipelineSpec],
) -> argparse.Namespace:
    extracted_defaults = extract_recipe_defaults_for_scope(recipe_specs)
    defaults = merge_recipe_defaults_for_scope(base_defaults, extracted_defaults)
    parser = build_parser(defaults)
    args = parser.parse_args()
    args._explicit_cli_args = tuple(sys.argv[1:])
    args._recipe_param_overrides = parse_explicit_cli_overrides(
        _recipe_param_defaults(extracted_defaults),
        extra_args=args._explicit_cli_args,
    )
    args._window_arg_overrides = parse_explicit_cli_overrides(
        {
            "input_len": defaults["INPUT_LEN"],
            "target_len": defaults["TARGET_LEN"],
            "batch_size": defaults["BATCH_SIZE"],
        },
        extra_args=args._explicit_cli_args,
    )
    return args


def _resolve_official_setup_compute_scope_specs(
    recipe_specs: list[PipelineSpec],
) -> list[PipelineSpec]:
    selected: list[PipelineSpec] = []
    for spec in recipe_specs:
        method = str(spec.pipeline_method).strip()
        kind = str(spec.pipeline_kind).strip()
        if kind in {"train", "wrap"}:
            selected.append(spec)
            continue
        raise ValueError(
            "Compute accounting supports train and wrap recipes in the public "
            f"benchmark scope. Method '{method}' declares pipeline_kind='{kind}'."
        )
    return selected


def _expected_pipeline_kinds_by_cell(
    *,
    scope_specs: list[PipelineSpec],
    architectures: list[str],
) -> dict[tuple[str, str], str]:
    expected: dict[tuple[str, str], str] = {}
    for spec in scope_specs:
        method = str(spec.pipeline_method).strip()
        kind = str(spec.pipeline_kind).strip()
        for arch in architectures:
            policy_skip = scope_policy_skip_reason_for_spec(spec, arch)
            if policy_skip is not None:
                continue
            expected[(arch, method)] = kind
    return expected


def _resolve_testing_compute_scope_specs(
    recipe_specs: list[PipelineSpec],
) -> list[PipelineSpec]:
    selected: list[PipelineSpec] = []
    for spec in recipe_specs:
        method = str(spec.pipeline_method).strip()
        kind = str(spec.pipeline_kind).strip()
        if kind in ALLOWED_PIPELINE_KINDS:
            selected.append(spec)
            continue
        allowed = ", ".join(ALLOWED_PIPELINE_KINDS)
        raise ValueError(
            f"Method '{method}' declares pipeline_kind='{kind}'. "
            f"Allowed kinds: {allowed}."
        )
    return selected


def _resolve_scope_signatures(
    *,
    client,
    experiment_id: str,
    dataset_spec,
    ds_args,
    scope_specs: list[PipelineSpec],
    architectures: list[str],
    data_sig: str,
) -> set[str]:
    all_signatures: set[str] = set()
    for spec in scope_specs:
        method = str(spec.pipeline_method).strip()
        for arch in architectures:
            policy_skip = scope_policy_skip_reason_for_spec(spec, arch)
            if policy_skip is not None:
                continue
            runner = create_pipeline_runner(spec, ds_args)
            try:
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    scope = runner.expected_tuning_scope(
                        client=client,
                        experiment_id=experiment_id,
                        dataset_spec=dataset_spec,
                        architecture=arch,
                        data_config_signature=data_sig,
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not resolve expected tuning scope for "
                    f"dataset='{dataset_spec.key}', architecture='{arch}', "
                    f"pipeline_method='{method}'."
                ) from exc
            all_signatures.update(scope.signature_set)
    return all_signatures


# ---------------------------------------------------------------------------
# Ledger row builder
# ---------------------------------------------------------------------------
LEDGER_FIELDS = [
    "run_id",
    "experiment_name",
    "dataset",
    "model_architecture",
    "pipeline_method",
    "pipeline_id",
    "pipeline_kind",
    "status",
    "lifecycle_stage",
    "start_time_iso",
    "end_time_iso",
    "evidence_duration_seconds",
    "evidence_duration_hours",
    "duration_source",
    "data_config_signature",
    "signature",
    "best_model",
    "backbone_current",
    "tested",
    "ledger_category",
]


def _build_ledger_row(
    run,
    *,
    experiment_name: str,
    dataset_key: str,
    duration_seconds: float,
    duration_source: str,
    tested: bool,
    ledger_category: str,
) -> dict:
    import math
    hours = duration_seconds / 3600.0 if not math.isnan(duration_seconds) else ""
    return {
        "run_id": run.info.run_id,
        "experiment_name": experiment_name,
        "dataset": dataset_key,
        "model_architecture": _safe_tag(run, "model_architecture"),
        "pipeline_method": _safe_tag(run, "pipeline_method"),
        "pipeline_id": _safe_tag(run, "pipeline_id"),
        "pipeline_kind": _safe_tag(run, "pipeline_kind"),
        "status": run.info.status,
        "lifecycle_stage": run.info.lifecycle_stage,
        "start_time_iso": _ts_to_iso(run.info.start_time),
        "end_time_iso": _ts_to_iso(run.info.end_time),
        "evidence_duration_seconds": duration_seconds if not math.isnan(duration_seconds) else "",
        "evidence_duration_hours": f"{hours:.6f}" if isinstance(hours, float) else "",
        "duration_source": duration_source,
        "data_config_signature": _safe_tag(run, "data_config_signature"),
        "signature": _safe_tag(run, "signature"),
        "best_model": _safe_tag(run, "best_model"),
        "backbone_current": _safe_tag(run, "backbone_current"),
        "tested": str(tested).lower(),
        "ledger_category": ledger_category,
    }


# ---------------------------------------------------------------------------
# Overhead audit helpers
# ---------------------------------------------------------------------------
OVERHEAD_FIELDS = [
    "run_id",
    "experiment_name",
    "experiment_prefix",
    "dataset",
    "model_architecture",
    "pipeline_method",
    "pipeline_kind",
    "status",
    "lifecycle_stage",
    "is_child_run",
    "start_time_iso",
    "end_time_iso",
    "evidence_duration_seconds",
    "evidence_duration_hours",
    "duration_source",
    "overhead_category",
]


def _build_overhead_row(
    run,
    *,
    experiment_name: str,
    experiment_prefix: str,
    dataset_key: str,
    duration_seconds: float,
    duration_source: str,
    overhead_category: str,
) -> dict:
    import math
    hours = duration_seconds / 3600.0 if not math.isnan(duration_seconds) else ""
    is_child = bool(_safe_tag(run, "mlflow.parentRunId"))
    return {
        "run_id": run.info.run_id,
        "experiment_name": experiment_name,
        "experiment_prefix": experiment_prefix,
        "dataset": dataset_key,
        "model_architecture": _safe_tag(run, "model_architecture"),
        "pipeline_method": _safe_tag(run, "pipeline_method"),
        "pipeline_kind": _safe_tag(run, "pipeline_kind"),
        "status": run.info.status,
        "lifecycle_stage": run.info.lifecycle_stage,
        "is_child_run": str(is_child).lower(),
        "start_time_iso": _ts_to_iso(run.info.start_time),
        "end_time_iso": _ts_to_iso(run.info.end_time),
        "evidence_duration_seconds": duration_seconds if not math.isnan(duration_seconds) else "",
        "evidence_duration_hours": f"{hours:.6f}" if isinstance(hours, float) else "",
        "duration_source": duration_source,
        "overhead_category": overhead_category,
    }


def _best_effort_duration(client, run) -> tuple[float, str]:
    dur, src = _resolve_run_duration_seconds(client, run)
    import math
    if not math.isnan(dur):
        return dur, src
    return _raw_span_seconds(run)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import math

    defaults = load_defaults()
    recipe_specs = load_benchmark_recipe_specs_for_scope(defaults)
    dataset_window_defaults = load_dataset_windows(defaults=defaults)
    args = _build_args_namespace(defaults, recipe_specs)
    scope_specs = _resolve_official_setup_compute_scope_specs(recipe_specs)
    scope_methods = tuple(spec.pipeline_method for spec in scope_specs)
    testing_scope_specs = _resolve_testing_compute_scope_specs(recipe_specs)

    architectures: list[str] = list(args.benchmark_architectures)
    datasets: list[str] = list(args.data_files)
    prefix: str = str(args.mlflow_experiment_prefix)

    tracking_uri = get_tracking_uri(args.logdir)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    dataset_specs_by_key = {
        spec.key: spec
        for spec in resolve_with_defaults(
            datasets=datasets,
            targets=args.data_targets,
            data_root=args.data_root,
        )
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Official per-run ledger
    # -----------------------------------------------------------------------
    print("=" * 78)
    print("Per-run ledger for official current-scope runs")
    print("=" * 78)

    all_ledger_rows: list[dict] = []
    all_overhead_rows: list[dict] = []

    # Aggregate accumulators
    agg_by_dataset: dict[str, dict] = {}
    agg_by_arch: dict[str, dict] = defaultdict(lambda: {"runs": 0, "seconds": 0.0})
    agg_by_method: dict[str, dict] = defaultdict(lambda: {"runs": 0, "seconds": 0.0})
    agg_by_dataset_arch: dict[tuple, dict] = defaultdict(lambda: {"runs": 0, "seconds": 0.0})
    agg_by_dataset_method: dict[tuple, dict] = defaultdict(lambda: {"runs": 0, "seconds": 0.0})

    # Testing aggregates
    test_agg_by_dataset: dict[str, dict] = {}
    test_agg_by_method: dict[str, dict] = defaultdict(lambda: {"runs": 0, "seconds": 0.0})

    for dataset_key in datasets:
        dataset_spec = dataset_specs_by_key[dataset_key]
        ds_args = resolve_dataset_window_args(
            args,
            dataset_spec=dataset_spec,
            dataset_window_defaults=dataset_window_defaults,
            explicit_arg_overrides={},
        )
        data_sig = compute_data_config_signature(
            dataset_spec=dataset_spec, args=ds_args
        )
        experiment_name = f"{prefix}-{dataset_key}"
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise RuntimeError(f"Expected MLflow experiment '{experiment_name}' not found.")
        experiment_id = experiment.experiment_id

        print(f"\n--- {experiment_name} ---")

        # Resolve current-scope signatures
        scope_sigs = _resolve_scope_signatures(
            client=client,
            experiment_id=experiment_id,
            dataset_spec=dataset_spec,
            ds_args=ds_args,
            scope_specs=scope_specs,
            architectures=architectures,
            data_sig=data_sig,
        )

        scope_expected_kinds = _expected_pipeline_kinds_by_cell(
            scope_specs=scope_specs,
            architectures=architectures,
        )

        # Query all finished parent runs in the current experiment. Runs outside
        # the official setup scope are accounted for as current-prefix overhead.
        filter_string = (
            "attribute.status = 'FINISHED'"
        )
        all_finished = search_runs_all(
            client, [experiment_id],
            filter_string=filter_string,
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        arch_set = set(architectures)
        method_set = set(scope_methods)
        parent_finished = [
            r for r in all_finished
            if not _safe_tag(r, "mlflow.parentRunId")
        ]
        in_scope = []
        out_of_scope = []
        for r in parent_finished:
            arch = _safe_tag(r, "model_architecture")
            method = _safe_tag(r, "pipeline_method")
            kind = _safe_tag(r, "pipeline_kind")
            expected_kind = scope_expected_kinds.get((arch, method))
            is_scope_member = (
                arch in arch_set
                and method in method_set
                and expected_kind is not None
                and kind == expected_kind
                and _safe_tag(r, "data_config_signature") == data_sig
                and _safe_tag(r, "signature") in scope_sigs
            )
            if is_scope_member:
                in_scope.append(r)
            else:
                out_of_scope.append(r)

        ds_runs = 0
        ds_seconds = 0.0

        for r in in_scope:
            dur, src = _resolve_official_setup_duration_seconds_strict(
                client,
                r,
                test_metric=str(args.test_metric),
            )
            tested = _safe_tag(r, "best_model") == "true" and _safe_tag(r, "backbone_current") == "true"
            kind = _safe_tag(r, "pipeline_kind")
            ledger_category = (
                "training_current_scope"
                if kind == "train"
                else "wrapper_current_scope"
            )
            row = _build_ledger_row(
                r,
                experiment_name=experiment_name,
                dataset_key=dataset_key,
                duration_seconds=dur,
                duration_source=src,
                tested=tested,
                ledger_category=ledger_category,
            )
            all_ledger_rows.append(row)
            ds_runs += 1
            ds_seconds += dur

            arch = _safe_tag(r, "model_architecture")
            method = _safe_tag(r, "pipeline_method")
            agg_by_arch[arch]["runs"] += 1
            agg_by_arch[arch]["seconds"] += dur
            agg_by_method[method]["runs"] += 1
            agg_by_method[method]["seconds"] += dur
            agg_by_dataset_arch[(dataset_key, arch)]["runs"] += 1
            agg_by_dataset_arch[(dataset_key, arch)]["seconds"] += dur
            agg_by_dataset_method[(dataset_key, method)]["runs"] += 1
            agg_by_dataset_method[(dataset_key, method)]["seconds"] += dur

        agg_by_dataset[dataset_key] = {"runs": ds_runs, "seconds": ds_seconds}
        print(f"  Setup (current-scope): {ds_runs} runs, {ds_seconds/3600:.1f} hrs")

        # Out-of-scope finished runs → overhead
        for r in out_of_scope:
            dur, src = _resolve_run_duration_seconds(client, r)
            all_overhead_rows.append(_build_overhead_row(
                r, experiment_name=experiment_name, experiment_prefix=prefix,
                dataset_key=dataset_key, duration_seconds=dur, duration_source=src,
                overhead_category="improv11_stale_out_of_scope",
            ))

        # --- Current-scope testing winners ---
        # Resolve expected testing kinds
        testing_expected_kinds: dict[tuple[str, str], str] = {}
        for spec in testing_scope_specs:
            method = str(spec.pipeline_method).strip()
            kind = str(spec.pipeline_kind).strip()
            if kind not in ALLOWED_PIPELINE_KINDS:
                continue
            for arch in architectures:
                policy_skip = scope_policy_skip_reason_for_spec(spec, arch)
                if policy_skip is not None:
                    continue
                testing_expected_kinds[(arch, method)] = kind

        filter_winners = (
            "attribute.status = 'FINISHED' "
            "AND tags.best_model = 'true' "
            "AND tags.backbone_current = 'true' "
            f"AND tags.data_config_signature = '{data_sig}'"
        )
        current_winners = search_runs_all(
            client, [experiment_id],
            filter_string=filter_winners,
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        current_winners = [
            r for r in current_winners
            if not _safe_tag(r, "mlflow.parentRunId")
        ]

        test_ds_runs = 0
        test_ds_seconds = 0.0

        for r in current_winners:
            tags = r.data.tags or {}
            try:
                resolved = resolve_pipeline_tags(tags, run_id=r.info.run_id)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not resolve pipeline tags for current winner "
                    f"{r.info.run_id} in {experiment_name}."
                ) from exc
            cell = (_safe_tag(r, "model_architecture"), resolved["pipeline_method"])
            expected_kind = testing_expected_kinds.get(cell)
            if expected_kind is None:
                continue
            if resolved["pipeline_kind"] != expected_kind:
                raise RuntimeError(
                    f"Current winner {r.info.run_id} in {experiment_name} has "
                    f"pipeline_kind={resolved['pipeline_kind']!r}, expected "
                    f"{expected_kind!r} for {cell}."
                )
            try:
                fully_tested = is_fully_tested(r, args=args, client=None)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not validate testing coverage for current winner "
                    f"{r.info.run_id} in {experiment_name}."
                ) from exc
            if not fully_tested:
                raise RuntimeError(
                    f"Current winner {r.info.run_id} in {experiment_name} is not "
                    "fully tested for the requested evaluation scope."
                )

            dur, src = _resolve_testing_duration_seconds(client, r)
            row = _build_ledger_row(
                r,
                experiment_name=experiment_name,
                dataset_key=dataset_key,
                duration_seconds=dur,
                duration_source=src,
                tested=True,
                ledger_category="testing_current_winner",
            )
            all_ledger_rows.append(row)
            if not math.isnan(dur):
                test_ds_runs += 1
                test_ds_seconds += dur

                method = resolved["pipeline_method"]
                test_agg_by_method[method]["runs"] += 1
                test_agg_by_method[method]["seconds"] += dur

        test_agg_by_dataset[dataset_key] = {"runs": test_ds_runs, "seconds": test_ds_seconds}
        print(f"  Testing (current winners): {test_ds_runs} runs, {test_ds_seconds/3600:.1f} hrs")

        # ---------------------------------------------------------------
        # Current-prefix non-current-scope overhead within this experiment
        # ---------------------------------------------------------------
        # Failed / killed / running runs
        for status_val in ["FAILED", "KILLED", "RUNNING"]:
            filt = f"attribute.status = '{status_val}'"
            status_runs = search_runs_all(
                client, [experiment_id],
                filter_string=filt,
                run_view_type=ViewType.ACTIVE_ONLY,
            )
            for r in status_runs:
                if _safe_tag(r, "mlflow.parentRunId"):
                    continue
                dur, src = _best_effort_duration(client, r)
                all_overhead_rows.append(_build_overhead_row(
                    r, experiment_name=experiment_name, experiment_prefix=prefix,
                    dataset_key=dataset_key, duration_seconds=dur, duration_source=src,
                    overhead_category=f"improv11_{status_val.lower()}",
                ))

        # Soft-deleted runs (ViewType.DELETED_ONLY)
        deleted_runs = search_runs_all(
            client, [experiment_id],
            filter_string="",
            run_view_type=ViewType.DELETED_ONLY,
        )
        for r in deleted_runs:
            if _safe_tag(r, "mlflow.parentRunId"):
                continue
            dur, src = _best_effort_duration(client, r)
            all_overhead_rows.append(_build_overhead_row(
                r, experiment_name=experiment_name, experiment_prefix=prefix,
                dataset_key=dataset_key, duration_seconds=dur, duration_source=src,
                overhead_category="improv11_deleted",
            ))

    # -----------------------------------------------------------------------
    # Older prefix and non-benchmark experiment overhead
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Older prefix and non-benchmark experiment overhead")
    print("=" * 78)

    all_experiments = client.search_experiments()
    current_experiment_names = {f"{prefix}-{ds}" for ds in datasets}

    for exp in sorted(all_experiments, key=lambda e: e.name):
        if exp.name in current_experiment_names:
            continue
        if exp.name == "Default":
            exp_prefix = "default"
            exp_dataset = ""
        elif exp.name == "Meta Analysis":
            exp_prefix = "meta_analysis"
            exp_dataset = ""
        elif exp.name.startswith("improv-"):
            parts = exp.name.split("-", 2)
            exp_prefix = f"improv-{parts[1]}" if len(parts) >= 2 else exp.name
            exp_dataset = parts[2] if len(parts) >= 3 else ""
        else:
            exp_prefix = exp.name
            exp_dataset = ""

        # Query all runs (active and deleted)
        for view_type, view_label in [
            (ViewType.ACTIVE_ONLY, "active"),
            (ViewType.DELETED_ONLY, "deleted"),
        ]:
            runs = search_runs_all(
                client, [exp.experiment_id],
                filter_string="",
                run_view_type=view_type,
            )
            parent_runs = [r for r in runs if not _safe_tag(r, "mlflow.parentRunId")]
            if not parent_runs:
                continue
            print(f"  {exp.name} ({view_label}): {len(parent_runs)} parent runs")

            for r in parent_runs:
                dur, src = _best_effort_duration(client, r)
                status = r.info.status
                category = f"{exp_prefix}_{view_label}_{status.lower()}"
                all_overhead_rows.append(_build_overhead_row(
                    r, experiment_name=exp.name, experiment_prefix=exp_prefix,
                    dataset_key=exp_dataset, duration_seconds=dur,
                    duration_source=src, overhead_category=category,
                ))

    # -----------------------------------------------------------------------
    # Write CSVs
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Writing output CSVs")
    print("=" * 78)

    # 1. Per-run ledger CSV
    ledger_path = OUTPUT_DIR / "compute_ledger_official_runs.csv"
    with open(ledger_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for row in sorted(all_ledger_rows, key=lambda r: (
            r["dataset"], r["ledger_category"], r["model_architecture"],
            r["pipeline_method"], r["start_time_iso"],
        )):
            writer.writerow(row)
    print(f"  {ledger_path} ({len(all_ledger_rows)} rows)")

    # 2. Aggregate summaries CSV
    agg_path = OUTPUT_DIR / "compute_aggregates.csv"
    agg_rows: list[dict] = []

    # By dataset (official training setup spans).
    for ds in datasets:
        if ds in agg_by_dataset:
            d = agg_by_dataset[ds]
            agg_rows.append({
                "group": "training_by_dataset", "key": ds,
                "runs": d["runs"], "hours": f"{d['seconds']/3600:.2f}",
            })

    # By architecture (official training setup spans).
    for arch in architectures:
        if arch in agg_by_arch:
            d = agg_by_arch[arch]
            agg_rows.append({
                "group": "training_by_architecture", "key": arch,
                "runs": d["runs"], "hours": f"{d['seconds']/3600:.2f}",
            })

    # By method (official training setup spans).
    for method in scope_methods:
        if method in agg_by_method:
            d = agg_by_method[method]
            agg_rows.append({
                "group": "training_by_method", "key": method,
                "runs": d["runs"], "hours": f"{d['seconds']/3600:.2f}",
            })

    # By dataset x architecture (official training setup spans).
    for ds in datasets:
        for arch in architectures:
            k = (ds, arch)
            if k in agg_by_dataset_arch:
                d = agg_by_dataset_arch[k]
                if d["runs"] > 0:
                    agg_rows.append({
                        "group": "training_by_dataset_architecture",
                        "key": f"{ds}/{arch}",
                        "runs": d["runs"], "hours": f"{d['seconds']/3600:.2f}",
                    })

    # By dataset x method (official training setup spans).
    for ds in datasets:
        for method in scope_methods:
            k = (ds, method)
            if k in agg_by_dataset_method:
                d = agg_by_dataset_method[k]
                if d["runs"] > 0:
                    agg_rows.append({
                        "group": "training_by_dataset_method",
                        "key": f"{ds}/{method}",
                        "runs": d["runs"], "hours": f"{d['seconds']/3600:.2f}",
                    })

    # Testing by dataset
    for ds in datasets:
        if ds in test_agg_by_dataset:
            d = test_agg_by_dataset[ds]
            agg_rows.append({
                "group": "testing_by_dataset", "key": ds,
                "runs": d["runs"], "hours": f"{d['seconds']/3600:.2f}",
            })

    # Testing by method
    for method in sorted(test_agg_by_method):
        d = test_agg_by_method[method]
        agg_rows.append({
            "group": "testing_by_method", "key": method,
            "runs": d["runs"], "hours": f"{d['seconds']/3600:.2f}",
        })

    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "key", "runs", "hours"])
        writer.writeheader()
        for row in agg_rows:
            writer.writerow(row)
    print(f"  {agg_path} ({len(agg_rows)} rows)")

    # 3. Overhead audit CSV
    overhead_path = OUTPUT_DIR / "compute_overhead_audit.csv"
    with open(overhead_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OVERHEAD_FIELDS)
        writer.writeheader()
        for row in sorted(all_overhead_rows, key=lambda r: (
            r["experiment_prefix"], r["experiment_name"],
            r["overhead_category"], r["start_time_iso"],
        )):
            writer.writerow(row)
    print(f"  {overhead_path} ({len(all_overhead_rows)} rows)")

    # -----------------------------------------------------------------------
    # Overhead summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("OVERHEAD AUDIT SUMMARY")
    print("=" * 78)

    cat_summary: dict[str, dict] = defaultdict(
        lambda: {"parent_runs": 0, "seconds_with_evidence": 0.0, "runs_with_evidence": 0}
    )
    for row in all_overhead_rows:
        cat = row["overhead_category"]
        cat_summary[cat]["parent_runs"] += 1
        dur = row["evidence_duration_seconds"]
        if dur != "" and not math.isnan(float(dur)):
            cat_summary[cat]["seconds_with_evidence"] += float(dur)
            cat_summary[cat]["runs_with_evidence"] += 1

    print(f"  {'category':<50}  {'parents':>8}  {'w/evidence':>10}  {'hours':>10}")
    print(f"  {'-'*50}  {'-'*8}  {'-'*10}  {'-'*10}")
    total_overhead_parents = 0
    total_overhead_seconds = 0.0
    total_overhead_with_evidence = 0
    for cat in sorted(cat_summary):
        d = cat_summary[cat]
        hrs = d["seconds_with_evidence"] / 3600.0
        print(
            f"  {cat:<50}  {d['parent_runs']:>8}  "
            f"{d['runs_with_evidence']:>10}  {hrs:>10.1f}"
        )
        total_overhead_parents += d["parent_runs"]
        total_overhead_seconds += d["seconds_with_evidence"]
        total_overhead_with_evidence += d["runs_with_evidence"]
    print(f"  {'-'*50}  {'-'*8}  {'-'*10}  {'-'*10}")
    print(
        f"  {'TOTAL':<50}  {total_overhead_parents:>8}  "
        f"{total_overhead_with_evidence:>10}  {total_overhead_seconds/3600:.1f}"
    )

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    total_train_runs = sum(d["runs"] for d in agg_by_dataset.values())
    total_train_seconds = sum(d["seconds"] for d in agg_by_dataset.values())
    total_test_runs = sum(d["runs"] for d in test_agg_by_dataset.values())
    total_test_seconds = sum(d["seconds"] for d in test_agg_by_dataset.values())

    print("\n" + "=" * 78)
    print("OFFICIAL COMPUTE TOTALS (confirmed)")
    print("=" * 78)
    print(f"  Setup runs (current scope):              {total_train_runs:,}")
    print(f"  Setup GPU-hours:                         {total_train_seconds/3600:,.0f}")
    print(f"  Testing winner runs:                     {total_test_runs:,}")
    print(f"  Testing GPU-hours (lower bound):         {total_test_seconds/3600:,.0f}")
    print(f"  Combined official:                       {(total_train_seconds+total_test_seconds)/3600:,.0f}")
    print()
    print(f"  Full-project overhead parent runs:       {total_overhead_parents:,}")
    print(f"  Full-project overhead GPU-hours (est.):  {total_overhead_seconds/3600:,.0f}")
    print(
        f"  Full-project total (official + overhead): "
        f"{(total_train_seconds+total_test_seconds+total_overhead_seconds)/3600:,.0f}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
