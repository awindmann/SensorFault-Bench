"""Compute accounting for the experiments of the paper.

The reference record is the compute accounting of version 0.1.0: the official
spans and the overhead audit of the reference campaign, recorded on
2026-05-06. Every other campaign of the config is an added campaign, and the
analyses of the config own the evaluation episodes beyond the official ones.
One read-only pass over the experiments of the campaigns reproduces the
reference campaign's official spans run by run and carries the reference
overhead audit over unchanged. It writes one report:

1. A per-run ledger of official training runs and official evaluations.
2. One row per counted evaluation episode: the official episode of every
   winner, and every other episode that starts in the window of an analysis
   of its campaign.
3. Aggregate summaries per campaign, and the evaluation episodes per analysis.
4. The overhead audit: the carried-over reference rows, then the failed,
   killed, running, and soft-deleted parent runs in the experiments of the
   added campaigns.

An official training span runs from the run's ``start_time`` to the last entry
of the first available metric in ``ep_val_loss``, ``best_val_loss``, and
``train_loss``. Evaluation episodes come from the metric history of
``${TEST_METRIC}_val`` and the degradation keys required by the campaign's
``TEST_METRIC`` and ``PERTURBATION_SCENARIOS``. Run eval-context tags are
mutable and are not read. Every evaluation logs its own validation anchor
before its degradation metrics, so anchor ``a_i`` opens the interval
``(a_i, a_{i+1}]`` and the last interval is open-ended. An interval with
entries of every required key is a complete episode, an interval with entries
of only some required keys is an incomplete episode, and an interval without
required-key entries is not an episode. Each winner has one official episode:
for the reference campaign the episode whose span equals the run's span in the
reference record, and for every added campaign the earliest complete episode.
Every other counted episode is additional when it starts after the reference
record's last evaluation, and prior otherwise. Every analysis window must hold
at least one episode. Evaluation spans are lower bounds because they exclude
model loading and artifact uploads.

The campaigns read the tracking server from ``LOGDIR`` in
``configs/defaults.yaml``, which must point at the server that holds the
recorded runs.

Read-only: does NOT call set_tag, log_metric, log_param, delete_run, or any
rerun/mutation command.

Run from the repo root, with the reference record taken from the v0.1.0
release:

    git show v0.1.0:evidence/compute_accounting/compute_ledger_official_runs.csv \\
      > runs/reference_ledger.csv
    git show v0.1.0:evidence/compute_accounting/compute_overhead_audit.csv \\
      > runs/reference_overhead.csv
    uv run python scripts/compute_accounting_ledger.py \\
      --config configs/reporting/compute_accounting.yaml \\
      --reference-ledger runs/reference_ledger.csv \\
      --reference-overhead runs/reference_overhead.csv \\
      --output-dir runs/compute_accounting
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import os
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow.entities import ViewType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import load_defaults
from data.datasets import ResolvedDatasetSpec, resolve_with_defaults
from pipelines.ranking import ALLOWED_PIPELINE_KINDS
from pipelines.runner import create_pipeline_runner, scope_policy_skip_reason_for_spec
from pipelines.selection import (
    resolve_pipeline_tags,
    resolve_requested_architectures,
    resolve_requested_methods,
)
from pipelines.signatures import compute_data_config_signature
from pipelines.specs import PipelineSpec
from pipelines.training import get_tracking_uri, search_runs_all
from run_testing import prepare_testing_entrypoint_args
from utils.parsing import (
    ComputeCampaign,
    ComputePaperAnalysis,
    compute_campaign_cli_tokens,
    parse_compute_accounting_config,
    parse_duration_milliseconds,
    parse_iso_timestamp_milliseconds,
    parse_perturbation_scenarios,
    parse_required_nonempty_string,
    parse_winner_candidate_tags,
    require_tested_param,
    resolve_dataset_window_args,
)
from utils.scoring import required_degradation_metric_keys

# Pause between experiments to limit the load on the shared MLflow server.
QUERY_PAUSE_SECONDS = 5
REFERENCE_CAMPAIGN_ID = "reference_seed42"
DURATION_METRIC_KEYS: tuple[str, ...] = ("ep_val_loss", "best_val_loss", "train_loss")
TRAINING_CATEGORY = "training_current_scope"
EVALUATION_CATEGORY = "testing_current_winner"
# Overhead categories of an added campaign are "<campaign_id>_<label>".
ATTEMPT_OVERHEAD_STATUSES = ("FAILED", "KILLED", "RUNNING")
DELETED_OVERHEAD_LABEL = "deleted"

LEDGER_FIELDS = [
    "campaign_id",
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
EPISODE_FIELDS = [
    "campaign_id",
    "run_id",
    "dataset",
    "model_architecture",
    "pipeline_method",
    "is_winner",
    "episode_index",
    "anchor_time_iso",
    "end_time_iso",
    "span_seconds",
    "episode_class",
    "complete",
    "analysis",
]
AGGREGATE_FIELDS = ["campaign_id", "group", "key", "runs", "hours"]
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
LEDGER_AGGREGATE_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("training_by_dataset", TRAINING_CATEGORY, ("dataset",)),
    ("training_by_architecture", TRAINING_CATEGORY, ("model_architecture",)),
    ("training_by_method", TRAINING_CATEGORY, ("pipeline_method",)),
    (
        "training_by_dataset_architecture",
        TRAINING_CATEGORY,
        ("dataset", "model_architecture"),
    ),
    ("training_by_dataset_method", TRAINING_CATEGORY, ("dataset", "pipeline_method")),
    ("testing_by_dataset", EVALUATION_CATEGORY, ("dataset",)),
    ("testing_by_method", EVALUATION_CATEGORY, ("pipeline_method",)),
)
ANALYSIS_AGGREGATE_GROUP = "evaluations_by_analysis"


@dataclass(frozen=True)
class _CampaignDataset:
    spec: ResolvedDatasetSpec
    args: argparse.Namespace
    data_config_signature: str
    experiment_name: str


@dataclass(frozen=True)
class _Campaign:
    campaign_id: str
    experiment_prefix: str
    architectures: tuple[str, ...]
    training_specs: tuple[PipelineSpec, ...]
    expected_cells: dict[tuple[str, str], str]
    anchor_key: str
    required_keys: tuple[str, ...]
    datasets: tuple[_CampaignDataset, ...]


@dataclass(frozen=True)
class _ReferenceRecord:
    training_ms: dict[str, int]
    evaluation_ms: dict[str, int]
    evaluation_end_ms: int
    overhead_rows: list[dict]


@dataclass(frozen=True)
class _Episode:
    anchor_ms: int
    end_ms: int
    end_key: str
    complete: bool

    @property
    def span_ms(self) -> int:
        return self.end_ms - self.anchor_ms


def _ts_to_iso(ts_ms: int | None) -> str:
    if ts_ms is None:
        return ""
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()


def _require_run_tag(run, key: str) -> str:
    tags = run.data.tags or {}
    raw_value = tags.get(key)
    if raw_value is None or not str(raw_value).strip():
        raise ValueError(f"Run {run.info.run_id} is missing required MLflow tag '{key}'.")
    return str(raw_value).strip()


def _safe_tag(run, key: str) -> str:
    """Best-effort tag read, restricted to overhead rows."""
    value = run.data.tags.get(key)
    return str(value).strip() if value is not None else ""


def _hours(seconds: Sequence[float]) -> float:
    return math.fsum(seconds) / 3600.0


# ---------------------------------------------------------------------------
# Campaign resolution
# ---------------------------------------------------------------------------
def _select_training_compute_scope_specs(
    specs: Sequence[PipelineSpec],
) -> list[PipelineSpec]:
    """Return the train-kind recipes. Wrap recipes add no training span."""
    selected: list[PipelineSpec] = []
    seen_methods: set[str] = set()
    for spec in specs:
        method = str(spec.pipeline_method).strip()
        if method in seen_methods:
            raise ValueError(f"Duplicate configured pipeline_method '{method}'.")
        seen_methods.add(method)
        kind = str(spec.pipeline_kind).strip()
        if kind == "wrap":
            continue
        if kind != "train":
            raise ValueError(
                "Training compute accounting supports train and wrap recipes, but "
                f"method '{method}' has pipeline_kind='{kind}'."
            )
        selected.append(spec)
    if not selected:
        raise ValueError("No train-kind recipes remain in the training compute scope.")
    return selected


def _resolve_testing_expected_kinds(
    *,
    scope_specs: Sequence[PipelineSpec],
    architectures: Sequence[str],
) -> dict[tuple[str, str], str]:
    """Map every applicable (architecture, method) cell to its pipeline kind."""
    expected_kinds: dict[tuple[str, str], str] = {}
    for spec in scope_specs:
        method = str(spec.pipeline_method).strip()
        if not method:
            raise ValueError("Configured pipeline spec is missing pipeline_method.")
        kind = str(spec.pipeline_kind).strip()
        if kind not in ALLOWED_PIPELINE_KINDS:
            raise ValueError(
                f"Configured pipeline method '{method}' has unknown pipeline_kind '{kind}'."
            )
        for arch in architectures:
            if scope_policy_skip_reason_for_spec(spec, arch) is None:
                expected_kinds[(arch, method)] = kind
    if not expected_kinds:
        raise RuntimeError("Expected testing scope is empty after policy skips.")
    return expected_kinds


def _resolve_campaign(campaign: ComputeCampaign, *, defaults: dict) -> _Campaign:
    """Resolve one campaign through the testing entry point's argument path."""
    tokens = compute_campaign_cli_tokens(campaign.overrides, defaults=defaults)
    args, recipe_specs, dataset_window_defaults = prepare_testing_entrypoint_args(
        cli_args=tokens
    )
    methods = resolve_requested_methods(
        args,
        configured_methods=tuple(spec.pipeline_method for spec in recipe_specs),
    )
    scope_specs = [spec for spec in recipe_specs if spec.pipeline_method in methods]
    architectures = resolve_requested_architectures(args)
    test_metric = parse_required_nonempty_string(args.test_metric, key="TEST_METRIC")
    scenarios = parse_perturbation_scenarios(
        args.perturbation_scenarios,
        key="PERTURBATION_SCENARIOS",
    )
    dataset_specs = resolve_with_defaults(
        datasets=args.data_files,
        targets=args.data_targets,
        data_root=args.data_root,
    )
    datasets = []
    for data_file, dataset_spec in zip(args.data_files, dataset_specs, strict=True):
        if dataset_spec.key != data_file:
            raise ValueError(
                f"Compute campaign '{campaign.campaign_id}' must name dataset "
                f"'{dataset_spec.key}' by its registered key, got '{data_file}'."
            )
        dataset_args = resolve_dataset_window_args(
            args,
            dataset_spec=dataset_spec,
            dataset_window_defaults=dataset_window_defaults,
            explicit_arg_overrides=args._window_arg_overrides,
        )
        datasets.append(_CampaignDataset(
            spec=dataset_spec,
            args=dataset_args,
            data_config_signature=compute_data_config_signature(
                dataset_spec=dataset_spec,
                args=dataset_args,
            ),
            experiment_name=f"{args.mlflow_experiment_prefix}-{dataset_spec.key}",
        ))
    return _Campaign(
        campaign_id=campaign.campaign_id,
        experiment_prefix=args.mlflow_experiment_prefix,
        architectures=tuple(architectures),
        training_specs=tuple(_select_training_compute_scope_specs(scope_specs)),
        expected_cells=_resolve_testing_expected_kinds(
            scope_specs=scope_specs,
            architectures=architectures,
        ),
        anchor_key=f"{test_metric}_val",
        required_keys=required_degradation_metric_keys(
            test_metric=test_metric,
            scenario_indices=range(len(scenarios)),
        ),
        datasets=tuple(datasets),
    )


def _load_reference_record(ledger_path: Path, overhead_path: Path) -> _ReferenceRecord:
    """Load the official spans and overhead rows of the reference record.

    The record ends when its last official evaluation run ended. Episodes that
    start later are added compute.
    """
    spans: dict[str, dict[str, int]] = {TRAINING_CATEGORY: {}, EVALUATION_CATEGORY: {}}
    evaluation_ends: list[int] = []
    with open(ledger_path, newline="") as handle:
        for row in csv.DictReader(handle):
            run_id = row["run_id"]
            category = row["ledger_category"]
            if category not in spans:
                raise ValueError(
                    f"{ledger_path} has unknown ledger_category '{category}' for run "
                    f"{run_id}."
                )
            if run_id in spans[category]:
                raise ValueError(
                    f"{ledger_path} has duplicate {category} rows for run {run_id}."
                )
            spans[category][run_id] = parse_duration_milliseconds(
                row["evidence_duration_seconds"],
                key=f"{ledger_path} run {run_id} evidence_duration_seconds",
            )
            if category == EVALUATION_CATEGORY:
                evaluation_ends.append(parse_iso_timestamp_milliseconds(
                    row["end_time_iso"],
                    key=f"{ledger_path} run {run_id} end_time_iso",
                ))
    if not evaluation_ends:
        raise ValueError(f"{ledger_path} has no {EVALUATION_CATEGORY} rows.")
    with open(overhead_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != OVERHEAD_FIELDS:
            raise ValueError(
                f"{overhead_path} has columns {reader.fieldnames}, expected "
                f"{OVERHEAD_FIELDS}."
            )
        overhead_rows = [
            {
                **row,
                "evidence_duration_seconds": parse_duration_milliseconds(
                    row["evidence_duration_seconds"],
                    key=f"{overhead_path} run {row['run_id']} evidence_duration_seconds",
                ) / 1000.0,
            }
            for row in reader
        ]
    return _ReferenceRecord(
        training_ms=spans[TRAINING_CATEGORY],
        evaluation_ms=spans[EVALUATION_CATEGORY],
        evaluation_end_ms=max(evaluation_ends),
        overhead_rows=overhead_rows,
    )


# ---------------------------------------------------------------------------
# MLflow reads
# ---------------------------------------------------------------------------
def _query_parent_runs(
    client,
    experiment_id: str,
    filter_string: str | None,
    *,
    view_type=ViewType.ACTIVE_ONLY,
) -> list:
    runs = search_runs_all(
        client,
        [experiment_id],
        filter_string=filter_string,
        run_view_type=view_type,
    )
    return [run for run in runs if not run.data.tags.get("mlflow.parentRunId")]


def _history_timestamps(client, run_id: str, key: str) -> list[int]:
    return [int(entry.timestamp) for entry in client.get_metric_history(run_id, key)]


def _resolve_run_duration_seconds(client, run) -> tuple[float, str]:
    """Return the span from ``start_time`` to the last duration-metric entry.

    The metric timestamps bound the training work more tightly than
    ``end_time``, which a lingering process can move later.
    """
    start = run.info.start_time
    if start is None:
        raise ValueError(f"Run {run.info.run_id} is missing start_time.")
    for key in DURATION_METRIC_KEYS:
        timestamps = _history_timestamps(client, run.info.run_id, key)
        if not timestamps:
            continue
        last_ts = max(timestamps)
        if last_ts <= start:
            raise ValueError(
                f"Run {run.info.run_id} metric '{key}' has no timestamp after "
                f"start_time ({last_ts} <= {start})."
            )
        return (last_ts - start) / 1000.0, f"metric:{key}"
    raise ValueError(
        f"Run {run.info.run_id} has no duration metric history. "
        f"Expected at least one of: {', '.join(DURATION_METRIC_KEYS)}."
    )


def _resolve_scope_signatures(
    *,
    client,
    experiment_id: str,
    dataset_spec,
    ds_args,
    scope_specs: Sequence[PipelineSpec],
    architectures: Sequence[str],
    data_sig: str,
) -> set[str]:
    """Return the tuning signatures of the expected training scope."""
    signatures: set[str] = set()
    for spec in scope_specs:
        method = str(spec.pipeline_method).strip()
        for arch in architectures:
            if scope_policy_skip_reason_for_spec(spec, arch) is not None:
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
                    "Could not resolve expected tuning scope for "
                    f"dataset='{dataset_spec.key}', architecture='{arch}', "
                    f"pipeline_method='{method}'."
                ) from exc
            signatures.update(scope.signature_set)
    return signatures


def _query_finished_train_runs(
    *,
    client,
    experiment_id: str,
    architectures: Sequence[str],
    scope_methods: Sequence[str],
    data_sig: str,
) -> list:
    filter_string = (
        "attribute.status = 'FINISHED' "
        "AND tags.pipeline_kind = 'train' "
        f"AND tags.data_config_signature = '{data_sig}'"
    )
    runs = _query_parent_runs(client, experiment_id, filter_string)
    filtered = []
    for run in runs:
        architecture = _require_run_tag(run, "model_architecture")
        method = _require_run_tag(run, "pipeline_method")
        if architecture not in architectures or method not in scope_methods:
            continue
        _require_run_tag(run, "signature")
        filtered.append(run)
    return filtered


# ---------------------------------------------------------------------------
# Official rows
# ---------------------------------------------------------------------------
def _ledger_row(
    run,
    *,
    campaign_id: str,
    dataset: _CampaignDataset,
    seconds: float,
    duration_source: str,
    ledger_category: str,
) -> dict:
    run_id = run.info.run_id
    pipeline_kind = _require_run_tag(run, "pipeline_kind")
    winner_tags = parse_winner_candidate_tags(run.data.tags, run_id=run_id)
    return {
        "campaign_id": campaign_id,
        "run_id": run_id,
        "experiment_name": dataset.experiment_name,
        "dataset": dataset.spec.key,
        "model_architecture": _require_run_tag(run, "model_architecture"),
        "pipeline_method": _require_run_tag(run, "pipeline_method"),
        "pipeline_id": _require_run_tag(run, "pipeline_id"),
        "pipeline_kind": pipeline_kind,
        "status": run.info.status,
        "lifecycle_stage": run.info.lifecycle_stage,
        "start_time_iso": _ts_to_iso(run.info.start_time),
        "end_time_iso": _ts_to_iso(run.info.end_time),
        "evidence_duration_seconds": seconds,
        "evidence_duration_hours": f"{seconds / 3600.0:.6f}",
        "duration_source": duration_source,
        "data_config_signature": _require_run_tag(run, "data_config_signature"),
        # Wrap runs are not tuned, so they carry no tuning signature.
        "signature": "" if pipeline_kind == "wrap" else _require_run_tag(run, "signature"),
        "best_model": "" if winner_tags is None else str(winner_tags[0]).lower(),
        "backbone_current": "" if winner_tags is None else str(winner_tags[1]).lower(),
        "tested": str(winner_tags == (True, True)).lower(),
        "ledger_category": ledger_category,
    }


def _training_rows(
    client,
    campaign: _Campaign,
    dataset: _CampaignDataset,
    *,
    experiment_id: str,
) -> list[dict]:
    """Return the official training rows of one campaign experiment."""
    scope_signatures = _resolve_scope_signatures(
        client=client,
        experiment_id=experiment_id,
        dataset_spec=dataset.spec,
        ds_args=dataset.args,
        scope_specs=campaign.training_specs,
        architectures=campaign.architectures,
        data_sig=dataset.data_config_signature,
    )
    finished_runs = _query_finished_train_runs(
        client=client,
        experiment_id=experiment_id,
        architectures=campaign.architectures,
        scope_methods=tuple(spec.pipeline_method for spec in campaign.training_specs),
        data_sig=dataset.data_config_signature,
    )
    ledger_rows: list[dict] = []
    for run in finished_runs:
        if _require_run_tag(run, "signature") not in scope_signatures:
            continue
        seconds, source = _resolve_run_duration_seconds(client, run)
        ledger_rows.append(_ledger_row(
            run,
            campaign_id=campaign.campaign_id,
            dataset=dataset,
            seconds=seconds,
            duration_source=source,
            ledger_category=TRAINING_CATEGORY,
        ))
    if not ledger_rows:
        raise RuntimeError(
            f"No current-scope finished parent train runs in '{dataset.experiment_name}'."
        )
    return ledger_rows


def _select_winners(
    runs: Sequence,
    *,
    expected_cells: Mapping[tuple[str, str], str],
) -> dict[tuple[str, str], Any]:
    """Map each expected testing cell to its tagged winner. Winners are never re-ranked."""
    winners: dict[tuple[str, str], Any] = {}
    for run in runs:
        run_id = run.info.run_id
        tags = run.data.tags
        if parse_winner_candidate_tags(tags, run_id=run_id) != (True, True):
            raise ValueError(
                f"Run {run_id} from the best_model=true query is not tagged "
                "best_model=true and backbone_current=true."
            )
        if not require_tested_param(run.data.params, run_id=run_id):
            raise ValueError(f"Winner run {run_id} has tested=false.")
        resolved = resolve_pipeline_tags(tags, run_id=run_id)
        cell = (_require_run_tag(run, "model_architecture"), resolved["pipeline_method"])
        if cell not in expected_cells:
            raise RuntimeError(
                f"Winner run {run_id} is outside the expected testing cells: {cell}."
            )
        if resolved["pipeline_kind"] != expected_cells[cell]:
            raise RuntimeError(
                f"Winner run {run_id} for testing cell {cell} has pipeline_kind="
                f"'{resolved['pipeline_kind']}', expected '{expected_cells[cell]}'."
            )
        if cell in winners:
            raise RuntimeError(
                f"Testing cell {cell} has two winners: "
                f"{winners[cell].info.run_id} and {run_id}."
            )
        winners[cell] = run
    missing = sorted(set(expected_cells) - set(winners))
    if missing:
        raise RuntimeError(
            f"{len(missing)} expected testing cell(s) have no winner: {missing}."
        )
    return winners


def _split_evaluation_episodes(
    *,
    run_id: str,
    anchors: Sequence[int],
    required_entries: Mapping[str, Sequence[int]],
) -> list[_Episode]:
    """Split required-key history into episodes at the validation anchors."""
    anchor_times = sorted(set(anchors))
    entries = sorted(
        (timestamp, key)
        for key, timestamps in required_entries.items()
        for timestamp in timestamps
    )
    if entries and (not anchor_times or entries[0][0] <= anchor_times[0]):
        timestamp, key = entries[0]
        raise ValueError(
            f"Run {run_id} has required metric '{key}' at {timestamp}, at or before "
            "its first evaluation anchor."
        )
    episodes = []
    for anchor, upper in zip(anchor_times, [*anchor_times[1:], None]):
        inside = [
            entry for entry in entries
            if entry[0] > anchor and (upper is None or entry[0] <= upper)
        ]
        if not inside:
            continue
        end_ms, end_key = max(inside)
        episodes.append(_Episode(
            anchor_ms=anchor,
            end_ms=end_ms,
            end_key=end_key,
            complete={key for _, key in inside} == set(required_entries),
        ))
    return episodes


def _evaluation_episodes(client, run_id: str, campaign: _Campaign) -> list[_Episode]:
    return _split_evaluation_episodes(
        run_id=run_id,
        anchors=_history_timestamps(client, run_id, campaign.anchor_key),
        required_entries={
            key: _history_timestamps(client, run_id, key)
            for key in campaign.required_keys
        },
    )


def _select_official_episode(
    episodes: Sequence[_Episode],
    *,
    run_id: str,
    reference_ms: int | None,
) -> _Episode:
    """Return the complete episode spanning ``reference_ms``, else the earliest one."""
    complete = [episode for episode in episodes if episode.complete]
    if reference_ms is None:
        if not complete:
            raise RuntimeError(f"Winner run {run_id} has no complete evaluation episode.")
        return complete[0]
    matches = [episode for episode in complete if episode.span_ms == reference_ms]
    if len(matches) != 1:
        raise RuntimeError(
            f"Reference winner run {run_id} has {len(matches)} complete evaluation "
            f"episodes spanning the recorded {reference_ms} ms, expected exactly one."
        )
    return matches[0]


def _evaluation_rows(
    client,
    campaign: _Campaign,
    dataset: _CampaignDataset,
    *,
    experiment_id: str,
    reference_evaluation_ms: Mapping[str, int] | None,
    record_end_ms: int,
    analyses: Sequence[ComputePaperAnalysis],
) -> tuple[list[dict], list[dict]]:
    """Return official evaluation rows and the counted episode rows of every tested run.

    A non-official episode counts when it starts in the window of one of the
    campaign's ``analyses``. It is additional when it starts after the
    reference record ends at ``record_end_ms``, and prior otherwise.
    """
    scope_filter = (
        "attribute.status = 'FINISHED' "
        f"AND tags.data_config_signature = '{dataset.data_config_signature}'"
    )
    winners = _select_winners(
        _query_parent_runs(
            client,
            experiment_id,
            f"{scope_filter} AND tags.best_model = 'true'",
        ),
        expected_cells=campaign.expected_cells,
    )
    winner_ids = {run.info.run_id for run in winners.values()}
    tested_runs = _query_parent_runs(
        client,
        experiment_id,
        f"{scope_filter} AND params.tested = 'true'",
    )
    untested = sorted(winner_ids - {run.info.run_id for run in tested_runs})
    if untested:
        raise RuntimeError(
            f"Winner run(s) missing from the params.tested query: {untested}."
        )
    ledger_rows: list[dict] = []
    episode_rows: list[dict] = []
    for run in tested_runs:
        run_id = run.info.run_id
        architecture = _require_run_tag(run, "model_architecture")
        pipeline_method = _require_run_tag(run, "pipeline_method")
        is_winner = run_id in winner_ids
        episodes = _evaluation_episodes(client, run_id, campaign)
        official = None
        if is_winner:
            reference_ms = None
            if reference_evaluation_ms is not None:
                if run_id not in reference_evaluation_ms:
                    raise RuntimeError(
                        f"Reference winner run {run_id} has no {EVALUATION_CATEGORY} "
                        "row in the reference ledger."
                    )
                reference_ms = reference_evaluation_ms[run_id]
            official = _select_official_episode(
                episodes,
                run_id=run_id,
                reference_ms=reference_ms,
            )
            ledger_rows.append(_ledger_row(
                run,
                campaign_id=campaign.campaign_id,
                dataset=dataset,
                seconds=official.span_ms / 1000.0,
                duration_source=f"metric:{campaign.anchor_key}->{official.end_key}",
                ledger_category=EVALUATION_CATEGORY,
            ))
        for index, episode in enumerate(episodes):
            if episode is official:
                episode_class, analysis_id = "official", ""
            else:
                # Windows of one campaign do not overlap, so at most one matches.
                matching = [
                    analysis.analysis_id for analysis in analyses
                    if analysis.window_start_ms <= episode.anchor_ms < analysis.window_end_ms
                ]
                if not matching:
                    continue
                (analysis_id,) = matching
                episode_class = "additional" if episode.anchor_ms > record_end_ms else "prior"
            episode_rows.append({
                "campaign_id": campaign.campaign_id,
                "run_id": run_id,
                "dataset": dataset.spec.key,
                "model_architecture": architecture,
                "pipeline_method": pipeline_method,
                "is_winner": str(is_winner).lower(),
                "episode_index": index,
                "anchor_time_iso": _ts_to_iso(episode.anchor_ms),
                "end_time_iso": _ts_to_iso(episode.end_ms),
                "span_seconds": episode.span_ms / 1000.0,
                "episode_class": episode_class,
                "complete": str(episode.complete).lower(),
                "analysis": analysis_id,
            })
    return ledger_rows, episode_rows


def _check_reference_reproduction(
    rows: Sequence[dict],
    *,
    training_ms: Mapping[str, int],
    evaluation_ms: Mapping[str, int],
) -> None:
    """Require the reference campaign to reproduce the reference record run by run."""
    for category, expected in (
        (TRAINING_CATEGORY, training_ms),
        (EVALUATION_CATEGORY, evaluation_ms),
    ):
        recomputed = {
            row["run_id"]: parse_duration_milliseconds(
                row["evidence_duration_seconds"],
                key=f"run {row['run_id']} evidence_duration_seconds",
            )
            for row in rows
            if row["ledger_category"] == category
        }
        missing = sorted(set(expected) - set(recomputed))
        extra = sorted(set(recomputed) - set(expected))
        changed = sorted(
            run_id for run_id in set(expected) & set(recomputed)
            if expected[run_id] != recomputed[run_id]
        )
        if missing or extra or changed:
            raise RuntimeError(
                f"Campaign '{REFERENCE_CAMPAIGN_ID}' does not reproduce the {category} "
                f"rows of the reference ledger: {len(missing)} missing, "
                f"{len(extra)} extra, {len(changed)} with changed durations. "
                f"First missing {missing[:5]}, extra {extra[:5]}, changed {changed[:5]}."
            )


# ---------------------------------------------------------------------------
# Overhead rows (best-effort durations, labeled per row)
# ---------------------------------------------------------------------------
def _overhead_metric_evidence(client, run) -> tuple[float, str] | None:
    """Return the first duration metric that ends after ``start_time``."""
    start = run.info.start_time
    if start is None:
        raise ValueError(f"Run {run.info.run_id} is missing start_time.")
    for key in DURATION_METRIC_KEYS:
        timestamps = _history_timestamps(client, run.info.run_id, key)
        if timestamps and max(timestamps) > start:
            return (max(timestamps) - start) / 1000.0, f"metric:{key}"
    return None


def _overhead_row(
    client,
    run,
    *,
    experiment_name: str,
    experiment_prefix: str,
    dataset_key: str,
    overhead_category: str,
) -> dict:
    """Build an overhead row from metric evidence, then the raw MLflow span."""
    evidence = _overhead_metric_evidence(client, run)
    start, end = run.info.start_time, run.info.end_time
    if evidence is None and end is not None and end > start:
        evidence = (end - start) / 1000.0, "raw_span"
    seconds, source = (None, "no_evidence") if evidence is None else evidence
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
        "is_child_run": str(bool(_safe_tag(run, "mlflow.parentRunId"))).lower(),
        "start_time_iso": _ts_to_iso(start),
        "end_time_iso": _ts_to_iso(end),
        "evidence_duration_seconds": "" if seconds is None else seconds,
        "evidence_duration_hours": "" if seconds is None else f"{seconds / 3600.0:.6f}",
        "duration_source": source,
        "overhead_category": overhead_category,
    }


def _campaign_overhead_rows(
    client,
    campaign: _Campaign,
    dataset: _CampaignDataset,
    *,
    experiment_id: str,
) -> list[dict]:
    """Return the overhead rows of one experiment of an added campaign.

    The rows are the failed, killed, running, and soft-deleted parent runs.
    """
    labeled_runs = []
    for status in ATTEMPT_OVERHEAD_STATUSES:
        labeled_runs.extend(
            (run, status.lower())
            for run in _query_parent_runs(
                client,
                experiment_id,
                f"attribute.status = '{status}'",
                view_type=ViewType.ACTIVE_ONLY,
            )
        )
    labeled_runs.extend(
        (run, DELETED_OVERHEAD_LABEL)
        for run in _query_parent_runs(
            client,
            experiment_id,
            None,
            view_type=ViewType.DELETED_ONLY,
        )
    )
    return [
        _overhead_row(
            client,
            run,
            experiment_name=dataset.experiment_name,
            experiment_prefix=campaign.experiment_prefix,
            dataset_key=dataset.spec.key,
            overhead_category=f"{campaign.campaign_id}_{label}",
        )
        for run, label in labeled_runs
    ]


# ---------------------------------------------------------------------------
# Aggregates, outputs, and totals
# ---------------------------------------------------------------------------
def _seconds(rows: Sequence[dict], field: str, **match: str) -> list[float]:
    """Return ``field`` of every row whose columns equal ``match``."""
    return [
        row[field] for row in rows
        if all(row[column] == value for column, value in match.items())
    ]


def _print_banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _aggregate_row(campaign_id: str, group: str, key: str, seconds: list[float]) -> dict:
    return {
        "campaign_id": campaign_id,
        "group": group,
        "key": key,
        "runs": len(seconds),
        "hours": f"{_hours(seconds):.2f}",
    }


def _aggregate_rows(
    campaign_ids: Sequence[str],
    ledger_rows: Sequence[dict],
    episode_rows: Sequence[dict],
    analyses: Sequence[ComputePaperAnalysis],
) -> list[dict]:
    rows: list[dict] = []
    for campaign_id in campaign_ids:
        for group, category, fields in LEDGER_AGGREGATE_GROUPS:
            totals: dict[str, list[float]] = defaultdict(list)
            for row in ledger_rows:
                if row["campaign_id"] == campaign_id and row["ledger_category"] == category:
                    key = "/".join(row[field] for field in fields)
                    totals[key].append(row["evidence_duration_seconds"])
            rows.extend(
                _aggregate_row(campaign_id, group, key, seconds)
                for key, seconds in sorted(totals.items())
            )
        rows.extend(
            _aggregate_row(
                campaign_id,
                ANALYSIS_AGGREGATE_GROUP,
                analysis.analysis_id,
                _seconds(episode_rows, "span_seconds", analysis=analysis.analysis_id),
            )
            for analysis in analyses
            if analysis.campaign_id == campaign_id
        )
    return rows


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path} ({len(rows)} rows)")


def _write_outputs(
    output_dir: Path,
    *,
    ledger_rows: Sequence[dict],
    episode_rows: Sequence[dict],
    aggregate_rows: Sequence[dict],
    overhead_rows: Sequence[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(
        output_dir / "compute_ledger_official_runs.csv",
        LEDGER_FIELDS,
        sorted(ledger_rows, key=lambda row: (
            row["campaign_id"], row["dataset"], row["ledger_category"],
            row["model_architecture"], row["pipeline_method"],
            row["start_time_iso"], row["run_id"],
        )),
    )
    _write_csv(
        output_dir / "compute_evaluation_episodes.csv",
        EPISODE_FIELDS,
        sorted(episode_rows, key=lambda row: (
            row["campaign_id"], row["dataset"], row["run_id"], row["episode_index"],
        )),
    )
    _write_csv(output_dir / "compute_aggregates.csv", AGGREGATE_FIELDS, aggregate_rows)
    _write_csv(
        output_dir / "compute_overhead_audit.csv",
        OVERHEAD_FIELDS,
        sorted(overhead_rows, key=lambda row: (
            row["experiment_prefix"], row["experiment_name"],
            row["overhead_category"], row["start_time_iso"], row["run_id"],
        )),
    )


def _evidence_seconds(rows: Sequence[dict]) -> list[float]:
    return [
        row["evidence_duration_seconds"] for row in rows
        if row["evidence_duration_seconds"] != ""
    ]


def _print_totals(
    campaign_ids: Sequence[str],
    ledger_rows: Sequence[dict],
    episode_rows: Sequence[dict],
    analyses: Sequence[ComputePaperAnalysis],
    *,
    reference_overhead_rows: Sequence[dict],
    added_overhead_rows: Sequence[dict],
) -> None:
    _print_banner(
        "CAMPAIGN TOTALS (single-MIG-device GPU-hours, evaluations are lower bounds)"
    )
    print(f"  {'campaign':<22}  {'train':>7}  {'hours':>8}  {'eval':>7}  {'hours':>8}")
    for campaign_id in campaign_ids:
        print(f"  {campaign_id:<22}" + "".join(
            f"  {len(seconds):>7,}  {_hours(seconds):>8.2f}"
            for seconds in (
                _seconds(
                    ledger_rows,
                    "evidence_duration_seconds",
                    campaign_id=campaign_id,
                    ledger_category=category,
                )
                for category in (TRAINING_CATEGORY, EVALUATION_CATEGORY)
            )
        ))

    _print_banner("ANALYSIS EVALUATION EPISODES (lower bounds)")
    print(f"  {'analysis':<22}  {'campaign':<22}  {'episodes':>8}  {'hours':>8}")
    for analysis in analyses:
        seconds = _seconds(episode_rows, "span_seconds", analysis=analysis.analysis_id)
        print(
            f"  {analysis.analysis_id:<22}  {analysis.campaign_id:<22}  "
            f"{len(seconds):>8,}  {_hours(seconds):>8.2f}"
        )

    _print_banner("OVERHEAD AUDIT SUMMARY (reference record and added campaigns)")
    print(f"  {'category':<50}  {'parents':>8}  {'w/evidence':>10}  {'hours':>8}")
    overhead_rows = [*reference_overhead_rows, *added_overhead_rows]
    for category in sorted({row["overhead_category"] for row in overhead_rows}):
        rows = [row for row in overhead_rows if row["overhead_category"] == category]
        seconds = _evidence_seconds(rows)
        print(
            f"  {category:<50}  {len(rows):>8,}  {len(seconds):>10,}  "
            f"{_hours(seconds):>8.2f}"
        )

    analysis_rows = [row for row in episode_rows if row["analysis"]]
    totals = (
        (
            "Official spans",
            len(ledger_rows),
            [row["evidence_duration_seconds"] for row in ledger_rows],
        ),
        (
            "Analysis evaluation episodes",
            len(analysis_rows),
            [row["span_seconds"] for row in analysis_rows],
        ),
        (
            "Reference overhead records",
            len(reference_overhead_rows),
            _evidence_seconds(reference_overhead_rows),
        ),
        (
            "Added overhead records",
            len(added_overhead_rows),
            _evidence_seconds(added_overhead_rows),
        ),
    )
    _print_banner("TOTAL (single-MIG-device GPU-hours)")
    for label, count, seconds in totals:
        print(f"  {label:<42}  {count:>8,}  {_hours(seconds):>10.2f}")
    print(
        f"  {'Total records':<42}  "
        f"{sum(count for _, count, _ in totals):>8,}  "
        f"{_hours([value for _, _, seconds in totals for value in seconds]):>10.2f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _account_experiment(
    client,
    campaign: _Campaign,
    dataset: _CampaignDataset,
    *,
    record: _ReferenceRecord,
    analyses: Sequence[ComputePaperAnalysis],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return the ledger, episode, and overhead rows of one campaign experiment.

    The reference campaign adds no overhead rows, because its overhead is the
    carried-over overhead audit of the reference record.
    """
    is_reference = campaign.campaign_id == REFERENCE_CAMPAIGN_ID
    print(
        f"\n--- {dataset.experiment_name} (input_len={dataset.args.input_len}, "
        f"target_len={dataset.args.target_len}, "
        f"data_config_sig={dataset.data_config_signature[:12]}) ---"
    )
    experiment = client.get_experiment_by_name(dataset.experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"Expected MLflow experiment '{dataset.experiment_name}' was not found."
        )
    experiment_id = experiment.experiment_id
    training = _training_rows(
        client,
        campaign,
        dataset,
        experiment_id=experiment_id,
    )
    evaluations, episodes = _evaluation_rows(
        client,
        campaign,
        dataset,
        experiment_id=experiment_id,
        reference_evaluation_ms=record.evaluation_ms if is_reference else None,
        record_end_ms=record.evaluation_end_ms,
        analyses=analyses,
    )
    overhead = [] if is_reference else _campaign_overhead_rows(
        client,
        campaign,
        dataset,
        experiment_id=experiment_id,
    )
    analysis_episodes = [row["span_seconds"] for row in episodes if row["analysis"]]
    for label, count, seconds in (
        ("Training runs (current scope)", len(training),
         _seconds(training, "evidence_duration_seconds")),
        ("Official evaluations", len(evaluations),
         _seconds(evaluations, "evidence_duration_seconds")),
        ("Analysis evaluation episodes", len(analysis_episodes), analysis_episodes),
        ("Added overhead parent runs", len(overhead), _evidence_seconds(overhead)),
    ):
        print(f"  {label:<32}  {count:>6,}  {_hours(seconds):>8.2f} h")
    return training + evaluations, episodes, overhead


def _parse_cli_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only compute accounting over the experiment campaigns.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Compute accounting YAML with the campaigns and paper analyses.",
    )
    parser.add_argument(
        "--reference-ledger",
        type=Path,
        required=True,
        help="compute_ledger_official_runs.csv of version 0.1.0.",
    )
    parser.add_argument(
        "--reference-overhead",
        type=Path,
        required=True,
        help="compute_overhead_audit.csv of version 0.1.0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory, which must not exist yet.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    cli = _parse_cli_args(argv)
    if cli.output_dir.exists():
        raise FileExistsError(f"Output directory '{cli.output_dir}' already exists.")
    defaults = load_defaults()
    with open(cli.config) as handle:
        registry, analyses = parse_compute_accounting_config(
            yaml.safe_load(handle),
            defaults=defaults,
        )
    campaign_ids = [campaign.campaign_id for campaign in registry]
    if REFERENCE_CAMPAIGN_ID not in campaign_ids:
        raise ValueError(
            f"Compute accounting config must contain campaign '{REFERENCE_CAMPAIGN_ID}' "
            "for the reference reproduction check."
        )
    record = _load_reference_record(cli.reference_ledger, cli.reference_overhead)
    campaigns = [_resolve_campaign(campaign, defaults=defaults) for campaign in registry]
    tracking_uris = {
        get_tracking_uri(dataset.args.logdir)
        for campaign in campaigns
        for dataset in campaign.datasets
    }
    if len(tracking_uris) != 1:
        raise ValueError(
            "Compute campaigns must share one MLflow tracking server, got "
            f"{sorted(tracking_uris)}."
        )
    mlflow.set_tracking_uri(tracking_uris.pop())
    client = mlflow.MlflowClient()

    ledger_rows: list[dict] = []
    episode_rows: list[dict] = []
    overhead_rows: list[dict] = []
    for campaign in campaigns:
        _print_banner(
            f"Campaign {campaign.campaign_id}: {len(campaign.expected_cells)} winner "
            "cells per dataset"
        )
        campaign_analyses = [
            analysis for analysis in analyses if analysis.campaign_id == campaign.campaign_id
        ]
        campaign_rows: list[dict] = []
        for dataset in campaign.datasets:
            rows, episodes, overhead = _account_experiment(
                client,
                campaign,
                dataset,
                record=record,
                analyses=campaign_analyses,
            )
            campaign_rows.extend(rows)
            episode_rows.extend(episodes)
            overhead_rows.extend(overhead)
            time.sleep(QUERY_PAUSE_SECONDS)
        if campaign.campaign_id == REFERENCE_CAMPAIGN_ID:
            _check_reference_reproduction(
                campaign_rows,
                training_ms=record.training_ms,
                evaluation_ms=record.evaluation_ms,
            )
            print(f"\n  {REFERENCE_CAMPAIGN_ID} reproduces the reference record run by run.")
        ledger_rows.extend(campaign_rows)
    counted = {row["analysis"] for row in episode_rows}
    empty = sorted(
        analysis.analysis_id for analysis in analyses if analysis.analysis_id not in counted
    )
    if empty:
        raise RuntimeError(f"Paper analyses without an episode in their window: {empty}.")

    _print_banner("Writing output CSVs")
    _write_outputs(
        cli.output_dir,
        ledger_rows=ledger_rows,
        episode_rows=episode_rows,
        aggregate_rows=_aggregate_rows(campaign_ids, ledger_rows, episode_rows, analyses),
        overhead_rows=[*record.overhead_rows, *overhead_rows],
    )
    _print_totals(
        campaign_ids,
        ledger_rows,
        episode_rows,
        analyses,
        reference_overhead_rows=record.overhead_rows,
        added_overhead_rows=overhead_rows,
    )


if __name__ == "__main__":
    main()
