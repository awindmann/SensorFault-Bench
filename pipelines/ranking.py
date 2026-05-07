from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from utils.parsing import parse_improvement_selection_mode


ALLOWED_PIPELINE_KINDS: tuple[str, ...] = (
    "train",
    "wrap",
    "finetune",
)

_TRAINING_PIPELINE_KINDS: tuple[str, ...] = (
    "train",
    "finetune",
)


def rank_key(
    run,
    *,
    metric_key: str = "best_val_loss",
    metric_value: Optional[float] = None,
    require_metric: bool = True,
    include_end_time: bool = True,
) -> tuple:
    value = metric_value
    used_missing_sentinel = False
    if value is None:
        value = run.data.metrics.get(metric_key)
    if value is None:
        if require_metric:
            raise ValueError(f"Run {run.info.run_id} is missing '{metric_key}'.")
        value = float("inf")
        used_missing_sentinel = True
    score = float(value)
    if not math.isfinite(score):
        if not (used_missing_sentinel and not require_metric and math.isinf(score)):
            raise ValueError(
                f"Run {run.info.run_id} has non-finite selection metric "
                f"'{metric_key}': {score}."
            )
    end_time = -(run.info.end_time or 0) if include_end_time else 0
    return (score, end_time, run.info.run_id)


def rank_key_for_run(
    run_or_record: Any,
    *,
    metric_key: str = "best_val_loss",
    metric_value: Optional[float] = None,
    require_metric: bool = True,
    include_end_time: bool = True,
) -> tuple:
    """Rank-key adapter that accepts either MLflow Run or RunRecord."""
    run = getattr(run_or_record, "run", run_or_record)
    if not hasattr(run, "info") or not hasattr(run, "data"):
        raise ValueError(
            "rank_key_for_run expected an MLflow Run-like object or RunRecord with .run."
        )
    return rank_key(
        run,
        metric_key=metric_key,
        metric_value=metric_value,
        require_metric=require_metric,
        include_end_time=include_end_time,
    )


def rank_key_for_row_values(
    *,
    selection_value: Any,
    end_time: Any,
    run_id: Any,
) -> tuple[float, float, str]:
    """Rank-key adapter for dataframe rows with precomputed selection values."""
    if selection_value is None:
        raise ValueError("Cannot compute row rank key: missing selection_value.")
    if run_id is None or not str(run_id).strip():
        raise ValueError("Cannot compute row rank key: missing run_id.")
    score = float(selection_value)
    if not math.isfinite(score):
        raise ValueError(
            "Cannot compute row rank key: selection_value must be finite."
        )
    end_time_value = end_time
    if end_time_value is None:
        normalized_end_time = 0.0
    else:
        normalized_end_time = float(end_time_value)
        if math.isnan(normalized_end_time):
            normalized_end_time = 0.0
    return (score, -normalized_end_time, str(run_id))


def rank_key_for_dataframe_row(
    row: Mapping[str, Any],
    *,
    selection_value_col: str = "selection_value",
    end_time_col: str = "end_time",
    run_id_col: str = "run_id",
) -> tuple[float, float, str]:
    missing_cols = [
        col
        for col in (selection_value_col, end_time_col, run_id_col)
        if col not in row
    ]
    if missing_cols:
        raise ValueError(
            "Cannot compute row rank key: missing required columns "
            f"{missing_cols}."
        )
    return rank_key_for_row_values(
        selection_value=row[selection_value_col],
        end_time=row[end_time_col],
        run_id=row[run_id_col],
    )


def validate_pipeline_tags_for_selection(
    tags: Mapping[str, Any],
    *,
    run_id: str,
) -> dict[str, str]:
    """Strictly validate pipeline tags used for selection/ranking paths."""
    pipeline_id_raw = tags.get("pipeline_id")
    pipeline_method_raw = tags.get("pipeline_method")
    pipeline_kind_raw = tags.get("pipeline_kind")
    robustness_method_raw = tags.get("robustness_method")

    missing = [
        key
        for key, value in (
            ("pipeline_id", pipeline_id_raw),
            ("pipeline_method", pipeline_method_raw),
            ("pipeline_kind", pipeline_kind_raw),
            ("robustness_method", robustness_method_raw),
        )
        if value is None or not str(value).strip()
    ]
    if missing:
        raise ValueError(
            f"Run {run_id} is missing required pipeline tag(s): {', '.join(missing)}."
        )

    pipeline_id = str(pipeline_id_raw).strip()
    pipeline_method = str(pipeline_method_raw).strip()
    pipeline_kind = str(pipeline_kind_raw).strip()
    robustness_method = str(robustness_method_raw).strip()

    if pipeline_kind not in ALLOWED_PIPELINE_KINDS:
        raise ValueError(
            f"Run {run_id} has unknown pipeline_kind '{pipeline_kind}'."
        )

    if pipeline_id == "baseline" and pipeline_method != "baseline":
        raise ValueError(
            f"Run {run_id} has pipeline_id='baseline' but pipeline_method='{pipeline_method}'."
        )
    if pipeline_method == "baseline" and pipeline_id != "baseline":
        raise ValueError(
            f"Run {run_id} has pipeline_method='baseline' but pipeline_id='{pipeline_id}'."
        )
    if pipeline_method == "baseline":
        if pipeline_kind != "train":
            raise ValueError(
                f"Run {run_id} has baseline pipeline_method but pipeline_kind='{pipeline_kind}'. "
                "Baseline runs must have pipeline_kind='train'."
            )
        if robustness_method != "baseline":
            raise ValueError(
                f"Run {run_id} has baseline pipeline_method but robustness_method='{robustness_method}'."
            )
    elif robustness_method != pipeline_method:
        raise ValueError(
            f"Run {run_id} has non-baseline tag mismatch: pipeline_method='{pipeline_method}' "
            f"but robustness_method='{robustness_method}'."
        )

    return {
        "pipeline_id": pipeline_id,
        "pipeline_method": pipeline_method,
        "pipeline_kind": pipeline_kind,
        "robustness_method": robustness_method,
    }


def _clean_selection_metric_key_for_kind(
    *,
    pipeline_kind: str,
    test_metric: str,
    run_id: str,
) -> str:
    resolved_kind = str(pipeline_kind).strip()
    if resolved_kind == "wrap":
        metric = str(test_metric).strip()
        if not metric:
            raise ValueError(
                f"Wrap run {run_id} is missing test_metric required for selection metric resolution."
            )
        return f"{metric}_val"
    if resolved_kind in _TRAINING_PIPELINE_KINDS:
        return "best_val_loss"
    raise ValueError(
        f"Run {run_id} has unknown pipeline_kind '{resolved_kind}'."
    )


def perturbed_selection_metric_keys(
    *,
    test_metric: str,
    run_id: str,
) -> tuple[str, str]:
    metric = str(test_metric).strip()
    if not metric:
        raise ValueError(
            f"Run {run_id} is missing test_metric required for perturbed selection metric resolution."
        )
    return (f"{metric}_pert_ws_val", f"{metric}_pert_mean_val")


def require_perturbed_selection_metric_pair(
    *,
    metrics: Mapping[str, Any],
    test_metric: str,
    run_id: str,
) -> dict[str, float]:
    pert_ws_key, pert_mean_key = perturbed_selection_metric_keys(
        test_metric=test_metric,
        run_id=run_id,
    )
    values: dict[str, float] = {}
    missing: list[str] = []
    for metric_key in (pert_ws_key, pert_mean_key):
        raw_value = metrics.get(metric_key)
        if raw_value is None:
            missing.append(metric_key)
            continue
        score_value = float(raw_value)
        if not math.isfinite(score_value):
            raise ValueError(
                f"Run {run_id} has non-finite selection metric '{metric_key}': {score_value}."
            )
        values[metric_key] = score_value
    if missing:
        raise ValueError(
            f"Run {run_id} is missing required perturbed selection metric(s) {missing}."
        )
    return values


def winner_selection_mode_for_method(
    *,
    robustness_method: str,
    improvement_selection_mode: str,
    run_id: str,
) -> str:
    resolved_method = str(robustness_method).strip()
    if not resolved_method:
        raise ValueError(
            f"Run {run_id} is missing robustness_method required for selection mode resolution."
        )
    selection_mode = parse_improvement_selection_mode(
        improvement_selection_mode,
        key="improvement_selection_mode",
    )
    if resolved_method == "baseline":
        return "clean"
    return selection_mode


def selection_metric_key_for_kind(
    *,
    pipeline_kind: str,
    robustness_method: str,
    test_metric: str,
    improvement_selection_mode: str,
    run_id: str,
) -> str:
    resolved_kind = str(pipeline_kind).strip()
    if resolved_kind not in ALLOWED_PIPELINE_KINDS:
        raise ValueError(
            f"Run {run_id} has unknown pipeline_kind '{resolved_kind}'."
        )
    resolved_method = str(robustness_method).strip()
    if not resolved_method:
        raise ValueError(
            f"Run {run_id} is missing robustness_method required for selection metric resolution."
        )
    selection_mode = winner_selection_mode_for_method(
        robustness_method=resolved_method,
        improvement_selection_mode=improvement_selection_mode,
        run_id=run_id,
    )
    if selection_mode == "clean":
        return _clean_selection_metric_key_for_kind(
            pipeline_kind=resolved_kind,
            test_metric=test_metric,
            run_id=run_id,
        )
    metric = str(test_metric).strip()
    if not metric:
        raise ValueError(
            f"Run {run_id} is missing test_metric required for perturbed selection metric resolution."
        )
    if selection_mode == "perturbed_worst":
        return f"{metric}_pert_ws_val"
    if selection_mode == "perturbed_mean":
        return f"{metric}_pert_mean_val"
    raise AssertionError(
        f"Unhandled improvement_selection_mode '{selection_mode}'."
    )


def resolve_selection_score(
    *,
    metrics: Mapping[str, Any],
    pipeline_kind: str,
    robustness_method: str,
    test_metric: str,
    improvement_selection_mode: str,
    run_id: str,
) -> float:
    metric_key = selection_metric_key_for_kind(
        pipeline_kind=pipeline_kind,
        robustness_method=robustness_method,
        test_metric=test_metric,
        improvement_selection_mode=improvement_selection_mode,
        run_id=run_id,
    )
    selection_mode = winner_selection_mode_for_method(
        robustness_method=robustness_method,
        improvement_selection_mode=improvement_selection_mode,
        run_id=run_id,
    )
    if selection_mode != "clean":
        perturbed_metrics = require_perturbed_selection_metric_pair(
            metrics=metrics,
            test_metric=test_metric,
            run_id=run_id,
        )
        return perturbed_metrics[metric_key]
    score = metrics.get(metric_key)
    if score is None:
        kind = str(pipeline_kind).strip()
        if kind == "wrap":
            raise ValueError(
                f"Wrap run {run_id} is missing required selection metric '{metric_key}'."
            )
        raise ValueError(
            f"Training-based run {run_id} is missing required selection metric "
            f"'{metric_key}'."
        )
    score_value = float(score)
    if not math.isfinite(score_value):
        raise ValueError(
            f"Run {run_id} has non-finite selection metric '{metric_key}': {score_value}."
        )
    return score_value


def sort_runs_by_metric(
    runs: list,
    *,
    metric_key: str = "best_val_loss",
    include_end_time: bool = True,
    missing_error_prefix: str = "Runs",
) -> list:
    missing = [run.info.run_id for run in runs if run.data.metrics.get(metric_key) is None]
    if missing:
        raise ValueError(
            f"{missing_error_prefix} missing {metric_key}: " + ", ".join(missing)
        )
    return sorted(
        runs,
        key=lambda run: rank_key(
            run,
            metric_key=metric_key,
            include_end_time=include_end_time,
        ),
    )
