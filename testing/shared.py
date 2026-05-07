from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from typing import Any

from utils.parsing import (
    parse_winner_candidate_tags,
)

_EVAL_DATALOADER_WORKER_WARNING_PATTERN = (
    r".*does not have many workers which may be a bottleneck.*"
)


def _configure_runtime_loggers_for_testing() -> None:
    logger_names = (
        "pytorch_lightning",
        "lightning.pytorch",
        "lightning_fabric",
        "mlflow.tracking._tracking_service.client",
    )
    for logger_name in logger_names:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


@contextmanager
def _suppress_lightning_worker_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_EVAL_DATALOADER_WORKER_WARNING_PATTERN,
            category=UserWarning,
        )
        yield


def _raise_non_finished_winner(run) -> None:
    """Raise when a non-FINISHED run has best_model=true."""
    tags = run.data.tags
    if tags is None:
        raise ValueError(
            f"Run {run.info.run_id} has best_model=true but is missing tags entirely."
        )
    arch = tags.get("model_architecture")
    method = tags.get("pipeline_method")
    pid = tags.get("pipeline_id")
    if not arch or not method or not pid:
        raise ValueError(
            f"Run {run.info.run_id} has best_model=true but is missing required "
            f"pipeline tags (model_architecture={arch!r}, pipeline_method={method!r}, "
            f"pipeline_id={pid!r})."
        )
    raise ValueError(
        f"Run {run.info.run_id} has best_model=true but "
        f"status={run.info.status}. "
        f"Variant: ({arch}, {method}, {pid}). "
        f"A non-FINISHED run cannot be a valid winner. "
        f"Re-run the {method} pipeline for {arch} to produce a "
        f"FINISHED replacement, then re-run testing to re-select winners."
    )


def _require_best_model_current_tags(
    tags: dict[str, Any],
    *,
    run_id: str,
) -> tuple[bool, bool]:
    """Require a run to be an explicit winner candidate in current winner pool."""
    winner_tags = parse_winner_candidate_tags(tags, run_id=run_id)
    if winner_tags is None:
        raise ValueError(
            f"Run {run_id} from best_model=true query is missing "
            "best_model/backbone_current tags."
        )
    best_model, backbone_current = winner_tags
    if not best_model or not backbone_current:
        raise ValueError(
            f"Run {run_id} from best_model=true query has invalid winner tags: "
            f"best_model={str(best_model).lower()}, "
            f"backbone_current={str(backbone_current).lower()}."
        )
    return winner_tags


def _require_single_pipeline_kind(
    kinds_by_pipeline: dict[str, Any],
    *,
    arch: str,
    pipeline_method: str,
) -> str:
    """Require a family to use exactly one pipeline_kind."""
    missing = sorted(
        str(pipeline_id)
        for pipeline_id, kind in kinds_by_pipeline.items()
        if kind is None or not str(kind).strip()
    )
    if missing:
        raise ValueError(
            f"Family ({arch}, {pipeline_method}) has pipeline_id(s) with missing pipeline_kind: "
            f"{', '.join(missing)}."
        )
    unique_kinds = sorted({str(kind).strip() for kind in kinds_by_pipeline.values()})
    if len(unique_kinds) != 1:
        kind_summary = ", ".join(
            f"{kind}=[{', '.join(sorted(str(pid) for pid, candidate_kind in kinds_by_pipeline.items() if str(candidate_kind).strip() == kind))}]"
            for kind in unique_kinds
        )
        raise ValueError(
            f"Cannot select winner for mixed-kind family ({arch}, {pipeline_method}). "
            f"Found multiple pipeline_kind values: {kind_summary}."
        )
    return unique_kinds[0]
