"""Helpers for downloading checkpoint artifacts from MLflow."""

from __future__ import annotations

import glob
import logging
import os
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

logger = logging.getLogger(__name__)


def _find_checkpoints(local_dir: str) -> list[str]:
    """Return sorted list of .ckpt file paths under *local_dir*."""
    return sorted(glob.glob(os.path.join(local_dir, "**", "*.ckpt"), recursive=True))


def _is_missing_artifact_error(exc: Exception) -> bool:
    """Return whether an MLflow artifact lookup failed because the resource is absent."""
    if isinstance(exc, (FileNotFoundError, KeyError)):
        return True
    if not isinstance(exc, MlflowException):
        return False
    error_code = getattr(exc, "error_code", None)
    return error_code in {
        RESOURCE_DOES_NOT_EXIST,
        "RESOURCE_DOES_NOT_EXIST",
    }


def download_best_checkpoint(client: Any, run_id: str, dst_path: str | None = None) -> str:
    """Download and return the local path to the best checkpoint for a run.

    PL 2.5+ (``log_model=True``) saves checkpoints at the artifact root
    (``epoch=XX-ep_val_loss=XX/``).  PL 2.4 and explicit logging use
    ``model/checkpoints/``.  This function tries targeted ``epoch=*``
    directories first, then falls back to ``model/checkpoints/``.

    Assumes ``save_top_k=1`` — only one checkpoint per directory.

    Raises ``ValueError`` if no ``.ckpt`` file is found under either layout.
    """
    # --- Strategy 1: PL 2.5+ root-level epoch dirs ---
    # List top-level artifacts and download only epoch=* dirs to avoid
    # pulling unrelated artifacts (hparams.json, figures, etc.).
    try:
        top_level = client.list_artifacts(run_id)
    except Exception as exc:
        if not _is_missing_artifact_error(exc):
            raise
        logger.debug("Run %s: root artifact listing unavailable.", run_id)
        top_level = []
    epoch_dirs = sorted(
        [a for a in top_level if a.is_dir and a.path.startswith("epoch=")],
        key=lambda a: a.path,
        reverse=True,
    )
    for art in epoch_dirs:
        try:
            local_dir = client.download_artifacts(run_id, art.path, dst_path=dst_path)
        except Exception as exc:
            if not _is_missing_artifact_error(exc):
                raise
            logger.debug("Run %s: failed to download artifact '%s'.", run_id, art.path)
            continue
        ckpt_files = _find_checkpoints(local_dir)
        if ckpt_files:
            return ckpt_files[0]

    # --- Strategy 2: explicit model/checkpoints/ logging ---
    try:
        local_dir = client.download_artifacts(run_id, "model/checkpoints", dst_path=dst_path)
    except Exception as exc:
        if not _is_missing_artifact_error(exc):
            raise
        logger.debug("Run %s: 'model/checkpoints' not found.", run_id)
        local_dir = None
    if local_dir is not None:
        ckpt_files = _find_checkpoints(local_dir)
        if ckpt_files:
            logger.info(
                "Run %s: checkpoint found at 'model/checkpoints/'.",
                run_id,
            )
            return ckpt_files[0]

    raise ValueError(
        f"No checkpoint found for run {run_id}. "
        f"Searched root-level epoch=* dirs and model/checkpoints/."
    )


def load_lightning_module_checkpoint(
    model_class: Any,
    checkpoint_path: str,
    **model_kwargs: Any,
) -> Any:
    """Load a Lightning checkpoint onto CPU before runtime device placement.

    Checkpoints may persist device-local storage metadata from the training
    runtime. Force CPU deserialization here so evaluation and reporting can
    reconstruct models even when the original CUDA device is unavailable or
    unstable, then move the model to the requested runtime device later.
    """
    return model_class.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        **model_kwargs,
    )
