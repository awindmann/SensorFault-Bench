"""Environment and infrastructure helpers shared across entrypoints."""

from __future__ import annotations

import os
import subprocess
import warnings

from utils.parsing import require_namespace_value

_MLFLOW_FILE_STORE_DEPRECATION_WARNING_RE = (
    r"The filesystem tracking backend \(e\.g\., './mlruns'\) is deprecated as of "
    r"February 2026\."
)


def current_git_commit() -> str:
    """Return the current HEAD commit hash, or ``unknown`` outside a git checkout."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return commit or "unknown"


def suppress_mlflow_file_store_deprecation_warning() -> None:
    """Hide MLflow's warning for intentional local file-backed tracking."""
    warnings.filterwarnings(
        "ignore",
        message=_MLFLOW_FILE_STORE_DEPRECATION_WARNING_RE,
        category=FutureWarning,
    )


def set_mlflow_storage_env(args) -> None:
    """Configure S3/MinIO environment variables from ``args.minio_endpoint``."""
    suppress_mlflow_file_store_deprecation_warning()
    minio_endpoint = require_namespace_value(args, key="minio_endpoint")
    if not minio_endpoint:
        return
    os.environ["AWS_ENDPOINT_URL"] = minio_endpoint
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = minio_endpoint
