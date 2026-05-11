import os
import subprocess
import warnings
from types import SimpleNamespace

import mlflow

from utils.env import (
    current_git_commit,
    set_mlflow_storage_env,
    suppress_mlflow_file_store_deprecation_warning,
)


def test_set_mlflow_storage_env_explicit_endpoint_overrides_existing_env(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://previous.example.invalid")
    monkeypatch.setenv("MLFLOW_S3_ENDPOINT_URL", "http://previous.example.invalid")

    set_mlflow_storage_env(
        SimpleNamespace(minio_endpoint="http://object-store.example.invalid")
    )

    assert os.environ["AWS_ENDPOINT_URL"] == "http://object-store.example.invalid"
    assert os.environ["MLFLOW_S3_ENDPOINT_URL"] == "http://object-store.example.invalid"


def test_current_git_commit_is_explicit_unknown_outside_git_checkout(monkeypatch):
    def _raise_no_git(*_args, **_kwargs):
        raise subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(subprocess, "check_output", _raise_no_git)

    assert current_git_commit() == "unknown"


def test_suppress_mlflow_file_store_deprecation_warning_hides_mlflow_warning(tmp_path):
    previous_tracking_uri = mlflow.get_tracking_uri()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            suppress_mlflow_file_store_deprecation_warning()
            mlflow.set_tracking_uri(f"file:{tmp_path / 'mlruns'}")
            mlflow.MlflowClient()
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)

    messages = [str(warning.message) for warning in caught]
    assert not any(
        message.startswith("The filesystem tracking backend") for message in messages
    )


def test_suppress_mlflow_file_store_deprecation_warning_keeps_other_futurewarnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        suppress_mlflow_file_store_deprecation_warning()
        warnings.warn("unrelated future warning", FutureWarning)

    assert [str(warning.message) for warning in caught] == ["unrelated future warning"]
