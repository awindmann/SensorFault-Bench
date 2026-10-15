"""Checks that the benchmark entry points start and list their options."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entrypoint", ("run_training.py", "run_testing.py", "run_analysis.py"))
def test_benchmark_entrypoint_help_commands_exit_successfully(entrypoint: str):
    result = subprocess.run(
        [sys.executable, entrypoint, "--help"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert f"usage: {entrypoint}" in result.stdout
    assert "--data-root" in result.stdout
    assert "--logdir" in result.stdout
    assert "--minio-endpoint" in result.stdout
    if entrypoint == "run_analysis.py":
        for training_flag in (
            "--rerun",
            "--max-epochs",
            "--save-checkpoint",
        ):
            assert training_flag in result.stdout
