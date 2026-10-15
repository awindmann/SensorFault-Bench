"""Shared fixtures of the test suite."""

from __future__ import annotations

import pytest

import run_analysis
import run_testing
import run_training
from config_loader import load_defaults
from pipelines.recipes import PIPELINE_RECIPE_PATHS_BY_METHOD


@pytest.fixture
def registered_methods_defaults(monkeypatch):
    """Serve the entry points the configured defaults with every registered method in scope.

    `BENCHMARK_METHODS` in `configs/defaults.yaml` selects which registered
    methods a run uses. Tests that name methods explicitly request this
    fixture, so they stay valid when a user narrows that selection.
    """
    defaults = {
        **load_defaults(),
        "BENCHMARK_METHODS": list(PIPELINE_RECIPE_PATHS_BY_METHOD),
    }
    for module in (run_training, run_testing, run_analysis):
        monkeypatch.setattr(module, "load_defaults", lambda: dict(defaults))
    return defaults
