from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.compute_accounting_ledger import (
    _expected_pipeline_kinds_by_cell,
    _resolve_official_setup_compute_scope_specs,
)


def _spec(method: str, kind: str) -> SimpleNamespace:
    return SimpleNamespace(pipeline_method=method, pipeline_kind=kind)


def test_official_compute_scope_keeps_train_and_wrap_methods():
    specs = [
        _spec("baseline", "train"),
        _spec("ensemble", "wrap"),
        _spec("randomized_smoothing", "wrap"),
    ]

    assert _resolve_official_setup_compute_scope_specs(specs) == specs


def test_official_compute_scope_rejects_unshipped_pipeline_kind():
    with pytest.raises(ValueError, match="pipeline_kind='finetune'"):
        _resolve_official_setup_compute_scope_specs([
            _spec("unsupported", "finetune"),
        ])


def test_expected_pipeline_kinds_uses_method_architecture_policy():
    expected = _expected_pipeline_kinds_by_cell(
        scope_specs=[
            _spec("baseline", "train"),
            _spec("ensemble", "wrap"),
            _spec("revin", "train"),
        ],
        architectures=["PatchTST", "GRU", "Chronos2"],
    )

    assert expected[("PatchTST", "baseline")] == "train"
    assert expected[("GRU", "ensemble")] == "wrap"
    assert expected[("GRU", "revin")] == "train"
    assert ("PatchTST", "revin") not in expected
    assert ("Chronos2", "ensemble") not in expected
