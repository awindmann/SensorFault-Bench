"""Checks that the perturbation regression fixture covers the shipped scenario configuration."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from config_loader import load_defaults


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "perturbation_regression.json"


def _load_fixture() -> dict:
    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _transfer_family_names() -> list[str]:
    recipe_path = REPO_ROOT / "configs" / "pipelines" / "fault_augmentation.yaml"
    with recipe_path.open(encoding="utf-8") as handle:
        recipe = yaml.safe_load(handle)
    return list(recipe["train_fault_profiles"]["holdout"]["scenarios"])


def test_fixture_covers_the_configured_scored_scenarios_and_transfer_families():
    fixture = _load_fixture()

    assert load_defaults()["PERTURBATION_SEVERITY_DISTRIBUTION"] == "uniform"
    assert [row["scenario"] for row in fixture["scored_scenarios"]] == list(
        load_defaults()["PERTURBATION_SCENARIOS"]
    )
    assert [row["scenario"] for row in fixture["transfer_families"]] == (
        _transfer_family_names()
    )
    assert len(fixture["scored_scenarios"]) == 8
    assert len(fixture["transfer_families"]) == 7
