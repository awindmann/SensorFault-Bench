"""Bit-level regression fixtures for the scored and transfer perturbation paths.

The fixture freezes the outputs of the canonical scenario-batch builder for the
eight scored scenarios and the seven transfer fault families on fixed inputs.
Any change to operator arithmetic, severity draws, channel selection, or the
per-sample random streams changes a hash and fails this test. The check uses the
scenario lists frozen in the fixture, so editing the configured scenarios does
not affect it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from data.data_module import instantiate_perturbations, _resolve_perturbation_classes
from data.perturbations import PERTURBATION_REGISTRY
from data.samplers import resolve_severity_distribution
from testing.evaluation import _build_degradation_scenario_batch


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "perturbation_regression.json"
EVAL_DATA_SEED = 20260925
N_SAMPLES = 8
INPUT_LEN = 48
TARGET_LEN = 12
CONTINUOUS_CHANNELS = [0, 1, 2, 3, 4]
DISCRETE_CHANNELS = [5]
CHANNEL_FRACTION_MAX = 0.5


def _fixed_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    samples = torch.arange(N_SAMPLES, dtype=torch.float64).view(-1, 1, 1)
    steps = torch.arange(INPUT_LEN, dtype=torch.float64).view(1, -1, 1)
    channels = torch.arange(len(CONTINUOUS_CHANNELS), dtype=torch.float64).view(1, 1, -1)
    continuous = torch.sin(0.37 * steps + 0.91 * channels + 0.53 * samples) + 0.05 * channels * steps / INPUT_LEN
    discrete = (torch.sin(0.2 * steps + samples) > 0).to(torch.float64)
    x = torch.cat([continuous, discrete], dim=-1).to(torch.float32)

    target_steps = torch.arange(TARGET_LEN, dtype=torch.float64).view(1, -1, 1)
    all_channels = torch.arange(x.size(-1), dtype=torch.float64).view(1, 1, -1)
    y = torch.cos(0.29 * target_steps + 0.41 * all_channels + 0.17 * samples).to(torch.float32)
    return x, y


def _scenario_payload(scenario_names: list[str]) -> list[dict[str, Any]]:
    x, y = _fixed_inputs()
    perturbations = instantiate_perturbations(
        _resolve_perturbation_classes(
            scenario_names,
            registry=PERTURBATION_REGISTRY,
            key="fixture_scenarios",
        ),
        registry=PERTURBATION_REGISTRY,
        channel_fraction_max=CHANNEL_FRACTION_MAX,
    )
    payload: list[dict[str, Any]] = []
    for scenario_idx, perturbation in enumerate(perturbations):
        x_pert, _y_pert, severities, affected = _build_degradation_scenario_batch(
            perturbation=perturbation,
            x_cpu=x,
            y_cpu=y,
            batch_sample_ids=list(range(N_SAMPLES)),
            eval_data_seed=EVAL_DATA_SEED,
            scenario_idx=scenario_idx,
            cont_idx=CONTINUOUS_CHANNELS,
            disc_idx=DISCRETE_CHANNELS,
            severity_distribution=resolve_severity_distribution("uniform"),
            context_name="perturbation regression fixture",
            capture_affected_channels=True,
        )
        payload.append(
            {
                "scenario": perturbation.name,
                "scenario_idx": scenario_idx,
                "severities": [float(value) for value in severities],
                "affected_channels": [
                    [int(channel) for channel in channels] for channels in affected
                ],
                "x_sha256": hashlib.sha256(
                    x_pert.contiguous().numpy().tobytes()
                ).hexdigest(),
            }
        )
    return payload


def compute_fixture_payload(
    scored_scenarios: list[str],
    transfer_families: list[str],
) -> dict[str, Any]:
    """Recompute the fixture for the given scored scenarios and transfer families."""
    return {
        "eval_data_seed": EVAL_DATA_SEED,
        "channel_fraction_max": CHANNEL_FRACTION_MAX,
        "scored_scenarios": _scenario_payload(scored_scenarios),
        "transfer_families": _scenario_payload(transfer_families),
    }


def _load_fixture() -> dict[str, Any]:
    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_scored_and_transfer_perturbations_reproduce_the_fixture_bit_for_bit():
    fixture = _load_fixture()
    current = compute_fixture_payload(
        [row["scenario"] for row in fixture["scored_scenarios"]],
        [row["scenario"] for row in fixture["transfer_families"]],
    )

    for group in ("scored_scenarios", "transfer_families"):
        for expected, observed in zip(fixture[group], current[group], strict=True):
            assert observed["scenario"] == expected["scenario"]
            assert observed["affected_channels"] == expected["affected_channels"], (
                expected["scenario"]
            )
            assert all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=0.0)
                for left, right in zip(
                    observed["severities"], expected["severities"], strict=True
                )
            ), expected["scenario"]
            assert observed["x_sha256"] == expected["x_sha256"], expected["scenario"]
