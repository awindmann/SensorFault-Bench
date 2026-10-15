"""Tests for the fault-direction and timing diagnostic scenarios.

`negative_drift`, `negative_spike`, and `local_timing_warp` are extended
scenarios for the sensitivity analyses. They are never part of the default
scored scenario set. The parity fixture freezes their outputs, and those of
the Beta(1, 2) severity distribution, on fixed inputs and generators.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch

import data.perturbations as perturbations
from data.perturbations import (
    DEFAULT_PERTURBATIONS,
    EXTENDED_PERTURBATIONS,
    PERTURBATION_REGISTRY,
    Drift,
    LocalTimingWarp,
    NegativeDrift,
    NegativeSpike,
    Spike,
    TimeCompress,
    boundary_preserving_local_timing_warp_coordinates,
    build_perturbation_scenario_params_signature,
)
from data.samplers import beta_1_2_severity


FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "diagnostic_perturbation_parity.json"
)
DIAGNOSTIC_SCENARIOS = ("negative_drift", "negative_spike", "local_timing_warp")
N_SAMPLES = 8
INPUT_LEN = 48
TARGET_LEN = 12
CONTINUOUS_CHANNELS = [0, 1, 2, 3, 4]
DISCRETE_CHANNELS = [5]


def _severity_channel_count_kwargs(severity: float) -> dict[str, float | str]:
    return {
        "channel_count_mode": "severity",
        "channel_count_value": severity,
    }


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


def _parity_payload(
    operators: dict[str, type],
    beta_distribution: Callable[[torch.Generator], float],
) -> dict[str, Any]:
    """Apply each diagnostic operator to the fixed inputs with fixed generators."""
    x, y = _fixed_inputs()
    scenarios: list[dict[str, Any]] = []
    for scenario_idx, name in enumerate(DIAGNOSTIC_SCENARIOS):
        operator = operators[name](channel_frac=0.5)
        x_rows: list[torch.Tensor] = []
        severities: list[float] = []
        affected: list[list[int]] = []
        for sample in range(N_SAMPLES):
            rng = torch.Generator().manual_seed(7000 + 100 * scenario_idx + sample)
            severity = float(torch.rand((), generator=rng).item())
            x_pert, y_pert, channels = operator(
                x[sample].clone(),
                y[sample].clone(),
                severity,
                rng,
                CONTINUOUS_CHANNELS,
                DISCRETE_CHANNELS,
                **_severity_channel_count_kwargs(severity),
            )
            if not torch.equal(y_pert, y[sample]):
                raise AssertionError(f"{name} changed the target.")
            x_rows.append(x_pert.to(torch.float32))
            severities.append(severity)
            affected.append([int(channel) for channel in channels])
        scenarios.append(
            {
                "scenario": name,
                "severities": severities,
                "affected_channels": affected,
                "x_sha256": hashlib.sha256(
                    torch.stack(x_rows).contiguous().numpy().tobytes()
                ).hexdigest(),
            }
        )
    beta_draws = [
        float(beta_distribution(torch.Generator().manual_seed(500 + draw)))
        for draw in range(64)
    ]
    return {"scenarios": scenarios, "beta_1_2_draws": beta_draws}


def test_diagnostics_match_the_reference_parity_fixture():
    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        expected = json.load(handle)

    observed = _parity_payload(
        {
            "negative_drift": NegativeDrift,
            "negative_spike": NegativeSpike,
            "local_timing_warp": LocalTimingWarp,
        },
        beta_1_2_severity,
    )

    assert observed == expected


def test_diagnostic_scenarios_are_registered_as_extended_only():
    default_names = {perturbation.name for perturbation in DEFAULT_PERTURBATIONS}
    extended_names = {perturbation.name for perturbation in EXTENDED_PERTURBATIONS}

    for name in DIAGNOSTIC_SCENARIOS:
        assert name in PERTURBATION_REGISTRY
        assert name in extended_names
        assert name not in default_names


def test_local_timing_warp_signature_records_operator_semantics():
    payload = json.loads(
        build_perturbation_scenario_params_signature(("local_timing_warp",))
    )

    assert payload == [
        {
            "channel_scope": "continuous",
            "class": "LocalTimingWarp",
            "defaults": {"max_rate_deviation": 0.9},
            "name": "local_timing_warp",
        }
    ]


@pytest.mark.parametrize("window_length", [47, 48])
@pytest.mark.parametrize("severity", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_local_timing_warp_coordinates_preserve_boundaries(window_length, severity):
    start = 7
    coordinates = boundary_preserving_local_timing_warp_coordinates(
        start,
        window_length,
        severity,
        max_rate_deviation=0.9,
    )

    assert coordinates.dtype == np.float64
    assert coordinates[0] == pytest.approx(start, abs=1e-12)
    assert coordinates[-1] == pytest.approx(start + window_length - 1, abs=1e-12)
    assert np.isfinite(coordinates).all()
    assert np.all(np.diff(coordinates) > 0.0)
    assert coordinates.min() >= start - 1e-12
    assert coordinates.max() <= start + window_length - 1 + 1e-12


def test_local_timing_warp_coordinates_have_expected_derivative_bounds():
    normalized = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
    derivative = 1.0 + 0.9 * np.cos(np.pi * normalized)

    assert derivative.min() == pytest.approx(0.1)
    assert derivative.max() == pytest.approx(1.9)


def test_local_timing_warp_is_identity_at_zero_severity():
    x = torch.randn(96, 3, generator=torch.Generator().manual_seed(17))
    y = torch.randn(4, 1, generator=torch.Generator().manual_seed(18))
    rng = torch.Generator().manual_seed(19)
    initial_state = rng.get_state().clone()

    x_pert, y_pert, channels = LocalTimingWarp(channel_frac=1.0)(
        x,
        y,
        0.0,
        rng,
        [0, 1, 2],
        [],
        **_severity_channel_count_kwargs(0.0),
    )

    assert torch.equal(x_pert, x)
    assert torch.equal(y_pert, y)
    assert channels == []
    assert torch.equal(rng.get_state(), initial_state)


def test_local_timing_warp_changes_only_selected_local_window(monkeypatch):
    start = 9
    monkeypatch.setattr(perturbations, "randint_py", lambda low, high, rng: start)
    x = torch.arange(96 * 4, dtype=torch.float32).reshape(96, 4).square()
    y = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    x_pert, y_pert, channels = LocalTimingWarp(channel_frac=0.5)(
        x,
        y,
        1.0,
        torch.Generator().manual_seed(21),
        [0, 1, 2, 3],
        [],
        **_severity_channel_count_kwargs(1.0),
    )

    window_length = 48
    untouched_channels = [channel for channel in range(4) if channel not in channels]
    assert len(channels) == 2
    assert torch.equal(y_pert, y)
    assert torch.equal(x_pert[:start], x[:start])
    assert torch.equal(x_pert[start + window_length:], x[start + window_length:])
    assert torch.equal(x_pert[:, untouched_channels], x[:, untouched_channels])
    assert not torch.equal(
        x_pert[start:start + window_length, channels],
        x[start:start + window_length, channels],
    )


def test_local_timing_warp_replays_time_compress_channel_and_start_draws(monkeypatch):
    starts: list[int] = []
    original_randint = perturbations.randint_py

    def _record_randint(low, high, rng):
        value = original_randint(low, high, rng)
        starts.append(value)
        return value

    monkeypatch.setattr(perturbations, "randint_py", _record_randint)
    x = torch.arange(96 * 4, dtype=torch.float32).reshape(96, 4)
    y = torch.zeros(2, 1)
    canonical_rng = torch.Generator().manual_seed(23)
    local_rng = torch.Generator().manual_seed(23)
    kwargs = _severity_channel_count_kwargs(0.75)

    _, _, canonical_channels = TimeCompress(channel_frac=0.5)(
        x, y, 0.75, canonical_rng, [0, 1, 2, 3], [], **kwargs
    )
    _, _, local_channels = LocalTimingWarp(channel_frac=0.5)(
        x, y, 0.75, local_rng, [0, 1, 2, 3], [], **kwargs
    )

    assert canonical_channels == local_channels
    assert starts[0] == starts[1]
    assert torch.equal(canonical_rng.get_state(), local_rng.get_state())


@pytest.mark.parametrize("input_length", [1, 2])
def test_local_timing_warp_rejects_inputs_without_two_window_endpoints(input_length):
    with pytest.raises(ValueError, match="at least two endpoints"):
        LocalTimingWarp(channel_frac=1.0)(
            torch.zeros(input_length, 1),
            torch.zeros(1, 1),
            1.0,
            torch.Generator().manual_seed(29),
            [0],
            [],
            **_severity_channel_count_kwargs(1.0),
        )


@pytest.mark.parametrize(
    ("positive_cls", "negative_cls"),
    [(Drift, NegativeDrift), (Spike, NegativeSpike)],
)
def test_negative_fault_exactly_mirrors_positive_fault_without_extra_rng_draw(
    positive_cls,
    negative_cls,
):
    x = torch.zeros(12, 4)
    y = torch.arange(3, dtype=torch.float32).reshape(3, 1)
    positive_rng = torch.Generator().manual_seed(103)
    negative_rng = torch.Generator().manual_seed(103)

    positive_x, positive_y, positive_channels = positive_cls(channel_frac=1.0)(
        x, y, 1.0, positive_rng, [0, 1, 2, 3], [], **_severity_channel_count_kwargs(1.0)
    )
    negative_x, negative_y, negative_channels = negative_cls(channel_frac=1.0)(
        x, y, 1.0, negative_rng, [0, 1, 2, 3], [], **_severity_channel_count_kwargs(1.0)
    )

    assert negative_channels == positive_channels
    assert torch.equal(positive_y, y)
    assert torch.equal(negative_y, y)
    assert torch.equal(negative_x, x - (positive_x - x))
    assert torch.equal(negative_rng.get_state(), positive_rng.get_state())


@pytest.mark.parametrize("negative_cls", [NegativeDrift, NegativeSpike])
def test_negative_fault_zero_severity_is_clean(negative_cls):
    x = torch.randn(8, 3, generator=torch.Generator().manual_seed(7))
    y = torch.randn(2, 1, generator=torch.Generator().manual_seed(8))
    rng = torch.Generator().manual_seed(9)
    initial_state = rng.get_state().clone()

    x_negative, y_negative, channels = negative_cls(channel_frac=1.0)(
        x, y, 0.0, rng, [0, 1, 2], [], **_severity_channel_count_kwargs(0.0)
    )

    assert torch.equal(x_negative, x)
    assert torch.equal(y_negative, y)
    assert channels == []
    assert torch.equal(rng.get_state(), initial_state)


def test_negative_faults_reuse_positive_constructor_defaults():
    assert inspect.signature(NegativeDrift.__init__) == inspect.signature(Drift.__init__)
    assert inspect.signature(NegativeSpike.__init__) == inspect.signature(Spike.__init__)


@pytest.mark.parametrize(
    ("positive_cls", "negative_cls"),
    [(Drift, NegativeDrift), (Spike, NegativeSpike)],
)
def test_negative_faults_reject_positive_operator_target_mutation(
    monkeypatch,
    positive_cls,
    negative_cls,
):
    def _mutate_target(
        self, x, y, severity, rng, cont_channels, disc_channels, *,
        channel_count_mode="severity", channel_count_value=None,
    ):
        del self, severity, rng, disc_channels, channel_count_mode, channel_count_value
        y.add_(1.0)
        return x, y, list(cont_channels)

    monkeypatch.setattr(positive_cls, "__call__", _mutate_target)

    with pytest.raises(ValueError, match="leave targets unchanged"):
        negative_cls(channel_frac=1.0)(
            torch.zeros(4, 1),
            torch.zeros(2, 1),
            1.0,
            torch.Generator().manual_seed(4),
            [0],
            [],
            **_severity_channel_count_kwargs(1.0),
        )
