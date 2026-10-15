"""Tests for analysis runs whose scenarios lie outside the core-figure vocabulary."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from testing.meta_analysis import (
    METHOD_SCENARIO_FAMILY_DELTA_COLUMNS,
    METHOD_SCENARIO_FAMILY_SUMMARY_COLUMNS,
    _build_method_scenario_family_delta_tables,
    _core_figure_registry,
    _unsupported_figure_scenarios,
)
from utils.parsing import scenarios_outside_figure_registry


def _analysis_args(perturbation_scenarios, *, fixed_channel_fraction=None) -> SimpleNamespace:
    return SimpleNamespace(
        perturbation_channel_fraction_max=0.5,
        perturbation_scenarios=list(perturbation_scenarios),
        fixed_channel_fraction=fixed_channel_fraction,
    )


def test_scenarios_outside_figure_registry_accepts_registry_subsets():
    assert scenarios_outside_figure_registry(
        ["noise", "drift"],
        registry_scenarios=["drift", "noise", "time_compress"],
        context="scenario figure scope",
    ) == ()


def test_scenarios_outside_figure_registry_reports_sorted_unknown_scenarios():
    assert scenarios_outside_figure_registry(
        ["local_timing_warp", "drift", "negative_spike"],
        registry_scenarios=["drift", "noise"],
        context="scenario figure scope",
    ) == ("local_timing_warp", "negative_spike")


@pytest.mark.parametrize(
    "scenarios, registry_scenarios, expected",
    [
        ([], ["drift"], "empty configured scenario list"),
        (["drift"], [], "empty figure-registry scenario vocabulary"),
        (["drift", ""], ["drift"], "scenarios\\[1\\]"),
        (["drift", "None"], ["drift"], "scenarios\\[1\\]"),
    ],
)
def test_scenarios_outside_figure_registry_rejects_malformed_inputs(
    scenarios,
    registry_scenarios,
    expected,
):
    with pytest.raises(ValueError, match=expected):
        scenarios_outside_figure_registry(
            scenarios,
            registry_scenarios=registry_scenarios,
            context="scenario figure scope",
        )


def test_scored_scenarios_render_every_scenario_figure():
    scored = _core_figure_registry().scenario_display_order

    assert _unsupported_figure_scenarios(_analysis_args(scored), full_coverage=True) == ()
    assert _unsupported_figure_scenarios(
        _analysis_args(["missing_data", "noise"]),
        full_coverage=True,
    ) == ()


def test_unregistered_scenario_raises_under_full_coverage():
    with pytest.raises(ValueError, match="local_timing_warp"):
        _unsupported_figure_scenarios(
            _analysis_args(["drift", "local_timing_warp"]),
            full_coverage=True,
        )


def test_unregistered_scenario_omits_scenario_keyed_outputs_under_partial_coverage(
    capsys,
):
    unsupported = _unsupported_figure_scenarios(
        _analysis_args(["negative_drift", "attenuation", "noise", "negative_spike"]),
        full_coverage=False,
    )

    assert unsupported == ("negative_drift", "negative_spike")
    message = capsys.readouterr().out
    assert "negative_drift" in message
    assert "method scenario-delta heatmap" in message
    assert "written empty" in message


def test_unregistered_scenario_raises_with_fixed_channel_fraction_under_partial_coverage():
    with pytest.raises(ValueError, match="fixed-channel-fraction summary"):
        _unsupported_figure_scenarios(
            _analysis_args(
                ["negative_drift", "attenuation", "noise"],
                fixed_channel_fraction=0.25,
            ),
            full_coverage=False,
        )


def test_scenario_family_tables_are_written_empty_for_unregistered_scenarios():
    delta_df, summary_df = _build_method_scenario_family_delta_tables(
        pd.DataFrame(
            columns=[
                "dataset",
                "robustness_method",
                "model_architecture",
                "scenario",
                "baseline_D",
                "delta_D",
            ]
        ),
        registry=_core_figure_registry(),
        delta_col="delta_D",
    )

    assert delta_df.empty
    assert summary_df.empty
    assert list(delta_df.columns) == list(METHOD_SCENARIO_FAMILY_DELTA_COLUMNS)
    assert list(summary_df.columns) == list(METHOD_SCENARIO_FAMILY_SUMMARY_COLUMNS)
