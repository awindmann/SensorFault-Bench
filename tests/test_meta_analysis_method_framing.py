from pathlib import Path

import pandas as pd
import pytest

from testing.meta_analysis import (
    FigureArtifactSpec,
    _build_method_scenario_family_delta_tables,
    _render_figure_registry,
)
from utils.parsing import parse_core_figure_registry_config


def _core_registry():
    return parse_core_figure_registry_config(
        {
            "BASELINE_RANK_PARETO_METRIC": "D_w",
            "CORE_IMPROVEMENT_TRAJECTORY_METHOD": "adversarial_training",
            "CORE_IMPROVEMENT_TRAJECTORY_METRIC": "D_w",
            "CORE_FIGURE_DATASET_SPEC": [
                "BeijingAir_Tiantan=BeijingAir",
                "Penmanshiel_Hourly_WT08=Penmanshiel",
                "ETTh1=ETTh1",
            ],
            "CORE_METHOD_DISPLAY": [
                "adversarial_training=PGD",
            ],
            "CORE_SCENARIO_DISPLAY_ORDER": ["drift", "noise", "missing_data"],
            "CORE_SCENARIO_DISPLAY": [
                "drift=Drift",
                "noise=Noise",
                "missing_data=MissingData",
            ],
            "CORE_SCENARIO_GROUPS": [
                "Value=drift,noise",
                "Availability=missing_data",
            ],
        },
        context="test core figure registry",
    )


class _FailingFigure:
    def write_image(self, _path: str | Path) -> None:
        raise ValueError("kaleido unavailable")


def test_figure_registry_export_is_best_effort_for_partial_coverage(tmp_path, capsys):
    records = _render_figure_registry(
        figure_specs=[
            FigureArtifactSpec(
                figure=_FailingFigure(),
                rel_parts=("2_baselines", "smoke"),
                filename="figure.pdf",
                figure_type="baseline_smoke",
            )
        ],
        output_root=tmp_path.as_posix(),
        full_coverage=False,
    )

    assert records == []
    assert "full_coverage=false" in capsys.readouterr().out


def test_figure_registry_export_raises_for_full_coverage(tmp_path):
    with pytest.raises(ValueError, match="kaleido unavailable"):
        _render_figure_registry(
            figure_specs=[
                FigureArtifactSpec(
                    figure=_FailingFigure(),
                    rel_parts=("2_baselines", "smoke"),
                    filename="figure.pdf",
                    figure_type="baseline_smoke",
                )
            ],
            output_root=tmp_path.as_posix(),
            full_coverage=True,
        )


def test_method_scenario_family_tables_frame_method_gains_by_fault_family():
    rows = [
        {
            "dataset": dataset,
            "robustness_method": "adversarial_training",
            "model_architecture": "GRU",
            "scenario": scenario,
            "baseline_D": baseline_d,
            "delta_D": delta_d,
        }
        for dataset, scenario, baseline_d, delta_d in [
            ("BeijingAir_Tiantan", "drift", 2.0, -0.30),
            ("BeijingAir_Tiantan", "noise", 2.2, -0.40),
            ("BeijingAir_Tiantan", "missing_data", 1.1, 0.10),
            ("Penmanshiel_Hourly_WT08", "drift", 1.8, -0.20),
            ("Penmanshiel_Hourly_WT08", "noise", 1.7, -0.10),
            ("Penmanshiel_Hourly_WT08", "missing_data", 1.2, 0.05),
            ("ETTh1", "drift", 1.1, 0.03),
            ("ETTh1", "noise", 1.2, 0.02),
            ("ETTh1", "missing_data", 1.5, -0.05),
        ]
    ]

    delta_df, summary_df = _build_method_scenario_family_delta_tables(
        pd.DataFrame(rows),
        registry=_core_registry(),
        delta_col="delta_D",
    )

    value_rows = delta_df.loc[delta_df["scenario_family"] == "Value"]
    beijing_value = value_rows.loc[
        value_rows["dataset"] == "BeijingAir_Tiantan"
    ].iloc[0]
    penmanshiel_value = value_rows.loc[
        value_rows["dataset"] == "Penmanshiel_Hourly_WT08"
    ].iloc[0]
    etth1_value = value_rows.loc[value_rows["dataset"] == "ETTh1"].iloc[0]

    assert beijing_value["baseline_family_D_rank_desc"] == 1
    assert beijing_value["method_family_gain_rank_asc"] == 1
    assert beijing_value["improved_scenario_count"] == 2
    assert penmanshiel_value["baseline_family_D_rank_desc"] == 2
    assert etth1_value["effect_direction"] == "worsens"

    value_summary = summary_df.loc[
        summary_df["scenario_family"] == "Value"
    ].iloc[0]
    assert value_summary["method_family_delta_rank"] == 1
    assert value_summary["effect_direction"] == "improves"
    assert value_summary["improved_dataset_count"] == 2
    assert value_summary["family_scenarios_display"] == "Drift, Noise"
    assert value_summary["baseline_impact_dataset_order"].startswith(
        "BeijingAir=2.1; Penmanshiel=1.75"
    )
    assert value_summary["largest_gain_dataset_order"].startswith(
        "BeijingAir=-0.35; Penmanshiel=-0.15"
    )
    assert (
        value_summary["practitioner_frame"]
        == "PGD: improves Value faults (Drift, Noise); baseline burden by "
        "dataset: BeijingAir=2.1; Penmanshiel=1.75; ETTh1=1.15; method "
        "deltas by dataset: BeijingAir=-0.35; Penmanshiel=-0.15; ETTh1=0.025."
    )


def test_method_scenario_family_tables_reject_unknown_scenarios():
    df = pd.DataFrame(
        [
            {
                "dataset": "BeijingAir_Tiantan",
                "robustness_method": "adversarial_training",
                "model_architecture": "GRU",
                "scenario": "unknown",
                "baseline_D": 1.2,
                "delta_D": -0.1,
            }
        ]
    )

    with pytest.raises(ValueError, match="unexpected scenario ids"):
        _build_method_scenario_family_delta_tables(
            df,
            registry=_core_registry(),
            delta_col="delta_D",
        )
