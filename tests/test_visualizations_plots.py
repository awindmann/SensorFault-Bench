import numpy as np
import pandas as pd
import pytest

from scripts.render_forecast_plots import (
    load_forecast_trace_figure,
    load_forecast_trace_record,
    render_traces,
)
from visualizations.semantics import PlotSemanticsRecord
from visualizations.plots import (
    _score_axis_range,
    compute_ranked_performance_robustness_summary,
    metric_display_name,
    plot_forecast_extreme,
    plot_error_distribution_overview,
    plot_backbone_scenario_heatmap,
    plot_improvement_comparison,
    plot_improvement_deltas_heatmap,
    plot_improvement_trajectory_subplots,
    plot_method_delta_pair_subplots,
    plot_per_method_delta_scatter,
    plot_pareto,
    plot_improvement_trajectories,
    plot_pareto_dataset_panels,
    plot_perturbed_vs_clean_error_panels,
    plot_method_scenario_delta_heatmap,
    plot_ranked_performance_robustness_pareto,
    plot_ranked_performance_robustness_pareto_panels,
    plot_perturbation_curves,
    plot_scenario_delta_heatmap,
    plot_scenario_radar,
    robustness_metric_display_name,
    trajectory_output_label_for_method,
)


def _plot_semantics(
    *,
    direction: str,
    axis_family: str,
    display_label: str,
    neutral_value: float | None = None,
) -> PlotSemanticsRecord:
    return PlotSemanticsRecord(
        direction=direction,
        axis_family=axis_family,
        neutral_value=neutral_value,
        display_label=display_label,
    )


def _perf_semantics(display_label: str = "MSE (Test)") -> PlotSemanticsRecord:
    return _plot_semantics(
        direction="minimize",
        axis_family="numeric",
        neutral_value=None,
        display_label=display_label,
    )


def _score_semantics(
    display_label: str = "Worst-Scenario Robustness",
) -> PlotSemanticsRecord:
    return _plot_semantics(
        direction="maximize",
        axis_family="score",
        neutral_value=1.0,
        display_label=display_label,
    )


def _lower_is_better_score_semantics(display_label: str) -> PlotSemanticsRecord:
    return _plot_semantics(
        direction="minimize",
        axis_family="score",
        neutral_value=None,
        display_label=display_label,
    )


def _delta_semantics(
    display_label: str,
    *,
    direction: str,
) -> PlotSemanticsRecord:
    return _plot_semantics(
        direction=direction,
        axis_family="delta",
        neutral_value=0.0,
        display_label=display_label,
    )


def _trajectory_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    backbone_df = pd.DataFrame(
        [
            {"dataset": "etth1", "model": "GRU", "MSE_test": 1.2, "D_mean": 0.35},
            {"dataset": "etth1", "model": "TSMixer", "MSE_test": 1.0, "D_mean": 0.40},
        ]
    )
    improvement_df = pd.DataFrame(
        [
            {
                "dataset": "etth1",
                "backbone_architecture": "GRU",
                "model": "adaptive_robust_loss|GRU",
                "robustness_method": "adaptive_robust_loss",
                "MSE_test": 1.1,
                "D_mean": 0.30,
            },
            {
                "dataset": "etth1",
                "backbone_architecture": "TSMixer",
                "model": "adaptive_robust_loss|TSMixer",
                "robustness_method": "adaptive_robust_loss",
                "MSE_test": 0.95,
                "D_mean": 0.34,
            },
        ]
    )
    return backbone_df, improvement_df


def _trajectory_subplot_frames(
    *,
    dataset_count: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, str]]]:
    panel_spec = [
        ("BeijingAir_Tiantan", "BeijingAir"),
        ("Penmanshiel_Hourly_WT08", "Penmanshiel"),
        ("ETTh1", "ETTh1"),
        ("traffic", "traffic"),
    ]
    selected_panels = panel_spec[:dataset_count]
    baseline_rows: list[dict[str, object]] = []
    improvement_rows: list[dict[str, object]] = []
    for dataset_idx, (dataset_name, _) in enumerate(selected_panels):
        signature = f"sig_{dataset_idx}"
        for arch_idx, architecture in enumerate(["GRU", "TSMixer"]):
            baseline_rows.append(
                {
                    "dataset": dataset_name,
                    "data_config_signature": signature,
                    "model_architecture": architecture,
                    "MSE_test": 0.30 + 0.10 * dataset_idx + 0.02 * arch_idx,
                    "D_w": 0.72 + 0.03 * dataset_idx + 0.02 * arch_idx,
                }
            )
            improvement_rows.append(
                {
                    "dataset": dataset_name,
                    "data_config_signature": signature,
                    "robustness_method": "adaptive_robust_loss",
                    "pipeline_id": f"adaptive_robust_loss_{dataset_idx}_{architecture.lower()}",
                    "model_architecture": architecture,
                    "backbone_architecture": architecture,
                    "MSE_test": 0.28 + 0.10 * dataset_idx + 0.02 * arch_idx,
                    "D_w": 0.66 + 0.03 * dataset_idx + 0.02 * arch_idx,
                }
            )
    return (
        pd.DataFrame(baseline_rows),
        pd.DataFrame(improvement_rows),
        panel_spec,
    )


def _ranked_pareto_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "BeijingAir_Tiantan",
                "model": "GRU",
                "architecture_family": "Recurrent",
                "MSE_test": 0.30,
                "R_worst_scenario": 0.90,
            },
            {
                "dataset": "BeijingAir_Tiantan",
                "model": "ModernTCN",
                "architecture_family": "Convolution",
                "MSE_test": 0.40,
                "R_worst_scenario": 0.95,
            },
            {
                "dataset": "Penmanshiel_Hourly_WT08",
                "model": "GRU",
                "architecture_family": "Recurrent",
                "MSE_test": 0.50,
                "R_worst_scenario": 0.92,
            },
            {
                "dataset": "Penmanshiel_Hourly_WT08",
                "model": "ModernTCN",
                "architecture_family": "Convolution",
                "MSE_test": 0.40,
                "R_worst_scenario": 0.80,
            },
            {
                "dataset": "Penmanshiel_Hourly_WT08",
                "model": "PatchTST",
                "architecture_family": "Attention",
                "MSE_test": 0.20,
                "R_worst_scenario": 0.70,
            },
            {
                "dataset": "ETTh1",
                "model": "GRU",
                "architecture_family": "Recurrent",
                "MSE_test": 0.45,
                "R_worst_scenario": 0.88,
            },
            {
                "dataset": "ETTh1",
                "model": "ModernTCN",
                "architecture_family": "Convolution",
                "MSE_test": 0.55,
                "R_worst_scenario": 0.84,
            },
        ]
    )


def test_compute_ranked_performance_robustness_summary_uses_rank_on_x_and_raw_robustness_on_y():
    summary = compute_ranked_performance_robustness_summary(
        _ranked_pareto_df(),
        dataset_col="dataset",
        model_col="model",
        perf_col="MSE_test",
        robust_col="R_worst_scenario",
    ).sort_values("model", ignore_index=True)

    expected = pd.DataFrame(
            [
                {
                    "model": "GRU",
                    "avg_perf_rank": 1.6666666666666667,
                    "mean_metric_value": 0.90,
                    "n_datasets": 3,
                },
            {
                "model": "ModernTCN",
                "avg_perf_rank": 2.0,
                "mean_metric_value": 0.8633333333333333,
                "n_datasets": 3,
            },
            {
                "model": "PatchTST",
                "avg_perf_rank": 1.0,
                "mean_metric_value": 0.70,
                "n_datasets": 1,
            },
        ]
    )

    pd.testing.assert_frame_equal(summary, expected)


def test_plot_ranked_performance_robustness_pareto_uses_raw_robustness_frontier():
    fig = plot_ranked_performance_robustness_pareto(
        _ranked_pareto_df(),
        dataset_col="dataset",
        model_col="model",
        arch_col="architecture_family",
        perf_col="MSE_test",
        robust_col="R_worst_scenario",
        y_semantics=_score_semantics("Worst-Scenario Robustness"),
        x_title="Average MSE Rank",
        y_title="Mean Worst-Scenario Robustness",
    )

    assert float(fig.layout.xaxis.range[0]) > float(fig.layout.xaxis.range[1])
    assert fig.layout.yaxis.title.text == "Mean Worst-Scenario Robustness"

    pareto_trace = next(trace for trace in fig.data if trace.name == "Pareto frontier")
    assert list(map(float, pareto_trace.x)) == pytest.approx([1.0, 1.6666666666666667])
    assert list(map(float, pareto_trace.y)) == pytest.approx([0.70, 0.90])


def test_plot_ranked_performance_robustness_pareto_defaults_to_requested_test_metric_label():
    fig = plot_ranked_performance_robustness_pareto(
        _ranked_pareto_df().rename(columns={"MSE_test": "MAE_test"}),
        dataset_col="dataset",
        model_col="model",
        arch_col="architecture_family",
        perf_col="MAE_test",
        robust_col="R_worst_scenario",
        y_semantics=_score_semantics("Worst-Scenario Robustness"),
    )

    assert fig.layout.xaxis.title.text == "Average MAE Rank"


def test_plot_ranked_performance_robustness_pareto_panels_use_requested_titles():
    ranked_df = _ranked_pareto_df()
    fig = plot_ranked_performance_robustness_pareto_panels(
        [
            (
                "Short Horizon (96/6)",
                ranked_df.loc[
                    ranked_df["dataset"].isin(
                        ["BeijingAir_Tiantan", "Penmanshiel_Hourly_WT08"]
                    )
                ].copy(),
            ),
            ("Long Horizon (96/96)", ranked_df.loc[ranked_df["dataset"] == "ETTh1"].copy()),
        ],
        dataset_col="dataset",
        model_col="model",
        arch_col="architecture_family",
        perf_col="MSE_test",
        robust_col="R_worst_scenario",
        y_semantics=_score_semantics("Worst-Scenario Robustness"),
        x_title="Average MSE Rank",
        y_title="Mean Worst-Scenario Robustness",
    )

    annotation_texts = {str(annotation.text) for annotation in fig.layout.annotations}
    assert "Short Horizon (96/6)" in annotation_texts
    assert "Long Horizon (96/96)" in annotation_texts
    assert fig.layout.xaxis.title.text == "Average MSE Rank"
    assert fig.layout.yaxis.title.text == "Mean Worst-Scenario Robustness"
    assert float(fig.layout.xaxis.range[0]) > float(fig.layout.xaxis.range[1])


def test_plot_ranked_performance_robustness_pareto_panels_default_to_requested_test_metric_label():
    ranked_df = _ranked_pareto_df().rename(columns={"MSE_test": "MAE_test"})
    fig = plot_ranked_performance_robustness_pareto_panels(
        [
            (
                "Short Horizon (96/6)",
                ranked_df.loc[
                    ranked_df["dataset"].isin(
                        ["BeijingAir_Tiantan", "Penmanshiel_Hourly_WT08"]
                    )
                ].copy(),
            ),
            ("Long Horizon (96/96)", ranked_df.loc[ranked_df["dataset"] == "ETTh1"].copy()),
        ],
        dataset_col="dataset",
        model_col="model",
        arch_col="architecture_family",
        perf_col="MAE_test",
        robust_col="R_worst_scenario",
        y_semantics=_score_semantics("Worst-Scenario Robustness"),
        y_title="Mean Worst-Scenario Robustness",
    )

    assert fig.layout.xaxis.title.text == "Average MAE Rank"


def test_plot_ranked_performance_robustness_pareto_panels_show_second_panel_only_legend_groups():
    fig = plot_ranked_performance_robustness_pareto_panels(
        [
            (
                "Short",
                pd.DataFrame(
                    [
                        {
                            "dataset": "short",
                            "model": "GRU",
                            "architecture_family": "Recurrent",
                            "MSE_test": 0.30,
                            "R_worst_scenario": 0.90,
                        }
                    ]
                ),
            ),
            (
                "Long",
                pd.DataFrame(
                    [
                        {
                            "dataset": "long",
                            "model": "Chronos2",
                            "architecture_family": "Foundation",
                            "MSE_test": 0.40,
                            "R_worst_scenario": 0.80,
                        }
                    ]
                ),
            ),
        ],
        dataset_col="dataset",
        model_col="model",
        arch_col="architecture_family",
        perf_col="MSE_test",
        robust_col="R_worst_scenario",
        y_semantics=_score_semantics("Worst-Scenario Robustness"),
    )

    legend_names = {
        str(trace.name)
        for trace in fig.data
        if bool(getattr(trace, "showlegend", False))
    }
    assert "Recurrent" in legend_names
    assert "Foundation" in legend_names
    assert "Pareto frontier" in legend_names


def test_plot_ranked_performance_robustness_pareto_rejects_missing_group_values():
    with pytest.raises(ValueError, match="missing grouping values"):
        plot_ranked_performance_robustness_pareto(
            pd.DataFrame(
                [
                    {
                        "dataset": "d1",
                        "model": "GRU",
                        "architecture_family": None,
                        "MSE_test": 0.30,
                        "R_worst_scenario": 0.90,
                    },
                    {
                        "dataset": "d1",
                        "model": "TSMixer",
                        "architecture_family": "Recurrent",
                        "MSE_test": 0.40,
                        "R_worst_scenario": 0.80,
                    },
                ]
            ),
            dataset_col="dataset",
            model_col="model",
            arch_col="architecture_family",
            perf_col="MSE_test",
            robust_col="R_worst_scenario",
            y_semantics=_score_semantics("Worst-Scenario Robustness"),
        )


def test_plot_pareto_dataset_panels_creates_2x2_with_shared_legend():
    datasets = ["BeijingAir", "Penmanshiel", "ETTh1", "traffic"]
    frames = []
    for ds in datasets:
        frames.append(
            (
                ds,
                pd.DataFrame(
                    [
                        {
                            "model": "GRU",
                            "architecture_family": "Recurrent",
                            "MSE_test": 0.30 + hash(ds) % 10 * 0.01,
                            "R_worst_scenario": 0.85,
                        },
                        {
                            "model": "PatchTST",
                            "architecture_family": "Attention",
                            "MSE_test": 0.25 + hash(ds) % 10 * 0.01,
                            "R_worst_scenario": 0.78,
                        },
                    ]
                ),
            )
        )

    fig = plot_pareto_dataset_panels(
        frames,
        perf_col="MSE_test",
        robust_col="R_worst_scenario",
        x_semantics=_perf_semantics("MSE (Test)"),
        y_semantics=_score_semantics("Worst-Scenario Robustness"),
        model_col="model",
        arch_col="architecture_family",
        perf_lower_is_better=True,
        x_title="MSE (Test)",
        y_title="Worst-Scenario Robustness",
    )

    annotation_texts = {str(a.text) for a in fig.layout.annotations}
    for ds in datasets:
        assert ds in annotation_texts, f"Missing panel title for {ds}"
    legend_names = {
        str(t.name) for t in fig.data if bool(getattr(t, "showlegend", False))
    }
    assert legend_names == {"Pareto frontier"}
    marker_trace_names = {
        str(t.name) for t in fig.data if str(getattr(t, "mode", "")) == "markers+text"
    }
    assert marker_trace_names == {"Attention", "Recurrent"}
    for axis in fig.select_xaxes():
        assert float(axis.range[0]) < float(axis.range[1])
    assert fig.layout.width == 1100
    assert fig.layout.height == 380


def test_plot_pareto_dataset_panels_accepts_single_panel():
    single_frame = [
        (
            "only_one",
            pd.DataFrame(
                [{"model": "X", "MSE_test": 0.3, "R_worst_scenario": 0.8}]
            ),
        )
    ]
    fig = plot_pareto_dataset_panels(
        single_frame,
        perf_col="MSE_test",
        robust_col="R_worst_scenario",
        x_semantics=_perf_semantics("MSE (Test)"),
        y_semantics=_score_semantics("Worst-Scenario Robustness"),
    )

    annotation_texts = {str(a.text) for a in fig.layout.annotations}
    assert "only_one" in annotation_texts
    assert float(fig.layout.xaxis.range[0]) < float(fig.layout.xaxis.range[1])
    assert fig.layout.width == 550
    assert fig.layout.height == 380


def test_plot_pareto_dataset_panels_rejects_more_than_four_panels():
    frames = [
        (
            f"dataset_{idx}",
            pd.DataFrame(
                [{"model": "X", "MSE_test": 0.3 + 0.01 * idx, "R_worst_scenario": 0.8}]
            ),
        )
        for idx in range(5)
    ]

    with pytest.raises(ValueError, match="between 1 and 4 panels"):
        plot_pareto_dataset_panels(
            frames,
            perf_col="MSE_test",
            robust_col="R_worst_scenario",
            x_semantics=_perf_semantics("MSE (Test)"),
            y_semantics=_score_semantics("Worst-Scenario Robustness"),
        )


def test_plot_pareto_dataset_panels_prefers_wide_three_panel_layout():
    frames = [
        (
            f"dataset_{idx}",
            pd.DataFrame(
                [{"model": "X", "MSE_test": 0.3 + 0.01 * idx, "R_worst_scenario": 0.8}]
            ),
        )
        for idx in range(3)
    ]

    fig = plot_pareto_dataset_panels(
        frames,
        perf_col="MSE_test",
        robust_col="R_worst_scenario",
        x_semantics=_perf_semantics("MSE (Test)"),
        y_semantics=_score_semantics("Worst-Scenario Robustness"),
    )

    assert fig.layout.width == 825
    assert fig.layout.height == 380


def test_plot_pareto_defaults_to_standard_mse_axis_direction():
    df = pd.DataFrame(
        [
            {"model": "TSMixer", "architecture_family": "Recurrent", "MSE_test": 0.30, "D_w": 0.50},
            {"model": "PatchTST", "architecture_family": "Attention", "MSE_test": 0.25, "D_w": 0.45},
        ]
    )

    fig = plot_pareto(
        df,
        perf_col="MSE_test",
        robust_col="D_w",
        x_semantics=_perf_semantics("MSE (Test)"),
        y_semantics=_plot_semantics(
            direction="minimize",
            axis_family="numeric",
            neutral_value=None,
            display_label="Worst-Scenario Degradation",
        ),
        model_col="model",
        arch_col="architecture_family",
    )

    assert float(fig.layout.xaxis.range[0]) < float(fig.layout.xaxis.range[1])
    assert fig.layout.legend.title.text == "Architecture"
    annotation_texts = {str(annotation.text) for annotation in fig.layout.annotations}
    assert {"Better \u2190", "Better \u2193"}.issubset(annotation_texts)


def test_plot_pareto_supports_lower_lower_direction_pair():
    df = pd.DataFrame(
        [
            {"model": "TSMixer", "architecture_family": "Recurrent", "MSE_test": 0.30, "D_w": 0.50},
            {"model": "GRU", "architecture_family": "Recurrent", "MSE_test": 0.40, "D_w": 0.45},
            {"model": "ModernTCN", "architecture_family": "Convolution", "MSE_test": 0.45, "D_w": 0.60},
        ]
    )

    fig = plot_pareto(
        df,
        perf_col="MSE_test",
        robust_col="D_w",
        x_semantics=_perf_semantics("MSE (Test)"),
        y_semantics=_plot_semantics(
            direction="minimize",
            axis_family="numeric",
            neutral_value=None,
            display_label="Worst-Scenario Degradation",
        ),
        model_col="model",
        arch_col="architecture_family",
        perf_lower_is_better=True,
        flip_perf_axis=True,
    )

    pareto_trace = next(trace for trace in fig.data if trace.name == "Pareto frontier")
    assert list(map(float, pareto_trace.x)) == pytest.approx([0.40, 0.30])
    assert list(map(float, pareto_trace.y)) == pytest.approx([0.45, 0.50])
    assert fig.layout.yaxis.title.text == "Worst-Scenario Degradation"


def test_plot_pareto_supports_higher_higher_direction_pair():
    df = pd.DataFrame(
        [
            {"model": "TSMixer", "family": "Recurrent", "Accuracy_test": 0.80, "Utility": 0.70},
            {"model": "GRU", "family": "Recurrent", "Accuracy_test": 0.75, "Utility": 0.75},
            {"model": "ModernTCN", "family": "Convolution", "Accuracy_test": 0.60, "Utility": 0.60},
        ]
    )

    fig = plot_pareto(
        df,
        perf_col="Accuracy_test",
        robust_col="Utility",
        x_semantics=_plot_semantics(
            direction="maximize",
            axis_family="numeric",
            neutral_value=None,
            display_label="Accuracy (Test)",
        ),
        y_semantics=_plot_semantics(
            direction="maximize",
            axis_family="numeric",
            neutral_value=None,
            display_label="Robust Utility",
        ),
        model_col="model",
        arch_col="family",
        perf_lower_is_better=False,
        flip_perf_axis=True,
    )

    pareto_trace = next(trace for trace in fig.data if trace.name == "Pareto frontier")
    assert list(map(float, pareto_trace.x)) == pytest.approx([0.75, 0.80])
    assert list(map(float, pareto_trace.y)) == pytest.approx([0.75, 0.70])
    assert fig.layout.xaxis.title.text == "Accuracy (Test)"


def test_plot_method_scenario_delta_heatmap_marks_unavailable_rows():
    scenario_order = [
        "drift",
        "attenuation",
        "noise",
        "spike",
        "time_stretch",
        "time_compress",
        "stuck_sensor",
        "missing_data",
    ]
    rows = [
        {
            "dataset": "BeijingAir_Tiantan",
            "robustness_method": "adversarial_training",
            "scenario": scenario,
            "delta_D": -0.10 + 0.01 * idx,
        }
        for idx, scenario in enumerate(scenario_order)
    ]
    fig = plot_method_scenario_delta_heatmap(
        pd.DataFrame(rows),
        dataset_spec=[
            ("BeijingAir_Tiantan", "BeijingAir"),
            ("Penmanshiel_Hourly_WT08", "Penmanshiel"),
            ("ETTh1", "ETTh1"),
            ("traffic", "Traffic"),
        ],
        method_display={
            "adversarial_training": "PGD",
            "ensemble": "Ens.",
            "revin": "RevIN",
        },
        scenario_display_order=scenario_order,
        scenario_display={
            "drift": "Drift",
            "attenuation": "Attenuation",
            "noise": "Noise",
            "spike": "Spike",
            "time_stretch": "TimeStretch",
            "time_compress": "TimeCompress",
            "stuck_sensor": "StuckSensor",
            "missing_data": "MissingData",
        },
        scenario_groups={
            "Value": ("drift", "attenuation", "noise", "spike"),
            "Timing": ("time_stretch", "time_compress"),
            "Availability": ("stuck_sensor", "missing_data"),
        },
        value_col="delta_D",
        value_semantics=_delta_semantics(
            "Δ Scenario Degradation",
            direction="minimize",
        ),
    )

    overlay_texts = []
    for trace in fig.data:
        text = getattr(trace, "text", None)
        if text is None:
            continue
        overlay_texts.extend(str(value) for row in text for value in row)
    assert "NA" in overlay_texts
    assert tuple(fig.layout.yaxis.ticktext) == ("PGD", "Ens.", "RevIN")
    assert tuple(fig.layout.xaxis.ticktext) == (
        "Drift",
        "Attenuation",
        "Noise",
        "Spike",
        "TimeStretch",
        "TimeCompress",
        "StuckSensor",
        "MissingData",
    )
    assert fig.layout.coloraxis.colorbar.title.text == "Delta Scenario Degradation"


def test_plot_method_scenario_delta_heatmap_rejects_unexpected_dataset_ids():
    df = pd.DataFrame(
        [
            {
                "dataset": "unknown_dataset",
                "robustness_method": "adversarial_training",
                "scenario": "drift",
                "delta_D": -0.10,
            }
        ]
    )

    with pytest.raises(ValueError, match="unexpected dataset ids"):
        plot_method_scenario_delta_heatmap(
            df,
            dataset_spec=[
                ("BeijingAir_Tiantan", "BeijingAir"),
                ("Penmanshiel_Hourly_WT08", "Penmanshiel"),
                ("ETTh1", "ETTh1"),
                ("traffic", "Traffic"),
            ],
            method_display={
                "adversarial_training": "AT",
                "ensemble": "Ensemble",
                "revin": "RevIN",
            },
            scenario_display_order=[
                "drift",
                "attenuation",
                "noise",
                "spike",
                "time_stretch",
                "time_compress",
                "stuck_sensor",
                "missing_data",
            ],
            scenario_display={
                "drift": "Drift",
                "attenuation": "Attenuation",
                "noise": "Noise",
                "spike": "Spike",
                "time_stretch": "TimeStretch",
                "time_compress": "TimeCompress",
                "stuck_sensor": "StuckSensor",
                "missing_data": "MissingData",
            },
            scenario_groups={
                "Value": ("drift", "attenuation", "noise", "spike"),
                "Timing": ("time_stretch", "time_compress"),
                "Availability": ("stuck_sensor", "missing_data"),
            },
            value_col="delta_D",
            value_semantics=_delta_semantics(
                "Δ Scenario Degradation",
                direction="minimize",
            ),
        )


def test_plot_method_scenario_delta_heatmap_rejects_display_dataset_ids():
    df = pd.DataFrame(
        [
            {
                "dataset": "BeijingAir",
                "robustness_method": "adversarial_training",
                "scenario": "drift",
                "delta_D": -0.10,
            },
            {
                "dataset": "BeijingAir_Tiantan",
                "robustness_method": "adversarial_training",
                "scenario": "attenuation",
                "delta_D": -0.09,
            },
        ]
    )

    with pytest.raises(ValueError, match="mixed raw/display dataset ids"):
        plot_method_scenario_delta_heatmap(
            df,
            dataset_spec=[
                ("BeijingAir_Tiantan", "BeijingAir"),
            ],
            method_display={
                "adversarial_training": "PGD",
            },
            scenario_display_order=["drift", "attenuation"],
            scenario_display={
                "drift": "Drift",
                "attenuation": "Attenuation",
            },
            scenario_groups={
                "Value": ("drift", "attenuation"),
            },
            value_col="delta_D",
            value_semantics=_delta_semantics(
                "Δ Scenario Degradation",
                direction="minimize",
            ),
        )


def test_plot_method_scenario_delta_heatmap_rejects_unexpected_method_ids():
    df = pd.DataFrame(
        [
            {
                "dataset": "BeijingAir_Tiantan",
                "robustness_method": "unknown_method",
                "scenario": "drift",
                "delta_D": -0.10,
            }
        ]
    )

    with pytest.raises(ValueError, match="unexpected robustness_method ids"):
        plot_method_scenario_delta_heatmap(
            df,
            dataset_spec=[
                ("BeijingAir_Tiantan", "BeijingAir"),
            ],
            method_display={
                "adversarial_training": "PGD",
            },
            scenario_display_order=["drift"],
            scenario_display={
                "drift": "Drift",
            },
            scenario_groups={
                "Value": ("drift",),
            },
            value_col="delta_D",
            value_semantics=_delta_semantics(
                "Δ Scenario Degradation",
                direction="minimize",
            ),
        )


def test_plot_method_scenario_delta_heatmap_rejects_unexpected_scenario_ids():
    df = pd.DataFrame(
        [
            {
                "dataset": "BeijingAir_Tiantan",
                "robustness_method": "adversarial_training",
                "scenario": "unknown_scenario",
                "delta_D": -0.10,
            }
        ]
    )

    with pytest.raises(ValueError, match="unexpected scenario ids"):
        plot_method_scenario_delta_heatmap(
            df,
            dataset_spec=[
                ("BeijingAir_Tiantan", "BeijingAir"),
            ],
            method_display={
                "adversarial_training": "PGD",
            },
            scenario_display_order=["drift"],
            scenario_display={
                "drift": "Drift",
            },
            scenario_groups={
                "Value": ("drift",),
            },
            value_col="delta_D",
            value_semantics=_delta_semantics(
                "Δ Scenario Degradation",
                direction="minimize",
            ),
        )


def test_plot_backbone_scenario_heatmap_renders_numeric_semantics():
    scenario_order = ["drift", "noise"]
    rows = [
        {
            "dataset": "BeijingAir_Tiantan",
            "model_architecture": "GRU",
            "scenario": "drift",
            "D": 1.02,
        },
        {
            "dataset": "BeijingAir_Tiantan",
            "model_architecture": "GRU",
            "scenario": "noise",
            "D": 1.08,
        },
        {
            "dataset": "BeijingAir_Tiantan",
            "model_architecture": "PatchTST",
            "scenario": "drift",
            "D": 1.10,
        },
        {
            "dataset": "BeijingAir_Tiantan",
            "model_architecture": "PatchTST",
            "scenario": "noise",
            "D": 1.24,
        },
    ]

    fig = plot_backbone_scenario_heatmap(
        pd.DataFrame(rows),
        dataset_spec=[("BeijingAir_Tiantan", "BeijingAir")],
        backbone_display={
            "GRU": "GRU",
            "PatchTST": "PatchTST",
        },
        scenario_display_order=scenario_order,
        scenario_display={
            "drift": "Drift",
            "noise": "Noise",
        },
        scenario_groups={
            "Value": ("drift", "noise"),
        },
        value_col="D",
        value_semantics=_perf_semantics("Scenario Degradation"),
    )

    assert tuple(fig.layout.yaxis.ticktext) == ("GRU", "PatchTST")
    assert tuple(fig.layout.xaxis.ticktext) == ("Drift", "Noise")
    assert fig.layout.yaxis.title.text == "Backbone"
    assert fig.layout.coloraxis.colorbar.title.text == "Scenario Degradation"
    assert fig.layout.coloraxis.cmid is None


def test_plot_backbone_scenario_heatmap_rejects_unexpected_backbone_ids():
    df = pd.DataFrame(
        [
            {
                "dataset": "BeijingAir_Tiantan",
                "model_architecture": "unknown_backbone",
                "scenario": "drift",
                "D": 1.10,
            }
        ]
    )

    with pytest.raises(ValueError, match="unexpected model_architecture ids"):
        plot_backbone_scenario_heatmap(
            df,
            dataset_spec=[("BeijingAir_Tiantan", "BeijingAir")],
            backbone_display={
                "GRU": "GRU",
            },
            scenario_display_order=["drift"],
            scenario_display={
                "drift": "Drift",
            },
            scenario_groups={
                "Value": ("drift",),
            },
            value_col="D",
            value_semantics=_perf_semantics("Scenario Degradation"),
        )


def test_plot_improvement_trajectories_marker_semantics():
    backbone_df, improvement_df = _trajectory_frames()
    fig = plot_improvement_trajectories(
        backbone_df,
        improvement_df,
        perf_col="MSE_test",
        robust_col="D_mean",
        x_semantics=_perf_semantics("MSE (Test)"),
        y_semantics=_lower_is_better_score_semantics("Mean Degradation"),
        model_col="model",
        method_col="backbone_architecture",
        backbone_col="backbone_architecture",
        show_pareto=False,
    )

    names = [trace.name for trace in fig.data]
    assert "GRU (baseline)" in names
    assert "GRU (robustness)" in names
    assert "TSMixer (baseline)" in names
    assert "TSMixer (robustness)" in names
    assert "Robustness method output" in names
    non_baseline_traces = [
        trace for trace in fig.data if not str(trace.name).endswith("(baseline)")
    ]
    assert all(trace.mode != "markers+text" for trace in non_baseline_traces)

    baseline_trace = next(trace for trace in fig.data if trace.name == "GRU (baseline)")
    robustness_trace = next(trace for trace in fig.data if trace.name == "GRU (robustness)")

    assert baseline_trace.marker.color != "white"
    assert baseline_trace.marker.line.color == baseline_trace.marker.color
    assert int(baseline_trace.marker.line.width) == 1
    assert robustness_trace.marker.color == "white"
    assert robustness_trace.marker.line.color == baseline_trace.marker.color
    assert int(robustness_trace.marker.line.width) == 2


def test_plot_improvement_trajectories_rejects_missing_grouping_column():
    backbone_df, improvement_df = _trajectory_frames()

    with pytest.raises(ValueError, match="missing grouping column"):
        plot_improvement_trajectories(
            backbone_df,
            improvement_df,
            perf_col="MSE_test",
            robust_col="D_mean",
            x_semantics=_perf_semantics("MSE (Test)"),
            y_semantics=_lower_is_better_score_semantics("Mean Degradation"),
            model_col="model",
            method_col="missing_group_col",
            backbone_col="backbone_architecture",
            show_pareto=False,
        )


def test_plot_improvement_trajectories_supports_custom_output_label():
    backbone_df, improvement_df = _trajectory_frames()
    fig = plot_improvement_trajectories(
        backbone_df,
        improvement_df,
        perf_col="MSE_test",
        robust_col="D_mean",
        x_semantics=_perf_semantics("MSE (Test)"),
        y_semantics=_lower_is_better_score_semantics("Mean Degradation"),
        model_col="model",
        method_col="backbone_architecture",
        backbone_col="backbone_architecture",
        show_pareto=False,
        robustness_output_label="+ Dropout",
    )

    names = [trace.name for trace in fig.data]
    assert "+ Dropout" in names
    assert "Robustness method output" not in names


def test_plot_improvement_trajectory_subplots_use_ordered_titles_and_shared_legend():
    backbone_df, improvement_df, panel_spec = _trajectory_subplot_frames()

    fig = plot_improvement_trajectory_subplots(
        backbone_df,
        improvement_df,
        panel_spec=panel_spec,
        perf_col="MSE_test",
        robust_col="D_w",
        x_semantics=_perf_semantics("MSE (Test)"),
        y_semantics=_lower_is_better_score_semantics("Worst-Scenario Degradation"),
        method_col="backbone_architecture",
        backbone_col="backbone_architecture",
        baseline_backbone_col="model_architecture",
        improvement_name_col="pipeline_id",
    )

    title_texts = [
        str(annotation.text)
        for annotation in fig.layout.annotations
        if not annotation.showarrow and str(annotation.text)
    ]
    legend_names = [trace.name for trace in fig.data if trace.showlegend]

    assert title_texts == ["BeijingAir", "Penmanshiel", "ETTh1", "traffic"]
    assert legend_names == ["Baseline", "Robustness method output", "Pareto frontier"]
    assert fig.layout.title.text is None
    assert fig.layout.width == 550
    assert fig.layout.height == 550
    assert any(
        axis.title.text == "Worst-Scenario Degradation"
        for axis in fig.select_yaxes()
    )
    assert all(float(axis.range[0]) < float(axis.range[1]) for axis in fig.select_xaxes())
    labeled_baseline_traces = [
        trace for trace in fig.data if trace.name.endswith("(baseline)") and trace.text is not None
    ]
    assert labeled_baseline_traces
    non_baseline_traces = [
        trace for trace in fig.data if not str(trace.name).endswith("(baseline)")
    ]
    assert all(trace.mode != "markers+text" for trace in non_baseline_traces)


@pytest.mark.parametrize(
    ("dataset_count", "expected_axis_count", "expected_height", "expected_titles"),
    [
        (1, 1, 380, ["BeijingAir"]),
        (2, 2, 320, ["BeijingAir", "Penmanshiel"]),
        (3, 4, 550, ["BeijingAir", "Penmanshiel", "ETTh1"]),
        (4, 4, 550, ["BeijingAir", "Penmanshiel", "ETTh1", "traffic"]),
    ],
)
def test_plot_improvement_trajectory_subplots_choose_expected_layout(
    dataset_count: int,
    expected_axis_count: int,
    expected_height: int,
    expected_titles: list[str],
):
    backbone_df, improvement_df, panel_spec = _trajectory_subplot_frames(
        dataset_count=dataset_count
    )

    fig = plot_improvement_trajectory_subplots(
        backbone_df,
        improvement_df,
        panel_spec=panel_spec,
        perf_col="MSE_test",
        robust_col="D_w",
        x_semantics=_perf_semantics("MSE (Test)"),
        y_semantics=_lower_is_better_score_semantics("Worst-Scenario Degradation"),
        method_col="backbone_architecture",
        backbone_col="backbone_architecture",
        baseline_backbone_col="model_architecture",
        improvement_name_col="pipeline_id",
    )

    title_texts = [
        str(annotation.text)
        for annotation in fig.layout.annotations
        if not annotation.showarrow and str(annotation.text)
    ]

    assert len(list(fig.select_xaxes())) == expected_axis_count
    assert len(list(fig.select_yaxes())) == expected_axis_count
    assert fig.layout.height == expected_height
    assert title_texts == expected_titles


def test_plot_perturbed_vs_clean_error_panels_render_four_panel_layout():
    panel_frames = [
        (
            dataset_name,
            pd.DataFrame(
                [
                    {
                        "model": "GRU",
                        "MSE_test": 0.30 + 0.05 * idx,
                        "MSE_w": 0.45 + 0.05 * idx,
                    },
                    {
                        "model": "PatchTST",
                        "MSE_test": 0.25 + 0.05 * idx,
                        "MSE_w": 0.40 + 0.05 * idx,
                    },
                ]
            ),
        )
        for idx, dataset_name in enumerate(["BeijingAir", "Penmanshiel", "ETTh1", "Traffic"])
    ]

    fig = plot_perturbed_vs_clean_error_panels(
        panel_frames,
        model_col="model",
        clean_col="MSE_test",
        pert_col="MSE_w",
        x_semantics=_perf_semantics("MSE_c"),
        y_semantics=_plot_semantics(
            direction="minimize",
            axis_family="numeric",
            neutral_value=None,
            display_label="MSE_w",
        ),
        x_title="MSE_c",
        y_title="MSE_w",
    )

    annotation_texts = {str(a.text) for a in fig.layout.annotations}
    assert annotation_texts.issuperset({"BeijingAir", "Penmanshiel", "ETTh1", "Traffic"})
    assert fig.layout.width == 1100
    assert fig.layout.height == 380
    assert not any(bool(getattr(trace, "showlegend", False)) for trace in fig.data)
    parity_traces = [trace for trace in fig.data if str(trace.name) == "Parity"]
    assert len(parity_traces) == 4
    for trace in parity_traces:
        assert str(trace.mode) == "lines"
        assert list(map(float, trace.x)) == pytest.approx(list(map(float, trace.y)))
        assert str(trace.line.dash) == "dash"
    x_axes = list(fig.select_xaxes())
    y_axes = list(fig.select_yaxes())
    assert len(x_axes) == 4
    assert len(y_axes) == 4
    for idx, (x_axis, y_axis) in enumerate(zip(x_axes, y_axes), start=1):
        axis_suffix = "" if idx == 1 else str(idx)
        assert float(x_axis.range[0]) < float(x_axis.range[1])
        assert list(map(float, x_axis.range)) == pytest.approx(list(map(float, y_axis.range)))
        assert str(x_axis.scaleanchor) == f"y{axis_suffix}"
        assert float(x_axis.scaleratio) == pytest.approx(1.0)
        assert str(x_axis.constrain) == "domain"


def test_plot_improvement_trajectory_subplots_reject_missing_signature_values():
    backbone_df, improvement_df, panel_spec = _trajectory_subplot_frames()
    backbone_df = backbone_df.copy()
    backbone_df.loc[0, "data_config_signature"] = np.nan

    with pytest.raises(ValueError, match="data_config_signature"):
        plot_improvement_trajectory_subplots(
            backbone_df,
            improvement_df,
            panel_spec=panel_spec,
            perf_col="MSE_test",
            robust_col="D_w",
            x_semantics=_perf_semantics("MSE (Test)"),
            y_semantics=_lower_is_better_score_semantics("Worst-Scenario Degradation"),
            method_col="backbone_architecture",
            backbone_col="backbone_architecture",
            baseline_backbone_col="model_architecture",
            improvement_name_col="pipeline_id",
        )


def test_plot_improvement_trajectory_subplots_reject_dataset_outside_panel_spec():
    backbone_df, improvement_df, panel_spec = _trajectory_subplot_frames()
    extra_row = backbone_df.iloc[[0]].copy()
    extra_row["dataset"] = "custom_dataset"
    backbone_df = pd.concat([backbone_df, extra_row], ignore_index=True)

    with pytest.raises(ValueError, match="outside the panel spec"):
        plot_improvement_trajectory_subplots(
            backbone_df,
            improvement_df,
            panel_spec=panel_spec,
            perf_col="MSE_test",
            robust_col="R_cvar",
            x_semantics=_perf_semantics("MSE (Test)"),
            y_semantics=_score_semantics("CVaR Robustness"),
            method_col="backbone_architecture",
            backbone_col="backbone_architecture",
            baseline_backbone_col="model_architecture",
            improvement_name_col="pipeline_id",
        )


def test_plot_improvement_trajectory_subplots_reject_duplicate_panel_spec_dataset_keys():
    backbone_df, improvement_df, _ = _trajectory_subplot_frames(dataset_count=1)

    with pytest.raises(ValueError, match="duplicate dataset key"):
        plot_improvement_trajectory_subplots(
            backbone_df,
            improvement_df,
            panel_spec=[
                ("BeijingAir_Tiantan", "BeijingAir"),
                ("BeijingAir_Tiantan", "Duplicate"),
            ],
            perf_col="MSE_test",
            robust_col="R_cvar",
            x_semantics=_perf_semantics("MSE (Test)"),
            y_semantics=_score_semantics("CVaR Robustness"),
            method_col="backbone_architecture",
            backbone_col="backbone_architecture",
            baseline_backbone_col="model_architecture",
            improvement_name_col="pipeline_id",
        )


def test_plot_improvement_trajectory_subplots_reject_empty_panel_spec():
    backbone_df, improvement_df, _ = _trajectory_subplot_frames(dataset_count=1)

    with pytest.raises(ValueError, match="panel spec must be non-empty"):
        plot_improvement_trajectory_subplots(
            backbone_df,
            improvement_df,
            panel_spec=[],
            perf_col="MSE_test",
            robust_col="R_cvar",
            x_semantics=_perf_semantics("MSE (Test)"),
            y_semantics=_score_semantics("CVaR Robustness"),
            method_col="backbone_architecture",
            backbone_col="backbone_architecture",
            baseline_backbone_col="model_architecture",
            improvement_name_col="pipeline_id",
        )


def test_plot_improvement_trajectory_subplots_reject_missing_signature_column():
    backbone_df, improvement_df, panel_spec = _trajectory_subplot_frames()
    backbone_df = backbone_df.drop(columns=["data_config_signature"])

    with pytest.raises(ValueError, match="missing columns"):
        plot_improvement_trajectory_subplots(
            backbone_df,
            improvement_df,
            panel_spec=panel_spec,
            perf_col="MSE_test",
            robust_col="R_cvar",
            x_semantics=_perf_semantics("MSE (Test)"),
            y_semantics=_score_semantics("CVaR Robustness"),
            method_col="backbone_architecture",
            backbone_col="backbone_architecture",
            baseline_backbone_col="model_architecture",
            improvement_name_col="pipeline_id",
        )


def test_plot_improvement_trajectory_subplots_reject_duplicate_baseline_join_rows():
    backbone_df, improvement_df, panel_spec = _trajectory_subplot_frames(dataset_count=1)
    backbone_df = pd.concat([backbone_df, backbone_df.iloc[[0]].copy()], ignore_index=True)

    with pytest.raises(ValueError, match="baseline rows must be unique"):
        plot_improvement_trajectory_subplots(
            backbone_df,
            improvement_df,
            panel_spec=panel_spec,
            perf_col="MSE_test",
            robust_col="D_w",
            x_semantics=_perf_semantics("MSE (Test)"),
            y_semantics=_lower_is_better_score_semantics("Worst-Scenario Degradation"),
            method_col="backbone_architecture",
            backbone_col="backbone_architecture",
            baseline_backbone_col="model_architecture",
            improvement_name_col="pipeline_id",
        )


def test_plot_improvement_trajectory_subplots_reject_duplicate_improvement_join_rows():
    backbone_df, improvement_df, panel_spec = _trajectory_subplot_frames(dataset_count=1)
    duplicate_row = improvement_df.iloc[[0]].copy()
    duplicate_row["model_architecture"] = "DuplicateArch"
    improvement_df = pd.concat([improvement_df, duplicate_row], ignore_index=True)

    with pytest.raises(ValueError, match="improvement rows must be unique"):
        plot_improvement_trajectory_subplots(
            backbone_df,
            improvement_df,
            panel_spec=panel_spec,
            perf_col="MSE_test",
            robust_col="D_w",
            x_semantics=_perf_semantics("MSE (Test)"),
            y_semantics=_lower_is_better_score_semantics("Worst-Scenario Degradation"),
            method_col="backbone_architecture",
            backbone_col="backbone_architecture",
            baseline_backbone_col="model_architecture",
            improvement_name_col="pipeline_id",
        )


def test_plot_improvement_trajectory_subplots_reject_unmatched_improvement_rows():
    backbone_df, improvement_df, panel_spec = _trajectory_subplot_frames(dataset_count=1)
    improvement_df = improvement_df.copy()
    improvement_df.loc[0, "backbone_architecture"] = "PatchTST"

    with pytest.raises(ValueError, match="do not match baseline trajectory join keys"):
        plot_improvement_trajectory_subplots(
            backbone_df,
            improvement_df,
            panel_spec=panel_spec,
            perf_col="MSE_test",
            robust_col="D_w",
            x_semantics=_perf_semantics("MSE (Test)"),
            y_semantics=_lower_is_better_score_semantics("Worst-Scenario Degradation"),
            method_col="backbone_architecture",
            backbone_col="backbone_architecture",
            baseline_backbone_col="model_architecture",
            improvement_name_col="pipeline_id",
        )


def test_plot_perturbation_curves_keeps_non_core_scenarios():
    df = pd.DataFrame(
        [
            {"pert_idx": 0, "severity": 0.25, "R_cell": 0.9},
            {"pert_idx": 0, "severity": 0.75, "R_cell": 0.6},
            {"pert_idx": 9, "severity": 0.25, "R_cell": 0.8},
            {"pert_idx": 9, "severity": 0.75, "R_cell": 0.5},
        ]
    )
    fig = plot_perturbation_curves(
        df,
        model_name="TSMixer",
        value_col="R_cell",
        y_semantics=_score_semantics("Robustness Score"),
        idx_to_name={0: "drift", 9: "linear_drift"},
    )

    trace_names = [str(trace.name) for trace in fig.data]
    assert trace_names == ["Drift", "LinearDrift"]


def test_trajectory_output_label_rejects_disabled_revin_input_only():
    with pytest.raises(ValueError, match="unknown robustness method"):
        trajectory_output_label_for_method("revin_input_only")


def test_trajectory_output_label_uses_configured_method_label():
    assert (
        trajectory_output_label_for_method("adversarial_training")
        == "+ PGD adversarial training"
    )


def test_plot_perturbation_curves_expands_y_axis_for_scores_above_one():
    df = pd.DataFrame(
        [
            {"pert_idx": 0, "severity": 0.25, "R_cell": 1.4},
            {"pert_idx": 0, "severity": 0.75, "R_cell": 1.1},
        ]
    )

    fig = plot_perturbation_curves(
        df,
        model_name="TSMixer",
        value_col="R_cell",
        y_semantics=_score_semantics("Robustness Score"),
        idx_to_name={0: "drift"},
    )

    assert fig.layout.yaxis.title.text == "Robustness Score"
    assert float(fig.layout.yaxis.range[1]) > 1.4


def test_plot_perturbation_curves_prepends_benign_anchor_when_missing():
    df = pd.DataFrame(
        [
            {"pert_idx": 0, "severity": 0.5, "R_cell": 0.8},
            {"pert_idx": 0, "severity": 1.0, "R_cell": 0.6},
            {"pert_idx": 1, "severity": 0.5, "R_cell": 0.7},
            {"pert_idx": 1, "severity": 1.0, "R_cell": 0.4},
        ]
    )

    fig = plot_perturbation_curves(
        df,
        model_name="TSMixer",
        value_col="R_cell",
        y_semantics=_score_semantics("Robustness Score"),
        idx_to_name={0: "drift", 1: "noise"},
        neutral_anchor_value=1.0,
    )

    for trace in fig.data:
        assert float(trace.x[0]) == pytest.approx(0.0)
        assert float(trace.y[0]) == pytest.approx(1.0)


def test_plot_perturbation_curves_does_not_duplicate_existing_zero_anchor():
    df = pd.DataFrame(
        [
            {"pert_idx": 0, "severity": 0.0, "R_cell": 1.0},
            {"pert_idx": 0, "severity": 0.5, "R_cell": 0.8},
            {"pert_idx": 0, "severity": 1.0, "R_cell": 0.6},
        ]
    )

    fig = plot_perturbation_curves(
        df,
        model_name="TSMixer",
        value_col="R_cell",
        y_semantics=_score_semantics("Robustness Score"),
        idx_to_name={0: "drift"},
        neutral_anchor_value=1.0,
    )

    drift_trace = fig.data[0]
    zero_levels = [float(level) for level in drift_trace.x if np.isclose(float(level), 0.0)]
    assert zero_levels == [0.0]


def test_plot_perturbation_curves_preserves_observed_severity_levels():
    df = pd.DataFrame(
        [
            {"pert_idx": 0, "severity": 0.2, "R_cell": 0.8},
            {"pert_idx": 0, "severity": 0.4, "R_cell": 0.6},
        ]
    )

    fig = plot_perturbation_curves(
        df,
        model_name="TSMixer",
        value_col="R_cell",
        y_semantics=_score_semantics("Robustness Score"),
        idx_to_name={0: "drift"},
        neutral_anchor_value=1.0,
    )

    drift_trace = fig.data[0]
    assert list(map(float, drift_trace.x)) == pytest.approx([0.0, 0.2, 0.4])
    assert list(map(float, drift_trace.y)) == pytest.approx([1.0, 0.8, 0.6])


def test_plot_scenario_radar_expands_radial_range_for_scores_above_one():
    fig = plot_scenario_radar(
        {"Model A": {"drift": 1.2, "noise": 1.6}},
        title="Radar",
    )

    assert float(fig.layout.polar.radialaxis.range[1]) > 1.6


def test_plot_scenario_radar_defaults_to_generic_metric_language():
    fig = plot_scenario_radar(
        {"Model A": {"drift": 0.8, "noise": 0.6}},
    )

    assert fig.layout.title.text == "Scenario Metric Profile"
    assert "value=%{r:.4f}" in str(fig.data[0].hovertemplate)


def test_plot_scenario_radar_empty_defaults_to_generic_no_data_title():
    fig = plot_scenario_radar({})

    assert fig.layout.title.text == "Scenario Metric Profile (No Data)"


def test_plot_scenario_delta_heatmap_defaults_to_generic_delta_title():
    scenario_delta_df = pd.DataFrame(
        [
            {
                "dataset": "BeijingAir_Tiantan",
                "robustness_method": "adversarial_training",
                "scenario": "drift",
                "delta_metric_value": -0.2,
            },
            {
                "dataset": "BeijingAir_Tiantan",
                "robustness_method": "adversarial_training",
                "scenario": "noise",
                "delta_metric_value": -0.1,
            },
        ]
    )

    fig = plot_scenario_delta_heatmap(
        scenario_delta_df,
        value_semantics=_delta_semantics(
            "Delta Scenario Degradation",
            direction="minimize",
        ),
    )

    assert fig.layout.title.text == "Scenario Delta vs Baseline"


def test_plot_improvement_comparison_uses_canonical_metric_display_names():
    deltas_long_df = pd.DataFrame(
        [
            {
                "robustness_method": "adaptive_robust_loss",
                "metric_name": "D_w",
                "delta_value": -0.2,
            },
            {
                "robustness_method": "adaptive_robust_loss",
                "metric_name": "MSE_test",
                "delta_value": -0.1,
            },
        ]
    )

    fig = plot_improvement_comparison(deltas_long_df)

    annotation_texts = {str(annotation.text) for annotation in fig.layout.annotations}
    assert "Worst-Scenario Degradation" in annotation_texts
    assert "MSE (Test)" in annotation_texts


def test_metric_display_name_supports_current_meta_analysis_scores():
    assert robustness_metric_display_name("mCE_snaive") == (
        "Mean CE vs SeasonalNaive"
    )
    assert robustness_metric_display_name("relative_mCE_snaive") == (
        "Mean Relative CE vs SeasonalNaive"
    )
    assert metric_display_name("rho_eff") == "Effective Robustness Residual"
    assert metric_display_name("tau_mean") == "Mean Corrupted-Error Effect"


def test_robustness_metric_display_name_rejects_retired_score_keys():
    with pytest.raises(ValueError, match="Unknown robustness metric key 'R_cvar'"):
        robustness_metric_display_name("R_cvar")


def test_plot_forecast_extreme_limits_large_multitarget_to_explicit_subset():
    n_channels = 20
    target = np.tile(np.arange(4, dtype=float).reshape(-1, 1), (1, n_channels))
    prediction_perturbed = target + 0.5
    prediction_clean = target + 0.25
    target[:, 2] = np.array([5.0, 15.0, 25.0, 35.0], dtype=float)
    prediction_perturbed[:, 2] = target[:, 2] + 1.0
    target[:, 7] = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
    prediction_perturbed[:, 7] = target[:, 7] + 1.5
    target[:, 19] = np.array([0.0, 100.0, 0.0, 100.0], dtype=float)
    prediction_perturbed[:, 19] = np.array([5.0, 95.0, 10.0, 105.0], dtype=float)

    clean_input = np.tile(np.arange(6, dtype=float).reshape(-1, 1), (1, n_channels))
    perturbed_input = clean_input.copy()
    perturbed_input[:, 2] += 1.0
    perturbed_input[:, 7] += 2.0
    feature_names = [f"ch_{idx}" for idx in range(n_channels)]

    fig = plot_forecast_extreme(
        time_index=np.arange(7, 11, dtype=float),
        target=target,
        prediction_perturbed=prediction_perturbed,
        prediction_clean=prediction_clean,
        clean_input=clean_input,
        perturbed_input=perturbed_input,
        input_time_index=np.arange(1, 7, dtype=float),
        input_feature_names=feature_names,
        target_feature_names=feature_names,
        affected_feature_names=["ch_7", "ch_2"],
        scenario="noise",
        severity=0.5,
        sample_score=1.2345,
        score_metric="MSE",
    )

    target_traces = [trace for trace in fig.data if trace.legendgroup == "target"]
    perturbed_prediction_traces = [
        trace for trace in fig.data if trace.legendgroup == "prediction_perturbed"
    ]
    clean_prediction_traces = [
        trace for trace in fig.data if trace.legendgroup == "prediction_clean"
    ]
    input_target_traces = [
        trace for trace in fig.data if trace.legendgroup == "inputs_target"
    ]
    perturbed_traces = [trace for trace in fig.data if trace.legendgroup == "perturbed"]

    assert len(target_traces) == 12
    assert len(perturbed_prediction_traces) == 12
    assert len(clean_prediction_traces) == 12
    assert len(input_target_traces) == 12
    assert len(perturbed_traces) == 2
    assert "Showing 12 of 20 target channels" in str(fig.layout.title.text)
    assert "MSE=1.2345" in str(fig.layout.title.text)


def test_plot_forecast_extreme_reserves_space_for_traffic_multiline_title():
    n_channels = 862
    target = np.tile(np.arange(4, dtype=float).reshape(-1, 1), (1, n_channels))
    prediction_perturbed = target + 0.5
    prediction_clean = target + 0.25
    clean_input = np.tile(np.arange(6, dtype=float).reshape(-1, 1), (1, n_channels))
    perturbed_input = clean_input.copy()
    perturbed_input[:, :12] += 1.0
    feature_names = [str(idx) for idx in range(n_channels)]

    fig = plot_forecast_extreme(
        time_index=np.arange(7, 11, dtype=float),
        target=target,
        prediction_perturbed=prediction_perturbed,
        prediction_clean=prediction_clean,
        clean_input=clean_input,
        perturbed_input=perturbed_input,
        input_time_index=np.arange(1, 7, dtype=float),
        input_feature_names=feature_names,
        target_feature_names=feature_names,
        affected_feature_names=feature_names[:12],
        title=(
            "traffic - PatchTST - baseline<br><sup>Scenario: missing_data, "
            "Severity: 0.491, clean MSE=0.2597, pert MSE=1.2422</sup>"
        ),
        scenario="missing_data",
        severity=0.491,
        sample_score=1.2422,
        score_metric="MSE",
    )

    assert "Showing 12 of 862 target channels" in str(fig.layout.title.text)
    assert fig.layout.margin.t >= 100
    assert fig.layout.height > 400
    assert fig.layout.legend.yanchor == "top"
    assert fig.layout.legend.xanchor == "left"


def test_render_forecast_trace_csv_replays_plot_without_mlflow(tmp_path):
    trace_path = tmp_path / "trace.csv"
    metadata = {
        "dataset": "ETTh1",
        "arch": "GRU",
        "robustness_method": "baseline",
        "scenario": "noise",
        "pert_idx": 0,
        "sample_id": 0,
        "source_sample_idx": 0,
        "severity": 0.5,
        "score_metric": "MSE",
        "clean_sample_score": 0.1,
        "perturbed_sample_score": 0.2,
    }
    rows = []
    for time_step in range(2):
        rows.append(
            {
                **metadata,
                "segment": "input",
                "time_index": float(time_step),
                "time_step": time_step,
                "feature": "sensor",
                "affected_feature": True,
                "clean_input": float(time_step),
                "perturbed_input": float(time_step) + 0.1,
                "ground_truth": None,
                "clean_prediction": None,
                "perturbed_prediction": None,
            }
        )
    for time_step in range(2):
        rows.append(
            {
                **metadata,
                "segment": "forecast",
                "time_index": float(time_step + 2),
                "time_step": time_step,
                "feature": "sensor",
                "affected_feature": True,
                "clean_input": None,
                "perturbed_input": None,
                "ground_truth": float(time_step),
                "clean_prediction": float(time_step) + 0.05,
                "perturbed_prediction": float(time_step) + 0.1,
            }
        )
    pd.DataFrame(rows).to_csv(trace_path, index=False)

    fig = load_forecast_trace_figure(trace_path)

    assert "clean MSE=0.1000" in str(fig.layout.title.text)
    assert "pert MSE=0.2000" in str(fig.layout.title.text)
    assert {str(trace.name) for trace in fig.data} >= {
        "sensor",
        "sensor (Pred Clean)",
        "sensor (Pred Perturbed)",
    }

    output_dir = tmp_path / "rendered"
    rendered = render_traces([trace_path], output_dir=output_dir, file_format="html")

    assert rendered == [
        {
            "trace_csv": trace_path.as_posix(),
            "output_path": (output_dir / "trace.html").as_posix(),
            "output_html": (output_dir / "trace.html").as_posix(),
        }
    ]
    assert (output_dir / "trace.html").is_file()


def _write_minimal_forecast_trace_csv(
    path,
    *,
    sample_id: int = 0,
    robustness_method: str | None = "baseline",
) -> None:
    metadata = {
        "dataset": "ETTh1",
        "arch": "GRU",
        "scenario": "noise",
        "pert_idx": 0,
        "sample_id": sample_id,
        "source_sample_idx": sample_id,
        "severity": 0.5,
        "score_metric": "MSE",
        "clean_sample_score": 0.1,
        "perturbed_sample_score": 0.2,
    }
    if robustness_method is not None:
        metadata["robustness_method"] = robustness_method
    rows = []
    for time_step in range(2):
        rows.append(
            {
                **metadata,
                "segment": "input",
                "time_index": float(time_step),
                "time_step": time_step,
                "feature": "sensor",
                "affected_feature": True,
                "clean_input": float(time_step),
                "perturbed_input": float(time_step) + 0.1,
                "ground_truth": None,
                "clean_prediction": None,
                "perturbed_prediction": None,
            }
        )
    for time_step in range(2):
        rows.append(
            {
                **metadata,
                "segment": "forecast",
                "time_index": float(time_step + 2),
                "time_step": time_step,
                "feature": "sensor",
                "affected_feature": True,
                "clean_input": None,
                "perturbed_input": None,
                "ground_truth": float(time_step),
                "clean_prediction": float(time_step) + 0.05,
                "perturbed_prediction": float(time_step) + 0.1,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_render_forecast_trace_csv_requires_method_unless_explicitly_supplied(tmp_path):
    trace_path = tmp_path / "legacy_trace.csv"
    _write_minimal_forecast_trace_csv(trace_path, robustness_method=None)

    with pytest.raises(ValueError, match="robustness_method"):
        load_forecast_trace_figure(trace_path)

    record = load_forecast_trace_record(
        trace_path,
        missing_robustness_method="baseline",
    )

    assert "GRU - baseline" in record["figure"].layout.title.text


def test_plot_forecast_extreme_rejects_shape_mismatch():
    target = np.array([[1.0, 2.0], [3.0, 4.0]])
    prediction_perturbed = np.array([[1.0], [3.0]])

    with pytest.raises(ValueError, match="prediction_perturbed to share the same shape"):
        plot_forecast_extreme(
            time_index=np.array([1.0, 2.0]),
            target=target,
            prediction_perturbed=prediction_perturbed,
            scenario="noise",
            severity=0.5,
        )


def test_plot_forecast_extreme_single_target_renders_clean_and_perturbed_predictions():
    target = np.array([[1.0], [2.0], [3.0]])
    prediction_clean = target + 0.25
    prediction_perturbed = target + 0.5

    fig = plot_forecast_extreme(
        time_index=np.arange(1, 4, dtype=float),
        target=target,
        prediction_clean=prediction_clean,
        prediction_perturbed=prediction_perturbed,
        target_feature_names=["sensor"],
        scenario="drift",
        severity=0.3,
        sample_score=0.25,
        score_metric="MSE",
    )

    trace_names = {str(trace.name) for trace in fig.data}
    assert "sensor" in trace_names
    assert "sensor (Pred Clean)" in trace_names
    assert "sensor (Pred Perturbed)" in trace_names


def test_plot_forecast_extreme_omits_input_when_mapping_unavailable():
    n_channels = 20
    target = np.tile(np.arange(4, dtype=float).reshape(-1, 1), (1, n_channels))
    prediction_perturbed = target + 0.5
    clean_input = np.ones((6, 5), dtype=float)
    input_names = [f"in_{i}" for i in range(5)]
    target_names = [f"tgt_{i}" for i in range(n_channels)]

    fig = plot_forecast_extreme(
        time_index=np.arange(7, 11, dtype=float),
        target=target,
        prediction_perturbed=prediction_perturbed,
        clean_input=clean_input,
        input_time_index=np.arange(1, 7, dtype=float),
        input_feature_names=input_names,
        target_feature_names=target_names,
        scenario="noise",
        severity=0.5,
    )

    input_traces = [
        trace for trace in fig.data
        if trace.legendgroup in ("inputs", "inputs_target")
    ]
    assert len(input_traces) == 0
    assert "Input history omitted" in str(fig.layout.title.text)


def test_plot_pareto_accepts_focused_architecture_dataframe_shape():
    df = pd.DataFrame(
        [
            {"plot_label": "TSMixer", "architecture_family": "Mixer", "MSE_test": 1.0, "R_cvar": 0.40},
            {"plot_label": "GRU", "architecture_family": "Recurrent", "MSE_test": 0.95, "R_cvar": 0.42},
            {"plot_label": "PatchTST", "architecture_family": "Attention", "MSE_test": 0.90, "R_cvar": 0.50},
            {"plot_label": "ModernTCN", "architecture_family": "Convolution", "MSE_test": 0.88, "R_cvar": 0.52},
        ]
    )

    fig = plot_pareto(
        df,
        perf_col="MSE_test",
        robust_col="R_cvar",
        x_semantics=_perf_semantics("MSE (Test)"),
        y_semantics=_score_semantics("CVaR Robustness"),
        model_col="plot_label",
        arch_col="architecture_family",
        perf_lower_is_better=True,
        flip_perf_axis=True,
        title="Focused Architecture Pareto",
    )

    trace_names = {str(trace.name) for trace in fig.data}
    assert "Pareto frontier" in trace_names
    assert {"Mixer", "Recurrent", "Attention", "Convolution"}.issubset(trace_names)


def test_plot_pareto_expands_robustness_axis_for_scores_above_one():
    df = pd.DataFrame(
        [
            {"plot_label": "TSMixer", "architecture_family": "Mixer", "MSE_test": 1.0, "R_cvar": 1.2},
            {"plot_label": "GRU", "architecture_family": "Recurrent", "MSE_test": 0.95, "R_cvar": 1.5},
        ]
    )

    fig = plot_pareto(
        df,
        perf_col="MSE_test",
        robust_col="R_cvar",
        x_semantics=_perf_semantics("MSE (Test)"),
        y_semantics=_score_semantics("CVaR Robustness"),
        model_col="plot_label",
        arch_col="architecture_family",
        perf_lower_is_better=True,
        flip_perf_axis=True,
        title="Focused Architecture Pareto",
    )

    assert float(fig.layout.yaxis.range[1]) > 1.5


def test_plot_improvement_deltas_heatmap_supports_mixed_color_semantics():
    deltas_df = pd.DataFrame(
        [
            {
                "robustness_method": "adaptive_robust_loss",
                "backbone_architecture": "TSMixer",
                "delta_MSE_test": 0.2,
                "delta_R_worst_scenario_onebin": 0.1,
            }
        ]
    )

    fig = plot_improvement_deltas_heatmap(
        deltas_df,
        metric_cols=["delta_MSE_test", "delta_R_worst_scenario_onebin"],
        metric_semantics={
            "delta_MSE_test": _delta_semantics("Δ MSE (Test)", direction="minimize"),
            "delta_R_worst_scenario_onebin": _delta_semantics(
                "Δ Worst-Scenario Robustness",
                direction="maximize",
            ),
        },
        colorbar_label="Utility-aligned Δ",
    )

    heatmap = fig.data[0]
    assert heatmap.z[0][0] == pytest.approx(-0.2)
    assert heatmap.z[0][1] == pytest.approx(0.1)
    assert heatmap.text[0][0] == "0.200"
    assert heatmap.text[0][1] == "0.100"
    assert fig.layout.coloraxis.colorbar.title.text == "Utility-aligned Δ"


def test_plot_improvement_deltas_heatmap_aligns_lower_is_better_delta_colors():
    deltas_df = pd.DataFrame(
        [
            {
                "robustness_method": "adaptive_robust_loss",
                "backbone_architecture": "TSMixer",
                "delta_D_w": -0.2,
                "delta_err_pert_ws": 0.1,
            }
        ]
    )

    fig = plot_improvement_deltas_heatmap(
        deltas_df,
        metric_cols=["delta_D_w", "delta_err_pert_ws"],
        metric_semantics={
            "delta_D_w": _delta_semantics(
                "Δ Worst-Scenario Degradation",
                direction="minimize",
            ),
            "delta_err_pert_ws": _delta_semantics(
                "Δ MSE_w",
                direction="minimize",
            ),
        },
    )

    heatmap = fig.data[0]
    assert heatmap.z[0][0] == pytest.approx(0.2)
    assert heatmap.z[0][1] == pytest.approx(-0.1)
    assert heatmap.text[0][0] == "-0.200"
    assert heatmap.text[0][1] == "0.100"


def test_plot_improvement_deltas_heatmap_raises_on_missing_metric_values():
    deltas_df = pd.DataFrame(
        [
            {
                "robustness_method": "adaptive_robust_loss",
                "backbone_architecture": "TSMixer",
                "delta_MSE_test": float("nan"),
                "delta_R_worst_scenario_onebin": 0.1,
            }
        ]
    )

    with pytest.raises(ValueError, match="missing metric values"):
        plot_improvement_deltas_heatmap(
            deltas_df,
            metric_cols=["delta_MSE_test", "delta_R_worst_scenario_onebin"],
            metric_semantics={
                "delta_MSE_test": _delta_semantics("Δ MSE (Test)", direction="minimize"),
                "delta_R_worst_scenario_onebin": _delta_semantics(
                    "Δ Worst-Scenario Robustness",
                    direction="maximize",
                ),
            },
        )


def test_plot_improvement_deltas_heatmap_raises_on_missing_metric_semantics():
    deltas_df = pd.DataFrame(
        [
            {
                "robustness_method": "adaptive_robust_loss",
                "backbone_architecture": "TSMixer",
                "delta_MSE_test": 0.2,
                "delta_R_worst_scenario_onebin": 0.1,
            }
        ]
    )

    with pytest.raises(ValueError, match="missing required keys"):
        plot_improvement_deltas_heatmap(
            deltas_df,
            metric_cols=["delta_MSE_test", "delta_R_worst_scenario_onebin"],
            metric_semantics={
                "delta_MSE_test": _delta_semantics("Δ MSE (Test)", direction="minimize"),
            },
        )


def test_plot_method_delta_pair_subplots_renders_two_heatmaps_with_expected_orientation():
    deltas_df = pd.DataFrame(
        [
            {
                "robustness_method": "adaptive_robust_loss",
                "backbone_architecture": "TSMixer",
                "pipeline_id": "adaptive_robust_loss_delta1",
                "delta_MSE_test": -0.2,
                "delta_R_worst_scenario": 0.05,
            },
            {
                "robustness_method": "adaptive_robust_loss",
                "backbone_architecture": "GRU",
                "pipeline_id": "adaptive_robust_loss_delta1",
                "delta_MSE_test": -0.1,
                "delta_R_worst_scenario": 0.02,
            },
            {
                "robustness_method": "randomized_smoothing",
                "backbone_architecture": "TSMixer",
                "pipeline_id": "randomized_smoothing_mask0_2",
                "delta_MSE_test": 0.1,
                "delta_R_worst_scenario": 0.03,
            },
            {
                "robustness_method": "randomized_smoothing",
                "backbone_architecture": "GRU",
                "pipeline_id": "randomized_smoothing_mask0_2",
                "delta_MSE_test": 0.2,
                "delta_R_worst_scenario": 0.04,
            },
        ]
    )

    fig = plot_method_delta_pair_subplots(
        deltas_df,
        perf_delta_col="delta_MSE_test",
        robust_delta_col="delta_R_worst_scenario",
        perf_semantics=_delta_semantics("Δ MSE_test", direction="minimize"),
        robust_semantics=_delta_semantics(
            "Δ Worst-Scenario Robustness",
            direction="maximize",
        ),
        method_col="robustness_method",
        baseline_col="backbone_architecture",
        title="ETTh1: Method Deltas vs Baseline (Worst-Scenario Robustness)",
    )

    subplot_titles = {
        annotation.text for annotation in fig.layout.annotations if annotation.text is not None
    }
    assert "Δ MSE_test" in subplot_titles
    assert "Δ Worst-Scenario Robustness" in subplot_titles
    assert len(subplot_titles) == 2

    assert len(fig.data) == 2
    perf_trace = fig.data[0]
    robust_trace = fig.data[1]
    assert perf_trace.type == "heatmap"
    assert robust_trace.type == "heatmap"
    assert tuple(perf_trace.x) == ("adaptive_robust_loss", "randomized_smoothing")
    assert tuple(perf_trace.y) == ("GRU", "TSMixer")
    assert tuple(robust_trace.x) == ("adaptive_robust_loss", "randomized_smoothing")
    assert tuple(robust_trace.y) == ("GRU", "TSMixer")
    perf_z = np.asarray(perf_trace.z, dtype=float)
    robust_z = np.asarray(robust_trace.z, dtype=float)
    assert np.allclose(perf_z, np.array([[-0.1, 0.2], [-0.2, 0.1]], dtype=float))
    assert np.allclose(robust_z, np.array([[0.02, 0.04], [0.05, 0.03]], dtype=float))
    assert perf_trace.text[0][0] == "-0.100"
    assert perf_trace.text[0][1] == "0.200"
    assert robust_trace.text[0][0] == "0.020"
    assert robust_trace.text[1][0] == "0.050"


def test_plot_method_delta_pair_subplots_raises_on_missing_backbone_column():
    deltas_df = pd.DataFrame(
        [
            {
                "robustness_method": "adaptive_robust_loss",
                "delta_MSE_test": -0.2,
                "delta_R_worst_scenario": 0.05,
            }
        ]
    )

    with pytest.raises(ValueError, match="missing required columns"):
        plot_method_delta_pair_subplots(
            deltas_df,
            perf_delta_col="delta_MSE_test",
            robust_delta_col="delta_R_worst_scenario",
            perf_semantics=_delta_semantics("Δ MSE_test", direction="minimize"),
            robust_semantics=_delta_semantics(
                "Δ Worst-Scenario Robustness",
                direction="maximize",
            ),
            method_col="robustness_method",
            baseline_col="backbone_architecture",
        )


def test_plot_method_delta_pair_subplots_raises_on_duplicate_baseline_method_rows():
    deltas_df = pd.DataFrame(
        [
            {
                "robustness_method": "adaptive_robust_loss",
                "backbone_architecture": "TSMixer",
                "pipeline_id": "adaptive_robust_loss_delta1",
                "delta_MSE_test": -0.2,
                "delta_R_worst_scenario": 0.05,
            },
            {
                "robustness_method": "adaptive_robust_loss",
                "backbone_architecture": "TSMixer",
                "pipeline_id": "adaptive_robust_loss_delta2",
                "delta_MSE_test": -0.1,
                "delta_R_worst_scenario": 0.04,
            },
        ]
    )

    with pytest.raises(ValueError, match="duplicate \\(baseline, method\\) rows"):
        plot_method_delta_pair_subplots(
            deltas_df,
            perf_delta_col="delta_MSE_test",
            robust_delta_col="delta_R_worst_scenario",
            perf_semantics=_delta_semantics("Δ MSE_test", direction="minimize"),
            robust_semantics=_delta_semantics(
                "Δ Worst-Scenario Robustness",
                direction="maximize",
            ),
            method_col="robustness_method",
            baseline_col="backbone_architecture",
        )


def test_plot_method_delta_pair_subplots_raises_on_missing_delta_values():
    deltas_df = pd.DataFrame(
        [
            {
                "robustness_method": "adaptive_robust_loss",
                "backbone_architecture": "TSMixer",
                "pipeline_id": "adaptive_robust_loss_delta1",
                "delta_MSE_test": float("nan"),
                "delta_R_worst_scenario": 0.05,
            }
        ]
    )

    with pytest.raises(ValueError, match="missing delta values"):
        plot_method_delta_pair_subplots(
            deltas_df,
            perf_delta_col="delta_MSE_test",
            robust_delta_col="delta_R_worst_scenario",
            perf_semantics=_delta_semantics("Δ MSE_test", direction="minimize"),
            robust_semantics=_delta_semantics(
                "Δ Worst-Scenario Robustness",
                direction="maximize",
            ),
            method_col="robustness_method",
            baseline_col="backbone_architecture",
        )


def test_plot_method_delta_pair_subplots_utility_aligns_lower_is_better_metric_colors():
    deltas_df = pd.DataFrame(
        [
            {
                "robustness_method": "adaptive_robust_loss",
                "backbone_architecture": "TSMixer",
                "pipeline_id": "adaptive_robust_loss_delta1",
                "delta_MSE_test": -0.05,
                "delta_CE_cell_ws_meansev_snaive": -0.04,
            },
            {
                "robustness_method": "randomized_smoothing",
                "backbone_architecture": "GRU",
                "pipeline_id": "randomized_smoothing_mask0_2",
                "delta_MSE_test": 0.10,
                "delta_CE_cell_ws_meansev_snaive": 0.03,
            },
        ]
    )

    fig = plot_method_delta_pair_subplots(
        deltas_df,
        perf_delta_col="delta_MSE_test",
        robust_delta_col="delta_CE_cell_ws_meansev_snaive",
        perf_semantics=_delta_semantics("Δ MSE_test", direction="minimize"),
        robust_semantics=_delta_semantics(
            "Δ Worst-Scenario Cell CE",
            direction="minimize",
        ),
        robust_higher_is_better=False,
        method_col="robustness_method",
        baseline_col="backbone_architecture",
    )

    subplot_titles = {
        annotation.text for annotation in fig.layout.annotations if annotation.text is not None
    }
    assert "Δ Worst-Scenario Cell CE" in subplot_titles
    robust_trace = fig.data[1]
    assert np.allclose(
        np.asarray(robust_trace.z, dtype=float),
        np.array([[np.nan, 0.03], [-0.04, np.nan]], dtype=float),
        equal_nan=True,
    )
    assert robust_trace.text[0][1] == "0.030"
    assert robust_trace.text[1][0] == "-0.040"


def test_plot_per_method_delta_scatter_labels_lower_is_better_metric_direction():
    deltas_df = pd.DataFrame(
        [
            {
                "model_architecture": "TSMixer",
                "robustness_method": "adaptive_robust_loss",
                "pipeline_id": "adaptive_robust_loss_delta1",
                "delta_MSE_test": -0.2,
                "delta_CE_cell_ws_meansev_snaive": -0.05,
            }
        ]
    )

    fig = plot_per_method_delta_scatter(
        deltas_df,
        delta_perf_col="delta_MSE_test",
        delta_robust_col="delta_CE_cell_ws_meansev_snaive",
        perf_semantics=_delta_semantics("Δ MSE_test", direction="minimize"),
        robust_semantics=_delta_semantics(
            "Δ Worst-Scenario Cell CE",
            direction="minimize",
        ),
        method_name="dataset_a: Winner Method Deltas",
        robust_higher_is_better=False,
        arch_col="model_architecture",
        normalize_perf=False,
    )

    assert fig.layout.yaxis.title.text == "Δ Worst-Scenario Cell CE"


def test_plot_error_distribution_overview_uses_raw_rows_without_weight_column():
    df = pd.DataFrame(
        [
            {
                "model_variant": "baseline|TSMixer",
                "error_kind": "err_clean",
                "error_value": 1.0,
            },
            {
                "model_variant": "baseline|TSMixer",
                "error_kind": "err_clean",
                "error_value": 3.0,
            },
            {
                "model_variant": "baseline|TSMixer",
                "error_kind": "err_pert",
                "error_value": 2.0,
            },
            {
                "model_variant": "baseline|TSMixer",
                "error_kind": "err_pert",
                "error_value": 8.0,
            },
        ]
    )

    fig = plot_error_distribution_overview(df, dataset="dataset_a")

    assert fig.layout.title.text == "dataset_a: Error Distributions by Model Variant"
    assert {trace.name for trace in fig.data} == {"err_clean", "err_pert"}
    clean_trace = next(trace for trace in fig.data if trace.name == "err_clean")
    pert_trace = next(trace for trace in fig.data if trace.name == "err_pert")
    assert np.sort(np.asarray(clean_trace.y, dtype=float)).tolist() == [1.0, 3.0]
    assert np.sort(np.asarray(pert_trace.y, dtype=float)).tolist() == [2.0, 8.0]


def test_plot_error_distribution_overview_requires_both_error_kinds_per_facet():
    df = pd.DataFrame(
        [
            {
                "model_variant": "baseline|TSMixer",
                "error_kind": "err_clean",
                "error_value": 1.0,
            },
            {
                "model_variant": "adaptive_robust_loss|TSMixer",
                "error_kind": "err_clean",
                "error_value": 0.9,
            },
            {
                "model_variant": "adaptive_robust_loss|TSMixer",
                "error_kind": "err_pert",
                "error_value": 1.1,
            },
        ]
    )

    with pytest.raises(ValueError, match="requires both err_clean and err_pert"):
        plot_error_distribution_overview(df, dataset="dataset_a")


def test_score_axis_range_defaults_to_zero_one_baseline():
    lo, hi = _score_axis_range([0.3, 0.6])
    assert lo == 0.0
    assert hi > 1.0  # padded above 1


def test_score_axis_range_expands_for_scores_above_one():
    lo, hi = _score_axis_range([0.5, 2.0])
    assert lo == 0.0
    assert hi > 2.0  # padded above the max


def test_score_axis_range_expands_for_negative_scores():
    lo, hi = _score_axis_range([-0.2, 0.8])
    assert lo < -0.2
    assert hi > 1.0


def test_score_axis_range_empty_returns_zero_one():
    assert _score_axis_range([]) == (0.0, 1.0)


def test_plot_perturbation_curves_handles_scores_above_one():
    df = pd.DataFrame(
        [
            {"pert_idx": 0, "severity": 0.25, "R_cell": 2.0},
            {"pert_idx": 0, "severity": 0.75, "R_cell": 0.5},
            {"pert_idx": 1, "severity": 0.25, "R_cell": 1.5},
            {"pert_idx": 1, "severity": 0.75, "R_cell": 0.3},
        ]
    )
    fig = plot_perturbation_curves(
        df,
        "TestModel",
        value_col="R_cell",
        y_semantics=_score_semantics("Robustness Score"),
    )
    y_range = fig.layout.yaxis.range
    assert y_range[0] == 0.0
    assert y_range[1] > 2.0
