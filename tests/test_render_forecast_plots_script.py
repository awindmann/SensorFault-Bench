from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.render_forecast_plots import (
    _assert_no_output_path_collisions,
    _export_sample_trace_csv,
    _load_forecast_extremes_samples,
    _load_scenario_samples,
    _pivot_segment,
    _parse_sample_spec,
    parse_cli_args,
)


def test_render_forecast_plots_cli_parses_direct_sample_with_dataset(tmp_path):
    args = parse_cli_args(
        [
            "from-runs",
            "--dataset",
            "ETTh1",
            "--sample",
            "GRU:noise:0:7:11:0.500",
            "--run-id",
            "run_123",
            "--output-dir",
            str(tmp_path),
        ]
    )
    sample = _parse_sample_spec(args.sample[0], index=0, dataset=args.dataset)

    assert args.dataset == "ETTh1"
    assert args.run_id == ["run_123"]
    assert sample == {
        "dataset": "ETTh1",
        "arch": "GRU",
        "scenario": "noise",
        "pert_idx": 0,
        "sample_id": 7,
        "source_sample_idx": 11,
        "severity": 0.5,
        "order": 0,
    }
    assert args.command == "from-runs"
    assert args.export_traces is True


def test_render_forecast_plots_cli_rejects_removed_trace_replay_grid(tmp_path):
    trace_path = tmp_path / "trace.csv"
    with pytest.raises(SystemExit):
        parse_cli_args(
            [
                "from-traces",
                "--trace-csv",
                str(trace_path),
                "--output-dir",
                str(tmp_path / "plots"),
                "--grid",
                "1x2",
            ]
        )


def test_render_forecast_plots_cli_requires_trace_source(tmp_path):
    with pytest.raises(SystemExit):
        parse_cli_args(
            [
                "from-traces",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_render_forecast_plots_cli_requires_dataset_for_direct_sample(tmp_path):
    with pytest.raises(SystemExit):
        parse_cli_args(
            [
                "from-runs",
                "--sample",
                "GRU:noise:0:7:11:0.500",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_render_forecast_plots_loads_filtered_scenario_samples(tmp_path):
    csv_path = tmp_path / "scenario_samples.csv"
    pd.DataFrame(
        [
            {
                "dataset": "ETTh1",
                "run_id": "run_a",
                "model_architecture": "GRU",
                "backbone_architecture": "GRU",
                "robustness_method": "baseline",
                "pipeline_method": "baseline",
                "pipeline_id": "baseline",
                "sample_id": 3,
                "source_sample_idx": 13,
                "pert_idx": 1,
                "scenario": "noise",
                "severity": 0.25,
                "err_pert": 1.5,
            },
            {
                "dataset": "traffic",
                "run_id": "run_b",
                "model_architecture": "PatchTST",
                "backbone_architecture": "PatchTST",
                "robustness_method": "baseline",
                "pipeline_method": "baseline",
                "pipeline_id": "baseline",
                "sample_id": 5,
                "source_sample_idx": 15,
                "pert_idx": 2,
                "scenario": "drift",
                "severity": 0.5,
                "err_pert": 2.5,
            },
        ]
    ).to_csv(csv_path, index=False)
    args = parse_cli_args(
        [
            "from-runs",
            "--dataset",
            "ETTh1",
            "--model",
            "GRU",
            "--method",
            "baseline",
            "--scenario-samples-csv",
            str(csv_path),
            "--output-dir",
            str(tmp_path),
        ]
    )

    samples = _load_scenario_samples(csv_path, cli_args=args, offset=0)

    assert samples == [
        {
            "dataset": "ETTh1",
            "arch": "GRU",
            "scenario": "noise",
            "pert_idx": 1,
            "sample_id": 3,
            "source_sample_idx": 13,
            "severity": 0.25,
            "order": 0,
            "sample_score": 1.5,
            "run_id": "run_a",
            "pipeline_id": "baseline",
            "robustness_method": "baseline",
            "pipeline_method": "baseline",
        }
    ]


def test_render_forecast_plots_rejects_forecast_extreme_metric_mismatch(tmp_path):
    csv_path = tmp_path / "forecast_extremes.csv"
    pd.DataFrame(
        [
            {
                "dataset": "ETTh1",
                "run_id": "run_a",
                "model_architecture": "GRU",
                "backbone_architecture": "GRU",
                "robustness_method": "baseline",
                "pipeline_method": "baseline",
                "pipeline_kind": "train",
                "pipeline_id": "baseline",
                "sample_id": 3,
                "source_sample_idx": 13,
                "pert_idx": 1,
                "scenario": "noise",
                "severity": 0.25,
                "sample_score": 1.5,
                "score_metric": "MAE",
            },
        ]
    ).to_csv(csv_path, index=False)
    args = parse_cli_args(
        [
            "from-runs",
            "--forecast-extremes-csv",
            str(csv_path),
            "--test-metric",
            "MSE",
            "--output-dir",
            str(tmp_path),
        ]
    )

    with pytest.raises(ValueError, match="score_metric"):
        _load_forecast_extremes_samples(csv_path, cli_args=args, offset=0)


def test_render_forecast_plots_requires_forecast_extreme_method_metadata(tmp_path):
    csv_path = tmp_path / "forecast_extremes.csv"
    pd.DataFrame(
        [
            {
                "dataset": "ETTh1",
                "run_id": "run_a",
                "model_architecture": "GRU",
                "backbone_architecture": "GRU",
                "sample_id": 3,
                "source_sample_idx": 13,
                "pert_idx": 1,
                "scenario": "noise",
                "severity": 0.25,
                "sample_score": 1.5,
                "score_metric": "MSE",
            },
        ]
    ).to_csv(csv_path, index=False)
    args = parse_cli_args(
        [
            "from-runs",
            "--forecast-extremes-csv",
            str(csv_path),
            "--test-metric",
            "MSE",
            "--output-dir",
            str(tmp_path),
        ]
    )

    with pytest.raises(ValueError, match="robustness_method"):
        _load_forecast_extremes_samples(csv_path, cli_args=args, offset=0)


def test_render_forecast_plots_detects_output_collisions(tmp_path):
    samples = [
        {
            "dataset": "ETTh1",
            "arch": "GRU",
            "scenario": "noise",
            "sample_id": 3,
            "severity": 0.25,
            "order": 0,
            "robustness_method": "baseline",
        },
        {
            "dataset": "ETTh1",
            "arch": "GRU",
            "scenario": "noise",
            "sample_id": 3,
            "severity": 0.25,
            "order": 1,
            "robustness_method": "randomized_smoothing",
        },
    ]

    with pytest.raises(ValueError, match="output collision"):
        _assert_no_output_path_collisions(
            samples,
            output_dir=tmp_path,
            file_format="pdf",
            export_traces=True,
        )


def _trace_payload() -> dict[str, object]:
    return {
        "sample_id": 7,
        "source_sample_idx": 11,
        "pert_idx": 2,
        "scenario": "noise",
        "severity": 0.125,
        "sample_score": 2.0,
        "clean_sample_score": 0.5,
        "perturbed_sample_score": 2.0,
        "clean_input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        "perturbed_input": np.array([[1.5, 2.0], [3.0, 4.5]], dtype=float),
        "target": np.array([[10.0], [11.0]], dtype=float),
        "prediction_clean": np.array([[10.1], [10.9]], dtype=float),
        "prediction_perturbed": np.array([[12.0], [13.0]], dtype=float),
        "input_feature_names": ("sensor_a", "sensor_b"),
        "target_feature_names": ("target",),
        "affected_feature_names": ("sensor_a",),
    }


def test_render_forecast_plots_exports_trace_csv_with_method_metadata(tmp_path):
    output_path = _export_sample_trace_csv(
        dataset="ETTh1",
        sample={"arch": "GRU", "robustness_method": "baseline"},
        payload=_trace_payload(),
        score_metric="MSE",
        input_time_index=np.array([1.0, 2.0], dtype=float),
        output_time_index=np.array([3.0, 4.0], dtype=float),
        output_dir=tmp_path,
    )

    assert output_path == tmp_path / "traces" / "noise" / "GRU_noise_sid7_sev0.125.csv"
    df = pd.read_csv(output_path)

    assert list(df.columns)[:4] == [
        "dataset",
        "arch",
        "robustness_method",
        "scenario",
    ]
    assert set(df["robustness_method"]) == {"baseline"}
    assert len(df) == 6


def test_render_forecast_plots_pivots_numeric_trace_feature_labels(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "segment": "input",
                "time_step": 0,
                "time_index": 1.0,
                "feature": 19,
                "clean_input": 0.1,
            },
            {
                "segment": "input",
                "time_step": 0,
                "time_index": 1.0,
                "feature": 188,
                "clean_input": 0.2,
            },
            {
                "segment": "input",
                "time_step": 1,
                "time_index": 2.0,
                "feature": 19,
                "clean_input": 0.3,
            },
            {
                "segment": "input",
                "time_step": 1,
                "time_index": 2.0,
                "feature": 188,
                "clean_input": 0.4,
            },
        ]
    )

    time_index, feature_names, values = _pivot_segment(
        frame,
        segment="input",
        value_column="clean_input",
        path=tmp_path / "numeric_features.csv",
    )

    assert time_index.tolist() == [1.0, 2.0]
    assert feature_names == ["19", "188"]
    assert values.tolist() == [[0.1, 0.2], [0.3, 0.4]]


def test_render_forecast_plots_rejects_inconsistent_sample_overrides():
    from scripts.render_forecast_plots import _consistent_sample_override

    samples = [
        {"batch_size": 16},
        {"batch_size": 64},
    ]

    with pytest.raises(ValueError, match="inconsistent batch_size"):
        _consistent_sample_override(
            samples,
            "batch_size",
            default=None,
            positive=True,
        )
