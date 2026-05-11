import pandas as pd
import pytest

from visualizations.semantics import (
    PlotSemanticsRecord,
    require_plot_semantics_mapping,
)
from utils.scoring import (
    build_fixed_channel_fraction_metric_key,
    build_degradation_scenario_metric_key,
    extract_required_degradation_scenario_metric_with_ci,
    extract_required_degradation_scenario_metrics,
    extract_required_overall_degradation_metrics,
    require_logged_fixed_channel_fraction_metric_bundle,
    require_logged_degradation_metric_bundle,
    required_fixed_channel_fraction_metric_keys,
    required_degradation_metric_keys,
    score_degradation_artifact_bundle,
    validate_clean_test_samples,
    validate_fixed_channel_fraction_artifact_bundle,
    validate_degradation_artifact_bundle,
    validate_degradation_scenario_samples,
    validate_degradation_scenario_summary,
)


def _logged_degradation_metrics(*, test_metric: str = "MSE") -> dict[str, float]:
    prefix = f"degradation/{test_metric}"
    return {
        f"{test_metric}_test": 0.25,
        f"{prefix}/D_w": 2.0,
        f"{prefix}/D_w_CI_lo": 1.8,
        f"{prefix}/D_w_CI_hi": 2.2,
        f"{prefix}/D_mean": 1.8,
        f"{prefix}/D_mean_CI_lo": 1.6,
        f"{prefix}/D_mean_CI_hi": 2.0,
        f"{prefix}/err_pert_ws": 0.31,
        f"{prefix}/err_pert_ws_CI_lo": 0.28,
        f"{prefix}/err_pert_ws_CI_hi": 0.35,
        f"{prefix}/err_pert_mean": 0.29,
        f"{prefix}/err_pert_mean_CI_lo": 0.26,
        f"{prefix}/err_pert_mean_CI_hi": 0.32,
        f"{prefix}/scenario/0/D": 1.6,
        f"{prefix}/scenario/0/D_CI_lo": 1.4,
        f"{prefix}/scenario/0/D_CI_hi": 1.8,
        f"{prefix}/scenario/0/err_pert": 0.27,
        f"{prefix}/scenario/0/err_pert_CI_lo": 0.24,
        f"{prefix}/scenario/0/err_pert_CI_hi": 0.30,
        f"{prefix}/scenario/1/D": 2.0,
        f"{prefix}/scenario/1/D_CI_lo": 1.8,
        f"{prefix}/scenario/1/D_CI_hi": 2.2,
        f"{prefix}/scenario/1/err_pert": 0.31,
        f"{prefix}/scenario/1/err_pert_CI_lo": 0.28,
        f"{prefix}/scenario/1/err_pert_CI_hi": 0.35,
    }


def test_required_degradation_metric_keys_include_run_and_scenario_metrics():
    keys = required_degradation_metric_keys(
        test_metric="MSE",
        scenario_indices=(1, 0, 1),
    )

    assert "MSE_test" in keys
    assert "degradation/MSE/D_w" in keys
    assert "degradation/MSE/scenario/0/D" in keys
    assert "degradation/MSE/scenario/1/err_pert_CI_hi" in keys


def test_extract_required_overall_degradation_metrics_requires_complete_bundle():
    metrics = {
        "degradation/MSE/D_w": 2.0,
        "degradation/MSE/D_w_CI_lo": 1.8,
        "degradation/MSE/D_w_CI_hi": 2.2,
        "degradation/MSE/D_mean": 1.7,
        "degradation/MSE/D_mean_CI_lo": 1.5,
        "degradation/MSE/D_mean_CI_hi": 1.9,
        "degradation/MSE/err_pert_ws": 0.6,
        "degradation/MSE/err_pert_ws_CI_lo": 0.5,
        "degradation/MSE/err_pert_ws_CI_hi": 0.7,
        "degradation/MSE/err_pert_mean": 0.45,
        "degradation/MSE/err_pert_mean_CI_lo": 0.4,
        "degradation/MSE/err_pert_mean_CI_hi": 0.5,
    }

    extracted = extract_required_overall_degradation_metrics(
        metrics,
        run_id="run_degradation",
        test_metric="MSE",
    )

    assert extracted["D_w"] == pytest.approx(2.0)
    assert extracted["D_mean_CI_lo"] == pytest.approx(1.5)
    assert extracted["err_pert_ws_CI_hi"] == pytest.approx(0.7)
    assert extracted["err_pert_mean"] == pytest.approx(0.45)


def test_extract_required_degradation_scenario_metrics_and_ci_require_integer_indexed_keys():
    metrics = {
        "degradation/MSE/scenario/1/D": 2.4,
        "degradation/MSE/scenario/1/D_CI_lo": 2.1,
        "degradation/MSE/scenario/1/D_CI_hi": 2.7,
        "degradation/MSE/scenario/1/err_pert": 0.72,
        "degradation/MSE/scenario/1/err_pert_CI_lo": 0.68,
        "degradation/MSE/scenario/1/err_pert_CI_hi": 0.77,
    }

    extracted = extract_required_degradation_scenario_metrics(
        metrics,
        run_id="run_degradation",
        test_metric="MSE",
        scenario_idx=1,
    )
    value, ci_lo, ci_hi = extract_required_degradation_scenario_metric_with_ci(
        metrics,
        run_id="run_degradation",
        test_metric="MSE",
        scenario_idx=1,
        metric_name="D",
    )

    assert extracted["err_pert"] == pytest.approx(0.72)
    assert extracted["err_pert_CI_hi"] == pytest.approx(0.77)
    assert value == pytest.approx(2.4)
    assert ci_lo == pytest.approx(2.1)
    assert ci_hi == pytest.approx(2.7)


def test_require_logged_degradation_metric_bundle_rejects_inconsistent_worst_scenario():
    with pytest.raises(ValueError, match="metrics imply worst_scenario='noise'"):
        require_logged_degradation_metric_bundle(
            _logged_degradation_metrics(),
            tags={"degradation/MSE/worst_scenario": "drift"},
            params={"degradation/MSE/worst_scenario": "drift"},
            run_id="run_degradation",
            test_metric="MSE",
            expected_idx_to_name={0: "drift", 1: "noise"},
        )


@pytest.mark.parametrize(
    ("metric_key", "bad_value", "message_fragment"),
    [
        ("degradation/MSE/D_w", 1.9, "inconsistent D_w"),
        ("degradation/MSE/err_pert_ws", 0.29, "inconsistent err_pert_ws"),
    ],
)
def test_require_logged_degradation_metric_bundle_rejects_inconsistent_run_level_aggregates(
    metric_key: str,
    bad_value: float,
    message_fragment: str,
):
    metrics = _logged_degradation_metrics()
    metrics[metric_key] = bad_value

    with pytest.raises(ValueError, match=message_fragment):
        require_logged_degradation_metric_bundle(
            metrics,
            tags={"degradation/MSE/worst_scenario": "noise"},
            params={"degradation/MSE/worst_scenario": "noise"},
            run_id="run_degradation",
            test_metric="MSE",
            expected_idx_to_name={0: "drift", 1: "noise"},
        )


def test_degradation_metric_key_builders_reject_non_integral_scenario_indices():
    with pytest.raises(ValueError, match="requires integer scenario_idx"):
        build_degradation_scenario_metric_key(
            test_metric="MSE",
            scenario_idx=1.9,
            metric_name="D",
        )

    with pytest.raises(ValueError, match="requires integer scenario_idx"):
        required_degradation_metric_keys(
            test_metric="MSE",
            scenario_indices=(0, 1.9),
        )


def test_validate_clean_test_samples_sorts_and_normalizes():
    df = pd.DataFrame(
        [
            {"sample_id": "1", "source_sample_idx": "7", "err_clean": "0.2"},
            {"sample_id": "0", "source_sample_idx": "3", "err_clean": "0.1"},
        ]
    )

    result = validate_clean_test_samples(df, context_name="clean_test_samples.csv")

    assert result["sample_id"].tolist() == [0, 1]
    assert result["source_sample_idx"].tolist() == [3, 7]
    assert result["err_clean"].tolist() == [0.1, 0.2]


def test_validate_clean_test_samples_rejects_near_integer_sample_ids():
    df = pd.DataFrame(
        [
            {
                "sample_id": 1.00000000001,
                "source_sample_idx": 3,
                "err_clean": 0.1,
            }
        ]
    )

    with pytest.raises(ValueError, match="must contain integer values"):
        validate_clean_test_samples(df, context_name="clean_test_samples.csv")


def test_validate_clean_test_samples_requires_zero_based_contiguous_sample_ids():
    df = pd.DataFrame(
        [
            {"sample_id": 5, "source_sample_idx": 3, "err_clean": 0.1},
            {"sample_id": 7, "source_sample_idx": 4, "err_clean": 0.2},
        ]
    )

    with pytest.raises(ValueError, match="contiguous zero-based IDs"):
        validate_clean_test_samples(df, context_name="clean_test_samples.csv")


def test_validate_degradation_scenario_samples_requires_complete_sample_scenario_matrix():
    df = pd.DataFrame(
        [
            {
                "sample_id": 0,
                "source_sample_idx": 3,
                "pert_idx": 0,
                "scenario": "drift",
                "severity": 0.2,
                "err_pert": 0.3,
            },
            {
                "sample_id": 1,
                "source_sample_idx": 7,
                "pert_idx": 0,
                "scenario": "drift",
                "severity": 0.4,
                "err_pert": 0.5,
            },
        ]
    )

    with pytest.raises(ValueError, match="pert_idx values \\[0\\] do not match expected \\[0, 1\\]"):
        validate_degradation_scenario_samples(
            df,
            expected_idx_to_name={0: "drift", 1: "noise"},
            context_name="scenario_samples.csv",
        )


def test_validate_degradation_scenario_samples_rejects_duplicate_expected_scenario_names():
    df = pd.DataFrame(
        [
            {
                "sample_id": 0,
                "source_sample_idx": 3,
                "pert_idx": 0,
                "scenario": "drift",
                "severity": 0.2,
                "err_pert": 0.3,
            },
            {
                "sample_id": 0,
                "source_sample_idx": 3,
                "pert_idx": 1,
                "scenario": "drift",
                "severity": 0.4,
                "err_pert": 0.5,
            },
        ]
    )

    with pytest.raises(ValueError, match="duplicate scenario name"):
        validate_degradation_scenario_samples(
            df,
            expected_idx_to_name={0: "drift", 1: "drift"},
            context_name="scenario_samples.csv",
        )


def test_validate_degradation_scenario_samples_requires_zero_based_contiguous_sample_ids():
    df = pd.DataFrame(
        [
            {
                "sample_id": 5,
                "source_sample_idx": 3,
                "pert_idx": 0,
                "scenario": "drift",
                "severity": 0.2,
                "err_pert": 0.3,
            },
            {
                "sample_id": 5,
                "source_sample_idx": 3,
                "pert_idx": 1,
                "scenario": "noise",
                "severity": 0.4,
                "err_pert": 0.5,
            },
            {
                "sample_id": 7,
                "source_sample_idx": 4,
                "pert_idx": 0,
                "scenario": "drift",
                "severity": 0.6,
                "err_pert": 0.7,
            },
            {
                "sample_id": 7,
                "source_sample_idx": 4,
                "pert_idx": 1,
                "scenario": "noise",
                "severity": 0.8,
                "err_pert": 0.9,
            },
        ]
    )

    with pytest.raises(ValueError, match="contiguous zero-based IDs"):
        validate_degradation_scenario_samples(
            df,
            expected_idx_to_name={0: "drift", 1: "noise"},
            context_name="scenario_samples.csv",
        )


def test_validate_degradation_scenario_summary_requires_expected_order():
    df = pd.DataFrame(
        [
            {
                "pert_idx": 1,
                "scenario": "noise",
                "n_test_samples": 2,
                "err_clean_global": 0.2,
                "err_pert": 0.4,
                "err_pert_CI_lo": 0.3,
                "err_pert_CI_hi": 0.5,
                "D": 2.0,
                "D_CI_lo": 1.5,
                "D_CI_hi": 2.5,
            },
            {
                "pert_idx": 0,
                "scenario": "drift",
                "n_test_samples": 2,
                "err_clean_global": 0.2,
                "err_pert": 0.3,
                "err_pert_CI_lo": 0.2,
                "err_pert_CI_hi": 0.4,
                "D": 1.5,
                "D_CI_lo": 1.0,
                "D_CI_hi": 2.0,
            },
        ]
    )

    with pytest.raises(ValueError, match="row order"):
        validate_degradation_scenario_summary(
            df,
            expected_idx_to_name={0: "drift", 1: "noise"},
            expected_n_test_samples=2,
            context_name="scenario_summary.csv",
        )


def test_validate_degradation_scenario_summary_requires_single_n_test_samples_value():
    df = pd.DataFrame(
        [
            {
                "pert_idx": 0,
                "scenario": "drift",
                "n_test_samples": 2,
                "err_clean_global": 0.2,
                "err_pert": 0.3,
                "err_pert_CI_lo": 0.2,
                "err_pert_CI_hi": 0.4,
                "D": 1.5,
                "D_CI_lo": 1.0,
                "D_CI_hi": 2.0,
            },
            {
                "pert_idx": 1,
                "scenario": "noise",
                "n_test_samples": 3,
                "err_clean_global": 0.2,
                "err_pert": 0.4,
                "err_pert_CI_lo": 0.3,
                "err_pert_CI_hi": 0.5,
                "D": 2.0,
                "D_CI_lo": 1.5,
                "D_CI_hi": 2.5,
            },
        ]
    )

    with pytest.raises(ValueError, match="one canonical n_test_samples value"):
        validate_degradation_scenario_summary(
            df,
            expected_idx_to_name={0: "drift", 1: "noise"},
            context_name="scenario_summary.csv",
        )


def test_validate_degradation_artifact_bundle_rejects_source_idx_mismatch():
    clean_df = pd.DataFrame(
        [
            {"sample_id": 0, "source_sample_idx": 3, "err_clean": 0.1},
            {"sample_id": 1, "source_sample_idx": 7, "err_clean": 0.2},
        ]
    )
    scenario_samples_df = pd.DataFrame(
        [
            {
                "sample_id": 0,
                "source_sample_idx": 4,
                "pert_idx": 0,
                "scenario": "drift",
                "severity": 0.2,
                "err_pert": 0.3,
            },
            {
                "sample_id": 0,
                "source_sample_idx": 4,
                "pert_idx": 1,
                "scenario": "noise",
                "severity": 0.7,
                "err_pert": 0.5,
            },
            {
                "sample_id": 1,
                "source_sample_idx": 7,
                "pert_idx": 0,
                "scenario": "drift",
                "severity": 0.4,
                "err_pert": 0.6,
            },
            {
                "sample_id": 1,
                "source_sample_idx": 7,
                "pert_idx": 1,
                "scenario": "noise",
                "severity": 0.9,
                "err_pert": 0.8,
            },
        ]
    )
    scenario_summary_df = pd.DataFrame(
        [
            {
                "pert_idx": 0,
                "scenario": "drift",
                "n_test_samples": 2,
                "err_clean_global": 0.15,
                "err_pert": 0.45,
                "err_pert_CI_lo": 0.35,
                "err_pert_CI_hi": 0.55,
                "D": 3.0,
                "D_CI_lo": 2.0,
                "D_CI_hi": 4.0,
            },
            {
                "pert_idx": 1,
                "scenario": "noise",
                "n_test_samples": 2,
                "err_clean_global": 0.15,
                "err_pert": 0.65,
                "err_pert_CI_lo": 0.55,
                "err_pert_CI_hi": 0.75,
                "D": 4.0,
                "D_CI_lo": 3.0,
                "D_CI_hi": 5.0,
            },
        ]
    )

    with pytest.raises(ValueError, match="source_sample_idx values do not align"):
        validate_degradation_artifact_bundle(
            clean_df,
            scenario_samples_df,
            scenario_summary_df,
            expected_idx_to_name={0: "drift", 1: "noise"},
            context_name="degradation_bundle",
        )


def test_validate_degradation_artifact_bundle_rejects_clean_length_mismatch():
    clean_df = pd.DataFrame(
        [
            {"sample_id": 0, "source_sample_idx": 3, "err_clean": 0.1},
            {"sample_id": 1, "source_sample_idx": 7, "err_clean": 0.2},
        ]
    )
    scenario_samples_df = pd.DataFrame(
        [
            {
                "sample_id": 0,
                "source_sample_idx": 3,
                "pert_idx": 0,
                "scenario": "drift",
                "severity": 0.2,
                "err_pert": 0.3,
            },
            {
                "sample_id": 0,
                "source_sample_idx": 3,
                "pert_idx": 1,
                "scenario": "noise",
                "severity": 0.7,
                "err_pert": 0.5,
            },
            {
                "sample_id": 1,
                "source_sample_idx": 7,
                "pert_idx": 0,
                "scenario": "drift",
                "severity": 0.4,
                "err_pert": 0.6,
            },
            {
                "sample_id": 1,
                "source_sample_idx": 7,
                "pert_idx": 1,
                "scenario": "noise",
                "severity": 0.9,
                "err_pert": 0.8,
            },
        ]
    )
    scenario_summary_df = pd.DataFrame(
        [
            {
                "pert_idx": 0,
                "scenario": "drift",
                "n_test_samples": 3,
                "err_clean_global": 0.15,
                "err_pert": 0.45,
                "err_pert_CI_lo": 0.35,
                "err_pert_CI_hi": 0.55,
                "D": 3.0,
                "D_CI_lo": 2.0,
                "D_CI_hi": 4.0,
            },
            {
                "pert_idx": 1,
                "scenario": "noise",
                "n_test_samples": 3,
                "err_clean_global": 0.15,
                "err_pert": 0.65,
                "err_pert_CI_lo": 0.55,
                "err_pert_CI_hi": 0.75,
                "D": 4.0,
                "D_CI_lo": 3.0,
                "D_CI_hi": 5.0,
            },
        ]
    )

    with pytest.raises(ValueError, match="clean_test_samples has 2 rows but expected n_test_samples=3"):
        validate_degradation_artifact_bundle(
            clean_df,
            scenario_samples_df,
            scenario_summary_df,
            expected_idx_to_name={0: "drift", 1: "noise"},
            expected_n_test_samples=3,
            context_name="degradation_bundle",
        )


def test_validate_degradation_artifact_bundle_rejects_err_clean_global_mismatch():
    clean_df = pd.DataFrame(
        [
            {"sample_id": 0, "source_sample_idx": 3, "err_clean": 0.1},
            {"sample_id": 1, "source_sample_idx": 7, "err_clean": 0.3},
        ]
    )
    scenario_samples_df = pd.DataFrame(
        [
            {
                "sample_id": 0,
                "source_sample_idx": 3,
                "pert_idx": 0,
                "scenario": "drift",
                "severity": 0.2,
                "err_pert": 0.3,
            },
            {
                "sample_id": 0,
                "source_sample_idx": 3,
                "pert_idx": 1,
                "scenario": "noise",
                "severity": 0.7,
                "err_pert": 0.5,
            },
            {
                "sample_id": 1,
                "source_sample_idx": 7,
                "pert_idx": 0,
                "scenario": "drift",
                "severity": 0.4,
                "err_pert": 0.6,
            },
            {
                "sample_id": 1,
                "source_sample_idx": 7,
                "pert_idx": 1,
                "scenario": "noise",
                "severity": 0.9,
                "err_pert": 0.8,
            },
        ]
    )
    scenario_summary_df = pd.DataFrame(
        [
            {
                "pert_idx": 0,
                "scenario": "drift",
                "n_test_samples": 2,
                "err_clean_global": 999.0,
                "err_pert": 0.45,
                "err_pert_CI_lo": 0.35,
                "err_pert_CI_hi": 0.55,
                "D": 3.0,
                "D_CI_lo": 2.0,
                "D_CI_hi": 4.0,
            },
            {
                "pert_idx": 1,
                "scenario": "noise",
                "n_test_samples": 2,
                "err_clean_global": 999.0,
                "err_pert": 0.65,
                "err_pert_CI_lo": 0.55,
                "err_pert_CI_hi": 0.75,
                "D": 4.0,
                "D_CI_lo": 3.0,
                "D_CI_hi": 5.0,
            },
        ]
    )

    with pytest.raises(ValueError, match="does not match clean_test_samples mean err_clean"):
        validate_degradation_artifact_bundle(
            clean_df,
            scenario_samples_df,
            scenario_summary_df,
            expected_idx_to_name={0: "drift", 1: "noise"},
            context_name="degradation_bundle",
        )


def test_plot_semantics_record_requires_zero_neutral_for_delta():
    record = PlotSemanticsRecord(
        direction="minimize",
        axis_family="delta",
        neutral_value=0,
        display_label="delta D_w",
    )

    assert record.direction == "minimize"
    assert record.axis_family == "delta"
    assert record.neutral_value == 0.0

    with pytest.raises(ValueError, match="neutral_value=0"):
        PlotSemanticsRecord(
            direction="minimize",
            axis_family="delta",
            neutral_value=None,
            display_label="delta D_w",
        )


def test_require_plot_semantics_mapping_requires_complete_record_mapping():
    semantics = {
        "D_w": PlotSemanticsRecord(
            direction="minimize",
            axis_family="numeric",
            neutral_value=None,
            display_label="Worst-case degradation",
        )
    }

    resolved = require_plot_semantics_mapping(
        semantics,
        required_keys=("D_w",),
        context="metric semantics",
    )

    assert list(resolved) == ["D_w"]

    with pytest.raises(ValueError, match="missing required keys"):
        require_plot_semantics_mapping(
            semantics,
            required_keys=("D_w", "D_mean"),
            context="metric semantics",
        )


def test_score_degradation_artifact_bundle_uses_first_scenario_on_exact_dw_tie():
    clean_df = pd.DataFrame(
        [
            {"sample_id": 0, "source_sample_idx": 10, "err_clean": 1.0},
            {"sample_id": 1, "source_sample_idx": 11, "err_clean": 3.0},
        ]
    )
    scenario_samples_df = pd.DataFrame(
        [
            {
                "sample_id": 0,
                "source_sample_idx": 10,
                "pert_idx": 0,
                "scenario": "drift",
                "severity": 0.2,
                "err_pert": 2.0,
            },
            {
                "sample_id": 0,
                "source_sample_idx": 10,
                "pert_idx": 1,
                "scenario": "noise",
                "severity": 0.4,
                "err_pert": 4.0,
            },
            {
                "sample_id": 1,
                "source_sample_idx": 11,
                "pert_idx": 0,
                "scenario": "drift",
                "severity": 0.6,
                "err_pert": 6.0,
            },
            {
                "sample_id": 1,
                "source_sample_idx": 11,
                "pert_idx": 1,
                "scenario": "noise",
                "severity": 0.8,
                "err_pert": 4.0,
            },
        ]
    )

    clean, scenario_samples, scenario_summary, metric_bundle, worst_scenario = (
        score_degradation_artifact_bundle(
            clean_df,
            scenario_samples_df,
            expected_idx_to_name={0: "drift", 1: "noise"},
            bootstrap_resamples=25,
            bootstrap_confidence_level=0.95,
            bootstrap_seed=7,
            context_name="degradation_bundle",
        )
    )

    assert clean["sample_id"].tolist() == [0, 1]
    assert scenario_samples["pert_idx"].tolist() == [0, 1, 0, 1]
    assert scenario_summary["D"].tolist() == pytest.approx([2.0, 2.0])
    assert metric_bundle["D_w"] == pytest.approx(2.0)
    assert metric_bundle["D_mean"] == pytest.approx(2.0)
    assert metric_bundle["err_pert_ws"] == pytest.approx(4.0)
    assert metric_bundle["err_pert_mean"] == pytest.approx(4.0)
    assert worst_scenario == "drift"


def test_score_degradation_artifact_bundle_rejects_non_positive_clean_denominator():
    clean_df = pd.DataFrame(
        [
            {"sample_id": 0, "source_sample_idx": 10, "err_clean": 0.0},
            {"sample_id": 1, "source_sample_idx": 11, "err_clean": 0.0},
        ]
    )
    scenario_samples_df = pd.DataFrame(
        [
            {
                "sample_id": 0,
                "source_sample_idx": 10,
                "pert_idx": 0,
                "scenario": "drift",
                "severity": 0.2,
                "err_pert": 1.0,
            },
            {
                "sample_id": 1,
                "source_sample_idx": 11,
                "pert_idx": 0,
                "scenario": "drift",
                "severity": 0.4,
                "err_pert": 1.0,
            },
        ]
    )

    with pytest.raises(ValueError, match="denominator must be > 0"):
        score_degradation_artifact_bundle(
            clean_df,
            scenario_samples_df,
            expected_idx_to_name={0: "drift"},
            bootstrap_resamples=25,
            bootstrap_confidence_level=0.95,
            bootstrap_seed=7,
            context_name="degradation_bundle",
        )


def _fixed_channel_fraction_sample_frames():
    clean_df = pd.DataFrame(
        [
            {"sample_id": 0, "source_sample_idx": 10, "err_clean": 1.0},
            {"sample_id": 1, "source_sample_idx": 11, "err_clean": 2.0},
        ]
    )
    scenario_samples_df = pd.DataFrame(
        [
            {
                "sample_id": 0,
                "source_sample_idx": 10,
                "pert_idx": 0,
                "scenario": "drift",
                "severity": 0.25,
                "err_pert": 2.5,
                "intensity_severity": 0.25,
                "requested_fixed_channel_fraction": 0.5,
                "derived_fixed_channel_count": 2,
                "channel_scope": "continuous",
                "eligible_channel_count": 4,
                "selected_channel_count": 2,
                "reported_affected_channel_count": 2,
            },
            {
                "sample_id": 0,
                "source_sample_idx": 10,
                "pert_idx": 1,
                "scenario": "missing_data",
                "severity": 0.0,
                "err_pert": 1.0,
                "intensity_severity": 0.0,
                "requested_fixed_channel_fraction": None,
                "derived_fixed_channel_count": None,
                "channel_scope": "all",
                "eligible_channel_count": None,
                "selected_channel_count": None,
                "reported_affected_channel_count": 0,
            },
            {
                "sample_id": 1,
                "source_sample_idx": 11,
                "pert_idx": 0,
                "scenario": "drift",
                "severity": 0.75,
                "err_pert": 3.5,
                "intensity_severity": 0.75,
                "requested_fixed_channel_fraction": 0.5,
                "derived_fixed_channel_count": 2,
                "channel_scope": "continuous",
                "eligible_channel_count": 4,
                "selected_channel_count": 2,
                "reported_affected_channel_count": 2,
            },
            {
                "sample_id": 1,
                "source_sample_idx": 11,
                "pert_idx": 1,
                "scenario": "missing_data",
                "severity": 0.6,
                "err_pert": 2.0,
                "intensity_severity": 0.6,
                "requested_fixed_channel_fraction": None,
                "derived_fixed_channel_count": None,
                "channel_scope": "all",
                "eligible_channel_count": None,
                "selected_channel_count": None,
                "reported_affected_channel_count": 0,
            },
        ]
    )
    scenario_summary_df = pd.DataFrame(
        [
            {
                "pert_idx": 0,
                "scenario": "drift",
                "n_test_samples": 2,
                "err_clean_global": 1.5,
                "err_pert": 3.0,
                "err_pert_CI_lo": 2.5,
                "err_pert_CI_hi": 3.5,
                "D": 2.0,
                "D_CI_lo": 1.8,
                "D_CI_hi": 2.2,
            },
            {
                "pert_idx": 1,
                "scenario": "missing_data",
                "n_test_samples": 2,
                "err_clean_global": 1.5,
                "err_pert": 1.5,
                "err_pert_CI_lo": 1.0,
                "err_pert_CI_hi": 2.0,
                "D": 1.0,
                "D_CI_lo": 0.8,
                "D_CI_hi": 1.2,
            },
        ]
    )
    return clean_df, scenario_samples_df, scenario_summary_df


def test_validate_fixed_channel_fraction_bundle_checks_diagnostics():
    clean_df, scenario_samples_df, scenario_summary_df = (
        _fixed_channel_fraction_sample_frames()
    )

    clean, scenario_samples, scenario_summary = (
        validate_fixed_channel_fraction_artifact_bundle(
            clean_df,
            scenario_samples_df,
            scenario_summary_df,
            expected_idx_to_name={0: "drift", 1: "missing_data"},
            expected_n_test_samples=2,
            fixed_channel_fraction=0.5,
            perturbation_channel_fraction_max=0.5,
            context_name="fixed-fraction bundle",
        )
    )

    assert clean["sample_id"].tolist() == [0, 1]
    selected_counts = scenario_samples["selected_channel_count"]
    assert selected_counts.iloc[[0, 2]].tolist() == [2.0, 2.0]
    assert selected_counts.iloc[[1, 3]].isna().all()
    assert scenario_summary["D"].tolist() == pytest.approx([2.0, 1.0])


def test_validate_fixed_channel_fraction_bundle_rejects_affected_count_drift():
    clean_df, scenario_samples_df, scenario_summary_df = (
        _fixed_channel_fraction_sample_frames()
    )
    scenario_samples_df.loc[0, "reported_affected_channel_count"] = 1

    with pytest.raises(ValueError, match="reported_affected_channel_count"):
        validate_fixed_channel_fraction_artifact_bundle(
            clean_df,
            scenario_samples_df,
            scenario_summary_df,
            expected_idx_to_name={0: "drift", 1: "missing_data"},
            expected_n_test_samples=2,
            fixed_channel_fraction=0.5,
            perturbation_channel_fraction_max=0.5,
            context_name="fixed-fraction bundle",
        )


def test_require_logged_fixed_channel_fraction_bundle_validates_aggregate_metrics():
    test_metric = "MSE"
    fixed_fraction = 0.5
    max_fraction = 0.5
    metrics = {}
    for key in required_fixed_channel_fraction_metric_keys(
        test_metric=test_metric,
        fixed_channel_fraction=fixed_fraction,
        perturbation_channel_fraction_max=max_fraction,
        scenario_indices=(0, 1),
    ):
        metrics[key] = 0.0
    metric_values = {
        "D_w": 2.0,
        "D_w_CI_lo": 1.8,
        "D_w_CI_hi": 2.2,
        "D_mean": 1.5,
        "D_mean_CI_lo": 1.2,
        "D_mean_CI_hi": 1.8,
        "err_pert_ws": 3.0,
        "err_pert_ws_CI_lo": 2.5,
        "err_pert_ws_CI_hi": 3.5,
        "err_pert_mean": 2.25,
        "err_pert_mean_CI_lo": 2.0,
        "err_pert_mean_CI_hi": 2.5,
    }
    for metric_name, value in metric_values.items():
        metrics[
            build_fixed_channel_fraction_metric_key(
                test_metric=test_metric,
                fixed_channel_fraction=fixed_fraction,
                perturbation_channel_fraction_max=max_fraction,
                metric_name=metric_name,
            )
        ] = value
    prefix = "fixed_channel_fraction/fraction_0p5/MSE"
    metrics[f"{prefix}/scenario/0/D"] = 2.0
    metrics[f"{prefix}/scenario/0/err_pert"] = 3.0
    metrics[f"{prefix}/scenario/1/D"] = 1.0
    metrics[f"{prefix}/scenario/1/err_pert"] = 1.5

    worst_scenario = require_logged_fixed_channel_fraction_metric_bundle(
        metrics,
        tags={f"{prefix}/worst_scenario": "drift"},
        run_id="run_fixed_fraction",
        test_metric=test_metric,
        fixed_channel_fraction=fixed_fraction,
        perturbation_channel_fraction_max=max_fraction,
        expected_idx_to_name={0: "drift", 1: "missing_data"},
    )

    assert worst_scenario == "drift"

    metrics[
        build_fixed_channel_fraction_metric_key(
            test_metric=test_metric,
            fixed_channel_fraction=fixed_fraction,
            perturbation_channel_fraction_max=max_fraction,
            metric_name="D_mean",
        )
    ] = 1.0
    with pytest.raises(ValueError, match="inconsistent fixed-channel-fraction D_mean"):
        require_logged_fixed_channel_fraction_metric_bundle(
            metrics,
            tags={f"{prefix}/worst_scenario": "drift"},
            run_id="run_fixed_fraction",
            test_metric=test_metric,
            fixed_channel_fraction=fixed_fraction,
            perturbation_channel_fraction_max=max_fraction,
            expected_idx_to_name={0: "drift", 1: "missing_data"},
        )
