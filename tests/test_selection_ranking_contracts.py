from __future__ import annotations

import pandas as pd
import pytest

from pipelines.ranking import (
    rank_key_for_dataframe_row,
    rank_key_for_row_values,
    resolve_selection_score,
    selection_metric_key_for_kind,
)
from pipelines.selection import select_group_winners


def test_clean_selection_uses_validation_metrics_not_test_metrics():
    score = resolve_selection_score(
        metrics={
            "best_val_loss": 0.25,
            "MSE_test": 0.01,
            "MSE_val": 0.4,
        },
        pipeline_kind="train",
        robustness_method="adaptive_robust_loss",
        test_metric="MSE",
        improvement_selection_mode="clean",
        run_id="train_run",
    )

    assert score == pytest.approx(0.25)
    assert (
        selection_metric_key_for_kind(
            pipeline_kind="wrap",
            robustness_method="randomized_smoothing",
            test_metric="MSE",
            improvement_selection_mode="clean",
            run_id="wrap_run",
        )
        == "MSE_val"
    )
    assert (
        resolve_selection_score(
            metrics={
                "best_val_loss": 0.01,
                "MSE_val": 0.4,
                "MSE_test": 0.02,
            },
            pipeline_kind="wrap",
            robustness_method="randomized_smoothing",
            test_metric="MSE",
            improvement_selection_mode="clean",
            run_id="wrap_run",
        )
        == pytest.approx(0.4)
    )


def test_baseline_selection_ignores_perturbed_selector_mode():
    assert (
        resolve_selection_score(
            metrics={
                "best_val_loss": 0.3,
                "MSE_pert_ws_val": 0.1,
                "MSE_pert_mean_val": 0.2,
                "MSE_test": 0.01,
            },
            pipeline_kind="train",
            robustness_method="baseline",
            test_metric="MSE",
            improvement_selection_mode="perturbed_worst",
            run_id="baseline_run",
        )
        == pytest.approx(0.3)
    )


@pytest.mark.parametrize(
    ("mode", "expected_key", "expected_score"),
    (
        ("perturbed_worst", "MSE_pert_ws_val", 0.6),
        ("perturbed_mean", "MSE_pert_mean_val", 0.4),
    ),
)
def test_perturbed_selection_requires_validation_perturbation_pair(
    mode: str,
    expected_key: str,
    expected_score: float,
):
    metrics = {
        "best_val_loss": 0.1,
        "MSE_test": 0.01,
        "MSE_pert_ws_val": 0.6,
        "MSE_pert_mean_val": 0.4,
    }

    assert (
        selection_metric_key_for_kind(
            pipeline_kind="train",
            robustness_method="adversarial_training",
            test_metric="MSE",
            improvement_selection_mode=mode,
            run_id="adv_run",
        )
        == expected_key
    )
    assert (
        resolve_selection_score(
            metrics=metrics,
            pipeline_kind="train",
            robustness_method="adversarial_training",
            test_metric="MSE",
            improvement_selection_mode=mode,
            run_id="adv_run",
        )
        == pytest.approx(expected_score)
    )

    incomplete_metrics = dict(metrics)
    incomplete_metrics.pop("MSE_pert_mean_val")
    with pytest.raises(ValueError, match="MSE_pert_mean_val"):
        resolve_selection_score(
            metrics=incomplete_metrics,
            pipeline_kind="train",
            robustness_method="adversarial_training",
            test_metric="MSE",
            improvement_selection_mode=mode,
            run_id="adv_run_incomplete",
        )


def test_rank_key_dataframe_adapters_use_canonical_tiebreaker():
    assert rank_key_for_row_values(
        selection_value=0.1,
        end_time=200,
        run_id="run_a",
    ) == (0.1, -200.0, "run_a")
    assert rank_key_for_dataframe_row(
        {
            "score": 0.2,
            "finished_at": 100,
            "candidate_id": "run_b",
        },
        selection_value_col="score",
        end_time_col="finished_at",
        run_id_col="candidate_id",
    ) == (0.2, -100.0, "run_b")


def test_select_group_winners_uses_validation_score_then_latest_then_run_id():
    df = pd.DataFrame(
        [
            {
                "dataset": "ETTh1",
                "run_id": "run_bad_test_metric",
                "best_val_loss": 0.2,
                "MSE_test": 0.01,
                "end_time": 999,
            },
            {
                "dataset": "ETTh1",
                "run_id": "run_z",
                "best_val_loss": 0.1,
                "MSE_test": 0.9,
                "end_time": 200,
            },
            {
                "dataset": "ETTh1",
                "run_id": "run_a",
                "best_val_loss": 0.1,
                "MSE_test": 0.8,
                "end_time": 200,
            },
            {
                "dataset": "traffic",
                "run_id": "traffic_run",
                "best_val_loss": 0.3,
                "MSE_test": 0.01,
                "end_time": 1,
            },
        ]
    )

    winners = select_group_winners(
        df,
        group_cols=["dataset"],
        selection_metric_col="best_val_loss",
    )

    assert {
        row["dataset"]: row["run_id"]
        for row in winners.to_dict("records")
    } == {
        "ETTh1": "run_a",
        "traffic": "traffic_run",
    }
