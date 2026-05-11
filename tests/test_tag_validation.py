from types import SimpleNamespace

import pytest

from data.perturbations import build_perturbation_scenario_params_signature
from pipelines import selection as selection_module
from pipelines.selection import resolve_pipeline_tags
from utils.parsing import (
    ROBUSTNESS_RESULTS_COMPLETE_TAG,
    build_degradation_eval_context_tag_payload,
    require_degradation_eval_context_from_args,
    require_shared_anchor_bootstrap_ci_context_from_args,
    robustness_results_complete_tag_value,
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


def test_resolve_pipeline_tags_requires_pipeline_method():
    tags = {
        "pipeline_id": "baseline",
        "robustness_method": "baseline",
        "pipeline_kind": "train",
    }
    with pytest.raises(ValueError, match="pipeline_method"):
        resolve_pipeline_tags(tags, run_id="bad-run")


def test_resolve_pipeline_tags_requires_pipeline_kind():
    tags = {
        "pipeline_id": "baseline",
        "pipeline_method": "baseline",
        "robustness_method": "baseline",
    }
    with pytest.raises(ValueError, match="pipeline_kind"):
        resolve_pipeline_tags(tags, run_id="bad-run")


def test_resolve_pipeline_tags_requires_pipeline_id():
    tags = {
        "pipeline_method": "baseline",
        "robustness_method": "baseline",
        "pipeline_kind": "train",
    }
    with pytest.raises(ValueError, match="pipeline_id"):
        resolve_pipeline_tags(tags, run_id="bad-run")


def test_resolve_pipeline_tags_rejects_baseline_id_with_non_baseline_method():
    tags = {
        "pipeline_id": "baseline",
        "pipeline_method": "adaptive_robust_loss",
        "robustness_method": "adaptive_robust_loss",
        "pipeline_kind": "train",
    }
    with pytest.raises(ValueError, match="pipeline_id='baseline'"):
        resolve_pipeline_tags(tags, run_id="bad-baseline")


def test_resolve_pipeline_tags_rejects_baseline_method_with_non_baseline_id():
    tags = {
        "pipeline_id": "adaptive_robust_loss_delta1_0",
        "pipeline_method": "baseline",
        "robustness_method": "baseline",
        "pipeline_kind": "train",
    }
    with pytest.raises(ValueError, match="pipeline_method='baseline'"):
        resolve_pipeline_tags(tags, run_id="bad-baseline")


def test_resolve_pipeline_tags_returns_required_fields():
    tags = {
        "pipeline_id": "adaptive_robust_loss_delta1_0",
        "pipeline_method": "adaptive_robust_loss",
        "robustness_method": "adaptive_robust_loss",
        "pipeline_kind": "train",
        "dataset": "dataset_a",
    }

    resolved = resolve_pipeline_tags(tags, run_id="run-1")

    assert resolved == {
        "pipeline_id": "adaptive_robust_loss_delta1_0",
        "pipeline_method": "adaptive_robust_loss",
        "pipeline_kind": "train",
        "robustness_method": "adaptive_robust_loss",
    }


def test_resolve_pipeline_tags_requires_robustness_method():
    tags = {
        "pipeline_id": "baseline",
        "pipeline_method": "baseline",
        "pipeline_kind": "train",
    }
    with pytest.raises(ValueError, match="missing robustness_method"):
        resolve_pipeline_tags(tags, run_id="bad-run")


def test_resolve_pipeline_tags_rejects_unknown_pipeline_kind():
    tags = {
        "pipeline_id": "adaptive_robust_loss_delta1_0",
        "pipeline_method": "adaptive_robust_loss",
        "robustness_method": "adaptive_robust_loss",
        "pipeline_kind": "unknown_kind",
    }
    with pytest.raises(ValueError, match="unknown pipeline_kind"):
        resolve_pipeline_tags(tags, run_id="bad-kind")


def test_resolve_pipeline_tags_rejects_non_baseline_method_mismatch():
    tags = {
        "pipeline_id": "adaptive_robust_loss_delta1_0",
        "pipeline_method": "adaptive_robust_loss",
        "robustness_method": "ensemble",
        "pipeline_kind": "train",
    }
    with pytest.raises(ValueError, match="non-baseline tag mismatch"):
        resolve_pipeline_tags(tags, run_id="bad-mismatch")


def test_resolve_pipeline_tags_rejects_baseline_robustness_mismatch():
    tags = {
        "pipeline_id": "baseline",
        "pipeline_method": "baseline",
        "robustness_method": "adaptive_robust_loss",
        "pipeline_kind": "train",
    }
    with pytest.raises(ValueError, match="must set robustness_method='baseline'"):
        resolve_pipeline_tags(tags, run_id="bad-baseline-robustness")


def test_resolve_pipeline_tags_rejects_non_train_baseline_kind():
    tags = {
        "pipeline_id": "baseline",
        "pipeline_method": "baseline",
        "robustness_method": "baseline",
        "pipeline_kind": "wrap",
    }
    with pytest.raises(ValueError, match="pipeline_kind='train'"):
        resolve_pipeline_tags(tags, run_id="bad-baseline-kind")


def test_is_fully_tested_uses_seed_data_for_canonical_eval_data_seed(monkeypatch):
    captured = {}

    def _fake_resolve(eval_data_seed, *, canonical_seed_data, eval_key, canonical_key):
        captured["eval_data_seed"] = eval_data_seed
        captured["canonical_seed_data"] = canonical_seed_data
        captured["eval_key"] = eval_key
        captured["canonical_key"] = canonical_key
        return 11

    def _stop_after_seed_resolution(*_args, **_kwargs):
        raise RuntimeError("stopped after seed resolution")

    monkeypatch.setattr(
        selection_module,
        "resolve_effective_eval_data_seed",
        _fake_resolve,
    )
    monkeypatch.setattr(
        selection_module,
        "require_degradation_eval_context_from_args",
        _stop_after_seed_resolution,
    )
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="run_1"),
        data=SimpleNamespace(
            params={"tested": "true"},
            tags={
                "robustness_scoring_semantics": (
                    selection_module.ROBUSTNESS_SCORING_SEMANTICS
                ),
                "seed_data": "11",
                "seed_eval": "33",
            },
        ),
    )

    with pytest.raises(RuntimeError, match="stopped after seed resolution"):
        selection_module.is_fully_tested(
            run,
            args=SimpleNamespace(eval_data_seed=None),
        )

    assert captured == {
        "eval_data_seed": None,
        "canonical_seed_data": "11",
        "eval_key": "args.eval_data_seed",
        "canonical_key": "seed_data tag",
    }


def test_is_fully_tested_returns_false_when_tested_param_is_missing():
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="run_missing_tested"),
        data=SimpleNamespace(params={}, tags={}),
    )

    assert selection_module.is_fully_tested(run, args=SimpleNamespace()) is False


def test_is_fully_tested_rejects_malformed_tested_param():
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="run_bad_tested"),
        data=SimpleNamespace(params={"tested": "not-a-bool"}, tags={}),
    )

    with pytest.raises(ValueError, match="invalid 'tested' param"):
        selection_module.is_fully_tested(run, args=SimpleNamespace())


def test_is_fully_tested_reuses_canonical_seed_data_degradation_bundle():
    scenarios = ["drift", "noise"]
    args = SimpleNamespace(
        eval_data_seed=None,
        test_metric="MSE",
        n_test_samples=64,
        perturbation_channel_fraction_max=0.5,
        perturbation_scenarios=scenarios,
        strict_iid=False,
        bootstrap_ci_resamples=99,
        bootstrap_ci_confidence_level=0.95,
        full_coverage=True,
    )
    eval_context = require_degradation_eval_context_from_args(
        args,
        eval_data_seed=11,
        context="args",
    )
    bootstrap_context = require_shared_anchor_bootstrap_ci_context_from_args(
        args,
        eval_data_seed=11,
        test_metric="MSE",
        context="args",
    )
    tags = {
        "robustness_scoring_semantics": selection_module.ROBUSTNESS_SCORING_SEMANTICS,
        "seed_data": "11",
        "seed_eval": "33",
        "perturbation_scenario_params_signature": (
            build_perturbation_scenario_params_signature(scenarios)
        ),
        ROBUSTNESS_RESULTS_COMPLETE_TAG: robustness_results_complete_tag_value(
            complete=True
        ),
        "degradation/MSE/worst_scenario": "noise",
    }
    tags.update(build_degradation_eval_context_tag_payload(eval_context))
    tags.update({key: str(value) for key, value in bootstrap_context.items()})
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="run_seed_data_reuse"),
        data=SimpleNamespace(
            params={
                "tested": "true",
                "degradation/MSE/worst_scenario": "noise",
            },
            tags=tags,
            metrics=_logged_degradation_metrics(),
        ),
    )

    assert selection_module.is_fully_tested(run, args=args) is True
