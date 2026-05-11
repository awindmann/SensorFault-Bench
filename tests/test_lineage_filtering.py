import json
from pathlib import Path

import pytest

from pipelines.selection import (
    build_base_index,
    classify_lineage_run as _classify_lineage_run,
)


class DummyInfo:
    def __init__(self, run_id, end_time=0, lifecycle_stage="active"):
        self.run_id = run_id
        self.end_time = end_time
        self.lifecycle_stage = lifecycle_stage


class DummyData:
    def __init__(self, tags=None, metrics=None, params=None):
        self.tags = tags or {}
        self.metrics = metrics or {}
        self.params = params or {}


class DummyRun:
    def __init__(
        self,
        run_id,
        *,
        tags=None,
        metrics=None,
        params=None,
        end_time=0,
        hparams=None,
    ):
        self.info = DummyInfo(run_id, end_time=end_time)
        self.data = DummyData(tags=tags, metrics=metrics, params=params)
        self.hparams = dict(hparams) if hparams is not None else None


def _make_run(run_id, *, tags=None, metrics=None, params=None, end_time=0, hparams=None):
    return DummyRun(
        run_id,
        tags=tags,
        metrics=metrics,
        params=params,
        end_time=end_time,
        hparams=hparams,
    )


def _build_base_maps(runs):
    return build_base_index({("Arch", "baseline"): runs})


_BASELINE_HPARAM_SPECS = {
    "Arch": {"lr": [0.001, 0.01]},
    "DLinear": {"lr": [0.001, 0.01]},
    "TSMixer": {"lr": [0.001, 0.01]},
}


def _baseline_run(run_id, *, best_val_loss: float, end_time: int, lr: float):
    return _make_run(
        run_id,
        tags={"lr": str(lr)},
        metrics={"best_val_loss": best_val_loss},
        end_time=end_time,
        hparams={"lr": lr},
    )


class _HparamsArtifactClient:
    def __init__(self, hparams_by_run_id):
        self.hparams_by_run_id = dict(hparams_by_run_id)

    def download_artifacts(self, run_id, artifact_path, dst_path=None):
        assert artifact_path == "hparams.json"
        if run_id not in self.hparams_by_run_id:
            raise FileNotFoundError(f"missing hparams for {run_id}")
        target_dir = Path(dst_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{run_id}_hparams.json"
        path.write_text(
            json.dumps(self.hparams_by_run_id[run_id]),
            encoding="utf-8",
        )
        return str(path)


def _hparams_from_run(run):
    if getattr(run, "hparams", None) is not None:
        return dict(run.hparams)
    return dict(getattr(run.data, "tags", {}) or {})


def _lineage_hparams_client(run, kwargs):
    runs = [run]
    runs.extend((kwargs.get("current_base_runs_by_key") or {}).values())
    for sorted_runs in (kwargs.get("sorted_base_runs_by_key") or {}).values():
        runs.extend(sorted_runs)
    return _HparamsArtifactClient(
        {candidate.info.run_id: _hparams_from_run(candidate) for candidate in runs}
    )


def classify_lineage_run(run, arch, **kwargs):
    if (
        "artifact_client" not in kwargs
        and kwargs.get("baseline_hparam_specs_by_arch") is not None
    ):
        kwargs["artifact_client"] = _lineage_hparams_client(run, kwargs)
    return _classify_lineage_run(run, arch, **kwargs)


def test_build_base_index_prefers_latest_end_time_on_ties():
    older = _make_run("base_old", metrics={"best_val_loss": 0.1}, end_time=100)
    newer = _make_run("base_new", metrics={"best_val_loss": 0.1}, end_time=200)
    sorted_runs, current_runs = _build_base_maps([older, newer])
    assert sorted_runs[("Arch", "baseline")][0].info.run_id == "base_new"
    assert current_runs[("Arch", "baseline")].info.run_id == "base_new"


def test_classify_lineage_backbone_set_policy():
    base1 = _make_run("base1", metrics={"best_val_loss": 0.1}, end_time=300)
    base2 = _make_run("base2", metrics={"best_val_loss": 0.2}, end_time=200)
    base3 = _make_run("base3", metrics={"best_val_loss": 0.3}, end_time=100)
    sorted_runs, current_runs = _build_base_maps([base1, base2, base3])

    tags_match = {
        "pipeline_method": "ensemble",
        "base_pipeline_method": "baseline",
        "backbone_run_ids": "base1,base2",
    }
    reason = classify_lineage_run(
        _make_run("ens_ok", tags=tags_match),
        "Arch",
        current_base_runs_by_key=current_runs,
        sorted_base_runs_by_key=sorted_runs,
    )
    assert reason is None

    tags_mismatch = dict(tags_match, backbone_run_ids="base2,base3")
    reason = classify_lineage_run(
        _make_run("ens_bad", tags=tags_mismatch),
        "Arch",
        current_base_runs_by_key=current_runs,
        sorted_base_runs_by_key=sorted_runs,
    )
    assert reason == "backbone_set_changed"

    tags_whitespace = dict(tags_match, backbone_run_ids="base1, base2")
    reason = classify_lineage_run(
        _make_run("ens_whitespace", tags=tags_whitespace),
        "Arch",
        current_base_runs_by_key=current_runs,
        sorted_base_runs_by_key=sorted_runs,
    )
    assert reason is None

    tags_duplicate = dict(tags_match, backbone_run_ids="base1,base1,base2")
    with pytest.raises(ValueError, match="duplicate id"):
        classify_lineage_run(
            _make_run("ens_duplicate", tags=tags_duplicate),
            "Arch",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
        )

    tags_insufficient = dict(tags_match, backbone_run_ids="base1,base2,base3,base4")
    reason = classify_lineage_run(
        _make_run("ens_short", tags=tags_insufficient),
        "Arch",
        current_base_runs_by_key=current_runs,
        sorted_base_runs_by_key=sorted_runs,
    )
    assert reason == "insufficient_current_backbones"


def test_classify_lineage_hparams_modes():
    base = _baseline_run("base1", best_val_loss=0.1, end_time=100, lr=0.001)
    sorted_runs, current_runs = _build_base_maps([base])

    baseline_grid_tags = {
        "pipeline_method": "train",
        "base_pipeline_method": "baseline",
        "hparams_mode": "baseline_grid",
    }
    reason = classify_lineage_run(
        _make_run("grid_ok", tags=baseline_grid_tags),
        "Arch",
        current_base_runs_by_key=current_runs,
        sorted_base_runs_by_key=sorted_runs,
    )
    assert reason is None

    inherit_tags = {
        "pipeline_method": "train",
        "base_pipeline_method": "baseline",
        "hparams_mode": "inherit_baseline",
    }
    reason = classify_lineage_run(
        _make_run("inherit_missing", tags=inherit_tags),
        "Arch",
        current_base_runs_by_key=current_runs,
        sorted_base_runs_by_key=sorted_runs,
    )
    assert reason == "missing_baseline_hparams_run_id"

    unknown_tags = dict(inherit_tags, hparams_mode="mystery_mode")
    reason = classify_lineage_run(
        _make_run("inherit_unknown", tags=unknown_tags),
        "Arch",
        current_base_runs_by_key=current_runs,
        sorted_base_runs_by_key=sorted_runs,
    )
    assert reason == "unknown_hparams_mode"


def test_classify_lineage_baseline_hparams_mismatch():
    base = _baseline_run("base1", best_val_loss=0.1, end_time=100, lr=0.001)
    sorted_runs, current_runs = _build_base_maps([base])

    tags = {
        "pipeline_method": "train",
        "base_pipeline_method": "baseline",
        "baseline_hparams_run_id": "base0",
        "hparams_mode": "inherit_baseline",
        "lr": "0.01",
    }
    reason = classify_lineage_run(
        _make_run("inherit_bad", tags=tags),
        "Arch",
        current_base_runs_by_key=current_runs,
        sorted_base_runs_by_key=sorted_runs,
        baseline_hparam_specs_by_arch=_BASELINE_HPARAM_SPECS,
    )
    assert reason == "baseline_changed"


def test_classify_lineage_inherit_requires_parseable_baseline_hparams():
    base = _baseline_run("base1", best_val_loss=0.1, end_time=100, lr=0.001)
    sorted_runs, current_runs = _build_base_maps([base])

    with pytest.raises(ValueError, match="missing required hyperparameter 'lr'"):
        classify_lineage_run(
            _make_run(
                "inherit_missing_lr",
                tags={
                    "pipeline_method": "train",
                    "base_pipeline_method": "baseline",
                    "hparams_mode": "inherit_baseline",
                    "baseline_hparams_run_id": "base1",
                },
            ),
            "Arch",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
            baseline_hparam_specs_by_arch=_BASELINE_HPARAM_SPECS,
        )


def test_stale_detection_after_baseline_change():
    base_old = _baseline_run("base_old", best_val_loss=0.2, end_time=100, lr=0.001)
    sorted_runs, current_runs = _build_base_maps([base_old])

    tags = {
        "pipeline_method": "train",
        "base_pipeline_method": "baseline",
        "baseline_hparams_run_id": "base_old",
        "hparams_mode": "inherit_baseline",
        "lr": "0.001",
    }
    improvement = _make_run("improve_run", tags=tags)
    reason = classify_lineage_run(
        improvement,
        "Arch",
        current_base_runs_by_key=current_runs,
        sorted_base_runs_by_key=sorted_runs,
        baseline_hparam_specs_by_arch=_BASELINE_HPARAM_SPECS,
    )
    assert reason is None

    base_new = _baseline_run("base_new", best_val_loss=0.1, end_time=200, lr=0.01)
    sorted_runs, current_runs = _build_base_maps([base_new, base_old])
    reason = classify_lineage_run(
        improvement,
        "Arch",
        current_base_runs_by_key=current_runs,
        sorted_base_runs_by_key=sorted_runs,
        baseline_hparam_specs_by_arch=_BASELINE_HPARAM_SPECS,
    )
    assert reason == "baseline_changed"


def test_inherit_lineage_stays_current_when_winner_run_changes_but_hparams_match():
    base_old = _baseline_run("base_old", best_val_loss=0.2, end_time=100, lr=0.001)
    sorted_runs, current_runs = _build_base_maps([base_old])

    improvement = _make_run(
        "improve_run",
        tags={
            "pipeline_method": "train",
            "base_pipeline_method": "baseline",
            "baseline_hparams_run_id": "base_old",
            "hparams_mode": "inherit_baseline",
            "lr": "0.001",
        },
    )
    reason = classify_lineage_run(
        improvement,
        "Arch",
        current_base_runs_by_key=current_runs,
        sorted_base_runs_by_key=sorted_runs,
        baseline_hparam_specs_by_arch=_BASELINE_HPARAM_SPECS,
    )
    assert reason is None

    # New winner has better val score but identical baseline hparams.
    base_new = _baseline_run("base_new", best_val_loss=0.1, end_time=200, lr=0.001)
    sorted_runs, current_runs = _build_base_maps([base_new, base_old])
    reason = classify_lineage_run(
        improvement,
        "Arch",
        current_base_runs_by_key=current_runs,
        sorted_base_runs_by_key=sorted_runs,
        baseline_hparam_specs_by_arch=_BASELINE_HPARAM_SPECS,
    )
    assert reason is None


# =============================================================================
# Additional lineage consistency tests
# =============================================================================


class TestTiedBaselines:
    """Edge case: tied baselines with same best_val_loss."""

    def test_tied_baselines_and_end_time_uses_run_id_as_tiebreaker(self):
        """When val_loss and end_time both tie, run_id breaks the tie deterministically."""
        run_a = _make_run("base_aaa", metrics={"best_val_loss": 0.1}, end_time=100)
        run_z = _make_run("base_zzz", metrics={"best_val_loss": 0.1}, end_time=100)
        sorted_runs, current_runs = _build_base_maps([run_z, run_a])

        # run_id is used alphabetically as final tiebreaker
        assert current_runs[("Arch", "baseline")].info.run_id == "base_aaa"
        assert sorted_runs[("Arch", "baseline")][0].info.run_id == "base_aaa"

    def test_improvement_tracks_correct_baseline_on_ties(self):
        """Improvement built on tied baseline should be current if it matches winner."""
        run_a = _baseline_run("base_aaa", best_val_loss=0.1, end_time=100, lr=0.001)
        run_z = _baseline_run("base_zzz", best_val_loss=0.1, end_time=100, lr=0.01)
        sorted_runs, current_runs = _build_base_maps([run_z, run_a])

        # Improvement built on winner (base_aaa) is current
        tags_winner = {
            "pipeline_method": "train",
            "base_pipeline_method": "baseline",
            "baseline_hparams_run_id": "base_aaa",
            "hparams_mode": "inherit_baseline",
            "lr": "0.001",
        }
        reason = classify_lineage_run(
            _make_run("improve_ok", tags=tags_winner),
            "Arch",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
            baseline_hparam_specs_by_arch=_BASELINE_HPARAM_SPECS,
        )
        assert reason is None

        # Improvement built on loser (base_zzz) is stale
        tags_loser = {
            "pipeline_method": "train",
            "base_pipeline_method": "baseline",
            "baseline_hparams_run_id": "base_zzz",
            "hparams_mode": "inherit_baseline",
            "lr": "0.01",
        }
        reason = classify_lineage_run(
            _make_run("improve_stale", tags=tags_loser),
            "Arch",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
            baseline_hparam_specs_by_arch=_BASELINE_HPARAM_SPECS,
        )
        assert reason == "baseline_changed"


class TestEnsemblePartialBackboneOverlap:
    """Edge case: ensemble with partial backbone overlap."""

    def test_ensemble_partial_overlap_is_stale(self):
        """Ensemble with only partial overlap (k-1 of k match) is stale."""
        base1 = _make_run("base1", metrics={"best_val_loss": 0.1}, end_time=300)
        base2 = _make_run("base2", metrics={"best_val_loss": 0.2}, end_time=200)
        base3 = _make_run("base3", metrics={"best_val_loss": 0.3}, end_time=100)
        sorted_runs, current_runs = _build_base_maps([base1, base2, base3])

        # Ensemble uses base1 + base3 but current top-2 is base1 + base2
        tags = {
            "pipeline_method": "ensemble",
            "base_pipeline_method": "baseline",
            "backbone_run_ids": "base1,base3",
        }
        reason = classify_lineage_run(
            _make_run("ens_partial", tags=tags),
            "Arch",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
        )
        assert reason == "backbone_set_changed"

    def test_ensemble_stale_after_single_backbone_change(self):
        """When one backbone in top-k changes, affected ensembles become stale."""
        base1 = _make_run("base1", metrics={"best_val_loss": 0.1}, end_time=300)
        base2 = _make_run("base2", metrics={"best_val_loss": 0.2}, end_time=200)
        base3 = _make_run("base3", metrics={"best_val_loss": 0.3}, end_time=100)
        sorted_runs, current_runs = _build_base_maps([base1, base2, base3])

        # Ensemble built on original top-3
        tags_top3 = {
            "pipeline_method": "ensemble",
            "base_pipeline_method": "baseline",
            "backbone_run_ids": "base1,base2,base3",
        }
        reason = classify_lineage_run(
            _make_run("ens_top3", tags=tags_top3),
            "Arch",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
        )
        assert reason is None

        # New run becomes 4th best - top-3 ensemble stays current
        base4 = _make_run("base4", metrics={"best_val_loss": 0.4}, end_time=400)
        sorted_runs, current_runs = _build_base_maps([base1, base2, base3, base4])
        reason = classify_lineage_run(
            _make_run("ens_top3", tags=tags_top3),
            "Arch",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
        )
        assert reason is None  # Still current because top-3 unchanged

        # New run becomes new best - all ensembles become stale
        base_new = _make_run("base_new", metrics={"best_val_loss": 0.05}, end_time=500)
        sorted_runs, current_runs = _build_base_maps([base1, base2, base3, base_new])
        reason = classify_lineage_run(
            _make_run("ens_top3", tags=tags_top3),
            "Arch",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
        )
        assert reason == "backbone_set_changed"

    def test_ensemble_order_independent(self):
        """Backbone IDs order should not matter for ensemble matching."""
        base1 = _make_run("base1", metrics={"best_val_loss": 0.1}, end_time=300)
        base2 = _make_run("base2", metrics={"best_val_loss": 0.2}, end_time=200)
        sorted_runs, current_runs = _build_base_maps([base1, base2])

        # Same backbones in different order
        tags_order1 = {
            "pipeline_method": "ensemble",
            "base_pipeline_method": "baseline",
            "backbone_run_ids": "base1,base2",
        }
        tags_order2 = {
            "pipeline_method": "ensemble",
            "base_pipeline_method": "baseline",
            "backbone_run_ids": "base2,base1",
        }

        reason1 = classify_lineage_run(
            _make_run("ens1", tags=tags_order1),
            "Arch",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
        )
        reason2 = classify_lineage_run(
            _make_run("ens2", tags=tags_order2),
            "Arch",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
        )
        assert reason1 is None
        assert reason2 is None


class TestBackboneLineageScenarios:
    """Distinct backbone lineage scenarios."""

    def test_finetune_backbone_lineage(self):
        """Finetune runs with backbone_run_id track lineage correctly."""
        base = _make_run("base1", metrics={"best_val_loss": 0.1}, end_time=100)
        sorted_runs, current_runs = _build_base_maps([base])

        # Finetune run with matching backbone
        finetune_tags = {
            "pipeline_method": "adaptive_robust_loss_finetune",
            "base_pipeline_method": "baseline",
            "backbone_run_id": "base1",
        }
        reason = classify_lineage_run(
            _make_run("finetune_ok", tags=finetune_tags),
            "Arch",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
        )
        assert reason is None

        # Finetune run with stale backbone
        stale_finetune_tags = {
            "pipeline_method": "adaptive_robust_loss_finetune",
            "base_pipeline_method": "baseline",
            "backbone_run_id": "old_base",
        }
        reason = classify_lineage_run(
            _make_run("finetune_stale", tags=stale_finetune_tags),
            "Arch",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
        )
        assert reason == "backbone_changed"

    def test_multiple_architectures_independent(self):
        """Different architectures have independent baseline lineage."""
        # DLinear baseline
        dlinear_base = _make_run(
            "dlinear_base",
            tags={"lr": "0.001"},
            metrics={"best_val_loss": 0.1},
            end_time=100,
        )
        # TSMixer baseline
        tsmixer_base = _make_run(
            "tsmixer_base",
            tags={"lr": "0.01"},
            metrics={"best_val_loss": 0.2},
            end_time=100,
        )

        runs_by_key = {
            ("DLinear", "baseline"): [dlinear_base],
            ("TSMixer", "baseline"): [tsmixer_base],
        }
        sorted_runs, current_runs = build_base_index(runs_by_key)

        # DLinear improvement
        dlinear_improve_tags = {
            "pipeline_method": "train",
            "base_pipeline_method": "baseline",
            "baseline_hparams_run_id": "dlinear_base",
            "hparams_mode": "inherit_baseline",
            "lr": "0.001",
        }
        reason = classify_lineage_run(
            _make_run("dlinear_improve", tags=dlinear_improve_tags),
            "DLinear",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
            baseline_hparam_specs_by_arch=_BASELINE_HPARAM_SPECS,
        )
        assert reason is None

        # TSMixer improvement
        tsmixer_improve_tags = {
            "pipeline_method": "train",
            "base_pipeline_method": "baseline",
            "baseline_hparams_run_id": "tsmixer_base",
            "hparams_mode": "inherit_baseline",
            "lr": "0.01",
        }
        reason = classify_lineage_run(
            _make_run("tsmixer_improve", tags=tsmixer_improve_tags),
            "TSMixer",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
            baseline_hparam_specs_by_arch=_BASELINE_HPARAM_SPECS,
        )
        assert reason is None

        # Cross-architecture mismatch is stale
        reason = classify_lineage_run(
            _make_run("cross_arch", tags=dlinear_improve_tags),
            "TSMixer",  # Wrong architecture
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
            baseline_hparam_specs_by_arch=_BASELINE_HPARAM_SPECS,
        )
        assert reason == "baseline_changed"


class TestMissingLineageTags:
    """Tests for runs with missing lineage tags (treated as stale per policy)."""

    def test_missing_hparams_mode_is_stale(self):
        """Run without hparams_mode and no other lineage tags is stale."""
        base = _make_run("base1", metrics={"best_val_loss": 0.1}, end_time=100)
        sorted_runs, current_runs = _build_base_maps([base])

        tags = {
            "pipeline_method": "train",
            "base_pipeline_method": "baseline",
            # No hparams_mode, no backbone tags
        }
        reason = classify_lineage_run(
            _make_run("no_lineage", tags=tags),
            "Arch",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
        )
        assert reason == "missing_hparams_mode"

    def test_missing_hparams_mode_with_baseline_hparams_tag_is_stale(self):
        base = _baseline_run("base1", best_val_loss=0.1, end_time=100, lr=0.001)
        sorted_runs, current_runs = _build_base_maps([base])

        reason = classify_lineage_run(
            _make_run(
                "no_mode_with_baseline_hparams",
                tags={
                    "pipeline_method": "train",
                    "base_pipeline_method": "baseline",
                    "baseline_hparams_run_id": "base1",
                    "lr": "0.001",
                },
            ),
            "Arch",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
            baseline_hparam_specs_by_arch=_BASELINE_HPARAM_SPECS,
        )
        assert reason == "missing_hparams_mode"

    def test_missing_base_pipeline_method_is_stale(self):
        """Non-baseline run without base_pipeline_method is stale."""
        base = _make_run("base1", metrics={"best_val_loss": 0.1}, end_time=100)
        sorted_runs, current_runs = _build_base_maps([base])

        tags = {
            "pipeline_method": "ensemble",
            # No base_pipeline_method
            "backbone_run_ids": "base1",
        }
        reason = classify_lineage_run(
            _make_run("no_base_method", tags=tags),
            "Arch",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
        )
        assert reason == "missing_base_pipeline_method"

    def test_baseline_runs_always_current(self):
        """Baseline runs are always current (no lineage check needed)."""
        base = _make_run("base1", metrics={"best_val_loss": 0.1}, end_time=100)
        sorted_runs, current_runs = _build_base_maps([base])

        tags = {"pipeline_method": "baseline"}
        reason = classify_lineage_run(
            _make_run("any_baseline", tags=tags),
            "Arch",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
        )
        assert reason is None

    def test_baseline_grid_without_backbone_is_current(self):
        """baseline_grid mode without backbone tags is always current."""
        base = _make_run("base1", metrics={"best_val_loss": 0.1}, end_time=100)
        sorted_runs, current_runs = _build_base_maps([base])

        tags = {
            "pipeline_method": "train",
            "base_pipeline_method": "baseline",
            "hparams_mode": "baseline_grid",
        }
        reason = classify_lineage_run(
            _make_run("grid_run", tags=tags),
            "Arch",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
        )
        assert reason is None


class TestNoCurrentBaseline:
    """Tests for edge case when no current baseline exists."""

    def test_backbone_run_id_without_current_baseline(self):
        """Run with backbone_run_id but no current baseline is stale."""
        sorted_runs = {("Arch", "baseline"): []}
        current_runs = {}

        tags = {
            "pipeline_method": "adaptive_robust_loss_finetune",
            "base_pipeline_method": "baseline",
            "backbone_run_id": "orphan_base",
        }
        reason = classify_lineage_run(
            _make_run("orphan_finetune", tags=tags),
            "Arch",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
        )
        assert reason == "missing_current_baseline"

    def test_ensemble_without_current_baseline(self):
        """Ensemble without current baseline is stale."""
        sorted_runs = {("Arch", "baseline"): []}
        current_runs = {}

        tags = {
            "pipeline_method": "ensemble",
            "base_pipeline_method": "baseline",
            "backbone_run_ids": "base1,base2",
        }
        reason = classify_lineage_run(
            _make_run("orphan_ensemble", tags=tags),
            "Arch",
            current_base_runs_by_key=current_runs,
            sorted_base_runs_by_key=sorted_runs,
        )
        assert reason == "missing_current_baseline"
