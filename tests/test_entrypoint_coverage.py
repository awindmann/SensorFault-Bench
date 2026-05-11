from types import SimpleNamespace

import pytest

import run_training
from config_loader import build_parser, load_defaults
from pipelines.selection import CoverageMismatchError
from pipelines.specs import PipelineSpec
from testing.evaluation import _require_requested_family_selected_variants


class _RunnerStub:
    def __init__(self, reports=()):
        self._reports = list(reports)
        self._last_report = None
        self.calls = 0

    def run(self, _client, _dataset_spec, _arch):
        self._last_report = (
            self._reports[self.calls]
            if self.calls < len(self._reports)
            else None
        )
        self.calls += 1

    def get_last_run_report(self):
        return self._last_report


def _patch_run_training_side_effects(monkeypatch, *, runner):
    monkeypatch.setattr(run_training, "set_mlflow_storage_env", lambda _args: None)
    monkeypatch.setattr(run_training, "get_tracking_uri", lambda logdir: f"file:{logdir}")
    monkeypatch.setattr(run_training.mlflow, "set_tracking_uri", lambda *_a, **_k: None)
    monkeypatch.setattr(run_training.mlflow, "MlflowClient", lambda: SimpleNamespace())
    monkeypatch.setattr(run_training, "print_coverage_report", lambda _report: None)
    monkeypatch.setattr(
        run_training,
        "print_coverage_summary_and_raise_on_incomplete",
        lambda _reports: None,
    )
    monkeypatch.setattr(
        run_training,
        "resolve_with_defaults",
        lambda datasets, targets, data_root: [
            SimpleNamespace(key=dataset, path=f"/tmp/{dataset}")
            for dataset in datasets
        ],
    )
    monkeypatch.setattr(run_training, "create_pipeline_runner", lambda _spec, _args: runner)


def test_run_training_requires_runner_coverage_report(monkeypatch):
    runner = _RunnerStub()
    _patch_run_training_side_effects(monkeypatch, runner=runner)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_training.py",
            "--data-files",
            "ETTh1",
            "--model",
            "GRU",
            "--method",
            "baseline",
        ],
    )

    with pytest.raises(ValueError, match="did not emit a RunExecutionReport"):
        run_training.main()

    assert runner.calls == 1


def test_run_training_rejects_scope_resolver_that_drops_requested_method(monkeypatch):
    monkeypatch.setattr(
        run_training,
        "resolve_benchmark_method_architecture_scope",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_training.py",
            "--data-files",
            "ETTh1",
            "--model",
            "GRU",
            "--method",
            "baseline",
        ],
    )

    with pytest.raises(ValueError, match="omitted requested method\\(s\\): baseline"):
        run_training.main()


def test_run_analysis_accepts_shared_training_scope_flags(monkeypatch):
    parser = build_parser(load_defaults())
    parsed = parser.parse_args(["--rerun", "--max-epochs", "1", "--no-save-checkpoint"])

    assert parsed.rerun is True
    assert parsed.max_epochs == 1
    assert parsed.save_checkpoint is False


def test_pretrain_finetune_kind_is_not_benchmark_runner_surface():
    with pytest.raises(ValueError, match="Invalid pipeline_kind"):
        PipelineSpec(
            pipeline_id="removed_pretrain",
            pipeline_method="removed_pretrain",
            pipeline_kind="pretrain_finetune",
            recipe_params={},
            model_hparams_mode="inherit_baseline",
        )


def test_explicit_method_scope_cannot_relax_to_zero_selected_variants():
    with pytest.raises(CoverageMismatchError, match="randomized_training/GRU"):
        _require_requested_family_selected_variants(
            selected_variants=set(),
            selected_methods=frozenset({"randomized_training"}),
            dataset_coverage_fractions={
                ("GRU", "randomized_training"): (0, 3),
            },
        )


def test_default_partial_coverage_can_still_keep_unrequested_family_drops():
    _require_requested_family_selected_variants(
        selected_variants={("GRU", "baseline", "baseline")},
        selected_methods=frozenset({"baseline"}),
        dataset_coverage_fractions={
            ("GRU", "baseline"): (1, 1),
            ("GRU", "randomized_training"): (0, 3),
        },
    )
