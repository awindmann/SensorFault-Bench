import argparse
import sys
from types import SimpleNamespace

import pytest

from config_loader import build_parser, load_defaults, parse_explicit_cli_overrides

import run_analysis
import run_testing
import run_training


class _RunnerStub:
    def __init__(self, captured):
        self._captured = captured
        self._last_report = None

    def run(self, _client, dataset_spec, arch):
        self._captured.append((dataset_spec.key, arch))
        self._last_report = SimpleNamespace(
            dataset=dataset_spec.key,
            architecture=arch,
            pipeline_method="baseline",
            pipeline_kind="train",
            expected_units=1,
            executed_units=1,
            skipped_existing_units=0,
            skipped_policy_units=0,
            failed_units=0,
            uncovered_units=0,
            is_complete=True,
        )

    def get_last_run_report(self):
        return self._last_report


def _recipe_scope_stub():
    return [
        SimpleNamespace(
            pipeline_id="baseline",
            pipeline_method="baseline",
            pipeline_kind="train",
            recipe_params={},
            model_hparams_mode="baseline_grid",
        ),
        SimpleNamespace(
            pipeline_id="ensemble",
            pipeline_method="ensemble",
            pipeline_kind="wrap",
            recipe_params={},
            model_hparams_mode="inherit_baseline",
        ),
    ]


def _dataset_windows(**_kwargs):
    return {
        "ETTh1": {"input_len": 168, "target_len": 24, "batch_size": 16},
        "Penmanshiel_Hourly_WT08": {"input_len": 288, "target_len": 36, "batch_size": 64},
        "BeijingAir_Tiantan": {"input_len": 168, "target_len": 24, "batch_size": 64},
    }


def _build_training_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=None)
    parser.add_argument("--benchmark-architectures", nargs="*", default=["GRU"])
    parser.add_argument("--method", nargs="+", default=None)
    parser.add_argument("--benchmark-methods", nargs="*", default=["baseline"])
    parser.add_argument("--minio-endpoint", default=None)
    parser.add_argument("--logdir", default="/tmp/mlruns")
    parser.add_argument("--data-files", nargs="*", default=["ETTh1", "Penmanshiel_Hourly_WT08"])
    parser.add_argument("--data-targets", nargs="*", default=[])
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--input-len", type=int, default=90)
    parser.add_argument("--target-len", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def _build_run_testing_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-files", nargs="*", default=["BeijingAir_Tiantan"])
    parser.add_argument("--data-targets", nargs="*", default=[])
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--input-len", type=int, default=90)
    parser.add_argument("--target-len", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--minio-endpoint", default=None)
    parser.add_argument("--raise-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full-coverage", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rerun", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--method", nargs="+", default=None)
    parser.add_argument("--benchmark-methods", nargs="*", default=["baseline"])
    parser.add_argument("--model", nargs="+", default=None)
    parser.add_argument("--benchmark-architectures", nargs="*", default=["GRU"])
    return parser


def _patch_run_testing_orchestration(monkeypatch):
    monkeypatch.setattr(
        run_testing,
        "load_defaults",
        lambda: {
            "INPUT_LEN": 90,
            "TARGET_LEN": 30,
            "DATA_ROOT": "data/processed",
            "METHOD": None,
            "BENCHMARK_METHODS": ["baseline"],
            "MODEL": None,
            "BENCHMARK_ARCHITECTURES": ["GRU"],
            "BATCH_SIZE": 8,
        },
    )
    monkeypatch.setattr(run_testing, "load_dataset_windows", _dataset_windows)
    monkeypatch.setattr(
        run_testing,
        "load_benchmark_recipe_specs_for_scope",
        lambda _defaults: _recipe_scope_stub(),
    )
    monkeypatch.setattr(run_testing, "extract_recipe_defaults_for_scope", lambda _specs: [])
    monkeypatch.setattr(run_testing, "merge_recipe_defaults_for_scope", lambda defaults, _extracted: defaults)
    monkeypatch.setattr(run_testing, "_configure_runtime_loggers_for_testing", lambda: None)
    monkeypatch.setattr(run_testing, "build_parser", lambda _defaults: _build_run_testing_parser())


def _patch_training_entrypoint(monkeypatch):
    monkeypatch.setattr(
        run_training,
        "load_defaults",
        lambda: {
            "INPUT_LEN": 90,
            "TARGET_LEN": 30,
            "DATA_ROOT": "data/processed",
            "METHOD": None,
            "BENCHMARK_METHODS": ["baseline"],
            "MODEL": None,
            "BENCHMARK_ARCHITECTURES": ["GRU"],
            "BATCH_SIZE": 8,
        },
    )
    monkeypatch.setattr(run_training, "load_dataset_windows", _dataset_windows)
    monkeypatch.setattr(
        run_training,
        "load_benchmark_recipe_specs_for_scope",
        lambda _defaults: [_recipe_scope_stub()[0]],
    )
    monkeypatch.setattr(run_training, "extract_recipe_defaults_for_scope", lambda _specs: [])
    monkeypatch.setattr(run_training, "merge_recipe_defaults_for_scope", lambda defaults, _extracted: defaults)
    monkeypatch.setattr(run_training, "parse_explicit_cli_overrides", lambda *_a, **_k: {})
    monkeypatch.setattr(run_training, "build_parser", lambda _defaults: _build_training_parser())
    monkeypatch.setattr(
        "pipelines.selection.load_hparams",
        lambda: {"GRU": {"lr": [0.001]}},
    )
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


def test_run_training_uses_dataset_window_defaults_per_dataset(monkeypatch):
    _patch_training_entrypoint(monkeypatch)
    monkeypatch.setattr(
        run_training,
        "resolve_with_defaults",
        lambda datasets, targets, data_root: [
            SimpleNamespace(key="ETTh1", path="/tmp/etth1.csv"),
            SimpleNamespace(key="Penmanshiel_Hourly_WT08", path="/tmp/pen.csv"),
        ],
    )

    captured_windows: list[tuple[int, int, int]] = []
    captured_runs: list[tuple[str, str]] = []

    def _create_runner(_spec, args):
        captured_windows.append((args.input_len, args.target_len, args.batch_size))
        return _RunnerStub(captured_runs)

    monkeypatch.setattr(run_training, "create_pipeline_runner", _create_runner)
    monkeypatch.setattr(sys, "argv", ["run_training.py"])

    run_training.main()

    assert captured_windows == [(168, 24, 16), (288, 36, 64)]
    assert captured_runs == [("ETTh1", "GRU"), ("Penmanshiel_Hourly_WT08", "GRU")]


def test_run_testing_main_uses_dataset_window_defaults(monkeypatch):
    _patch_run_testing_orchestration(monkeypatch)
    monkeypatch.setattr(
        run_testing,
        "resolve_with_defaults",
        lambda datasets, targets, data_root: [
            SimpleNamespace(key="BeijingAir_Tiantan", path="/tmp/beijing.parquet")
        ],
    )

    captured: list[tuple[int, int, int]] = []

    def _test_on_dataset(*, dataset_spec, args, recipe_specs_for_scope):
        captured.append((args.input_len, args.target_len, args.batch_size))
        return 0, {}

    monkeypatch.setattr(run_testing, "test_on_dataset", _test_on_dataset)
    monkeypatch.setattr(sys, "argv", ["run_testing.py", "--full-coverage"])

    run_testing.main()

    assert captured == [(168, 24, 64)]


def test_benchmark_defaults_expose_current_scope_keys():
    defaults = load_defaults()

    assert defaults["MODEL"] is None
    assert defaults["METHOD"] is None
    assert defaults["BENCHMARK_ARCHITECTURES"]
    assert defaults["BENCHMARK_METHODS"]


def test_model_parser_accepts_multi_model_scope():
    parser = build_parser(load_defaults())

    assert parser.parse_args([]).model is None
    assert parser.parse_args(["--model", "GRU", "PatchTST"]).model == [
        "GRU",
        "PatchTST",
    ]
    assert parser.parse_args(["--model", "GRU"]).model == ["GRU"]


def test_model_parser_rejects_empty_model_flag():
    parser = build_parser(load_defaults())

    with pytest.raises(SystemExit):
        parser.parse_args(["--model"])


def test_full_coverage_defaults_to_partial_and_strict_flag_opts_in():
    defaults = load_defaults()
    parser = build_parser(defaults)

    assert defaults["FULL_COVERAGE"] is False
    assert parser.parse_args([]).full_coverage is False
    assert parser.parse_args(["--full-coverage"]).full_coverage is True
    assert parser.parse_args(["--no-full-coverage"]).full_coverage is False


def test_default_benchmark_testing_and_analysis_use_partial_coverage():
    testing_args, _, _ = run_testing.prepare_testing_entrypoint_args([])
    analysis_args = run_analysis.prepare_analysis_entrypoint_args([])

    assert testing_args.full_coverage is False
    assert testing_args._explicit_cli_args == ()
    assert analysis_args.full_coverage is False
    assert analysis_args._explicit_cli_args == ()


def test_parse_explicit_cli_overrides_accepts_multi_scope_values():
    overrides = parse_explicit_cli_overrides(
        {"MODEL": None, "METHOD": None, "BENCHMARK_ARCHITECTURES": []},
        extra_args=["--model", "GRU", "PatchTST", "--method", "baseline", "ensemble"],
    )

    assert overrides["model"] == ["GRU", "PatchTST"]
    assert overrides["method"] == ["baseline", "ensemble"]


def test_benchmark_parser_exposes_current_scope_flags():
    parser = build_parser(load_defaults())

    assert "--model" in parser._option_string_actions
    assert "--method" in parser._option_string_actions
    assert "--benchmark-architectures" in parser._option_string_actions
    assert "--benchmark-methods" in parser._option_string_actions


def test_run_analysis_main_runs_standalone_analysis(monkeypatch):
    args = SimpleNamespace(
        _explicit_cli_args=(),
        minio_endpoint="",
    )
    monkeypatch.setattr(
        run_analysis,
        "prepare_analysis_entrypoint_args",
        lambda: args,
    )
    monkeypatch.setattr(run_analysis, "_configure_runtime_loggers_for_testing", lambda: None)
    monkeypatch.setattr(run_analysis, "set_mlflow_storage_env", lambda _args: None)

    captured: list[object] = []

    def _meta_analysis(meta_args, *, coverage_fractions_by_dataset=None):
        captured.append(coverage_fractions_by_dataset)

    monkeypatch.setattr(run_analysis, "meta_analysis", _meta_analysis)

    run_analysis.main()

    assert captured == [None]


def test_run_analysis_main_uses_real_prepared_parser_path(monkeypatch):
    monkeypatch.setattr(
        run_analysis,
        "load_defaults",
        lambda: {
            **load_defaults(),
            "DATA_FILES": ["ETTh1"],
            "DATA_TARGETS": [],
            "BENCHMARK_METHODS": ["baseline", "ensemble"],
        },
    )
    monkeypatch.setattr(
        run_analysis,
        "load_benchmark_recipe_specs_for_scope",
        lambda _defaults: _recipe_scope_stub(),
    )
    monkeypatch.setattr(run_analysis, "_configure_runtime_loggers_for_testing", lambda: None)
    monkeypatch.setattr(run_analysis, "set_mlflow_storage_env", lambda _args: None)

    captured: list[tuple[bool, tuple[str, ...], dict]] = []

    def _meta_analysis(meta_args, *, coverage_fractions_by_dataset=None):
        captured.append(
            (
                meta_args.full_coverage,
                meta_args._explicit_cli_args,
                meta_args._window_arg_overrides,
            )
        )

    monkeypatch.setattr(run_analysis, "meta_analysis", _meta_analysis)
    monkeypatch.setattr(sys, "argv", ["run_analysis.py", "--full-coverage"])

    run_analysis.main()

    assert captured == [(True, ("--full-coverage",), {})]


def test_run_analysis_accepts_scope_flags_needed_for_coverage_recompute():
    parser = build_parser(load_defaults())
    parsed = parser.parse_args(
        [
            "--model",
            "GRU",
            "--benchmark-architectures",
            "GRU",
            "--method",
            "baseline",
            "--benchmark-methods",
            "baseline",
        ]
    )

    assert parsed.model == ["GRU"]
    assert parsed.benchmark_architectures == ["GRU"]
    assert parsed.method == ["baseline"]
    assert parsed.benchmark_methods == ["baseline"]


@pytest.mark.parametrize("argv", [["run_analysis.py", "--rerun", "--max-epochs", "1"]])
def test_run_analysis_accepts_shared_training_execution_flags(argv):
    parser = build_parser(load_defaults())
    parsed = parser.parse_args(argv[1:])

    assert parsed.rerun is True
    assert parsed.max_epochs == 1


def test_run_testing_no_raise_error_allows_dataset_failures(monkeypatch):
    _patch_run_testing_orchestration(monkeypatch)
    monkeypatch.setattr(
        run_testing,
        "resolve_with_defaults",
        lambda datasets, targets, data_root: [
            SimpleNamespace(key="BeijingAir_Tiantan", path="/tmp/beijing.parquet"),
            SimpleNamespace(key="ETTh1", path="/tmp/etth1.csv"),
        ],
    )

    seen_datasets: list[str] = []

    def _test_on_dataset(*, dataset_spec, args, recipe_specs_for_scope):
        seen_datasets.append(dataset_spec.key)
        if dataset_spec.key == "BeijingAir_Tiantan":
            raise RuntimeError("boom")
        return 0, {("GRU", "baseline"): (1, 1)}

    monkeypatch.setattr(run_testing, "test_on_dataset", _test_on_dataset)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_testing.py", "--no-raise-error", "--full-coverage"],
    )

    run_testing.main()

    assert seen_datasets == ["BeijingAir_Tiantan", "ETTh1"]
