from types import SimpleNamespace

import pytest

import run_analysis
import run_testing
import run_training
import testing.evaluation as evaluation
from config_loader import build_parser, load_defaults, load_benchmark_scope
from pipelines.recipes import PIPELINE_RECIPE_PATHS_BY_METHOD
from pipelines.selection import (
    resolve_benchmark_method_architecture_scope,
    resolve_requested_methods,
)


BENCHMARK_METHOD_ORDER = (
    "baseline",
    "randomized_training",
    "adversarial_training",
    "adaptive_robust_loss",
    "ensemble",
    "randomized_smoothing",
    "fault_augmentation",
    "revin",
)
BENCHMARK_ARCHITECTURES = (
    "DLinear",
    "TSMixer",
    "PatchTST",
    "GRU",
    "ModernTCN",
    "SeasonalNaive",
    "Chronos2",
)


def _expected_benchmark_scope(defaults=None, benchmark_scope=None):
    defaults = load_defaults() if defaults is None else defaults
    benchmark_scope = load_benchmark_scope() if benchmark_scope is None else benchmark_scope
    applicability = benchmark_scope["methods"]["method_architecture_applicability"]
    return {
        method: tuple(
            architecture
            for architecture in defaults["BENCHMARK_ARCHITECTURES"]
            if architecture in applicability[method]
        )
        for method in defaults["BENCHMARK_METHODS"]
    }


def _complete_report(*, dataset: str, architecture: str, method: str, kind: str):
    return SimpleNamespace(
        dataset=dataset,
        architecture=architecture,
        pipeline_method=method,
        pipeline_kind=kind,
        expected_units=1,
        executed_units=1,
        skipped_existing_units=0,
        skipped_policy_units=0,
        failed_units=0,
        uncovered_units=0,
        is_complete=True,
    )


class _RunnerStub:
    def __init__(self, spec, calls):
        self._spec = spec
        self._calls = calls
        self._last_report = None

    def run(self, _client, dataset_spec, architecture):
        self._calls.append((self._spec.pipeline_method, dataset_spec.key, architecture))
        self._last_report = _complete_report(
            dataset=dataset_spec.key,
            architecture=architecture,
            method=self._spec.pipeline_method,
            kind=self._spec.pipeline_kind,
        )

    def get_last_run_report(self):
        return self._last_report


def _patch_training_runtime(monkeypatch):
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        "pipelines.selection.load_hparams",
        lambda: {architecture: {"lr": [0.001]} for architecture in BENCHMARK_ARCHITECTURES},
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
    monkeypatch.setattr(
        run_training,
        "resolve_with_defaults",
        lambda datasets, targets, data_root: [
            SimpleNamespace(key=dataset, path=f"/tmp/{dataset}")
            for dataset in datasets
        ],
    )
    monkeypatch.setattr(
        run_training,
        "create_pipeline_runner",
        lambda spec, _args: _RunnerStub(spec, calls),
    )
    return calls


def test_benchmark_defaults_expose_method_scope_without_removed_method_flags():
    defaults = load_defaults()
    parser = build_parser(defaults)

    assert defaults["METHOD"] is None
    assert tuple(defaults["BENCHMARK_METHODS"]) == BENCHMARK_METHOD_ORDER
    assert "IMPROVEMENT_METHOD" not in defaults
    assert "SKIP_RECIPES" not in defaults
    assert parser.parse_args([]).method is None
    assert parser.parse_args(["--method", "baseline"]).method == ["baseline"]
    assert "--improvement-method" not in parser._option_string_actions
    assert "--skip-recipes" not in parser._option_string_actions


def test_resolve_requested_methods_matches_model_scope_shape():
    configured = tuple(PIPELINE_RECIPE_PATHS_BY_METHOD)
    args = SimpleNamespace(
        method=None,
        benchmark_methods=["baseline", "randomized_training"],
    )
    assert resolve_requested_methods(args, configured_methods=configured) == [
        "baseline",
        "randomized_training",
    ]

    explicit_args = SimpleNamespace(
        method=["adaptive_robust_loss", "randomized_training"],
        benchmark_methods=list(BENCHMARK_METHOD_ORDER),
    )
    assert resolve_requested_methods(explicit_args, configured_methods=configured) == [
        "adaptive_robust_loss",
        "randomized_training",
    ]

    duplicate_args = SimpleNamespace(
        method=["baseline", "Baseline"],
        benchmark_methods=list(BENCHMARK_METHOD_ORDER),
    )
    with pytest.raises(ValueError, match="METHOD contains duplicate value 'baseline'"):
        resolve_requested_methods(duplicate_args, configured_methods=configured)

    all_args = SimpleNamespace(
        method=["all"],
        benchmark_methods=list(BENCHMARK_METHOD_ORDER),
    )
    with pytest.raises(ValueError, match="METHOD no longer supports 'all'"):
        resolve_requested_methods(all_args, configured_methods=configured)


def test_empty_benchmark_architectures_do_not_fall_back_to_hparam_keys(monkeypatch):
    monkeypatch.setattr(
        "pipelines.selection.load_hparams",
        lambda: {"GRU": {"lr": [0.001]}, "PatchTST": {"lr": [0.001]}},
    )
    with pytest.raises(ValueError, match="BENCHMARK_ARCHITECTURES must contain"):
        run_training.resolve_requested_architectures(
            SimpleNamespace(model=None, benchmark_architectures=[])
        )


def test_run_training_no_args_expands_benchmark_applicability(monkeypatch):
    calls = _patch_training_runtime(monkeypatch)
    monkeypatch.setattr("sys.argv", ["run_training.py"])

    run_training.main()

    defaults = load_defaults()
    expected_scope = _expected_benchmark_scope(defaults)
    expected_calls = [
        (method, dataset, architecture)
        for method in defaults["BENCHMARK_METHODS"]
        for dataset in defaults["DATA_FILES"]
        for architecture in expected_scope[method]
    ]

    assert calls == expected_calls


def test_run_training_scoped_methods_do_not_train_implicit_baseline(monkeypatch):
    calls = _patch_training_runtime(monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_training.py",
            "--data-files",
            "ETTh1",
            "--model",
            "GRU",
            "--method",
            "adaptive_robust_loss",
            "randomized_training",
        ],
    )

    run_training.main()

    assert calls == [
        ("adaptive_robust_loss", "ETTh1", "GRU"),
        ("randomized_training", "ETTh1", "GRU"),
    ]


def test_run_training_scoped_methods_filter_default_architectures(monkeypatch):
    calls = _patch_training_runtime(monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_training.py",
            "--data-files",
            "ETTh1",
            "--method",
            "randomized_training",
            "adaptive_robust_loss",
        ],
    )

    run_training.main()

    expected_architectures = ("DLinear", "TSMixer", "PatchTST", "GRU", "ModernTCN")
    assert calls == [
        *[
            ("randomized_training", "ETTh1", architecture)
            for architecture in expected_architectures
        ],
        *[
            ("adaptive_robust_loss", "ETTh1", architecture)
            for architecture in expected_architectures
        ],
    ]


@pytest.mark.parametrize(
    "argv",
    [
        ["run_training.py", "--method", "not_a_method"],
        ["run_training.py", "--method", "f2f"],
        ["run_training.py", "--method", "all"],
        ["run_training.py", "--model", "LSTM"],
        ["run_training.py", "--model", "Chronos2"],
        ["run_training.py", "--method", "revin", "--model", "PatchTST"],
        [
            "run_training.py",
            "--method",
            "revin",
            "--benchmark-architectures",
            "DLinear",
            "PatchTST",
        ],
        [
            "run_training.py",
            "--method",
            "adversarial_training",
            "--model",
            "Chronos2",
        ],
    ],
)
def test_run_training_rejects_unknown_all_and_explicit_unsupported_pairs(
    monkeypatch,
    argv,
):
    calls = _patch_training_runtime(monkeypatch)
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(ValueError):
        run_training.main()
    assert calls == []


def test_run_testing_parser_uses_benchmark_method_scope(monkeypatch):
    monkeypatch.setattr(run_testing, "load_defaults", load_defaults)
    monkeypatch.setattr(
        run_testing,
        "load_dataset_windows",
        lambda **_kw: {
            dataset: {"input_len": 96, "target_len": 96}
            for dataset in load_defaults()["DATA_FILES"]
        },
    )
    args, recipe_specs, _ = run_testing.prepare_testing_entrypoint_args(
        [
            "--method",
            "randomized_training",
            "adaptive_robust_loss",
            "--model",
            "DLinear",
            "TSMixer",
            "PatchTST",
            "GRU",
            "ModernTCN",
        ]
    )

    assert args.method == ["randomized_training", "adaptive_robust_loss"]
    assert tuple(spec.pipeline_method for spec in recipe_specs) == BENCHMARK_METHOD_ORDER
    assert args._method_architecture_scope == {
        "randomized_training": (
            "DLinear",
            "TSMixer",
            "PatchTST",
            "GRU",
            "ModernTCN",
        ),
        "adaptive_robust_loss": (
            "DLinear",
            "TSMixer",
            "PatchTST",
            "GRU",
            "ModernTCN",
        ),
    }


def test_run_testing_omitted_method_uses_benchmark_methods(monkeypatch):
    monkeypatch.setattr(run_testing, "load_defaults", load_defaults)
    monkeypatch.setattr(
        run_testing,
        "load_dataset_windows",
        lambda **_kw: {
            dataset: {"input_len": 96, "target_len": 96}
            for dataset in load_defaults()["DATA_FILES"]
        },
    )
    args, recipe_specs, _ = run_testing.prepare_testing_entrypoint_args(["--full-coverage"])

    assert args.method is None
    assert tuple(spec.pipeline_method for spec in recipe_specs) == BENCHMARK_METHOD_ORDER
    assert args._method_architecture_scope == _expected_benchmark_scope()


def test_testing_internal_default_recipe_scope_is_benchmark(monkeypatch):
    sentinel = RuntimeError("benchmark recipe scope used")

    monkeypatch.setattr(
        evaluation,
        "compute_data_config_signature",
        lambda **_kw: "signature",
    )
    monkeypatch.setattr(evaluation, "load_hparams", lambda: {"DLinear": {}})
    monkeypatch.setattr(evaluation, "load_defaults", lambda: {"BENCHMARK_METHODS": ["baseline"]})

    def _load_benchmark(defaults):
        assert defaults == {"BENCHMARK_METHODS": ["baseline"]}
        raise sentinel

    monkeypatch.setattr(evaluation, "load_benchmark_recipe_specs_for_scope", _load_benchmark)

    with pytest.raises(RuntimeError, match="benchmark recipe scope used"):
        evaluation._resolve_dataset_testing_coverage_scope(
            SimpleNamespace(key="ETTh1"),
            SimpleNamespace(full_coverage=True),
        )


def test_testing_scope_resolver_receives_requested_architecture_order(monkeypatch):
    sentinel = RuntimeError("captured architecture order")
    captured = {}

    monkeypatch.setattr(
        evaluation,
        "compute_data_config_signature",
        lambda **_kw: "signature",
    )
    monkeypatch.setattr(
        evaluation,
        "load_hparams",
        lambda: {
            "DLinear": {},
            "GRU": {},
            "PatchTST": {},
        },
    )
    monkeypatch.setattr(
        evaluation,
        "resolve_requested_architectures",
        lambda _args: ["PatchTST", "GRU", "DLinear"],
    )
    monkeypatch.setattr(
        evaluation,
        "resolve_requested_methods",
        lambda _args, *, configured_methods: ["baseline"],
    )
    monkeypatch.setattr(
        evaluation,
        "has_explicit_architecture_scope",
        lambda _args: True,
    )

    def _capture_scope(
        *,
        methods,
        architectures,
        explicit_architectures,
        benchmark_scope=None,
        context,
    ):
        assert benchmark_scope is None
        captured["methods"] = tuple(methods)
        captured["architectures"] = tuple(architectures)
        captured["explicit_architectures"] = explicit_architectures
        captured["context"] = context
        raise sentinel

    monkeypatch.setattr(
        evaluation,
        "resolve_benchmark_method_architecture_scope",
        _capture_scope,
    )

    with pytest.raises(RuntimeError, match="captured architecture order"):
        evaluation._resolve_dataset_testing_coverage_scope(
            SimpleNamespace(key="ETTh1"),
            SimpleNamespace(full_coverage=True),
            recipe_specs_for_scope=[
                SimpleNamespace(pipeline_method="baseline", pipeline_kind="train")
            ],
        )

    assert captured == {
        "methods": ("baseline",),
        "architectures": ("PatchTST", "GRU", "DLinear"),
        "explicit_architectures": True,
        "context": "run_testing.py",
    }


def test_run_testing_explicit_method_does_not_select_implicit_baseline(monkeypatch):
    monkeypatch.setattr(run_testing, "load_defaults", load_defaults)
    monkeypatch.setattr(
        run_testing,
        "load_dataset_windows",
        lambda **_kw: {
            dataset: {"input_len": 96, "target_len": 96}
            for dataset in load_defaults()["DATA_FILES"]
        },
    )
    args, _, _ = run_testing.prepare_testing_entrypoint_args(
        ["--method", "randomized_training", "--model", "GRU"]
    )

    assert args._method_architecture_scope == {"randomized_training": ("GRU",)}


def test_run_testing_explicit_method_filters_default_architectures(monkeypatch):
    monkeypatch.setattr(run_testing, "load_defaults", load_defaults)
    monkeypatch.setattr(
        run_testing,
        "load_dataset_windows",
        lambda **_kw: {
            dataset: {"input_len": 96, "target_len": 96}
            for dataset in load_defaults()["DATA_FILES"]
        },
    )
    args, _, _ = run_testing.prepare_testing_entrypoint_args(
        ["--method", "randomized_training"]
    )

    assert args._method_architecture_scope == {
        "randomized_training": ("DLinear", "TSMixer", "PatchTST", "GRU", "ModernTCN")
    }


def test_run_testing_parser_rejects_removed_method_flags():
    with pytest.raises(SystemExit):
        run_testing.prepare_testing_entrypoint_args(["--improvement-method", "ensemble"])
    with pytest.raises(SystemExit):
        run_testing.prepare_testing_entrypoint_args(["--skip-recipes", "revin"])


@pytest.mark.parametrize(
    "argv",
    [
        ["--method", "not_a_method"],
        ["--method", "f2f"],
        ["--method", "all"],
        ["--method", "baseline", "Baseline"],
        ["--model", "LSTM"],
        ["--model", "Chronos2"],
        ["--method", "revin", "--model", "PatchTST"],
        ["--method", "revin", "--benchmark-architectures", "DLinear", "PatchTST"],
        ["--method", "adversarial_training", "--model", "Chronos2"],
    ],
)
def test_run_testing_rejects_bad_method_scope_before_selection(monkeypatch, argv):
    monkeypatch.setattr(run_testing, "load_defaults", load_defaults)
    monkeypatch.setattr(
        run_testing,
        "load_dataset_windows",
        lambda **_kw: {
            dataset: {"input_len": 96, "target_len": 96}
            for dataset in load_defaults()["DATA_FILES"]
        },
    )

    with pytest.raises(ValueError):
        run_testing.prepare_testing_entrypoint_args(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["--method", "not_a_method"],
        ["--method", "f2f"],
        ["--method", "all"],
        ["--method", "baseline", "Baseline"],
        ["--model", "LSTM"],
        ["--model", "Chronos2"],
        ["--method", "revin", "--model", "PatchTST"],
        ["--method", "adversarial_training", "--model", "Chronos2"],
    ],
)
def test_run_analysis_rejects_bad_method_scope_before_meta_analysis(argv):
    with pytest.raises(ValueError):
        run_analysis.prepare_analysis_entrypoint_args(argv)


def test_run_analysis_accepts_recipe_param_overrides_for_coverage_recompute():
    args = run_analysis.prepare_analysis_entrypoint_args(
        [
            "--method",
            "randomized_smoothing",
            "--model",
            "GRU",
            "--rs-noise-std",
            "1.0",
        ]
    )

    assert args.rs_noise_std == 1.0
    assert args._recipe_param_overrides["rs_noise_std"] == 1.0


def test_shared_applicability_validation_rejects_explicit_bad_pair():
    with pytest.raises(ValueError, match="revin/PatchTST"):
        resolve_benchmark_method_architecture_scope(
            methods=["revin"],
            architectures=["PatchTST"],
            explicit_architectures=True,
            context="test",
        )
    assert resolve_benchmark_method_architecture_scope(
        methods=["randomized_training"],
        architectures=BENCHMARK_ARCHITECTURES,
        explicit_architectures=False,
        context="test",
    ) == {
        "randomized_training": ("DLinear", "TSMixer", "PatchTST", "GRU", "ModernTCN")
    }
    with pytest.raises(ValueError, match="adversarial_training/Chronos2"):
        resolve_benchmark_method_architecture_scope(
            methods=["baseline", "adversarial_training"],
            architectures=["Chronos2"],
            explicit_architectures=True,
            context="test",
        )
