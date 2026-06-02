from types import SimpleNamespace

import pytest

import models
import run_testing
import run_training
from config_loader import build_parser, load_benchmark_scope, load_defaults, load_hparams
from improvements import (
    get_improvement_registration,
    get_registration_by_loader_kind,
    list_available_improvements,
)
from improvements.base import BackboneReference, ImprovementSpec as ImprovementMetadataSpec
from pipelines.recipes import PIPELINE_RECIPE_PATHS_BY_METHOD, load_pipeline_spec_for_method
from pipelines.runner import PipelineRunner
from pipelines.selection import resolve_requested_architectures, resolve_requested_methods
from pipelines.specs import PipelineSpec
from testing.evaluation import (
    _resolve_model_loading_identity_for_run,
    load_model_with_loader,
)


_BENCHMARK_SCOPE = load_benchmark_scope()
BENCHMARK_ARCHITECTURES = set(_BENCHMARK_SCOPE["architectures"]["display_order"])
BENCHMARK_METHODS = set(_BENCHMARK_SCOPE["methods"]["display_order"])


def test_benchmark_architecture_discovery_matches_benchmark_scope():
    hparams = load_hparams()

    assert set(hparams) == BENCHMARK_ARCHITECTURES
    assert set(models.__all__) == BENCHMARK_ARCHITECTURES


def test_unknown_architecture_raises_through_normal_scope_parser():
    with pytest.raises(ValueError, match="Unsupported MODEL\\[0\\] 'NotAnArchitecture'"):
        resolve_requested_architectures(
            SimpleNamespace(
                model=["NotAnArchitecture"],
                benchmark_architectures=[],
            )
        )


def test_benchmark_method_discovery_matches_configured_scope():
    assert set(PIPELINE_RECIPE_PATHS_BY_METHOD) == BENCHMARK_METHODS
    assert set(list_available_improvements()) == {"ensemble", "randomized_smoothing"}

    with pytest.raises(ValueError, match="Unknown pipeline_method 'not_a_method'"):
        load_pipeline_spec_for_method("not_a_method")
    with pytest.raises(KeyError):
        get_improvement_registration("not_an_improvement")


def test_unknown_method_raises_through_normal_scope_parser():
    with pytest.raises(ValueError, match="Unsupported METHOD\\[0\\] 'not_a_method'"):
        resolve_requested_methods(
            SimpleNamespace(
                method=["not_a_method"],
                benchmark_methods=list(BENCHMARK_METHODS),
            ),
            configured_methods=tuple(BENCHMARK_METHODS),
        )


def test_run_training_rejects_unknown_method_before_training(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_training.py", "--method", "not_a_method"])

    with pytest.raises(ValueError, match="Unsupported METHOD\\[0\\] 'not_a_method'"):
        run_training.main()


def test_run_testing_rejects_unknown_method_before_selection():
    with pytest.raises(ValueError, match="Unsupported METHOD\\[0\\] 'not_a_method'"):
        run_testing.prepare_testing_entrypoint_args(["--method", "not_a_method"])


def test_benchmark_entrypoint_defaults_expose_current_scope_controls():
    defaults = load_defaults()
    parser = build_parser(defaults)

    assert defaults["METHOD"] is None
    assert defaults["MODEL"] is None
    assert parser.parse_args([]).method is None
    assert parser.parse_args(["--method", "baseline"]).method == ["baseline"]
    assert parser.parse_args(["--model", "GRU"]).model == ["GRU"]


def test_unknown_loader_kind_fails_as_unknown_loader():
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="unknown-loader"),
        data=SimpleNamespace(
            tags={
                "model_architecture": "GRU",
                "loader_kind": "UnknownLoader",
                "pipeline_id": "baseline",
                "pipeline_method": "baseline",
                "pipeline_kind": "train",
                "robustness_method": "baseline",
            },
        ),
    )

    with pytest.raises(ValueError, match="Unknown loader_kind 'UnknownLoader'"):
        load_model_with_loader(object(), run, SimpleNamespace())


def test_model_loading_identity_requires_loader_kind_tag():
    with pytest.raises(ValueError, match="missing required loader_kind"):
        _resolve_model_loading_identity_for_run(
            {"model_architecture": "GRU"},
            run_id="missing-loader-kind",
        )


def test_invalid_pipeline_kind_is_not_available():
    with pytest.raises(ValueError, match="Invalid pipeline_kind"):
        PipelineSpec(
            pipeline_id="not_a_pipeline",
            pipeline_method="not_a_method",
            pipeline_kind="unsupported_kind",
            recipe_params={},
            model_hparams_mode="inherit_baseline",
        )


def test_unknown_loader_kind_is_not_registered():
    with pytest.raises(KeyError):
        get_registration_by_loader_kind("UnknownLoader")


def test_backbone_reference_rejects_unsupported_metadata_keys():
    payload = {
        "run_id": "run-1",
        "run_name": "baseline_GRU",
        "dataset": "ETTh1",
        "model_architecture": "GRU",
        "pipeline_id": "baseline",
        "pipeline_method": "baseline",
        "pipeline_kind": "train",
        "data_config_signature": "sig",
        "unexpected_field": "value",
    }

    with pytest.raises(ValueError, match="unsupported key\\(s\\): unexpected_field"):
        BackboneReference.from_dict(payload)


def test_improvement_spec_rejects_unsupported_metadata_keys():
    payload = {
        "parameters": {},
        "backbones": [],
        "unexpected_field": "value",
    }

    with pytest.raises(ValueError, match="unsupported key\\(s\\): unexpected_field"):
        ImprovementMetadataSpec.from_dict(payload)


def test_baseline_hparams_json_rejects_invalid_json(tmp_path):
    hparams_path = tmp_path / "hparams.json"
    hparams_path.write_text("{", encoding="utf-8")

    class _Client:
        def download_artifacts(self, run_id, artifact_path, dst_path=None):
            assert run_id == "bad-hparams"
            assert artifact_path == "hparams.json"
            assert dst_path is not None
            return str(hparams_path)

    run = SimpleNamespace(info=SimpleNamespace(run_id="bad-hparams"))
    runner = PipelineRunner(SimpleNamespace(), SimpleNamespace())

    with pytest.raises(ValueError, match="hparams.json is not valid JSON"):
        runner._extract_hparams_from_run(_Client(), run, "GRU")


def test_baseline_hparams_json_is_required_even_when_params_exist():
    class _Client:
        def download_artifacts(self, run_id, artifact_path, dst_path=None):
            assert run_id == "missing-hparams"
            assert artifact_path == "hparams.json"
            assert dst_path is not None
            raise FileNotFoundError("no hparams artifact")

    run = SimpleNamespace(
        info=SimpleNamespace(run_id="missing-hparams"),
        data=SimpleNamespace(params={"d_hidden": "64"}),
    )
    runner = PipelineRunner(SimpleNamespace(), SimpleNamespace())

    with pytest.raises(ValueError, match="missing required hparams.json"):
        runner._extract_hparams_from_run(_Client(), run, "GRU")


def test_baseline_hparams_json_rejects_unreadable_artifact(tmp_path):
    class _Client:
        def download_artifacts(self, run_id, artifact_path, dst_path=None):
            assert run_id == "unreadable-hparams"
            assert artifact_path == "hparams.json"
            assert dst_path is not None
            return str(tmp_path)

    run = SimpleNamespace(info=SimpleNamespace(run_id="unreadable-hparams"))
    runner = PipelineRunner(SimpleNamespace(), SimpleNamespace())

    with pytest.raises(ValueError, match="hparams.json could not be read"):
        runner._extract_hparams_from_run(_Client(), run, "GRU")


def test_baseline_hparams_json_rejects_non_mapping_payload(tmp_path):
    hparams_path = tmp_path / "hparams.json"
    hparams_path.write_text("[1, 2, 3]", encoding="utf-8")

    class _Client:
        def download_artifacts(self, run_id, artifact_path, dst_path=None):
            assert run_id == "list-hparams"
            assert artifact_path == "hparams.json"
            assert dst_path is not None
            return str(hparams_path)

    run = SimpleNamespace(info=SimpleNamespace(run_id="list-hparams"))
    runner = PipelineRunner(SimpleNamespace(), SimpleNamespace())

    with pytest.raises(ValueError, match="hparams.json must contain a mapping"):
        runner._extract_hparams_from_run(_Client(), run, "GRU")


def test_baseline_hparams_json_requires_current_architecture_keys(tmp_path):
    hparams_path = tmp_path / "hparams.json"
    hparams_path.write_text(
        '{"d_hidden": 64, "n_layers": 1, "autoregressive": false, "dropout": 0.0}',
        encoding="utf-8",
    )

    class _Client:
        def download_artifacts(self, run_id, artifact_path, dst_path=None):
            assert run_id == "missing-current-keys"
            assert artifact_path == "hparams.json"
            assert dst_path is not None
            return str(hparams_path)

    run = SimpleNamespace(info=SimpleNamespace(run_id="missing-current-keys"))
    runner = PipelineRunner(SimpleNamespace(), SimpleNamespace())

    with pytest.raises(
        ValueError,
        match="missing keys.*lr.*avoid silently falling back to model defaults",
    ):
        runner._extract_hparams_from_run(_Client(), run, "GRU")


def test_baseline_hparams_json_reads_valid_mapping(tmp_path):
    hparams_path = tmp_path / "hparams.json"
    hparams_path.write_text(
        (
            '{"d_hidden": 64, "n_layers": 1, "autoregressive": false, '
            '"dropout": 0.0, "lr": 0.001}'
        ),
        encoding="utf-8",
    )

    class _Client:
        def download_artifacts(self, run_id, artifact_path, dst_path=None):
            assert run_id == "valid-hparams"
            assert artifact_path == "hparams.json"
            assert dst_path is not None
            return str(hparams_path)

    run = SimpleNamespace(info=SimpleNamespace(run_id="valid-hparams"))
    runner = PipelineRunner(SimpleNamespace(), SimpleNamespace())

    assert runner._extract_hparams_from_run(_Client(), run, "GRU") == {
        "d_hidden": 64,
        "n_layers": 1,
        "autoregressive": False,
        "dropout": 0.0,
        "lr": 0.001,
    }
