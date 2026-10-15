import json
from pathlib import Path
from types import SimpleNamespace

import models
import yaml

from config_loader import load_defaults, load_hparams
from data.datasets import channel_order_tags, resolve_with_defaults
from improvements import get_registration_by_loader_kind
from models.chronos2 import Chronos2
from pipelines.ranking import (
    resolve_selection_score,
    validate_pipeline_tags_for_selection,
)
from pipelines.selection import classify_lineage_run as _classify_lineage_run
from pipelines.selection import load_benchmark_recipe_specs_for_scope
from testing.evaluation import (
    _resolve_model_loading_identity_for_run,
    load_model_with_loader,
)
from testing.shared import _require_best_model_current_tags
from utils.artifacts import download_best_checkpoint
from utils.parsing import (
    SHARED_ANCHOR_BOOTSTRAP_CI_SEMANTICS,
    parse_backbone_run_ids,
    require_degradation_eval_context_tags,
    require_robustness_results_complete_tag,
    require_shared_anchor_bootstrap_ci_context_tags,
    require_tested_param,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCOPE_PATH = REPO_ROOT / "configs" / "benchmark_scope.yaml"
COMPATIBILITY_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "current_run_compatibility.yaml"
)

BENCHMARK_DATASETS = (
    "BeijingAir_Tiantan",
    "Penmanshiel_Hourly_WT08",
    "ETTh1",
    "traffic",
)
BENCHMARK_ARCHITECTURES = (
    "DLinear",
    "GRU",
    "ModernTCN",
    "PatchTST",
    "TSMixer",
    "SeasonalNaive",
    "Chronos2",
)
METHOD_COMPARISON_ARCHITECTURES = (
    "DLinear",
    "GRU",
    "ModernTCN",
    "PatchTST",
    "TSMixer",
)
BASELINE_ONLY_ARCHITECTURES = ("SeasonalNaive", "Chronos2")
BENCHMARK_METHODS = (
    "baseline",
    "adversarial_training",
    "ensemble",
    "randomized_training",
    "randomized_smoothing",
    "adaptive_robust_loss",
    "revin",
    "fault_augmentation",
)
BENCHMARK_METHOD_RUN_ORDER = (
    "baseline",
    "randomized_training",
    "adversarial_training",
    "adaptive_robust_loss",
    "ensemble",
    "randomized_smoothing",
    "fault_augmentation",
    "revin",
)


class _HparamsArtifactClient:
    def __init__(self, hparams_by_run_id):
        self.hparams_by_run_id = dict(hparams_by_run_id)

    def download_artifacts(self, run_id, artifact_path, dst_path=None):
        assert artifact_path == "hparams.json"
        if run_id not in self.hparams_by_run_id:
            raise FileNotFoundError(f"missing hparams for {run_id}")
        path = Path(dst_path) / f"{run_id}_hparams.json"
        path.write_text(
            json.dumps(self.hparams_by_run_id[run_id]),
            encoding="utf-8",
        )
        return str(path)


def _hparams_from_run(run):
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


REQUIRED_METADATA_TAGS = (
    "dataset",
    "data_config_signature",
    "model_architecture",
    "loader_kind",
    "seed_master",
    "seed_data",
    "seed_model",
    "seed_eval",
    "seed_policy",
)
EXPECTED_FIXTURE_CATEGORIES = (
    "baseline",
    "pretrained",
    "randomized_smoothing",
    "revin",
    "train_kind",
    "wrap_kind",
)


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} must load as a mapping.")
    return payload


def _require_unique_nonempty_strings(values, *, key: str) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise AssertionError(f"{key} must be a list.")
    parsed = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in parsed):
        raise AssertionError(f"{key} must contain only non-empty strings.")
    if len(set(parsed)) != len(parsed):
        raise AssertionError(f"{key} must not contain duplicates.")
    return parsed


def _load_benchmark_scope() -> dict:
    return _load_yaml(BENCHMARK_SCOPE_PATH)


def _load_compatibility_fixtures() -> list[dict]:
    payload = _load_yaml(COMPATIBILITY_FIXTURE_PATH)
    if "runs" not in payload:
        raise AssertionError("Compatibility fixture is missing runs.")
    runs = payload["runs"]
    if not isinstance(runs, list) or not runs:
        raise AssertionError("Compatibility fixture must define a non-empty runs list.")
    return runs


def _run_from_fixture(record: dict) -> SimpleNamespace:
    if "params" not in record:
        raise AssertionError(f"Compatibility fixture {record['id']} is missing params.")
    params = record["params"]
    if not isinstance(params, dict):
        raise AssertionError(f"Compatibility fixture {record['id']} params must be a mapping.")
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id=record["id"],
            status="FINISHED",
            lifecycle_stage="active",
            end_time=1000,
        ),
        data=SimpleNamespace(
            tags=dict(record["tags"]),
            metrics=dict(record["metrics"]),
            params=dict(params),
        ),
    )


def _records_by_id() -> dict[str, dict]:
    return {record["id"]: record for record in _load_compatibility_fixtures()}


def _assert_mlflow_string_metadata(
    values: dict,
    *,
    record_id: str,
    store_name: str,
) -> None:
    for key, value in values.items():
        assert isinstance(value, str), (
            f"{record_id} {store_name}.{key} must be an MLflow-compatible string."
        )
        assert value.strip(), f"{record_id} {store_name}.{key} must be non-empty."


def _expected_dispatch_for_record(record: dict) -> tuple[str, str]:
    tags = record["tags"]
    loader_kind = tags["loader_kind"]
    if loader_kind == "pretrained":
        return ("pretrained", tags["model_architecture"])
    if tags["pipeline_kind"] == "wrap":
        return ("wrap", loader_kind)
    return ("standard", tags["model_architecture"])


def test_benchmark_scope_manifest_shape_matches_benchmark_runtime_defaults():
    manifest = _load_benchmark_scope()
    defaults = load_defaults()

    assert set(manifest) == {
        "schema_version",
        "datasets",
        "architectures",
        "methods",
    }
    assert manifest["schema_version"] == 1

    datasets = manifest["datasets"]
    dataset_order = _require_unique_nonempty_strings(
        datasets["display_order"],
        key="datasets.display_order",
    )
    assert dataset_order == BENCHMARK_DATASETS
    assert set(datasets["display_labels"]) == set(dataset_order)
    assert set(defaults["DATA_FILES"]) == set(dataset_order)

    architectures = manifest["architectures"]
    architecture_order = _require_unique_nonempty_strings(
        architectures["display_order"],
        key="architectures.display_order",
    )
    assert architecture_order == BENCHMARK_ARCHITECTURES
    assert set(architectures["display_labels"]) == set(architecture_order)
    assert set(defaults["BENCHMARK_ARCHITECTURES"]) == set(architecture_order)

    methods = manifest["methods"]
    method_order = _require_unique_nonempty_strings(
        methods["display_order"],
        key="methods.display_order",
    )
    assert method_order == BENCHMARK_METHODS
    assert set(methods["display_labels"]) == set(method_order)
    assert set(methods["method_architecture_applicability"]) == set(method_order)

    for section_name, section in (
        ("datasets", datasets),
        ("architectures", architectures),
        ("methods", methods),
    ):
        labels = section["display_labels"]
        assert all(
            isinstance(label, str) and label.strip()
            for label in labels.values()
        ), f"{section_name}.display_labels must be non-empty strings."


def test_shipped_benchmark_methods_keep_run_order():
    assert tuple(load_defaults()["BENCHMARK_METHODS"]) == BENCHMARK_METHOD_RUN_ORDER


def test_benchmark_scope_architecture_roles_partition_benchmark_architectures():
    manifest = _load_benchmark_scope()
    roles = manifest["architectures"]["roles"]

    method_comparison = _require_unique_nonempty_strings(
        roles["method_comparison"],
        key="architectures.roles.method_comparison",
    )
    baseline_only = _require_unique_nonempty_strings(
        roles["baseline_only_reference"],
        key="architectures.roles.baseline_only_reference",
    )

    assert method_comparison == METHOD_COMPARISON_ARCHITECTURES
    assert baseline_only == BASELINE_ONLY_ARCHITECTURES
    assert set(method_comparison).isdisjoint(baseline_only)
    assert set(method_comparison) | set(baseline_only) == set(BENCHMARK_ARCHITECTURES)


def test_benchmark_scope_method_architecture_applicability_matrix():
    manifest = _load_benchmark_scope()
    applicability = manifest["methods"]["method_architecture_applicability"]
    method_architectures = set(METHOD_COMPARISON_ARCHITECTURES)

    assert tuple(applicability["baseline"]) == BENCHMARK_ARCHITECTURES
    for method in BENCHMARK_METHODS:
        architectures = tuple(applicability[method])
        assert len(set(architectures)) == len(architectures)
        if method == "baseline":
            continue
        if method == "revin":
            assert set(architectures) == method_architectures - {"PatchTST"}
        else:
            assert set(architectures) == method_architectures
        assert set(architectures).isdisjoint(BASELINE_ONLY_ARCHITECTURES)


def test_current_run_compatibility_fixtures_cover_required_paths():
    runs = _load_compatibility_fixtures()

    categories = {record["category"] for record in runs}
    assert categories == set(EXPECTED_FIXTURE_CATEGORIES)

    methods = {record["tags"]["pipeline_method"] for record in runs}
    assert methods == set(BENCHMARK_METHODS)
    datasets = {record["tags"]["dataset"] for record in runs}
    assert datasets == set(BENCHMARK_DATASETS)
    baseline_architectures = {
        record["tags"]["model_architecture"]
        for record in runs
        if record["tags"]["pipeline_method"] == "baseline"
    }
    assert baseline_architectures == set(BENCHMARK_ARCHITECTURES)


def test_current_run_compatibility_pipeline_ids_match_benchmark_recipe_grids():
    specs_by_method = {
        str(spec.pipeline_method): spec
        for spec in load_benchmark_recipe_specs_for_scope(load_defaults())
    }
    pipeline_ids_by_method = {
        method: {
            spec.format_pipeline_id(param_values)
            for param_values in spec.expand_params()
        }
        for method, spec in specs_by_method.items()
    }

    for record in _load_compatibility_fixtures():
        tags = record["tags"]
        method = tags["pipeline_method"]
        pipeline_id = tags["pipeline_id"]
        assert method in pipeline_ids_by_method
        assert pipeline_id in pipeline_ids_by_method[method]


def test_current_run_compatibility_fixtures_preserve_required_metadata():
    records = _records_by_id()
    for record in records.values():
        run = _run_from_fixture(record)
        tags = run.data.tags
        params = run.data.params
        _assert_mlflow_string_metadata(
            tags,
            record_id=run.info.run_id,
            store_name="tags",
        )
        _assert_mlflow_string_metadata(
            params,
            record_id=run.info.run_id,
            store_name="params",
        )
        for key in REQUIRED_METADATA_TAGS:
            assert key in tags
            assert str(tags[key]).strip()
        assert require_tested_param(params, run_id=run.info.run_id) is True
        assert require_robustness_results_complete_tag(
            tags,
            run_id=run.info.run_id,
        ) is True

        degradation_context = require_degradation_eval_context_tags(
            tags,
            run_id=run.info.run_id,
            expected_test_metric="MSE",
        )
        assert degradation_context["n_test_samples"] == 3000
        assert degradation_context["eval_data_seed"] == int(tags["eval_data_seed"])
        assert degradation_context["eval_data_seed"] == int(tags["seed_data"])

        bootstrap_context = require_shared_anchor_bootstrap_ci_context_tags(
            tags,
            run_id=run.info.run_id,
            require_seed=True,
        )
        assert (
            bootstrap_context["bootstrap_ci_semantics"]
            == SHARED_ANCHOR_BOOTSTRAP_CI_SEMANTICS
        )

        pipeline_method = tags["pipeline_method"]
        if pipeline_method == "baseline":
            continue
        if tags["pipeline_kind"] == "wrap":
            assert tags["stage"] == "improve"
            assert tags["base_pipeline_method"] == "baseline"
            assert "backbone_run_id" in tags or "backbone_run_ids" in tags
            for backbone_id in parse_backbone_run_ids(
                tags.get("backbone_run_ids"),
                run_id=run.info.run_id,
            ):
                assert backbone_id in records
            if "backbone_run_id" in tags:
                assert tags["backbone_run_id"] in records
        else:
            assert tags["stage"] == "train"
            assert tags["base_pipeline_method"] == "baseline"
            assert tags["hparams_mode"] == "inherit_baseline"
            assert str(tags["baseline_hparams_run_id"]).strip()
            assert tags["baseline_hparams_run_id"] in records


def test_current_run_fixtures_classify_as_current_lineage_candidates():
    records = _records_by_id()
    baseline_run = _run_from_fixture(records["kept_baseline_gru"])
    ensemble_backbone_ids = parse_backbone_run_ids(
        records["kept_ensemble_gru"]["tags"]["backbone_run_ids"],
        run_id="kept_ensemble_gru",
    )
    ensemble_backbone_runs = [
        _run_from_fixture(records[backbone_id])
        for backbone_id in ensemble_backbone_ids
    ]
    current_base_runs_by_key = {("GRU", "baseline"): baseline_run}
    sorted_base_runs_by_key = {("GRU", "baseline"): ensemble_backbone_runs}
    scoped_train_run_ids_by_key = {
        ("GRU", "baseline"): {run.info.run_id for run in ensemble_backbone_runs}
    }
    baseline_hparam_specs_by_arch = {"GRU": load_hparams()["GRU"]}

    for record in records.values():
        if record["tags"]["pipeline_method"] == "baseline":
            continue
        run = _run_from_fixture(record)

        assert (
            classify_lineage_run(
                run,
                "GRU",
                current_base_runs_by_key=current_base_runs_by_key,
                sorted_base_runs_by_key=sorted_base_runs_by_key,
                baseline_hparam_specs_by_arch=baseline_hparam_specs_by_arch,
                scoped_train_run_ids_by_key=scoped_train_run_ids_by_key,
            )
            is None
        )

    stale_hparams_record = {
        **records["kept_randomized_training_gru"],
        "id": "stale_randomized_training_gru",
        "tags": {
            **records["kept_randomized_training_gru"]["tags"],
            "d_hidden": "256",
        },
    }
    stale_run = _run_from_fixture(stale_hparams_record)

    assert (
        classify_lineage_run(
            stale_run,
            "GRU",
            current_base_runs_by_key=current_base_runs_by_key,
            sorted_base_runs_by_key=sorted_base_runs_by_key,
            baseline_hparam_specs_by_arch=baseline_hparam_specs_by_arch,
            scoped_train_run_ids_by_key=scoped_train_run_ids_by_key,
        )
        == "baseline_changed"
    )


def test_current_run_fixtures_are_selectable_through_existing_metadata():
    for record in _load_compatibility_fixtures():
        run = _run_from_fixture(record)
        tags = run.data.tags
        resolved = validate_pipeline_tags_for_selection(tags, run_id=run.info.run_id)
        _require_best_model_current_tags(tags, run_id=run.info.run_id)
        model_architecture, loader_kind = _resolve_model_loading_identity_for_run(
            tags,
            run_id=run.info.run_id,
        )

        selection_score = resolve_selection_score(
            metrics=run.data.metrics,
            pipeline_kind=resolved["pipeline_kind"],
            robustness_method=resolved["robustness_method"],
            test_metric="MSE",
            improvement_selection_mode="clean",
            run_id=run.info.run_id,
        )
        expected_metric = (
            "MSE_val" if resolved["pipeline_kind"] == "wrap" else "best_val_loss"
        )

        assert model_architecture == tags["model_architecture"]
        assert loader_kind == tags["loader_kind"]
        assert selection_score == run.data.metrics[expected_metric]


class _ArtifactClient:
    def __init__(self, tmp_path: Path, records: list[dict]):
        self._top_level: dict[str, list[SimpleNamespace]] = {}
        self._downloads: dict[tuple[str, str], str] = {}
        for record in records:
            run_id = record["id"]
            layout = record["artifact_layout"]
            if layout == "root_epoch_checkpoint":
                artifact_path = "epoch=010-ep_val_loss=0.100"
                checkpoint_dir = tmp_path / run_id / artifact_path
                checkpoint_dir.mkdir(parents=True)
                (checkpoint_dir / "best.ckpt").write_text(run_id, encoding="utf-8")
                self._top_level[run_id] = [
                    SimpleNamespace(is_dir=True, path=artifact_path)
                ]
                self._downloads[(run_id, artifact_path)] = str(checkpoint_dir)
            elif layout == "model_checkpoints":
                artifact_path = "model/checkpoints"
                checkpoint_dir = tmp_path / run_id / artifact_path
                checkpoint_dir.mkdir(parents=True)
                (checkpoint_dir / "best.ckpt").write_text(run_id, encoding="utf-8")
                self._top_level[run_id] = []
                self._downloads[(run_id, artifact_path)] = str(checkpoint_dir)
            elif layout == "pretrained_snapshot":
                artifact_path = Chronos2.pretrained_artifact_path
                snapshot_dir = tmp_path / run_id / artifact_path
                snapshot_dir.mkdir(parents=True)
                self._top_level[run_id] = []
                self._downloads[(run_id, artifact_path)] = str(snapshot_dir)
            else:
                raise AssertionError(f"Unsupported artifact layout '{layout}'.")

    def list_artifacts(self, run_id: str):
        return list(self._top_level[run_id])

    def download_artifacts(
        self,
        run_id: str,
        artifact_path: str,
        dst_path: str | None = None,
    ) -> str:
        del dst_path
        key = (run_id, artifact_path)
        if key not in self._downloads:
            raise FileNotFoundError(artifact_path)
        return self._downloads[key]


def test_current_run_fixture_checkpoints_resolve_existing_artifact_layouts(tmp_path):
    runs = _load_compatibility_fixtures()
    client = _ArtifactClient(tmp_path, runs)

    for record in runs:
        if record["artifact_layout"] == "pretrained_snapshot":
            continue
        checkpoint_path = Path(download_best_checkpoint(client, record["id"]))

        assert checkpoint_path.name == "best.ckpt"
        assert checkpoint_path.read_text(encoding="utf-8") == record["id"]


def test_current_run_fixtures_dispatch_through_existing_loader_contracts(
    monkeypatch,
    tmp_path,
):
    runs = _load_compatibility_fixtures()
    client = _ArtifactClient(tmp_path, runs)
    captured_dispatch: dict[str, tuple[str, str]] = {}
    from models.pretrained_loader import PRETRAINED_MODEL_LOADERS

    assert get_registration_by_loader_kind("Ensemble").name == "ensemble"
    assert (
        get_registration_by_loader_kind("RandomizedSmoothing").name
        == "randomized_smoothing"
    )
    assert callable(PRETRAINED_MODEL_LOADERS["chronos2"])

    class _LoadedModel:
        pass

    def _fake_standard_loader(_client, run, target_class):
        captured_dispatch[run.info.run_id] = ("standard", target_class)
        return _LoadedModel(), "/tmp/standard-root"

    def _fake_registration(loader_kind):
        if loader_kind not in {"Ensemble", "RandomizedSmoothing"}:
            raise KeyError(loader_kind)

        def _builder(_client, run):
            captured_dispatch[run.info.run_id] = ("wrap", loader_kind)
            return _LoadedModel(), f"/tmp/{loader_kind}"

        return SimpleNamespace(builder=_builder)

    def _fake_chronos_loader(scoped_client, run, _args, _dm):
        snapshot_dir = scoped_client.download_artifacts(
            run.info.run_id,
            Chronos2.pretrained_artifact_path,
        )
        captured_dispatch[run.info.run_id] = ("pretrained", "Chronos2")
        return _LoadedModel(), snapshot_dir

    monkeypatch.setattr("testing.evaluation._load_model_from_run", _fake_standard_loader)
    monkeypatch.setattr(
        "testing.evaluation.get_registration_by_loader_kind",
        _fake_registration,
    )
    monkeypatch.setattr(
        "testing.evaluation.PRETRAINED_MODEL_LOADERS",
        {"chronos2": _fake_chronos_loader},
    )

    for record in runs:
        if _expected_dispatch_for_record(record)[0] == "standard":
            assert hasattr(models, record["tags"]["model_architecture"])
        run = _run_from_fixture(record)
        datamodule = None
        if record["category"] == "pretrained":
            dataset_spec = resolve_with_defaults("ETTh1", data_root=str(tmp_path))[0]
            datamodule = SimpleNamespace(dataset_spec=dataset_spec)
            run.data.tags.update(channel_order_tags(dataset_spec))
        model, default_root_dir = load_model_with_loader(
            client,
            run,
            SimpleNamespace(),
            dm=datamodule,
        )
        try:
            assert default_root_dir
            assert hasattr(model, "_artifact_tempdir_handle")
        finally:
            model._artifact_tempdir_handle.cleanup()

    expected_dispatch = {
        record["id"]: _expected_dispatch_for_record(record)
        for record in runs
    }
    assert captured_dispatch == expected_dispatch
