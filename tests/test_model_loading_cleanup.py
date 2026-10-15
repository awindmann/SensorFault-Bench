import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import improvements.ensemble as ensemble_module
import improvements.randomized_smoothing as rs_module
import models
import testing.evaluation as evaluation
from improvements.base import BackboneReference, ImprovementSpec
from testing.evaluation import (
    _ScopedArtifactClient,
    _evaluate_run,
    _load_model_from_run,
    ensure_selection_metric_for_run,
    load_model_with_loader,
)


class _ModelCleanupStub:
    def __init__(self):
        self.trainer = None
        self._logger = object()
        self._test_step_outputs = [object()]
        self.to_devices: list[str] = []
        self._artifact_tempdir_handle = None

    def to(self, device):
        self.to_devices.append(str(device))
        return self


def _attach_tempdir(model, tmp_path):
    tempdir = tempfile.TemporaryDirectory(dir=tmp_path)
    model._artifact_tempdir_handle = tempdir
    return Path(tempdir.name)


def _backbone_reference(*, run_id="backbone-run", pipeline_method="baseline"):
    return BackboneReference(
        run_id=run_id,
        run_name=run_id,
        dataset="dummy",
        model_architecture="GRU",
        pipeline_id=pipeline_method,
        pipeline_method=pipeline_method,
        pipeline_kind="train",
        data_config_signature="sig",
    )


def _wrap_run(run_id="wrap-run"):
    return SimpleNamespace(info=SimpleNamespace(run_id=run_id))


def test_scoped_artifact_client_uses_root_by_default_and_preserves_explicit_destination(
    tmp_path,
):
    root = tmp_path / "root"
    root.mkdir()
    download_calls: list[tuple[str, str]] = []

    class _Client:
        def download_artifacts(self, run_id, artifact_path, dst_path=None):
            download_calls.append((artifact_path, str(dst_path)))
            destination = Path(dst_path)
            destination.mkdir(parents=True, exist_ok=True)
            downloaded = destination / f"{artifact_path}.txt"
            downloaded.write_text(run_id, encoding="utf-8")
            return str(downloaded)

    scoped_client = _ScopedArtifactClient(_Client(), str(root))

    implicit_path = Path(scoped_client.download_artifacts("run-1", "implicit"))
    assert implicit_path == root / "implicit.txt"

    explicit_inside = root / "custom"
    explicit_path = Path(
        scoped_client.download_artifacts(
            "run-1",
            "explicit",
            dst_path=str(explicit_inside),
        )
    )
    assert explicit_path == explicit_inside / "explicit.txt"

    outside_download = tmp_path / "outside-download"
    outside_path = Path(
        scoped_client.download_artifacts(
            "run-1",
            "outside",
            dst_path=str(outside_download),
        )
    )
    assert outside_path == outside_download / "outside.txt"
    assert download_calls == [
        ("implicit", str(root)),
        ("explicit", str(explicit_inside)),
        ("outside", str(outside_download)),
    ]

    checkpoint_path = root / "nested" / "best.ckpt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    scoped_client.unlink_downloaded_checkpoint(
        checkpoint_path,
        run_id="run-1",
        context="standard loader",
    )

    assert not checkpoint_path.exists()

    outside_path = tmp_path / "outside.ckpt"
    outside_path.write_text("checkpoint", encoding="utf-8")
    with pytest.raises(ValueError, match="outside scoped artifact root"):
        scoped_client.unlink_downloaded_checkpoint(
            outside_path,
            run_id="run-1",
            context="standard loader",
        )
    assert outside_path.exists()


def test_load_model_from_run_unlinks_checkpoint_after_deserialization(
    monkeypatch,
    tmp_path,
):
    checkpoint_dir = tmp_path / "downloaded"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "best.ckpt"
    checkpoint_path.write_text("checkpoint", encoding="utf-8")
    scoped_client = _ScopedArtifactClient(SimpleNamespace(), str(tmp_path))
    run = SimpleNamespace(info=SimpleNamespace(run_id="run-1"))

    def _fake_download_best_checkpoint(client, run_id):
        assert client is scoped_client
        assert run_id == "run-1"
        return str(checkpoint_path)

    def _fake_load_checkpoint(model_class, path):
        assert model_class is models.GRU
        assert path == str(checkpoint_path)
        assert checkpoint_path.exists()
        return _ModelCleanupStub()

    monkeypatch.setattr(
        evaluation,
        "download_best_checkpoint",
        _fake_download_best_checkpoint,
    )
    monkeypatch.setattr(
        evaluation,
        "load_lightning_module_checkpoint",
        _fake_load_checkpoint,
    )

    model, default_root_dir = _load_model_from_run(scoped_client, run, "GRU")

    assert isinstance(model, _ModelCleanupStub)
    assert default_root_dir == str(checkpoint_dir)
    assert not checkpoint_path.exists()


def test_load_model_with_loader_cleans_scoped_tempdir_when_standard_load_raises(
    monkeypatch,
):
    captured: dict[str, Path] = {}

    class _Client:
        def download_artifacts(self, run_id, artifact_path, dst_path=None):
            assert run_id == "run-1"
            assert artifact_path == "model/checkpoints"
            assert dst_path is not None
            root = Path(dst_path)
            captured["root"] = root
            checkpoint_dir = root / "model" / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            checkpoint_path = checkpoint_dir / "best.ckpt"
            checkpoint_path.write_text("checkpoint", encoding="utf-8")
            return str(checkpoint_dir)

        def list_artifacts(self, run_id):
            assert run_id == "run-1"
            return []

    def _raise_checkpoint_load(_model_class, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        assert checkpoint_path.exists()
        raise RuntimeError("checkpoint load failed")

    monkeypatch.setattr(
        evaluation,
        "load_lightning_module_checkpoint",
        _raise_checkpoint_load,
    )
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="run-1"),
        data=SimpleNamespace(
            tags={
                "model_architecture": "GRU",
                "loader_kind": "GRU",
                "pipeline_id": "baseline",
                "pipeline_method": "baseline",
                "pipeline_kind": "train",
                "robustness_method": "baseline",
            }
        ),
    )

    with pytest.raises(RuntimeError, match="checkpoint load failed"):
        load_model_with_loader(_Client(), run, SimpleNamespace())

    assert "root" in captured
    assert not captured["root"].exists()


def test_build_ensemble_model_unlinks_downloaded_backbone_checkpoint(
    monkeypatch,
    tmp_path,
):
    references = [
        _backbone_reference(run_id="backbone-run-1"),
        _backbone_reference(
            run_id="backbone-run-2",
            pipeline_method="randomized_training",
        ),
    ]
    checkpoint_paths = []
    for idx, _reference in enumerate(references):
        checkpoint_path = tmp_path / f"best-{idx}.ckpt"
        checkpoint_path.write_text("checkpoint", encoding="utf-8")
        checkpoint_paths.append(checkpoint_path)
    client = _ScopedArtifactClient(SimpleNamespace(), str(tmp_path))
    run = _wrap_run()

    monkeypatch.setattr(
        ensemble_module,
        "load_improvement_spec",
        lambda _client, _run: ImprovementSpec(
            parameters={
                "ensemble_combine_method": "mean",
                "ensemble_top_k": "2",
                "ensemble_scope": "same_architecture",
            },
            backbones=references,
        ),
    )
    monkeypatch.setattr(
        ensemble_module,
        "resolve_wrap_backbone_references",
        lambda _run, _spec: references,
    )
    checkpoint_by_run_id = {
        reference.run_id: checkpoint_path
        for reference, checkpoint_path in zip(references, checkpoint_paths)
    }

    def _download_backbone_checkpoint(_client, _run, reference):
        return str(checkpoint_by_run_id[reference.run_id])

    monkeypatch.setattr(
        ensemble_module,
        "download_backbone_reference_checkpoint",
        _download_backbone_checkpoint,
    )

    def _fake_load_checkpoint(model_class, path):
        assert model_class is models.GRU
        checkpoint_path = Path(path)
        assert checkpoint_path in checkpoint_paths
        assert checkpoint_path.exists()
        return _ModelCleanupStub()

    class _FakeEnsemble:
        def __init__(self, *, backbones, combine_method):
            self.backbones = backbones
            self.combine_method = combine_method

    monkeypatch.setattr(
        ensemble_module,
        "load_lightning_module_checkpoint",
        _fake_load_checkpoint,
    )
    monkeypatch.setattr(ensemble_module, "EnsembleModel", _FakeEnsemble)

    model, default_root_dir = ensemble_module.build_ensemble_model(client, run)

    assert isinstance(model, _FakeEnsemble)
    assert model.combine_method == "mean"
    assert len(model.backbones) == 2
    assert default_root_dir
    assert all(not checkpoint_path.exists() for checkpoint_path in checkpoint_paths)


def test_build_randomized_smoothing_model_unlinks_downloaded_backbone_checkpoint(
    monkeypatch,
    tmp_path,
):
    reference = _backbone_reference(pipeline_method="baseline")
    checkpoint_path = tmp_path / "best.ckpt"
    checkpoint_path.write_text("checkpoint", encoding="utf-8")
    client = _ScopedArtifactClient(SimpleNamespace(), str(tmp_path))
    run = _wrap_run()

    monkeypatch.setattr(
        rs_module,
        "load_improvement_spec",
        lambda _client, _run: ImprovementSpec(
            parameters={
                "rs_noise_std": "0.1",
                "rs_sample_count": "4",
                "rs_backbone_method": "baseline",
                "rs_trim_alpha": "0.25",
            },
            backbones=[reference],
        ),
    )
    monkeypatch.setattr(
        rs_module,
        "resolve_wrap_backbone_references",
        lambda _run, _spec: [reference],
    )
    monkeypatch.setattr(
        rs_module,
        "download_backbone_reference_checkpoint",
        lambda _client, _run, _reference: str(checkpoint_path),
    )

    def _fake_load_checkpoint(model_class, path):
        assert model_class is models.GRU
        assert path == str(checkpoint_path)
        assert checkpoint_path.exists()
        return _ModelCleanupStub()

    class _FakeRandomizedSmoothing:
        def __init__(self, *, wrapped_backbone, noise_std, sample_count, trim_alpha):
            self.wrapped_backbone = wrapped_backbone
            self.noise_std = noise_std
            self.sample_count = sample_count
            self.trim_alpha = trim_alpha

    monkeypatch.setattr(
        rs_module,
        "load_lightning_module_checkpoint",
        _fake_load_checkpoint,
    )
    monkeypatch.setattr(rs_module, "RandomizedSmoothing", _FakeRandomizedSmoothing)

    model, default_root_dir = rs_module.build_randomized_smoothing_model(client, run)

    assert isinstance(model, _FakeRandomizedSmoothing)
    assert model.sample_count == 4
    assert default_root_dir
    assert not checkpoint_path.exists()


def test_load_model_with_loader_passes_scoped_cleanup_hook(monkeypatch, tmp_path):
    captured: dict[str, Path] = {}

    class _Client:
        def download_artifacts(self, run_id, artifact_path, dst_path=None):
            assert run_id == "run-1"
            assert artifact_path == "dummy"
            assert dst_path is not None
            root = Path(dst_path)
            captured["root"] = root
            artifact_path = root / "artifact.txt"
            artifact_path.write_text("artifact", encoding="utf-8")
            return str(artifact_path)

    def _builder(scoped_client, run):
        assert hasattr(scoped_client, "unlink_downloaded_checkpoint")
        artifact_path = Path(scoped_client.download_artifacts(run.info.run_id, "dummy"))
        return _ModelCleanupStub(), str(artifact_path.parent)

    monkeypatch.setattr(
        evaluation,
        "get_registration_by_loader_kind",
        lambda _loader_kind: SimpleNamespace(builder=_builder),
    )
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="run-1"),
        data=SimpleNamespace(
            tags={
                "model_architecture": "GRU",
                "loader_kind": "Ensemble",
            }
        ),
    )

    model, default_root_dir = load_model_with_loader(_Client(), run, SimpleNamespace())
    try:
        assert default_root_dir == str(captured["root"])
        assert captured["root"].exists()
        assert hasattr(model, "_artifact_tempdir_handle")
    finally:
        evaluation._teardown_model_after_eval(model)
    assert not captured["root"].exists()


def test_load_model_with_loader_cleans_scoped_tempdir_when_loader_raises(
    monkeypatch,
):
    captured: dict[str, Path] = {}

    class _Client:
        def download_artifacts(self, run_id, artifact_path, dst_path=None):
            assert run_id == "run-1"
            assert artifact_path == "dummy"
            assert dst_path is not None
            root = Path(dst_path)
            captured["root"] = root
            artifact_path = root / "artifact.txt"
            artifact_path.write_text("artifact", encoding="utf-8")
            return str(artifact_path)

    def _builder(scoped_client, run):
        scoped_client.download_artifacts(run.info.run_id, "dummy")
        raise RuntimeError("loader exploded")

    monkeypatch.setattr(
        evaluation,
        "get_registration_by_loader_kind",
        lambda _loader_kind: SimpleNamespace(builder=_builder),
    )
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="run-1"),
        data=SimpleNamespace(
            tags={
                "model_architecture": "GRU",
                "loader_kind": "Ensemble",
            }
        ),
    )

    with pytest.raises(RuntimeError, match="loader exploded"):
        load_model_with_loader(_Client(), run, SimpleNamespace())

    assert "root" in captured
    assert not captured["root"].exists()


def test_selection_metric_backfill_tears_down_when_perturbed_logging_raises(
    monkeypatch,
    tmp_path,
):
    model = _ModelCleanupStub()
    tempdir_path = _attach_tempdir(model, tmp_path)
    candidate_run = SimpleNamespace(
        info=SimpleNamespace(run_id="run-1"),
        data=SimpleNamespace(tags={}, metrics={}),
    )
    args = SimpleNamespace(
        test_metric="MSE",
        refresh_selection_metrics=True,
        rerun=False,
        perturbation_scenarios=("spike",),
    )

    monkeypatch.setattr(
        evaluation,
        "resolve_pipeline_tags",
        lambda _tags, *, run_id: {
            "pipeline_kind": "wrap",
            "robustness_method": "ensemble",
        },
    )
    monkeypatch.setattr(
        evaluation,
        "require_improvement_selection_mode",
        lambda _args, *, context: "perturbed_validation",
    )
    monkeypatch.setattr(
        evaluation,
        "selection_metric_key_for_kind",
        lambda **_kwargs: "MSE_pert_ws_val",
    )
    monkeypatch.setattr(
        evaluation,
        "perturbed_selection_metric_keys",
        lambda **_kwargs: ("MSE_pert_ws_val", "MSE_pert_mean_val"),
    )
    monkeypatch.setattr(
        evaluation,
        "require_selection_perturbation_context_from_args",
        lambda _args, *, context: {
            "selection_metric_semantics": "perturbed_validation_error"
        },
    )
    monkeypatch.setattr(
        evaluation,
        "build_perturbation_scenario_params_signature",
        lambda _scenarios: "scenario-sig",
    )

    def _raise_selection_metrics(*_args, **_kwargs):
        raise RuntimeError("selection metrics failed")

    monkeypatch.setattr(
        evaluation,
        "_log_perturbed_validation_selection_metrics",
        _raise_selection_metrics,
    )

    class _Client:
        def get_run(self, run_id):
            assert run_id == "run-1"
            return candidate_run

    with pytest.raises(RuntimeError, match="selection metrics failed"):
        ensure_selection_metric_for_run(
            candidate_run,
            arch="GRU",
            pipeline_method="ensemble",
            pipeline_id="ensemble_top2",
            args=args,
            client=_Client(),
            dataset_name="dummy",
            dm_holder={"selection": object()},
            dm_factory=lambda: object(),
            load_model_fn=lambda _run: (model, str(tmp_path)),
        )

    assert not tempdir_path.exists()
    assert model._artifact_tempdir_handle is None
    assert model._test_step_outputs == []
    assert model.to_devices == ["cpu"]


def test_evaluate_run_tears_down_when_seed_validation_fails(tmp_path):
    model = _ModelCleanupStub()
    tempdir_path = _attach_tempdir(model, tmp_path)
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="run-1"),
        data=SimpleNamespace(tags={}, metrics={}),
    )

    with pytest.raises(ValueError, match="missing seed tag"):
        _evaluate_run(
            run,
            model,
            str(tmp_path),
            "dummy",
            SimpleNamespace(),
            object(),
            eval_data_seed=1,
        )

    assert not tempdir_path.exists()
    assert model._artifact_tempdir_handle is None
    assert model._test_step_outputs == []
    assert model.to_devices == ["cpu"]
