import os
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, TensorDataset

from data.data_module import TSDataModule
from models.chronos2 import Chronos2
from models.pretrained_loader import load_chronos2_model
from pipelines.runner import scope_policy_skip_reason_for_spec
from pipelines.selection import resolve_requested_architectures
from pipelines.specs import PipelineSpec
from testing.evaluation import load_model_with_loader


@pytest.fixture
def fake_chronos2_pipeline(monkeypatch):
    class _FakeInnerModel:
        def __init__(self) -> None:
            self.bound_devices: list[str] = []
            self.eval_calls = 0
            self.weight = torch.nn.Parameter(torch.ones(1))

        def to(self, device):
            self.bound_devices.append(str(device))
            return self

        def parameters(self):
            return (self.weight,)

        def eval(self):
            self.eval_calls += 1
            return self

    class _FakeChronos2Pipeline:
        context_length = 16
        prediction_length = 8
        last_from_pretrained = None
        return_non_instance_for: set[str] = set()
        instances: list["_FakeChronos2Pipeline"] = []

        def __init__(self, source: str, **load_kwargs) -> None:
            self.source = source
            self.load_kwargs = dict(load_kwargs)
            self.model_context_length = type(self).context_length
            self.model_prediction_length = type(self).prediction_length
            self.model = _FakeInnerModel()
            self.predict_calls: list[dict[str, object]] = []
            self.saved_dirs: list[str] = []
            type(self).instances.append(self)

        @classmethod
        def from_pretrained(cls, source: str, **kwargs):
            cls.last_from_pretrained = (source, dict(kwargs))
            if source in cls.return_non_instance_for:
                return object()
            return cls(source, **kwargs)

        def predict_quantiles(
            self,
            inputs,
            prediction_length=None,
            quantile_levels=None,
            **predict_kwargs,
        ):
            self.predict_calls.append(
                {
                    "context_shape": tuple(inputs.shape),
                    "prediction_length": int(prediction_length),
                    "quantile_levels": tuple(float(level) for level in (quantile_levels or [])),
                    "context_length": predict_kwargs.get("context_length"),
                    "limit_prediction_length": bool(predict_kwargs.get("limit_prediction_length", False)),
                    "cross_learning": bool(predict_kwargs.get("cross_learning", False)),
                    "grad_enabled": bool(torch.is_grad_enabled()),
                }
            )
            batch_size, n_features, _ = inputs.shape
            mean = []
            for batch_index in range(batch_size):
                values = torch.arange(
                    n_features * int(prediction_length),
                    dtype=inputs.dtype,
                    device=inputs.device,
                ).reshape(n_features, int(prediction_length))
                mean.append(values + float(batch_index))
            return [], mean

        def save_pretrained(self, path: str) -> None:
            self.saved_dirs.append(path)
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as handle:
                handle.write("{\"model_type\": \"chronos2\"}")

    monkeypatch.setattr(
        "models.chronos2._import_chronos2_pipeline_class",
        lambda: _FakeChronos2Pipeline,
    )
    return _FakeChronos2Pipeline


def _base_kwargs() -> dict:
    return {
        "d_input_features": 3,
        "d_target_features": 2,
        "d_seq_in": 4,
        "d_seq_out": 2,
        "target_indices": (0, 2),
        "chronos_model_id": "amazon/chronos-2",
        "chronos_model_revision": "0f8a440441931157957e2be1a9bce66627d99c76",
    }


def _make_dm_stub() -> TSDataModule:
    dm = object.__new__(TSDataModule)
    dm.n_inputs = 3
    dm.n_outputs = 2
    dm.input_len = 4
    dm.target_len = 2
    dm.target_column_indices = (0, 2)
    return dm


class _TinyDataModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        x = torch.randn(4, 4, 3)
        y = torch.randn(4, 2, 2)
        self._train = TensorDataset(x, y)
        self._val = TensorDataset(x[:2], y[:2])

    def train_dataloader(self):
        return DataLoader(self._train, batch_size=2)

    def val_dataloader(self):
        return DataLoader(self._val, batch_size=2)


class _CountingChronos2(Chronos2):
    def __init__(self, **kwargs) -> None:
        self.training_batches_seen = 0
        super().__init__(**kwargs)

    def training_step(self, batch, batch_idx):
        self.training_batches_seen += 1
        return super().training_step(batch, batch_idx)


def test_chronos2_forward_shape_and_target_projection(fake_chronos2_pipeline):
    model = Chronos2(**_base_kwargs())
    x = torch.randn(2, model.d_seq_in, model.d_input_features)

    y_pred = model(x)

    assert y_pred.shape == (2, model.d_seq_out, model.d_target_features)
    fake_instance = fake_chronos2_pipeline.instances[-1]
    assert fake_instance.predict_calls[-1]["context_shape"] == (2, 3, 4)
    assert fake_instance.predict_calls[-1]["cross_learning"] is False
    assert fake_instance.predict_calls[-1]["limit_prediction_length"] is False


def test_chronos2_forward_does_not_revalidate_lengths(fake_chronos2_pipeline):
    model = Chronos2(**_base_kwargs())
    x = torch.randn(2, model.d_seq_in, model.d_input_features)

    def _boom():
        raise AssertionError("_validate_model_lengths should not run during forward.")

    model._validate_model_lengths = _boom

    y_pred = model(x)

    assert y_pred.shape == (2, model.d_seq_out, model.d_target_features)


def test_chronos2_freezes_loaded_pipeline(fake_chronos2_pipeline):
    model = Chronos2(**_base_kwargs())

    fake_instance = fake_chronos2_pipeline.instances[-1]

    assert fake_chronos2_pipeline.last_from_pretrained == (
        "amazon/chronos-2",
        {"revision": "0f8a440441931157957e2be1a9bce66627d99c76"},
    )
    assert fake_instance.model.weight.requires_grad is False
    assert fake_instance.model.eval_calls == 1


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"chronos_model_id": ""}, "chronos_model_id"),
        ({"chronos_model_revision": ""}, "chronos_model_revision"),
        (
            {"target_indices": None, "d_target_features": 2, "d_input_features": 3},
            "requires target channels to be present in inputs",
        ),
    ],
)
def test_chronos2_invalid_config_raises(fake_chronos2_pipeline, override, message):
    kwargs = _base_kwargs()
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        Chronos2(**kwargs)


def test_chronos2_rejects_non_chronos2_source(fake_chronos2_pipeline):
    fake_chronos2_pipeline.return_non_instance_for = {"not-chronos2"}
    kwargs = _base_kwargs()
    kwargs["chronos_model_id"] = "not-chronos2"

    with pytest.raises(ValueError, match="did not resolve to a Chronos2Pipeline"):
        Chronos2(**kwargs)


def test_chronos2_rejects_input_len_above_context_limit(fake_chronos2_pipeline):
    fake_chronos2_pipeline.context_length = 3

    with pytest.raises(ValueError, match="exceeds model_context_length"):
        Chronos2(**_base_kwargs())


def test_chronos2_rejects_target_len_above_prediction_limit(fake_chronos2_pipeline):
    fake_chronos2_pipeline.prediction_length = 1

    with pytest.raises(ValueError, match="exceeds model_prediction_length"):
        Chronos2(**_base_kwargs())


def test_chronos2_rejects_lr_scheduler(fake_chronos2_pipeline):
    kwargs = _base_kwargs()
    kwargs["lr_scheduler"] = True

    with pytest.raises(ValueError, match="does not support lr_scheduler=True"):
        Chronos2(**kwargs)


@pytest.mark.parametrize(
    "optimizer_key",
    ["optimizer", "lr", "beta1", "weight_decay", "scheduler_factor", "grad_clip"],
)
def test_chronos2_rejects_unsupported_optimizer_kwargs(
    fake_chronos2_pipeline,
    optimizer_key,
):
    kwargs = _base_kwargs()
    kwargs[optimizer_key] = 0.1

    with pytest.raises(ValueError, match=f"optimizer config '{optimizer_key}'"):
        Chronos2(**kwargs)


def test_chronos2_fit_logs_best_val_loss_and_snapshot(fake_chronos2_pipeline, tmp_path):
    tracking_dir = tmp_path / "mlruns"
    tracking_uri = f"file:{tracking_dir}"
    mlflow.set_tracking_uri(tracking_uri)
    logger = MLFlowLogger(
        tracking_uri=tracking_uri,
        experiment_name="chronos2-fit-test",
    )
    model = _CountingChronos2(**_base_kwargs())
    dm = _TinyDataModule()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=logger,
        max_epochs=5,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=dm)

    client = mlflow.MlflowClient()
    run = client.get_run(logger.run_id)
    artifacts = client.list_artifacts(logger.run_id, Chronos2.pretrained_artifact_path)

    assert "best_val_loss" in run.data.metrics
    assert artifacts
    assert model.training_batches_seen == 2
    assert model.automatic_optimization is False
    assert model.configure_optimizers() is None


def test_chronos2_training_step_uses_no_grad_for_pipeline(fake_chronos2_pipeline):
    model = Chronos2(**_base_kwargs())
    x = torch.randn(2, model.d_seq_in, model.d_input_features)
    y = torch.randn(2, model.d_seq_out, model.d_target_features)

    model.training_step((x, y), 0)

    fake_instance = fake_chronos2_pipeline.instances[-1]
    assert fake_instance.predict_calls[-1]["grad_enabled"] is False


def test_chronos2_on_fit_end_skips_snapshot_logging_in_test_mode(fake_chronos2_pipeline):
    model = Chronos2(**_base_kwargs())
    model.test_mode = True

    model.on_fit_end()


def test_chronos2_validation_start_rejects_multi_device(fake_chronos2_pipeline):
    model = Chronos2(**_base_kwargs())
    model._trainer = SimpleNamespace(
        strategy=SimpleNamespace(
            parallel_devices=[torch.device("cpu"), torch.device("cpu")],
        )
    )

    with pytest.raises(ValueError, match="supports exactly one runtime device"):
        model.on_validation_start()


def test_chronos2_validation_start_accepts_single_device_strategy(fake_chronos2_pipeline):
    """SingleDeviceStrategy in Lightning 2.6+ only has root_device, no parallel_devices."""
    model = Chronos2(**_base_kwargs())
    model._trainer = SimpleNamespace(
        strategy=SimpleNamespace(root_device=torch.device("cpu"))
    )
    model.on_validation_start()


def test_pretrained_loader_reconstructs_chronos2(fake_chronos2_pipeline, tmp_path):
    tracking_dir = tmp_path / "mlruns"
    tracking_uri = f"file:{tracking_dir}"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    exp_id = client.create_experiment("chronos2-loader-test")

    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")

    with mlflow.start_run(experiment_id=exp_id) as run:
        run_id = run.info.run_id
        client.log_artifacts(run_id, str(snapshot_dir), artifact_path=Chronos2.pretrained_artifact_path)
        client.log_param(run_id, "chronos_model_id", "amazon/chronos-2")
        client.log_param(run_id, "chronos_model_revision", "0f8a440441931157957e2be1a9bce66627d99c76")
        client.log_param(run_id, "d_seq_in", "4")
        client.log_param(run_id, "d_seq_out", "2")
        client.log_param(run_id, "target_indices", "[0, 2]")
        client.log_param(run_id, "loss", "MSE")
        client.set_tag(run_id, "loader_kind", "pretrained")
        client.set_tag(run_id, "model_architecture", "Chronos2")
        client.set_tag(run_id, "input_channel_count", "3")
        client.set_tag(run_id, "target_channel_count", "2")

    run = client.get_run(run_id)
    model, default_root_dir = load_chronos2_model(
        client,
        run,
        args=SimpleNamespace(),
        datamodule=_make_dm_stub(),
    )

    assert isinstance(model, Chronos2)
    assert model.chronos_model_id == "amazon/chronos-2"
    assert model.chronos_model_revision == "0f8a440441931157957e2be1a9bce66627d99c76"
    assert model.target_indices == (0, 2)
    assert Path(default_root_dir).is_dir()
    assert Path(default_root_dir) != Path(model._chronos_snapshot_path)
    assert fake_chronos2_pipeline.last_from_pretrained is not None
    assert fake_chronos2_pipeline.last_from_pretrained[1] == {"local_files_only": True}


def test_pretrained_loader_allows_missing_chronos_revision_for_snapshot(
    fake_chronos2_pipeline,
    tmp_path,
):
    tracking_dir = tmp_path / "mlruns"
    tracking_uri = f"file:{tracking_dir}"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    exp_id = client.create_experiment("chronos2-loader-missing-revision-test")

    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")

    with mlflow.start_run(experiment_id=exp_id) as run:
        run_id = run.info.run_id
        client.log_artifacts(
            run_id,
            str(snapshot_dir),
            artifact_path=Chronos2.pretrained_artifact_path,
        )
        client.log_param(run_id, "chronos_model_id", "amazon/chronos-2")
        client.log_param(run_id, "d_seq_in", "4")
        client.log_param(run_id, "d_seq_out", "2")
        client.log_param(run_id, "target_indices", "[0, 2]")
        client.log_param(run_id, "loss", "MSE")
        client.set_tag(run_id, "loader_kind", "pretrained")
        client.set_tag(run_id, "model_architecture", "Chronos2")
        client.set_tag(run_id, "input_channel_count", "3")
        client.set_tag(run_id, "target_channel_count", "2")

    run = client.get_run(run_id)
    model, _ = load_chronos2_model(
        client,
        run,
        args=SimpleNamespace(),
        datamodule=_make_dm_stub(),
    )

    assert isinstance(model, Chronos2)
    assert model.chronos_model_revision is None
    assert fake_chronos2_pipeline.last_from_pretrained is not None
    assert fake_chronos2_pipeline.last_from_pretrained[1] == {"local_files_only": True}


def test_pretrained_loader_requires_snapshot_artifact(fake_chronos2_pipeline, tmp_path):
    tracking_dir = tmp_path / "mlruns"
    tracking_uri = f"file:{tracking_dir}"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    exp_id = client.create_experiment("chronos2-loader-missing-artifact-test")

    with mlflow.start_run(experiment_id=exp_id) as run:
        run_id = run.info.run_id
        client.log_param(run_id, "chronos_model_id", "amazon/chronos-2")
        client.log_param(run_id, "d_seq_in", "4")
        client.log_param(run_id, "d_seq_out", "2")
        client.log_param(run_id, "target_indices", "[0, 2]")
        client.log_param(run_id, "loss", "MSE")
        client.set_tag(run_id, "loader_kind", "pretrained")
        client.set_tag(run_id, "model_architecture", "Chronos2")
        client.set_tag(run_id, "input_channel_count", "3")
        client.set_tag(run_id, "target_channel_count", "2")

    run = client.get_run(run_id)
    with pytest.raises(ValueError, match="missing required pretrained snapshot artifact"):
        load_chronos2_model(
            client,
            run,
            args=SimpleNamespace(),
            datamodule=_make_dm_stub(),
        )


def test_load_model_with_loader_rejects_unknown_pretrained_architecture():
    class _Client:
        def download_artifacts(self, run_id, artifact_path, dst_path=None):
            raise AssertionError("download_artifacts should not be called for unknown pretrained architectures.")

    run = SimpleNamespace(
        info=SimpleNamespace(run_id="run_123"),
        data=SimpleNamespace(
            tags={
                "loader_kind": "pretrained",
                "model_architecture": "UnknownFoundation",
            }
        ),
    )

    with pytest.raises(ValueError, match="Unknown pretrained model_architecture 'UnknownFoundation'"):
        load_model_with_loader(
            _Client(),
            run,
            SimpleNamespace(),
            dm=_make_dm_stub(),
        )


def test_load_model_with_loader_requires_datamodule_for_pretrained_runs():
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="run_123"),
        data=SimpleNamespace(
            tags={
                "loader_kind": "pretrained",
                "model_architecture": "Chronos2",
            }
        ),
    )

    with pytest.raises(ValueError, match="requires a datamodule for reconstruction"):
        load_model_with_loader(
            SimpleNamespace(),
            run,
            SimpleNamespace(),
        )


def test_chronos2_declares_pretrained_loader_kind():
    assert Chronos2.loader_kind == "pretrained"


def test_benchmark_architecture_scope_includes_chronos2(monkeypatch):
    monkeypatch.setattr(
        "pipelines.selection.load_hparams",
        lambda: {
            "GRU": {"lr": [0.001]},
            "Chronos2": {
                "chronos_model_id": ["amazon/chronos-2"],
                "chronos_model_revision": ["0f8a440441931157957e2be1a9bce66627d99c76"],
            },
        },
    )

    assert "Chronos2" in resolve_requested_architectures(
        SimpleNamespace(
            model=None,
            benchmark_architectures=["GRU", "Chronos2"],
        )
    )


def test_benchmark_architecture_scope_explicit_chronos2_ignores_benchmark_scope(monkeypatch):
    monkeypatch.setattr(
        "pipelines.selection.load_hparams",
        lambda: {
            "GRU": {"lr": [0.001]},
            "Chronos2": {
                "chronos_model_id": ["amazon/chronos-2"],
                "chronos_model_revision": ["0f8a440441931157957e2be1a9bce66627d99c76"],
            },
        },
    )

    assert resolve_requested_architectures(
        SimpleNamespace(model=["Chronos2"], benchmark_architectures=["GRU"])
    ) == ["Chronos2"]


def test_non_baseline_recipes_skip_chronos2_by_policy():
    spec = PipelineSpec.from_yaml(Path("configs/pipelines/randomized_training.yaml"))

    assert (
        scope_policy_skip_reason_for_spec(spec, "Chronos2")
        == "unsupported_benchmark_method_architecture"
    )
