"""Runs the worked examples of the extension guides in docs/."""

from __future__ import annotations

import io
import re
import sys
import types
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from config_loader import load_benchmark_scope, load_core_figure_registry, load_defaults
from data.data_module import TSDataModule
from data.datasets.base import DatasetSpec
from models import DLinear
from pipelines.recipes import PIPELINE_RECIPE_PATHS_BY_METHOD
from pipelines.specs import PipelineSpec


REPO_ROOT = Path(__file__).resolve().parents[1]


def _block(path: str, marker: str, language: str) -> str:
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    block = re.search(
        rf"<!-- {marker}:start -->\s*```{language}\n(.*?)```\s*<!-- {marker}:end -->",
        text,
        re.S,
    )
    assert block is not None, marker
    lines = block.group(1).splitlines()
    indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
    return "\n".join(line[indent:] for line in lines) + "\n"


def _linear_forecaster_class() -> type:
    namespace: dict[str, object] = {}
    exec(_block("docs/extending_models.md", "model-example", "python"), namespace)
    return namespace["LinearForecaster"]


def _shape_kwargs(**overrides) -> dict[str, object]:
    kwargs = {
        "d_input_features": 4,
        "d_target_features": 1,
        "d_seq_in": 12,
        "d_seq_out": 3,
        "target_indices": (2,),
        "lr": 0.001,
    }
    return {**kwargs, **overrides}


def test_model_example_forecasts_targets_and_returns_a_mean_loss():
    linear_forecaster = _linear_forecaster_class()
    x = torch.randn(5, 12, 4)
    y = torch.randn(5, 3, 1)

    model = linear_forecaster(**_shape_kwargs())
    outputs = model._shared_step(x, y)

    assert outputs["pred"].shape == (5, 3, 1)
    assert outputs["loss"].ndim == 0
    assert model._shared_step(x, None)["loss"] is None
    assert torch.equal(model(x), outputs["pred"])

    separate_targets = linear_forecaster(**_shape_kwargs(d_target_features=2, target_indices=None))
    assert separate_targets(x).shape == (5, 3, 2)


def test_pretrained_model_example_keeps_its_weights_and_saves_a_checkpoint(tmp_path, monkeypatch):
    linear_forecaster = _linear_forecaster_class()
    module = types.ModuleType("models.linear_forecaster")
    module.LinearForecaster = linear_forecaster
    monkeypatch.setitem(sys.modules, "models.linear_forecaster", module)
    namespace: dict[str, object] = {}
    exec(_block("docs/extending_models.md", "pretrained-model-example", "python"), namespace)
    pretrained_linear = namespace["PretrainedLinear"]

    kwargs = _shape_kwargs()
    del kwargs["lr"]
    weights_path = tmp_path / "pretrained_linear.pt"
    torch.save(linear_forecaster(**_shape_kwargs()).state_dict(), weights_path)
    reference = torch.load(weights_path)
    model = pretrained_linear(weights_path=str(weights_path), **kwargs)
    windows = TensorDataset(torch.randn(16, 12, 4), torch.randn(16, 3, 1))
    checkpoint = ModelCheckpoint(dirpath=tmp_path / "checkpoints", monitor="ep_val_loss", mode="min")
    trainer = pl.Trainer(
        max_epochs=pretrained_linear.max_fit_epochs,
        accelerator="cpu",
        logger=False,
        callbacks=[checkpoint],
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, DataLoader(windows, batch_size=4), DataLoader(windows, batch_size=4))

    assert pretrained_linear.uses_base_optimizer is False
    assert trainer.global_step > 0
    saved = torch.load(checkpoint.best_model_path, weights_only=False)["state_dict"]
    for key, value in reference.items():
        assert torch.equal(model.state_dict()[key], value), key
        assert torch.equal(saved[key], value), key


def test_method_recipe_example_parses_and_registers_through_the_documented_owners(tmp_path):
    recipe_path = tmp_path / "mae_training.yaml"
    recipe_path.write_text(
        _block("docs/extending_methods.md", "method-recipe-example", "yaml"),
        encoding="utf-8",
    )
    spec = PipelineSpec.from_yaml(recipe_path)

    assert (spec.pipeline_method, spec.pipeline_kind, spec.pipeline_id) == (
        "mae_training",
        "train",
        "mae_training",
    )
    assert spec.model_hparams_mode == "inherit_baseline"
    DLinear(**_shape_kwargs(loss=spec.model_kwargs["loss"]))
    assert "baseline" in PIPELINE_RECIPE_PATHS_BY_METHOD
    assert "baseline" in load_defaults()["BENCHMARK_METHODS"]
    assert "CORE_METHOD_DISPLAY" in load_core_figure_registry()
    assert "families" in load_benchmark_scope()["architectures"]


def test_dataset_example_spec_matches_the_example_file():
    namespace: dict[str, object] = {"DatasetSpec": DatasetSpec}
    exec("spec = " + _block("docs/extending_datasets.md", "dataset-spec-example", "python").rstrip().rstrip(","), namespace)
    spec = namespace["spec"]
    frame = TSDataModule._set_csv_datetime_index(
        pd.read_csv(io.StringIO(_block("docs/extending_datasets.md", "dataset-file-example", "text")))
    )

    assert list(frame.columns) == list(spec.input_channels)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert set(spec.continuous_channels) | set(spec.discrete_channels) == set(spec.input_channels)
    assert frame[list(spec.discrete_channels)].dtypes.map(pd.api.types.is_integer_dtype).all()
    assert yaml.safe_load(
        (REPO_ROOT / "docs/extending_datasets.md").read_text(encoding="utf-8").split("```yaml\n")[1].split("```")[0]
    ) == {"MyPlant": {"input_len": 48, "target_len": 12, "batch_size": 32}}
