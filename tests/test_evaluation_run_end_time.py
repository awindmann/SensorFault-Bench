"""Evaluation keeps the end time that breaks selection ties."""

from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
from mlflow import MlflowClient
from torch.utils.data import DataLoader, TensorDataset

import testing.evaluation as evaluation
from utils.parsing import build_mlflow_tracking_uri


class _TinyModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def validation_step(self, batch, _batch_idx):
        (x,) = batch
        self.log("MSE_val", self.layer(x).pow(2).mean())


def _store(tmp_path):
    logdir = str(tmp_path / "mlruns")
    client = MlflowClient(tracking_uri=build_mlflow_tracking_uri(logdir))
    experiment_id = client.create_experiment("toy")
    return SimpleNamespace(logdir=logdir), client, experiment_id


def test_eval_logger_keeps_run_end_time_after_validation(tmp_path):
    args, client, experiment_id = _store(tmp_path)
    run_id = client.create_run(experiment_id, start_time=500).info.run_id
    client.set_terminated(run_id, status="FINISHED", end_time=1_000)

    logger = evaluation._make_eval_logger(run_id, "toy", args, log_test_commit=False)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.validate(
        _TinyModule(),
        dataloaders=DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=2),
        verbose=False,
    )

    run = client.get_run(run_id)
    assert "MSE_val" in run.data.metrics
    assert run.info.end_time == 1_000
    assert run.info.status == "FINISHED"


def test_eval_logger_requires_run_end_time(tmp_path):
    args, client, experiment_id = _store(tmp_path)
    run_id = client.create_run(experiment_id).info.run_id

    with pytest.raises(ValueError, match="has no end time"):
        evaluation._make_eval_logger(run_id, "toy", args, log_test_commit=False)
