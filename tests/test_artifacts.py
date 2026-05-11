from types import SimpleNamespace

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

from utils.artifacts import download_best_checkpoint


def test_download_best_checkpoint_propagates_list_artifacts_errors():
    class _Client:
        def list_artifacts(self, run_id):
            raise RuntimeError("tracking server unavailable")

    with pytest.raises(RuntimeError, match="tracking server unavailable"):
        download_best_checkpoint(_Client(), "run-1")


def test_download_best_checkpoint_checks_model_checkpoint_dir_when_root_listing_is_missing(tmp_path):
    checkpoint_dir = tmp_path / "model-checkpoints"
    ckpt_path = checkpoint_dir / "best.ckpt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text("checkpoint", encoding="utf-8")
    calls: list[str] = []

    class _Client:
        def list_artifacts(self, run_id):
            raise MlflowException(
                "root listing missing",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        def download_artifacts(self, run_id, artifact_path, dst_path=None):
            calls.append(artifact_path)
            assert artifact_path == "model/checkpoints"
            return str(checkpoint_dir)

    resolved = download_best_checkpoint(_Client(), "run-1")

    assert resolved == str(ckpt_path)
    assert calls == ["model/checkpoints"]


def test_download_best_checkpoint_propagates_non_missing_epoch_download_errors():
    epoch_dir = SimpleNamespace(is_dir=True, path="epoch=001")

    class _Client:
        def list_artifacts(self, run_id):
            return [epoch_dir]

        def download_artifacts(self, run_id, artifact_path, dst_path=None):
            raise RuntimeError("artifact store unavailable")

    with pytest.raises(RuntimeError, match="artifact store unavailable"):
        download_best_checkpoint(_Client(), "run-1")


def test_download_best_checkpoint_checks_model_checkpoint_dir_on_missing_epoch_directory(tmp_path):
    epoch_dir = SimpleNamespace(is_dir=True, path="epoch=001")
    checkpoint_dir = tmp_path / "model-checkpoints"
    ckpt_path = checkpoint_dir / "best.ckpt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text("checkpoint", encoding="utf-8")
    calls: list[str] = []

    class _Client:
        def list_artifacts(self, run_id):
            return [epoch_dir]

        def download_artifacts(self, run_id, artifact_path, dst_path=None):
            calls.append(artifact_path)
            if artifact_path == "epoch=001":
                raise MlflowException(
                    "artifact missing",
                    error_code=RESOURCE_DOES_NOT_EXIST,
            )
            assert artifact_path == "model/checkpoints"
            return str(checkpoint_dir)

    resolved = download_best_checkpoint(_Client(), "run-1")

    assert resolved == str(ckpt_path)
    assert calls == ["epoch=001", "model/checkpoints"]


def test_download_best_checkpoint_propagates_non_missing_model_checkpoint_errors():
    class _Client:
        def list_artifacts(self, run_id):
            return []

        def download_artifacts(self, run_id, artifact_path, dst_path=None):
            assert artifact_path == "model/checkpoints"
            raise RuntimeError("artifact store unavailable")

    with pytest.raises(RuntimeError, match="artifact store unavailable"):
        download_best_checkpoint(_Client(), "run-1")


def test_download_best_checkpoint_prefers_latest_epoch_directory(tmp_path):
    epoch_001_dir = SimpleNamespace(is_dir=True, path="epoch=001")
    epoch_010_dir = SimpleNamespace(is_dir=True, path="epoch=010")
    ignored_file = SimpleNamespace(is_dir=False, path="metrics.json")
    epoch_001_root = tmp_path / "epoch-001"
    epoch_010_root = tmp_path / "epoch-010"
    epoch_001_ckpt = epoch_001_root / "best.ckpt"
    epoch_010_ckpt = epoch_010_root / "best.ckpt"
    epoch_001_ckpt.parent.mkdir(parents=True, exist_ok=True)
    epoch_010_ckpt.parent.mkdir(parents=True, exist_ok=True)
    epoch_001_ckpt.write_text("earlier", encoding="utf-8")
    epoch_010_ckpt.write_text("new", encoding="utf-8")
    calls: list[str] = []

    class _Client:
        def list_artifacts(self, run_id):
            return [ignored_file, epoch_001_dir, epoch_010_dir]

        def download_artifacts(self, run_id, artifact_path, dst_path=None):
            calls.append(artifact_path)
            if artifact_path == "epoch=010":
                return str(epoch_010_root)
            if artifact_path == "epoch=001":
                return str(epoch_001_root)
            raise AssertionError(f"unexpected artifact path {artifact_path}")

    resolved = download_best_checkpoint(_Client(), "run-1")

    assert resolved == str(epoch_010_ckpt)
    assert calls == ["epoch=010"]
