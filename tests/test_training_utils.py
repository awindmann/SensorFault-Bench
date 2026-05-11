import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from config_loader import load_defaults
from models.seasonal_naive import SeasonalNaive
from pipelines.runner import PipelineRunner
from pipelines.signatures import build_signature
from pipelines.specs import PipelineSpec
from pipelines.training import (
    _assert_no_reserved_tag_overlap,
    _hparams_enable_lr_scheduler,
    _log_hparams_artifact,
    build_model_name,
    ensure_experiment_data_signature,
    get_tracking_uri,
    merge_optimizer_hparams,
    normalize_mlflow_run_name,
    optimizer_hparams_from_args,
    optimizer_identity_hparams,
)
from utils.parsing import resolve_mlflow_local_save_dir


def _default_optimizer_args(**overrides):
    defaults = load_defaults()
    values = {
        "lr_scheduler": defaults["LR_SCHEDULER"],
        "optimizer": defaults["OPTIMIZER"],
        "optimizer_beta1": defaults["OPTIMIZER_BETA1"],
        "optimizer_beta2": defaults["OPTIMIZER_BETA2"],
        "optimizer_weight_decay": defaults["OPTIMIZER_WEIGHT_DECAY"],
        "optimizer_eps": defaults["OPTIMIZER_EPS"],
        "scheduler_type": defaults["SCHEDULER_TYPE"],
        "scheduler_factor": defaults["SCHEDULER_FACTOR"],
        "scheduler_patience": defaults["SCHEDULER_PATIENCE"],
        "scheduler_min_lr": defaults["SCHEDULER_MIN_LR"],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_get_tracking_uri_accepts_local_paths_and_file_uris():
    assert get_tracking_uri("logs") == "file:logs"
    assert get_tracking_uri("/tmp/mlruns") == "file:/tmp/mlruns"
    assert get_tracking_uri("file:/tmp/mlruns") == "file:/tmp/mlruns"
    assert get_tracking_uri("file:///tmp/mlruns") == "file:///tmp/mlruns"


def test_resolve_mlflow_local_save_dir_normalizes_file_uris():
    assert resolve_mlflow_local_save_dir("logs") == "logs"
    assert resolve_mlflow_local_save_dir("/tmp/mlruns") == "/tmp/mlruns"
    assert resolve_mlflow_local_save_dir("file:/tmp/mlruns") == "/tmp/mlruns"
    assert resolve_mlflow_local_save_dir("file:///tmp/mlruns") == "/tmp/mlruns"


def test_resolve_mlflow_local_save_dir_rejects_non_local_file_hosts():
    with pytest.raises(ValueError, match="must reference the local machine"):
        resolve_mlflow_local_save_dir("file://example.com/tmp/mlruns")


def test_build_model_name_compacts_structured_and_oversized_values():
    hparams = {
        "profile": "structured_profile",
        "step_size": 0.25,
        "scenario_names": ["drift", "linear_drift", "scaling"],
        "family_defaults": {
            "drift": {"max_offset": 0.5},
            "scaling": {"max_factor": 2.0},
        },
        "family_defaults_signature": (
            '{"drift":{"max_offset":0.5},"scaling":{"max_factor":2.0}}'
        ),
    }

    model_name = build_model_name("GRU", hparams)

    assert len(model_name) < 160
    assert "max_offset" not in model_name
    assert "linear_drift" not in model_name
    assert model_name == build_model_name("GRU", dict(hparams))


def test_normalize_mlflow_run_name_truncates_to_schema_limit():
    run_name = "x" * 260

    normalized = normalize_mlflow_run_name(run_name)
    digest = hashlib.sha256(run_name.encode("utf-8")).hexdigest()[:12]

    assert len(normalized) == 250
    assert normalized.startswith("x" * 235)
    assert normalized.endswith(f"__h{digest}")


def test_optimizer_hparams_from_args_resolves_yaml_owned_defaults():
    args = _default_optimizer_args()

    hparams = optimizer_hparams_from_args(args, model_architecture="DLinear")

    assert hparams["optimizer"] == args.optimizer
    assert hparams["lr_scheduler"] is args.lr_scheduler
    assert hparams["beta1"] == args.optimizer_beta1
    assert hparams["beta2"] == args.optimizer_beta2
    assert hparams["weight_decay"] == args.optimizer_weight_decay
    assert hparams["eps"] == args.optimizer_eps
    assert "scheduler_factor" not in hparams


def test_merge_optimizer_hparams_preserves_explicit_scheduler_override():
    args = _default_optimizer_args(lr_scheduler=False)

    hparams = merge_optimizer_hparams(
        {"lr": 0.2, "lr_scheduler": True},
        args,
        model_architecture="GRU",
    )

    assert hparams["lr"] == 0.2
    assert hparams["lr_scheduler"] is True
    assert hparams["scheduler_type"] == args.scheduler_type
    assert hparams["scheduler_factor"] == args.scheduler_factor
    assert hparams["scheduler_patience"] == args.scheduler_patience
    assert hparams["min_lr"] == args.scheduler_min_lr


def test_merge_optimizer_hparams_skips_non_base_optimizer_architectures():
    args = _default_optimizer_args()

    hparams = merge_optimizer_hparams(
        {"season_length": 24},
        args,
        model_architecture="SeasonalNaive",
    )

    assert hparams == {"season_length": 24}


def test_merge_optimizer_hparams_rejects_scheduler_for_non_base_optimizer():
    args = _default_optimizer_args(lr_scheduler=True)

    with pytest.raises(ValueError, match="does not use the BaseLitModule optimizer"):
        merge_optimizer_hparams(
            {"season_length": 24},
            args,
            model_architecture="SeasonalNaive",
        )


def test_merge_optimizer_hparams_rejects_optimizer_knobs_for_non_base_optimizer():
    args = _default_optimizer_args()

    with pytest.raises(ValueError, match="does not accept optimizer hparam"):
        merge_optimizer_hparams(
            {"season_length": 24, "lr": 0.1},
            args,
            model_architecture="SeasonalNaive",
        )


def test_seasonal_naive_accepts_and_ignores_optimizer_kwargs():
    model = SeasonalNaive(
        season_length=24,
        d_input_features=1,
        d_target_features=1,
        d_seq_in=4,
        d_seq_out=2,
        optimizer="Adam",
        lr=0.1,
        lr_scheduler=True,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.5,
        eps=1e-8,
    )

    assert model.hparams.lr == pytest.approx(0.1)
    assert model.hparams.lr_scheduler is True
    assert model.hparams.weight_decay == pytest.approx(0.5)
    assert model.configure_optimizers().param_groups[0]["lr"] == 0.0


def test_reserved_run_metadata_tags_cannot_be_overridden():
    with pytest.raises(ValueError, match="model_architecture"):
        _assert_no_reserved_tag_overlap(
            {"model_architecture": "GRU"},
            context="test hparams",
        )
    with pytest.raises(ValueError, match="signature"):
        _assert_no_reserved_tag_overlap(
            {"signature": "sig"},
            context="test extra_tags",
        )


def test_lr_scheduler_monitor_uses_effective_hparams():
    assert _hparams_enable_lr_scheduler({"lr_scheduler": "true"}) is True
    assert _hparams_enable_lr_scheduler({}) is False
    with pytest.raises(ValueError, match="lr_scheduler"):
        _hparams_enable_lr_scheduler({"lr_scheduler": None})


def test_hparams_artifact_logging_is_required_and_logs_empty_payload(tmp_path):
    recorded: dict[str, str] = {}

    class _Client:
        def log_artifact(self, run_id, path):
            recorded["run_id"] = run_id
            recorded["payload"] = Path(path).read_text(encoding="utf-8")

    _log_hparams_artifact(_Client(), run_id="run_1", hparams={})

    assert recorded["run_id"] == "run_1"
    assert json.loads(recorded["payload"]) == {}

    class _FailingClient:
        def log_artifact(self, run_id, path):
            raise RuntimeError(f"cannot log {run_id}: {tmp_path.name}")

    with pytest.raises(RuntimeError, match="cannot log run_2"):
        _log_hparams_artifact(_FailingClient(), run_id="run_2", hparams={})


class _ExperimentClient:
    def __init__(self, *, tags=None, runs=()):
        self.experiment = (
            None
            if tags is None
            else SimpleNamespace(experiment_id="exp_existing", tags=dict(tags))
        )
        self.runs = list(runs)
        self.created_names: list[str] = []
        self.set_tags: list[tuple[str, str, str]] = []

    def get_experiment_by_name(self, _name):
        return self.experiment

    def create_experiment(self, name):
        self.created_names.append(name)
        self.experiment = SimpleNamespace(experiment_id="exp_created", tags={})
        return "exp_created"

    def get_experiment(self, experiment_id):
        assert self.experiment is not None
        assert self.experiment.experiment_id == experiment_id
        return self.experiment

    def search_runs(self, experiment_ids, max_results, run_view_type):
        assert experiment_ids == [self.experiment.experiment_id]
        assert max_results == 1
        assert run_view_type is not None
        return self.runs

    def set_experiment_tag(self, experiment_id, key, value):
        self.set_tags.append((experiment_id, key, value))
        assert self.experiment is not None
        self.experiment.tags[key] = value


def test_ensure_experiment_data_signature_sets_tag_for_new_experiment():
    client = _ExperimentClient()

    experiment_id = ensure_experiment_data_signature(
        client,
        experiment_name="benchmark_ETTh1",
        data_config_signature="sig_current",
    )

    assert experiment_id == "exp_created"
    assert client.created_names == ["benchmark_ETTh1"]
    assert client.set_tags == [
        ("exp_created", "data_config_signature", "sig_current")
    ]


def test_ensure_experiment_data_signature_accepts_existing_matching_tag():
    client = _ExperimentClient(tags={"data_config_signature": "sig_current"})

    assert ensure_experiment_data_signature(
        client,
        experiment_name="benchmark_ETTh1",
        data_config_signature="sig_current",
    ) == "exp_existing"
    assert client.set_tags == []


def test_ensure_experiment_data_signature_rejects_mismatch_and_untagged_runs():
    mismatch_client = _ExperimentClient(tags={"data_config_signature": "sig_old"})
    with pytest.raises(ValueError, match="sig_old.*sig_current"):
        ensure_experiment_data_signature(
            mismatch_client,
            experiment_name="benchmark_ETTh1",
            data_config_signature="sig_current",
        )

    untagged_client = _ExperimentClient(tags={}, runs=[SimpleNamespace()])
    with pytest.raises(ValueError, match="existing runs but no data_config_signature"):
        ensure_experiment_data_signature(
            untagged_client,
            experiment_name="benchmark_ETTh1",
            data_config_signature="sig_current",
        )


def test_optimizer_identity_hparams_strips_default_optimizer_values():
    args = _default_optimizer_args()
    signature_hparams = {
        "moving_avg": 9,
        "individual": False,
        "init_weights": False,
        "lr": 0.001,
        "loss": "MSE",
    }

    effective_hparams = merge_optimizer_hparams(
        signature_hparams,
        args,
        model_architecture="DLinear",
    )
    identity_hparams = optimizer_identity_hparams(effective_hparams)

    assert identity_hparams == signature_hparams
    assert build_signature(
        "DLinear",
        "ETTh1",
        identity_hparams,
        pipeline_id="baseline",
        data_config_signature="data_sig",
    ) == build_signature(
        "DLinear",
        "ETTh1",
        signature_hparams,
        pipeline_id="baseline",
        data_config_signature="data_sig",
    )


def test_optimizer_identity_hparams_keeps_non_default_optimizer_values():
    args = _default_optimizer_args(optimizer_weight_decay=0.01)
    base_hparams = {"d_hidden": 64, "n_layers": 1, "dropout": 0.0, "lr": 0.001}

    effective_hparams = merge_optimizer_hparams(
        base_hparams,
        args,
        model_architecture="GRU",
    )
    identity_hparams = optimizer_identity_hparams(effective_hparams)

    assert identity_hparams["optimizer"] == "Adam"
    assert identity_hparams["weight_decay"] == pytest.approx(0.01)
    assert identity_hparams["beta1"] == pytest.approx(0.9)


def test_chronos_provenance_pin_is_not_part_of_run_signature():
    baseline_hparams = {
        "chronos_model_id": "amazon/chronos-2",
        "loss": "MSE",
    }
    pinned_hparams = dict(baseline_hparams)
    pinned_hparams["chronos_model_revision"] = "0f8a440441931157957e2be1a9bce66627d99c76"

    assert build_signature(
        "Chronos2",
        "ETTh1",
        baseline_hparams,
        pipeline_id="baseline",
        data_config_signature="data_sig",
    ) == build_signature(
        "Chronos2",
        "ETTh1",
        pinned_hparams,
        pipeline_id="baseline",
        data_config_signature="data_sig",
    )


def test_training_candidates_use_signature_identity_but_effective_optimizer_hparams():
    args = _default_optimizer_args(loss="MSE")
    spec = PipelineSpec(
        pipeline_id="baseline",
        pipeline_method="baseline",
        pipeline_kind="train",
        recipe_params={},
        model_hparams_mode="baseline_grid",
    )
    runner = PipelineRunner(spec, args)
    signature_hparams = {
        "moving_avg": 9,
        "individual": False,
        "init_weights": False,
        "lr": 0.001,
        "loss": "MSE",
    }

    candidates = runner._build_training_candidates(
        architecture="DLinear",
        dataset_name="ETTh1",
        data_config_signature="data_sig",
        param_sets=[{}],
        hparams_mode="baseline_grid",
        hparams_grid={
            "moving_avg": [9],
            "individual": [False],
            "init_weights": [False],
            "lr": [0.001],
        },
        backbone_run_id=None,
        finetune_epochs=None,
        finetune_lr_factor=None,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.signature == build_signature(
        "DLinear",
        "ETTh1",
        signature_hparams,
        pipeline_id="baseline",
        data_config_signature="data_sig",
    )
    assert candidate.hparams["optimizer"] == "Adam"
    assert candidate.hparams["beta1"] == pytest.approx(0.9)
    assert candidate.model_name == f"baseline_{build_model_name('DLinear', signature_hparams)}"
