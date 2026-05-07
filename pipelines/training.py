"""Shared training utilities for pipeline execution."""

import itertools
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import mlflow
import pytorch_lightning as pl
import torch
from mlflow.entities import Run, ViewType
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from data.data_module import TSDataModule, validate_train_noise_config
from data.datasets import spec_to_tags
import models
from .ranking import sort_runs_by_metric
from .signatures import compute_data_config_signature
from utils.artifacts import download_best_checkpoint, load_lightning_module_checkpoint
from utils.env import current_git_commit as _current_git_commit
from utils.env import set_mlflow_storage_env as _set_mlflow_storage_env
from utils.parsing import (
    build_mlflow_tracking_uri,
    has_explicit_value,
    normalize_mlflow_run_name,
    parse_optimizer_name,
    parse_optional_nonempty_string,
    parse_required_finite_float,
    parse_required_nonnegative_int,
    parse_scheduler_type,
    parse_value,
    require_namespace_bool,
    require_namespace_value,
    resolve_mlflow_local_save_dir,
    sanitize_model_name_fragment,
    serialize_model_name_value,
)
from utils.rng import set_seed, derive_component_seeds


torch.backends.cudnn.benchmark = False


class _NaNDivergenceError(RuntimeError):
    """Raised when training produces no valid ``best_val_loss``."""


class _FailOnNaNDivergence(pl.Callback):
    """Raise at the end of training if the model never recorded a valid
    ``best_val_loss``.  Because this fires inside ``trainer.fit()``, the
    MLFlowLogger finalizes the run as FAILED instead of FINISHED."""

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if math.isinf(pl_module._best_val_loss):
            raise _NaNDivergenceError(
                "Training diverged: best_val_loss was never recorded "
                "(all validation losses were NaN or training produced no valid epoch)."
            )


class _LogAdaptiveLossArtifact(pl.Callback):
    """Log learned adaptive-loss parameters as a JSON artifact at end of training."""

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        from metrics.adaptive_robust_loss import AdaptiveRobustLoss

        loss_fn = getattr(pl_module, "loss_fn", None)
        if not isinstance(loss_fn, AdaptiveRobustLoss):
            return
        if math.isinf(pl_module._best_val_loss):
            return  # diverged, no artifact
        with torch.no_grad():
            alpha = loss_fn.get_alpha().detach().cpu()
            scale = loss_fn.get_scale().detach().cpu()
        d_seq_out = pl_module.d_seq_out
        d_target = pl_module.d_target_features
        artifact = {
            "d_seq_out": d_seq_out,
            "d_target_features": d_target,
            "alpha": alpha.reshape(d_seq_out, d_target).tolist(),
            "scale": scale.reshape(d_seq_out, d_target).tolist(),
        }
        with tempfile.TemporaryDirectory(prefix="robust-") as tmpdir:
            path = os.path.join(tmpdir, "adaptive_loss_params.json")
            with open(path, "w") as f:
                json.dump(artifact, f, indent=2)
            logger = getattr(trainer, "logger", None)
            experiment = getattr(logger, "experiment", None)
            run_id = getattr(logger, "run_id", None)
            if experiment is None or run_id is None:
                raise RuntimeError(
                    "Adaptive-loss artifact logging requires an active MLflow logger."
                )
            experiment.log_artifact(run_id, path, artifact_path="diagnostics")


def iter_limited_product(values, limit=None):
    """Yield (1-indexed) cartesian products of values up to an optional limit."""
    for index, combo in enumerate(itertools.product(*values), start=1):
        if limit is not None and index > limit:
            break
        yield index, combo


_RESERVED_TAG_KEYS = {
    "batch_size",
    "best_model",
    "continuous_input_channel_count",
    "continuous_input_channels",
    "data_config_signature",
    "dataset",
    "dataset_path",
    "date",
    "discrete_input_channel_count",
    "discrete_input_channels",
    "finetune_epochs",
    "finetune_lr_factor",
    "input_channel_count",
    "input_channels",
    "loader_kind",
    "loss_function",
    "model_architecture",
    "pipeline_id",
    "pipeline_kind",
    "pipeline_method",
    "robustness_method",
    "seed_data",
    "seed_eval",
    "seed_master",
    "seed_model",
    "seed_policy",
    "shuffle_batches_before_split",
    "signature",
    "split_mode",
    "stage",
    "target_alias",
    "target_channel_count",
    "target_channels",
    "train_commit",
    "train_noise_channels",
    "train_noise_std",
    "train_perturbation_channel_fraction_max",
    "train_perturbation_probability",
    "train_perturbation_profile",
    "train_perturbation_scenarios_signature",
    "train_perturbation_severity_max",
}

_BASE_OPTIMIZER_HPARAM_KEYS = (
    "optimizer",
    "lr_scheduler",
    "scheduler_type",
    "scheduler_factor",
    "scheduler_patience",
    "min_lr",
    "beta1",
    "beta2",
    "weight_decay",
    "eps",
)

_NON_BASE_OPTIMIZER_HPARAM_KEYS = (
    *_BASE_OPTIMIZER_HPARAM_KEYS,
    "grad_clip",
    "grad_clip_after_warmup",
    "initial_lr",
    "lr",
    "peak_lr",
    "warmup_div",
)

_DEFAULT_OPTIMIZER_IDENTITY_HPARAMS = {
    "optimizer": "Adam",
    "lr_scheduler": False,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 0.0,
    "eps": 1e-8,
}


def _model_uses_base_optimizer(model_architecture: str) -> bool:
    try:
        model_class = getattr(models, model_architecture)
    except AttributeError as exc:
        raise ValueError(f"The model '{model_architecture}' does not exist.") from exc
    return bool(getattr(model_class, "uses_base_optimizer", True))


def _explicit_non_base_optimizer_hparam_keys(hparams: dict[str, Any]) -> list[str]:
    return sorted(
        key
        for key in set(_NON_BASE_OPTIMIZER_HPARAM_KEYS)
        if key in hparams and has_explicit_value(hparams[key])
    )


def _optimizer_identity_value_matches_default(
    key: str,
    value: Any,
    default: Any,
) -> bool:
    if key == "optimizer":
        return parse_optimizer_name(value, key=key).lower() == str(default).lower()
    if key == "lr_scheduler":
        return parse_value(value, bool, key=key) is bool(default)
    numeric = parse_required_finite_float(value, key=key)
    return math.isclose(
        numeric,
        float(default),
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def optimizer_hparams_from_args(
    args: Any,
    *,
    model_architecture: str,
    scheduler_enabled: Optional[bool] = None,
) -> dict[str, Any]:
    """Resolve YAML/CLI-owned optimizer defaults into model hparams."""
    args_lr_scheduler = require_namespace_bool(args, key="lr_scheduler")
    use_scheduler = (
        args_lr_scheduler
        if scheduler_enabled is None
        else bool(scheduler_enabled)
    )
    if not _model_uses_base_optimizer(model_architecture):
        if use_scheduler:
            raise ValueError(
                f"{model_architecture} does not use the BaseLitModule optimizer "
                "and cannot enable lr_scheduler."
            )
        return {}

    hparams: dict[str, Any] = {
        "optimizer": parse_optimizer_name(
            require_namespace_value(args, key="optimizer"),
            key="optimizer",
        ),
        "lr_scheduler": use_scheduler,
        "beta1": parse_required_finite_float(
            require_namespace_value(args, key="optimizer_beta1"),
            key="optimizer_beta1",
        ),
        "beta2": parse_required_finite_float(
            require_namespace_value(args, key="optimizer_beta2"),
            key="optimizer_beta2",
        ),
        "weight_decay": parse_required_finite_float(
            require_namespace_value(args, key="optimizer_weight_decay"),
            key="optimizer_weight_decay",
        ),
        "eps": parse_required_finite_float(
            require_namespace_value(args, key="optimizer_eps"),
            key="optimizer_eps",
        ),
    }
    if hparams["beta1"] <= 0.0 or hparams["beta1"] >= 1.0:
        raise ValueError(
            "optimizer_beta1 must satisfy 0 < beta1 < 1; "
            f"got {hparams['beta1']}."
        )
    if hparams["beta2"] <= 0.0 or hparams["beta2"] >= 1.0:
        raise ValueError(
            "optimizer_beta2 must satisfy 0 < beta2 < 1; "
            f"got {hparams['beta2']}."
        )
    if hparams["weight_decay"] < 0.0:
        raise ValueError(
            f"optimizer_weight_decay must be >= 0; got {hparams['weight_decay']}."
        )
    if hparams["eps"] <= 0.0:
        raise ValueError(f"optimizer_eps must be > 0; got {hparams['eps']}.")

    if use_scheduler:
        scheduler_factor = parse_required_finite_float(
            require_namespace_value(args, key="scheduler_factor"),
            key="scheduler_factor",
        )
        if scheduler_factor <= 0.0:
            raise ValueError(f"scheduler_factor must be > 0; got {scheduler_factor}.")
        scheduler_min_lr = parse_required_finite_float(
            require_namespace_value(args, key="scheduler_min_lr"),
            key="scheduler_min_lr",
        )
        if scheduler_min_lr < 0.0:
            raise ValueError(f"scheduler_min_lr must be >= 0; got {scheduler_min_lr}.")
        hparams.update(
            {
                "scheduler_type": parse_scheduler_type(
                    require_namespace_value(args, key="scheduler_type"),
                    key="scheduler_type",
                ),
                "scheduler_factor": scheduler_factor,
                "scheduler_patience": parse_required_nonnegative_int(
                    require_namespace_value(args, key="scheduler_patience"),
                    key="scheduler_patience",
                ),
                "min_lr": scheduler_min_lr,
            }
        )
    return hparams


def merge_optimizer_hparams(
    hparams: dict[str, Any],
    args: Any,
    *,
    model_architecture: str,
) -> dict[str, Any]:
    """Merge YAML/CLI optimizer defaults without overriding explicit hparams."""
    merged = dict(hparams)
    if not _model_uses_base_optimizer(model_architecture):
        unsupported = _explicit_non_base_optimizer_hparam_keys(merged)
        if unsupported:
            raise ValueError(
                f"{model_architecture} does not use the BaseLitModule optimizer "
                "and does not accept optimizer hparam(s): "
                f"{', '.join(unsupported)}."
            )
    scheduler_enabled: Optional[bool] = None
    if "lr_scheduler" in merged:
        parsed = parse_value(merged["lr_scheduler"], bool, key="lr_scheduler")
        if parsed is None:
            raise ValueError("lr_scheduler must not be null when present in hparams.")
        scheduler_enabled = parsed
    defaults = optimizer_hparams_from_args(
        args,
        model_architecture=model_architecture,
        scheduler_enabled=scheduler_enabled,
    )
    for key, value in defaults.items():
        merged.setdefault(key, value)
    return merged


def optimizer_identity_hparams(hparams: dict[str, Any]) -> dict[str, Any]:
    """Return hparams for run identity while preserving default signatures."""
    identity_hparams = dict(hparams)
    if not any(key in identity_hparams for key in _DEFAULT_OPTIMIZER_IDENTITY_HPARAMS):
        return identity_hparams
    for key, default in _DEFAULT_OPTIMIZER_IDENTITY_HPARAMS.items():
        if key not in identity_hparams:
            return identity_hparams
        if not _optimizer_identity_value_matches_default(
            key,
            identity_hparams[key],
            default,
        ):
            return identity_hparams
    for key in _DEFAULT_OPTIMIZER_IDENTITY_HPARAMS:
        identity_hparams.pop(key, None)
    return identity_hparams


def apply_optimizer_hparams_to_model(
    model: pl.LightningModule,
    hparams: dict[str, Any],
) -> None:
    """Attach optimizer hparams to a checkpoint-loaded model before finetuning."""
    for key in _BASE_OPTIMIZER_HPARAM_KEYS:
        if key in hparams:
            setattr(model.hparams, key, hparams[key])


def _assert_no_reserved_tag_overlap(payload: dict | None, *, context: str) -> None:
    if not payload:
        return
    collisions = sorted(set(payload) & _RESERVED_TAG_KEYS)
    if collisions:
        raise ValueError(
            f"{context} contains reserved tag(s): {', '.join(collisions)}. "
            "Remove them to avoid overriding run metadata."
        )


def _validate_training_pipeline_identity(
    *,
    pipeline_id: str,
    pipeline_method: str,
    pipeline_kind: str,
    robustness_method: str | None,
    context: str,
) -> None:
    if (pipeline_id == "baseline") != (pipeline_method == "baseline"):
        raise ValueError(
            "Baseline tagging mismatch: pipeline_id and pipeline_method must both be "
            f"'baseline' (or both be non-baseline) for {context}."
        )
    if pipeline_method == "baseline":
        if pipeline_kind != "train":
            raise ValueError(
                f"Baseline {context} must use pipeline_kind='train', got "
                f"{pipeline_kind!r}."
            )
        if robustness_method != "baseline":
            raise ValueError(
                f"Baseline {context} must set robustness_method='baseline', got "
                f"{robustness_method!r}."
            )
        return
    if robustness_method is None or str(robustness_method).strip() != pipeline_method:
        raise ValueError(
            f"Non-baseline {context} must set robustness_method equal to "
            f"pipeline_method={pipeline_method!r}, got {robustness_method!r}."
        )


def _hparams_enable_lr_scheduler(hparams: dict[str, Any]) -> bool:
    if "lr_scheduler" not in hparams:
        return False
    parsed = parse_value(hparams["lr_scheduler"], bool, key="lr_scheduler")
    if parsed is None:
        raise ValueError("lr_scheduler must not be null when present in hparams.")
    return parsed


def _require_optional_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dictionary when provided.")
    return dict(value)


def _resolve_loader_kind_value(
    *,
    model_architecture: str,
    requested_loader_kind: Any,
) -> str:
    explicit = parse_optional_nonempty_string(
        requested_loader_kind,
        key="loader_kind",
    )
    if explicit is not None:
        return explicit

    model_class = getattr(models, model_architecture, None)
    if model_class is None:
        raise ValueError(f"The model '{model_architecture}' does not exist.")
    class_default = parse_optional_nonempty_string(
        getattr(model_class, "loader_kind", None),
        key="loader_kind",
    )
    if class_default is not None:
        return class_default
    return model_architecture


def _resolve_max_fit_epochs(
    *,
    model: pl.LightningModule,
    max_epochs: int,
) -> int:
    resolved = int(max_epochs)
    model_limit = getattr(model, "max_fit_epochs", None)
    if model_limit is None:
        return resolved
    limit = int(model_limit)
    if limit <= 0:
        raise ValueError(
            f"Model {model.__class__.__name__} declared invalid max_fit_epochs={limit}."
        )
    return min(resolved, limit)


@dataclass
class ResolvedRuns:
    canonical: Optional[Run]
    failed: List[Run]
    running: List[Run]
    duplicates: List[Run]


def build_model_name(model_architecture: str, hparams: Dict[str, object]) -> str:
    """Derive a readable MLflow run name from model architecture and hyperparameters."""
    parts = [model_architecture]
    for key, value in hparams.items():
        abbrev = "".join(fragment[0] for fragment in key.split("_"))
        parts.append(f"{abbrev}{serialize_model_name_value(value)}")
    name = "_".join(parts)
    return sanitize_model_name_fragment(name)


def search_runs_all(
    client: Any,
    experiment_ids: List[str],
    *,
    filter_string: Optional[str] = None,
    run_view_type: ViewType = ViewType.ALL,
    max_results: int = 1000,
    order_by: Optional[List[str]] = None,
) -> List[Run]:
    """Fetch all runs across MLflow pages for the provided query."""
    page_token: Optional[str] = None
    all_runs: List[Run] = []
    seen_ids: set[str] = set()
    while True:
        kwargs: Dict[str, Any] = {
            "run_view_type": run_view_type,
            "max_results": max_results,
        }
        if filter_string is not None:
            kwargs["filter_string"] = filter_string
        if order_by is not None:
            kwargs["order_by"] = order_by
        if page_token is not None:
            kwargs["page_token"] = page_token
        page = client.search_runs(experiment_ids, **kwargs)
        runs = list(page)
        for run in runs:
            run_id = run.info.run_id
            if run_id in seen_ids:
                continue
            seen_ids.add(run_id)
            all_runs.append(run)
        next_page_token = getattr(page, "token", None)
        if not next_page_token:
            break
        if next_page_token == page_token:
            raise RuntimeError(
                "MLflow search_runs pagination returned a repeated page token."
            )
        page_token = next_page_token
    return all_runs


def resolve_runs(
    client,
    experiment_id,
    signature,
):
    """Locate existing MLflow runs sharing the same signature."""
    if experiment_id is None:
        return ResolvedRuns(canonical=None, failed=[], running=[], duplicates=[])

    filter_string = f"tags.signature = '{signature}'"
    runs = search_runs_all(
        client,
        [experiment_id],
        filter_string=filter_string,
        max_results=1000,
        run_view_type=ViewType.ALL,
    )
    runs = [run for run in runs if getattr(run.info, "lifecycle_stage", "active") == "active"]
    runs = [run for run in runs if not run.data.tags.get("mlflow.parentRunId")]

    def _status_token(run: Run) -> str:
        status_value = getattr(run.info, "status", None)
        if status_value is None:
            return ""
        return str(status_value).upper()

    in_progress_statuses = {"RUNNING", "SCHEDULED"}

    finished = [
        run for run in runs
        if _status_token(run) == "FINISHED"
        and "best_val_loss" in run.data.metrics
    ]
    diverged = [
        run for run in runs
        if _status_token(run) == "FINISHED"
        and "best_val_loss" not in run.data.metrics
    ]
    running = [run for run in runs if _status_token(run) in in_progress_statuses]
    failed = [
        run
        for run in runs
        if _status_token(run) not in {"FINISHED", *in_progress_statuses}
    ]
    failed.extend(diverged)

    finished = sort_runs_by_metric(
        finished,
        metric_key="best_val_loss",
        missing_error_prefix="Signature-dedup finished runs",
    )
    canonical = finished[0] if finished else None
    duplicates = finished[1:] if len(finished) > 1 else []

    failed.sort(key=lambda run: run.info.start_time or 0, reverse=True)

    return ResolvedRuns(canonical=canonical, failed=failed, running=running, duplicates=duplicates)


def delete_runs(client, runs, reason):
    """Delete the provided MLflow runs if they are still active.

    Clears ``best_model`` and ``backbone_current`` tags before soft-deleting
    so that stale winner tags do not persist on deleted runs (MLflow soft
    delete preserves tags).
    """
    seen_ids: set[str] = set()
    for run in runs:
        if not run:
            continue
        run_id = run.info.run_id
        if run_id in seen_ids:
            continue
        seen_ids.add(run_id)
        if getattr(run.info, "lifecycle_stage", "active") != "active":
            continue
        try:
            if run.data.tags is None:
                raise ValueError(
                    f"Run {run_id} is missing tags during delete_runs ({reason})."
                )
            if run.data.tags.get("best_model") == "true":
                client.set_tag(run_id, "best_model", "false")
            if run.data.tags.get("backbone_current") == "true":
                client.set_tag(run_id, "backbone_current", "false")
            client.delete_run(run_id)
            print(f"Deleted MLflow run {run_id} ({reason}).")
        except Exception as exc:
            print(f"Failed to delete MLflow run {run_id}: {exc}")


def get_tracking_uri(logdir: str) -> str:
    """Build the MLflow tracking URI from the configured logdir."""
    return build_mlflow_tracking_uri(logdir)


def ensure_experiment_data_signature(
    client,
    experiment_name: str,
    data_config_signature: str,
) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
        experiment = client.get_experiment(experiment_id)
    experiment_id = experiment.experiment_id

    existing = (experiment.tags or {}).get("data_config_signature")
    if existing:
        if existing != data_config_signature:
            raise ValueError(
                f"Experiment '{experiment_name}' has data_config_signature '{existing}', "
                f"but current config is '{data_config_signature}'. "
                "Use a different --mlflow-experiment-prefix to avoid mixing incomparable settings."
            )
        return experiment_id

    has_runs = bool(
        client.search_runs([experiment_id], max_results=1, run_view_type=ViewType.ALL)
    )
    if has_runs:
        raise ValueError(
            f"Experiment '{experiment_name}' has existing runs but no data_config_signature tag. "
            "Create a new --mlflow-experiment-prefix to avoid mixing incomparable settings."
        )
    client.set_experiment_tag(experiment_id, "data_config_signature", data_config_signature)
    return experiment_id


def _build_datamodule(
    *,
    dataset_spec,
    args,
    datamodule_kwargs: Optional[dict] = None,
) -> TSDataModule:
    dm_overrides = _require_optional_mapping(
        datamodule_kwargs,
        name="datamodule_kwargs",
    )
    if dm_overrides:
        validate_train_noise_config(
            dm_overrides.get("train_noise_std"),
            dm_overrides.get("train_noise_channels"),
        )
    dm_kwargs = {
        "dataset_spec": dataset_spec,
        "input_len": args.input_len,
        "target_len": args.target_len,
        "n_train_samples": args.n_train_samples,
        "n_val_samples": args.n_val_samples,
        "perturbation_channel_fraction_max": args.perturbation_channel_fraction_max,
        "perturbation_scenarios": args.perturbation_scenarios,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "purged_fraction": args.purged_fraction,
        "shuffle_batches_before_split": args.shuffle_batches_before_split,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "s3_endpoint": args.minio_endpoint,
        "strict_iid": args.strict_iid,
    }
    dm_kwargs.update(dm_overrides)
    return TSDataModule(**dm_kwargs)


def _extract_train_augmentation_identity_tags(
    datamodule_kwargs: Optional[dict[str, Any]],
) -> dict[str, Any]:
    if not datamodule_kwargs:
        return {}
    if not isinstance(datamodule_kwargs, dict):
        raise ValueError("datamodule_kwargs must be a dictionary when provided.")
    identity_keys = (
        "train_noise_std",
        "train_noise_channels",
        "train_perturbation_profile",
        "train_perturbation_scenarios_signature",
        "train_perturbation_probability",
        "train_perturbation_severity_max",
        "train_perturbation_channel_fraction_max",
    )
    tags: dict[str, Any] = {}
    for key in identity_keys:
        if key not in datamodule_kwargs:
            continue
        tags[key] = datamodule_kwargs[key]
    return tags


def _cleanup_local_checkpoints(*, args, callbacks: list, trainer: pl.Trainer) -> None:
    if not args.save_checkpoint:
        return
    checkpoint_cb = next((cb for cb in callbacks if isinstance(cb, ModelCheckpoint)), None)
    if checkpoint_cb is None:
        return
    ckpt_path = checkpoint_cb._ModelCheckpoint__resolve_ckpt_dir(trainer)
    if not os.path.exists(ckpt_path):
        return
    shutil.rmtree(ckpt_path)
    for folder in [os.path.dirname(ckpt_path), os.path.dirname(os.path.dirname(ckpt_path))]:
        if os.path.exists(folder) and not os.listdir(folder):
            os.rmdir(folder)


def _log_hparams_artifact(client: Any, *, run_id: str, hparams: dict) -> None:
    with tempfile.TemporaryDirectory(prefix="robust-") as tmpdir:
        path = os.path.join(tmpdir, "hparams.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(hparams, handle, indent=2, sort_keys=True)
        client.log_artifact(run_id, path)


def _fit_and_finalize(
    *,
    args,
    client: Any,
    dataset_name: str,
    run_name: str,
    run_id: Optional[str],
    tags: dict[str, str],
    model: pl.LightningModule,
    datamodule: TSDataModule,
    max_epochs: int,
    hparams_to_log: dict,
    stage: str,
    signature: Optional[str],
    params_to_log: Optional[dict[str, Any]] = None,
) -> str:
    callbacks: list = []
    if args.save_checkpoint:
        callbacks.append(
            ModelCheckpoint(
                monitor="ep_val_loss",
                filename="{epoch}-{ep_val_loss:.5f}",
                save_top_k=1,
                mode="min",
            )
        )
    if args.early_stopping:
        callbacks.append(
            EarlyStopping(monitor="ep_val_loss", patience=args.early_stopping_patience)
        )
    if _hparams_enable_lr_scheduler(hparams_to_log):
        callbacks.append(LearningRateMonitor())
    callbacks.append(_FailOnNaNDivergence())
    callbacks.append(_LogAdaptiveLossArtifact())
    normalized_run_name = normalize_mlflow_run_name(run_name)
    logger_kwargs = {
        "tracking_uri": get_tracking_uri(args.logdir),
        "save_dir": resolve_mlflow_local_save_dir(args.logdir),
        "experiment_name": f"{args.mlflow_experiment_prefix}-{dataset_name}",
        "run_name": None if run_id else normalized_run_name,
        "tags": tags,
        "log_model": True if args.save_checkpoint else False,
    }
    if run_id:
        logger_kwargs["run_id"] = run_id
    logger = MLFlowLogger(**logger_kwargs)
    if run_id:
        if not hasattr(client, "set_tag"):
            raise AttributeError(
                "MLflow client missing set_tag; cannot log tags for existing run."
            )
        for key, value in tags.items():
            client.set_tag(run_id, key, value)

    trainer = pl.Trainer(
        enable_checkpointing=True if args.save_checkpoint else False,
        max_epochs=_resolve_max_fit_epochs(model=model, max_epochs=max_epochs),
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
    )
    if trainer.strategy.root_device.type == "cuda":
        torch.set_float32_matmul_precision("medium")

    try:
        trainer.fit(model=model, datamodule=datamodule)
    except _NaNDivergenceError:
        final_run_id = logger.run_id
        msg = f"Run {final_run_id} diverged: best_val_loss was never recorded."
        if require_namespace_bool(args, key="raise_error"):
            raise RuntimeError(msg)
        print(msg)
        return final_run_id

    _cleanup_local_checkpoints(args=args, callbacks=callbacks, trainer=trainer)
    print("Done.")

    final_run_id = logger.run_id
    if params_to_log:
        for key, value in params_to_log.items():
            if value is None:
                continue
            try:
                client.log_param(final_run_id, key, value)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to log required param '{key}' for run {final_run_id}."
                ) from exc
    _log_hparams_artifact(client, run_id=final_run_id, hparams=hparams_to_log)

    client.set_tag(final_run_id, "stage", stage)
    client.set_tag(final_run_id, "batch_size", str(args.batch_size))
    if signature is not None:
        client.set_tag(final_run_id, "signature", signature)
    return final_run_id


def train_single_run(
    *,
    model_architecture,
    hparams,
    dataset_spec,
    args,
    data_config_signature=None,
    signature=None,
    model_name_override=None,
    run_id=None,
    stage="train",
    robustness_method="baseline",
    extra_tags=None,
    pipeline_id="baseline",
    pipeline_method="baseline",
    pipeline_kind="train",
    loader_kind=None,
    datamodule_kwargs=None,
    model_kwargs=None,
):
    # Use consistent MLflow run naming across training flows.
    hparams = _require_optional_mapping(hparams, name="hparams")
    datamodule_kwargs = _require_optional_mapping(
        datamodule_kwargs,
        name="datamodule_kwargs",
    )
    model_kwargs = _require_optional_mapping(model_kwargs, name="model_kwargs")
    _validate_training_pipeline_identity(
        pipeline_id=pipeline_id,
        pipeline_method=pipeline_method,
        pipeline_kind=pipeline_kind,
        robustness_method=robustness_method,
        context="training runs",
    )

    effective_hparams = dict(hparams)
    if model_kwargs:
        effective_hparams.update(model_kwargs)
    effective_hparams = merge_optimizer_hparams(
        effective_hparams,
        args,
        model_architecture=model_architecture,
    )
    identity_hparams = optimizer_identity_hparams(effective_hparams)

    if model_name_override is None:
        model_name = build_model_name(model_architecture, identity_hparams)
    else:
        model_name = str(model_name_override).strip()
        if not model_name:
            raise ValueError("model_name_override must be a non-empty string.")

    if data_config_signature is None:
        data_config_signature = compute_data_config_signature(
            dataset_spec=dataset_spec, args=args
        )
    data_split_seed = require_namespace_value(args, key="data_split_seed")
    seeds = derive_component_seeds(
        base_seed=args.seed,
        data_base_seed=data_split_seed,
        dataset_key=dataset_spec.key,
        data_config_signature=data_config_signature,
        architecture=model_architecture,
        pipeline_id=pipeline_id,
        pipeline_method=pipeline_method,
        pipeline_kind=pipeline_kind,
    )
    set_seed(seeds["model_seed"])
    pl.seed_everything(seeds["model_seed"], workers=True)

    _set_mlflow_storage_env(args)
    tracking_uri = get_tracking_uri(args.logdir)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    dm_kwargs = dict(datamodule_kwargs)
    dm_kwargs["seed"] = seeds["data_seed"]
    dm_kwargs.setdefault(
        "train_noise_generator",
        torch.Generator().manual_seed(seeds["model_seed"]),
    )
    dm_kwargs.setdefault(
        "train_perturbation_generator",
        torch.Generator().manual_seed(seeds["model_seed"]),
    )
    dm = _build_datamodule(dataset_spec=dataset_spec, args=args, datamodule_kwargs=dm_kwargs)
    dataset_name = dataset_spec.key
    spec_tags = spec_to_tags(dataset_spec, n_inputs=dm.n_inputs, n_outputs=dm.n_outputs)
    loss_value = effective_hparams.get("loss")
    if loss_value is None:
        loss_value = require_namespace_value(args, key="loss")
    loader_kind_value = _resolve_loader_kind_value(
        model_architecture=model_architecture,
        requested_loader_kind=loader_kind,
    )

    tags = {
        **spec_tags,
        "model_architecture": model_architecture,
        "dataset": dataset_name,
        "date": datetime.today().strftime('%Y-%m-%d'),
        "best_model": "false",
        "stage": stage,
        "loader_kind": loader_kind_value,
        "shuffle_batches_before_split": args.shuffle_batches_before_split,
        "pipeline_id": pipeline_id,
        "pipeline_method": pipeline_method,
        "pipeline_kind": pipeline_kind,
        "data_config_signature": data_config_signature,
        "seed_master": args.seed,
        "seed_data": seeds["data_seed"],
        "seed_model": seeds["model_seed"],
        "seed_eval": seeds["eval_seed"],
        "seed_policy": "v2",
    }
    if loss_value is not None:
        tags["loss_function"] = loss_value
    if robustness_method is not None:
        tags["robustness_method"] = robustness_method
    _assert_no_reserved_tag_overlap(effective_hparams, context="train_single_run hparams")
    _assert_no_reserved_tag_overlap(extra_tags, context="train_single_run extra_tags")
    tags.update(effective_hparams)
    tags.update(_extract_train_augmentation_identity_tags(datamodule_kwargs))
    if signature is not None:
        tags["signature"] = signature
    if extra_tags:
        tags.update(extra_tags)
    tags["train_commit"] = _current_git_commit()

    tags = {k:str(v) for k,v in tags.items()}

    # load model
    try:
        model_class = getattr(models, model_architecture)
        model = model_class(
            d_input_features=dm.n_inputs,
            d_target_features=dm.n_outputs,
            d_seq_in=args.input_len,
            d_seq_out=args.target_len,
            target_indices=dm.target_column_indices,
            **effective_hparams
        )
    except AttributeError:
        raise ValueError(f"The model '{model_architecture}' does not exist.")
    model.set_model_seed(seeds["model_seed"])

    return _fit_and_finalize(
        args=args,
        client=client,
        dataset_name=dataset_name,
        run_name=model_name,
        run_id=run_id,
        tags=tags,
        model=model,
        datamodule=dm,
        max_epochs=int(args.max_epochs),
        hparams_to_log=effective_hparams,
        stage=stage,
        signature=signature,
    )


def finetune_single_run(
    *,
    model_architecture,
    backbone_run_id,
    hparams,
    dataset_spec,
    args,
    data_config_signature=None,
    signature=None,
    model_name_override=None,
    run_id=None,
    stage="train",
    robustness_method="baseline",
    extra_tags=None,
    pipeline_id="baseline",
    pipeline_method="baseline",
    pipeline_kind="train",
    loader_kind=None,
    datamodule_kwargs=None,
    model_kwargs=None,
    finetune_epochs=None,
    finetune_lr_factor=None,
):
    hparams = _require_optional_mapping(hparams, name="hparams")
    datamodule_kwargs = _require_optional_mapping(
        datamodule_kwargs,
        name="datamodule_kwargs",
    )
    model_kwargs = _require_optional_mapping(model_kwargs, name="model_kwargs")
    _validate_training_pipeline_identity(
        pipeline_id=pipeline_id,
        pipeline_method=pipeline_method,
        pipeline_kind=pipeline_kind,
        robustness_method=robustness_method,
        context="finetune runs",
    )
    if finetune_epochs is None:
        raise ValueError("finetune_epochs must be provided for finetune_single_run.")
    if finetune_lr_factor is None:
        raise ValueError("finetune_lr_factor must be provided for finetune_single_run.")

    effective_hparams = dict(hparams)
    if model_kwargs:
        effective_hparams.update(model_kwargs)
    effective_hparams = merge_optimizer_hparams(
        effective_hparams,
        args,
        model_architecture=model_architecture,
    )
    identity_hparams = optimizer_identity_hparams(effective_hparams)

    if model_name_override is None:
        model_name = build_model_name(model_architecture, identity_hparams)
    else:
        model_name = str(model_name_override).strip()
        if not model_name:
            raise ValueError("model_name_override must be a non-empty string.")

    if data_config_signature is None:
        data_config_signature = compute_data_config_signature(
            dataset_spec=dataset_spec, args=args
        )
    data_split_seed = require_namespace_value(args, key="data_split_seed")
    seeds = derive_component_seeds(
        base_seed=args.seed,
        data_base_seed=data_split_seed,
        dataset_key=dataset_spec.key,
        data_config_signature=data_config_signature,
        architecture=model_architecture,
        pipeline_id=pipeline_id,
        pipeline_method=pipeline_method,
        pipeline_kind=pipeline_kind,
    )
    set_seed(seeds["model_seed"])
    pl.seed_everything(seeds["model_seed"], workers=True)

    _set_mlflow_storage_env(args)
    tracking_uri = get_tracking_uri(args.logdir)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    dm_kwargs = dict(datamodule_kwargs)
    dm_kwargs["seed"] = seeds["data_seed"]
    dm_kwargs.setdefault(
        "train_noise_generator",
        torch.Generator().manual_seed(seeds["model_seed"]),
    )
    dm_kwargs.setdefault(
        "train_perturbation_generator",
        torch.Generator().manual_seed(seeds["model_seed"]),
    )
    dm = _build_datamodule(dataset_spec=dataset_spec, args=args, datamodule_kwargs=dm_kwargs)

    with tempfile.TemporaryDirectory(prefix=f"robust-backbone-{backbone_run_id[:8]}-") as ckpt_tmpdir:
        checkpoint_path = download_best_checkpoint(client, backbone_run_id, dst_path=ckpt_tmpdir)

        try:
            model_class = getattr(models, model_architecture)
            model = load_lightning_module_checkpoint(
                model_class,
                checkpoint_path,
                **model_kwargs,
            )
        except AttributeError:
            raise ValueError(f"The model '{model_architecture}' does not exist.")
    model.set_model_seed(seeds["model_seed"])

    model.apply_loss_overrides_from_kwargs(model_kwargs)
    apply_optimizer_hparams_to_model(model, effective_hparams)

    original_lr = getattr(model.hparams, "lr", None)
    if original_lr is None:
        raise ValueError(
            f"Backbone {backbone_run_id} is missing 'lr' in hparams. "
            "Cannot apply finetune_lr_factor."
        )
    finetune_lr = float(original_lr) * float(finetune_lr_factor)
    model.hparams.lr = finetune_lr
    effective_hparams["lr"] = finetune_lr

    dataset_name = dataset_spec.key
    spec_tags = spec_to_tags(dataset_spec, n_inputs=dm.n_inputs, n_outputs=dm.n_outputs)
    loss_value = effective_hparams.get("loss")
    if loss_value is None:
        loss_value = require_namespace_value(args, key="loss")
    loader_kind_value = _resolve_loader_kind_value(
        model_architecture=model_architecture,
        requested_loader_kind=loader_kind,
    )

    tags = {
        **spec_tags,
        "model_architecture": model_architecture,
        "dataset": dataset_name,
        "date": datetime.today().strftime("%Y-%m-%d"),
        "best_model": "false",
        "stage": stage,
        "loader_kind": loader_kind_value,
        "shuffle_batches_before_split": args.shuffle_batches_before_split,
        "pipeline_id": pipeline_id,
        "pipeline_method": pipeline_method,
        "pipeline_kind": pipeline_kind,
        "data_config_signature": data_config_signature,
        "backbone_run_id": backbone_run_id,
        "seed_master": args.seed,
        "seed_data": seeds["data_seed"],
        "seed_model": seeds["model_seed"],
        "seed_eval": seeds["eval_seed"],
        "seed_policy": "v2",
    }
    if finetune_epochs is not None:
        tags["finetune_epochs"] = finetune_epochs
    if finetune_lr_factor is not None:
        tags["finetune_lr_factor"] = finetune_lr_factor
    if loss_value is not None:
        tags["loss_function"] = loss_value
    if robustness_method is not None:
        tags["robustness_method"] = robustness_method
    _assert_no_reserved_tag_overlap(effective_hparams, context="finetune_single_run hparams")
    _assert_no_reserved_tag_overlap(extra_tags, context="finetune_single_run extra_tags")
    tags.update(effective_hparams)
    tags.update(_extract_train_augmentation_identity_tags(datamodule_kwargs))
    if signature is not None:
        tags["signature"] = signature
    if extra_tags:
        tags.update(extra_tags)
    tags["train_commit"] = _current_git_commit()

    tags = {k: str(v) for k, v in tags.items()}
    return _fit_and_finalize(
        args=args,
        client=client,
        dataset_name=dataset_name,
        run_name=model_name,
        run_id=run_id,
        tags=tags,
        model=model,
        datamodule=dm,
        max_epochs=int(finetune_epochs),
        hparams_to_log=effective_hparams,
        stage=stage,
        signature=signature,
        params_to_log={
            "backbone_run_id": backbone_run_id,
            "finetune_epochs": int(finetune_epochs),
            "finetune_lr_factor": float(finetune_lr_factor),
            "finetune_lr": finetune_lr,
        },
    )


__all__ = [
    "ResolvedRuns",
    "apply_optimizer_hparams_to_model",
    "build_model_name",
    "delete_runs",
    "ensure_experiment_data_signature",
    "get_tracking_uri",
    "iter_limited_product",
    "finetune_single_run",
    "merge_optimizer_hparams",
    "normalize_mlflow_run_name",
    "optimizer_hparams_from_args",
    "optimizer_identity_hparams",
    "resolve_runs",
    "search_runs_all",
    "train_single_run",
]
