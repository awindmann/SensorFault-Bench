from __future__ import annotations

import copy
import gc
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

import models
from config_loader import load_defaults, load_hparams
from data.data_module import TSDataModule
from data.dataset import PerturbedDataset
from data.perturbations import (
    build_perturbation_scenario_params_signature,
    fixed_channel_count_for_fraction,
    require_perturbation_channel_scope,
)
from data.samplers import uniform_severity
from improvements import get_registration_by_loader_kind
from models.pretrained_loader import PRETRAINED_MODEL_LOADERS
from pipelines.ranking import (
    perturbed_selection_metric_keys,
    rank_key_for_row_values,
    rank_key_for_run,
    selection_metric_key_for_kind,
    winner_selection_mode_for_method,
)
from pipelines.runner import PipelineRunner
from pipelines.selection import (
    audit_and_apply_testing_coverage_policy,
    build_base_index,
    classify_lineage_run,
    CoverageMismatchError,
    expand_testing_method_scope_for_wrap_dependencies,
    has_explicit_architecture_scope,
    is_fully_tested,
    is_fixed_channel_fraction_complete,
    load_benchmark_recipe_specs_for_scope,
    require_seed_tags,
    resolve_pipeline_tags,
    resolve_benchmark_method_architecture_scope,
    resolve_requested_architectures,
    resolve_requested_methods,
    scope_exclusion_reason as shared_scope_exclusion_reason,
)
from pipelines.signatures import compute_data_config_signature
from pipelines.specs import PipelineSpec
from mlflow.entities import ViewType
from pipelines.training import search_runs_all
from metrics.loss import resolve_stateless_loss
from utils.env import current_git_commit
from utils.parsing import (
    DEGRADATION_SCORING_SEMANTICS,
    ROBUSTNESS_RESULTS_COMPLETE_TAG,
    SelectionPerturbationContextNotReadyError,
    build_shared_anchor_bootstrap_ci_tag_payload,
    build_selection_perturbation_context_tag_payload,
    build_degradation_eval_context_tag_payload,
    build_winner_selection_provenance_tag_payload,
    build_mlflow_tracking_uri,
    build_seeded_eval_input_artifact_prefix,
    build_seeded_degradation_artifact_prefix,
    format_fixed_channel_fraction_token,
    optional_nonempty_tag_value,
    parse_perturbation_scenarios,
    parse_perturbation_channel_fraction_max,
    parse_optional_name_tuple,
    parse_optional_nonempty_string,
    parse_optional_unit_float,
    parse_runtime_precision,
    parse_required_positive_int,
    parse_required_bool,
    require_selection_perturbation_context_from_args,
    require_selection_perturbation_context_tags,
    require_shared_anchor_bootstrap_ci_context_from_args,
    require_shared_anchor_bootstrap_ci_context_tags,
    require_degradation_eval_context_from_args,
    require_degradation_eval_context_tags,
    require_dataframe_columns,
    require_integer_series,
    require_nonempty_tag_value,
    require_improvement_selection_mode,
    require_namespace_bool,
    require_namespace_value,
    require_stage_tag,
    robustness_results_complete_tag_value,
    resolve_mlflow_local_save_dir,
    resolve_effective_eval_data_seed,
)
from utils.artifacts import (
    download_best_checkpoint,
    load_lightning_module_checkpoint,
    require_downloaded_checkpoint_unlinker,
    unlink_downloaded_checkpoint,
)
from utils.rng import derive_seed, set_seed

from testing.shared import (
    _configure_runtime_loggers_for_testing,
    _raise_non_finished_winner,
    _require_best_model_current_tags,
    _require_single_pipeline_kind,
    _suppress_lightning_worker_warning,
)
from utils.scoring import (
    build_canonical_degradation_context_signature,
    build_degradation_metric_key,
    build_degradation_scenario_metric_key,
    build_fixed_channel_fraction_artifact_prefix,
    build_fixed_channel_fraction_context_signature,
    build_fixed_channel_fraction_metric_key,
    build_fixed_channel_fraction_scenario_metric_key,
    build_fixed_channel_fraction_tag_key,
    download_validated_degradation_artifact_bundle,
    score_degradation_artifact_bundle,
    validate_fixed_channel_fraction_artifact_bundle,
)
from visualizations.plots import (
    plot_robustness_vs_performance,
    plot_scenario_radar,
)


def _build_testing_datamodule(
    *,
    dataset_spec,
    args,
    canonical_data_seed: int,
    eval_data_seed: int,
    val_seed: int | None = None,
    setup_dm: bool = True,
) -> TSDataModule:
    set_seed(eval_data_seed)
    pl.seed_everything(eval_data_seed, workers=True)
    # Keep train/val tied to the canonical experiment data seed.
    # Route eval_data_seed only into the test realization.
    dm = TSDataModule(
        dataset_spec=dataset_spec,
        input_len=args.input_len,
        target_len=args.target_len,
        n_train_samples=args.n_train_samples,
        n_val_samples=args.n_val_samples,
        n_test_samples=args.n_test_samples,
        perturbation_channel_fraction_max=args.perturbation_channel_fraction_max,
        perturbation_scenarios=args.perturbation_scenarios,
        train_split=args.train_split,
        val_split=args.val_split,
        purged_fraction=args.purged_fraction,
        shuffle_batches_before_split=args.shuffle_batches_before_split,
        strict_iid=require_namespace_bool(args, key="strict_iid"),
        batch_size=args.batch_size,
        num_workers=0,
        seed=canonical_data_seed,
        val_seed=val_seed,
        test_seed=eval_data_seed,
        s3_endpoint=args.minio_endpoint,
        train_noise_std=0.0,
    )
    if setup_dm:
        dm.setup()
    return dm


class _ScopedArtifactClient:
    def __init__(self, client, dst_root: str):
        if client is None:
            raise ValueError("Scoped artifact client requires a backing client.")
        if not dst_root:
            raise ValueError("Scoped artifact client requires a non-empty dst_root.")
        self._client = client
        self._dst_root = str(dst_root)

    def download_artifacts(self, run_id: str, path: str, dst_path: str | None = None) -> str:
        target_dst = self._dst_root if dst_path is None else dst_path
        return self._client.download_artifacts(run_id, path, dst_path=target_dst)

    def unlink_downloaded_checkpoint(
        self,
        checkpoint_path: str | os.PathLike[str],
        *,
        run_id: str,
        context: str,
    ) -> None:
        local_path = os.path.abspath(os.fspath(checkpoint_path))
        dst_root = os.path.abspath(self._dst_root)
        if os.path.commonpath([local_path, dst_root]) != dst_root:
            raise ValueError(
                f"Refusing to remove checkpoint outside scoped artifact root "
                f"for run {run_id}: checkpoint='{local_path}', root='{dst_root}'."
            )
        unlink_downloaded_checkpoint(
            local_path,
            run_id=run_id,
            context=context,
        )

    def __getattr__(self, name: str):
        return getattr(self._client, name)


torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class DatasetTestingCoverageScope:
    client: Any
    dataset_name: str
    experiment_id: str
    data_config_signature: str
    data_seed: int
    n_runs: int
    parent_runs: Sequence[Any]
    all_runs: Sequence[Any]
    resolved_by_run_id: Mapping[str, dict[str, Any]]
    runs_by_variant: Mapping[tuple[str, str, str], Sequence[Any]]
    selection_current_base_runs_by_key: Mapping[tuple[str, str], Any]
    requested_architectures: frozenset[str]
    selected_methods: frozenset[str]
    stale_run_ids: frozenset[str]
    stale_reasons: Mapping[str, str]
    dataset_coverage_fractions: Mapping[tuple[str, str], tuple[int, int]]
    eval_data_seed: int


def _clear_out_of_scope_best_model_tags(
    client,
    *,
    parent_runs: Sequence[Any],
    new_winner_ids: set[str],
    cleanup_architectures: set[str],
    cleanup_methods: set[str],
) -> None:
    """Clear stale winner tags inside the active testing scope only."""
    normalized_architectures = {
        str(architecture).strip() for architecture in cleanup_architectures
    }
    normalized_methods = {str(method).strip() for method in cleanup_methods}
    if not normalized_architectures:
        raise ValueError("best_model cleanup requires a non-empty architecture scope.")
    if not normalized_methods:
        raise ValueError("best_model cleanup requires a non-empty method scope.")
    if "" in normalized_architectures:
        raise ValueError("best_model cleanup architecture scope contains an empty value.")
    if "" in normalized_methods:
        raise ValueError("best_model cleanup method scope contains an empty value.")

    for run in parent_runs:
        run_id = run.info.run_id
        if run_id in new_winner_ids:
            continue
        run_tags = run.data.tags
        if run_tags is None:
            raise ValueError(
                f"Run {run_id} in parent run pool is missing tags during best_model cleanup."
            )
        best_model_raw = run_tags.get("best_model")
        if best_model_raw is None or not str(best_model_raw).strip():
            continue
        best_model = parse_required_bool(
            best_model_raw,
            key="best_model tag",
            context=f"Run {run_id}",
        )
        if not best_model:
            continue
        model_architecture = require_nonempty_tag_value(
            run_tags,
            key="model_architecture",
            run_id=run_id,
        )
        pipeline_method = require_nonempty_tag_value(
            run_tags,
            key="pipeline_method",
            run_id=run_id,
        )
        _require_best_model_current_tags(run_tags, run_id=run_id)
        resolved_tags = resolve_pipeline_tags(run_tags, run_id=run_id)
        if model_architecture not in normalized_architectures:
            continue
        if resolved_tags["pipeline_method"] not in normalized_methods:
            continue
        client.set_tag(run_id, "best_model", "false")
        client.set_tag(run_id, "backbone_current", "false")
        _replace_winner_selection_provenance_tags(client, run)


_WINNER_SELECTION_PROVENANCE_TAG_KEYS: tuple[str, ...] = (
    "winner_selection_mode",
    "winner_selection_metric_name",
    "winner_selection_metric_semantics",
    "winner_selection_perturbation_channel_fraction_max",
    "winner_selection_perturbation_scenarios_signature",
)


def _replace_winner_selection_provenance_tags(
    client,
    run,
    *,
    tag_payload: Mapping[str, str] | None = None,
) -> None:
    """Replace winner-selection provenance tags with the canonical current payload."""
    tags = run.data.tags
    if tags is None:
        raise ValueError(
            f"Run {run.info.run_id} is missing tags required for winner-selection provenance updates."
        )
    delete_tag = getattr(client, "delete_tag", None)
    for key in _WINNER_SELECTION_PROVENANCE_TAG_KEYS:
        if key not in tags:
            continue
        tags.pop(key, None)
        if callable(delete_tag):
            delete_tag(run.info.run_id, key)
    if tag_payload is None:
        return
    for key, value in tag_payload.items():
        client.set_tag(run.info.run_id, key, value)
        tags[key] = value


def _load_model_from_run(client, run, target_class):
    if not hasattr(models, target_class):
        raise ValueError(f"Unknown model class '{target_class}' for run {run.info.run_id}.")

    cleanup_checkpoint = require_downloaded_checkpoint_unlinker(
        client,
        context=f"standard loader for {target_class}",
    )
    model_class = getattr(models, target_class)
    checkpoint_path = download_best_checkpoint(client, run.info.run_id)
    model = load_lightning_module_checkpoint(model_class, checkpoint_path)
    cleanup_checkpoint(
        checkpoint_path,
        run_id=run.info.run_id,
        context=f"standard loader for {target_class}",
    )
    default_root_dir = os.path.dirname(checkpoint_path)
    return model, default_root_dir


def _resolve_model_loading_identity_for_run(
    run_tags: Mapping[str, Any],
    *,
    run_id: str,
) -> tuple[str, str]:
    model_arch = require_nonempty_tag_value(
        run_tags,
        key="model_architecture",
        run_id=run_id,
    )
    loader_kind = require_nonempty_tag_value(
        run_tags,
        key="loader_kind",
        run_id=run_id,
    )
    return model_arch, loader_kind


def load_model_with_loader(client, run, args, dm: TSDataModule | None = None):
    artifact_tmpdir = tempfile.TemporaryDirectory(prefix="robust-eval-")
    scoped_client = _ScopedArtifactClient(client, artifact_tmpdir.name)
    try:
        run_tags = run.data.tags
        if run_tags is None:
            raise ValueError(
                f"Run {run.info.run_id} is missing tags required for model loading."
            )
        model_arch, loader_kind = _resolve_model_loading_identity_for_run(
            run_tags,
            run_id=run.info.run_id,
        )

        registration = None
        try:
            registration = get_registration_by_loader_kind(loader_kind)
        except KeyError:
            registration = None
        if loader_kind == "pretrained":
            if dm is None:
                raise ValueError(
                    f"Run {run.info.run_id} uses loader_kind='pretrained' and requires "
                    "a datamodule for reconstruction."
                )
            loader = PRETRAINED_MODEL_LOADERS.get(model_arch.lower())
            if loader is None:
                raise ValueError(
                    f"Unknown pretrained model_architecture '{model_arch}' for run "
                    f"{run.info.run_id}."
                )
            model, default_root_dir = loader(scoped_client, run, args, dm)
        elif registration is not None:
            model, default_root_dir = registration.builder(scoped_client, run)
        else:
            if loader_kind.lower() != model_arch.lower():
                raise ValueError(
                    f"Unknown loader_kind '{loader_kind}' for run {run.info.run_id}."
                )

            pipeline_kind = resolve_pipeline_tags(
                run_tags,
                run_id=run.info.run_id,
            )["pipeline_kind"]
            if pipeline_kind == "wrap":
                raise ValueError(
                    f"Improvement run {run.info.run_id} has loader_kind='{loader_kind}', "
                    "which does not map to an improvement loader."
                )
            model, default_root_dir = _load_model_from_run(
                scoped_client, run, model_arch
            )
    except Exception:
        artifact_tmpdir.cleanup()
        raise
    try:
        model._artifact_tempdir_handle = artifact_tmpdir
    except Exception as exc:
        artifact_tmpdir.cleanup()
        raise ValueError(
            f"Loaded model for run {run.info.run_id} does not allow artifact tempdir attachment."
        ) from exc
    return model, default_root_dir


def _prepare_model_for_evaluation(
    model,
    args,
    dm,
    *,
    eval_seed: int | None = None,
):
    model.set_test_mode(test_metric=args.test_metric)
    hparams = getattr(model, "hparams", {})
    d_seq_in = getattr(hparams, "d_seq_in", None)
    d_seq_out = getattr(hparams, "d_seq_out", None)
    d_inputs = getattr(hparams, "d_input_features", None)
    d_outputs = getattr(hparams, "d_target_features", d_inputs)

    if args.input_len != d_seq_in:
        raise ValueError(
            f"Model input length ({d_seq_in}) does not match input length in config ({args.input_len})."
        )
    if args.target_len != d_seq_out:
        raise ValueError(
            f"Model prediction length ({d_seq_out}) does not match target length in config ({args.target_len})."
        )
    if d_inputs is not None and d_inputs != dm.n_inputs:
        raise ValueError(
            f"Model expects {d_inputs} input features but datamodule has {dm.n_inputs}."
        )
    if d_outputs is not None and d_outputs != dm.n_outputs:
        raise ValueError(
            f"Model expects {d_outputs} target features but datamodule has {dm.n_outputs}."
        )
    if getattr(model, "target_indices", None) is None and dm.target_column_indices is not None:
        model.target_indices = dm.target_column_indices
    if hasattr(model, "bind_eval_context"):
        model.bind_eval_context(
            input_columns=dm.input_columns,
            target_columns=dm.output_columns,
            continuous_channels=dm.continuous_channels,
            input_means=dm.input_means,
            input_stds=dm.input_stds,
        )
    if hasattr(model, "clear_noise_sample_ids"):
        model.clear_noise_sample_ids()
    if eval_seed is not None and hasattr(model, "set_noise_generator"):
        generator = torch.Generator().manual_seed(eval_seed)
        model.set_noise_generator(generator)


def _require_single_runtime_device_request(raw_devices, *, context_name: str) -> None:
    if raw_devices is None:
        raise ValueError(f"{context_name} requires args.devices to resolve a runtime device.")
    if isinstance(raw_devices, bool):
        raise ValueError(
            f"{context_name} requires a single runtime device request; got boolean devices={raw_devices!r}."
        )
    if isinstance(raw_devices, int):
        if int(raw_devices) != 1:
            raise ValueError(
                f"{context_name} supports exactly one runtime device; got devices={raw_devices}."
            )
        return
    if isinstance(raw_devices, (list, tuple)):
        if len(raw_devices) != 1:
            raise ValueError(
                f"{context_name} supports exactly one runtime device; got devices={raw_devices!r}."
            )
        return
    token = str(raw_devices).strip()
    if not token:
        raise ValueError(f"{context_name} requires a non-empty args.devices value.")
    normalized = [part.strip() for part in token.split(",") if part.strip()]
    if len(normalized) != 1:
        raise ValueError(
            f"{context_name} supports exactly one runtime device; got devices={raw_devices!r}."
        )


def _resolve_requested_runtime_device(args, *, context_name: str) -> torch.device:
    accelerator = parse_optional_nonempty_string(
        require_namespace_value(args, key="accelerator"),
        key="args.accelerator",
        context=context_name,
        disallow_none_token=True,
    )
    if accelerator is None:
        raise ValueError(f"{context_name} requires args.accelerator to resolve a runtime device.")
    _require_single_runtime_device_request(
        require_namespace_value(args, key="devices"),
        context_name=context_name,
    )
    accelerator_token = accelerator.lower()
    if accelerator_token == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if accelerator_token in {"gpu", "cuda"}:
        if not torch.cuda.is_available():
            raise ValueError(
                f"{context_name} requested accelerator={accelerator!r}, but CUDA is unavailable."
            )
        return torch.device("cuda", torch.cuda.current_device())
    if accelerator_token == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError(
                f"{context_name} requested accelerator={accelerator!r}, but MPS is unavailable."
            )
        return torch.device("mps")
    if accelerator_token == "cpu":
        return torch.device("cpu")
    raise ValueError(
        f"{context_name} has unsupported args.accelerator={accelerator!r} for manual evaluation."
    )


def _resolve_manual_eval_precision_from_trainer(
    trainer: pl.Trainer,
    *,
    runtime_device: torch.device,
    context_name: str,
):
    precision_plugin = getattr(trainer, "precision_plugin", None)
    if precision_plugin is None or not hasattr(precision_plugin, "precision"):
        raise ValueError(
            f"{context_name} requires trainer.precision_plugin.precision to resolve "
            "manual evaluation precision."
        )
    return parse_runtime_precision(
        getattr(precision_plugin, "precision"),
        device_type=runtime_device.type,
        key="trainer.precision_plugin.precision",
        context=context_name,
    )


@contextmanager
def _manual_eval_runtime_context(
    model,
    *,
    runtime_device: torch.device,
    runtime_precision,
):
    if runtime_precision.model_dtype is None:
        model.to(runtime_device)
    else:
        model.to(device=runtime_device, dtype=runtime_precision.model_dtype)
    autocast_context = nullcontext()
    if runtime_precision.autocast_dtype is not None:
        autocast_context = torch.autocast(
            device_type=runtime_device.type,
            dtype=runtime_precision.autocast_dtype,
        )
    with autocast_context:
        yield


def _move_tensor_to_manual_eval_runtime(
    tensor: torch.Tensor,
    *,
    runtime_device: torch.device,
    runtime_precision,
) -> torch.Tensor:
    if runtime_precision.input_dtype is not None and tensor.is_floating_point():
        return tensor.to(device=runtime_device, dtype=runtime_precision.input_dtype)
    return tensor.to(runtime_device)


def _prepare_tensor_for_host_export(tensor: torch.Tensor) -> torch.Tensor:
    host_tensor = tensor.detach().cpu()
    if (
        host_tensor.is_floating_point()
        and host_tensor.dtype in {torch.float16, torch.bfloat16}
    ):
        return host_tensor.to(dtype=torch.float32)
    return host_tensor


def _bind_model_noise_sample_ids(
    model,
    sample_ids: Sequence[int],
    *,
    context_key: str,
) -> None:
    if hasattr(model, "bind_noise_sample_ids"):
        model.bind_noise_sample_ids(sample_ids, context_key=context_key)


def _clear_model_noise_sample_ids(model) -> None:
    if hasattr(model, "clear_noise_sample_ids"):
        model.clear_noise_sample_ids()


def _build_degradation_batch_sample_ids(
    *,
    sample_offset: int,
    batch_size: int,
) -> list[int]:
    return [
        int(sample_offset + row_idx)
        for row_idx in range(int(batch_size))
    ]


def _require_degradation_source_sample_idx(
    base_dataset,
    *,
    sample_offset: int,
    batch_size: int,
    context_name: str,
) -> np.ndarray:
    source_sample_idx = np.asarray(
        base_dataset.sample_idxs[sample_offset: sample_offset + batch_size],
        dtype=np.int64,
    )
    if int(source_sample_idx.size) != int(batch_size):
        raise ValueError(
            "Base test dataset sample_idxs do not align with the "
            f"{context_name} loader."
        )
    return source_sample_idx


def _predict_with_bound_noise_sample_ids(
    model,
    inputs: torch.Tensor,
    sample_ids: Sequence[int],
    *,
    context_key: str,
) -> torch.Tensor:
    _bind_model_noise_sample_ids(
        model,
        sample_ids,
        context_key=context_key,
    )
    try:
        return model(inputs).detach()
    finally:
        _clear_model_noise_sample_ids(model)


def _score_degradation_predictions(
    metric_fn,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
    context_name: str,
) -> np.ndarray:
    errors = _prepare_tensor_for_host_export(metric_fn(predictions, targets)).numpy()
    if errors.ndim != 1 or errors.shape[0] != int(batch_size):
        raise ValueError(
            "test_metric_fn must return one scalar per sample in "
            f"{context_name}; received shape {errors.shape} "
            f"for batch_size={batch_size}."
        )
    return errors


@contextmanager
def _configure_eval_matmul_precision_for_device(device: torch.device | str | None):
    if device is None:
        yield
        return
    resolved_device = device
    if not isinstance(resolved_device, torch.device):
        resolved_device = torch.device(str(resolved_device))
    if resolved_device.type != "cuda":
        yield
        return
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("medium")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)


def _require_degradation_runtime(
    dm: TSDataModule,
    *,
    context_name: str,
) -> tuple[Any, Any, tuple[str, ...], list[Any]]:
    perturbed_dataset, base_dataset = _require_degradation_test_dataset(dm)
    scenario_names = parse_perturbation_scenarios(
        getattr(dm, "perturbation_names", None),
        key="datamodule perturbation_names",
    )
    perturbations = getattr(getattr(dm, "pert_sampler", None), "perturbations", None)
    if not isinstance(perturbations, list) or len(perturbations) != len(scenario_names):
        raise ValueError(
            "Datamodule perturbation sampler is missing perturbations required for "
            f"{context_name}."
        )
    return perturbed_dataset, base_dataset, tuple(scenario_names), perturbations


def _require_perturbed_validation_runtime(
    dm: TSDataModule,
    *,
    perturbation_seed: int,
    context_name: str,
) -> tuple[Any, Any, tuple[str, ...], list[Any]]:
    base_dataset = getattr(dm, "ds_val", None)
    if base_dataset is None:
        raise ValueError(
            "Datamodule validation dataset is not initialized for perturbed validation selection."
        )
    pert_sampler = getattr(dm, "pert_sampler", None)
    if pert_sampler is None:
        raise ValueError(
            f"Datamodule perturbation sampler is missing perturbations required for {context_name}."
        )
    perturbations = getattr(pert_sampler, "perturbations", None)
    scenario_names = parse_perturbation_scenarios(
        getattr(dm, "perturbation_names", None),
        key="datamodule perturbation_names",
    )
    if not isinstance(perturbations, list) or len(perturbations) != len(scenario_names):
        raise ValueError(
            f"Datamodule perturbation sampler is missing perturbations required for {context_name}."
        )
    perturbed_dataset = PerturbedDataset(
        base_dataset,
        pert_sampler,
        seed=int(perturbation_seed),
    )
    return perturbed_dataset, base_dataset, tuple(scenario_names), perturbations


def _build_perturbed_split_error_frames(
    *,
    model,
    dm: TSDataModule,
    args,
    split_name: str,
    perturbation_seed: int,
    context_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    if split_name == "test":
        perturbed_dataset, base_dataset, scenario_names, perturbations = (
            _require_degradation_runtime(
                dm,
                context_name=context_name,
            )
        )
    elif split_name == "val":
        perturbed_dataset, base_dataset, scenario_names, perturbations = (
            _require_perturbed_validation_runtime(
                dm,
                perturbation_seed=perturbation_seed,
                context_name=context_name,
            )
        )
    else:
        raise ValueError(
            f"Unsupported perturbed split '{split_name}' in {context_name}."
        )

    runtime_device = _resolve_requested_runtime_device(
        args,
        context_name=context_name,
    )
    runtime_precision = parse_runtime_precision(
        require_namespace_value(args, key="precision"),
        device_type=runtime_device.type,
        key="args.precision",
        context=context_name,
    )
    clean_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    with _configure_eval_matmul_precision_for_device(runtime_device):
        model.eval()
        with _manual_eval_runtime_context(
            model,
            runtime_device=runtime_device,
            runtime_precision=runtime_precision,
        ):
            metric_fn = resolve_stateless_loss(args.test_metric)
            n_clean_samples = int(len(base_dataset))
            clean_loader = DataLoader(
                base_dataset,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )

            sample_offset = 0
            with torch.no_grad():
                for x_cpu, y_cpu in clean_loader:
                    batch_size = int(x_cpu.size(0))
                    source_sample_idx = _require_degradation_source_sample_idx(
                        base_dataset,
                        sample_offset=sample_offset,
                        batch_size=batch_size,
                        context_name=context_name,
                    )
                    x_device = _move_tensor_to_manual_eval_runtime(
                        x_cpu,
                        runtime_device=runtime_device,
                        runtime_precision=runtime_precision,
                    )
                    y_device = _move_tensor_to_manual_eval_runtime(
                        y_cpu,
                        runtime_device=runtime_device,
                        runtime_precision=runtime_precision,
                    )
                    batch_sample_ids = _build_degradation_batch_sample_ids(
                        sample_offset=sample_offset,
                        batch_size=batch_size,
                    )
                    pred_clean = _predict_with_bound_noise_sample_ids(
                        model,
                        x_device,
                        batch_sample_ids,
                        context_key=f"{context_name}:clean",
                    )
                    err_clean = _score_degradation_predictions(
                        metric_fn,
                        pred_clean,
                        y_device,
                        batch_size=batch_size,
                        context_name=context_name,
                    )

                    for row_idx in range(batch_size):
                        sample_id = int(sample_offset + row_idx)
                        clean_rows.append(
                            {
                                "sample_id": sample_id,
                                "source_sample_idx": int(source_sample_idx[row_idx]),
                                "err_clean": float(err_clean[row_idx]),
                            }
                        )

                    for scenario_idx, (scenario_name, perturbation) in enumerate(
                        zip(scenario_names, perturbations)
                    ):
                        x_pert_cpu, y_pert_cpu, severities, _ = _build_degradation_scenario_batch(
                            perturbation=perturbation,
                            x_cpu=x_cpu,
                            y_cpu=y_cpu,
                            batch_sample_ids=batch_sample_ids,
                            eval_data_seed=perturbation_seed,
                            scenario_idx=scenario_idx,
                            cont_idx=perturbed_dataset.cont_idx,
                            disc_idx=perturbed_dataset.disc_idx,
                            context_name=context_name,
                        )
                        pred_pert = _predict_with_bound_noise_sample_ids(
                            model,
                            _move_tensor_to_manual_eval_runtime(
                                x_pert_cpu,
                                runtime_device=runtime_device,
                                runtime_precision=runtime_precision,
                            ),
                            batch_sample_ids,
                            context_key=f"{context_name}:scenario:{scenario_idx}",
                        )
                        err_pert = _score_degradation_predictions(
                            metric_fn,
                            pred_pert,
                            _move_tensor_to_manual_eval_runtime(
                                y_pert_cpu,
                                runtime_device=runtime_device,
                                runtime_precision=runtime_precision,
                            ),
                            batch_size=batch_size,
                            context_name=f"{context_name} scenario evaluation",
                        )
                        for row_idx in range(batch_size):
                            sample_id = int(sample_offset + row_idx)
                            source_idx = int(source_sample_idx[row_idx])
                            scenario_rows.append(
                                {
                                    "sample_id": sample_id,
                                    "source_sample_idx": source_idx,
                                    "pert_idx": int(scenario_idx),
                                    "scenario": str(scenario_name),
                                    "severity": float(severities[row_idx]),
                                    "err_pert": float(err_pert[row_idx]),
                                }
                            )
                    sample_offset += batch_size

    if sample_offset != n_clean_samples:
        raise ValueError(
            f"{context_name} consumed {sample_offset} clean anchors but expected "
            f"{n_clean_samples}."
        )

    clean_df = pd.DataFrame(
        clean_rows,
        columns=["sample_id", "source_sample_idx", "err_clean"],
    )
    scenario_samples_df = pd.DataFrame(
        scenario_rows,
        columns=[
            "sample_id",
            "source_sample_idx",
            "pert_idx",
            "scenario",
            "severity",
            "err_pert",
        ],
    )
    return clean_df, scenario_samples_df, tuple(scenario_names)


def _build_degradation_scenario_batch(
    *,
    perturbation,
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    batch_sample_ids: Sequence[int],
    eval_data_seed: int,
    scenario_idx: int,
    cont_idx,
    disc_idx,
    context_name: str,
    capture_affected_channels: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[float], list[Sequence[int] | None] | None]:
    x_pert_rows: list[torch.Tensor] = []
    y_pert_rows: list[torch.Tensor] = []
    severities: list[float] = []
    affected_channels_by_row: list[Sequence[int] | None] | None = (
        [] if capture_affected_channels else None
    )
    for row_idx, sample_id in enumerate(batch_sample_ids):
        rng = torch.Generator().manual_seed(
            _degradation_scenario_seed(
                eval_data_seed,
                scenario_idx=int(scenario_idx),
                sample_id=int(sample_id),
            )
        )
        severity = float(uniform_severity(rng))
        x_pert, y_pert, affected_channels = perturbation(
            x_cpu[row_idx].clone(),
            y_cpu[row_idx].clone(),
            severity,
            rng,
            cont_idx,
            disc_idx,
            channel_count_mode="severity",
            channel_count_value=severity,
        )
        if y_pert.shape != y_cpu[row_idx].shape or not torch.allclose(
            y_pert,
            y_cpu[row_idx],
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                f"{context_name} requires perturbations to leave targets unchanged."
            )
        x_pert_rows.append(x_pert.to(dtype=torch.float32))
        y_pert_rows.append(y_pert.to(dtype=torch.float32))
        severities.append(severity)
        if affected_channels_by_row is not None:
            affected_channels_by_row.append(affected_channels)
    return (
        torch.stack(x_pert_rows, dim=0),
        torch.stack(y_pert_rows, dim=0),
        severities,
        affected_channels_by_row,
    )


def _eligible_channels_for_perturbation_scope(
    perturbation,
    *,
    cont_idx: Sequence[int],
    disc_idx: Sequence[int],
    context_name: str,
) -> tuple[str, list[int] | None]:
    scope = require_perturbation_channel_scope(
        perturbation,
        context=context_name,
    )
    if scope == "continuous":
        return scope, [int(idx) for idx in cont_idx]
    if scope == "discrete":
        return scope, [int(idx) for idx in disc_idx]
    if scope == "all":
        return scope, None
    raise ValueError(f"{context_name} has unsupported channel_scope={scope!r}.")


def _build_fixed_channel_fraction_scenario_batch(
    *,
    perturbation,
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    batch_sample_ids: Sequence[int],
    eval_data_seed: int,
    scenario_idx: int,
    cont_idx,
    disc_idx,
    fixed_channel_fraction: float,
    perturbation_channel_fraction_max: float,
    context_name: str,
) -> tuple[torch.Tensor, torch.Tensor, list[float], list[dict[str, Any]]]:
    scope, eligible_channels = _eligible_channels_for_perturbation_scope(
        perturbation,
        cont_idx=cont_idx,
        disc_idx=disc_idx,
        context_name=context_name,
    )
    derived_fixed_channel_count: int | None = None
    eligible_channel_count: int | None = None
    if eligible_channels is not None:
        eligible_channel_count = len(eligible_channels)
        derived_fixed_channel_count = fixed_channel_count_for_fraction(
            eligible_channel_count,
            perturbation_channel_fraction_max,
            fixed_channel_fraction,
        )

    x_pert_rows: list[torch.Tensor] = []
    y_pert_rows: list[torch.Tensor] = []
    severities: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    for row_idx, sample_id in enumerate(batch_sample_ids):
        rng = torch.Generator().manual_seed(
            _degradation_scenario_seed(
                eval_data_seed,
                scenario_idx=int(scenario_idx),
                sample_id=int(sample_id),
            )
        )
        intensity_severity = float(uniform_severity(rng))
        x_pert, y_pert, affected_channels = perturbation(
            x_cpu[row_idx].clone(),
            y_cpu[row_idx].clone(),
            intensity_severity,
            rng,
            cont_idx,
            disc_idx,
            channel_count_mode="fixed_fraction",
            channel_count_value=fixed_channel_fraction,
        )
        if y_pert.shape != y_cpu[row_idx].shape or not torch.allclose(
            y_pert,
            y_cpu[row_idx],
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                f"{context_name} requires perturbations to leave targets unchanged."
            )
        selected_channel_count: int | None = None
        if derived_fixed_channel_count is not None:
            selected_channel_count = (
                0 if intensity_severity == 0.0 else int(derived_fixed_channel_count)
            )
        x_pert_rows.append(x_pert.to(dtype=torch.float32))
        y_pert_rows.append(y_pert.to(dtype=torch.float32))
        severities.append(intensity_severity)
        diagnostics.append(
            {
                "intensity_severity": float(intensity_severity),
                "requested_fixed_channel_fraction": (
                    float(fixed_channel_fraction) if scope != "all" else None
                ),
                "derived_fixed_channel_count": derived_fixed_channel_count,
                "channel_scope": scope,
                "eligible_channel_count": eligible_channel_count,
                "selected_channel_count": selected_channel_count,
                "reported_affected_channel_count": int(len(affected_channels)),
            }
        )
    return (
        torch.stack(x_pert_rows, dim=0),
        torch.stack(y_pert_rows, dim=0),
        severities,
        diagnostics,
    )


def _preflight_fixed_channel_fraction_channel_counts(
    perturbations: Sequence[Any],
    *,
    cont_idx: Sequence[int],
    disc_idx: Sequence[int],
    fixed_channel_fraction: float,
    perturbation_channel_fraction_max: float,
    context_name: str,
) -> None:
    for scenario_idx, perturbation in enumerate(perturbations):
        scope, eligible_channels = _eligible_channels_for_perturbation_scope(
            perturbation,
            cont_idx=cont_idx,
            disc_idx=disc_idx,
            context_name=f"{context_name} scenario/{scenario_idx}",
        )
        if scope == "all":
            continue
        if eligible_channels is None:
            raise ValueError(
                f"{context_name} scenario/{scenario_idx} has channel_scope={scope!r} "
                "but no eligible channel list."
            )
        fixed_channel_count_for_fraction(
            len(eligible_channels),
            perturbation_channel_fraction_max,
            fixed_channel_fraction,
        )


def _prime_model_for_degradation_evaluation(
    model,
    args,
    dm,
    *,
    eval_seed: int | None = None,
) -> None:
    if eval_seed is not None:
        set_seed(eval_seed)
        pl.seed_everything(eval_seed, workers=True)
    _prepare_model_for_evaluation(
        model,
        args,
        dm,
        eval_seed=eval_seed,
    )


def _make_eval_logger(
    run_id: str,
    dataset_name: str,
    args,
    *,
    log_test_commit: bool = True,
) -> MLFlowLogger:
    tracking_uri = build_mlflow_tracking_uri(args.logdir)
    logger = MLFlowLogger(
        tracking_uri=tracking_uri,
        save_dir=resolve_mlflow_local_save_dir(args.logdir),
        experiment_name=f"{dataset_name}",
        run_id=run_id,
    )
    if log_test_commit:
        logger.experiment.set_tag(logger.run_id, "test_commit", current_git_commit())
    return logger


def _resolve_bootstrap_ci_context(args, *, eval_data_seed: int) -> dict[str, Any]:
    test_metric = parse_optional_nonempty_string(
        require_namespace_value(args, key="test_metric"),
        key="args.test_metric",
        context="shared-anchor bootstrap context",
        disallow_none_token=True,
    )
    if test_metric is None:
        raise ValueError("args.test_metric is required for bootstrap CI.")
    return require_shared_anchor_bootstrap_ci_context_from_args(
        args,
        eval_data_seed=eval_data_seed,
        test_metric=test_metric,
        context="args",
    )


def _log_common_eval_params(
    logger: MLFlowLogger,
    args,
    dm: TSDataModule,
    *,
    bootstrap_ci_context: Mapping[str, Any] | None = None,
    eval_data_seed: int | None = None,
    seed_master: int | None = None,
    seed_data: int | None = None,
    seed_eval: int | None = None,
    log_degradation_context: bool = True,
) -> None:
    if log_degradation_context:
        if bootstrap_ci_context is None:
            raise ValueError(
                "bootstrap_ci_context is required when logging degradation evaluation context."
            )
        if eval_data_seed is None:
            raise ValueError(
                "eval_data_seed is required when logging degradation evaluation context."
            )
        eval_context = require_degradation_eval_context_from_args(
            args,
            eval_data_seed=eval_data_seed,
            context="args",
        )
        expected_scenarios = parse_perturbation_scenarios(
            getattr(dm, "perturbation_names", None),
            key="datamodule perturbation_names",
        )
        configured_scenarios = eval_context["perturbation_idx_name_map"]
        configured_order = tuple(
            configured_scenarios[idx] for idx in sorted(configured_scenarios)
        )
        if tuple(expected_scenarios) != configured_order:
            raise ValueError(
                "Datamodule perturbation_names do not match configured perturbation_scenarios "
                "order required for degradation evaluation."
            )
        logger.experiment.set_tag(
            logger.run_id,
            "robustness_scoring_semantics",
            DEGRADATION_SCORING_SEMANTICS,
        )
        logger.experiment.set_tag(
            logger.run_id,
            ROBUSTNESS_RESULTS_COMPLETE_TAG,
            robustness_results_complete_tag_value(complete=False),
        )
        for key, value in build_degradation_eval_context_tag_payload(
            eval_context,
            context_name="degradation_eval_context",
        ).items():
            logger.experiment.set_tag(logger.run_id, key, value)
        logger.experiment.set_tag(
            logger.run_id,
            "perturbation_scenario_params_signature",
            build_perturbation_scenario_params_signature(configured_order),
        )
        for key, value in build_shared_anchor_bootstrap_ci_tag_payload(
            bootstrap_ci_context,
            context_name="shared_anchor_bootstrap_ci_context",
        ).items():
            logger.experiment.set_tag(logger.run_id, key, value)
    logger.experiment.log_param(logger.run_id, "input_len", args.input_len)
    logger.experiment.log_param(logger.run_id, "target_len", args.target_len)
    logger.experiment.log_param(logger.run_id, "train_split", args.train_split)
    logger.experiment.log_param(logger.run_id, "val_split", args.val_split)
    master_seed = args.seed if seed_master is None else seed_master
    logger.experiment.log_param(logger.run_id, "seed_master", master_seed)
    if seed_data is not None:
        logger.experiment.log_param(logger.run_id, "seed_data", seed_data)
    if seed_eval is not None:
        logger.experiment.log_param(logger.run_id, "seed_eval", seed_eval)
    data_split_seed = require_namespace_value(args, key="data_split_seed")
    if data_split_seed is not None:
        logger.experiment.log_param(logger.run_id, "data_split_seed", data_split_seed)
    logger.experiment.set_tag(
        logger.run_id,
        "shuffle_batches_before_split",
        str(args.shuffle_batches_before_split),
    )


def _degradation_scenario_seed(
    eval_data_seed: int,
    *,
    scenario_idx: int,
    sample_id: int,
) -> int:
    return derive_seed(
        int(eval_data_seed),
        f"degradation:scenario:{int(scenario_idx)}:sample:{int(sample_id)}",
    )


def _log_optional_degradation_diagnostic_figures(
    *,
    logger: MLFlowLogger,
    artifact_path_prefix: str,
    model_name: str,
    mean_clean_error: float,
    scenario_summary_df: pd.DataFrame,
) -> None:
    if logger is None or not hasattr(logger, "experiment"):
        raise ValueError("MLflow logger with experiment handle is required during figure logging.")

    required_cols = {"scenario", "D"}
    missing_cols = sorted(required_cols - set(scenario_summary_df.columns))
    if missing_cols:
        raise ValueError(
            "Optional degradation diagnostic plots require scenario summary columns "
            f"{sorted(required_cols)}; missing {missing_cols}."
        )

    ordered_summary = scenario_summary_df.sort_values(
        ["pert_idx", "scenario"],
        kind="mergesort",
    ).reset_index(drop=True)
    scenario_names = [str(name) for name in ordered_summary["scenario"].tolist()]
    scenario_D_values = [float(value) for value in ordered_summary["D"].tolist()]
    scenario_D_map = {
        str(name): float(value)
        for name, value in zip(scenario_names, scenario_D_values)
    }
    performance_values = [float(mean_clean_error)] * len(scenario_names)

    try:
        fig_vs_perf = plot_robustness_vs_performance(
            "Scenario Degradation (D)",
            model_name,
            pert_names=scenario_names,
            metric_values=scenario_D_values,
            performance_values=performance_values,
        )
        logger.experiment.log_figure(
            logger.run_id,
            fig_vs_perf,
            artifact_file=f"{artifact_path_prefix}/d_vs_clean_error.pdf",
        )
    except Exception as exc:
        print(
            "Warning: Failed to log optional degradation degradation-vs-clean-error "
            f"figure for run {logger.run_id}: {exc}"
        )

    try:
        radar_fig = plot_scenario_radar(
            {str(model_name): scenario_D_map},
            title=f"{model_name} Scenario Degradation Profile",
        )
        logger.experiment.log_figure(
            logger.run_id,
            radar_fig,
            artifact_file=f"{artifact_path_prefix}/d_radar.pdf",
        )
    except Exception as exc:
        print(
            "Warning: Failed to log optional degradation scenario-radar figure "
            f"for run {logger.run_id}: {exc}"
        )


def _require_degradation_test_dataset(dm: TSDataModule):
    perturbed_dataset = getattr(dm, "ds_test", None)
    if perturbed_dataset is None:
        raise ValueError(
            "Datamodule test dataset is not initialized for degradation evaluation."
        )
    base_dataset = getattr(perturbed_dataset, "base_ds", None)
    if base_dataset is None:
        raise ValueError(
            "Datamodule test dataset is missing base_ds required for degradation evaluation."
        )
    sample_idxs = getattr(base_dataset, "sample_idxs", None)
    if sample_idxs is None:
        raise ValueError(
            "Base test dataset is missing sample_idxs required for degradation evaluation."
        )
    if not hasattr(perturbed_dataset, "cont_idx") or not hasattr(perturbed_dataset, "disc_idx"):
        raise ValueError(
            "Datamodule test dataset is missing perturbation channel indices required "
            "for degradation evaluation."
        )
    return perturbed_dataset, base_dataset


def _run_degradation_evaluation(
    *,
    model,
    trainer: pl.Trainer,
    logger: MLFlowLogger,
    dm: TSDataModule,
    args,
    eval_data_seed: int,
    bootstrap_ci_context: Mapping[str, Any],
) -> None:
    if logger is None or not hasattr(logger, "experiment"):
        raise ValueError("MLflow logger with experiment handle is required during test logging.")

    clean_df, scenario_samples_df, scenario_names = _build_perturbed_split_error_frames(
        model=model,
        dm=dm,
        args=args,
        split_name="test",
        perturbation_seed=eval_data_seed,
        context_name="degradation evaluation",
    )
    idx_to_name = {
        idx: str(name) for idx, name in enumerate(scenario_names)
    }
    clean_df, scenario_samples_df, scenario_summary_df, metric_bundle, worst_scenario_name = (
        score_degradation_artifact_bundle(
            clean_df,
            scenario_samples_df,
            expected_idx_to_name=idx_to_name,
            bootstrap_resamples=bootstrap_ci_context["bootstrap_ci_resamples"],
            bootstrap_confidence_level=bootstrap_ci_context[
                "bootstrap_ci_confidence_level"
            ],
            bootstrap_seed=bootstrap_ci_context["bootstrap_ci_seed"],
            context_name=f"Run {logger.run_id} degradation bundle",
        )
    )

    mean_clean_error = float(clean_df["err_clean"].mean())
    logger.experiment.log_metric(logger.run_id, f"{args.test_metric}_test", mean_clean_error)
    for metric_name in ("D_w", "D_mean", "err_pert_ws", "err_pert_mean"):
        logger.experiment.log_metric(
            logger.run_id,
            build_degradation_metric_key(
                test_metric=args.test_metric,
                metric_name=metric_name,
            ),
            float(metric_bundle[metric_name]),
        )
        logger.experiment.log_metric(
            logger.run_id,
            build_degradation_metric_key(
                test_metric=args.test_metric,
                metric_name=f"{metric_name}_CI_lo",
            ),
            float(metric_bundle[f"{metric_name}_CI_lo"]),
        )
        logger.experiment.log_metric(
            logger.run_id,
            build_degradation_metric_key(
                test_metric=args.test_metric,
                metric_name=f"{metric_name}_CI_hi",
            ),
            float(metric_bundle[f"{metric_name}_CI_hi"]),
        )
    for scenario_idx in sorted(idx_to_name):
        for metric_name in ("D", "D_CI_lo", "D_CI_hi", "err_pert", "err_pert_CI_lo", "err_pert_CI_hi"):
            logger.experiment.log_metric(
                logger.run_id,
                build_degradation_scenario_metric_key(
                    test_metric=args.test_metric,
                    scenario_idx=scenario_idx,
                    metric_name=metric_name,
                ),
                float(metric_bundle[f"scenario/{scenario_idx}/{metric_name}"]),
            )
    logger.experiment.set_tag(
        logger.run_id,
        build_degradation_metric_key(
            test_metric=args.test_metric,
            metric_name="worst_scenario",
        ),
        worst_scenario_name,
    )

    eval_input_artifact_prefix = build_seeded_eval_input_artifact_prefix(
        test_metric=args.test_metric,
        eval_data_seed=eval_data_seed,
    )
    artifact_path_prefix = build_seeded_degradation_artifact_prefix(
        test_metric=args.test_metric,
        eval_data_seed=eval_data_seed,
    )
    with tempfile.TemporaryDirectory(prefix="robust-") as tmpdir:
        clean_path = os.path.join(tmpdir, "clean_test_samples.csv")
        clean_df.to_csv(clean_path, index=False)
        logger.experiment.log_artifact(
            logger.run_id,
            clean_path,
            artifact_path=eval_input_artifact_prefix,
        )
        scenario_samples_path = os.path.join(tmpdir, "scenario_samples.csv")
        scenario_samples_df.to_csv(scenario_samples_path, index=False)
        logger.experiment.log_artifact(
            logger.run_id,
            scenario_samples_path,
            artifact_path=artifact_path_prefix,
        )
        scenario_summary_path = os.path.join(tmpdir, "scenario_summary.csv")
        scenario_summary_df.to_csv(scenario_summary_path, index=False)
        logger.experiment.log_artifact(
            logger.run_id,
            scenario_summary_path,
            artifact_path=artifact_path_prefix,
        )

    logger.experiment.set_tag(
        logger.run_id,
        ROBUSTNESS_RESULTS_COMPLETE_TAG,
        robustness_results_complete_tag_value(complete=True),
    )
    logger.experiment.log_param(logger.run_id, "tested", "true")
    model_name = getattr(model, "model_architecture", None) or model.__class__.__name__
    _log_optional_degradation_diagnostic_figures(
        logger=logger,
        artifact_path_prefix=artifact_path_prefix,
        model_name=str(model_name),
        mean_clean_error=mean_clean_error,
        scenario_summary_df=scenario_summary_df,
    )


def _validate_canonical_clean_anchor_replay(
    *,
    clean_df: pd.DataFrame,
    base_dataset,
    context_name: str,
) -> None:
    expected_sample_ids = [int(value) for value in clean_df["sample_id"].tolist()]
    if expected_sample_ids != list(range(len(expected_sample_ids))):
        raise ValueError(
            f"{context_name} canonical clean sample_id values must be contiguous from zero."
        )
    if len(base_dataset) < len(clean_df):
        raise ValueError(
            f"{context_name} base test dataset has {len(base_dataset)} rows but canonical "
            f"clean anchors require {len(clean_df)}."
        )
    source_sample_idx = getattr(base_dataset, "sample_idxs", None)
    if source_sample_idx is None:
        raise ValueError(
            f"{context_name} base test dataset is missing sample_idxs required for "
            "canonical anchor replay."
        )
    observed = [int(value) for value in source_sample_idx[: len(clean_df)]]
    expected = [int(value) for value in clean_df["source_sample_idx"].tolist()]
    if observed != expected:
        mismatches = [
            {
                "sample_id": sample_id,
                "expected_source_sample_idx": expected_idx,
                "observed_source_sample_idx": observed_idx,
            }
            for sample_id, (expected_idx, observed_idx) in enumerate(zip(expected, observed))
            if expected_idx != observed_idx
        ][:8]
        raise ValueError(
            f"{context_name} current datamodule does not reproduce canonical clean "
            f"anchor order. Examples: {mismatches}."
        )


def _build_fixed_channel_fraction_error_frames(
    *,
    model,
    dm: TSDataModule,
    args,
    clean_df: pd.DataFrame,
    eval_data_seed: int,
    fixed_channel_fraction: float,
    perturbation_channel_fraction_max: float,
    context_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    perturbed_dataset, base_dataset, scenario_names, perturbations = (
        _require_degradation_runtime(
            dm,
            context_name=context_name,
        )
    )
    _validate_canonical_clean_anchor_replay(
        clean_df=clean_df,
        base_dataset=base_dataset,
        context_name=context_name,
    )
    _preflight_fixed_channel_fraction_channel_counts(
        perturbations,
        cont_idx=perturbed_dataset.cont_idx,
        disc_idx=perturbed_dataset.disc_idx,
        fixed_channel_fraction=fixed_channel_fraction,
        perturbation_channel_fraction_max=perturbation_channel_fraction_max,
        context_name=context_name,
    )

    runtime_device = _resolve_requested_runtime_device(
        args,
        context_name=context_name,
    )
    runtime_precision = parse_runtime_precision(
        require_namespace_value(args, key="precision"),
        device_type=runtime_device.type,
        key="args.precision",
        context=context_name,
    )
    metric_fn = resolve_stateless_loss(args.test_metric)
    target_n_samples = int(len(clean_df))
    scenario_rows: list[dict[str, Any]] = []
    clean_loader = DataLoader(
        base_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    sample_offset = 0
    with _configure_eval_matmul_precision_for_device(runtime_device):
        model.eval()
        with _manual_eval_runtime_context(
            model,
            runtime_device=runtime_device,
            runtime_precision=runtime_precision,
        ):
            with torch.no_grad():
                for x_cpu, y_cpu in clean_loader:
                    if sample_offset >= target_n_samples:
                        break
                    batch_count = int(x_cpu.size(0))
                    remaining = target_n_samples - sample_offset
                    if batch_count > remaining:
                        batch_count = int(remaining)
                        x_cpu = x_cpu[:batch_count]
                        y_cpu = y_cpu[:batch_count]
                    source_sample_idx = _require_degradation_source_sample_idx(
                        base_dataset,
                        sample_offset=sample_offset,
                        batch_size=batch_count,
                        context_name=context_name,
                    )
                    batch_sample_ids = _build_degradation_batch_sample_ids(
                        sample_offset=sample_offset,
                        batch_size=batch_count,
                    )
                    for scenario_idx, (scenario_name, perturbation) in enumerate(
                        zip(scenario_names, perturbations)
                    ):
                        x_pert_cpu, y_pert_cpu, severities, diagnostics = (
                            _build_fixed_channel_fraction_scenario_batch(
                                perturbation=perturbation,
                                x_cpu=x_cpu,
                                y_cpu=y_cpu,
                                batch_sample_ids=batch_sample_ids,
                                eval_data_seed=eval_data_seed,
                                scenario_idx=scenario_idx,
                                cont_idx=perturbed_dataset.cont_idx,
                                disc_idx=perturbed_dataset.disc_idx,
                                fixed_channel_fraction=fixed_channel_fraction,
                                perturbation_channel_fraction_max=(
                                    perturbation_channel_fraction_max
                                ),
                                context_name=context_name,
                            )
                        )
                        pred_pert = _predict_with_bound_noise_sample_ids(
                            model,
                            _move_tensor_to_manual_eval_runtime(
                                x_pert_cpu,
                                runtime_device=runtime_device,
                                runtime_precision=runtime_precision,
                            ),
                            batch_sample_ids,
                            context_key=f"degradation evaluation:scenario:{scenario_idx}",
                        )
                        err_pert = _score_degradation_predictions(
                            metric_fn,
                            pred_pert,
                            _move_tensor_to_manual_eval_runtime(
                                y_pert_cpu,
                                runtime_device=runtime_device,
                                runtime_precision=runtime_precision,
                            ),
                            batch_size=batch_count,
                            context_name=f"{context_name} scenario evaluation",
                        )
                        for row_idx in range(batch_count):
                            sample_id = int(sample_offset + row_idx)
                            row = {
                                "sample_id": sample_id,
                                "source_sample_idx": int(source_sample_idx[row_idx]),
                                "pert_idx": int(scenario_idx),
                                "scenario": str(scenario_name),
                                "severity": float(severities[row_idx]),
                                "err_pert": float(err_pert[row_idx]),
                            }
                            row.update(diagnostics[row_idx])
                            scenario_rows.append(row)
                    sample_offset += batch_count

    if sample_offset != target_n_samples:
        raise ValueError(
            f"{context_name} consumed {sample_offset} canonical anchors but expected "
            f"{target_n_samples}."
        )

    scenario_samples_df = pd.DataFrame(
        scenario_rows,
        columns=[
            "sample_id",
            "source_sample_idx",
            "pert_idx",
            "scenario",
            "severity",
            "err_pert",
            "intensity_severity",
            "requested_fixed_channel_fraction",
            "derived_fixed_channel_count",
            "channel_scope",
            "eligible_channel_count",
            "selected_channel_count",
            "reported_affected_channel_count",
        ],
    )
    return clean_df.copy(), scenario_samples_df, tuple(scenario_names)


def _run_fixed_channel_fraction_evaluation(
    *,
    run,
    model,
    logger: MLFlowLogger,
    dm: TSDataModule,
    args,
    eval_data_seed: int,
    fixed_channel_fraction: float,
    bootstrap_ci_context: Mapping[str, Any],
    canonical_clean_df: pd.DataFrame,
    canonical_context_signature: str,
    perturbation_scenario_params_signature: str,
) -> None:
    if logger is None or not hasattr(logger, "experiment"):
        raise ValueError(
            "MLflow logger with experiment handle is required during fixed-channel-fraction "
            "logging."
        )
    perturbation_channel_fraction_max = float(args.perturbation_channel_fraction_max)
    logger.experiment.set_tag(
        logger.run_id,
        build_fixed_channel_fraction_tag_key(
            fixed_channel_fraction=fixed_channel_fraction,
            perturbation_channel_fraction_max=perturbation_channel_fraction_max,
            tag_name="complete",
        ),
        "false",
    )
    clean_df, scenario_samples_df, scenario_names = (
        _build_fixed_channel_fraction_error_frames(
            model=model,
            dm=dm,
            args=args,
            clean_df=canonical_clean_df,
            eval_data_seed=eval_data_seed,
            fixed_channel_fraction=fixed_channel_fraction,
            perturbation_channel_fraction_max=perturbation_channel_fraction_max,
            context_name=f"Run {run.info.run_id} fixed-channel-fraction",
        )
    )
    idx_to_name = {idx: str(name) for idx, name in enumerate(scenario_names)}
    canonical_projection = scenario_samples_df.loc[
        :,
        [
            "sample_id",
            "source_sample_idx",
            "pert_idx",
            "scenario",
            "severity",
            "err_pert",
        ],
    ]
    clean_df, _, scenario_summary_df, metric_bundle, worst_scenario_name = (
        score_degradation_artifact_bundle(
            clean_df,
            canonical_projection,
            expected_idx_to_name=idx_to_name,
            bootstrap_resamples=bootstrap_ci_context["bootstrap_ci_resamples"],
            bootstrap_confidence_level=bootstrap_ci_context[
                "bootstrap_ci_confidence_level"
            ],
            bootstrap_seed=bootstrap_ci_context["bootstrap_ci_seed"],
            context_name=f"Run {run.info.run_id} fixed-channel-fraction bundle",
        )
    )
    clean_df, scenario_samples_df, scenario_summary_df = (
        validate_fixed_channel_fraction_artifact_bundle(
            clean_df,
            scenario_samples_df,
            scenario_summary_df,
            expected_idx_to_name=idx_to_name,
            expected_n_test_samples=int(len(clean_df)),
            fixed_channel_fraction=fixed_channel_fraction,
            perturbation_channel_fraction_max=perturbation_channel_fraction_max,
            context_name=f"Run {run.info.run_id} fixed-channel-fraction bundle",
        )
    )

    for metric_name in ("D_w", "D_mean", "err_pert_ws", "err_pert_mean"):
        for suffix in ("", "_CI_lo", "_CI_hi"):
            key_name = f"{metric_name}{suffix}"
            logger.experiment.log_metric(
                logger.run_id,
                build_fixed_channel_fraction_metric_key(
                    test_metric=args.test_metric,
                    fixed_channel_fraction=fixed_channel_fraction,
                    perturbation_channel_fraction_max=perturbation_channel_fraction_max,
                    metric_name=key_name,
                ),
                float(metric_bundle[key_name]),
            )
    for scenario_idx in sorted(idx_to_name):
        for metric_name in (
            "D",
            "D_CI_lo",
            "D_CI_hi",
            "err_pert",
            "err_pert_CI_lo",
            "err_pert_CI_hi",
        ):
            logger.experiment.log_metric(
                logger.run_id,
                build_fixed_channel_fraction_scenario_metric_key(
                    test_metric=args.test_metric,
                    fixed_channel_fraction=fixed_channel_fraction,
                    perturbation_channel_fraction_max=perturbation_channel_fraction_max,
                    scenario_idx=scenario_idx,
                    metric_name=metric_name,
                ),
                float(metric_bundle[f"scenario/{scenario_idx}/{metric_name}"]),
            )
    logger.experiment.set_tag(
        logger.run_id,
        build_fixed_channel_fraction_metric_key(
            test_metric=args.test_metric,
            fixed_channel_fraction=fixed_channel_fraction,
            perturbation_channel_fraction_max=perturbation_channel_fraction_max,
            metric_name="worst_scenario",
        ),
        worst_scenario_name,
    )

    artifact_path_prefix = build_fixed_channel_fraction_artifact_prefix(
        test_metric=args.test_metric,
        eval_data_seed=eval_data_seed,
        fixed_channel_fraction=fixed_channel_fraction,
        perturbation_channel_fraction_max=perturbation_channel_fraction_max,
    )
    fraction_token = format_fixed_channel_fraction_token(
        fixed_channel_fraction,
        max_value=perturbation_channel_fraction_max,
    )
    context_payload = {
        "evaluation_family": "fixed_channel_fraction",
        "fixed_channel_fraction": float(fixed_channel_fraction),
        "fixed_channel_fraction_token": fraction_token,
        "test_metric": str(args.test_metric),
        "eval_data_seed": int(eval_data_seed),
        "n_test_samples": int(len(clean_df)),
        "perturbation_channel_fraction_max": perturbation_channel_fraction_max,
        "perturbation_scenarios_signature": require_degradation_eval_context_from_args(
            args,
            eval_data_seed=eval_data_seed,
            context="args",
        )["perturbation_scenarios_signature"],
        "perturbation_scenarios_count": int(len(idx_to_name)),
        "perturbation_idx_name_map": {
            str(idx): str(name) for idx, name in idx_to_name.items()
        },
        "perturbation_scenario_params_signature": perturbation_scenario_params_signature,
        "canonical_context_signature": canonical_context_signature,
        "bootstrap_ci_semantics": bootstrap_ci_context["bootstrap_ci_semantics"],
        "bootstrap_ci_resamples": int(bootstrap_ci_context["bootstrap_ci_resamples"]),
        "bootstrap_ci_confidence_level": float(
            bootstrap_ci_context["bootstrap_ci_confidence_level"]
        ),
        "bootstrap_ci_seed": int(bootstrap_ci_context["bootstrap_ci_seed"]),
        "artifact_prefix": artifact_path_prefix,
    }
    context_signature = build_fixed_channel_fraction_context_signature(
        context_payload
    )
    with tempfile.TemporaryDirectory(prefix="robust-") as tmpdir:
        clean_path = os.path.join(tmpdir, "clean_test_samples.csv")
        clean_df.to_csv(clean_path, index=False)
        logger.experiment.log_artifact(
            logger.run_id,
            clean_path,
            artifact_path=artifact_path_prefix,
        )
        scenario_samples_path = os.path.join(tmpdir, "scenario_samples.csv")
        scenario_samples_df.to_csv(scenario_samples_path, index=False)
        logger.experiment.log_artifact(
            logger.run_id,
            scenario_samples_path,
            artifact_path=artifact_path_prefix,
        )
        scenario_summary_path = os.path.join(tmpdir, "scenario_summary.csv")
        scenario_summary_df.to_csv(scenario_summary_path, index=False)
        logger.experiment.log_artifact(
            logger.run_id,
            scenario_summary_path,
            artifact_path=artifact_path_prefix,
        )
        context_path = os.path.join(tmpdir, "context.json")
        with open(context_path, "w", encoding="utf-8") as handle:
            json.dump(
                context_payload,
                handle,
                sort_keys=True,
                separators=(",", ":"),
            )
        logger.experiment.log_artifact(
            logger.run_id,
            context_path,
            artifact_path=artifact_path_prefix,
        )

    logger.experiment.set_tag(
        logger.run_id,
        build_fixed_channel_fraction_tag_key(
            fixed_channel_fraction=fixed_channel_fraction,
            perturbation_channel_fraction_max=perturbation_channel_fraction_max,
            tag_name="fixed_channel_fraction",
        ),
        str(float(fixed_channel_fraction)),
    )
    logger.experiment.set_tag(
        logger.run_id,
        build_fixed_channel_fraction_tag_key(
            fixed_channel_fraction=fixed_channel_fraction,
            perturbation_channel_fraction_max=perturbation_channel_fraction_max,
            tag_name="context_signature",
        ),
        context_signature,
    )
    logger.experiment.set_tag(
        logger.run_id,
        build_fixed_channel_fraction_tag_key(
            fixed_channel_fraction=fixed_channel_fraction,
            perturbation_channel_fraction_max=perturbation_channel_fraction_max,
            tag_name="complete",
        ),
        "true",
    )


def _log_perturbed_validation_selection_metrics(
    run,
    *,
    model,
    dm: TSDataModule,
    args,
    client,
) -> None:
    seeds = require_seed_tags(run)
    eval_seed = seeds["seed_eval"]
    pert_ws_key, pert_mean_key = perturbed_selection_metric_keys(
        test_metric=args.test_metric,
        run_id=run.info.run_id,
    )
    selection_context = require_selection_perturbation_context_from_args(
        args,
        context="args",
    )
    _prime_model_for_degradation_evaluation(
        model,
        args,
        dm,
        eval_seed=eval_seed,
    )
    clean_df, scenario_samples_df, scenario_names = _build_perturbed_split_error_frames(
        model=model,
        dm=dm,
        args=args,
        split_name="val",
        perturbation_seed=eval_seed,
        context_name="perturbed validation selection",
    )
    if clean_df.empty:
        raise ValueError(
            f"Run {run.info.run_id} perturbed validation selection produced no clean validation anchors."
        )
    if scenario_samples_df.empty:
        raise ValueError(
            f"Run {run.info.run_id} perturbed validation selection produced no perturbed validation samples."
        )
    scenario_mean_errors = (
        scenario_samples_df.groupby("scenario", sort=False)["err_pert"].mean()
    )
    missing_scenarios = [
        scenario_name
        for scenario_name in scenario_names
        if scenario_name not in scenario_mean_errors.index
    ]
    extra_scenarios = sorted(
        set(str(name) for name in scenario_mean_errors.index) - set(scenario_names)
    )
    if missing_scenarios or extra_scenarios:
        raise ValueError(
            f"Run {run.info.run_id} perturbed validation selection has mismatched scenarios: "
            f"missing={missing_scenarios}, extra={extra_scenarios}."
        )
    ordered_scenario_errors = scenario_mean_errors.reindex(list(scenario_names))
    if ordered_scenario_errors.isna().any():
        raise ValueError(
            f"Run {run.info.run_id} perturbed validation selection has non-finite scenario means."
        )
    pert_ws_val = float(ordered_scenario_errors.max())
    pert_mean_val = float(ordered_scenario_errors.mean())
    client.log_metric(run.info.run_id, pert_ws_key, pert_ws_val)
    client.log_metric(run.info.run_id, pert_mean_key, pert_mean_val)
    for key, value in build_selection_perturbation_context_tag_payload(
        selection_context,
        context_name="selection_perturbation_context",
    ).items():
        client.set_tag(run.info.run_id, key, value)
    client.set_tag(
        run.info.run_id,
        "selection_perturbation_scenario_params_signature",
        build_perturbation_scenario_params_signature(scenario_names),
    )


def ensure_selection_metric_for_run(
    candidate_run,
    *,
    arch: str,
    pipeline_method: str,
    pipeline_id: str,
    args,
    client,
    dataset_name: str,
    dm_holder: dict[str, TSDataModule],
    dm_factory,
    load_model_fn,
) -> float:
    resolved = resolve_pipeline_tags(candidate_run.data.tags, run_id=candidate_run.info.run_id)
    selection_mode = require_improvement_selection_mode(
        args,
        context="selection metric ensure args",
    )
    metric_key = selection_metric_key_for_kind(
        pipeline_kind=str(resolved["pipeline_kind"]),
        robustness_method=str(resolved["robustness_method"]),
        test_metric=args.test_metric,
        improvement_selection_mode=selection_mode,
        run_id=candidate_run.info.run_id,
    )
    perturbed_metric_keys = set(
        perturbed_selection_metric_keys(
            test_metric=args.test_metric,
            run_id=candidate_run.info.run_id,
        )
    )
    expected_selection_context: dict[str, Any] | None = None
    if metric_key in perturbed_metric_keys:
        expected_selection_context = require_selection_perturbation_context_from_args(
            args,
            context="args",
        )
        expected_selection_params_signature = (
            build_perturbation_scenario_params_signature(
                require_namespace_value(args, key="perturbation_scenarios")
            )
        )
    refresh_selection_metrics = require_namespace_bool(
        args,
        key="refresh_selection_metrics",
        context="selection metric ensure args",
    )
    rerun = require_namespace_bool(
        args,
        key="rerun",
        context="selection metric ensure args",
    )
    selection_refresh_requested = bool(refresh_selection_metrics) or bool(rerun)
    refresh_requested = (
        selection_refresh_requested
        and metric_key != "best_val_loss"
    )

    def _ready_metric_value(run) -> float | None:
        value = run.data.metrics.get(metric_key)
        if value is None:
            return None
        if expected_selection_context is not None:
            missing_companions = sorted(
                key
                for key in perturbed_metric_keys
                if run.data.metrics.get(key) is None
            )
            if missing_companions:
                return None
            try:
                require_selection_perturbation_context_tags(
                    run.data.tags,
                    run_id=run.info.run_id,
                    expected_context=expected_selection_context,
                )
            except SelectionPerturbationContextNotReadyError:
                return None
            params_signature = optional_nonempty_tag_value(
                run.data.tags,
                key="selection_perturbation_scenario_params_signature",
            )
            if params_signature != expected_selection_params_signature:
                return None
        return float(value)

    fresh_run = None
    if hasattr(client, "get_run"):
        fresh_run = client.get_run(candidate_run.info.run_id)

    if not refresh_requested:
        ready_value = _ready_metric_value(candidate_run)
        if ready_value is not None:
            return ready_value
        if fresh_run is not None:
            ready_value = _ready_metric_value(fresh_run)
            if ready_value is not None:
                return ready_value

    if metric_key == "best_val_loss":
        raise ValueError(
            f"Training-based run {candidate_run.info.run_id} for variant "
            f"({arch}, {pipeline_method}, {pipeline_id}) missing '{metric_key}'."
        )

    dm = dm_holder.get("selection")
    if dm is None:
        dm = dm_factory()
        dm_holder["selection"] = dm

    selection_run = fresh_run if fresh_run is not None else candidate_run
    model, default_root_dir = load_model_fn(selection_run)
    try:
        if metric_key == f"{args.test_metric}_val":
            validate_run(
                selection_run,
                model,
                default_root_dir,
                dataset_name,
                args,
                dm,
            )
        else:
            _log_perturbed_validation_selection_metrics(
                selection_run,
                model=model,
                dm=dm,
                args=args,
                client=client,
            )
    finally:
        try:
            _teardown_model_after_eval(model)
        finally:
            del model

    if hasattr(client, "get_run"):
        refreshed_run = client.get_run(candidate_run.info.run_id)
        ready_value = _ready_metric_value(refreshed_run)
    else:
        refreshed_run = candidate_run
        ready_value = _ready_metric_value(refreshed_run)
    if ready_value is not None:
        return ready_value

    value = refreshed_run.data.metrics.get(metric_key)
    if value is None:
        raise ValueError(
            f"Run {refreshed_run.info.run_id} is missing required selection metric '{metric_key}'."
        )
    if expected_selection_context is not None:
        missing_companions = sorted(
            key
            for key in perturbed_metric_keys
            if refreshed_run.data.metrics.get(key) is None
        )
        if missing_companions:
            raise ValueError(
                f"Run {refreshed_run.info.run_id} is missing required perturbed selection metric(s) "
                f"{missing_companions}."
            )
        require_selection_perturbation_context_tags(
            refreshed_run.data.tags,
            run_id=refreshed_run.info.run_id,
            expected_context=expected_selection_context,
        )
    return float(value)


def _build_winner_selection_provenance_tag_payload_for_run(
    run,
    *,
    args,
    test_metric: str,
) -> dict[str, str]:
    """Build persisted provenance for the current winner-selection semantics."""
    resolved = resolve_pipeline_tags(run.data.tags, run_id=run.info.run_id)
    selection_mode = winner_selection_mode_for_method(
        robustness_method=str(resolved["robustness_method"]),
        improvement_selection_mode=require_improvement_selection_mode(
            args,
            context="winner selection provenance args",
        ),
        run_id=run.info.run_id,
    )
    metric_name = selection_metric_key_for_kind(
        pipeline_kind=str(resolved["pipeline_kind"]),
        robustness_method=str(resolved["robustness_method"]),
        test_metric=test_metric,
        improvement_selection_mode=selection_mode,
        run_id=run.info.run_id,
    )
    provenance: dict[str, Any] = {
        "winner_selection_mode": selection_mode,
        "winner_selection_metric_name": metric_name,
    }
    if selection_mode != "clean":
        selection_context = require_selection_perturbation_context_from_args(
            args,
            context="args",
        )
        provenance.update(
            {
                "winner_selection_metric_semantics": selection_context[
                    "selection_metric_semantics"
                ],
                "winner_selection_perturbation_channel_fraction_max": selection_context[
                    "selection_perturbation_channel_fraction_max"
                ],
                "winner_selection_perturbation_scenarios_signature": selection_context[
                    "selection_perturbation_scenarios_signature"
                ],
            }
        )
    return build_winner_selection_provenance_tag_payload(
        provenance,
        context_name=f"Run {run.info.run_id} winner selection provenance",
    )


def _raise_fixed_channel_fraction_canonical_not_ready(run, *, args) -> None:
    run_id = run.info.run_id
    params = run.data.params
    tags = run.data.tags
    if params is None:
        raise ValueError(
            f"Run {run_id} is missing params required for canonical fixed-channel-fraction "
            "precondition checks."
        )
    if tags is None:
        raise ValueError(
            f"Run {run_id} is missing tags required for canonical fixed-channel-fraction "
            "precondition checks."
        )
    diagnostics: dict[str, Any] = {
        "tested_param": params.get("tested"),
        "robustness_results_complete": tags.get(ROBUSTNESS_RESULTS_COMPLETE_TAG),
    }
    try:
        expected_eval_data_seed = resolve_effective_eval_data_seed(
            require_namespace_value(args, key="eval_data_seed"),
            canonical_seed_data=tags.get("seed_data"),
            eval_key="args.eval_data_seed",
            canonical_key="seed_data tag",
        )
        expected_eval_context = require_degradation_eval_context_from_args(
            args,
            eval_data_seed=expected_eval_data_seed,
            context="args",
        )
        diagnostics["expected_eval_context"] = expected_eval_context
        diagnostics["logged_eval_context"] = require_degradation_eval_context_tags(
            tags,
            run_id=run_id,
        )
        diagnostics["expected_perturbation_scenario_params_signature"] = (
            build_perturbation_scenario_params_signature(
                require_namespace_value(args, key="perturbation_scenarios")
            )
        )
        diagnostics["logged_perturbation_scenario_params_signature"] = (
            require_nonempty_tag_value(
                tags,
                key="perturbation_scenario_params_signature",
                run_id=run_id,
            )
        )
        expected_bootstrap_ci_context = require_shared_anchor_bootstrap_ci_context_from_args(
            args,
            eval_data_seed=expected_eval_data_seed,
            test_metric=expected_eval_context["test_metric"],
            context="args",
        )
        diagnostics["expected_bootstrap_ci_context"] = expected_bootstrap_ci_context
        diagnostics["logged_bootstrap_ci_context"] = (
            require_shared_anchor_bootstrap_ci_context_tags(
                tags,
                run_id=run_id,
                require_seed=True,
            )
        )
    except Exception as exc:
        diagnostics["context_diagnostic_error"] = str(exc)
    raise ValueError(
        f"Run {run_id} is not canonically tested for the current fixed-channel-fraction "
        f"context. Diagnostics: {diagnostics}. Run canonical testing without "
        "--fixed-channel-fraction before running fixed-channel-fraction evaluation."
    )


def _collect_degradation_forecast_samples(
    *,
    model,
    dm: TSDataModule,
    sample_rows: pd.DataFrame,
    test_metric: str,
    eval_data_seed: int,
    runtime_device: torch.device,
    runtime_precision,
) -> list[dict[str, Any]]:
    require_dataframe_columns(
        sample_rows,
        {
            "sample_id",
            "source_sample_idx",
            "pert_idx",
            "scenario",
            "severity",
        },
        context="forecast-sample rows",
    )
    if sample_rows.empty:
        return []

    working = sample_rows.copy()
    for column in ("sample_id", "source_sample_idx", "pert_idx"):
        working[column] = require_integer_series(
            working,
            column,
            context="forecast-sample rows",
            sample_cols=("sample_id", "source_sample_idx", "pert_idx", "scenario"),
            min_value=0,
        )
    working["scenario"] = working["scenario"].astype(str).str.strip()
    working["severity"] = pd.to_numeric(working["severity"], errors="raise").astype(float)
    if "sample_score" in working.columns:
        working["sample_score"] = pd.to_numeric(
            working["sample_score"],
            errors="raise",
        ).astype(float)

    perturbed_dataset, base_dataset, scenario_names, perturbations = (
        _require_degradation_runtime(
            dm,
            context_name="forecast-sample rendering",
        )
    )

    input_feature_names = parse_optional_name_tuple(
        getattr(dm, "input_feature_names", None),
        key="datamodule.input_feature_names",
    )
    if input_feature_names is None:
        raise ValueError(
            "forecast-sample rendering requires datamodule.input_feature_names."
        )
    target_feature_names = parse_optional_name_tuple(
        getattr(dm, "target_feature_names", None),
        key="datamodule.target_feature_names",
    )
    if target_feature_names is None:
        raise ValueError(
            "forecast-sample rendering requires datamodule.target_feature_names."
        )
    metric_fn = resolve_stateless_loss(test_metric)
    batch_size = parse_required_positive_int(
        getattr(dm, "batch_size", None),
        key="datamodule.batch_size",
    )
    sorted_rows = working.sort_values(
        ["sample_id", "pert_idx"],
        kind="mergesort",
    ).reset_index(drop=True)
    row_order: list[tuple[int, int]] = []
    selected_rows_by_key: dict[tuple[int, int], Any] = {}
    selected_perturbations_by_sample_id: dict[int, set[int]] = defaultdict(set)
    for row in sorted_rows.itertuples(index=False):
        key = (int(row.sample_id), int(row.pert_idx))
        if key in selected_rows_by_key:
            raise ValueError(
                "forecast-sample rows contain duplicate "
                f"(sample_id, pert_idx)={key}."
            )
        if int(row.pert_idx) < 0 or int(row.pert_idx) >= len(perturbations):
            raise ValueError(
                f"forecast-sample pert_idx={int(row.pert_idx)} is outside the configured "
                f"scenario range [0, {len(perturbations)})."
            )
        row_order.append(key)
        selected_rows_by_key[key] = row
        selected_perturbations_by_sample_id[int(row.sample_id)].add(int(row.pert_idx))

    pending_keys = set(row_order)
    records_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    clean_loader = DataLoader(
        base_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    sample_offset = 0
    resolved_runtime_device = (
        runtime_device
        if isinstance(runtime_device, torch.device)
        else torch.device(str(runtime_device))
    )
    resolved_runtime_precision = parse_runtime_precision(
        runtime_precision,
        device_type=resolved_runtime_device.type,
        key="precision",
        context="Forecast-sample rendering",
    )
    with _configure_eval_matmul_precision_for_device(resolved_runtime_device):
        model.eval()
        with _manual_eval_runtime_context(
            model,
            runtime_device=resolved_runtime_device,
            runtime_precision=resolved_runtime_precision,
        ):
            with torch.no_grad():
                for x_cpu, y_cpu in clean_loader:
                    batch_count = int(x_cpu.size(0))
                    batch_sample_ids = _build_degradation_batch_sample_ids(
                        sample_offset=sample_offset,
                        batch_size=batch_count,
                    )
                    if batch_sample_ids and batch_sample_ids[-1] >= len(base_dataset):
                        raise ValueError(
                            "forecast-sample replay exceeded the base test dataset "
                            f"range [0, {len(base_dataset)})."
                        )
                    source_sample_idx = _require_degradation_source_sample_idx(
                        base_dataset,
                        sample_offset=sample_offset,
                        batch_size=batch_count,
                        context_name="forecast-sample replay",
                    )
                    requested_row_indices = [
                        row_idx
                        for row_idx, sample_id in enumerate(batch_sample_ids)
                        if int(sample_id) in selected_perturbations_by_sample_id
                    ]
                    if not requested_row_indices:
                        sample_offset += batch_count
                        continue

                    pred_clean_batch = _predict_with_bound_noise_sample_ids(
                        model,
                        _move_tensor_to_manual_eval_runtime(
                            x_cpu,
                            runtime_device=resolved_runtime_device,
                            runtime_precision=resolved_runtime_precision,
                        ),
                        batch_sample_ids,
                        context_key="degradation:clean",
                    )
                    err_clean_batch = _score_degradation_predictions(
                        metric_fn,
                        pred_clean_batch,
                        _move_tensor_to_manual_eval_runtime(
                            y_cpu,
                            runtime_device=resolved_runtime_device,
                            runtime_precision=resolved_runtime_precision,
                        ),
                        batch_size=batch_count,
                        context_name="forecast-sample clean replay",
                    )
                    pred_clean_batch_cpu = _prepare_tensor_for_host_export(
                        pred_clean_batch
                    )
                    for row_idx in requested_row_indices:
                        sample_id = int(batch_sample_ids[row_idx])
                        if sample_id < 0 or sample_id >= len(base_dataset):
                            raise ValueError(
                                f"forecast-sample sample_id={sample_id} is outside the base test "
                                f"dataset range [0, {len(base_dataset)})."
                            )
                        expected_source_sample_idx = int(source_sample_idx[row_idx])
                        for pert_idx in selected_perturbations_by_sample_id[sample_id]:
                            row = selected_rows_by_key[(sample_id, int(pert_idx))]
                            if expected_source_sample_idx != int(row.source_sample_idx):
                                raise ValueError(
                                    "forecast-sample source_sample_idx does not match the "
                                    f"reconstructed sample window for sample_id={sample_id}: "
                                    f"expected {expected_source_sample_idx}, got "
                                    f"{int(row.source_sample_idx)}."
                                )

                    requested_scenario_indices = sorted(
                        {
                            int(pert_idx)
                            for row_idx in requested_row_indices
                            for pert_idx in selected_perturbations_by_sample_id[
                                int(batch_sample_ids[row_idx])
                            ]
                        }
                    )
                    for scenario_idx in requested_scenario_indices:
                        perturbation = perturbations[scenario_idx]
                        x_pert_cpu, y_pert_cpu, severities, affected_channels_by_row = (
                            _build_degradation_scenario_batch(
                                perturbation=perturbation,
                                x_cpu=x_cpu,
                                y_cpu=y_cpu,
                                batch_sample_ids=batch_sample_ids,
                                eval_data_seed=eval_data_seed,
                                scenario_idx=scenario_idx,
                                cont_idx=perturbed_dataset.cont_idx,
                                disc_idx=perturbed_dataset.disc_idx,
                                context_name="Canonical forecast-sample rendering",
                                capture_affected_channels=True,
                            )
                        )
                        pred_pert_batch = _predict_with_bound_noise_sample_ids(
                            model,
                            _move_tensor_to_manual_eval_runtime(
                                x_pert_cpu,
                                runtime_device=resolved_runtime_device,
                                runtime_precision=resolved_runtime_precision,
                            ),
                            batch_sample_ids,
                            context_key=f"degradation:scenario:{scenario_idx}",
                        )
                        err_pert_batch = _score_degradation_predictions(
                            metric_fn,
                            pred_pert_batch,
                            _move_tensor_to_manual_eval_runtime(
                                y_pert_cpu,
                                runtime_device=resolved_runtime_device,
                                runtime_precision=resolved_runtime_precision,
                            ),
                            batch_size=batch_count,
                            context_name="forecast-sample replay",
                        )
                        pred_pert_batch_cpu = _prepare_tensor_for_host_export(pred_pert_batch)

                        for row_idx in requested_row_indices:
                            sample_id = int(batch_sample_ids[row_idx])
                            key = (sample_id, int(scenario_idx))
                            if key not in pending_keys:
                                continue
                            row = selected_rows_by_key[key]
                            if int(scenario_idx) < 0 or int(scenario_idx) >= len(perturbations):
                                raise ValueError(
                                    f"forecast-sample pert_idx={scenario_idx} is outside the configured "
                                    f"scenario range [0, {len(perturbations)})."
                                )
                            expected_scenario = str(scenario_names[scenario_idx])
                            if str(row.scenario) != expected_scenario:
                                raise ValueError(
                                    "forecast-sample scenario name does not match pert_idx: "
                                    f"pert_idx={scenario_idx} expected '{expected_scenario}' but got "
                                    f"'{row.scenario}'."
                                )
                            severity_value = float(severities[row_idx])
                            if not math.isclose(
                                severity_value,
                                float(row.severity),
                                rel_tol=0.0,
                                abs_tol=1e-9,
                            ):
                                raise ValueError(
                                    "forecast-sample severity does not match the canonical rendering seed "
                                    f"for sample_id={sample_id}, pert_idx={scenario_idx}: expected "
                                    f"{float(row.severity):.12f}, got {severity_value:.12f}."
                                )
                            observed_sample_score = float(err_pert_batch[row_idx])
                            if "sample_score" in working.columns:
                                expected_sample_score = float(row.sample_score)
                                if not math.isclose(
                                    observed_sample_score,
                                    expected_sample_score,
                                    rel_tol=1e-2,
                                    abs_tol=1e-9,
                                ):
                                    raise ValueError(
                                        "forecast-sample sample_score does not match the "
                                        "canonical rendering prediction score for "
                                        f"sample_id={sample_id}, pert_idx={scenario_idx}: "
                                        f"expected {expected_sample_score:.12f}, got "
                                        f"{observed_sample_score:.12f}."
                                    )
                            affected_feature_names: list[str] = []
                            affected_channels = affected_channels_by_row[row_idx]
                            if affected_channels is not None:
                                for idx in affected_channels:
                                    idx_int = int(idx)
                                    if 0 <= idx_int < len(input_feature_names):
                                        affected_feature_names.append(input_feature_names[idx_int])
                            records_by_key[key] = {
                                "sample_id": sample_id,
                                "source_sample_idx": int(source_sample_idx[row_idx]),
                                "pert_idx": int(scenario_idx),
                                "scenario": expected_scenario,
                                "severity": severity_value,
                                "sample_score": observed_sample_score,
                                "clean_sample_score": float(err_clean_batch[row_idx]),
                                "perturbed_sample_score": observed_sample_score,
                                "clean_input": dm.destandardize_inputs(
                                    x_cpu[row_idx].unsqueeze(0)
                                ).squeeze(0).cpu().numpy(),
                                "perturbed_input": dm.destandardize_inputs(
                                    x_pert_cpu[row_idx].unsqueeze(0)
                                ).squeeze(0).cpu().numpy(),
                                "target": dm.destandardize_targets(
                                    y_cpu[row_idx].unsqueeze(0)
                                ).squeeze(0).cpu().numpy(),
                                "prediction_clean": dm.destandardize_targets(
                                    pred_clean_batch_cpu[row_idx].unsqueeze(0)
                                ).squeeze(0).cpu().numpy(),
                                "prediction_perturbed": dm.destandardize_targets(
                                    pred_pert_batch_cpu[row_idx].unsqueeze(0)
                                ).squeeze(0).cpu().numpy(),
                                "input_feature_names": tuple(input_feature_names),
                                "target_feature_names": tuple(target_feature_names),
                                "affected_feature_names": tuple(affected_feature_names),
                            }
                            pending_keys.remove(key)
                    sample_offset += batch_count
                    if not pending_keys:
                        break
        if pending_keys:
            missing = sorted(pending_keys)[:8]
            raise ValueError(
                "forecast-sample replay could not render all requested "
                f"sample rows. Missing examples: {missing}."
            )
    return [records_by_key[key] for key in row_order]


def _make_eval_trainer(
    logger: MLFlowLogger,
    default_root_dir: str,
    args,
    *,
    max_epochs: int,
    enable_checkpointing: bool,
) -> pl.Trainer:
    return pl.Trainer(
        enable_checkpointing=enable_checkpointing,
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=0,
        logger=logger,
        default_root_dir=default_root_dir,
    )


def post_model_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _teardown_model_after_eval(model) -> None:
    # Lightning's ``trainer`` property raises until a module is attached, so
    # cleanup must not probe it via ``hasattr``.
    try:
        trainer = getattr(model, "trainer")
    except AttributeError:
        trainer = None
    except RuntimeError as exc:
        if "is not attached to a `Trainer`" not in str(exc):
            raise
        trainer = None
    if trainer is not None:
        model.trainer = None
    if hasattr(model, "_logger"):
        model._logger = None
    model.to("cpu")
    for attr_name in ("_test_step_outputs",):
        if not hasattr(model, attr_name):
            continue
        value = getattr(model, attr_name)
        if isinstance(value, list) or isinstance(value, dict):
            value.clear()
        elif value is not None and hasattr(value, "clear"):
            value.clear()
    artifact_tmpdir = getattr(model, "_artifact_tempdir_handle", None)
    if artifact_tmpdir is not None:
        artifact_tmpdir.cleanup()
        setattr(model, "_artifact_tempdir_handle", None)
    post_model_cleanup()


def _evaluate_run(
    best_run,
    model,
    default_root_dir,
    dataset_name,
    args,
    dm,
    eval_data_seed: int,
):
    try:
        seeds = require_seed_tags(best_run)
        eval_seed = seeds["seed_eval"]
        bootstrap_ci_context = _resolve_bootstrap_ci_context(
            args,
            eval_data_seed=eval_data_seed,
        )
        _prime_model_for_degradation_evaluation(
            model,
            args,
            dm,
            eval_seed=eval_seed,
        )
        logger = _make_eval_logger(best_run.info.run_id, dataset_name, args)
        trainer = _make_eval_trainer(
            logger,
            default_root_dir,
            args,
            max_epochs=args.max_epochs,
            enable_checkpointing=True if args.save_checkpoint else False,
        )
        # Disable hyperparameter logging to avoid MLflow conflicts with improvement runs
        model._log_hyperparams = False

        _log_common_eval_params(
            logger,
            args,
            dm,
            bootstrap_ci_context=bootstrap_ci_context,
            eval_data_seed=eval_data_seed,
            seed_master=seeds.get("seed_master"),
            seed_data=seeds.get("seed_data"),
            seed_eval=eval_seed,
        )

        root_device = getattr(getattr(trainer, "strategy", None), "root_device", None)
        with _configure_eval_matmul_precision_for_device(root_device):
            with _suppress_lightning_worker_warning():
                trainer.validate(model=model, datamodule=dm, verbose=False)
        _prepare_model_for_evaluation(
            model,
            args,
            dm,
            eval_seed=eval_seed,
        )
        _run_degradation_evaluation(
            model=model,
            trainer=trainer,
            logger=logger,
            dm=dm,
            args=args,
            eval_data_seed=eval_data_seed,
            bootstrap_ci_context=bootstrap_ci_context,
        )
    finally:
        _teardown_model_after_eval(model)


def validate_run(best_run, model, default_root_dir, dataset_name, args, dm):
    seeds = require_seed_tags(best_run)
    eval_seed = seeds["seed_eval"]
    _prime_model_for_degradation_evaluation(
        model,
        args,
        dm,
        eval_seed=eval_seed,
    )
    logger = _make_eval_logger(best_run.info.run_id, dataset_name, args)
    trainer = _make_eval_trainer(
        logger,
        default_root_dir,
        args,
        max_epochs=1,
        enable_checkpointing=False,
    )
    model._log_hyperparams = False

    try:
        _log_common_eval_params(
            logger,
            args,
            dm,
            seed_master=seeds.get("seed_master"),
            seed_data=seeds.get("seed_data"),
            seed_eval=eval_seed,
            log_degradation_context=False,
        )

        root_device = getattr(getattr(trainer, "strategy", None), "root_device", None)
        with _configure_eval_matmul_precision_for_device(root_device):
            with _suppress_lightning_worker_warning():
                trainer.validate(model=model, datamodule=dm, verbose=False)
    finally:
        _teardown_model_after_eval(model)


def _resolve_dataset_testing_coverage_scope(
    dataset_spec,
    args,
    *,
    recipe_specs_for_scope: Optional[list[PipelineSpec]] = None,
) -> DatasetTestingCoverageScope:
    full_coverage = require_namespace_bool(
        args,
        key="full_coverage",
        context="Testing args",
    )
    data_config_signature = compute_data_config_signature(
        dataset_spec=dataset_spec,
        args=args,
    )
    baseline_hparam_specs_by_arch = load_hparams()
    if baseline_hparam_specs_by_arch is None:
        raise ValueError(
            "Baseline hyperparameter configuration is missing. "
            "Expected load_hparams() to return a mapping."
        )
    if not isinstance(baseline_hparam_specs_by_arch, Mapping):
        raise ValueError(
            "Baseline hyperparameter configuration must be a mapping."
        )
    if recipe_specs_for_scope is None:
        recipe_specs_for_scope = load_benchmark_recipe_specs_for_scope(load_defaults())
    recipe_spec_by_method: dict[str, PipelineSpec] = {}
    for recipe_spec in recipe_specs_for_scope:
        method_raw = recipe_spec.pipeline_method
        if method_raw is None or not str(method_raw).strip():
            raise ValueError("Recipe scope contains pipeline with missing pipeline_method.")
        method = str(method_raw).strip()
        kind_raw = recipe_spec.pipeline_kind
        if kind_raw is None or not str(kind_raw).strip():
            raise ValueError(
                f"Recipe scope entry '{method}' is missing required pipeline_kind."
            )
        if method in recipe_spec_by_method:
            raise ValueError(
                f"Duplicate configured pipeline_method '{method}' in recipe scope."
            )
        recipe_spec_by_method[method] = recipe_spec
    requested_architectures_list = resolve_requested_architectures(args)
    selected_methods_list = resolve_requested_methods(
        args,
        configured_methods=tuple(recipe_spec_by_method.keys()),
    )
    selected_methods = set(selected_methods_list)
    method_architecture_scope = resolve_benchmark_method_architecture_scope(
        methods=selected_methods_list,
        architectures=tuple(requested_architectures_list),
        explicit_architectures=has_explicit_architecture_scope(args),
        context="run_testing.py",
    )
    selected_methods = set(method_architecture_scope)
    requested_architectures = frozenset(
        architecture
        for architectures in method_architecture_scope.values()
        for architecture in architectures
    )
    lineage_methods = expand_testing_method_scope_for_wrap_dependencies(
        args,
        requested_methods=selected_methods,
        recipe_spec_by_method=recipe_spec_by_method,
    )
    if selected_methods - {"baseline"}:
        lineage_methods.add("baseline")

    tracking_uri = build_mlflow_tracking_uri(args.logdir)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    dataset_name = dataset_spec.key
    experiment = client.get_experiment_by_name(
        f"{args.mlflow_experiment_prefix}-{dataset_name}"
    )
    if experiment is None:
        raise ValueError(
            f"No experiments with prefix {args.mlflow_experiment_prefix} on {dataset_name} found."
        )
    experiment_tags = experiment.tags
    if experiment_tags is None:
        raise ValueError(
            f"Experiment '{experiment.name}' is missing tags."
        )
    experiment_signature = experiment_tags.get("data_config_signature")
    if not experiment_signature:
        raise ValueError(
            f"Experiment '{experiment.name}' is missing data_config_signature tag. "
            "Train the benchmark with the current benchmark workflow before evaluating."
        )
    if experiment_signature != data_config_signature:
        raise ValueError(
            "Data config signature mismatch between experiment and current args. "
            f"experiment='{experiment_signature}', current='{data_config_signature}'. "
            "Ensure data_split_seed and data parameters match the training runs."
        )

    all_runs = search_runs_all(
        client,
        [experiment.experiment_id],
        max_results=1000,  # with pagination
    )
    n_runs = len(all_runs)

    parent_runs = [
        run
        for run in all_runs
        if getattr(run.info, "lifecycle_stage", "active") == "active"
        and run.info.status == "FINISHED"
        and not run.data.tags.get("mlflow.parentRunId")
    ]
    if not parent_runs:
        raise ValueError(
            f"No finished parent runs found in experiment '{experiment.name}'."
        )

    mismatched_data_sig = []
    for run in parent_runs:
        run_tags = run.data.tags
        if run_tags is None:
            raise ValueError(
                f"Run {run.info.run_id} is missing tags in parent run pool."
            )
        if run_tags.get("data_config_signature") != experiment_signature:
            mismatched_data_sig.append(run.info.run_id)
    if mismatched_data_sig:
        preview = ", ".join(mismatched_data_sig[:5])
        raise ValueError(
            "Runs with mismatched or missing data_config_signature detected. "
            f"Example run IDs: {preview}."
        )

    seed_data_values = []
    seed_policy_values = set()
    for run in parent_runs:
        seeds = require_seed_tags(run)
        seed_data_values.append(seeds["seed_data"])
        seed_policy_values.add(seeds["seed_policy"])

    unique_seed_data = sorted(set(seed_data_values))
    if len(unique_seed_data) != 1:
        preview = ", ".join(str(value) for value in unique_seed_data[:5])
        raise ValueError(
            "Multiple seed_data values detected in the experiment. "
            f"Found: {preview}. Split experiments so each evaluation pool has a single data seed."
        )
    if len(seed_policy_values) != 1:
        preview = ", ".join(sorted(str(value) for value in seed_policy_values))
        raise ValueError(
            "Multiple seed_policy values detected in the experiment. "
            f"Found: {preview}. Do not mix seed policies in a single evaluation."
        )
    data_seed = unique_seed_data[0]
    eval_data_seed = resolve_effective_eval_data_seed(
        require_namespace_value(args, key="eval_data_seed"),
        canonical_seed_data=data_seed,
        eval_key="args.eval_data_seed",
        canonical_key="seed_data",
    )
    if eval_data_seed != data_seed:
        print(
            f"Using eval_data_seed override {eval_data_seed} "
            f"(canonical seed_data={data_seed})."
        )
    else:
        print(f"Using canonical eval_data_seed={eval_data_seed}.")

    tuning_scope_cache: dict[tuple[str, str, str], Any] = {}

    def _expected_tuning_scope(
        *,
        arch: str,
        pipeline_method: str,
        pipeline_kind: str,
    ):
        cache_key = (str(arch), str(pipeline_method), str(pipeline_kind))
        cached = tuning_scope_cache.get(cache_key)
        if cached is not None:
            return cached
        spec = recipe_spec_by_method.get(str(pipeline_method))
        if spec is None:
            raise ValueError(
                f"Pipeline method '{pipeline_method}' is not configured for strict coverage scope."
            )
        if str(spec.pipeline_kind).strip() != str(pipeline_kind).strip():
            raise ValueError(
                f"Pipeline method '{pipeline_method}' resolved to kind '{spec.pipeline_kind}', "
                f"but run selection requires kind '{pipeline_kind}'."
            )
        scope_runner = PipelineRunner(spec, args)
        scope = scope_runner.expected_tuning_scope(
            client=client,
            experiment_id=experiment.experiment_id,
            dataset_spec=dataset_spec,
            architecture=arch,
            data_config_signature=data_config_signature,
        )
        tuning_scope_cache[cache_key] = scope
        return scope

    def _scope_exclusion_reason(
        run,
        *,
        arch: str,
        pipeline_method: str,
        pipeline_kind: str,
    ) -> Optional[str]:
        return shared_scope_exclusion_reason(
            run,
            arch=arch,
            pipeline_method=pipeline_method,
            pipeline_kind=pipeline_kind,
            recipe_spec_by_method=recipe_spec_by_method,
            expected_tuning_scope=_expected_tuning_scope,
        )

    runs_by_variant = defaultdict(list)
    resolved_by_run_id: dict[str, dict[str, Any]] = {}
    base_runs_by_key = defaultdict(list)
    scoped_train_run_ids_by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    skipped_incomplete = 0
    skipped_out_of_scope = 0

    for run in parent_runs:
        tags = run.data.tags
        if tags is None:
            raise ValueError(
                f"Run {run.info.run_id} is missing tags in parent run selection."
            )
        arch = tags.get("model_architecture")
        if not arch:
            raise ValueError(
                f"Run {run.info.run_id} is missing required model_architecture tag."
            )
        arch_token = str(arch)
        if arch_token not in requested_architectures:
            continue

        resolved = resolve_pipeline_tags(tags, run_id=run.info.run_id)
        pipeline_method_token = str(resolved["pipeline_method"])
        if pipeline_method_token not in lineage_methods:
            continue
        resolved_by_run_id[run.info.run_id] = resolved
        scope_reason = _scope_exclusion_reason(
            run,
            arch=arch_token,
            pipeline_method=pipeline_method_token,
            pipeline_kind=str(resolved["pipeline_kind"]),
        )
        if scope_reason is not None:
            skipped_out_of_scope += 1
            continue

        if (
            str(resolved["pipeline_kind"]) != "wrap"
            and run.data.metrics.get("best_val_loss") is None
        ):
            skipped_incomplete += 1
            continue

        if pipeline_method_token in selected_methods:
            runs_by_variant[
                (arch, resolved["pipeline_method"], resolved["pipeline_id"])
            ].append(run)
        stage = require_stage_tag(tags, run_id=run.info.run_id)
        if stage == "train" and str(resolved["pipeline_kind"]) in ("train", "finetune"):
            base_runs_by_key[(arch, str(resolved["pipeline_method"]))].append(run)
            scoped_train_run_ids_by_key[
                (str(arch), str(resolved["pipeline_method"]))
            ].add(run.info.run_id)

    if skipped_incomplete:
        print(
            f"Skipping {skipped_incomplete} run(s) missing best_val_loss (training incomplete)."
        )
    if skipped_out_of_scope:
        print(
            f"Skipping {skipped_out_of_scope} run(s) outside the active tuning scope "
            "(current seed + max_hp_trials_per_model + recipe grid)."
        )

    sorted_base_runs_by_key, current_base_runs_by_key = build_base_index(
        base_runs_by_key
    )
    if sorted_base_runs_by_key:
        print(
            f"Resolved current baselines for {len(current_base_runs_by_key)} "
            "architecture/method pairs."
        )

    stale_run_ids = set()
    stale_reasons: dict[str, str] = {}
    hparams_artifact_cache: dict[str, dict[str, Any]] = {}

    def _mark_stale(run_id: str, reason: str) -> None:
        stale_run_ids.add(run_id)
        stale_reasons[run_id] = reason

    filtered_runs_by_variant = defaultdict(list)
    for (arch, pipeline_method, pipeline_id), runs in runs_by_variant.items():
        for run in runs:
            reason = classify_lineage_run(
                run,
                arch,
                current_base_runs_by_key=current_base_runs_by_key,
                sorted_base_runs_by_key=sorted_base_runs_by_key,
                baseline_hparam_specs_by_arch=baseline_hparam_specs_by_arch,
                scoped_train_run_ids_by_key=scoped_train_run_ids_by_key,
                artifact_client=client,
                hparams_artifact_cache=hparams_artifact_cache,
            )
            if reason:
                _mark_stale(run.info.run_id, reason)
                continue
            filtered_runs_by_variant[(arch, pipeline_method, pipeline_id)].append(run)

    runs_by_variant = {
        key: value for key, value in filtered_runs_by_variant.items() if value
    }

    coverage_policy = audit_and_apply_testing_coverage_policy(
        args=args,
        full_coverage=full_coverage,
        requested_architectures=requested_architectures,
        requested_methods=selected_methods,
        recipe_spec_by_method=recipe_spec_by_method,
        runs_by_variant=runs_by_variant,
        resolved_by_run_id=resolved_by_run_id,
        expected_tuning_scope=_expected_tuning_scope,
    )
    runs_by_variant = coverage_policy["runs_by_variant"]
    for run_id, reason in coverage_policy["dropped_run_reasons"].items():
        if run_id in stale_run_ids:
            continue
        _mark_stale(run_id, reason)

    if "coverage_fractions" not in coverage_policy:
        raise ValueError(
            "Testing coverage policy did not return required coverage_fractions."
        )
    dataset_coverage_fractions = coverage_policy["coverage_fractions"]

    selection_base_runs_by_key = defaultdict(list)
    for (arch, pipeline_method, _), runs in runs_by_variant.items():
        for run in runs:
            run_tags = run.data.tags
            if run_tags is None:
                raise ValueError(
                    f"Run {run.info.run_id} is missing tags while rebuilding selection scope."
                )
            stage = require_stage_tag(run_tags, run_id=run.info.run_id)
            resolved = resolved_by_run_id.get(run.info.run_id)
            if resolved is None:
                raise ValueError(
                    f"Run {run.info.run_id} is missing resolved pipeline tags "
                    "while rebuilding selection scope."
                )
            if stage != "train" or str(resolved["pipeline_kind"]) not in ("train", "finetune"):
                continue
            selection_base_runs_by_key[(str(arch), str(pipeline_method))].append(run)
    _, selection_current_base_runs_by_key = build_base_index(selection_base_runs_by_key)

    return DatasetTestingCoverageScope(
        client=client,
        dataset_name=dataset_name,
        experiment_id=experiment.experiment_id,
        data_config_signature=data_config_signature,
        data_seed=int(data_seed),
        n_runs=n_runs,
        parent_runs=tuple(parent_runs),
        all_runs=tuple(all_runs),
        resolved_by_run_id=MappingProxyType(dict(resolved_by_run_id)),
        runs_by_variant=MappingProxyType(
            {key: tuple(value) for key, value in runs_by_variant.items()}
        ),
        selection_current_base_runs_by_key=MappingProxyType(
            dict(selection_current_base_runs_by_key)
        ),
        requested_architectures=frozenset(requested_architectures),
        selected_methods=frozenset(selected_methods),
        stale_run_ids=frozenset(stale_run_ids),
        stale_reasons=MappingProxyType(dict(stale_reasons)),
        dataset_coverage_fractions=MappingProxyType(dict(dataset_coverage_fractions)),
        eval_data_seed=int(eval_data_seed),
    )


def _require_requested_family_selected_variants(
    *,
    selected_variants: set[tuple[str, str, str]],
    selected_methods: set[str] | frozenset[str],
    dataset_coverage_fractions: Mapping[tuple[str, str], tuple[int, int]],
) -> None:
    selected_method_tokens = {str(method) for method in selected_methods}
    selected_family_keys = {
        (str(arch), str(method))
        for arch, method, _ in selected_variants
    }
    missing_requested_families: list[tuple[str, str, int, int]] = []
    for (raw_arch, raw_method), fraction in sorted(dataset_coverage_fractions.items()):
        method = str(raw_method)
        if method not in selected_method_tokens:
            continue
        arch = str(raw_arch)
        seen, expected = fraction
        expected_count = int(expected)
        if expected_count <= 0:
            continue
        if (arch, method) not in selected_family_keys:
            missing_requested_families.append(
                (arch, method, int(seen), expected_count)
            )

    if not missing_requested_families:
        return

    preview = ", ".join(
        f"{method}/{arch} ({seen}/{expected} coverage)"
        for arch, method, seen, expected in missing_requested_families[:5]
    )
    remainder = len(missing_requested_families) - 5
    if remainder > 0:
        preview += f", ... (+{remainder} more)"
    raise CoverageMismatchError(
        "Coverage relaxation removed all candidate variants for explicitly "
        f"requested method family/families: {preview}. Re-run testing with "
        "matching completed runs or narrow the requested method/model scope."
    )


def test_on_dataset(
    dataset_spec,
    args,
    *,
    recipe_specs_for_scope: Optional[list[PipelineSpec]] = None,
):
    _configure_runtime_loggers_for_testing()
    args.full_coverage = require_namespace_bool(args, key="full_coverage")
    perturbation_channel_fraction_max = parse_perturbation_channel_fraction_max(
        require_namespace_value(
            args,
            key="perturbation_channel_fraction_max",
        ),
        key="perturbation_channel_fraction_max",
    )
    fixed_channel_fraction = parse_optional_unit_float(
        getattr(args, "fixed_channel_fraction", None),
        key="fixed_channel_fraction",
        max_value=perturbation_channel_fraction_max,
    )
    args.perturbation_channel_fraction_max = perturbation_channel_fraction_max
    args.fixed_channel_fraction = fixed_channel_fraction
    selection_args = args
    if fixed_channel_fraction is not None:
        selection_args = copy.copy(args)
        selection_args.rerun = False
    scope = _resolve_dataset_testing_coverage_scope(
        dataset_spec,
        args,
        recipe_specs_for_scope=recipe_specs_for_scope,
    )
    client = scope.client
    dataset_name = scope.dataset_name
    experiment_id = scope.experiment_id
    all_runs = scope.all_runs
    parent_runs = scope.parent_runs
    resolved_by_run_id = scope.resolved_by_run_id
    runs_by_variant = scope.runs_by_variant
    selection_current_base_runs_by_key = scope.selection_current_base_runs_by_key
    requested_architectures = scope.requested_architectures
    selected_methods = scope.selected_methods
    n_runs = scope.n_runs
    data_seed = scope.data_seed
    eval_data_seed = scope.eval_data_seed
    dataset_coverage_fractions = scope.dataset_coverage_fractions
    # Mutable copies — these are mutated by _mark_stale() closures below
    stale_run_ids = set(scope.stale_run_ids)
    stale_reasons = dict(scope.stale_reasons)

    dm = _build_testing_datamodule(
        dataset_spec=dataset_spec,
        args=args,
        canonical_data_seed=data_seed,
        eval_data_seed=eval_data_seed,
        val_seed=data_seed,
    )
    fixed_channel_fraction_dm_by_n: dict[int, TSDataModule] = {int(args.n_test_samples): dm}

    def _dm_for_effective_test_samples(n_test_samples: int) -> TSDataModule:
        parsed_n = parse_required_positive_int(
            n_test_samples,
            key="fixed_channel_fraction_n_test_samples",
        )
        if parsed_n not in fixed_channel_fraction_dm_by_n:
            fixed_channel_fraction_args = copy.copy(args)
            fixed_channel_fraction_args.n_test_samples = parsed_n
            fixed_channel_fraction_dm_by_n[parsed_n] = _build_testing_datamodule(
                dataset_spec=dataset_spec,
                args=fixed_channel_fraction_args,
                canonical_data_seed=data_seed,
                eval_data_seed=eval_data_seed,
                val_seed=data_seed,
            )
        return fixed_channel_fraction_dm_by_n[parsed_n]

    selection_dm_holder: dict[str, TSDataModule] = {"selection": dm}

    def _best_run(runs, *, score_fn, include_end_time: bool = True):
        best_key = None
        best_run = None
        for run in runs:
            score = score_fn(run)
            key = rank_key_for_run(
                run,
                metric_value=score,
                include_end_time=include_end_time,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_run = run
        return best_run

    def _mark_stale(run_id: str, reason: str) -> None:
        stale_run_ids.add(run_id)
        stale_reasons[run_id] = reason

    def _score_for_kind(candidate_run, pipeline_kind, arch, pipeline_method, pipeline_id):
        return ensure_selection_metric_for_run(
            candidate_run,
            arch=arch,
            pipeline_method=pipeline_method,
            pipeline_id=pipeline_id,
            args=selection_args,
            client=client,
            dataset_name=dataset_name,
            dm_holder=selection_dm_holder,
            dm_factory=lambda: dm,
            load_model_fn=_load_candidate_model,
        )

    if stale_run_ids:
        print(f"Excluding {len(stale_run_ids)} stale run(s) from selection.")

    current_run_ids = set()
    current_baseline_ids = {
        run.info.run_id
        for (arch, pipeline_method), run in selection_current_base_runs_by_key.items()
        if pipeline_method == "baseline" and run is not None
    }
    for (arch, pipeline_method, _), runs in runs_by_variant.items():
        for run in runs:
            if run.info.run_id in stale_run_ids:
                continue
            if str(pipeline_method) == "baseline":
                if run.info.run_id in current_baseline_ids:
                    current_run_ids.add(run.info.run_id)
                continue
            current_run_ids.add(run.info.run_id)
    all_runs_by_id = {run.info.run_id: run for run in all_runs}
    all_run_ids = set(all_runs_by_id.keys())
    for run_id in sorted(current_run_ids):
        if run_id not in all_run_ids:
            continue
        run = all_runs_by_id[run_id]
        run_tags = run.data.tags
        if run_tags is None:
            raise ValueError(
                f"Run {run_id} is missing tags while tagging current backbone runs."
            )
        if require_stage_tag(run_tags, run_id=run_id) == "uq":
            continue
        if run_tags.get("backbone_current") != "true":
            client.set_tag(run_id, "backbone_current", "true")

    if stale_run_ids:
        missing = 0
        for run_id in sorted(stale_run_ids):
            if run_id not in all_run_ids:
                missing += 1
                continue
            run = all_runs_by_id[run_id]
            run_tags = run.data.tags
            if run_tags is None:
                raise ValueError(
                    f"Run {run_id} is missing tags while tagging stale runs."
                )
            if require_stage_tag(run_tags, run_id=run_id) == "uq":
                continue
            if run_id not in stale_reasons:
                raise ValueError(f"Stale run {run_id} is missing a stale reason.")
            reason = stale_reasons[run_id]
            if run_tags.get("backbone_current") != "false":
                client.set_tag(run_id, "backbone_current", "false")
            if run_tags.get("backbone_current_reason") != reason:
                client.set_tag(run_id, "backbone_current_reason", reason)
            if run_tags.get("best_model") != "false":
                client.set_tag(run_id, "best_model", "false")
        if missing:
            print(f"Skipped tagging {missing} stale run(s) missing from the experiment.")

    # Clear backbone_current on non-current baselines so that tags set by
    # a previous testing pass don't persist across re-runs.
    for (arch, pipeline_method, _), runs in runs_by_variant.items():
        if str(pipeline_method) != "baseline":
            continue
        for run in runs:
            if run.info.run_id not in current_baseline_ids and run.info.run_id in all_run_ids:
                if run.data.tags.get("backbone_current") != "false":
                    client.set_tag(run.info.run_id, "backbone_current", "false")

    # Summarize the filtered parent-run pool that remains eligible for
    # selection/evaluation after scope, lineage, and coverage auditing.
    parent_run_ids = {run.info.run_id for run in parent_runs}
    in_scope_run_ids = {
        run.info.run_id
        for runs in runs_by_variant.values()
        for run in runs
    }
    ignored_parent_run_count = len(parent_run_ids - in_scope_run_ids)
    unique_archs = set(arch for arch, _, _ in runs_by_variant.keys())
    unique_pipelines = set(pid for _, _, pid in runs_by_variant.keys())
    unique_families = set(fam for _, fam, _ in runs_by_variant.keys())
    print(
        f"Discovered {n_runs} total runs; active finished parent runs: "
        f"{len(parent_run_ids)}."
    )
    print(
        f"In-scope pool: {len(in_scope_run_ids)} parent runs across "
        f"{len(unique_archs)} architectures, {len(unique_pipelines)} pipeline "
        f"variants, and {len(unique_families)} families."
    )
    if ignored_parent_run_count:
        print(
            f"Ignoring {ignored_parent_run_count} active finished parent run(s) "
            "outside the active selection/evaluation scope."
        )

    # Select and test best run per pipeline_method (plus baseline).
    tested_run_ids = set()
    best_runs_by_variant = {}
    runs_by_family = defaultdict(list)

    def _resolved_run_tags(run) -> dict:
        cached = resolved_by_run_id.get(run.info.run_id)
        if cached is not None:
            return cached
        return resolve_pipeline_tags(run.data.tags, run_id=run.info.run_id)

    def _load_candidate_model(candidate_run):
        return load_model_with_loader(client, candidate_run, args, dm)

    def _select_best_run(runs, variant_kind, arch, pipeline_method, pipeline_id):
        return _best_run(
            runs,
            score_fn=lambda r: _score_for_kind(r, variant_kind, arch, pipeline_method, pipeline_id),
            include_end_time=True,
        )

    def _family_candidate_key(candidate_run, pipeline_kind, pipeline_id, arch, pipeline_method):
        score = _score_for_kind(candidate_run, pipeline_kind, arch, pipeline_method, pipeline_id)
        key = rank_key_for_row_values(
            selection_value=score,
            end_time=candidate_run.info.end_time,
            run_id=candidate_run.info.run_id,
        )
        return (*key, str(pipeline_id))

    for (arch, pipeline_method, pipeline_id), runs in runs_by_variant.items():
        if str(arch) not in requested_architectures:
            continue

        kinds = {_resolved_run_tags(r).get("pipeline_kind") for r in runs}
        if len(kinds) != 1:
            raise ValueError(
                f"Variant ({arch}, {pipeline_method}, {pipeline_id}) has inconsistent pipeline_kind values: "
                f"{sorted(str(k) for k in kinds)}."
            )
        (variant_kind,) = tuple(kinds)

        best_run = _select_best_run(runs, variant_kind, arch, pipeline_method, pipeline_id)

        best_runs_by_variant[(arch, pipeline_method, pipeline_id)] = (best_run, runs)
        if str(pipeline_method) in selected_methods:
            runs_by_family[(arch, pipeline_method)].append((pipeline_id, best_run))

    # Select winning pipeline_id per family (baseline always included).
    selected_variants = set()

    for (arch, pipeline_method), entries in runs_by_family.items():
        if pipeline_method == "baseline":
            # Always keep baseline for each architecture.
            selected_variants.add((arch, pipeline_method, "baseline"))
            continue
        if not entries:
            continue

        kinds_by_pipeline = {
            str(pipeline_id): _resolved_run_tags(candidate_run).get("pipeline_kind")
            for pipeline_id, candidate_run in entries
        }
        _require_single_pipeline_kind(
            kinds_by_pipeline,
            arch=arch,
            pipeline_method=pipeline_method,
        )
        # Pick the best pipeline_id within this family.
        best = None  # (score, -end_time, run_id, pipeline_id)
        for pipeline_id, candidate_run in entries:
            pipeline_kind = _resolved_run_tags(candidate_run).get("pipeline_kind")
            candidate = _family_candidate_key(
                candidate_run, pipeline_kind, pipeline_id, arch, pipeline_method
            )
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise ValueError(f"No candidate variants found for ({arch}, {pipeline_method}).")
        _, _, _, winner_pipeline_id = best
        selected_variants.add((arch, pipeline_method, winner_pipeline_id))

    if getattr(args, "method", None) is not None:
        _require_requested_family_selected_variants(
            selected_variants=selected_variants,
            selected_methods=selected_methods,
            dataset_coverage_fractions=dataset_coverage_fractions,
        )

    # Mark best_model tags for all candidate variants: only selected winners are best_model=true.
    # Written unconditionally because earlier loops (e.g. line 4554) may have
    # mutated server-side tags, making in-memory run objects stale.
    new_winner_ids: set[str] = set()
    for (arch, pipeline_method, pipeline_id), (best_run, runs) in best_runs_by_variant.items():
        is_selected = (arch, pipeline_method, pipeline_id) in selected_variants
        for run in runs:
            is_winner = is_selected and run.info.run_id == best_run.info.run_id
            desired = "true" if is_winner else "false"
            client.set_tag(run.info.run_id, "best_model", desired)
            client.set_tag(run.info.run_id, "backbone_current", desired)
            winner_selection_tag_payload: Mapping[str, str] | None = None
            if is_winner:
                winner_selection_tag_payload = (
                    _build_winner_selection_provenance_tag_payload_for_run(
                        run,
                        args=selection_args,
                        test_metric=args.test_metric,
                    )
                )
                new_winner_ids.add(run.info.run_id)
            _replace_winner_selection_provenance_tags(
                client,
                run,
                tag_payload=winner_selection_tag_payload,
            )

    # Clear stale best_model=true tags on all previous winners that are not
    # newly selected. The best_model tag is transient session state.
    # meta_analysis queries it globally, so no stale tags must persist.
    _clear_out_of_scope_best_model_tags(
        client,
        parent_runs=parent_runs,
        new_winner_ids=new_winner_ids,
        cleanup_architectures={str(arch) for arch, _, _ in best_runs_by_variant},
        cleanup_methods={str(method) for method in selected_methods},
    )

    # Validate no non-FINISHED active runs retain stale best_model=true
    # (e.g. from a previously interrupted testing session).
    for run in search_runs_all(
        client,
        [experiment_id],
        filter_string="tags.best_model = 'true'",
        run_view_type=ViewType.ACTIVE_ONLY,
    ):
        if run.data.tags.get("mlflow.parentRunId"):
            continue
        if run.info.status != "FINISHED":
            _raise_non_finished_winner(run)

    # Evaluate selected variants only
    sorted_variants = sorted(selected_variants)
    total_variants = len(sorted_variants)
    arch_list = list(dict.fromkeys(a for a, _, _ in sorted_variants))
    arch_num = {a: j for j, a in enumerate(arch_list, 1)}
    n_archs = len(arch_list)
    variants_per_arch = Counter(a for a, _, _ in sorted_variants)
    n_skipped = 0
    n_evaluated = 0
    _prev_arch = None
    _arch_i = 0

    for trial_i, (arch, pipeline_method, pipeline_id) in enumerate(sorted_variants, 1):
        if arch != _prev_arch:
            _prev_arch = arch
            _arch_i = 0
        _arch_i += 1

        best_run, runs = best_runs_by_variant.get((arch, pipeline_method, pipeline_id), (None, None))
        if best_run is None:
            raise ValueError(f"Selected variant ({arch}, {pipeline_method}, {pipeline_id}) has no resolved run.")
        best_run = client.get_run(best_run.info.run_id)

        if fixed_channel_fraction is not None:
            if not is_fully_tested(best_run, args=args, client=client):
                _raise_fixed_channel_fraction_canonical_not_ready(best_run, args=args)
            if not args.rerun and is_fixed_channel_fraction_complete(
                best_run,
                args=args,
                client=client,
                fixed_fraction=fixed_channel_fraction,
            ):
                n_skipped += 1
                continue

            run_eval_context = require_degradation_eval_context_tags(
                best_run.data.tags,
                run_id=best_run.info.run_id,
            )
            run_bootstrap_ci_context = require_shared_anchor_bootstrap_ci_context_tags(
                best_run.data.tags,
                run_id=best_run.info.run_id,
                require_seed=True,
            )
            run_params_signature = require_nonempty_tag_value(
                best_run.data.tags,
                key="perturbation_scenario_params_signature",
                run_id=best_run.info.run_id,
            )
            canonical_context_signature = build_canonical_degradation_context_signature(
                degradation_eval_context=run_eval_context,
                bootstrap_ci_context=run_bootstrap_ci_context,
                perturbation_scenario_params_signature=run_params_signature,
            )
            test_metric = str(run_eval_context["test_metric"])
            canonical_clean_df, _, _ = download_validated_degradation_artifact_bundle(
                client,
                run_id=best_run.info.run_id,
                test_metric=test_metric,
                eval_data_seed=int(run_eval_context["eval_data_seed"]),
                expected_idx_to_name=run_eval_context["perturbation_idx_name_map"],
                expected_n_test_samples=int(run_eval_context["n_test_samples"]),
                expected_clean_metric_value=best_run.data.metrics[f"{test_metric}_test"],
                context_name=(
                    f"Run {best_run.info.run_id} canonical degradation artifacts "
                    "for fixed-channel-fraction"
                ),
            )
            fixed_channel_fraction_dm = _dm_for_effective_test_samples(
                int(run_eval_context["n_test_samples"])
            )
            n_evaluated += 1
            variant = f": {pipeline_id}" if pipeline_id != pipeline_method else ""
            print(
                f"[{trial_i}/{total_variants}] {arch} ({arch_num[arch]}/{n_archs}) | "
                f"{pipeline_method}{variant} ({_arch_i}/{variants_per_arch[arch]}) -- "
                "fixed-channel-fraction"
            )
            model, _default_root_dir = load_model_with_loader(
                client,
                best_run,
                args,
                fixed_channel_fraction_dm,
            )
            seeds = require_seed_tags(best_run)
            try:
                _prime_model_for_degradation_evaluation(
                    model,
                    args,
                    fixed_channel_fraction_dm,
                    eval_seed=seeds["seed_eval"],
                )
                logger = _make_eval_logger(
                    best_run.info.run_id,
                    dataset_name,
                    args,
                    log_test_commit=False,
                )
                model._log_hyperparams = False
                _run_fixed_channel_fraction_evaluation(
                    run=best_run,
                    model=model,
                    logger=logger,
                    dm=fixed_channel_fraction_dm,
                    args=args,
                    eval_data_seed=int(run_eval_context["eval_data_seed"]),
                    fixed_channel_fraction=fixed_channel_fraction,
                    bootstrap_ci_context=run_bootstrap_ci_context,
                    canonical_clean_df=canonical_clean_df,
                    canonical_context_signature=canonical_context_signature,
                    perturbation_scenario_params_signature=run_params_signature,
                )
            finally:
                _teardown_model_after_eval(model)
            continue

        # Skip if already tested (with all eval context tags present)
        if not args.rerun and is_fully_tested(best_run, args=args, client=client):
            n_skipped += 1
            continue

        n_evaluated += 1
        variant = f": {pipeline_id}" if pipeline_id != pipeline_method else ""
        print(
            f"[{trial_i}/{total_variants}] {arch} ({arch_num[arch]}/{n_archs}) | "
            f"{pipeline_method}{variant} ({_arch_i}/{variants_per_arch[arch]}) -- evaluating"
        )
        model, default_root_dir = load_model_with_loader(client, best_run, args, dm)

        try:
            # Set test variant tag before evaluation so downstream analysis can
            # attribute failures to the intended selected variant.
            client.set_tag(best_run.info.run_id, "test_variant", pipeline_id)
            tested_run_ids.add(best_run.info.run_id)
            _evaluate_run(
                best_run,
                model,
                default_root_dir,
                dataset_name,
                args,
                dm,
                eval_data_seed=eval_data_seed,
            )
        finally:
            try:
                _teardown_model_after_eval(model)
            finally:
                del model

    print(
        f"Evaluation complete: {n_evaluated} evaluated, "
        f"{n_skipped} skipped (already tested), "
        f"{total_variants} variants ({n_archs} architectures)."
    )

    print("Done.")
    return n_runs, dataset_coverage_fractions
