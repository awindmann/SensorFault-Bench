"""Signature computation for run deduplication and data config validation."""

import hashlib
import json
from typing import Any, Optional

from utils.parsing import parse_dataset_split_mode, require_namespace_value


_PROVENANCE_ONLY_HPARAMS_BY_ARCHITECTURE = {
    "Chronos2": frozenset({"chronos_model_revision"}),
}


def _signature_hparams(model_architecture: str, hparams: dict[str, Any]) -> dict[str, Any]:
    excluded = _PROVENANCE_ONLY_HPARAMS_BY_ARCHITECTURE.get(
        str(model_architecture),
        frozenset(),
    )
    return {key: hparams[key] for key in sorted(hparams) if key not in excluded}


def compute_data_config_signature(
    *,
    dataset_spec: Any,
    args: Any,
) -> str:
    """Hash the data/config contract for comparability.

    The signature includes:
    - Dataset key and channel configuration
    - Sequence lengths (input_len, target_len)
    - Split policy (train_split, val_split, purged_fraction, etc.)

    Explicitly excluded (run budget / reproducibility knobs):
    - n_train_samples, n_val_samples, n_test_samples (but seed is included when sampling is enabled)
    - stride, seed (unless sampling is enabled via n_*_samples)

    Args:
        dataset_spec: Resolved dataset specification with key, channels, etc.
        args: Namespace containing data configuration arguments.

    Returns:
        SHA256 hex digest of the normalized config.
    """
    if dataset_spec.input_channels:
        input_channels = sorted(dataset_spec.input_channels)
    else:
        input_channels = "all"

    if dataset_spec.target_channels:
        target_channels = sorted(dataset_spec.target_channels)
    else:
        target_channels = "all"

    split_mode = parse_dataset_split_mode(getattr(dataset_spec, "split_mode", None))
    if split_mode == "within_batches" and args.shuffle_batches_before_split:
        raise ValueError(
            "shuffle_batches_before_split must be false when split_mode='within_batches'."
        )
    normalized = {
        "dataset": dataset_spec.key,
        "target_channels": target_channels,
        "input_channels": input_channels,
        "input_len": args.input_len,
        "target_len": args.target_len,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "strict_iid": args.strict_iid,
    }
    if split_mode == "temporal":
        # Batch shuffling does not affect temporal splits, so temporal datasets
        # keep the canonical marker while split-mode-changing datasets include
        # their realized split semantics explicitly.
        normalized["shuffle_batches_before_split"] = True
        normalized["purged_fraction"] = args.purged_fraction
    if split_mode == "within_batches":
        normalized["split_mode"] = split_mode
        normalized["purged_fraction"] = args.purged_fraction
    if split_mode == "across_batches":
        normalized["split_mode"] = split_mode
        normalized["shuffle_batches_before_split"] = args.shuffle_batches_before_split
    n_train_samples = require_namespace_value(args, key="n_train_samples")
    n_val_samples = require_namespace_value(args, key="n_val_samples")
    n_test_samples = require_namespace_value(args, key="n_test_samples")
    sampling_enabled = any(
        value is not None for value in (n_train_samples, n_val_samples, n_test_samples)
    )
    if sampling_enabled:
        sampling_seed = require_namespace_value(args, key="data_split_seed")
        if sampling_seed is None:
            sampling_seed = require_namespace_value(args, key="seed")
        if sampling_seed is None:
            raise ValueError("Sampling is enabled but args.seed is missing.")
        normalized["sampling_seed"] = sampling_seed
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_signature(
    model_architecture: str,
    dataset_key: str,
    hparams: dict[str, Any],
    *,
    pipeline_id: str = "baseline",
    data_config_signature: str,
    recipe_params: Optional[dict[str, Any]] = None,
) -> str:
    """Stable signature for skip/rerun semantics.
    
    Args:
        model_architecture: Model class name (e.g., "GRU").
        dataset_key: Dataset identifier.
        hparams: Hyperparameter dict (sorted for stability).
        pipeline_id: Pipeline variant identifier (default "baseline").
        data_config_signature: Hash of data configuration (required).
        recipe_params: Optional dict of recipe-specific parameters.

    Returns:
        SHA256 hex digest of the normalized signature payload.
    """
    normalized = {
        "model_architecture": model_architecture,
        "dataset": dataset_key,
        "pipeline_id": pipeline_id,
        "data_config_signature": data_config_signature,
        "hyperparameters": _signature_hparams(model_architecture, hparams),
    }
    if recipe_params:
        normalized["recipe_params"] = recipe_params
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["compute_data_config_signature", "build_signature"]
