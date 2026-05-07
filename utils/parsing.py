from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from decimal import Decimal
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np
import pandas as pd
import torch
from utils.rng import derive_seed

_TRUE_BOOL_TOKENS = frozenset(("true", "1", "yes", "y", "t"))
_FALSE_BOOL_TOKENS = frozenset(("false", "0", "no", "n", "f"))
_NULL_TOKENS = frozenset(("none", "null", ""))
_REMOTE_MLFLOW_URI_PREFIXES = ("http://", "https://")
_LOCAL_FILE_URI_PREFIX = "file:"
_S3_URI_PREFIX = "s3:"
BOOTSTRAP_CI_SEMANTICS = "cell_stratified_percentile"
SHARED_ANCHOR_BOOTSTRAP_CI_SEMANTICS = "shared_anchor_percentile"
METHOD_DELTA_PAIR_BOOTSTRAP_CI_SEMANTICS = "method_delta_pair_percentile"
DEGRADATION_SCORING_SEMANTICS = "uniform_severity_degradation"
ROBUSTNESS_RESULTS_COMPLETE_TAG = "robustness_results_complete"
IMPROVEMENT_SELECTION_MODES = ("clean", "perturbed_worst", "perturbed_mean")
SELECTION_METRIC_SEMANTICS = "perturbed_validation_error"


@dataclass(frozen=True)
class CoreFigureRegistry:
    baseline_rank_pareto_metric: str
    core_improvement_trajectory_method: str
    core_improvement_trajectory_metric: str
    dataset_spec: tuple[tuple[str, str], ...]
    method_display: dict[str, str]
    method_order: tuple[str, ...]
    scenario_display_order: tuple[str, ...]
    scenario_display: dict[str, str]
    scenario_groups: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class RuntimePrecisionConfig:
    precision: str
    model_dtype: torch.dtype | None
    input_dtype: torch.dtype | None
    autocast_dtype: torch.dtype | None


class SelectionPerturbationContextNotReadyError(ValueError):
    """Current run selection-context tags are missing, stale, or malformed."""


def normalize_data_root(value: Any, *, key: str = "DATA_ROOT") -> str:
    """Normalize an explicit dataset root path or S3 URI."""
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a non-empty string.")
    token = value.strip()
    if token == "":
        raise ValueError(f"{key} must be a non-empty string.")
    if token.lower().startswith("s3://"):
        suffix = token[5:].lstrip("/")
        if suffix == "":
            raise ValueError(f"{key} must include an S3 bucket, got {value!r}.")
        normalized = "s3://" + suffix
        return normalized.rstrip("/")
    if token.lower().startswith(_S3_URI_PREFIX):
        raise ValueError(f"{key} must use an explicit s3:// URI, got {value!r}.")
    if "://" in token:
        raise ValueError(f"{key} must be a local path or s3:// URI, got {value!r}.")
    normalized = Path(token).as_posix()
    if normalized == "/":
        return normalized
    return normalized.rstrip("/")


def parse_relative_dataset_path(value: Any, *, key: str = "dataset path") -> str:
    """Parse a registry-owned dataset filename relative to DATA_ROOT."""
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a non-empty relative path string.")
    token = value.strip()
    if token == "":
        raise ValueError(f"{key} must be a non-empty relative path string.")
    if token.lower().startswith(_S3_URI_PREFIX) or "://" in token:
        raise ValueError(f"{key} must be relative to DATA_ROOT, got {value!r}.")
    path = Path(token)
    if path.is_absolute():
        raise ValueError(f"{key} must be relative to DATA_ROOT, got {value!r}.")
    if ".." in path.parts:
        raise ValueError(f"{key} must not traverse outside DATA_ROOT, got {value!r}.")
    return path.as_posix()


def join_data_root_path(
    data_root: Any,
    relative_path: Any,
    *,
    key: str = "DATA_ROOT",
) -> str:
    """Join an explicit DATA_ROOT with a registry-owned relative dataset path."""
    root = normalize_data_root(data_root, key=key)
    path = parse_relative_dataset_path(relative_path)
    if root.lower().startswith("s3://"):
        return f"{root}/{path}"
    return (Path(root) / path).as_posix()


def robustness_results_complete_tag_value(*, complete: bool) -> str:
    """Render the degradation-results completeness tag value."""
    return "true" if complete else "false"


def parse_robustness_results_complete(
    value: Any,
    *,
    key: str = ROBUSTNESS_RESULTS_COMPLETE_TAG,
) -> bool:
    """Parse the robustness-results completeness tag into a boolean."""
    return parse_required_bool(
        value,
        key=key,
    )


def require_robustness_results_complete_tag(
    tags: Mapping[str, Any] | None,
    *,
    run_id: str,
) -> bool:
    """Require the robustness-results completeness tag and return whether it is complete."""
    status = optional_nonempty_tag_value(tags, key=ROBUSTNESS_RESULTS_COMPLETE_TAG)
    if status is None:
        raise ValueError(
            f"Run {run_id} is missing required {ROBUSTNESS_RESULTS_COMPLETE_TAG} tag."
        )
    return parse_robustness_results_complete(
        status,
        key=ROBUSTNESS_RESULTS_COMPLETE_TAG,
    )


def build_shared_anchor_bootstrap_ci_seed_key(test_metric: str) -> str:
    """Build the deterministic seed-derivation key for shared-anchor bootstrap CI."""
    metric = parse_optional_nonempty_string(
        test_metric,
        key="test_metric",
        context="Shared-anchor bootstrap CI seed derivation",
        disallow_none_token=True,
    )
    if metric is None:
        raise ValueError(
            "test_metric is required for shared-anchor bootstrap CI seed derivation."
        )
    return (
        "bootstrap_ci:"
        f"degradation:{metric}:{SHARED_ANCHOR_BOOTSTRAP_CI_SEMANTICS}"
    )


def build_method_delta_pair_bootstrap_ci_seed_key(
    test_metric: str,
    *,
    dataset: str,
    data_config_signature: str,
    robustness_method: str,
) -> str:
    """Build the deterministic seed-derivation key for method-delta pair bootstrap CI."""
    metric = parse_optional_nonempty_string(
        test_metric,
        key="test_metric",
        context="Method-delta pair bootstrap CI seed derivation",
        disallow_none_token=True,
    )
    if metric is None:
        raise ValueError(
            "test_metric is required for method-delta pair bootstrap CI seed derivation."
        )
    normalized_dataset = parse_optional_nonempty_string(
        dataset,
        key="dataset",
        context="Method-delta pair bootstrap CI seed derivation",
        disallow_none_token=True,
    )
    if normalized_dataset is None:
        raise ValueError(
            "dataset is required for method-delta pair bootstrap CI seed derivation."
        )
    normalized_signature = parse_optional_nonempty_string(
        data_config_signature,
        key="data_config_signature",
        context="Method-delta pair bootstrap CI seed derivation",
        disallow_none_token=True,
    )
    if normalized_signature is None:
        raise ValueError(
            "data_config_signature is required for method-delta pair bootstrap CI seed derivation."
        )
    normalized_method = parse_optional_nonempty_string(
        robustness_method,
        key="robustness_method",
        context="Method-delta pair bootstrap CI seed derivation",
        disallow_none_token=True,
    )
    if normalized_method is None:
        raise ValueError(
            "robustness_method is required for method-delta pair bootstrap CI seed derivation."
        )
    return (
        "bootstrap_ci:"
        f"method_delta:{metric}:{normalized_dataset}:{normalized_signature}:"
        f"{normalized_method}:{METHOD_DELTA_PAIR_BOOTSTRAP_CI_SEMANTICS}"
    )


def coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        token = value.strip()
        if token == "":
            return None
        try:
            return int(token)
        except ValueError:
            try:
                parsed = float(token)
            except ValueError:
                return None
            return int(parsed) if parsed.is_integer() else None
    return None


def padded_feature_names(
    names: Sequence[str | None] | None,
    count: int,
    *,
    prefix: str = "Target",
) -> list[str]:
    """Return a list of *count* feature names, padding missing/blank entries."""
    if names is None:
        raw_names: Sequence[str | None] = ()
    else:
        raw_names = names
    result = [
        name if isinstance(name, str) and name.strip() else f"{prefix} {idx + 1}"
        for idx, name in enumerate(raw_names)
    ]
    result.extend(
        f"{prefix} {idx + 1}" for idx in range(len(result), count)
    )
    return result[:count]


def require_denormalized_forecast_payload(
    payload: Any,
    *,
    run_id: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Run {run_id} forecast payload must be a dictionary.")
    if payload.get("denormalized") is not True:
        raise ValueError(
            f"Run {run_id} forecast payload must declare denormalized=true. "
            "Re-run testing to regenerate forecast artifacts."
        )
    return payload


def _normalize_token(value: Any) -> tuple[str, str]:
    token = str(value).strip()
    return token, token.lower()


def _is_null_token(lowered_token: str) -> bool:
    return lowered_token in _NULL_TOKENS


def _parse_bool_token(lowered_token: str) -> Optional[bool]:
    if lowered_token in _TRUE_BOOL_TOKENS:
        return True
    if lowered_token in _FALSE_BOOL_TOKENS:
        return False
    return None


def build_mlflow_tracking_uri(logdir: Any) -> str:
    """Normalize a configured MLflow logdir into a tracking URI."""
    token = str(logdir).strip()
    if token == "":
        raise ValueError("logdir must be a non-empty string.")
    if token.startswith(_REMOTE_MLFLOW_URI_PREFIXES):
        return token
    if token.startswith(_LOCAL_FILE_URI_PREFIX):
        return token
    return f"{_LOCAL_FILE_URI_PREFIX}{token}"


def resolve_mlflow_local_save_dir(logdir: Any) -> str | None:
    """Return the local filesystem path used for MLflow save_dir.

    Remote HTTP(S) tracking URIs return ``None`` because Lightning should not
    construct a local file-backed logger path for those backends.
    """
    token = str(logdir).strip()
    if token == "":
        raise ValueError("logdir must be a non-empty string.")
    if token.startswith(_REMOTE_MLFLOW_URI_PREFIXES):
        return None
    if not token.startswith(_LOCAL_FILE_URI_PREFIX):
        return token

    parsed = urlparse(token)
    host = parsed.netloc
    if host not in ("", "localhost"):
        raise ValueError(
            "MLflow file URIs must reference the local machine. "
            f"Received host '{host}'."
        )
    path = url2pathname(parsed.path)
    if path == "":
        raise ValueError(f"MLflow file URI '{token}' must include a local path.")
    return path


def _parse_collection_literal(
    token: str,
    *,
    expected_type: tuple[type, ...],
    key: str,
    error_prefix: str,
    include_parse_errors: bool = False,
) -> Any:
    parse_errors: list[str] = []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(token)
        except Exception as exc:
            parse_errors.append(f"{getattr(parser, '__name__', str(parser))}: {exc}")
            continue
        if isinstance(parsed, expected_type):
            return parsed
        if not include_parse_errors:
            break
    expected_label = (
        expected_type[0].__name__
        if len(expected_type) == 1
        else " or ".join(t.__name__ for t in expected_type)
    )
    if include_parse_errors:
        raise ValueError(
            f"{error_prefix} '{key}'='{token}' to {expected_label}. "
            f"Parse errors: {parse_errors}"
        )
    raise ValueError(f"{key} must be a {expected_label}.")


def parse_value(value: Any, expected_type: type, *, key: str):
    """Strict parsing helper that raises on invalid values."""
    if value is None:
        return None
    if expected_type is bool:
        if isinstance(value, bool):
            return value
        _, lowered = _normalize_token(value)
        parsed_bool = _parse_bool_token(lowered)
        if parsed_bool is not None:
            return parsed_bool
        raise ValueError(f"Cannot parse {key}='{value}' as bool.")
    if expected_type is int:
        token, lowered = _normalize_token(value)
        if _is_null_token(lowered):
            return None
        try:
            return int(token)
        except ValueError as exc:
            raise ValueError(f"Cannot parse {key}='{value}' as int.") from exc
    if expected_type is float:
        token, lowered = _normalize_token(value)
        if _is_null_token(lowered):
            return None
        return float(token)
    if expected_type is str:
        return str(value)
    if expected_type in (list, tuple):
        if isinstance(value, (list, tuple)):
            parsed = list(value)
        else:
            token, lowered = _normalize_token(value)
            if _is_null_token(lowered):
                return None
            parsed = _parse_collection_literal(
                token,
                expected_type=(list, tuple),
                key=key,
                error_prefix="Cannot parse",
            )
            parsed = list(parsed)
        return tuple(parsed) if expected_type is tuple else parsed
    return expected_type(value)


def parse_feature_indices(
    value: Any,
    *,
    n_features: int,
    key: str = "target_indices",
    allow_none: bool = False,
) -> Optional[tuple[int, ...]]:
    """Parse strict feature-index sequences used for target/input mapping."""
    if not isinstance(n_features, int):
        raise ValueError(
            f"n_features must be an int, got {type(n_features).__name__}."
        )
    if n_features <= 0:
        raise ValueError(f"n_features must be > 0, got {n_features}.")

    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{key} is required.")

    if isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be an iterable of integer indices.")
    try:
        iterator = iter(value)
    except TypeError as exc:
        raise ValueError(f"{key} must be an iterable of integer indices.") from exc

    parsed: list[int] = []
    seen: set[int] = set()
    for raw_idx in iterator:
        scalar_value = raw_idx
        if hasattr(raw_idx, "numel") and callable(getattr(raw_idx, "numel")):
            try:
                numel = int(raw_idx.numel())
            except Exception as exc:
                raise ValueError(
                    f"{key} entries must be scalar values, got {raw_idx!r}."
                ) from exc
            if numel != 1:
                raise ValueError(
                    f"{key} entries must be scalar values, got {raw_idx!r}."
                )
            try:
                scalar_value = raw_idx.item()
            except Exception as exc:
                raise ValueError(
                    f"{key} entries must be scalar values, got {raw_idx!r}."
                ) from exc

        if isinstance(scalar_value, bool):
            raise ValueError(f"{key} must contain integer values, not bools.")

        try:
            numeric_value = float(scalar_value)
        except (TypeError, ValueError):
            numeric_value = None
        if numeric_value is not None and not numeric_value.is_integer():
            raise ValueError(
                f"{key} must contain integer values, got {scalar_value!r}."
            )

        try:
            idx = int(scalar_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{key} must contain integer values, got {scalar_value!r}."
            ) from exc

        if idx < 0 or idx >= n_features:
            raise ValueError(
                f"{key} values must be in range [0, {n_features - 1}], got {idx}."
            )
        if idx in seen:
            raise ValueError(f"{key} must be unique, found duplicate {idx}.")
        seen.add(idx)
        parsed.append(idx)

    if len(parsed) == 0:
        raise ValueError(f"{key} must be non-empty when provided.")
    return tuple(parsed)


def parse_revin_settings(
    *,
    use_revin: Any,
    revin_affine: Any,
    revin_denorm: Any,
    revin_eps: Any,
) -> tuple[bool, bool, bool, float]:
    """Parse strict RevIN constructor settings."""
    if not isinstance(use_revin, bool):
        raise ValueError(
            f"use_revin must be a bool, got {type(use_revin).__name__}."
        )
    if not isinstance(revin_affine, bool):
        raise ValueError(
            f"revin_affine must be a bool, got {type(revin_affine).__name__}."
        )
    if not isinstance(revin_denorm, bool):
        raise ValueError(
            f"revin_denorm must be a bool, got {type(revin_denorm).__name__}."
        )
    try:
        revin_eps_value = float(revin_eps)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"revin_eps must be a positive float, got {revin_eps!r}."
        ) from exc
    if revin_eps_value <= 0.0:
        raise ValueError(f"revin_eps must be > 0, got {revin_eps_value}.")
    return use_revin, revin_affine, revin_denorm, revin_eps_value


def parse_optional_name_tuple(
    value: Any,
    *,
    key: str,
) -> Optional[tuple[str, ...]]:
    """Parse an optional sequence of unique non-empty names."""
    if value is None:
        return None

    if isinstance(value, str):
        raw_values = [value]
    elif isinstance(value, Sequence):
        raw_values = list(value)
    else:
        raise ValueError(f"{key} must be a string or sequence of strings.")

    names: list[str] = []
    seen: set[str] = set()
    for idx, raw_name in enumerate(raw_values):
        if not isinstance(raw_name, str):
            raise ValueError(
                f"{key} entry at index {idx} must be a string; got {type(raw_name).__name__}."
            )
        name = raw_name.strip()
        if not name:
            raise ValueError(f"{key} contains an empty name at index {idx}.")
        if name in seen:
            raise ValueError(f"{key} contains duplicate name '{name}'.")
        seen.add(name)
        names.append(name)
    return tuple(names)


def validate_channel_name_partition(
    input_channels: Any,
    continuous_channels: Any,
    discrete_channels: Any,
    *,
    context: str,
) -> tuple[Optional[tuple[str, ...]], Optional[tuple[str, ...]], Optional[tuple[str, ...]]]:
    """Validate that continuous/discrete channels form a complete input partition."""
    normalized_inputs = parse_optional_name_tuple(
        input_channels,
        key=f"{context}.input_channels",
    )
    normalized_continuous = parse_optional_name_tuple(
        continuous_channels,
        key=f"{context}.continuous_channels",
    )
    normalized_discrete = parse_optional_name_tuple(
        discrete_channels,
        key=f"{context}.discrete_channels",
    )

    if normalized_continuous is None and normalized_discrete is None:
        return normalized_inputs, None, None
    if normalized_continuous is None or normalized_discrete is None:
        raise ValueError(
            f"{context} must provide continuous_channels and discrete_channels together."
        )
    if normalized_inputs is None:
        raise ValueError(
            f"{context} cannot declare continuous/discrete channels without input_channels."
        )

    input_set = set(normalized_inputs)
    overlap = sorted(set(normalized_continuous) & set(normalized_discrete))
    if overlap:
        raise ValueError(
            f"{context} continuous_channels and discrete_channels overlap: {overlap}."
        )

    declared_set = set(normalized_continuous) | set(normalized_discrete)
    missing = [name for name in normalized_inputs if name not in declared_set]
    unknown = sorted(declared_set - input_set)
    if missing or unknown:
        problems: list[str] = []
        if missing:
            problems.append(f"missing from partition: {missing}")
        if unknown:
            problems.append(f"not present in input_channels: {unknown}")
        raise ValueError(
            f"{context} must partition input_channels exactly; " + "; ".join(problems) + "."
        )

    return normalized_inputs, normalized_continuous, normalized_discrete


VALID_NOISE_CHANNELS = ("target_only", "continuous", "all")
VALID_ENSEMBLE_COMBINE_METHODS = ("median", "mean")


def validate_noise_channels(value: Any, *, key: str = "noise_channels") -> str:
    """Parse the strict RT/RS noise-channel scope enum."""
    return parse_required_choice(value, key=key, allowed=VALID_NOISE_CHANNELS)


def validate_ensemble_combine_method(
    value: Any,
    *,
    key: str = "ensemble_combine_method",
) -> str:
    """Parse the strict ensemble forecast-combination operator enum."""
    return parse_required_choice(
        value,
        key=key,
        allowed=VALID_ENSEMBLE_COMBINE_METHODS,
    )


def validate_trim_alpha(
    value: Any,
    sample_count: Any,
    *,
    key: str = "rs_trim_alpha",
) -> float:
    """Parse and validate symmetric alpha trimming for RS sample aggregation."""
    parsed = parse_value(value, float, key=key)
    if parsed is None:
        raise ValueError(f"{key} must be provided.")
    alpha = _require_finite_float(parsed, key=key)
    if alpha <= 0.0 or alpha >= 0.5:
        raise ValueError(f"{key} must satisfy 0 < {key} < 0.5; got {alpha}.")

    resolved_sample_count = parse_required_positive_int(
        sample_count,
        key="rs_sample_count",
    )
    trim_count = math.floor(alpha * resolved_sample_count)
    if trim_count < 1:
        raise ValueError(
            f"{key}={alpha} with rs_sample_count={resolved_sample_count} trims zero "
            "samples per tail; require floor(rs_trim_alpha * rs_sample_count) >= 1."
        )
    # Unreachable with alpha < 0.5 in practice, but spec-required guard.
    retained_count = resolved_sample_count - 2 * trim_count
    if retained_count < 1:
        raise ValueError(
            f"{key}={alpha} with rs_sample_count={resolved_sample_count} removes all "
            "samples; require rs_sample_count - 2 * floor(rs_trim_alpha * "
            "rs_sample_count) >= 1."
        )
    return alpha


def build_noise_channel_mask(
    input_columns: Any,
    target_columns: Any,
    continuous_channels: Any,
    noise_channels: Any,
    *,
    key: str = "noise_channels",
) -> torch.Tensor | None:
    """Resolve a strict RT/RS channel scope into a float mask over model inputs."""
    normalized_input_columns = parse_optional_name_tuple(
        input_columns,
        key=f"{key}.input_columns",
    )
    if normalized_input_columns is None:
        raise ValueError(f"{key} requires input_columns.")
    if len(normalized_input_columns) == 0:
        raise ValueError(f"{key} requires at least one input column.")

    normalized_noise_channels = validate_noise_channels(noise_channels, key=key)
    if normalized_noise_channels == "all":
        return None

    input_pos = {
        name: idx for idx, name in enumerate(normalized_input_columns)
    }
    if normalized_noise_channels == "target_only":
        normalized_target_columns = parse_optional_name_tuple(
            target_columns,
            key=f"{key}.target_columns",
        )
        if normalized_target_columns is None:
            raise ValueError(f"{key}='target_only' requires target_columns.")
        target_set = set(normalized_target_columns)
        selected_names = [
            name for name in normalized_input_columns if name in target_set
        ]
        if not selected_names:
            raise ValueError(
                f"{key}='target_only' but no target channels map to model inputs."
            )
    elif normalized_noise_channels == "continuous":
        normalized_continuous_channels = parse_optional_name_tuple(
            continuous_channels,
            key=f"{key}.continuous_channels",
        )
        if normalized_continuous_channels is None:
            raise ValueError(f"{key}='continuous' requires continuous_channels.")
        if len(normalized_continuous_channels) == 0:
            raise ValueError(
                f"{key}='continuous' but no continuous channels were provided."
            )
        missing = [
            name
            for name in normalized_continuous_channels
            if name not in input_pos
        ]
        if missing:
            raise ValueError(
                f"{key}='continuous' includes channels not present in model inputs: {missing}."
            )
        selected_names = list(normalized_continuous_channels)
    else:
        raise AssertionError(f"Unhandled noise channel scope '{normalized_noise_channels}'.")

    mask = torch.zeros(len(normalized_input_columns), dtype=torch.float32)
    for name in selected_names:
        mask[input_pos[name]] = 1.0
    return mask


def validate_input_stats(
    input_means: Any,
    input_stds: Any,
    expected_channels: int,
    *,
    key: str = "input_stats",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and return cloned (means, stds) tensors for raw-space noise."""
    if input_means is None or input_stds is None:
        raise ValueError(f"{key} requires input_means and input_stds.")
    means = torch.as_tensor(input_means, dtype=torch.float32)
    stds = torch.as_tensor(input_stds, dtype=torch.float32)
    if means.ndim != 1 or stds.ndim != 1:
        raise ValueError(f"{key} input_means/input_stds must be 1D tensors.")
    if means.numel() != expected_channels or stds.numel() != expected_channels:
        raise ValueError(
            f"{key} input_means/input_stds length ({means.numel()}, {stds.numel()}) "
            f"must match expected channels ({expected_channels})."
        )
    if torch.any(stds <= 0):
        raise ValueError(f"{key} input_stds must be strictly positive.")
    return means.clone(), stds.clone()


def parse_required_positive_int(value: Any, *, key: str) -> int:
    """Parse a required positive integer value."""
    parsed = parse_value(value, int, key=key)
    if parsed is None:
        raise ValueError(f"{key} must be provided.")
    result = int(parsed)
    if result <= 0:
        raise ValueError(f"{key} must be > 0; got {result}.")
    return result


def parse_required_nonnegative_int(
    value: Any,
    *,
    key: str,
    context: str | None = None,
) -> int:
    """Parse a required non-negative integer value."""
    subject = context if context else "Value"
    try:
        parsed = parse_value(value, int, key=key)
    except ValueError as exc:
        raise ValueError(f"{subject} requires integer {key}, got {value!r}.") from exc
    if parsed is None:
        raise ValueError(f"{subject} requires integer {key}, got {value!r}.")
    result = int(parsed)
    if result < 0:
        raise ValueError(f"{subject} requires {key} >= 0, got {result}.")
    return result


def parse_required_odd_positive_int(value: Any, *, key: str) -> int:
    """Parse a required positive odd integer value."""
    result = parse_required_positive_int(value, key=key)
    if result % 2 == 0:
        raise ValueError(f"{key} must be odd; got {result}.")
    return result


def parse_required_dropout(value: Any, *, key: str = "dropout") -> float:
    """Parse and validate dropout rate in [0, 1)."""
    try:
        parsed = parse_value(value, float, key=key)
    except ValueError as exc:
        raise ValueError(
            f"{key} must be a float in [0, 1); got {value!r}."
        ) from exc
    if parsed is None:
        raise ValueError(f"{key} must be provided.")
    dropout = float(parsed)
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError(f"{key} must satisfy 0 <= {key} < 1; got {dropout}.")
    return dropout


def parse_required_choice(
    value: Any,
    *,
    key: str,
    allowed: Sequence[str],
) -> str:
    """Parse a required string constrained to one of the allowed values."""
    parsed = parse_value(value, str, key=key)
    if parsed is None:
        raise ValueError(f"{key} must be provided.")
    normalized = str(parsed).strip().lower()
    allowed_normalized = tuple(str(item).strip().lower() for item in allowed)
    if normalized not in allowed_normalized:
        allowed_repr = ", ".join(f"'{item}'" for item in sorted(set(allowed_normalized)))
        raise ValueError(
            f"Unsupported {key} '{value}'. Supported: [{allowed_repr}]."
        )
    return normalized


def parse_required_canonical_choice(
    value: Any,
    *,
    key: str,
    allowed: Sequence[str],
) -> str:
    """Parse a required string and return the canonical allowed token."""
    parsed = parse_value(value, str, key=key)
    if parsed is None:
        raise ValueError(f"{key} must be provided.")
    token = str(parsed).strip()
    if not token:
        raise ValueError(f"{key} must be provided.")

    allowed_by_lower: dict[str, str] = {}
    for raw_allowed in allowed:
        canonical = str(raw_allowed).strip()
        if not canonical:
            raise ValueError(f"{key} allowed choices must be non-empty strings.")
        lowered = canonical.lower()
        existing = allowed_by_lower.get(lowered)
        if existing is not None and existing != canonical:
            raise ValueError(
                f"{key} allowed choices contain ambiguous duplicate '{canonical}'."
            )
        allowed_by_lower[lowered] = canonical

    resolved = allowed_by_lower.get(token.lower())
    if resolved is None:
        allowed_repr = ", ".join(f"'{item}'" for item in sorted(allowed_by_lower.values()))
        raise ValueError(
            f"Unsupported {key} '{value}'. Supported: [{allowed_repr}]."
        )
    return resolved


def parse_canonical_choice_list(
    value: Any,
    *,
    key: str,
    allowed: Sequence[str],
) -> list[str]:
    """Parse a list of strings against canonical allowed values."""
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{key} must be a list; got {type(value).__name__}.")
    parsed: list[str] = []
    seen: set[str] = set()
    for idx, raw_item in enumerate(value):
        if not isinstance(raw_item, str):
            raise TypeError(
                f"{key} entries must be strings; got {type(raw_item).__name__} at index {idx}."
            )
        item = parse_required_canonical_choice(
            raw_item,
            key=f"{key}[{idx}]",
            allowed=allowed,
        )
        if item in seen:
            raise ValueError(f"{key} contains duplicate value '{item}'.")
        seen.add(item)
        parsed.append(item)
    return parsed


def parse_model_architecture_scope(
    value: Any,
    *,
    benchmark_architectures: Any,
    allowed: Sequence[str],
) -> list[str]:
    """Resolve MODEL plus BENCHMARK_ARCHITECTURES into canonical architectures."""
    if not allowed:
        raise ValueError("No architectures configured in baseline_hparams.yaml.")

    if value is None:
        parsed_benchmark_architectures = parse_canonical_choice_list(
            benchmark_architectures,
            key="BENCHMARK_ARCHITECTURES",
            allowed=allowed,
        )
        if not parsed_benchmark_architectures:
            raise ValueError("BENCHMARK_ARCHITECTURES must contain at least one architecture.")
        return parsed_benchmark_architectures

    if isinstance(value, str):
        raw_items: Sequence[Any] = (value,)
    elif isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raise TypeError(f"MODEL must be a list of model names; got {type(value).__name__}.")

    if not raw_items:
        raise ValueError("MODEL must contain at least one model name when provided.")

    parsed: list[str] = []
    seen: set[str] = set()
    for idx, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, str):
            raise TypeError(
                "MODEL entries must be strings; "
                f"got {type(raw_item).__name__} at index {idx}."
            )
        token = raw_item.strip()
        if not token:
            raise ValueError(f"MODEL[{idx}] must be a non-empty model name.")
        if token.lower() == "all":
            raise ValueError(
                "MODEL no longer supports 'all'. Omit --model to use "
                "BENCHMARK_ARCHITECTURES, or pass one or more explicit model names."
            )
        item = parse_required_canonical_choice(
            token,
            key=f"MODEL[{idx}]",
            allowed=allowed,
        )
        if item in seen:
            raise ValueError(f"MODEL contains duplicate value '{item}'.")
        seen.add(item)
        parsed.append(item)
    return parsed


def parse_method_scope(
    value: Any,
    *,
    benchmark_methods: Any,
    allowed: Sequence[str],
) -> list[str]:
    """Resolve METHOD plus BENCHMARK_METHODS into canonical pipeline methods."""
    if not allowed:
        raise ValueError("No pipeline methods configured for method-scope parsing.")

    if value is None:
        parsed_benchmark_methods = parse_canonical_choice_list(
            benchmark_methods,
            key="BENCHMARK_METHODS",
            allowed=allowed,
        )
        if not parsed_benchmark_methods:
            raise ValueError("BENCHMARK_METHODS must contain at least one method.")
        return parsed_benchmark_methods

    if isinstance(value, str):
        raw_items: Sequence[Any] = (value,)
    elif isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raise TypeError(f"METHOD must be a list of method names; got {type(value).__name__}.")

    if not raw_items:
        raise ValueError("METHOD must contain at least one method name when provided.")

    parsed: list[str] = []
    seen: set[str] = set()
    for idx, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, str):
            raise TypeError(
                "METHOD entries must be strings; "
                f"got {type(raw_item).__name__} at index {idx}."
            )
        token = raw_item.strip()
        if not token:
            raise ValueError(f"METHOD[{idx}] must be a non-empty method name.")
        if token.lower() == "all":
            raise ValueError(
                "METHOD no longer supports 'all'. Omit --method to use "
                "BENCHMARK_METHODS, or pass one or more explicit method names."
            )
        item = parse_required_canonical_choice(
            token,
            key=f"METHOD[{idx}]",
            allowed=allowed,
        )
        if item in seen:
            raise ValueError(f"METHOD contains duplicate value '{item}'.")
        seen.add(item)
        parsed.append(item)
    return parsed


def parse_method_architecture_applicability(
    value: Any,
    *,
    benchmark_methods: Sequence[str],
    benchmark_architectures: Sequence[str],
    key: str = "method_architecture_applicability",
) -> dict[str, tuple[str, ...]]:
    """Validate benchmark method->architecture applicability metadata."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping.")
    methods = parse_canonical_choice_list(
        list(benchmark_methods),
        key="benchmark_methods",
        allowed=benchmark_methods,
    )
    architectures = parse_canonical_choice_list(
        list(benchmark_architectures),
        key="benchmark_architectures",
        allowed=benchmark_architectures,
    )
    parsed: dict[str, tuple[str, ...]] = {}
    for method in methods:
        if method not in value:
            raise ValueError(f"{key} is missing required method '{method}'.")
        parsed[method] = tuple(
            parse_canonical_choice_list(
                value[method],
                key=f"{key}.{method}",
                allowed=architectures,
            )
        )
        if not parsed[method]:
            raise ValueError(f"{key}.{method} must contain at least one architecture.")

    unknown_methods = sorted(str(method) for method in value.keys() if method not in methods)
    if unknown_methods:
        raise ValueError(
            f"{key} contains unknown method(s): {', '.join(unknown_methods)}."
        )
    return parsed


def resolve_applicable_method_architecture_scope(
    *,
    methods: Sequence[str],
    architectures: Sequence[str],
    applicability: Mapping[str, Sequence[str]],
    explicit_architectures: bool,
    context: str,
) -> dict[str, tuple[str, ...]]:
    """Resolve runnable benchmark method/architecture cells and reject explicit bad pairs."""
    if not methods:
        raise ValueError(f"{context}: method scope must not be empty.")
    if not architectures:
        raise ValueError(f"{context}: architecture scope must not be empty.")
    if not isinstance(applicability, Mapping):
        raise ValueError(f"{context}: applicability must be a mapping.")

    known_architectures = {
        str(arch)
        for method_architectures in applicability.values()
        for arch in method_architectures
    }
    resolved: dict[str, tuple[str, ...]] = {}
    unsupported_pairs: list[tuple[str, str]] = []

    for raw_method in methods:
        method = str(raw_method)
        if method not in applicability:
            raise ValueError(f"{context}: unknown benchmark method '{method}'.")
        method_architectures = tuple(str(arch) for arch in applicability[method])
        method_architecture_set = set(method_architectures)
        supported_architectures: list[str] = []
        for raw_architecture in architectures:
            architecture = str(raw_architecture)
            if architecture not in known_architectures:
                raise ValueError(
                    f"{context}: unknown benchmark architecture '{architecture}'."
                )
            if architecture in method_architecture_set:
                supported_architectures.append(architecture)
            else:
                unsupported_pairs.append((method, architecture))
        if supported_architectures:
            resolved[method] = tuple(supported_architectures)

    if unsupported_pairs and explicit_architectures:
        preview = ", ".join(
            f"{method}/{architecture}"
            for method, architecture in unsupported_pairs
        )
        raise ValueError(
            f"{context}: unsupported benchmark method/architecture pair(s): {preview}."
        )
    if not resolved:
        raise ValueError(
            f"{context}: no applicable benchmark method/architecture pairs remain."
        )
    return resolved


def parse_improvement_selection_mode(
    value: Any,
    *,
    key: str = "improvement_selection_mode",
) -> str:
    """Parse the configured robustness-improvement selector mode."""
    return parse_required_canonical_choice(
        value,
        key=key,
        allowed=IMPROVEMENT_SELECTION_MODES,
    )


VALID_SPLIT_MODES = ("temporal", "across_batches", "within_batches")


def parse_dataset_split_mode(value: Any, *, key: str = "split_mode") -> str:
    """Parse the explicit dataset split-mode contract."""
    return parse_required_choice(value, key=key, allowed=VALID_SPLIT_MODES)


def parse_dataset_window_defaults(
    value: Any,
    *,
    required_datasets: Sequence[str] = (),
) -> dict[str, dict[str, int]]:
    """Parse and validate dataset-specific window and batch-size defaults from YAML."""
    if value is None:
        raise ValueError("Dataset window defaults config must not be empty.")
    if not isinstance(value, Mapping):
        raise ValueError("Dataset window defaults config must be a mapping.")
    if not value:
        raise ValueError("Dataset window defaults config must be a non-empty mapping.")

    normalized: dict[str, dict[str, int]] = {}
    for raw_dataset, raw_window in value.items():
        dataset_name = parse_optional_nonempty_string(
            raw_dataset,
            key="dataset key",
            context="Dataset window defaults",
            disallow_none_token=True,
        )
        if dataset_name is None:
            raise ValueError("Dataset window defaults dataset key must be provided.")
        if dataset_name in normalized:
            raise ValueError(
                f"Dataset window defaults contains duplicate dataset '{dataset_name}'."
            )
        if not isinstance(raw_window, Mapping):
            raise ValueError(
                f"Dataset window defaults entry for '{dataset_name}' must be a mapping."
            )

        unknown_keys = sorted(set(raw_window.keys()) - {"input_len", "target_len", "batch_size"})
        if unknown_keys:
            raise ValueError(
                f"Dataset window defaults entry for '{dataset_name}' has unsupported key(s): "
                + ", ".join(str(key) for key in unknown_keys)
            )
        missing_keys = [
            key for key in ("input_len", "target_len") if key not in raw_window
        ]
        if missing_keys:
            raise ValueError(
                f"Dataset window defaults entry for '{dataset_name}' is missing required key(s): "
                + ", ".join(missing_keys)
            )

        normalized_entry = {
            "input_len": parse_required_positive_int(
                raw_window.get("input_len"),
                key=f"{dataset_name}.input_len",
            ),
            "target_len": parse_required_positive_int(
                raw_window.get("target_len"),
                key=f"{dataset_name}.target_len",
            ),
        }
        if "batch_size" in raw_window:
            normalized_entry["batch_size"] = parse_required_positive_int(
                raw_window.get("batch_size"),
                key=f"{dataset_name}.batch_size",
            )
        normalized[dataset_name] = normalized_entry

    missing_required = [
        dataset_name for dataset_name in required_datasets if dataset_name not in normalized
    ]
    if missing_required:
        raise ValueError(
            "Dataset window defaults config is missing required dataset(s): "
            + ", ".join(missing_required)
        )
    return normalized


def resolve_dataset_window_args(
    args: Any,
    *,
    dataset_spec: Any,
    dataset_window_defaults: Mapping[str, Mapping[str, Any]],
    explicit_arg_overrides: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Return a shallow args copy with dataset-scoped input/target lengths and batch size.

    Resolution precedence: explicit CLI override > dataset YAML. Registered
    benchmark datasets must have YAML-owned window defaults. Shipped benchmark
    configs also declare dataset-specific paper-track batch sizes.
    """
    dataset_name = parse_optional_nonempty_string(
        getattr(dataset_spec, "key", None),
        key="dataset_spec.key",
        context="Dataset window resolution",
        disallow_none_token=True,
    )
    if dataset_name is None:
        raise ValueError("dataset_spec.key is required for dataset window resolution.")

    if explicit_arg_overrides is None:
        raise ValueError(
            "explicit_arg_overrides is required for dataset window resolution."
        )
    if not isinstance(explicit_arg_overrides, Mapping):
        raise TypeError(
            "explicit_arg_overrides must be a mapping for dataset window resolution."
        )
    overrides = explicit_arg_overrides
    has_input = overrides.get("input_len") is not None
    has_target = overrides.get("target_len") is not None
    has_batch_size = overrides.get("batch_size") is not None
    if has_input != has_target:
        raise ValueError(
            "input_len and target_len must be provided together when overriding "
            "dataset-specific window defaults."
        )

    if has_input:
        input_len = parse_required_positive_int(overrides["input_len"], key="input_len")
        target_len = parse_required_positive_int(overrides["target_len"], key="target_len")
    else:
        dataset_defaults = dataset_window_defaults.get(dataset_name)
        if dataset_defaults is not None:
            # Already validated by parse_dataset_window_defaults. Extract directly.
            input_len = dataset_defaults["input_len"]
            target_len = dataset_defaults["target_len"]
        else:
            raise ValueError(
                "Dataset window defaults config is missing dataset "
                f"'{dataset_name}'. Add an entry to configs/dataset_windows.yaml "
                "or pass both --input-len and --target-len explicitly."
            )

    resolved_args = copy.copy(args)
    setattr(resolved_args, "input_len", input_len)
    setattr(resolved_args, "target_len", target_len)
    if has_batch_size:
        batch_size = parse_required_positive_int(overrides["batch_size"], key="batch_size")
    else:
        dataset_defaults = dataset_window_defaults.get(dataset_name)
        batch_size = (
            dataset_defaults.get("batch_size")
            if dataset_defaults is not None
            else None
        )
    if batch_size is not None:
        setattr(resolved_args, "batch_size", batch_size)
    return resolved_args


def validate_split_mode_batch_column(
    split_mode: str,
    batch_column: Optional[str],
) -> None:
    """Validate mutual constraint between split_mode and batch_column."""
    if batch_column is not None and not batch_column:
        raise ValueError("batch_column must be a non-empty string or None.")
    if batch_column is None and split_mode != "temporal":
        raise ValueError(
            "Datasets without batch_column must use split_mode='temporal'."
        )
    if batch_column is not None and split_mode == "temporal":
        raise ValueError(
            "Datasets with batch_column must use split_mode='across_batches' "
            "or split_mode='within_batches'."
        )


def normalize_yaml_value(value: Any) -> Any:
    """Recursively normalize values into YAML-serializable primitives."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {
            str(key): normalize_yaml_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [normalize_yaml_value(item) for item in value]
    return str(value)


def parse_max_hp_trials_per_model(value: Any) -> Optional[int]:
    """Parse ``max_hp_trials_per_model`` with strict validation."""
    parsed = parse_value(value, int, key="max_hp_trials_per_model")
    if parsed is None:
        return None
    if parsed <= 0:
        raise ValueError("max_hp_trials_per_model must be positive or None")
    return parsed


def parse_eval_data_seed(value: Any, *, key: str = "eval_data_seed") -> Optional[int]:
    """Parse an optional literal evaluation data-seed override."""
    parsed = parse_value(value, int, key=key)
    if parsed is None:
        return None
    return parsed


def resolve_meta_analysis_eval_data_seed_scope(
    eval_data_seed: Any,
    *,
    key: str = "eval_data_seed",
    default_eval_data_seed: Any = None,
    default_key: str | None = None,
) -> tuple[Optional[int], str]:
    """Resolve canonical-vs-explicit meta-analysis evaluation-seed scope."""
    resolved_seed = parse_eval_data_seed(eval_data_seed, key=key)
    if default_key is not None and resolved_seed is None:
        resolved_seed = parse_eval_data_seed(
            default_eval_data_seed,
            key=default_key,
        )
    resolved_mode = "canonical" if resolved_seed is None else str(int(resolved_seed))
    return resolved_seed, resolved_mode


def require_meta_analysis_eval_data_seed_scope_tags(
    tags: Mapping[str, Any],
    *,
    run_id: str,
) -> tuple[Optional[int], str]:
    """Extract meta-analysis eval-seed scope tags.

    Canonical meta-analysis runs may carry an optional ``eval_data_seed`` tag from
    older logged metadata, but the persisted ``eval_data_seed_mode`` remains the
    authoritative scope signal.
    """
    if tags is None:
        raise ValueError(
            f"Run {run_id} is missing tags required for meta-analysis eval-seed scope."
        )
    mode_raw = tags.get("eval_data_seed_mode")
    if mode_raw is None or not str(mode_raw).strip():
        raise ValueError(f"Run {run_id} eval_data_seed_mode is required.")
    mode = str(mode_raw).strip()
    if mode == "canonical":
        parse_eval_data_seed(
            tags.get("eval_data_seed"),
            key="eval_data_seed",
        )
        return None, "canonical"

    try:
        expected_seed = parse_eval_data_seed(
            mode,
            key="eval_data_seed_mode",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Run {run_id} has invalid eval_data_seed_mode tag {mode!r}; "
            "expected 'canonical' or an integer seed."
        ) from exc
    if expected_seed is None:
        raise ValueError(
            f"Run {run_id} has invalid eval_data_seed_mode tag {mode!r}; "
            "expected 'canonical' or an integer seed."
        )

    actual_seed = parse_eval_data_seed(
        tags.get("eval_data_seed"),
        key="eval_data_seed",
    )
    if actual_seed is None:
        raise ValueError(
            f"Run {run_id} has eval_data_seed_mode={mode!r} but is missing "
            "required eval_data_seed tag."
        )
    if actual_seed != expected_seed:
        raise ValueError(
            f"Run {run_id} has eval_data_seed_mode={mode!r} but eval_data_seed="
            f"{actual_seed}; expected {expected_seed}."
        )
    return int(actual_seed), str(int(expected_seed))


def parse_bootstrap_ci_resamples(
    value: Any,
    *,
    key: str = "bootstrap_ci_resamples",
) -> Optional[int]:
    """Parse the configured bootstrap-resample count."""
    parsed = parse_value(value, int, key=key)
    if parsed is None:
        return None
    if parsed <= 0:
        raise ValueError(f"{key} must be > 0, got {parsed}.")
    return parsed


def parse_bootstrap_ci_confidence_level(
    value: Any,
    *,
    key: str = "bootstrap_ci_confidence_level",
) -> Optional[float]:
    """Parse the configured bootstrap confidence level."""
    parsed = parse_value(value, float, key=key)
    if parsed is None:
        return None
    if not (0.0 < float(parsed) < 1.0):
        raise ValueError(f"{key} must satisfy 0 < {key} < 1; got {float(parsed):g}.")
    return float(parsed)


def _build_bootstrap_ci_tag_payload(
    context: Mapping[str, Any],
    *,
    expected_semantics: str,
    require_seed: bool = True,
    context_name: str = "bootstrap_ci_context",
) -> dict[str, str]:
    """Normalize bootstrap-CI context for MLflow tag logging."""
    if not isinstance(context, Mapping):
        raise ValueError(f"{context_name} must be a mapping.")
    semantics_raw = context.get("bootstrap_ci_semantics")
    if semantics_raw is None or not str(semantics_raw).strip():
        raise ValueError(f"{context_name}.bootstrap_ci_semantics is required.")
    semantics = str(semantics_raw).strip()
    if semantics != expected_semantics:
        raise ValueError(
            f"{context_name}.bootstrap_ci_semantics must be "
            f"{expected_semantics!r}; got {semantics!r}."
        )
    resamples = parse_bootstrap_ci_resamples(
        context.get("bootstrap_ci_resamples"),
        key=f"{context_name}.bootstrap_ci_resamples",
    )
    if resamples is None:
        raise ValueError(f"{context_name}.bootstrap_ci_resamples is required.")
    confidence_level = parse_bootstrap_ci_confidence_level(
        context.get("bootstrap_ci_confidence_level"),
        key=f"{context_name}.bootstrap_ci_confidence_level",
    )
    if confidence_level is None:
        raise ValueError(f"{context_name}.bootstrap_ci_confidence_level is required.")
    payload = {
        "bootstrap_ci_semantics": semantics,
        "bootstrap_ci_resamples": str(int(resamples)),
        "bootstrap_ci_confidence_level": str(float(confidence_level)),
    }
    if require_seed:
        seed = parse_value(
            context.get("bootstrap_ci_seed"),
            int,
            key=f"{context_name}.bootstrap_ci_seed",
        )
        if seed is None:
            raise ValueError(f"{context_name}.bootstrap_ci_seed is required.")
        payload["bootstrap_ci_seed"] = str(int(seed))
    return payload


def build_shared_anchor_bootstrap_ci_tag_payload(
    context: Mapping[str, Any],
    *,
    require_seed: bool = True,
    context_name: str = "shared_anchor_bootstrap_ci_context",
) -> dict[str, str]:
    """Normalize shared-anchor bootstrap-CI context for MLflow tag logging."""
    return _build_bootstrap_ci_tag_payload(
        context,
        expected_semantics=SHARED_ANCHOR_BOOTSTRAP_CI_SEMANTICS,
        require_seed=require_seed,
        context_name=context_name,
    )


def resolve_effective_eval_data_seed(
    eval_data_seed: Any,
    *,
    canonical_seed_data: Any,
    eval_key: str = "eval_data_seed",
    canonical_key: str = "seed_data",
) -> int:
    """Resolve the persisted evaluation data seed from override or data seed."""
    override = parse_eval_data_seed(eval_data_seed, key=eval_key)
    if override is not None:
        return override
    canonical = parse_value(canonical_seed_data, int, key=canonical_key)
    if canonical is None:
        raise ValueError(f"{canonical_key} is required for eval_data_seed resolution.")
    return canonical


def require_eval_data_seed_tag(
    tags: Mapping[str, Any],
    *,
    run_id: str,
) -> int:
    """Require the integer ``eval_data_seed`` tag for evaluated runs."""
    if tags is None:
        raise ValueError(
            f"Run {run_id} is missing tags required for eval_data_seed parsing."
        )
    parsed = parse_eval_data_seed(tags.get("eval_data_seed"), key="eval_data_seed")
    if parsed is None:
        raise ValueError(f"Run {run_id} is missing required eval_data_seed tag.")
    return parsed


def build_seeded_eval_input_artifact_prefix(
    *,
    test_metric: Any,
    eval_data_seed: Any,
) -> str:
    """Build the canonical seed-scoped evaluation-input prefix."""
    metric = parse_optional_nonempty_string(
        test_metric,
        key="test_metric",
        context="Evaluation input artifact path",
        disallow_none_token=True,
    )
    if metric is None:
        raise ValueError("Evaluation input artifact path requires test_metric.")
    seed = parse_eval_data_seed(eval_data_seed, key="eval_data_seed")
    if seed is None:
        raise ValueError("Evaluation input artifact path requires eval_data_seed.")
    return f"robustness_inputs/{metric}/seed_data_{seed}"


def build_seeded_degradation_artifact_prefix(
    *,
    test_metric: Any,
    eval_data_seed: Any,
) -> str:
    """Build the literal `robustness/degradation/...` artifact prefix."""
    metric = parse_optional_nonempty_string(
        test_metric,
        key="test_metric",
        context="`degradation` robustness artifact path",
        disallow_none_token=True,
    )
    if metric is None:
        raise ValueError("`degradation` robustness artifact path requires test_metric.")
    seed = parse_eval_data_seed(eval_data_seed, key="eval_data_seed")
    if seed is None:
        raise ValueError("`degradation` robustness artifact path requires eval_data_seed.")
    return f"robustness/degradation/{metric}/seed_data_{seed}"


def require_shared_anchor_bootstrap_ci_context_from_args(
    args: Any,
    *,
    eval_data_seed: Any,
    test_metric: Any,
    context: str = "args",
) -> dict[str, Any]:
    """Build required bootstrap-CI context for the degradation benchmark."""
    parsed_eval_data_seed = parse_eval_data_seed(
        eval_data_seed,
        key=f"{context}.eval_data_seed",
    )
    if parsed_eval_data_seed is None:
        raise ValueError(f"{context}.eval_data_seed is required for bootstrap CI.")
    parsed_test_metric = parse_optional_nonempty_string(
        test_metric,
        key=f"{context}.test_metric",
        context="Shared-anchor bootstrap CI context",
        disallow_none_token=True,
    )
    if parsed_test_metric is None:
        raise ValueError(f"{context}.test_metric is required for bootstrap CI.")
    resamples = parse_bootstrap_ci_resamples(
        require_namespace_value(args, key="bootstrap_ci_resamples", context=context),
        key=f"{context}.bootstrap_ci_resamples",
    )
    if resamples is None:
        raise ValueError(f"{context}.bootstrap_ci_resamples is required.")
    confidence_level = parse_bootstrap_ci_confidence_level(
        require_namespace_value(args, key="bootstrap_ci_confidence_level", context=context),
        key=f"{context}.bootstrap_ci_confidence_level",
    )
    if confidence_level is None:
        raise ValueError(f"{context}.bootstrap_ci_confidence_level is required.")
    seed = derive_seed(
        parsed_eval_data_seed,
        build_shared_anchor_bootstrap_ci_seed_key(parsed_test_metric),
    )
    return {
        "bootstrap_ci_semantics": SHARED_ANCHOR_BOOTSTRAP_CI_SEMANTICS,
        "bootstrap_ci_resamples": int(resamples),
        "bootstrap_ci_confidence_level": float(confidence_level),
        "bootstrap_ci_seed": int(seed),
    }


def _require_bootstrap_ci_context_tags(
    tags: Mapping[str, Any],
    *,
    run_id: str,
    expected_semantics: str,
    expected_context: Mapping[str, Any] | None = None,
    require_seed: bool = True,
) -> dict[str, Any]:
    """Extract and validate bootstrap-CI provenance tags."""
    if tags is None:
        raise ValueError(
            f"Run {run_id} is missing tags required for bootstrap CI parsing."
        )
    semantics_raw = tags.get("bootstrap_ci_semantics")
    if semantics_raw is None or not str(semantics_raw).strip():
        raise ValueError(
            f"Run {run_id} is missing required bootstrap_ci_semantics tag."
        )
    semantics = str(semantics_raw).strip()
    if semantics != expected_semantics:
        raise ValueError(
            f"Run {run_id} has unsupported bootstrap_ci_semantics={semantics!r}; "
            f"expected {expected_semantics!r}."
        )
    resamples = parse_bootstrap_ci_resamples(
        tags.get("bootstrap_ci_resamples"),
        key="bootstrap_ci_resamples",
    )
    if resamples is None:
        raise ValueError(
            f"Run {run_id} is missing required bootstrap_ci_resamples tag."
        )
    confidence_level = parse_bootstrap_ci_confidence_level(
        tags.get("bootstrap_ci_confidence_level"),
        key="bootstrap_ci_confidence_level",
    )
    if confidence_level is None:
        raise ValueError(
            f"Run {run_id} is missing required bootstrap_ci_confidence_level tag."
        )
    context_payload: dict[str, Any] = {
        "bootstrap_ci_semantics": semantics,
        "bootstrap_ci_resamples": int(resamples),
        "bootstrap_ci_confidence_level": float(confidence_level),
    }
    if require_seed:
        seed = parse_value(
            tags.get("bootstrap_ci_seed"),
            int,
            key="bootstrap_ci_seed",
        )
        if seed is None:
            raise ValueError(
                f"Run {run_id} is missing required bootstrap_ci_seed tag."
            )
        context_payload["bootstrap_ci_seed"] = int(seed)
    if expected_context:
        if not shared_anchor_bootstrap_ci_context_matches(
            context_payload,
            expected_context=expected_context,
            require_seed=require_seed,
        ):
            raise ValueError(
                f"Run {run_id} does not match the expected bootstrap-CI context."
            )
    return context_payload


def require_shared_anchor_bootstrap_ci_context_tags(
    tags: Mapping[str, Any],
    *,
    run_id: str,
    expected_context: Mapping[str, Any] | None = None,
    require_seed: bool = True,
) -> dict[str, Any]:
    """Extract and validate shared-anchor bootstrap-CI provenance tags."""
    return _require_bootstrap_ci_context_tags(
        tags,
        run_id=run_id,
        expected_semantics=SHARED_ANCHOR_BOOTSTRAP_CI_SEMANTICS,
        expected_context=expected_context,
        require_seed=require_seed,
    )


def shared_anchor_bootstrap_ci_context_matches(
    context: Mapping[str, Any],
    *,
    expected_context: Mapping[str, Any],
    require_seed: bool = True,
) -> bool:
    """Return whether a parsed shared-anchor bootstrap-CI context matches expectations."""
    if not isinstance(context, Mapping):
        raise ValueError(
            "shared-anchor bootstrap-CI context comparison requires a mapping."
        )
    if not isinstance(expected_context, Mapping):
        raise ValueError(
            "expected shared-anchor bootstrap-CI context must be a mapping."
        )
    if not expected_context:
        return True
    for key, expected_value in expected_context.items():
        actual_value = context.get(key)
        if key == "bootstrap_ci_semantics":
            expected_semantics = parse_required_choice(
                expected_value,
                key="expected bootstrap_ci_semantics",
                allowed=(SHARED_ANCHOR_BOOTSTRAP_CI_SEMANTICS,),
            )
            if actual_value != expected_semantics:
                return False
            continue
        if key == "bootstrap_ci_resamples":
            expected_resamples = parse_bootstrap_ci_resamples(
                expected_value,
                key="expected bootstrap_ci_resamples",
            )
            if expected_resamples is None:
                raise ValueError("expected bootstrap_ci_resamples is required.")
            if actual_value != int(expected_resamples):
                return False
            continue
        if key == "bootstrap_ci_confidence_level":
            expected_confidence = parse_bootstrap_ci_confidence_level(
                expected_value,
                key="expected bootstrap_ci_confidence_level",
            )
            if expected_confidence is None:
                raise ValueError(
                    "expected bootstrap_ci_confidence_level is required."
                )
            if actual_value is None or not math.isclose(
                float(actual_value),
                float(expected_confidence),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return False
            continue
        if key == "bootstrap_ci_seed":
            if not require_seed:
                raise ValueError(
                    "bootstrap_ci_seed expectation requires require_seed=True."
                )
            expected_seed = parse_value(
                expected_value,
                int,
                key="expected bootstrap_ci_seed",
            )
            if expected_seed is None:
                raise ValueError("expected bootstrap_ci_seed is required.")
            if actual_value != int(expected_seed):
                return False
            continue
        raise ValueError(
            f"Unsupported shared-anchor bootstrap-CI comparison key '{key}'."
        )
    return True


def require_degradation_eval_context_from_args(
    args: Any,
    *,
    eval_data_seed: Any,
    context: str = "args",
) -> dict[str, Any]:
    """Build the degradation evaluation identity from runtime args/config."""
    if args is None:
        raise ValueError(f"{context} is required for degradation evaluation context.")
    parsed_eval_data_seed = parse_eval_data_seed(
        eval_data_seed,
        key=f"{context}.eval_data_seed",
    )
    if parsed_eval_data_seed is None:
        raise ValueError(
            f"{context}.eval_data_seed is required for degradation evaluation context."
        )
    test_metric = require_namespace_nonempty_string(
        args,
        key="test_metric",
        context=context,
        disallow_none_token=True,
    )
    n_test_samples = parse_required_positive_int(
        require_namespace_value(args, key="n_test_samples", context=context),
        key=f"{context}.n_test_samples",
    )
    perturbation_channel_fraction_max = parse_perturbation_channel_fraction_max(
        require_namespace_value(
            args,
            key="perturbation_channel_fraction_max",
            context=context,
        ),
        key=f"{context}.perturbation_channel_fraction_max",
    )
    perturbation_scenarios = parse_perturbation_scenarios(
        require_namespace_value(args, key="perturbation_scenarios", context=context),
        key=f"{context}.perturbation_scenarios",
    )
    strict_iid = require_namespace_bool(args, key="strict_iid", context=context)
    if strict_iid:
        raise ValueError(
            f"{context}.strict_iid must be false for degradation evaluation. "
            "Strict-IID sampling changes test-anchor coverage and cannot define the "
            "canonical reusable robustness results."
        )
    perturbation_idx_name_map = {
        idx: name for idx, name in enumerate(perturbation_scenarios)
    }
    return {
        "robustness_scoring_semantics": DEGRADATION_SCORING_SEMANTICS,
        "test_metric": test_metric,
        "eval_data_seed": int(parsed_eval_data_seed),
        "n_test_samples": int(n_test_samples),
        "perturbation_channel_fraction_max": float(perturbation_channel_fraction_max),
        "perturbation_scenarios_signature": build_perturbation_scenarios_signature(
            perturbation_scenarios
        ),
        "perturbation_scenarios_count": len(perturbation_scenarios),
        "perturbation_idx_name_map": perturbation_idx_name_map,
    }


def require_selection_perturbation_context_from_args(
    args: Any,
    *,
    context: str = "args",
) -> dict[str, Any]:
    """Build the perturbed-validation selector context from runtime args/config."""
    if args is None:
        raise ValueError(
            f"{context} is required for perturbed validation selection context."
        )
    perturbation_channel_fraction_max = parse_perturbation_channel_fraction_max(
        require_namespace_value(
            args,
            key="perturbation_channel_fraction_max",
            context=context,
        ),
        key=f"{context}.perturbation_channel_fraction_max",
    )
    perturbation_scenarios = parse_perturbation_scenarios(
        require_namespace_value(args, key="perturbation_scenarios", context=context),
        key=f"{context}.perturbation_scenarios",
    )
    return {
        "selection_metric_semantics": SELECTION_METRIC_SEMANTICS,
        "selection_perturbation_channel_fraction_max": float(
            perturbation_channel_fraction_max
        ),
        "selection_perturbation_scenarios_signature": build_perturbation_scenarios_signature(
            perturbation_scenarios
        ),
    }


def _parse_selection_perturbation_context(
    context: Mapping[str, Any],
    *,
    context_name: str,
) -> dict[str, Any]:
    """Parse the perturbed-selection context into canonical native values."""
    if not isinstance(context, Mapping):
        raise ValueError(f"{context_name} must be a mapping.")
    semantics_raw = context.get("selection_metric_semantics")
    if semantics_raw is None or not str(semantics_raw).strip():
        raise ValueError(f"{context_name}.selection_metric_semantics is required.")
    semantics = str(semantics_raw).strip()
    if semantics != SELECTION_METRIC_SEMANTICS:
        raise ValueError(
            f"{context_name}.selection_metric_semantics must be "
            f"{SELECTION_METRIC_SEMANTICS!r}; got {semantics!r}."
        )
    return {
        "selection_metric_semantics": semantics,
        "selection_perturbation_channel_fraction_max": parse_perturbation_channel_fraction_max(
            context.get("selection_perturbation_channel_fraction_max"),
            key=f"{context_name}.selection_perturbation_channel_fraction_max",
        ),
        "selection_perturbation_scenarios_signature": parse_perturbation_scenarios_signature(
            context.get("selection_perturbation_scenarios_signature"),
            key=f"{context_name}.selection_perturbation_scenarios_signature",
        ),
    }


def build_selection_perturbation_context_tag_payload(
    context: Mapping[str, Any],
    *,
    context_name: str = "selection_perturbation_context",
) -> dict[str, str]:
    """Normalize perturbed-validation selector identity for MLflow tag logging."""
    parsed_context = _parse_selection_perturbation_context(
        context,
        context_name=context_name,
    )
    return {
        "selection_metric_semantics": parsed_context["selection_metric_semantics"],
        "selection_perturbation_channel_fraction_max": str(
            float(parsed_context["selection_perturbation_channel_fraction_max"])
        ),
        "selection_perturbation_scenarios_signature": parsed_context[
            "selection_perturbation_scenarios_signature"
        ],
    }


def selection_perturbation_context_matches(
    context: Mapping[str, Any],
    *,
    expected_context: Mapping[str, Any],
) -> bool:
    """Return whether a parsed perturbed-selection context matches expectations."""
    if not isinstance(context, Mapping):
        raise ValueError("selection perturbation context comparison requires a mapping.")
    if not isinstance(expected_context, Mapping):
        raise ValueError("expected selection perturbation context must be a mapping.")
    if not expected_context:
        return True
    for key, expected_value in expected_context.items():
        if key == "selection_perturbation_channel_fraction_max":
            expected_float = parse_perturbation_channel_fraction_max(
                expected_value,
                key="expected selection_perturbation_channel_fraction_max",
            )
            actual_value = context.get(key)
            if actual_value is None or not math.isclose(
                float(actual_value),
                expected_float,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return False
            continue
        if key == "selection_perturbation_scenarios_signature":
            expected_signature = parse_perturbation_scenarios_signature(
                expected_value,
                key="expected selection_perturbation_scenarios_signature",
            )
            if context.get(key) != expected_signature:
                return False
            continue
        if key == "selection_metric_semantics":
            if str(context.get(key)).strip() != str(expected_value).strip():
                return False
            continue
        raise ValueError(f"Unsupported selection perturbation context key '{key}'.")
    return True


def require_selection_perturbation_context_tags(
    tags: Mapping[str, Any],
    *,
    run_id: str,
    expected_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract and validate required perturbed-validation selector-context tags."""
    if tags is None:
        raise SelectionPerturbationContextNotReadyError(
            f"Run {run_id} is missing tags required for perturbed validation selection context."
        )
    try:
        context = _parse_selection_perturbation_context(
            tags,
            context_name=f"Run {run_id}",
        )
    except ValueError as exc:
        raise SelectionPerturbationContextNotReadyError(str(exc)) from exc
    if expected_context:
        if not selection_perturbation_context_matches(
            context,
            expected_context=expected_context,
        ):
            raise SelectionPerturbationContextNotReadyError(
                f"Run {run_id} does not match the expected perturbed selection context."
            )
    return context


def _parse_winner_selection_provenance(
    context: Mapping[str, Any],
    *,
    context_name: str,
) -> dict[str, Any]:
    """Parse the persisted winner-selection provenance into canonical values."""
    if not isinstance(context, Mapping):
        raise ValueError(f"{context_name} must be a mapping.")
    selection_mode = parse_improvement_selection_mode(
        context.get("winner_selection_mode"),
        key=f"{context_name}.winner_selection_mode",
    )
    selection_metric_name = parse_required_nonempty_string(
        context.get("winner_selection_metric_name"),
        key=f"{context_name}.winner_selection_metric_name",
    )
    parsed: dict[str, Any] = {
        "winner_selection_mode": selection_mode,
        "winner_selection_metric_name": selection_metric_name,
    }
    if selection_mode == "clean":
        return parsed
    semantics_raw = context.get("winner_selection_metric_semantics")
    if semantics_raw is None or not str(semantics_raw).strip():
        raise ValueError(f"{context_name}.winner_selection_metric_semantics is required.")
    semantics = str(semantics_raw).strip()
    if semantics != SELECTION_METRIC_SEMANTICS:
        raise ValueError(
            f"{context_name}.winner_selection_metric_semantics must be "
            f"{SELECTION_METRIC_SEMANTICS!r}; got {semantics!r}."
        )
    parsed.update(
        {
            "winner_selection_metric_semantics": semantics,
            "winner_selection_perturbation_channel_fraction_max": parse_perturbation_channel_fraction_max(
                context.get("winner_selection_perturbation_channel_fraction_max"),
                key=(
                    f"{context_name}."
                    "winner_selection_perturbation_channel_fraction_max"
                ),
            ),
            "winner_selection_perturbation_scenarios_signature": parse_perturbation_scenarios_signature(
                context.get("winner_selection_perturbation_scenarios_signature"),
                key=(
                    f"{context_name}."
                    "winner_selection_perturbation_scenarios_signature"
                ),
            ),
        }
    )
    return parsed


def build_winner_selection_provenance_tag_payload(
    context: Mapping[str, Any],
    *,
    context_name: str = "winner_selection_provenance",
) -> dict[str, str]:
    """Normalize winner-selection provenance for MLflow tag logging."""
    parsed_context = _parse_winner_selection_provenance(
        context,
        context_name=context_name,
    )
    tag_payload = {
        "winner_selection_mode": parsed_context["winner_selection_mode"],
        "winner_selection_metric_name": parsed_context["winner_selection_metric_name"],
    }
    if parsed_context["winner_selection_mode"] == "clean":
        return tag_payload
    tag_payload.update(
        {
            "winner_selection_metric_semantics": parsed_context[
                "winner_selection_metric_semantics"
            ],
            "winner_selection_perturbation_channel_fraction_max": str(
                float(
                    parsed_context[
                        "winner_selection_perturbation_channel_fraction_max"
                    ]
                )
            ),
            "winner_selection_perturbation_scenarios_signature": parsed_context[
                "winner_selection_perturbation_scenarios_signature"
            ],
        }
    )
    return tag_payload


def winner_selection_provenance_matches(
    context: Mapping[str, Any],
    *,
    expected_context: Mapping[str, Any],
) -> bool:
    """Return whether parsed winner-selection provenance matches expectations."""
    if not isinstance(context, Mapping):
        raise ValueError("winner selection provenance comparison requires a mapping.")
    if not isinstance(expected_context, Mapping):
        raise ValueError("expected winner selection provenance must be a mapping.")
    if not expected_context:
        return True
    for key, expected_value in expected_context.items():
        if key == "winner_selection_perturbation_channel_fraction_max":
            expected_float = parse_perturbation_channel_fraction_max(
                expected_value,
                key="expected winner_selection_perturbation_channel_fraction_max",
            )
            actual_value = context.get(key)
            if actual_value is None or not math.isclose(
                float(actual_value),
                expected_float,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return False
            continue
        if key == "winner_selection_perturbation_scenarios_signature":
            expected_signature = parse_perturbation_scenarios_signature(
                expected_value,
                key="expected winner_selection_perturbation_scenarios_signature",
            )
            if context.get(key) != expected_signature:
                return False
            continue
        if key in {
            "winner_selection_mode",
            "winner_selection_metric_name",
            "winner_selection_metric_semantics",
        }:
            actual_value = context.get(key)
            if actual_value is None or str(actual_value).strip() != str(expected_value).strip():
                return False
            continue
        raise ValueError(f"Unsupported winner selection provenance key '{key}'.")
    return True


def require_winner_selection_provenance_tags(
    tags: Mapping[str, Any],
    *,
    run_id: str,
    expected_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract and validate required winner-selection provenance tags."""
    if tags is None:
        raise ValueError(
            f"Run {run_id} is missing tags required for winner-selection provenance."
        )
    context = _parse_winner_selection_provenance(
        tags,
        context_name=f"Run {run_id}",
    )
    if expected_context:
        if not winner_selection_provenance_matches(
            context,
            expected_context=expected_context,
        ):
            raise ValueError(
                f"Run {run_id} does not match the expected winner-selection provenance."
            )
    return context


def build_degradation_eval_context_tag_payload(
    context: Mapping[str, Any],
    *,
    context_name: str = "degradation_eval_context",
    include_eval_data_seed: bool = True,
    validate_optional_eval_data_seed: bool = True,
) -> dict[str, str]:
    """Normalize degradation evaluation identity for MLflow tag logging."""
    if not isinstance(context, Mapping):
        raise ValueError(f"{context_name} must be a mapping.")
    semantics_raw = context.get("robustness_scoring_semantics")
    if semantics_raw is None or not str(semantics_raw).strip():
        raise ValueError(f"{context_name}.robustness_scoring_semantics is required.")
    semantics = str(semantics_raw).strip()
    if semantics != DEGRADATION_SCORING_SEMANTICS:
        raise ValueError(
            f"{context_name}.robustness_scoring_semantics must be "
            f"{DEGRADATION_SCORING_SEMANTICS!r}; got {semantics!r}."
        )
    test_metric = parse_optional_nonempty_string(
        context.get("test_metric"),
        key=f"{context_name}.test_metric",
        context="Degradation evaluation context",
        disallow_none_token=True,
    )
    if test_metric is None:
        raise ValueError(f"{context_name}.test_metric is required.")
    eval_seed = None
    raw_eval_seed = context.get("eval_data_seed")
    has_optional_eval_seed = raw_eval_seed is not None and str(raw_eval_seed).strip() != ""
    if include_eval_data_seed or (
        validate_optional_eval_data_seed and has_optional_eval_seed
    ):
        eval_seed = parse_eval_data_seed(
            raw_eval_seed,
            key=f"{context_name}.eval_data_seed",
        )
        if include_eval_data_seed and eval_seed is None:
            raise ValueError(f"{context_name}.eval_data_seed is required.")
    n_test_samples = parse_required_positive_int(
        context.get("n_test_samples"),
        key=f"{context_name}.n_test_samples",
    )
    perturbation_channel_fraction_max = parse_perturbation_channel_fraction_max(
        context.get("perturbation_channel_fraction_max"),
        key=f"{context_name}.perturbation_channel_fraction_max",
    )
    perturbation_scenarios_signature = parse_perturbation_scenarios_signature(
        context.get("perturbation_scenarios_signature"),
        key=f"{context_name}.perturbation_scenarios_signature",
    )
    perturbation_scenarios_count = parse_required_positive_int(
        context.get("perturbation_scenarios_count"),
        key=f"{context_name}.perturbation_scenarios_count",
    )
    perturbation_idx_name_map = require_order_sensitive_perturbation_idx_name_map(
        context.get("perturbation_idx_name_map"),
        scenarios_signature=perturbation_scenarios_signature,
        scenarios_count=perturbation_scenarios_count,
        key=f"{context_name}.perturbation_idx_name_map",
        signature_key=f"{context_name}.perturbation_scenarios_signature",
        count_key=f"{context_name}.perturbation_scenarios_count",
    )
    signature_names = parse_perturbation_scenarios_from_signature(
        perturbation_scenarios_signature,
        key=f"{context_name}.perturbation_scenarios_signature",
    )
    if perturbation_scenarios_count != len(signature_names):
        raise ValueError(
            f"{context_name}.perturbation_scenarios_count={perturbation_scenarios_count} "
            f"does not match perturbation_scenarios_signature length {len(signature_names)}."
        )
    payload = {
        "robustness_scoring_semantics": semantics,
        "test_metric": test_metric,
        "n_test_samples": str(int(n_test_samples)),
        "perturbation_channel_fraction_max": str(float(perturbation_channel_fraction_max)),
        "perturbation_scenarios_signature": perturbation_scenarios_signature,
        "perturbation_scenarios_count": str(int(perturbation_scenarios_count)),
        "perturbation_idx_name_map": build_perturbation_idx_name_map(
            perturbation_idx_name_map,
            key=f"{context_name}.perturbation_idx_name_map",
        ),
    }
    if eval_seed is not None:
        payload["eval_data_seed"] = str(int(eval_seed))
    return payload


def require_degradation_eval_context_tags(
    tags: Mapping[str, Any],
    *,
    run_id: str,
    expected_test_metric: str | None = None,
    expected_context: Mapping[str, Any] | None = None,
    require_eval_data_seed: bool = True,
) -> dict[str, Any]:
    """Extract and validate required degradation evaluation identity tags."""
    if tags is None:
        raise ValueError(
            f"Run {run_id} is missing tags required for degradation evaluation context."
        )
    tag_payload = build_degradation_eval_context_tag_payload(
        tags,
        context_name=f"Run {run_id}",
        include_eval_data_seed=require_eval_data_seed,
    )
    context = {
        "robustness_scoring_semantics": tag_payload["robustness_scoring_semantics"],
        "test_metric": tag_payload["test_metric"],
        "n_test_samples": parse_required_positive_int(
            tag_payload["n_test_samples"],
            key="n_test_samples",
        ),
        "perturbation_channel_fraction_max": parse_perturbation_channel_fraction_max(
            tag_payload["perturbation_channel_fraction_max"],
            key="perturbation_channel_fraction_max",
        ),
        "perturbation_scenarios_signature": parse_perturbation_scenarios_signature(
            tag_payload["perturbation_scenarios_signature"],
            key="perturbation_scenarios_signature",
        ),
        "perturbation_scenarios_count": parse_required_positive_int(
            tag_payload["perturbation_scenarios_count"],
            key="perturbation_scenarios_count",
        ),
        "perturbation_idx_name_map": parse_perturbation_idx_name_map(
            tag_payload["perturbation_idx_name_map"],
            key="perturbation_idx_name_map",
        ),
    }
    if require_eval_data_seed or "eval_data_seed" in tag_payload:
        context["eval_data_seed"] = parse_eval_data_seed(
            tag_payload["eval_data_seed"],
            key="eval_data_seed",
        )
    if expected_test_metric is not None and context["test_metric"] != str(expected_test_metric):
        raise ValueError(
            f"Run {run_id} has test_metric tag '{context['test_metric']}' but expected "
            f"'{expected_test_metric}'."
        )
    if expected_context:
        if not degradation_eval_context_matches(
            context,
            expected_context=expected_context,
        ):
            raise ValueError(
                f"Run {run_id} does not match the expected degradation evaluation context."
            )
    return context


def degradation_eval_context_matches(
    context: Mapping[str, Any],
    *,
    expected_context: Mapping[str, Any],
) -> bool:
    """Return whether a parsed degradation evaluation context matches expectations."""
    if not isinstance(context, Mapping):
        raise ValueError("degradation evaluation context comparison requires a mapping.")
    if not isinstance(expected_context, Mapping):
        raise ValueError("expected degradation evaluation context must be a mapping.")
    if not expected_context:
        return True
    for key, expected_value in expected_context.items():
        if key == "perturbation_channel_fraction_max":
            expected_float = parse_perturbation_channel_fraction_max(
                expected_value,
                key="expected perturbation_channel_fraction_max",
            )
            actual_value = context.get(key)
            if actual_value is None or not math.isclose(
                float(actual_value),
                expected_float,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return False
            continue
        if key == "eval_data_seed":
            expected_seed = parse_eval_data_seed(
                expected_value,
                key="expected eval_data_seed",
            )
            if expected_seed is None:
                raise ValueError("expected eval_data_seed is required.")
            if context.get(key) != expected_seed:
                return False
            continue
        if key in {"n_test_samples", "perturbation_scenarios_count"}:
            expected_int = parse_required_positive_int(
                expected_value,
                key=f"expected {key}",
            )
            if context.get(key) != expected_int:
                return False
            continue
        if key == "perturbation_idx_name_map":
            expected_idx_name_map = require_order_sensitive_perturbation_idx_name_map(
                expected_value,
                scenarios_signature=context["perturbation_scenarios_signature"],
                scenarios_count=context["perturbation_scenarios_count"],
                key="expected perturbation_idx_name_map",
                signature_key="expected perturbation_scenarios_signature",
                count_key="expected perturbation_scenarios_count",
            )
            if context.get(key) != expected_idx_name_map:
                return False
            continue
        if key == "robustness_scoring_semantics":
            expected_semantics = parse_required_choice(
                expected_value,
                key="expected robustness_scoring_semantics",
                allowed=(DEGRADATION_SCORING_SEMANTICS,),
            )
            if context.get(key) != expected_semantics:
                return False
            continue
        if key == "test_metric":
            expected_metric = parse_optional_nonempty_string(
                expected_value,
                key="expected test_metric",
                context="Degradation evaluation context comparison",
                disallow_none_token=True,
            )
            if expected_metric is None:
                raise ValueError("expected test_metric is required.")
            if context.get(key) != expected_metric:
                return False
            continue
        if key == "perturbation_scenarios_signature":
            expected_signature = parse_perturbation_scenarios_signature(
                expected_value,
                key="expected perturbation_scenarios_signature",
            )
            if context.get(key) != expected_signature:
                return False
            continue
        raise ValueError(
            f"Unsupported degradation evaluation-context comparison key '{key}'."
        )
    return True


def degradation_n_test_samples_meet_policy(
    n_test_samples: Any,
    *,
    expected_n_test_samples: Any,
    full_coverage: bool,
) -> bool:
    """Return whether a run's n_test_samples exactly matches the reuse policy."""
    actual = parse_required_positive_int(
        n_test_samples,
        key="n_test_samples",
    )
    expected = parse_required_positive_int(
        expected_n_test_samples,
        key="expected n_test_samples",
    )
    if actual == expected:
        return True
    return False


def build_export_manifest_eval_context(
    defaults: Mapping[str, Any],
    *,
    n_test_samples: Any,
    context_name: str = "export_manifest_eval_context",
) -> dict[str, Any]:
    """Normalize the export-manifest degradation context from resolved defaults."""
    if not isinstance(defaults, Mapping):
        raise ValueError(f"{context_name} defaults must be a mapping.")

    def _require_defaults_key(key: str) -> Any:
        if key not in defaults:
            raise ValueError(
                f"{context_name} defaults are missing required '{key}'."
            )
        return defaults[key]

    test_metric = parse_optional_nonempty_string(
        _require_defaults_key("TEST_METRIC"),
        key=f"{context_name}.TEST_METRIC",
        context="Export manifest eval context",
        disallow_none_token=True,
    )
    if test_metric is None:
        raise ValueError(f"{context_name}.TEST_METRIC is required.")

    bootstrap_ci_resamples = parse_bootstrap_ci_resamples(
        _require_defaults_key("BOOTSTRAP_CI_RESAMPLES"),
        key=f"{context_name}.BOOTSTRAP_CI_RESAMPLES",
    )
    if bootstrap_ci_resamples is None:
        raise ValueError(f"{context_name}.BOOTSTRAP_CI_RESAMPLES is required.")

    bootstrap_ci_confidence_level = parse_bootstrap_ci_confidence_level(
        _require_defaults_key("BOOTSTRAP_CI_CONFIDENCE_LEVEL"),
        key=f"{context_name}.BOOTSTRAP_CI_CONFIDENCE_LEVEL",
    )
    if bootstrap_ci_confidence_level is None:
        raise ValueError(
            f"{context_name}.BOOTSTRAP_CI_CONFIDENCE_LEVEL is required."
        )

    return {
        "robustness_scoring_semantics": DEGRADATION_SCORING_SEMANTICS,
        "test_metric": test_metric,
        "n_test_samples": parse_required_positive_int(
            n_test_samples,
            key=f"{context_name}.n_test_samples",
        ),
        "eval_data_seed": parse_eval_data_seed(
            _require_defaults_key("EVAL_DATA_SEED"),
            key=f"{context_name}.EVAL_DATA_SEED",
        ),
        "perturbation_channel_fraction_max": parse_perturbation_channel_fraction_max(
            _require_defaults_key("PERTURBATION_CHANNEL_FRACTION_MAX"),
            key=f"{context_name}.PERTURBATION_CHANNEL_FRACTION_MAX",
        ),
        "perturbation_scenarios": list(
            parse_perturbation_scenarios(
                _require_defaults_key("PERTURBATION_SCENARIOS"),
                key=f"{context_name}.PERTURBATION_SCENARIOS",
            )
        ),
        "bootstrap_ci_resamples": int(bootstrap_ci_resamples),
        "bootstrap_ci_confidence_level": float(bootstrap_ci_confidence_level),
    }


def require_tsmixer_hparams(values: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize required TSMixer hyperparameters."""
    if not isinstance(values, Mapping):
        raise ValueError("TSMixer hyperparameters must be provided as a mapping.")

    def _parse(key: str, expected_type: type) -> Any:
        raw = values.get(key)
        try:
            return parse_value(raw, expected_type, key=key)
        except ValueError as exc:
            raise ValueError(f"invalid {key} for TSMixer: {exc}") from exc

    def _require_positive_int(key: str) -> int:
        val = _parse(key, int)
        if val is None:
            raise ValueError(f"missing {key} for TSMixer; specify {key} explicitly")
        if val <= 0:
            raise ValueError(f"{key} must be > 0 for TSMixer, got {key}={val}")
        return val

    def _require_enum(key: str, allowed: set[str]) -> str:
        val = _parse(key, str)
        if val is None:
            raise ValueError(f"missing {key} for TSMixer; specify {key} explicitly")
        val = val.strip()
        if val not in allowed:
            raise ValueError(
                f"TSMixer {key} must be exactly one of {sorted(allowed)}, "
                f"got {key}={val!r}"
            )
        return val

    n_block = _require_positive_int("n_block")
    ff_dim = _require_positive_int("ff_dim")

    dropout = _parse("dropout", float)
    if dropout is None:
        raise ValueError("missing dropout for TSMixer; specify dropout explicitly")
    if not (0.0 <= dropout < 1.0):
        raise ValueError(
            f"TSMixer dropout must satisfy 0 <= dropout < 1, got dropout={dropout}"
        )

    norm_type = _require_enum("norm_type", {"L", "B"})
    activation = _require_enum("activation", {"relu", "gelu"})

    return {
        "n_block": n_block,
        "ff_dim": ff_dim,
        "dropout": dropout,
        "norm_type": norm_type,
        "activation": activation,
    }


def _hparam_spec_allows_none(spec_value: Any) -> bool:
    if spec_value is None:
        return True
    if isinstance(spec_value, list):
        return any(item is None for item in spec_value)
    return False


def infer_hparam_expected_type(spec_value: Any, *, key: str) -> type:
    """Infer the expected Python type for one hyperparameter spec entry."""
    if isinstance(spec_value, list):
        observed = {type(item) for item in spec_value if item is not None}
        if not observed:
            return type(None)
        if observed == {bool}:
            return bool
        if observed.issubset({int, float}):
            return float if float in observed else int
        if len(observed) == 1:
            return next(iter(observed))
        raise ValueError(
            "Cannot infer expected type from mixed hyperparameter spec values for "
            f"'{key}': " + ", ".join(sorted(t.__name__ for t in observed))
        )
    return type(spec_value)


def coerce_hparam_value(
    raw_value: Any,
    expected_type: type,
    *,
    key: str,
    allow_none: bool = False,
) -> Any:
    """Coerce one hyperparameter value to the expected type."""
    if raw_value is None:
        if allow_none:
            return None
        raise ValueError(
            f"Cannot coerce hyperparameter '{key}' to null/none; this key does not allow null."
        )
    if expected_type is type(None):
        if isinstance(raw_value, str):
            _, lowered = _normalize_token(raw_value)
            if _is_null_token(lowered):
                return None
        raise ValueError(
            f"Cannot coerce hyperparameter '{key}'='{raw_value}' to None; expected null/none."
        )
    if isinstance(raw_value, str):
        stripped, lowered = _normalize_token(raw_value)
        if _is_null_token(lowered):
            if allow_none:
                return None
            raise ValueError(
                f"Cannot coerce hyperparameter '{key}'='{raw_value}' to null/none; this key does not allow null."
            )
        if expected_type is bool:
            parsed_bool = _parse_bool_token(lowered)
            if parsed_bool is not None:
                return parsed_bool
            raise ValueError(
                f"Cannot coerce hyperparameter '{key}'='{raw_value}' to bool."
            )
        if expected_type is int:
            try:
                return int(stripped)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot coerce hyperparameter '{key}'='{raw_value}' to int."
                ) from exc
        if expected_type is float:
            try:
                return float(stripped)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot coerce hyperparameter '{key}'='{raw_value}' to float."
                ) from exc
        if expected_type in (list, dict, tuple):
            return _parse_collection_literal(
                stripped,
                expected_type=(expected_type,),
                key=key,
                error_prefix="Cannot coerce hyperparameter",
                include_parse_errors=True,
            )
    if expected_type is bool:
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, (int, float)) and raw_value in (0, 1):
            return bool(int(raw_value))
        raise ValueError(f"Cannot coerce hyperparameter '{key}'='{raw_value}' to bool.")
    if expected_type is int:
        if isinstance(raw_value, bool):
            raise ValueError(f"Cannot coerce hyperparameter '{key}'='{raw_value}' to int.")
        if isinstance(raw_value, int):
            return raw_value
        if isinstance(raw_value, float):
            if raw_value.is_integer():
                return int(raw_value)
            raise ValueError(
                f"Cannot coerce hyperparameter '{key}'='{raw_value}' to int."
            )
    if expected_type is float:
        if isinstance(raw_value, bool):
            raise ValueError(
                f"Cannot coerce hyperparameter '{key}'='{raw_value}' to float."
            )
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
    if expected_type in (list, dict, tuple):
        if isinstance(raw_value, expected_type):
            return raw_value
        raise ValueError(
            f"Cannot coerce hyperparameter '{key}'='{raw_value}' to {expected_type.__name__}."
        )
    try:
        return expected_type(raw_value)
    except Exception as exc:
        raise ValueError(
            f"Cannot coerce hyperparameter '{key}'='{raw_value}' to {expected_type}."
        ) from exc


def extract_required_typed_hparams(
    values: Mapping[str, Any],
    hparam_spec: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """Extract and type-coerce all required hyperparameter keys from a mapping."""
    typed: dict[str, Any] = {}
    for key, spec_value in hparam_spec.items():
        if key not in values:
            raise ValueError(f"{context} is missing required hyperparameter '{key}'.")
        expected_type = infer_hparam_expected_type(spec_value, key=key)
        typed[key] = coerce_hparam_value(
            values[key],
            expected_type,
            key=key,
            allow_none=_hparam_spec_allows_none(spec_value),
        )
    return typed


def require_numeric_series(
    series: pd.Series,
    *,
    column_name: str,
    context: str,
    allow_nan: bool = False,
    allow_infinite: bool = False,
) -> pd.Series:
    """Parse a series as numeric and enforce NaN/inf policy."""
    try:
        parsed = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as exc:
        coerced = pd.to_numeric(series, errors="coerce")
        bad_mask = series.notna() & coerced.isna()
        bad_examples = series.loc[bad_mask].head(5).tolist()
        raise ValueError(
            f"{context} because column '{column_name}' has non-numeric values. "
            f"Examples: {bad_examples}."
        ) from exc

    if not allow_nan and parsed.isna().any():
        bad_examples = series.loc[parsed.isna()].head(5).tolist()
        raise ValueError(
            f"{context} because column '{column_name}' has missing values. "
            f"Examples: {bad_examples}."
        )

    values = parsed.to_numpy(dtype=float)
    if not allow_infinite and not np.isfinite(values).all():
        bad_mask = ~np.isfinite(values)
        bad_examples = series.loc[bad_mask].head(5).tolist()
        raise ValueError(
            f"{context} because column '{column_name}' has non-finite values. "
            f"Examples: {bad_examples}."
        )
    return parsed


def require_integer_series(
    df: pd.DataFrame,
    column: str,
    *,
    context: str,
    sample_cols: Sequence[str] | None = None,
    min_value: int | None = None,
) -> pd.Series:
    """Parse a dataframe column as integer-valued numeric data."""
    if sample_cols is None:
        parsed = pd.to_numeric(df[column], errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"{context} column '{column}' contains non-numeric values.")
    else:
        parsed = require_numeric_series(
            df[column],
            column_name=column,
            context=context,
        )
    values = parsed.to_numpy(dtype=float)
    integer_mask = values == np.floor(values)
    if not bool(integer_mask.all()):
        if sample_cols is None:
            raise ValueError(f"{context} column '{column}' must contain integer values.")
        examples = sample_dataframe_records(df.loc[~integer_mask], sample_cols)
        raise ValueError(
            f"{context}: rows contain non-integer '{column}' values. Examples: {examples}."
        )
    integers = parsed.astype(int)
    if min_value is not None:
        bad_mask = integers < int(min_value)
        if bad_mask.any():
            if sample_cols is None:
                examples = integers[bad_mask].head(8).tolist()
                raise ValueError(
                    f"{context} column '{column}' must be >= {min_value}. "
                    f"Examples: {examples}."
                )
            examples = sample_dataframe_records(df.loc[bad_mask], sample_cols)
            raise ValueError(
                f"{context}: rows contain '{column}' values below {min_value}. "
                f"Examples: {examples}."
            )
    return integers


def require_dataframe_columns(
    df: pd.DataFrame,
    required_cols: set[str] | Sequence[str],
    *,
    context: str,
) -> None:
    missing_cols = sorted(set(required_cols) - set(df.columns))
    if missing_cols:
        raise ValueError(f"{context}: missing columns {missing_cols}.")


def sample_dataframe_records(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    if df.empty:
        return []
    unique_columns = list(dict.fromkeys(columns))
    return df.loc[:, unique_columns].head(limit).to_dict(orient="records")


def require_nonempty_string_series(
    df: pd.DataFrame,
    column: str,
    *,
    context: str,
    sample_cols: Sequence[str],
) -> pd.Series:
    missing_mask = df[column].isna()
    if missing_mask.any():
        examples = sample_dataframe_records(df.loc[missing_mask], sample_cols)
        raise ValueError(
            f"{context}: rows contain missing '{column}' values. Examples: {examples}."
        )
    cleaned = df[column].astype(str).str.strip()
    empty_mask = cleaned == ""
    if empty_mask.any():
        examples = sample_dataframe_records(df.loc[empty_mask], sample_cols)
        raise ValueError(
            f"{context}: rows contain empty '{column}' values. Examples: {examples}."
        )
    return cleaned


def _normalized_duplicate_key_frame(
    df: pd.DataFrame,
    key_cols: Sequence[str],
) -> pd.DataFrame:
    normalized = pd.DataFrame(index=df.index)
    for column in key_cols:
        series = df[column]
        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            normalized[column] = series.astype(str).str.strip()
        else:
            normalized[column] = series
    return normalized


def assert_no_duplicate_rows(
    df: pd.DataFrame,
    key_cols: Sequence[str],
    *,
    context: str,
) -> None:
    unique_key_cols = list(dict.fromkeys(key_cols))
    normalized = _normalized_duplicate_key_frame(df, unique_key_cols)
    duplicate_mask = normalized.duplicated(unique_key_cols, keep=False)
    if not duplicate_mask.any():
        return
    examples = (
        normalized.loc[duplicate_mask, unique_key_cols]
        .drop_duplicates()
        .head(5)
        .to_dict(orient="records")
    )
    raise ValueError(f"{context}. Examples: {examples}.")


def _require_finite_float(value: Any, *, key: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{key} must be finite; got {numeric}.")
    return numeric


def parse_required_finite_float(value: Any, *, key: str) -> float:
    """Parse and validate one required finite float."""
    parsed = parse_value(value, float, key=key)
    if parsed is None:
        raise ValueError(f"{key} is required.")
    return _require_finite_float(parsed, key=key)


def _parse_required_positive_bounded_float(
    value: Any,
    *,
    key: str,
    max_value: Optional[float] = None,
) -> float:
    """Parse a required finite float satisfying value > 0 (and optionally <= max_value)."""
    parsed = parse_value(value, float, key=key)
    if parsed is None:
        raise ValueError(f"{key} is required.")
    result = _require_finite_float(parsed, key=key)
    if max_value is not None:
        if result <= 0.0 or result > max_value:
            raise ValueError(f"{key} must satisfy 0 < {key} <= {max_value}; got {result}.")
    else:
        if result <= 0.0:
            raise ValueError(f"{key} must be > 0; got {result}.")
    return result


def _parse_required_nonnegative_bounded_float(
    value: Any,
    *,
    key: str,
    max_value: Optional[float] = None,
) -> float:
    """Parse a required finite float satisfying value >= 0 (and optionally <= max_value)."""
    parsed = parse_value(value, float, key=key)
    if parsed is None:
        raise ValueError(f"{key} is required.")
    result = _require_finite_float(parsed, key=key)
    if max_value is not None:
        if result < 0.0 or result > max_value:
            raise ValueError(
                f"{key} must satisfy 0 <= {key} <= {max_value}; got {result}."
            )
    else:
        if result < 0.0:
            raise ValueError(f"{key} must be >= 0; got {result}.")
    return result


def parse_required_nonnegative_float(
    value: Any,
    *,
    key: str,
) -> float:
    """Parse and validate one required finite float satisfying value >= 0."""
    return _parse_required_nonnegative_bounded_float(value, key=key)


def parse_required_unit_interval_float(
    value: Any,
    *,
    key: str,
) -> float:
    """Parse and validate one required finite float satisfying 0 <= value <= 1."""
    return _parse_required_nonnegative_bounded_float(value, key=key, max_value=1.0)


def parse_perturbation_channel_fraction_max(
    value: Any,
    *,
    key: str = "perturbation_channel_fraction_max",
) -> float:
    """Parse and validate maximum affected-channel fraction."""
    return _parse_required_positive_bounded_float(value, key=key, max_value=1.0)


def parse_optional_unit_float(
    value: Any,
    *,
    key: str,
    max_value: float = 1.0,
) -> Optional[float]:
    """Parse an optional finite float satisfying 0 < value <= max_value."""
    parsed_max_value = _parse_required_positive_bounded_float(
        max_value,
        key=f"{key}_max_value",
        max_value=1.0,
    )
    if value is None:
        return None
    if isinstance(value, str):
        token, lowered = _normalize_token(value)
        if lowered in ("none", "null"):
            return None
        if token == "":
            raise ValueError(f"{key} must not be empty; use null/None when unset.")
    try:
        return _parse_required_positive_bounded_float(
            value,
            key=key,
            max_value=parsed_max_value,
        )
    except ValueError as exc:
        raise ValueError(
            f"{key} must be a finite float satisfying 0 < {key} <= "
            f"{parsed_max_value}; got {value!r}."
        ) from exc


def _canonical_float_token(value: float) -> str:
    normalized = Decimal(str(float(value))).normalize()
    token = format(normalized, "f")
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    if token == "-0":
        token = "0"
    return token


def format_fixed_channel_fraction_token(
    value: Any,
    *,
    key: str = "fixed_channel_fraction",
    max_value: float = 1.0,
) -> str:
    """Build the canonical fraction token used in ablation metric and artifact paths."""
    parsed = parse_optional_unit_float(value, key=key, max_value=max_value)
    if parsed is None:
        raise ValueError(f"{key} is required for fixed-channel-fraction paths.")
    fraction_token = _canonical_float_token(parsed).replace(".", "p")
    return f"fraction_{fraction_token}"


def parse_train_perturbation_probability(
    value: Any,
    *,
    key: str = "train_perturbation_probability",
) -> float:
    """Parse train-time perturbation application probability."""
    return _parse_required_positive_bounded_float(value, key=key, max_value=1.0)


def parse_train_perturbation_severity_max(
    value: Any,
    *,
    key: str = "train_perturbation_severity_max",
) -> float:
    """Parse maximum train-time perturbation severity."""
    return _parse_required_positive_bounded_float(value, key=key, max_value=1.0)


def parse_train_perturbation_profile(
    value: Any,
    *,
    key: str = "train_perturbation_profile",
) -> str:
    """Parse a named train-fault profile key."""
    parsed = parse_value(value, str, key=key)
    if parsed is None:
        raise ValueError(f"{key} is required.")
    profile = str(parsed).strip()
    if not profile:
        raise ValueError(f"{key} must be a non-empty string.")
    if any(ch != "_" and not ch.isalnum() for ch in profile):
        raise ValueError(
            f"{key} must be underscore-safe (letters, digits, underscore only); got '{profile}'."
        )
    return profile


class AdvtrainConfig:
    """Parsed and validated adversarial-training configuration.

    All fields are already typed. Consumers should not re-cast.
    ``loss_token`` is None when loss validation was deferred to a later stage.
    """

    __slots__ = (
        "epsilon",
        "step_size",
        "attack_steps",
        "random_start",
        "attack_channels",
        "loss_token",
    )

    def __init__(
        self,
        *,
        epsilon: float,
        step_size: float,
        attack_steps: int,
        random_start: bool,
        attack_channels: str,
        loss_token: Optional[str],
    ) -> None:
        self.epsilon = epsilon
        self.step_size = step_size
        self.attack_steps = attack_steps
        self.random_start = random_start
        self.attack_channels = attack_channels
        self.loss_token = loss_token


def parse_advtrain_epsilon(
    value: Any,
    *,
    key: str = "advtrain_epsilon",
) -> float:
    """Parse the L_inf adversarial budget for standard adversarial training."""
    return _parse_required_positive_bounded_float(
        value,
        key=key,
    )


_ADVTRAIN_LOSS_DEFERRED = object()


def parse_advtrain_config(
    raw: Mapping[str, Any],
    *,
    context: str = "adversarial training",
    loss_value: Any = _ADVTRAIN_LOSS_DEFERRED,
) -> AdvtrainConfig:
    """Parse and validate all adversarial-training hyperparameters at once.

    *raw* is a dict-like that must contain the five ``advtrain_*`` keys.
    *loss_value* is the training-loss token (e.g. from ``hparams.loss`` or
    ``args.loss``). When omitted, loss validation is deferred.
    """
    loss_token: Optional[str] = None
    if loss_value is not _ADVTRAIN_LOSS_DEFERRED:
        loss_token = parse_advtrain_loss(
            loss_value,
            key="loss",
            context=context,
        )
    epsilon = parse_advtrain_epsilon(
        raw.get("advtrain_epsilon"),
        key="advtrain_epsilon",
    )
    attack_steps = parse_required_positive_int(
        raw.get("advtrain_attack_steps"),
        key="advtrain_attack_steps",
    )
    step_size = _parse_required_positive_bounded_float(
        raw.get("advtrain_step_size"),
        key="advtrain_step_size",
    )
    return AdvtrainConfig(
        epsilon=epsilon,
        step_size=step_size,
        attack_steps=attack_steps,
        random_start=parse_required_bool(
            raw.get("advtrain_random_start"),
            key="advtrain_random_start",
            context=context,
        ),
        attack_channels=parse_advtrain_attack_channels(
            raw.get("advtrain_attack_channels"),
            key="advtrain_attack_channels",
        ),
        loss_token=loss_token,
    )


def parse_advtrain_attack_channels(
    value: Any,
    *,
    key: str = "advtrain_attack_channels",
) -> str:
    """Parse the attack-channel scope for adversarial training."""
    return parse_required_choice(
        value,
        key=key,
        allowed=("all", "continuous"),
    )


def parse_advtrain_loss(
    value: Any,
    *,
    key: str = "loss",
    context: str | None = None,
) -> str:
    """Parse the restricted loss surface allowed for adversarial training."""
    token = parse_required_nonempty_string(
        value,
        key=key,
        context=context,
    )
    token_upper = token.upper()
    if token_upper in {"MSE", "MAE"}:
        return token
    subject = context if context else "Value"
    raise ValueError(
        f"{subject} has unsupported {key} '{token}' for adversarial training. "
        "Allowed losses: MSE, MAE."
    )


def parse_perturbation_scenarios(
    value: Any,
    *,
    key: str = "perturbation_scenarios",
) -> tuple[str, ...]:
    """Parse and validate explicit perturbation scenario names."""
    parsed = parse_value(value, list, key=key)
    if parsed is None:
        raise ValueError(f"{key} is required.")
    if not parsed:
        raise ValueError(f"{key} must be a non-empty list of scenario names.")
    names: list[str] = []
    seen: set[str] = set()
    for idx, raw_name in enumerate(parsed):
        if not isinstance(raw_name, str):
            raise ValueError(
                f"{key} entry at index {idx} must be a string; got {type(raw_name).__name__}."
            )
        name = raw_name.strip()
        if not name:
            raise ValueError(
                f"{key} contains an empty scenario name at index {idx}."
            )
        if name in seen:
            raise ValueError(f"{key} contains duplicate scenario name '{name}'.")
        seen.add(name)
        names.append(name)
    return tuple(names)


def canonicalize_train_perturbation_scenarios(
    names: Sequence[str],
    *,
    registry_names: Sequence[str],
    key: str = "train_perturbation_scenarios",
) -> tuple[str, ...]:
    """Sort a resolved train-fault subset by the unified registry order."""
    scenarios = parse_perturbation_scenarios(list(names), key=key)
    registry = parse_perturbation_scenarios(
        list(registry_names),
        key=f"{key}_registry",
    )
    order = {name: idx for idx, name in enumerate(registry)}
    unknown = [name for name in scenarios if name not in order]
    if unknown:
        raise ValueError(
            f"{key} contains unknown scenario name(s): {sorted(unknown)}. "
            f"Known scenarios: {', '.join(registry)}."
        )
    return tuple(sorted(scenarios, key=order.__getitem__))


def parse_train_fault_profiles(
    raw: Any,
    *,
    registry_names: Sequence[str],
    key: str = "train_fault_profiles",
) -> dict[str, tuple[str, ...]]:
    """Parse and validate named train-fault profile definitions from YAML."""
    if raw is None:
        raise ValueError(f"{key} is required.")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{key} must be a mapping from profile key to definition.")
    if not raw:
        raise ValueError(f"{key} must define at least one profile.")

    profiles: dict[str, tuple[str, ...]] = {}
    for raw_profile, raw_entry in raw.items():
        profile = parse_train_perturbation_profile(
            raw_profile,
            key=f"{key} profile",
        )
        if profile in profiles:
            raise ValueError(f"{key} contains duplicate profile '{profile}'.")
        if not isinstance(raw_entry, Mapping):
            raise ValueError(
                f"{key}.{profile} must be a mapping with required key 'scenarios'."
            )
        supported = {"scenarios"}
        unknown = sorted(set(raw_entry.keys()) - supported)
        if unknown:
            raise ValueError(
                f"{key}.{profile} has unsupported key(s): {', '.join(unknown)}."
            )
        if "scenarios" not in raw_entry:
            raise ValueError(f"{key}.{profile} is missing required key 'scenarios'.")
        profiles[profile] = canonicalize_train_perturbation_scenarios(
            raw_entry["scenarios"],
            registry_names=registry_names,
            key=f"{key}.{profile}.scenarios",
        )
    return profiles


def resolve_train_perturbation_profile_config(
    profile: Any,
    *,
    profiles: Mapping[str, Sequence[str]],
    registry_names: Sequence[str],
    scenarios: Sequence[str] | None = None,
    scenarios_signature: Any = None,
    profile_key: str = "train_perturbation_profile",
    profiles_key: str = "train_fault_profiles",
    scenarios_key: str = "train_perturbation_scenarios",
    signature_key: str = "train_perturbation_scenarios_signature",
) -> tuple[str, tuple[str, ...], str]:
    """Resolve a FAug profile to its canonical scenarios and signature."""
    resolved_profile = parse_train_perturbation_profile(profile, key=profile_key)
    if not isinstance(profiles, Mapping):
        raise ValueError(
            f"{profiles_key} must be a mapping from profile key to scenario lists."
        )

    resolved_profile_scenarios = profiles.get(resolved_profile)
    if resolved_profile_scenarios is None:
        raise ValueError(
            f"Unknown {profile_key} '{resolved_profile}'. "
            f"Known profiles: {', '.join(sorted(profiles.keys()))}."
        )
    expected_scenarios = parse_perturbation_scenarios(
        list(resolved_profile_scenarios),
        key=f"{profiles_key}.{resolved_profile}.scenarios",
    )

    if scenarios is None:
        resolved_scenarios = expected_scenarios
    else:
        resolved_scenarios = canonicalize_train_perturbation_scenarios(
            scenarios,
            registry_names=registry_names,
            key=scenarios_key,
        )
        if resolved_scenarios != expected_scenarios:
            raise ValueError(
                f"{scenarios_key} does not match the configured {profile_key}."
            )

    resolved_signature = build_perturbation_scenarios_signature(resolved_scenarios)
    if scenarios_signature is not None:
        parsed_signature = parse_perturbation_scenarios_signature(
            scenarios_signature,
            key=signature_key,
        )
        if parsed_signature != resolved_signature:
            raise ValueError(
                f"{signature_key} does not match {scenarios_key}."
            )

    return resolved_profile, resolved_scenarios, resolved_signature


def build_perturbation_scenarios_signature(
    names: Sequence[str],
) -> str:
    """Build deterministic order-preserving signature for perturbation scenarios."""
    if isinstance(names, str):
        raise ValueError("perturbation scenario names must be a sequence, not a string.")
    if names is None:
        raise ValueError("perturbation scenario names are required for signature building.")
    normalized = parse_perturbation_scenarios(
        list(names),
        key="perturbation_scenarios_signature_source",
    )
    return json.dumps(list(normalized), separators=(",", ":"), ensure_ascii=True)


def parse_perturbation_scenarios_signature(
    value: Any,
    *,
    key: str = "perturbation_scenarios_signature",
) -> str:
    """Parse and validate canonical perturbation-scenario signature."""
    parsed = parse_value(value, str, key=key)
    if parsed is None:
        raise ValueError(f"{key} is required.")
    signature = str(parsed).strip()
    if not signature:
        raise ValueError(f"{key} must be a non-empty string.")
    try:
        decoded = json.loads(signature)
    except Exception as exc:
        raise ValueError(
            f"{key} must be a canonical JSON-serialized scenario list."
        ) from exc
    canonical = build_perturbation_scenarios_signature(decoded)
    if signature != canonical:
        raise ValueError(
            f"{key} must be canonical JSON with deterministic ordering."
        )
    return canonical


def parse_perturbation_scenarios_from_signature(
    value: Any,
    *,
    key: str = "perturbation_scenarios_signature",
) -> tuple[str, ...]:
    """Decode one canonical scenario-signature string into ordered scenario names."""
    signature = parse_perturbation_scenarios_signature(value, key=key)
    try:
        decoded = json.loads(signature)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"{key} must decode to a JSON array.") from exc
    return parse_perturbation_scenarios(decoded, key=key)


def build_perturbation_idx_name_map(
    value: Mapping[Any, Any],
    *,
    key: str = "perturbation_idx_name_map_source",
) -> str:
    """Build canonical JSON mapping from perturbation index to scenario name."""
    if value is None:
        raise ValueError(f"{key} is required.")
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping from perturbation index to scenario name.")
    if not value:
        raise ValueError(f"{key} must be a non-empty mapping.")
    normalized: dict[int, str] = {}
    seen_names: set[str] = set()
    for raw_idx, raw_name in value.items():
        parsed_idx = parse_value(raw_idx, int, key=f"{key} index")
        if parsed_idx is None:
            raise ValueError(f"{key} contains a missing perturbation index.")
        idx = int(parsed_idx)
        if idx < 0:
            raise ValueError(f"{key} contains negative perturbation index {idx}.")
        if idx in normalized:
            raise ValueError(f"{key} contains duplicate perturbation index {idx}.")
        if not isinstance(raw_name, str):
            raise ValueError(
                f"{key} value for index {idx} must be a string; got {type(raw_name).__name__}."
            )
        name = raw_name.strip()
        if not name:
            raise ValueError(f"{key} value for index {idx} must be a non-empty string.")
        if name in seen_names:
            raise ValueError(f"{key} contains duplicate scenario name '{name}'.")
        seen_names.add(name)
        normalized[idx] = name
    canonical = {str(idx): normalized[idx] for idx in sorted(normalized)}
    return json.dumps(canonical, separators=(",", ":"), ensure_ascii=True)


def parse_perturbation_idx_name_map(
    value: Any,
    *,
    key: str = "perturbation_idx_name_map",
) -> dict[int, str]:
    """Parse and validate canonical perturbation index->scenario mapping."""
    parsed = parse_value(value, str, key=key)
    if parsed is None:
        raise ValueError(f"{key} is required.")
    raw = str(parsed).strip()
    if not raw:
        raise ValueError(f"{key} must be a non-empty string.")
    try:
        decoded = json.loads(raw)
    except Exception as exc:
        raise ValueError(f"{key} must be a canonical JSON object mapping indices to names.") from exc
    canonical = build_perturbation_idx_name_map(decoded, key=key)
    if raw != canonical:
        raise ValueError(f"{key} must be canonical JSON with sorted integer-string keys.")
    normalized: dict[int, str] = {}
    for idx_raw, name in decoded.items():
        idx = parse_value(idx_raw, int, key=f"{key} index")
        if idx is None:
            raise ValueError(f"{key} contains a missing perturbation index.")
        normalized[int(idx)] = str(name).strip()
    return normalized


def require_order_sensitive_perturbation_idx_name_map(
    value: Any,
    *,
    scenarios_signature: Any,
    scenarios_count: Any | None = None,
    key: str = "perturbation_idx_name_map",
    signature_key: str = "perturbation_scenarios_signature",
    count_key: str = "perturbation_scenarios_count",
) -> dict[int, str]:
    """Require an order-preserving idx->name map consistent with the scenario signature."""
    if isinstance(value, Mapping):
        parsed_idx_name_map = parse_perturbation_idx_name_map(
            build_perturbation_idx_name_map(value, key=key),
            key=key,
        )
    else:
        parsed_idx_name_map = parse_perturbation_idx_name_map(value, key=key)
    parsed_signature = parse_perturbation_scenarios_signature(
        scenarios_signature,
        key=signature_key,
    )
    ordered_signature_names = parse_perturbation_scenarios_from_signature(
        parsed_signature,
        key=signature_key,
    )
    expected_indices = list(range(len(parsed_idx_name_map)))
    actual_indices = sorted(parsed_idx_name_map)
    if actual_indices != expected_indices:
        raise ValueError(
            f"{key} must use contiguous zero-based indices {expected_indices}; "
            f"got {actual_indices}."
        )
    ordered_map_names = tuple(parsed_idx_name_map[idx] for idx in expected_indices)
    resolved_signature = build_perturbation_scenarios_signature(ordered_map_names)
    if resolved_signature != parsed_signature:
        raise ValueError(
            f"{key} order {ordered_map_names!r} does not match {signature_key}="
            f"{parsed_signature}."
        )
    if tuple(ordered_signature_names) != ordered_map_names:
        raise ValueError(
            f"{key} order {ordered_map_names!r} does not match decoded {signature_key}="
            f"{tuple(ordered_signature_names)!r}."
        )
    if scenarios_count is not None:
        parsed_count = parse_required_positive_int(scenarios_count, key=count_key)
        if parsed_count != len(parsed_idx_name_map):
            raise ValueError(
                f"{count_key}={parsed_count} does not match {key} length "
                f"{len(parsed_idx_name_map)}."
            )
    return parsed_idx_name_map


def _require_perturbation_coupling_mapping(
    values: Mapping[str, Any],
    *,
    run_id: str,
    source_name: str,
    expected_max: Any | None = None,
    expected_scenarios_signature: Any | None = None,
) -> dict[str, Any]:
    """Parse required perturbation-coupling values and enforce optional expectations."""
    if values is None:
        raise ValueError(
            f"Run {run_id} is missing {source_name} required for perturbation coupling."
        )
    if not isinstance(values, Mapping):
        raise ValueError(
            f"Run {run_id} {source_name} must be a mapping for perturbation coupling checks."
        )
    try:
        channel_fraction_max = parse_perturbation_channel_fraction_max(
            values.get("perturbation_channel_fraction_max"),
            key="perturbation_channel_fraction_max",
        )
    except ValueError as exc:
        raise ValueError(
            f"Run {run_id} has invalid perturbation coupling {source_name}: {exc}"
        ) from exc

    try:
        scenarios_signature = parse_perturbation_scenarios_signature(
            values.get("perturbation_scenarios_signature"),
            key="perturbation_scenarios_signature",
        )
    except ValueError as exc:
        raise ValueError(
            f"Run {run_id} has invalid perturbation coupling {source_name}: {exc}"
        ) from exc

    if expected_max is not None:
        expected_channel_fraction_max = parse_perturbation_channel_fraction_max(
            expected_max,
            key="expected perturbation_channel_fraction_max",
        )
        if not math.isclose(
            channel_fraction_max,
            expected_channel_fraction_max,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Run {run_id} has perturbation_channel_fraction_max="
                f"{channel_fraction_max} but expected {expected_channel_fraction_max}."
            )

    if expected_scenarios_signature is not None:
        expected_signature = parse_perturbation_scenarios_signature(
            expected_scenarios_signature,
            key="expected perturbation_scenarios_signature",
        )
        if scenarios_signature != expected_signature:
            raise ValueError(
                f"Run {run_id} has perturbation_scenarios_signature="
                f"{scenarios_signature} but expected {expected_signature}."
            )

    return {
        "perturbation_channel_fraction_max": channel_fraction_max,
        "perturbation_scenarios_signature": scenarios_signature,
    }


def require_perturbation_coupling_params(
    params: Mapping[str, Any],
    *,
    run_id: str,
    expected_max: Any | None = None,
    expected_scenarios_signature: Any | None = None,
) -> dict[str, Any]:
    """Parse required perturbation-coupling params and enforce optional expectations."""
    return _require_perturbation_coupling_mapping(
        params,
        run_id=run_id,
        source_name="params",
        expected_max=expected_max,
        expected_scenarios_signature=expected_scenarios_signature,
    )


def require_perturbation_coupling_tags(
    tags: Mapping[str, Any],
    *,
    run_id: str,
    expected_max: Any | None = None,
    expected_scenarios_signature: Any | None = None,
) -> dict[str, Any]:
    """Parse required perturbation-coupling tags and enforce optional expectations."""
    return _require_perturbation_coupling_mapping(
        tags,
        run_id=run_id,
        source_name="tags",
        expected_max=expected_max,
        expected_scenarios_signature=expected_scenarios_signature,
    )


def optional_nonempty_tag_value(tags: Mapping[str, Any], *, key: str) -> Optional[str]:
    """Return a tag value normalized to non-empty string, or ``None``."""
    raw_value = tags.get(key)
    if raw_value is None:
        return None
    normalized = str(raw_value).strip()
    if not normalized:
        return None
    return normalized


def require_mapping(
    value: Any,
    *,
    key: str,
    context: str | None = None,
) -> Mapping[str, Any]:
    """Require one metadata object to be a mapping."""
    ctx = context if context else "Value"
    if value is None:
        raise ValueError(f"{ctx} is missing {key}.")
    if not isinstance(value, Mapping):
        raise ValueError(f"{ctx} has non-mapping {key}.")
    return value


def get_tag_or_param_value(
    tags: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    key: str,
) -> Any:
    """Return one metadata value, preferring tags over params."""
    if key in tags:
        return tags.get(key)
    if key in params:
        return params.get(key)
    return None


def require_tag_value_with_optional_param_match(
    tags: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    key: str,
    context: str,
    disallow_none_token: bool = False,
) -> str:
    """Require one tag value and validate any duplicate param copy matches exactly."""
    tag_value = parse_required_nonempty_string(
        tags.get(key) if key in tags else None,
        key=key,
        context=context,
        disallow_none_token=disallow_none_token,
    )
    param_value = parse_optional_nonempty_string(
        params.get(key) if key in params else None,
        key=key,
        context=context,
        disallow_none_token=disallow_none_token,
    )
    if param_value is not None and param_value != tag_value:
        raise ValueError(
            f"{context} has conflicting {key} metadata: tag={tag_value!r}, "
            f"param={param_value!r}."
        )
    return tag_value


def _require_typed_value(
    raw_value: Any,
    *,
    key: str,
    expected_type: type,
    context: str,
    allow_none: bool,
) -> Any:
    if raw_value is None:
        raise ValueError(f"{context} is missing required '{key}'.")
    parsed = parse_value(raw_value, expected_type, key=key)
    if parsed is None and not allow_none:
        raise ValueError(f"{context} has null '{key}'.")
    return parsed


def require_typed_mapping_value(
    values: Mapping[str, Any],
    *,
    key: str,
    expected_type: type,
    context: str,
    allow_none: bool = False,
) -> Any:
    """Require one typed metadata value from a mapping."""
    return _require_typed_value(
        values.get(key) if key in values else None,
        key=key,
        expected_type=expected_type,
        context=context,
        allow_none=allow_none,
    )


def require_typed_tag_or_param_value(
    tags: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    key: str,
    expected_type: type,
    context: str,
    allow_none: bool = False,
) -> Any:
    """Require one typed metadata value from tag/param storage."""
    return _require_typed_value(
        get_tag_or_param_value(tags, params, key=key),
        key=key,
        expected_type=expected_type,
        context=context,
        allow_none=allow_none,
    )


def parse_semicolon_delimited_strings(
    value: Any,
    *,
    key: str,
    context: str | None = None,
) -> tuple[str, ...]:
    """Parse optional semicolon-delimited string metadata."""
    subject = context if context else "Value"
    if value is None:
        return ()
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return ()
        raw_items = normalized.split(";")
    elif isinstance(value, (list, tuple)):
        raw_items = list(value)
        if not raw_items:
            return ()
    else:
        raise ValueError(
            f"{subject} has invalid {key}; expected semicolon-delimited string or string sequence."
        )

    items: list[str] = []
    for idx, raw_item in enumerate(raw_items):
        item = parse_optional_nonempty_string(
            raw_item,
            key=f"{key}[{idx}]",
            context=subject,
        )
        if item is None:
            raise ValueError(f"{subject} has invalid {key}[{idx}].")
        items.append(item)
    return tuple(items)


def parse_named_display_string_pairs(
    value: Any,
    *,
    key: str,
    context: str | None = None,
) -> tuple[tuple[str, str], ...]:
    """Parse one ordered list of ``name=display`` strings."""
    subject = context if context else "Value"
    parsed = parse_value(value, list, key=key)
    if parsed is None:
        raise ValueError(f"{subject} is missing required {key}.")
    if not parsed:
        raise ValueError(f"{subject} has empty {key}.")
    pairs: list[tuple[str, str]] = []
    seen_names: set[str] = set()
    seen_display_values: set[str] = set()
    for idx, raw_entry in enumerate(parsed):
        entry = parse_required_nonempty_string(
            raw_entry,
            key=f"{key}[{idx}]",
            context=subject,
            disallow_none_token=True,
        )
        name, sep, display = entry.partition("=")
        if not sep:
            raise ValueError(
                f"{subject} has invalid {key}[{idx}]={entry!r}; expected 'name=display'."
            )
        normalized_name = name.strip()
        normalized_display = display.strip()
        if not normalized_name:
            raise ValueError(f"{subject} has empty name in {key}[{idx}].")
        if not normalized_display:
            raise ValueError(f"{subject} has empty display value in {key}[{idx}].")
        if normalized_name in seen_names:
            raise ValueError(
                f"{subject} has duplicate name '{normalized_name}' in {key}."
            )
        if normalized_display in seen_display_values:
            raise ValueError(
                f"{subject} has duplicate display value '{normalized_display}' in {key}."
            )
        seen_names.add(normalized_name)
        seen_display_values.add(normalized_display)
        pairs.append((normalized_name, normalized_display))
    return tuple(pairs)


def parse_named_group_string_pairs(
    value: Any,
    *,
    key: str,
    context: str | None = None,
) -> dict[str, tuple[str, ...]]:
    """Parse one ordered list of ``group=item_a,item_b`` strings."""
    subject = context if context else "Value"
    parsed = parse_value(value, list, key=key)
    if parsed is None:
        raise ValueError(f"{subject} is missing required {key}.")
    if not parsed:
        raise ValueError(f"{subject} has empty {key}.")
    groups: dict[str, tuple[str, ...]] = {}
    for idx, raw_entry in enumerate(parsed):
        entry = parse_required_nonempty_string(
            raw_entry,
            key=f"{key}[{idx}]",
            context=subject,
            disallow_none_token=True,
        )
        group_name, sep, members = entry.partition("=")
        if not sep:
            raise ValueError(
                f"{subject} has invalid {key}[{idx}]={entry!r}; expected 'group=item_a,item_b'."
            )
        normalized_group_name = group_name.strip()
        if not normalized_group_name:
            raise ValueError(f"{subject} has empty group name in {key}[{idx}].")
        if normalized_group_name in groups:
            raise ValueError(
                f"{subject} has duplicate group '{normalized_group_name}' in {key}."
            )
        member_names: list[str] = []
        seen_members: set[str] = set()
        for member_idx, raw_member in enumerate(members.split(",")):
            normalized_member = raw_member.strip()
            if not normalized_member:
                raise ValueError(
                    f"{subject} has empty member in {key}[{idx}] at position {member_idx}."
                )
            if normalized_member in seen_members:
                raise ValueError(
                    f"{subject} has duplicate member '{normalized_member}' in {key}[{idx}]."
                )
            seen_members.add(normalized_member)
            member_names.append(normalized_member)
        groups[normalized_group_name] = tuple(member_names)
    return groups


def parse_required_string_sequence(
    value: Any,
    *,
    key: str,
    context: str | None = None,
) -> tuple[str, ...]:
    """Parse one required list of unique non-empty strings."""
    subject = context if context else "Value"
    parsed = parse_value(value, list, key=key)
    if parsed is None:
        raise ValueError(f"{subject} is missing required {key}.")
    if not parsed:
        raise ValueError(f"{subject} has empty {key}.")
    items: list[str] = []
    seen: set[str] = set()
    for idx, raw_item in enumerate(parsed):
        item = parse_required_nonempty_string(
            raw_item,
            key=f"{key}[{idx}]",
            context=subject,
            disallow_none_token=True,
        )
        if item in seen:
            raise ValueError(f"{subject} has duplicate '{item}' in {key}.")
        seen.add(item)
        items.append(item)
    return tuple(items)


def validate_raw_display_id_values(
    values: Sequence[Any],
    *,
    raw_ids: Sequence[str],
    display_mapping: Mapping[str, str],
    context: str,
    id_label: str,
) -> None:
    """Raise when *values* contain display labels, mixed raw/display ids, or unknown ids."""
    raw_id_set = {
        parse_required_nonempty_string(
            raw_id,
            key=f"{id_label}_raw_ids[{idx}]",
            context=context,
            disallow_none_token=True,
        )
        for idx, raw_id in enumerate(raw_ids)
    }
    display_value_set = {
        parse_required_nonempty_string(
            display_value,
            key=f"{id_label}_display[{raw_id}]",
            context=context,
            disallow_none_token=True,
        )
        for raw_id, display_value in display_mapping.items()
    }
    present = {
        parse_required_nonempty_string(
            value,
            key=f"{id_label}_values[{idx}]",
            context=context,
            disallow_none_token=True,
        )
        for idx, value in enumerate(values)
    }
    display_only = sorted(
        value
        for value in present
        if value in display_value_set and value not in raw_id_set
    )
    if display_only:
        raw_present = sorted(present & raw_id_set)
        if raw_present:
            raise ValueError(
                f"{context} received mixed raw/display {id_label} ids: "
                f"display={display_only}, raw={raw_present}."
            )
        raise ValueError(
            f"{context} received display labels where raw {id_label} ids are required: "
            f"{display_only}."
        )
    unknown_ids = sorted(present - raw_id_set - display_value_set)
    if unknown_ids:
        raise ValueError(
            f"{context} received unexpected {id_label} ids: {unknown_ids}."
        )


def validate_scoped_raw_display_id_values(
    values: Sequence[Any],
    *,
    raw_ids: Sequence[str],
    display_mapping: Mapping[str, str],
    known_raw_ids: Sequence[str],
    context: str,
    id_label: str,
) -> None:
    """Validate scoped ids while allowing repo-known raw ids outside the local scope."""
    normalized_values = [
        parse_required_nonempty_string(
            value,
            key=f"{id_label}_values[{idx}]",
            context=context,
            disallow_none_token=True,
        )
        for idx, value in enumerate(values)
    ]
    raw_id_set = {
        parse_required_nonempty_string(
            raw_id,
            key=f"{id_label}_raw_ids[{idx}]",
            context=context,
            disallow_none_token=True,
        )
        for idx, raw_id in enumerate(raw_ids)
    }
    display_value_set = {
        parse_required_nonempty_string(
            display_value,
            key=f"{id_label}_display[{raw_id}]",
            context=context,
            disallow_none_token=True,
        )
        for raw_id, display_value in display_mapping.items()
    }
    known_raw_id_set = {
        parse_required_nonempty_string(
            raw_id,
            key=f"{id_label}_known_raw_ids[{idx}]",
            context=context,
            disallow_none_token=True,
        )
        for idx, raw_id in enumerate(known_raw_ids)
    }
    unknown_ids = sorted(
        set(normalized_values) - raw_id_set - display_value_set - known_raw_id_set
    )
    if unknown_ids:
        raise ValueError(
            f"{context} received unexpected {id_label} ids: {unknown_ids}."
        )
    validate_raw_display_id_values(
        [
            value
            for value in normalized_values
            if value in raw_id_set or value in display_value_set
        ],
        raw_ids=raw_ids,
        display_mapping=display_mapping,
        context=context,
        id_label=id_label,
    )


def parse_core_figure_registry_config(
    value: Any,
    *,
    context: str = "core figure registry config",
) -> CoreFigureRegistry:
    """Parse the dedicated core-figure dataset/method/scenario registry config."""
    config = require_mapping(
        value,
        key="core_figure_registry_config",
        context=context,
    )
    supported_keys = {
        "BASELINE_RANK_PARETO_METRIC",
        "CORE_IMPROVEMENT_TRAJECTORY_METHOD",
        "CORE_IMPROVEMENT_TRAJECTORY_METRIC",
        "CORE_FIGURE_DATASET_SPEC",
        "CORE_METHOD_DISPLAY",
        "CORE_SCENARIO_DISPLAY_ORDER",
        "CORE_SCENARIO_DISPLAY",
        "CORE_SCENARIO_GROUPS",
    }
    unknown_keys = sorted(set(config.keys()) - supported_keys)
    if unknown_keys:
        raise ValueError(
            f"{context} has unsupported key(s): {', '.join(str(key) for key in unknown_keys)}."
        )
    dataset_spec = parse_named_display_string_pairs(
        config.get("CORE_FIGURE_DATASET_SPEC"),
        key="CORE_FIGURE_DATASET_SPEC",
        context=context,
    )
    method_display_pairs = parse_named_display_string_pairs(
        config.get("CORE_METHOD_DISPLAY"),
        key="CORE_METHOD_DISPLAY",
        context=context,
    )
    scenario_display_order = parse_required_string_sequence(
        config.get("CORE_SCENARIO_DISPLAY_ORDER"),
        key="CORE_SCENARIO_DISPLAY_ORDER",
        context=context,
    )
    scenario_display_pairs = parse_named_display_string_pairs(
        config.get("CORE_SCENARIO_DISPLAY"),
        key="CORE_SCENARIO_DISPLAY",
        context=context,
    )
    scenario_groups = parse_named_group_string_pairs(
        config.get("CORE_SCENARIO_GROUPS"),
        key="CORE_SCENARIO_GROUPS",
        context=context,
    )

    method_display = dict(method_display_pairs)
    method_order = tuple(method_display.keys())
    allowed_metric_keys = ("D_w", "D_mean", "err_pert_ws")
    baseline_rank_pareto_metric = parse_required_canonical_choice(
        config.get("BASELINE_RANK_PARETO_METRIC"),
        key="BASELINE_RANK_PARETO_METRIC",
        allowed=allowed_metric_keys,
    )
    core_improvement_trajectory_method = parse_required_canonical_choice(
        config.get("CORE_IMPROVEMENT_TRAJECTORY_METHOD"),
        key="CORE_IMPROVEMENT_TRAJECTORY_METHOD",
        allowed=method_order,
    )
    core_improvement_trajectory_metric = parse_required_canonical_choice(
        config.get("CORE_IMPROVEMENT_TRAJECTORY_METRIC"),
        key="CORE_IMPROVEMENT_TRAJECTORY_METRIC",
        allowed=allowed_metric_keys,
    )

    scenario_display = dict(scenario_display_pairs)
    scenario_order_set = set(scenario_display_order)
    scenario_display_set = set(scenario_display)
    if scenario_order_set != scenario_display_set:
        raise ValueError(
            f"{context} has inconsistent scenario display config: "
            f"order={sorted(scenario_order_set)}, display={sorted(scenario_display_set)}."
        )
    grouped_scenarios = [
        scenario_name
        for group_members in scenario_groups.values()
        for scenario_name in group_members
    ]
    if len(grouped_scenarios) != len(set(grouped_scenarios)):
        raise ValueError(f"{context} assigns at least one scenario to multiple groups.")
    grouped_scenario_set = set(grouped_scenarios)
    if grouped_scenario_set != scenario_order_set:
        raise ValueError(
            f"{context} has inconsistent scenario groups: "
            f"groups={sorted(grouped_scenario_set)}, order={sorted(scenario_order_set)}."
        )
    if tuple(grouped_scenarios) != tuple(scenario_display_order):
        raise ValueError(
            f"{context} scenario groups must cover CORE_SCENARIO_DISPLAY_ORDER "
            "exactly once and in order."
        )

    return CoreFigureRegistry(
        baseline_rank_pareto_metric=baseline_rank_pareto_metric,
        core_improvement_trajectory_method=core_improvement_trajectory_method,
        core_improvement_trajectory_metric=core_improvement_trajectory_metric,
        dataset_spec=dataset_spec,
        method_display=method_display,
        method_order=method_order,
        scenario_display_order=scenario_display_order,
        scenario_display=scenario_display,
        scenario_groups=scenario_groups,
    )


def parse_optional_nonempty_string(
    value: Any,
    *,
    key: str,
    context: str | None = None,
    disallow_none_token: bool = False,
) -> Optional[str]:
    """Normalize optional string metadata and raise on malformed non-null values."""
    if value is None:
        return None
    normalized = str(value).strip()
    subject = context if context else "Value"
    if not normalized:
        raise ValueError(f"{subject} has empty {key}.")
    if disallow_none_token and normalized.lower() == "none":
        raise ValueError(
            f"{subject} has invalid {key} token 'none'; use null/None for missing values."
        )
    return normalized


def parse_backbone_run_ids(
    value: Any,
    *,
    run_id: str,
    key: str = "backbone_run_ids",
) -> list[str]:
    """Parse a comma-delimited backbone run id tag into a unique ordered list."""
    normalized = parse_optional_nonempty_string(
        value,
        key=key,
        context=f"Run {run_id}",
        disallow_none_token=True,
    )
    if normalized is None:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for idx, raw_token in enumerate(normalized.split(","), start=1):
        token = raw_token.strip()
        if not token:
            raise ValueError(
                f"Run {run_id} has malformed {key} tag: empty token at position {idx}."
            )
        if token in seen:
            raise ValueError(
                f"Run {run_id} has malformed {key} tag: duplicate id '{token}'."
            )
        seen.add(token)
        ordered.append(token)
    return ordered


def parse_required_nonempty_string(
    value: Any,
    *,
    key: str,
    context: str | None = None,
    disallow_none_token: bool = False,
) -> str:
    """Parse one required non-empty string value."""
    parsed = parse_optional_nonempty_string(
        value,
        key=key,
        context=context,
        disallow_none_token=disallow_none_token,
    )
    if parsed is None:
        subject = context if context else "Value"
        raise ValueError(f"{subject} is missing required {key}.")
    return parsed


def parse_optimizer_name(
    value: Any,
    *,
    key: str = "optimizer",
    context: str | None = "optimizer defaults",
) -> str:
    """Parse the benchmark optimizer enum."""
    parsed = parse_optional_nonempty_string(
        value,
        key=key,
        context=context,
        disallow_none_token=True,
    )
    if parsed is None:
        raise ValueError(f"{key} must be a non-empty string.")
    if parsed.lower() != "adam":
        raise ValueError(f"Unknown optimizer '{parsed}'.")
    return parsed


def parse_scheduler_type(
    value: Any,
    *,
    key: str = "scheduler_type",
    context: str | None = "optimizer defaults",
) -> str:
    """Parse the benchmark LR-scheduler enum."""
    parsed = parse_optional_nonempty_string(
        value,
        key=key,
        context=context,
        disallow_none_token=True,
    )
    if parsed is None:
        raise ValueError(f"{key} must be a non-empty string.")
    if parsed != "plateau":
        raise ValueError(f"Unknown scheduler_type '{parsed}'.")
    return parsed


def parse_reference_normalization_anchor_model(
    value: Any,
    *,
    key: str = "reference_normalization_anchor_model",
    context: str | None = None,
) -> str:
    """Parse the explicit anchor-model identifier for reference-normalized diagnostics."""
    subject = context if context else "Reference normalization anchor model"
    parsed = parse_required_nonempty_string(
        value,
        key=key,
        context=subject,
        disallow_none_token=True,
    )
    if parsed != "SeasonalNaive":
        raise ValueError(
            f"{subject} has unsupported {key} '{parsed}'. "
            "Expected 'SeasonalNaive'."
        )
    return parsed


def parse_runtime_precision(
    value: Any,
    *,
    device_type: Any,
    key: str = "precision",
    context: str | None = None,
) -> RuntimePrecisionConfig:
    """Parse a runtime precision setting into explicit eval-time semantics."""
    subject = context if context else "Runtime precision"
    resolved_device_type = parse_required_nonempty_string(
        device_type,
        key="device_type",
        context=subject,
        disallow_none_token=True,
    ).lower()
    if isinstance(value, bool):
        raise ValueError(f"{subject} requires a numeric or string {key}, not a boolean.")
    if isinstance(value, int):
        raw_precision = str(int(value))
    else:
        raw_precision = parse_required_nonempty_string(
            value,
            key=key,
            context=subject,
            disallow_none_token=True,
        )
    precision_token = raw_precision.lower()
    alias_map = {
        "32": "32-true",
        "64": "64-true",
        "bf16": "bf16-mixed",
    }
    if precision_token == "16":
        precision_token = "bf16-mixed" if resolved_device_type == "cpu" else "16-mixed"
    elif precision_token == "16-mixed" and resolved_device_type == "cpu":
        precision_token = "bf16-mixed"
    else:
        precision_token = alias_map.get(precision_token, precision_token)

    config = {
        "16-mixed": RuntimePrecisionConfig(
            precision="16-mixed",
            model_dtype=None,
            input_dtype=None,
            autocast_dtype=torch.float16,
        ),
        "bf16-mixed": RuntimePrecisionConfig(
            precision="bf16-mixed",
            model_dtype=None,
            input_dtype=None,
            autocast_dtype=torch.bfloat16,
        ),
        "16-true": RuntimePrecisionConfig(
            precision="16-true",
            model_dtype=torch.float16,
            input_dtype=torch.float16,
            autocast_dtype=None,
        ),
        "32-true": RuntimePrecisionConfig(
            precision="32-true",
            model_dtype=torch.float32,
            input_dtype=torch.float32,
            autocast_dtype=None,
        ),
        "64-true": RuntimePrecisionConfig(
            precision="64-true",
            model_dtype=torch.float64,
            input_dtype=torch.float64,
            autocast_dtype=None,
        ),
        "bf16-true": RuntimePrecisionConfig(
            precision="bf16-true",
            model_dtype=torch.bfloat16,
            input_dtype=torch.bfloat16,
            autocast_dtype=None,
        ),
    }.get(precision_token)
    if config is None:
        raise ValueError(
            f"{subject} supports precision values "
            "16, 16-mixed, 16-true, 32, 32-true, 64, 64-true, bf16, "
            f"bf16-mixed, bf16-true; got {value!r}."
        )
    return config


def build_metric_w_name(
    metric_name: Any,
    *,
    key: str = "metric_name",
    context: str | None = None,
) -> str:
    """Build the canonical short worst-scenario companion field name."""
    parsed_metric_name = parse_required_nonempty_string(
        metric_name,
        key=key,
        context=context,
        disallow_none_token=True,
    )
    return f"{parsed_metric_name}_w"


def has_explicit_value(value: Any) -> bool:
    """Return whether a config/tag value was explicitly set to a non-null token."""
    if value is None:
        return False
    if isinstance(value, str):
        _, lowered = _normalize_token(value)
        return not _is_null_token(lowered)
    return True


def require_loss_metadata(
    tags: dict[str, Any],
    params: dict[str, Any],
    *,
    run_id: str,
) -> str:
    """Resolve loss from run tags/params with conflict detection.

    Checks ``loss_function`` tag and ``loss`` param and raises on conflict or
    if neither is present.
    """
    run_context = f"Run {run_id}"
    loss_tag_value = parse_optional_nonempty_string(
        tags.get("loss_function"),
        key="loss_function",
        context=run_context,
    )
    loss_param_value = parse_optional_nonempty_string(
        params.get("loss"),
        key="loss",
        context=run_context,
    )
    if (
        loss_tag_value is not None
        and loss_param_value is not None
        and loss_tag_value != loss_param_value
    ):
        raise ValueError(
            f"Run {run_id} has conflicting loss metadata: "
            f"loss_function='{loss_tag_value}' vs loss='{loss_param_value}'."
        )
    if loss_tag_value is not None:
        return loss_tag_value
    if loss_param_value is not None:
        return loss_param_value
    raise ValueError(
        f"Run {run_id} is missing required loss_function/loss metadata."
    )


def require_namespace_value(
    namespace: Any,
    *,
    key: str,
    context: str = "args",
) -> Any:
    """Require an argparse-style namespace attribute without adding code defaults."""
    if namespace is None:
        raise ValueError(f"{context} is required.")
    if not hasattr(namespace, key):
        raise ValueError(f"{context}.{key} is required.")
    return getattr(namespace, key)


def require_namespace_nonempty_string(
    namespace: Any,
    *,
    key: str,
    context: str = "args",
    disallow_none_token: bool = False,
) -> str:
    """Require a namespace attribute and normalize it to a non-empty string."""
    value = require_namespace_value(
        namespace,
        key=key,
        context=context,
    )
    parsed = parse_optional_nonempty_string(
        value,
        key=key,
        context=context,
        disallow_none_token=disallow_none_token,
    )
    if parsed is None:
        raise ValueError(f"{context}.{key} must be a non-empty string.")
    return parsed


def require_namespace_bool(
    namespace: Any,
    *,
    key: str,
    context: str = "args",
) -> bool:
    """Require a namespace attribute and parse it as a boolean."""
    value = require_namespace_value(
        namespace,
        key=key,
        context=context,
    )
    return parse_required_bool(
        value,
        key=key,
        context=context,
    )


def require_improvement_selection_mode(
    namespace: Any,
    *,
    context: str = "args",
) -> str:
    """Require and parse the configured robustness-improvement selector mode."""
    return parse_improvement_selection_mode(
        require_namespace_value(
            namespace,
            key="improvement_selection_mode",
            context=context,
        ),
        key=f"{context}.improvement_selection_mode",
    )


def require_nonempty_tag_value(
    tags: Mapping[str, Any],
    *,
    key: str,
    run_id: str,
) -> str:
    """Return a required tag value normalized to a non-empty string."""
    value = optional_nonempty_tag_value(tags, key=key)
    if value is None:
        raise ValueError(f"Run {run_id} is missing required {key} tag.")
    return value


def tag_is_truthy(tags: Mapping[str, Any], *, key: str) -> bool:
    """Return ``True`` when the tag value is one of the accepted true tokens."""
    raw_value = tags.get(key)
    if raw_value is None:
        return False
    return str(raw_value).strip().lower() in _TRUE_BOOL_TOKENS


def parse_required_bool(value: Any, *, key: str, context: str | None = None) -> bool:
    """Parse a required boolean value and raise on missing/invalid input."""
    subject = context if context else "Value"
    if value is None or not str(value).strip():
        raise ValueError(f"{subject} is missing required {key}.")
    try:
        parsed = parse_value(value, bool, key=key)
    except ValueError as exc:
        raise ValueError(
            f"{subject} has invalid {key} '{value}'. Expected a boolean value."
        ) from exc
    return bool(parsed)


def require_stage_tag(tags: dict[str, Any], *, run_id: str) -> str:
    """Extract and validate the required ``stage`` tag."""
    stage_value = tags.get("stage")
    if stage_value is None or not str(stage_value).strip():
        raise ValueError(f"Run {run_id} is missing required stage tag.")
    return str(stage_value).strip().lower()


def parse_winner_candidate_tags(
    tags: dict[str, Any],
    *,
    run_id: str,
) -> Optional[tuple[bool, bool]]:
    """Parse ``best_model``/``backbone_current`` winner tags.

    Returns ``None`` when both tags are absent/empty, allowing callers to skip
    non-candidate runs. Raises when exactly one tag is present, when either tag
    is invalid, or when the boolean values disagree.
    """
    best_model_raw = tags.get("best_model")
    backbone_current_raw = tags.get("backbone_current")
    has_best_model = best_model_raw is not None and str(best_model_raw).strip() != ""
    has_backbone_current = (
        backbone_current_raw is not None and str(backbone_current_raw).strip() != ""
    )

    if not has_best_model and not has_backbone_current:
        return None

    if has_best_model != has_backbone_current:
        present = "best_model" if has_best_model else "backbone_current"
        absent = "backbone_current" if has_best_model else "best_model"
        raise ValueError(f"Run {run_id} has {present} tag but is missing {absent} tag.")

    best_model = parse_required_bool(
        best_model_raw,
        key="best_model tag",
        context=f"Run {run_id}",
    )
    backbone_current = parse_required_bool(
        backbone_current_raw,
        key="backbone_current tag",
        context=f"Run {run_id}",
    )

    if best_model != backbone_current:
        raise ValueError(
            f"Run {run_id} has inconsistent winner tags: "
            f"best_model={str(best_model).lower()} and "
            f"backbone_current={str(backbone_current).lower()}."
        )

    return best_model, backbone_current


def require_robustness_scoring_semantics_tag(
    tags: dict[str, Any],
    *,
    run_id: str,
    expected: str = DEGRADATION_SCORING_SEMANTICS,
) -> str:
    """Require explicit robustness scoring semantics tag and exact value."""
    raw = tags.get("robustness_scoring_semantics")
    if raw is None or not str(raw).strip():
        raise ValueError(
            f"Run {run_id} is missing required robustness_scoring_semantics tag."
        )
    value = str(raw).strip()
    if value != expected:
        raise ValueError(
            f"Run {run_id} has robustness_scoring_semantics='{value}' but expected '{expected}'."
        )
    return value


def parse_scenario_metric_key(
    metric_key: str,
    *,
    scenario_prefix: str,
    run_id: str,
) -> Optional[tuple[int, str]]:
    """Parse ``scenario/<pert_idx>/<metric_label>`` metric keys."""
    if not metric_key.startswith(scenario_prefix):
        return None

    remainder = metric_key[len(scenario_prefix):]
    if "/" not in remainder:
        raise ValueError(
            f"Run {run_id} has malformed scenario metric key '{metric_key}'. "
            "Expected suffix '<pert_idx>/<metric_label>'."
        )
    scenario_token, metric_label = remainder.split("/", 1)
    scenario_token = str(scenario_token).strip()
    if not scenario_token:
        raise ValueError(
            f"Run {run_id} has malformed scenario metric key '{metric_key}'. "
            "Expected scenario/<pert_idx>/... with integer pert_idx."
        )
    try:
        scenario_idx = int(scenario_token)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Run {run_id} has malformed scenario metric key '{metric_key}'. "
            "Expected scenario/<pert_idx>/... with integer pert_idx."
        ) from exc
    if scenario_idx < 0:
        raise ValueError(
            f"Run {run_id} has malformed scenario metric key '{metric_key}'. "
            "Expected scenario/<pert_idx>/... with non-negative pert_idx."
        )

    metric_label = str(metric_label).strip()
    if not metric_label:
        raise ValueError(
            f"Run {run_id} has malformed scenario metric key '{metric_key}'. "
            "Missing metric label suffix."
        )

    return scenario_idx, metric_label


def parse_dropout_value(raw_value: Any, *, tol: float = 1e-8) -> float | None:
    """Parse a dropout value, returning None for missing/unparseable values."""
    try:
        value = parse_value(raw_value, float, key="dropout")
    except ValueError:
        return None
    if value is None:
        return None
    if abs(value) <= tol:
        return 0.0
    return value


def require_tested_param(params: dict[str, Any], *, run_id: str) -> bool:
    """Extract and validate the required ``tested`` param."""
    return parse_required_bool(
        params.get("tested"),
        key="'tested' param",
        context=f"Run {run_id}",
    )


# ---------------------------------------------------------------------------
# MLflow run-name / model-name serialization helpers
# ---------------------------------------------------------------------------

_MLFLOW_RUN_NAME_MAX_LENGTH = 250
_MODEL_NAME_INLINE_VALUE_MAX_LENGTH = 24
_RUN_NAME_DIGEST_HEX_LENGTH = 12


def _short_stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[
        :_RUN_NAME_DIGEST_HEX_LENGTH
    ]


def sanitize_model_name_fragment(value: str) -> str:
    """Replace characters that are problematic in MLflow run/model names."""
    return (
        value.replace(".", "-")
        .replace("[", "(")
        .replace("]", ")")
        .replace(", ", "-")
    )


def serialize_model_name_value(value: object) -> str:
    """Serialize a hyperparameter value for inclusion in a model name token."""
    normalized = normalize_yaml_value(value)
    if isinstance(normalized, (dict, list)):
        raw = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return f"h{_short_stable_digest(raw)}"

    raw = str(normalized)
    sanitized = sanitize_model_name_fragment(raw)
    if len(sanitized) > _MODEL_NAME_INLINE_VALUE_MAX_LENGTH:
        return f"h{_short_stable_digest(raw)}"
    return sanitized


def normalize_mlflow_run_name(
    run_name: str,
    *,
    max_length: int = _MLFLOW_RUN_NAME_MAX_LENGTH,
) -> str:
    """Normalize a run name to fit MLflow's SQL schema length limit."""
    name = str(run_name).strip()
    if not name:
        raise ValueError("run_name must be a non-empty string.")

    digest_suffix = f"__h{_short_stable_digest(name)}"
    if max_length <= len(digest_suffix):
        raise ValueError(
            "max_length must exceed the digest suffix length for run-name normalization."
        )
    if len(name) <= max_length:
        return name
    prefix_length = max_length - len(digest_suffix)
    return f"{name[:prefix_length]}{digest_suffix}"
