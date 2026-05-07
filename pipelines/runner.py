"""Pipeline runner for executing training recipes."""

from collections import Counter
from copy import copy
import json
import random
import tempfile
from dataclasses import dataclass
from itertools import product
from typing import Any, Optional

from mlflow.entities import ViewType

from config_loader import load_hparams
from data.perturbations import (
    PERTURBATION_REGISTRY,
)
from improvements import get_improvement_registration, list_available_improvements
from improvements.base import WrapExecutionOutcome
from utils.parsing import (
    parse_advtrain_loss,
    parse_advtrain_config,
    optional_nonempty_tag_value,
    parse_perturbation_channel_fraction_max,
    parse_required_choice,
    parse_required_bool,
    parse_required_dropout,
    parse_required_odd_positive_int,
    parse_required_positive_int,
    parse_max_hp_trials_per_model,
    parse_train_fault_profiles,
    parse_train_perturbation_probability,
    parse_train_perturbation_severity_max,
    resolve_train_perturbation_profile_config,
    require_tsmixer_hparams,
    parse_value,
)
from utils.rng import derive_tuning_seed

from .recipes import (
    PIPELINE_RECIPE_PATHS_BY_METHOD,
    load_pipeline_spec_for_method,
    require_pipeline_method_value,
)
from .scope import load_benchmark_method_architecture_applicability
from .specs import PipelineSpec
from .signatures import build_signature, compute_data_config_signature
from .ranking import rank_key, sort_runs_by_metric
from .training import (
    build_model_name,
    delete_runs,
    ensure_experiment_data_signature,
    merge_optimizer_hparams,
    optimizer_identity_hparams,
    resolve_runs,
    search_runs_all,
    finetune_single_run,
    train_single_run,
)


def param_overrides_for_spec(spec: PipelineSpec, args: Any) -> dict[str, Any]:
    if not hasattr(args, "_recipe_param_overrides"):
        raise ValueError(
            "Missing required args._recipe_param_overrides. "
            "Populate explicit recipe override metadata before building tuning scope."
        )
    explicit_overrides = getattr(args, "_recipe_param_overrides")
    if explicit_overrides is None:
        raise ValueError("args._recipe_param_overrides must not be None.")
    if not isinstance(explicit_overrides, dict):
        raise ValueError(
            "args._recipe_param_overrides must be a dictionary of explicit overrides."
        )
    recipe_params = spec.recipe_params
    if recipe_params is None:
        raise ValueError(
            f"Pipeline spec '{spec.pipeline_method}' has recipe_params=None; expected a mapping."
        )
    if not isinstance(recipe_params, dict):
        raise ValueError(
            f"Pipeline spec '{spec.pipeline_method}' recipe_params must be a mapping."
        )
    return {
        key: explicit_overrides[key]
        for key in recipe_params.keys()
        if key in explicit_overrides
    }


def resolve_wrap_base_pipeline_method(
    pipeline_method: str,
    param_values: dict[str, Any],
) -> str:
    """Resolve the base pipeline method for a wrap recipe from param values.

    Standalone helper usable by coverage auditing without instantiating a runner.
    """
    if pipeline_method == "randomized_smoothing":
        if "rs_backbone_method" not in param_values:
            raise ValueError(
                "randomized_smoothing recipe is missing required rs_backbone_method."
            )
        base_method = param_values["rs_backbone_method"]
        if base_method is None or not str(base_method).strip():
            raise ValueError("rs_backbone_method must be a non-empty string.")
        return str(base_method).strip()
    return "baseline"


def _require_pipeline_kind_value(
    pipeline_kind: Any,
    *,
    context: str,
) -> str:
    if pipeline_kind is None:
        raise ValueError(f"{context}: pipeline_kind must be a non-empty string.")
    kind = str(pipeline_kind).strip()
    if not kind:
        raise ValueError(f"{context}: pipeline_kind must be a non-empty string.")
    return kind


def scope_policy_skip_reason_for_spec(
    spec: PipelineSpec,
    architecture: str,
) -> Optional[str]:
    pipeline_method = require_pipeline_method_value(
        spec.pipeline_method,
        context="scope_policy_skip_reason_for_spec",
    )
    resolved_architecture = str(architecture).strip()
    if not resolved_architecture:
        raise ValueError("scope_policy_skip_reason_for_spec requires architecture.")
    applicability = load_benchmark_method_architecture_applicability()
    if pipeline_method not in applicability:
        raise ValueError(f"Unknown benchmark method '{pipeline_method}'.")
    known_architectures = {
        candidate
        for method_architectures in applicability.values()
        for candidate in method_architectures
    }
    if resolved_architecture not in known_architectures:
        raise ValueError(f"Unknown benchmark architecture '{resolved_architecture}'.")
    if resolved_architecture not in applicability[pipeline_method]:
        return "unsupported_benchmark_method_architecture"
    return None


@dataclass(frozen=True)
class _TuningCandidate:
    model_name: str
    pipeline_id: str
    pipeline_method: str
    pipeline_kind: str
    signature: str
    hparams: dict[str, Any]
    datamodule_kwargs: dict[str, Any]
    model_kwargs: dict[str, Any]
    param_values: dict[str, Any]


_TUNING_STRATEGY_RANDOM_SUBGRID = "random_subgrid"
_TUNING_CAMPAIGN_RESUME = "resume"
_TUNING_CAMPAIGN_FRESH_RERUN = "fresh_rerun"
_TUNING_SEED_POLICY = "v2"
_BASELINE_RECIPE_PATH = PIPELINE_RECIPE_PATHS_BY_METHOD["baseline"]


@dataclass(frozen=True)
class TuningScope:
    scope_key: str
    tuning_seed: int
    reference_budget: int
    target_budget: int
    pool_exhausted: bool
    signature_set: frozenset[str]


@dataclass(frozen=True)
class RunExecutionReport:
    dataset: str
    architecture: str
    pipeline_method: str
    pipeline_kind: str
    expected_units: int
    executed_units: int
    skipped_existing_units: int
    skipped_policy_units: int
    failed_units: int
    uncovered_units: int
    is_complete: bool


def _build_run_report_for_spec(
    *,
    spec: PipelineSpec,
    dataset: str,
    architecture: str,
    expected_units: int,
    executed_units: int,
    skipped_existing_units: int,
    skipped_policy_units: int,
    failed_units: int,
) -> RunExecutionReport:
    pipeline_method = require_pipeline_method_value(
        spec.pipeline_method,
        context="_build_run_report_for_spec",
    )
    pipeline_kind = _require_pipeline_kind_value(
        spec.pipeline_kind,
        context="_build_run_report_for_spec",
    )
    return PipelineRunner._finalize_run_report(
        dataset=dataset,
        architecture=architecture,
        pipeline_method=pipeline_method,
        pipeline_kind=pipeline_kind,
        expected_units=expected_units,
        executed_units=executed_units,
        skipped_existing_units=skipped_existing_units,
        skipped_policy_units=skipped_policy_units,
        failed_units=failed_units,
    )


class PipelineRunner:
    """Execute a pipeline spec and log to MLflow.

    Attributes:
        spec: The pipeline specification to execute.
        args: Command-line arguments / configuration namespace.
    """

    def __init__(self, spec: PipelineSpec, args: Any):
        self.spec = spec
        self.args = args
        self._last_run_report: Optional[RunExecutionReport] = None
        self._cached_train_fault_profiles: Optional[dict[str, tuple[str, ...]]] = None

    def get_last_run_report(self) -> Optional[RunExecutionReport]:
        return self._last_run_report

    @staticmethod
    def _finalize_run_report(
        *,
        dataset: str,
        architecture: str,
        pipeline_method: str,
        pipeline_kind: str,
        expected_units: int,
        executed_units: int,
        skipped_existing_units: int,
        skipped_policy_units: int,
        failed_units: int,
    ) -> RunExecutionReport:
        counters = {
            "expected_units": expected_units,
            "executed_units": executed_units,
            "skipped_existing_units": skipped_existing_units,
            "skipped_policy_units": skipped_policy_units,
            "failed_units": failed_units,
        }
        for name, value in counters.items():
            if value < 0:
                raise ValueError(f"Execution counter '{name}' must be non-negative.")
        resolved = (
            executed_units
            + skipped_existing_units
            + skipped_policy_units
            + failed_units
        )
        uncovered_units = expected_units - resolved
        if uncovered_units < 0:
            raise ValueError(
                "Execution counter identity violated: "
                f"expected={expected_units}, executed={executed_units}, "
                f"skipped_existing={skipped_existing_units}, "
                f"skipped_policy={skipped_policy_units}, failed={failed_units}."
            )
        is_complete = uncovered_units == 0 and failed_units == 0
        return RunExecutionReport(
            dataset=str(dataset),
            architecture=str(architecture),
            pipeline_method=str(pipeline_method),
            pipeline_kind=str(pipeline_kind),
            expected_units=int(expected_units),
            executed_units=int(executed_units),
            skipped_existing_units=int(skipped_existing_units),
            skipped_policy_units=int(skipped_policy_units),
            failed_units=int(failed_units),
            uncovered_units=int(uncovered_units),
            is_complete=bool(is_complete),
        )

    def _build_empty_tuning_scope(
        self,
        *,
        dataset_name: str,
        architecture: str,
        data_config_signature: str,
    ) -> TuningScope:
        scope_key = self._build_tuning_scope_key(
            dataset_name=dataset_name,
            architecture=architecture,
            data_config_signature=data_config_signature,
            pipeline_method=self.spec.pipeline_method,
            pipeline_kind=self.spec.pipeline_kind,
        )
        tuning_seed = derive_tuning_seed(
            base_seed=self.args.seed,
            dataset_key=dataset_name,
            architecture=architecture,
            data_config_signature=data_config_signature,
            pipeline_method=self.spec.pipeline_method,
            pipeline_kind=self.spec.pipeline_kind,
            tuning_strategy=_TUNING_STRATEGY_RANDOM_SUBGRID,
        )
        return TuningScope(
            scope_key=scope_key,
            tuning_seed=tuning_seed,
            reference_budget=0,
            target_budget=0,
            pool_exhausted=False,
            signature_set=frozenset(),
        )

    def _scope_policy_skip_reason(self, architecture: str) -> Optional[str]:
        return scope_policy_skip_reason_for_spec(self.spec, architecture)

    def _is_fault_augmentation_method(self) -> bool:
        method = require_pipeline_method_value(
            self.spec.pipeline_method,
            context="PipelineRunner._is_fault_augmentation_method",
        )
        return method == "fault_augmentation"

    def _is_adversarial_training_method(self) -> bool:
        return self.spec.pipeline_method == "adversarial_training"

    def _uses_adversarial_training_loss_validation(self) -> bool:
        return self.spec.pipeline_method == "adversarial_training"

    def _normalize_adversarial_training_params(
        self,
        param_values: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(param_values)
        cfg = parse_advtrain_config(
            param_values,
            context="Adversarial training recipe params",
        )
        normalized.update(
            {
                "advtrain_epsilon": cfg.epsilon,
                "advtrain_step_size": cfg.step_size,
                "advtrain_attack_steps": cfg.attack_steps,
                "advtrain_random_start": cfg.random_start,
                "advtrain_attack_channels": cfg.attack_channels,
            }
        )
        return normalized

    def _load_train_fault_profiles(self) -> dict[str, tuple[str, ...]]:
        if self._cached_train_fault_profiles is None:
            raw_profiles = self.spec.train_fault_profiles
            if raw_profiles is None:
                raise ValueError(
                    "fault_augmentation recipe is missing required "
                    "train_fault_profiles."
                )
            self._cached_train_fault_profiles = parse_train_fault_profiles(
                raw_profiles,
                registry_names=tuple(PERTURBATION_REGISTRY.keys()),
            )
        return self._cached_train_fault_profiles

    def _normalize_fault_augmentation_params(
        self,
        param_values: dict[str, Any],
    ) -> dict[str, Any]:
        profile, resolved_scenarios, scenarios_signature = (
            resolve_train_perturbation_profile_config(
                param_values.get("train_perturbation_profile"),
                profiles=self._load_train_fault_profiles(),
                registry_names=tuple(PERTURBATION_REGISTRY.keys()),
                profile_key="train_perturbation_profile",
                profiles_key="train_fault_profiles",
                scenarios_key="train_perturbation_scenarios",
                signature_key="train_perturbation_scenarios_signature",
            )
        )
        probability = parse_train_perturbation_probability(
            param_values.get("train_perturbation_probability"),
            key="train_perturbation_probability",
        )
        severity_max = parse_train_perturbation_severity_max(
            param_values.get("train_perturbation_severity_max"),
            key="train_perturbation_severity_max",
        )
        channel_fraction_max = parse_perturbation_channel_fraction_max(
            param_values.get("train_perturbation_channel_fraction_max"),
            key="train_perturbation_channel_fraction_max",
        )

        normalized = dict(param_values)
        normalized.update(
            {
                "train_perturbation_profile": profile,
                "train_perturbation_scenarios": list(resolved_scenarios),
                "train_perturbation_scenarios_signature": scenarios_signature,
                "train_perturbation_probability": probability,
                "train_perturbation_severity_max": severity_max,
                "train_perturbation_channel_fraction_max": channel_fraction_max,
            }
        )
        return normalized

    def _normalize_recipe_param_sets(
        self,
        param_sets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self._is_adversarial_training_method():
            return [
                self._normalize_adversarial_training_params(dict(param_values))
                for param_values in param_sets
            ]
        if not self._is_fault_augmentation_method():
            return [dict(param_values) for param_values in param_sets]
        return [
            self._normalize_fault_augmentation_params(dict(param_values))
            for param_values in param_sets
        ]

    def _invalid_hparams_reason(
        self,
        architecture: str,
        hparams: dict[str, Any],
    ) -> Optional[str]:
        def _parse(key: str, expected_type: type, default: Any = None) -> Any:
            value = hparams.get(key, default)
            try:
                return parse_value(value, expected_type, key=key)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid hyperparameters for {architecture}: {exc}"
                ) from exc

        arch = architecture.strip().lower()
        if arch == "gru":
            dropout = _parse("dropout", float)
            if dropout is None:
                return "missing dropout for GRU; specify dropout explicitly"
            if dropout > 0:
                n_layers = _parse("n_layers", int)
                if n_layers is None:
                    return "missing n_layers for GRU; specify n_layers explicitly"
                if n_layers <= 1:
                    return (
                        f"dropout={dropout} with n_layers={n_layers} "
                        "is invalid for single-layer RNNs"
                    )

        if arch == "patchtst":
            required_keys = (
                "d_model",
                "d_ff",
                "n_layers_enc",
                "n_heads",
                "patch_len",
                "stride",
                "dropout",
                "factor",
                "activation",
            )
            missing_keys = [key for key in required_keys if key not in hparams]
            if missing_keys:
                # Missing required keys means malformed hparams grid/config, not
                # a candidate-value mismatch. Keep this as a hard error.
                raise ValueError(
                    "Invalid hyperparameters for PatchTST: missing required key(s): "
                    + ", ".join(sorted(missing_keys))
                )
            if self.spec.pipeline_kind == "train" and "lr" not in hparams:
                raise ValueError(
                    "Invalid hyperparameters for PatchTST: missing required key(s): lr"
                )

            try:
                d_model = parse_required_positive_int(
                    hparams["d_model"],
                    key="d_model",
                )
                parse_required_positive_int(hparams["d_ff"], key="d_ff")
                parse_required_positive_int(hparams["n_layers_enc"], key="n_layers_enc")
                n_heads = parse_required_positive_int(hparams["n_heads"], key="n_heads")
                patch_len = parse_required_positive_int(hparams["patch_len"], key="patch_len")
                parse_required_positive_int(hparams["stride"], key="stride")
                parse_required_dropout(hparams["dropout"], key="dropout")
                parse_required_positive_int(hparams["factor"], key="factor")
                parse_required_choice(
                    hparams["activation"],
                    key="activation",
                    allowed=("gelu", "relu"),
                )
            except ValueError as exc:
                return str(exc)

            lr = _parse("lr", float) if "lr" in hparams else None

            if d_model % n_heads != 0:
                return (
                    f"d_model={d_model} must be divisible by n_heads={n_heads}"
                )
            if self.spec.pipeline_kind == "train" and lr is None:
                raise ValueError(
                    "Invalid hyperparameters for PatchTST: lr must be provided."
                )

            d_seq_in = parse_value(
                getattr(self.args, "input_len", None),
                int,
                key="input_len",
            )
            if d_seq_in is None:
                raise ValueError(
                    "Invalid hyperparameters for PatchTST: args.input_len must be provided."
                )
            if patch_len > d_seq_in:
                return f"patch_len={patch_len} exceeds d_seq_in={d_seq_in}"

        if arch == "moderntcn":
            required_keys = (
                "d_model",
                "num_blocks",
                "large_size",
                "small_size",
                "ffn_ratio",
                "patch_size",
                "patch_stride",
                "dropout",
                "head_dropout",
                "individual",
            )
            missing_keys = [key for key in required_keys if key not in hparams]
            if missing_keys:
                raise ValueError(
                    "Invalid hyperparameters for ModernTCN: missing required key(s): "
                    + ", ".join(sorted(missing_keys))
                )
            if self.spec.pipeline_kind == "train" and "lr" not in hparams:
                raise ValueError(
                    "Invalid hyperparameters for ModernTCN: missing required key(s): lr"
                )

            try:
                large_size = parse_required_odd_positive_int(
                    hparams["large_size"],
                    key="large_size",
                )
                small_size = parse_required_odd_positive_int(
                    hparams["small_size"],
                    key="small_size",
                )
                patch_size = parse_required_positive_int(
                    hparams["patch_size"],
                    key="patch_size",
                )
                patch_stride = parse_required_positive_int(
                    hparams["patch_stride"],
                    key="patch_stride",
                )
                parse_required_positive_int(hparams["d_model"], key="d_model")
                parse_required_positive_int(hparams["num_blocks"], key="num_blocks")
                parse_required_positive_int(hparams["ffn_ratio"], key="ffn_ratio")
                parse_required_dropout(hparams["dropout"], key="dropout")
                parse_required_dropout(hparams["head_dropout"], key="head_dropout")
                parse_required_bool(
                    hparams["individual"],
                    key="individual",
                    context="ModernTCN",
                )
            except ValueError as exc:
                return str(exc)

            if small_size > large_size:
                return (
                    f"small_size={small_size} must be <= large_size={large_size}"
                )
            if patch_stride > patch_size:
                return (
                    f"patch_stride={patch_stride} must be <= patch_size={patch_size}"
                )

            d_seq_in = parse_value(
                getattr(self.args, "input_len", None),
                int,
                key="input_len",
            )
            if d_seq_in is None:
                raise ValueError(
                    "Invalid hyperparameters for ModernTCN: args.input_len must be provided."
                )
            if patch_size > d_seq_in:
                return f"patch_size={patch_size} exceeds d_seq_in={d_seq_in}"

        if arch == "tsmixer":
            try:
                require_tsmixer_hparams(hparams)
            except ValueError as exc:
                return str(exc)

        if self._uses_adversarial_training_loss_validation():
            loss_value = hparams.get("loss")
            if loss_value is None:
                loss_value = self.args.loss
            parse_advtrain_loss(
                loss_value,
                key="loss",
                context=f"{self.spec.pipeline_method}/{architecture}",
            )

        return None

    def _build_training_candidates(
        self,
        *,
        architecture: str,
        dataset_name: str,
        data_config_signature: str,
        param_sets: list[dict[str, Any]],
        hparams_mode: str,
        hparams_grid: dict[str, Any],
        backbone_run_id: Optional[str],
        finetune_epochs: Optional[int],
        finetune_lr_factor: Optional[float],
    ) -> list[_TuningCandidate]:
        candidates: list[_TuningCandidate] = []
        seen_signatures: set[str] = set()
        skip_counts: Counter[str] = Counter()
        is_finetune_recipe = self.spec.pipeline_kind == "finetune"
        for param_values in param_sets:
            formatted_pipeline_id = self.spec.format_pipeline_id(param_values)
            pipeline_id = formatted_pipeline_id
            pipeline_kind = self.spec.pipeline_kind
            datamodule_kwargs = self.spec.render_kwargs(
                self.spec.datamodule_kwargs, param_values
            )
            if self._is_fault_augmentation_method():
                datamodule_kwargs["train_fault_profiles"] = self.spec.train_fault_profiles
            model_kwargs = self.spec.render_kwargs(
                self.spec.model_kwargs, param_values
            )
            if "loss" not in model_kwargs and getattr(self.args, "loss", None) is not None:
                model_kwargs["loss"] = self.args.loss

            signature_params: Optional[dict[str, Any]] = param_values or None
            if is_finetune_recipe:
                if backbone_run_id is None:
                    raise ValueError(
                        "backbone_run_id is required for finetune candidate generation."
                    )
                if finetune_epochs is None:
                    raise ValueError(
                        "finetune_epochs is required for finetune candidate generation."
                    )
                if finetune_lr_factor is None:
                    raise ValueError(
                        "finetune_lr_factor is required for finetune candidate generation."
                    )
                pipeline_id = pipeline_id + self._format_finetune_schedule_suffix(
                    finetune_epochs=int(finetune_epochs),
                    finetune_lr_factor=float(finetune_lr_factor),
                )
                signature_params = dict(param_values)
                signature_params["finetune_epochs"] = int(finetune_epochs)
                signature_params["finetune_lr_factor"] = float(finetune_lr_factor)
                signature_params["backbone_run_id"] = str(backbone_run_id)

            if hparams_mode == "baseline_grid":
                list_keys = [k for k, v in hparams_grid.items() if isinstance(v, list)]
                list_values = [hparams_grid[k] for k in list_keys]
                direct_values = {
                    k: v for k, v in hparams_grid.items() if not isinstance(v, list)
                }
                for combo in product(*list_values):
                    combo_dict = dict(zip(list_keys, combo))
                    merged_hparams = dict(direct_values)
                    merged_hparams.update(combo_dict)
                    if model_kwargs:
                        merged_hparams.update(model_kwargs)
                    merged_hparams = merge_optimizer_hparams(
                        merged_hparams,
                        self.args,
                        model_architecture=architecture,
                    )
                    identity_hparams = optimizer_identity_hparams(merged_hparams)
                    invalid_reason = self._invalid_hparams_reason(
                        architecture, merged_hparams
                    )
                    if invalid_reason:
                        skip_counts[f"{architecture}: {invalid_reason}"] += 1
                        continue
                    model_name = build_model_name(architecture, identity_hparams)
                    model_name = f"{pipeline_id}_{model_name}"
                    signature = build_signature(
                        architecture,
                        dataset_name,
                        identity_hparams,
                        pipeline_id=pipeline_id,
                        data_config_signature=data_config_signature,
                        recipe_params=signature_params,
                    )
                    if signature in seen_signatures:
                        raise ValueError(
                            f"Duplicate candidate signature '{signature}' generated for "
                            f"pipeline_id='{pipeline_id}' ({architecture}, {dataset_name})."
                        )
                    seen_signatures.add(signature)
                    candidates.append(
                        _TuningCandidate(
                            model_name=model_name,
                            pipeline_id=pipeline_id,
                            pipeline_method=self.spec.pipeline_method,
                            pipeline_kind=pipeline_kind,
                            signature=signature,
                            hparams=merged_hparams,
                            datamodule_kwargs=datamodule_kwargs,
                            model_kwargs=model_kwargs,
                            param_values=dict(param_values),
                        )
                    )
                continue

            if hparams_mode != "inherit_baseline":
                raise ValueError(
                    f"Unknown model_hparams.mode '{hparams_mode}' in pipeline spec."
                )
            effective_hparams = dict(hparams_grid)
            if model_kwargs:
                effective_hparams.update(model_kwargs)
            effective_hparams = merge_optimizer_hparams(
                effective_hparams,
                self.args,
                model_architecture=architecture,
            )
            identity_hparams = optimizer_identity_hparams(effective_hparams)
            invalid_reason = self._invalid_hparams_reason(architecture, effective_hparams)
            if invalid_reason:
                skip_counts[f"{architecture}: {invalid_reason}"] += 1
                continue
            model_name = build_model_name(architecture, identity_hparams)
            model_name = f"{pipeline_id}_{model_name}"
            signature = build_signature(
                architecture,
                dataset_name,
                identity_hparams,
                pipeline_id=pipeline_id,
                data_config_signature=data_config_signature,
                recipe_params=signature_params,
            )
            if signature in seen_signatures:
                raise ValueError(
                    f"Duplicate candidate signature '{signature}' generated for "
                    f"pipeline_id='{pipeline_id}' ({architecture}, {dataset_name})."
                )
            seen_signatures.add(signature)
            candidates.append(
                _TuningCandidate(
                    model_name=model_name,
                    pipeline_id=pipeline_id,
                    pipeline_method=self.spec.pipeline_method,
                    pipeline_kind=pipeline_kind,
                    signature=signature,
                    hparams=effective_hparams,
                    datamodule_kwargs=datamodule_kwargs,
                    model_kwargs=model_kwargs,
                    param_values=dict(param_values),
                )
            )
        if skip_counts:
            for reason, count in sorted(skip_counts.items()):
                print(f"Skipped {count} candidate(s): {reason}.")
        return candidates

    @staticmethod
    def _format_finetune_schedule_suffix(
        *,
        finetune_epochs: int,
        finetune_lr_factor: float,
    ) -> str:
        return f"_ft{int(finetune_epochs)}_lrf{float(finetune_lr_factor)}"

    def _resolve_active_finetune_schedule(self) -> tuple[int, float]:
        finetune_epochs = getattr(self.args, "finetune_epochs", None)
        finetune_lr_factor = getattr(self.args, "finetune_lr_factor", None)
        if finetune_epochs is None:
            raise ValueError("finetune_epochs must be set for finetune-kind recipes.")
        if finetune_lr_factor is None:
            raise ValueError("finetune_lr_factor must be set for finetune-kind recipes.")
        return int(finetune_epochs), float(finetune_lr_factor)

    def _filter_candidates_before_scheduling(
        self,
        *,
        dataset_spec: Any,
        architecture: str,
        data_config_signature: str,
        candidates: list[_TuningCandidate],
    ) -> list[_TuningCandidate]:
        return candidates

    @staticmethod
    def _build_tuning_scope_key(
        *,
        dataset_name: str,
        architecture: str,
        data_config_signature: str,
        pipeline_method: str,
        pipeline_kind: str,
    ) -> str:
        return (
            f"{dataset_name}:{architecture}:{data_config_signature}:"
            f"{pipeline_method}:{pipeline_kind}:{_TUNING_STRATEGY_RANDOM_SUBGRID}"
        )

    def _collect_consumed_signatures(
        self,
        *,
        client: Any,
        experiment_id: str,
        dataset_name: str,
        architecture: str,
        data_config_signature: str,
        pipeline_method: str,
        pipeline_kind: str,
        tuning_campaign: Optional[str] = None,
        candidate_signatures: Optional[set[str]] = None,
        expected_base_pipeline_method: Optional[str] = None,
        expected_hparams_mode: Optional[str] = None,
        expected_tuning_scope_key: Optional[str] = None,
        expected_tuning_seed: Optional[int] = None,
        expected_tuning_campaign: Optional[str] = None,
    ) -> tuple[set[str], list[Any]]:
        if tuning_campaign not in (
            None,
            _TUNING_CAMPAIGN_RESUME,
            _TUNING_CAMPAIGN_FRESH_RERUN,
        ):
            raise ValueError(
                "tuning_campaign must be one of: "
                f"{_TUNING_CAMPAIGN_RESUME}, {_TUNING_CAMPAIGN_FRESH_RERUN}, or None."
            )
        filter_parts = [
            "tags.stage = 'train'",
            f"tags.dataset = '{dataset_name}'",
            f"tags.model_architecture = '{architecture}'",
            f"tags.pipeline_method = '{pipeline_method}'",
            f"tags.pipeline_kind = '{pipeline_kind}'",
            f"tags.data_config_signature = '{data_config_signature}'",
        ]
        if tuning_campaign is not None:
            filter_parts.append(
                f"tags.tuning_strategy = '{_TUNING_STRATEGY_RANDOM_SUBGRID}'"
            )
            filter_parts.append(f"tags.tuning_campaign = '{tuning_campaign}'")
        filter_string = " AND ".join(filter_parts)
        runs = search_runs_all(
            client,
            [experiment_id],
            filter_string=filter_string,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=1000,
        )
        best_run_by_signature: dict[str, Any] = {}
        duplicate_runs: list[Any] = []
        for run in runs:
            if run.data.tags.get("mlflow.parentRunId"):
                continue
            status = str(getattr(run.info, "status", "") or "").upper()
            if status in {"RUNNING", "SCHEDULED"}:
                # Do not consume in-progress attempts. They must still be checked
                # by per-signature duplicate/run-state handling before training.
                continue
            if status != "FINISHED":
                continue
            metrics = run.data.metrics or {}
            if "best_val_loss" not in metrics:
                print(
                    f"Ignoring run {run.info.run_id} for resume budgeting: "
                    "missing best_val_loss."
                )
                continue
            signature_raw = (run.data.tags or {}).get("signature")
            if signature_raw is None or not str(signature_raw).strip():
                raise ValueError(
                    f"Run {run.info.run_id} in tuning scope is missing required signature tag."
                )
            signature = str(signature_raw).strip()
            if candidate_signatures is not None and signature not in candidate_signatures:
                continue
            reason = self._run_reuse_block_reason(
                run,
                expected_base_pipeline_method=expected_base_pipeline_method,
                expected_hparams_mode=expected_hparams_mode,
                expected_tuning_scope_key=expected_tuning_scope_key,
                expected_tuning_seed=expected_tuning_seed,
                expected_tuning_campaign=expected_tuning_campaign,
            )
            if reason:
                print(
                    f"Ignoring run {run.info.run_id} for resume budgeting "
                    f"(signature={signature}): {reason}."
                )
                continue
            existing = best_run_by_signature.get(signature)
            if existing is not None:
                existing_key = rank_key(existing, require_metric=False)
                current_key = rank_key(run, require_metric=False)
                if current_key < existing_key:
                    duplicate_runs.append(existing)
                    best_run_by_signature[signature] = run
                else:
                    duplicate_runs.append(run)
            else:
                best_run_by_signature[signature] = run
        return set(best_run_by_signature.keys()), duplicate_runs

    @classmethod
    def _non_baseline_lineage_reuse_block_reason(
        cls,
        tags: dict[str, Any],
        *,
        expected_base_pipeline_method: Optional[str] = None,
        expected_hparams_mode: Optional[str] = None,
    ) -> Optional[str]:
        base_pipeline_method = optional_nonempty_tag_value(
            tags,
            key="base_pipeline_method",
        )
        if base_pipeline_method is None:
            return "missing_base_pipeline_method"
        if (
            expected_base_pipeline_method is not None
            and base_pipeline_method != expected_base_pipeline_method
        ):
            return (
                "base_pipeline_method_mismatch:"
                f"{base_pipeline_method}!={expected_base_pipeline_method}"
            )

        hparams_mode = optional_nonempty_tag_value(tags, key="hparams_mode")
        if hparams_mode is None:
            return "missing_hparams_mode"
        if expected_hparams_mode is not None and hparams_mode != expected_hparams_mode:
            return f"hparams_mode_mismatch:{hparams_mode}!={expected_hparams_mode}"
        if hparams_mode == "baseline_grid":
            return None
        if hparams_mode == "inherit_baseline":
            baseline_hparams_run_id = optional_nonempty_tag_value(
                tags,
                key="baseline_hparams_run_id",
            )
            if baseline_hparams_run_id is None:
                return "missing_baseline_hparams_run_id"
            return None
        return f"unknown_hparams_mode:{hparams_mode}"

    @classmethod
    def _tuning_scope_reuse_block_reason(
        cls,
        tags: dict[str, Any],
        *,
        expected_tuning_scope_key: Optional[str] = None,
        expected_tuning_seed: Optional[int] = None,
        expected_tuning_campaign: Optional[str] = None,
    ) -> Optional[str]:
        if (
            expected_tuning_scope_key is None
            and expected_tuning_seed is None
            and expected_tuning_campaign is None
        ):
            return None
        tuning_strategy = optional_nonempty_tag_value(tags, key="tuning_strategy")
        if tuning_strategy is None:
            return "missing_tuning_strategy"
        if tuning_strategy != _TUNING_STRATEGY_RANDOM_SUBGRID:
            return (
                "tuning_strategy_mismatch:"
                f"{tuning_strategy}!={_TUNING_STRATEGY_RANDOM_SUBGRID}"
            )

        if expected_tuning_scope_key is not None:
            scope_key = optional_nonempty_tag_value(tags, key="tuning_scope_key")
            if scope_key is None:
                return "missing_tuning_scope_key"
            if scope_key != expected_tuning_scope_key:
                return (
                    "tuning_scope_key_mismatch:"
                    f"{scope_key}!={expected_tuning_scope_key}"
                )

        if expected_tuning_seed is not None:
            tuning_seed = optional_nonempty_tag_value(tags, key="tuning_seed")
            if tuning_seed is None:
                return "missing_tuning_seed"
            expected_seed_token = str(expected_tuning_seed)
            if tuning_seed != expected_seed_token:
                return (
                    "tuning_seed_mismatch:"
                    f"{tuning_seed}!={expected_seed_token}"
                )

        if expected_tuning_campaign is not None:
            campaign = optional_nonempty_tag_value(tags, key="tuning_campaign")
            if campaign is None:
                return "missing_tuning_campaign"
            if campaign != expected_tuning_campaign:
                return (
                    "tuning_campaign_mismatch:"
                    f"{campaign}!={expected_tuning_campaign}"
                )
        return None

    @classmethod
    def _run_reuse_block_reason(
        cls,
        run: Any,
        *,
        expected_base_pipeline_method: Optional[str] = None,
        expected_hparams_mode: Optional[str] = None,
        expected_tuning_scope_key: Optional[str] = None,
        expected_tuning_seed: Optional[int] = None,
        expected_tuning_campaign: Optional[str] = None,
    ) -> Optional[str]:
        tags = run.data.tags or {}
        tuning_scope_reason = cls._tuning_scope_reuse_block_reason(
            tags,
            expected_tuning_scope_key=expected_tuning_scope_key,
            expected_tuning_seed=expected_tuning_seed,
            expected_tuning_campaign=expected_tuning_campaign,
        )
        if tuning_scope_reason is not None:
            return tuning_scope_reason
        pipeline_method = optional_nonempty_tag_value(tags, key="pipeline_method")
        if pipeline_method is None:
            return "missing_pipeline_method"
        if pipeline_method == "baseline":
            return None
        return cls._non_baseline_lineage_reuse_block_reason(
            tags,
            expected_base_pipeline_method=expected_base_pipeline_method,
            expected_hparams_mode=expected_hparams_mode,
        )

    def _resolve_tuning_campaign(
        self,
        *,
        rerun_requested: bool,
    ) -> str:
        if rerun_requested:
            return _TUNING_CAMPAIGN_FRESH_RERUN
        return _TUNING_CAMPAIGN_RESUME

    def _baseline_reference_candidates(
        self,
        *,
        dataset_name: str,
        architecture: str,
        data_config_signature: str,
    ) -> list["_TuningCandidate"]:
        if not _BASELINE_RECIPE_PATH.exists():
            raise FileNotFoundError(
                f"Baseline recipe not found at '{_BASELINE_RECIPE_PATH}'."
            )
        baseline_spec = PipelineSpec.from_yaml(_BASELINE_RECIPE_PATH)
        baseline_runner = PipelineRunner(baseline_spec, self.args)
        baseline_hparams_grid = load_hparams().get(architecture)
        if baseline_hparams_grid is None:
            raise ValueError(
                f"No hyperparameters found for architecture '{architecture}' "
                "in configs/baseline_hparams.yaml."
            )
        baseline_param_sets = baseline_spec.expand_params(
            overrides=param_overrides_for_spec(baseline_spec, self.args)
        )
        return baseline_runner._build_training_candidates(
            architecture=architecture,
            dataset_name=dataset_name,
            data_config_signature=data_config_signature,
            param_sets=baseline_param_sets,
            hparams_mode="baseline_grid",
            hparams_grid=baseline_hparams_grid,
            backbone_run_id=None,
            finetune_epochs=None,
            finetune_lr_factor=None,
        )

    def _count_baseline_reference_candidates(
        self,
        *,
        dataset_name: str,
        architecture: str,
        data_config_signature: str,
    ) -> int:
        return len(self._baseline_reference_candidates(
            dataset_name=dataset_name,
            architecture=architecture,
            data_config_signature=data_config_signature,
        ))

    @staticmethod
    def _compute_budget_targets(
        *,
        max_hp_trials_per_model: Optional[int],
        baseline_candidate_count: int,
        method_candidate_count: int,
    ) -> tuple[int, int, bool]:
        if baseline_candidate_count < 0:
            raise ValueError("baseline_candidate_count must be non-negative.")
        if method_candidate_count < 0:
            raise ValueError("method_candidate_count must be non-negative.")
        if max_hp_trials_per_model is None:
            reference_budget = baseline_candidate_count
        else:
            reference_budget = min(max_hp_trials_per_model, baseline_candidate_count)
        target_budget = min(reference_budget, method_candidate_count)
        pool_exhausted = method_candidate_count < reference_budget
        return reference_budget, target_budget, pool_exhausted

    @staticmethod
    def _select_seeded_candidates(
        *,
        candidates: list[_TuningCandidate],
        target_budget: int,
        tuning_seed: int,
    ) -> list[_TuningCandidate]:
        if target_budget < 0:
            raise ValueError("target_budget must be non-negative.")
        if target_budget == 0:
            return []
        indices = list(range(len(candidates)))
        rng = random.Random(tuning_seed)
        rng.shuffle(indices)
        selected_indices = indices[:target_budget]
        return [candidates[idx] for idx in selected_indices]

    @staticmethod
    def _select_random_unseen_candidates(
        *,
        candidates: list[_TuningCandidate],
        consumed_signatures: set[str],
        remaining_budget: int,
        tuning_seed: int,
    ) -> list[_TuningCandidate]:
        if remaining_budget <= 0:
            return []
        indices = list(range(len(candidates)))
        rng = random.Random(tuning_seed)
        rng.shuffle(indices)
        selected: list[_TuningCandidate] = []
        for idx in indices:
            candidate = candidates[idx]
            if candidate.signature in consumed_signatures:
                continue
            selected.append(candidate)
            if len(selected) >= remaining_budget:
                break
        return selected

    def expected_tuning_scope(
        self,
        *,
        client: Any,
        experiment_id: str,
        dataset_spec: Any,
        architecture: str,
        data_config_signature: str,
    ) -> TuningScope:
        if self.spec.pipeline_kind == "wrap":
            raise ValueError(
                "expected_tuning_scope is only defined for training-kind pipelines."
            )
        dataset_name = dataset_spec.key
        skip_reason = self._scope_policy_skip_reason(architecture)
        if skip_reason is not None:
            print(
                f"Expected tuning scope is empty for '{self.spec.pipeline_method}' on "
                f"{architecture} ({dataset_name}): {skip_reason}."
            )
            return self._build_empty_tuning_scope(
                dataset_name=dataset_name,
                architecture=architecture,
                data_config_signature=data_config_signature,
            )
        param_overrides = param_overrides_for_spec(self.spec, self.args)
        param_sets = self._normalize_recipe_param_sets(
            self.spec.expand_params(overrides=param_overrides)
        )
        active_finetune_epochs = None
        active_finetune_lr_factor = None
        if self.spec.pipeline_kind == "finetune":
            (
                active_finetune_epochs,
                active_finetune_lr_factor,
            ) = self._resolve_active_finetune_schedule()
        hparams_mode = self.spec.model_hparams_mode
        backbone_run_id: Optional[str] = None
        if hparams_mode == "baseline_grid":
            hparams_grid = load_hparams().get(architecture)
            if hparams_grid is None:
                raise ValueError(
                    f"No hyperparameters found for architecture '{architecture}' "
                    "in configs/baseline_hparams.yaml."
                )
        elif hparams_mode == "inherit_baseline":
            if self.spec.pipeline_kind == "finetune":
                backbone_run = self._find_best_baseline_run(
                    client,
                    experiment_id,
                    dataset_spec,
                    architecture,
                    data_config_signature=data_config_signature,
                )
                hparams_grid = self._extract_hparams_from_run(
                    client,
                    backbone_run,
                    architecture,
                )
                backbone_run_id = backbone_run.info.run_id
            else:
                hparams_grid, _ = self.find_baseline_hparams(
                    client,
                    experiment_id,
                    dataset_spec,
                    architecture,
                    data_config_signature=data_config_signature,
                )
                backbone_run_id = None
        else:
            raise ValueError(
                f"Unknown model_hparams.mode '{self.spec.model_hparams_mode}' in pipeline spec."
            )
        candidates = self._build_training_candidates(
            architecture=architecture,
            dataset_name=dataset_name,
            data_config_signature=data_config_signature,
            param_sets=param_sets,
            hparams_mode=hparams_mode,
            hparams_grid=hparams_grid,
            backbone_run_id=backbone_run_id,
            finetune_epochs=active_finetune_epochs,
            finetune_lr_factor=active_finetune_lr_factor,
        )
        baseline_candidate_count = self._count_baseline_reference_candidates(
            dataset_name=dataset_name,
            architecture=architecture,
            data_config_signature=data_config_signature,
        )
        max_hp_trials_per_model = parse_max_hp_trials_per_model(
            getattr(self.args, "max_hp_trials_per_model", None)
        )
        (
            reference_budget,
            target_budget,
            pool_exhausted,
        ) = self._compute_budget_targets(
            max_hp_trials_per_model=max_hp_trials_per_model,
            baseline_candidate_count=baseline_candidate_count,
            method_candidate_count=len(candidates),
        )
        scope_key = self._build_tuning_scope_key(
            dataset_name=dataset_name,
            architecture=architecture,
            data_config_signature=data_config_signature,
            pipeline_method=self.spec.pipeline_method,
            pipeline_kind=self.spec.pipeline_kind,
        )
        tuning_seed = derive_tuning_seed(
            base_seed=self.args.seed,
            dataset_key=dataset_name,
            architecture=architecture,
            data_config_signature=data_config_signature,
            pipeline_method=self.spec.pipeline_method,
            pipeline_kind=self.spec.pipeline_kind,
            tuning_strategy=_TUNING_STRATEGY_RANDOM_SUBGRID,
        )
        selected_candidates = self._select_seeded_candidates(
            candidates=candidates,
            target_budget=target_budget,
            tuning_seed=tuning_seed,
        )
        return TuningScope(
            scope_key=scope_key,
            tuning_seed=tuning_seed,
            reference_budget=reference_budget,
            target_budget=target_budget,
            pool_exhausted=pool_exhausted,
            signature_set=frozenset(
                candidate.signature for candidate in selected_candidates
            ),
        )

    @staticmethod
    def _build_tuning_tags(
        *,
        reference_budget: int,
        target_budget: int,
        trial_number: int,
        tuning_seed: int,
        scope_key: str,
        campaign: str,
        pool_exhausted: bool,
        early_stopping: bool,
    ) -> dict[str, str]:
        tags = {
            "tuning_strategy": _TUNING_STRATEGY_RANDOM_SUBGRID,
            "tuning_budget_target": str(target_budget),
            "tuning_reference_budget": str(reference_budget),
            "tuning_trial_number": str(trial_number),
            "tuning_seed": str(tuning_seed),
            "tuning_seed_policy": _TUNING_SEED_POLICY,
            "tuning_scope_key": scope_key,
            "tuning_campaign": campaign,
            "tuning_full_fidelity": "true",
            "tuning_early_stopping": str(bool(early_stopping)).lower(),
        }
        if pool_exhausted:
            tags["tuning_pool_exhausted"] = "true"
        return tags

    def _handle_existing_runs(
        self,
        client: Any,
        experiment_id: str,
        signature: str,
        model_name: str,
        rerun_requested: bool,
        expected_base_pipeline_method: Optional[str] = None,
        expected_hparams_mode: Optional[str] = None,
        expected_tuning_scope_key: Optional[str] = None,
        expected_tuning_seed: Optional[int] = None,
        expected_tuning_campaign: Optional[str] = None,
    ) -> bool:
        """Check for existing runs and handle skip/rerun logic.

        Args:
            client: MLflow client instance.
            experiment_id: MLflow experiment ID.
            signature: Run signature for deduplication.
            model_name: Human-readable model name for logging.
            rerun_requested: Whether --rerun flag was set.

        Returns:
            True if training should be skipped (existing finished run found),
            False if training should proceed.

        Raises:
            RuntimeError: If RUNNING/SCHEDULED runs are found with the same signature.
        """
        resolved = resolve_runs(client, experiment_id, signature)

        if resolved.running:
            running_ids = ", ".join(run.info.run_id for run in resolved.running)
            raise RuntimeError(
                f"Encountered unexpected RUNNING/SCHEDULED MLflow runs for signature '{signature}'. "
                f"Active run IDs: {running_ids}"
            )

        if rerun_requested:
            parents_to_delete = [
                resolved.canonical,
                *resolved.failed,
                *resolved.duplicates,
            ]
            delete_runs(
                client,
                parents_to_delete,
                reason="rerun requested",
            )
            return False

        finished_candidates = [
            run
            for run in [resolved.canonical, *resolved.duplicates]
            if run is not None and run.info.status == "FINISHED"
        ]
        reusable_finished = [
            run
            for run in finished_candidates
            if self._run_reuse_block_reason(
                run,
                expected_base_pipeline_method=expected_base_pipeline_method,
                expected_hparams_mode=expected_hparams_mode,
                expected_tuning_scope_key=expected_tuning_scope_key,
                expected_tuning_seed=expected_tuning_seed,
                expected_tuning_campaign=expected_tuning_campaign,
            )
            is None
        ]
        reusable_existing = reusable_finished[0] if reusable_finished else None

        duplicates_to_delete = [
            run
            for run in resolved.duplicates
        ]
        if reusable_existing is not None:
            duplicates_to_delete = [
                run
                for run in duplicates_to_delete
                if run.info.run_id != reusable_existing.info.run_id
            ]

        if duplicates_to_delete:
            delete_runs(
                client,
                duplicates_to_delete,
                reason="duplicate signature",
            )
        if resolved.failed:
            delete_runs(
                client,
                list(resolved.failed),
                reason="retry after failure",
            )
        if reusable_existing is not None:
            if (
                resolved.canonical is not None
                and resolved.canonical.info.run_id != reusable_existing.info.run_id
            ):
                print(
                    f"Model '{model_name}' found reusable existing run "
                    f"(Run ID: {reusable_existing.info.run_id}) for signature '{signature}', "
                    f"while top-scored run {resolved.canonical.info.run_id} is non-reusable. "
                    "Skipping retraining."
                )
            else:
                print(
                    f"Model '{model_name}' already trained "
                    f"(Run ID: {reusable_existing.info.run_id}). Skipping this run."
                )
            return True
        if resolved.canonical and resolved.canonical.info.status == "FINISHED":
            reuse_block_reason = self._run_reuse_block_reason(
                resolved.canonical,
                expected_base_pipeline_method=expected_base_pipeline_method,
                expected_hparams_mode=expected_hparams_mode,
                expected_tuning_scope_key=expected_tuning_scope_key,
                expected_tuning_seed=expected_tuning_seed,
                expected_tuning_campaign=expected_tuning_campaign,
            )
            if reuse_block_reason:
                print(
                    f"Model '{model_name}' has existing finished run "
                    f"(Run ID: {resolved.canonical.info.run_id}) but it is not reusable: "
                    f"{reuse_block_reason}. Scheduling retraining."
                )
                return False
            print(
                f"Model '{model_name}' already trained "
                f"(Run ID: {resolved.canonical.info.run_id}). Skipping this run."
            )
            return True

        return False

    def _run_variant(
        self,
        *,
        client: Any,
        experiment_id: str,
        dataset_spec: Any,
        architecture: str,
        model_name: str,
        data_config_signature: str,
        signature: str,
        pipeline_id: str,
        hparams: dict[str, Any],
        datamodule_kwargs: dict[str, Any],
        model_kwargs: dict[str, Any],
        param_values: dict[str, Any],
        extra_tags: Optional[dict[str, str]] = None,
    ) -> Optional[str]:
        return train_single_run(
            model_architecture=architecture,
            hparams=hparams,
            dataset_spec=dataset_spec,
            args=self.args,
            data_config_signature=data_config_signature,
            signature=signature,
            model_name_override=model_name,
            pipeline_id=pipeline_id,
            pipeline_method=self.spec.pipeline_method,
            pipeline_kind=self.spec.pipeline_kind,
            datamodule_kwargs=datamodule_kwargs,
            model_kwargs=model_kwargs,
            robustness_method=self.spec.pipeline_method,
            extra_tags=extra_tags,
        )

    @staticmethod
    def _is_baseline_candidate(run) -> bool:
        tags = run.data.tags or {}
        pipeline_id = tags.get("pipeline_id")
        pipeline_method = tags.get("pipeline_method")
        pipeline_kind = tags.get("pipeline_kind")
        if not (pipeline_id and pipeline_method and pipeline_kind):
            missing = [
                name
                for name, value in (
                    ("pipeline_id", pipeline_id),
                    ("pipeline_method", pipeline_method),
                    ("pipeline_kind", pipeline_kind),
                )
                if not value
            ]
            raise ValueError(
                f"Run {run.info.run_id} is missing pipeline_* tags ({', '.join(missing)}). "
            )
        if pipeline_id == "baseline" and pipeline_method != "baseline":
            raise ValueError(
                f"Run {run.info.run_id} has pipeline_id='baseline' but pipeline_method='{pipeline_method}'. "
                "Baseline runs must set pipeline_method='baseline'."
            )
        if pipeline_method == "baseline" and pipeline_id != "baseline":
            raise ValueError(
                f"Run {run.info.run_id} has pipeline_method='baseline' but pipeline_id='{pipeline_id}'. "
                "Baseline runs must set pipeline_id='baseline'."
            )
        return pipeline_id == "baseline" and pipeline_method == "baseline"

    def _find_best_baseline_run(
        self,
        client: Any,
        experiment_id: str,
        dataset_spec: Any,
        architecture: str,
        *,
        data_config_signature: str,
    ):
        data_sig = str(data_config_signature).strip()
        if not data_sig:
            raise ValueError(
                "_find_best_baseline_run: data_config_signature must be a "
                "non-empty string."
            )
        filter_parts = [
            "tags.stage = 'train'",
            "tags.pipeline_id = 'baseline'",
            f"tags.model_architecture = '{architecture}'",
            f"tags.data_config_signature = '{data_sig}'",
            "attribute.status = 'FINISHED'",
        ]
        filter_string = " AND ".join(filter_parts)
        runs = search_runs_all(
            client,
            [experiment_id],
            filter_string=filter_string,
            max_results=1000,
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        runs = [r for r in runs if not r.data.tags.get("mlflow.parentRunId")]
        if not runs:
            raise ValueError(
                f"No baseline run found for architecture '{architecture}' with "
                f"matching data_config_signature. "
                f"Train baseline first (pipeline_id=baseline) before running "
                f"improvement pipelines."
            )
        for run in runs:
            tags = run.data.tags or {}
            pipeline_method = tags.get("pipeline_method")
            pipeline_kind = tags.get("pipeline_kind")
            if pipeline_method != "baseline":
                raise ValueError(
                    f"Run {run.info.run_id} has pipeline_id=baseline but pipeline_method='{pipeline_method}'. "
                    "Baseline runs must set pipeline_method='baseline'."
                )
            if not pipeline_kind:
                raise ValueError(
                    f"Run {run.info.run_id} is missing pipeline_kind tag. "
                    "Baseline runs must set pipeline_kind."
                )
            if pipeline_kind != "train":
                raise ValueError(
                    f"Run {run.info.run_id} has pipeline_kind='{pipeline_kind}' but expected 'train'."
                )
            signature = tags.get("signature")
            if not signature or not str(signature).strip():
                raise ValueError(
                    f"Run {run.info.run_id} is missing required signature tag. "
                    "Baseline runs must have a signature for tuning-scope filtering."
                )

        # Filter to baselines whose signature is in the current budget-limited
        # tuning scope.  This ensures inherit_baseline selects from the same
        # pool that testing considers valid, preventing stale-lineage
        # mismatches after grid revisions.
        baseline_candidates = self._baseline_reference_candidates(
            dataset_name=dataset_spec.key,
            architecture=architecture,
            data_config_signature=data_sig,
        )
        max_hp_trials_per_model = parse_max_hp_trials_per_model(
            getattr(self.args, "max_hp_trials_per_model", None)
        )
        _, target_budget, _ = self._compute_budget_targets(
            max_hp_trials_per_model=max_hp_trials_per_model,
            baseline_candidate_count=len(baseline_candidates),
            method_candidate_count=len(baseline_candidates),
        )
        tuning_seed = derive_tuning_seed(
            base_seed=self.args.seed,
            dataset_key=dataset_spec.key,
            architecture=architecture,
            data_config_signature=data_sig,
            pipeline_method="baseline",
            pipeline_kind="train",
            tuning_strategy=_TUNING_STRATEGY_RANDOM_SUBGRID,
        )
        selected = self._select_seeded_candidates(
            candidates=baseline_candidates,
            target_budget=target_budget,
            tuning_seed=tuning_seed,
        )
        valid_signatures = frozenset(c.signature for c in selected)
        in_scope_runs = [
            r for r in runs
            if r.data.tags.get("signature") in valid_signatures
        ]
        if not in_scope_runs:
            raise ValueError(
                f"No in-scope baseline run found for architecture '{architecture}'. "
                f"{len(runs)} baseline run(s) exist but none match the current "
                f"tuning scope ({len(valid_signatures)} valid signatures). "
                f"Retrain baselines with the updated grid first."
            )
        if len(in_scope_runs) < len(runs):
            print(
                f"inherit_baseline: filtered {len(runs)} → {len(in_scope_runs)} "
                f"baseline(s) for {architecture} (excluded "
                f"{len(runs) - len(in_scope_runs)} out-of-scope baseline(s) "
                f"from the active tuning scope)."
            )

        return sort_runs_by_metric(
            in_scope_runs,
            metric_key="best_val_loss",
            missing_error_prefix="Baseline runs",
        )[0]

    def _extract_hparams_from_run(
        self,
        client: Any,
        run: Any,
        architecture: str,
    ) -> dict[str, Any]:
        hparam_spec = load_hparams().get(architecture, {})
        hparam_keys = set(hparam_spec.keys())
        run_id = run.info.run_id

        with tempfile.TemporaryDirectory(prefix="robust-") as tmpdir:
            try:
                artifact_path = client.download_artifacts(run_id, "hparams.json", dst_path=tmpdir)
            except Exception as exc:
                raise ValueError(
                    f"Baseline run {run_id} is missing required hparams.json for "
                    f"architecture '{architecture}'."
                ) from exc
            try:
                with open(artifact_path, encoding="utf-8") as f:
                    hparams = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Baseline run {run_id} hparams.json is not valid JSON for "
                    f"architecture '{architecture}'."
                ) from exc
            except OSError as exc:
                raise ValueError(
                    f"Baseline run {run_id} hparams.json could not be read for "
                    f"architecture '{architecture}'."
                ) from exc

        if isinstance(hparams, dict):
            if hparam_keys:
                missing = sorted(hparam_keys - set(hparams.keys()))
                if missing:
                    raise ValueError(
                        f"Baseline run {run_id} hparams.json is missing keys for architecture '{architecture}': "
                        f"{', '.join(missing)}. "
                        "Rerun the baseline pipeline with the current code (or re-log hparams.json) "
                        "to avoid silently falling back to model defaults."
                    )
            return hparams

        raise ValueError(
            f"Baseline run {run_id} hparams.json must contain a mapping for "
            f"architecture '{architecture}'."
        )

    def find_baseline_hparams(
        self,
        client: Any,
        experiment_id: str,
        dataset_spec: Any,
        architecture: str,
        *,
        data_config_signature: str,
    ) -> tuple[dict[str, Any], str]:
        """Find best baseline run and extract its hyperparameters.

        Locates baseline runs matching the current data configuration and
        selects the one with lowest best_val_loss. Extracts hyperparameters
        from the run's hparams.json artifact.

        Args:
            client: MLflow client instance.
            experiment_id: MLflow experiment ID to search.
            dataset_spec: Dataset specification for data_config_signature.
            architecture: Model architecture name (e.g., "GRU").
            data_config_signature: Data configuration signature for comparability.

        Returns:
            Tuple of (hyperparameters dict, baseline run_id).

        Raises:
            ValueError: If no baseline run found or hparams cannot be extracted.
        """
        best = self._find_best_baseline_run(
            client,
            experiment_id,
            dataset_spec,
            architecture,
            data_config_signature=data_config_signature,
        )
        hparams = self._extract_hparams_from_run(client, best, architecture)
        return hparams, best.info.run_id

    def run(
        self,
        client: Any,
        dataset_spec: Any,
        architecture: str,
    ) -> Optional[str]:
        """Execute the pipeline for one (dataset, architecture) combination.

        Args:
            client: MLflow client instance.
            dataset_spec: Dataset specification object.
            architecture: Model architecture name (e.g., "GRU").

        Returns:
            Run ID of the created run, or None if skipped.

        Raises:
            ValueError: If the pipeline spec is invalid or a baseline is missing.
        """
        self._last_run_report = None
        data_config_signature = compute_data_config_signature(
            dataset_spec=dataset_spec, args=self.args
        )
        dataset_name = dataset_spec.key
        param_overrides = param_overrides_for_spec(self.spec, self.args)
        param_sets = self._normalize_recipe_param_sets(
            self.spec.expand_params(overrides=param_overrides)
        )
        skip_reason = self._scope_policy_skip_reason(architecture)
        if skip_reason is not None:
            raise ValueError(
                f"Unsupported benchmark method/architecture pair: "
                f"{self.spec.pipeline_method}/{architecture} ({dataset_name})."
            )
        experiment_name = f"{self.args.mlflow_experiment_prefix}-{dataset_name}"
        experiment_id = ensure_experiment_data_signature(
            client,
            experiment_name,
            data_config_signature,
        )
        rerun_requested = bool(getattr(self.args, "rerun", False))
        is_finetune_recipe = self.spec.pipeline_kind == "finetune"
        max_hp_trials_per_model = parse_max_hp_trials_per_model(
            getattr(self.args, "max_hp_trials_per_model", None)
        )

        finetune_epochs = None
        finetune_lr_factor = None
        if is_finetune_recipe:
            finetune_epochs, finetune_lr_factor = self._resolve_active_finetune_schedule()

        backbone_run = None
        baseline_hparams_run_id = None
        if is_finetune_recipe and self.spec.model_hparams_mode != "inherit_baseline":
            raise ValueError(
                "Finetune-kind recipes must use model_hparams.mode='inherit_baseline'; "
                f"got '{self.spec.model_hparams_mode}'."
            )
        if is_finetune_recipe and not getattr(self.args, "save_checkpoint", True):
            raise ValueError(
                "Finetune-kind recipes require save_checkpoint=true so that "
                "testing can reload the finetuned model from its own checkpoint."
            )
        hparams_mode = self.spec.model_hparams_mode
        if hparams_mode == "baseline_grid":
            hparams_grid = load_hparams().get(architecture)
            if hparams_grid is None:
                raise ValueError(
                    f"No hyperparameters found for architecture '{architecture}' "
                    "in configs/baseline_hparams.yaml."
                )
        elif hparams_mode == "inherit_baseline":
            if is_finetune_recipe:
                backbone_run = self._find_best_baseline_run(
                    client,
                    experiment_id,
                    dataset_spec,
                    architecture,
                    data_config_signature=data_config_signature,
                )
                hparams_grid = self._extract_hparams_from_run(client, backbone_run, architecture)
                baseline_hparams_run_id = backbone_run.info.run_id
            else:
                hparams_grid, baseline_hparams_run_id = self.find_baseline_hparams(
                    client, experiment_id, dataset_spec, architecture,
                    data_config_signature=data_config_signature,
                )
        else:
            raise ValueError(
                f"Unknown model_hparams.mode '{self.spec.model_hparams_mode}' in pipeline spec."
            )

        def _extra_tags_for_pipeline() -> Optional[dict[str, str]]:
            method = require_pipeline_method_value(
                self.spec.pipeline_method,
                context="PipelineRunner.run",
            ).lower()
            if method == "baseline":
                return None
            tags: dict[str, str] = {"base_pipeline_method": "baseline"}
            tags["hparams_mode"] = str(hparams_mode)
            if hparams_mode == "inherit_baseline" and baseline_hparams_run_id:
                tags["baseline_hparams_run_id"] = str(baseline_hparams_run_id)
            return tags

        expected_base_pipeline_method: Optional[str] = None
        expected_hparams_mode: Optional[str] = None
        if require_pipeline_method_value(
            self.spec.pipeline_method,
            context="PipelineRunner.run",
        ).lower() != "baseline":
            expected_base_pipeline_method = "baseline"
            expected_hparams_mode = str(hparams_mode)

        backbone_run_id: Optional[str] = None
        if backbone_run is not None:
            backbone_run_id = backbone_run.info.run_id
        candidates = self._build_training_candidates(
            architecture=architecture,
            dataset_name=dataset_name,
            data_config_signature=data_config_signature,
            param_sets=param_sets,
            hparams_mode=hparams_mode,
            hparams_grid=hparams_grid,
            backbone_run_id=backbone_run_id,
            finetune_epochs=finetune_epochs,
            finetune_lr_factor=finetune_lr_factor,
        )
        candidates = self._filter_candidates_before_scheduling(
            dataset_spec=dataset_spec,
            architecture=architecture,
            data_config_signature=data_config_signature,
            candidates=candidates,
        )

        tuning_scope_key = self._build_tuning_scope_key(
            dataset_name=dataset_name,
            architecture=architecture,
            data_config_signature=data_config_signature,
            pipeline_method=self.spec.pipeline_method,
            pipeline_kind=self.spec.pipeline_kind,
        )
        tuning_seed = derive_tuning_seed(
            base_seed=self.args.seed,
            dataset_key=dataset_name,
            architecture=architecture,
            data_config_signature=data_config_signature,
            pipeline_method=self.spec.pipeline_method,
            pipeline_kind=self.spec.pipeline_kind,
            tuning_strategy=_TUNING_STRATEGY_RANDOM_SUBGRID,
        )
        tuning_reference_budget = 0
        tuning_budget_target = 0
        tuning_campaign = _TUNING_CAMPAIGN_RESUME
        tuning_pool_exhausted = False
        consumed_signatures: set[str] = set()
        selected_trial_number_by_signature: dict[str, int] = {}
        expected_units = 0
        executed_units = 0
        skipped_existing_units = 0
        skipped_policy_units = 0
        failed_units = 0

        baseline_candidate_count = self._count_baseline_reference_candidates(
            dataset_name=dataset_name,
            architecture=architecture,
            data_config_signature=data_config_signature,
        )
        (
            tuning_reference_budget,
            tuning_budget_target,
            tuning_pool_exhausted,
        ) = self._compute_budget_targets(
            max_hp_trials_per_model=max_hp_trials_per_model,
            baseline_candidate_count=baseline_candidate_count,
            method_candidate_count=len(candidates),
        )
        selected_candidates = self._select_seeded_candidates(
            candidates=candidates,
            target_budget=tuning_budget_target,
            tuning_seed=tuning_seed,
        )
        selected_trial_number_by_signature = {
            candidate.signature: trial_number
            for trial_number, candidate in enumerate(selected_candidates, start=1)
        }
        expected_units = len(selected_candidates)
        tuning_campaign = self._resolve_tuning_campaign(
            rerun_requested=rerun_requested,
        )
        if not rerun_requested:
            candidate_signatures = {
                candidate.signature for candidate in selected_candidates
            }
            consumed_signatures, consumed_duplicates = self._collect_consumed_signatures(
                client=client,
                experiment_id=experiment_id,
                dataset_name=dataset_name,
                architecture=architecture,
                data_config_signature=data_config_signature,
                pipeline_method=self.spec.pipeline_method,
                pipeline_kind=self.spec.pipeline_kind,
                tuning_campaign=None,
                candidate_signatures=candidate_signatures,
                expected_base_pipeline_method=expected_base_pipeline_method,
                expected_hparams_mode=expected_hparams_mode,
                expected_tuning_scope_key=tuning_scope_key,
                expected_tuning_seed=tuning_seed,
                expected_tuning_campaign=None,
            )
            if consumed_duplicates:
                delete_runs(
                    client,
                    consumed_duplicates,
                    reason="consumed duplicate signature",
                )
            skipped_existing_units += len(consumed_signatures)
        remaining_budget = max(tuning_budget_target - len(consumed_signatures), 0)
        if tuning_pool_exhausted:
            print(
                f"Tuning pool exhausted for '{self.spec.pipeline_method}' on {architecture} "
                f"({dataset_name}): {len(candidates)} candidate(s) available, "
                f"reference budget is {tuning_reference_budget}."
            )
        candidates = [
            candidate
            for candidate in selected_candidates
            if candidate.signature not in consumed_signatures
        ]
        print(
            f"Random-subgrid tuning budget for '{self.spec.pipeline_method}' on {architecture} "
            f"({dataset_name}): target={tuning_budget_target}, "
            f"consumed={len(consumed_signatures)}, scheduled={len(candidates)}, "
            f"remaining={remaining_budget}, "
            f"campaign={tuning_campaign}."
        )

        last_run_id = None
        for candidate in candidates:
            print(f"Running {candidate.pipeline_id} on {architecture} ({dataset_name})")
            if self._handle_existing_runs(
                client,
                experiment_id,
                candidate.signature,
                candidate.model_name,
                rerun_requested,
                expected_base_pipeline_method=expected_base_pipeline_method,
                expected_hparams_mode=expected_hparams_mode,
                expected_tuning_scope_key=tuning_scope_key,
                expected_tuning_seed=tuning_seed,
                expected_tuning_campaign=None,
            ):
                skipped_existing_units += 1
                continue

            extra_tags = _extra_tags_for_pipeline()
            trial_number = selected_trial_number_by_signature.get(candidate.signature)
            if trial_number is None:
                raise ValueError(
                    f"Missing deterministic trial number for signature '{candidate.signature}'."
                )
            tuning_tags = self._build_tuning_tags(
                reference_budget=tuning_reference_budget,
                target_budget=tuning_budget_target,
                trial_number=trial_number,
                tuning_seed=tuning_seed,
                scope_key=tuning_scope_key,
                campaign=tuning_campaign,
                pool_exhausted=tuning_pool_exhausted,
                early_stopping=bool(getattr(self.args, "early_stopping", False)),
            )
            if extra_tags is None:
                extra_tags = dict(tuning_tags)
            else:
                merged_tags = dict(extra_tags)
                merged_tags.update(tuning_tags)
                extra_tags = merged_tags

            try:
                if is_finetune_recipe:
                    if backbone_run is None:
                        raise ValueError(
                            "backbone_run must be set for finetune training."
                        )
                    ft_epochs = finetune_epochs
                    ft_lr_factor = finetune_lr_factor
                    if ft_epochs is None:
                        raise ValueError(
                            "finetune_epochs must be resolved before finetune training."
                        )
                    if ft_lr_factor is None:
                        raise ValueError(
                            "finetune_lr_factor must be resolved before finetune training."
                        )
                    finetune_run_id = finetune_single_run(
                        model_architecture=architecture,
                        backbone_run_id=backbone_run.info.run_id,
                        hparams=candidate.hparams,
                        dataset_spec=dataset_spec,
                        args=self.args,
                        data_config_signature=data_config_signature,
                        signature=candidate.signature,
                        model_name_override=candidate.model_name,
                        pipeline_id=candidate.pipeline_id,
                        pipeline_method=candidate.pipeline_method,
                        pipeline_kind=candidate.pipeline_kind,
                        datamodule_kwargs=candidate.datamodule_kwargs,
                        model_kwargs=candidate.model_kwargs,
                        robustness_method=candidate.pipeline_method,
                        finetune_epochs=ft_epochs,
                        finetune_lr_factor=ft_lr_factor,
                        extra_tags=extra_tags,
                    )
                    if finetune_run_id is None:
                        raise ValueError(
                            f"Finetune execution returned no run_id for signature "
                            f"'{candidate.signature}' ({architecture}, {dataset_name})."
                        )
                    last_run_id = finetune_run_id
                else:
                    variant_run_id = self._run_variant(
                        client=client,
                        experiment_id=experiment_id,
                        dataset_spec=dataset_spec,
                        architecture=architecture,
                        model_name=candidate.model_name,
                        data_config_signature=data_config_signature,
                        signature=candidate.signature,
                        pipeline_id=candidate.pipeline_id,
                        hparams=candidate.hparams,
                        datamodule_kwargs=candidate.datamodule_kwargs,
                        model_kwargs=candidate.model_kwargs,
                        param_values=candidate.param_values,
                        extra_tags=extra_tags,
                    )
                    if variant_run_id is None:
                        skipped_policy_units += 1
                        continue
                    last_run_id = variant_run_id
                executed_units += 1
            except Exception as exc:
                failed_units += 1
                if getattr(self.args, "raise_error", False):
                    self._last_run_report = _build_run_report_for_spec(
                        spec=self.spec,
                        dataset=dataset_name,
                        architecture=architecture,
                        expected_units=expected_units,
                        executed_units=executed_units,
                        skipped_existing_units=skipped_existing_units,
                        skipped_policy_units=skipped_policy_units,
                        failed_units=failed_units,
                    )
                    raise
                if "CUDA out of memory" in str(exc):
                    print(
                        "-" * 80
                        + f"\nCUDA out of memory, skipping {architecture}.\n"
                        + "-" * 80
                    )
                else:
                    print(str(exc))
                    print(
                        "-" * 80
                        + f"\nUnknown error, skipping {architecture}.\n"
                        + "-" * 80
                    )

        self._last_run_report = _build_run_report_for_spec(
            spec=self.spec,
            dataset=dataset_name,
            architecture=architecture,
            expected_units=expected_units,
            executed_units=executed_units,
            skipped_existing_units=skipped_existing_units,
            skipped_policy_units=skipped_policy_units,
            failed_units=failed_units,
        )
        return last_run_id


class WrapPipelineRunner:
    """Execute wrap-style improvement pipelines (e.g., ensemble) using baseline backbones."""

    def __init__(self, spec: PipelineSpec, args: Any):
        if spec.pipeline_kind != "wrap":
            raise ValueError(
                f"WrapPipelineRunner requires pipeline_kind='wrap' (got '{spec.pipeline_kind}')."
            )
        self.spec = spec
        self.args = args
        self.registration = self._resolve_improvement_registration()
        self._tuning_scope_cache: dict[tuple[str, str, str, str], TuningScope] = {}
        self._last_run_report: Optional[RunExecutionReport] = None

    def get_last_run_report(self) -> Optional[RunExecutionReport]:
        return self._last_run_report

    def _resolve_improvement_registration(self):
        token = self.spec.pipeline_method
        if not token:
            raise ValueError(
                "Wrap pipeline spec is missing pipeline_method; this is required."
            )
        try:
            return get_improvement_registration(token)
        except KeyError:
            options = ", ".join(sorted(list_available_improvements().keys()))
        raise ValueError(
            "Unknown improvement pipeline for pipeline_kind='wrap'. "
            f"pipeline_method='{self.spec.pipeline_method}'. "
                f"Available improvements: {options}"
        )

    def _list_improvement_runs(self, client: Any, experiment_id: str) -> list:
        runs = search_runs_all(
            client,
            [experiment_id],
            filter_string="tags.stage = 'improve' AND attribute.status = 'FINISHED'",
            max_results=1000,
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        return list(runs)

    def _list_base_runs(
        self,
        *,
        client: Any,
        experiment_id: str,
        dataset_spec: Any,
        architecture: str,
        data_config_signature: str,
        base_pipeline_method: str,
    ) -> list:
        filter_parts = [
            "tags.stage = 'train'",
            f"tags.pipeline_method = '{base_pipeline_method}'",
            "tags.pipeline_kind = 'train'",
            f"tags.model_architecture = '{architecture}'",
            f"tags.data_config_signature = '{data_config_signature}'",
            "attribute.status = 'FINISHED'",
        ]
        filter_string = " AND ".join(filter_parts)
        runs = search_runs_all(
            client,
            [experiment_id],
            filter_string=filter_string,
            max_results=1000,
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        runs = [run for run in runs if not run.data.tags.get("mlflow.parentRunId")]
        for run in runs:
            tags = run.data.tags or {}
            pipeline_method = tags.get("pipeline_method")
            pipeline_kind = tags.get("pipeline_kind")
            if pipeline_method != base_pipeline_method:
                raise ValueError(
                    f"Run {run.info.run_id} has pipeline_method='{pipeline_method}' but expected "
                    f"pipeline_method='{base_pipeline_method}'."
                )
            if pipeline_kind != "train":
                raise ValueError(
                    f"Run {run.info.run_id} has pipeline_kind='{pipeline_kind}' but expected 'train'."
                )
        scope_key = (
            str(dataset_spec.key),
            str(architecture),
            str(data_config_signature),
            str(base_pipeline_method),
        )
        expected_scope = self._tuning_scope_cache.get(scope_key)
        if expected_scope is None:
            base_spec = load_pipeline_spec_for_method(base_pipeline_method)
            if str(base_spec.pipeline_kind).strip() != "train":
                raise ValueError(
                    f"Base pipeline '{base_pipeline_method}' must be train-kind for wrap improvements."
                )
            base_runner = PipelineRunner(base_spec, self.args)
            expected_scope = base_runner.expected_tuning_scope(
                client=client,
                experiment_id=experiment_id,
                dataset_spec=dataset_spec,
                architecture=architecture,
                data_config_signature=data_config_signature,
            )
            self._tuning_scope_cache[scope_key] = expected_scope

        scoped_runs = []
        for run in runs:
            tags = run.data.tags or {}
            signature_raw = tags.get("signature")
            signature = str(signature_raw).strip() if signature_raw is not None else ""
            if not signature:
                raise ValueError(
                    f"Run {run.info.run_id} is missing required signature tag."
                )
            tuning_strategy = optional_nonempty_tag_value(
                tags, key="tuning_strategy"
            )
            if tuning_strategy is None:
                raise ValueError(
                    f"Run {run.info.run_id} is missing required tuning_strategy tag."
                )
            if tuning_strategy != _TUNING_STRATEGY_RANDOM_SUBGRID:
                continue
            tuning_scope_key = optional_nonempty_tag_value(
                tags, key="tuning_scope_key"
            )
            if tuning_scope_key is None:
                raise ValueError(
                    f"Run {run.info.run_id} is missing required tuning_scope_key tag."
                )
            if tuning_scope_key != expected_scope.scope_key:
                continue
            tuning_seed = optional_nonempty_tag_value(tags, key="tuning_seed")
            if tuning_seed is None:
                raise ValueError(
                    f"Run {run.info.run_id} is missing required tuning_seed tag."
                )
            try:
                int(tuning_seed)
            except ValueError as exc:
                raise ValueError(
                    f"Run {run.info.run_id} has non-integer tuning_seed '{tuning_seed}'."
                ) from exc
            if tuning_seed != str(expected_scope.tuning_seed):
                continue
            if signature not in expected_scope.signature_set:
                continue
            scoped_runs.append(run)
        if not scoped_runs and runs:
            raise ValueError(
                f"Found {len(runs)} finished base run(s) for pipeline_method='{base_pipeline_method}' "
                f"on dataset '{dataset_spec.key}' / architecture '{architecture}', but none matched "
                "the active deterministic tuning scope "
                f"(scope_key='{expected_scope.scope_key}', tuning_seed='{expected_scope.tuning_seed}', "
                f"expected_signatures={len(expected_scope.signature_set)}). "
                "Run the corresponding train-kind recipe with the same seed/trial-budget/overrides "
                "before launching wrap improvements."
            )
        return scoped_runs

    @staticmethod
    def _sort_baselines(runs: list) -> list:
        return sort_runs_by_metric(
            runs,
            metric_key="best_val_loss",
            missing_error_prefix="Baseline runs",
        )

    def run(
        self,
        client: Any,
        dataset_spec: Any,
        architecture: str,
    ) -> Optional[str]:
        self._last_run_report = None
        data_config_signature = compute_data_config_signature(
            dataset_spec=dataset_spec, args=self.args
        )
        dataset_name = dataset_spec.key
        param_sets = self.spec.expand_params(
            overrides=param_overrides_for_spec(self.spec, self.args)
        )
        expected_units = len(param_sets)
        if scope_policy_skip_reason_for_spec(self.spec, architecture) is not None:
            raise ValueError(
                f"Unsupported benchmark method/architecture pair: "
                f"{self.spec.pipeline_method}/{architecture} ({dataset_name})."
            )
        experiment_name = f"{self.args.mlflow_experiment_prefix}-{dataset_name}"
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(
                f"No experiment found for dataset '{dataset_name}' "
                f"(prefix: {self.args.mlflow_experiment_prefix}). "
                "Train baselines first before running wrap improvements."
        )
        experiment_id = ensure_experiment_data_signature(
            client,
            experiment_name,
            data_config_signature,
        )
        improvement_runs = self._list_improvement_runs(client, experiment_id)
        executed_units = 0
        skipped_existing_units = 0
        skipped_policy_units = 0
        failed_units = 0

        last_run_id = None
        for param_values in param_sets:
            base_method = resolve_wrap_base_pipeline_method(
                self.spec.pipeline_method, param_values
            )
            base_runs = self._list_base_runs(
                client=client,
                experiment_id=experiment_id,
                dataset_spec=dataset_spec,
                architecture=architecture,
                data_config_signature=data_config_signature,
                base_pipeline_method=str(base_method),
            )
            if not base_runs:
                raise ValueError(
                    f"No base runs found for pipeline_method '{base_method}' on dataset '{dataset_name}' "
                    f"(architecture '{architecture}', data_config_signature '{data_config_signature}')."
                )
            base_runs = self._sort_baselines(base_runs)
            updated_args = copy(self.args)
            for key, value in param_values.items():
                setattr(updated_args, key, value)
            improvement = self.registration.recipe_cls(updated_args)
            if not hasattr(improvement, "top_k"):
                raise ValueError(
                    f"Wrap improvement '{self.spec.pipeline_method}' must define top_k."
                )
            top_k_raw = getattr(improvement, "top_k")
            if top_k_raw is None:
                raise ValueError(
                    f"Wrap improvement '{self.spec.pipeline_method}' has top_k=None."
                )
            try:
                top_k = int(top_k_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Wrap improvement '{self.spec.pipeline_method}' has invalid top_k='{top_k_raw}'."
                ) from exc
            if top_k <= 0:
                raise ValueError(
                    f"Invalid top_k={top_k} for pipeline '{self.spec.pipeline_id}'. "
                    "Wrap pipelines require at least one baseline (top_k >= 1)."
                )
            if top_k > len(base_runs):
                raise ValueError(
                    f"Requested top_k={top_k} but only {len(base_runs)} base runs are available "
                    f"for '{architecture}' on '{dataset_name}' with pipeline_method='{base_method}'."
                )
            try:
                outcome = improvement.run(
                    client=client,
                    experiment=experiment,
                    dataset_name=dataset_name,
                    architecture=architecture,
                    base_runs=base_runs,
                    improvement_runs=improvement_runs,
                )
            except Exception as exc:
                failed_units += 1
                if getattr(self.args, "raise_error", False):
                    self._last_run_report = _build_run_report_for_spec(
                        spec=self.spec,
                        dataset=dataset_name,
                        architecture=architecture,
                        expected_units=expected_units,
                        executed_units=executed_units,
                        skipped_existing_units=skipped_existing_units,
                        skipped_policy_units=skipped_policy_units,
                        failed_units=failed_units,
                    )
                    raise
                print(str(exc))
                print(
                    "-" * 80
                    + f"\nUnknown error, skipping {architecture}.\n"
                    + "-" * 80
                )
                continue
            if not isinstance(outcome, WrapExecutionOutcome):
                raise ValueError(
                    f"Wrap improvement '{self.spec.pipeline_method}' returned unsupported outcome "
                    f"type {type(outcome)!r}."
                )
            if outcome.status == "executed":
                if not outcome.run_id:
                    raise ValueError(
                        "Wrap execution outcome with status='executed' must include run_id."
                    )
                executed_units += 1
                last_run_id = str(outcome.run_id)
                continue
            if outcome.status == "skipped_existing":
                skipped_existing_units += 1
                continue
            if outcome.status == "skipped_policy":
                skipped_policy_units += 1
                continue
            if outcome.status == "failed":
                failed_units += 1
                if getattr(self.args, "raise_error", False):
                    self._last_run_report = _build_run_report_for_spec(
                        spec=self.spec,
                        dataset=dataset_name,
                        architecture=architecture,
                        expected_units=expected_units,
                        executed_units=executed_units,
                        skipped_existing_units=skipped_existing_units,
                        skipped_policy_units=skipped_policy_units,
                        failed_units=failed_units,
                    )
                    raise RuntimeError(f"Wrap execution failed: {outcome.reason}")
                continue
            raise ValueError(
                f"Uncategorized wrap outcome status '{outcome.status}' for "
                f"{self.spec.pipeline_method} on {architecture}/{dataset_name}."
            )
        self._last_run_report = _build_run_report_for_spec(
            spec=self.spec,
            dataset=dataset_name,
            architecture=architecture,
            expected_units=expected_units,
            executed_units=executed_units,
            skipped_existing_units=skipped_existing_units,
            skipped_policy_units=skipped_policy_units,
            failed_units=failed_units,
        )
        return last_run_id


def print_coverage_report(report: RunExecutionReport) -> None:
    print(
        "Coverage "
        f"[dataset={report.dataset} arch={report.architecture} "
        f"method={report.pipeline_method} kind={report.pipeline_kind}] "
        f"expected={report.expected_units} "
        f"executed={report.executed_units} "
        f"skipped_existing={report.skipped_existing_units} "
        f"skipped_policy={report.skipped_policy_units} "
        f"failed={report.failed_units} "
        f"uncovered={report.uncovered_units} "
        f"complete={str(report.is_complete).lower()}"
    )


def print_coverage_summary_and_raise_on_incomplete(
    coverage_reports: list[RunExecutionReport],
) -> None:
    totals_by_arch: dict[str, dict[str, int]] = {}
    global_totals = {
        "expected": 0,
        "executed": 0,
        "skipped_existing": 0,
        "skipped_policy": 0,
        "failed": 0,
        "uncovered": 0,
    }
    for report in coverage_reports:
        arch_key = str(report.architecture)
        if arch_key not in totals_by_arch:
            totals_by_arch[arch_key] = {
                "expected": 0,
                "executed": 0,
                "skipped_existing": 0,
                "skipped_policy": 0,
                "failed": 0,
                "uncovered": 0,
            }
        arch_totals = totals_by_arch[arch_key]
        arch_totals["expected"] += int(report.expected_units)
        arch_totals["executed"] += int(report.executed_units)
        arch_totals["skipped_existing"] += int(report.skipped_existing_units)
        arch_totals["skipped_policy"] += int(report.skipped_policy_units)
        arch_totals["failed"] += int(report.failed_units)
        arch_totals["uncovered"] += int(report.uncovered_units)
        global_totals["expected"] += int(report.expected_units)
        global_totals["executed"] += int(report.executed_units)
        global_totals["skipped_existing"] += int(report.skipped_existing_units)
        global_totals["skipped_policy"] += int(report.skipped_policy_units)
        global_totals["failed"] += int(report.failed_units)
        global_totals["uncovered"] += int(report.uncovered_units)

    print("Coverage totals by architecture:")
    for arch in sorted(totals_by_arch.keys()):
        totals = totals_by_arch[arch]
        print(
            f"- {arch}: expected={totals['expected']} executed={totals['executed']} "
            f"skipped_existing={totals['skipped_existing']} "
            f"skipped_policy={totals['skipped_policy']} failed={totals['failed']} "
            f"uncovered={totals['uncovered']}"
        )
    print(
        "Coverage totals (global): "
        f"expected={global_totals['expected']} executed={global_totals['executed']} "
        f"skipped_existing={global_totals['skipped_existing']} "
        f"skipped_policy={global_totals['skipped_policy']} "
        f"failed={global_totals['failed']} uncovered={global_totals['uncovered']}"
    )
    incomplete_reports = [report for report in coverage_reports if not report.is_complete]
    if incomplete_reports:
        examples = ", ".join(
            f"{report.dataset}/{report.architecture}/{report.pipeline_method}"
            for report in incomplete_reports[:5]
        )
        raise ValueError(
            "Coverage is incomplete for one or more dataset/architecture/method combinations: "
            f"{examples}."
        )


def create_pipeline_runner(spec: PipelineSpec, args: Any):
    require_pipeline_method_value(
        spec.pipeline_method,
        context="create_pipeline_runner",
    )
    pipeline_kind = _require_pipeline_kind_value(
        spec.pipeline_kind,
        context="create_pipeline_runner",
    )
    if pipeline_kind == "wrap":
        return WrapPipelineRunner(spec, args)
    return PipelineRunner(spec, args)


__all__ = [
    "PipelineRunner",
    "RunExecutionReport",
    "TuningScope",
    "WrapPipelineRunner",
    "create_pipeline_runner",
    "load_pipeline_spec_for_method",
    "param_overrides_for_spec",
    "print_coverage_report",
    "print_coverage_summary_and_raise_on_incomplete",
    "resolve_wrap_base_pipeline_method",
    "scope_policy_skip_reason_for_spec",
]
