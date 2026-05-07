"""Pipeline specification and recipe parsing."""

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Optional

import yaml


VALID_PIPELINE_KINDS = {"train", "finetune", "wrap"}
# These methods require inherit_baseline because their recipe owns a baseline-backed path.
ADVERSARIAL_TRAINING_METHODS = frozenset(
    {
        "adversarial_training",
    }
)
_ADVTRAIN_RECIPE_PARAM_KEYS = frozenset(
    {
        "advtrain_epsilon",
        "advtrain_step_size",
        "advtrain_attack_steps",
        "advtrain_random_start",
        "advtrain_attack_channels",
    }
)
_ADVTRAIN_MODEL_KWARG_KEYS = frozenset(
    {"adversarial_training_mode", *_ADVTRAIN_RECIPE_PARAM_KEYS}
)


def _validate_exact_key_set(
    *,
    actual_keys: set[str],
    required_keys: frozenset[str],
    context: str,
) -> None:
    missing = sorted(required_keys - actual_keys)
    unknown = sorted(actual_keys - required_keys)
    if not missing and not unknown:
        return

    parts: list[str] = []
    if missing:
        parts.append(f"missing required key(s): {', '.join(missing)}")
    if unknown:
        parts.append(f"unsupported key(s): {', '.join(unknown)}")
    raise ValueError(f"{context} has invalid key set: {'; '.join(parts)}.")


@dataclass
class PipelineSpec:
    """Specification for a training pipeline variant.

    Attributes:
        pipeline_id: Unique identifier for this pipeline variant.
            May contain placeholders like {param} that are resolved from params.
        pipeline_method: Coarse grouping key (e.g., "randomized_training").
        pipeline_kind: One of "train", "finetune", or "wrap".
        datamodule_kwargs: Keyword arguments passed to TSDataModule.
        model_kwargs: Keyword arguments passed to model constructor.
        train_fault_profiles: Optional FAug-only named profile registry owned by
            the recipe itself.
        recipe_params: Grid of parameters for expansion. Each key maps to a dict with
            required "default" and optional "grid".
        model_hparams_mode: Either "inherit_baseline" or "baseline_grid".
    """

    pipeline_id: str
    pipeline_method: str
    pipeline_kind: str

    # Training modifications
    datamodule_kwargs: dict[str, Any] = field(default_factory=dict)
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    train_fault_profiles: dict[str, Any] = field(default_factory=dict)

    # Method-specific params (for grid expansion)
    recipe_params: dict[str, Any] = field(default_factory=dict)

    # Hyperparameter handling
    model_hparams_mode: str = "inherit_baseline"

    def __post_init__(self) -> None:
        if self.pipeline_kind not in VALID_PIPELINE_KINDS:
            raise ValueError(
                f"Invalid pipeline_kind '{self.pipeline_kind}'. "
                f"Must be one of {VALID_PIPELINE_KINDS}."
            )
        if not isinstance(self.pipeline_method, str) or not self.pipeline_method.strip():
            raise ValueError("pipeline_method must be a non-empty string.")
        if not self.model_hparams_mode:
            raise ValueError("model_hparams.mode must be set in the pipeline recipe.")
        if self.model_hparams_mode not in {"baseline_grid", "inherit_baseline"}:
            raise ValueError(f"Invalid model_hparams.mode '{self.model_hparams_mode}'.")
        if (
            self.pipeline_method in ADVERSARIAL_TRAINING_METHODS
            and self.model_hparams_mode != "inherit_baseline"
        ):
            raise ValueError(
                f"{self.pipeline_method} requires model_hparams.mode='inherit_baseline'."
            )
        if not isinstance(self.datamodule_kwargs, dict):
            raise ValueError("datamodule_kwargs must be a mapping when provided.")
        if not isinstance(self.model_kwargs, dict):
            raise ValueError("model_kwargs must be a mapping when provided.")
        if not isinstance(self.recipe_params, dict):
            raise ValueError("recipe_params must be a mapping when provided.")
        if not isinstance(self.train_fault_profiles, dict):
            raise ValueError("train_fault_profiles must be a mapping when provided.")
        if self.pipeline_method == "fault_augmentation":
            if len(self.train_fault_profiles) == 0:
                raise ValueError(
                    "fault_augmentation recipes must define train_fault_profiles."
                )
        elif len(self.train_fault_profiles) > 0:
            raise ValueError(
                "train_fault_profiles is only supported for "
                "pipeline_method='fault_augmentation'."
            )
        if self.pipeline_method == "adversarial_training":
            _validate_exact_key_set(
                actual_keys=set(self.recipe_params.keys()),
                required_keys=_ADVTRAIN_RECIPE_PARAM_KEYS,
                context="adversarial_training recipe_params",
            )
            _validate_exact_key_set(
                actual_keys=set(self.model_kwargs.keys()),
                required_keys=_ADVTRAIN_MODEL_KWARG_KEYS,
                context="adversarial_training model_kwargs",
            )
            mode = self.model_kwargs.get("adversarial_training_mode")
            if mode != "pgd_linf":
                raise ValueError(
                    "adversarial_training model_kwargs must set "
                    "adversarial_training_mode='pgd_linf'."
                )

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineSpec":
        """Load spec from YAML file.

        Raises:
            FileNotFoundError: If the recipe file does not exist.
            ValueError: If required keys are missing or values are invalid.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Recipe {path} is empty or invalid YAML.")

        required = ["pipeline_id", "pipeline_method", "pipeline_kind", "recipe_params", "model_hparams"]
        for key in required:
            if key not in data:
                raise ValueError(f"Recipe {path} missing required key: {key}")
        supported = {
            "pipeline_id",
            "pipeline_method",
            "pipeline_kind",
            "recipe_params",
            "model_hparams",
            "datamodule_kwargs",
            "model_kwargs",
            "train_fault_profiles",
        }
        unknown = sorted(set(data.keys()) - supported)
        if unknown:
            raise ValueError(
                f"Recipe {path} has unsupported key(s): {', '.join(unknown)}"
            )

        # Validate pipeline_kind
        if data["pipeline_kind"] not in VALID_PIPELINE_KINDS:
            raise ValueError(
                f"Recipe {path} has invalid pipeline_kind '{data['pipeline_kind']}'. "
                f"Must be one of {VALID_PIPELINE_KINDS}."
            )
        if not isinstance(data["model_hparams"], dict):
            raise ValueError(f"Recipe {path} model_hparams must be a mapping.")
        train_fault_profiles = data.get("train_fault_profiles", {})
        if train_fault_profiles is None:
            raise ValueError(f"Recipe {path} train_fault_profiles must not be null.")
        if not isinstance(train_fault_profiles, dict):
            raise ValueError(
                f"Recipe {path} train_fault_profiles must be a mapping."
            )
        params = data.get("recipe_params")
        if params is None:
            raise ValueError(f"Recipe {path} recipe_params must not be null.")
        if not isinstance(params, dict):
            raise ValueError(f"Recipe {path} recipe_params must be a mapping.")

        return cls(
            pipeline_id=data["pipeline_id"],
            pipeline_method=data["pipeline_method"],
            pipeline_kind=data["pipeline_kind"],
            datamodule_kwargs=data.get("datamodule_kwargs", {}),
            model_kwargs=data.get("model_kwargs", {}),
            train_fault_profiles=train_fault_profiles,
            recipe_params=params,
            model_hparams_mode=data["model_hparams"].get("mode"),
        )

    def expand_params(
        self,
        overrides: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Expand params grid into list of concrete configurations.

        Returns:
            List of dicts, each representing one parameter combination.
            Returns [{}] if no params are defined.

        Raises:
            ValueError: If a parameter entry is malformed.
        """
        if not self.recipe_params:
            return [{}]

        overrides = overrides or {}
        keys = list(self.recipe_params.keys())
        values = []
        for k in keys:
            entry = self.recipe_params[k]
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Parameter '{k}' must be a mapping with 'default' and optional 'grid'."
                )
            if "default" not in entry:
                raise ValueError(
                    f"Parameter '{k}' is missing required 'default' value in recipe."
                )
            if k in overrides and overrides[k] is not None:
                values.append([overrides[k]])
                continue
            default_value = entry["default"]
            grid = entry.get("grid", None)
            if grid is None:
                values.append([default_value])
                continue
            if isinstance(grid, list):
                if len(grid) == 0:
                    raise ValueError(
                        f"Parameter '{k}' has an empty grid; omit 'grid' to use "
                        "the default as a single value."
                    )
                values.append(grid)
            else:
                values.append([grid])

        if limit is not None and limit < 0:
            raise ValueError("max_hp_trials_per_model must be non-negative or None")

        total_combinations = 1
        for opts in values:
            total_combinations *= len(opts)
        if limit == 0:
            return []

        combos = []
        for combo in product(*values):
            combos.append(dict(zip(keys, combo)))
            if limit is not None and len(combos) >= limit:
                break
        return combos

    def count_param_combinations(
        self,
        overrides: Optional[dict[str, Any]] = None,
    ) -> int:
        """Return the total number of recipe parameter combinations."""
        if not self.recipe_params:
            return 1

        overrides = overrides or {}
        total = 1
        for key, entry in self.recipe_params.items():
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Parameter '{key}' must be a mapping with 'default' and optional 'grid'."
                )
            if "default" not in entry:
                raise ValueError(
                    f"Parameter '{key}' is missing required 'default' value in recipe."
                )
            if key in overrides and overrides[key] is not None:
                total *= 1
                continue
            grid = entry.get("grid", None)
            if grid is None:
                total *= 1
                continue
            if isinstance(grid, list):
                if len(grid) == 0:
                    raise ValueError(
                        f"Parameter '{key}' has an empty grid; omit 'grid' to use "
                        "the default as a single value."
                    )
                total *= len(grid)
            else:
                total *= 1
        return total

    def format_pipeline_id(self, param_values: dict[str, Any]) -> str:
        """Format pipeline_id with concrete param values.

        Args:
            param_values: Dict mapping param names to concrete values.

        Returns:
            Formatted pipeline_id string.

        Raises:
            ValueError: If pipeline_id references an unknown param.
        """
        try:
            return self.pipeline_id.format(**param_values)
        except KeyError as exc:
            missing = exc.args[0]
            raise ValueError(
                f"pipeline_id '{self.pipeline_id}' references unknown param '{missing}'. "
                f"Provide it under `recipe_params:` in the recipe."
            ) from exc

    def get_default(self, param_name: str) -> Any:
        """Get default value for a recipe parameter."""
        if param_name in self.recipe_params:
            entry = self.recipe_params[param_name]
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Parameter '{param_name}' must be a mapping with 'default' and optional 'grid'."
                )
            if "default" not in entry:
                raise ValueError(
                    f"Parameter '{param_name}' is missing required 'default' value in recipe."
                )
            return entry["default"]
        raise ValueError(
            f"Parameter '{param_name}' is not defined in recipe_params."
        )

    def render_kwargs(self, obj: dict[str, Any], param_values: dict[str, Any]) -> dict[str, Any]:
        """Render datamodule/model kwargs using param_values.

        Type rule:
        - If value == "{k}" exactly, substitute param_values[k] with original type.
        - Otherwise, if value is a string, apply {k} string formatting.
        - Non-string values are passed through unchanged.

        Args:
            obj: Dict of kwargs to render (e.g., datamodule_kwargs).
            param_values: Dict mapping param names to concrete values.

        Returns:
            Rendered kwargs dict with placeholders substituted.

        Raises:
            ValueError: If a kwarg references an unknown param.
        """
        result = {}
        for key, value in obj.items():
            if isinstance(value, str):
                # Check if it's an exact placeholder match like "{param}"
                if (
                    value.startswith("{")
                    and value.endswith("}")
                    and value.count("{") == 1
                ):
                    param_key = value[1:-1]
                    if param_key not in param_values:
                        raise ValueError(
                            f"Kwarg '{key}' references unknown param '{param_key}'. "
                            f"Provide it under `recipe_params:` in the recipe."
                        )
                    # Preserve original type (float, int, etc.)
                    result[key] = param_values[param_key]
                    continue
                # Otherwise do string formatting (result stays a string)
                try:
                    result[key] = value.format(**param_values)
                except KeyError as exc:
                    missing = exc.args[0]
                    raise ValueError(
                        f"Kwarg '{key}' references unknown param '{missing}' in '{value}'. "
                        f"Provide it under `recipe_params:` in the recipe."
                    ) from exc
            else:
                result[key] = value
        return result


__all__ = ["PipelineSpec", "VALID_PIPELINE_KINDS"]
