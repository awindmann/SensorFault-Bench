from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Optional

from .specs import PipelineSpec


PIPELINE_CONFIGS_DIR = Path("configs/pipelines")
PIPELINE_RECIPE_PATHS_BY_METHOD: dict[str, Path] = {
    "baseline": PIPELINE_CONFIGS_DIR / "baseline.yaml",
    "randomized_training": PIPELINE_CONFIGS_DIR / "randomized_training.yaml",
    "adversarial_training": PIPELINE_CONFIGS_DIR / "adversarial_training.yaml",
    "fault_augmentation": PIPELINE_CONFIGS_DIR / "fault_augmentation.yaml",
    "adaptive_robust_loss": PIPELINE_CONFIGS_DIR / "adaptive_robust_loss.yaml",
    "revin": PIPELINE_CONFIGS_DIR / "revin.yaml",
    "ensemble": PIPELINE_CONFIGS_DIR / "ensemble.yaml",
    "randomized_smoothing": PIPELINE_CONFIGS_DIR / "randomized_smoothing.yaml",
}


def require_pipeline_method_value(
    pipeline_method: Any,
    *,
    context: str,
) -> str:
    if pipeline_method is None:
        raise ValueError(f"{context}: pipeline_method must be a non-empty string.")
    method = str(pipeline_method).strip()
    if not method:
        raise ValueError(f"{context}: pipeline_method must be a non-empty string.")
    return method


def load_recipe_specs(
    recipe_order: Optional[Collection[str]] = None,
) -> list[PipelineSpec]:
    if recipe_order is None:
        raise ValueError(
            "load_recipe_specs requires an explicit recipe order. Runtime benchmark "
            "method order comes from configs/defaults.yaml BENCHMARK_METHODS."
        )
    recipe_names = tuple(recipe_order)
    specs: list[PipelineSpec] = []
    for recipe_name in recipe_names:
        recipe_path = PIPELINE_CONFIGS_DIR / recipe_name
        if not recipe_path.exists():
            raise FileNotFoundError(
                f"Recipe '{recipe_name}' not found under {PIPELINE_CONFIGS_DIR}."
            )
        specs.append(PipelineSpec.from_yaml(recipe_path))
    return specs


def load_pipeline_spec_for_method(
    pipeline_method: str,
    *,
    allowed_methods: Optional[Collection[str]] = None,
    context: str = "load_pipeline_spec_for_method",
) -> PipelineSpec:
    method = require_pipeline_method_value(pipeline_method, context=context)
    if allowed_methods is not None and method not in allowed_methods:
        known = ", ".join(sorted(allowed_methods))
        raise ValueError(
            f"Unknown pipeline_method '{method}'. Known methods: {known}."
        )
    recipe_path = PIPELINE_RECIPE_PATHS_BY_METHOD.get(method)
    if recipe_path is None:
        known = ", ".join(sorted(PIPELINE_RECIPE_PATHS_BY_METHOD.keys()))
        raise ValueError(
            f"Unknown pipeline_method '{method}'. Known methods: {known}."
        )
    if not recipe_path.exists():
        raise FileNotFoundError(
            f"Pipeline recipe for method '{method}' not found at '{recipe_path}'."
        )
    spec = PipelineSpec.from_yaml(recipe_path)
    declared_method = str(spec.pipeline_method).strip()
    if declared_method != method:
        raise ValueError(
            f"Recipe '{recipe_path}' declares pipeline_method='{declared_method}', "
            f"expected '{method}'."
        )
    return spec


__all__ = [
    "PIPELINE_CONFIGS_DIR",
    "PIPELINE_RECIPE_PATHS_BY_METHOD",
    "require_pipeline_method_value",
    "load_pipeline_spec_for_method",
    "load_recipe_specs",
]
