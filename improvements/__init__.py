from typing import Dict, Iterable

from .base import (
    BackboneReference,
    BaseImprovement,
    ImprovementBuilder,
    ImprovementRegistration,
    ImprovementSpec,
    WrapExecutionOutcome,
    build_backbone_reference,
    download_backbone_reference_checkpoint,
    load_improvement_spec,
    resolve_backbone_source_run,
)
from .ensemble import EnsembleImprovement, build_ensemble_model
from .randomized_smoothing import RandomizedSmoothingImprovement, build_randomized_smoothing_model


_IMPROVEMENTS: Iterable[ImprovementRegistration] = (
    ImprovementRegistration(
        name="ensemble",
        recipe_cls=EnsembleImprovement,
        builder=build_ensemble_model,
    ),
    ImprovementRegistration(
        name="randomized_smoothing",
        recipe_cls=RandomizedSmoothingImprovement,
        builder=build_randomized_smoothing_model,
    ),
)


def _build_registries() -> tuple[
    Dict[str, ImprovementRegistration],
    Dict[str, ImprovementRegistration],
]:
    benchmark_name_registry: Dict[str, ImprovementRegistration] = {}
    loader_kind_registry: Dict[str, ImprovementRegistration] = {}
    for entry in _IMPROVEMENTS:
        registry_name = getattr(entry.recipe_cls, "registry_name", None)
        if registry_name is None or not str(registry_name).strip():
            raise ValueError(
                f"Improvement recipe {entry.recipe_cls.__name__} must define a non-empty registry_name."
            )
        loader_token = getattr(entry.recipe_cls, "loader_kind_key", None)
        if loader_token is None or not str(loader_token).strip():
            raise ValueError(
                f"Improvement recipe {entry.recipe_cls.__name__} must define a non-empty loader_kind_key."
            )
        benchmark_tokens = {
            entry.name,
            str(registry_name).strip(),
        }
        loader_kind_registry[str(loader_token).strip().lower()] = entry
        for token in filter(None, benchmark_tokens):
            benchmark_name_registry[token.lower()] = entry
    return benchmark_name_registry, loader_kind_registry


_BENCHMARK_NAME_REGISTRY, _LOADER_KIND_REGISTRY = _build_registries()


def list_available_improvements() -> Dict[str, ImprovementRegistration]:
    """Return a mapping of user-facing tokens to improvement registrations."""
    return dict(_BENCHMARK_NAME_REGISTRY)


def get_improvement_registration(token: str) -> ImprovementRegistration:
    return _BENCHMARK_NAME_REGISTRY[token.lower()]


def get_registration_by_loader_kind(loader_kind: str) -> ImprovementRegistration:
    return _LOADER_KIND_REGISTRY[loader_kind.lower()]


__all__ = [
    "BackboneReference",
    "BaseImprovement",
    "ImprovementBuilder",
    "ImprovementRegistration",
    "ImprovementSpec",
    "WrapExecutionOutcome",
    "EnsembleImprovement",
    "RandomizedSmoothingImprovement",
    "list_available_improvements",
    "get_improvement_registration",
    "get_registration_by_loader_kind",
    "build_backbone_reference",
    "download_backbone_reference_checkpoint",
    "load_improvement_spec",
    "resolve_backbone_source_run",
]
