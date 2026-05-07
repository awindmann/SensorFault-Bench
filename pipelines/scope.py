from __future__ import annotations

from typing import Any, Mapping, Sequence

from config_loader import load_benchmark_scope
from utils.parsing import (
    parse_method_architecture_applicability,
    resolve_applicable_method_architecture_scope,
)


def load_benchmark_method_architecture_applicability(
    benchmark_scope: Mapping[str, Any] | None = None,
) -> dict[str, tuple[str, ...]]:
    manifest = load_benchmark_scope() if benchmark_scope is None else benchmark_scope
    if not isinstance(manifest, Mapping):
        raise ValueError("configs/benchmark_scope.yaml must be a mapping.")
    if "methods" not in manifest:
        raise ValueError("configs/benchmark_scope.yaml is missing methods metadata.")
    if "architectures" not in manifest:
        raise ValueError("configs/benchmark_scope.yaml is missing architectures metadata.")
    methods_section = manifest["methods"]
    architectures_section = manifest["architectures"]
    if not isinstance(methods_section, Mapping):
        raise ValueError("configs/benchmark_scope.yaml is missing methods metadata.")
    if not isinstance(architectures_section, Mapping):
        raise ValueError("configs/benchmark_scope.yaml is missing architectures metadata.")
    if "method_architecture_applicability" not in methods_section:
        raise ValueError(
            "configs/benchmark_scope.yaml is missing method applicability metadata."
        )
    if "display_order" not in methods_section:
        raise ValueError("configs/benchmark_scope.yaml is missing method display order.")
    if "display_order" not in architectures_section:
        raise ValueError(
            "configs/benchmark_scope.yaml is missing architecture display order."
        )
    return parse_method_architecture_applicability(
        methods_section["method_architecture_applicability"],
        benchmark_methods=methods_section["display_order"],
        benchmark_architectures=architectures_section["display_order"],
        key="configs/benchmark_scope.yaml methods.method_architecture_applicability",
    )


def resolve_benchmark_method_architecture_scope(
    *,
    methods: Sequence[str],
    architectures: Sequence[str],
    explicit_architectures: bool,
    benchmark_scope: Mapping[str, Any] | None = None,
    context: str,
) -> dict[str, tuple[str, ...]]:
    applicability = load_benchmark_method_architecture_applicability(benchmark_scope)
    return resolve_applicable_method_architecture_scope(
        methods=methods,
        architectures=architectures,
        applicability=applicability,
        explicit_architectures=explicit_architectures,
        context=context,
    )
