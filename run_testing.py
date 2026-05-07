from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from config_loader import (
    build_parser,
    load_dataset_windows,
    load_defaults,
    parse_explicit_cli_overrides,
)
from data.datasets import resolve_with_defaults
from pipelines.selection import (
    CoverageMismatchError,
    extract_recipe_defaults_for_scope,
    has_explicit_architecture_scope,
    load_benchmark_recipe_specs_for_scope,
    merge_recipe_defaults_for_scope,
    resolve_benchmark_method_architecture_scope,
    resolve_requested_architectures,
    resolve_requested_methods,
)
from testing.evaluation import test_on_dataset
from testing.shared import _configure_runtime_loggers_for_testing
from utils.env import set_mlflow_storage_env
from utils.parsing import (
    require_namespace_bool,
    resolve_dataset_window_args,
)


def _normalize_orchestration_booleans(args: Any) -> None:
    args.full_coverage = require_namespace_bool(args, key="full_coverage")
    args.rerun = require_namespace_bool(args, key="rerun")


def prepare_testing_entrypoint_args(
    cli_args: Sequence[str] | None = None,
) -> tuple[Any, Sequence[Any], dict[str, dict[str, int]]]:
    explicit_cli_args = tuple(sys.argv[1:] if cli_args is None else cli_args)
    defaults = load_defaults()
    recipe_specs = load_benchmark_recipe_specs_for_scope(defaults)
    dataset_window_defaults = load_dataset_windows(defaults=defaults)
    extracted_defaults = extract_recipe_defaults_for_scope(recipe_specs)
    defaults = merge_recipe_defaults_for_scope(defaults, extracted_defaults)
    parser = build_parser(defaults)
    args = parser.parse_args(list(explicit_cli_args))

    recipe_param_defaults: dict[str, Any] = {}
    for defaults_dict in extracted_defaults:
        recipe_param_defaults.update(defaults_dict)
    args._explicit_cli_args = explicit_cli_args
    args._recipe_param_overrides = parse_explicit_cli_overrides(
        recipe_param_defaults,
        extra_args=explicit_cli_args,
    )
    args._window_arg_overrides = parse_explicit_cli_overrides(
        {
            "input_len": defaults["INPUT_LEN"],
            "target_len": defaults["TARGET_LEN"],
            "batch_size": defaults["BATCH_SIZE"],
        },
        extra_args=explicit_cli_args,
    )
    _normalize_orchestration_booleans(args)
    configured_methods = tuple(str(spec.pipeline_method).strip() for spec in recipe_specs)
    requested_methods = resolve_requested_methods(
        args,
        configured_methods=configured_methods,
    )
    requested_architectures = resolve_requested_architectures(args)
    args._method_architecture_scope = resolve_benchmark_method_architecture_scope(
        methods=requested_methods,
        architectures=requested_architectures,
        explicit_architectures=has_explicit_architecture_scope(args),
        context="run_testing.py",
    )
    return args, recipe_specs, dataset_window_defaults


def main() -> None:
    args, recipe_specs, dataset_window_defaults = prepare_testing_entrypoint_args()
    _configure_runtime_loggers_for_testing()

    set_mlflow_storage_env(args)

    resolved_specs = resolve_with_defaults(
        datasets=args.data_files,
        targets=args.data_targets,
        data_root=args.data_root,
    )
    dataset_failures: list[tuple[str, str, str]] = []
    for i, dataset_spec in enumerate(resolved_specs):
        print(f"\nTesting dataset {i+1}/{len(resolved_specs)}: {dataset_spec.key} ({dataset_spec.path}).")
        dataset_args = resolve_dataset_window_args(
            args,
            dataset_spec=dataset_spec,
            dataset_window_defaults=dataset_window_defaults,
            explicit_arg_overrides=args._window_arg_overrides,
        )
        try:
            test_on_dataset(
                dataset_spec=dataset_spec,
                args=dataset_args,
                recipe_specs_for_scope=recipe_specs,
            )
        except Exception as exc:
            if isinstance(exc, CoverageMismatchError):
                raise
            if args.raise_error:
                raise
            error_type = type(exc).__name__
            error_message = str(exc).strip() or repr(exc)
            dataset_failures.append((str(dataset_spec.key), error_type, error_message))
            print(
                f"Error during testing of dataset {dataset_spec.key}: "
                f"{error_type}: {error_message}"
            )

    if dataset_failures:
        preview = "; ".join(
            f"{dataset_name} ({error_type}): {message}"
            for dataset_name, error_type, message in dataset_failures[:5]
        )
        remainder = len(dataset_failures) - 5
        if remainder > 0:
            preview += f"; ... (+{remainder} more)"
        print(
            "Testing completed with dataset failures under --no-raise-error. "
            f"Failures: {preview}"
        )
        return


if __name__ == "__main__":
    main()
