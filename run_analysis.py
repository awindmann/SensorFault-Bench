from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import Any

from config_loader import (
    build_parser,
    load_defaults,
    parse_explicit_cli_overrides,
)
from pipelines.selection import (
    extract_recipe_defaults_for_scope,
    has_explicit_architecture_scope,
    load_benchmark_recipe_specs_for_scope,
    merge_recipe_defaults_for_scope,
    resolve_benchmark_method_architecture_scope,
    resolve_requested_architectures,
    resolve_requested_methods,
)
from testing.meta_analysis import meta_analysis
from testing.shared import _configure_runtime_loggers_for_testing
from utils.env import set_mlflow_storage_env


def _recipe_param_defaults(
    extracted_defaults: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    recipe_param_defaults: dict[str, Any] = {}
    for defaults_dict in extracted_defaults:
        recipe_param_defaults.update(defaults_dict)
    return recipe_param_defaults


def prepare_analysis_entrypoint_args(
    cli_args: Sequence[str] | None = None,
):
    explicit_cli_args = tuple(sys.argv[1:] if cli_args is None else cli_args)
    defaults = load_defaults()
    recipe_specs = load_benchmark_recipe_specs_for_scope(defaults)
    extracted_defaults = extract_recipe_defaults_for_scope(recipe_specs)
    recipe_param_defaults = _recipe_param_defaults(extracted_defaults)
    defaults = merge_recipe_defaults_for_scope(defaults, extracted_defaults)
    parser = build_parser(defaults)
    args = parser.parse_args(list(explicit_cli_args))
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
    configured_methods = tuple(
        str(spec.pipeline_method).strip() for spec in recipe_specs
    )
    requested_methods = resolve_requested_methods(
        args,
        configured_methods=configured_methods,
    )
    requested_architectures = resolve_requested_architectures(args)
    args._method_architecture_scope = resolve_benchmark_method_architecture_scope(
        methods=requested_methods,
        architectures=requested_architectures,
        explicit_architectures=has_explicit_architecture_scope(args),
        context="run_analysis.py",
    )
    return args


def main() -> None:
    args = prepare_analysis_entrypoint_args()
    _configure_runtime_loggers_for_testing()

    set_mlflow_storage_env(args)
    meta_analysis(args, coverage_fractions_by_dataset=None)


if __name__ == "__main__":
    main()
