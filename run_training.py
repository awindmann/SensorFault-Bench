import sys
from typing import Any

import mlflow

from config_loader import (
    build_parser,
    load_dataset_windows,
    load_defaults,
    parse_explicit_cli_overrides,
)
from data.datasets import resolve_with_defaults
from pipelines.runner import (
    create_pipeline_runner,
    print_coverage_report,
    print_coverage_summary_and_raise_on_incomplete,
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
from pipelines.specs import PipelineSpec
from pipelines.training import get_tracking_uri
from utils.env import set_mlflow_storage_env
from utils.parsing import (
    resolve_dataset_window_args,
)


def _recipe_specs_by_method(specs: list[PipelineSpec]) -> dict[str, PipelineSpec]:
    specs_by_method: dict[str, PipelineSpec] = {}
    for spec in specs:
        method = str(spec.pipeline_method).strip()
        if not method:
            raise ValueError("Pipeline spec is missing required pipeline_method.")
        if method in specs_by_method:
            raise ValueError(f"Duplicate configured pipeline_method '{method}'.")
        specs_by_method[method] = spec
    return specs_by_method


def _recipe_param_defaults(extracted_defaults: list[dict[str, Any]]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for defaults_dict in extracted_defaults:
        defaults.update(defaults_dict)
    return defaults


def main() -> None:
    base_defaults = load_defaults()
    dataset_window_defaults = load_dataset_windows(defaults=base_defaults)
    pipeline_specs = load_benchmark_recipe_specs_for_scope(base_defaults)
    specs_by_method = _recipe_specs_by_method(pipeline_specs)
    extracted_defaults = extract_recipe_defaults_for_scope(pipeline_specs)
    defaults = merge_recipe_defaults_for_scope(base_defaults, extracted_defaults)
    parser = build_parser(defaults)
    args = parser.parse_args()
    args._explicit_cli_args = tuple(sys.argv[1:])
    args._recipe_param_overrides = parse_explicit_cli_overrides(
        _recipe_param_defaults(extracted_defaults),
        extra_args=args._explicit_cli_args,
    )
    args._window_arg_overrides = parse_explicit_cli_overrides(
        {
            "input_len": defaults["INPUT_LEN"],
            "target_len": defaults["TARGET_LEN"],
            "batch_size": defaults["BATCH_SIZE"],
        },
        extra_args=args._explicit_cli_args,
    )

    methods = resolve_requested_methods(
        args,
        configured_methods=tuple(specs_by_method.keys()),
    )
    architectures = resolve_requested_architectures(args)
    method_architecture_scope = resolve_benchmark_method_architecture_scope(
        methods=methods,
        architectures=architectures,
        explicit_architectures=has_explicit_architecture_scope(args),
        context="run_training.py",
    )

    set_mlflow_storage_env(args)

    tracking_uri = get_tracking_uri(args.logdir)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    dataset_specs = resolve_with_defaults(
        datasets=args.data_files,
        targets=args.data_targets,
        data_root=args.data_root,
    )

    coverage_reports = []
    for method in methods:
        if method not in method_architecture_scope:
            continue
        spec = specs_by_method[method]
        print(f"Running pipeline: {spec.pipeline_id}")
        if spec.model_hparams_mode == "inherit_baseline":
            print(
                "Warning: This recipe inherits hyperparameters from the best "
                "existing baseline run (selected by lowest best_val_loss). "
                "CLI hyperparameter tweaks do not affect that selection."
            )
        for dataset_spec in dataset_specs:
            dataset_args = resolve_dataset_window_args(
                args,
                dataset_spec=dataset_spec,
                dataset_window_defaults=dataset_window_defaults,
                explicit_arg_overrides=args._window_arg_overrides,
            )
            runner = create_pipeline_runner(spec, dataset_args)
            for arch in method_architecture_scope[method]:
                runner.run(client, dataset_spec, arch)
                report = runner.get_last_run_report()
                if report is None:
                    raise ValueError(
                        "Pipeline runner did not emit a RunExecutionReport for "
                        f"{dataset_spec.key}/{arch}/{spec.pipeline_method}."
                    )
                coverage_reports.append(report)
                print_coverage_report(report)

    print_coverage_summary_and_raise_on_incomplete(coverage_reports)


if __name__ == "__main__":
    main()
