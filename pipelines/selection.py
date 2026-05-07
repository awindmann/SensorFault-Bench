from __future__ import annotations

from collections import defaultdict
import json
import tempfile
from typing import Any, Callable, Mapping, Optional, Sequence

import pandas as pd

from config_loader import load_defaults, load_hparams
from data.perturbations import build_perturbation_scenario_params_signature
from pipelines.recipes import (
    PIPELINE_CONFIGS_DIR,
    PIPELINE_RECIPE_PATHS_BY_METHOD,
    load_pipeline_spec_for_method,
)
from pipelines.ranking import (
    rank_key_for_dataframe_row,
    sort_runs_by_metric,
    validate_pipeline_tags_for_selection,
)
from pipelines.runner import (
    param_overrides_for_spec,
    resolve_wrap_base_pipeline_method,
    scope_policy_skip_reason_for_spec,
)
from pipelines.scope import (
    load_benchmark_method_architecture_applicability,
    resolve_benchmark_method_architecture_scope,
)
from pipelines.specs import PipelineSpec
from utils.parsing import (
    DEGRADATION_SCORING_SEMANTICS,
    build_perturbation_scenarios_signature,
    coerce_int,
    extract_required_typed_hparams,
    shared_anchor_bootstrap_ci_context_matches,
    degradation_eval_context_matches,
    degradation_n_test_samples_meet_policy,
    optional_nonempty_tag_value,
    parse_backbone_run_ids,
    parse_method_scope,
    parse_model_architecture_scope,
    require_degradation_eval_context_from_args,
    require_degradation_eval_context_tags,
    require_robustness_results_complete_tag,
    require_shared_anchor_bootstrap_ci_context_from_args,
    require_shared_anchor_bootstrap_ci_context_tags,
    parse_optional_unit_float,
    parse_perturbation_channel_fraction_max,
    parse_perturbation_idx_name_map,
    parse_perturbation_scenarios,
    resolve_effective_eval_data_seed,
    require_tested_param,
    require_namespace_bool,
    require_namespace_value,
    require_nonempty_tag_value,
    require_perturbation_coupling_tags,
)
from utils.scoring import (
    build_canonical_degradation_context_signature,
    build_fixed_channel_fraction_tag_key,
    download_validated_degradation_artifact_bundle,
    download_validated_fixed_channel_fraction_artifact_bundle,
    require_logged_degradation_metric_bundle,
    require_logged_fixed_channel_fraction_metric_bundle,
)


ROBUSTNESS_SCORING_SEMANTICS = DEGRADATION_SCORING_SEMANTICS


def load_recipe_specs_for_scope() -> list[PipelineSpec]:
    return load_benchmark_recipe_specs_for_scope(load_defaults())


def load_benchmark_recipe_specs_for_scope(defaults: Mapping[str, Any]) -> list[PipelineSpec]:
    if "BENCHMARK_METHODS" not in defaults:
        raise ValueError("defaults is missing required key 'BENCHMARK_METHODS'.")
    methods = parse_method_scope(
        None,
        benchmark_methods=defaults["BENCHMARK_METHODS"],
        allowed=tuple(PIPELINE_RECIPE_PATHS_BY_METHOD.keys()),
    )
    return [
        load_pipeline_spec_for_method(
            method,
            allowed_methods=set(PIPELINE_RECIPE_PATHS_BY_METHOD.keys()),
        )
        for method in methods
    ]


def extract_recipe_defaults_for_scope(specs: list[PipelineSpec]) -> list[dict]:
    extracted = []
    for spec in specs:
        defaults = {}
        for key, entry in spec.recipe_params.items():
            if not isinstance(entry, dict) or "default" not in entry:
                raise ValueError(
                    f"Recipe param '{key}' in pipeline '{spec.pipeline_method}' "
                    "is missing required 'default'."
                )
            defaults[key] = entry["default"]
        extracted.append(defaults)
    return extracted


def merge_recipe_defaults_for_scope(
    global_defaults: dict,
    extracted_defaults: list[dict],
) -> dict:
    merged = dict(global_defaults)
    seen: dict[str, Any] = {}
    for defaults_dict in extracted_defaults:
        for key, value in defaults_dict.items():
            upper = key.upper()
            if upper in seen and seen[upper] != value:
                raise ValueError(
                    f"Conflicting defaults for '{key}': {seen[upper]} vs {value}."
                )
            seen[upper] = value
            merged[upper] = value
    return merged


def resolve_requested_architectures(args) -> list[str]:
    available = list(load_hparams().keys())
    requested_model = require_namespace_value(args, key="model")
    benchmark_architectures = (
        require_namespace_value(
            args,
            key="benchmark_architectures",
        )
        if requested_model is None
        else ()
    )
    return parse_model_architecture_scope(
        requested_model,
        benchmark_architectures=benchmark_architectures,
        allowed=available,
    )


def resolve_requested_methods(
    args,
    *,
    configured_methods: Sequence[str],
) -> list[str]:
    requested_method = require_namespace_value(args, key="method")
    benchmark_methods = (
        require_namespace_value(args, key="benchmark_methods")
        if requested_method is None
        else ()
    )
    return parse_method_scope(
        requested_method,
        benchmark_methods=benchmark_methods,
        allowed=tuple(configured_methods),
    )


def has_explicit_model_architecture_scope(args) -> bool:
    return require_namespace_value(args, key="model") is not None


def _explicit_cli_option_present(args, *, option: str) -> bool:
    raw_args = getattr(args, "_explicit_cli_args", ())
    if raw_args is None:
        return False
    if not isinstance(raw_args, (list, tuple)):
        raise ValueError("args._explicit_cli_args must be a list or tuple of CLI tokens.")
    flag = f"--{option}"
    for token in raw_args:
        if not isinstance(token, str):
            raise ValueError("args._explicit_cli_args must contain only string CLI tokens.")
        if token == flag or token.startswith(f"{flag}="):
            return True
    return False


def has_explicit_architecture_scope(args) -> bool:
    return has_explicit_model_architecture_scope(args) or _explicit_cli_option_present(
        args,
        option="benchmark-architectures",
    )


def expand_testing_method_scope_for_wrap_dependencies(
    args,
    *,
    requested_methods: set[str],
    recipe_spec_by_method: Mapping[str, PipelineSpec],
) -> set[str]:
    expanded_methods = {str(method) for method in requested_methods}
    if not expanded_methods:
        raise ValueError("requested_methods must be non-empty.")
    missing_methods = sorted(
        method for method in expanded_methods if method not in recipe_spec_by_method
    )
    if missing_methods:
        raise ValueError(
            "Cannot expand testing method scope: missing recipe specs for "
            f"{missing_methods}."
        )

    # Single pass: wrap methods depend on train-kind methods only, so no
    # transitive wrap-to-wrap chains exist and one iteration suffices.
    for pipeline_method in sorted(expanded_methods):
        spec = recipe_spec_by_method[pipeline_method]
        pipeline_kind = str(spec.pipeline_kind).strip()
        if pipeline_kind != "wrap":
            continue
        overrides = param_overrides_for_spec(spec, args)
        for param_values in spec.expand_params(overrides=overrides):
            base_method = resolve_wrap_base_pipeline_method(
                pipeline_method,
                param_values,
            )
            base_spec = recipe_spec_by_method.get(base_method)
            if base_spec is None:
                raise ValueError(
                    f"Wrap method '{pipeline_method}' depends on unknown "
                    f"base pipeline_method '{base_method}'."
                )
            base_kind = str(base_spec.pipeline_kind).strip()
            if base_kind != "train":
                raise ValueError(
                    f"Wrap method '{pipeline_method}' requires train-kind "
                    f"dependencies, but base pipeline_method '{base_method}' "
                    f"has pipeline_kind='{base_kind}'."
                )
            expanded_methods.add(base_method)
    return expanded_methods


class CoverageMismatchError(RuntimeError):
    """Raised when deterministic expected-vs-seen coverage checks fail."""


def _coverage_preview(values: Sequence[Any] | Any, *, limit: int = 10) -> str:
    if isinstance(values, str):
        tokens = [values]
    else:
        try:
            tokens = sorted(str(value) for value in values)
        except TypeError:
            tokens = [str(values)]
    if not tokens:
        return "none"
    preview = tokens[:limit]
    rendered = ", ".join(preview)
    if len(tokens) > limit:
        rendered += f", ... (+{len(tokens) - limit} more)"
    return rendered


def _collect_family_variant_runs(
    runs_by_variant: Mapping[tuple[str, str, str], Sequence[Any]],
    *,
    arch: str,
    pipeline_method: str,
) -> dict[str, list[Any]]:
    collected: dict[str, list[Any]] = {}
    for (candidate_arch, candidate_method, pipeline_id), runs in runs_by_variant.items():
        if str(candidate_arch) != str(arch) or str(candidate_method) != str(pipeline_method):
            continue
        if not runs:
            continue
        collected[str(pipeline_id)] = list(runs)
    return collected


def _collect_training_family_signature_details(
    *,
    arch: str,
    pipeline_method: str,
    spec: PipelineSpec,
    runs_by_variant: Mapping[tuple[str, str, str], Sequence[Any]],
    resolved_by_run_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    variant_runs = _collect_family_variant_runs(
        runs_by_variant,
        arch=str(arch),
        pipeline_method=str(pipeline_method),
    )
    return _build_training_family_signature_details(
        arch=arch,
        pipeline_method=pipeline_method,
        spec=spec,
        variant_runs=variant_runs,
        resolved_by_run_id=resolved_by_run_id,
    )


def _build_training_family_signature_details(
    *,
    arch: str,
    pipeline_method: str,
    spec: PipelineSpec,
    variant_runs: Mapping[str, Sequence[Any]],
    resolved_by_run_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_kind = str(spec.pipeline_kind).strip()
    signature_to_run_ids: dict[str, list[str]] = defaultdict(list)
    family_run_ids: set[str] = set()
    for runs in variant_runs.values():
        for run in runs:
            resolved = resolved_by_run_id.get(run.info.run_id)
            if resolved is None:
                raise ValueError(
                    f"Run {run.info.run_id} is missing resolved pipeline tags "
                    "during coverage auditing."
                )
            actual_kind = str(resolved["pipeline_kind"]).strip()
            if actual_kind != expected_kind:
                raise ValueError(
                    f"Coverage audit found run {run.info.run_id} under "
                    f"({arch}, {pipeline_method}) with pipeline_kind='{actual_kind}', "
                    f"expected '{expected_kind}'."
                )
            tags = run.data.tags
            if tags is None:
                raise ValueError(
                    f"Run {run.info.run_id} is missing tags during signature coverage auditing."
                )
            signature = require_nonempty_tag_value(
                tags,
                key="signature",
                run_id=run.info.run_id,
            )
            signature_to_run_ids[str(signature)].append(run.info.run_id)
            family_run_ids.add(run.info.run_id)
    duplicates = {
        signature: sorted(run_ids)
        for signature, run_ids in signature_to_run_ids.items()
        if len(run_ids) > 1
    }
    return {
        "variant_runs": variant_runs,
        "family_run_ids": family_run_ids,
        "seen_signatures": set(signature_to_run_ids.keys()),
        "duplicates": duplicates,
    }


def _late_filter_inherit_baseline_family_variant_runs(
    *,
    arch: str,
    pipeline_method: str,
    pipeline_kind: str,
    recipe_spec_by_method: Mapping[str, PipelineSpec],
    expected_tuning_scope: Callable[..., Any],
    variant_runs: Mapping[str, Sequence[Any]],
) -> tuple[dict[str, list[Any]], dict[str, str]]:
    filtered_variant_runs: dict[str, list[Any]] = {}
    dropped_run_reasons: dict[str, str] = {}
    for pipeline_id, runs in variant_runs.items():
        kept_runs: list[Any] = []
        for run in runs:
            reason = scope_exclusion_reason(
                run,
                arch=arch,
                pipeline_method=pipeline_method,
                pipeline_kind=pipeline_kind,
                recipe_spec_by_method=recipe_spec_by_method,
                expected_tuning_scope=expected_tuning_scope,
                enforce_inherit_baseline_scope=True,
            )
            if reason is not None:
                dropped_run_reasons[run.info.run_id] = reason
                continue
            kept_runs.append(run)
        if kept_runs:
            filtered_variant_runs[str(pipeline_id)] = kept_runs
    return filtered_variant_runs, dropped_run_reasons


def _resolve_base_pipeline_method(tags: Mapping[str, Any]) -> Optional[str]:
    base_method = tags.get("base_pipeline_method")
    if base_method is None:
        return None
    normalized = str(base_method).strip()
    return normalized or None


def _collect_wrap_family_variant_details(
    *,
    arch: str,
    pipeline_method: str,
    spec: PipelineSpec,
    runs_by_variant: Mapping[tuple[str, str, str], Sequence[Any]],
    resolved_by_run_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    variant_runs = _collect_family_variant_runs(
        runs_by_variant,
        arch=str(arch),
        pipeline_method=str(pipeline_method),
    )
    expected_kind = str(spec.pipeline_kind).strip()
    family_run_ids: set[str] = set()
    seen_base_method_by_pipeline_id: dict[str, str] = {}
    for pipeline_id, runs in variant_runs.items():
        base_methods: set[str] = set()
        for run in runs:
            resolved = resolved_by_run_id.get(run.info.run_id)
            if resolved is None:
                raise ValueError(
                    f"Run {run.info.run_id} is missing resolved pipeline tags "
                    "during wrap coverage auditing."
                )
            actual_kind = str(resolved["pipeline_kind"]).strip()
            if actual_kind != expected_kind:
                raise ValueError(
                    f"Coverage audit found wrap run {run.info.run_id} under "
                    f"({arch}, {pipeline_method}, {pipeline_id}) with "
                    f"pipeline_kind='{actual_kind}', expected '{expected_kind}'."
                )
            tags = run.data.tags
            if tags is None:
                raise ValueError(
                    f"Run {run.info.run_id} is missing tags during wrap coverage auditing."
                )
            base_method = _resolve_base_pipeline_method(tags)
            if base_method is None:
                raise ValueError(
                    f"Wrap run {run.info.run_id} is missing required base_pipeline_method "
                    "during coverage auditing."
                )
            base_methods.add(str(base_method))
            family_run_ids.add(run.info.run_id)
        if len(base_methods) != 1:
            raise ValueError(
                f"Wrap variant ({arch}, {pipeline_method}, {pipeline_id}) has "
                f"inconsistent base_pipeline_method values: {sorted(base_methods)}."
            )
        seen_base_method_by_pipeline_id[str(pipeline_id)] = next(iter(base_methods))
    return {
        "variant_runs": variant_runs,
        "family_run_ids": family_run_ids,
        "seen_pipeline_ids": set(seen_base_method_by_pipeline_id.keys()),
        "seen_base_method_by_pipeline_id": seen_base_method_by_pipeline_id,
    }


def _build_expected_wrap_variants_for_coverage(
    *,
    spec: PipelineSpec,
    args: Any,
) -> dict[str, dict[str, Any]]:
    pipeline_kind = str(spec.pipeline_kind).strip()
    if pipeline_kind != "wrap":
        raise ValueError(
            "_build_expected_wrap_variants_for_coverage requires a wrap pipeline spec."
        )
    overrides = param_overrides_for_spec(spec, args)
    expected_variants: dict[str, dict[str, Any]] = {}
    for param_values in spec.expand_params(overrides=overrides):
        pipeline_id = spec.format_pipeline_id(param_values)
        if pipeline_id in expected_variants:
            raise ValueError(
                f"Wrap coverage expected unique pipeline_id values for "
                f"pipeline_method='{spec.pipeline_method}', found duplicate '{pipeline_id}'."
            )
        base_method = resolve_wrap_base_pipeline_method(
            str(spec.pipeline_method), param_values,
        )
        expected_variants[str(pipeline_id)] = {
            "base_pipeline_method": str(base_method).strip(),
        }
    return expected_variants


def scope_exclusion_reason(
    run,
    *,
    arch: str,
    pipeline_method: str,
    pipeline_kind: str,
    recipe_spec_by_method: Mapping[str, PipelineSpec],
    expected_tuning_scope: Callable[..., Any],
    enforce_inherit_baseline_scope: bool = False,
) -> Optional[str]:
    kind = str(pipeline_kind).strip()
    if kind not in {"train", "finetune"}:
        return None
    spec = recipe_spec_by_method.get(str(pipeline_method))
    if spec is None:
        raise ValueError(
            f"Pipeline method '{pipeline_method}' is not configured for scope filtering."
        )
    if str(spec.pipeline_kind).strip() != kind:
        raise ValueError(
            f"Pipeline method '{pipeline_method}' resolved to kind '{spec.pipeline_kind}', "
            f"but scope filtering requires kind '{kind}'."
        )
    if pipeline_method != "baseline":
        run_hparams_mode = optional_nonempty_tag_value(
            run.data.tags, key="hparams_mode",
        )
        if run_hparams_mode is not None:
            spec_mode = str(spec.model_hparams_mode).strip()
            if run_hparams_mode != spec_mode:
                return "hparams_mode_mismatch"
    if (
        str(spec.model_hparams_mode).strip() == "inherit_baseline"
        and not enforce_inherit_baseline_scope
    ):
        return None
    scope = expected_tuning_scope(
        arch=arch,
        pipeline_method=str(pipeline_method),
        pipeline_kind=kind,
    )
    tags = run.data.tags
    if tags is None:
        raise ValueError(
            f"Run {run.info.run_id} is missing tags required for scope filtering."
        )
    signature = require_nonempty_tag_value(
        tags,
        key="signature",
        run_id=run.info.run_id,
    )
    tuning_strategy = require_nonempty_tag_value(
        tags,
        key="tuning_strategy",
        run_id=run.info.run_id,
    )
    if tuning_strategy != "random_subgrid":
        return "tuning_strategy_mismatch"
    tuning_scope_key = require_nonempty_tag_value(
        tags,
        key="tuning_scope_key",
        run_id=run.info.run_id,
    )
    if tuning_scope_key != scope.scope_key:
        return "tuning_scope_key_mismatch"
    tuning_seed = require_nonempty_tag_value(
        tags,
        key="tuning_seed",
        run_id=run.info.run_id,
    )
    try:
        int(tuning_seed)
    except ValueError as exc:
        raise ValueError(
            f"Run {run.info.run_id} has non-integer tuning_seed '{tuning_seed}'."
        ) from exc
    if tuning_seed != str(scope.tuning_seed):
        return "tuning_seed_mismatch"
    if signature not in scope.signature_set:
        return "signature_out_of_scope"
    return None


def audit_and_apply_testing_coverage_policy(
    *,
    args,
    full_coverage: bool,
    requested_architectures: set[str],
    requested_methods: set[str],
    recipe_spec_by_method: Mapping[str, PipelineSpec],
    runs_by_variant: Mapping[tuple[str, str, str], Sequence[Any]],
    resolved_by_run_id: Mapping[str, Mapping[str, Any]],
    expected_tuning_scope: Callable[..., Any],
) -> dict[str, Any]:
    requested_arches_sorted = sorted(str(arch) for arch in requested_architectures)
    requested_methods_sorted = sorted(str(method) for method in requested_methods)
    coverage_mismatches: list[str] = []
    arch_drop_reasons: dict[str, dict[str, Any]] = {}
    family_drop_reasons: dict[tuple[str, str], dict[str, Any]] = {}
    variant_drop_reasons: dict[tuple[str, str, str], dict[str, Any]] = {}
    dropped_run_reasons: dict[str, str] = {}
    baseline_blocked_arches: set[str] = set()
    coverage_fractions: dict[tuple[str, str], tuple[int, int]] = {}
    seen_methods_by_arch: dict[str, set[str]] = defaultdict(set)
    effective_runs_by_variant: dict[tuple[str, str, str], list[Any]] = {
        (str(key[0]), str(key[1]), str(key[2])): list(runs)
        for key, runs in runs_by_variant.items()
    }

    for (arch, pipeline_method, _), runs in effective_runs_by_variant.items():
        if not runs:
            continue
        seen_methods_by_arch[str(arch)].add(str(pipeline_method))

    baseline_spec = recipe_spec_by_method.get("baseline")
    if "baseline" in requested_methods and baseline_spec is None:
        raise ValueError(
            "Requested coverage scope includes baseline, but no baseline recipe spec is configured."
        )

    for arch in requested_arches_sorted:
        configured_methods_for_arch: list[str] = []
        for method in requested_methods_sorted:
            spec = recipe_spec_by_method.get(method)
            if spec is None:
                raise ValueError(
                    f"Requested method '{method}' is missing from configured recipe scope."
                )
            if scope_policy_skip_reason_for_spec(spec, arch) is not None:
                continue
            configured_methods_for_arch.append(str(method))
        seen = seen_methods_by_arch.get(arch, set())
        missing = sorted(set(configured_methods_for_arch) - seen)
        if missing:
            print(f"Coverage audit: {arch} missing methods: {', '.join(missing)}")

    if "baseline" in requested_methods:
        for arch in requested_arches_sorted:
            seen_methods = seen_methods_by_arch.get(arch, set())
            if baseline_spec is None:
                continue
            baseline_skip_reason = scope_policy_skip_reason_for_spec(baseline_spec, arch)
            baseline_expected_signatures: set[str] = set()
            if baseline_skip_reason is None:
                baseline_scope = expected_tuning_scope(
                    arch=arch,
                    pipeline_method="baseline",
                    pipeline_kind=str(baseline_spec.pipeline_kind).strip(),
                )
                baseline_expected_signatures = set(baseline_scope.signature_set)
            if baseline_expected_signatures:
                baseline_details = _collect_training_family_signature_details(
                    arch=arch,
                    pipeline_method="baseline",
                    spec=baseline_spec,
                    runs_by_variant=runs_by_variant,
                    resolved_by_run_id=resolved_by_run_id,
                )
                if baseline_details["duplicates"]:
                    duplicate_preview = _coverage_preview(
                        f"{signature} x{len(run_ids)}"
                        for signature, run_ids in baseline_details["duplicates"].items()
                    )
                    raise CoverageMismatchError(
                        f"Coverage integrity failure for baseline architecture '{arch}': "
                        f"duplicate signatures detected ({duplicate_preview})."
                    )
                unexpected_signatures = (
                    baseline_details["seen_signatures"] - baseline_expected_signatures
                )
                if unexpected_signatures:
                    message = (
                        f"Coverage integrity failure for baseline architecture '{arch}': "
                        "unexpected signatures escaped scope filtering "
                        f"({_coverage_preview(unexpected_signatures)})."
                    )
                    if full_coverage:
                        raise CoverageMismatchError(message)
                    baseline_blocked_arches.add(arch)
                    arch_drop_reasons[arch] = {
                        "reason_code": "coverage_drop_architecture_unexpected_signatures",
                        "message": message,
                    }
                    continue
                coverage_fractions[(arch, "baseline")] = (
                    len(baseline_details["seen_signatures"]),
                    len(baseline_expected_signatures),
                )
                missing_signatures = (
                    baseline_expected_signatures - baseline_details["seen_signatures"]
                )
                if missing_signatures:
                    baseline_blocked_arches.add(arch)
                    message = (
                        f"Baseline coverage mismatch for '{arch}': "
                        f"{len(baseline_details['seen_signatures'])}/"
                        f"{len(baseline_expected_signatures)} signatures present "
                        f"({len(missing_signatures)} missing)."
                    )
                    if full_coverage:
                        coverage_mismatches.append(message)
                    else:
                        arch_drop_reasons[arch] = {
                            "reason_code": "coverage_drop_architecture_baseline_mismatch",
                            "message": message,
                        }
            elif "baseline" in seen_methods:
                baseline_blocked_arches.add(arch)
                message = (
                    f"Baseline coverage mismatch for architecture '{arch}': "
                    "saw baseline runs even though baseline is out of the active expected scope."
                )
                if full_coverage:
                    coverage_mismatches.append(message)
                else:
                    arch_drop_reasons[arch] = {
                        "reason_code": "coverage_drop_architecture_unexpected_baseline",
                        "message": message,
                    }

    for arch in requested_arches_sorted:
        for method in requested_methods_sorted:
            if method == "baseline":
                continue
            spec = recipe_spec_by_method.get(method)
            if spec is None:
                raise ValueError(
                    f"Requested method '{method}' is missing from configured recipe scope."
                )
            kind = str(spec.pipeline_kind).strip()
            if kind == "wrap":
                continue
            if arch in arch_drop_reasons or arch in baseline_blocked_arches:
                family_drop_reasons[(arch, method)] = {
                    "reason_code": "coverage_drop_family_baseline_dependency",
                    "message": (
                        f"Coverage drop for ({arch}, {method}): "
                        "baseline coverage for this architecture was dropped."
                    ),
                }
                continue
            variant_runs = _collect_family_variant_runs(
                effective_runs_by_variant,
                arch=arch,
                pipeline_method=method,
            )
            skip_reason = scope_policy_skip_reason_for_spec(spec, arch)
            expected_signatures: set[str] = set()
            if skip_reason is None:
                scope = expected_tuning_scope(
                    arch=arch,
                    pipeline_method=method,
                    pipeline_kind=kind,
                )
                expected_signatures = set(scope.signature_set)
            if (
                expected_signatures
                and str(spec.model_hparams_mode).strip() == "inherit_baseline"
            ):
                # inherit_baseline families defer scope filtering during ingestion
                # because their valid scope depends on baseline coverage. Once the
                # current expected scope is known, enforce it here before any
                # family-level integrity checks.
                variant_runs, late_scope_drop_reasons = (
                    _late_filter_inherit_baseline_family_variant_runs(
                        arch=arch,
                        pipeline_method=method,
                        pipeline_kind=kind,
                        recipe_spec_by_method=recipe_spec_by_method,
                        expected_tuning_scope=expected_tuning_scope,
                        variant_runs=variant_runs,
                    )
                )
                dropped_run_reasons.update(late_scope_drop_reasons)
                for key in list(effective_runs_by_variant.keys()):
                    if key[0] == arch and key[1] == method:
                        del effective_runs_by_variant[key]
                for pipeline_id, runs in variant_runs.items():
                    effective_runs_by_variant[(arch, method, pipeline_id)] = list(runs)
            family_details = _build_training_family_signature_details(
                arch=arch,
                pipeline_method=method,
                spec=spec,
                variant_runs=variant_runs,
                resolved_by_run_id=resolved_by_run_id,
            )
            if family_details["duplicates"]:
                duplicate_preview = _coverage_preview(
                    f"{signature} x{len(run_ids)}"
                    for signature, run_ids in family_details["duplicates"].items()
                )
                raise CoverageMismatchError(
                    f"Coverage integrity failure for ({arch}, {method}): "
                    f"duplicate signatures detected ({duplicate_preview})."
                )
            if not expected_signatures:
                if family_details["family_run_ids"]:
                    message = (
                        f"Coverage drop for ({arch}, {method}): "
                        "family is out of the active expected scope but current-lineage runs exist."
                    )
                    if full_coverage:
                        coverage_mismatches.append(message)
                    else:
                        family_drop_reasons[(arch, method)] = {
                            "reason_code": "coverage_drop_family_unexpected_method",
                            "message": message,
                        }
                continue
            coverage_fractions[(arch, method)] = (
                len(family_details["seen_signatures"]),
                len(expected_signatures),
            )
            unexpected_signatures = family_details["seen_signatures"] - expected_signatures
            if unexpected_signatures:
                message = (
                    f"Coverage integrity failure for ({arch}, {method}): "
                    "unexpected signatures escaped scope filtering "
                    f"({_coverage_preview(unexpected_signatures)})."
                )
                if full_coverage:
                    raise CoverageMismatchError(message)
                family_drop_reasons[(arch, method)] = {
                    "reason_code": "coverage_drop_family_unexpected_signatures",
                    "message": message,
                }
                continue
            missing_signatures = expected_signatures - family_details["seen_signatures"]
            if missing_signatures:
                message = (
                    f"Coverage mismatch for ({arch}, {method}): "
                    f"{len(family_details['seen_signatures'])}/"
                    f"{len(expected_signatures)} signatures present "
                    f"({len(missing_signatures)} missing)."
                )
                if full_coverage:
                    coverage_mismatches.append(message)
                else:
                    family_drop_reasons[(arch, method)] = {
                        "reason_code": "coverage_drop_family_missing_signatures",
                        "message": message,
                    }

    for arch in requested_arches_sorted:
        for method in requested_methods_sorted:
            spec = recipe_spec_by_method.get(method)
            if spec is None:
                raise ValueError(
                    f"Requested method '{method}' is missing from configured recipe scope."
                )
            kind = str(spec.pipeline_kind).strip()
            if kind != "wrap":
                continue
            if arch in arch_drop_reasons or arch in baseline_blocked_arches:
                family_drop_reasons[(arch, method)] = {
                    "reason_code": "coverage_drop_family_baseline_dependency",
                    "message": (
                        f"Coverage drop for ({arch}, {method}): "
                        "baseline coverage for this architecture was dropped."
                    ),
                }
                continue
            family_details = _collect_wrap_family_variant_details(
                arch=arch,
                pipeline_method=method,
                spec=spec,
                runs_by_variant=effective_runs_by_variant,
                resolved_by_run_id=resolved_by_run_id,
            )
            skip_reason = scope_policy_skip_reason_for_spec(spec, arch)
            expected_variants: dict[str, dict[str, Any]] = {}
            if skip_reason is None:
                expected_variants = _build_expected_wrap_variants_for_coverage(
                    spec=spec,
                    args=args,
                )
            if not expected_variants:
                if family_details["family_run_ids"]:
                    message = (
                        f"Coverage drop for ({arch}, {method}): "
                        "wrap family is out of the active expected scope but current-lineage runs exist."
                    )
                    if full_coverage:
                        coverage_mismatches.append(message)
                    else:
                        family_drop_reasons[(arch, method)] = {
                            "reason_code": "coverage_drop_family_unexpected_method",
                            "message": message,
                        }
                continue

            expected_pipeline_ids = set(expected_variants.keys())
            seen_pipeline_ids = set(family_details["seen_pipeline_ids"])
            common_pipeline_ids = seen_pipeline_ids & expected_pipeline_ids
            unexpected_pipeline_ids = seen_pipeline_ids - expected_pipeline_ids
            for pipeline_id in unexpected_pipeline_ids:
                variant_drop_reasons[(arch, method, pipeline_id)] = {
                    "reason_code": "coverage_drop_variant_unexpected_wrap_variant",
                    "message": (
                        f"Coverage drop for ({arch}, {method}, {pipeline_id}): "
                        "variant is outside the active expected wrap grid."
                    ),
                }
            mismatch_base_ids = {
                pipeline_id
                for pipeline_id in common_pipeline_ids
                if family_details["seen_base_method_by_pipeline_id"][pipeline_id]
                != expected_variants[pipeline_id]["base_pipeline_method"]
            }
            covered_expected_pipeline_ids = common_pipeline_ids - mismatch_base_ids
            coverage_fractions[(arch, method)] = (
                len(covered_expected_pipeline_ids),
                len(expected_pipeline_ids),
            )
            if full_coverage:
                missing_pipeline_ids = expected_pipeline_ids - seen_pipeline_ids
                if missing_pipeline_ids or mismatch_base_ids:
                    message = (
                        f"Wrap coverage mismatch for ({arch}, {method}): "
                        f"covered_expected_variants="
                        f"{len(covered_expected_pipeline_ids)}/{len(expected_pipeline_ids)}"
                    )
                    if missing_pipeline_ids:
                        message += (
                            f", missing={_coverage_preview(missing_pipeline_ids)}"
                        )
                    if mismatch_base_ids:
                        message += (
                            f", base_method_mismatch={_coverage_preview(mismatch_base_ids)}"
                        )
                    message += "."
                    coverage_mismatches.append(message)
                continue

            valid_expected_pipeline_ids: set[str] = set()
            valid_seen_pipeline_ids = set(seen_pipeline_ids - unexpected_pipeline_ids)
            dependency_out_of_scope_pipeline_ids: set[str] = set()

            for pipeline_id, metadata in expected_variants.items():
                base_method = metadata["base_pipeline_method"]
                if (arch, base_method) in family_drop_reasons:
                    dependency_out_of_scope_pipeline_ids.add(pipeline_id)
                    variant_drop_reasons[(arch, method, pipeline_id)] = {
                        "reason_code": "coverage_drop_variant_dependency_out_of_scope",
                        "message": (
                            f"Coverage drop for ({arch}, {method}, {pipeline_id}): "
                            f"upstream family ({arch}, {base_method}) was dropped."
                        ),
                    }
                    valid_seen_pipeline_ids.discard(pipeline_id)
                    continue
                valid_expected_pipeline_ids.add(pipeline_id)

            for pipeline_id in common_pipeline_ids:
                if pipeline_id in dependency_out_of_scope_pipeline_ids:
                    continue
                seen_base_method = family_details["seen_base_method_by_pipeline_id"][pipeline_id]
                expected_base_method = expected_variants[pipeline_id]["base_pipeline_method"]
                if seen_base_method == expected_base_method:
                    continue
                variant_drop_reasons[(arch, method, pipeline_id)] = {
                    "reason_code": "coverage_drop_variant_base_pipeline_method_mismatch",
                    "message": (
                        f"Coverage drop for ({arch}, {method}, {pipeline_id}): "
                        f"seen base_pipeline_method='{seen_base_method}' does not match "
                        f"expected '{expected_base_method}'."
                    ),
                }
                valid_seen_pipeline_ids.discard(pipeline_id)

            if not valid_expected_pipeline_ids:
                if family_details["family_run_ids"] or dependency_out_of_scope_pipeline_ids:
                    family_drop_reasons[(arch, method)] = {
                        "reason_code": "coverage_drop_family_all_variants_out_of_scope",
                        "message": (
                            f"Coverage drop for ({arch}, {method}): "
                            "all expected wrap variants were outside the active scope."
                        ),
                    }
                continue

            missing_pipeline_ids = valid_expected_pipeline_ids - valid_seen_pipeline_ids
            if missing_pipeline_ids:
                family_drop_reasons[(arch, method)] = {
                    "reason_code": "coverage_drop_family_missing_wrap_variants",
                    "message": (
                        f"Coverage drop for ({arch}, {method}): "
                        f"seen_variants={len(valid_seen_pipeline_ids)}/"
                        f"{len(valid_expected_pipeline_ids)}, "
                        f"missing={_coverage_preview(missing_pipeline_ids)}."
                    ),
                }

    if coverage_mismatches:
        summary_lines = ["Testing expected-vs-seen coverage mismatch detected:"]
        for msg in coverage_mismatches:
            summary_lines.append(f"  - {msg}")
        raise CoverageMismatchError("\n".join(summary_lines))

    filtered_runs_by_variant: dict[tuple[str, str, str], list[Any]] = {}
    for key, runs in effective_runs_by_variant.items():
        arch, pipeline_method, pipeline_id = (str(key[0]), str(key[1]), str(key[2]))
        reason = None
        if arch in arch_drop_reasons:
            reason = arch_drop_reasons[arch]["reason_code"]
        elif (arch, pipeline_method) in family_drop_reasons:
            reason = family_drop_reasons[(arch, pipeline_method)]["reason_code"]
        elif (arch, pipeline_method, pipeline_id) in variant_drop_reasons:
            reason = variant_drop_reasons[(arch, pipeline_method, pipeline_id)]["reason_code"]
        if reason is not None:
            for run in runs:
                dropped_run_reasons[run.info.run_id] = reason
            continue
        filtered_runs_by_variant[(arch, pipeline_method, pipeline_id)] = list(runs)

    surviving_baseline_arches = sorted(
        {
            arch
            for (arch, pipeline_method, _), runs in filtered_runs_by_variant.items()
            if pipeline_method == "baseline" and runs
        }
    )
    if (
        "baseline" in requested_methods
        and not full_coverage
        and not surviving_baseline_arches
    ):
        drop_diagnostics: list[str] = []
        for arch in requested_arches_sorted:
            if arch in arch_drop_reasons:
                drop_diagnostics.append(f"{arch}: {arch_drop_reasons[arch]['message']}")
            else:
                drop_diagnostics.append(f"{arch}: no surviving baseline coverage")
        raise CoverageMismatchError(
            "Coverage drop removed all architectures with baseline coverage. "
            f"Examples: {drop_diagnostics[:5]}"
        )

    if not full_coverage:
        n_drops = len(arch_drop_reasons) + len(family_drop_reasons) + len(variant_drop_reasons)
        if n_drops:
            print(
                f"Coverage relaxation: dropped {len(arch_drop_reasons)} architecture(s), "
                f"{len(family_drop_reasons)} family/families, "
                f"{len(variant_drop_reasons)} variant(s). See table below."
            )

    if coverage_fractions:
        partial = {
            k: v for k, v in coverage_fractions.items() if v[0] != v[1]
        }
        n_total = len(coverage_fractions)
        n_ok = n_total - len(partial)
        print(f"Coverage: {n_ok}/{n_total} families fully covered.")
        if partial:
            print("Partial coverage:")
            for (arch, method), (seen, expected) in sorted(partial.items()):
                ratio = seen / expected if expected > 0 else 0.0
                print(f"  {arch:>20s} | {method:<20s} | {seen}/{expected} ({ratio:.0%})")

    return {
        "runs_by_variant": filtered_runs_by_variant,
        "dropped_run_reasons": dropped_run_reasons,
        "arch_drop_reasons": arch_drop_reasons,
        "family_drop_reasons": family_drop_reasons,
        "variant_drop_reasons": variant_drop_reasons,
        "coverage_fractions": coverage_fractions,
    }


def build_base_index(runs_by_key: Mapping[Any, Sequence[Any]]) -> tuple[dict, dict]:
    sorted_by_key = {}
    current_by_key = {}
    for key, runs in runs_by_key.items():
        sorted_runs = sort_runs_by_metric(
            list(runs),
            metric_key="best_val_loss",
            missing_error_prefix=f"Baseline runs for key {key}",
        )
        sorted_by_key[key] = sorted_runs
        if sorted_runs:
            current_by_key[key] = sorted_runs[0]
    return sorted_by_key, current_by_key


def _current_baseline_id(current_base_runs_by_key: Mapping[Any, Any], arch: str, base_method: str) -> Optional[str]:
    current = current_base_runs_by_key.get((arch, base_method))
    return current.info.run_id if current else None


def _current_topk_ids(sorted_base_runs_by_key: Mapping[Any, Sequence[Any]], arch: str, base_method: str, k: int) -> Optional[set[str]]:
    if k <= 0:
        return set()
    sorted_runs = sorted_base_runs_by_key.get((arch, base_method), [])
    if len(sorted_runs) < k:
        return None
    return {run.info.run_id for run in sorted_runs[:k]}


def _resolve_lineage_baseline_hparam_spec(
    run,
    *,
    arch: str,
    baseline_hparam_specs_by_arch: Mapping[str, Any],
) -> Mapping[str, Any]:
    run_id = run.info.run_id
    tags = run.data.tags
    if tags is None:
        raise ValueError(
            f"Run {run_id} is missing tags required for baseline-hparams lineage checks."
        )
    spec_raw = baseline_hparam_specs_by_arch.get(arch)
    if spec_raw is None:
        raise ValueError(
            f"No baseline hyperparameter spec found for architecture '{arch}' "
            "in configs/baseline_hparams.yaml."
        )
    if not isinstance(spec_raw, Mapping):
        raise ValueError(
            f"Baseline hyperparameter spec for architecture '{arch}' must be a mapping."
        )
    require_nonempty_tag_value(
        tags,
        key="pipeline_method",
        run_id=run_id,
    )
    return spec_raw


def _extract_required_baseline_hparams(
    client: Any,
    run,
    *,
    arch: str,
    hparam_spec: Mapping[str, Any],
    cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    run_id = run.info.run_id
    if client is None:
        raise ValueError(
            "Lineage classification for inherit_baseline requires an artifact client "
            "to read hparams.json."
        )
    if cache is not None and run_id in cache:
        return dict(cache[run_id])
    with tempfile.TemporaryDirectory(prefix="robust-lineage-") as tmpdir:
        try:
            artifact_path = client.download_artifacts(
                run_id,
                "hparams.json",
                dst_path=tmpdir,
            )
        except Exception as exc:
            raise ValueError(
                f"Run {run_id} is missing required hparams.json for "
                f"baseline-hparams lineage checks on architecture '{arch}'."
            ) from exc
        try:
            with open(artifact_path, encoding="utf-8") as handle:
                hparams = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Run {run_id} hparams.json is not valid JSON for "
                f"baseline-hparams lineage checks on architecture '{arch}'."
            ) from exc
        except OSError as exc:
            raise ValueError(
                f"Run {run_id} hparams.json could not be read for "
                f"baseline-hparams lineage checks on architecture '{arch}'."
            ) from exc
    if not isinstance(hparams, Mapping):
        raise ValueError(
            f"Run {run_id} hparams.json must contain a mapping for "
            f"baseline-hparams lineage checks on architecture '{arch}'."
        )
    typed_hparams = extract_required_typed_hparams(
        hparams,
        hparam_spec,
        context=f"Run {run_id} baseline-hparams lineage extraction",
    )
    if cache is not None:
        cache[run_id] = dict(typed_hparams)
    return typed_hparams


def classify_lineage_run(
    run,
    arch: str,
    *,
    current_base_runs_by_key: Mapping[Any, Any],
    sorted_base_runs_by_key: Mapping[Any, Sequence[Any]],
    baseline_hparam_specs_by_arch: Optional[Mapping[str, Any]] = None,
    scoped_train_run_ids_by_key: Optional[Mapping[tuple[str, str], set[str]]] = None,
    artifact_client: Any | None = None,
    hparams_artifact_cache: dict[str, dict[str, Any]] | None = None,
) -> Optional[str]:
    tags = run.data.tags
    if tags is None:
        raise ValueError(
            f"Run {run.info.run_id} is missing tags in lineage classification."
        )
    pipeline_method = require_nonempty_tag_value(
        tags,
        key="pipeline_method",
        run_id=run.info.run_id,
    )
    if pipeline_method == "baseline":
        return None

    base_method = _resolve_base_pipeline_method(tags)
    if base_method is None:
        return "missing_base_pipeline_method"
    current_baseline_id = _current_baseline_id(current_base_runs_by_key, arch, base_method)

    backbone_ids = parse_backbone_run_ids(
        tags.get("backbone_run_ids"),
        run_id=run.info.run_id,
    )
    backbone_id = tags.get("backbone_run_id")
    baseline_hparams_run_id = tags.get("baseline_hparams_run_id")
    hparams_mode_raw = tags.get("hparams_mode")
    scoped_backbone_ids: Optional[set[str]] = None
    if scoped_train_run_ids_by_key is not None:
        scoped_backbone_ids = scoped_train_run_ids_by_key.get((arch, base_method), set())

    if backbone_ids:
        if scoped_backbone_ids is not None and not scoped_backbone_ids:
            return "missing_scope_backbone_pool"
        if scoped_backbone_ids is not None and not set(backbone_ids).issubset(scoped_backbone_ids):
            return "backbone_out_of_scope"
        if not current_baseline_id:
            return "missing_current_baseline"
        current_top_k = _current_topk_ids(
            sorted_base_runs_by_key, arch, base_method, len(backbone_ids)
        )
        if current_top_k is None:
            return "insufficient_current_backbones"
        if set(backbone_ids) != current_top_k:
            return "backbone_set_changed"
        return None

    if backbone_id:
        if scoped_backbone_ids is not None and not scoped_backbone_ids:
            return "missing_scope_backbone_pool"
        if scoped_backbone_ids is not None and str(backbone_id) not in scoped_backbone_ids:
            return "backbone_out_of_scope"
        if not current_baseline_id:
            return "missing_current_baseline"
        if str(backbone_id) != str(current_baseline_id):
            return "backbone_changed"
        return None

    if hparams_mode_raw is None or not str(hparams_mode_raw).strip():
        return "missing_hparams_mode"
    hparams_mode = str(hparams_mode_raw).strip()
    if hparams_mode == "baseline_grid":
        return None
    if hparams_mode == "inherit_baseline":
        if baseline_hparams_run_id is None or not str(baseline_hparams_run_id).strip():
            return "missing_baseline_hparams_run_id"
        if not current_baseline_id:
            return "missing_current_baseline"
        if baseline_hparam_specs_by_arch is None:
            raise ValueError(
                "Lineage classification for inherit_baseline requires baseline hyperparameter specs."
            )
        current_baseline_run = current_base_runs_by_key.get((arch, base_method))
        if current_baseline_run is None:
            return "missing_current_baseline"
        hparam_spec = _resolve_lineage_baseline_hparam_spec(
            run,
            arch=arch,
            baseline_hparam_specs_by_arch=baseline_hparam_specs_by_arch,
        )
        run_hparams = _extract_required_baseline_hparams(
            artifact_client,
            run,
            arch=arch,
            hparam_spec=hparam_spec,
            cache=hparams_artifact_cache,
        )
        current_hparams = _extract_required_baseline_hparams(
            artifact_client,
            current_baseline_run,
            arch=arch,
            hparam_spec=hparam_spec,
            cache=hparams_artifact_cache,
        )
        if run_hparams != current_hparams:
            return "baseline_changed"
        return None
    return "unknown_hparams_mode"


def resolve_pipeline_tags(tags: Mapping[str, Any], *, run_id: str | None = None) -> dict[str, str]:
    if run_id is not None:
        run_label = run_id
    else:
        run_name = tags.get("mlflow.runName")
        if run_name is not None and str(run_name).strip():
            run_label = str(run_name)
        else:
            run_label = "<unknown run>"
    robustness_method_raw = tags.get("robustness_method")
    if robustness_method_raw is None or not str(robustness_method_raw).strip():
        raise ValueError(
            f"Run {run_label} is missing robustness_method tag."
        )
    pipeline_method_raw = tags.get("pipeline_method")
    if (
        pipeline_method_raw is not None
        and str(pipeline_method_raw).strip() == "baseline"
        and str(robustness_method_raw).strip() != "baseline"
    ):
        raise ValueError(
            f"Run {run_label} has baseline pipeline_method but robustness_method="
            f"'{str(robustness_method_raw).strip()}'. "
            "Baseline runs must set robustness_method='baseline'."
        )
    resolved = validate_pipeline_tags_for_selection(tags, run_id=str(run_label))
    return {
        "pipeline_id": resolved["pipeline_id"],
        "pipeline_method": resolved["pipeline_method"],
        "pipeline_kind": resolved["pipeline_kind"],
        "robustness_method": resolved["robustness_method"],
    }


def select_group_winners(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    selection_metric_col: str = "best_val_loss",
) -> pd.DataFrame:
    metric_col = str(selection_metric_col).strip()
    if not metric_col:
        raise ValueError("selection_metric_col must be a non-empty string.")
    required_cols = set(group_cols) | {metric_col, "end_time", "run_id"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(
            "Cannot select group winners: missing required columns "
            f"{missing_cols}."
        )
    winner_df = df.copy()
    winner_df[metric_col] = pd.to_numeric(winner_df[metric_col], errors="raise")
    winner_df["end_time"] = pd.to_numeric(winner_df["end_time"], errors="raise")
    for group_col in group_cols:
        if winner_df[group_col].isna().any():
            raise ValueError(
                f"Cannot select group winners: column '{group_col}' contains missing values."
            )
    if winner_df["run_id"].isna().any():
        raise ValueError("Cannot select group winners: run_id contains missing values.")
    candidate_key_cols = list(group_cols)
    if "run_id" not in candidate_key_cols:
        candidate_key_cols.append("run_id")
    duplicate_candidate_mask = winner_df.duplicated(
        subset=candidate_key_cols,
        keep=False,
    )
    if duplicate_candidate_mask.any():
        duplicate_examples = (
            winner_df.loc[duplicate_candidate_mask, candidate_key_cols]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "Cannot select group winners: duplicate candidate rows share the same "
            f"group/run_id identity. Examples: {duplicate_examples}."
        )
    winner_df["selection_value"] = winner_df[metric_col]
    winner_df["_rank_key"] = winner_df.apply(
        lambda row: rank_key_for_dataframe_row(
            row,
            selection_value_col="selection_value",
            end_time_col="end_time",
            run_id_col="run_id",
        ),
        axis=1,
    )
    winner_df = winner_df.sort_values(["_rank_key"]).reset_index(drop=True)
    selected = winner_df.drop_duplicates(subset=group_cols, keep="first").reset_index(
        drop=True
    )
    return selected.drop(columns=["selection_value", "_rank_key"], errors="ignore")


def require_seed_tags(run) -> dict[str, Any]:
    tags = run.data.tags
    if tags is None:
        raise ValueError(
            f"Run {run.info.run_id} is missing tags required for seed validation."
        )
    seed_master = coerce_int(tags.get("seed_master"))
    seed_data = coerce_int(tags.get("seed_data"))
    seed_model = coerce_int(tags.get("seed_model"))
    seed_eval = coerce_int(tags.get("seed_eval"))
    seed_policy_raw = tags.get("seed_policy")
    seed_policy = str(seed_policy_raw).strip() if seed_policy_raw is not None else ""
    missing = []
    if seed_master is None:
        missing.append("seed_master")
    if seed_data is None:
        missing.append("seed_data")
    if seed_model is None:
        missing.append("seed_model")
    if seed_eval is None:
        missing.append("seed_eval")
    if not seed_policy:
        missing.append("seed_policy")
    if missing:
        raise ValueError(
            f"Run {run.info.run_id} is missing seed tag(s): {', '.join(missing)}. "
            "Rerun the model with current benchmark seed-tag logging before evaluating."
        )
    return {
        "seed_master": seed_master,
        "seed_data": seed_data,
        "seed_model": seed_model,
        "seed_eval": seed_eval,
        "seed_policy": seed_policy,
    }


def expected_perturbation_coupling_from_args(args) -> dict[str, Any]:
    perturbation_channel_fraction_max = parse_perturbation_channel_fraction_max(
        require_namespace_value(args, key="perturbation_channel_fraction_max"),
        key="perturbation_channel_fraction_max",
    )
    perturbation_scenarios = parse_perturbation_scenarios(
        require_namespace_value(args, key="perturbation_scenarios"),
        key="perturbation_scenarios",
    )
    perturbation_scenarios_signature = build_perturbation_scenarios_signature(
        perturbation_scenarios
    )
    perturbation_scenario_params_signature = (
        build_perturbation_scenario_params_signature(perturbation_scenarios)
    )
    return {
        "perturbation_channel_fraction_max": perturbation_channel_fraction_max,
        "perturbation_scenarios": perturbation_scenarios,
        "perturbation_scenarios_signature": perturbation_scenarios_signature,
        "perturbation_scenario_params_signature": (
            perturbation_scenario_params_signature
        ),
    }


def require_run_perturbation_idx_name_map(
    *,
    tags: Mapping[str, Any],
    run_id: str,
    expected_scenarios: Sequence[str] | None = None,
) -> dict[int, str]:
    try:
        idx_name_map = parse_perturbation_idx_name_map(
            tags.get("perturbation_idx_name_map"),
            key="perturbation_idx_name_map",
        )
    except ValueError as exc:
        raise ValueError(
            f"Run {run_id} has invalid perturbation_idx_name_map tag: {exc}"
        ) from exc
    if expected_scenarios is not None:
        expected_names = parse_perturbation_scenarios(
            list(expected_scenarios),
            key="expected perturbation_scenarios",
        )
        map_names = tuple(idx_name_map[idx] for idx in sorted(idx_name_map))
        if len(map_names) != len(expected_names) or set(map_names) != set(expected_names):
            raise ValueError(
                f"Run {run_id} perturbation_idx_name_map scenarios {sorted(map_names)} "
                f"do not match expected perturbation_scenarios {sorted(expected_names)}."
            )
    return idx_name_map


def require_matching_perturbation_coupling_params(
    run,
    *,
    args,
    context: str,
) -> dict[str, Any]:
    expected_coupling = expected_perturbation_coupling_from_args(args)
    run_id = run.info.run_id
    tags = run.data.tags
    if tags is None:
        raise ValueError(f"Run {run_id} {context} is missing tags.")
    try:
        coupling = require_perturbation_coupling_tags(
            tags,
            run_id=run_id,
            expected_max=expected_coupling["perturbation_channel_fraction_max"],
            expected_scenarios_signature=expected_coupling[
                "perturbation_scenarios_signature"
            ],
        )
        params_signature = require_nonempty_tag_value(
            tags,
            key="perturbation_scenario_params_signature",
            run_id=run_id,
        )
        if params_signature != expected_coupling["perturbation_scenario_params_signature"]:
            raise ValueError(
                "perturbation_scenario_params_signature does not match current "
                "perturbation scenario defaults."
            )
        coupling["perturbation_scenario_params_signature"] = params_signature
        return coupling
    except ValueError as exc:
        raise ValueError(
            f"Run {run_id} {context} has incompatible perturbation coupling params: {exc}"
        ) from exc


def is_fully_tested(run, *, args, client: Any | None = None) -> bool:
    params = run.data.params
    if params is None:
        return False
    if "tested" not in params:
        return False
    if not require_tested_param(params, run_id=run.info.run_id):
        return False
    tags = run.data.tags
    if tags is None:
        raise ValueError(
            f"Run {run.info.run_id} has tested='true' but missing tags."
        )
    semantics = require_nonempty_tag_value(
        tags,
        key="robustness_scoring_semantics",
        run_id=run.info.run_id,
    )
    if semantics != ROBUSTNESS_SCORING_SEMANTICS:
        return False

    expected_eval_data_seed = resolve_effective_eval_data_seed(
        require_namespace_value(args, key="eval_data_seed"),
        canonical_seed_data=tags.get("seed_data"),
        eval_key="args.eval_data_seed",
        canonical_key="seed_data tag",
    )
    expected_eval_context = require_degradation_eval_context_from_args(
        args,
        eval_data_seed=expected_eval_data_seed,
        context="args",
    )
    expected_bootstrap_ci_context = require_shared_anchor_bootstrap_ci_context_from_args(
        args,
        eval_data_seed=expected_eval_data_seed,
        test_metric=expected_eval_context["test_metric"],
        context="args",
    )
    run_eval_context = require_degradation_eval_context_tags(
        tags,
        run_id=run.info.run_id,
    )
    run_params_signature = require_nonempty_tag_value(
        tags,
        key="perturbation_scenario_params_signature",
        run_id=run.info.run_id,
    )
    expected_params_signature = build_perturbation_scenario_params_signature(
        require_namespace_value(args, key="perturbation_scenarios")
    )
    if run_params_signature != expected_params_signature:
        return False
    run_bootstrap_ci_context = require_shared_anchor_bootstrap_ci_context_tags(
        tags,
        run_id=run.info.run_id,
        require_seed=True,
    )
    expected_eval_identity = dict(expected_eval_context)
    expected_eval_identity.pop("n_test_samples")
    if not degradation_eval_context_matches(
        run_eval_context,
        expected_context=expected_eval_identity,
    ):
        return False
    if not shared_anchor_bootstrap_ci_context_matches(
        run_bootstrap_ci_context,
        expected_context=expected_bootstrap_ci_context,
        require_seed=True,
    ):
        return False

    is_complete_bundle = require_robustness_results_complete_tag(
        tags,
        run_id=run.info.run_id,
    )
    if not is_complete_bundle:
        return False

    require_logged_degradation_metric_bundle(
        run.data.metrics,
        tags=tags,
        params=params,
        run_id=run.info.run_id,
        test_metric=expected_eval_context["test_metric"],
        expected_idx_to_name=expected_eval_context["perturbation_idx_name_map"],
    )
    if client is not None:
        test_metric = str(expected_eval_context["test_metric"])
        try:
            download_validated_degradation_artifact_bundle(
                client,
                run_id=run.info.run_id,
                test_metric=test_metric,
                eval_data_seed=int(run_eval_context["eval_data_seed"]),
                expected_idx_to_name=run_eval_context["perturbation_idx_name_map"],
                expected_n_test_samples=int(run_eval_context["n_test_samples"]),
                expected_clean_metric_value=run.data.metrics[f"{test_metric}_test"],
                context_name=f"Run {run.info.run_id} canonical degradation artifacts",
            )
        except Exception as exc:
            raise ValueError(
                f"Run {run.info.run_id} has tested='true' but invalid canonical "
                f"degradation artifacts: {exc}"
            ) from exc
    run_n_test_samples = int(run_eval_context["n_test_samples"])
    expected_n_test_samples = int(expected_eval_context["n_test_samples"])
    full_coverage = require_namespace_bool(args, key="full_coverage")
    if not degradation_n_test_samples_meet_policy(
        run_n_test_samples,
        expected_n_test_samples=expected_n_test_samples,
        full_coverage=full_coverage,
    ):
        return False
    return True


def is_fixed_channel_fraction_complete(
    run,
    *,
    args,
    client: Any,
    fixed_fraction: Any,
) -> bool:
    """Return whether a run has a complete fixed-channel-fraction bundle."""
    if client is None:
        raise ValueError(
            "client is required for fixed-channel-fraction completion because "
            "completion validation must read context.json and artifacts."
        )
    if not is_fully_tested(run, args=args, client=client):
        return False
    tags = run.data.tags
    if tags is None:
        raise ValueError(
            f"Run {run.info.run_id} is missing tags required for fixed-channel-fraction "
            "completion."
        )
    metrics = run.data.metrics
    if metrics is None:
        raise ValueError(
            f"Run {run.info.run_id} is missing metrics required for fixed-channel-fraction "
            "completion."
        )
    max_fraction = parse_perturbation_channel_fraction_max(
        require_namespace_value(args, key="perturbation_channel_fraction_max"),
        key="args.perturbation_channel_fraction_max",
    )
    parsed_fraction = parse_optional_unit_float(
        fixed_fraction,
        key="fixed_channel_fraction",
        max_value=max_fraction,
    )
    if parsed_fraction is None:
        raise ValueError("fixed_fraction is required for fixed-channel-fraction completion.")

    run_eval_context = require_degradation_eval_context_tags(
        tags,
        run_id=run.info.run_id,
    )
    run_bootstrap_ci_context = require_shared_anchor_bootstrap_ci_context_tags(
        tags,
        run_id=run.info.run_id,
        require_seed=True,
    )
    run_params_signature = require_nonempty_tag_value(
        tags,
        key="perturbation_scenario_params_signature",
        run_id=run.info.run_id,
    )
    canonical_context_signature = build_canonical_degradation_context_signature(
        degradation_eval_context=run_eval_context,
        bootstrap_ci_context=run_bootstrap_ci_context,
        perturbation_scenario_params_signature=run_params_signature,
    )
    complete_tag = build_fixed_channel_fraction_tag_key(
        fixed_channel_fraction=parsed_fraction,
        perturbation_channel_fraction_max=max_fraction,
        tag_name="complete",
    )
    complete_value_raw = tags.get(complete_tag)
    if complete_value_raw is None:
        return False
    complete_value = str(complete_value_raw).strip().lower()
    if complete_value == "false":
        return False
    if complete_value != "true":
        raise ValueError(
            f"Run {run.info.run_id} has malformed fixed-channel-fraction "
            f"completion tag {complete_tag}={complete_value_raw!r}."
        )
    fraction_tag = build_fixed_channel_fraction_tag_key(
        fixed_channel_fraction=parsed_fraction,
        perturbation_channel_fraction_max=max_fraction,
        tag_name="fixed_channel_fraction",
    )
    logged_fraction = parse_optional_unit_float(
        require_nonempty_tag_value(
            tags,
            key=fraction_tag,
            run_id=run.info.run_id,
        ),
        key=fraction_tag,
        max_value=max_fraction,
    )
    if logged_fraction is None:
        raise ValueError(
            f"Run {run.info.run_id} has complete fixed-channel-fraction metadata but "
            f"{fraction_tag} is null."
        )
    if abs(float(logged_fraction) - float(parsed_fraction)) > 1e-12:
        raise ValueError(
            f"Run {run.info.run_id} has complete fixed-channel-fraction metadata for "
            f"{fraction_tag}={logged_fraction}, expected {parsed_fraction}."
        )
    context_signature_tag = build_fixed_channel_fraction_tag_key(
        fixed_channel_fraction=parsed_fraction,
        perturbation_channel_fraction_max=max_fraction,
        tag_name="context_signature",
    )
    context_signature = require_nonempty_tag_value(
        tags,
        key=context_signature_tag,
        run_id=run.info.run_id,
    )

    require_logged_fixed_channel_fraction_metric_bundle(
        metrics,
        tags=tags,
        run_id=run.info.run_id,
        test_metric=str(run_eval_context["test_metric"]),
        fixed_channel_fraction=parsed_fraction,
        perturbation_channel_fraction_max=max_fraction,
        expected_idx_to_name=run_eval_context["perturbation_idx_name_map"],
    )
    try:
        test_metric = str(run_eval_context["test_metric"])
        canonical_clean_df, _, _ = download_validated_degradation_artifact_bundle(
            client,
            run_id=run.info.run_id,
            test_metric=test_metric,
            eval_data_seed=int(run_eval_context["eval_data_seed"]),
            expected_idx_to_name=run_eval_context["perturbation_idx_name_map"],
            expected_n_test_samples=int(run_eval_context["n_test_samples"]),
            expected_clean_metric_value=metrics[f"{test_metric}_test"],
            context_name=(
                f"Run {run.info.run_id} canonical degradation artifacts for "
                "fixed-channel-fraction completion"
            ),
        )
        download_validated_fixed_channel_fraction_artifact_bundle(
            client,
            run_id=run.info.run_id,
            test_metric=test_metric,
            eval_data_seed=int(run_eval_context["eval_data_seed"]),
            fixed_channel_fraction=parsed_fraction,
            perturbation_channel_fraction_max=max_fraction,
            expected_idx_to_name=run_eval_context["perturbation_idx_name_map"],
            expected_n_test_samples=int(run_eval_context["n_test_samples"]),
            expected_clean_df=canonical_clean_df,
            expected_context_signature=context_signature,
            expected_perturbation_scenarios_signature=str(
                run_eval_context["perturbation_scenarios_signature"]
            ),
            expected_perturbation_scenario_params_signature=run_params_signature,
            expected_canonical_context_signature=canonical_context_signature,
            expected_bootstrap_ci_context=run_bootstrap_ci_context,
            context_name=(
                f"Run {run.info.run_id} fixed-channel-fraction artifacts"
            ),
        )
    except Exception as exc:
        raise ValueError(
            f"Run {run.info.run_id} has fixed-channel-fraction completion tag but "
            f"invalid artifacts: {exc}"
        ) from exc
    return True


__all__ = [
    "CoverageMismatchError",
    "PIPELINE_CONFIGS_DIR",
    "ROBUSTNESS_SCORING_SEMANTICS",
    "audit_and_apply_testing_coverage_policy",
    "build_base_index",
    "classify_lineage_run",
    "expected_perturbation_coupling_from_args",
    "extract_recipe_defaults_for_scope",
    "is_fully_tested",
    "is_fixed_channel_fraction_complete",
    "load_benchmark_method_architecture_applicability",
    "load_benchmark_recipe_specs_for_scope",
    "load_recipe_specs_for_scope",
    "merge_recipe_defaults_for_scope",
    "require_matching_perturbation_coupling_params",
    "require_run_perturbation_idx_name_map",
    "require_seed_tags",
    "resolve_pipeline_tags",
    "resolve_benchmark_method_architecture_scope",
    "resolve_requested_architectures",
    "resolve_requested_methods",
    "expand_testing_method_scope_for_wrap_dependencies",
    "scope_exclusion_reason",
    "select_group_winners",
]
