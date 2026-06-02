"""Randomized smoothing robustness improvement method (test-time)."""

from __future__ import annotations

import os
from typing import Callable, List, Optional

import mlflow

import models
from models.randomized_smoothing import RandomizedSmoothing
from .base import (
    BaseImprovement,
    ImprovementSpec,
    download_backbone_reference_checkpoint,
    load_improvement_spec,
    resolve_wrap_backbone_references,
)
from utils.artifacts import (
    load_lightning_module_checkpoint,
    require_downloaded_checkpoint_unlinker,
)
from utils.parsing import (
    parse_required_choice,
    parse_required_nonnegative_float,
    parse_required_positive_int,
    require_namespace_value,
    validate_trim_alpha,
)


VALID_RS_BACKBONE_METHODS = ("baseline", "randomized_training")


class RandomizedSmoothingImprovement(BaseImprovement):
    """Wrap a backbone with randomized smoothing for alpha-trimmed regression."""

    registry_name = "randomized_smoothing"
    loader_kind_key = RandomizedSmoothing.__name__

    def __init__(self, args):
        super().__init__(args)
        self.noise_std = parse_required_nonnegative_float(
            require_namespace_value(args, key="rs_noise_std"),
            key="rs_noise_std",
        )
        self.sample_count = parse_required_positive_int(
            require_namespace_value(args, key="rs_sample_count"),
            key="rs_sample_count",
        )
        self.rs_backbone_method = parse_required_choice(
            require_namespace_value(args, key="rs_backbone_method"),
            key="rs_backbone_method",
            allowed=VALID_RS_BACKBONE_METHODS,
        )
        self.trim_alpha = validate_trim_alpha(
            require_namespace_value(args, key="rs_trim_alpha"),
            self.sample_count,
        )
        self.tag = (
            "rs_"
            f"backbone{self.rs_backbone_method}_std{self.noise_std}_samples{self.sample_count}"
            f"_alpha{self.trim_alpha}"
        )

    def existing_run(self, improvement_runs: List, base_selection: List) -> Optional[str]:
        matches = self.matching_runs(improvement_runs, base_selection)
        if not matches:
            return None
        return matches[0].info.run_id

    def matching_runs(self, improvement_runs: List, base_selection: List) -> List:
        base_id = base_selection[0].info.run_id if base_selection else None
        expected_base_method = str(self.rs_backbone_method)
        pipeline_method = self._require_registry_name()
        matches = []
        for candidate in improvement_runs:
            if getattr(candidate.info, "lifecycle_stage", "active") != "active":
                continue
            tags = candidate.data.tags
            if tags is None:
                raise ValueError(
                    f"Randomized smoothing candidate run {candidate.info.run_id} is missing tags."
                )
            if tags.get("pipeline_method") != pipeline_method:
                continue
            if tags.get("base_pipeline_method") != expected_base_method:
                continue
            if tags.get("backbone_run_id") != base_id:
                continue
            pipeline_id = tags.get("pipeline_id")
            if pipeline_id is None or not str(pipeline_id).strip():
                raise ValueError(
                    f"Randomized smoothing run {candidate.info.run_id} is missing required pipeline_id tag."
                )
            if str(pipeline_id) != self.tag:
                continue
            candidate_noise_std = self._require_matching_param(
                candidate,
                key="rs_noise_std",
                parser=lambda value: parse_required_nonnegative_float(
                    value,
                    key="rs_noise_std",
                ),
            )
            if candidate_noise_std != self.noise_std:
                continue
            candidate_sample_count = self._require_matching_param(
                candidate,
                key="rs_sample_count",
                parser=lambda value: parse_required_positive_int(
                    value,
                    key="rs_sample_count",
                ),
            )
            if candidate_sample_count != self.sample_count:
                continue
            candidate_backbone_method = self._require_matching_param(
                candidate,
                key="rs_backbone_method",
                parser=lambda value: parse_required_choice(
                    value,
                    key="rs_backbone_method",
                    allowed=VALID_RS_BACKBONE_METHODS,
                ),
            )
            if candidate_backbone_method != self.rs_backbone_method:
                continue
            candidate_trim_alpha = self._require_matching_param(
                candidate,
                key="rs_trim_alpha",
                parser=lambda value: validate_trim_alpha(
                    value,
                    candidate_sample_count,
                ),
            )
            if candidate_trim_alpha != self.trim_alpha:
                continue
            matches.append(candidate)
        return matches

    @staticmethod
    def _require_matching_param(
        candidate,
        *,
        key: str,
        parser: Callable[[object], object] | None = None,
    ) -> object:
        params = candidate.data.params
        value = params.get(key)
        if value is None:
            raise ValueError(
                f"Randomized smoothing run {candidate.info.run_id} is missing "
                f"required param '{key}'."
            )
        if parser is None:
            return value
        try:
            return parser(value)
        except ValueError as exc:
            raise ValueError(
                f"Randomized smoothing run {candidate.info.run_id} has invalid param "
                f"'{key}': {exc}"
            ) from exc

    def create_spec(self, client, dataset_name, architecture, base_selection, active_run) -> ImprovementSpec:
        base_run = base_selection[0]
        best_loss = base_run.data.metrics.get(self.sort_metric)
        if best_loss is None:
            raise ValueError(
                f"Base run {base_run.info.run_id} is missing '{self.sort_metric}'."
            )
        mlflow.log_param("rs_noise_std", str(self.noise_std))
        mlflow.log_param("rs_sample_count", str(self.sample_count))
        mlflow.log_param("rs_backbone_method", str(self.rs_backbone_method))
        mlflow.log_param("rs_trim_alpha", str(self.trim_alpha))
        mlflow.log_param("backbone_run_id", base_run.info.run_id)
        mlflow.log_metric(self.sort_metric, float(best_loss))

        parameters = {
            "rs_noise_std": self.noise_std,
            "rs_sample_count": self.sample_count,
            "rs_backbone_method": self.rs_backbone_method,
            "rs_trim_alpha": self.trim_alpha,
        }
        backbone_reference = self.capture_backbone_reference(base_run)
        return ImprovementSpec(parameters=parameters, backbones=[backbone_reference])


def build_randomized_smoothing_model(client, run):
    spec = load_improvement_spec(client, run)
    required_keys = {
        "rs_noise_std",
        "rs_sample_count",
        "rs_backbone_method",
        "rs_trim_alpha",
    }
    missing = required_keys - set(spec.parameters.keys())
    if missing:
        raise ValueError(
            f"Randomized smoothing run {run.info.run_id} missing required spec keys "
            f"{sorted(missing)}. Re-run improvements to generate updated specs."
        )
    if not spec.backbones:
        raise ValueError(
            f"Randomized smoothing run {run.info.run_id} has no backbone artifacts."
        )
    references = resolve_wrap_backbone_references(run, spec)
    if len(references) != 1:
        raise ValueError(
            f"Randomized smoothing run {run.info.run_id} expected 1 backbone, got {len(references)}."
        )

    reference = references[0]
    backbone_method = parse_required_choice(
        spec.parameters.get("rs_backbone_method"),
        key="rs_backbone_method",
        allowed=VALID_RS_BACKBONE_METHODS,
    )
    if reference.pipeline_method != backbone_method:
        raise ValueError(
            f"Randomized smoothing run {run.info.run_id} stores rs_backbone_method="
            f"'{backbone_method}' but backbone reference {reference.run_id} stores "
            f"pipeline_method='{reference.pipeline_method}'."
        )
    cleanup_checkpoint = require_downloaded_checkpoint_unlinker(
        client,
        context=f"randomized smoothing loader for wrap run {run.info.run_id}",
    )
    checkpoint_path = download_backbone_reference_checkpoint(client, run, reference)
    if not hasattr(models, reference.model_architecture):
        raise ValueError(
            f"Unknown backbone architecture '{reference.model_architecture}' in improvement spec "
            f"for run {run.info.run_id}."
        )

    base_class = getattr(models, reference.model_architecture)
    backbone = load_lightning_module_checkpoint(base_class, checkpoint_path)
    cleanup_checkpoint(
        checkpoint_path,
        run_id=reference.run_id,
        context=f"randomized smoothing backbone for wrap run {run.info.run_id}",
    )
    sample_count = parse_required_positive_int(
        spec.parameters.get("rs_sample_count"),
        key="rs_sample_count",
    )
    noise_std = parse_required_nonnegative_float(
        spec.parameters.get("rs_noise_std"),
        key="rs_noise_std",
    )
    trim_alpha = validate_trim_alpha(
        spec.parameters.get("rs_trim_alpha"),
        sample_count,
    )
    wrapper = RandomizedSmoothing(
        wrapped_backbone=backbone,
        noise_std=noise_std,
        sample_count=sample_count,
        trim_alpha=trim_alpha,
    )
    return wrapper, os.getcwd()
