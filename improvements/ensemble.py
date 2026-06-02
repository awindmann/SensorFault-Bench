from __future__ import annotations

import os
from typing import List, Optional

import mlflow

from .base import (
    BackboneReference,
    BaseImprovement,
    ImprovementSpec,
    download_backbone_reference_checkpoint,
    load_improvement_spec,
    resolve_wrap_backbone_references,
)
from models.ensemble import Ensemble as EnsembleModel
import models
from utils.artifacts import (
    load_lightning_module_checkpoint,
    require_downloaded_checkpoint_unlinker,
)
from utils.parsing import (
    parse_required_positive_int,
    require_namespace_value,
    validate_ensemble_combine_method,
)


class EnsembleImprovement(BaseImprovement):
    registry_name = "ensemble"
    loader_kind_key = EnsembleModel.__name__

    def __init__(self, args):
        super().__init__(args)
        self.top_k = require_namespace_value(args, key="ensemble_top_k")
        self.combine_method = require_namespace_value(args, key="ensemble_combine_method")
        self.scope = require_namespace_value(args, key="ensemble_scope")
        if self.top_k is None:
            raise ValueError("ensemble_top_k must be set.")
        if self.combine_method is None:
            raise ValueError("ensemble_combine_method must be set.")
        if self.scope is None:
            raise ValueError("ensemble_scope must be set.")
        self.top_k = int(self.top_k)
        self.combine_method = validate_ensemble_combine_method(
            self.combine_method,
            key="ensemble_combine_method",
        )
        self.tag = f"ensemble_top{self.top_k}_{self.combine_method}"

    def should_skip(self, base_runs: List) -> bool:
        if self.top_k < 2:
            return True
        valid_runs = [r for r in base_runs if self.sort_metric in r.data.metrics]
        return len(valid_runs) < self.top_k

    def existing_run(self, improvement_runs: List, base_selection: List) -> Optional[str]:
        matches = self.matching_runs(improvement_runs, base_selection)
        if not matches:
            return None
        return matches[0].info.run_id

    def matching_runs(self, improvement_runs: List, base_selection: List) -> List:
        selected_ids = {run.info.run_id for run in base_selection}
        base_tags = base_selection[0].data.tags
        if base_tags is None:
            raise ValueError(
                f"Base run {base_selection[0].info.run_id} is missing tags."
            )
        expected_base_method = base_tags.get("pipeline_method")
        if expected_base_method is None or not str(expected_base_method).strip():
            raise ValueError(
                f"Base run {base_selection[0].info.run_id} is missing pipeline_method tag."
            )
        expected_base_method = str(expected_base_method).strip()
        top_k = len(base_selection)
        pipeline_method = self._require_registry_name()
        matches = []
        for candidate in improvement_runs:
            if getattr(candidate.info, "lifecycle_stage", "active") != "active":
                continue
            tags = candidate.data.tags
            if tags is None:
                raise ValueError(
                    f"Ensemble candidate run {candidate.info.run_id} is missing tags."
                )
            if tags.get("pipeline_method") != pipeline_method:
                continue
            if tags.get("base_pipeline_method") != expected_base_method:
                continue
            recorded_tag = tags.get("backbone_run_ids")
            if recorded_tag is None or not str(recorded_tag).strip():
                continue
            recorded_tag_ids = {rid.strip() for rid in str(recorded_tag).split(",") if rid.strip()}
            if recorded_tag_ids != selected_ids:
                continue
            if candidate.data.params.get("ensemble_combine_method") != self.combine_method:
                continue
            if candidate.data.params.get("ensemble_top_k") != str(top_k):
                continue
            matches.append(candidate)
        return matches

    def extra_tags(self, dataset_name: str, architecture: str, base_selection: List) -> dict:
        tags = super().extra_tags(dataset_name, architecture, base_selection) or {}
        tags["ensemble_scope"] = self.scope
        return tags

    def create_spec(
        self,
        client,
        dataset_name: str,
        architecture: str,
        base_selection: List,
        active_run,
    ) -> ImprovementSpec:
        best_losses = [
            run.data.metrics.get(self.sort_metric)
            for run in base_selection
            if run.data.metrics.get(self.sort_metric) is not None
        ]
        if not best_losses:
            raise ValueError("Selected backbone runs do not expose the sorting metric.")

        run_ids = [run.info.run_id for run in base_selection]
        mlflow.log_param("ensemble_top_k", str(len(base_selection)))
        mlflow.log_param("ensemble_combine_method", self.combine_method)
        mlflow.log_param("ensemble_scope", self.scope)
        joined_ids = ",".join(run_ids)
        mlflow.log_param("ensemble_backbone_runs", joined_ids)
        mlflow.log_metric(self.sort_metric, min(best_losses))
        backbone_references: List[BackboneReference] = []
        for base_run in base_selection:
            backbone_references.append(self.capture_backbone_reference(base_run))

        parameters = {
            "ensemble_combine_method": self.combine_method,
            "ensemble_top_k": len(base_selection),
            "ensemble_scope": self.scope,
        }

        return ImprovementSpec(
            parameters=parameters,
            backbones=backbone_references,
        )


def build_ensemble_model(client, run):
    spec = load_improvement_spec(client, run)
    required_keys = {"ensemble_combine_method", "ensemble_top_k", "ensemble_scope"}
    missing = required_keys - set(spec.parameters.keys())
    if missing:
        raise ValueError(
            f"Ensemble improvement run {run.info.run_id} missing required spec keys "
            f"{sorted(missing)}. Re-run improvements to generate updated specs."
        )
    combine_method = spec.parameters.get("ensemble_combine_method")
    if combine_method is None or not str(combine_method).strip():
        raise ValueError(
            f"Ensemble improvement run {run.info.run_id} is missing required "
            "'ensemble_combine_method' in its spec parameters."
        )
    combine_method = validate_ensemble_combine_method(
        combine_method,
        key="ensemble_combine_method",
    )
    backbones = resolve_wrap_backbone_references(run, spec)
    ensemble_top_k = parse_required_positive_int(
        spec.parameters.get("ensemble_top_k"),
        key="ensemble_top_k",
    )
    if len(backbones) != ensemble_top_k:
        raise ValueError(
            f"Ensemble improvement run {run.info.run_id} stores ensemble_top_k={ensemble_top_k} "
            f"but references {len(backbones)} backbones in its spec."
        )
    cleanup_checkpoint = require_downloaded_checkpoint_unlinker(
        client,
        context=f"ensemble loader for wrap run {run.info.run_id}",
    )
    backbone_modules = []
    for reference in backbones:
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
            context=f"ensemble backbone for wrap run {run.info.run_id}",
        )
        backbone_modules.append(backbone)
    ensemble_model = EnsembleModel(backbones=backbone_modules, combine_method=combine_method)
    return ensemble_model, os.getcwd()
